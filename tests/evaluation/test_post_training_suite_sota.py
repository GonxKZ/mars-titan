"""Pruebas TDD de `Stage2PostTrainingPipeline` y `TFMTraceAndTableExporter`.

Verifica de forma estricta:
  1. Desacoplamiento Orden+Nivel (FM-7 / F7-b): `OrderLevelDecoupler` (Huber y LAD) preserva
     exactamente el Rank IC (invarianza estricta de orden) reduciendo drásticamente el error
     de nivel MAE/RMSE fuera de muestra.
  2. `Stage2PostTrainingPipeline` extremo a extremo sobre 4 familias independientes
     (`titans_original`, `holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`):
     - Calibración conforme `NormalizedConformalPID` estratificada por deciles Mondrian de
       volatilidad (intervalos 80%/90%/95% no cruzados) + `VennAbersCalibrator` (`[p0, p1]`).
     - Combinación causal ortogonal `OAS-ICIR` + contracción risk-parity + gobernador
       `MetacognitiveGovernor.decide_torch` + cartera convexa `ConvexPortfolioOptimizerPPlus`.
     - Informe completo 6D (32 métricas) + contraste pareado Stage-2 vs Stage-1 (Newey-West t,
       delta IC, delta CRPS, delta Net Sharpe 1x/2x/3x, delta MaxDD, delta Capacity).
  3. Causalidad estricta bit-a-bit y sello `2024+` (`FINAL_TEST_START_US`):
     - Rechaza `prediction_at >= 2024`, `matures_at >= 2024`, `session_years >= 2024` y solapes
       entre `cal_split` y `eval_split`.
     - Mutar sesiones futuras `s > s0` en `eval_split` produce 0 bits de cambio en `<= s0`.
  4. `TFMTraceAndTableExporter`: lectura real de `telemetry.jsonl` + `experiment_memory.jsonl` +
     evaluaciones 6D + veredictos `GainValidationGate` -> exportación de las 4 tablas TFM en
     Markdown y LaTeX (`booktabs`, caracteres escapados) + `export_execution_trace_summary`.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation.post_training_suite_sota import (
    FOUR_FAMILIES,
    OrderLevelDecoupler,
    Stage2Config,
    Stage2PostTrainingPipeline,
    Stage2SplitData,
    TFMTraceAndTableExporter,
    read_compressed_trace_bundle,
)
from mars_titan.models.mmarrs.long_running_harness import (
    DEFAULT_MATCHED_CONTROLS,
    GainValidationGate,
    SealedTestViolationError,
)
from mars_titan.models.mmarrs.model import build_variation
from mars_titan.models.mmarrs.telemetry import ResearchTelemetryMonitor, TelemetryConfig

_DAY_US = 86_400 * 1_000_000
_BASE_2022_US = 1_641_000_000_000_000  # 2022-01-01 aprox (estrictamente < 2024)


def _make_multi_family_split(
    *,
    n_sessions: int,
    n_assets: int,
    start_us: int,
    seed: int,
    families: tuple[str, ...] = FOUR_FAMILIES,
    scale_miscalibration: float = 40.0,
) -> Stage2SplitData:
    """Genera un tramo sintético con señal ortogonal por familia y volatilidad heterogénea."""
    rng = np.random.default_rng(seed)
    n_total = n_sessions * n_assets
    bounds = [(s * n_assets, (s + 1) * n_assets) for s in range(n_sessions)]

    pred_at = np.empty(n_total, dtype=np.int64)
    mat_at = np.empty(n_total, dtype=np.int64)
    vol = np.empty(n_total, dtype=np.float64)
    liq = np.empty(n_total, dtype=np.float64)
    exposures = rng.standard_normal((n_total, 4)).astype(np.float64)
    target = np.empty(n_total, dtype=np.float64)
    family_preds = {fam: np.empty(n_total, dtype=np.float64) for fam in families}
    q9_raw = np.empty((n_total, 9), dtype=np.float64)
    hop_e = np.abs(rng.standard_normal(n_total)) * 0.4
    r_ent = np.full(n_total, 1.2, dtype=np.float64)
    l_eng = np.abs(rng.standard_normal(n_total)) * 0.1

    z_quantiles = np.array([-2.326, -1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645, 2.326])

    for s, (lo, hi) in enumerate(bounds):
        t_s = start_us + s * 2 * _DAY_US
        pred_at[lo:hi] = t_s
        mat_at[lo:hi] = t_s + _DAY_US

        # Volatilidad heterogénea entre 0.8% y 5% diario para ejercitar deciles Mondrian
        v_s = np.exp(rng.uniform(np.log(0.008), np.log(0.05), size=n_assets))
        vol[lo:hi] = v_s
        liq[lo:hi] = np.exp(rng.uniform(np.log(5e6), np.log(8e7), size=n_assets))

        # 3 componentes latentes ortogonales de alfa puro + factor común + ruido
        a1 = rng.standard_normal(n_assets)
        a2 = rng.standard_normal(n_assets)
        a3 = rng.standard_normal(n_assets)
        fac_ret = exposures[lo:hi] @ np.array([0.003, -0.002, 0.0015, 0.001])
        idio_alpha = (0.35 * a1 + 0.35 * a2 + 0.35 * a3) * 0.007
        y_s = fac_ret + idio_alpha + rng.standard_normal(n_assets) * v_s
        target[lo:hi] = y_s

        # Cada familia captura un subconjunto ortogonal distinto (con amplitud descalibrada!)
        for f_idx, fam in enumerate(families):
            if f_idx == 0:
                sig = 0.7 * a1 + 0.25 * a2 + rng.standard_normal(n_assets) * 1.1
            elif f_idx == 1:
                sig = 0.7 * a2 + 0.25 * a3 + rng.standard_normal(n_assets) * 1.1
            elif f_idx == 2:
                sig = 0.55 * a1 + 0.55 * a3 + rng.standard_normal(n_assets) * 0.9
            else:
                # 4a familia con señal más débil + ruido para verificar contracción OAS-ICIR
                sig = 0.10 * a1 + rng.standard_normal(n_assets) * 2.5
            family_preds[fam][lo:hi] = sig * (0.01 * scale_miscalibration)

        # Cuantiles Stage-1 ligeramente sub-cubiertos (0.65x sigma) para que CQR-NPID los corrija
        base_med = family_preds[families[0]][lo:hi] / scale_miscalibration
        q9_raw[lo:hi] = base_med[:, None] + (0.65 * v_s[:, None]) * z_quantiles[None, :]

    return Stage2SplitData(
        family_preds=family_preds,
        target=target,
        bounds=bounds,
        prediction_at=pred_at,
        matures_at=mat_at,
        quantiles=q9_raw,
        volatility=vol,
        liquidity=liq,
        exposures=exposures,
        markets=np.full(n_total, "US", dtype=object),
        hopfield_energy=hop_e,
        router_entropy=r_ent,
        loop_energy=l_eng,
    )


def test_order_level_decoupler_preserves_rank_ic_and_fixes_level():
    """FM-7 / F7-b: Huber y LAD reducen drásticamente el MAE/RMSE con delta Rank IC == 0."""
    rng = np.random.default_rng(19)
    n_cal, n_eval = 1200, 800
    s_cal = rng.standard_normal(n_cal)
    y_cal = 0.006 * s_cal + 0.0005 + rng.standard_normal(n_cal) * 0.015
    s_eval = rng.standard_normal(n_eval)
    y_eval = 0.006 * s_eval + 0.0005 + rng.standard_normal(n_eval) * 0.015

    raw_ic = oa.rank_ic_vector(s_eval, y_eval)
    raw_mae = float(np.mean(np.abs(y_eval - s_eval)))

    for loss_kind in ("huber", "lad"):
        dec = OrderLevelDecoupler(loss=loss_kind, huber_delta=0.02).fit(s_cal, y_cal)
        assert dec.lambda_scale > 0.0
        assert dec.lambda_scale == pytest.approx(0.006, abs=0.003)

        p_eval = dec.transform(s_eval)
        dec_ic = oa.rank_ic_vector(p_eval, y_eval)
        dec_mae = float(np.mean(np.abs(y_eval - p_eval)))
        # Invarianza estricta de orden: Rank IC exactamente preservado
        assert dec_ic == pytest.approx(raw_ic, abs=1e-12)
        # Error de nivel dramáticamente reducido frente a la señal sin desacoplar
        assert dec_mae < raw_mae * 0.10


def test_stage2_pipeline_end_to_end_four_families_calibration_and_paired_gain():
    """Stage-2 completo (FM-7 + CQR-NPID Mondrian 80/90/95 + Venn-Abers + OAS-ICIR + P+/A5-A8)."""
    cal_split = _make_multi_family_split(
        n_sessions=24, n_assets=40, start_us=_BASE_2022_US, seed=101
    )
    eval_start = int(cal_split.matures_at.max()) + 2 * _DAY_US
    eval_split = _make_multi_family_split(
        n_sessions=24, n_assets=40, start_us=eval_start, seed=202
    )

    cfg = Stage2Config(
        n_vol_strata=5,
        conformal_nominals=(0.80, 0.90, 0.95),
        conformal_scale_mode="hybrid",
        blend_mode="oas_icir_rp",
        device="cpu",
    )
    pipe = Stage2PostTrainingPipeline(cfg)
    result = pipe.run(cal_split, eval_split)

    # 1) Desacoplamiento Orden+Nivel ajustado en calibración
    assert result["decoupler"]["lambda_scale"] > 0.0
    assert result["stage2_report"]["dim1_value"]["mae_bps"] < (
        result["stage1_report"]["dim1_value"]["mae_bps"] * 0.25
    )

    # 2) Calibración conforme Mondrian 80/90/95 no cruzada + mejora frente a cuantiles crudos
    conf = result["conformal_calibration"]
    for nom_key in ("cov_80", "cov_90", "cov_95"):
        assert nom_key in conf
        assert 0.50 <= conf[nom_key] <= 1.0
    assert conf["cov_80"] <= conf["cov_90"] + 0.02
    assert conf["cov_90"] <= conf["cov_95"] + 0.02
    q9_cal = result["calibrated_quantiles_9"]
    assert q9_cal.shape == (len(eval_split.target), 9)
    assert np.all(np.diff(q9_cal, axis=1) >= -1e-10)
    assert len(conf["decile_cov_80"]) >= 2

    # 3) Venn-Abers: [p0, p1] válidos con anchura epistémica >= 0
    va = result["venn_abers"]
    assert np.all((va["p0"] >= 0.0) & (va["p0"] <= va["p1"]) & (va["p1"] <= 1.0))
    assert np.all((va["p_minimax"] >= va["p0"] - 1e-12) & (va["p_minimax"] <= va["p1"] + 1e-12))
    assert np.all(va["uncertainty"] >= -1e-12)

    # 4) Combinación ortogonal 4 familias + cartera P+ neutral al dólar y acotada
    mean_w = result["blending"]["mean_weights"]
    assert set(mean_w.keys()) == set(FOUR_FAMILIES)
    assert sum(mean_w.values()) == pytest.approx(1.0, abs=1e-6)
    # La 4a familia ruidosa recibe menor peso que la media de las 3 familias fuertes
    strong_avg = np.mean([mean_w[f] for f in FOUR_FAMILIES[:3]])
    assert mean_w[FOUR_FAMILIES[3]] < strong_avg

    weights = result["portfolio_weights"]
    assert weights.shape == (len(eval_split.target),)
    assert np.isfinite(weights).all()
    for lo, hi in eval_split.bounds:
        wt = weights[lo:hi]
        assert abs(float(np.sum(wt))) <= 1e-3
        assert float(np.sum(np.abs(wt))) <= cfg.gross_limit + 1e-3

    # 5) Informe 6D completo + contraste pareado Stage-2 vs Stage-1
    for rep_key in ("stage1_report", "stage2_report"):
        rep = result[rep_key]
        assert set(rep.keys()) == {
            "dim1_value",
            "dim2_ranking",
            "dim3_trend",
            "dim4_uncertainty",
            "dim5_error_autopsy",
            "dim6_economics",
        }
        assert np.isfinite(rep["dim2_ranking"]["rank_ic_mean"])
        assert np.isfinite(rep["dim4_uncertainty"]["crps"])
        assert np.isfinite(rep["dim6_economics"]["net_sharpe_1x"])

    contrast = result["paired_gain_contrast"]
    for req_field in (
        "stage1_reference_family",
        "rank_ic_t_nw_paired",
        "net_return_1x_t_nw_paired",
        "delta_rank_ic",
        "delta_pure_alpha_ic",
        "delta_crps",
        "crps_improvement",
        "delta_net_sharpe_1x",
        "delta_net_sharpe_2x",
        "delta_net_sharpe_3x",
        "delta_max_drawdown_1x",
        "delta_capacity_musd",
    ):
        assert req_field in contrast
    # El blend ortogonal 4-familias + desacoplamiento mejora CRPS y Rank IC frente a Stage-1
    assert contrast["crps_improvement"] > 0.0
    assert contrast["delta_rank_ic"] > 0.0


def test_stage2_pipeline_strict_causality_and_2024_seal():
    """Rechaza cualquier marca `>= 2024` y garantiza 0 bits de fuga desde sesiones futuras."""
    cal_split = _make_multi_family_split(
        n_sessions=12, n_assets=30, start_us=_BASE_2022_US, seed=7
    )
    eval_start = int(cal_split.matures_at.max()) + 2 * _DAY_US
    eval_split = _make_multi_family_split(
        n_sessions=12, n_assets=30, start_us=eval_start, seed=8
    )
    pipe = Stage2PostTrainingPipeline(Stage2Config(n_vol_strata=3, device="cpu"))

    # 1) Sello 2024+: prediction_at >= FINAL_TEST_START_US -> SealedTestViolationError
    bad_pred_at = eval_split.prediction_at.copy()
    bad_pred_at[-30:] = FINAL_TEST_START_US
    bad_mat_at = eval_split.matures_at.copy()
    bad_mat_at[-30:] = FINAL_TEST_START_US + _DAY_US
    sealed_eval = Stage2SplitData(
        family_preds=eval_split.family_preds,
        target=eval_split.target,
        bounds=eval_split.bounds,
        prediction_at=bad_pred_at,
        matures_at=bad_mat_at,
        quantiles=eval_split.quantiles,
        volatility=eval_split.volatility,
        exposures=eval_split.exposures,
        markets=eval_split.markets,
    )
    with pytest.raises(SealedTestViolationError):
        pipe.run(cal_split, sealed_eval)

    # 2) session_years con 2024 -> SealedTestViolationError
    sealed_years_eval = Stage2SplitData(
        family_preds=eval_split.family_preds,
        target=eval_split.target,
        bounds=eval_split.bounds,
        prediction_at=eval_split.prediction_at,
        matures_at=eval_split.matures_at,
        quantiles=eval_split.quantiles,
        volatility=eval_split.volatility,
        session_years=np.array([2023] * 11 + [2024], dtype=np.int64),
    )
    with pytest.raises(SealedTestViolationError):
        pipe.run(cal_split, sealed_years_eval)

    # 3) Violación de frontera temporal cal_split -> eval_split -> ValueError
    overlap_eval = _make_multi_family_split(
        n_sessions=8, n_assets=30, start_us=int(cal_split.prediction_at.min()), seed=9
    )
    with pytest.raises(ValueError, match="cronol"):
        pipe.run(cal_split, overlap_eval)

    # 4) Causalidad bit-a-bit: mutar sesiones futuras s > 5 en eval_split no cambia nada en s <= 5
    res_orig = pipe.run(cal_split, eval_split)
    cutoff_row = eval_split.bounds[5][1]

    mut_preds = {k: v.copy() for k, v in eval_split.family_preds.items()}
    for k in mut_preds:
        mut_preds[k][cutoff_row:] *= -5.0
    mut_target = eval_split.target.copy()
    mut_target[cutoff_row:] = -mut_target[cutoff_row:] * 10.0
    mut_q = eval_split.quantiles.copy()
    mut_q[cutoff_row:] += 0.5
    mut_eval = Stage2SplitData(
        family_preds=mut_preds,
        target=mut_target,
        bounds=eval_split.bounds,
        prediction_at=eval_split.prediction_at,
        matures_at=eval_split.matures_at,
        quantiles=mut_q,
        volatility=eval_split.volatility,
        liquidity=eval_split.liquidity,
        exposures=eval_split.exposures,
        markets=eval_split.markets,
        hopfield_energy=eval_split.hopfield_energy,
        router_entropy=eval_split.router_entropy,
        loop_energy=eval_split.loop_energy,
    )
    res_mut = pipe.run(cal_split, mut_eval)

    np.testing.assert_array_equal(
        res_orig["stage2_predictions"][:cutoff_row],
        res_mut["stage2_predictions"][:cutoff_row],
    )
    np.testing.assert_array_equal(
        res_orig["calibrated_quantiles_9"][:cutoff_row],
        res_mut["calibrated_quantiles_9"][:cutoff_row],
    )
    np.testing.assert_array_equal(
        res_orig["venn_abers"]["p_minimax"][:cutoff_row],
        res_mut["venn_abers"]["p_minimax"][:cutoff_row],
    )
    np.testing.assert_array_equal(
        res_orig["portfolio_weights"][:cutoff_row],
        res_mut["portfolio_weights"][:cutoff_row],
    )


def test_tfm_trace_and_table_exporter_markdown_latex_and_trace_summary(tmp_path: Path):
    """Exporta las 4 tablas TFM en Markdown y LaTeX + resumen consolidado de trazas JSON."""
    # 1) Ejecutar Stage-2 real sobre 4 familias
    cal_split = _make_multi_family_split(
        n_sessions=16, n_assets=32, start_us=_BASE_2022_US, seed=31
    )
    eval_start = int(cal_split.matures_at.max()) + 2 * _DAY_US
    eval_split = _make_multi_family_split(
        n_sessions=16, n_assets=32, start_us=eval_start, seed=32
    )
    s2_out = Stage2PostTrainingPipeline(Stage2Config(n_vol_strata=4, device="cpu")).run(
        cal_split, eval_split
    )

    # 2) Generar `telemetry.jsonl` real con `ResearchTelemetryMonitor`
    tel_dir = tmp_path / "tel_mmarrs"
    torch.manual_seed(11)
    model = build_variation(
        "v13_mmarrs_titan_full",
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = ResearchTelemetryMonitor(
        model,
        TelemetryConfig(log_dir=str(tel_dir), log_every_steps=1, enable_tensorboard=False),
        optimizer=opt,
    )
    batch = dict(
        prices=torch.randn(10, 16),
        fundamentals=torch.randn(10, 8),
        macro=torch.randn(10, 6),
        news=torch.randn(10, 8),
        charts=torch.randn(10, 8),
        modality_mask=torch.ones(10, 5),
        physics_v4=torch.randn(10, 12),
        high_snr_context=torch.randn(10, 10),
        exposures=torch.randn(10, 4),
        rmt_loadings=torch.randn(10, 4),
    )
    for step in range(2):
        mon.begin_step(step)
        opt.zero_grad(set_to_none=True)
        out = model.forward_session(batch)
        loss = (out["pred_return"] ** 2).mean()
        loss.backward()
        mon.before_optimizer_step(step)
        opt.step()
        mon.end_step(
            step,
            loss_dict=dict(total=loss, ic=loss * 0.5),
            model_output=out,
            titans_update_diag=dict(alpha_min=0.06, alpha_mean=0.14, eta_max=0.88, theta_max=0.20),
            cognitive_core_diag=dict(
                tool_usage_distribution=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10],
                tool_entropy=1.68,
                pruned_context_ratio=0.25,
            ),
            extra_metrics=dict(three_factor_dopamine=0.12, replay_priority=0.85),
        )
    mon.close()

    # 3) Crear `experiment_memory.jsonl` con registros KEEP/REJECT/BUG_CAUGHT
    mem_path = tmp_path / "experiment_memory.jsonl"
    mem_records = [
        dict(
            experiment_id="exp_001",
            timestamp_utc="2026-11-23T10:00:00Z",
            family="mmarrs_titan",
            variation="v13_mmarrs_titan_full",
            market="US",
            hypothesis="Full multimodal MoE+loop beats matched controls",
            changes_made="Activated 5M+HRR+MoE+PC-TRM",
            metrics_6d=s2_out["family_stage1_reports"]["mmarrs_titan"],
            vs_controls_delta=dict(delta_rank_ic=0.012),
            verdict="KEEP",
            failure_mode_or_root_cause="",
            lessons_learned="Aux-free MoE stays balanced",
            next_hypothesis="Test Stage-2 conformal PID",
            hardware_env="RTX_PRO_6000_sm120",
            fingerprint_sha256="a" * 64,
        ),
        dict(
            experiment_id="exp_002",
            timestamp_utc="2026-11-23T10:15:00Z",
            family="holographic_brain",
            variation="hb_no_replay_control",
            market="US",
            hypothesis="Disable replay buffer",
            changes_made="use_replay=False",
            metrics_6d=s2_out["family_stage1_reports"]["holographic_brain"],
            vs_controls_delta=dict(delta_rank_ic=-0.004),
            verdict="REJECT",
            failure_mode_or_root_cause="Catastrophic drift across regime shifts",
            lessons_learned="Keep prioritized replay enabled",
            next_hypothesis="Enable 3-factor neuromodulation",
            hardware_env="RTX_PRO_6000_sm120",
            fingerprint_sha256="b" * 64,
        ),
    ]
    mem_path.write_text(
        "\n".join(json.dumps(r) for r in mem_records) + "\n", encoding="utf-8"
    )

    # 4) Veredictos de GainValidationGate (PASS y REJECT)
    gate = GainValidationGate(
        min_rank_ic_gain=0.001,
        min_pure_alpha_ic_gain=0.001,
        min_net_sharpe_1x_gain=0.05,
        min_dsr=0.55,
        min_newey_west_t=1.96,
    )
    ctrl_rep = dict(
        dim2_ranking=dict(rank_ic_mean=0.025, pure_alpha_ic_mean=0.020, rank_ic_t_nw=2.1),
        dim4_uncertainty=dict(crps=0.0125, cov_80=0.80),
        dim6_economics=dict(net_sharpe_1x=0.90, net_sharpe_2x=0.50, dsr=0.60),
    )
    controls_5 = {k: ctrl_rep for k in DEFAULT_MATCHED_CONTROLS}
    pass_rep = dict(
        dim2_ranking=dict(rank_ic_mean=0.042, pure_alpha_ic_mean=0.036, rank_ic_t_nw=3.4),
        dim4_uncertainty=dict(crps=0.0108, cov_80=0.81),
        dim6_economics=dict(net_sharpe_1x=1.55, net_sharpe_2x=1.10, dsr=0.88),
    )
    rej_rep = dict(
        dim2_ranking=dict(rank_ic_mean=0.020, pure_alpha_ic_mean=0.015, rank_ic_t_nw=1.2),
        dim4_uncertainty=dict(crps=0.0140, cov_80=0.64),
        dim6_economics=dict(net_sharpe_1x=0.60, net_sharpe_2x=0.10, dsr=0.40),
    )
    verdicts = {
        "v13_mmarrs_titan_full": gate.evaluate("v13_mmarrs_titan_full", pass_rep, controls_5),
        "v07_moe_double_loop": gate.evaluate("v07_moe_double_loop", rej_rep, controls_5),
    }

    exporter = TFMTraceAndTableExporter(
        family_evaluations=s2_out["family_stage1_reports"],
        gain_verdicts=verdicts,
        stage2_results={"US_2023_4fam": s2_out},
        telemetry_paths={"mmarrs_titan": mon.jsonl_path},
        experiment_memory_path=mem_path,
    )

    out_dir = tmp_path / "tfm_tables"
    md_tables = exporter.export_markdown_tables(out_dir=out_dir)
    tex_tables = exporter.export_latex_tables(out_dir=out_dir)
    summary = exporter.export_execution_trace_summary(output_path=out_dir / "trace_summary.json")

    expected_keys = (
        "master_four_family_comparison",
        "paired_control_gain_validation",
        "diagnostic_trace_summary",
        "stage1_vs_stage2_gain",
    )
    for key in expected_keys:
        assert key in md_tables
        assert key in tex_tables
        assert (out_dir / f"{key}.md").is_file()
        assert (out_dir / f"{key}.tex").is_file()
        # Markdown GFM válido
        assert "| ---" in md_tables[key]
        assert "nan" not in md_tables[key].lower()
        # LaTeX booktabs válido y sin guiones bajos sin escapar fuera de comandos
        tex = tex_tables[key]
        assert "\\begin{table}" in tex
        assert "\\begin{tabular}" in tex
        assert "\\toprule" in tex
        assert "\\midrule" in tex
        assert "\\bottomrule" in tex
        assert "\\end{table}" in tex
        assert "nan" not in tex.lower()

    # Verificar contenido específico de cada tabla
    for fam in FOUR_FAMILIES:
        assert fam in md_tables["master_four_family_comparison"]
        assert fam.replace("_", "\\_") in tex_tables["master_four_family_comparison"]
    assert "PASS" in md_tables["paired_control_gain_validation"]
    assert "REJECT" in md_tables["paired_control_gain_validation"]
    assert "mmarrs_titan" in md_tables["diagnostic_trace_summary"]
    assert "US_2023_4fam" in md_tables["stage1_vs_stage2_gain"]

    # Verificar resumen de ejecución consolidado
    assert (out_dir / "trace_summary.json").is_file()
    assert summary["telemetry_summary"]["mmarrs_titan"]["n_logged_steps"] == 2
    assert summary["telemetry_summary"]["mmarrs_titan"]["cognitive_tool_entropy_mean"] == (
        pytest.approx(1.68)
    )
    assert summary["experiment_memory_summary"]["total_experiments"] == 2
    assert summary["experiment_memory_summary"]["verdict_counts"]["KEEP"] == 1
    assert summary["experiment_memory_summary"]["verdict_counts"]["REJECT"] == 1
    assert "US_2023_4fam" in summary["stage2_summary"]


def test_single_family_stage2_and_exporter_without_optional_files(tmp_path: Path):
    """Soporta ejecución Stage-2 con 1 sola familia y exportador sin archivos opcionales."""
    cal_1 = _make_multi_family_split(
        n_sessions=10,
        n_assets=24,
        start_us=_BASE_2022_US,
        seed=51,
        families=("titans_cognitive_core",),
    )
    eval_start = int(cal_1.matures_at.max()) + 2 * _DAY_US
    eval_1 = _make_multi_family_split(
        n_sessions=10,
        n_assets=24,
        start_us=eval_start,
        seed=52,
        families=("titans_cognitive_core",),
    )
    res = Stage2PostTrainingPipeline(Stage2Config(n_vol_strata=3, device="cpu")).run(cal_1, eval_1)
    assert res["blending"]["mean_weights"] == {"titans_cognitive_core": pytest.approx(1.0)}
    assert np.isfinite(res["stage2_report"]["dim2_ranking"]["rank_ic_mean"])

    exporter = TFMTraceAndTableExporter(
        family_evaluations=res["family_stage1_reports"],
        stage2_results=res,
        experiment_memory_path=tmp_path / "does_not_exist.jsonl",
    )
    md = exporter.export_markdown_tables()
    tex = exporter.export_latex_tables()
    summary = exporter.export_execution_trace_summary()
    assert "titans_cognitive_core" in md["master_four_family_comparison"]
    assert "\\toprule" in tex["stage1_vs_stage2_gain"]
    assert summary["experiment_memory_summary"]["total_experiments"] == 0


def test_compressed_lossless_jsonl_gz_trace_bundle_and_ragged_large_n_stage2(tmp_path: Path):
    """Exporta trazas comprimidas `.jsonl.gz` sin pérdidas y soporta paneles irregulares."""
    # 1) Panel irregular con altas/bajas (asset_ids cambiantes) y N=600 en eval para verificar P+
    def _ragged_split(n_sessions: int, start_us: int, base_n: int, seed_offset: int):
        r = np.random.default_rng(707 + seed_offset)
        counts = [base_n + (i % 3) * 4 for i in range(n_sessions)]
        n_tot = sum(counts)
        bounds = []
        cur = 0
        for c in counts:
            bounds.append((cur, cur + c))
            cur += c
        pred_at = np.empty(n_tot, dtype=np.int64)
        mat_at = np.empty(n_tot, dtype=np.int64)
        aids = np.empty(n_tot, dtype=np.int64)
        vol = np.full(n_tot, 0.02, dtype=np.float64)
        exp = r.standard_normal((n_tot, 6)).astype(np.float64)
        y = np.empty(n_tot, dtype=np.float64)
        fams = {f: np.empty(n_tot, dtype=np.float64) for f in ("titans_original", "mmarrs_titan")}
        for s_i, (lo, hi) in enumerate(bounds):
            n_s = hi - lo
            pred_at[lo:hi] = start_us + s_i * 2 * _DAY_US
            mat_at[lo:hi] = start_us + s_i * 2 * _DAY_US + _DAY_US
            # 2 activos salen y 2 entran entre sesiones consecutivas (panel ragged)
            aids[lo:hi] = np.arange(s_i * 2, s_i * 2 + n_s, dtype=np.int64)
            sig = r.standard_normal(n_s)
            y[lo:hi] = 0.008 * sig + r.standard_normal(n_s) * 0.02
            fams["titans_original"][lo:hi] = sig + r.standard_normal(n_s) * 0.8
            fams["mmarrs_titan"][lo:hi] = sig + r.standard_normal(n_s) * 0.5
        return Stage2SplitData(
            family_preds=fams,
            target=y,
            bounds=bounds,
            prediction_at=pred_at,
            matures_at=mat_at,
            volatility=vol,
            exposures=exp,
            asset_ids=aids,
        )

    cal_r = _ragged_split(6, _BASE_2022_US, base_n=80, seed_offset=0)
    ev_start = int(cal_r.matures_at.max()) + 2 * _DAY_US
    # N_s = 600 prueba invarianza O(1) de Lipschitz en P+ (sin congelar ||w||_1 ~ 0!)
    ev_r = _ragged_split(4, ev_start, base_n=600, seed_offset=10)
    res = Stage2PostTrainingPipeline(Stage2Config(n_vol_strata=3, device="cpu")).run(cal_r, ev_r)
    assert len(res["session_traces"]) == 4
    # Verificar que la cartera P+ a escala N=600 NO está congelada (||w||_1 significativo)
    l1_norms = [float(np.sum(np.abs(res["portfolio_weights"][lo:hi]))) for lo, hi in ev_r.bounds]
    assert min(l1_norms) > 0.25
    assert res["asset_aligned_turnover_mean"] > 0.0

    # 2) Paquete comprimido `.jsonl.gz` sin pérdidas + lectura transparente de `.jsonl.gz`
    bundle_path = tmp_path / "stage2_traces.jsonl.gz"
    summary_gz = tmp_path / "trace_summary.json.gz"
    exporter = TFMTraceAndTableExporter(
        family_evaluations=res["family_stage1_reports"],
        stage2_results={"US_ragged": res},
    )
    meta = exporter.export_compressed_trace_bundle(bundle_path)
    assert bundle_path.is_file()
    assert meta["compressed"] is True
    assert meta["compressed_bytes"] < meta["uncompressed_bytes"]
    assert meta["compression_ratio"] > 1.5
    assert meta["record_counts"]["stage2_session_trace"] == 4
    assert meta["record_counts"]["family_evaluation_6d"] == 2

    loaded = read_compressed_trace_bundle(bundle_path)
    assert len(loaded) == meta["n_records"]
    assert loaded[0]["record_type"] == "bundle_header"
    sess_recs = [r for r in loaded if r["record_type"] == "stage2_session_trace"]
    assert [r["session_idx"] for r in sess_recs] == [0, 1, 2, 3]

    # También comprobar escritura comprimida `.json.gz` y lectura transparente de `.jsonl.gz`
    exporter.export_execution_trace_summary(output_path=summary_gz)
    assert summary_gz.is_file()
    with gzip.open(summary_gz, "rt", encoding="utf-8") as fh:
        loaded_sum = json.load(fh)
    assert "US_ragged" in loaded_sum["stage2_summary"]

    # Verificar lectura transparente de experiment_memory.jsonl.gz comprimido
    exp_gz = tmp_path / "experiment_memory.jsonl.gz"
    with gzip.open(exp_gz, "wt", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                dict(
                    experiment_id="exp_gz_1",
                    family="mmarrs_titan",
                    variation="v13",
                    verdict="KEEP",
                    failure_mode_or_root_cause="",
                ),
                separators=(",", ":"),
            )
            + "\n"
        )
    exp_loader = TFMTraceAndTableExporter(experiment_memory_path=exp_gz)
    assert exp_loader.export_execution_trace_summary()["experiment_memory_summary"][
        "total_experiments"
    ] == 1


def test_p0_4_stage2_forwards_asset_ids_and_cn_market_friction_to_both_stage_reports():
    """P0-4: `asset_ids` y `market='CN'` llegan a informes 6D Stage-1 y Stage-2."""
    def _make_cn_ragged_split(n_sessions: int, start_us: int, seed_offset: int, lock_first: bool):
        r = np.random.default_rng(909 + seed_offset)
        counts = [36 + (i % 3) * 4 for i in range(n_sessions)]
        n_tot = sum(counts)
        bounds: list[tuple[int, int]] = []
        cur = 0
        for c in counts:
            bounds.append((cur, cur + c))
            cur += c
        pred_at = np.empty(n_tot, dtype=np.int64)
        mat_at = np.empty(n_tot, dtype=np.int64)
        aids = np.empty(n_tot, dtype=np.int64)
        vol = np.full(n_tot, 0.02, dtype=np.float64)
        liq = np.full(n_tot, 2e7, dtype=np.float64)
        exp = r.standard_normal((n_tot, 4)).astype(np.float64)
        y = np.empty(n_tot, dtype=np.float64)
        fams = {f: np.empty(n_tot, dtype=np.float64) for f in ("titans_original", "mmarrs_titan")}
        for s_i, (lo, hi) in enumerate(bounds):
            n_s = hi - lo
            pred_at[lo:hi] = start_us + s_i * 2 * _DAY_US
            mat_at[lo:hi] = start_us + s_i * 2 * _DAY_US + _DAY_US
            # 3 activos salen y entran en cada sesión para producir `exited_turnover_mean > 0`
            aids[lo:hi] = np.arange(s_i * 3, s_i * 3 + n_s, dtype=np.int64)
            sig = r.standard_normal(n_s)
            y[lo:hi] = 0.008 * sig + r.standard_normal(n_s) * 0.02
            fams["titans_original"][lo:hi] = sig + r.standard_normal(n_s) * 0.8
            fams["mmarrs_titan"][lo:hi] = sig + r.standard_normal(n_s) * 0.5
        if lock_first:
            y[0] = 0.105  # supera límite diario CN (+10% >= 0.098)
        return Stage2SplitData(
            family_preds=fams,
            target=y,
            bounds=bounds,
            prediction_at=pred_at,
            matures_at=mat_at,
            volatility=vol,
            liquidity=liq,
            exposures=exp,
            markets=np.full(n_tot, "CN", dtype=object),
            asset_ids=aids,
        )

    with pytest.raises(ValueError, match="market"):
        Stage2Config(market="US+CN")

    cal_s = _make_cn_ragged_split(8, _BASE_2022_US, seed_offset=0, lock_first=False)
    ev_start = int(cal_s.matures_at.max()) + 2 * _DAY_US
    ev_s = _make_cn_ragged_split(8, ev_start, seed_offset=10, lock_first=True)

    res_cn = Stage2PostTrainingPipeline(
        Stage2Config(market="CN", n_vol_strata=3, device="cpu")
    ).run(cal_s, ev_s)
    res_us = Stage2PostTrainingPipeline(
        Stage2Config(market="US", n_vol_strata=3, device="cpu")
    ).run(cal_s, ev_s)

    for rep_cn, rep_us in (
        (res_cn["stage1_report"], res_us["stage1_report"]),
        (res_cn["stage2_report"], res_us["stage2_report"]),
        (
            res_cn["family_stage1_reports"]["titans_original"],
            res_us["family_stage1_reports"]["titans_original"],
        ),
    ):
        d6_cn = rep_cn["dim6_economics"]
        d6_us = rep_us["dim6_economics"]
        assert d6_cn["market"] == "CN"
        assert d6_us["market"] == "US"
        assert float(d6_cn["cn_stamp_tax_bps_cost"]) > 0.0
        assert float(d6_cn["cn_limit_lock_rate"]) > 0.0
        assert float(d6_cn["linear_cost_bps"]) > float(d6_us["linear_cost_bps"])
        assert float(d6_cn["exited_turnover_mean"]) > 0.0

    # El turnover del informe Stage-2 coincide con `asset_aligned_turnover_mean`
    assert res_cn["stage2_report"]["dim6_economics"]["turnover_mean"] == pytest.approx(
        res_cn["asset_aligned_turnover_mean"], rel=1e-5
    )


def test_p1_5_fit_vol_strata_vectorized_exact_parity_including_ties_and_unseen_markets():
    """P1-5: `_fit_vol_strata` vectorizado iguala bit a bit la referencia (empates y mercados)."""
    rng = np.random.default_rng(1234)
    n_cal, n_eval = 2000, 20_000
    # Calibración con estratos dispersos por mercado para forzar empates y clamping
    vol_cal = np.concatenate(
        [
            np.exp(rng.uniform(np.log(0.005), np.log(0.012), size=400)),  # US bajos
            np.exp(rng.uniform(np.log(0.030), np.log(0.060), size=600)),  # US altos (hueco medio)
            np.exp(rng.uniform(np.log(0.010), np.log(0.040), size=1000)),  # CN
        ]
    )
    mkt_cal = np.array(["US"] * 1000 + ["CN"] * 1000, dtype=str)

    vol_eval = np.exp(rng.uniform(np.log(0.004), np.log(0.080), size=n_eval))
    mkt_choices = np.array(["US", "CN", "EU"], dtype=str)  # "EU" no visto -> fallback_mkt
    mkt_eval = mkt_choices[rng.integers(0, 3, size=n_eval)]

    g_cal, g_eval, s_clamped, edges = Stage2PostTrainingPipeline._fit_vol_strata(
        vol_cal, vol_eval, mkt_cal, mkt_eval, n_strata=10
    )

    # Referencia escalar exacta del bucle original
    probs = np.linspace(0.0, 1.0, 11)[1:-1]
    raw_edges = np.unique(np.quantile(vol_cal, probs))
    s_cal_ref = np.searchsorted(raw_edges, vol_cal, side="right").astype(np.int64)
    s_eval_ref = np.searchsorted(raw_edges, vol_eval, side="right").astype(np.int64)
    valid_by_mkt = {
        m: np.sort(np.unique(s_cal_ref[mkt_cal == m])) for m in sorted(set(mkt_cal.tolist()))
    }
    fallback = sorted(valid_by_mkt)[0]
    ref_clamped = np.empty(n_eval, dtype=np.int64)
    ref_g = np.empty(n_eval, dtype=object)
    for i in range(n_eval):
        m = str(mkt_eval[i])
        use_m = m if m in valid_by_mkt else fallback
        allowed = valid_by_mkt[use_m]
        nearest = int(allowed[int(np.argmin(np.abs(allowed - s_eval_ref[i])))])
        ref_clamped[i] = nearest
        ref_g[i] = f"{use_m}|s{nearest}"

    assert edges == pytest.approx([float(x) for x in raw_edges.tolist()], abs=1e-12)
    np.testing.assert_array_equal(s_clamped, ref_clamped)
    np.testing.assert_array_equal(g_eval, ref_g.astype(str))
    assert len(g_cal) == n_cal
    import inspect

    assert "range(len(vol_eval))" not in inspect.getsource(
        Stage2PostTrainingPipeline._fit_vol_strata
    )


def test_wave3c_cross_family_disagreement_stack_and_stage2_kappa_control():
    """#195 #5C: Cross-family epistemic disagreement risk into P+ with bit-exact kappa=0 control."""
    from mars_titan.evaluation.post_training_suite_sota import (
        run_cross_family_disagreement_stack,
    )

    # 1) Direct helper `run_cross_family_disagreement_stack`: kappa=0 vs kappa>0 + CKA weights
    rng = np.random.default_rng(195)
    n_assets = 40
    base_sig = rng.standard_normal(n_assets)
    # First 10 assets have EXTREME cross-family disagreement; last 30 have consensus!
    f1 = base_sig.copy()
    f2 = base_sig.copy()
    f3 = base_sig.copy()
    f4 = base_sig.copy()
    f1[:10] += 3.0
    f2[:10] -= 3.0
    f3[:10] += 2.5
    f4[:10] -= 2.5
    fam_preds = dict(
        titans_original=f1,
        holographic_brain=f2,
        mmarrs_titan=f3,
        titans_cognitive_core=f4,
    )
    vol = np.full(n_assets, 0.02, dtype=np.float64)
    exp = rng.standard_normal((n_assets, 4))

    res_k0 = run_cross_family_disagreement_stack(
        fam_preds, realized_vol=vol, exposures=exp, disagreement_kappa=0.0
    )
    res_k2 = run_cross_family_disagreement_stack(
        fam_preds, realized_vol=vol, exposures=exp, disagreement_kappa=2.0
    )
    np.testing.assert_allclose(res_k0["sigma_eff"], vol, atol=1e-7, rtol=1e-7)
    assert float(np.mean(res_k2["sigma_eff"][:10])) > float(np.mean(res_k2["sigma_eff"][10:])) * 1.2
    # High-disagreement assets get strictly shrunk |w_i| under kappa=2.0 vs kappa=0.0
    assert float(np.mean(np.abs(res_k2["portfolio_weights"][:10]))) < float(
        np.mean(np.abs(res_k0["portfolio_weights"][:10]))
    )

    # Redundant families in CKA matrix get downweighted cleanly
    cka_redundant = np.array(
        [
            [1.0, 0.99, 0.1, 0.1],
            [0.99, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ],
        dtype=np.float64,
    )
    res_cka = run_cross_family_disagreement_stack(
        fam_preds, realized_vol=vol, exposures=exp, disagreement_kappa=1.0, cka_matrix=cka_redundant
    )
    w_cka = np.asarray(res_cka["family_weights"], dtype=np.float64)
    assert w_cka[2] > w_cka[0] and w_cka[3] > w_cka[1]

    # 2) Integrated in Stage2PostTrainingPipeline: kappa=0.0 is BIT-EXACT identical to default!
    cal_split = _make_multi_family_split(
        n_sessions=10, n_assets=32, start_us=_BASE_2022_US, seed=61
    )
    ev_start = int(cal_split.matures_at.max()) + 2 * _DAY_US
    eval_split = _make_multi_family_split(
        n_sessions=10, n_assets=32, start_us=ev_start, seed=62
    )
    res_def = Stage2PostTrainingPipeline(Stage2Config(n_vol_strata=3, device="cpu")).run(
        cal_split, eval_split
    )
    res_zero = Stage2PostTrainingPipeline(
        Stage2Config(n_vol_strata=3, device="cpu", disagreement_kappa=0.0)
    ).run(cal_split, eval_split)
    np.testing.assert_array_equal(res_def["portfolio_weights"], res_zero["portfolio_weights"])

    res_pos = Stage2PostTrainingPipeline(
        Stage2Config(
            n_vol_strata=3,
            device="cpu",
            disagreement_kappa=2.0,
            cka_weights=(0.25, 0.25, 0.25, 0.25),
        )
    ).run(cal_split, eval_split)
    assert "epistemic_disagreement" in res_pos
    assert float(res_pos["epistemic_disagreement"]["mean_sigma_inflation_ratio"]) > 1.0
    assert not np.allclose(res_zero["portfolio_weights"], res_pos["portfolio_weights"], atol=1e-5)
    assert np.isfinite(res_pos["stage2_report"]["dim6_economics"]["net_sharpe_1x"])

    # 3) #233 P1-3: 2D multi-horizon [F, N, H] predictions + horizon_idx slicing & zero-vol safe!
    fam_2d = {
        k: np.stack([v, v * 0.8, v * -0.5], axis=-1) for k, v in fam_preds.items()
    }
    res_h0 = run_cross_family_disagreement_stack(
        fam_2d, realized_vol=vol, exposures=exp, disagreement_kappa=1.0, horizon_idx=0
    )
    res_h2 = run_cross_family_disagreement_stack(
        fam_2d, realized_vol=vol, exposures=exp, disagreement_kappa=1.0, horizon_idx=2
    )
    assert res_h0["portfolio_weights"].shape == (n_assets,)
    assert res_h2["portfolio_weights"].shape == (n_assets,)
    assert np.isfinite(res_h0["portfolio_weights"]).all()
    assert not np.allclose(res_h0["portfolio_weights"], res_h2["portfolio_weights"], atol=1e-4)

    # All-zero realized_vol falls back to 0.02 cleanly without emitting RuntimeWarning
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_zv = run_cross_family_disagreement_stack(
            fam_preds, realized_vol=np.zeros(n_assets), disagreement_kappa=1.0
        )
    assert np.isfinite(res_zv["portfolio_weights"]).all()
    assert not any("Mean of empty slice" in str(w.message) for w in caught)



