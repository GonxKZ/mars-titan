"""Pruebas TDD del motor SOTA de auto-ajuste y control óptimo de ajuste (`autotuner_sota.py`).

Cubre estrictamente:
  1. `SearchSpaceSpec`: 9 hiperparámetros canónicos MMARRS/Titans y bloqueo duro de datos `2024+`.
  2. `TPEHyperbandAutoTuner`: búsqueda TPE determinista sin Optuna + poda ASHA + auditoría completa
     (incluyendo ensayos podados y fallidos) + recorte de multiplicidad y DSR (`granular_eval`).
  3. Detección de acantilados de sensibilidad (`check_parameter_sensitivity_cliff` y selección
     robusta de meseta frente a picos frágiles).
  4. `OptimalFitController`: diagnóstico dual (`underfitting` / `healthy` / `overfitting`), parada
     temprana suavizada (`smoothing=3`, `patience=5`, `lr_decay=0.5x` hasta 2 veces) y promedio
     exacto de pesos Top-3 (Model Soup).
  5. `shrink_and_perturb` (Ash & Adams, NeurIPS): contracción de norma, finitud y determinismo.
  6. `zeropower_via_newtonschulz5`, `MuonHybridOptimizer`, `ScheduleFreeAdamW` y
     `build_sota_optimizer`: ejecución real sobre `MMARRSTitan`, reducción de pérdida, finitud,
     seguridad `bfloat16` y auditoría AST de cero sincronizaciones host en `step()`.
  7. `rank_ensemble_predictions` y `evaluate_rank_ensemble`: mejora de Rank IC en 3 semillas.
"""

from __future__ import annotations

import ast
import copy
import inspect
import math

import numpy as np
import pytest
import torch
from torch import nn

from mars_titan.evaluation.granular_evaluator import deflated_sharpe_ratio
from mars_titan.evaluation.online_aggregator import rank_ic_vector
from mars_titan.models.mmarrs.autotuner_sota import (
    DEFAULT_SEARCH_PARAM_NAMES,
    SOTA_OPTIMIZER_TYPES,
    MuonHybridOptimizer,
    OptimalFitConfig,
    OptimalFitController,
    ParamSpec,
    ScheduleFreeAdamW,
    SearchSpaceSpec,
    TPEHyperbandAutoTuner,
    TrialContext,
    append_audit_record,
    build_sota_optimizer,
    check_parameter_sensitivity_cliff,
    evaluate_rank_ensemble,
    rank_ensemble_predictions,
    read_audit_ledger,
    shrink_and_perturb,
    validate_purged_validation_split,
    zeropower_via_newtonschulz5,
)
from mars_titan.models.mmarrs.granular_heads import MMARRSMultiTaskLoss
from mars_titan.models.mmarrs.hardware_blackwell import BlackwellRuntimeConfig
from mars_titan.models.mmarrs.long_running_harness import SealedTestViolationError
from mars_titan.models.mmarrs.model import build_variation


def _synthetic_session_batch(
    rng: np.random.Generator,
    n_assets: int = 20,
    *,
    prices_dim: int = 16,
    fundamentals_dim: int = 8,
    macro_dim: int = 6,
    news_dim: int = 8,
    charts_dim: int = 8,
    physics_dim: int = 12,
    context_dim: int = 10,
    exposures_dim: int = 4,
    rmt_k: int = 4,
) -> dict[str, torch.Tensor]:
    def _t(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(rng.standard_normal(shape), dtype=torch.float32)

    mask = torch.ones((n_assets, 5), dtype=torch.float32)
    mask[0::5, 3] = 0.0
    prices = _t((n_assets, prices_dim))
    physics = _t((n_assets, physics_dim))
    # Señal sintética limpia pero con ruido para verificar descenso real de pérdida
    y1 = 0.015 * prices[:, 0] + 0.010 * physics[:, 0] + 0.004 * _t((n_assets,))
    targets = torch.stack([y1 * math.sqrt(h) for h in (1, 2, 5, 10, 21, 63)], dim=-1)
    return dict(
        prices=prices,
        fundamentals=_t((n_assets, fundamentals_dim)),
        macro=_t((n_assets, macro_dim)),
        news=_t((n_assets, news_dim)),
        charts=_t((n_assets, charts_dim)),
        modality_mask=mask,
        physics_v4=physics,
        high_snr_context=_t((n_assets, context_dim)),
        exposures=_t((n_assets, exposures_dim)),
        rmt_loadings=_t((n_assets, rmt_k)),
        regime_vec=_t((context_dim,)),
        ach_signal=torch.abs(_t((n_assets, 1))) * 0.5,
        ne_signal=torch.tensor([0.1], dtype=torch.float32),
        target_returns=targets,
    )


def test_search_space_spec_defaults_and_sealed_2024_guard() -> None:
    """El espacio canónico cubre los 9 hiperparámetros MMARRS y bloquea `2024+`."""
    space = SearchSpaceSpec.default_mmarrs_space()
    expected = (
        "lr",
        "weight_decay",
        "hidden_dim",
        "router_lr_scale",
        "modality_dropout",
        "ema_decay",
        "w_crps",
        "w_sharpe",
        "w_energy",
    )
    assert DEFAULT_SEARCH_PARAM_NAMES == expected
    assert space.param_names == expected

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    p_a = space.sample_random(rng_a)
    p_b = space.sample_random(rng_b)
    assert p_a == p_b
    assert set(p_a.keys()) == set(expected)
    assert 1e-4 <= p_a["lr"] <= 5e-3
    assert p_a["hidden_dim"] in (64, 96, 128, 192, 256)
    assert 0.0 <= p_a["modality_dropout"] <= 0.35

    # Partición purgada válida (2000-2022 train, 2023 val) pasa sin errores
    validate_purged_validation_split(
        train_years=[2000, 2015, 2022],
        val_years=[2023, 2023],
        sealed_test_start_year=2024,
    )
    # Cualquier año >= 2024 en validación o entrenamiento lanza SealedTestViolationError
    with pytest.raises(SealedTestViolationError):
        validate_purged_validation_split(
            train_years=[2000, 2022],
            val_years=[2023, 2024],
            sealed_test_start_year=2024,
        )
    with pytest.raises(SealedTestViolationError):
        validate_purged_validation_split(
            train_years=[2000, 2024],
            val_years=[2023],
            sealed_test_start_year=2024,
        )
    # Fuga temporal (train >= val) lanza ValueError
    with pytest.raises(ValueError, match="orden cronológico|temporal"):
        validate_purged_validation_split(
            train_years=[2000, 2023],
            val_years=[2023],
            sealed_test_start_year=2024,
        )


def test_tpe_hyperband_autotuner_finds_near_optimum_prunes_and_audits_all_trials(
    tmp_path,
) -> None:
    """TPE+ASHA converge cerca del óptimo ruidoso, poda ensayos, audita todos y aplica DSR."""
    space = SearchSpaceSpec(
        params=(
            ParamSpec("lr", "log_float", low=1e-4, high=1e-2),
            ParamSpec("weight_decay", "log_float", low=1e-5, high=1e-2),
            ParamSpec("modality_dropout", "float", low=0.0, high=0.30),
            ParamSpec("hidden_dim", "categorical", choices=(32, 64, 128)),
        )
    )
    opt_log_lr = math.log(1.2e-3)
    opt_log_wd = math.log(3e-4)
    opt_drop = 0.12

    def noisy_objective(params: dict[str, object], ctx: TrialContext) -> dict[str, object]:
        # Inyectar un fallo deliberado en trial 4 para verificar auditoría honesta de fallos
        if ctx.trial_id == 4:
            raise RuntimeError("synthetic OOM failure")
        t_rng = np.random.default_rng(10_000 + ctx.trial_id)
        d_lr = (math.log(float(params["lr"])) - opt_log_lr) / 1.4
        d_wd = (math.log(float(params["weight_decay"])) - opt_log_wd) / 1.8
        d_dp = (float(params["modality_dropout"]) - opt_drop) / 0.12
        cat_pen = 0.0 if params["hidden_dim"] == 64 else 0.18
        base_signal = 0.85 - 0.35 * (d_lr**2 + d_wd**2 + d_dp**2) - cat_pen

        score = -1.0
        for rung in ctx.rungs:
            # Ruido decreciente con el presupuesto (recurso = épocas)
            noise = float(t_rng.normal(0.0, 0.04 / math.sqrt(rung)))
            # Curva creciente con recurso para que ASHA discrimine temprano
            score = base_signal * (0.7 + 0.3 * (rung / ctx.max_resource)) + noise
            ctx.check_prune(score, resource=rung)

        # Serie sintética de retornos diarios de validación coherente con score
        daily_mu = max(score, -0.2) * 0.0022
        rets = daily_mu + t_rng.normal(0.0, 0.01, size=252)
        return dict(score=score, returns=rets, metrics=dict(val_ic=score * 0.1))

    tuner_a = TPEHyperbandAutoTuner(
        space,
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        n_startup_trials=8,
        seed=42,
    )
    summary_a = tuner_a.optimize(
        noisy_objective,
        n_trials=32,
        train_years=[2020, 2021, 2022],
        val_years=[2023],
    )

    # Determinismo exacto con misma semilla
    tuner_b = TPEHyperbandAutoTuner(
        space,
        min_resource=1,
        max_resource=9,
        reduction_factor=3,
        n_startup_trials=8,
        seed=42,
    )
    summary_b = tuner_b.optimize(
        noisy_objective,
        n_trials=32,
        train_years=[2020, 2021, 2022],
        val_years=[2023],
    )
    assert summary_a["ledger_sha256"] == summary_b["ledger_sha256"]
    assert summary_a["best_params"] == summary_b["best_params"]

    # Auditoría 100% honesta: TODOS los 32 ensayos registrados (completados + podados + fallidos)
    assert summary_a["n_total_trials"] == 32
    assert len(tuner_a.ledger) == 32
    assert summary_a["n_failed"] == 1
    assert summary_a["n_pruned"] >= 4
    assert summary_a["n_completed"] + summary_a["n_pruned"] + summary_a["n_failed"] == 32

    # Convergencia cerca del óptimo real (máximo teórico ~0.85)
    assert float(summary_a["best_raw_score"]) > 0.72
    best_p = summary_a["best_params"]
    assert best_p["hidden_dim"] == 64

    # Control de multiplicidad: haircut estrictamente menor que bruto y DSR exacto de granular_eval
    assert float(summary_a["multiplicity_haircut_offset"]) > 0.0
    assert float(summary_a["haircut_score"]) < float(summary_a["best_raw_score"])
    dsr_rep = summary_a["dsr_report"]
    assert 0.0 <= float(dsr_rep["dsr"]) <= 1.0
    best_rec = tuner_a.get_trial(int(summary_a["best_trial_id"]))
    assert best_rec is not None and best_rec.returns is not None
    emp_sharpes = []
    for rec in tuner_a.ledger:
        if rec.returns is not None:
            r_arr = np.asarray(rec.returns, dtype=np.float64)
            r_arr = r_arr[np.isfinite(r_arr)]
            if len(r_arr) >= 5 and float(np.std(r_arr, ddof=1)) > 1e-12:
                emp_sharpes.append(float(np.mean(r_arr) / np.std(r_arr, ddof=1)))
    assert len(emp_sharpes) >= 2
    dsr_ref = deflated_sharpe_ratio(
        best_rec.returns, n_trials=32, trial_sharpes=emp_sharpes, trial_sharpes_annualized=False
    )
    dsr_single = deflated_sharpe_ratio(best_rec.returns, n_trials=1)
    assert float(dsr_rep["dsr"]) == pytest.approx(dsr_ref["dsr"], abs=1e-12)
    assert float(dsr_rep["sr_std_used"]) == pytest.approx(dsr_ref["sr_std_used"], abs=1e-12)
    assert float(dsr_rep["sr_threshold"]) > float(dsr_single["sr_threshold"])
    assert float(dsr_rep["dsr"]) <= float(dsr_single["dsr"])

    # Exportar y releer el ledger JSONL sin pérdida
    ledger_path = tmp_path / "autotuner_ledger.jsonl"
    tuner_a.write_audit_ledger(ledger_path)
    rows = read_audit_ledger(ledger_path)
    assert len(rows) == 32
    assert any(r["status"] == "failed" and r["trial_id"] == 4 for r in rows)
    assert any(r["status"] == "pruned" for r in rows)

    # Guard de sello 2024+ activo también en optimize()
    with pytest.raises(SealedTestViolationError):
        tuner_a.optimize(noisy_objective, n_trials=2, train_years=[2022], val_years=[2024])


def test_sensitivity_cliff_detection_flags_brittle_spike_and_prefers_plateau() -> None:
    """Detecta picos frágiles (acantilados) y selecciona la meseta robusta frente al pico."""
    space = SearchSpaceSpec(
        params=(
            ParamSpec("x", "float", low=0.0, high=1.0),
            ParamSpec("y", "float", low=0.0, high=1.0),
        )
    )

    def spike_vs_plateau(params: dict[str, object]) -> float:
        x = float(params["x"])
        y = float(params["y"])
        # Aguja muy estrecha (sobreajuste frágil) centrada en (0.20, 0.20) con pico 1.0
        spike = 1.0 * math.exp(-(((x - 0.20) ** 2 + (y - 0.20) ** 2) / (2.0 * 0.015**2)))
        # Meseta amplia y estable centrada en (0.70, 0.70) con altura 0.72
        plateau = 0.72 * math.exp(-(((x - 0.70) ** 2 + (y - 0.70) ** 2) / (2.0 * 0.28**2)))
        return max(spike, plateau)

    spike_report = check_parameter_sensitivity_cliff(
        space,
        {"x": 0.20, "y": 0.20},
        spike_vs_plateau,
        delta_frac=0.12,
    )
    plateau_report = check_parameter_sensitivity_cliff(
        space,
        {"x": 0.70, "y": 0.70},
        spike_vs_plateau,
        delta_frac=0.12,
    )

    assert spike_report["is_cliff"] is True
    assert set(spike_report["cliff_params"]) == {"x", "y"}
    assert float(spike_report["max_relative_drop"]) > 0.80

    assert plateau_report["is_cliff"] is False
    assert plateau_report["cliff_params"] == []
    assert float(plateau_report["robustness_fraction"]) >= 0.80

    # El selector robusto rechaza el pico frágil (1.0) y elige la meseta estable (0.72)
    tuner = TPEHyperbandAutoTuner(space, seed=7)
    p_spike = {"x": 0.20, "y": 0.20}
    p_plateau = {"x": 0.70, "y": 0.70}
    tuner.record_external_trial(p_spike, score=spike_vs_plateau(p_spike))
    tuner.record_external_trial(p_plateau, score=spike_vs_plateau(p_plateau))
    chosen = tuner.select_robust_best(spike_vs_plateau, top_k_candidates=2, delta_frac=0.12)
    assert chosen["selected_params"] == {"x": 0.70, "y": 0.70}
    assert chosen["selected_sensitivity"]["is_cliff"] is False
    assert chosen["rejected_cliff_trial_ids"] == [0]


def test_optimal_fit_controller_regimes_lr_decay_early_stop_and_exact_top3_soup() -> None:
    """Distingue underfit/healthy/overfit, decae LR 2x, para y promedia exactamente Top-3 pesos."""
    # Modelo simple con peso flotante y buffer entero para verificar Model Soup exacto
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    model.register_buffer("step_counter", torch.tensor(0, dtype=torch.int64))
    decay_params = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay_params = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW(
        [
            dict(params=decay_params, lr=1e-3, weight_decay=1e-4),
            dict(params=no_decay_params, lr=1e-3, weight_decay=0.0),
        ]
    )

    cfg = OptimalFitConfig(
        smoothing=3,
        patience=5,
        lr_decay_factor=0.5,
        max_lr_decays=2,
        top_k_soup=3,
        min_delta=1e-4,
    )
    ctrl = OptimalFitController(cfg)

    def _set_weights(val: float, counter: int) -> None:
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(val)
            model.step_counter.fill_(counter)

    # 1) Fase UNDERFITTING: IC train bajo + ratio Karpathy ||ΔW||/||W|| diminuto
    _set_weights(0.10, 1)
    d_under = ctrl.step(
        epoch=0,
        train_ic=0.002,
        val_ic=0.001,
        val_crps=0.040,
        train_loss=2.50,
        update_weight_ratio=1e-6,
        model=model,
        optimizer=opt,
    )
    assert d_under.regime == "underfitting"
    assert any("lr" in a or "shrink_and_perturb" in a for a in d_under.recommended_actions)
    assert not d_under.should_stop

    # 2) Fase HEALTHY: train_ic y val_ic crecen juntos, val_crps baja, update ratio ~1e-3
    healthy_seq = [
        (1, 0.025, 0.022, 0.036, 0.20),
        (2, 0.042, 0.038, 0.033, 0.30),
        (3, 0.060, 0.055, 0.030, 0.40),  # Top-2 (val_ic=0.055, w=0.40)
        (4, 0.072, 0.068, 0.028, 0.50),  # Top-1 (val_ic=0.068, w=0.50, counter=4)
        (5, 0.075, 0.062, 0.028, 0.60),  # Top-3 (val_ic=0.062, w=0.60)
    ]
    for ep, tr_ic, va_ic, crps, w_val in healthy_seq:
        _set_weights(w_val, ep)
        dec = ctrl.step(
            epoch=ep,
            train_ic=tr_ic,
            val_ic=va_ic,
            val_crps=crps,
            train_loss=1.5 - 0.1 * ep,
            update_weight_ratio=1.2e-3,
            model=model,
            optimizer=opt,
        )
        assert dec.regime == "healthy"
        assert not dec.should_stop

    # 3) Fase OVERFITTING sostenido: brecha train-val se abre y val_crps sube -> 2 decays + stop
    lr_history: list[float] = [float(opt.param_groups[0]["lr"])]
    overfit_decisions = []
    for offset in range(1, 20):
        ep = 5 + offset
        _set_weights(0.90 + 0.01 * offset, ep)
        dec = ctrl.step(
            epoch=ep,
            train_ic=0.10 + 0.01 * offset,
            val_ic=max(0.005, 0.045 - 0.003 * offset),
            val_crps=0.031 + 0.002 * offset,
            train_loss=0.8 - 0.02 * offset,
            update_weight_ratio=2e-3,
            model=model,
            optimizer=opt,
        )
        overfit_decisions.append(dec)
        lr_history.append(float(opt.param_groups[0]["lr"]))
        if dec.should_stop:
            break

    assert any(d.regime == "overfitting" for d in overfit_decisions)
    assert overfit_decisions[-1].should_stop is True
    assert overfit_decisions[-1].lr_decays_used == 2
    assert overfit_decisions[-1].soup_restored is True
    # Verificar 2 decaimientos exactos 0.5x: 1e-3 -> 5e-4 -> 2.5e-4
    assert min(lr_history) == pytest.approx(2.5e-4, rel=1e-6)
    # Weight decay subió en grupo con decay, pero grupo no-decay sigue estrictamente en 0.0
    assert opt.param_groups[0]["weight_decay"] > 1e-4
    assert opt.param_groups[1]["weight_decay"] == 0.0

    # 4) Verificar promedio EXACTO de pesos Top-3: épocas 4 (0.50), 5 (0.60), 3 (0.40) -> media 0.50
    expected_w = (0.50 + 0.60 + 0.40) / 3.0
    soup_sd = ctrl.build_model_soup_state_dict()
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param.detach(), torch.full_like(param, expected_w), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            soup_sd[name], torch.full_like(param, expected_w), atol=1e-6, rtol=1e-6
        )
    # Buffer entero proviene exactamente de la mejor época (#1 = época 4) sin corromper dtype
    assert model.step_counter.dtype == torch.int64
    assert int(model.step_counter.item()) == 4


def test_shrink_and_perturb_reduces_norm_finite_and_deterministic() -> None:
    """`shrink_and_perturb` reduce norma L2 de pesos, mantiene finitud y es 100% determinista."""
    torch.manual_seed(17)
    m1 = build_variation(
        "v07_moe_double_loop",
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=24,
    )
    m2 = copy.deepcopy(m1)
    m3 = copy.deepcopy(m1)

    rep1 = shrink_and_perturb(m1, shrink=0.8, perturb=0.02, seed=2026)
    rep2 = shrink_and_perturb(m2, shrink=0.8, perturb=0.02, seed=2026)
    rep3 = shrink_and_perturb(m3, shrink=0.8, perturb=0.02, seed=2027)

    assert rep1["all_finite"] is True
    assert rep2["all_finite"] is True
    assert rep3["all_finite"] is True
    assert rep1["norm_after"] < rep1["norm_before"]
    assert 0.70 < rep1["norm_ratio"] < 0.95
    assert rep1["n_tensors_updated"] > 0

    for (n1, p1), (_, p2), (_, p3) in zip(
        m1.named_parameters(), m2.named_parameters(), m3.named_parameters(), strict=True
    ):
        assert torch.isfinite(p1).all(), f"parámetro no finito: {n1}"
        torch.testing.assert_close(p1, p2, atol=0.0, rtol=0.0)
        if p1.ndim >= 2:
            assert not torch.allclose(p1, p3, atol=1e-7)


def test_newton_schulz_orthogonalization_properties_and_bf16_safety() -> None:
    """Newton-Schulz de 5 pasos produce matrices aproximadamente ortogonales y es seguro en BF16."""
    torch.manual_seed(99)
    for shape in ((24, 24), (48, 16), (16, 48)):
        g = torch.randn(shape, dtype=torch.float32)
        u = zeropower_via_newtonschulz5(g, steps=5)
        assert u.shape == shape
        assert u.dtype == torch.float32
        assert torch.isfinite(u).all()
        # Verificar aproximación a ortogonalidad: valores singulares cercanos a 1
        svals = torch.linalg.svdvals(u)
        assert float(svals.min()) > 0.55
        assert float(svals.max()) < 1.35

    # Matriz nula no produce NaN
    z = torch.zeros((16, 16), dtype=torch.float32)
    uz = zeropower_via_newtonschulz5(z, steps=5)
    assert torch.isfinite(uz).all()
    assert float(uz.abs().max()) == 0.0

    # Seguridad en bfloat16 (RTX PRO 6000 bf16_amp)
    g_bf16 = torch.randn((32, 16), dtype=torch.bfloat16)
    u_bf16 = zeropower_via_newtonschulz5(g_bf16, steps=5)
    assert u_bf16.dtype == torch.bfloat16
    assert torch.isfinite(u_bf16).all()


def test_muon_hybrid_optimizer_on_real_mmarrs_titan_and_sota_factory() -> None:
    """`MuonHybridOptimizer` y `build_sota_optimizer` entrenan `MMARRSTitan` sin NaN ni syncs."""
    assert set(SOTA_OPTIMIZER_TYPES) == {"adamw_fused", "muon_hybrid", "schedule_free_adamw"}

    torch.manual_seed(42)
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
        hidden_dim=24,
    )
    rt_cfg = BlackwellRuntimeConfig(precision_mode="fp32_strict", router_lr_scale=1.0 / 3.0)

    # 1) Verificar particionado exacto en MuonHybridOptimizer (2D ocultos -> Muon, routers -> 1/3)
    muon_opt = build_sota_optimizer(
        model,
        optimizer_type="muon_hybrid",
        lr=3e-3,
        weight_decay=1e-4,
        router_lr_scale=1.0 / 3.0,
        cfg=rt_cfg,
    )
    assert isinstance(muon_opt, MuonHybridOptimizer)
    muon_groups = [g for g in muon_opt.param_groups if g.get("use_muon")]
    router_groups = [g for g in muon_opt.param_groups if "router" in str(g.get("group_role", ""))]
    adamw_groups = [g for g in muon_opt.param_groups if not g.get("use_muon")]
    assert len(muon_groups) >= 1
    assert len(router_groups) >= 1
    assert len(adamw_groups) >= 1
    for g in muon_groups:
        assert all(p.ndim in (2, 3) for p in g["params"])
    for g in router_groups:
        assert g["lr"] == pytest.approx(1e-3, rel=1e-5)

    # 2) Entrenar `MMARRSTitan` real en sesiones sintéticas: la pérdida baja y todo queda finito
    rng = np.random.default_rng(321)
    sessions = [_synthetic_session_batch(rng, n_assets=20) for _ in range(3)]
    loss_fn = MMARRSMultiTaskLoss()

    def _eval_loss() -> float:
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for b in sessions:
                out = model.forward_session(b, state=None, enable_early_exit=False)
                ld = loss_fn(
                    step_outputs=out["loop_outputs"],
                    targets=dict(returns=b["target_returns"]),
                    exposures=b["exposures"],
                    energy_total=out["energy"]["total"],
                )
                tot += float(ld["total"].item())
        return tot / len(sessions)

    loss_before = _eval_loss()
    model.train()
    for _ in range(6):
        for b in sessions:
            muon_opt.zero_grad(set_to_none=True)
            out = model.forward_session(b, state=None, enable_early_exit=False)
            ld = loss_fn(
                step_outputs=out["loop_outputs"],
                targets=dict(returns=b["target_returns"]),
                exposures=b["exposures"],
                energy_total=out["energy"]["total"],
            )
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            muon_opt.step()
    loss_after = _eval_loss()

    assert math.isfinite(loss_before) and math.isfinite(loss_after)
    assert loss_after < loss_before
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"parámetro no finito tras Muon: {name}"

    # Roundtrip state_dict
    sd = muon_opt.state_dict()
    muon_opt.load_state_dict(sd)

    # 3) Verificar adamw_fused y schedule_free_adamw + cambio train()/eval() en ScheduleFreeAdamW
    adamw_opt = build_sota_optimizer(
        model, optimizer_type="adamw_fused", lr=3e-3, weight_decay=1e-4, router_lr_scale=1.0 / 3.0
    )
    assert isinstance(adamw_opt, torch.optim.AdamW)

    torch.manual_seed(11)
    toy = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
    x_toy = torch.randn(32, 8)
    y_toy = (x_toy[:, :1] * 0.5 - x_toy[:, 1:2] * 0.3)
    sf_opt = build_sota_optimizer(
        toy, optimizer_type="schedule_free_adamw", lr=2e-2, weight_decay=1e-4, sf_warmup_steps=2
    )
    assert isinstance(sf_opt, ScheduleFreeAdamW)
    sf_opt.train()
    init_loss = float(nn.functional.mse_loss(toy(x_toy), y_toy).item())
    for _ in range(25):
        sf_opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(toy(x_toy), y_toy)
        loss.backward()
        sf_opt.step()
    train_weights = [p.detach().clone() for p in toy.parameters()]
    sf_opt.eval()
    eval_weights = [p.detach().clone() for p in toy.parameters()]
    eval_loss = float(nn.functional.mse_loss(toy(x_toy), y_toy).item())
    assert eval_loss < init_loss * 0.4
    assert any(
        not torch.allclose(tw, ew) for tw, ew in zip(train_weights, eval_weights, strict=True)
    )
    sf_opt.train()
    for tw, p in zip(train_weights, toy.parameters(), strict=True):
        torch.testing.assert_close(tw, p.detach(), atol=1e-6, rtol=1e-6)

    # 4) Auditoría AST: CERO sincronizaciones host (.item/.cpu/.numpy/synchronize) en hot-path opt
    forbidden_attrs = {"item", "cpu", "numpy", "tolist", "synchronize"}
    for fn_obj in (
        zeropower_via_newtonschulz5,
        MuonHybridOptimizer.step,
        ScheduleFreeAdamW.step,
        ScheduleFreeAdamW.train,
        ScheduleFreeAdamW.eval,
    ):
        src = inspect.getsource(fn_obj)
        tree = ast.parse(inspect.cleandoc(src))
        hits = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs
        ]
        assert hits == [], f"Host-sync prohibido en {fn_obj.__qualname__}: {hits}"

    with pytest.raises(ValueError, match="optimizer_type"):
        build_sota_optimizer(model, optimizer_type="unknown_opt")


def test_rank_ensemble_predictions_improves_rank_ic_over_noisy_seeds() -> None:
    """El ensemble por rangos de 3 semillas supera en Rank IC a las semillas ruidosas."""
    rng = np.random.default_rng(2026)
    n_sessions, n_assets = 30, 80
    true_alpha = rng.standard_normal((n_sessions, n_assets))
    targets = 0.20 * true_alpha + rng.standard_normal((n_sessions, n_assets))

    # 3 semillas con señal débil común + ruido independiente y escalas distintas (para probar rango)
    scales = (1.0, 15.0, 0.05)
    seed_preds = [
        scales[i] * (0.45 * true_alpha + rng.standard_normal((n_sessions, n_assets)))
        for i in range(3)
    ]

    for method in ("rank_zscore", "rank_uniform", "gaussian_rank"):
        ens = rank_ensemble_predictions(seed_preds, method=method)
        assert ens.shape == (n_sessions, n_assets)
        assert np.isfinite(ens).all()
        report = evaluate_rank_ensemble(seed_preds, targets, method=method)
        assert float(report["ensemble_rank_ic"]) > float(report["mean_seed_rank_ic"]) + 0.005
        assert float(report["ensemble_rank_ic"]) > float(report["max_seed_rank_ic"])

    # Verificar modo apilado 1D con `bounds` y manejo seguro de NaNs
    flat_seeds = [p.reshape(-1) for p in seed_preds]
    flat_y = targets.reshape(-1)
    bounds = [(t * n_assets, (t + 1) * n_assets) for t in range(n_sessions)]
    ens_flat = rank_ensemble_predictions(flat_seeds, bounds=bounds, method="rank_zscore")
    assert ens_flat.shape == flat_y.shape
    ics_flat = [rank_ic_vector(ens_flat[lo:hi], flat_y[lo:hi]) for lo, hi in bounds]
    ics_s0 = [rank_ic_vector(flat_seeds[0][lo:hi], flat_y[lo:hi]) for lo, hi in bounds]
    assert float(np.mean(ics_flat)) > float(np.mean(ics_s0)) + 0.005

    # NaN parcial en una semilla no contamina las demás semillas
    noisy_with_nan = [p.copy() for p in seed_preds]
    noisy_with_nan[0][0, 0] = np.nan
    ens_nan = rank_ensemble_predictions(noisy_with_nan)
    assert np.isfinite(ens_nan[0, 0])


def test_edge_cases_9param_default_space_asha_rungs_bf16_optimizers_and_nan_safety() -> None:
    """Verifica espacio completo de 9 parámetros, poda solo en rungs ASHA, BF16 y seguridad NaN."""
    # 1) Búsqueda en los 9 parámetros canónicos MMARRS + poda estrictamente en rungs {1, 3}
    full_space = SearchSpaceSpec.default_mmarrs_space()
    tuner = TPEHyperbandAutoTuner(
        full_space, min_resource=1, max_resource=9, reduction_factor=3, n_startup_trials=6, seed=9
    )

    def _per_epoch_obj(params: dict[str, object], ctx: TrialContext) -> float:
        lr_pen = abs(math.log(float(params["lr"])) - math.log(1e-3))
        base = 1.0 - 0.3 * lr_pen - 0.2 * float(params["modality_dropout"])
        val = 0.0
        for ep in range(1, 10):
            val = base * (0.6 + 0.4 * (ep / 9.0))
            ctx.check_prune(val, resource=ep)
        return val

    rep = tuner.optimize(_per_epoch_obj, n_trials=14, train_years=[2022], val_years=[2023])
    assert rep["n_total_trials"] == 14
    assert rep["best_params"] is not None
    assert set(rep["best_params"].keys()) == set(DEFAULT_SEARCH_PARAM_NAMES)
    # Verificar que NINGÚN ensayo fue podado fuera de los rungs geométricos {1, 3}
    for rec in tuner.ledger:
        if rec.status == "pruned":
            assert rec.max_resource_reached in {1, 3}

    # 2) Seguridad BF16 en MuonHybridOptimizer y ScheduleFreeAdamW + state_dict roundtrip
    torch.manual_seed(5)
    net_bf16 = nn.Sequential(nn.Linear(12, 16), nn.SiLU(), nn.Linear(16, 4)).to(torch.bfloat16)
    x_bf = torch.randn(16, 12, dtype=torch.bfloat16)
    y_bf = torch.randn(16, 4, dtype=torch.bfloat16)
    for opt_kind in ("muon_hybrid", "schedule_free_adamw"):
        m_c = copy.deepcopy(net_bf16)
        opt_c = build_sota_optimizer(m_c, optimizer_type=opt_kind, lr=1e-2, weight_decay=1e-3)
        for _ in range(4):
            opt_c.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(m_c(x_bf), y_bf)
            loss.backward()
            opt_c.step()
        if isinstance(opt_c, ScheduleFreeAdamW):
            opt_c.eval()
            opt_c.train()
        opt_c.load_state_dict(opt_c.state_dict())
        for p in m_c.parameters():
            assert p.dtype == torch.bfloat16
            assert torch.isfinite(p).all()

    # 3) OptimalFitController con auto_adjust_underfitting=True y resistencia a NaN
    tiny_model = nn.Linear(4, 2)
    tiny_opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    ctrl = OptimalFitController(
        OptimalFitConfig(
            auto_adjust_underfitting=True, underfit_lr_boost=1.5, max_underfit_boosts=1
        )
    )
    d0 = ctrl.step(
        epoch=0,
        train_ic=0.001,
        val_ic=0.001,
        update_weight_ratio=1e-6,
        model=tiny_model,
        optimizer=tiny_opt,
    )
    assert d0.regime == "underfitting"
    assert float(tiny_opt.param_groups[0]["lr"]) == pytest.approx(1.5e-3, rel=1e-6)
    d_nan = ctrl.step(
        epoch=1,
        train_ic=float("nan"),
        val_ic=float("nan"),
        model=tiny_model,
        optimizer=tiny_opt,
    )
    assert not d_nan.improved
    assert ctrl.top_checkpoint_epochs == (0,)


def test_schedule_free_adamw_eval_guard_and_opt_in_auto_train() -> None:
    """`ScheduleFreeAdamW.step()` lanza RuntimeError en eval(); opt-in con allow_auto_train."""
    torch.manual_seed(123)
    m_normal = nn.Sequential(nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 2))
    m_guarded = copy.deepcopy(m_normal)
    m_direct = copy.deepcopy(m_normal)
    opt_normal = build_sota_optimizer(
        m_normal, optimizer_type="schedule_free_adamw", lr=2e-2, weight_decay=0.1
    )
    opt_guarded = ScheduleFreeAdamW(m_guarded, lr=2e-2, weight_decay=0.1)
    opt_direct = ScheduleFreeAdamW(m_direct, lr=2e-2, weight_decay=0.1, allow_auto_train=True)
    assert isinstance(opt_normal, ScheduleFreeAdamW)

    x = torch.randn(24, 8)
    y = torch.randn(24, 2)
    for _ in range(10):
        for mod, opt in ((m_normal, opt_normal), (m_guarded, opt_guarded), (m_direct, opt_direct)):
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(mod(x), y).backward()
            opt.step()

    opt_normal.eval()
    opt_guarded.eval()
    opt_direct.eval()

    # Default (allow_auto_train=False): llamar step() en eval() DEBE lanzar RuntimeError
    nn.functional.mse_loss(m_guarded(x), y).backward()
    with pytest.raises(RuntimeError, match="eval"):
        opt_guarded.step()

    # Con allow_auto_train=True e idéntico gradiente: coincide bit-a-bit con train() explícito
    opt_normal.train()
    opt_normal.zero_grad(set_to_none=True)
    nn.functional.mse_loss(m_normal(x), y).backward()
    for pn, pd in zip(m_normal.parameters(), m_direct.parameters(), strict=True):
        assert pn.grad is not None
        pd.grad = pn.grad.detach().clone()

    opt_normal.step()
    opt_direct.step()
    assert all(g["train_mode"] for g in opt_direct.param_groups)
    for pn, pd in zip(m_normal.parameters(), m_direct.parameters(), strict=True):
        torch.testing.assert_close(pd.detach(), pn.detach(), atol=1e-6, rtol=1e-6)
        assert torch.isfinite(pd).all()


def test_p0_7_model_soup_averages_only_named_parameters_not_floating_buffers() -> None:
    """P0-7: Model Soup promedia SOLO `named_parameters()`; buffers flotantes vienen de época #1."""

    class _EpisodicNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 3)
            self.register_buffer("slot_keys", torch.zeros(5, 4, dtype=torch.float32))
            self.register_buffer("balance_bias", torch.zeros(3, dtype=torch.float32))
            self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.int64))

    net = _EpisodicNet()
    ctrl = OptimalFitController(OptimalFitConfig(top_k_soup=3, patience=2, max_lr_decays=0))

    snapshots = [
        (0, 0.04, 0.20, 1.0, 10),
        (1, 0.08, 0.50, 9.0, 20),  # mejor época (#1)
        (2, 0.06, 0.80, 3.0, 30),
    ]
    for ep, val_ic, w_val, buf_val, ptr in snapshots:
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(w_val)
            net.slot_keys.fill_(buf_val)
            net.balance_bias.fill_(buf_val * 2.0)
            net.write_ptr.fill_(ptr)
        ctrl.step(epoch=ep, train_ic=val_ic + 0.01, val_ic=val_ic, model=net)

    soup = ctrl.build_model_soup_state_dict()
    expected_w = (0.20 + 0.50 + 0.80) / 3.0
    for name, p in net.named_parameters():
        torch.testing.assert_close(soup[name], torch.full_like(p, expected_w), atol=1e-6, rtol=1e-6)
    # Buffers flotantes NO deben promediarse (9.0 exacto de época #1, no (1+9+3)/3=4.33!)
    torch.testing.assert_close(soup["slot_keys"], torch.full((5, 4), 9.0), atol=0.0, rtol=0.0)
    torch.testing.assert_close(soup["balance_bias"], torch.full((3,), 18.0), atol=0.0, rtol=0.0)
    assert int(soup["write_ptr"].item()) == 20


def test_p0_7_muon_ns5_bf16_extreme_3d_moe_batched_and_per_param_step() -> None:
    """P0-7: NS-5 seguro en BF16 extremo, soporta 3D MoE por bmm y step por parámetro en AdamW."""
    # 1) BF16 extremo (1e18, 1e-18, rango-1) jamás diverge o produce NaN
    for scale in (1e18, 1e-18, 1.0):
        g = torch.randn(32, 64, dtype=torch.bfloat16) * scale
        u = zeropower_via_newtonschulz5(g, steps=5, use_bf16=True)
        assert u.dtype == torch.bfloat16 and torch.isfinite(u).all()
        if scale == 1.0:
            svals = torch.linalg.svdvals(u.float())
            assert float(svals.max()) < 1.35

    # 2) 3D stacked MoE experts [E, D_out, D_in] incluidos en Muon y batcheados vía bmm
    class _StackedMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Linear(16, 16, bias=False)
            self.w2 = nn.Linear(16, 16, bias=False)
            self.experts = nn.Parameter(torch.randn(4, 16, 16) * 0.1)
            self.bias_1d = nn.Parameter(torch.zeros(16))
            self.unused_1d = nn.Parameter(torch.ones(16))

    m = _StackedMoE()
    opt = MuonHybridOptimizer(m, lr=1e-2, weight_decay=1e-4)
    muon_params = [p for g in opt.param_groups if g.get("use_muon") for p in g["params"]]
    assert any(p is m.experts for p in muon_params), "3D MoE experts deben entrar en grupo Muon"

    # Per-parameter step tracking en AdamW: unused_1d sin grad durante 3 pasos, luego recibe grad
    for step_i in range(4):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 16)
        out = m.w2(m.w1(x)) + torch.einsum("nd,edk->nk", x, m.experts) + m.bias_1d
        if step_i == 3:
            out = out + m.unused_1d
        out.pow(2).mean().backward()
        opt.step()
    assert opt.state[m.bias_1d]["step"] == 4
    assert opt.state[m.unused_1d]["step"] == 1

    # 3) CUDA verificación: BF16 NS-5 + foreach Muon & ScheduleFree sin NaN
    if torch.cuda.is_available():
        g_cuda = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 1e18
        u_cuda = zeropower_via_newtonschulz5(g_cuda, steps=5, use_bf16=True)
        assert u_cuda.dtype == torch.bfloat16 and torch.isfinite(u_cuda).all()
        # Rank-1 matriz en BF16 donde sigma_max sin margen 1.01 podría pasar > 1.0
        v_r1 = torch.randn(64, 1, device="cuda", dtype=torch.float32)
        g_r1 = (v_r1 @ v_r1.T).to(torch.bfloat16)
        u_r1 = zeropower_via_newtonschulz5(g_r1, steps=5, use_bf16=True)
        assert torch.isfinite(u_r1).all()
        assert float(torch.linalg.svdvals(u_r1.float()).max()) < 1.35

        m_gpu = _StackedMoE().cuda()
        opt_gpu = MuonHybridOptimizer(m_gpu, lr=1e-2, weight_decay=1e-4, use_bf16_ns=True)
        for _ in range(3):
            opt_gpu.zero_grad(set_to_none=True)
            xg = torch.randn(16, 16, device="cuda")
            loss_g = (
                m_gpu.w2(m_gpu.w1(xg))
                + torch.einsum("nd,edk->nk", xg, m_gpu.experts)
                + m_gpu.bias_1d
                + m_gpu.unused_1d
            ).pow(2).mean()
            loss_g.backward()
            opt_gpu.step()
        for p in m_gpu.parameters():
            assert torch.isfinite(p).all()

        m_sf = _StackedMoE().cuda()
        sf_gpu = ScheduleFreeAdamW(m_sf, lr=1e-2, weight_decay=1e-4)
        for _ in range(3):
            sf_gpu.zero_grad(set_to_none=True)
            xg = torch.randn(16, 16, device="cuda")
            (m_sf.w2(m_sf.w1(xg)) + m_sf.bias_1d).pow(2).mean().backward()
            sf_gpu.step()
        sf_gpu.eval()
        sf_gpu.train()
        for p in m_sf.parameters():
            assert torch.isfinite(p).all()


def test_p1_5_shrink_and_perturb_single_device_sync_ast() -> None:
    """P1-5: `shrink_and_perturb` acumula en GPU/CPU sin `.cpu().item()` dentro del bucle."""
    src = inspect.getsource(shrink_and_perturb)
    tree = ast.parse(inspect.cleandoc(src))
    for_loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
    in_loop_syncs = []
    for lp in for_loops:
        for sub in ast.walk(lp):
            if isinstance(sub, ast.Attribute) and sub.attr in ("item", "cpu", "numpy", "tolist"):
                in_loop_syncs.append(sub.attr)
    assert in_loop_syncs == [], f"Syncs dentro del bucle de shrink_and_perturb: {in_loop_syncs}"

    if torch.cuda.is_available():
        m_g = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 8)).cuda()
        rep = shrink_and_perturb(m_g, shrink=0.8, perturb=0.02, seed=99)
        assert rep["all_finite"] is True
        assert rep["norm_after"] < rep["norm_before"]



def test_audit_ledger_lossless_jsonl_gz_roundtrip_append_and_size_reduction(tmp_path) -> None:
    """`write_audit_ledger`/`read_audit_ledger` soportan `.jsonl.gz` sin pérdida y comprimido."""
    space = SearchSpaceSpec.default_mmarrs_space()
    tuner = TPEHyperbandAutoTuner(
        space, min_resource=1, max_resource=3, reduction_factor=3, n_startup_trials=6, seed=21
    )

    def _obj(params: dict[str, object], ctx: TrialContext) -> dict[str, object]:
        sc = 0.5 - abs(math.log(float(params["lr"])) - math.log(1e-3))
        for r in ctx.rungs:
            ctx.check_prune(sc * (0.8 + 0.2 * r / ctx.max_resource), resource=r)
        return dict(score=sc, metrics=dict(val_ic=sc * 0.05, family="mmarrs_titan"))

    tuner.optimize(_obj, n_trials=24, train_years=[2022], val_years=[2023])

    plain_path = tmp_path / "ledger.jsonl"
    gz_path = tmp_path / "ledger.jsonl.gz"
    gz_path_2 = tmp_path / "ledger_copy.jsonl.gz"

    tuner.write_audit_ledger(plain_path)
    tuner.write_audit_ledger(gz_path)
    tuner.write_audit_ledger(gz_path_2)

    plain_rows = read_audit_ledger(plain_path)
    gz_rows = read_audit_ledger(gz_path)
    assert len(plain_rows) == 24
    assert gz_rows == plain_rows
    # Compresión real significativa y determinista byte-a-byte (mtime=0)
    assert gz_path.stat().st_size < plain_path.stat().st_size * 0.55
    assert gz_path.read_bytes() == gz_path_2.read_bytes()
    # Verificar cabecera mágica gzip RFC 1952
    assert gz_path.read_bytes()[:2] == b"\x1f\x8b"
    # Verificar append-only RFC 1952 sobre `.jsonl.gz` sin corrupción
    extra_rec = dict(trial_id=999, status="completed", score=0.42, params={"lr": 1e-3})
    append_audit_record(gz_path, extra_rec)
    reloaded = read_audit_ledger(gz_path)
    assert len(reloaded) == 25
    assert reloaded[:24] == plain_rows
    assert reloaded[-1]["trial_id"] == 999
    assert reloaded[-1]["score"] == pytest.approx(0.42)


