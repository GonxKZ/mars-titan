# ruff: noqa: E501
"""Suite TDD para `online_aggregator.py` (causalidad, paridad, agregadores, operable y CLI)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation import online_aggregator as oa  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPO_V11_SCREENING = ROOT / "scripts" / "dgx_screening_report.py"

DAY_US = 86_400_000_000
START_2012_US = 1_325_462_400_000_000  # 2012-01-02 00:00:00 UTC


def _load_screening_ref():
    spec = importlib.util.spec_from_file_location("dgx_screening_report_ref", REPO_V11_SCREENING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_split_parquet(
    path: Path,
    *,
    markets: list[str],
    n_sessions: int,
    n_assets: int,
    start_day_offset: int,
    target_fn,
    pred_fn,
    alpha_fn=None,
    matures_lag_days: int | None = None,
    shuffle_seed: int | None = None,
) -> Path:
    rows_sample_id = []
    rows_asset_id = []
    rows_market = []
    rows_pred_at = []
    rows_target = []
    rows_pred = []
    rows_alpha = []
    rows_matures = []

    for m_idx, market in enumerate(markets):
        market_hour_offset = m_idx * 3_600_000_000  # distinto instante por mercado si hace falta
        for s in range(n_sessions):
            t_us = START_2012_US + (start_day_offset + s) * DAY_US + market_hour_offset
            y = target_fn(market, s, n_assets)
            p = pred_fn(market, s, n_assets, y)
            a = alpha_fn(market, s, n_assets) if alpha_fn is not None else None
            for i in range(n_assets):
                asset = f"{market}/A{i:04d}"
                rows_sample_id.append(f"{asset}/{t_us}")
                rows_asset_id.append(asset)
                rows_market.append(market)
                rows_pred_at.append(t_us)
                rows_target.append(float(y[i]))
                rows_pred.append(float(p[i]))
                if a is not None:
                    rows_alpha.append(float(a[i]))
                if matures_lag_days is not None:
                    rows_matures.append(int(t_us + matures_lag_days * DAY_US))

    idx = np.arange(len(rows_sample_id))
    if shuffle_seed is not None:
        idx = np.random.default_rng(shuffle_seed).permutation(idx)

    cols = {
        "sample_id": pa.array([rows_sample_id[i] for i in idx], type=pa.string()),
        "asset_id": pa.array([rows_asset_id[i] for i in idx], type=pa.string()),
        "market": pa.array([rows_market[i] for i in idx], type=pa.string()),
        "prediction_at": pa.array([rows_pred_at[i] for i in idx], type=pa.int64()).cast(
            pa.timestamp("us", tz="UTC")
        ),
        "target": pa.array([rows_target[i] for i in idx], type=pa.float64()),
        "prediction": pa.array([rows_pred[i] for i in idx], type=pa.float64()),
    }
    if alpha_fn is not None:
        cols["alpha"] = pa.array([rows_alpha[i] for i in idx], type=pa.float64())
    if matures_lag_days is not None:
        cols["matures_at"] = pa.array([rows_matures[i] for i in idx], type=pa.int64())

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)
    return path


def _make_synthetic_runs(
    tmp_path: Path,
    *,
    markets: list[str] = ("US",),
    split_lengths: dict[str, int] | None = None,
    n_assets: int = 40,
    seed: int = 123,
    include_alpha: bool = False,
    matures_lag_days: int | None = None,
    eval_target_flip: bool = False,
    eval_perturb_from: int | None = None,
    shuffle_rows: bool = False,
) -> dict[str, Path]:
    """Crea 3 modelos sintéticos heterogéneos (m_strong, m_ortho, m_weak) con señal controlada."""
    if split_lengths is None:
        split_lengths = {"validation": 25, "calibration": 30, "evaluation": 45}
    rng = np.random.default_rng(seed)

    # Pre-generar factores latentes y ruido por (split, market, session)
    latents = {}
    offsets = {}
    day_cursor = 0
    for split in oa.SPLITS:
        n_sess = split_lengths[split]
        offsets[split] = day_cursor
        day_cursor += n_sess + 2
        for market in markets:
            for s in range(n_sess):
                z1 = rng.standard_normal(n_assets)
                z2 = rng.standard_normal(n_assets)
                alpha = (
                    0.004 * rng.standard_normal(n_assets) if include_alpha else np.zeros(n_assets)
                )
                eps = rng.standard_normal(n_assets)
                # target y = r - alpha; si include_alpha, -alpha correlaciona con y
                y = 0.01 * (0.28 * z1 + 0.14 * z2 + eps) - alpha
                if split == "evaluation":
                    if eval_target_flip:
                        y = -y
                    elif eval_perturb_from is not None and s >= eval_perturb_from:
                        y = -5.0 * y + 0.05
                # Predicciones de 3 modelos con escalas distintas
                p_strong = 0.50 * (z1 + 0.8 * rng.standard_normal(n_assets)) - 0.8 * alpha
                p_ortho = 0.05 * (z2 + 0.9 * rng.standard_normal(n_assets))
                p_weak = 0.08 * (0.3 * z1 + 1.5 * rng.standard_normal(n_assets))
                latents[(split, market, s)] = (y, alpha, p_strong, p_ortho, p_weak)

    model_dirs = {}
    for m_idx, name in enumerate(("m_strong", "m_ortho", "m_weak")):
        run_dir = tmp_path / "runs" / name
        for split in oa.SPLITS:
            n_sess = split_lengths[split]

            def _target_fn(market, s, n_a, sp=split):
                return latents[(sp, market, s)][0]

            def _pred_fn(market, s, n_a, y, sp=split, idx=m_idx):
                return latents[(sp, market, s)][2 + idx]

            def _alpha_fn(market, s, n_a, sp=split):
                return latents[(sp, market, s)][1]

            _write_split_parquet(
                run_dir / f"{split}-predictions.parquet",
                markets=list(markets),
                n_sessions=n_sess,
                n_assets=n_assets,
                start_day_offset=offsets[split],
                target_fn=_target_fn,
                pred_fn=_pred_fn,
                alpha_fn=_alpha_fn if include_alpha else None,
                matures_lag_days=matures_lag_days,
                shuffle_seed=(seed + m_idx + 17) if shuffle_rows else None,
            )
        model_dirs[name] = run_dir
    return model_dirs


# ---------------------------------------------------------------------------
# 1. Esquema Parquet, load_split y paridad con dgx_screening_report.py
# ---------------------------------------------------------------------------


def test_load_split_matches_dgx_screening_report_and_rejects_sealed_2024(tmp_path):
    ref = _load_screening_ref()
    run_dir = _make_synthetic_runs(tmp_path, markets=["US", "CN"], n_assets=35)["m_strong"]

    got = oa.load_split(run_dir, "evaluation")
    expected = ref.load_split(run_dir, "evaluation")

    assert got["sample"] == expected["sample"]
    assert got["sessions"] == expected["sessions"]
    np.testing.assert_allclose(got["target"], expected["target"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got["prediction"], expected["prediction"], rtol=0, atol=1e-6)
    assert got["prediction"].dtype == np.float32
    assert got["target"].dtype == np.float32

    # Sello 2024: una fila con prediction_at >= 2024-01-01 UTC debe ser rechazada
    sealed_path = tmp_path / "sealed_2024.parquet"
    t_2024 = oa.FINAL_TEST_START_US
    tbl = pa.table(
        {
            "sample_id": pa.array(["US/A0/1"]),
            "asset_id": pa.array(["US/A0"]),
            "market": pa.array(["US"]),
            "prediction_at": pa.array([t_2024], type=pa.int64()).cast(pa.timestamp("us", tz="UTC")),
            "target": pa.array([0.01]),
            "prediction": pa.array([0.01]),
        }
    )
    pq.write_table(tbl, sealed_path)
    with pytest.raises(ValueError, match="2024"):
        oa.load_split(sealed_path)

    # Sello 2024 también sobre matures_at aunque prediction_at < 2024-01-01 UTC
    sealed_mat_path = tmp_path / "sealed_2024_matures.parquet"
    tbl_mat = pa.table(
        {
            "sample_id": pa.array(["US/A0/1"]),
            "asset_id": pa.array(["US/A0"]),
            "market": pa.array(["US"]),
            "prediction_at": pa.array([t_2024 - DAY_US], type=pa.int64()).cast(
                pa.timestamp("us", tz="UTC")
            ),
            "matures_at": pa.array([t_2024], type=pa.int64()),
            "target": pa.array([0.01]),
            "prediction": pa.array([0.01]),
        }
    )
    pq.write_table(tbl_mat, sealed_mat_path)
    with pytest.raises(ValueError, match="2024"):
        oa.load_split(sealed_mat_path, maturity_col="matures_at")


def test_ranks_ic_and_newey_west_hac_match_reference_and_respect_market_blocks(tmp_path):
    ref = _load_screening_ref()
    runs = _make_synthetic_runs(tmp_path, markets=["US", "CN"], n_assets=35, seed=7)

    split_a = oa.load_split(runs["m_strong"], "evaluation")
    split_b = oa.load_split(runs["m_weak"], "evaluation")
    ref_a = ref.load_split(runs["m_strong"], "evaluation")
    ref_b = ref.load_split(runs["m_weak"], "evaluation")

    # Paridad de average_ranks con empates
    tied = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0], dtype=np.float32)
    np.testing.assert_array_equal(oa.average_ranks(tied), ref.average_ranks(tied))

    # Paridad de IC por sesión (tolerancia FP32 <= 1e-5)
    ic_a = oa.session_rank_ic(split_a, min_assets=30)
    ic_b = oa.session_rank_ic(split_b, min_assets=30)
    ref_ic_a = ref.session_ic(ref_a)
    ref_ic_b = ref.session_ic(ref_b)
    assert set(ic_a) == set(ref_ic_a)
    for k in ref_ic_a:
        assert math.isclose(ic_a[k], ref_ic_a[k], abs_tol=1e-5)

    # Paridad de contraste pareado HAC Newey-West segmentado por mercado
    got_pair = oa.paired_contrast(ic_a, ic_b)
    ref_pair = ref.paired(ref_ic_a, ref_ic_b)
    assert got_pair["sessions"] == ref_pair["sessions"]
    assert got_pair["hac_lags"] == ref_pair["hac_lags"]
    assert math.isclose(got_pair["mean"], ref_pair["mean"], abs_tol=1e-5)
    assert math.isclose(got_pair["se_hac"], ref_pair["se_hac"], abs_tol=1e-5)
    assert math.isclose(got_pair["t_hac"], ref_pair["t"], abs_tol=1e-3)

    # Comprobar explícitamente que la segmentación por mercado difiere de cruzar bloques
    # con autocorrelación fuerte dentro de cada mercado pero signo opuesto entre mercados
    us_vals = [0.05 + 0.02 * ((-1) ** (i // 4)) for i in range(40)]
    cn_vals = [-0.05 - 0.02 * ((-1) ** (i // 4)) for i in range(40)]
    series_blocked = {("US", i): v for i, v in enumerate(us_vals)}
    series_blocked.update({("CN", i): v for i, v in enumerate(cn_vals)})
    zeros = {k: 0.0 for k in series_blocked}
    c_blocked = oa.paired_contrast(series_blocked, zeros)
    t_unblocked, se_unblocked, _ = oa.newey_west_hac(
        np.array([series_blocked[k] for k in sorted(series_blocked)]), blocks=None
    )
    assert not math.isclose(c_blocked["se_hac"], se_unblocked, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# 2. PRUEBA OBLIGATORIA DE CAUSALIDAD (etiquetas no maduras t y futuras > t)
# ---------------------------------------------------------------------------


def test_causality_future_and_current_label_perturbation_invariant(tmp_path):
    """Perturbar etiquetas desde la sesión t0 (incluyendo t0 no madura) NO puede alterar
    ni un bit de los pesos ni de las predicciones emitidas en t <= t0 para ningún agregador.
    Con maturity_lag=2, tampoco puede alterar t0 + 1."""
    t0 = 20
    base_dir = tmp_path / "base"
    pert_dir = tmp_path / "pert"
    runs_base = _make_synthetic_runs(base_dir, n_assets=35, seed=11)
    runs_pert = _make_synthetic_runs(pert_dir, n_assets=35, seed=11, eval_perturb_from=t0)

    res_base = oa.run_aggregation(runs_base, maturity_lag=1, min_assets=30)
    res_pert = oa.run_aggregation(runs_pert, maturity_lag=1, min_assets=30)

    eval_sessions = res_base["aligned"]["evaluation"]["sessions"]
    # Filas hasta el final de la sesión t0 inclusive (sesiones 0..t0)
    end_row_t0 = eval_sessions[t0][3]

    for name in (
        "rank_average",
        "ridge_rolling",
        "ridge_ewma",
        "ridge_cov_active",
        "hedge",
        "hedge_fixed_share",
        "adanormalhedge",
        "regime_weights",
        "session_abstention",
    ):
        w_base = res_base["aggregators"][name]["evaluation_weights"]
        w_pert = res_pert["aggregators"][name]["evaluation_weights"]
        p_base = res_base["aggregators"][name]["evaluation_prediction"]
        p_pert = res_pert["aggregators"][name]["evaluation_prediction"]

        # Exactamente idénticos bit a bit hasta la sesión t0 inclusive
        np.testing.assert_array_equal(
            w_base[: t0 + 1],
            w_pert[: t0 + 1],
            err_msg=f"FUGA CAUSAL en pesos de {name} para sesiones <= {t0}",
        )
        np.testing.assert_array_equal(
            p_base[:end_row_t0],
            p_pert[:end_row_t0],
            err_msg=f"FUGA CAUSAL en predicciones de {name} para sesiones <= {t0}",
        )

    # Además: con lag=1, los agregadores adaptativos SÍ deben cambiar tras madurar t0 (t >= t0 + 1)
    for adaptive in ("ridge_rolling", "ridge_ewma", "hedge", "hedge_fixed_share"):
        w_base = res_base["aggregators"][adaptive]["evaluation_weights"]
        w_pert = res_pert["aggregators"][adaptive]["evaluation_weights"]
        assert not np.array_equal(w_base[t0 + 1 :], w_pert[t0 + 1 :]), (
            f"{adaptive} no reaccionó tras madurar la perturbación en t0 + 1"
        )

    # Y comprobar reacción post-madurez de AdaNormalHedge en dos expertos equilibrados
    losses_eq = np.full((20, 2), 0.45, dtype=np.float64)
    losses_eq_pert = losses_eq.copy()
    losses_eq_pert[10:, 0] = 0.05
    losses_eq_pert[10:, 1] = 0.95
    w_anh_b = oa.adanormalhedge_weights(losses_eq, maturity_lag=1)
    w_anh_p = oa.adanormalhedge_weights(losses_eq_pert, maturity_lag=1)
    np.testing.assert_array_equal(w_anh_b[:11], w_anh_p[:11])
    assert not np.array_equal(w_anh_b[11:], w_anh_p[11:])

    # Verificar también maturity_lag=2: t0 y t0+1 deben ser bit a bit idénticos
    res_base_d2 = oa.run_aggregation(runs_base, maturity_lag=2, min_assets=30)
    res_pert_d2 = oa.run_aggregation(runs_pert, maturity_lag=2, min_assets=30)
    end_row_t1 = eval_sessions[t0 + 1][3]
    for name in ("ridge_rolling", "ridge_ewma", "hedge", "adanormalhedge", "regime_weights"):
        np.testing.assert_array_equal(
            res_base_d2["aggregators"][name]["evaluation_weights"][: t0 + 2],
            res_pert_d2["aggregators"][name]["evaluation_weights"][: t0 + 2],
        )
        np.testing.assert_array_equal(
            res_base_d2["aggregators"][name]["evaluation_prediction"][:end_row_t1],
            res_pert_d2["aggregators"][name]["evaluation_prediction"][:end_row_t1],
        )


def test_explicit_maturity_timestamps_enforced_and_invalid_rejected():
    times = np.array([100, 200, 300, 400], dtype=np.int64)
    # Con retardo de 2 sesiones en timestamps: sesión 0 madura en 300
    matures = np.array([300, 400, 500, 600], dtype=np.int64)
    assert list(oa.mature_session_indices(times, matures, 0)) == []
    assert list(oa.mature_session_indices(times, matures, 1)) == []
    assert list(oa.mature_session_indices(times, matures, 2)) == [0]
    assert list(oa.mature_session_indices(times, matures, 3)) == [0, 1]

    # Una etiqueta que madura <= su propia decisión debe ser rechazada
    bad_matures = np.array([100, 300, 400, 500], dtype=np.int64)
    with pytest.raises(ValueError, match="madur"):
        oa.mature_session_indices(times, bad_matures, 1)

    # maturity_lag < 1 debe ser rechazado
    with pytest.raises(ValueError, match="retardo|maturity_lag"):
        oa.mature_session_indices(times, None, 1, maturity_lag=0)


# ---------------------------------------------------------------------------
# 3. Cero fuga del tramo de evaluación en selección de hiperparámetros y mejor modelo
# ---------------------------------------------------------------------------


def test_hyperparameter_and_best_model_selection_never_reads_evaluation(tmp_path):
    runs_base = _make_synthetic_runs(tmp_path / "base", n_assets=35, seed=21)
    runs_flip = _make_synthetic_runs(tmp_path / "flip", n_assets=35, seed=21, eval_target_flip=True)

    res_base = oa.run_aggregation(runs_base, min_assets=30)
    res_flip = oa.run_aggregation(runs_flip, min_assets=30)

    assert res_base["best_individual"]["model"] == res_flip["best_individual"]["model"]
    assert res_base["best_individual"]["model"] == "m_strong"
    assert (
        res_base["best_individual"]["decision_scores"]["m_strong"]
        > res_base["best_individual"]["decision_scores"]["m_weak"] + 0.05
    )
    assert res_base["selected_hyperparams"] == res_flip["selected_hyperparams"]
    # En el pipeline completo, los árbitros por pérdida de IC deben premiar al modelo fuerte, nunca al débil
    for arbiter in ("hedge", "hedge_fixed_share", "adanormalhedge"):
        mw = res_base["aggregators"][arbiter]["mean_evaluation_weights"]
        assert mw["m_strong"] > mw["m_weak"] + 0.30, (
            f"{arbiter} no premió a m_strong sobre m_weak en run_aggregation: {mw}"
        )
    # En la primera sesión de evaluación, los pesos solo dependen de validación+calibración
    for name in res_base["aggregators"]:
        if "evaluation_weights" in res_base["aggregators"][name]:
            np.testing.assert_array_equal(
                res_base["aggregators"][name]["evaluation_weights"][0],
                res_flip["aggregators"][name]["evaluation_weights"][0],
            )


# ---------------------------------------------------------------------------
# 4. Propiedades matemáticas de cada agregador
# ---------------------------------------------------------------------------


def test_rank_average_is_invariant_to_monotonic_rescaling(tmp_path):
    runs = _make_synthetic_runs(tmp_path / "orig", n_assets=35, seed=5)
    res_orig = oa.run_aggregation(runs, min_assets=30)

    # Reescalar m_strong por 100x y desplazar +50 en todas las particiones
    runs_scaled = {}
    for name, src_dir in runs.items():
        dst_dir = tmp_path / "scaled" / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for split in oa.SPLITS:
            tbl = pq.read_table(src_dir / f"{split}-predictions.parquet")
            if name == "m_strong":
                pred = tbl["prediction"].to_numpy() * 100.0 + 50.0
                idx = tbl.schema.get_field_index("prediction")
                tbl = tbl.set_column(idx, "prediction", pa.array(pred, type=pa.float64()))
            pq.write_table(tbl, dst_dir / f"{split}-predictions.parquet")
        runs_scaled[name] = dst_dir

    res_scaled = oa.run_aggregation(runs_scaled, min_assets=30)
    np.testing.assert_array_equal(
        res_orig["aggregators"]["rank_average"]["evaluation_prediction"],
        res_scaled["aggregators"]["rank_average"]["evaluation_prediction"],
    )


def test_hedge_and_adanormalhedge_concentrate_on_better_expert_and_satisfy_potential():
    # Experto 0 tiene IC=0.25 constante, experto 1 tiene IC=-0.05 constante
    n_sessions = 60
    ics = np.column_stack([np.full(n_sessions, 0.25), np.full(n_sessions, -0.05)])
    losses = (1.0 - ics) / 2.0

    w_hedge = oa.hedge_weights(losses, eta=4.0, maturity_lag=1)
    w_anh, state = oa.adanormalhedge_weights(losses, maturity_lag=1, return_state=True)

    assert w_hedge.shape == (n_sessions, 2)
    assert w_anh.shape == (n_sessions, 2)
    np.testing.assert_allclose(w_hedge.sum(axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(w_anh.sum(axis=1), 1.0, atol=1e-7)
    assert (w_hedge >= 0).all() and (w_anh >= 0).all()
    # Al final del horizonte ambos concentran fuertemente en el experto 0
    assert w_hedge[-1, 0] > 0.90
    assert w_anh[-1, 0] > 0.90

    # Comprobar Lema 2 de Luo & Schapire (2015): sum q_i Phi(R_i, C_i) <= B
    pot, bound = state.potential_bound()
    assert pot <= bound + 1e-9

    # Fórmula exacta del potencial de Luo & Schapire (2015): constante 3*C en el denominador
    r_pt = np.array([2.0, -0.5], dtype=np.float64)
    c_pt = np.array([3.0, 1.0], dtype=np.float64)
    got_w = oa._anh_weights(r_pt, c_pt)
    expected_w0 = 0.5 * (math.exp(9.0 / 12.0) - math.exp(1.0 / 12.0))
    expected_w1 = 0.5 * (math.exp(0.25 / 6.0) - 1.0)
    assert math.isclose(float(got_w[0]), expected_w0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(float(got_w[1]), expected_w1, rel_tol=1e-12, abs_tol=1e-12)

    # Fixed-Share mantiene un suelo mínimo de peso y recupera rápido tras cambio de régimen
    losses_sw = np.vstack(
        [
            np.tile([0.10, 0.90], (40, 1)),
            np.tile([0.90, 0.10], (15, 1)),
        ]
    )
    w_plain = oa.hedge_weights(losses_sw, eta=6.0, fixed_share_alpha=0.0, maturity_lag=1)
    w_fs = oa.hedge_weights(losses_sw, eta=6.0, fixed_share_alpha=0.05, maturity_lag=1)
    assert w_fs[40, 1] >= 0.025 - 1e-9
    assert w_plain[40, 1] < 1e-6
    assert w_fs[50, 1] > 0.80
    assert w_plain[50, 1] < 0.05

    # Coherencia de use_emitted_weights bajo retardo d=1 (idéntico) y d=3 (usa p_{t_0} emitido)
    rng = np.random.default_rng(19)
    losses_del = rng.uniform(0.2, 0.8, size=(30, 3))
    w_d1_em = oa.adanormalhedge_weights(losses_del, maturity_lag=1, use_emitted_weights=True)
    w_d1_cur = oa.adanormalhedge_weights(losses_del, maturity_lag=1, use_emitted_weights=False)
    np.testing.assert_array_equal(w_d1_em, w_d1_cur)
    w_d3_default = oa.adanormalhedge_weights(losses_del, maturity_lag=3)
    w_d3_em = oa.adanormalhedge_weights(losses_del, maturity_lag=3, use_emitted_weights=True)
    w_d3_cur = oa.adanormalhedge_weights(losses_del, maturity_lag=3, use_emitted_weights=False)
    np.testing.assert_array_equal(w_d3_default, w_d3_em)
    assert not np.allclose(w_d3_em, w_d3_cur)


def test_ridge_cov_active_rescues_weak_orthogonal_and_zeros_placebo():
    """Frente a un núcleo fuerte (IC=0.12), un clon del núcleo (rho=0.98), un órgano débil
    ortogonal (IC=0.035, rho=0) y un placebo puro (IC=0), el apilado Ridge consciente de
    covarianza sobre conjunto activo rescata peso para el órgano ortogonal y anula el placebo."""
    rng = np.random.default_rng(42)
    n_sessions, n_assets = 120, 120
    z_list = []
    ic_rows = []
    for _ in range(n_sessions):
        f1 = rng.standard_normal(n_assets)
        f2 = rng.standard_normal(n_assets)
        eps = rng.standard_normal(n_assets)
        y = 0.14 * f1 + 0.06 * f2 + eps
        p_core = f1 + 0.15 * rng.standard_normal(n_assets)
        p_clone = p_core + 0.15 * rng.standard_normal(n_assets)
        p_ortho = f2 + 0.30 * rng.standard_normal(n_assets)
        p_noise = rng.standard_normal(n_assets)
        Z = np.column_stack([oa.standardized_ranks(p) for p in (p_core, p_clone, p_ortho, p_noise)])
        z_list.append(Z)
        ic_rows.append([oa.rank_ic_vector(p, y) for p in (p_core, p_clone, p_ortho, p_noise)])

    ics = np.asarray(ic_rows, dtype=np.float64)
    covs = np.asarray([(Z.T @ Z) / n_assets for Z in z_list], dtype=np.float64)

    w_cov = oa.ridge_stacking_weights(
        ics,
        covs,
        lam=0.2,
        half_life=63.0,
        maturity_lag=1,
        use_current_cov=True,
        se_shrinkage=1.0,
    )
    mean_eval_w = w_cov[60:].mean(axis=0)
    # Órgano ortogonal rescatado (> 5 %) y placebo prácticamente nulo (< 2 %)
    assert mean_eval_w[2] > 0.05
    assert mean_eval_w[3] < 0.02
    assert mean_eval_w[0] + mean_eval_w[1] > 0.50

    # Ridge estándar calibrado en escala también comprime el placebo (< 8 %) y converge a 1/M con lam grande
    w_roll = oa.ridge_stacking_weights(ics, covs, lam=0.2, window=60, maturity_lag=1)
    assert w_roll[60:, 3].mean() < 0.08
    w_big_lam = oa.ridge_stacking_weights(ics, covs, lam=1e6, window=60, maturity_lag=1)
    np.testing.assert_allclose(w_big_lam[60:], 0.25, atol=1e-3)

    # `use_current_cov=True` reacciona en el mismo instante t a un colapso repentino de correlación en t
    ics_const = np.full((35, 3), 0.15, dtype=np.float64)
    covs_shock = np.tile(np.eye(3, dtype=np.float64), (35, 1, 1))
    covs_shock[30] = np.array([[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 1.0]])
    w_cur = oa.ridge_stacking_weights(
        ics_const, covs_shock, lam=0.1, maturity_lag=1, use_current_cov=True, se_shrinkage=0.0
    )
    w_hist = oa.ridge_stacking_weights(
        ics_const, covs_shock, lam=0.1, maturity_lag=1, use_current_cov=False, se_shrinkage=0.0
    )
    assert w_cur[30, 2] > w_cur[30, 0] + 0.10
    np.testing.assert_allclose(w_hist[30], np.full(3, 1.0 / 3.0), atol=1e-6)


def test_regime_weights_adapt_causally_to_observable_volatility_regime():
    """En baja dispersión manda modelo 0; en alta dispersión manda modelo 1.
    Verifica adaptación causal y ausencia de fuga al perturbar el futuro."""
    rng = np.random.default_rng(99)
    n_sessions, _n_assets = 100, 60
    ics = np.zeros((n_sessions, 2), dtype=np.float64)
    target_disp = np.zeros(n_sessions, dtype=np.float64)
    # Regímenes en bloques largos de 15 sesiones para que la dispersión pasada prediga el régimen
    for s in range(n_sessions):
        high = (s // 15) % 2 == 1
        target_disp[s] = (0.04 if high else 0.01) + 0.001 * rng.standard_normal()
        if high:
            ics[s] = [0.02 + 0.01 * rng.standard_normal(), 0.20 + 0.01 * rng.standard_normal()]
        else:
            ics[s] = [0.20 + 0.01 * rng.standard_normal(), 0.02 + 0.01 * rng.standard_normal()]

    w_base, regimes_base = oa.regime_conditioned_weights(
        ics,
        target_disp,
        window=5,
        tau=5.0,
        eta=6.0,
        maturity_lag=1,
        min_history=10,
    )
    # Perturbar dispersión e IC desde t0=50 en adelante: todo <= 50 debe ser bit a bit idéntico
    t0 = 50
    ics_pert = ics.copy()
    disp_pert = target_disp.copy()
    ics_pert[t0:] = -ics_pert[t0:]
    disp_pert[t0:] = 10.0
    w_pert, regimes_pert = oa.regime_conditioned_weights(
        ics_pert,
        disp_pert,
        window=5,
        tau=5.0,
        eta=6.0,
        maturity_lag=1,
        min_history=10,
    )
    np.testing.assert_array_equal(w_base[: t0 + 1], w_pert[: t0 + 1])
    np.testing.assert_array_equal(regimes_base[: t0 + 1], regimes_pert[: t0 + 1])

    # En la segunda mitad, en régimen alto pesa más modelo 1; en régimen bajo pesa más modelo 0
    late = np.arange(n_sessions) >= 40
    high_mask = late & (regimes_base == 1)
    low_mask = late & (regimes_base == 0)
    assert high_mask.any() and low_mask.any()
    assert w_base[high_mask, 1].mean() > w_base[high_mask, 0].mean() + 0.10
    assert w_base[low_mask, 0].mean() > w_base[low_mask, 1].mean() + 0.10


def test_abstention_honest_metrics_and_matched_magnitude_control():
    """La abstención de sesión cuenta IC=0 al abstenerse; y la abstención por activo
    compara contra el control emparejado por |predicción| (sin trampa de range enhancement)."""
    # 1) Abstención por sesión: bloque negativo de 30 sesiones tras 30 sesiones positivas
    ic_series = np.concatenate([np.full(30, 0.12), np.full(30, -0.10)])
    out = oa.evaluate_session_abstention(
        ic_series, window=5, threshold=0.0, maturity_lag=1, cost_per_session=0.02
    )
    assert 0.0 < out["coverage"] < 1.0
    assert out["unconditional_ic"] > float(np.mean(ic_series))
    assert out["net_utility"] > float(np.mean(ic_series) - 0.02)
    # Perturbar sesión t0 no cambia exposición en <= t0
    ic_pert = ic_series.copy()
    ic_pert[35:] = 0.50
    out_pert = oa.evaluate_session_abstention(
        ic_pert, window=5, threshold=0.0, maturity_lag=1, cost_per_session=0.02
    )
    np.testing.assert_array_equal(out["exposure"][:36], out_pert["exposure"][:36])

    # 2) Abstención selectiva por activo: con señal homocedástica sin ventaja de incertidumbre,
    # NO debe superar sistemáticamente al control emparejado de magnitud |z|
    rng = np.random.default_rng(7)
    n_sessions, n_assets = 40, 200
    preds_models = []
    targets = []
    for _ in range(n_sessions):
        f = rng.standard_normal(n_assets)
        y = 0.15 * f + rng.standard_normal(n_assets)
        m1 = f + rng.standard_normal(n_assets)
        m2 = f + rng.standard_normal(n_assets)
        m3 = f + rng.standard_normal(n_assets)
        preds_models.append(np.column_stack([m1, m2, m3]))
        targets.append(y)
    asset_res = oa.evaluate_asset_abstention(preds_models, targets, keep_fraction=0.5, seed=0)
    assert math.isclose(asset_res["coverage"], 0.5, abs_tol=0.02)
    # El control |z| ya sube respecto al placebo aleatorio (range enhancement)
    assert asset_res["active_ic_magnitude_control"] > asset_res["active_ic_random_placebo"] + 0.01
    # Y la puntuación de desacuerdo sin heteroscedasticidad NO gana artificialmente sobre |z|
    assert (
        abs(
            asset_res["active_ic_signal_to_disagreement"] - asset_res["active_ic_magnitude_control"]
        )
        < 0.03
    )

    # 3) session_rank_ic DEBE puntuar 0.0 cuando abstained=True y ponderar e_t * IC_t con exposición continua
    n_a = 40
    y_s0 = np.linspace(-1.0, 1.0, n_a, dtype=np.float32)
    y_s1 = np.linspace(-2.0, 2.0, n_a, dtype=np.float32)
    p_s0 = y_s0.copy()
    p_s1 = y_s1.copy()
    split_mock = dict(
        target=np.concatenate([y_s0, y_s1]),
        prediction=np.concatenate([p_s0, p_s1]),
        sessions=[("US", 100, 0, n_a), ("US", 200, n_a, 2 * n_a)],
    )
    ic_full = oa.session_rank_ic(split_mock, min_assets=30)
    assert math.isclose(ic_full[("US", 100)], 1.0, abs_tol=1e-5)
    assert math.isclose(ic_full[("US", 200)], 1.0, abs_tol=1e-5)

    ic_abst = oa.session_rank_ic(
        split_mock, min_assets=30, abstained=np.array([True, False], dtype=bool)
    )
    assert ic_abst[("US", 100)] == 0.0
    assert math.isclose(ic_abst[("US", 200)], 1.0, abs_tol=1e-5)

    ic_exp = oa.session_rank_ic(
        split_mock, min_assets=30, exposure=np.array([0.25, 0.75], dtype=np.float64)
    )
    assert math.isclose(ic_exp[("US", 100)], 0.25, abs_tol=1e-5)
    assert math.isclose(ic_exp[("US", 200)], 0.75, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 5. Objetivo operable (apagado por defecto, bit-idéntico; encendido sin fuga)
# ---------------------------------------------------------------------------


def test_operable_mode_off_is_bit_identical_and_on_uses_calibration_only(tmp_path):
    _make_synthetic_runs(tmp_path / "no_alpha", include_alpha=False, n_assets=35, seed=31)
    runs_with_alpha = _make_synthetic_runs(
        tmp_path / "with_alpha", include_alpha=True, n_assets=35, seed=31
    )

    # Con operable=False, la presencia de columna `alpha` en el parquet NO cambia nada
    # (usando exactamente el mismo target — aquí clonamos runs_with_alpha quitando columna alpha)
    runs_stripped = {}
    for name, src_dir in runs_with_alpha.items():
        dst = tmp_path / "stripped" / name
        dst.mkdir(parents=True, exist_ok=True)
        for split in oa.SPLITS:
            tbl = pq.read_table(src_dir / f"{split}-predictions.parquet").drop(["alpha"])
            pq.write_table(tbl, dst / f"{split}-predictions.parquet")
        runs_stripped[name] = dst

    r_with_off = oa.run_aggregation(runs_with_alpha, operable=False, min_assets=30)
    r_stripped_off = oa.run_aggregation(runs_stripped, operable=False, min_assets=30)
    for name in r_with_off["aggregators"]:
        np.testing.assert_array_equal(
            r_with_off["aggregators"][name]["evaluation_prediction"],
            r_stripped_off["aggregators"][name]["evaluation_prediction"],
        )

    # Con operable=True, los factores k_m se eligen SOLO en calibración (invariantes a flip de evaluación)
    runs_alpha_flip = _make_synthetic_runs(
        tmp_path / "with_alpha_flip",
        include_alpha=True,
        eval_target_flip=True,
        n_assets=35,
        seed=31,
    )
    r_op = oa.run_aggregation(runs_with_alpha, operable=True, alpha_col="alpha", min_assets=30)
    r_op_flip = oa.run_aggregation(runs_alpha_flip, operable=True, alpha_col="alpha", min_assets=30)
    assert r_op["operable"]["scales"] == r_op_flip["operable"]["scales"]
    assert r_op["operable"]["enabled"] is True
    assert all(k in oa.OPERABLE_SCALES for k in r_op["operable"]["scales"].values())

    # Verificar que k_m realmente escala las predicciones en _transform_split_to_operable
    # (modelo sobre-escalado tipo titans_mac_nogain con alpha informativa en y + alpha:
    # k_m óptimo <= 0.10 y supera con claridad a k=1.0)
    rng_os = np.random.default_rng(311)
    n_a_os = 40
    split_lens_os = {"validation": 15, "calibration": 30, "evaluation": 35}
    lat_os = {}
    off_os = {}
    cursor = 0
    for sp in oa.SPLITS:
        n_s = split_lens_os[sp]
        off_os[sp] = cursor
        cursor += n_s + 2
        for s in range(n_s):
            z1 = rng_os.standard_normal(n_a_os)
            alpha = 0.008 * rng_os.standard_normal(n_a_os)
            # y_op = y + alpha tiene señal de z1 Y de +alpha (como en datos reales: +alpha IC > 0)
            y_op = 0.01 * (0.25 * z1 + rng_os.standard_normal(n_a_os)) + 0.50 * alpha
            y = y_op - alpha
            # m_strong emite con dispersión 30x superior a alpha -> necesita k_m pequeño (<= 0.10)
            p_strong = 0.30 * (z1 + 0.5 * rng_os.standard_normal(n_a_os))
            p_other = 0.004 * (z1 + 1.5 * rng_os.standard_normal(n_a_os))
            lat_os[(sp, "US", s)] = (y, alpha, p_strong, p_other)

    runs_overscaled = {}
    for m_idx, name in enumerate(("m_strong", "m_other")):
        dst = tmp_path / "overscaled" / name
        for sp in oa.SPLITS:
            _write_split_parquet(
                dst / f"{sp}-predictions.parquet",
                markets=["US"],
                n_sessions=split_lens_os[sp],
                n_assets=n_a_os,
                start_day_offset=off_os[sp],
                target_fn=lambda m, s, n, _sp=sp: lat_os[(_sp, m, s)][0],
                pred_fn=lambda m, s, n, y, _sp=sp, _idx=m_idx: lat_os[(_sp, m, s)][2 + _idx],
                alpha_fn=lambda m, s, n, _sp=sp: lat_os[(_sp, m, s)][1],
            )
        runs_overscaled[name] = dst

    r_os = oa.run_aggregation(runs_overscaled, operable=True, alpha_col="alpha", min_assets=30)
    k_strong = r_os["operable"]["scales"]["m_strong"]
    assert k_strong <= 0.10, (
        f"Se esperaba k_m <= 0.10 para modelo sobre-escalado, obtenido {k_strong}"
    )
    raw_eval = oa.load_split(runs_overscaled["m_strong"], "evaluation", alpha_col="alpha")
    m_idx = r_os["model_names"].index("m_strong")
    expected_p_op = (np.float32(k_strong) * raw_eval["prediction"] + raw_eval["alpha"]).astype(
        np.float32
    )
    np.testing.assert_array_equal(
        r_os["aligned"]["evaluation"]["predictions"][:, m_idx], expected_p_op
    )
    unscaled_ic = float(
        np.mean(
            list(
                oa.session_rank_ic(
                    dict(
                        target=(raw_eval["target"] + raw_eval["alpha"]).astype(np.float32),
                        prediction=(raw_eval["prediction"] + raw_eval["alpha"]).astype(np.float32),
                        sessions=raw_eval["sessions"],
                    ),
                    min_assets=30,
                ).values()
            )
        )
    )
    assert r_os["individuals"]["m_strong"]["evaluation"]["all"]["mean"] > unscaled_ic + 0.01


# ---------------------------------------------------------------------------
# 6. Alineación con activos faltantes, determinismo ante orden y CLI completa
# ---------------------------------------------------------------------------


def test_missing_assets_strict_raises_and_inner_aligns_cleanly(tmp_path):
    runs = _make_synthetic_runs(tmp_path, n_assets=40, seed=41)
    # Quitar 3 filas de evaluación en m_weak
    p = runs["m_weak"] / "evaluation-predictions.parquet"
    tbl = pq.read_table(p)
    pq.write_table(tbl.slice(3, tbl.num_rows - 3), p)

    with pytest.raises(ValueError, match="alinea|muestra|sample"):
        oa.run_aggregation(runs, align="strict", min_assets=30)

    res_inner = oa.run_aggregation(runs, align="inner", min_assets=30)
    assert res_inner["alignment"]["evaluation"]["dropped_rows"]["m_strong"] == 3
    assert res_inner["alignment"]["evaluation"]["dropped_rows"]["m_weak"] == 0
    assert (
        res_inner["aggregators"]["rank_average"]["summary"]["evaluation"]["all"]["sessions"] == 45
    )


def test_row_order_invariance_bit_for_bit_deterministic(tmp_path):
    runs_ord = _make_synthetic_runs(tmp_path / "ord", n_assets=35, seed=55, shuffle_rows=False)
    runs_shf = _make_synthetic_runs(tmp_path / "shf", n_assets=35, seed=55, shuffle_rows=True)

    r1 = oa.run_aggregation(runs_ord, min_assets=30)
    r2 = oa.run_aggregation(runs_shf, min_assets=30)
    for name in r1["aggregators"]:
        np.testing.assert_array_equal(
            r1["aggregators"][name]["evaluation_prediction"],
            r2["aggregators"][name]["evaluation_prediction"],
        )


def test_cli_end_to_end_writes_json_and_markdown_reports(tmp_path):
    runs = _make_synthetic_runs(tmp_path / "data", markets=["US", "CN"], n_assets=35, seed=61)
    out_dir = tmp_path / "out"
    argv = [
        "--model",
        f"m_strong={runs['m_strong']}",
        "--model",
        f"m_ortho={runs['m_ortho']}",
        "--model",
        f"m_weak={runs['m_weak']}",
        "--min-assets",
        "30",
        "--output",
        str(out_dir),
    ]
    rc = oa.main(argv)
    assert rc == 0
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    assert report_json.is_file() and report_md.is_file()

    doc = json.loads(report_json.read_text(encoding="utf-8"))
    assert doc["best_individual"]["selection_split"] == "calibration"
    assert "rank_average" in doc["aggregators"]
    assert "adanormalhedge" in doc["aggregators"]
    assert "placebo_permuted_labels" in doc["placebos"]
    assert "placebo_shuffled_weights" in doc["placebos"]
    assert "US" in doc["aggregators"]["rank_average"]["summary"]["evaluation"]["by_market"]
    assert "CN" in doc["aggregators"]["rank_average"]["summary"]["evaluation"]["by_market"]
    assert "vs_best_individual" in doc["aggregators"]["adanormalhedge"]["contrasts"]["evaluation"]
    assert "vs_rank_average" in doc["aggregators"]["adanormalhedge"]["contrasts"]["evaluation"]


# ---------------------------------------------------------------------------
# 7. Casos adversarios de Pasada 3 (frontera de madurez, 4.198 activos FP32, constante, NaN)
# ---------------------------------------------------------------------------


def test_calibration_evaluation_maturity_frontier_and_lag_embargo(tmp_path):
    """(a) Rechaza `matures_at` de calibración posterior al inicio de evaluación;
    (b) Con `maturity_lag=3` sin `matures_at`, perturbar las 2 últimas sesiones de calibración
    no altera `best_individual`, `selected_hyperparams` ni `operable.scales`."""
    # (a) Frontera explícita violada: gap entre splits es de 2 días, ponemos matures_lag_days=5
    runs_bad = _make_synthetic_runs(
        tmp_path / "bad_frontier", matures_lag_days=5, n_assets=35, seed=71
    )
    with pytest.raises(ValueError, match="Frontera causal"):
        oa.run_aggregation(runs_bad, min_assets=30)

    # (b) Embargo implícito con maturity_lag=3: alterar últimas 2 sesiones de calibración
    runs_base = _make_synthetic_runs(
        tmp_path / "emb_base", include_alpha=True, n_assets=35, seed=72
    )
    runs_tail = _make_synthetic_runs(
        tmp_path / "emb_tail", include_alpha=True, n_assets=35, seed=72
    )
    for _name, rdir in runs_tail.items():
        p = rdir / "calibration-predictions.parquet"
        tbl = pq.read_table(p)
        times = tbl["prediction_at"].cast(pa.int64()).to_numpy()
        last_two = set(np.unique(times)[-2:].tolist())
        targ = tbl["target"].to_numpy().copy()
        mask = np.array([int(t) in last_two for t in times], dtype=bool)
        targ[mask] = -10.0 * targ[mask]
        idx = tbl.schema.get_field_index("target")
        tbl = tbl.set_column(idx, "target", pa.array(targ, type=pa.float64()))
        pq.write_table(tbl, p)

    r_b = oa.run_aggregation(
        runs_base, maturity_lag=3, operable=True, alpha_col="alpha", min_assets=30
    )
    r_t = oa.run_aggregation(
        runs_tail, maturity_lag=3, operable=True, alpha_col="alpha", min_assets=30
    )
    assert r_b["best_individual"]["model"] == r_t["best_individual"]["model"]
    assert r_b["selected_hyperparams"] == r_t["selected_hyperparams"]
    assert r_b["operable"]["scales"] == r_t["operable"]["scales"]


def test_fp32_parity_on_full_us_4198_assets_and_constant_expert_resilience(tmp_path):
    """Verifica precisión FP32 en el universo real de 4.198 activos y robustez ante experto constante."""
    ref = _load_screening_ref()
    rng = np.random.default_rng(2013)
    y = rng.standard_normal(4198).astype(np.float32)
    p = (0.12 * y + rng.standard_normal(4198)).astype(np.float32)
    ic_fp32 = oa.rank_ic_vector(p, y)
    ic_fp64 = ref.correlation(ref.average_ranks(p), ref.average_ranks(y))
    assert math.isclose(ic_fp32, ic_fp64, abs_tol=1e-5)

    # Experto constante en todo evaluación no produce NaN ni matriz singular
    runs = _make_synthetic_runs(tmp_path / "const", n_assets=35, seed=73)
    p_weak = runs["m_weak"] / "evaluation-predictions.parquet"
    tbl = pq.read_table(p_weak)
    idx = tbl.schema.get_field_index("prediction")
    tbl = tbl.set_column(idx, "prediction", pa.array(np.zeros(tbl.num_rows), type=pa.float64()))
    pq.write_table(tbl, p_weak)

    res = oa.run_aggregation(runs, min_assets=30)
    for name, entry in res["aggregators"].items():
        assert np.isfinite(entry["evaluation_prediction"]).all(), f"NaN en {name}"
        assert np.isfinite(entry["evaluation_weights"]).all(), f"Pesos NaN en {name}"


def test_newey_west_rejects_nonfinite_and_inner_align_reports_scorable_sessions(tmp_path):
    with pytest.raises(ValueError, match="finitas"):
        oa.newey_west_hac(np.array([0.1, float("nan"), 0.2]))

    runs = _make_synthetic_runs(tmp_path / "drop_sess", n_assets=35, seed=74)
    # Eliminar 10 filas de la primera sesión de evaluación en m_weak -> baja a 25 < 30
    p = runs["m_weak"] / "evaluation-predictions.parquet"
    tbl = pq.read_table(p)
    pq.write_table(tbl.slice(10, tbl.num_rows - 10), p)

    res = oa.run_aggregation(runs, align="inner", min_assets=30)
    assert res["alignment"]["evaluation"]["sessions"] == 45
    assert res["alignment"]["evaluation"]["scorable_sessions"] == 44


def test_zero_label_model_disagreement_regime_is_strictly_causal():
    """El observable de desacuerdo entre modelos (`observable_is_target_based=False`)
    es estrictamente causal frente a perturbaciones futuras."""
    rng = np.random.default_rng(75)
    n_sessions = 50
    ics = np.column_stack(
        [
            0.12 + 0.02 * rng.standard_normal(n_sessions),
            0.10 + 0.02 * rng.standard_normal(n_sessions),
        ]
    )
    disagree = np.full(n_sessions, 0.25, dtype=np.float64)
    disagree[15:25] = 0.60
    t0 = 28
    disagree_pert = disagree.copy()
    disagree_pert[t0 + 1 :] = 9.0
    ics_pert = ics.copy()
    ics_pert[t0:] = -ics_pert[t0:]

    w1, r1 = oa.regime_conditioned_weights(
        ics, disagree, window=5, min_history=10, observable_is_target_based=False
    )
    w2, r2 = oa.regime_conditioned_weights(
        ics_pert, disagree_pert, window=5, min_history=10, observable_is_target_based=False
    )
    np.testing.assert_array_equal(w1[: t0 + 1], w2[: t0 + 1])
    np.testing.assert_array_equal(r1[: t0 + 1], r2[: t0 + 1])


# ---------------------------------------------------------------------------
# 8. Contratos endurecidos tras auditoría independiente (Objeciones 2–5)
# ---------------------------------------------------------------------------


def test_selection_split_validation_requires_validation_and_pooled_is_consistent(tmp_path):
    """--no-validation + --selection-split validation lanza ValueError limpio (nunca KeyError),
    y selection_split='pooled' usa validation + calibration de forma coherente en best_individual y _tune_grid."""
    runs = _make_synthetic_runs(tmp_path / "base", n_assets=35, seed=81)
    with pytest.raises(ValueError, match="validation"):
        oa.run_aggregation(
            runs, splits=("calibration", "evaluation"), selection_split="validation", min_assets=30
        )

    # En modo pooled, invertir target en validation debe afectar decision_scores Y puede mover tune_splits,
    # mientras que en selection_split="calibration" no altera decision_scores de calibración.
    runs_val_flip = _make_synthetic_runs(tmp_path / "vflip", n_assets=35, seed=81)
    for _name, rdir in runs_val_flip.items():
        p = rdir / "validation-predictions.parquet"
        tbl = pq.read_table(p)
        idx = tbl.schema.get_field_index("target")
        tbl = tbl.set_column(idx, "target", pa.array(-tbl["target"].to_numpy(), type=pa.float64()))
        pq.write_table(tbl, p)

    r_pool_base = oa.run_aggregation(runs, selection_split="pooled", min_assets=30)
    r_pool_flip = oa.run_aggregation(runs_val_flip, selection_split="pooled", min_assets=30)
    assert (
        r_pool_base["best_individual"]["decision_scores"]
        != r_pool_flip["best_individual"]["decision_scores"]
    )
    # Verificar que en pooled los hiperparámetros se evalúan también sobre validation
    r_cal_base = oa.run_aggregation(runs, selection_split="calibration", min_assets=30)
    assert (
        r_cal_base["best_individual"]["decision_scores"]
        != r_pool_base["best_individual"]["decision_scores"]
    )


def test_ridge_and_abstention_reject_invalid_window_and_half_life():
    ics = np.full((10, 2), 0.1, dtype=np.float64)
    covs = np.tile(np.eye(2, dtype=np.float64), (10, 1, 1))
    with pytest.raises(ValueError, match="window"):
        oa.ridge_stacking_weights(ics, covs, window=0)
    with pytest.raises(ValueError, match="window"):
        oa.ridge_stacking_weights(ics, covs, window=-5)
    with pytest.raises(ValueError, match="half_life"):
        oa.ridge_stacking_weights(ics, covs, half_life=0.0)
    with pytest.raises(ValueError, match="half_life"):
        oa.ridge_stacking_weights(ics, covs, half_life=-2.0)
    with pytest.raises(ValueError, match="window"):
        oa.evaluate_session_abstention(np.full(10, 0.1), window=0)


def test_asymmetric_matures_at_rejected_across_models_and_splits(tmp_path):
    """Presencia asimétrica de matures_at entre modelos (incluso si falta en el 1.º alfabético)
    o entre tramos debe lanzar ValueError explícito, nunca degradar silenciosamente."""
    runs_with = _make_synthetic_runs(tmp_path / "with_m", matures_lag_days=1, n_assets=35, seed=82)
    runs_no = _make_synthetic_runs(tmp_path / "no_m", matures_lag_days=None, n_assets=35, seed=82)

    # Caso 1: primer modelo alfabético (m_ortho) sin matures_at, segundo (m_strong) con matures_at
    d_a = oa.load_split(runs_no["m_ortho"], "evaluation", maturity_col="matures_at")
    d_b = oa.load_split(runs_with["m_strong"], "evaluation", maturity_col="matures_at")
    assert d_a["matures_at"] is None and d_b["matures_at"] is not None
    with pytest.raises(ValueError, match="matures_at"):
        oa.align_models_split({"a_no_mat": d_a, "b_with_mat": d_b}, align="strict")
    with pytest.raises(ValueError, match="matures_at"):
        oa.align_models_split({"a_no_mat": d_a, "b_with_mat": d_b}, align="inner")

    # Caso 2: validation sin matures_at mientras calibration y evaluation sí tienen matures_at
    runs_mixed = {}
    for name in ("m_strong", "m_ortho", "m_weak"):
        dst = tmp_path / "mixed" / name
        dst.mkdir(parents=True, exist_ok=True)
        for sp in oa.SPLITS:
            src = (
                runs_no[name] if sp == "validation" else runs_with[name]
            ) / f"{sp}-predictions.parquet"
            pq.write_table(pq.read_table(src), dst / f"{sp}-predictions.parquet")
        runs_mixed[name] = dst
    with pytest.raises(ValueError, match="matures_at"):
        oa.run_aggregation(runs_mixed, min_assets=30)


def test_session_abstention_tuning_respects_cost_and_min_call_coverage(tmp_path):
    """La sintonización de session_abstention en calibración maximiza utilidad neta
    `e_t * (IC_t - abstention_cost)` sujeta a `coverage >= MIN_CALL_COVERAGE (0.50)`."""
    # Calibración de 80 sesiones: primeras 35 con IC realista (~+0.20) y últimas 45 con IC +0.010:
    # - Ventana causal w=21 cae a +0.010 (entre 0.0 y +0.02) en las últimas ~24 sesiones (cov ~70 % >= 50 %).
    # - Con coste 0.0: esas sesiones aportan +0.010 > 0 -> mejor mantenerse activo (threshold=-0.02).
    # - Con coste 0.05: aportan +0.010 - 0.05 = -0.04 < 0 -> mejor abstenerse (threshold=+0.02).
    rng = np.random.default_rng(83)
    n_assets = 40
    y_base = np.arange(n_assets, dtype=np.float64)
    p_low_ic = None
    for _ in range(5000):
        cand = rng.permutation(n_assets).astype(np.float64)
        ic_c = oa.rank_ic_vector(cand, y_base)
        if 0.008 <= ic_c <= 0.014:
            p_low_ic = cand
            break
    assert p_low_ic is not None

    split_lengths = {"validation": 10, "calibration": 80, "evaluation": 20}
    latents = {}
    offsets = {}
    day_cursor = 0
    for sp in oa.SPLITS:
        n_s = split_lengths[sp]
        offsets[sp] = day_cursor
        day_cursor += n_s + 2
        for s in range(n_s):
            if sp == "calibration" and s >= 35:
                latents[(sp, "US", s)] = (y_base, p_low_ic, p_low_ic)
            else:
                y = rng.standard_normal(n_assets)
                p1 = 0.25 * y + rng.standard_normal(n_assets)
                p2 = 0.25 * y + rng.standard_normal(n_assets)
                latents[(sp, "US", s)] = (y, p1, p2)

    runs = {}
    for m_idx, name in enumerate(("m1", "m2")):
        rdir = tmp_path / "cost_runs" / name
        for sp in oa.SPLITS:
            _write_split_parquet(
                rdir / f"{sp}-predictions.parquet",
                markets=["US"],
                n_sessions=split_lengths[sp],
                n_assets=n_assets,
                start_day_offset=offsets[sp],
                target_fn=lambda m, s, n, _sp=sp: latents[(_sp, m, s)][0],
                pred_fn=lambda m, s, n, y, _sp=sp, _idx=m_idx: latents[(_sp, m, s)][1 + _idx],
            )
        runs[name] = rdir

    r_zero_cost = oa.run_aggregation(runs, abstention_cost=0.0, min_assets=30)
    r_high_cost = oa.run_aggregation(runs, abstention_cost=0.05, min_assets=30)
    assert r_zero_cost["selected_hyperparams"]["session_abstention"]["threshold"] == -0.02
    assert r_high_cost["selected_hyperparams"]["session_abstention"]["threshold"] == 0.02
    assert (
        r_high_cost["selected_hyperparams"]["session_abstention"]["calibration_coverage"]
        >= oa.MIN_CALL_COVERAGE
    )
    assert r_high_cost["selected_hyperparams"]["session_abstention"]["calibration_coverage"] < 1.0

    # Si toda la calibración es negativa (cualquier umbral en ABSTENTION_THRESHOLDS viola MIN_CALL_COVERAGE),
    # no debe seleccionar un umbral con cobertura < 0.50 sino activar el respaldo conservador always-on.
    runs_neg = {}
    for m_idx, name in enumerate(("m1", "m2")):
        rdir = tmp_path / "neg_runs" / name
        for sp in oa.SPLITS:
            _write_split_parquet(
                rdir / f"{sp}-predictions.parquet",
                markets=["US"],
                n_sessions=split_lengths[sp],
                n_assets=n_assets,
                start_day_offset=offsets[sp],
                target_fn=lambda m, s, n, _sp=sp: latents[(_sp, m, s)][0],
                pred_fn=lambda m, s, n, y, _sp=sp, _idx=m_idx: -latents[(_sp, m, s)][1 + _idx],
            )
        runs_neg[name] = rdir
    r_neg = oa.run_aggregation(runs_neg, abstention_cost=0.02, min_assets=30)
    assert (
        r_neg["selected_hyperparams"]["session_abstention"]["calibration_coverage"]
        >= oa.MIN_CALL_COVERAGE
    )
    assert r_neg["selected_hyperparams"]["session_abstention"]["threshold"] == -2.0


# ---------------------------------------------------------------------------
# Pruebas añadidas al adoptar el módulo (mutantes que sobrevivían a la suite original)
# ---------------------------------------------------------------------------


def test_rank_average_is_the_plain_mean_of_the_session_standardized_ranks(tmp_path):
    """Mata el mutante que pone todo el peso de la media de rangos en un solo modelo."""
    runs = _make_synthetic_runs(tmp_path, n_assets=35, seed=5)
    result = oa.run_aggregation(runs, min_assets=30)
    average = result["aggregators"]["rank_average"]
    weights = average["mean_evaluation_weights"]
    assert all(math.isclose(w, 1 / len(weights), rel_tol=1e-12) for w in weights.values())
    aligned = result["aligned"]["evaluation"]
    expected = {}
    for market, instant, start, end in aligned["sessions"]:
        z = np.column_stack(
            [oa.standardized_ranks(aligned["predictions"][start:end, m]) for m in range(3)]
        )
        # La suma en FP32 decide los empates entre activos como la emisión del agregador.
        combined = oa.combine_session_scores(z, np.full(3, 1 / 3))
        expected[(market, instant)] = oa.rank_ic_vector(combined, aligned["target"][start:end])
    measured = average["evaluation_session_ics"]
    assert set(measured) == set(expected)
    assert max(abs(measured[key] - expected[key]) for key in measured) < 1e-12


def test_the_permuted_labels_placebo_learns_other_weights_than_ridge_cov_active(tmp_path):
    """Mata el mutante que deja sin permutar los IC maduros del placebo."""
    runs = _make_synthetic_runs(tmp_path, n_assets=35, seed=9)
    result = oa.run_aggregation(runs, min_assets=30)
    learned = result["aggregators"]["ridge_cov_active"]["mean_evaluation_weights"]
    placebo = result["placebos"]["placebo_permuted_labels"]["mean_evaluation_weights"]
    assert set(learned) == set(placebo)
    assert max(abs(learned[m] - placebo[m]) for m in learned) > 0.05


def test_ridge_cov_active_uses_the_covariance_known_at_its_own_session_and_no_later_one():
    """Mata el mutante que usa la covarianza madura en lugar de la de la propia sesión."""
    rng = np.random.default_rng(3)
    n, m = 40, 3
    ics = rng.normal(0.05, 0.02, size=(n, m))
    base = np.eye(m) + 0.5 * (np.ones((m, m)) - np.eye(m))
    covs = np.repeat(base[None], n, axis=0)
    options = dict(
        lam=0.5,
        half_life=None,
        session_times=START_2012_US + np.arange(n, dtype=np.int64) * DAY_US,
        session_matures=None,
        maturity_lag=1,
        scored=np.ones(n, dtype=bool),
        use_current_cov=True,
        se_shrinkage=1.0,
    )
    weights = oa.ridge_stacking_weights(ics, covs, **options)
    changed = covs.copy()
    changed[-1] = np.eye(m)
    again = oa.ridge_stacking_weights(ics, changed, **options)
    assert not np.allclose(weights[-1], again[-1])
    np.testing.assert_array_equal(weights[:-1], again[:-1])


def test_the_operable_factor_grid_is_declared_and_its_boundaries_are_flagged(tmp_path):
    """La rejilla de factores operables se pasa al ajustar y avisa si el elegido cae en un borde."""
    runs = _make_synthetic_runs(tmp_path, include_alpha=True, n_assets=35, seed=31)
    narrow = oa.run_aggregation(runs, operable=True, min_assets=30, operable_scales=(0.5, 0.7))
    assert set(narrow["operable"]["scales"].values()) <= {0.5, 0.7}
    assert narrow["operable"]["scales_on_boundary"] == sorted(narrow["operable"]["scales"])
    wide = oa.run_aggregation(
        runs, operable=True, min_assets=30, operable_scales=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    )
    assert set(wide["operable"]["scales"].values()) <= {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}
    with pytest.raises(ValueError, match="rejilla de factores"):
        oa.run_aggregation(runs, operable=True, min_assets=30, operable_scales=(1.0, 0.5))
