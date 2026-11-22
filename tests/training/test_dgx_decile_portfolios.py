"""Carteras por deciles de unas predicciones transversales: rendimiento, rotación y costes."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/dgx_decile_portfolios.py"
SPEC = importlib.util.spec_from_file_location("dgx_decile_portfolios", SCRIPT_PATH)
cd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cd)


def _make_panel(
    n_sessions: int,
    n_assets: int,
    seed: int = 42,
    with_liquidity: bool = False,
    date_sessions: bool = False,
) -> pd.DataFrame:
    """Genera un panel sintético determinista con target normal estándar."""
    rng = np.random.default_rng(seed)
    rows = n_sessions * n_assets
    if date_sessions:
        base = dt.date(2024, 1, 2)
        sess_vals = [base + dt.timedelta(days=i) for i in range(n_sessions)]
        sessions = np.repeat(sess_vals, n_assets)
    else:
        sessions = np.repeat(np.arange(1, n_sessions + 1, dtype=np.int64), n_assets)

    asset_names = [f"A{i:03d}" for i in range(n_assets)]
    assets = np.tile(asset_names, n_sessions)
    targets = rng.normal(loc=0.0, scale=0.02, size=rows)
    predictions = rng.normal(loc=0.0, scale=1.0, size=rows)

    data: dict[str, object] = {
        "session": sessions,
        "asset": assets,
        "prediction": predictions,
        "target": targets,
    }
    if with_liquidity:
        data["liquidity"] = rng.lognormal(mean=2.0, sigma=1.0, size=rows)
    return pd.DataFrame(data)


def test_perfect_positive_and_negative_predictions() -> None:
    """Con pred=target el L/S es positivo y monotonía=1; con pred=-target, negativo y -1."""
    df = _make_panel(n_sessions=25, n_assets=40, seed=101)

    df_pos = df.copy()
    df_pos["prediction"] = df_pos["target"]
    ls_pos = cd.long_short(df_pos, groups=10)
    res_pos = cd.summary(df_pos, groups=10, lags=5)

    assert len(ls_pos) == 25
    assert (ls_pos > 0.0).all()
    assert res_pos["hit_rate"] == pytest.approx(1.0)
    assert res_pos["monotonicity"] == pytest.approx(1.0)
    assert res_pos["mean"] > 0.0

    df_neg = df.copy()
    df_neg["prediction"] = -df_neg["target"]
    ls_neg = cd.long_short(df_neg, groups=10)
    res_neg = cd.summary(df_neg, groups=10, lags=5)

    assert len(ls_neg) == 25
    assert (ls_neg < 0.0).all()
    assert res_neg["hit_rate"] == pytest.approx(0.0)
    assert res_neg["monotonicity"] == pytest.approx(-1.0)
    assert res_neg["mean"] < 0.0


def test_random_prediction_not_significant() -> None:
    """Con predicciones aleatorias independientes del target, |t| < 3."""
    df = _make_panel(n_sessions=120, n_assets=50, seed=2026)
    res = cd.summary(df, groups=10, lags=10)

    assert res["sessions"] == 120
    assert abs(res["t_stat"]) < 3.0
    assert abs(res["t_newey_west"]) < 3.0


def test_turnover_zero_when_extreme_groups_unchanged() -> None:
    """La rotación es 0 cuando los grupos extremos no cambian entre sesiones."""
    rng = np.random.default_rng(7)
    n_sessions = 15
    n_assets = 40
    frames = []
    assets = [f"A{i:03d}" for i in range(n_assets)]
    for sess in range(1, n_sessions + 1):
        # A000..A003 siempre en decil 1; A036..A039 siempre en decil 10.
        # Los intermedios cambian aleatoriamente para confirmar que no alteran la rotación.
        preds = np.empty(n_assets, dtype=np.float64)
        preds[:4] = -10.0 + np.arange(4) * 0.1
        preds[4:36] = rng.uniform(-1.0, 1.0, size=32)
        preds[36:] = 10.0 + np.arange(4) * 0.1
        frames.append(
            pd.DataFrame(
                {
                    "session": sess,
                    "asset": assets,
                    "prediction": preds,
                    "target": rng.normal(0.0, 0.01, size=n_assets),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    to = cd.turnover(df, groups=10, holding="overnight")
    assert math.isnan(to.iloc[0]["long"])
    assert math.isnan(to.iloc[0]["short"])
    assert math.isnan(to.iloc[0]["mean"])
    assert (to.iloc[1:]["long"] == 0.0).all()
    assert (to.iloc[1:]["short"] == 0.0).all()
    assert (to.iloc[1:]["mean"] == 0.0).all()

    res = cd.summary(df, groups=10, holding="overnight")
    assert res["turnover_mean"] == pytest.approx(0.0)
    assert res["two_leg_turnover_mean"] == pytest.approx(0.0)
    assert res["breakeven_cost_bps"] is None


def test_turnover_random_independent_near_0_9() -> None:
    """Con predicciones independientes entre sesiones y 10 grupos, rotación ~ 0,9."""
    df = _make_panel(n_sessions=160, n_assets=100, seed=31415)
    to = cd.turnover(df, groups=10, holding="overnight")

    assert len(to) == 160
    assert to.iloc[0].isna().all()
    assert to["long"].mean() == pytest.approx(0.9, abs=0.025)
    assert to["short"].mean() == pytest.approx(0.9, abs=0.025)
    assert to["mean"].mean() == pytest.approx(0.9, abs=0.02)


def test_turnover_matches_direct_l1_weights_with_varying_universe() -> None:
    """Comprueba la fórmula 0,5 * suma(|w_t - w_{t-1}|) con universos de distinto tamaño."""
    rng = np.random.default_rng(99)
    frames = []
    for sess, n_assets in [(1, 30), (2, 35), (3, 32), (4, 40)]:
        assets = [f"A{i:03d}" for i in range(n_assets)]
        frames.append(
            pd.DataFrame(
                {
                    "session": sess,
                    "asset": assets,
                    "prediction": rng.normal(size=n_assets),
                    "target": rng.normal(0.0, 0.01, size=n_assets),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    grouped = cd.deciles(df, groups=10)
    to = cd.turnover(df, groups=10, holding="overnight")

    sessions = [1, 2, 3, 4]
    for idx in range(1, len(sessions)):
        s_prev, s_curr = sessions[idx - 1], sessions[idx]
        for col, grp in [("long", 10), ("short", 1)]:
            prev_a = grouped.loc[
                (grouped["session"] == s_prev) & (grouped["group"] == grp), "asset"
            ].tolist()
            curr_a = grouped.loc[
                (grouped["session"] == s_curr) & (grouped["group"] == grp), "asset"
            ].tolist()
            w_prev = {a: 1.0 / len(prev_a) for a in prev_a}
            w_curr = {a: 1.0 / len(curr_a) for a in curr_a}
            universe = set(w_prev) | set(w_curr)
            expected = 0.5 * sum(abs(w_curr.get(a, 0.0) - w_prev.get(a, 0.0)) for a in universe)
            assert to.loc[s_curr, col] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_breakeven_cost_and_net_sharpe_hand_calculation() -> None:
    """Verifica coste de equilibrio, Sharpe neto y Newey-West contra cálculo manual exacto."""
    # 5 sesiones x 30 activos (exactamente 3 activos por decil).
    # En cada transición cambian 2 de 3 activos en pata larga y 1 de 3 en pata corta:
    # rotación larga = 2/3, rotación corta = 1/3, media por pata = 0,5, dos patas = 1,0.
    n_assets = 30
    assets = [f"A{i:02d}" for i in range(n_assets)]
    # Definimos los 3 activos de cada pata por sesión para controlar exactamente la rotación.
    long_legs = [
        ["A27", "A28", "A29"],
        ["A27", "A25", "A26"],  # 1 común -> rot = 2/3
        ["A25", "A23", "A24"],  # 1 común -> rot = 2/3
        ["A23", "A21", "A22"],  # 1 común -> rot = 2/3
        ["A21", "A19", "A20"],  # 1 común -> rot = 2/3
    ]
    short_legs = [
        ["A00", "A01", "A02"],
        ["A00", "A01", "A03"],  # 2 comunes -> rot = 1/3
        ["A00", "A03", "A04"],  # 2 comunes -> rot = 1/3
        ["A03", "A04", "A05"],  # 2 comunes -> rot = 1/3
        ["A04", "A05", "A06"],  # 2 comunes -> rot = 1/3
    ]
    # Rendimientos L/S diarios conocidos: [0.0020, 0.0010, 0.0030, 0.0015, 0.0025].
    ls_targets = [0.0020, 0.0010, 0.0030, 0.0015, 0.0025]

    frames = []
    for s_idx in range(5):
        long_set = set(long_legs[s_idx])
        short_set = set(short_legs[s_idx])
        middle = [a for a in assets if a not in long_set and a not in short_set]
        preds = {}
        targs = {}
        for j, a in enumerate(short_legs[s_idx]):
            preds[a] = float(j)  # decil 1
            targs[a] = 0.0
        for j, a in enumerate(middle):
            preds[a] = 10.0 + float(j)  # deciles 2..9
            targs[a] = 0.0001 * (j // 3 + 1)
        for j, a in enumerate(long_legs[s_idx]):
            preds[a] = 100.0 + float(j)  # decil 10
            targs[a] = ls_targets[s_idx]

        frames.append(
            pd.DataFrame(
                {
                    "session": s_idx + 1,
                    "asset": assets,
                    "prediction": [preds[a] for a in assets],
                    "target": [targs[a] for a in assets],
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    costs = (0, 5, 10, 20)
    res = cd.summary(df, groups=10, costs_bps=costs, lags=2, holding="overnight")

    gross = np.array(ls_targets, dtype=np.float64)
    mean_gross = float(np.mean(gross))  # 0.0020 (20 bps)
    std_gross = float(np.std(gross, ddof=1))
    sharpe_gross = (mean_gross / std_gross) * math.sqrt(252.0)
    t_simple = (mean_gross / std_gross) * math.sqrt(5.0)

    # Cálculo manual de Newey-West con lags=2 (núcleo Bartlett y factor T/(T-1)).
    u = gross - mean_gross
    g0 = float(np.dot(u, u)) / 5.0
    g1 = float(np.dot(u[1:], u[:-1])) / 5.0
    g2 = float(np.dot(u[2:], u[:-2])) / 5.0
    omega = g0 + 2.0 * (1.0 - 1.0 / 3.0) * g1 + 2.0 * (1.0 - 2.0 / 3.0) * g2
    t_nw_manual = mean_gross / math.sqrt(omega / 4.0)

    # Rotación: larga = 2/3, corta = 1/3 -> media = 0,5, dos patas = 1,0.
    # Coste de equilibrio = (0.0020 / 1.0) * 10000 = 20.0 bps.
    paid_two_leg = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    assert res["sessions"] == 5
    assert res["mean"] == pytest.approx(mean_gross, rel=1e-12, abs=1e-12)
    assert res["std"] == pytest.approx(std_gross, rel=1e-12, abs=1e-12)
    assert res["t_stat"] == pytest.approx(t_simple, rel=1e-12, abs=1e-12)
    assert res["t_newey_west"] == pytest.approx(t_nw_manual, rel=1e-12, abs=1e-12)
    assert res["sharpe_gross"] == pytest.approx(sharpe_gross, rel=1e-12, abs=1e-12)
    assert res["hit_rate"] == pytest.approx(1.0)
    assert res["turnover_mean"] == pytest.approx(0.5, rel=1e-12, abs=1e-12)
    assert res["two_leg_turnover_mean"] == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert res["breakeven_cost_bps"] == pytest.approx(20.0, rel=1e-12, abs=1e-12)
    assert res["monotonicity"] == pytest.approx(1.0)

    for c_bps in costs:
        net = gross - (c_bps / 10_000.0) * paid_two_leg
        expected_net_sharpe = (float(np.mean(net)) / float(np.std(net, ddof=1))) * math.sqrt(252.0)
        assert res["net_sharpe"][c_bps] == pytest.approx(expected_net_sharpe, rel=1e-12, abs=1e-12)

    # Con lags=0, Newey-West coincide exactamente con t simple.
    res_lag0 = cd.summary(df, groups=10, costs_bps=(5,), lags=0)
    assert res_lag0["t_newey_west"] == pytest.approx(res_lag0["t_stat"], rel=1e-12, abs=1e-12)


def test_liquidity_weighting_and_top_third_summary() -> None:
    """La ponderación por liquidez iguala a pesos iguales con liquidez constante."""
    df_const = _make_panel(n_sessions=12, n_assets=36, seed=55)
    df_const["liquidity"] = 123.45

    ls_eq = cd.long_short(df_const, groups=10, weighting="equal")
    ls_liq = cd.long_short(df_const, groups=10, weighting="liquidity")
    pd.testing.assert_series_equal(ls_eq, ls_liq)

    # Con liquideces distintas, el resultado difiere y coincide con el promedio ponderado.
    df_var = _make_panel(n_sessions=3, n_assets=30, seed=56, with_liquidity=True)
    ls_var = cd.long_short(df_var, groups=10, weighting="liquidity")
    grouped = cd.deciles(df_var, groups=10)
    for sess in [1, 2, 3]:
        s_long = grouped.loc[(grouped["session"] == sess) & (grouped["group"] == 10)]
        s_short = grouped.loc[(grouped["session"] == sess) & (grouped["group"] == 1)]
        expected = float(
            np.average(s_long["target"], weights=s_long["liquidity"])
            - np.average(s_short["target"], weights=s_short["liquidity"])
        )
        assert ls_var.loc[sess] == pytest.approx(expected, rel=1e-12, abs=1e-12)

    # Resumen con liquidity incluye liquid_top_third evaluado en el tercio más líquido.
    # 90 activos por sesión -> el tercio más líquido tiene exactamente 30 activos (>= 30).
    df_90 = _make_panel(n_sessions=10, n_assets=90, seed=77)
    # Los 30 activos más líquidos tienen señal perfecta (pred=target), los 60 menos líquidos
    # tienen señal invertida (pred=-target).
    liq = np.tile(np.arange(1.0, 91.0), 10)
    df_90["liquidity"] = liq
    df_90["prediction"] = np.where(df_90["liquidity"] > 60.0, df_90["target"], -df_90["target"])

    res_liq = cd.summary(df_90, groups=10, lags=3)
    assert "liquid_top_third" in res_liq
    top3 = res_liq["liquid_top_third"]
    assert top3["sessions"] == 10
    assert top3["mean"] > 0.0
    assert top3["hit_rate"] == pytest.approx(1.0)
    assert top3["monotonicity"] == pytest.approx(1.0)


def test_min_30_assets_and_group_balance_with_ties() -> None:
    """Sesiones con < 30 activos no cuentan; grupos equilibrados y empates por rango medio."""
    rng = np.random.default_rng(12)
    frames = []
    for sess, count in [(1, 10), (2, 29), (3, 30), (4, 35)]:
        frames.append(
            pd.DataFrame(
                {
                    "session": sess,
                    "asset": [f"A{i:02d}" for i in range(count)],
                    "prediction": rng.normal(size=count),
                    "target": rng.normal(0.0, 0.01, size=count),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    dec = cd.deciles(df, groups=10)
    assert sorted(dec["session"].unique().tolist()) == [3, 4]
    assert cd.long_short(df, groups=10).index.tolist() == [3, 4]
    assert cd.turnover(df, groups=10).index.tolist() == [3, 4]
    assert cd.summary(df, groups=10)["sessions"] == 2

    # Comprueba tamaños equilibrados (diferencia máxima de 1 activo por grupo sin empates).
    sizes_s4 = dec.loc[dec["session"] == 4, "group"].value_counts()
    assert len(sizes_s4) == 10
    assert int(sizes_s4.max() - sizes_s4.min()) <= 1

    # Comprueba empates con rango medio: activos empatados reciben exactamente el mismo grupo.
    preds_ties = np.arange(30, dtype=np.float64)
    preds_ties[2:5] = 2.0  # Posiciones 3, 4, 5 -> rango medio 4.0 -> todos en grupo 2.
    df_ties = pd.DataFrame(
        {
            "session": 1,
            "asset": [f"A{i:02d}" for i in range(30)],
            "prediction": preds_ties,
            "target": np.linspace(-0.01, 0.01, 30),
        }
    )
    dec_ties = cd.deciles(df_ties, groups=10)
    tied_groups = dec_ties.loc[dec_ties["asset"].isin(["A02", "A03", "A04"]), "group"]
    assert tied_groups.nunique() == 1
    assert int(tied_groups.iloc[0]) == 2


def test_validation_rejects_duplicates_nonfinite_and_invalid_inputs() -> None:
    """Rechaza duplicados, no finitos, columnas faltantes y liquidity <= 0 con ValueError."""
    base = _make_panel(n_sessions=2, n_assets=30, seed=3)

    # Duplicado dentro de una sesión (incluso en un DataFrame pequeño).
    dup = pd.DataFrame(
        {
            "session": [1, 1],
            "asset": ["A01", "A01"],
            "prediction": [0.1, 0.2],
            "target": [0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="duplicados"):
        cd.deciles(dup)

    # Valores no finitos (NaN, +inf, -inf) en prediction, target y liquidity.
    for col in ("prediction", "target"):
        for bad_val in (np.nan, np.inf, -np.inf):
            bad_df = base.copy()
            bad_df.loc[0, col] = bad_val
            with pytest.raises(ValueError, match="no finitos"):
                cd.summary(bad_df)

    bad_liq = base.copy()
    bad_liq["liquidity"] = 1.0
    bad_liq.loc[0, "liquidity"] = np.inf
    with pytest.raises(ValueError, match="no finitos"):
        cd.summary(bad_liq)

    # liquidity <= 0 con weighting="liquidity" y ausencia de liquidity.
    with pytest.raises(ValueError, match="liquidity"):
        cd.long_short(base, weighting="liquidity")

    zero_liq = base.copy()
    zero_liq["liquidity"] = 1.0
    zero_liq.loc[5, "liquidity"] = 0.0
    with pytest.raises(ValueError, match="liquidity > 0"):
        cd.long_short(zero_liq, weighting="liquidity")

    # Columna obligatoria faltante.
    with pytest.raises(ValueError, match="obligatorias"):
        cd.deciles(base.drop(columns=["target"]))

    # Parámetros inválidos.
    with pytest.raises(ValueError):
        cd.deciles(base, groups=1)
    with pytest.raises(ValueError):
        cd.long_short(base, weighting="invalid")
    with pytest.raises(ValueError):
        cd.summary(base, lags=-1)
    with pytest.raises(ValueError):
        cd.summary(base, costs_bps=(-5,))


def test_determinism_and_no_lookahead() -> None:
    """Resultados deterministas ante permutación de filas y sin usar información futura."""
    df = _make_panel(n_sessions=12, n_assets=90, seed=404, with_liquidity=True, date_sessions=True)
    res_a = cd.summary(df, groups=10, costs_bps=(5, 10, 20), lags=4)

    shuffled = df.sample(frac=1.0, random_state=9876).reset_index(drop=True)
    res_b = cd.summary(shuffled, groups=10, costs_bps=(5, 10, 20), lags=4)
    assert res_a == res_b

    # Modificar sesiones futuras no debe alterar deciles, L/S ni rotación de sesiones previas.
    cutoff = sorted(df["session"].unique())[5]
    altered = df.copy()
    future_mask = altered["session"] > cutoff
    altered.loc[future_mask, "prediction"] = -altered.loc[future_mask, "prediction"] * 7.0
    altered.loc[future_mask, "target"] = altered.loc[future_mask, "target"] + 0.5
    altered.loc[future_mask, "liquidity"] = altered.loc[future_mask, "liquidity"] * 100.0

    dec_orig = cd.deciles(df, groups=10)
    dec_alt = cd.deciles(altered, groups=10)
    pd.testing.assert_frame_equal(
        dec_orig.loc[dec_orig["session"] <= pd.Timestamp(cutoff)].reset_index(drop=True),
        dec_alt.loc[dec_alt["session"] <= pd.Timestamp(cutoff)].reset_index(drop=True),
    )

    ls_orig = cd.long_short(df, groups=10)
    ls_alt = cd.long_short(altered, groups=10)
    pd.testing.assert_series_equal(
        ls_orig.loc[ls_orig.index <= pd.Timestamp(cutoff)],
        ls_alt.loc[ls_alt.index <= pd.Timestamp(cutoff)],
    )

    to_orig = cd.turnover(df, groups=10)
    to_alt = cd.turnover(altered, groups=10)
    pd.testing.assert_frame_equal(
        to_orig.loc[to_orig.index <= pd.Timestamp(cutoff)],
        to_alt.loc[to_alt.index <= pd.Timestamp(cutoff)],
    )


def test_cli_writes_expected_json_for_parquet_and_csv(tmp_path: Path) -> None:
    """La interfaz CLI lee Parquet y CSV y escribe el JSON exacto esperado."""
    df = _make_panel(n_sessions=15, n_assets=90, seed=808, with_liquidity=True, date_sessions=True)
    parquet_path = tmp_path / "predicciones.parquet"
    csv_path = tmp_path / "predicciones.csv"
    out_parquet_json = tmp_path / "resultado_parquet.json"
    out_csv_json = tmp_path / "resultado_csv.json"

    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    df.to_csv(csv_path, index=False)

    proc_pq = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(parquet_path),
            "--output",
            str(out_parquet_json),
            "--groups",
            "10",
            "--costs",
            "5",
            "10",
            "20",
            "--lags",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_pq.returncode == 0, proc_pq.stderr
    assert out_parquet_json.exists()

    proc_csv = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(csv_path),
            "--output",
            str(out_csv_json),
            "--groups",
            "10",
            "--costs",
            "5",
            "10",
            "20",
            "--lags",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_csv.returncode == 0, proc_csv.stderr
    assert out_csv_json.exists()

    loaded_pq = json.loads(out_parquet_json.read_text(encoding="utf-8"))
    loaded_csv = json.loads(out_csv_json.read_text(encoding="utf-8"))
    expected = json.loads(
        json.dumps(cd.summary(df, groups=10, costs_bps=(5, 10, 20), lags=5), allow_nan=False)
    )

    assert loaded_pq == expected
    assert loaded_csv["sessions"] == expected["sessions"]
    assert loaded_csv["mean"] == pytest.approx(expected["mean"], rel=1e-12, abs=1e-12)
    assert loaded_csv["sharpe_gross"] == pytest.approx(
        expected["sharpe_gross"], rel=1e-12, abs=1e-12
    )
    assert loaded_csv["breakeven_cost_bps"] == pytest.approx(
        expected["breakeven_cost_bps"], rel=1e-12, abs=1e-12
    )
    assert loaded_csv["monotonicity"] == pytest.approx(
        expected["monotonicity"], rel=1e-12, abs=1e-12
    )
    assert "liquid_top_third" in loaded_pq
    assert set(loaded_pq["net_sharpe"].keys()) == {"5", "10", "20"}
    assert set(loaded_pq["group_means"].keys()) == {str(i) for i in range(1, 11)}


def test_edge_cases_small_and_degenerate_inputs(tmp_path: Path) -> None:
    """Comprueba casos límite: sin sesiones válidas, una sesión, empates totales y errores CLI."""
    # Todas las sesiones tienen menos de 30 activos.
    small = _make_panel(n_sessions=3, n_assets=20, seed=11, with_liquidity=True)
    assert cd.deciles(small).empty
    assert cd.long_short(small).empty
    assert cd.turnover(small).empty
    res_empty = cd.summary(small)
    assert res_empty["sessions"] == 0
    assert res_empty["mean"] is None
    assert res_empty["breakeven_cost_bps"] is None
    assert res_empty["liquid_top_third"]["sessions"] == 0

    # Una única sesión válida: media y monotonía definidas; std es None; rotación intraday = 2.0,
    # overnight = None.
    one = _make_panel(n_sessions=1, n_assets=30, seed=13)
    one["prediction"] = one["target"]
    res_one = cd.summary(one, holding="overnight")
    assert res_one["sessions"] == 1
    assert res_one["mean"] > 0.0
    assert res_one["std"] is None
    assert res_one["turnover_mean"] is None
    assert res_one["monotonicity"] == pytest.approx(1.0)
    res_one_intraday = cd.summary(one, holding="intraday")
    assert res_one_intraday["turnover_mean"] == pytest.approx(2.0)
    assert res_one_intraday["two_leg_turnover_mean"] == pytest.approx(4.0)

    # Predicciones todas empatadas: todos los activos van al mismo grupo intermedio.
    tied = _make_panel(n_sessions=2, n_assets=30, seed=17)
    tied["prediction"] = 0.0
    dec_tied = cd.deciles(tied, groups=10)
    assert dec_tied["group"].nunique() == 1
    res_tied = cd.summary(tied, groups=10)
    assert res_tied["sessions"] == 0
    assert res_tied["dropped_sessions_ties"] == 2

    # Tipos inválidos de session y asset.
    bad_sess = one.copy()
    bad_sess["session"] = 1.5
    with pytest.raises(ValueError, match="session"):
        cd.deciles(bad_sess)

    bad_asset = one.copy()
    bad_asset["asset"] = np.arange(len(bad_asset))
    with pytest.raises(ValueError, match="asset"):
        cd.deciles(bad_asset)

    # Error limpio en CLI cuando el archivo de entrada tiene datos inválidos.
    bad_csv = tmp_path / "malo.csv"
    bad_df = one.copy()
    bad_df.loc[0, "prediction"] = np.nan
    bad_df.to_csv(bad_csv, index=False)
    out_json = tmp_path / "no_debe_crearse.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(bad_csv), "--output", str(out_json)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "Error:" in proc.stderr
    assert not out_json.exists()


def test_intraday_turnover_is_roundtrip_and_markets_separated() -> None:
    """intraday da rotación 2.0 por pata, separa US y CN sin cruces y cuenta empates."""
    us = _make_panel(n_sessions=6, n_assets=30, seed=11)
    us["market"] = "US"
    # Activos fijos en los extremos para que overnight sea 0.0 dentro de cada mercado.
    us["prediction"] = np.tile(np.arange(30, dtype=np.float64), 6)
    us["z_return"] = us["target"] + 0.002

    cn = _make_panel(n_sessions=6, n_assets=30, seed=22)
    cn["market"] = "CN"
    cn["asset"] = "CN_" + cn["asset"]
    cn["prediction"] = np.tile(np.arange(30, dtype=np.float64), 6)
    cn["z_return"] = cn["target"] - 0.001

    joint = pd.concat([us, cn], ignore_index=True)

    # Por defecto holding='intraday': cada sesión paga ida y vuelta (2.0 por pata, 4.0 dos patas).
    to_intra = cd.turnover(joint, groups=10)
    assert (to_intra["long"] == 2.0).all()
    assert (to_intra["short"] == 2.0).all()
    assert (to_intra["mean"] == 2.0).all()

    res_intra = cd.summary(joint, groups=10, return_col="z_return")
    assert res_intra["holding"] == "intraday"
    assert res_intra["return_col"] == "z_return"
    assert res_intra["turnover_mean"] == pytest.approx(2.0)
    assert res_intra["two_leg_turnover_mean"] == pytest.approx(4.0)
    assert set(res_intra["by_market"].keys()) == {"CN", "US"}
    assert res_intra["dropped_sessions_ties"] == 0

    # En overnight con mercados US+CN, la primera sesión de CN y la primera de US son NaN
    # (nunca se calcula transición CN -> US) y el resto tiene rotación 0.0.
    to_over = cd.turnover(joint, groups=10, holding="overnight")
    assert math.isnan(to_over.loc[("CN", 1), "mean"])
    assert math.isnan(to_over.loc[("US", 1), "mean"])
    assert (to_over.loc[("CN", slice(2, 6)), "mean"] == 0.0).all()
    assert (to_over.loc[("US", slice(2, 6)), "mean"] == 0.0).all()

    # Empates que colapsan un decil extremo incrementan dropped_sessions_ties explícitamente.
    tied_panel = us.copy()
    tied_panel.loc[tied_panel["session"] == 3, "prediction"] = 0.0
    res_tie = cd.summary(tied_panel, groups=10)
    assert res_tie["sessions"] == 5
    assert res_tie["dropped_sessions_ties"] == 1
    assert res_tie["dropped_sessions_nan"] == 0

    # Pesos de Bartlett cuando lags >= n: usa max_lag = n - 1 -> 1 - lag / (max_lag + 1).
    vals = np.array([0.01, 0.02, 0.03, 0.015], dtype=np.float64)
    t_got = cd._newey_west_t_stat(vals, lags=10)
    u = vals - np.mean(vals)
    max_lag = 3
    lrv = float(np.dot(u, u)) / 4.0 + sum(
        2.0 * (1.0 - k / (max_lag + 1)) * float(np.dot(u[k:], u[:-k])) / 4.0
        for k in range(1, max_lag + 1)
    )
    t_exp = float(np.mean(vals)) / math.sqrt(lrv / 3.0)
    assert t_got == pytest.approx(t_exp, rel=1e-12, abs=1e-12)


def test_survivorship_delisting_stress_test_grid_and_gate() -> None:
    """Verifica la prueba de estrés de sesgo de supervivencia (2/5/8 % x 0/-30/-100 %, Gate 4.4)."""
    df = _make_panel(n_sessions=60, n_assets=50, seed=2025)
    # Señal fuerte para que supere el umbral de 5 % de bajas anuales con choque de -30 %.
    df["prediction"] = df["target"] + 0.005
    df["target"] = df["target"] + 0.004

    stress = cd.survivorship_stress_test(
        df,
        annual_hazards=(0.02, 0.05, 0.08),
        terminal_returns=(0.0, -0.30, -1.00),
        groups=10,
        costs_bps=(5, 10, 20),
        seed=42,
    )
    assert "baseline" in stress and "scenarios" in stress
    assert len(stress["scenarios"]) == 9
    s_5_30 = stress["scenarios"]["hazard_0.05_return_-0.30"]
    s_8_100 = stress["scenarios"]["hazard_0.08_return_-1.00"]
    assert s_5_30["delisted_assets"] > 0
    assert s_5_30["surviving_rows_fraction"] < 1.0
    assert s_8_100["surviving_rows_fraction"] <= s_5_30["surviving_rows_fraction"]
    assert isinstance(stress["passes_5pct_minus30pct_gate"], bool)
