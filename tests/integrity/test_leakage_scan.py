"""Escáner de fugas con información futura inyectada a propósito en paneles de prueba.

Los paneles no proceden de ningún modelo ni se ajusta nada. Solo se comprueba que el
escáner marca las entradas que contienen el objetivo y no marca las legítimas.
"""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.integrity import leakage_scan

ASSETS, SESSIONS = 60, 40


def panel(seed=20261010):
    rng = np.random.default_rng(seed)
    rows = []
    for market in ("US", "CN"):
        for session in range(SESSIONS):
            moment = 1_700_000_000_000_000 + session * 86_400_000_000
            for asset in range(ASSETS):
                rows.append((market, f"{market}/{asset:03d}", moment))
    frame = pd.DataFrame(rows, columns=["market", "asset_id", "prediction_at"])
    frame["target"] = rng.normal(0, 0.02, len(frame))
    # Señal débil y realista: correlación de rangos del orden de 0,05.
    frame["weak_signal"] = 0.05 * frame["target"] / 0.02 + rng.normal(0, 1, len(frame))
    frame["noise"] = rng.normal(0, 1, len(frame))
    frame["past_return"] = leakage_scan.placebo_past_target(frame, 1)
    return frame, rng


def by_column(result):
    return {entry["column"]: entry for entry in result["columns"]}


def test_legitimate_columns_are_not_flagged():
    frame, _ = panel()
    result = leakage_scan.scan(frame, ["weak_signal", "noise", "past_return"], lag=1)
    assert result["passed"] and result["flagged"] == []
    columns = by_column(result)
    assert columns["weak_signal"]["mean_abs_ic"] < 0.2
    # Copiar el rendimiento pasado es legítimo: alto con el placebo y bajo con el futuro.
    assert columns["past_return"]["past_mean_abs_ic"] == pytest.approx(1.0)
    assert columns["past_return"]["mean_abs_ic"] < 0.2


@pytest.mark.parametrize(
    "canary",
    [
        lambda f, rng: f["target"] + rng.normal(0, 0.02, len(f)),  # Objetivo con ruido.
        lambda f, rng: np.exp(50 * f["target"]),  # Transformación monótona.
        lambda f, rng: -f["target"].rank(),  # Rango invertido.
    ],
)
def test_columns_that_contain_the_future_target_are_flagged(canary):
    frame, rng = panel()
    frame["canary"] = canary(frame, rng)
    result = leakage_scan.scan(frame, ["noise", "canary"], lag=1)
    assert result["flagged"] == ["canary"] and not result["passed"]
    assert by_column(result)["canary"]["mean_abs_ic"] >= 0.6


def test_a_leak_in_a_single_session_is_flagged_by_the_session_threshold():
    frame, _ = panel()
    frame["sparse_leak"] = frame["noise"]
    one = (frame["market"] == "US") & (frame["prediction_at"] == frame["prediction_at"].max())
    frame.loc[one, "sparse_leak"] = frame.loc[one, "target"]
    entry = by_column(leakage_scan.scan(frame, ["sparse_leak"], lag=1))["sparse_leak"]
    assert entry["flagged"] and entry["max_abs_ic"] == pytest.approx(1.0)
    assert entry["mean_abs_ic"] < 0.2


def test_absences_are_excluded_per_column_and_match_a_direct_spearman():
    frame, rng = panel()
    leak = frame["target"] + rng.normal(0, 0.01, len(frame))
    leak[rng.random(len(frame)) < 0.5] = np.nan
    frame["leak_with_gaps"] = leak
    frame.loc[rng.random(len(frame)) < 0.01, "noise"] = np.inf
    result = leakage_scan.scan(frame, ["leak_with_gaps", "noise"], lag=1)
    entry = by_column(result)["leak_with_gaps"]
    assert entry["flagged"] and entry["rows"] == int(leak.notna().sum())
    expected = [
        group[["leak_with_gaps", "target"]].dropna().corr(method="spearman").iloc[0, 1]
        for _, group in frame.groupby(["market", "prediction_at"])
        if group["leak_with_gaps"].notna().sum() >= 20
    ]
    assert entry["mean_ic"] == pytest.approx(float(np.mean(expected)), rel=1e-12)
    assert by_column(result)["noise"]["rows"] < len(frame)


def test_constant_columns_and_small_sessions_have_no_defined_correlation():
    frame, _ = panel()
    frame["constant"] = 1.0
    result = leakage_scan.scan(frame, ["constant", "noise"], lag=1, min_assets=ASSETS + 1)
    columns = by_column(result)
    assert columns["constant"]["sessions"] == 0 and columns["constant"]["mean_abs_ic"] is None
    assert columns["noise"]["sessions"] == 0 and not result["flagged"]


def test_invalid_inputs_are_rejected():
    frame, _ = panel()
    with pytest.raises(ValueError, match="Opciones"):
        leakage_scan.scan(frame, ["noise"], lag=1, threshold=0.1)
    with pytest.raises(ValueError, match="clave"):
        leakage_scan.scan(frame, ["target"], lag=1)
    with pytest.raises(ValueError, match="retardo"):
        leakage_scan.scan(frame, ["noise"], lag=0)
    frame.loc[0, "target"] = np.nan
    with pytest.raises(ValueError, match="finito"):
        leakage_scan.scan(frame, ["noise"], lag=1)


def test_the_cli_reads_parquet_and_signals_a_leak(tmp_path):
    frame, rng = panel()
    frame["canary"] = frame["target"] + rng.normal(0, 0.005, len(frame))
    path = tmp_path / "inputs.parquet"
    pq.write_table(pa.Table.from_pandas(frame.drop(columns=["past_return"])), path)
    output = tmp_path / "scan.json"
    assert leakage_scan.main([str(path), "--lag", "1", "--output", str(output)]) == 1
    written = json.loads(output.read_text())
    assert written["flagged"] == ["canary"] and written["placebo_lag"] == 1


def test_blocks_of_any_width_give_the_same_findings_as_the_whole_table():
    frame, rng = panel()
    frame["canary"] = frame["target"] + rng.normal(0, 0.02, len(frame))
    frame.loc[rng.random(len(frame)) < 0.2, "weak_signal"] = np.nan
    columns = ["weak_signal", "noise", "past_return", "canary"]
    whole = leakage_scan.scan(frame, columns, lag=1)
    keys = frame[["market", "asset_id", "prediction_at", "target"]]
    for width in (1, 3):
        blocks = (
            (columns[i : i + width], frame[columns[i : i + width]].to_numpy(np.float32))
            for i in range(0, len(columns), width)
        )
        result = leakage_scan.scan_blocks(keys, blocks, lag=1)
        # float32 conserva el orden de estos valores, así que los rangos y las sumas coinciden.
        assert result["columns"] == whole["columns"]
        assert result["flagged"] == whole["flagged"] == ["canary"]


def test_blocks_must_match_the_rows_and_never_repeat_a_column():
    frame, _ = panel()
    keys = frame[["market", "asset_id", "prediction_at", "target"]]
    values = frame[["noise"]].to_numpy()
    with pytest.raises(ValueError, match="filas"):
        leakage_scan.scan_blocks(keys, [(["noise"], values[:-1])], lag=1)
    with pytest.raises(ValueError, match="repetidas"):
        leakage_scan.scan_blocks(keys, [(["noise"], values), (["noise"], values)], lag=1)
    with pytest.raises(ValueError, match="No hay columnas"):
        leakage_scan.scan_blocks(keys, [], lag=1)


def pandas_session_ic(frame, column, target, min_assets):
    """Referencia con pandas: rangos medios por sesión y sumas centradas, columna a columna."""
    sessions = frame.groupby(["market", "prediction_at"], sort=True).ngroup()
    values = frame[column].where(np.isfinite(frame[column]))
    mask = values.notna() & target.notna()
    x = values.where(mask).groupby(sessions).rank(method="average")
    y = target.where(mask).groupby(sessions).rank(method="average")
    dx = x - x.groupby(sessions).transform("mean")
    dy = y - y.groupby(sessions).transform("mean")
    count = mask.groupby(sessions).sum().to_numpy(dtype=np.float64)
    covariance = (dx * dy).groupby(sessions).sum().to_numpy()
    variance_x = (dx * dx).groupby(sessions).sum().to_numpy()
    variance_y = (dy * dy).groupby(sessions).sum().to_numpy()
    defined = (count >= min_assets) & (variance_x > 0) & (variance_y > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(defined, covariance / np.sqrt(variance_x * variance_y), np.nan), count


def test_the_numpy_session_ranks_match_an_independent_pandas_reference():
    frame, rng = panel()
    rows = len(frame)
    # Empates, ausencias, infinitos, una columna constante por sesión y una fuga con huecos.
    frame["ties"] = np.round(rng.normal(0, 1, rows), 1)
    frame["gaps"] = np.where(
        rng.random(rows) < 0.3, np.nan, frame["target"] + rng.normal(0, 0.01, rows)
    )
    frame["infinite"] = np.where(rng.random(rows) < 0.05, np.inf, rng.normal(0, 1, rows))
    frame["per_session"] = frame.groupby(["market", "prediction_at"]).ngroup().astype(float)
    frame["binary"] = (rng.random(rows) < 0.5).astype(float)
    columns = ["ties", "gaps", "infinite", "per_session", "binary", "noise", "past_return"]
    result = {e["column"]: e for e in leakage_scan.scan(frame, columns, lag=1, chunk=3)["columns"]}
    past = leakage_scan.placebo_past_target(frame, 1)
    for column in columns:
        ic, count = pandas_session_ic(frame, column, frame["target"], 20)
        placebo, _ = pandas_session_ic(frame, column, past, 20)
        entry, defined = result[column], np.isfinite(ic)
        assert entry["sessions"] == int(defined.sum()) and entry["rows"] == int(count.sum())
        if not defined.any():
            assert entry["mean_ic"] is None
            continue
        assert entry["mean_ic"] == pytest.approx(float(np.mean(ic[defined])), rel=1e-12, abs=1e-15)
        assert entry["max_abs_ic"] == pytest.approx(float(np.max(np.abs(ic[defined]))), rel=1e-12)
        valid = np.isfinite(placebo)
        assert entry["past_mean_abs_ic"] == pytest.approx(
            float(np.mean(np.abs(placebo[valid]))), rel=1e-12
        )


def small_sessions(seed=7, sessions=6, assets=25):
    """Sesiones de 25 activos, como las que hacen grande un |IC| por azar."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        dict(
            market="US",
            asset_id=[f"US/{a:02d}" for _ in range(sessions) for a in range(assets)],
            prediction_at=np.repeat(np.arange(sessions) * 86_400_000_000, assets),
            target=rng.normal(0, 0.02, sessions * assets),
        )
    )
    values = np.column_stack(
        [frame["target"], rng.normal(size=len(frame)), rng.integers(0, 3, len(frame))]
    )
    return frame, values, ["copy", "noise", "groups"]


def test_the_permutation_stays_inside_each_session():
    codes = np.repeat([0, 1, 2], [5, 9, 4])
    sessions = leakage_scan._Sessions(codes)
    rng = np.random.default_rng(1)
    orders = [leakage_scan._within_sessions(sessions, rng) for _ in range(50)]
    for order in orders:
        assert sorted(order) == list(range(len(codes)))
        np.testing.assert_array_equal(sessions.codes[order], sessions.codes)
    assert len({tuple(order) for order in orders}) > 1


def test_the_null_scores_like_the_scanner_and_is_reproducible():
    frame, values, names = small_sessions()
    result = leakage_scan.permutation_null(frame, values, names, permutations=200, seed=3)
    scanned = by_column(
        leakage_scan.scan(frame.assign(**dict(zip(names, values.T, strict=True))), names, lag=1)
    )
    observed = {entry["column"]: entry for entry in result["observed"]["columns"]}
    for name in names:
        assert observed[name]["mean_abs_ic"] == pytest.approx(scanned[name]["mean_abs_ic"])
        assert observed[name]["flagged"] == scanned[name]["flagged"]
    assert "copy" in result["observed"]["flagged"]
    assert observed["copy"]["positive_sessions"] == 1 and result["observed"]["max_mean_abs_ic"] == 1
    null = result["null"]
    assert len(null["flagged_counts"]) == 200 and null["seed"] == 3
    # Ninguna permutación de 25 activos ordena seis sesiones igual que el objetivo.
    assert null["share_max_at_least_observed"] == 0
    assert 0 < null["max_mean_abs_ic_quantiles"]["0.5"] < 0.5
    again = leakage_scan.permutation_null(frame, values, names, permutations=200, seed=3)
    assert again == result
    other = leakage_scan.permutation_null(frame, values, names, permutations=200, seed=4)
    assert other["null"]["flagged_counts"] != null["flagged_counts"]


def test_the_null_by_blocks_scores_every_column_with_the_same_orders():
    frame, values, names = small_sessions()
    # Una columna con huecos tiene otras filas presentes y otros rangos del objetivo.
    gaps = np.where(np.arange(len(frame)) % 10 == 0, np.nan, values[:, 1])
    values, names = np.column_stack([values, gaps]), [*names, "gaps"]
    settings = leakage_scan._settings({})
    sessions = leakage_scan._Sessions(
        frame.groupby(list(leakage_scan.KEYS), sort=True).ngroup().to_numpy()
    )
    target = frame["target"].to_numpy()[sessions.order]
    ordered = values[sessions.order]
    unused = np.full(len(sessions.starts), np.nan)
    rng = np.random.default_rng(11)
    counts, maxima = [], []
    for _ in range(60):
        order = leakage_scan._within_sessions(sessions, rng)
        ic, present = leakage_scan._session_ic(sessions, ordered, target[order], 20, {})
        found = [
            leakage_scan._finding(name, ic[:, j], present[:, j], unused, settings)
            for j, name in enumerate(names)
        ]
        counts.append(sum(entry["flagged"] for entry in found))
        maxima.append(max(entry["mean_abs_ic"] for entry in found))
    for chunk in (1, 2, 4):
        null = leakage_scan.permutation_null(
            frame, values, names, permutations=60, seed=11, chunk=chunk
        )["null"]
        assert null["flagged_counts"] == counts
        assert null["max_mean_abs_ic_quantiles"] == {
            str(q): float(np.quantile(maxima, q)) for q in (0.05, 0.5, 0.95)
        }


def test_the_null_rejects_bad_inputs():
    frame, values, names = small_sessions()
    with pytest.raises(ValueError, match="al menos una permutación"):
        leakage_scan.permutation_null(frame, values, names, permutations=0, seed=1)
    with pytest.raises(ValueError, match="filas de las claves"):
        leakage_scan.permutation_null(frame, values[:-1], names, permutations=1, seed=1)
    with pytest.raises(ValueError, match="correlación definida"):
        leakage_scan.permutation_null(frame, np.ones_like(values), names, permutations=1, seed=1)
