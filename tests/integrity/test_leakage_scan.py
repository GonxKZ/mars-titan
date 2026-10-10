"""Escáner de fugas con canarios inyectados en paneles escritos en la prueba.

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
