"""Escáner de fugas sobre la matriz tabular de una vista, con un corpus técnico y sin ajustar nada.

Los objetivos del corpus técnico son iguales en todos los activos de una sesión, así que aquí
se reescriben con valores aleatorios. En los gráficos, que están presentes en todas las filas,
una columna lleva el objetivo con ruido a propósito y otra solo ruido.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.data.price_windows import PRICE_WINDOW_CHANNELS
from mars_titan.data.storage import sha256
from mars_titan.integrity import window_leakage
from mars_titan.training import masked_campaign
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.tabular_corpus import _matrix
from tests.training.historical_temporal_fixture import historical_temporal_fixture

# Con 64 activos por sesión, el |IC| medio por azar ronda 0,1, lejos del umbral de 0,2.
ASSETS = 64
LEAK, NOISE = "charts.7", "charts.8"


def presence(market, symbol, row):
    """Noticias, fundamentales y macro alternos para que haya bloques ausentes."""
    index = int(symbol[1:])
    return (row + index) % 2 == 0, (row + index) % 3 != 0, row % 2 == 1


def rewrite(manifest, *, seed=0):
    """Objetivos aleatorios por fila, la fuga en `LEAK` y ruido en `NOISE`.

    Con 64 activos, el |IC| de una columna sin relación tiene una desviación típica de 0,126
    por sesión y el umbral de 0,2 sobre la media de seis sesiones no está tan lejos. Con la
    semilla 20261013, `presence.news` llegó a 0,208 por azar (unas 3,5 desviaciones). Con las
    semillas 0 a 9 las columnas sin fuga quedan entre 0,06 y 0,14 y `LEAK` es la única marcada.
    """
    rng = np.random.default_rng(seed)
    meta = json.loads(manifest.read_text())
    for asset in meta["assets"]:
        folder = Path(meta["roots"]["labels"]) / asset["market"] / asset["symbol"]
        labels = pq.read_table(folder / "labels.parquet")
        rows = labels.to_pylist()
        for row in rows:
            if row["target"] is not None:
                row["target"] = float(rng.normal(0, 0.02))
        pq.write_table(pa.Table.from_pylist(rows, schema=labels.schema), folder / "labels.parquet")
        asset["labels_sha256"] = sha256(folder / "labels.parquet")
        path = Path(meta["roots"]["samples"]) / asset["market"] / asset["symbol"]
        samples = pq.read_table(path / "samples.parquet")
        records = samples.to_pylist()
        targets = {row["sample_row"]: row["target"] for row in rows}
        for number, record in enumerate(records):
            target = targets.get(number)
            noise = rng.normal(0, 1, 2)
            record["charts"][7] = noise[0] if target is None else target + 0.002 * noise[0]
            record["charts"][8] = noise[1]
        pq.write_table(
            pa.Table.from_pylist(records, schema=samples.schema),
            path / "samples.parquet",
            row_group_size=3,
        )
        asset["samples_sha256"] = sha256(path / "samples.parquet")
    manifest.write_text(json.dumps(meta))
    return manifest


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    # Con el contrato de ventanas de la v3.1, los precios llevan el canal de presencia.
    fixture = historical_temporal_fixture(
        tmp_path_factory.mktemp("corpus"),
        assets=ASSETS,
        presence=presence,
        market_absent={"US": []},
    )
    return rewrite(fixture.parent)


@pytest.fixture(scope="module")
def scanned(manifest, tmp_path_factory):
    destination = tmp_path_factory.mktemp("scan") / "out" / "fold-000.json"
    report = window_leakage.scan_window(
        manifest, destination, input_policy=HISTORICAL_MASKED, lag=2
    )
    return destination, report


def dataset(manifest):
    return CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)


def test_the_matrix_is_the_tabular_matrix_with_absent_blocks_missing(manifest, tmp_path):
    keys, matrix, names = window_leakage.window_matrix(
        dataset(manifest), tmp_path / "matrix.npy", batch_size=7
    )
    batches = list(dataset(manifest).batches(partition="validation", batch_size=5, epoch=0, seed=0))
    expected = np.concatenate([_matrix(b, np.float32, presence=True) for b in batches])
    bits = np.concatenate([b["presence"] for b in batches])
    assert matrix.shape == expected.shape and np.isfortran(matrix)
    assert names[: len(PRICE_WINDOW_CHANNELS)] == [f"prices.0.{c}" for c in PRICE_WINDOW_CHANNELS]
    assert names[-5:] == [f"presence.{name}" for name in MODALITIES]
    _, spans = window_leakage.column_layout(batches[0], True)
    assert names[spans["prices"][1] - 1] == f"prices.63.{PRICE_WINDOW_CHANNELS[-1]}"
    for index, name in enumerate(MODALITIES):
        start, stop = spans[name]
        assert stop - start == batches[0]["inputs"][name][0].size
        assert names[start].startswith(f"{name}.") and names[stop - 1].startswith(f"{name}.")
        absent = ~bits[:, index]
        assert np.isnan(matrix[absent, start:stop]).all()
        np.testing.assert_array_equal(matrix[~absent, start:stop], expected[~absent, start:stop])
    assert not bits[:, 1].all() and bits[:, 1].any() and not bits[:, 3].all()
    np.testing.assert_array_equal(matrix[:, -5:], bits.astype(np.float32))
    ids = [s for b in batches for s in b["sample_ids"]]
    assert keys["asset_id"].tolist() == ["/".join(s.split("/")[:2]) for s in ids]
    assert keys["market"].tolist() == [m for b in batches for m in b["market"]]
    np.testing.assert_array_equal(keys["target"], np.concatenate([b["target"] for b in batches]))
    np.testing.assert_array_equal(
        keys["prediction_at"], np.concatenate([b["prediction_at"] for b in batches])
    )
    assert len(keys) == dataset(manifest).manifest["counts"]["validation"]


def test_a_target_copied_into_an_input_column_is_flagged_and_noise_is_not(manifest, scanned):
    destination, report = scanned
    assert report["flagged"] == [LEAK] and not report["passed"]
    columns = {entry["column"]: entry for entry in report["columns"]}
    assert columns[LEAK]["mean_abs_ic"] > 0.9
    assert columns[NOISE]["sessions"] == report["sessions"] and not columns[NOISE]["flagged"]
    assert columns[NOISE]["mean_abs_ic"] < 0.2
    assert report["kind"] == window_leakage.KIND and report["partition"] == "validation"
    assert report["view_sha256"] == sha256(manifest) and report["placebo_lag"] == 2
    assert report["sessions"] >= 5 and report["rows"] == ASSETS * report["sessions"]
    # Precios, fundamentales y macro son iguales en todos los activos del corpus técnico.
    assert 2 <= report["checked_columns"] < report["features"] == len(report["columns"])
    assert report["last_prediction_at"] < "2024"
    assert not list(destination.parent.glob(".*"))
    assert json.loads(destination.read_text())["flagged"] == [LEAK]


def fake_scan(calls, outcome):
    """Escaneo sustituto que publica el informe que fija `outcome` para cada ventana."""

    def scan(manifest, destination, *, input_policy, lag, chunk=32):
        window = Path(destination).stem
        calls.append(window)
        flagged, checked = outcome[window]
        report = dict(
            view_sha256=sha256(Path(manifest)),
            placebo_lag=lag,
            rows=10,
            sessions=2,
            features=3,
            checked_columns=checked,
            flagged=flagged,
            columns=[dict(column=LEAK, mean_abs_ic=0.9, max_abs_ic=1.0, past_mean_abs_ic=0.0)],
            seconds=0.0,
        )
        Path(destination).write_text(json.dumps(report))
        return report

    return scan


@pytest.fixture
def scope(tmp_path, monkeypatch):
    """Ámbito de dos ventanas con vistas sustitutas y un escaneo sustituto."""
    windows = {}
    for window in ("fold-000", "fold-001"):
        path = tmp_path / "views" / window / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(window=window)))
        windows[window] = dict(path=str(path), sha256=sha256(path))
    campaign = tmp_path / "campaign.json"
    campaign.write_text("{}")
    monkeypatch.setattr(
        masked_campaign, "load_campaign", lambda path: dict(input_policy=HISTORICAL_MASKED)
    )
    monkeypatch.setattr(
        masked_campaign,
        "scope_views",
        lambda directory, name, loaded: dict(report_sha256="0" * 64, windows=windows),
    )
    calls, outcome = [], {"fold-000": ([], 3), "fold-001": ([LEAK], 3)}
    monkeypatch.setattr(window_leakage, "scan_window", fake_scan(calls, outcome))
    return campaign, windows, calls, outcome


def test_the_scope_summary_lists_each_window_and_resumes_by_view(tmp_path, scope):
    campaign, windows, calls, _ = scope
    output = tmp_path / "scan"
    first = window_leakage.scan_scope(campaign, "US", tmp_path, output, lag=2, windows=["fold-000"])
    assert not first["complete"] and not first["passed"] and first["flagged"] == {}
    summary = window_leakage.scan_scope(campaign, "US", tmp_path, output, lag=2)
    assert calls == ["fold-000", "fold-001"] and summary["complete"]
    assert summary["flagged"] == {"fold-001": [LEAK]} and not summary["passed"]
    assert summary["windows"]["fold-001"]["strongest"][0]["column"] == LEAK
    written = json.loads((output / "US" / "summary.json").read_text())
    assert written["windows"]["fold-000"]["report_sha256"] == sha256(output / "US/fold-000.json")
    assert written["placebo_lag"] == 2 and written["partition"] == "validation"
    # Una vista distinta para la misma ventana se vuelve a escanear.
    path = Path(windows["fold-000"]["path"])
    path.write_text(json.dumps(dict(window="fold-000", changed=True)))
    windows["fold-000"]["sha256"] = sha256(path)
    window_leakage.scan_scope(campaign, "US", tmp_path, output, lag=2)
    assert calls == ["fold-000", "fold-001", "fold-000"]
    with pytest.raises(ValueError, match="retardo"):
        window_leakage.scan_scope(campaign, "US", tmp_path, output, lag=3, windows=["fold-001"])
    with pytest.raises(ValueError, match="ámbito"):
        window_leakage.scan_scope(campaign, "US", tmp_path, output, lag=2, windows=["fold-009"])


def test_a_clean_scope_passes_and_the_cli_returns_its_status(tmp_path, scope):
    campaign, _, _, outcome = scope
    arguments = ["--campaign", str(campaign), "--scope", "US", "--views", str(tmp_path)]
    arguments += ["--lag", "2", "--output", str(tmp_path / "scan")]
    assert window_leakage.main(arguments) == 1
    outcome["fold-001"] = ([], 3)
    assert window_leakage.main([*arguments[:-1], str(tmp_path / "clean")]) == 0
    assert json.loads((tmp_path / "clean/US/summary.json").read_text())["passed"]


def test_a_window_without_checked_columns_does_not_pass(tmp_path, scope):
    campaign, _, _, outcome = scope
    outcome["fold-001"] = ([], 0)
    summary = window_leakage.scan_scope(campaign, "US", tmp_path, tmp_path / "scan", lag=2)
    assert summary["complete"] and summary["flagged"] == {} and not summary["passed"]
    assert summary["unchecked"] == ["fold-001"]


def test_the_scope_runs_the_real_scan_on_each_window(manifest, tmp_path, monkeypatch):
    windows = {"fold-000": dict(path=str(manifest), sha256=sha256(manifest))}
    monkeypatch.setattr(
        masked_campaign, "load_campaign", lambda path: dict(input_policy=HISTORICAL_MASKED)
    )
    monkeypatch.setattr(
        masked_campaign,
        "scope_views",
        lambda directory, name, loaded: dict(report_sha256="0" * 64, windows=windows),
    )
    campaign = tmp_path / "campaign.json"
    campaign.write_text("{}")
    summary = window_leakage.scan_scope(campaign, "US", tmp_path, tmp_path / "scan", lag=2)
    assert summary["flagged"] == {"fold-000": [LEAK]} and summary["complete"]
    assert summary["windows"]["fold-000"]["strongest"][0]["column"] == LEAK
