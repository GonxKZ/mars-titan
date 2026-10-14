"""Integridad de una comparación publicada, desde las filas y desde los agregados por ventana.

Reutiliza el estudio de juguete de `tests/evaluation/test_walk_forward_comparison.py`, cuyas
predicciones se escriben con valores conocidos. Cada mutación rompe a propósito el código
principal de la comparación o un artefacto publicado y comprueba que la integridad lo
detecta. No hay modelos ajustados ni pasos de optimizador.
"""

import copy
import json
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import forecast_scores, window_aggregates
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.integrity import comparison_integrity as integrity
from tests.evaluation.test_comparison_sources import save
from tests.evaluation.test_walk_forward_comparison import Study


def publish(study, output, **options):
    walk.write_walk_forward(study.config_path, study.sources_path, study.scope, output, **options)
    return output


def window_sources(study, window):
    """Manifiesto y configuración limitados a una ventana, como en la retención v2."""
    limited = copy.deepcopy(study.sources)
    limited["windows"] = {window: limited["windows"][window]}
    for seeds in limited["arms"].values():
        for seed, windows in seeds.items():
            seeds[seed] = {window: windows[window]}
    path = study.sources_path.with_name(f"{window}.json")
    save(path, limited)
    config = walk.restrict_windows(walk.load_config(study.config_path), study.scope, [window])
    sources = walk.load_sources(path, config, study.scope)
    return walk.scope_config(config, study.scope), sources


def aggregate(study, folder):
    """Agregados y registro de integridad de cada ventana del estudio."""
    return {
        window: integrity.write_checked(folder, *window_sources(study, window), window)
        for window in study.sources["windows"]
    }


def biased_mean(monkeypatch):
    """Error de agregación en el código principal: la primera media de cada panel se encoge."""
    original = forecast_scores._mean
    calls = {"n": 0}

    def biased(panel, values):
        calls["n"] += 1
        result = original(panel, values)
        if calls["n"] % 2 == 1:
            result = result * panel.session_samples / (panel.session_samples + 1)
        return result

    monkeypatch.setattr(forecast_scores, "_mean", biased)
    return original


@pytest.mark.parametrize("scope", ["US", "US+CN"])
def test_a_faithful_comparison_passes_from_its_rows(tmp_path, scope):
    study = Study(tmp_path / "study", scope=scope)
    output = publish(study, tmp_path / "result")
    result = integrity.published(study.config_path, study.sources_path, scope, output)
    assert result["passed"], json.dumps(result, indent=1)[:3000]
    assert result["rows"]["passed"] and result["summaries"]["passed"]
    # Ridge (1 semilla), gru y titans (2 semillas) y el control cero, en 2 ventanas.
    files = result["scores"]["files"]
    assert len(files) == 2 * (1 + 2 + 2) + 2
    assert {entry["arm"] for entry in files if entry["seed"] is None} == {"zero"}
    assert result["excluded_rows"] == dict(files=0, differing=[], passed=True)


def test_a_2024_row_is_caught_even_if_the_comparison_stops_checking_segments(tmp_path, monkeypatch):
    def move(arm, seed, fold, partition, values):
        if (fold, partition) == ("fold-001", "evaluation"):
            moments = values["prediction_at"].to_pylist()
            moments[0] = datetime.fromisoformat("2024-01-02T21:00:00+00:00")
            values["prediction_at"] = pa.array(moments, pa.timestamp("us", tz="UTC"))

    study = Study(tmp_path / "study", scope="US")
    study.changes.append(move)
    study.publish()

    def unchecked(table, window, partition, markets, label):
        market = table["market"].to_numpy(zero_copy_only=False)
        return market, table["prediction_at"].cast(pa.int64()).to_numpy()

    # La comparación principal deja de exigir el tramo y la reserva de 2024.
    monkeypatch.setattr(walk, "_check_segment", unchecked)
    output = publish(study, tmp_path / "result")
    result = integrity.published(study.config_path, study.sources_path, "US", output)
    assert not result["passed"] and not result["rows"]["passed"]
    moved = [entry for entry in result["rows"]["files"] if not entry["passed"]]
    assert len(moved) == 5 and all(entry["rows_in_2024"] == 1 for entry in moved)


def test_a_biased_primary_metric_is_caught_by_the_recalculation(tmp_path, monkeypatch):
    study = Study(tmp_path / "study", scope="US")
    original = biased_mean(monkeypatch)
    output = publish(study, tmp_path / "result")
    monkeypatch.setattr(forecast_scores, "_mean", original)
    result = integrity.published(study.config_path, study.sources_path, "US", output)
    assert result["rows"]["passed"] and not result["scores"]["passed"]
    assert not result["passed"]


def test_a_comparison_from_aggregates_passes_with_the_record_of_each_window(tmp_path):
    study = Study(tmp_path / "study", scope="US+CN")
    folder = tmp_path / "aggregates"
    written = aggregate(study, folder)
    for window, (stored, checked) in written.items():
        record = json.loads(checked["path"].read_text())
        assert record["passed"] and record["window"] == window
        assert record["aggregates"]["sha256"] == stored["sha256"]
        assert checked["path"] == folder / "US+CN" / f"{window}.integrity.json"
    output = publish(study, tmp_path / "result", aggregates=folder)
    result = integrity.published(
        study.config_path, study.sources_path, "US+CN", output, aggregates=folder
    )
    assert result["passed"], json.dumps(result, indent=1)[:3000]
    assert result["from_aggregates"] and set(result["windows"]["records"]) == set(written)
    # Las sesiones publicadas son las de los agregados, también las del control cero.
    assert len(result["scores"]["files"]) == 2 * (1 + 2 + 2 + 1)


def test_a_window_that_fails_its_recalculation_stops_before_its_rows_are_released(
    tmp_path, monkeypatch
):
    study = Study(tmp_path / "study", scope="US")
    folder = tmp_path / "aggregates"
    config, sources = window_sources(study, "fold-000")
    biased_mean(monkeypatch)
    with pytest.raises(ValueError, match="La integridad de US fold-000 falla"):
        integrity.write_checked(folder, config, sources, "fold-000")
    record = json.loads((folder / "US" / "fold-000.integrity.json").read_text())
    assert record["passed"] is False and record["rows"]["passed"]
    assert record["scores"]["failures"] > 0


def _tamper_record(path, change):
    record = json.loads(path.read_text())
    change(record)
    atomic_json(path, record)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "falta el registro de integridad"),
        (lambda record: record.update(passed=False), "el registro registra fallos"),
        (
            lambda record: record["aggregates"].update(sha256="0" * 64),
            "el registro describe otros agregados",
        ),
        (
            lambda record: record["identity"]["code"].update(
                {"evaluation/forecast_scores.py": "0" * 64}
            ),
            "el registro procede de otras fuentes o código",
        ),
    ],
    ids=["missing", "failed", "other-aggregates", "other-code"],
)
def test_the_publication_needs_a_passed_record_of_the_current_aggregates(tmp_path, change, message):
    study = Study(tmp_path / "study", scope="US")
    folder = tmp_path / "aggregates"
    aggregate(study, folder)
    output = publish(study, tmp_path / "result", aggregates=folder)
    path = folder / "US" / "fold-001.integrity.json"
    if change == "missing":
        path.unlink()
    else:
        _tamper_record(path, change)
    result = integrity.published(
        study.config_path, study.sources_path, "US", output, aggregates=folder
    )
    assert not result["passed"] and not result["windows"]["passed"]
    assert result["windows"]["failures"] == [f"fold-001: {message}"]
    # La otra ventana sigue superada y las sesiones publicadas no cambian.
    assert result["windows"]["records"]["fold-000"]["passed"] and result["scores"]["passed"]


def test_published_sessions_must_be_exactly_those_of_the_aggregates(tmp_path):
    study = Study(tmp_path / "study", scope="US")
    folder = tmp_path / "aggregates"
    aggregate(study, folder)
    output = publish(study, tmp_path / "result", aggregates=folder)
    # Un solo ulp de diferencia en una sesión publicada, con la huella del informe al día.
    table = pq.read_table(output / "sessions.parquet")
    mse = table.column("mse").to_numpy().copy()
    mse[0] = np.nextafter(mse[0], np.inf)
    table = table.set_column(table.schema.get_field_index("mse"), "mse", pa.array(mse))
    pq.write_table(table, output / "sessions.parquet", compression="zstd")
    report = json.loads((output / "comparison.json").read_text())
    report["artifacts"]["sessions.parquet"] = sha256(output / "sessions.parquet")
    atomic_json(output / "comparison.json", report)
    result = integrity.published(
        study.config_path, study.sources_path, "US", output, aggregates=folder
    )
    assert not result["passed"] and not result["scores"]["passed"]
    failing = [entry for entry in result["scores"]["files"] if not entry["passed"]]
    assert len(failing) == 1 and failing[0]["columns"]["mse"]["mismatched_sessions"] == 1


def test_a_record_needs_sources_limited_to_its_window(tmp_path):
    study = Study(tmp_path / "study", scope="US")
    folder = tmp_path / "aggregates"
    config = walk.scope_config(walk.load_config(study.config_path), "US")
    sources = walk.load_sources(study.sources_path, walk.load_config(study.config_path), "US")
    window_aggregates.write(folder, config, sources, "fold-000")
    with pytest.raises(ValueError, match="limitarse a la ventana"):
        integrity.write_window(folder, config, sources, "fold-000")


def test_rows_the_report_says_it_dropped_must_match_the_recount(tmp_path):
    study = Study(tmp_path / "study", scope="US+CN")
    output = publish(study, tmp_path / "result")
    report = json.loads((output / "comparison.json").read_text())
    window = report["arms"]["gru"]["seeds"]["42"]["windows"]["fold-001"]
    window["excluded_rows"] = {"evaluation": {"CN": 1}}
    atomic_json(output / "comparison.json", report)
    result = integrity.published(study.config_path, study.sources_path, "US+CN", output)
    assert not result["passed"] and result["rows"]["passed"]
    assert result["excluded_rows"]["differing"] == ["gru/42/fold-001/evaluation"]


@pytest.mark.parametrize("from_aggregates", [False, True], ids=["rows", "aggregates"])
def test_a_published_summary_must_follow_from_its_sessions(tmp_path, from_aggregates):
    study = Study(tmp_path / "study", scope="US")
    folder = tmp_path / "aggregates"
    options = {}
    if from_aggregates:
        aggregate(study, folder)
        options = dict(aggregates=folder)
    output = publish(study, tmp_path / "result", **options)
    # Las sesiones no cambian, pero el MAE publicado de un modelo ya no sale de ellas.
    report = json.loads((output / "comparison.json").read_text())
    point = report["arms"]["gru"]["seeds"]["42"]["overall"]["US"]["summary"]["point"]
    point["mae"] *= 0.9
    atomic_json(output / "comparison.json", report)
    result = integrity.published(study.config_path, study.sources_path, "US", output, **options)
    assert result["scores"]["passed"] and not result["summaries"]["passed"]
    assert not result["passed"] and result["summaries"]["failures"] == 1


def test_window_records_with_liquidity_strata_need_their_edition(tmp_path):
    """Los agregados guardan los estratos de liquidez con la identidad de su edición, así que
    el registro de cada ventana y la comprobación de la publicación la necesitan."""
    from tests.evaluation.liquidity_fixture import section, study_edition

    study = Study(tmp_path / "study", scope="US+CN")
    study.config[walk.LIQUIDITY_FIELD] = section(min_rows=1, min_sessions=1)
    study.publish()
    edition = study_edition(tmp_path / "edition")
    folder = tmp_path / "aggregates"
    for window in study.sources["windows"]:
        config, sources = window_sources(study, window)
        liquidity = walk.liquidity_source(config, edition)
        _, checked = integrity.write_checked(folder, config, sources, window, liquidity=liquidity)
        assert checked["passed"]
    output = publish(study, tmp_path / "result", aggregates=folder, edition=edition)
    report = json.loads((output / "comparison.json").read_text())
    assert report[walk.LIQUIDITY_FIELD]["status"] == "computed"
    options = dict(aggregates=folder)
    result = integrity.published(
        study.config_path, study.sources_path, "US+CN", output, edition=edition, **options
    )
    assert result["passed"], json.dumps(result, indent=1)[:3000]
    # Sin la edición, la identidad de los agregados ya no es la de sus registros.
    with pytest.raises(ValueError, match="liquidity"):
        integrity.published(study.config_path, study.sources_path, "US+CN", output, **options)
