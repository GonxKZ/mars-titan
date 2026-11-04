"""Controlador regional completo con una vista de fixture y pesos fijados."""

import json

import pyarrow.parquet as pq
import pytest
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_spaced_regional_selected import components
from test_spaced_window import run

from tests.memory.spaced_window_fixture import study_fixture

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@pytest.fixture(autouse=True)
def fixtures_without_scientific_execution(monkeypatch):
    from mars_titan.training import spaced_inference, spaced_window

    monkeypatch.setattr(spaced_window, "require_learning_allowed", lambda *_: None)
    monkeypatch.setattr(spaced_inference, "require_learning_allowed", lambda *_: None)


@pytest.fixture(scope="module")
def regional_window(tmp_path_factory):
    return study_fixture(
        tmp_path_factory.mktemp("regional-window"),
        components=components("tree"),
        reader_contract="regional",
    )


@pytest.mark.parametrize(
    "arm", ["reservoir", "utility", "permuted_utility", "uniform_frontier", "recent"]
)
def test_regional_window_uses_selected_architecture_and_completes_all_three_phases(
    native, regional_window, tmp_path, arm
):
    report = run(regional_window, tmp_path / "run", arm=arm)
    assert report["status"] == "completed"
    assert report["kind"] == "spaced_regional_utility_frozen_window_v1"
    assert report["identity"]["reader_contract"] == "regional"
    for name, prediction in report["predictions"].items():
        assert prediction["rows"] == regional_window["dataset"].manifest["counts"][name]
        assert prediction["metrics"]["spaced_regional_work"]["decision_rows"] > 0


def test_regional_baseline_and_e2_reservoir_keep_exactly_the_same_prediction_rows(
    native, regional_window, tmp_path
):
    reports = []
    for arm in ("baseline", "reservoir"):
        report = run(regional_window, tmp_path / arm, arm=arm)
        assert report["status"] == "completed"
        reports.append(report)
    for phase in reports[0]["predictions"]:
        left = pq.read_table(tmp_path / "baseline" / f"{phase}-predictions.parquet")
        right = pq.read_table(tmp_path / "reservoir" / f"{phase}-predictions.parquet")
        assert left.equals(right)
    assert (
        reports[0]["identity"]["selected"]["consumer_id"]
        == reports[1]["identity"]["selected"]["consumer_id"]
    )


def test_regional_confirmation_and_resume_preserve_the_completed_prefix(
    native, regional_window, tmp_path
):
    output = tmp_path / "run"

    class AfterValidation:
        @property
        def requested(self):
            return (output / "validation-predictions.parquet").exists()

    first = run(regional_window, output, stop=AfterValidation())
    assert first["status"] == "paused" and set(first["predictions"]) == {"validation"}
    stamp = (output / "validation-predictions.parquet").stat().st_mtime_ns
    final = run(regional_window, output)
    assert final["status"] == "completed"
    assert (output / "validation-predictions.parquet").stat().st_mtime_ns == stamp
    assert run(regional_window, output) == final


def test_regional_run_cannot_be_reopened_as_a_standard_kind(native, regional_window, tmp_path):
    output = tmp_path / "run"
    run(regional_window, output)
    path = output / "run.json"
    report = json.loads(path.read_text())
    report["kind"] = "spaced_utility_frozen_window_v1"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        run(regional_window, output)


def test_standard_study_does_not_adopt_a_regional_anchor(native, regional_window, tmp_path):
    document = json.loads(regional_window["study"].read_text())
    document.pop("reader_contract")
    document.update(
        schema_version=1,
        kind="spaced_utility_frozen_study_v1",
        variant_id="spaced_utility_consolidation_v1",
    )
    path = tmp_path / "standard-study.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="contrato"):
        run(dict(regional_window, study=path), tmp_path / "run")


def test_regional_ledger_file_is_verified_before_a_completed_run_is_reused(
    native, regional_window, tmp_path
):
    output = tmp_path / "run"
    run(regional_window, output)
    path = output / "validation-evidence.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="evidencia"):
        run(regional_window, output)
    assert path.read_text() == "{}"
