"""Controlador E2 sobre una ventana real de fixture y sin entrenamiento."""

import json
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from tests.memory.spaced_window_fixture import study_fixture


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    return study_fixture(tmp_path_factory.mktemp("spaced-study"))


@pytest.fixture(autouse=True)
def technical_fixtures_only(monkeypatch):
    from mars_titan.training import spaced_inference, spaced_window

    # Se mantienen la supervisión manual y la guarda global de optimizadores de pytest.
    monkeypatch.setattr(spaced_window, "require_learning_allowed", lambda *_: None)
    monkeypatch.setattr(spaced_inference, "require_learning_allowed", lambda *_: None)


def run(window, output, *, arm="utility", **options):
    from mars_titan.training.spaced_window import run_spaced_window

    return run_spaced_window(
        window["view"],
        window["anchor"],
        window["source_manifest"],
        window["study"],
        arm=arm,
        output=output,
        device="cpu",
        indices=window["indices"],
        **options,
    )


def test_temporal_fixture_preserves_all_phases_and_audited_source(window):
    assert window["dataset"].manifest["technical_fixture"] is True
    assert set(window["dataset"].manifest["counts"]) == {
        "train",
        "validation",
        "calibration",
        "evaluation",
    }
    assert all(
        window["dataset"].manifest["counts"][part] > 0
        for part in ("validation", "calibration", "evaluation")
    )
    assert window["source"]["prefixes"].source_id == window["dataset"].identity


def test_controller_produces_the_same_rows_for_baseline_and_original_reservoir(
    window, native, tmp_path, monkeypatch
):
    from mars_titan.training import spaced_inference, spaced_window

    monkeypatch.setattr(spaced_window, "require_learning_allowed", lambda *_: None)
    monkeypatch.setattr(spaced_inference, "require_learning_allowed", lambda *_: None)
    results = []
    for arm in ("baseline", "reservoir"):
        report = spaced_window.run_spaced_window(
            window["view"],
            window["anchor"],
            window["source_manifest"],
            window["study"],
            arm=arm,
            output=tmp_path / arm,
            device="cpu",
            indices=window["indices"],
        )
        assert report["status"] == "completed"
        assert report["final_test_opened"] is False
        assert set(report["predictions"]) == {"validation", "calibration", "evaluation"}
        results.append(report)
    for part in results[0]["predictions"]:
        assert results[0]["predictions"][part]["rows"] == results[1]["predictions"][part]["rows"]
        assert (
            results[0]["predictions"][part]["metrics"]["session_mae"]
            == results[1]["predictions"][part]["metrics"]["session_mae"]
        )
        assert pq.read_table(tmp_path / "baseline" / f"{part}-predictions.parquet").equals(
            pq.read_table(tmp_path / "reservoir" / f"{part}-predictions.parquet")
        )


@pytest.mark.parametrize("arm", ["utility", "permuted_utility", "uniform_frontier", "recent"])
def test_each_retention_policy_completes_the_same_three_phases(window, native, tmp_path, arm):
    report = run(window, tmp_path / arm, arm=arm)
    assert report["status"] == "completed"
    for name, result in report["predictions"].items():
        assert result["rows"] == window["dataset"].manifest["counts"][name]
        assert result["evidence"] is not None


def test_completed_window_verifies_files_without_repeating_predictions(
    window, native, tmp_path, monkeypatch
):
    from mars_titan.training.spaced_inference import SpacedInference

    output = tmp_path / "run"
    expected = run(window, output)
    monkeypatch.setattr(
        SpacedInference, "evaluate", lambda *_a, **_k: pytest.fail("Repitió una fase")
    )
    assert run(window, output) == expected


def test_interruption_keeps_confirmed_phase_and_resumes_the_next_one(window, native, tmp_path):
    output = tmp_path / "run"

    class StopAfterValidation:
        @property
        def requested(self):
            return (output / "validation-predictions.parquet").exists()

    first = run(window, output, stop=StopAfterValidation())
    assert first["status"] == "paused" and set(first["predictions"]) == {"validation"}
    recorded = (output / "validation-predictions.parquet").stat().st_mtime_ns
    resumed = run(window, output)
    assert resumed["status"] == "completed"
    assert (output / "validation-predictions.parquet").stat().st_mtime_ns == recorded


@pytest.mark.parametrize(
    "filename",
    [
        "evaluation-predictions.parquet",
        "evaluation-evidence.json",
        "evaluation-session-metrics.parquet",
    ],
)
def test_modified_outputs_are_rejected_without_rewriting_them(window, native, tmp_path, filename):
    output = tmp_path / "run"
    run(window, output)
    path = output / filename
    altered = b"{}" if filename.endswith(".json") else b"alterado"
    path.write_bytes(altered)
    with pytest.raises(ValueError):
        run(window, output)
    assert path.read_bytes() == altered


def test_cpu_job_declares_attention_mode_and_restores_the_callers_setting(window, native, tmp_path):
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(True)
    try:
        assert run(window, tmp_path / "run")["status"] == "completed"
        assert torch.backends.mha.get_fastpath_enabled() is True
    finally:
        torch.backends.mha.set_fastpath_enabled(before)


def test_owned_preparation_failure_is_recorded_with_its_cost(window, native, tmp_path):
    study = tmp_path / "invalid-capacity.json"
    config = json.loads(window["study"].read_text())
    config["retention"]["capacity"] = 8
    study.write_text(json.dumps(config))
    fixture = dict(window, study=study)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="capacidad"):
        run(fixture, output)
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "failed" and record["attempts"][0]["seconds"] >= 0


def test_phase_row_budget_rejects_the_whole_window_without_truncating(window, native, tmp_path):
    study = tmp_path / "small.json"
    config = json.loads(window["study"].read_text())
    config["compute_budget"]["max_phase_rows"] = 1
    study.write_text(json.dumps(config))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="no se recortan"):
        run(dict(window, study=study), output)
    assert not output.exists()


def test_exhausted_budget_pauses_before_emitting_any_phase(window, native, tmp_path):
    study = tmp_path / "tiny.json"
    config = json.loads(window["study"].read_text())
    config["compute_budget"]["max_seconds"] = 1e-9
    study.write_text(json.dumps(config))
    record = run(dict(window, study=study), tmp_path / "run")
    assert record["status"] == "paused" and not record["predictions"]


def test_already_requested_stop_keeps_a_recoverable_empty_prefix(window, native, tmp_path):
    record = run(window, tmp_path / "run", stop=SimpleNamespace(requested=True))
    assert record["status"] == "paused" and record["predictions"] == {}


def test_per_session_metrics_are_published_with_the_original_aggregation(window, native, tmp_path):
    output = tmp_path / "run"
    record = run(window, output)
    for phase in record["predictions"].values():
        sessions = phase["sessions"]
        table = pq.read_table(output / sessions["path"])
        assert set(table.column_names) == {
            "market",
            "prediction_at",
            "samples",
            "mae",
            "mse",
            "rmse",
        }
        assert sessions["rows"] == table.num_rows == phase["metrics"]["session_count"]
        assert sum(table["samples"].to_pylist()) == phase["rows"]
        assert sum(table["mae"].to_pylist()) / table.num_rows == pytest.approx(
            phase["metrics"]["session_mae"]
        )


@pytest.mark.parametrize("field,value", [("schema_version", True), ("status", "unknown")])
def test_resume_rejects_invalid_report_states(window, native, tmp_path, field, value):
    output = tmp_path / "run"
    run(window, output)
    path = output / "run.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        run(window, output)


def test_control_identity_states_the_actual_policy_and_evidence_switch(window, native, tmp_path):
    for arm in ("baseline", "recent"):
        report = run(window, tmp_path / arm, arm=arm)
        assert report["identity"]["evidence_enabled"] is (arm != "baseline")
        assert report["identity"]["retention"]["policy"] == (
            "reservoir" if arm == "baseline" else arm
        )


def test_exhausted_budget_is_not_reset_by_reopening_the_same_output(window, native, tmp_path):
    study = tmp_path / "tiny.json"
    config = json.loads(window["study"].read_text())
    config["compute_budget"]["max_seconds"] = 1e-9
    study.write_text(json.dumps(config))
    fixture, output = dict(window, study=study), tmp_path / "run"
    first = run(fixture, output)
    second = run(fixture, output)
    assert second["status"] == "paused" and not second["predictions"]
    assert len(second["attempts"]) == 2
    assert sum(v["seconds"] for v in second["attempts"]) >= sum(
        v["seconds"] for v in first["attempts"]
    )


def test_existing_foreign_output_is_left_intact(window, native, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    path = output / "original.txt"
    path.write_text("conservar")
    with pytest.raises(ValueError):
        run(window, output)
    assert path.read_text() == "conservar" and not (output / "run.json").exists()


def test_committed_attempt_cost_prevents_new_predictions_after_exhaustion(
    window, native, tmp_path, monkeypatch
):
    from mars_titan.training.spaced_inference import SpacedInference

    output = tmp_path / "run"

    class StopAfterValidation:
        @property
        def requested(self):
            return (output / "validation-predictions.parquet").exists()

    assert run(window, output, stop=StopAfterValidation())["status"] == "paused"
    path = output / "run.json"
    record = json.loads(path.read_text())
    # Coste simulado del fixture para comprobar el presupuesto sin esperar 900 segundos.
    record["attempts"][0]["seconds"] = 900.0
    path.write_text(json.dumps(record))
    monkeypatch.setattr(
        SpacedInference,
        "evaluate",
        lambda *_a, **_k: pytest.fail("Predijo con el presupuesto agotado"),
    )
    resumed = run(window, output)
    assert resumed["status"] == "paused" and set(resumed["predictions"]) == {"validation"}


def test_prediction_failure_preserves_its_error_and_recoverable_attempt(
    window, native, tmp_path, monkeypatch
):
    from mars_titan.training.spaced_inference import SpacedInference

    def fail(*_a, **_k):
        raise RuntimeError("Fallo técnico de predicción")

    monkeypatch.setattr(SpacedInference, "evaluate", fail)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Fallo técnico"):
        run(window, output)
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "RuntimeError"
    assert report["attempts"][0]["seconds"] >= 0


def test_output_paths_are_checked_before_the_shared_file_verifier(
    window, native, tmp_path, monkeypatch
):
    from mars_titan.training import spaced_window

    output = tmp_path / "run"
    run(window, output)
    path = output / "run.json"
    report = json.loads(path.read_text())
    report["predictions"]["validation"]["path"] = "../outside.parquet"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        spaced_window, "_verify", lambda *_: pytest.fail("Leyó una ruta no validada")
    )
    with pytest.raises(ValueError, match="ruta"):
        run(window, output)
