"""Agregación de errores por semilla sin fabricar réplicas temporales."""

import importlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from tests.tooling.test_neural_plasticity_analysis import panel


def api():
    return importlib.import_module("benchmarks.quantile_multiseed_analysis")


def scored(prediction):
    from benchmarks.neural_plasticity_analysis import checked_scores

    table = panel().set_column(5, "prediction", pa.array(prediction))
    return checked_scores(table).series("mae")


def test_average_uses_errors_per_session_instead_of_cancelling_predictions():
    scores = {42: scored([2.0, 2.0, 11.0]), 43: scored([0.0, 0.0, 9.0])}
    series, summary = api().average_seeds(scores, [42, 43])
    assert series.values.tolist() == [1.0, 1.0]
    assert len(series.values) == 2
    assert summary["mean_session_mae"] == 1.0
    assert summary["seed_sample_std"] == 0.0
    assert summary["seeds"] == [42, 43]
    # El ensemble de estas predicciones sería perfecto y responde a otra pregunta.
    assert scored([1.0, 1.0, 10.0]).values.tolist() == [0.0, 0.0]


def test_seed_variation_is_reported_separately_from_the_number_of_sessions():
    scores = {42: scored([1.0, 1.0, 10.0]), 43: scored([3.0, 3.0, 12.0])}
    series, summary = api().average_seeds(scores, [42, 43])
    assert series.values.tolist() == [1.0, 1.0]
    assert summary["seed_session_mae"] == {"42": 0.0, "43": 2.0}
    assert summary["mean_session_mae"] == 1.0
    assert summary["seed_sample_std"] == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("seeds", [[42], [42, 42], [42, 44], [True, 43], [], [42, 43, 44]])
def test_incomplete_duplicate_or_unidentified_seed_sets_are_rejected(seeds):
    with pytest.raises(ValueError):
        api().average_seeds({42: scored([0.0] * 3), 43: scored([0.0] * 3)}, seeds)


@pytest.mark.parametrize(
    "fault", ["population", "period", "market", "mask", "metric", "loss", "empty"]
)
def test_seed_averages_require_identical_population_sessions_and_masks(fault):
    first = scored([0.0] * 3)
    if fault == "population":
        other = replace(first, cohort_sha256="f" * 64)
    elif fault == "period":
        other = replace(first, session_period=first.session_period + 1)
    elif fault == "market":
        other = replace(first, session_market=first.session_market + 1)
    elif fault == "mask":
        other = replace(first, defined=np.array([True, False]), values=np.array([1.0, 0.0]))
    elif fault == "metric":
        other = replace(first, metric="mse")
    elif fault == "loss":
        other = replace(first, loss=False)
    else:
        first = other = replace(first, defined=np.array([False, False]), values=np.zeros(2))
    with pytest.raises(ValueError):
        api().average_seeds({42: first, 43: other}, [42, 43])


def test_aggregation_uses_session_means_instead_of_row_means():
    first = scored([0.0] * 3)
    series, summary = api().average_seeds({42: first, 43: first}, [42, 43])
    assert summary["mean_session_mae"] == 5.5
    assert len(series.values) == 2


def test_multiseed_analysis_keeps_the_hold_before_opening_inputs(tmp_path, learning_hold):
    from mars_titan.training.learning_hold import LearningHoldError

    learning_hold(False)
    with pytest.raises(LearningHoldError):
        api().analyze(tmp_path / "absent", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def multiseed_fixture(work):
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.training.titans_walk_forward import _protocol
    from tests.tooling.test_neural_plasticity_analysis import complete_fixture

    prepared = complete_fixture(work, "none", "quantile_routing_v1")
    protocol = work / "recipes/protocol.json"
    document = json.loads(protocol.read_text())
    document["seeds"] = [42, 43]
    atomic_json(protocol, document)
    prepared["artifacts"]["recipes/protocol.json"] = sha256(protocol)
    prepared["configuration"]["seeds"] = [42, 43]
    prepared["study_version"] = "quantile_routing_v2"
    old = deepcopy(prepared["jobs"])
    for job in old:
        new = dict(job, id=job["id"].replace("-42", "-43"), seed=43)
        shutil.copytree(work / "runs" / job["id"], work / "runs" / new["id"])
        prepared["jobs"].append(new)
    atomic_json(work / "prepared.json", prepared)
    digest = sha256(work / "prepared.json")
    atomic_json(work / "execution-plan.json", dict(prepared_sha256=digest, prior_receipts=[]))
    template = json.loads((work / "launches" / f"{old[0]['id']}.json").read_text())
    protocol_hash = _protocol(protocol, "fold-000", 42)[1]
    for job in prepared["jobs"]:
        path = work / "runs" / job["id"] / "run.json"
        report = json.loads(path.read_text())
        report["request"].update(seed=job["seed"], protocol_sha256=protocol_hash)
        atomic_json(path, report)
        receipt = dict(template, id=job["id"], job=job, prepared_sha256=digest)
        atomic_json(work / "launches" / f"{job['id']}.json", receipt)
    return prepared


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "unit_second_seed",
        "unit_second_calibration",
        "missing_seed",
        "duplicate",
        "steps",
        "metric",
    ],
)
def test_integral_analysis_checks_every_seed_before_interpreting(
    tmp_path, monkeypatch, learning_doubles, fault
):
    import pyarrow.parquet as pq
    import torch

    from mars_titan.data.storage import atomic_json, sha256

    work = tmp_path / "input"
    work.mkdir()
    prepared = multiseed_fixture(work)
    if fault == "unit_second_seed":
        folder = work / "runs/all_feedback-43"
        path = folder / "checkpoint.fixture"
        state = torch.load(path, weights_only=True)
        state["state"]["model"]["weights"] += 1
        torch.save(state, path)
        report = json.loads((folder / "run.json").read_text())
        report["checkpoint"]["sha256"] = sha256(path)
        atomic_json(folder / "run.json", report)
    elif fault == "missing_seed":
        prepared["jobs"].pop()
    elif fault == "duplicate":
        prepared["jobs"].append(dict(prepared["jobs"][0]))
    elif fault == "unit_second_calibration":
        folder = work / "runs/all_feedback-43"
        report = json.loads((folder / "run.json").read_text())
        path = folder / report["predictions"]["calibration"]["path"]
        table = pq.read_table(path).set_column(5, "prediction", pa.array([2.0, 0.0, 0.0]))
        pq.write_table(table, path)
        report["predictions"]["calibration"]["sha256"] = sha256(path)
        atomic_json(folder / "run.json", report)
    elif fault in {"steps", "metric"}:
        path = work / "runs/both_isolated-43/run.json"
        report = json.loads(path.read_text())
        if fault == "steps":
            report["fit"]["global_step"] = 3
        else:
            report["predictions"]["evaluation"]["metrics"]["session_mae"] = 123.0
        atomic_json(path, report)
    monkeypatch.setattr(api(), "verify_prepared", lambda path: prepared)
    output = tmp_path / "output"
    if fault in {"missing_seed", "duplicate", "steps", "metric"}:
        with pytest.raises(ValueError):
            api().analyze(work, output)
        assert not output.exists()
        return
    result = api().analyze(work, output)
    assert result["primary"]["contrasts"][0]["estimate"] == -5.5
    assert result["primary"]["resampling"]["sessions"] == 2
    assert result["models"]["original"]["seeds"] == [42, 43]
    assert len(result["runs"]) == 10
    assert result["unity_control"]["passed"] is (fault == "none")
    assert result["unity_control"]["by_seed"]["43"]["passed"] is (fault == "none")
    assert result["primary_interpretation"] == (
        "menor_error_en_desarrollo"
        if fault == "none"
        else "atribucion_bloqueada_por_control_unitario"
    )
    assert result["seed_variation"]["paired_delta"]["42"] == -5.5
    assert result["scientific_level"] == "multi_seed_reused_development_pilot"
    assert result["final_test_opened"] is False
    assert len(list(output.glob("*-sessions.parquet"))) == 16
    assert json.loads((output / "results.json").read_text()) == result


def test_command_line_writes_the_same_real_fixture_analysis(
    tmp_path, monkeypatch, learning_doubles, capsys
):
    work = tmp_path / "input"
    work.mkdir()
    prepared = multiseed_fixture(work)
    monkeypatch.setattr(api(), "verify_prepared", lambda path: prepared)
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["analyze", "--work", str(work), "--output", str(output)])
    api().main()
    assert json.loads(capsys.readouterr().out) == dict(models=5, runs=10)
    assert json.loads((output / "results.json").read_text())["unity_control"]["passed"]


def test_wrong_study_version_and_existing_output_are_rejected(
    tmp_path, monkeypatch, learning_doubles
):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        api().analyze(tmp_path / "input", output)
    output.rmdir()
    monkeypatch.setattr(api(), "verify_prepared", lambda path: dict(study_version="v1"))
    with pytest.raises(ValueError, match="protocolo"):
        api().analyze(tmp_path / "input", output)
    assert not output.exists()
