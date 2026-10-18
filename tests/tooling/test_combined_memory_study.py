"""Catálogo, presupuesto y contratos de la comparación combinada, sin aprendizaje."""

import copy
import json
import sys

import pytest

from benchmarks import combined_memory_study as study
from mars_titan.training.learning_hold import LearningHoldError


def test_factorial_is_complete_and_each_child_uses_a_matching_parent():
    jobs = study.jobs()
    by_id = {job["id"]: job for job in jobs}
    assert len(jobs) == len(by_id)
    for seed in (42, 43, 44):
        cells = [
            job
            for job in jobs
            if job.get("factorial") and job["fold"] == "fold-018" and job["seed"] == seed
        ]
        assert {job["bits"] for job in cells} == set(range(8))
    available = set()
    for job in jobs:
        if "parent" in job:
            parent = by_id[job["parent"]]
            assert parent["id"] in available
            assert (parent["fold"], parent["seed"]) == (job["fold"], job["seed"])
        available.add(job["id"])
    assert any(j.get("bits") == 7 and j["kind"] == "readout" for j in jobs)
    assert {j.get("refinements") for j in jobs if j["kind"] == "readout"} >= {1, 2, 4}


def test_separate_factors_do_not_silently_enable_other_extensions():
    assert study.extensions(0) == {}
    assert study.extensions(1) == study.PT1
    assert study.extensions(2) == {"memory_multiscale": {"form": "cascade", "coupling": 0.0625}}
    assert study.extensions(4) == {"presence_modulation": "film_v1"}
    first = study.extensions(7)
    first["memory_stability"]["alpha_floor"] = 0.9
    assert study.extensions(7)["memory_stability"]["alpha_floor"] == 0.002


def test_budget_counts_failed_attempts_and_rejects_exhaustion(tmp_path):
    (tmp_path / "measurements").mkdir()
    path = tmp_path / "measurements/failed.json"
    path.write_text(json.dumps(dict(status="failed", process_seconds=12599.5)))
    assert study.remaining_seconds(tmp_path) == 0.5
    path.write_text(json.dumps(dict(status="failed", process_seconds=14401)))
    assert study.remaining_seconds(tmp_path) == 0
    for invalid in (-1, float("nan"), True):
        path.write_text(json.dumps(dict(status="failed", process_seconds=invalid)))
        with pytest.raises(ValueError, match="coste"):
            study.remaining_seconds(tmp_path)


def test_global_hold_precedes_any_input_read(tmp_path, monkeypatch):
    path = tmp_path / "hold.json"
    path.write_text(json.dumps(dict(training_allowed=False)))
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(path))
    with pytest.raises(LearningHoldError):
        study.run_job(tmp_path / "absent", "absent")


def test_prepared_receipt_rejects_config_and_artifact_drift(tmp_path):
    artifact = tmp_path / "input.json"
    artifact.write_text("{}")
    receipt = dict(
        configuration_sha256=study.sha256(study.CONFIG),
        jobs=study.jobs(),
        artifacts={"input.json": study.sha256(artifact)},
    )
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(receipt))
    assert study.verify_prepared(tmp_path) == receipt
    changed = copy.deepcopy(receipt)
    changed["configuration_sha256"] = "a" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        study.verify_prepared(tmp_path)
    path.write_text(json.dumps(receipt))
    artifact.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        study.verify_prepared(tmp_path)


def test_prepare_and_dispatch_every_case_with_valid_recipes(tmp_path, monkeypatch):
    from benchmarks import combined_memory_reference
    from mars_titan.evaluation.splits import build_folds
    from mars_titan.posttraining import chronological_windows
    from mars_titan.training import (
        candidate_walk_forward,
        cm_v1_factorial,
        mars_titan_walk_forward,
        titans_walk_forward,
    )
    from tests.tooling.test_combined_memory_cohort import population

    config = study.configuration()
    config["folds"] = {key: 8 for key in config["folds"]}
    monkeypatch.setattr(study, "configuration", lambda: copy.deepcopy(config))
    source, rows = population()
    for row in rows:
        row["folds"]["fold-015"] = copy.deepcopy(row["folds"]["fold-018"])
    for asset in source["assets"]:
        asset.update(samples=4000, purged_by_boundary=dict(train=0, evaluation=0))
    protocol = json.loads(
        (study.ROOT / "configs/evaluation/historical-masked-us-walk-forward-v2.json").read_text()
    )
    source.update(
        final_test_opened=False,
        input_policy="historical_masked_2000_v1",
        counts=dict(train=1, evaluation=1),
        coverage=[dict(market="US", symbol=a["symbol"]) for a in source["assets"]],
    )
    for fold in config["folds"]:
        folder = tmp_path / "sources" / fold
        folder.mkdir(parents=True)
        source["temporal_view"] = dict(
            protocol=protocol, fold=next(f for f in build_folds(protocol) if f["id"] == fold)
        )
        (folder / "manifest.json").write_text(json.dumps(source))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(rows=rows)))
    work = tmp_path / "work"
    receipt = study.prepare(tmp_path / "sources", audit, work)
    assert study.verify_prepared(work) == receipt
    called = []

    def record(*args, **kwargs):
        import torch

        assert not torch.backends.mha.get_fastpath_enabled()
        assert kwargs["device"] == "cuda:0"
        called.append(kwargs)
        return dict(status="prepared")

    for module, name in (
        (titans_walk_forward, "run_titans_window"),
        (candidate_walk_forward, "fit_window"),
        (cm_v1_factorial, "run_cm_v1_core_window"),
        (cm_v1_factorial, "run_cm_v1_window"),
        (mars_titan_walk_forward, "run_mars_titan_window"),
        (chronological_windows, "run_titans_posttraining"),
        (combined_memory_reference, "frozen_reference"),
    ):
        monkeypatch.setattr(module, name, record)
    for job in receipt["jobs"]:
        assert study.execute(work, job, object())["status"] == "prepared"
    assert len(called) == len(receipt["jobs"])


@pytest.mark.cuda
@pytest.mark.parametrize("failed", [False, True])
def test_cuda_job_records_success_or_failure_without_fitting(
    tmp_path, monkeypatch, learning_doubles, failed
):
    import torch

    receipt = dict(configuration_sha256=study.sha256(study.CONFIG), jobs=study.jobs(), artifacts={})
    (tmp_path / "prepared.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)

    def technical_only(work, job, budget):
        assert torch.cuda.current_device() == 0
        assert budget.steps == 0 and not budget.requested
        if failed:
            raise ValueError("fallo técnico deliberado")
        return dict(status="completed", run_id="fixture")

    monkeypatch.setattr(study, "execute", technical_only)
    first = receipt["jobs"][0]["id"]
    if failed:
        with pytest.raises(ValueError, match="deliberado"):
            study.run_job(tmp_path, first)
    else:
        assert study.run_job(tmp_path, first)["status"] == "completed"
    reports = list((tmp_path / "measurements").glob("*.json"))
    assert len(reports) == 1
    result = json.loads(reports[0].read_text())
    assert result["status"] == ("failed" if failed else "completed")
    assert result["optimizer_steps"] == 0 and result["process_seconds"] > 0
    assert result["final_test_opened"] is False
    assert study.remaining_seconds(tmp_path) < 12600


@pytest.mark.parametrize("command", ["prepare", "run"])
def test_cli_routes_only_the_requested_operation(tmp_path, monkeypatch, capsys, command):
    called = []

    def record(*args):
        called.append(args)
        return dict(status="fixture")

    monkeypatch.setattr(study, "prepare", record)
    monkeypatch.setattr(study, "run_job", record)
    extra = (
        ["--sources", "sources", "--audit", "audit"] if command == "prepare" else ["--job", "cell"]
    )
    monkeypatch.setattr(sys, "argv", ["study", command, "--work", str(tmp_path), *extra])
    study.main()
    assert len(called) == 1
    assert json.loads(capsys.readouterr().out) == dict(status="fixture")
    assert called[0][-1] == (tmp_path if command == "prepare" else "cell")


@pytest.mark.parametrize(
    "arguments",
    [
        ["prepare"],
        ["prepare", "--sources", "sources"],
        ["prepare", "--sources", "sources", "--audit", "audit", "--job", "cell"],
        ["run"],
        ["run", "--job", "cell", "--audit", "audit"],
        ["run", "--job", "cell", "--sources", "sources"],
    ],
)
def test_cli_rejects_incomplete_or_mixed_operations(tmp_path, monkeypatch, arguments):
    def forbidden(*args):
        raise AssertionError("Los argumentos inválidos llegaron al ejecutor")

    monkeypatch.setattr(study, "prepare", forbidden)
    monkeypatch.setattr(study, "run_job", forbidden)
    monkeypatch.setattr(sys, "argv", ["study", *arguments, "--work", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2


def test_frozen_reference_does_not_fall_back_when_cuda_is_unavailable(
    tmp_path, monkeypatch, learning_doubles
):
    import torch

    from benchmarks.combined_memory_reference import frozen_reference

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda:0"):
        frozen_reference(tmp_path / "missing-view", tmp_path / "parent", tmp_path / "out", seed=42)
