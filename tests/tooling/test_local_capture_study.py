"""Catálogo y análisis de la captura local sin ajustes ni corpus financiero."""

import importlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.tooling.test_neural_plasticity_study import prepared


def test_local_capture_seals_three_controls_and_five_complete_seed_pairs(tmp_path):
    from benchmarks.neural_plasticity_study import verify_prepared

    work, receipt = prepared(tmp_path, "local_capture_v1")
    assert receipt["configuration"]["seeds"] == [42, 43, 44, 45, 46]
    assert receipt["configuration"]["max_job_seconds"] == 300
    assert receipt["configuration"]["max_total_seconds"] == 14400
    assert receipt["configuration"]["primary_reference"] == "capture_global"
    assert receipt["configuration"]["primary_candidate"] == "capture_local"
    assert len(receipt["jobs"]) == 15
    variants = {job["arm"]: job["changes"] for job in receipt["jobs"]}
    assert set(variants) == {"capture_global", "capture_local", "capture_constant"}
    global_options = variants["capture_global"]["memory_plasticity"]
    local_options = variants["capture_local"]["memory_plasticity"]
    assert global_options | {"capture_scope": "row_local"} == local_options
    assert all(value["memory_plasticity"]["write"] == "none" for value in variants.values())
    assert all(value["memory_multiscale"]["coupling"] == 1 / 16 for value in variants.values())
    assert json.loads((work / "recipes/protocol.json").read_text())["seeds"] == [42, 43, 44, 45, 46]
    assert verify_prepared(work) == receipt


@pytest.mark.parametrize("fault", ["none", "initialization", "malformed", "missing", "one_model"])
def test_initialization_control_is_not_reported_as_a_trained_unity_control(tmp_path, fault):
    import torch

    module = importlib.import_module("benchmarks.quantile_multiseed_analysis")
    records = {}
    for seed in (42, 43):
        for name in ("capture_global", "capture_local", "capture_constant"):
            folder = tmp_path / "runs" / f"{name}-{seed}"
            folder.mkdir(parents=True)
            identity = dict(initial_parameters_sha256=("a" if seed == 42 else "b") * 64)
            if name == "capture_local" and seed == 43:
                if fault == "initialization":
                    identity["initial_parameters_sha256"] = "c" * 64
                elif fault == "malformed":
                    identity["initial_parameters_sha256"] = "x" * 64
                elif fault == "missing":
                    identity.clear()
            torch.save(dict(identity=identity), folder / "checkpoint.fixture")
            records[name, seed] = (
                dict(id=folder.name),
                dict(checkpoint=dict(path="checkpoint.fixture")),
                None,
            )
    if fault == "one_model":
        records = {key: value for key, value in records.items() if key[0] == "capture_local"}
    if fault in {"malformed", "missing", "one_model"}:
        with pytest.raises(ValueError):
            module.paired_initialization(tmp_path, records, [42, 43])
        return
    result = module.paired_initialization(tmp_path, records, [42, 43])
    assert result["kind"] == "paired_initial_parameters"
    assert result["passed"] is (fault == "none")
    assert result["by_seed"]["42"]["passed"] is True
    assert result["by_seed"]["43"]["passed"] is (fault == "none")
    assert "shared_parameters_equal" not in result


@pytest.mark.parametrize("fault", ["none", "initialization", "steps", "missing", "unknown"])
def test_complete_local_analysis_pairs_every_seed_and_keeps_all_controls(
    tmp_path, monkeypatch, learning_doubles, fault
):
    import torch

    from mars_titan.data.storage import atomic_json, sha256
    from tests.tooling.test_quantile_multiseed_analysis import multiseed_fixture

    module = importlib.import_module("benchmarks.quantile_multiseed_analysis")
    work = tmp_path / "input"
    work.mkdir()
    receipt = multiseed_fixture(work, "local_capture_v1")
    for seed in (42, 43):
        folder = work / f"runs/capture_constant-{seed}"
        report = json.loads((folder / "run.json").read_text())
        for phase in ("validation", "calibration", "evaluation"):
            path = folder / report["predictions"][phase]["path"]
            table = pq.read_table(path)
            table = table.set_column(5, "prediction", pa.array(table["target"].to_numpy() / 2))
            pq.write_table(table, path)
            report["predictions"][phase].update(sha256=sha256(path), metrics=dict(session_mae=2.75))
        atomic_json(folder / "run.json", report)
    if fault == "missing":
        receipt["jobs"].pop()
    elif fault == "unknown":
        for job in receipt["jobs"]:
            if job["arm"] == "capture_constant":
                job["arm"] = "unknown"
    elif fault == "steps":
        path = work / "runs/capture_local-43/run.json"
        report = json.loads(path.read_text())
        report["fit"]["global_step"] += 1
        atomic_json(path, report)
    elif fault == "initialization":
        folder = work / "runs/capture_local-43"
        path = folder / "checkpoint.fixture"
        payload = torch.load(path, weights_only=True)
        payload["identity"]["initial_parameters_sha256"] = "b" * 64
        torch.save(payload, path)
        report = json.loads((folder / "run.json").read_text())
        report["checkpoint"]["sha256"] = sha256(path)
        atomic_json(folder / "run.json", report)
    monkeypatch.setattr(module, "verify_prepared", lambda path: receipt)
    output = tmp_path / "output"
    if fault in {"missing", "steps", "unknown"}:
        with pytest.raises(ValueError):
            module.analyze(work, output)
        assert not output.exists()
        return
    result = module.analyze(work, output)
    assert result["kind"] == "local_capture_multiseed_development_results"
    assert result["primary"]["contrasts"][0]["estimate"] == -5.5
    assert result["primary"]["resampling"]["sessions"] == 2
    assert len(result["runs"]) == 6
    assert "unity_control" not in result
    assert result["initialization_control"]["passed"] is (fault == "none")
    assert result["primary_interpretation"] == (
        "menor_error_en_desarrollo"
        if fault == "none"
        else "atribucion_bloqueada_por_inicializacion"
    )
    contrasts = {value["name"] for value in result["secondary"]["contrasts"]}
    assert contrasts == {
        "capture_global_minus_zero",
        "capture_local_minus_zero",
        "capture_constant_minus_zero",
        "local_minus_constant",
        "global_minus_constant",
    }
    estimates = {value["name"]: value["estimate"] for value in result["secondary"]["contrasts"]}
    assert estimates["local_minus_constant"] == -2.75
    assert estimates["global_minus_constant"] == 2.75
    assert estimates["capture_local_minus_zero"] == -5.5
    assert len(list(output.glob("*-sessions.parquet"))) == 10
    assert result["final_test_opened"] is False
    assert json.loads((output / "results.json").read_text()) == result
