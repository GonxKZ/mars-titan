"""La medida del determinismo compara huellas, separa los modos y no cambia parámetros (#794)."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from benchmarks import determinism_cost as cost
from mars_titan.training import determinism

TINY = dict(rows=4, instants=1, repetitions=2, warmup=0, threads=1)


@pytest.fixture(autouse=True)
def restore_flags():
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.mha.get_fastpath_enabled(),
        torch.get_num_threads(),
    )
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, False
    yield
    torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[2:4]
    torch.backends.mha.set_fastpath_enabled(flags[4])
    torch.set_num_threads(flags[5])


def record(predictions, gradients, seconds=(1.0, 1.0)):
    return dict(
        digests=[
            dict(predictions=p, gradients=g) for p, g in zip(predictions, gradients, strict=True)
        ],
        graph_seconds=dict(samples=list(seconds)),
    )


def test_each_mode_repeats_its_digests_and_leaves_the_parameters_unchanged(monkeypatch):
    plain = cost.worker(cost.DEFAULT, "cpu", **TINY)
    assert plain["determinism"] is None and plain["numerics"]["deterministic"] is False
    assert plain["parameters_unchanged"] is True and len(plain["digests"]) == 2
    assert plain["digests"][0] == plain["digests"][1]
    monkeypatch.setenv(determinism.WORKSPACE_VARIABLE, ":4096:8")
    fixed = cost.worker(determinism.TORCH_DETERMINISTIC, "cpu", **TINY)
    assert fixed["determinism"] == determinism.determinism_identity(determinism.TORCH_DETERMINISTIC)
    assert fixed["numerics"]["deterministic"] is True
    assert fixed["digests"][0] == fixed["digests"][1]


@pytest.mark.parametrize("warn_only", [False, True])
def test_the_default_worker_keeps_inherited_determinism(warn_only):
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    result = cost.worker(cost.DEFAULT, "cpu", **TINY)
    assert result["determinism"] is None
    assert result["numerics"]["deterministic"] is True
    assert result["numerics"]["cudnn_deterministic"] is True
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only


def test_the_comparison_finds_each_kind_of_mismatch():
    same = record(["p", "p"], ["g", "g"], seconds=(1.0, 3.0))
    result = cost.compare(
        {cost.DEFAULT: [same, same], determinism.TORCH_DETERMINISTIC: [same, same]}
    )
    assert all(all(value.values()) for value in result["modes"].values())
    assert result["same_predictions_across_modes"] and result["same_gradients_across_modes"]
    assert result["cost_ratio_of_medians"] == 1.0
    # La repetición distinta va en el segundo proceso y la otra huella en el primero.
    within = record(["p", "q"], ["g", "g"])
    across = record(["q", "q"], ["g", "g"], seconds=(4.0, 4.0))
    result = cost.compare(
        {cost.DEFAULT: [same, within], determinism.TORCH_DETERMINISTIC: [across, same]}
    )
    assert result["modes"][cost.DEFAULT] == dict(within_process=False, across_processes=True)
    assert result["modes"][determinism.TORCH_DETERMINISTIC] == dict(
        within_process=True, across_processes=False
    )
    assert not result["same_predictions_across_modes"] and result["same_gradients_across_modes"]
    # Mediana de 1, 1, 1 y 3 frente a la de 4, 4, 1 y 3.
    assert result["median_graph_seconds"] == {
        cost.DEFAULT: 1.0,
        determinism.TORCH_DETERMINISTIC: 3.5,
    }
    assert result["cost_ratio_of_medians"] == 3.5


def test_only_the_deterministic_process_receives_the_cublas_workspace(monkeypatch, tmp_path):
    seen = []

    def run(command, *, cwd, env, check):
        seen.append((command, env.get(determinism.WORKSPACE_VARIABLE)))
        path = command[command.index("--result") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dict(mode=command[command.index("--worker") + 1]), handle)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv(determinism.WORKSPACE_VARIABLE, ":16:8")
    monkeypatch.setattr(cost.subprocess, "run", run)
    for index, mode in enumerate(cost.MODES):
        assert cost.launch(mode, "cpu", TINY, tmp_path, index) == dict(mode=mode)
    assert [workspace for _, workspace in seen] == [None, ":4096:8"]
    command = seen[0][0]
    assert command[1:4] == ["-m", "benchmarks.determinism_cost", "--worker"]
    assert command[command.index("--rows") + 1] == "4"
    assert os.environ[determinism.WORKSPACE_VARIABLE] == ":16:8"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--rows", "0"], "presupuesto"),
        (["--repetitions", "1"], "presupuesto"),
        (["--threads", "9"], "presupuesto"),
        (["--worker", cost.DEFAULT], "resultado nuevo"),
        ([], "recibo nuevo"),
    ],
)
def test_the_command_refuses_options_outside_its_budget(arguments, message):
    with pytest.raises(SystemExit, match=message):
        cost.main(arguments)


def test_the_command_needs_a_new_receipt_and_a_new_result(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("La orden no debe medir sobre un archivo existente")

    monkeypatch.setattr(cost, "launch", unexpected)
    monkeypatch.setattr(cost, "worker", unexpected)
    existing = tmp_path / "receipt.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="recibo nuevo"):
        cost.main(["--output", str(existing)])
    with pytest.raises(SystemExit, match="resultado nuevo"):
        cost.main(["--worker", cost.DEFAULT, "--result", str(existing)])


def test_the_worker_command_writes_the_record_of_its_mode(monkeypatch, tmp_path):
    calls, result = [], tmp_path / "result.json"

    def worker(mode, device, **options):
        calls.append((mode, device, options))
        return dict(mode=mode, digests=[])

    monkeypatch.setattr(cost, "worker", worker)
    options = ["--rows", "4", "--instants", "1", "--repetitions", "2", "--warmup", "0"]
    cost.main(["--worker", cost.DEFAULT, "--result", str(result), *options, "--threads", "1"])
    assert calls == [(cost.DEFAULT, "cpu", TINY)]
    assert json.loads(result.read_text(encoding="utf-8")) == dict(mode=cost.DEFAULT, digests=[])


def test_the_command_compares_two_processes_per_mode_in_a_new_receipt(
    monkeypatch, tmp_path, capsys
):
    calls, output = [], tmp_path / "receipts" / "determinism.json"
    # Los dos modos predicen lo mismo y difieren en los gradientes.
    default, fixed = record(["p", "p"], ["g", "g"]), record(["p", "p"], ["h", "h"])
    measured = dict(zip(cost.MODES, (default, fixed), strict=True))

    def launch(mode, device, options, scratch, index):
        calls.append((mode, device, options, index, Path(scratch).parent))
        return measured[mode]

    monkeypatch.setattr(cost, "launch", launch)
    options = ["--rows", "4", "--instants", "1", "--repetitions", "2", "--warmup", "0"]
    cost.main(["--output", str(output), *options, "--threads", "1"])
    expected = [
        (mode, "cpu", TINY, index, output.parent) for mode in cost.MODES for index in (0, 1)
    ]
    assert calls == expected
    receipt = json.loads(output.read_text(encoding="utf-8"))
    records = {mode: [measured[mode]] * cost.PROCESSES for mode in cost.MODES}
    assert receipt["processes"] == records and receipt["options"] == TINY
    assert receipt["comparison"] == cost.compare(records)
    assert not receipt["comparison"]["same_gradients_across_modes"]
    assert len(receipt["commit"]) == 40 and len(receipt["sources_sha256"]) == 4
    # Los resultados de cada proceso se borran y solo queda el recibo.
    assert [path.name for path in output.parent.iterdir()] == ["determinism.json"]
    assert json.loads(capsys.readouterr().out) == receipt["comparison"]


def test_a_receipt_with_nan_leaves_no_file(monkeypatch, tmp_path):
    output = tmp_path / "receipts" / "determinism.json"
    measured = record(["p", "p"], ["g", "g"], seconds=(float("nan"), 1.0))
    monkeypatch.setattr(cost, "launch", lambda *arguments: measured)
    with pytest.raises(ValueError):
        cost.main(["--output", str(output)])
    assert not list(output.parent.iterdir())


def test_the_warmup_runs_before_the_measured_passes(monkeypatch):
    passes, original = [], cost.graph_pass

    def counted(*arguments):
        passes.append(arguments[0])
        return original(*arguments)

    monkeypatch.setattr(cost, "graph_pass", counted)
    result = cost.worker(cost.DEFAULT, "cpu", **(TINY | dict(warmup=1)))
    assert len(passes) == 3 and len({id(model) for model in passes}) == 1
    assert len(result["digests"]) == len(result["graph_seconds"]["samples"]) == 2


def test_a_measure_that_changes_a_parameter_is_refused(monkeypatch):
    calls, original = [], cost.graph_pass

    def altered(model, *arguments):
        result = original(model, *arguments)
        calls.append(model)
        # Tras la última pasada, porque el predictor ya rechaza otra con parámetros cambiados.
        if len(calls) == TINY["repetitions"]:
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
        return result

    monkeypatch.setattr(cost, "graph_pass", altered)
    with pytest.raises(RuntimeError, match="modificado parámetros"):
        cost.worker(cost.DEFAULT, "cpu", **TINY)


def test_a_non_finite_gradient_in_one_parameter_stops_the_pass():
    spec = cost.specification()
    inputs = cost.batches(spec, 4, 1, device="cpu")
    targets = [torch.zeros(4)]
    config = cost.FinancialConfig(spec, variant="mac_online", seed=42, **cost.BASE)
    model = cost.FinancialPredictor(config, dtype=torch.float32).train()
    elapsed, digests = cost.graph_pass(model, inputs, targets, lambda: None)
    assert elapsed > 0 and set(digests) == {"predictions", "gradients"}
    assert all(value.grad is None for value in model.parameters())
    # Basta un parámetro con NaN, aunque los demás tengan un gradiente finito.
    next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(ValueError, match="gradiente no finito"):
        cost.graph_pass(model, inputs, targets, lambda: None)
