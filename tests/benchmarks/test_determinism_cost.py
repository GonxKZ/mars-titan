"""La medida del determinismo compara huellas, separa los modos y no cambia parámetros (#794)."""

import json
import os
import subprocess

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
