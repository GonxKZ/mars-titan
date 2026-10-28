"""La medida de coste de R3 (#786) cubre sus modelos sin optimizador ni cambios de parámetros."""

import json

import pytest
import torch

from benchmarks import familiarity_gate_cost as cost

SMALL = dict(rows=4, episodes=8, repetitions=1, warmup=0, threads=1)


def test_a_small_measure_covers_the_declared_models_and_keeps_the_parameters(tmp_path):
    output = tmp_path / "cost.json"
    receipt = cost.measure(output, **SMALL)
    models = receipt["models"]
    assert list(models) == ["reference", "familiarity_gate", "constant_gate"]
    gates = {name: value["familiarity_gate"] for name, value in models.items()}
    assert gates == dict(reference=None, familiarity_gate="familiarity", constant_gate="constant")
    added = {name: value["added_parameters"] for name, value in models.items()}
    assert added == dict(reference=0, familiarity_gate=2, constant_gate=1)
    for value in models.values():
        assert value["starts_at_reference"] and value["parameters_unchanged"]
        for key in ("forward_with_graph_seconds", "backward_seconds", "inference_seconds"):
            assert len(value[key]["samples"]) == 1 and value[key]["median"] > 0
    # Una fila de FP32 por paso añade cosenos y productos de ocho vecinos, el cambio y cuatro
    # escalares.
    extra = models["familiarity_gate"]["estimated_bytes"] - models["reference"]["estimated_bytes"]
    assert extra == 4 * (2 * 8 + 64 + 4) * 4
    assert receipt["shape"] == dict(rows=4, episodes=8, hidden_size=64, neighbors=8)
    assert receipt["kernel_policy"]["precision"] == "fp32_strict"
    assert json.loads(output.read_text()) == receipt
    with pytest.raises(FileExistsError):
        cost.measure(output, **SMALL)


@pytest.mark.parametrize(
    "changes",
    [dict(rows=0), dict(rows=True), dict(episodes=4), dict(repetitions=0), dict(threads=9)],
    ids=["rows", "boolean", "episodes", "repetitions", "threads"],
)
def test_shapes_outside_the_budget_are_refused(changes, tmp_path):
    output = tmp_path / "cost.json"
    with pytest.raises(ValueError, match="presupuesto"):
        cost.measure(output, **(SMALL | changes))
    with pytest.raises(ValueError, match="cpu o cuda:0"):
        cost.measure(output, **SMALL, device="cuda:1")
    assert not output.exists()


def test_a_gate_that_does_not_start_at_the_reference_is_refused(monkeypatch, tmp_path):
    original = cost.reader

    def shifted(gate, device):
        model = original(gate, device)
        if gate == "constant":
            with torch.no_grad():
                model.familiarity_bias.fill_(0.5)
        return model

    monkeypatch.setattr(cost, "reader", shifted)
    output = tmp_path / "cost.json"
    with pytest.raises(RuntimeError, match="constant_gate no empieza"):
        cost.measure(output, **SMALL)
    assert not output.exists()


def test_a_measure_that_changes_a_parameter_is_refused(monkeypatch, tmp_path):
    original = cost.inference

    def altered(model, data, state):
        seconds = original(model, data, state)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        return seconds

    monkeypatch.setattr(cost, "inference", altered)
    output = tmp_path / "cost.json"
    with pytest.raises(RuntimeError, match="modificado parámetros"):
        cost.measure(output, **SMALL)
    assert not output.exists()
