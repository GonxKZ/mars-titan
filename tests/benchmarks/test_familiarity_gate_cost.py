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


def test_the_warmup_runs_every_model_and_stays_out_of_the_samples(monkeypatch, tmp_path):
    calls, graph, plain = [], cost.graph_pass, cost.inference

    def graph_counted(model, *arguments):
        calls.append(("graph", model))
        return graph(model, *arguments)

    def plain_counted(model, *arguments):
        calls.append(("plain", model))
        return plain(model, *arguments)

    monkeypatch.setattr(cost, "graph_pass", graph_counted)
    monkeypatch.setattr(cost, "inference", plain_counted)
    receipt = cost.measure(tmp_path / "cost.json", **(SMALL | dict(warmup=2)))
    # Dos rondas de calentamiento y una medida, con las dos pasadas de los tres modelos.
    for kind in ("graph", "plain"):
        models = [model for name, model in calls if name == kind]
        assert len(models) == 3 * 3 and len({id(model) for model in models}) == 3
    assert receipt["warmup"] == 2
    for value in receipt["models"].values():
        for key in ("forward_with_graph_seconds", "backward_seconds", "inference_seconds"):
            assert len(value[key]["samples"]) == 1


def test_a_non_finite_gradient_in_one_parameter_stops_the_measure():
    model = cost.reader("familiarity", "cpu")
    generator = torch.Generator().manual_seed(3)
    state = torch.randn(4, cost.HIDDEN, generator=generator)
    head = torch.randn(cost.HIDDEN, generator=generator)
    arguments = (cost.snapshot(8, "cpu"), state, head, torch.zeros(4))
    # Con gradientes finitos la pasada devuelve sus dos tiempos y deja los gradientes vacíos.
    assert all(seconds > 0 for seconds in cost.graph_pass(model, *arguments))
    assert all(value.grad is None for value in model.parameters())
    # Basta un parámetro con NaN, aunque los demás tengan un gradiente finito.
    model.familiarity_bias.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(ValueError, match="gradiente no finito"):
        cost.graph_pass(model, *arguments)


def test_only_the_first_cuda_device_is_synchronized(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", calls.append)
    cost._synchronize("cpu")
    cost._synchronize("cuda:0")
    assert calls == ["cuda:0"]


def test_the_command_line_passes_its_shape_and_prints_each_ratio(monkeypatch, tmp_path, capsys):
    calls, output = [], tmp_path / "cost.json"

    def measured(path, **options):
        calls.append((path, options))
        ratios = dict(reference=1.0, familiarity_gate=1.5)
        return dict(
            models={
                name: dict(relative_to_reference=dict(backward_seconds=value))
                for name, value in ratios.items()
            }
        )

    monkeypatch.setattr(cost, "measure", measured)
    monkeypatch.setattr("sys.argv", ["familiarity_gate_cost.py", "--output", str(output)])
    cost.main()
    defaults = dict(rows=512, episodes=1024, repetitions=30, warmup=3, threads=2, device="cpu")
    assert calls == [(output, defaults)]
    assert capsys.readouterr().out.splitlines() == [
        'reference {"backward_seconds": 1.0}',
        'familiarity_gate {"backward_seconds": 1.5}',
    ]
    shape = dict(rows=4, episodes=8, repetitions=1, warmup=0, threads=1, device="cuda:0")
    options = [item for key, value in shape.items() for item in (f"--{key}", str(value))]
    monkeypatch.setattr("sys.argv", ["familiarity_gate_cost.py", "--output", str(output), *options])
    cost.main()
    assert calls[1] == (output, shape)
    # Sin destino la medida no empieza.
    monkeypatch.setattr("sys.argv", ["familiarity_gate_cost.py"])
    with pytest.raises(SystemExit):
        cost.main()
    assert len(calls) == 2
