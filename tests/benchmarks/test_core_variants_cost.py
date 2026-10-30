"""La medida de coste de #786 recorre cada modelo declarado con formas pequeñas y no ajusta nada."""

import pytest
import torch

from benchmarks import core_variants_cost as cost
from mars_titan.models.titans import neural_memory


@pytest.fixture(autouse=True)
def restored_runtime():
    threads, fastpath = torch.get_num_threads(), torch.backends.mha.get_fastpath_enabled()
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def test_a_small_measure_covers_every_declared_model_and_keeps_the_parameters(tmp_path):
    output = tmp_path / "cost.json"
    original = neural_memory.huber_elements
    receipt = cost.measure(output, rows=4, instants=3, repetitions=1, warmup=0, threads=1)
    assert neural_memory.huber_elements is original
    names = list(cost.arms())
    assert names == ["reference", "huber_memory", "differential_attention", "scale_invariant_time"]
    models = receipt["models"]
    assert list(models) == names and all(m["parameters_unchanged"] for m in models.values())
    # Huber añade w y b de 64 canales, U1 dos proyecciones de 64 × 64 y λ por cabeza, y U2 nada.
    added = {name: models[name]["added_parameters"] for name in names}
    assert added == dict(
        reference=0,
        huber_memory=128,
        differential_attention=2 * 64 * 64 + 4,
        scale_invariant_time=0,
    )
    fraction = receipt["huber_clipped_fraction_at_start"]
    assert set(fraction) == {"huber_memory"} and 0 < fraction["huber_memory"] < 1
    assert output.exists() and receipt["shape"] == dict(rows=4, instants=3, hidden_size=64)
    with pytest.raises(FileExistsError):
        cost.measure(output, rows=4, instants=3, repetitions=1, warmup=0, threads=1)
    with pytest.raises(ValueError, match="presupuesto"):
        cost.measure(
            tmp_path / "other.json", rows=4, instants=1, repetitions=1, warmup=0, threads=1
        )


def test_two_declarations_cannot_give_one_model_different_changes(monkeypatch, tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text('{"arms": {"reference": {"predictor_changes": {}}}}', encoding="utf-8")
    second.write_text(
        '{"arms": {"reference": {"predictor_changes": {"gate_bias": {}}}}}', encoding="utf-8"
    )
    monkeypatch.setattr(cost, "ROOT", tmp_path)
    monkeypatch.setattr(cost, "DECLARATIONS", ("a.json", "b.json"))
    with pytest.raises(ValueError, match="cambios distintos"):
        cost.arms()


def test_a_measure_that_changes_a_parameter_is_refused(monkeypatch, tmp_path):
    original = cost.inference

    def altered(model, inputs):
        seconds = original(model, inputs)
        # El predictor rechaza el cambio en su siguiente frontera de sesión. Tras la última
        # pasada de cada modelo solo queda la huella final de la medida para detectarlo.
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        return seconds

    monkeypatch.setattr(cost, "inference", altered)
    output = tmp_path / "cost.json"
    with pytest.raises(RuntimeError, match="modificado parámetros"):
        cost.measure(output, rows=4, instants=2, repetitions=1, warmup=0, threads=1)
    assert not output.exists()


def test_the_warmup_runs_every_model_before_the_measured_passes(monkeypatch, tmp_path):
    calls, graph, plain = [], cost.graph_pass, cost.inference

    def graphed(model, inputs, targets):
        calls.append(("graph", model))
        return graph(model, inputs, targets)

    def inferred(model, inputs):
        calls.append(("inference", model))
        return plain(model, inputs)

    monkeypatch.setattr(cost, "graph_pass", graphed)
    monkeypatch.setattr(cost, "inference", inferred)
    cost.measure(tmp_path / "cost.json", rows=4, instants=2, repetitions=2, warmup=1, threads=1)
    models = list(dict.fromkeys(model for _, model in calls))
    passes = [(kind, model) for model in models for kind in ("graph", "inference")]
    assert len(models) == len(cost.arms())
    # Un calentamiento con grafo y sin grafo de cada modelo en el orden declarado, después la
    # primera repetición en ese orden y la segunda empezando por el segundo modelo. La pasada
    # final sin grafo es la de la fracción de Huber.
    assert calls[: len(passes)] == passes and calls[len(passes) : 2 * len(passes)] == passes
    assert calls[2 * len(passes) : 3 * len(passes)] == passes[2:] + passes[:2]
    assert calls[3 * len(passes) :] == [("inference", models[1])]


def test_a_non_finite_gradient_in_one_parameter_stops_the_measure(monkeypatch, tmp_path):
    original = cost.graph_pass

    def poisoned(model, inputs, targets):
        # Solo el primer gradiente que llega queda en NaN, así que el resto sigue finito.
        hit = []

        def hook(grad):
            if hit:
                return grad
            hit.append(True)
            return torch.full_like(grad, float("nan"))

        handles = [p.register_hook(hook) for p in model.parameters() if p.requires_grad]
        try:
            return original(model, inputs, targets)
        finally:
            for handle in handles:
                handle.remove()

    monkeypatch.setattr(cost, "graph_pass", poisoned)
    output = tmp_path / "cost.json"
    with pytest.raises(ValueError, match="gradiente no finito"):
        cost.measure(output, rows=4, instants=2, repetitions=1, warmup=0, threads=1)
    assert not output.exists()


def test_the_command_line_passes_its_shape_and_prints_each_ratio(monkeypatch, tmp_path, capsys):
    calls = []

    def measured(output, **shape):
        calls.append((output, shape))
        return dict(
            models=dict(reference=dict(relative_to_control=dict(inference_seconds=1.0))),
            huber_clipped_fraction_at_start=dict(huber_memory=0.25),
        )

    monkeypatch.setattr(cost, "measure", measured)
    output = tmp_path / "cost.json"
    shape = ["--rows", "8", "--instants", "3", "--repetitions", "4", "--warmup", "1"]
    for arguments in (
        ["--output", str(output)],
        ["--output", str(output), *shape, "--threads", "1"],
    ):
        monkeypatch.setattr("sys.argv", ["core_variants_cost.py", *arguments])
        cost.main()
    assert calls == [
        (output, dict(rows=256, instants=8, repetitions=10, warmup=2, threads=2)),
        (output, dict(rows=8, instants=3, repetitions=4, warmup=1, threads=1)),
    ]
    printed = ['reference {"inference_seconds": 1.0}', 'huber {"huber_memory": 0.25}']
    assert capsys.readouterr().out.splitlines() == printed * 2
    monkeypatch.setattr("sys.argv", ["core_variants_cost.py"])
    with pytest.raises(SystemExit):
        cost.main()
    assert len(calls) == 2
