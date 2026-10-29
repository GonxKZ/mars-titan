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
