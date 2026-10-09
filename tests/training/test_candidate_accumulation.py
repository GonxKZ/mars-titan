"""Acumulación por bloques y recomputación en el entrenador de la GRU candidata, sin pasos.

El optimizador solo registra gradientes. Se comparan la acumulación al madurar cada etiqueta y
la recomputación de cada bloque con el backward único del tramo de #389, que sigue siendo el
modo por defecto.
"""

import os
import weakref

import pytest
import torch

from mars_titan.data.input_policy import MODALITIES
from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
from mars_titan.training import candidate_run
from mars_titan.training.checkpoints import load_training_state
from tests.training.test_candidate_run import adapter, entries, named_records, trainer
from tests.training.test_financial_run import StopAtStep, corpus

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
    ),
    pytest.mark.usefixtures("learning_doubles"),
]

# Solo cambia el orden de las sumas y la división final por el total del tramo. Las cotas se
# fijaron con margen sobre la mayor diferencia observada en este corpus (ver la guía).
TOLERANCES = {
    torch.float64: dict(rtol=1e-12, atol=1e-15),
    torch.float32: dict(rtol=1e-5, atol=1e-7),
}


class SavedActivations(torch.autograd.graph.saved_tensors_hooks):
    """Bytes guardados por autograd que siguen vivos, por almacenamiento y sin parámetros."""

    def __init__(self, parameters):
        self.excluded = {value.untyped_storage().data_ptr() for value in parameters}
        self.live, self.current, self.peak = {}, 0, 0
        super().__init__(self._pack, lambda handle: handle)

    def _pack(self, tensor):
        # Guardar una vista separada evita el ciclo entre una salida y su propio nodo, que
        # retendría el grafo. La vista muere cuando autograd libera lo guardado.
        handle = tensor.detach()
        storage = handle.untyped_storage()
        key = storage.data_ptr()
        if key not in self.excluded:
            count, size = self.live.get(key, (0, storage.nbytes()))
            if count == 0:
                self.current += size
                self.peak = max(self.peak, self.current)
            self.live[key] = (count + 1, size)
            weakref.finalize(handle, self._release, key)
        return handle

    def _release(self, key):
        count, size = self.live.pop(key)
        if count > 1:
            self.live[key] = (count - 1, size)
        else:
            self.current -= size


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("accumulation"))


@pytest.mark.parametrize(
    "options",
    [
        *(dict(accumulation_rows=rows) for rows in (0, 1, True, 2.0, 65_537)),
        dict(recompute=1),
        dict(recompute=None),
    ],
)
def test_recipe_rejects_invalid_accumulation_or_recompute(options):
    with pytest.raises(ValueError, match="acumulación, recomputación"):
        candidate_run.CandidateRecipe(block_rows=2, **options)


def test_each_option_has_its_own_identity_and_the_default_does_not_split():
    default = candidate_run.CandidateRecipe()
    accumulated = candidate_run.CandidateRecipe(accumulation_rows=128)
    recomputed = candidate_run.CandidateRecipe(recompute=True)
    assert (default.accumulation_rows, default.recompute) == (None, False)
    assert default.identity()["loss_reduction"] == "mean_over_matured_labels_of_segment"
    assert accumulated.identity()["loss_reduction"] != default.identity()["loss_reduction"]
    assert recomputed.identity()["loss_reduction"] == default.identity()["loss_reduction"]
    assert recomputed.identity() != default.identity()


def segments(audit):
    """Agrupar los grupos de cada evento con el paso del tramo que los consume."""
    groups, result = [], []
    for entry in audit:
        if entry[0] == "accumulate":
            groups.extend(entry[2])
        elif entry[0] == "update":
            result.append((groups, entry[3]))
            groups = []
    return result


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "options",
    [
        dict(accumulation_rows=2),
        dict(accumulation_rows=4),
        dict(accumulation_rows=1024),
        dict(recompute=True),
        dict(accumulation_rows=2, recompute=True),
    ],
)
def test_variant_gradient_matches_the_segment_gradient(shared, tmp_path, dtype, options):
    _, streams = shared
    engines = {}
    for name, chosen in (("segment", {}), ("variant", options)):
        engines[name] = trainer(
            streams, tmp_path / name, model=adapter(streams, dtype=dtype), **chosen
        )
        engines[name].run()
    segment, blocks = engines["segment"], engines["variant"]
    # La misma instantánea, las mismas etiquetas y la admisión al final de cada evento.
    for kind in ("prediction", "label", "admit", "update"):
        assert entries(segment.audit, kind) == entries(blocks.audit, kind)
    assert not entries(segment.audit, "accumulate")
    rows = options.get("accumulation_rows")
    if rows is None:
        assert not entries(blocks.audit, "accumulate")
    for groups, used in segments(blocks.audit) if rows else ():
        assert sum(groups) == len(used) and all(0 < size <= rows for size in groups)
    expected, actual = named_records(segment), named_records(blocks)
    assert len(expected) == len(actual) > 3
    for left, right in zip(expected, actual, strict=True):
        assert left.keys() == right.keys()
        for name, value in left.items():
            if value is None:
                assert right[name] is None
            elif rows is None:
                # La recomputación repite el mismo forward y el mismo orden del backward.
                assert torch.equal(right[name], value), name
            else:
                torch.testing.assert_close(right[name], value, **TOLERANCES[dtype])
    reference = segment.history[1]["train"]
    observed = blocks.history[1]["train"]
    loss = reference.pop("mean_loss"), observed.pop("mean_loss")
    reference.pop("loss_sum", None), observed.pop("loss_sum", None)
    assert observed == reference
    assert loss[1] == pytest.approx(loss[0], rel=TOLERANCES[dtype]["rtol"])
    assert segment.run_id != blocks.run_id


def first_block(engine, source, size):
    for event in source.batched_events(block_rows=size):
        if event.at >= source.phase.decision_start and event.inputs:
            batch = validated_cpu_batch(event.inputs[0], engine.specification)
            if len(batch.flow_ids) >= size:
                return event, batch
    raise AssertionError("El corpus no tiene un bloque completo")


@pytest.mark.parametrize("recompute", [False, True])
def test_a_block_keeps_its_graph_until_its_last_label_matures(shared, tmp_path, recompute):
    _, streams = shared
    source = streams["train"]
    engine = trainer(
        streams, tmp_path / "run", admission="m0", accumulation_rows=2, recompute=recompute
    )
    event, batch = first_block(engine, source, 2)
    engine.model.train()
    run = candidate_run._Pass()
    quantiles = engine._forward(batch, engine._empty, grad=True)
    keys = list(zip(batch.flow_ids[:2], batch.prediction_at[:2], strict=True))
    run.outstanding[0] = 2
    for row, key in enumerate(keys):
        run.pending[key] = candidate_run._Pending(float(quantiles[row, 2].detach()), block=0)
        run.graphs[key] = quantiles[row]
    targets = (0.01, -0.02)
    # Cada fila madura en un evento distinto y el segundo backward recorre el mismo grafo.
    for row, (flow, at) in enumerate(keys):
        matured = ObservationEvent(event.at + row + 1, (), ((flow, at, targets[row]),), False)
        engine._labels(run, source, matured, train=True)
        assert run.outstanding == ({0: 1} if row == 0 else {})
        engine._accumulate(run, matured.at)
    assert run.accumulated == 2 and not run.graphs and not run.predictions
    accumulated = [value.grad for value in engine.trainable]
    engine.optimizer.zero_grad(set_to_none=True)
    whole = engine._forward(batch, engine._empty, grad=True)[:2]
    engine._loss(whole, torch.tensor(targets, dtype=whole.dtype), reduction="sum").backward()
    assert any(value is not None for value in accumulated)
    for value, gradient in zip(engine.trainable, accumulated, strict=True):
        if gradient is None:
            assert value.grad is None
        else:
            torch.testing.assert_close(gradient, value.grad, rtol=1e-12, atol=1e-15)


def test_groups_never_split_a_prediction_block(shared, tmp_path):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", admission="m0", accumulation_rows=3)
    weight = torch.ones(5, dtype=torch.float64, requires_grad=True)
    run = candidate_run._Pass()
    # Tres bloques de 2, 2 y 1 filas. El segundo no cabe con el primero y sí con el tercero.
    for block, size in enumerate((2, 2, 1)):
        output = weight * torch.arange(1.0, size + 1, dtype=torch.float64)[:, None]
        for row in range(size):
            run.predictions.append(output[row])
            run.targets.append(0.0)
            run.blocks.append(block)
    engine._accumulate(run, 7)
    assert entries(engine.audit, "accumulate") == [("accumulate", 7, (2, 3))]
    assert run.accumulated == 5 and weight.grad is not None


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_rows_of_a_block_do_not_see_other_rows(shared, tmp_path, dtype):
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", model=adapter(streams, dtype=dtype), block_rows=4)
    _, batch = first_block(engine, streams["train"], 3)
    encoded = engine.codec.encode(batch)
    memory = engine.model.snapshot(
        torch.from_numpy(encoded.keys.copy()),
        torch.from_numpy(encoded.values.copy()),
        torch.tensor([0.01, -0.02, 0.03][: len(encoded.keys)], dtype=dtype),
        torch.arange(1, len(encoded.keys) + 1),
        engine.model.representation_id(),
    )
    tensors = DecisionBatch.from_validated(batch, dtype=dtype)

    def forward(inputs):
        values = engine.native.CandidateInputs(
            *(inputs[name] for name in MODALITIES), tensors.presence
        )
        return engine.model.forward(values, memory, 2).quantiles

    base = forward(tensors.inputs)
    changed = dict(tensors.inputs)
    # Precios y gráficos están presentes en todas las filas del corpus técnico.
    for name in ("prices", "charts"):
        changed[name] = changed[name].clone()
        changed[name][1:] = changed[name][1:] * 3 + 1
    other = forward(changed)
    # La misma instantánea para todas las filas y ninguna mezcla entre filas del bloque.
    assert torch.equal(base[0], other[0]) and not torch.equal(base[1:], other[1:])
    gradients = []
    for quantiles in (base, other):
        engine.optimizer.zero_grad(set_to_none=True)
        engine._loss(quantiles[:1], torch.zeros(1, dtype=dtype), reduction="sum").backward()
        gradients.append({id(v): v.grad.clone() for v in engine.trainable if v.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    assert all(torch.equal(gradients[0][key], gradients[1][key]) for key in gradients[0])


def test_accumulation_and_recompute_bound_the_live_activations(shared, tmp_path):
    _, streams = shared
    peaks = {}
    for name, options in (
        ("segment", {}),
        ("blocks", dict(accumulation_rows=2)),
        ("recompute", dict(recompute=True)),
    ):
        engine = trainer(streams, tmp_path / name, update_instants=4, **options)
        tracker = SavedActivations(engine.model.named_parameters().values())
        with tracker:
            engine.run()
        assert tracker.current == 0
        peaks[name] = tracker.peak
    assert 0 < peaks["blocks"] * 2 < peaks["segment"]
    # Con la recomputación, autograd solo guarda fuera del bloque la pila y la pérdida. Las
    # entradas retenidas y el forward repetido se miden en candidate_memory_check.py.
    assert 0 < peaks["recompute"] * 20 < peaks["segment"]


@pytest.mark.parametrize("recompute", [False, True])
def test_resume_with_accumulation_reproduces_the_continuous_run(shared, tmp_path, recompute):
    _, streams = shared
    options = dict(accumulation_rows=2, recompute=recompute, checkpoint_updates=1)
    continuous = trainer(streams, tmp_path / "continuous", **options)
    expected = continuous.run()
    stop = StopAtStep(3)
    first = trainer(streams, tmp_path / "interrupted", **options)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    assert state["run"]["counters"]["updates"] == 3
    second = trainer(streams, tmp_path / "interrupted", **options)
    assert second.run(resume=True)["history"] == expected["history"]
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    for left, right in zip(joined, named_records(continuous), strict=True):
        for name, value in left.items():
            assert (value is None and right[name] is None) or torch.equal(value, right[name])
