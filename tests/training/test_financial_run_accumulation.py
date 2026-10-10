"""Acumulación por bloques de flujos en el recorrido de Titans-MAC, sin modificar pesos.

Con `accumulation_rows` el entrenador emite cada predicción con el mismo cálculo que el
recorrido por defecto y corta su grafo. Al actualizar recalcula el tramo por bloques de
flujos desde su estado inicial y acumula la pérdida media ponderada por filas. Se comprueba
que las predicciones, etiquetas, métricas y llamadas coinciden bit a bit, que el gradiente
acumulado coincide con el del tramo completo y que el grafo vivo al calcular la pérdida
se reduce. El optimizador solo registra gradientes.
"""

import hashlib

import pytest
import torch

from mars_titan.models.baselines.multimodal import SCALAR_HEAD
from mars_titan.models.quantile_head import PINBALL, QUANTILE_HEAD
from mars_titan.models.titans.config import MAX_BLOCK_ROWS, canonical
from mars_titan.models.titans.financial import VARIANTS, FinancialConfig, FinancialPredictor
from mars_titan.training.financial_run import ChronologicalRecipe, _Pass, load_recipe
from mars_titan.training.graph_memory import saved_graph_bytes
from tests.training.chronological_fixture import phases
from tests.training.test_financial_run import (
    StopAtStep,
    corpus,
    entries,
    explicit_fastpath,  # noqa: F401
    named_records,
    shared,  # noqa: F401
    trainer,
)

# Huellas de la identidad de receta capturadas antes de añadir la opción.
IDENTITY_DEFAULT = "262063b4a7723d8f1a45fd52f6613551430ffded9d795e6f8fee4e861e435afe"
IDENTITY_RECIPES = {
    "configs/titans/chronological-training.json": (
        "7ac1ddd0dbd716c95fbf8df665c4c1e7b1ea0f52b7e383cfb15b41ee64aab81a"
    ),
    "configs/titans/chronological-training-quantile.json": (
        "ee2235a9d0969994636c8a8ec194ded91f2be78a92aa435ffc8e9cee98e8f486"
    ),
}


def digest(recipe):
    return hashlib.sha256(canonical(recipe.identity()).encode()).hexdigest()


def model(streams, variant, head):
    specification = streams["train"].specification()
    config = FinancialConfig(specification, variant=variant, hidden_size=32, seed=42, head=head)
    return FinancialPredictor(config, dtype=torch.float64)


def engine_for(streams, output, variant, head, **options):
    loss = PINBALL if head == QUANTILE_HEAD else "mae"
    return trainer(
        streams, output, model=model(streams, variant, head), loss=loss, epochs=2, **options
    )


def without_loss(metrics):
    return {key: value for key, value in metrics.items() if key != "mean_loss"}


def test_default_identity_keeps_its_previous_digest():
    assert digest(ChronologicalRecipe()) == IDENTITY_DEFAULT
    for path, expected in IDENTITY_RECIPES.items():
        recipe, _ = load_recipe(path)
        assert recipe.accumulation_rows is None
        assert digest(recipe) == expected
    identity = ChronologicalRecipe(accumulation_rows=64).identity()
    assert identity["accumulation_rows"] == 64
    assert identity["gradient_accumulation"] == "flow_blocks_replayed_from_segment_start_v1"
    assert "accumulation_rows" not in ChronologicalRecipe().identity()


@pytest.mark.parametrize("value", [0, MAX_BLOCK_ROWS + 1, True, 1.0, "2"])
def test_accumulation_rows_must_be_a_bounded_integer(value):
    with pytest.raises(ValueError):
        ChronologicalRecipe(accumulation_rows=value)


def test_accumulation_rows_cannot_exceed_the_predictor_batch(shared, tmp_path):  # noqa: F811
    _, streams = shared
    specification = streams["train"].specification()
    config = FinancialConfig(specification, variant="mac_online", hidden_size=32, max_batch=2)
    small = FinancialPredictor(config, dtype=torch.float64)
    with pytest.raises(ValueError, match="lote del predictor"):
        trainer(streams, tmp_path / "run", model=small, accumulation_rows=3)


@pytest.mark.parametrize("head", [SCALAR_HEAD, QUANTILE_HEAD])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("rows", [1, 2])
def test_accumulated_gradient_matches_the_full_segment(
    shared,  # noqa: F811
    tmp_path,
    monkeypatch,
    variant,
    head,
    rows,
):
    _, streams = shared
    full = engine_for(streams, tmp_path / "full", variant, head)
    expected = full.run()
    blocks = engine_for(streams, tmp_path / "blocks", variant, head, accumulation_rows=rows)
    sizes, inside = [], []
    replay, prepare = blocks._replay, blocks.predictor.prepare

    def replay_spy(current):
        inside.append(True)
        try:
            return replay(current)
        finally:
            inside.pop()

    def prepare_spy(batch, state, **options):
        if inside and options["differentiable"]:
            sizes.append(len(batch.flow_ids))
        return prepare(batch, state, **options)

    monkeypatch.setattr(blocks, "_replay", replay_spy)
    monkeypatch.setattr(blocks.predictor, "prepare", prepare_spy)
    report = blocks.run()
    assert expected["status"] == report["status"] == "completed"
    # Predicciones, etiquetas y composición de cada actualización, bit a bit.
    assert blocks.audit == full.audit
    assert blocks.optimizer.calls == full.optimizer.calls == len(entries(full.audit, "update"))
    assert blocks.optimizer.zero_calls == full.optimizer.zero_calls
    for left, right in zip(report["history"], expected["history"], strict=True):
        assert left["validation"] == right["validation"]
        if right["train"] is not None:
            assert without_loss(left["train"]) == without_loss(right["train"])
            assert left["train"]["mean_loss"] == pytest.approx(right["train"]["mean_loss"])
    for left, right in zip(named_records(blocks), named_records(full), strict=True):
        assert left.keys() == right.keys()
        for key, value in right.items():
            if value is None:
                assert left[key] is None, key
            else:
                torch.testing.assert_close(left[key], value, rtol=1e-10, atol=1e-13)
    # La repetición recorre bloques de como mucho `rows` flujos, nunca el tramo entero.
    assert sizes and max(sizes) <= rows


def peak_saved_bytes(engine, monkeypatch):
    """Mayor grafo vivo al calcular una pérdida: pérdida, grafos retenidos y estado rápido."""
    parameters, peaks, current = list(engine.predictor.parameters()), [], {}
    loss, update = engine._loss, engine._update

    def tracked(run, at):
        current["run"] = run
        return update(run, at)

    def measured(prediction, target):
        run = current["run"]
        held = [v for v in (*run.graphs.values(), *run.predictions) if torch.is_tensor(v)]
        states = run.flows.values()
        fast = [t for s in states for t in (*s.mac.memory.weights, *s.mac.memory.momentum)]
        peaks.append(saved_graph_bytes([prediction, *held, *fast], exclude=parameters))
        return loss(prediction, target)

    monkeypatch.setattr(engine, "_update", tracked)
    monkeypatch.setattr(engine, "_loss", measured)
    cursor = dict(epoch=0, phase="train", event=0, stage="start")
    engine._train_pass(_Pass(), cursor, StopAtStep(10**9), None)
    return max(peaks)


def test_live_graph_at_the_loss_shrinks_with_the_block(shared, tmp_path, monkeypatch):  # noqa: F811
    _, streams = shared
    full = peak_saved_bytes(trainer(streams, tmp_path / "full"), monkeypatch)
    blocks = peak_saved_bytes(trainer(streams, tmp_path / "one", accumulation_rows=1), monkeypatch)
    # Con tres flujos, un bloque de uno conserva cerca de un tercio del grafo del tramo.
    assert 0 < blocks < 0.5 * full


def test_resume_with_accumulation_reproduces_the_continuous_run(shared, tmp_path):  # noqa: F811
    _, streams = shared
    options = dict(epochs=2, fixed=True, checkpoint_updates=2, accumulation_rows=1)
    continuous = trainer(streams, tmp_path / "continuous", **options)
    expected = continuous.run()
    stop = StopAtStep(15)
    first = trainer(streams, tmp_path / "interrupted", **options)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    second = trainer(streams, tmp_path / "interrupted", **options)
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert len(joined) == len(named_records(continuous))
    for left, right in zip(joined, named_records(continuous), strict=True):
        for key, value in right.items():
            if value is None:
                assert left[key] is None
            else:
                torch.testing.assert_close(left[key], value, rtol=0, atol=0)
    assert resumed["history"] == expected["history"]


@pytest.mark.parametrize(
    "fill",
    [
        lambda current: current.segment.append(object()),
        lambda current: current.starts.update({"US/A0000": None}),
        lambda current: current.graphs.update({("US/A0000", 1): object()}),
    ],
)
def test_recovery_is_only_exported_at_the_barrier_after_a_step(shared, tmp_path, fill):  # noqa: F811
    _, streams = shared
    engine = trainer(streams, tmp_path / "run", accumulation_rows=1)
    current = _Pass()
    fill(current)
    with pytest.raises(ValueError, match="barrera"):
        engine._export(current)


def test_saved_graph_bytes_counts_unique_storages_and_skips_excluded_tensors():
    weight = torch.ones(3, 4, dtype=torch.float64, requires_grad=True)
    inputs = torch.ones(5, 3, dtype=torch.float64, requires_grad=True)
    hidden = torch.tanh(inputs @ weight)
    output = (hidden * hidden).sum()
    # mm guarda la entrada (120 bytes) y el peso (96), tanh su salida (160) y mul la reutiliza.
    assert saved_graph_bytes([output], exclude=[weight]) == 120 + 160
    assert saved_graph_bytes([output]) == 120 + 96 + 160
    assert saved_graph_bytes([output, hidden], exclude=[weight]) == 280
    output.backward()
    assert saved_graph_bytes([output], exclude=[weight]) == 0
    assert saved_graph_bytes([inputs]) == 0


def test_flows_born_in_the_segment_send_gradient_to_the_initial_fast_weights(tmp_path):
    # Sin calentamiento, los flujos nacen en el primer tramo con M0 diferenciable.
    _, streams = corpus(tmp_path, train=phases(warmup="2022-11-15")[0])
    full = trainer(streams, tmp_path / "full")
    full.run()
    blocks = trainer(streams, tmp_path / "blocks", accumulation_rows=1)
    blocks.run()
    expected, actual = named_records(full), named_records(blocks)
    assert any(r["mac.memory.initial_weights.0"] is not None for r in expected)
    assert blocks.audit == full.audit
    for left, right in zip(actual, expected, strict=True):
        for key, value in right.items():
            if value is None:
                assert left[key] is None, key
            else:
                torch.testing.assert_close(left[key], value, rtol=1e-10, atol=1e-13)
