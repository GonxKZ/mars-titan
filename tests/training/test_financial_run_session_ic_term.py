"""Recorrido cronológico con el término de IC por sesión en la pérdida, sin pasos."""

import math
from collections import Counter

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.memory import financial_observations as observations
from mars_titan.models.baselines.multimodal import SCALAR_HEAD
from mars_titan.models.quantile_head import (
    LEVELS,
    MEDIAN_GAIN,
    PINBALL,
    QUANTILE_HEAD,
    median,
    pinball_loss,
)
from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.session_ic_objective import SESSION_IC_TERM, SessionICObjective
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.models.titans.test_session_ic_term import (
    numpy_session_term_reference,
    numpy_top_third_weights,
    reference,
    weighted_reference,
)
from tests.training.chronological_fixture import chronological_corpus, decision, phases
from tests.training.test_financial_run import SELECTION, RecordingOptimizer, corpus, entries
from tests.training.test_financial_run_auxiliary_horizons import AUXILIARY
from tests.training.test_financial_run_auxiliary_horizons import trainer as auxiliary_trainer
from tests.training.test_financial_run_weight_options import ScheduleFreeDouble

ASSETS = 5
TERM = dict(weight=2.0, minimum_rows=3)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def streams(root, *, perturb_after=None):
    dataset = CorpusDataset(
        chronological_corpus(root / "corpus", assets=ASSETS, perturb_after=perturb_after),
        input_policy=HISTORICAL_MASKED,
    )
    result = {}
    for phase in phases():
        manifest = observations.prepare_observation_index(
            dataset, root / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = observations.FinancialObservationSource(dataset, manifest)
    return result


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return streams(tmp_path_factory.mktemp("session-ic-term"))


def predictor(sources, *, head=QUANTILE_HEAD, **changes):
    # La memoria de la receta de campaña conserva la dispersión de la salida entre activos.
    options = dict(
        head=head,
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
    )
    config = FinancialConfig(
        sources["train"].specification(), hidden_size=32, variant="mac_online", **options | changes
    )
    return FinancialPredictor(config, dtype=torch.float64)


def trainer(sources, output, *, model=None, factory=RecordingOptimizer, **options):
    recipe = dict(
        truncation=3,
        epochs=1,
        block_rows=8,
        selection=SELECTION,
        loss=PINBALL,
        session_ic_term=TERM,
    )
    return ChronologicalTrainer(
        model or predictor(sources),
        ChronologicalRecipe(**recipe | options),
        train=sources["train"],
        validation=sources["validation"],
        output=output,
        optimizer_factory=factory,
        audit=True,
    )


def records(engine):
    names = {id(p): name for name, p in engine.predictor.named_parameters()}
    return [{names[k]: v for k, v in record.items()} for record in engine.optimizer.records]


class Spy:
    """Registra cada llamada al término sin cambiar su valor."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        # El recorrido lee del término su columna de liquidez.
        return getattr(self.inner, name)

    def term(self, point, target, sessions, liquidity=None, offset=None):
        value = self.inner.term(point, target, sessions, liquidity, offset)
        self.calls.append(
            (point.detach().clone(), target.detach().clone(), list(sessions), value.detach())
        )
        return value


def test_each_update_adds_the_term_of_its_whole_sessions_to_the_pinball_loss(shared, tmp_path):
    engine = trainer(shared, tmp_path / "run")
    engine.session_ic = spy = Spy(engine.session_ic)
    before = engine.predictor._parameter_id
    report = engine.run()
    assert report["status"] == "completed"
    train = report["history"][1]["train"]
    updates = entries(engine.audit, "update")
    assert len(spy.calls) == len(updates) == train["updates"] > 2
    labels = Counter(
        (flow.split("/")[0], at) for _, _, flow, at, _ in entries(engine.audit, "label", "train")
    )
    seen = Counter(key for _, _, sessions, _ in spy.calls for key in sessions)
    # Cada sesión entra entera en una sola actualización, con todas sus etiquetas maduras.
    assert seen == labels and max(labels.values()) == ASSETS
    for point, target, sessions, value in spy.calls:
        # El punto es la mediana emitida, un valor por etiqueta.
        assert point.shape == target.shape == (len(sessions),)
        expected = reference(
            point.tolist(), target.tolist(), sessions, TERM["weight"], TERM["minimum_rows"]
        )
        assert value.item() == pytest.approx(expected, rel=1e-7, abs=1e-12)
    assert any(value.item() != 0 for *_, value in spy.calls)
    gradients = records(engine)
    assert all(record["head.weight"].abs().sum() > 0 for record in gradients)
    assert all(record["fusion.0.weight"].abs().sum() > 0 for record in gradients)
    # El recorrido con RecordingOptimizer no modifica los pesos.
    assert engine.predictor._parameter_id == before


def test_the_loss_of_a_segment_is_the_declared_loss_plus_the_term(shared, tmp_path):
    engine = trainer(shared, tmp_path / "loss")
    generator = torch.Generator().manual_seed(7)
    prediction = torch.randn(12, 5, dtype=torch.float64, generator=generator).sort(dim=1).values
    target = torch.randn(12, dtype=torch.float64, generator=generator)
    keys = [(f"US/A{row % 6:04d}", row // 6) for row in range(12)]
    sessions = [("US", at) for _, at in keys]
    expected = pinball_loss(prediction, target).item() + reference(
        median(prediction).tolist(), target.tolist(), sessions, TERM["weight"], 3
    )
    assert engine._task_loss(prediction, target, keys).item() == pytest.approx(expected, rel=1e-9)
    # Sin el término, la pérdida del tramo es la declarada, bit a bit.
    plain = trainer(shared, tmp_path / "plain", session_ic_term=None)
    assert plain.session_ic is None
    torch.testing.assert_close(
        plain._task_loss(prediction, target, keys), pinball_loss(prediction, target), rtol=0, atol=0
    )


def test_the_term_changes_the_gradients_and_not_what_is_emitted(shared, tmp_path):
    # Sin recorte del gradiente, que reescalaría todas las filas con la norma del conjunto.
    with_term = trainer(shared, tmp_path / "with", max_grad_norm=None)
    with_term.run()
    without = trainer(shared, tmp_path / "without", max_grad_norm=None, session_ic_term=None)
    without.run()
    # RecordingOptimizer no mueve los pesos: las dos ejecuciones emiten lo mismo.
    assert entries(with_term.audit, "prediction", "train") == entries(
        without.audit, "prediction", "train"
    )
    left, right = records(with_term), records(without)
    assert len(left) == len(right) > 2
    assert any(
        not torch.equal(first["head.weight"], second["head.weight"])
        for first, second in zip(left, right, strict=True)
    )
    # El término solo llega a la fila de la mediana de la cabeza: las colas reciben lo mismo.
    for first, second in zip(left, right, strict=True):
        torch.testing.assert_close(
            first["head.weight"][[0, 1, 3, 4]], second["head.weight"][[0, 1, 3, 4]], rtol=0, atol=0
        )


def test_the_scalar_head_uses_its_output_as_the_point(shared, tmp_path):
    model = predictor(shared, head=SCALAR_HEAD)
    engine = trainer(shared, tmp_path / "scalar", model=model, loss="mae")
    engine.session_ic = spy = Spy(engine.session_ic)
    assert engine.run()["status"] == "completed"
    assert spy.calls and all(point.ndim == 1 for point, *_ in spy.calls)
    generator = torch.Generator().manual_seed(3)
    prediction = torch.randn(10, dtype=torch.float64, generator=generator)
    target = torch.randn(10, dtype=torch.float64, generator=generator)
    keys = [(f"US/A{row % 5:04d}", row // 5) for row in range(10)]
    expected = (prediction - target).abs().mean().item() + reference(
        prediction.tolist(), target.tolist(), [("US", at) for _, at in keys], TERM["weight"], 3
    )
    value = trainer(shared, tmp_path / "direct", model=model, loss="mae")._task_loss(
        prediction, target, keys
    )
    assert value.item() == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("truncation", [3, 1])
def test_replay_by_flow_blocks_reproduces_the_direct_gradients(shared, tmp_path, truncation):
    direct = trainer(shared, tmp_path / "direct", truncation=truncation)
    direct.run()
    replayed = trainer(shared, tmp_path / "replayed", truncation=truncation, accumulation_rows=8)
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
    # El término necesita todas las etiquetas de cada sesión en el mismo bloque.
    split = trainer(shared, tmp_path / "split", accumulation_rows=2)
    with pytest.raises(ValueError, match="repetir todos los flujos"):
        split.run()
    # Sin el término, el mismo reparto en bloques pequeños sí se admite.
    plain = trainer(shared, tmp_path / "plain", accumulation_rows=2, session_ic_term=None)
    assert plain.run()["status"] == "completed"


def test_future_suffix_does_not_change_past_predictions_or_gradients(tmp_path):
    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        engine = trainer(streams(tmp_path / name, perturb_after=perturb), tmp_path / name / "run")
        engine.run()
        engines.append(engine)
    base, future = engines

    def past(engine):
        return [e for e in entries(engine.audit, "prediction", "train") if e[3] <= cut]

    assert past(base) and past(base) == past(future)
    assert entries(base.audit, "prediction", "train") != entries(
        future.audit, "prediction", "train"
    )
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps
    for left, right in zip(records(base)[: len(steps)], records(future), strict=False):
        for key, value in left.items():
            torch.testing.assert_close(right[key], value, rtol=0, atol=0)


def test_the_recipe_identity_only_names_the_term_when_declared(shared, tmp_path):
    plain = ChronologicalRecipe(loss=PINBALL, selection=SELECTION)
    explicit = ChronologicalRecipe(loss=PINBALL, selection=SELECTION, session_ic_term=None)
    declared = ChronologicalRecipe(loss=PINBALL, selection=SELECTION, session_ic_term=TERM)
    assert "session_ic_term" not in plain.identity()
    assert explicit.identity() == plain.identity()
    identity = declared.identity()
    assert identity["session_ic_term"]["name"] == SESSION_IC_TERM
    assert identity["session_ic_term"]["weight"] == 2.0
    assert {k: v for k, v in identity.items() if k != "session_ic_term"} == plain.identity()
    other = ChronologicalRecipe(
        loss=PINBALL, selection=SELECTION, session_ic_term=dict(weight=3.0, minimum_rows=3)
    )
    assert other.identity() != identity
    engine = trainer(shared, tmp_path / "identity")
    assert engine.identity["recipe"]["session_ic_term"] == identity["session_ic_term"]
    assert "mars_titan.models.titans.session_ic_objective" in engine.identity["implementation"]


@pytest.mark.parametrize(
    "value",
    [
        dict(weight=1.0),
        dict(weight=0.0, minimum_rows=8),
        dict(weight=1.0, minimum_rows=2),
        dict(weight=1.0, minimum_rows=8, otra=1),
        "session_ic_term_v1",
        2.0,
    ],
)
def test_invalid_terms_are_rejected_by_the_recipe(value):
    with pytest.raises(ValueError):
        ChronologicalRecipe(loss=PINBALL, selection=SELECTION, session_ic_term=value)


def test_the_term_is_not_combined_with_distillation_or_an_output_objective(shared, tmp_path):
    with pytest.raises(ValueError, match="destilación"):
        ChronologicalRecipe(
            loss=PINBALL, selection=SELECTION, session_ic_term=TERM, distillation_weight=0.5
        )
    model = predictor(
        shared, head=SCALAR_HEAD, session_ic_objective=SessionICObjective(minimum_rows=3)
    )
    with pytest.raises(ValueError, match="objetivo de salida"):
        trainer(shared, tmp_path / "objective", model=model, loss="mae")


def test_the_term_is_added_with_cross_asset_attention_and_its_replay(shared, tmp_path):
    def model():
        mixer = CrossAssetAttentionConfig(mode="same_session")
        return predictor(shared, cross_asset_attention=mixer)

    direct = trainer(shared, tmp_path / "direct", model=model())
    direct.session_ic = spy = Spy(direct.session_ic)
    assert direct.run()["status"] == "completed"
    assert spy.calls and any(value.item() != 0 for *_, value in spy.calls)
    replayed = trainer(shared, tmp_path / "replayed", model=model(), accumulation_rows=8)
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)


def test_the_term_does_not_change_the_gradient_of_the_auxiliary_heads(tmp_path):
    _, sources = corpus(tmp_path / "corpus")
    reference = auxiliary_trainer(sources, tmp_path / "reference")
    engine = auxiliary_trainer(sources, tmp_path / "term", session_ic_term=TERM)
    engine.session_ic = spy = Spy(engine.session_ic)
    reports = reference.run(), engine.run()
    assert all(report["status"] == "completed" for report in reports)
    assert spy.calls
    counters = [report["history"][1]["train"] for report in reports]
    assert counters[0]["auxiliary_labels_in_loss"] == counters[1]["auxiliary_labels_in_loss"] > 0
    names = {id(p): name for name, p in engine.predictor.named_parameters()}
    others = {id(p): name for name, p in reference.predictor.named_parameters()}
    left = [{names[k]: v for k, v in record.items()} for record in engine.optimizer.records]
    right = [{others[k]: v for k, v in record.items()} for record in reference.optimizer.records]
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name in AUXILIARY:
            assert (first[name] is None) == (second[name] is None)
            if first[name] is not None:
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [dict(weight_average=dict(decay=0.9)), dict(schedule_free=dict(warmup_steps=0, beta1=0.9))],
)
def test_the_term_runs_with_the_weight_options(shared, tmp_path, options):
    factory = ScheduleFreeDouble if "schedule_free" in options else RecordingOptimizer
    engine = trainer(shared, tmp_path / "run", factory=factory, **options)
    engine.session_ic = spy = Spy(engine.session_ic)
    assert engine.run()["status"] == "completed"
    assert spy.calls and any(value.item() != 0 for *_, value in spy.calls)
    assert set(options) <= set(engine.identity["recipe"])


LIQUID = dict(weight=2.0, minimum_rows=3, liquidity_column=-1, liquidity_power=1.0)


def test_the_weighted_term_receives_the_liquidity_of_each_labelled_row(shared, tmp_path):
    engine = trainer(shared, tmp_path / "run", session_ic_term=LIQUID)
    assert engine.liquidity
    received, losses = {}, []
    graphs, task = engine._graphs, engine._task_loss

    def spy_graphs(prepared, batch):
        column = batch.inputs["charts"][:, -1].tolist()
        for flow, at, value in zip(batch.flow_ids, batch.prediction_at, column, strict=True):
            received[flow, at] = value
        outputs = graphs(prepared, batch)
        # Cinco cuantiles y, al final, la liquidez de la fila sin gradiente.
        assert outputs.shape == (len(batch.flow_ids), 6)
        torch.testing.assert_close(outputs[:, :5], prepared.quantiles, rtol=0, atol=0)
        return outputs

    def spy_task(prediction, target, keys):
        losses.append((prediction.detach().clone(), target.clone(), list(keys)))
        return task(prediction, target, keys)

    engine._graphs, engine._task_loss = spy_graphs, spy_task
    engine.session_ic = spy = Spy(engine.session_ic)
    assert engine.run()["status"] == "completed"
    assert len(losses) == len(spy.calls) > 2
    for prediction, target, keys in losses:
        assert prediction.shape == (len(keys), 6)
        assert prediction[:, -1].tolist() == [received[key] for key in keys]
        sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
        expected = pinball_loss(prediction[:, :5], target).item() + weighted_reference(
            median(prediction[:, :5]).tolist(),
            target.tolist(),
            sessions,
            prediction[:, -1].tolist(),
            LIQUID["weight"],
            LIQUID["minimum_rows"],
            LIQUID["liquidity_power"],
        )
        assert task(prediction, target, keys).item() == pytest.approx(expected, rel=1e-7, abs=1e-12)
    # La liquidez distingue filas en el corpus de prueba: no es una columna constante.
    assert len(set(received.values())) > 1
    gradients = records(engine)
    assert all(record["head.weight"].abs().sum() > 0 for record in gradients)
    identity = engine.identity["recipe"]["session_ic_term"]
    assert identity["liquidity_column"] == -1 and identity["liquidity_power"] == 1.0


@pytest.mark.parametrize("truncation", [3, 1])
def test_replay_reproduces_the_gradients_of_the_weighted_term(shared, tmp_path, truncation):
    direct = trainer(shared, tmp_path / "direct", truncation=truncation, session_ic_term=LIQUID)
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / "replayed",
        truncation=truncation,
        accumulation_rows=8,
        session_ic_term=LIQUID,
    )
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
    # Y la ponderación cambia los gradientes respecto del término sin ponderar.
    plain = trainer(shared, tmp_path / "plain", truncation=truncation)
    plain.run()
    assert any(
        not torch.equal(first["head.weight"], second["head.weight"])
        for first, second in zip(left, records(plain), strict=True)
    )


def test_the_weighted_term_with_the_scalar_head_and_its_column_check(shared, tmp_path):
    model = predictor(shared, head=SCALAR_HEAD)
    engine = trainer(shared, tmp_path / "scalar", model=model, loss="mae", session_ic_term=LIQUID)
    engine.session_ic = spy = Spy(engine.session_ic)
    assert engine.run()["status"] == "completed"
    assert spy.calls and all(point.ndim == 1 for point, *_ in spy.calls)
    width = shared["train"].specification().dimensions["charts"]
    beyond = dict(LIQUID, liquidity_column=-(width + 1))
    with pytest.raises(ValueError, match="bloque del gráfico"):
        trainer(shared, tmp_path / "beyond", session_ic_term=beyond)
    assert trainer(shared, tmp_path / "edge", session_ic_term=dict(LIQUID, liquidity_column=-width))


def test_the_gain_of_the_median_is_trained_by_the_quantile_loss_and_not_by_the_term(
    shared, tmp_path
):
    def model():
        return predictor(shared, median_gain=MEDIAN_GAIN)

    with_term = trainer(shared, tmp_path / "with", model=model(), max_grad_norm=None)
    with_term.run()
    without = trainer(
        shared, tmp_path / "without", model=model(), max_grad_norm=None, session_ic_term=None
    )
    without.run()
    left, right = records(with_term), records(without)
    assert len(left) == len(right) > 2
    scale = max(record["head.median_gain"].abs().item() for record in right)
    assert scale > 0
    for first, second in zip(left, right, strict=True):
        # El término no depende de la escala de la mediana: a la ganancia solo le llega pinball.
        torch.testing.assert_close(
            first["head.median_gain"], second["head.median_gain"], rtol=1e-6, atol=1e-8 * scale
        )
    # Y el término sí cambia el gradiente de la dirección de la mediana.
    assert any(
        not torch.equal(first["head.weight"][2], second["head.weight"][2])
        for first, second in zip(left, right, strict=True)
    )
    assert with_term.identity["predictor"]["median_gain"]["name"] == MEDIAN_GAIN


SHIFTED = dict(weight=2.0, minimum_rows=3, offset_column=-2, offset_scale=0.5)


def spied(engine):
    """Espía las salidas que van a la pérdida y cada pérdida de tramo, sin cambiarlas."""
    charts, losses = {}, []
    graphs, task = engine._graphs, engine._task_loss

    def spy_graphs(prepared, batch):
        for row, key in enumerate(zip(batch.flow_ids, batch.prediction_at, strict=True)):
            charts[key] = batch.inputs["charts"][row].tolist()
        outputs = graphs(prepared, batch)
        torch.testing.assert_close(outputs[:, :5], prepared.quantiles, rtol=0, atol=0)
        return outputs

    def spy_task(prediction, target, keys):
        losses.append((prediction.detach().clone(), target.clone(), list(keys)))
        return task(prediction, target, keys)

    engine._graphs, engine._task_loss = spy_graphs, spy_task
    return charts, losses, task


def test_the_shifted_term_receives_the_known_offset_of_each_labelled_row(shared, tmp_path):
    engine = trainer(shared, tmp_path / "run", session_ic_term=SHIFTED)
    assert engine.offset and not engine.liquidity
    charts, losses, task = spied(engine)
    assert engine.run()["status"] == "completed"
    assert len(losses) > 2
    seen = set()
    for prediction, target, keys in losses:
        # Cinco cuantiles y, al final, el desplazamiento conocido de la fila.
        assert prediction.shape == (len(keys), 6)
        expected_offset = [
            0.5 * charts[key][-2] if math.isfinite(charts[key][-2]) else 0.0 for key in keys
        ]
        assert prediction[:, -1].tolist() == expected_offset
        seen.update(expected_offset)
        sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
        offset = prediction[:, -1]
        expected = pinball_loss(prediction[:, :5], target).item() + reference(
            (median(prediction[:, :5]) + offset).tolist(),
            (target + offset).tolist(),
            sessions,
            SHIFTED["weight"],
            SHIFTED["minimum_rows"],
        )
        assert task(prediction, target, keys).item() == pytest.approx(expected, rel=1e-7, abs=1e-12)
    # El desplazamiento distingue filas en el corpus de prueba.
    assert len(seen) > 1
    identity = engine.identity["recipe"]["session_ic_term"]
    assert identity["offset_column"] == -2 and identity["offset_scale"] == 0.5


def test_the_liquidity_and_the_offset_travel_together(shared, tmp_path):
    both = dict(LIQUID, offset_column=-2, offset_scale=0.5)
    engine = trainer(shared, tmp_path / "run", session_ic_term=both)
    assert engine.offset and engine.liquidity
    charts, losses, task = spied(engine)
    assert engine.run()["status"] == "completed"
    for prediction, target, keys in losses:
        assert prediction.shape == (len(keys), 7)
        assert prediction[:, 5].tolist() == [charts[key][-1] for key in keys]
        assert prediction[:, 6].tolist() == [0.5 * charts[key][-2] for key in keys]
        sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
        offset = prediction[:, 6]
        expected = pinball_loss(prediction[:, :5], target).item() + weighted_reference(
            (median(prediction[:, :5]) + offset).tolist(),
            (target + offset).tolist(),
            sessions,
            prediction[:, 5].tolist(),
            LIQUID["weight"],
            LIQUID["minimum_rows"],
            LIQUID["liquidity_power"],
        )
        assert task(prediction, target, keys).item() == pytest.approx(expected, rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("truncation", [3, 1])
def test_replay_reproduces_the_gradients_of_the_shifted_term(shared, tmp_path, truncation):
    direct = trainer(shared, tmp_path / "direct", truncation=truncation, session_ic_term=SHIFTED)
    direct.run()
    replayed = trainer(
        shared,
        tmp_path / "replayed",
        truncation=truncation,
        accumulation_rows=8,
        session_ic_term=SHIFTED,
    )
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
    # Y el desplazamiento cambia los gradientes respecto del término de siempre.
    plain = trainer(shared, tmp_path / "plain", truncation=truncation)
    plain.run()
    assert any(
        not torch.equal(first["head.weight"], second["head.weight"])
        for first, second in zip(left, records(plain), strict=True)
    )


def test_the_shifted_term_also_trains_the_gain_of_the_median(shared, tmp_path):
    def model():
        return predictor(shared, median_gain=MEDIAN_GAIN)

    shifted = trainer(
        shared, tmp_path / "with", model=model(), max_grad_norm=None, session_ic_term=SHIFTED
    )
    shifted.run()
    without = trainer(
        shared, tmp_path / "without", model=model(), max_grad_norm=None, session_ic_term=None
    )
    without.run()
    left, right = records(shifted), records(without)
    assert len(left) == len(right) > 2
    # Sumado al alfa conocido, el orden depende de la escala de la mediana: el término la ajusta.
    assert any(
        not torch.allclose(first["head.median_gain"], second["head.median_gain"], rtol=1e-6)
        for first, second in zip(left, right, strict=True)
    )


def test_the_offset_column_belongs_to_the_chart_block(shared, tmp_path):
    width = shared["train"].specification().dimensions["charts"]
    beyond = dict(SHIFTED, offset_column=-(width + 1))
    with pytest.raises(ValueError, match="bloque del gráfico"):
        trainer(shared, tmp_path / "beyond", session_ic_term=beyond)
    assert trainer(shared, tmp_path / "edge", session_ic_term=dict(SHIFTED, offset_column=-width))


def test_the_two_terms_are_added_to_the_declared_loss(shared, tmp_path):
    both = dict(SHIFTED, plain_weight=1.5)
    engine = trainer(shared, tmp_path / "run", session_ic_term=both)
    assert engine.offset and not engine.liquidity
    _charts, losses, task = spied(engine)
    assert engine.run()["status"] == "completed"
    assert len(losses) > 2
    apart = 0
    for prediction, target, keys in losses:
        sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
        offset, point = prediction[:, -1], median(prediction[:, :5])
        operable = reference(
            (point + offset).tolist(),
            (target + offset).tolist(),
            sessions,
            SHIFTED["weight"],
            SHIFTED["minimum_rows"],
        )
        label = reference(point.tolist(), target.tolist(), sessions, 1.5, SHIFTED["minimum_rows"])
        expected = pinball_loss(prediction[:, :5], target).item() + operable + label
        assert task(prediction, target, keys).item() == pytest.approx(expected, rel=1e-7, abs=1e-12)
        apart += abs(label) > 1e-6
    # El término de la etiqueta cuenta de verdad en el corpus de prueba.
    assert apart > 0
    identity = engine.identity["recipe"]["session_ic_term"]
    assert identity["plain_weight"] == 1.5 and identity["offset_column"] == -2
    # Sin el peso de la etiqueta la identidad es la del término desplazado de siempre.
    alone = trainer(shared, tmp_path / "alone", session_ic_term=SHIFTED)
    assert "plain_weight" not in alone.identity["recipe"]["session_ic_term"]


def _numpy_weighted_pinball(quantiles, target, sessions, liquidity, liquidity_weight):
    """Media ponderada independiente del pinball con pesos del tercio líquido por sesión."""
    import numpy as np

    q = np.asarray(quantiles, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)[:, None]
    tau = np.asarray(LEVELS, dtype=np.float64)[None, :]
    residual = y - q
    per_row = np.maximum(tau * residual, (tau - 1.0) * residual).mean(axis=-1)
    weights = np.ones(len(per_row), dtype=np.float64)
    groups = {}
    for row, key in enumerate(sessions):
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        w_session = numpy_top_third_weights([liquidity[row] for row in rows], liquidity_weight)
        for idx, row in enumerate(rows):
            weights[row] = w_session[idx]
    return float((weights * per_row).sum() / weights.sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wp1_neutral_options_match_baseline_task_loss_and_gradients_bitwise(
    shared, tmp_path, dtype
):
    generator = torch.Generator().manual_seed(19)
    prediction = torch.randn(18, 5, dtype=dtype, generator=generator).sort(dim=1).values
    target = torch.randn(18, dtype=dtype, generator=generator)
    keys = [(f"{'US' if row % 2 == 0 else 'CN'}/A{row:04d}", row // 6) for row in range(18)]
    base_engine = trainer(shared, tmp_path / "base", session_ic_term=TERM)
    base_pred = prediction.clone().requires_grad_(True)
    base_loss = base_engine._task_loss(base_pred, target, keys)
    (base_grad,) = torch.autograd.grad(base_loss, base_pred)

    for idx, opts in enumerate(
        (
            dict(TERM, liquidity_weight=1.0),
            dict(TERM, stability=0.0),
            dict(TERM, liquidity_weight=1.0, stability=0.0),
            dict(TERM, liquidity_column=-1, liquidity_weight=1.0, stability=0.0),
        )
    ):
        eng = trainer(shared, tmp_path / f"neutral-{idx}", session_ic_term=opts)
        assert not eng.liquidity
        assert (
            eng.identity["recipe"]["session_ic_term"]
            == (base_engine.identity["recipe"]["session_ic_term"])
        )
        pred = prediction.clone().requires_grad_(True)
        loss = eng._task_loss(pred, target, keys)
        (grad,) = torch.autograd.grad(loss, pred)
        torch.testing.assert_close(loss, base_loss, rtol=0, atol=0)
        torch.testing.assert_close(grad, base_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_liquidity_weight_weights_pinball_and_ic_and_handles_missing_and_constant(
    shared, tmp_path, dtype
):
    generator = torch.Generator().manual_seed(29)
    # 3 sesiones completas (6 filas cada una, dos mercados) + 1 sesión corta (2 filas).
    keys = (
        [(f"US/A{i:04d}", 1) for i in range(6)]
        + [(f"US/A{i:04d}", 2) for i in range(6)]
        + [(f"CN/B{i:04d}", 1) for i in range(6)]
        + [(f"CN/B{i:04d}", 9) for i in range(2)]
    )
    rows = len(keys)
    sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
    quantiles = torch.randn(rows, 5, dtype=dtype, generator=generator).sort(dim=1).values
    target = torch.randn(rows, dtype=dtype, generator=generator)
    liquidity = torch.tensor(
        [
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            float("nan"),
            -5.0,
            -3.0,
            -1.0,
            1.0,
            float("inf"),
            0.5,
            1.5,
        ],
        dtype=dtype,
    )
    offset = 0.2 * torch.randn(rows, dtype=dtype, generator=generator)

    # Modo etiqueta + liquidity_weight + stability.
    opts = dict(
        weight=2.0, minimum_rows=3, liquidity_column=-1, liquidity_weight=3.0, stability=0.5
    )
    eng = trainer(shared, tmp_path / "label", session_ic_term=opts)
    assert eng.liquidity
    packed = torch.cat([quantiles, liquidity.unsqueeze(1)], dim=1).requires_grad_(True)
    loss = eng._task_loss(packed, target, keys)
    exp_pinball = _numpy_weighted_pinball(
        quantiles.tolist(), target.tolist(), sessions, liquidity.tolist(), 3.0
    )
    exp_ic = numpy_session_term_reference(
        median(quantiles).tolist(),
        target.tolist(),
        sessions,
        weight=2.0,
        minimum_rows=3,
        liquidity=liquidity.tolist(),
        liquidity_weight=3.0,
        stability=0.5,
    )
    rel = 1e-5 if dtype == torch.float32 else 1e-8
    assert loss.item() == pytest.approx(exp_pinball + exp_ic, rel=rel, abs=1e-6)
    (grad,) = torch.autograd.grad(loss, packed)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[:, -1]) == 0
    # Comprobar directamente que pinball pondera las colas ([0, 1, 3, 4], sin término de IC):
    # en US/1, filas 4-5 son el tercio líquido (peso 3) y 0-3 pesan 1.
    plain_eng = trainer(shared, tmp_path / "plain", session_ic_term=TERM)
    plain_q = quantiles.clone().requires_grad_(True)
    (plain_grad,) = torch.autograd.grad(plain_eng._task_loss(plain_q, target, keys), plain_q)
    # Razón entre gradiente de colas (fila líquida vs no líquida) escalada por la razón de pesos.
    liq_ratio = (grad[5, 0] / plain_grad[5, 0]).item()
    nonliq_ratio = (grad[0, 0] / plain_grad[0, 0]).item()
    assert liq_ratio / nonliq_ratio == pytest.approx(3.0, rel=1e-5)
    # En CN/1 las filas con NaN e inf (índices 12 y 17) pesan 1, igual que la fila 0.
    nan_ratio = (grad[12, 0] / plain_grad[12, 0]).item()
    inf_ratio = (grad[17, 0] / plain_grad[17, 0]).item()
    assert nan_ratio / nonliq_ratio == pytest.approx(1.0, rel=1e-5)
    assert inf_ratio / nonliq_ratio == pytest.approx(1.0, rel=1e-5)

    # Modo operable + plain_weight + liquidity_weight + stability.
    both_opts = dict(opts, offset_column=-2, offset_scale=0.5, plain_weight=1.5)
    both_eng = trainer(shared, tmp_path / "both", session_ic_term=both_opts)
    packed_both = torch.cat(
        [quantiles, liquidity.unsqueeze(1), offset.unsqueeze(1)], dim=1
    ).requires_grad_(True)
    loss_both = both_eng._task_loss(packed_both, target, keys)
    exp_op = numpy_session_term_reference(
        (median(quantiles) + offset).tolist(),
        (target + offset).tolist(),
        sessions,
        weight=2.0,
        minimum_rows=3,
        liquidity=liquidity.tolist(),
        liquidity_weight=3.0,
        stability=0.5,
    )
    exp_plain = numpy_session_term_reference(
        median(quantiles).tolist(),
        target.tolist(),
        sessions,
        weight=1.5,
        minimum_rows=3,
        liquidity=liquidity.tolist(),
        liquidity_weight=3.0,
        stability=0.5,
    )
    assert loss_both.item() == pytest.approx(exp_pinball + exp_op + exp_plain, rel=rel, abs=1e-6)
    (grad_both,) = torch.autograd.grad(loss_both, packed_both)
    assert torch.isfinite(grad_both).all()

    # Predicción constante cero y liquidez completamente ausente -> gradientes finitos.
    zeros = torch.zeros_like(quantiles)
    all_nan = torch.full_like(liquidity, float("nan"))
    packed_zero = torch.cat([zeros, all_nan.unsqueeze(1)], dim=1).requires_grad_(True)
    loss_zero = eng._task_loss(packed_zero, target, keys)
    (grad_zero,) = torch.autograd.grad(loss_zero, packed_zero)
    assert torch.isfinite(loss_zero) and torch.isfinite(grad_zero).all()

    # Columna fuera del bloque del gráfico rechazada incluso con liquidity_weight=1.0.
    width = shared["train"].specification().dimensions["charts"]
    with pytest.raises(ValueError, match="bloque del gráfico"):
        trainer(
            shared,
            tmp_path / "oob",
            session_ic_term=dict(TERM, liquidity_column=-(width + 1), liquidity_weight=1.0),
        )


def test_absent_charts_presence_marks_liquidity_unknown_in_graphs(shared, tmp_path):
    from types import SimpleNamespace

    from mars_titan.models.titans.financial import DecisionBatch
    from mars_titan.training.financial_run import validated_cpu_batches

    opts = dict(weight=2.0, minimum_rows=3, liquidity_column=-1, liquidity_weight=3.0)
    eng = trainer(shared, tmp_path / "charts-absent", session_ic_term=opts)
    event = next(event for event in shared["train"].batched_events(block_rows=8) if event.inputs)
    cpu = validated_cpu_batches(event.inputs, eng.predictor.config.inputs)[0]
    batch = DecisionBatch.from_validated(
        cpu, device=eng.device, dtype=eng.predictor.head.weight.dtype
    )
    state = eng.predictor.initial_state(batch.flow_ids, differentiable=False)
    prepared = eng.predictor.prepare(batch, state, differentiable=False, session_complete=True)
    masked_presence = batch.presence.clone()
    masked_presence[0, MODALITIES.index("charts")] = False
    masked_batch = SimpleNamespace(inputs=batch.inputs, presence=masked_presence)
    out = eng._graphs(prepared, masked_batch)
    assert torch.isnan(out[0, -1])
    assert torch.isfinite(out[1:, -1]).all()
    # La fila con charts ausente recibe peso 1 y gradiente finito en _task_loss.
    target = torch.zeros(len(batch.flow_ids), dtype=out.dtype)
    keys = list(zip(batch.flow_ids, batch.prediction_at, strict=True))
    packed = out.detach().clone().requires_grad_(True)
    loss = eng._task_loss(packed, target, keys)
    (grad,) = torch.autograd.grad(loss, packed)
    assert torch.isfinite(loss) and torch.isfinite(grad).all()


def test_wp1_options_run_end_to_end_with_replay_scalar_head_and_causal_no_leakage(shared, tmp_path):
    wp1 = dict(
        weight=2.0,
        minimum_rows=3,
        liquidity_column=-1,
        liquidity_weight=3.0,
        stability=0.5,
        offset_column=-2,
        offset_scale=0.5,
        plain_weight=1.5,
    )
    direct = trainer(shared, tmp_path / "direct", truncation=3, session_ic_term=wp1)
    charts, losses, task = spied(direct)
    assert direct.run()["status"] == "completed"
    assert len(losses) > 2
    for prediction, target, keys in losses:
        assert prediction.shape == (len(keys), 7)
        sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
        quantiles, liq, off = prediction[:, :5], prediction[:, 5], prediction[:, 6]
        exp_pinball = _numpy_weighted_pinball(
            quantiles.tolist(), target.tolist(), sessions, liq.tolist(), 3.0
        )
        exp_op = numpy_session_term_reference(
            (median(quantiles) + off).tolist(),
            (target + off).tolist(),
            sessions,
            weight=2.0,
            minimum_rows=3,
            liquidity=liq.tolist(),
            liquidity_weight=3.0,
            stability=0.5,
        )
        exp_plain = numpy_session_term_reference(
            median(quantiles).tolist(),
            target.tolist(),
            sessions,
            weight=1.5,
            minimum_rows=3,
            liquidity=liq.tolist(),
            liquidity_weight=3.0,
            stability=0.5,
        )
        assert task(prediction, target, keys).item() == pytest.approx(
            exp_pinball + exp_op + exp_plain, rel=1e-7, abs=1e-12
        )
    identity = direct.identity["recipe"]["session_ic_term"]
    assert identity["liquidity_weight"] == 3.0 and identity["stability"] == 0.5

    # Replay por bloques reproduce los gradientes directos.
    replayed = trainer(
        shared, tmp_path / "replayed", truncation=3, accumulation_rows=8, session_ic_term=wp1
    )
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)

    # Cabeza escalar con mae también funciona con liquidity_weight y stability.
    scalar_model = predictor(shared, head=SCALAR_HEAD)
    scalar_engine = trainer(
        shared,
        tmp_path / "scalar",
        model=scalar_model,
        loss="mae",
        session_ic_term=dict(
            weight=2.0, minimum_rows=3, liquidity_column=-1, liquidity_weight=3.0, stability=0.5
        ),
    )
    assert scalar_engine.run()["status"] == "completed"

    # Sin fugas del futuro: perturbar el futuro tras `cut` no cambia ni un bit hasta `cut`.
    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        eng = trainer(
            streams(tmp_path / name, perturb_after=perturb),
            tmp_path / name / "run",
            session_ic_term=wp1,
        )
        eng.run()
        engines.append(eng)
    base, future = engines
    past_base = [e for e in entries(base.audit, "prediction", "train") if e[3] <= cut]
    past_future = [e for e in entries(future.audit, "prediction", "train") if e[3] <= cut]
    assert past_base and past_base == past_future
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps
    for first, second in zip(records(base)[: len(steps)], records(future), strict=False):
        for key, value in first.items():
            torch.testing.assert_close(second[key], value, rtol=0, atol=0)


def test_liquidity_weight_weights_scalar_and_point_declared_losses(shared, tmp_path):
    generator = torch.Generator().manual_seed(43)
    keys = [(f"US/A{i:04d}", 1) for i in range(6)] + [(f"US/A{i:04d}", 2) for i in range(6)]
    sessions = [(flow.split("/", 1)[0], at) for flow, at in keys]
    target = torch.randn(12, dtype=torch.float64, generator=generator)
    liquidity = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2, dtype=torch.float64)
    row_w = torch.tensor([1.0, 1.0, 1.0, 1.0, 4.0, 4.0] * 2, dtype=torch.float64)
    opts = dict(weight=2.0, minimum_rows=3, liquidity_column=-1, liquidity_weight=4.0)
    scalar_pred = torch.randn(12, dtype=torch.float64, generator=generator)
    packed_scalar = torch.stack([scalar_pred, liquidity], dim=1)
    ic_scalar = numpy_session_term_reference(
        scalar_pred.tolist(),
        target.tolist(),
        sessions,
        weight=2.0,
        minimum_rows=3,
        liquidity=liquidity.tolist(),
        liquidity_weight=4.0,
    )
    for kind, per_row in (
        ("mae", (scalar_pred - target).abs()),
        ("mse", (scalar_pred - target).square()),
        (
            "huber",
            torch.nn.functional.huber_loss(scalar_pred, target, reduction="none", delta=0.01),
        ),
    ):
        eng = trainer(
            shared,
            tmp_path / f"scalar-{kind}",
            model=predictor(shared, head=SCALAR_HEAD),
            loss=kind,
            session_ic_term=opts,
        )
        expected = ((row_w * per_row).sum() / row_w.sum()).item() + ic_scalar
        assert eng._task_loss(packed_scalar, target, keys).item() == pytest.approx(
            expected, rel=1e-8
        )

    from mars_titan.training.quantile_objectives import POINT_COMPONENT, POINT_MAE

    quantiles = torch.randn(12, 5, dtype=torch.float64, generator=generator).sort(dim=1).values
    packed_q = torch.cat([quantiles, liquidity.unsqueeze(1)], dim=1)
    ic_q = numpy_session_term_reference(
        median(quantiles).tolist(),
        target.tolist(),
        sessions,
        weight=2.0,
        minimum_rows=3,
        liquidity=liquidity.tolist(),
        liquidity_weight=4.0,
    )
    l1_med = (median(quantiles) - target).abs()
    for kind, scale in ((POINT_MAE, 1.0), (POINT_COMPONENT, 0.1)):
        eng = trainer(shared, tmp_path / f"point-{kind}", loss=kind, session_ic_term=opts)
        expected = ((row_w * (l1_med * scale)).sum() / row_w.sum()).item() + ic_q
        assert eng._task_loss(packed_q, target, keys).item() == pytest.approx(expected, rel=1e-8)
