"""Recorrido cronológico con el término de IC por sesión en la pérdida, sin pasos."""

from collections import Counter

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as observations
from mars_titan.models.baselines.multimodal import SCALAR_HEAD
from mars_titan.models.quantile_head import (
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
from tests.models.titans.test_session_ic_term import reference, weighted_reference
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

    def term(self, point, target, sessions, liquidity=None):
        value = self.inner.term(point, target, sessions, liquidity)
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
