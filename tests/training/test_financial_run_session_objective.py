"""Recorrido cronológico con rangos por sesión y pérdida de IC (F4 de #749), sin pasos."""

from collections import Counter

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as observations
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.session_ic_objective import SessionICObjective
from mars_titan.models.titans.session_ranks import SESSION_RANK_INPUTS
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.chronological_fixture import chronological_corpus, decision, phases
from tests.training.test_financial_run import SELECTION, RecordingOptimizer, entries

ASSETS = 5
OBJECTIVE = SessionICObjective(minimum_rows=3)


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
    return streams(tmp_path_factory.mktemp("session-objective"))


def predictor(sources, **changes):
    # La memoria de la receta de campaña conserva la escala de la salida de MAC. Con la memoria
    # v1 del fixture la salida se iguala entre activos y la puntuación pierde su dispersión.
    options = dict(
        session_rank_inputs=SESSION_RANK_INPUTS,
        session_ic_objective=OBJECTIVE,
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
    )
    config = FinancialConfig(
        sources["train"].specification(), hidden_size=32, variant="mac_online", **options | changes
    )
    return FinancialPredictor(config, dtype=torch.float64)


def trainer(sources, output, *, model=None, **options):
    recipe = dict(truncation=3, epochs=1, block_rows=8, selection=SELECTION) | options
    return ChronologicalTrainer(
        model or predictor(sources),
        ChronologicalRecipe(**recipe),
        train=sources["train"],
        validation=sources["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


def records(engine):
    names = {id(p): name for name, p in engine.predictor.named_parameters()}
    return [{names[k]: v for k, v in record.items()} for record in engine.optimizer.records]


def test_each_update_scores_whole_sessions_and_reaches_trunk_and_scale(shared, tmp_path):
    engine = trainer(shared, tmp_path / "run")
    calls = []
    objective = engine.predictor.output_objective
    original = objective.loss

    def spy(outputs, target, sessions):
        calls.append((outputs.shape, list(sessions)))
        return original(outputs, target, sessions)

    objective.loss = spy
    before = engine.predictor._parameter_id
    report = engine.run()
    assert report["status"] == "completed"
    train = report["history"][1]["train"]
    assert train["labels_without_graph"] == 0
    updates = entries(engine.audit, "update")
    assert len(calls) == len(updates) == train["updates"] > 2
    labels = Counter(
        (flow.split("/")[0], at) for _, _, flow, at, _ in entries(engine.audit, "label", "train")
    )
    seen = Counter(key for _, sessions in calls for key in sessions)
    # Cada sesión entra entera en una sola actualización, con todas sus etiquetas maduras.
    assert seen == labels and max(labels.values()) == ASSETS
    assert all(shape[1] == 2 for shape, _ in calls)
    gradients = records(engine)
    assert all(record["output_objective.scale"].abs() > 0 for record in gradients)
    assert all(record["head.weight"].abs().sum() > 0 for record in gradients)
    assert all(record["fusion.0.weight"].abs().sum() > 0 for record in gradients)
    assert engine.predictor._parameter_id == before


def test_an_instant_split_into_blocks_is_refused(shared, tmp_path):
    engine = trainer(shared, tmp_path / "split", block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        engine.run()


@pytest.mark.parametrize("options", [dict(loss="mse"), dict(distillation_weight=0.5)])
def test_the_objective_replaces_only_the_plain_absolute_error(shared, tmp_path, options):
    with pytest.raises(ValueError, match="objetivo de salida"):
        trainer(shared, tmp_path / "loss", **options)


@pytest.mark.parametrize("truncation", [3, 1])
def test_replay_by_flow_blocks_reproduces_the_direct_gradients(shared, tmp_path, truncation):
    # Con un instante por tramo, la fila de A0000 sin etiqueta del 28 de noviembre no entra en
    # la pérdida y la repetición debe recalcularla igualmente para conservar la sesión.
    direct = trainer(shared, tmp_path / "direct", truncation=truncation)
    direct.run()
    replayed = trainer(shared, tmp_path / "replayed", truncation=truncation, accumulation_rows=8)
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
    split = trainer(shared, tmp_path / "split", accumulation_rows=2)
    with pytest.raises(ValueError, match="repetir todos los flujos"):
        split.run()


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


def test_validation_emits_the_scaled_standardized_score(shared, tmp_path):
    engine = trainer(shared, tmp_path / "rows")
    rows = []
    metrics = engine.evaluate(shared["validation"], rows=rows)
    assert metrics["samples"] == len(rows) > 0
    by_session = {}
    for _, at, issued, _, levels in rows:
        assert levels is None
        by_session.setdefault(at, []).append(issued)
    scale = torch.nn.functional.softplus(engine.predictor.output_objective.scale).item()
    for values in by_session.values():
        if len(values) == ASSETS:
            assert abs(sum(values)) < 1e-12
            assert max(abs(value) for value in values) <= scale * ASSETS**0.5
