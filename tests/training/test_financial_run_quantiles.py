"""Entrenador cronológico de Titans con `quantile_head_v1`, sin modificar pesos.

Se reutilizan el corpus técnico y el optimizador que solo registra gradientes de
`test_financial_run.py`. La pérdida del tramo debe ser la pinball de los cinco
niveles y el error de selección debe usar la mediana emitida.
"""

import pytest
import torch

from mars_titan.models.quantile_head import (
    MEDIAN_INDEX,
    PINBALL,
    QUANTILE_HEAD,
    pinball_loss,
)
from mars_titan.models.titans.financial import VARIANTS, FinancialConfig, FinancialPredictor
from mars_titan.training import financial_run
from mars_titan.training.financial_run import ChronologicalRecipe, load_recipe
from tests.training.test_financial_run import (
    entries,
    explicit_fastpath,  # noqa: F401
    named_records,
    shared,  # noqa: F401
    trainer,
)


def quantile_predictor(streams, variant="mac_online", seed=42):
    specification = streams["train"].specification()
    config = FinancialConfig(
        specification, variant=variant, hidden_size=32, seed=seed, head=QUANTILE_HEAD
    )
    return FinancialPredictor(config, dtype=torch.float64)


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_segments_use_pinball_on_five_levels_and_errors_use_the_median(
    shared,  # noqa: F811
    tmp_path,
    monkeypatch,
    variant,
):
    _, streams = shared
    calls = []

    def spy(prediction, target, **options):
        calls.append((prediction.shape, prediction.requires_grad))
        return pinball_loss(prediction, target, **options)

    monkeypatch.setattr(financial_run, "pinball_loss", spy)
    model = quantile_predictor(streams, variant)
    engine = trainer(streams, tmp_path / "run", model=model, loss=PINBALL)
    initial = model._parameter_id
    report = engine.run()
    assert report["status"] == "completed"
    train = report["history"][1]["train"]
    assert len(calls) == train["updates"] == len(engine.optimizer.records) > 3
    assert all(len(shape) == 2 and shape[1] == 5 and grad for shape, grad in calls)
    assert sum(shape[0] for shape, _ in calls) == train["labels_in_loss"]
    assert model._parameter_id == initial
    for record in named_records(engine):
        assert record["head.weight"][[0, 1, 3, 4]].abs().sum() > 0
    assert engine.identity["predictor"]["output"] == QUANTILE_HEAD
    assert engine.identity["recipe"]["loss"] == PINBALL
    assert "mars_titan.models.quantile_head" in engine.identity["implementation"]


def test_validation_issues_the_median_of_the_frozen_quantiles(shared, tmp_path):  # noqa: F811
    from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    _, streams = shared
    model = quantile_predictor(streams)
    engine = trainer(streams, tmp_path / "run", model=model, loss=PINBALL, block_rows=3)
    engine.evaluate(streams["validation"])
    issued = {(e[2], e[3]): e[4] for e in entries(engine.audit, "prediction", "validation")}
    model.requires_grad_(False)
    consumer = FrozenFinancialConsumer(model)
    stream, states, expected = streams["validation"], {}, {}
    for event in stream.events():
        for raw in event.inputs:
            batch = DecisionBatch.from_validated(
                validated_cpu_batch(raw, model.config.inputs), dtype=torch.float64
            )
            flow = batch.flow_ids[0]
            state = states.get(flow) or model.initial_state((flow,))
            warmup = event.at < stream.phase.decision_start
            result = consumer.prepare(batch, state, context_id="0" * 64, warmup=warmup)
            states[flow] = result.next_state
            if not warmup:
                assert torch.equal(result.point_predictions, result.quantiles[:, MEDIAN_INDEX])
                expected[flow, event.at] = float(result.point_predictions[0])
    assert issued.keys() == expected.keys() and issued
    for key, value in expected.items():
        assert issued[key] == pytest.approx(value, rel=1e-12, abs=1e-14)


def test_head_and_loss_must_be_declared_together(shared, tmp_path):  # noqa: F811
    _, streams = shared
    with pytest.raises(ValueError, match="pinball"):
        trainer(streams, tmp_path / "scalar", loss=PINBALL)
    for loss in ("mae", "mse", "huber"):
        with pytest.raises(ValueError, match="pinball"):
            trainer(streams, tmp_path / loss, model=quantile_predictor(streams), loss=loss)
    assert not any(tmp_path.iterdir())


def test_scalar_and_quantile_runs_have_distinct_identities(shared, tmp_path):  # noqa: F811
    _, streams = shared
    scalar = trainer(streams, tmp_path / "scalar")
    quantile = trainer(
        streams, tmp_path / "quantile", model=quantile_predictor(streams), loss=PINBALL
    )
    assert scalar.run_id != quantile.run_id
    assert scalar.identity["predictor"]["output"] == "scalar_corpus_target"
    assert "output_head" not in scalar.identity["predictor"]
    assert scalar.identity["recipe"]["loss"] == "mae"


@pytest.mark.parametrize("huber_delta", [0.0, -1.0, float("nan")])
def test_pinball_recipe_keeps_validating_its_shared_fields(huber_delta):
    with pytest.raises(ValueError, match="Huber"):
        ChronologicalRecipe(loss=PINBALL, huber_delta=huber_delta)


def test_declared_quantile_recipe_builds_the_four_paired_controls(shared):  # noqa: F811
    recipe, document = load_recipe("configs/titans/chronological-training-quantile.json")
    scalar, reference = load_recipe("configs/titans/chronological-training.json")
    assert recipe.loss == PINBALL and document["predictor"]["head"] == QUANTILE_HEAD
    assert document["status"] == "propuesta_sin_ejecutar"
    differing = {
        key for key in recipe.identity() if recipe.identity()[key] != scalar.identity()[key]
    }
    assert differing == {"loss"}
    assert {k: v for k, v in document["predictor"].items() if k != "head"} == reference["predictor"]
    _, streams = shared
    specification = streams["train"].specification()
    options = {k: v for k, v in document["predictor"].items() if k != "dtype"}
    assert tuple(document["variants"]) == VARIANTS
    for variant in document["variants"]:
        config = FinancialConfig(specification, variant=variant, **options)
        assert config.identity()["output"] == QUANTILE_HEAD
