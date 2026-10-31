"""Recorrido cronológico con las cabezas oponentes de A4 (#749), sin pasos de optimizador."""

import pytest
import torch

from mars_titan.models.titans import opponent_heads
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.opponent_heads import OpponentHeads
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    corpus,
    entries,
    named_records,
)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("opponent-heads"))[1]


def predictor(streams, heads):
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        opponent_heads=heads,
    )
    return FinancialPredictor(config, dtype=torch.float64)


def trainer(streams, output, heads, **options):
    values = dict(truncation=3, epochs=1, block_rows=2, max_grad_norm=None, selection=SELECTION)
    return ChronologicalTrainer(
        predictor(streams, heads),
        ChronologicalRecipe(**values | options),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


@pytest.mark.parametrize("level", [0.25, 0.5])
def test_each_update_uses_both_heads_and_the_trunk_keeps_the_l1_gradient(
    shared, tmp_path, level, monkeypatch
):
    calls, original = [], opponent_heads.opponent_loss

    def spy(prediction, target, config):
        calls.append((prediction.shape, config.level))
        return original(prediction, target, config)

    monkeypatch.setattr(opponent_heads, "opponent_loss", spy)
    heads = OpponentHeads(family="quantile", level=level)
    engine = trainer(shared, tmp_path / "opponent", heads)
    before = engine.predictor._parameter_id
    assert engine.run()["status"] == "completed"
    assert engine.predictor._parameter_id == before
    assert len(calls) == len(entries(engine.audit, "update")) == engine.optimizer.calls > 2
    assert all(len(shape) == 2 and shape[1] == 2 and seen == level for shape, seen in calls)
    control = trainer(shared, tmp_path / "control", None)
    control.run()
    records, reference = named_records(engine), named_records(control)
    assert len(records) == len(reference)
    for left, right in zip(records, reference, strict=True):
        weight, bias = left.pop("output_objective.weight"), left.pop("output_objective.bias")
        # Con τ = 0,5 la semiamplitud no recibe gradiente. Con τ < 0,5 empieza a abrirse.
        assert (torch.count_nonzero(weight) > 0) == (level < 0.5)
        assert (bias.abs() > 0).item() == (level < 0.5)
        assert left.keys() == right.keys()
        for name, value in left.items():
            if value is None or right[name] is None:
                assert value is right[name], name
            else:
                torch.testing.assert_close(value, right[name], rtol=1e-12, atol=1e-18)


def test_the_recipe_must_keep_mae_without_distillation(shared, tmp_path):
    with pytest.raises(ValueError, match="sustituye la pérdida mae"):
        trainer(shared, tmp_path / "huber", OpponentHeads(), loss="huber", huber_delta=0.01)


def test_posttraining_is_refused_until_it_is_declared(shared, tmp_path):
    with pytest.raises(ValueError, match="A4 no tienen todavía un postentrenamiento"):
        ChronologicalTrainer(
            predictor(shared, OpponentHeads()),
            ChronologicalRecipe(truncation=3, epochs=1, block_rows=2, selection=SELECTION),
            train=shared["train"],
            validation=shared["validation"],
            output=tmp_path / "adapters",
            optimizer_factory=RecordingOptimizer,
            posttraining=dict(kind="adapters"),
        )
