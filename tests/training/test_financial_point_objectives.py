"""Recorrido cronológico de controles puntuales, sin actualizar pesos."""

import pytest
import torch

from mars_titan.training.financial_run import ChronologicalRecipe
from tests.training.test_financial_run import (
    StopAtStep,
    explicit_fastpath,  # noqa: F401
    named_records,
    shared,  # noqa: F401
    trainer,
)
from tests.training.test_financial_run_quantiles import quantile_predictor

KINDS = ("quantile_point_mae_v1", "quantile_point_component_v1")
DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("device", DEVICES)
def test_point_controls_preserve_all_weights_and_leave_tail_rows_without_gradients(
    shared,  # noqa: F811
    tmp_path,
    kind,
    device,
):
    _, streams = shared
    model = quantile_predictor(streams).to(device)
    initial = model._parameter_id
    engine = trainer(streams, tmp_path / "run", model=model, loss=kind)
    report = engine.run()
    assert report["status"] == "completed"
    assert model._parameter_id == initial
    records = named_records(engine)
    assert len(records) > 3
    for record in records:
        assert torch.count_nonzero(record["head.weight"][[0, 1, 3, 4]]) == 0
        assert torch.count_nonzero(record["head.bias"][[0, 1, 3, 4]]) == 0
    assert any(torch.count_nonzero(record["head.weight"][2]) for record in records)
    assert engine.identity["recipe"]["loss"] == kind
    assert "mars_titan.training.quantile_objectives" in engine.identity["implementation"]


@pytest.mark.parametrize("kind", KINDS)
def test_point_objectives_cannot_be_used_with_a_scalar_head(shared, tmp_path, kind):  # noqa: F811
    with pytest.raises(ValueError):
        trainer(shared[1], tmp_path / "wrong", loss=kind)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("huber_delta", [0.0, -1.0, float("nan")])
def test_point_recipes_keep_validating_shared_hyperparameters(kind, huber_delta):
    with pytest.raises(ValueError, match="Huber"):
        ChronologicalRecipe(loss=kind, huber_delta=huber_delta)


@pytest.mark.parametrize("kind", KINDS)
def test_distillation_cannot_reintroduce_an_auxiliary_signal_in_a_point_control(kind):
    with pytest.raises(ValueError, match="destilación"):
        ChronologicalRecipe(loss=kind, distillation_weight=0.5)


def test_changing_the_objective_source_blocks_the_run_before_any_update(
    shared,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from mars_titan.training import quantile_objectives

    _, streams = shared
    engine = trainer(
        streams,
        tmp_path / "run",
        model=quantile_predictor(streams),
        loss="quantile_point_mae_v1",
    )
    changed = tmp_path / "changed.py"
    changed.write_text("# Otra revisión del objetivo de control.\n")
    monkeypatch.setattr(quantile_objectives, "__file__", str(changed))
    with pytest.raises(ValueError, match="código"):
        engine.run()
    assert engine.optimizer.calls == 0


@pytest.mark.parametrize("kind", KINDS)
def test_resuming_a_point_control_reproduces_the_next_gradients_and_selection(
    shared,  # noqa: F811
    tmp_path,
    kind,
):
    _, streams = shared
    options = dict(loss=kind, epochs=2, fixed=True, checkpoint_updates=2)

    def build(folder):
        return trainer(streams, folder, model=quantile_predictor(streams), **options)

    continuous = build(tmp_path / "continuous")
    expected = continuous.run()
    first = build(tmp_path / "resumed")
    stop = StopAtStep(15)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    second = build(tmp_path / "resumed")
    actual = second.run(resume=True)
    assert actual["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    reference = named_records(continuous)
    assert len(joined) == len(reference)
    for left, right in zip(joined, reference, strict=True):
        for name, value in left.items():
            if value is None:
                assert right[name] is None
            else:
                torch.testing.assert_close(value, right[name], rtol=0, atol=0)
    assert actual["history"] == expected["history"]
    assert actual["best_epoch"] == expected["best_epoch"]
