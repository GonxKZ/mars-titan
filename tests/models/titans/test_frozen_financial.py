"""Composición congelada y contrato efectivo, sin ajustar parámetros."""

import importlib

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, reader, source
from test_financial_adapter import setup

from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot


@pytest.fixture(autouse=True)
def frozen_backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def consumer(model, readout=None):
    api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    return api.FrozenFinancialConsumer(model, readout=readout)


def frozen(variant="mac_online", *, refinements=None):
    model, batch = setup(variant)
    model.eval().requires_grad_(False)
    extension = None if refinements is None else reader(refinements=refinements)
    if extension is not None:
        extension.eval().requires_grad_(False)
    return consumer(model, extension), batch


def bank(batch):
    return EpisodeSnapshot.create(
        **source(16),
        cutoff=batch.prediction_at[0],
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=torch.float64,
    )


@pytest.mark.parametrize(
    "variant", ["transformer_direct", "mac_disabled", "mac_frozen", "mac_online"]
)
def test_none_composition_is_exact_and_warmup_only_returns_state(variant):
    engine, batch = frozen(variant)
    state = engine.predictor.initial_state(batch.flow_ids)
    expected = engine.predictor.prepare(batch, state)
    result = engine.prepare(batch, state, context_id=CONTEXT)
    torch.testing.assert_close(result.point_predictions, expected.point_predictions, rtol=0, atol=0)
    assert result.next_state.observed_steps.tolist() == [1, 1]
    warmup = engine.prepare(batch, state, context_id=CONTEXT, warmup=True)
    assert warmup.point_predictions is None and warmup.readout is None
    assert warmup.next_state.observed_steps.tolist() == [1, 1]


@pytest.mark.parametrize("k", [1, 2, 4])
def test_refinement_preserves_one_mac_update_and_uses_existing_head(k):
    engine, batch = frozen(refinements=k)
    state = engine.predictor.initial_state(batch.flow_ids)
    rng = torch.get_rng_state().clone()
    result = engine.prepare(batch, state, snapshot=bank(batch), context_id=CONTEXT)
    assert result.next_state.mac.memory.steps.tolist() == [1, 1]
    assert len(result.readout.reads) == k
    with torch.no_grad():
        expected = engine.predictor.head(result.readout.state).squeeze(-1)
    torch.testing.assert_close(result.point_predictions, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not result.point_predictions.requires_grad
    assert all(p.grad is None for p in engine.predictor.parameters())


@pytest.mark.parametrize("change", ["train", "submodule", "requires_grad", "hook", "override"])
def test_undeclared_computation_changes_are_rejected(change):
    engine, batch = frozen()
    state = engine.predictor.initial_state(batch.flow_ids)
    if change == "train":
        engine.predictor.train()
    elif change == "submodule":
        engine.predictor.price_encoder.train()
    elif change == "requires_grad":
        engine.predictor.head.weight.requires_grad_(True)
    elif change == "hook":
        engine.predictor.head.register_forward_hook(lambda m, x, y: y)
    else:
        engine.predictor.head.forward = lambda x: x[:, :1]
    with pytest.raises(ValueError):
        engine.prepare(batch, state, context_id=CONTEXT)


def test_unfrozen_parameters_are_rejected_before_execution():
    model, _ = setup()
    model.eval()
    with pytest.raises(ValueError):
        consumer(model)


def test_fused_transformer_backend_is_rejected_without_changing_the_global_flag():
    model, _ = setup()
    model.eval().requires_grad_(False)
    torch.backends.mha.set_fastpath_enabled(True)
    with pytest.raises(ValueError, match="fastpath"):
        consumer(model)
    assert torch.backends.mha.get_fastpath_enabled()


def test_backend_change_after_construction_cannot_keep_the_execution_identity():
    engine, batch = frozen()
    assert engine.identity()["numerics"]["mha_fastpath"] is False
    torch.backends.mha.set_fastpath_enabled(True)
    with pytest.raises(ValueError):
        engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), context_id=CONTEXT)
    assert torch.backends.mha.get_fastpath_enabled()


def test_numeric_flags_are_part_of_recovery_identity():
    engine, _ = frozen()
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium" if previous == "highest" else "highest")
        with pytest.raises(ValueError):
            engine.verify()
        other, _ = frozen()
        assert other.model_id != engine.model_id
    finally:
        torch.set_float32_matmul_precision(previous)


def test_strong_boundary_verification_detects_data_bypass():
    engine, _ = frozen()
    engine.predictor.head.weight.data.add_(1)
    with pytest.raises(ValueError):
        engine.verify()


def test_future_snapshot_is_rejected_at_the_actual_decision_cutoff():
    engine, batch = frozen(refinements=1)
    snapshot = EpisodeSnapshot.create(
        **source(),
        cutoff=batch.prediction_at[0] + 1,
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError):
        engine.prepare(
            batch,
            engine.predictor.initial_state(batch.flow_ids),
            snapshot=snapshot,
            context_id=CONTEXT,
        )
