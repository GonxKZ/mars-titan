"""Variante regional sobre las mismas entradas, estados y cabeza financieros."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.training.search_cases import checked_predictor_changes
from mars_titan.training.titans_walk_forward import _predictor

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def test_regional_configuration_is_explicit_and_travels_through_the_recipe():
    base, _ = setup()
    change = dict(regional_fusion=asdict(RegionalFusionConfig(topology="ring")))
    assert checked_predictor_changes(change) == change
    config = replace(base.config, **change)
    assert type(config.regional_fusion) is RegionalFusionConfig
    assert "regional_fusion" not in base.config.identity()
    assert config.identity() != base.config.identity()
    with pytest.raises(ValueError):
        replace(config, regional_fusion=dict(topology="ring"))
    with pytest.raises(ValueError):
        replace(config, presence_modulation="film_v1")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graph_changes_only_working_computation_and_advances_mac_once(device, dtype):
    base, _ = setup()
    original_rng = torch.get_rng_state().clone()
    config = replace(base.config, regional_fusion=RegionalFusionConfig(topology="ring"))
    model = api().FinancialPredictor(config, device=device, dtype=dtype).eval()
    assert torch.equal(original_rng, torch.get_rng_state())
    base = api().FinancialPredictor(base.config, device=device, dtype=dtype).eval()
    for name, value in base.named_parameters():
        assert torch.equal(value, dict(model.named_parameters())[name])
    assert model._tensor_bytes_per_flow() == base._tensor_bytes_per_flow()
    state = None
    for step in range(2):
        batch = api().DecisionBatch.from_corpus(
            raw_batch(at=1_609_459_200_000_000 + 10 * step),
            config.inputs,
            device=device,
            dtype=dtype,
        )
        if state is None:
            state = model.initial_state(batch.flow_ids, differentiable=True)
        result = model.prepare(batch, state, differentiable=True)
        state = result.next_state
        assert state.mac.memory.steps.tolist() == [step + 1, step + 1]
    result.point_predictions.square().sum().backward()
    regional_gradients = [
        value.grad
        for name, value in model.named_parameters()
        if name.startswith("regional_fusion.")
    ]
    assert all(value is not None and torch.isfinite(value).all() for value in regional_gradients)
    assert any(torch.count_nonzero(value) for value in regional_gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_regional_predictor_recovers_the_next_prediction_without_hidden_state(device):
    base, _ = setup()
    config = replace(base.config, regional_fusion=RegionalFusionConfig())
    model = (
        api()
        .FinancialPredictor(config, device=device, dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    engine = FrozenFinancialConsumer(model)
    batch = api().DecisionBatch.from_corpus(
        raw_batch(), config.inputs, device=device, dtype=torch.float64
    )
    first = engine.prepare(batch, model.initial_state(batch.flow_ids), context_id="a" * 64)
    restored = model.restore_state(model.export_state_cpu(first.next_state), device=device)
    next_batch = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_459_200_000_010), config.inputs, device=device, dtype=torch.float64
    )
    expected = engine.prepare(next_batch, first.next_state, context_id="b" * 64)
    model.prepare(batch, model.initial_state(batch.flow_ids))
    actual = engine.prepare(next_batch, restored, context_id="b" * 64)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    engine.verify()


def test_frozen_audit_rejects_a_regional_schedule_changed_after_sealing():
    base, _ = setup()
    config = replace(base.config, regional_fusion=RegionalFusionConfig())
    model = api().FinancialPredictor(config, dtype=torch.float64).eval().requires_grad_(False)
    engine = FrozenFinancialConsumer(model)
    model.regional_fusion.config = replace(model.regional_fusion.config, cycles=1)
    with pytest.raises(ValueError, match="identidad"):
        engine.verify()


@pytest.mark.parametrize(
    "variant", ["transformer_direct", "mac_disabled", "mac_frozen", "mac_online"]
)
@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_region_graph_is_usable_with_all_four_backbone_controls_and_input_contracts(
    variant, historical, device
):
    base, _ = setup(variant, historical=historical)
    config = replace(base.config, regional_fusion=RegionalFusionConfig())
    model = api().FinancialPredictor(config, device=device, dtype=torch.float64).eval()
    values = raw_batch(absent=historical)
    if not historical:
        values.pop("presence")
    batch = api().DecisionBatch.from_corpus(
        values, config.inputs, device=device, dtype=torch.float64
    )
    output = model.prepare(batch, model.initial_state(batch.flow_ids))
    assert torch.isfinite(output.point_predictions).all()
    assert output.next_state.observed_steps.tolist() == [1, 1]
    if model.mac is not None:
        assert output.next_state.mac.memory.steps.tolist() == (
            [1, 1] if variant == "mac_online" else [0, 0]
        )


@pytest.mark.parametrize("device", DEVICES)
def test_declared_arms_build_and_predict_through_the_existing_training_factory(device):
    arms = json.loads(Path("configs/experiments/regional-fusion-v1.json").read_text())
    recipe = json.loads(Path(arms["base_recipe"]).read_text())
    base, _ = setup()
    counts = []
    for arm in arms["arms"].values():
        model, pairing = _predictor(
            recipe,
            base.config.inputs,
            "mac_online",
            42,
            device,
            changes=arm["predictor_changes"] or None,
        )
        assert pairing is None
        batch = api().DecisionBatch.from_corpus(
            raw_batch(), model.config.inputs, device=device, dtype=torch.float32
        )
        result = model.prepare(batch, model.initial_state(batch.flow_ids))
        assert torch.isfinite(result.quantiles).all() and result.quantiles.shape == (2, 5)
        counts.append(sum(value.numel() for value in model.parameters()))
    assert counts[0] < counts[1] and len(set(counts[1:])) == 1
