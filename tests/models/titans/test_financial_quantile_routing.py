"""Integración y recuperación de las vías de aprendizaje de la cabeza."""

from dataclasses import asdict, replace

import pytest
import torch
from torch.nn import functional as F

from mars_titan.models.quantile_head import QUANTILE_HEAD, pinball_loss
from mars_titan.models.quantile_routing import QuantileRouting
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.training.search_cases import checked_predictor_changes
from mars_titan.training.titans_walk_forward import unfused_attention
from tests.models.titans.test_financial_adapter import raw_batch, specification

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def configuration():
    return FinancialConfig(specification(), hidden_size=32, layers=1, head=QUANTILE_HEAD)


def test_route_identity_and_json_contract_cannot_silently_change_scalar_heads():
    base = configuration()
    changes = dict(quantile_routing=asdict(QuantileRouting(False, False)))
    assert checked_predictor_changes(changes) == changes
    current = replace(base, **changes)
    assert current.quantile_routing == QuantileRouting(False, False)
    assert "quantile_routing" not in base.identity()
    assert current.identity()["quantile_routing"]["surrogate_gradient"] is True
    for bad in (False, {}, dict(median_from_tails=False)):
        with pytest.raises(ValueError):
            replace(base, quantile_routing=bad)
    with pytest.raises(ValueError):
        replace(current, head="scalar")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_financial_control_keeps_forward_backward_and_tensor_state(device, dtype):
    configs = [configuration(), replace(configuration(), quantile_routing=QuantileRouting())]
    with unfused_attention():
        models = [FinancialPredictor(c, device=device, dtype=dtype) for c in configs]
        batch = DecisionBatch.from_corpus(
            raw_batch(), configs[0].inputs, device=device, dtype=dtype
        )
        outputs = [
            model.prepare(
                batch, model.initial_state(batch.flow_ids, differentiable=True), differentiable=True
            )
            for model in models
        ]
        torch.testing.assert_close(outputs[0].quantiles, outputs[1].quantiles, rtol=0, atol=0)
        assert models[0]._tensor_bytes_per_flow() == models[1]._tensor_bytes_per_flow()
        for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        target = torch.tensor([0.1, -0.2], device=device, dtype=dtype)
        gradients = [
            torch.autograd.grad(
                pinball_loss(out.quantiles, target), tuple(model.parameters()), allow_unused=True
            )
            for model, out in zip(models, outputs, strict=True)
        ]
        for left, right in zip(*gradients, strict=True):
            if left is None:
                assert right is None
            else:
                torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_complete_isolation_sends_only_the_primary_term_to_the_financial_encoder(device):
    base = configuration()
    with unfused_attention():
        models = [
            FinancialPredictor(c, device=device, dtype=torch.float64)
            for c in (base, replace(base, quantile_routing=QuantileRouting(False, False)))
        ]
        batch = DecisionBatch.from_corpus(
            raw_batch(), base.inputs, device=device, dtype=torch.float64
        )
        outputs = [
            model.prepare(
                batch, model.initial_state(batch.flow_ids, differentiable=True), differentiable=True
            )
            for model in models
        ]
        target = torch.tensor([0.1, -0.2], device=device, dtype=torch.float64)
        losses = [
            F.l1_loss(outputs[0].point_predictions, target) / 10,
            pinball_loss(outputs[1].quantiles, target),
        ]
        grads = [
            torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
            for loss, model in zip(losses, models, strict=True)
        ]
        for (name, _), left, right in zip(models[0].named_parameters(), *grads, strict=True):
            if name.startswith("head."):
                continue
            if left is None:
                assert right is None
            else:
                torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_active_route_roundtrip_reproduces_the_next_prediction(device):
    config = replace(configuration(), quantile_routing=QuantileRouting(False, False))
    with unfused_attention():
        model = FinancialPredictor(config, device=device, dtype=torch.float64)
        assert model.head.routing == config.quantile_routing
        batch = DecisionBatch.from_corpus(
            raw_batch(), config.inputs, device=device, dtype=torch.float64
        )
        first = model.prepare(batch, model.initial_state(batch.flow_ids))
        restored = model.restore_state(model.export_state_cpu(first.next_state), device=device)
        following = DecisionBatch.from_corpus(
            raw_batch(at=1_609_459_200_000_010), config.inputs, device=device, dtype=torch.float64
        )
        outputs = [
            model.prepare(following, state).quantiles for state in (first.next_state, restored)
        ]
        torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("consumer", [False, True])
@pytest.mark.parametrize("replacement", [None, {}])
def test_replacing_the_live_route_is_rejected_before_emitting_or_verifying(consumer, replacement):
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    config = replace(configuration(), quantile_routing=QuantileRouting(False, False))
    with unfused_attention():
        model = FinancialPredictor(config, dtype=torch.float64).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        batch = DecisionBatch.from_corpus(raw_batch(), config.inputs, dtype=torch.float64)
        frozen = FrozenFinancialConsumer(model) if consumer else None
        model.head.routing = replacement
        with pytest.raises(ValueError):
            if consumer:
                frozen.verify(strong=False)
            else:
                model.prepare(batch, model.initial_state(batch.flow_ids))
