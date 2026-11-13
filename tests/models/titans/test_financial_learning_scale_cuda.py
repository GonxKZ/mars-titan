"""Paridad FP32, gradientes y recuperación de las escalas en CUDA, sin pasos de optimizador."""

import io

import pytest
import torch
from test_financial_adapter import VARIANTS, raw_batch, specification
from test_financial_learning_scale import DECLARED, statistics

from mars_titan.models.quantile_head import QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.financial import (
    REFERENCE_ENCODERS,
    DecisionBatch,
    FinancialConfig,
    FinancialPredictor,
)

pytestmark = pytest.mark.cuda
CONFIGURATIONS = [(variant, None) for variant in VARIANTS] + [
    ("transformer_direct", kind) for kind in REFERENCE_ENCODERS
]


@pytest.fixture(autouse=True)
def strict_precision():
    fastpath = torch.backends.mha.get_fastpath_enabled()
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.mha.set_fastpath_enabled(fastpath)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def build(config, device, *, fitted=True):
    model = FinancialPredictor(config, device=device, dtype=torch.float32)
    if fitted and config.learning_scale is not None:
        model.set_learning_statistics(statistics(config.learning_scale))
    return model


def batch(config, device, at):
    return DecisionBatch.from_corpus(
        raw_batch(at=at), config.inputs, device=device, dtype=torch.float32
    )


@pytest.mark.parametrize("variant,encoder", CONFIGURATIONS)
@pytest.mark.parametrize("scaled", [False, True])
def test_cuda_matches_cpu_emission_and_gradients_with_optional_scales(variant, encoder, scaled):
    config = FinancialConfig(
        specification(),
        variant=variant,
        reference_encoder=encoder,
        hidden_size=32,
        layers=1,
        seed=42,
        head=QUANTILE_HEAD,
        learning_scale=dict(DECLARED, zero_median=False) if scaled else None,
    )
    outputs, gradients = [], []
    for device in ("cpu", "cuda:0"):
        model = build(config, device)
        inputs = batch(config, device, 1_609_459_200_000_000)
        prepared = model.prepare(
            inputs,
            model.initial_state(inputs.flow_ids, differentiable=True),
            differentiable=True,
        )
        outputs.append(prepared.quantiles.detach().cpu())
        target = torch.tensor([0.01, -0.02], device=device)
        loss = pinball_loss(prepared.quantiles, target)
        gradients.append(
            [
                None if value is None else value.detach().cpu()
                for value in torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
            ]
        )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
    assert any(value is not None and value.abs().sum() > 0 for value in gradients[1])
    for reference, observed in zip(*gradients, strict=True):
        if reference is None:
            assert observed is None
        else:
            assert observed is not None and torch.isfinite(observed).all()
            torch.testing.assert_close(reference, observed, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("variant,encoder", CONFIGURATIONS)
@pytest.mark.parametrize("head", ["scalar", QUANTILE_HEAD])
def test_cuda_recovers_scales_and_temporal_state_without_changing_the_next_prediction(
    variant, encoder, head
):
    config = FinancialConfig(
        specification(),
        variant=variant,
        reference_encoder=encoder,
        hidden_size=32,
        layers=1,
        seed=42,
        head=head,
        learning_scale=DECLARED,
    )
    model = build(config, "cuda:0")
    first = batch(config, "cuda:0", 1_609_459_200_000_000)
    prepared = model.prepare(first, model.initial_state(first.flow_ids))
    assert torch.equal(prepared.point_predictions, torch.zeros(2, device="cuda:0"))
    saved = io.BytesIO()
    torch.save(dict(model=model.state_dict(), state=model.export_state(prepared.next_state)), saved)
    saved.seek(0)
    # El transporte conserva los contadores en CPU. El predictor mueve sus pesos rápidos.
    payload = torch.load(saved, map_location="cpu", weights_only=False)
    twin = build(config, "cuda:0", fitted=False)
    twin.load_state_dict(payload["model"])
    restored = twin.restore_state(payload["state"], device="cuda:0")
    later = batch(config, "cuda:0", 1_609_545_600_000_000)
    expected, actual = model.prepare(later, prepared.next_state), twin.prepare(later, restored)
    assert torch.equal(expected.detached_tokens, actual.detached_tokens)
    assert torch.equal(expected.point_predictions, actual.point_predictions)
    if head == QUANTILE_HEAD:
        assert torch.equal(expected.quantiles, actual.quantiles)
