"""Integración del backend opcional con el predictor completo, sin ajuste."""

from dataclasses import replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
KINDS = [
    ("kronlstm_v1", "kronlstm_precomputed_v1", "scalar"),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("legacy,kind,rule", KINDS)
def test_precomputed_predictor_preserves_full_initialization_output_and_gradients(
    device, dtype, legacy, kind, rule
):
    original, _ = setup("transformer_direct")
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    models = [
        api()
        .FinancialPredictor(
            replace(original.config, temporal_encoder=name), device=device, dtype=dtype
        )
        .eval()
        for name in (legacy, kind)
    ]
    left, right = (dict(model.named_parameters()) for model in models)
    assert left.keys() == right.keys() and sum(value.numel() for value in left.values()) > 1000
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    assert models[1].config.identity()["temporal_encoder"] == kind
    assert models[1].price_encoder.configuration["backend"] == "pytorch_precomputed_cpu_qkv64_v1"
    assert models[1].price_encoder.configuration["rule"] == rule
    outputs, gradients = [], []
    for model in models:
        batch = api().DecisionBatch.from_corpus(
            raw_batch(), model.config.inputs, device=device, dtype=dtype
        )
        state = model.initial_state(batch.flow_ids)
        result = model.prepare(batch, state, differentiable=True)
        assert state.observed_steps.tolist() == [0, 0]
        assert result.next_state.observed_steps.tolist() == [1, 1]
        outputs.append(result.point_predictions)
        gradients.append(
            torch.autograd.grad(result.point_predictions.square().sum(), tuple(model.parameters()))
        )
    forward = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-12)
    backward = (
        dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-10)
    )
    torch.testing.assert_close(outputs[0], outputs[1], **forward)
    for a, b in zip(*gradients, strict=True):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, **backward)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)


@pytest.mark.parametrize("kind", [pair[1] for pair in KINDS])
def test_precomputed_predictor_is_usable_by_the_frozen_consumer(kind):
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    original, _ = setup("transformer_direct")
    model = api().FinancialPredictor(
        replace(original.config, temporal_encoder=kind), dtype=torch.float64
    )
    model.eval().requires_grad_(False)
    batch = api().DecisionBatch.from_corpus(raw_batch(), model.config.inputs, dtype=torch.float64)
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        frozen = FrozenFinancialConsumer(model)
        state = model.initial_state(batch.flow_ids)
        expected = model.prepare(batch, state)
        actual = frozen.prepare(batch, state, context_id="c" * 64)
        torch.testing.assert_close(
            actual.point_predictions, expected.point_predictions, rtol=0, atol=0
        )
        frozen.verify(strong=True)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
