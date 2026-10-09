"""Paridad de `QuantileHead` con `Candidate::quantiles` del candidato GRU nativo.

Se copian los pesos `head_weight` y `head_bias` del candidato a la cabeza común y
se comparan salidas y gradientes en FP32 y FP64 sobre estados de trabajo de
anchura 128. Requiere el enlace `_episodic_native` compilado con el candidato.
No ajusta ni modifica parámetros del candidato.
"""

import os

import pytest
import torch

from mars_titan.memory.native_backend import load_native
from mars_titan.models.quantile_head import QuantileHead, pinball_loss

pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)


def native_candidate(dtype, seed):
    native = load_native()
    if getattr(native, "candidate_abi_version", None) != 1:
        pytest.skip("El enlace se compiló sin el candidato GRU")
    config = native.CandidateConfig()
    config.normalization_id = "quantile-head-parity-v1"
    config.parameter_seed = seed
    return native.Candidate(config, "float32" if dtype == torch.float32 else "float64", "cpu")


def paired(dtype, seed=42):
    model = native_candidate(dtype, seed)
    parameters = model.named_parameters()
    head = QuantileHead(parameters["head_weight"].shape[1], dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(parameters["head_weight"])
        head.bias.copy_(parameters["head_bias"])
    return model, head


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [42, 7])
def test_outputs_and_gradients_match_the_native_candidate(dtype, seed):
    model, head = paired(dtype, seed)
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(17, 128, generator=generator, dtype=dtype) * 2
    target = torch.randn(17, generator=generator, dtype=dtype)
    native_state = state.clone().requires_grad_()
    common_state = state.clone().requires_grad_()
    expected = model.quantiles(native_state)
    actual = head(common_state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual.diff(dim=1) >= 0).all()
    parameters = model.named_parameters()
    native_gradients = torch.autograd.grad(
        pinball_loss(expected, target),
        (native_state, parameters["head_weight"], parameters["head_bias"]),
    )
    common_gradients = torch.autograd.grad(
        pinball_loss(actual, target), (common_state, head.weight, head.bias)
    )
    for left, right in zip(common_gradients, native_gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(value.grad is None for value in parameters.values())


def test_full_native_forward_exposes_the_same_head_on_its_refined_state():
    model, head = paired(torch.float64)
    native = load_native()
    inputs = native.CandidateInputs(
        torch.randn(2, 64, 5, dtype=torch.float64) * 0.01,
        torch.zeros(2, 384, dtype=torch.float64),
        torch.zeros(2, 512, dtype=torch.float64),
        torch.zeros(2, 45, dtype=torch.float64),
        torch.zeros(2, 420, dtype=torch.float64),
        torch.ones(2, 5, dtype=torch.bool),
    )
    with torch.no_grad():
        prediction = model.forward(inputs, model.empty_memory(), 2)
        torch.testing.assert_close(head(prediction.state), prediction.quantiles, rtol=0, atol=0)
