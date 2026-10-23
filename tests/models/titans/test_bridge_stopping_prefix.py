"""El presupuesto J2 devuelve su último prefijo aunque no alcance el umbral."""

import importlib

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ("adaptive_reference_full", "adaptive_compact"))
def test_maximum_j2_is_returned_without_claiming_the_criterion_was_met(backend):
    module = importlib.import_module("mars_titan.models.titans.bridge_recurrence")
    configuration = getattr(module, "BridgeStoppingConfig", None)
    execute = getattr(module, "adaptive_mix", None)
    assert callable(configuration) and callable(execute), "Falta la parada regional opcional"
    local = torch.tensor([[[1.0]]], device="cuda:0", dtype=torch.float64, requires_grad=True)
    result = execute(
        local,
        torch.ones_like(local),
        torch.zeros((1, 1, 1), device="cuda:0", dtype=torch.float64),
        torch.ones((1, 1), device="cuda:0", dtype=torch.bool),
        steps=2,
        gain=0.5,
        stopping=configuration(atol=0.0, rtol=0.0, backend=backend),
    )
    assert result.selected_steps.tolist() == result.evaluated_steps.tolist() == [2]
    assert result.criterion_met.tolist() == [False]
    assert torch.equal(result.values, torch.full_like(local, 0.25))
    (gradient,) = torch.autograd.grad(result.values.sum(), local)
    assert torch.equal(gradient, torch.full_like(local, 0.25))
