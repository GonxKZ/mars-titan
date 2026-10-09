"""Pinball de padres de cuantiles en cuda:0 frente a CPU, sin optimizador. Se omite sin GPU."""

import pytest
import torch

from mars_titan.models.quantile_head import pinball_loss
from tests.posttraining.test_quantile_adaptation import (
    batch,
    matrix_cases,
    model_for,
    quantile_parent,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere cuda:0")
CASES = [
    ("gru", "head+fusion"),
    ("transformer", "head+readout+fusion"),
    ("lstm", "full_continuation"),
]


def pinball_pass(kind, arm, device):
    """Niveles, pérdida y gradientes de un caso, con el padre intacto y sin ningún paso."""
    cases, _ = matrix_cases(kind)
    case = cases[f"seed-42/{arm}"]["case"]
    parent = quantile_parent(kind).to(device)
    before = {k: v.clone() for k, v in parent.state_dict().items() if torch.is_tensor(v)}
    model = model_for(parent, case).train()
    inputs, presence, target = batch()
    levels = model({k: v.to(device) for k, v in inputs.items()}, presence.to(device)).double()
    target = target.to(device)
    loss = pinball_loss(levels, target)
    loss.backward()
    assert (torch.diff(levels.detach(), dim=1) >= 0).all()
    gradients = {}
    for name, value in model.named_parameters():
        assert (value.grad is not None) == value.requires_grad, name
        if value.requires_grad:
            assert torch.isfinite(value.grad).all(), name
            gradients[name] = value.grad.detach().cpu()
    after = parent.state_dict()
    assert all(torch.equal(value, after[key]) for key, value in before.items())
    # La pinball cambia de pendiente en q = y. Lejos de ese punto, los gradientes son comparables.
    margin = float((target[:, None] - levels.detach()).abs().min())
    return levels.detach().cpu(), loss.detach().cpu(), gradients, margin


@pytest.mark.parametrize(("kind", "arm"), CASES)
def test_cuda_pinball_matches_cpu_and_keeps_the_quantile_order(kind, arm, monkeypatch):
    # TF32 cambiaría la referencia float32 frente a CPU.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    levels, loss, gradients, margin = pinball_pass(kind, arm, torch.device("cpu"))
    assert margin > 1e-4
    actual = pinball_pass(kind, arm, torch.device("cuda:0"))
    torch.testing.assert_close(actual[0], levels, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual[1], loss, rtol=1e-4, atol=1e-7)
    assert actual[2].keys() == gradients.keys()
    for name, value in gradients.items():
        torch.testing.assert_close(actual[2][name], value, rtol=1e-3, atol=1e-6, msg=name)
