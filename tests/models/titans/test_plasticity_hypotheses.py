"""Contraejemplos a interpretar coherencia y captura como utilidad o ruido."""

import pytest
import torch

from mars_titan.models.titans.plasticity import advance, initial_history
from mars_titan.models.titans.plasticity_config import MemoryPlasticity

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("device", DEVICES)
def test_predictable_alternation_can_have_low_coherence(device):
    config = MemoryPlasticity(beta=0.5, epsilon=0.25, gradient_contract="scalar_controls_v2")
    gradient = torch.ones(1, 1, 1, dtype=torch.float64, device=device)
    history = initial_history(gradient)
    for index in range(40):
        result = advance(
            config,
            gradient * (-1) ** index,
            history,
            torch.tensor([index], device=device),
        )
        history = result.history
    # El ciclo determinista tiene m→±1/3 y v→1. Su coherencia tiende a (1/9)/(5/4).
    torch.testing.assert_close(
        result.coherence,
        torch.full_like(result.coherence, 4 / 45),
        rtol=1e-10,
        atol=0,
    )
    assert result.write_gain.item() < 0.2


@pytest.mark.parametrize("device", DEVICES)
def test_equal_local_gradient_has_less_capture_when_other_rows_are_inactive(device):
    config = MemoryPlasticity(
        beta=0,
        trace_decay=0,
        epsilon=1,
        capture="eligibility",
        gradient_contract="scalar_controls_v2",
    )
    gradient = torch.ones(2, 8, 8, dtype=torch.float64, device=device)
    gradient[1, 1:] = 0
    result = advance(
        config,
        gradient,
        initial_history(gradient),
        torch.zeros(2, dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(result.write_gain[0, 0], result.write_gain[1, 0], rtol=0, atol=0)
    # Mismo g local, pero la media de coherencia de ocho filas divide por ocho la señal.
    torch.testing.assert_close(
        result.capture_gain[:, 0, 0],
        torch.tensor([73 / 90, 17 / 90], device=device, dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )
    excess = result.capture_gain[:, 0, 0] - config.capture_floor
    torch.testing.assert_close(excess[1], excess[0] / 8, rtol=1e-14, atol=1e-14)
