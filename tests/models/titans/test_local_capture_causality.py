"""Permutación y prefijos de la captura local, incluidas las ventanas causales."""

import pytest
import torch
from test_local_capture import DEVICES, local

from mars_titan.models.titans.config import MACConfig, MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.financial import MEMORY_PROJECTIONS, PAPER_PROJECTIONS
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.plasticity import advance, initial_history
from mars_titan.models.titans.state import mac_tensors, map_mac_rows


def configured(device):
    return TitansMAC(
        MACConfig(
            MemoryConfig(
                4,
                depth=2,
                multiscale=MemoryMultiscale(),
                plasticity=local(),
                **MEMORY_PROJECTIONS[PAPER_PROJECTIONS],
            ),
            heads=1,
            persistent_tokens=1,
            max_segment=2,
            attention_layout="sequence_contiguous_v1",
        ),
        device=device,
        dtype=torch.float64,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_local_statistics_follow_joint_internal_row_permutation(device):
    config = local(beta=0.7, trace_decay=0.6, epsilon=0.5)
    generator = torch.Generator(device=device).manual_seed(842)
    first, gradient = (
        torch.randn(2, 4, 4, device=device, dtype=torch.float64, generator=generator)
        for _ in range(2)
    )
    steps = torch.zeros(2, dtype=torch.int64, device=device)
    history = advance(config, first, initial_history(first), steps).history
    expected = advance(config, gradient, history, steps + 1)
    order = torch.tensor([2, 0, 1, 3], device=device)
    inverse = order.argsort()
    actual = advance(
        config,
        gradient.index_select(1, order),
        tuple(value.index_select(1, order) for value in history),
        steps + 1,
    )
    for left, right in zip(
        (*expected.history, expected.write_gain, expected.capture_gain, expected.coherence),
        (*actual.history, actual.write_gain, actual.capture_gain, actual.coherence),
        strict=True,
    ):
        torch.testing.assert_close(left, right.index_select(1, inverse), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_local_capture_joint_flow_permutation_keeps_the_next_output_and_all_states(device):
    model = configured(device)
    tokens = torch.arange(48, device=device, dtype=torch.float64).reshape(3, 4, 4) / 30
    _, prefix = model(tokens[:, :2], model.initial_state(3))
    order = torch.tensor([2, 0, 1], device=device)
    inverse = order.argsort()
    permuted = map_mac_rows(prefix, lambda value: value.index_select(0, order))
    expected, next_state = model(tokens[:, 2:], prefix)
    actual, permuted_next = model(tokens[:, 2:].index_select(0, order), permuted)
    torch.testing.assert_close(actual.index_select(0, inverse), expected, rtol=0, atol=0)
    restored_order = map_mac_rows(permuted_next, lambda value: value.index_select(0, inverse))
    for left, right in zip(mac_tensors(next_state), mac_tensors(restored_order), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_local_capture_prefix_output_state_and_gradients_cannot_use_a_future_suffix(device):
    model = configured(device)
    tokens = (
        torch.arange(32, device=device, dtype=torch.float64).reshape(2, 4, 4) / 30
    ).requires_grad_(True)
    state = model.initial_state(2)
    output, prefix = model(tokens[:, :2], state, differentiable=True)
    loss = output.square().sum() + sum(
        value.square().sum() for value in mac_tensors(prefix) if value.is_floating_point()
    )
    (gradient,) = torch.autograd.grad(loss, tokens)
    assert torch.count_nonzero(gradient[:, 2:]) == 0
    assert torch.isfinite(gradient).all() and gradient[:, :2].abs().sum() > 0
    changed = tokens.detach().clone()
    changed[:, 2:] = 1000
    repeated, repeated_prefix = model(changed[:, :2], state)
    assert torch.equal(output, repeated)
    for left, right in zip(mac_tensors(prefix), mac_tensors(repeated_prefix), strict=True):
        assert torch.equal(left, right)
