"""Límites numéricos de los rangos y del objetivo por sesión, sin pasos de optimizador."""

import pytest
import torch

from mars_titan.models.titans.session_ic_objective import (
    RELATIVE_FLOOR,
    SessionICObjective,
    SessionScale,
    session_ic_loss,
    standardized,
)
from mars_titan.models.titans.session_ranks import gaussian_ranks


@pytest.mark.parametrize("initial", [1e-12, 1e-3, 100.0, 1000.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_declared_scale_initializes_without_overflow(initial, dtype):
    head = SessionScale(SessionICObjective(initial_scale=initial), dtype=dtype)
    assert torch.nn.functional.softplus(head.scale).item() == pytest.approx(initial, rel=2e-6)


def test_constant_zero_scores_have_a_finite_head_gradient_norm_in_fp32():
    weight = torch.zeros(64, dtype=torch.float32, requires_grad=True)
    feature = torch.linspace(-1, 1, 8, dtype=torch.float32)
    scores = (feature[:, None] * weight[None, :]).sum(dim=1)
    outputs = torch.stack((scores, scores.detach()), dim=1)
    loss = session_ic_loss(outputs, feature / 100, [("US", 1)] * 8, SessionICObjective())
    loss.backward()
    assert torch.isfinite(weight.grad).all()
    assert torch.isfinite(torch.linalg.vector_norm(weight.grad))


@pytest.mark.parametrize("dtype, amplitude", [(torch.float32, 1e30), (torch.float64, 1e200)])
def test_large_finite_scores_preserve_their_order_without_squaring_overflow(dtype, amplitude):
    unit = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=dtype)
    score = (unit * amplitude).requires_grad_()
    actual = standardized(score)
    expected = unit / torch.sqrt(unit.square().mean() * (1 + RELATIVE_FLOOR))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=0)
    (actual * unit.flip(0)).sum().backward()
    assert torch.isfinite(score.grad).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_observed_nonfinite_rank_values_are_rejected(value):
    with pytest.raises(ValueError, match="finito"):
        gaussian_ranks(torch.tensor([[value], [1.0]]), torch.ones(2, 1, dtype=torch.bool))


def test_absent_nonfinite_values_do_not_contaminate_present_ranks():
    result = gaussian_ranks(
        torch.tensor([[float("nan")], [1.0], [2.0]]),
        torch.tensor([[False], [True], [True]]),
    )
    assert torch.isfinite(result).all()
    assert result[0, 0] == 0 and result[1, 0] < 0 < result[2, 0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rank_scores_reject_precision_outside_the_declared_contract(dtype):
    with pytest.raises(ValueError, match="precisión"):
        gaussian_ranks(torch.ones(4096, 1, dtype=dtype), torch.ones(4096, 1, dtype=torch.bool))


def test_presence_and_values_must_share_a_device():
    with pytest.raises(ValueError, match="dispositivo"):
        gaussian_ranks(torch.ones(2, 1), torch.empty(2, 1, dtype=torch.bool, device="meta"))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_scores_are_rejected_before_normalization(value):
    with pytest.raises(ValueError, match="finitas"):
        standardized(torch.tensor([1.0, value]))


def test_a_scale_below_the_precision_range_is_rejected():
    with pytest.raises(ValueError, match="mínimo normal"):
        SessionScale(SessionICObjective(initial_scale=1e-100), dtype=torch.float32)
    with pytest.raises(ValueError, match="precisión"):
        SessionScale(SessionICObjective(), dtype=torch.float16)


@pytest.mark.parametrize("name", ["initial_scale", "scale_weight"])
def test_configuration_rejects_integers_outside_floating_point_range(name):
    with pytest.raises(ValueError, match="finito"):
        SessionICObjective(**{name: 10**400})


def test_empty_and_nonvector_normalization_are_rejected():
    for values in (torch.empty(0), torch.ones(2, 2), torch.ones(2, dtype=torch.float16)):
        with pytest.raises(ValueError, match="vector"):
            standardized(values)


def test_stable_scaling_matches_the_declared_formula_and_its_derivative():
    values = torch.tensor([-0.2, 0.1, 0.3, 0.7], dtype=torch.float64, requires_grad=True)
    centered = values - values.mean()
    direct = centered / torch.sqrt(
        centered.square().mean()
        + RELATIVE_FLOOR * values.square().mean()
        + torch.finfo(values.dtype).eps ** 2
    )
    actual = standardized(values)
    torch.testing.assert_close(actual, direct, rtol=1e-14, atol=1e-15)
    weights = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64)
    expected = torch.autograd.grad((weights * direct).sum(), values, retain_graph=True)[0]
    gradient = torch.autograd.grad((weights * actual).sum(), values)[0]
    torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-14)
