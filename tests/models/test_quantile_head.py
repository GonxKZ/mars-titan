"""Cabeza `quantile_head_v1`: parametrización, orden, pinball y gradientes analíticos.

La referencia de la parametrización se escribe columna a columna a partir de
`Candidate::quantiles` en `native/src/candidate.cpp`:

    mediana = raw[:, 2]
    q1 = mediana - softplus(raw[:, 1])
    q3 = mediana + softplus(raw[:, 3])
    q0 = q1 - softplus(raw[:, 0])
    q4 = q3 + softplus(raw[:, 4])

Así se comprueba la fórmula aunque el enlace nativo no esté compilado. Ninguna
prueba aplica pasos de optimizador.
"""

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_pinball_loss

from mars_titan.evaluation.forecast_panel import central_intervals
from mars_titan.models import quantile_head
from mars_titan.models.quantile_head import (
    CONTRACT,
    LEVELS,
    MEDIAN_INDEX,
    PINBALL,
    QUANTILE_COLUMNS,
    QUANTILE_HEAD,
    QuantileHead,
    median,
    ordered_quantiles,
    pinball_loss,
)

DTYPES = (torch.float32, torch.float64)


def native_formula(raw):
    """Transcripción literal de Candidate::quantiles, sin compartir código con el módulo."""
    softplus = torch.nn.functional.softplus
    middle = raw[:, 2]
    lower = middle - softplus(raw[:, 1])
    upper = middle + softplus(raw[:, 3])
    return torch.stack(
        [lower - softplus(raw[:, 0]), lower, middle, upper, upper + softplus(raw[:, 4])], 1
    )


def random_raw(rows, dtype, seed=3, scale=3.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, 5, generator=generator, dtype=dtype) * scale


@pytest.mark.parametrize("dtype", DTYPES)
def test_parametrization_reproduces_the_native_formula_bit_for_bit(dtype):
    raw = random_raw(64, dtype)
    torch.testing.assert_close(ordered_quantiles(raw), native_formula(raw), rtol=0, atol=0)
    assert ordered_quantiles(raw).dtype == dtype


def test_contract_matches_the_decision_and_the_panel_intervals():
    assert LEVELS == (0.025, 0.1, 0.5, 0.9, 0.975)
    assert LEVELS[MEDIAN_INDEX] == 0.5
    assert len(QUANTILE_COLUMNS) == len(set(QUANTILE_COLUMNS)) == len(LEVELS)
    assert CONTRACT["levels"] == list(LEVELS) and CONTRACT["columns"] == list(QUANTILE_COLUMNS)
    assert CONTRACT["name"] == QUANTILE_HEAD and PINBALL == "pinball"
    # El panel debe reconocer los intervalos centrales del 80 % y del 95 %.
    pairs = central_intervals(LEVELS)
    assert pairs == [(1, 3), (0, 4)]
    assert [round(LEVELS[b] - LEVELS[a], 12) for a, b in pairs] == [0.8, 0.95]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1e-30, 1e-3, 1.0, 30.0, 1e6, 1e30])
def test_quantiles_never_cross_even_with_extreme_free_values(dtype, scale):
    raw = random_raw(512, dtype, seed=11, scale=scale)
    raw[:8, :] = torch.tensor([-1, 1, -1, 1, -1], dtype=dtype) * scale
    quantiles = ordered_quantiles(raw)
    finite = torch.isfinite(quantiles).all(dim=1)
    assert finite.any()
    assert (quantiles[finite].diff(dim=1) >= 0).all()
    assert torch.equal(median(quantiles), raw[:, MEDIAN_INDEX])


def test_large_negative_increments_can_tie_but_never_reverse():
    raw = torch.tensor([[-200.0, -200.0, 0.25, -200.0, -200.0]], dtype=torch.float32)
    quantiles = ordered_quantiles(raw)
    assert torch.equal(quantiles, torch.full((1, 5), 0.25))


@pytest.mark.parametrize("column", range(5))
def test_each_free_value_only_moves_its_side_of_the_distribution(column):
    raw = random_raw(16, torch.float64, seed=5)
    moved = raw.clone()
    moved[:, column] += 0.75
    before, after = ordered_quantiles(raw), ordered_quantiles(moved)
    changed = (before != after).any(dim=0).tolist()
    expected = {
        0: [True, False, False, False, False],
        1: [True, True, False, False, False],
        2: [True, True, True, True, True],
        3: [False, False, False, True, True],
        4: [False, False, False, False, True],
    }[column]
    assert changed == expected
    if column == MEDIAN_INDEX:
        torch.testing.assert_close(after - before, torch.full_like(before, 0.75))


def test_rows_are_independent_and_permutation_equivariant():
    raw = random_raw(32, torch.float64, seed=7)
    order = torch.randperm(32, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(ordered_quantiles(raw[order]), ordered_quantiles(raw)[order])
    alone = torch.cat([ordered_quantiles(raw[i : i + 1]) for i in range(32)])
    torch.testing.assert_close(alone, ordered_quantiles(raw), rtol=0, atol=0)


def test_hand_computed_pinball_values():
    quantiles = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    # y = 1,5: residuos 3,5 2,5 1,5 0,5 y -0,5. El último nivel paga (1 - 0,975) · 0,5.
    # (0,0875 + 0,25 + 0,75 + 0,45 + 0,0125) / 5 = 1,55 / 5 = 0,31.
    value = pinball_loss(quantiles, torch.tensor([1.5], dtype=torch.float64))
    assert value.item() == pytest.approx(0.31, rel=1e-15)
    below = pinball_loss(quantiles, torch.tensor([-3.0], dtype=torch.float64))
    # Residuos -1, -2, -3, -4 y -5 con pesos 0,975, 0,9, 0,5, 0,1 y 0,025.
    assert below.item() == pytest.approx((0.975 + 1.8 + 1.5 + 0.4 + 0.125) / 5, rel=1e-15)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinball_matches_scikit_learn_and_the_panel_formula(dtype):
    raw = random_raw(200, dtype, seed=13)
    quantiles = ordered_quantiles(raw)
    target = torch.randn(200, generator=torch.Generator().manual_seed(2), dtype=dtype) * 2
    rows = pinball_loss(quantiles, target, reduction="none")
    q, y = quantiles.double().numpy(), target.double().numpy()
    reference = np.mean(
        [mean_pinball_loss(y, q[:, j], alpha=level) for j, level in enumerate(LEVELS)]
    )
    tolerance = 1e-6 if dtype == torch.float32 else 1e-13
    assert pinball_loss(quantiles, target).item() == pytest.approx(reference, rel=tolerance)
    residual = y[:, None] - q
    panel = np.maximum(np.array(LEVELS) * residual, (np.array(LEVELS) - 1) * residual).mean(1)
    np.testing.assert_allclose(rows.double().numpy(), panel, rtol=tolerance, atol=0)
    assert rows.mean().item() == pinball_loss(quantiles, target).item()


def test_pinball_gradient_matches_the_analytic_subgradient_including_ties():
    quantiles = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.5, 1.0], [0.2, 0.3, 0.3, 0.4, 0.9]], dtype=torch.float64
    ).requires_grad_()
    target = torch.tensor([0.0, 0.3], dtype=torch.float64)
    (gradient,) = torch.autograd.grad(pinball_loss(quantiles, target), quantiles)
    levels = torch.tensor(LEVELS, dtype=torch.float64)
    residual = target[:, None] - quantiles.detach()
    # d/dq de max(tau u, (tau - 1) u) con u = y - q. En u = 0 se toma la media tau - 1/2.
    indicator = torch.where(residual < 0, 1.0, torch.where(residual > 0, 0.0, 0.5))
    expected = -(levels - indicator.double()) / (5 * len(target))
    torch.testing.assert_close(gradient, expected, rtol=0, atol=1e-17)
    assert residual[0, 2] == 0 and gradient[0, 2] == 0


def test_raw_gradient_matches_the_jacobian_of_the_softplus_parametrization():
    raw = random_raw(6, torch.float64, seed=17).requires_grad_()
    target = torch.randn(6, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    (gradient,) = torch.autograd.grad(pinball_loss(ordered_quantiles(raw), target), raw)
    quantiles = ordered_quantiles(raw.detach())
    levels = torch.tensor(LEVELS, dtype=torch.float64)
    upstream = -(levels - (target[:, None] < quantiles).double()) / (5 * len(target))
    sigma = raw.detach().sigmoid()
    jacobian = torch.zeros(6, 5, 5, dtype=torch.float64)  # [fila, cuantil, valor libre]
    jacobian[:, :, 2] = 1
    jacobian[:, 0, 1] = jacobian[:, 1, 1] = -sigma[:, 1]
    jacobian[:, 0, 0] = -sigma[:, 0]
    jacobian[:, 3, 3] = jacobian[:, 4, 3] = sigma[:, 3]
    jacobian[:, 4, 4] = sigma[:, 4]
    expected = torch.einsum("rq,rqv->rv", upstream, jacobian)
    torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-16)


def test_gradcheck_through_head_and_loss():
    head = QuantileHead(4, dtype=torch.float64)
    state = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(5, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: pinball_loss(head(x), target), (state,))
    assert torch.autograd.gradcheck(lambda x: head(x), (state,))


def test_median_term_is_one_tenth_of_l1_including_zero_residuals():
    quantiles = ordered_quantiles(random_raw(9, torch.float64, seed=19)).requires_grad_()
    target = quantiles.detach()[:, MEDIAN_INDEX].clone()
    target[::2] += torch.linspace(-1, 1, 5, dtype=torch.float64)
    (pinball,) = torch.autograd.grad(pinball_loss(quantiles, target), quantiles)
    middle = quantiles.detach()[:, MEDIAN_INDEX].clone().requires_grad_()
    (l1,) = torch.autograd.grad(torch.nn.functional.l1_loss(middle, target), middle)
    torch.testing.assert_close(pinball[:, MEDIAN_INDEX] * 10, l1, rtol=1e-15, atol=0)
    assert (l1[1::2] == 0).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_pinball_is_translation_invariant_and_positively_homogeneous(dtype):
    quantiles = ordered_quantiles(random_raw(64, dtype, seed=23))
    target = torch.randn(64, generator=torch.Generator().manual_seed(6), dtype=dtype)
    value = pinball_loss(quantiles, target)
    shift, scale = 0.375, 4.0
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=0)
    torch.testing.assert_close(pinball_loss(quantiles + shift, target + shift), value, **tolerance)
    torch.testing.assert_close(
        pinball_loss(quantiles * scale, target * scale), value * scale, rtol=0, atol=0
    )
    order = torch.randperm(64, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(
        pinball_loss(quantiles[order], target[order], reduction="none"),
        pinball_loss(quantiles, target, reduction="none")[order],
        rtol=0,
        atol=0,
    )
    assert pinball_loss(quantiles, quantiles[:, MEDIAN_INDEX]) >= 0


def test_a_constant_bias_on_the_median_shifts_every_quantile():
    head = QuantileHead(3, dtype=torch.float64)
    state = torch.randn(4, 3, dtype=torch.float64)
    before = head(state)
    with torch.no_grad():
        head.bias[MEDIAN_INDEX] += 0.5
    torch.testing.assert_close(head(state) - before, torch.full_like(before, 0.5))


def test_head_keeps_linear_names_shapes_and_default_initialization():
    torch.manual_seed(31)
    head = QuantileHead(16)
    torch.manual_seed(31)
    linear = torch.nn.Linear(16, 5)
    assert dict(head.named_parameters()).keys() == {"weight", "bias"}
    assert head.weight.shape == (5, 16) and head.bias.shape == (5,)
    torch.testing.assert_close(head.weight, linear.weight, rtol=0, atol=0)
    torch.testing.assert_close(head.bias, linear.bias, rtol=0, atol=0)
    state = torch.randn(3, 16)
    torch.testing.assert_close(head(state), ordered_quantiles(linear(state)), rtol=0, atol=0)


@pytest.mark.parametrize("width", [0, -1, 4097, 2.0, True])
def test_invalid_widths_are_rejected(width):
    with pytest.raises(ValueError, match="anchura"):
        QuantileHead(width)


@pytest.mark.parametrize(
    "quantiles,target,options",
    [
        (torch.zeros(3, 4), torch.zeros(3), {}),
        (torch.zeros(3, 5), torch.zeros(2), {}),
        (torch.zeros(3, 5), torch.zeros(3, 1), {}),
        (torch.zeros(0, 5), torch.zeros(0), {}),
        (torch.zeros(3, 5), torch.zeros(3, dtype=torch.float64), {}),
        (torch.zeros(3, 5, dtype=torch.int64), torch.zeros(3, dtype=torch.int64), {}),
        (torch.zeros(3, 5), torch.zeros(3), dict(reduction="sum")),
        (torch.zeros(1, 3, 5), torch.zeros(1), {}),
        (np.zeros((3, 5)), torch.zeros(3), {}),
    ],
)
def test_pinball_rejects_ambiguous_inputs(quantiles, target, options):
    with pytest.raises(ValueError):
        pinball_loss(quantiles, target, **options)


@pytest.mark.parametrize("value", [torch.zeros(3, 4), torch.zeros(()), np.zeros((3, 5))])
def test_parametrization_and_median_reject_other_widths(value):
    with pytest.raises(ValueError):
        ordered_quantiles(value)
    with pytest.raises(ValueError):
        median(value)


def test_nonfinite_free_values_propagate_instead_of_being_hidden():
    raw = torch.tensor([[0.0, 0.0, math.nan, 0.0, 0.0]])
    assert torch.isnan(ordered_quantiles(raw)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere cuda:0 explícito")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_head_and_loss_match_cpu(dtype):
    head = QuantileHead(32, dtype=dtype)
    device_head = QuantileHead(32, dtype=dtype, device="cuda:0")
    device_head.load_state_dict(head.state_dict())
    state = torch.randn(64, 32, generator=torch.Generator().manual_seed(29), dtype=dtype)
    target = torch.randn(64, generator=torch.Generator().manual_seed(30), dtype=dtype)
    values = []
    for model, device in ((head, "cpu"), (device_head, "cuda:0")):
        quantiles = model(state.to(device))
        loss = pinball_loss(quantiles, target.to(device))
        gradients = torch.autograd.grad(loss, (model.weight, model.bias))
        values.append([quantiles, loss, *gradients])
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=0)
    for cpu, cuda in zip(*values, strict=True):
        torch.testing.assert_close(cuda.cpu(), cpu, **tolerance)
    assert (values[1][0].diff(dim=1) >= 0).all()


def test_cached_levels_work_after_inference_mode_and_keep_the_loss():
    quantiles = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    target = torch.tensor([0.25], dtype=torch.float32)
    expected = torch.tensor(LEVELS, dtype=torch.float32)
    quantile_head._levels.cache_clear()
    with torch.inference_mode():
        first = pinball_loss(quantiles.clone(), target.clone())
    trained = quantiles.clone().requires_grad_(True)
    loss = pinball_loss(trained, target)
    loss.backward()
    residual = target.unsqueeze(-1) - quantiles
    reference = torch.maximum(expected * residual, (expected - 1) * residual).mean()
    assert torch.equal(first, reference) and torch.equal(loss.detach(), reference)
    assert trained.grad is not None
