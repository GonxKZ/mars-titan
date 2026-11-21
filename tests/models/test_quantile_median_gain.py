"""Ganancia aprendida de la mediana de la cabeza de cuantiles (`learned_median_gain_v1`)."""

import math

import pytest
import torch
from torch.nn import functional

from mars_titan.models.quantile_head import (
    MEDIAN_GAIN,
    MEDIAN_INDEX,
    QuantileHead,
    median,
    ordered_quantiles,
    pinball_loss,
)
from mars_titan.models.quantile_routing import QuantileRouting
from mars_titan.models.titans.session_ic_objective import SessionICTerm


def pair(width=16, seed=31, **options):
    """La misma cabeza con y sin ganancia, con los mismos pesos sorteados."""
    torch.manual_seed(seed)
    plain = QuantileHead(width, dtype=torch.float64, **options)
    torch.manual_seed(seed)
    gained = QuantileHead(width, dtype=torch.float64, median_gain=True, **options)
    return plain, gained


def test_without_the_gain_the_head_keeps_its_parameters_and_its_state():
    plain, gained = pair()
    assert MEDIAN_GAIN == "learned_median_gain_v1"
    assert dict(plain.named_parameters()).keys() == {"weight", "bias"}
    assert set(plain.state_dict()) == {"weight", "bias"} and plain.median_gain is None
    assert dict(gained.named_parameters()).keys() == {"weight", "bias", "median_gain"}
    # La ganancia no consume el generador: los pesos sorteados son los mismos.
    torch.testing.assert_close(gained.weight, plain.weight, rtol=0, atol=0)
    torch.testing.assert_close(gained.bias, plain.bias, rtol=0, atol=0)
    assert gained.median_gain.shape == () and gained.median_gain.item() == 0.0
    assert gained.median_gain.dtype == torch.float64 and gained.median_gain.requires_grad
    # Empieza en cero: emite lo mismo que la cabeza sin ganancia, bit a bit.
    state = torch.randn(7, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(gained(state), plain(state), rtol=0, atol=0)
    with pytest.raises((RuntimeError, ValueError)):
        plain.load_state_dict(gained.state_dict())
    with pytest.raises(ValueError, match="booleano"):
        QuantileHead(8, median_gain=1)


@pytest.mark.parametrize("gain", [math.log(0.1), 0.0, math.log(4.0)])
def test_the_gain_scales_the_median_and_the_tails_stay_increments_over_it(gain):
    plain, gained = pair()
    with torch.no_grad():
        gained.median_gain.fill_(gain)
    state = torch.randn(9, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    raw = functional.linear(state, plain.weight, plain.bias)
    quantiles = gained(state)
    center = math.exp(gain) * raw[:, MEDIAN_INDEX]
    lower, upper = center - functional.softplus(raw[:, 1]), center + functional.softplus(raw[:, 3])
    expected = torch.stack(
        (
            lower - functional.softplus(raw[:, 0]),
            lower,
            center,
            upper,
            upper + functional.softplus(raw[:, 4]),
        ),
        dim=1,
    )
    torch.testing.assert_close(quantiles, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(median(quantiles), center, rtol=1e-14, atol=1e-14)
    assert bool((quantiles[:, 1:] >= quantiles[:, :-1]).all())
    # La mediana conserva el signo y el orden entre filas; las anchuras no cambian.
    reference = ordered_quantiles(raw)
    assert torch.equal(median(quantiles).argsort(), median(reference).argsort())
    torch.testing.assert_close(
        quantiles[:, 3] - quantiles[:, 2], reference[:, 3] - reference[:, 2], rtol=1e-12, atol=1e-12
    )


def test_a_scale_free_term_does_not_move_the_gain_and_pinball_does():
    _, gained = pair()
    generator = torch.Generator().manual_seed(6)
    state = torch.randn(60, 16, dtype=torch.float64, generator=generator)
    target = 0.05 * torch.randn(60, dtype=torch.float64, generator=generator)
    sessions = [("US", row // 20) for row in range(60)]
    quantiles = gained(state)
    term = SessionICTerm(weight=3.0, minimum_rows=5).term(median(quantiles), target, sessions)
    (free,) = torch.autograd.grad(term, gained.median_gain, retain_graph=True)
    (weights,) = torch.autograd.grad(term, gained.weight, retain_graph=True)
    # La correlación no depende de la escala común de la mediana: la ganancia no recibe nada
    # (salvo el suelo relativo de la normalización) y la dirección de la mediana sí.
    assert abs(free.item()) < 1e-8 * weights.abs().sum().item()
    assert weights[MEDIAN_INDEX].abs().sum() > 0
    # La mediana libre está muy por encima de la escala del objetivo: pinball pide encogerla.
    (scale,) = torch.autograd.grad(pinball_loss(quantiles, target), gained.median_gain)
    assert scale.item() > 0


def test_pinball_alone_brings_the_gain_to_the_scale_of_the_target():
    torch.manual_seed(1)
    head = QuantileHead(1, dtype=torch.float64, median_gain=True)
    generator = torch.Generator().manual_seed(8)
    signal = torch.randn(4000, 1, dtype=torch.float64, generator=generator)
    target = 0.1 * signal[:, 0] + 0.02 * torch.randn(4000, dtype=torch.float64, generator=generator)
    with torch.no_grad():
        # La mediana libre es la señal tal cual, diez veces más dispersa que el objetivo.
        head.weight.zero_()
        head.bias.zero_()
        head.weight[MEDIAN_INDEX, 0] = 1.0
        head.bias[[0, 1, 3, 4]] = -3.0
    optimizer = torch.optim.Adam([head.median_gain], lr=0.02)
    for _ in range(400):
        optimizer.zero_grad()
        pinball_loss(head(signal), target).backward()
        optimizer.step()
    assert math.exp(head.median_gain.item()) == pytest.approx(0.1, rel=0.08)


def test_the_gain_also_applies_with_the_learning_routes_and_the_output_scale():
    routing = QuantileRouting(median_from_tails=False, encoder_from_spreads=False)
    plain, gained = pair(routing=routing, scaled=True)
    with torch.no_grad():
        gained.median_gain.fill_(math.log(0.25))
        plain.output_scale.fill_(0.02)
        gained.output_scale.fill_(0.02)
    state = torch.randn(5, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        expected = plain(state)
        actual = gained(state)
    # Misma anchura a cada lado y la mediana a un cuarto, todo por la escala de salida.
    torch.testing.assert_close(actual[:, 2], 0.25 * expected[:, 2], rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(
        actual[:, 4] - actual[:, 2], expected[:, 4] - expected[:, 2], rtol=1e-12, atol=1e-14
    )
    # Con las rutas, el gradiente de las colas no llega a la mediana ni a su ganancia.
    quantiles = gained(state)
    (gain,) = torch.autograd.grad(quantiles[:, [0, 1, 3, 4]].sum(), gained.median_gain)
    assert gain.item() == 0.0
    (through,) = torch.autograd.grad(gained(state)[:, 2].sum(), gained.median_gain)
    assert through.item() != 0.0
