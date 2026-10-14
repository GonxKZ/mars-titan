"""Los K refinamientos de una llamada repiten el cálculo, no las validaciones.

Dentro de `forward`, la instantánea, el presupuesto y la puerta no cambian entre pasos, y cada
lectura sale de `_read` con la forma del estado, finita y con relleno cero en las filas
ausentes. Por eso solo el primer paso las valida. El resultado debe ser bit a bit el de
componer `_read` y la `refine` pública, que sigue validándolo todo.
"""

import pytest
import torch
from test_episodic_readout import CONTEXT, api
from test_episodic_readout_fixed_episodes import random_reader, random_snapshot

OPTIONS = dict(context_id=CONTEXT, cutoff=10)


def composed(model, data, z, steps, selection, differentiable):
    """El mismo recorrido con la lectura y el refinamiento públicos, paso a paso."""
    with torch.set_grad_enabled(differentiable):
        base = z if differentiable else z.detach()
        state, positions, reads = base, None, []
        for _ in range(steps):
            read, chosen = model._read(state, data, CONTEXT, 10, positions)
            if selection == "first_read":
                positions = chosen
            state = model.refine(state, base, read)
            reads.append(read)
    return state, reads


def gradients(model, z, state):
    for parameter in model.parameters():
        parameter.grad = None
    z.grad = None
    state.sum().backward()
    return [z.grad.clone()] + [parameter.grad.clone() for parameter in model.parameters()]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("selection", ["per_step", "first_read"])
@pytest.mark.parametrize("differentiable", [False, True])
def test_forward_matches_the_public_read_and_refine_bit_for_bit(
    dtype, steps, selection, differentiable
):
    model = random_reader(5, dtype, neighbors=4, refinements=steps, episode_selection=selection)
    data = random_snapshot(10, dtype)
    generator = torch.Generator().manual_seed(8)
    z = torch.randn((3, 32), generator=generator, dtype=dtype).requires_grad_(differentiable)
    result = model(z, data, differentiable=differentiable, **OPTIONS)
    state, reads = composed(model, data, z, steps, selection, differentiable)
    assert torch.equal(result.state, state)
    assert len(result.reads) == len(reads) == steps
    for got, expected in zip(result.reads, reads, strict=True):
        for name in ("values", "weights", "ids", "presence"):
            assert torch.equal(getattr(got, name), getattr(expected, name))
    if differentiable:
        for got, expected in zip(
            gradients(model, z, result.state), gradients(model, z, state), strict=True
        ):
            assert torch.equal(got, expected)


def test_refinements_validate_once_and_skip_the_checks_their_reads_satisfy(monkeypatch):
    module = api()
    model = random_reader(5, torch.float64, neighbors=4, refinements=4)
    data = random_snapshot(10, torch.float64)
    z = torch.randn((3, 32), generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    calls = dict(validate=0, refine=0, zero_fill=0, gate=0)
    validate, refine = model._validate_read, model.refine
    require_true, check_finite = module.require_true, module.check_finite

    def counted(name, function):
        def call(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return call

    def gate(value, name):
        calls["gate"] += name == "La puerta de refinamiento"
        return check_finite(value, name)

    monkeypatch.setattr(model, "_validate_read", counted("validate", validate))
    monkeypatch.setattr(model, "refine", counted("refine", refine))
    monkeypatch.setattr(module, "require_true", counted("zero_fill", require_true))
    monkeypatch.setattr(module, "check_finite", gate)
    model(z, data, **OPTIONS)
    assert calls == dict(validate=1, refine=0, zero_fill=0, gate=1)
    # La lectura y el refinamiento públicos lo siguen validando todo.
    read = model.read(z, data, **OPTIONS)
    model.refine(z, z, read)
    assert calls == dict(validate=2, refine=1, zero_fill=1, gate=2)
