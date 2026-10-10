"""Decaimiento de AdamW anclado al padre (#444).

Las pruebas que dan pasos completos se omiten mientras la protección del aprendizaje esté
vigente (`tests/conftest.py`). Las demás calculan el gradiente y la actualización sobre
copias, con la misma función `adam` que llama el paso de AdamW, y no modifican ningún peso
ni el estado del optimizador.
"""

import io
import math

import pytest
import torch
from torch import nn
from torch.optim.adam import adam

from mars_titan.training import anchored_decay
from mars_titan.training.anchored_decay import INITIAL, AnchoredAdamW

# Presupuesto de la matriz v3 y una tasa grande para que el decaimiento se note en un paso.
BUDGET = (1e-4, 0.01)
LARGE = (0.1, 0.5)
RATES = pytest.mark.parametrize("lr, decay", [BUDGET, LARGE], ids=["budget", "large"])


def _model(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 2)).to(dtype)


def _gradients(model, seed=1):
    """Gradiente de una pérdida L1 sobre datos fijos, sin tocar los pesos."""
    generator = torch.Generator().manual_seed(seed)
    dtype = next(model.parameters()).dtype
    inputs = torch.randn(32, 6, generator=generator).to(dtype)
    target = torch.randn(32, 2, generator=generator).to(dtype)
    model.zero_grad(set_to_none=True)
    (model(inputs) - target).abs().mean().backward()
    return model


def _planned(optimizer):
    """Valor de cada parámetro tras el primer paso, calculado sobre copias.

    Reproduce el paso de AdamW con su función `adam` desde un estado vacío. En el anclado
    parte de las copias que devuelve `decayed_parameters`, igual que el gancho previo al
    paso. Nada de lo que recibe cambia.
    """
    decayed = {}
    if isinstance(optimizer, AnchoredAdamW):
        decayed = {id(value): copy for value, copy in optimizer.decayed_parameters()}
    result = {}
    for group in optimizer.param_groups:
        params = [value for value in group["params"] if value.grad is not None]
        values = [decayed.get(id(value), value.detach().clone()) for value in params]
        beta1, beta2 = group["betas"]
        adam(
            values,
            [value.grad.detach().clone() for value in params],
            [torch.zeros_like(value) for value in values],
            [torch.zeros_like(value) for value in values],
            [],
            [torch.tensor(0.0) for _ in values],
            foreach=group["foreach"],
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            decoupled_weight_decay=group["decoupled_weight_decay"],
            amsgrad=group["amsgrad"],
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
        )
        result.update({id(value): planned for value, planned in zip(params, values, strict=True)})
    return result


def _snapshot(model):
    return [value.detach().clone() for value in model.parameters()]


def _same(left, right):
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )


def _zero_anchors(optimizer):
    for group in optimizer.param_groups:
        group["anchors"] = [torch.zeros_like(value) for value in group["params"]]


def test_lambda_zero_plans_the_same_update_as_adamw_without_decay():
    model = _gradients(_model())
    before = _snapshot(model)
    plain = torch.optim.AdamW(model.parameters(), lr=BUDGET[0], weight_decay=0.0)
    anchored = AnchoredAdamW(model.parameters(), lr=BUDGET[0], weight_decay=0.0)
    assert anchored.decayed_parameters() == []
    # Los hiperparámetros que recibe AdamW son los mismos: solo se añaden el ancla y su λ.
    for ours, theirs in zip(anchored.param_groups, plain.param_groups, strict=True):
        extra = {"params", "anchors", "anchored_weight_decay"}
        assert {k: v for k, v in ours.items() if k not in extra} == {
            k: v for k, v in theirs.items() if k != "params"
        }
        assert ours["anchored_weight_decay"] == 0.0
    planned, reference = _planned(anchored), _planned(plain)
    assert planned.keys() == reference.keys()
    assert all(torch.equal(planned[key], reference[key]) for key in reference)
    assert _same(_snapshot(model), before) and not anchored.state and not plain.state


@RATES
def test_zero_anchors_reproduce_the_adamw_decay_bit_for_bit(lr, decay):
    model = _gradients(_model())
    anchored = AnchoredAdamW(model.parameters(), lr=lr, weight_decay=decay)
    _zero_anchors(anchored)
    rate = lr * decay
    # La copia decaída es la de AdamW, `param.mul_(1 - lr * weight_decay)`.
    for value, copy in anchored.decayed_parameters():
        assert torch.equal(copy, value.detach().clone().mul_(1 - rate))
    planned = _planned(anchored)
    reference = _planned(torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay))
    undecayed = _planned(torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0))
    assert all(torch.equal(planned[key], reference[key]) for key in reference)
    # La prueba distingue: sin decaimiento el paso sería otro.
    assert any(not torch.equal(reference[key], undecayed[key]) for key in reference)


@RATES
def test_the_parent_is_a_fixed_point_while_adamw_contracts_it(lr, decay):
    model = _gradients(_model())
    anchored = AnchoredAdamW(model.parameters(), lr=lr, weight_decay=decay)
    decayed = anchored.decayed_parameters()
    assert len(decayed) == len(list(model.parameters()))
    assert all(torch.equal(copy, value) for value, copy in decayed)
    # En el padre el paso anclado es exactamente el de AdamW sin decaimiento.
    planned = _planned(anchored)
    undecayed = _planned(torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0))
    assert all(torch.equal(planned[key], undecayed[key]) for key in undecayed)
    # AdamW, en cambio, resta ηλθ a cada peso del padre además de su actualización.
    contracted = _planned(torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay))
    for value in model.parameters():
        shift = contracted[id(value)] - undecayed[id(value)]
        torch.testing.assert_close(shift, -lr * decay * value.detach(), rtol=1e-3, atol=1e-7)


def test_repeated_decay_contracts_only_the_deviation_from_the_parent():
    rate, steps = 1e-3, 1000
    parent = torch.linspace(-2.0, 2.0, 41, dtype=torch.float64)
    deviation = torch.linspace(0.5, -0.5, 41, dtype=torch.float64)
    anchored, plain = [parent + deviation], [parent + deviation]
    zeros = [torch.zeros_like(parent)]
    for _ in range(steps):
        anchored_decay._decay_(anchored, [parent], rate)
        anchored_decay._decay_(plain, zeros, rate)
    factor = (1 - rate) ** steps
    torch.testing.assert_close(anchored[0] - parent, deviation * factor, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(plain[0], (parent + deviation) * factor, rtol=1e-12, atol=0.0)
    # Con el presupuesto declarado, AdamW contrae al padre en 1 − (1 − ηλ)^T. Entre 3.300 y
    # 6.000 actualizaciones son entre un 0,33 % y un 0,60 % del peso.
    rate = BUDGET[0] * BUDGET[1]
    assert [round(100 * (1 - (1 - rate) ** t), 2) for t in (3300, 6000)] == [0.33, 0.6]


@pytest.mark.parametrize("lr, decay", [LARGE], ids=["large"])
def test_anchored_step_matches_a_full_rank_residual_adapter(lr, decay):
    """θ con ancla θ₀ frente a AdamW sobre δ con θ = θ₀ + δ, en float64."""
    model = _model(dtype=torch.float64)
    parent = _snapshot(model)
    generator = torch.Generator().manual_seed(5)
    offsets = [0.1 * torch.randn(v.shape, generator=generator, dtype=v.dtype) for v in parent]
    anchored = AnchoredAdamW(model.parameters(), lr=lr, weight_decay=decay)
    with torch.no_grad():
        for value, offset in zip(model.parameters(), offsets, strict=True):
            value.add_(offset)
    _gradients(model)
    corrections = [nn.Parameter(offset.clone()) for offset in offsets]
    for correction, value in zip(corrections, model.parameters(), strict=True):
        correction.grad = value.grad.detach().clone()
    adapter = _planned(torch.optim.AdamW(corrections, lr=lr, weight_decay=decay))
    planned = _planned(anchored)
    for value, start, correction in zip(model.parameters(), parent, corrections, strict=True):
        torch.testing.assert_close(
            planned[id(value)], start + adapter[id(correction)], rtol=0.0, atol=1e-15
        )


def test_parameters_without_gradient_do_not_decay():
    model = _gradients(_model())
    frozen = model[2].bias
    frozen.grad = None
    anchored = AnchoredAdamW(model.parameters(), lr=LARGE[0], weight_decay=LARGE[1])
    _zero_anchors(anchored)
    decayed = {id(value) for value, _ in anchored.decayed_parameters()}
    assert id(frozen) not in decayed and len(decayed) == len(list(model.parameters())) - 1


def test_anchors_survive_a_restricted_checkpoint_and_belong_to_the_parent():
    model = _gradients(_model())
    parent = _snapshot(model)
    anchored = AnchoredAdamW(model.parameters(), lr=LARGE[0], weight_decay=LARGE[1])
    stream = io.BytesIO()
    torch.save(anchored.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    # La reanudación carga primero los pesos ya ajustados y construye el optimizador con
    # ellos. Las anclas buenas son las guardadas, que siguen siendo el padre.
    with torch.no_grad():
        for value in model.parameters():
            value.add_(1.0)
    resumed = AnchoredAdamW(model.parameters(), lr=LARGE[0], weight_decay=LARGE[1])
    assert not _same(resumed.param_groups[0]["anchors"], parent)
    resumed.load_state_dict(saved)
    anchors = resumed.param_groups[0]["anchors"]
    assert _same(anchors, parent)
    # Las anclas no comparten memoria con el estado leído (PyTorch copia los grupos).
    saved["param_groups"][0]["anchors"][0].add_(5.0)
    assert _same(resumed.param_groups[0]["anchors"], parent)
    assert resumed.param_groups[0]["anchored_weight_decay"] == LARGE[1]


def test_inconsistent_anchors_are_rejected_on_load():
    model = _model()
    anchored = AnchoredAdamW(model.parameters(), lr=LARGE[0], weight_decay=LARGE[1])
    for change in (
        lambda anchors: anchors.pop(),
        lambda anchors: anchors.__setitem__(0, torch.zeros(3)),
        lambda anchors: anchors.__setitem__(0, anchors[0].double()),
    ):
        state = anchored.state_dict()
        state["param_groups"] = [dict(state["param_groups"][0])]
        state["param_groups"][0]["anchors"] = list(state["param_groups"][0]["anchors"])
        change(state["param_groups"][0]["anchors"])
        with pytest.raises(ValueError, match="anclas guardadas"):
            AnchoredAdamW(model.parameters(), lr=1e-3, weight_decay=0.1).load_state_dict(state)


def test_invalid_declarations_and_closures_are_rejected():
    params = list(_model().parameters())
    for value in (-0.1, math.inf, math.nan, True, torch.tensor(0.1)):
        with pytest.raises(ValueError, match="finitos y no negativos"):
            AnchoredAdamW(params, lr=1e-3, weight_decay=value)
    with pytest.raises(ValueError, match="tasa de aprendizaje anclada"):
        AnchoredAdamW(params, lr=torch.tensor(1e-3), weight_decay=0.1)
    with pytest.raises(ValueError, match="decaimiento hacia cero"):
        AnchoredAdamW([dict(params=params, weight_decay=0.1)], lr=1e-3, weight_decay=0.1)
    with pytest.raises(ValueError, match="finitos y no negativos"):
        AnchoredAdamW([dict(params=params, anchored_weight_decay=-1.0)], lr=1e-3, weight_decay=0)
    with pytest.raises(ValueError, match="solo se ancla"):
        anchored_decay.adamw(params, learning_rate=1e-3, weight_decay=0.1, anchor="parent")
    # El gancho rechaza la clausura antes de tocar ningún peso.
    model = _gradients(_model())
    before = _snapshot(model)
    optimizer = AnchoredAdamW(model.parameters(), lr=LARGE[0], weight_decay=LARGE[1])
    _zero_anchors(optimizer)
    for args, kwargs in (((lambda: 0.0,), {}), ((), dict(closure=lambda: 0.0))):
        with pytest.raises(ValueError, match="clausura"):
            anchored_decay._apply_decay(optimizer, args, kwargs)
    assert _same(_snapshot(model), before)


def test_factory_keeps_plain_adamw_without_anchor_and_describes_the_anchor():
    params = list(_model().parameters())
    plain = anchored_decay.adamw(params, learning_rate=1e-3, weight_decay=0.01)
    reference = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    assert type(plain) is torch.optim.AdamW and plain.defaults == reference.defaults
    assert anchored_decay.describe(plain, None) is None
    anchored = anchored_decay.adamw(params, learning_rate=1e-3, weight_decay=0.01, anchor=INITIAL)
    assert type(anchored) is AnchoredAdamW and anchored.defaults["weight_decay"] == 0.0
    described = anchored_decay.describe(anchored, INITIAL)
    assert described["anchor"] == INITIAL and len(described["implementation_sha256"]) == 64
    # Un optimizador de PyTorch que no aplica el ancla declarada, o al revés, se rechaza.
    for optimizer, anchor in ((plain, INITIAL), (anchored, None)):
        with pytest.raises(ValueError, match="no aplica el ancla"):
            anchored_decay.describe(optimizer, anchor)
    # Los dobles sin pasos de las pruebas solo registran el ancla declarada.
    assert anchored_decay.describe(object(), INITIAL)["anchor"] == INITIAL


# Pasos completos: se omiten bajo la protección del aprendizaje.


def _run(optimizer_factory, model, steps, *, seed=1):
    optimizer = optimizer_factory(model)
    for step in range(steps):
        _gradients(model, seed=seed + step)
        optimizer.step()
    return optimizer


@pytest.mark.parametrize("foreach", [False, True])
def test_full_steps_with_lambda_zero_equal_adamw_without_decay(foreach):
    left, right = _model(), _model()
    _run(
        lambda m: torch.optim.AdamW(m.parameters(), lr=0.05, weight_decay=0.0, foreach=foreach),
        left,
        5,
    )
    _run(
        lambda m: AnchoredAdamW(m.parameters(), lr=0.05, weight_decay=0.0, foreach=foreach),
        right,
        5,
    )
    assert _same(_snapshot(left), _snapshot(right))


@pytest.mark.parametrize("foreach", [False, True])
def test_full_steps_with_zero_anchors_equal_adamw(foreach):
    def anchored(model):
        optimizer = AnchoredAdamW(model.parameters(), lr=0.05, weight_decay=0.5, foreach=foreach)
        _zero_anchors(optimizer)
        return optimizer

    left, right = _model(), _model()
    _run(
        lambda m: torch.optim.AdamW(m.parameters(), lr=0.05, weight_decay=0.5, foreach=foreach),
        left,
        5,
    )
    _run(anchored, right, 5)
    assert _same(_snapshot(left), _snapshot(right))


def test_resumed_steps_equal_the_uninterrupted_run():
    def factory(model):
        return AnchoredAdamW(model.parameters(), lr=0.05, weight_decay=0.5)

    whole = _model()
    _run(factory, whole, 5)
    part = _model()
    optimizer = _run(factory, part, 3)
    stream = io.BytesIO()
    torch.save(dict(model=part.state_dict(), optimizer=optimizer.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    resumed = _model(seed=9)
    resumed.load_state_dict(saved["model"])
    optimizer = factory(resumed)
    optimizer.load_state_dict(saved["optimizer"])
    for step in range(3, 5):
        _gradients(resumed, seed=1 + step)
        optimizer.step()
    assert _same(_snapshot(whole), _snapshot(resumed))
