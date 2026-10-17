"""Decaimiento de AdamW anclado al padre (#444).

El decaimiento se comprueba sobre copias, con `decayed_parameters`, frente a su fórmula
cerrada en FP64 y frente al factor que AdamW aplica antes de Adam, `1 - lr * weight_decay`.
Las pruebas de la fórmula no comprueban la llamada real al gancho. Las pruebas que
dan pasos completos se omiten mientras la protección del aprendizaje esté vigente, porque el
gancho global de `tests/conftest.py` intercepta `optimizer.step()`.
"""

import io
import math

import pytest
import torch
from torch import nn

from mars_titan.training import anchored_decay
from mars_titan.training.anchored_decay import INITIAL, AnchoredAdamW
from mars_titan.training.learning_hold import learning_blocked

# Presupuesto de la matriz v3 y una tasa grande para que el decaimiento se note en un paso.
BUDGET = (1e-4, 0.01)
LARGE = (0.1, 0.5)
RATES = pytest.mark.parametrize("lr, decay", [BUDGET, LARGE], ids=["budget", "large"])
# Las pruebas de pasos completos se omiten antes de calcular ningún gradiente mientras rija
# el bloqueo. El gancho global también las detendría, pero después del backward.
STEPS = pytest.mark.skipif(
    learning_blocked(), reason="Da pasos de optimizador: se omite mientras rija el bloqueo"
)


def _model(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 2)).to(dtype)


def _gradients(model, seed=1):
    """Gradiente de una pérdida L1 sobre datos fijos, sin tocar los pesos.

    Solo lo usan las pruebas de pasos completos, que se omiten bajo la protección.
    """
    generator = torch.Generator().manual_seed(seed)
    parameter = next(model.parameters())
    inputs = torch.randn(32, 6, generator=generator).to(parameter)
    target = torch.randn(32, 2, generator=generator).to(parameter)
    model.zero_grad(set_to_none=True)
    (model(inputs) - target).abs().mean().backward()
    return model


def _marked(model):
    """Marcar todos los parámetros con un gradiente nulo, sin pérdida ni datos.

    El decaimiento solo actúa sobre los parámetros con gradiente, como en AdamW, y no
    depende de su valor.
    """
    for value in model.parameters():
        value.grad = torch.zeros_like(value)
    return model


def _moved(model, seed=3, scale=0.1):
    """Separar los pesos de sus anclas con desplazamientos fijos, sin optimizador."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in model.parameters():
            value.add_(scale * torch.randn(value.shape, generator=generator, dtype=value.dtype))
    return model


def _snapshot(model):
    return [value.detach().clone() for value in model.parameters()]


def _same(left, right):
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )


def _zero_anchors(optimizer):
    for group in optimizer.param_groups:
        group["anchors"] = [torch.zeros_like(value) for value in group["params"]]


def test_lambda_zero_leaves_the_update_of_adamw_without_decay():
    model = _marked(_model())
    before = _snapshot(model)
    plain = torch.optim.AdamW(model.parameters(), lr=BUDGET[0], weight_decay=0.0)
    anchored = AnchoredAdamW(model.parameters(), lr=BUDGET[0], weight_decay=0.0)
    # Sin λ el gancho no decae nada.
    assert anchored.decayed_parameters() == [] and list(anchored._decaying()) == []
    # Los hiperparámetros que recibe AdamW son los mismos: solo se añaden el ancla y su λ.
    for ours, theirs in zip(anchored.param_groups, plain.param_groups, strict=True):
        extra = {"params", "anchors", "anchored_weight_decay"}
        assert {k: v for k, v in ours.items() if k not in extra} == {
            k: v for k, v in theirs.items() if k != "params"
        }
        assert ours["anchored_weight_decay"] == 0.0
    assert _same(_snapshot(model), before) and not anchored.state and not plain.state


@RATES
def test_zero_anchors_reproduce_the_adamw_decay_bit_for_bit(lr, decay):
    model = _moved(_marked(_model()))
    anchored = AnchoredAdamW(model.parameters(), lr=lr, weight_decay=decay)
    _zero_anchors(anchored)
    rate = lr * decay
    # AdamW multiplica cada peso por (1 − ηλ) antes de Adam, `param.mul_(1 - lr * weight_decay)`.
    # Con anclas nulas la copia decaída es exactamente ese producto.
    decayed = anchored.decayed_parameters()
    assert len(decayed) == len(list(model.parameters()))
    for value, copy in decayed:
        assert torch.equal(copy, value.detach().clone().mul_(1 - rate))
    # AdamW recibe λ = 0 con el decaimiento desacoplado activo, así que no decae dos veces.
    reference = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    for ours, theirs in zip(anchored.param_groups, reference.param_groups, strict=True):
        assert ours["weight_decay"] == 0.0 and ours["anchored_weight_decay"] == decay
        assert ours["decoupled_weight_decay"] is theirs["decoupled_weight_decay"] is True
        assert theirs["weight_decay"] == decay and ours["lr"] == theirs["lr"] == lr


@RATES
def test_the_decay_matches_its_closed_form_in_float64(lr, decay):
    """θ₀ + (θ − θ₀)(1 − ηλ) en FP32 frente a la misma fórmula en FP64."""
    model = _marked(_model())
    anchored = AnchoredAdamW(model.parameters(), lr=lr, weight_decay=decay)
    # Al construir el optimizador θ = θ₀: el padre es un punto fijo exacto.
    decayed = anchored.decayed_parameters()
    assert len(decayed) == len(list(model.parameters()))
    assert all(torch.equal(copy, value) for value, copy in decayed)
    parent = _snapshot(model)
    moved = _snapshot(_moved(model))
    # Las anclas son el padre y no se mueven con los pesos.
    assert _same(anchored.param_groups[0]["anchors"], parent)
    for (value, copy), anchor in zip(anchored.decayed_parameters(), parent, strict=True):
        start, current = anchor.double(), value.detach().double()
        exact = start + (current - start) * (1 - lr * decay)
        # Tres operaciones FP32 sobre pesos de orden uno: unos pocos ulp de diferencia.
        torch.testing.assert_close(copy.double(), exact, rtol=1e-6, atol=1e-7)
        # El decaimiento anclado acerca el peso al padre, salvo el redondeo de un ulp.
        assert ((copy.double() - start).abs() <= (current - start).abs() + 1e-7).all()
    # Las copias no tocan los pesos.
    assert _same(_snapshot(model), moved)


@RATES
def test_the_parent_is_a_fixed_point_while_adamw_contracts_it(lr, decay):
    model = _marked(_model())
    anchored = AnchoredAdamW(model.parameters(), lr=lr, weight_decay=decay)
    for value, copy in anchored.decayed_parameters():
        assert torch.equal(copy, value)
        # AdamW, en cambio, resta ηλθ a cada peso del padre antes de su actualización.
        contracted = value.detach().clone().mul_(1 - lr * decay)
        torch.testing.assert_close(
            contracted.double() - value.detach().double(),
            -lr * decay * value.detach().double(),
            rtol=1e-3,
            atol=1e-7,
        )
        assert not torch.equal(contracted, value) or lr * decay * value.abs().max() < 1e-7


@pytest.mark.parametrize("lr, decay", [LARGE], ids=["large"])
def test_the_anchored_decay_is_the_decay_of_a_full_rank_residual_adapter(lr, decay):
    """θ con ancla θ₀ frente a la corrección δ de un adaptador con θ = θ₀ + δ, en float64.

    AdamW decae la corrección a δ(1 − ηλ), así que el peso efectivo del adaptador queda en
    θ₀ + δ(1 − ηλ), lo mismo que el decaimiento anclado. El resto del paso es la
    actualización de Adam con ∂L/∂δ = ∂L/∂θ, porque la derivada de θ₀ + δ respecto a δ es
    la identidad. Se comprueba la parte del decaimiento sin dar el paso.
    """
    model = _marked(_model(dtype=torch.float64))
    parent = _snapshot(model)
    anchored = AnchoredAdamW(model.parameters(), lr=lr, weight_decay=decay)
    _moved(model)
    offsets = [
        value.detach() - start for value, start in zip(model.parameters(), parent, strict=True)
    ]
    corrections = [offset.clone().mul_(1 - lr * decay) for offset in offsets]
    for (_, copy), start, correction in zip(
        anchored.decayed_parameters(), parent, corrections, strict=True
    ):
        torch.testing.assert_close(copy, start + correction, rtol=0.0, atol=1e-15)


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


def test_parameters_without_gradient_do_not_decay():
    model = _marked(_model())
    frozen = model[2].bias
    frozen.grad = None
    anchored = AnchoredAdamW(model.parameters(), lr=LARGE[0], weight_decay=LARGE[1])
    _zero_anchors(anchored)
    decayed = {id(value) for value, _ in anchored.decayed_parameters()}
    assert id(frozen) not in decayed and len(decayed) == len(list(model.parameters())) - 1


def test_anchors_survive_a_restricted_checkpoint_and_belong_to_the_parent():
    model = _marked(_model())
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
    model = _marked(_model())
    before = _snapshot(model)
    optimizer = AnchoredAdamW(model.parameters(), lr=LARGE[0], weight_decay=LARGE[1])
    _zero_anchors(optimizer)
    for args, kwargs in (
        ((optimizer, lambda: 0.0), {}),
        ((optimizer,), dict(closure=lambda: 0.0)),
    ):
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


@STEPS
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


@STEPS
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


@STEPS
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
