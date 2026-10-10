"""Paridad de los adaptadores propios con PEFT 0.21.0 sobre los mismos pesos.

Cada caso adapta un padre pequeño con `adapted_copy`, como la etapa de adaptación, y copia
los mismos valores en el adaptador equivalente de PEFT. Solo se ejecutan pasos hacia
delante y `torch.autograd.grad` en CPU, con TF32 desactivado y algoritmos deterministas.
No se construye ningún optimizador y los pesos solo cambian por las copias explícitas de la
preparación.

Tolerancias declaradas antes de ejecutar, elemento a elemento, |a − b| ≤ atol + rtol · |b|:

- FP64: atol = 1e-12 y rtol = 0 en salidas y gradientes.
- FP32: las de la paridad de Titans, (rtol, atol) = (1e-5, 1e-6) en salidas y
  (1e-4, 1e-6) en gradientes, con b la evaluación FP64 de los mismos valores. Además cada
  implementación debe quedar por sí sola dentro de esa tolerancia frente a su propia
  evaluación FP64, de modo que la tolerancia cubre el redondeo de cada una y no una
  diferencia de semántica.

Los valores se sortean en FP64 y se redondean a FP32 antes de usarlos en las dos
precisiones, así el caso FP64 es exactamente la evaluación FP64 del caso FP32.

DoRA tiene una desviación declarada en `WeightDecomposedDelta`: PEFT separa del grafo la
norma de la dirección (sección 4.3 del artículo) y el proyecto no. Las salidas, el
gradiente de la entrada y el de la magnitud coinciden. Los de las matrices de bajo rango
solo coinciden con la norma separada, y la diferencia es la proyección sobre la dirección
de cada fila que se calcula aquí a mano.
"""

import pytest
import torch
from torch import nn

from mars_titan.models.predictive_adaptation import AdapterTarget, adapted_copy
from tests.suite_support import reference_module

pytestmark = pytest.mark.external_reference

TOLERANCES = {
    torch.float64: {"output": (0.0, 1e-12), "gradient": (0.0, 1e-12)},
    torch.float32: {"output": (1e-5, 1e-6), "gradient": (1e-4, 1e-6)},
}
RANK, ALPHA = 4, 8.0
SCALING = ALPHA / RANK


@pytest.fixture(scope="module")
def peft():
    module = reference_module("peft")
    assert module.__version__ == "0.21.0"
    return module


@pytest.fixture(autouse=True)
def strict_arithmetic(monkeypatch):
    """CPU, sin TF32 y con algoritmos deterministas, restaurando el estado al terminar.

    El generador global se bifurca porque las capas de PyTorch y la inicialización de PEFT
    lo consumen al construirse, aunque después se copien valores fijos.
    """
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.use_deterministic_algorithms(previous)


def draw(generator, *shape, scale=1.0):
    """Valores FP64 representables en FP32, para que FP64 evalúe exactamente el caso FP32."""
    values = torch.randn(*shape, generator=generator, dtype=torch.float64) * scale
    return values.float().double()


class Projection(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.lin = nn.Linear(weight.shape[1], weight.shape[0], dtype=weight.dtype)
        with torch.no_grad():
            self.lin.weight.copy_(weight)
            self.lin.bias.copy_(bias)

    def forward(self, x):
        return self.lin(x)


class Recurrent(nn.Module):
    def __init__(self, weights, dtype):
        super().__init__()
        self.gru = nn.GRU(6, 5, batch_first=True, dtype=dtype)
        with torch.no_grad():
            for name, value in weights.items():
                getattr(self.gru, name).copy_(value)

    def forward(self, x):
        return self.gru(x)[0]


def gradients(output, probe, inputs):
    return torch.autograd.grad(output, inputs, grad_outputs=probe)


def linear_values(dtype):
    generator = torch.Generator().manual_seed(11)
    values = dict(
        weight=draw(generator, 8, 16, scale=0.3),
        bias=draw(generator, 8, scale=0.1),
        x=draw(generator, 32, 16),
        probe=draw(generator, 32, 8),
        up=draw(generator, 8, RANK, scale=0.2),
        down=draw(generator, RANK, 16, scale=0.2),
        gain=draw(generator, 8, scale=0.2),
        columns=draw(generator, 16, scale=0.2),
        shift=draw(generator, 8, scale=0.05),
    )
    return {name: value.to(dtype) for name, value in values.items()}


def adapted(values, *targets):
    """Padre adaptado con los destinos declarados, como en la etapa de adaptación."""
    parent = Projection(values["weight"], values["bias"])
    return adapted_copy(parent, targets, seed=0)


def lora_case(peft, dtype):
    values = linear_values(dtype)
    ours = adapted(values, AdapterTarget("lin", "weight", "low_rank", rank=RANK, alpha=ALPHA))
    adapter = ours.lin.parametrizations.weight[0]
    config = peft.LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["lin"], lora_dropout=0.0)
    theirs = peft.get_peft_model(Projection(values["weight"], values["bias"]), config)
    layer = theirs.base_model.model.lin
    with torch.no_grad():
        adapter.up.copy_(values["up"])
        adapter.down.copy_(values["down"])
        layer.lora_B["default"].weight.copy_(values["up"])
        layer.lora_A["default"].weight.copy_(values["down"])
    x = values["x"].clone().requires_grad_(True)
    left, right = ours(x), theirs(x)
    mine = gradients(left, values["probe"], (x, adapter.up, adapter.down))
    other = (layer.lora_B["default"].weight, layer.lora_A["default"].weight)
    other = gradients(right, values["probe"], (x, *other))
    return dict(
        output=(left, right),
        input_gradient=(mine[0], other[0]),
        up_gradient=(mine[1], other[1]),
        down_gradient=(mine[2], other[2]),
    )


def row_block_case(peft, dtype):
    """LoRA sobre un bloque de filas equivale a PEFT con las filas de fuera de B a cero."""
    values = linear_values(dtype)
    rows = (2, 6)
    target = AdapterTarget("lin", "weight", "low_rank", rank=RANK, alpha=ALPHA, rows=rows)
    ours = adapted(values, target)
    adapter = ours.lin.parametrizations.weight[0]
    config = peft.LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["lin"], lora_dropout=0.0)
    theirs = peft.get_peft_model(Projection(values["weight"], values["bias"]), config)
    layer = theirs.base_model.model.lin
    padded = torch.zeros_like(values["up"])
    padded[rows[0] : rows[1]] = values["up"][rows[0] : rows[1]]
    with torch.no_grad():
        adapter.up.copy_(values["up"][rows[0] : rows[1]])
        adapter.down.copy_(values["down"])
        layer.lora_B["default"].weight.copy_(padded)
        layer.lora_A["default"].weight.copy_(values["down"])
    x = values["x"].clone().requires_grad_(True)
    left, right = ours(x), theirs(x)
    mine = gradients(left, values["probe"], (x, adapter.up, adapter.down))
    other = (layer.lora_B["default"].weight, layer.lora_A["default"].weight)
    other = gradients(right, values["probe"], (x, *other))
    return dict(
        output=(left, right),
        input_gradient=(mine[0], other[0]),
        up_gradient=(mine[1], other[1][rows[0] : rows[1]]),
        down_gradient=(mine[2], other[2]),
    )


def ia3_rows_case(peft, dtype):
    """(IA)³ fuera de la FFN escala la salida entera: diag(1 + g) W y (1 + g) b."""
    values = linear_values(dtype)
    ours = adapted(
        values,
        AdapterTarget("lin", "weight", "gain_rows"),
        AdapterTarget("lin", "bias", "gain_rows", shares="weight"),
    )
    adapter = ours.lin.parametrizations.weight[0]
    config = peft.IA3Config(target_modules=["lin"], feedforward_modules=[])
    theirs = peft.get_peft_model(Projection(values["weight"], values["bias"]), config)
    layer = theirs.base_model.model.lin
    scale = layer.ia3_l["default"]
    with torch.no_grad():
        adapter.gain.copy_(values["gain"])
        scale.copy_((1 + values["gain"]).reshape(scale.shape))
    x = values["x"].clone().requires_grad_(True)
    left, right = ours(x), theirs(x)
    mine = gradients(left, values["probe"], (x, adapter.gain))
    other = gradients(right, values["probe"], (x, scale))
    return dict(
        output=(left, right),
        input_gradient=(mine[0], other[0]),
        gain_gradient=(mine[1], other[1].reshape(mine[1].shape)),
    )


def ia3_columns_case(peft, dtype):
    """(IA)³ en la FFN escala la entrada: W diag(1 + g), sin tocar el sesgo."""
    values = linear_values(dtype)
    ours = adapted(values, AdapterTarget("lin", "weight", "gain_columns"))
    adapter = ours.lin.parametrizations.weight[0]
    config = peft.IA3Config(target_modules=["lin"], feedforward_modules=["lin"])
    theirs = peft.get_peft_model(Projection(values["weight"], values["bias"]), config)
    layer = theirs.base_model.model.lin
    scale = layer.ia3_l["default"]
    with torch.no_grad():
        adapter.gain.copy_(values["columns"])
        scale.copy_((1 + values["columns"]).reshape(scale.shape))
    x = values["x"].clone().requires_grad_(True)
    left, right = ours(x), theirs(x)
    mine = gradients(left, values["probe"], (x, adapter.gain))
    other = gradients(right, values["probe"], (x, scale))
    return dict(
        output=(left, right),
        input_gradient=(mine[0], other[0]),
        gain_gradient=(mine[1], other[1].reshape(mine[1].shape)),
    )


def dora_removed_component(values, gradient_of_weight):
    """Parte del gradiente de V que PEFT descarta al separar ||V|| del grafo.

    Con W' = c V / ||V|| por filas y G = dL/dW', el gradiente completo respecto a V es
    (c / ||V||) (G − (G · V̂) V̂), con V̂ = V / ||V||. Con la norma separada queda
    (c / ||V||) G. La diferencia es −(c / ||V||) (G · V̂) V̂.
    """
    direction = values["weight"] + SCALING * values["up"] @ values["down"]
    norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    unit = direction / norm
    magnitude = torch.linalg.vector_norm(values["weight"], dim=1, keepdim=True) + values[
        "shift"
    ].unsqueeze(1)
    return -(magnitude / norm) * (gradient_of_weight * unit).sum(1, keepdim=True) * unit


def dora_parts(peft, dtype):
    values = linear_values(dtype)
    ours = adapted(values, AdapterTarget("lin", "weight", "dora", rank=RANK, alpha=ALPHA))
    adapter = ours.lin.parametrizations.weight[0]
    config = peft.LoraConfig(
        r=RANK, lora_alpha=ALPHA, target_modules=["lin"], lora_dropout=0.0, use_dora=True
    )
    theirs = peft.get_peft_model(Projection(values["weight"], values["bias"]), config)
    layer = theirs.base_model.model.lin
    magnitude = layer.lora_magnitude_vector["default"].weight
    with torch.no_grad():
        adapter.up.copy_(values["up"])
        adapter.down.copy_(values["down"])
        adapter.magnitude.copy_(values["shift"])
        layer.lora_B["default"].weight.copy_(values["up"])
        layer.lora_A["default"].weight.copy_(values["down"])
        # PEFT guarda la magnitud completa, el proyecto el desplazamiento sobre ||W||.
        magnitude.copy_(torch.linalg.vector_norm(values["weight"], dim=1) + values["shift"])
    x = values["x"].clone().requires_grad_(True)
    left, right = ours(x), theirs(x)
    probe = values["probe"]
    mine = gradients(left, probe, (x, adapter.magnitude, adapter.up, adapter.down))
    lora = (layer.lora_B["default"].weight, layer.lora_A["default"].weight)
    other = gradients(right, probe, (x, magnitude, *lora))
    return values, (left, right), mine, other


def dora_case(peft, dtype):
    values, outputs, mine, other = dora_parts(peft, dtype)
    # G = dL/dW' para y = x W'^T + b con dL/dy = probe.
    removed = dora_removed_component(values, values["probe"].T @ values["x"])
    return dict(
        output=outputs,
        input_gradient=(mine[0], other[0]),
        magnitude_gradient=(mine[1], other[1]),
        up_gradient=(mine[2], other[2] + SCALING * removed @ values["down"].T),
        down_gradient=(mine[3], other[3] + SCALING * values["up"].T @ removed),
    )


def gru_values(dtype):
    generator = torch.Generator().manual_seed(5)
    values = dict(
        weight_ih_l0=draw(generator, 15, 6, scale=0.3),
        weight_hh_l0=draw(generator, 15, 5, scale=0.3),
        bias_ih_l0=draw(generator, 15, scale=0.1),
        bias_hh_l0=draw(generator, 15, scale=0.1),
        x=draw(generator, 4, 7, 6),
        probe=draw(generator, 4, 7, 5),
        up_ih=draw(generator, 15, 2, scale=0.2),
        down_ih=draw(generator, 2, 6, scale=0.2),
        up_hh=draw(generator, 15, 2, scale=0.2),
        down_hh=draw(generator, 2, 5, scale=0.2),
    )
    return {name: value.to(dtype) for name, value in values.items()}


def gru_case(peft, dtype):
    """LoRA sobre weight_ih_l0 y weight_hh_l0, que PEFT solo cubre con target_parameters."""
    values = gru_values(dtype)
    weights = {name: values[name] for name in values if name.endswith("_l0")}
    tensors = ("weight_ih_l0", "weight_hh_l0")
    targets = [AdapterTarget("gru", name, "low_rank", rank=2, alpha=4.0) for name in tensors]
    ours = adapted_copy(Recurrent(weights, dtype), targets, seed=0)
    config = peft.LoraConfig(
        r=2, lora_alpha=4.0, target_modules=[], target_parameters=[f"gru.{n}" for n in tensors]
    )
    theirs = peft.get_peft_model(Recurrent(weights, dtype), config)
    wrappers = {
        module.parameter_name: module
        for module in theirs.modules()
        if isinstance(module, peft.tuners.lora.ParamWrapper)
    }
    assert sorted(wrappers) == sorted(tensors)
    mine, other = [], []
    with torch.no_grad():
        for name, key in zip(tensors, ("ih", "hh"), strict=True):
            adapter = ours.gru.parametrizations[name][0]
            adapter.up.copy_(values[f"up_{key}"])
            adapter.down.copy_(values[f"down_{key}"])
            wrappers[name].lora_B["default"].weight.copy_(values[f"up_{key}"])
            wrappers[name].lora_A["default"].weight.copy_(values[f"down_{key}"])
            mine += [adapter.up, adapter.down]
            other += [wrappers[name].lora_B["default"].weight]
            other += [wrappers[name].lora_A["default"].weight]
    x = values["x"].clone().requires_grad_(True)
    left, right = ours(x), theirs(x)
    mine = gradients(left, values["probe"], (x, *mine))
    other = gradients(right, values["probe"], (x, *other))
    names = ("input", "up_ih", "down_ih", "up_hh", "down_hh")
    result = dict(output=(left, right))
    for name, ours_gradient, their_gradient in zip(names, mine, other, strict=True):
        result[f"{name}_gradient"] = (ours_gradient, their_gradient)
    return result


CASES = {
    "lora": lora_case,
    "lora_row_block": row_block_case,
    "ia3_rows": ia3_rows_case,
    "ia3_columns": ia3_columns_case,
    "dora": dora_case,
    "lora_gru": gru_case,
}


def check(actual, expected, dtype, kind, label, scale=None):
    """|actual − expected| ≤ atol + rtol · |scale| elemento a elemento, con scale = expected
    salvo que se indique la evaluación FP64."""
    rtol, atol = TOLERANCES[dtype][kind]
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, label
    difference = (actual.double() - expected.double()).abs()
    bound = atol + rtol * (expected if scale is None else scale).double().abs()
    assert bool((difference <= bound).all()), (
        f"{label}: máximo {difference.max().item():.3e}, peor cociente "
        f"{(difference / bound).max().item():.3f}"
    )


def kind(name):
    return "output" if name == "output" else "gradient"


@pytest.mark.parametrize("case", CASES)
def test_fp64_outputs_and_gradients_match_peft(peft, case):
    for name, (ours, theirs) in CASES[case](peft, torch.float64).items():
        check(ours, theirs, torch.float64, kind(name), f"{case}.{name}")


@pytest.mark.parametrize("case", CASES)
def test_fp32_differences_stay_within_the_rounding_of_each_implementation(peft, case):
    reference = CASES[case](peft, torch.float64)
    for name, (ours, theirs) in CASES[case](peft, torch.float32).items():
        expected, other = reference[name]
        label = f"{case}.{name}"
        check(ours, theirs, torch.float32, kind(name), label, scale=expected)
        # El mismo margen basta para que cada implementación FP32 siga a su evaluación FP64.
        check(ours.double(), expected, torch.float32, kind(name), f"{label} propio")
        check(theirs.double(), other, torch.float32, kind(name), f"{label} PEFT")


def test_peft_dora_gradients_are_ours_with_the_direction_norm_detached(peft):
    """Las ecuaciones del proyecto con ||V|| separada dan exactamente los gradientes de PEFT."""
    values, _, _, other = dora_parts(peft, torch.float64)
    up = values["up"].clone().requires_grad_(True)
    down = values["down"].clone().requires_grad_(True)
    direction = values["weight"] + SCALING * up @ down
    reference = torch.linalg.vector_norm(values["weight"], dim=1)
    scale = (reference + values["shift"]) / torch.linalg.vector_norm(direction, dim=1).detach()
    output = values["x"] @ (direction * scale.unsqueeze(1)).T + values["bias"]
    detached = gradients(output, values["probe"], (up, down))
    check(detached[0], other[2], torch.float64, "gradient", "dora.up_detached")
    check(detached[1], other[3], torch.float64, "gradient", "dora.down_detached")


def test_the_dora_deviation_is_not_negligible(peft):
    """La prueba distingue las dos versiones: separar la norma cambia de verdad el gradiente."""
    values, _, mine, other = dora_parts(peft, torch.float64)
    for ours, theirs in zip(mine[2:], other[2:], strict=True):
        relative = (ours - theirs).norm() / theirs.norm()
        assert relative > 1e-2
