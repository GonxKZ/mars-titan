"""Adaptador residual común, adaptadores sobre un padre congelado y objetivos predictivos."""

import copy
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.nn.utils import parametrize

# Formas por tensor (parametrizaciones) y por módulo (rama añadida a su salida).
# `docs/engineering/adapter-variety.md` relaciona cada una con su ecuación y su fuente.
TENSOR_FORMS = ("residual", "low_rank", "dora", "gain_rows", "gain_columns")
MODULE_FORMS = ("parallel_adapter", "serial_adapter")
ADAPTER_FORMS = TENSOR_FORMS + MODULE_FORMS
# Formas que declaran rango r y escala alpha. Las demás no tienen hiperparámetros propios.
RANKED_FORMS = ("low_rank", "dora", *MODULE_FORMS)
# Raíz de los módulos de adaptación por módulo. Sus parámetros son correcciones.
MODULE_ROOT = "posttraining_modules"


def _policy_inputs(centers, values, scale):
    if (
        not isinstance(centers, torch.Tensor)
        or centers.ndim != 1
        or not 1 <= len(centers) <= 4096
        or not centers.is_floating_point()
        or not isinstance(values, torch.Tensor)
        or values.shape != (21,)
        or not values.is_floating_point()
        or values.device != centers.device
        or type(scale) not in (int, float)
        or not math.isfinite(scale)
        or scale <= 0
        or not torch.isfinite(centers).all()
        or not torch.isfinite(values).all()
        or not torch.all(torch.diff(values) > 0)
    ):
        raise ValueError("Los centros, acciones o escala de la política no son válidos")
    return centers.double(), values.double()


def gaussian_log_probabilities(centers, values, scale):
    """Calcular logits gaussianos sin el término común mu², con anchura fija."""
    centers, values = _policy_inputs(centers, values, scale)
    z, m = values / scale, centers / scale
    logits = z[None, :] * m[:, None] - 0.5 * z.square()[None, :]
    result = torch.log_softmax(logits, dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("Los logits han desbordado la representación float64")
    return result


def objective(centers, targets, values, scale, mode, *, actions=None, generator=None):
    """Devolver pérdidas por fila. REINFORCE minimiza costes, no su negativo."""
    centers, values = _policy_inputs(centers, values, scale)
    if (
        mode not in {"reinforce", "expected", "mae"}
        or not isinstance(targets, torch.Tensor)
        or targets.shape != centers.shape
        or targets.device != centers.device
        or not targets.is_floating_point()
        or targets.requires_grad
        or not torch.isfinite(targets).all()
        or (mode != "reinforce" and actions is not None)
    ):
        raise ValueError("La modalidad de ajuste o las etiquetas no son válidas")
    targets = targets.double()
    if mode == "mae":
        loss = (centers - targets).abs() / scale
    else:
        log_probability = gaussian_log_probabilities(centers, values, scale)
        probability = log_probability.exp()
        costs = (values[None, :] - targets[:, None]).abs() / scale
        expected = (probability * costs).sum(dim=1)
        if mode == "expected":
            loss = expected
        else:
            if actions is None:
                actions = torch.multinomial(probability, 1, generator=generator).squeeze(1)
            if (
                not isinstance(actions, torch.Tensor)
                or actions.shape != centers.shape
                or actions.device != centers.device
                or actions.dtype not in {torch.int32, torch.int64}
                or (actions < 0).any()
                or (actions >= 21).any()
            ):
                raise ValueError("Las acciones muestreadas no pertenecen a la rejilla")
            selected = actions.long()[:, None]
            advantage = costs.gather(1, selected).squeeze(1).detach() - expected.detach()
            loss = advantage * log_probability.gather(1, selected).squeeze(1)
    if not torch.isfinite(loss).all():
        raise ValueError("La pérdida de adaptación no es finita")
    return loss, actions


class LinearResidualPolicy(nn.Module):
    """Ajustar una corrección común sin modificar las predicciones del padre almacenadas."""

    def __init__(self, mean, scale, *, target_scale):
        super().__init__()
        mean, scale = np.asarray(mean), np.asarray(scale)
        if (
            mean.ndim != 1
            or not 1 <= len(mean) <= 16384
            or scale.shape != mean.shape
            or np.iscomplexobj(mean)
            or np.iscomplexobj(scale)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
            or type(target_scale) not in (int, float)
            or not math.isfinite(target_scale)
            or target_scale <= 0
        ):
            raise ValueError("La normalización y la escala de retorno no son válidas")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float64))
        self.target_scale = float(target_scale)
        self.correction = nn.Linear(len(mean), 1, dtype=torch.float32)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, features, parent):
        if (
            features.ndim != 2
            or features.shape[1] != len(self.mean)
            or not 1 <= len(features) <= 4096
            or parent.shape != (len(features),)
            or features.device != self.mean.device
            or parent.device != self.mean.device
            or not features.is_floating_point()
            or not parent.is_floating_point()
            or not torch.isfinite(features).all()
            or not torch.isfinite(parent).all()
        ):
            raise ValueError("Las entradas y el padre no forman un lote válido del adaptador")
        standardized = ((features.detach().double() - self.mean) / self.scale).float()
        if not torch.isfinite(standardized).all():
            raise ValueError("La estandarización del adaptador ha desbordado float32")
        result = (
            parent.detach().double()
            + self.target_scale * self.correction(standardized).squeeze(1).double()
        )
        if not torch.isfinite(result).all():
            raise ValueError("El centro predictivo del adaptador no es finito")
        return result


def _row_block(rows, size):
    start, stop = (0, size) if rows is None else rows
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= size:
        raise ValueError("El bloque de filas del adaptador no pertenece al tensor")
    return start, stop


def _replace_rows(original, block, rows):
    start, stop = rows
    if (start, stop) == (0, len(original)):
        return block
    # El resto de filas se copia sin operar, así conserva exactamente sus valores.
    return torch.cat((original[:start], block, original[stop:]))


def _add_rows(original, delta, rows):
    start, stop = rows
    return _replace_rows(original, original[start:stop] + delta, rows)


def _ranked(rank, alpha, generator, limit):
    if (
        type(rank) is not int
        or not 1 <= rank <= min(64, limit)
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 0
        or not isinstance(generator, torch.Generator)
        or generator.device.type != "cpu"
    ):
        raise ValueError("El rango, la escala o el generador del adaptador no son válidos")


def _down(rank, columns, generator, dtype):
    """Proyección de bajada como una capa lineal de PyTorch, con un generador propio."""
    down = torch.empty(rank, columns, dtype=dtype)
    bound = 1 / math.sqrt(columns)
    down.uniform_(-bound, bound, generator=generator)
    return nn.Parameter(down)


class ResidualDelta(nn.Module):
    """Sumar a un bloque de filas una corrección completa que empieza en cero."""

    def __init__(self, shape, *, rows=None, dtype=torch.float32):
        super().__init__()
        shape = tuple(shape)
        if not 1 <= len(shape) <= 2 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("La corrección residual necesita un vector o una matriz")
        self.rows = _row_block(rows, shape[0])
        rows_count = self.rows[1] - self.rows[0]
        self.delta = nn.Parameter(torch.zeros((rows_count, *shape[1:]), dtype=dtype))

    def forward(self, original):
        return _add_rows(original, self.delta, self.rows)


class LowRankDelta(nn.Module):
    """Sumar (alpha / r) U V a un bloque de filas de W, con U inicializada a cero.

    Es la factorización de LoRA (Hu et al., 2022). V se inicializa como una capa
    lineal de PyTorch con un generador propio, de modo que no consume el RNG global.
    Con U nula la matriz efectiva es exactamente W y el gradiente de V es cero en
    el primer paso. Los parámetros entrenables son r (filas + columnas).
    """

    def __init__(self, shape, rank, *, alpha, rows=None, generator, dtype=torch.float32):
        super().__init__()
        shape = tuple(shape)
        if len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("El adaptador de bajo rango necesita una matriz")
        _ranked(rank, alpha, generator, min(shape))
        self.rows = _row_block(rows, shape[0])
        self.rank, self.scaling = rank, float(alpha) / rank
        self.down = _down(rank, shape[1], generator, dtype)
        self.up = nn.Parameter(torch.zeros(self.rows[1] - self.rows[0], rank, dtype=dtype))

    def forward(self, original):
        return _add_rows(original, (self.up @ self.down) * self.scaling, self.rows)


class WeightDecomposedDelta(nn.Module):
    """DoRA (Liu et al., 2024): magnitud por fila y dirección con una corrección de bajo rango.

    Para el bloque de filas W_b del tensor, con V = W_b + (alpha / r) U D:

        W'_b = ((||W_b||_f + m) / ||V||_f) ⊙ V

    donde ||·||_f es la norma euclídea de cada fila (una por unidad de salida, la misma
    que usan el artículo y su código para una capa `nn.Linear`). La magnitud del artículo
    es ||W_b||_f + m: se parametriza como desplazamiento m, nulo al inicio, sobre la norma
    del padre. La familia de funciones y el gradiente son los mismos. Solo cambia hacia
    dónde empuja el decaimiento de AdamW, que aquí lleva al padre como en LoRA. Con U nula,
    V y W_b tienen los mismos bits, las dos normas salen del mismo núcleo y el cociente es
    exactamente 1, así que W' = W bit a bit. La norma de V no se separa del grafo, a
    diferencia de la sección 4.3 del artículo: con estas matrices la memoria no lo exige
    y el gradiente es el de la función declarada.
    """

    def __init__(self, shape, rank, *, alpha, rows=None, generator, dtype=torch.float32):
        super().__init__()
        shape = tuple(shape)
        if len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("DoRA necesita una matriz")
        _ranked(rank, alpha, generator, min(shape))
        self.rows = _row_block(rows, shape[0])
        self.rank, self.scaling = rank, float(alpha) / rank
        count = self.rows[1] - self.rows[0]
        self.down = _down(rank, shape[1], generator, dtype)
        self.up = nn.Parameter(torch.zeros(count, rank, dtype=dtype))
        self.magnitude = nn.Parameter(torch.zeros(count, dtype=dtype))

    def forward(self, original):
        start, stop = self.rows
        block = original[start:stop]
        direction = block + (self.up @ self.down) * self.scaling
        reference = torch.linalg.vector_norm(block, dim=1)
        norm = torch.linalg.vector_norm(direction, dim=1)
        scale = (reference + self.magnitude) / norm
        return _replace_rows(original, direction * scale.unsqueeze(1), self.rows)


class RowGain(nn.Module):
    """Multiplicar un bloque de filas (o de entradas de un vector) por 1 + g, con g nulo.

    Es la reescala de (IA)³ (Liu et al., 2022) sobre la salida de una proyección:
    diag(1 + g) W, y el mismo factor sobre su sesgo cuando otro tensor lo comparte con
    `SharedRowGain`. Con g nulo el factor vale exactamente 1.
    """

    def __init__(self, shape, *, rows=None, dtype=torch.float32):
        super().__init__()
        shape = tuple(shape)
        if not 1 <= len(shape) <= 2 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("La ganancia por filas necesita un vector o una matriz")
        self.rows = _row_block(rows, shape[0])
        self.gain = nn.Parameter(torch.zeros(self.rows[1] - self.rows[0], dtype=dtype))

    def scale(self, original):
        start, stop = self.rows
        factor = (1 + self.gain).reshape(-1, *([1] * (original.ndim - 1)))
        return _replace_rows(original, original[start:stop] * factor, self.rows)

    def forward(self, original):
        return self.scale(original)


class SharedRowGain(nn.Module):
    """Aplicar a otro tensor la ganancia de un `RowGain`, sin un parámetro propio.

    La referencia no se registra como submódulo: el parámetro vive en un solo sitio y
    `copy.deepcopy` la conserva emparejada con la copia de su fuente.
    """

    def __init__(self, source):
        super().__init__()
        if not isinstance(source, RowGain):
            raise ValueError("La ganancia compartida necesita una ganancia por filas")
        object.__setattr__(self, "_source", source)

    def forward(self, original):
        return self._source.scale(original)


class ColumnGain(nn.Module):
    """Multiplicar las columnas de W por 1 + g: W diag(1 + g), con g nulo al inicio.

    Equivale a reescalar las activaciones que entran en la capa, como el vector de (IA)³
    sobre la activación intermedia de la FFN o el de los valores antes de `out_proj`.
    """

    def __init__(self, shape, *, dtype=torch.float32):
        super().__init__()
        shape = tuple(shape)
        if len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("La ganancia por columnas necesita una matriz")
        self.gain = nn.Parameter(torch.zeros(shape[1], dtype=dtype))

    def forward(self, original):
        return original * (1 + self.gain)


PLACEMENTS = {"parallel_adapter": "parallel", "serial_adapter": "serial"}


class BottleneckAdapter(nn.Module):
    """Rama s · up(ReLU(down(x) + b_d)) + s · b_u sumada a la salida de un bloque.

    En paralelo (He et al., 2022) x es la entrada del bloque y en serie (Houlsby et al.,
    2019) su salida. La proyección de subida y su sesgo empiezan en cero, así que la
    rama vale exactamente cero y el bloque devuelve los mismos valores que el padre. Como
    en LoRA, s = alpha / r y la bajada se inicializa como una capa lineal con un
    generador propio. El gancho es un método del propio adaptador: `copy.deepcopy` lo
    vuelve a enlazar con la copia.
    """

    def __init__(self, shape, rank, *, alpha, form, generator, dtype=torch.float32):
        super().__init__()
        shape = tuple(shape)
        if (
            form not in PLACEMENTS
            or len(shape) != 2
            or any(type(size) is not int or size < 1 for size in shape)
        ):
            raise ValueError("El adaptador en cuello de botella necesita salida y entrada")
        _ranked(rank, alpha, generator, min(shape))
        self.placement, self.rank, self.scaling = PLACEMENTS[form], rank, float(alpha) / rank
        self.down = _down(rank, shape[1], generator, dtype)
        self.down_bias = nn.Parameter(torch.zeros(rank, dtype=dtype))
        self.up = nn.Parameter(torch.zeros(shape[0], rank, dtype=dtype))
        self.up_bias = nn.Parameter(torch.zeros(shape[0], dtype=dtype))

    def branch(self, value):
        hidden = functional.relu(functional.linear(value, self.down, self.down_bias))
        return functional.linear(hidden, self.up, self.up_bias) * self.scaling

    def hook(self, module, inputs, output):
        source = inputs[0] if self.placement == "parallel" else output
        return output + self.branch(source)


def _linear_block(module):
    layers = (
        [layer for layer in module if isinstance(layer, nn.Linear)]
        if isinstance(module, nn.Sequential)
        else []
    )
    if not layers:
        raise ValueError("El adaptador por módulo necesita un bloque secuencial con capas lineales")
    return layers


def module_shape(module, form):
    """(salida, entrada) de la rama: entrada del bloque en paralelo y su salida en serie."""
    layers = _linear_block(module)
    output = layers[-1].out_features
    return (output, layers[0].in_features if PLACEMENTS[form] == "parallel" else output)


@dataclass(frozen=True)
class AdapterTarget:
    """Tensor o bloque de un padre que recibe una corrección, sin modificar su valor original.

    Las formas por módulo usan `tensor="output"`: la rama se suma a la salida del bloque.
    `shares` nombra otro tensor del mismo módulo cuya ganancia por filas se reutiliza.
    """

    module: str
    tensor: str
    form: str
    rank: int | None = None
    alpha: float | None = None
    rows: tuple[int, int] | None = None
    shares: str | None = None

    def __post_init__(self):
        ranked = self.form in RANKED_FORMS
        if (
            not isinstance(self.module, str)
            or not isinstance(self.tensor, str)
            or not self.tensor.isidentifier()
            or self.form not in ADAPTER_FORMS
            or (self.form in MODULE_FORMS) != (self.tensor == "output")
            or ranked != (self.rank is not None and self.alpha is not None)
            or (not ranked and (self.rank, self.alpha) != (None, None))
            or (self.rows is not None and (not isinstance(self.rows, tuple) or len(self.rows) != 2))
            or (
                self.shares is not None
                and (
                    self.form != "gain_rows"
                    or not isinstance(self.shares, str)
                    or not self.shares.isidentifier()
                    or self.shares == self.tensor
                )
            )
        ):
            raise ValueError("El destino del adaptador no está bien declarado")

    def trainable_parameters(self, shape):
        start, stop = _row_block(self.rows, shape[0])
        rows = stop - start
        if self.form == "residual":
            return rows * math.prod(shape[1:])
        if self.form == "gain_rows":
            return 0 if self.shares is not None else rows
        if self.form == "gain_columns":
            return shape[1]
        if self.form in MODULE_FORMS:
            return self.rank * (shape[0] + shape[1]) + self.rank + shape[0]
        return self.rank * (rows + shape[1]) + (rows if self.form == "dora" else 0)

    def identity(self, shape):
        result = dict(
            module=self.module,
            tensor=self.tensor,
            shape=list(shape),
            form=self.form,
            rank=self.rank,
            alpha=self.alpha,
            rows=list(_row_block(self.rows, shape[0])),
            trainable_parameters=self.trainable_parameters(shape),
        )
        # Sin ganancia compartida la identidad conserva literalmente su forma anterior.
        if self.shares is not None:
            result["shares"] = self.shares
        return result


def target_shape(model, target):
    """Forma del tensor adaptado o, en las formas por módulo, (salida, entrada) de la rama."""
    module = model.get_submodule(target.module)
    if target.form in MODULE_FORMS:
        return module_shape(module, target.form)
    return tuple(getattr(module, target.tensor).shape)


def _checked_targets(model, targets, seed):
    targets = tuple(targets)
    if (
        not isinstance(model, nn.Module)
        or not targets
        or any(not isinstance(target, AdapterTarget) for target in targets)
        or len({(t.module, t.tensor) for t in targets}) != len(targets)
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("Los destinos del adaptador o su semilla no son válidos")
    hooked = getattr(model, MODULE_ROOT, None)
    declared = {}
    for target in targets:
        module = model.get_submodule(target.module)
        if target.form in MODULE_FORMS:
            shape = module_shape(module, target.form)
            if (
                target.rows not in (None, (0, shape[0]))
                or (hooked is not None and _module_key(target.module) in hooked)
                or module is model
            ):
                raise ValueError("El bloque del adaptador por módulo no es válido o ya tiene uno")
            continue
        value = getattr(module, target.tensor, None)
        if not isinstance(value, nn.Parameter) or parametrize.is_parametrized(
            module, target.tensor
        ):
            raise ValueError("El destino del adaptador no es un parámetro original del padre")
        rows = _row_block(target.rows, value.shape[0])
        if target.form == "gain_columns" and rows != (0, value.shape[0]):
            raise ValueError("La ganancia por columnas se aplica a todas las filas del tensor")
        if target.form == "dora" and not bool(
            (torch.linalg.vector_norm(value.detach()[rows[0] : rows[1]], dim=1) > 0).all()
        ):
            raise ValueError("DoRA no admite filas nulas en el tensor del padre")
        if target.shares is not None:
            source = declared.get((target.module, target.shares))
            if (
                source is None
                or source.form != "gain_rows"
                or source.shares is not None
                or _row_block(source.rows, getattr(module, target.shares).shape[0])
                != _row_block(target.rows, value.shape[0])
                or value.ndim != 1
            ):
                raise ValueError("La ganancia compartida no corresponde a un destino anterior")
        declared[(target.module, target.tensor)] = target
    return targets


def _module_key(name):
    return name.replace(".", "__")


def adapted_copy(model, targets, *, seed):
    """Copiar el padre, congelar todos sus pesos y añadir correcciones nulas declaradas.

    El padre recibido no cambia. Solo los parámetros de los adaptadores requieren
    gradiente. Un destino repetido o inexistente se rechaza antes de copiar.
    """
    targets = _checked_targets(model, targets, seed)
    return _attach(copy.deepcopy(model), targets, seed)


def attach_adapters(model, targets, *, seed):
    """Congelar un modelo propio y añadirle en su sitio las correcciones nulas declaradas.

    Es `adapted_copy` sin la copia, para un padre que se acaba de reconstruir desde su
    checkpoint y que nadie más usa. Un módulo que sella sus parámetros debe volver a
    sellarlos después, porque las parametrizaciones cambian sus nombres y tensores.
    """
    return _attach(model, _checked_targets(model, targets, seed), seed)


def _tensor_adapter(target, value, generator):
    if target.form == "residual":
        return ResidualDelta(value.shape, rows=target.rows, dtype=value.dtype)
    if target.form in ("low_rank", "dora"):
        kind = LowRankDelta if target.form == "low_rank" else WeightDecomposedDelta
        return kind(
            value.shape,
            target.rank,
            alpha=target.alpha,
            rows=target.rows,
            generator=generator,
            dtype=value.dtype,
        )
    if target.form == "gain_rows":
        return RowGain(value.shape, rows=target.rows, dtype=value.dtype)
    return ColumnGain(value.shape, dtype=value.dtype)


def _attach(result, targets, seed):
    result.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for target in targets:
        module = result.get_submodule(target.module)
        if target.form in MODULE_FORMS:
            layers = _linear_block(module)
            adapter = BottleneckAdapter(
                module_shape(module, target.form),
                target.rank,
                alpha=target.alpha,
                form=target.form,
                generator=generator,
                dtype=layers[0].weight.dtype,
            ).to(device=layers[0].weight.device)
            if not hasattr(result, MODULE_ROOT):
                result.add_module(MODULE_ROOT, nn.ModuleDict())
            getattr(result, MODULE_ROOT)[_module_key(target.module)] = adapter
            module.register_forward_hook(adapter.hook)
            continue
        value = getattr(module, target.tensor)
        if target.shares is not None:
            adapter = SharedRowGain(module.parametrizations[target.shares][0])
        else:
            adapter = _tensor_adapter(target, value, generator)
        parametrize.register_parametrization(
            module, target.tensor, adapter.to(device=value.device), unsafe=False
        )
    return result


def trainable_parameters(model):
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def _parametrized(name):
    return name.startswith("parametrizations.") or ".parametrizations." in name


def is_adapter_name(name):
    """Nombre de un parámetro de corrección: parametrización o módulo de adaptación."""
    return (_parametrized(name) and not name.endswith(".original")) or name.startswith(
        MODULE_ROOT + "."
    )


def adapter_names(model):
    """Nombres de los parámetros de corrección, en el orden de `named_parameters`."""
    return [name for name, _ in model.named_parameters() if is_adapter_name(name)]


def base_digest(model):
    """Huella de los tensores originales del padre, sin las correcciones de los adaptadores.

    Comprueba que el ajuste solo cambia los adaptadores. Una parametrización conserva el
    tensor original con el sufijo `.original`, que aquí recupera su nombre del padre.
    """
    adapters = set(adapter_names(model))
    # Una parametrización mueve el tensor al final del orden de su módulo: se ordena por nombre.
    values = {
        ("." + name).replace(".parametrizations.", ".").removesuffix(".original")[1:]: value
        for name, value in model.named_parameters()
        if name not in adapters
    }
    digest = hashlib.sha256()
    for name in sorted(values):
        array = values[name].detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(array.shape)}:{array.dtype}".encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()
