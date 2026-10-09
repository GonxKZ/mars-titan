"""Adaptador residual común, adaptadores sobre un padre congelado y objetivos predictivos."""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

ADAPTER_FORMS = ("residual", "low_rank")


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


def _add_rows(original, delta, rows):
    start, stop = rows
    if (start, stop) == (0, len(original)):
        return original + delta
    # El resto de filas se copia sin operar, así conserva exactamente sus valores.
    return torch.cat((original[:start], original[start:stop] + delta, original[stop:]))


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
        if (
            type(rank) is not int
            or not 1 <= rank <= min(64, *shape)
            or type(alpha) not in (int, float)
            or not math.isfinite(alpha)
            or alpha <= 0
            or not isinstance(generator, torch.Generator)
            or generator.device.type != "cpu"
        ):
            raise ValueError("El rango, la escala o el generador del adaptador no son válidos")
        self.rows = _row_block(rows, shape[0])
        self.rank, self.scaling = rank, float(alpha) / rank
        down = torch.empty(rank, shape[1], dtype=dtype)
        bound = 1 / math.sqrt(shape[1])
        down.uniform_(-bound, bound, generator=generator)
        self.down = nn.Parameter(down)
        self.up = nn.Parameter(torch.zeros(self.rows[1] - self.rows[0], rank, dtype=dtype))

    def forward(self, original):
        return _add_rows(original, (self.up @ self.down) * self.scaling, self.rows)


@dataclass(frozen=True)
class AdapterTarget:
    """Tensor de un padre que recibe una corrección, sin modificar su valor original."""

    module: str
    tensor: str
    form: str
    rank: int | None = None
    alpha: float | None = None
    rows: tuple[int, int] | None = None

    def __post_init__(self):
        if (
            not isinstance(self.module, str)
            or not isinstance(self.tensor, str)
            or not self.tensor.isidentifier()
            or self.form not in ADAPTER_FORMS
            or (self.form == "low_rank") != (self.rank is not None and self.alpha is not None)
            or (self.form == "residual" and (self.rank, self.alpha) != (None, None))
            or (self.rows is not None and (not isinstance(self.rows, tuple) or len(self.rows) != 2))
        ):
            raise ValueError("El destino del adaptador no está bien declarado")

    def trainable_parameters(self, shape):
        start, stop = _row_block(self.rows, shape[0])
        rows = stop - start
        if self.form == "residual":
            return rows * math.prod(shape[1:])
        return self.rank * (rows + shape[1])

    def identity(self, shape):
        return dict(
            module=self.module,
            tensor=self.tensor,
            shape=list(shape),
            form=self.form,
            rank=self.rank,
            alpha=self.alpha,
            rows=list(_row_block(self.rows, shape[0])),
            trainable_parameters=self.trainable_parameters(shape),
        )


def adapted_copy(model, targets, *, seed):
    """Copiar el padre, congelar todos sus pesos y añadir correcciones nulas declaradas.

    El padre recibido no cambia. Solo los parámetros de los adaptadores requieren
    gradiente. Un destino repetido o inexistente se rechaza antes de copiar.
    """
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
    for target in targets:
        module = model.get_submodule(target.module)
        value = getattr(module, target.tensor, None)
        if not isinstance(value, nn.Parameter) or parametrize.is_parametrized(
            module, target.tensor
        ):
            raise ValueError("El destino del adaptador no es un parámetro original del padre")
    result = copy.deepcopy(model).requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for target in targets:
        module = result.get_submodule(target.module)
        value = getattr(module, target.tensor)
        if target.form == "residual":
            adapter = ResidualDelta(value.shape, rows=target.rows, dtype=value.dtype)
        else:
            adapter = LowRankDelta(
                value.shape,
                target.rank,
                alpha=target.alpha,
                rows=target.rows,
                generator=generator,
                dtype=value.dtype,
            )
        parametrize.register_parametrization(
            module, target.tensor, adapter.to(device=value.device), unsafe=False
        )
    return result


def trainable_parameters(model):
    return sum(value.numel() for value in model.parameters() if value.requires_grad)
