"""Ridge estandarizado por bloques con intercepto no penalizado y cálculo CUDA."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mars_titan.data.embeddings import require_cuda
from mars_titan.training.learning_hold import require_learning_allowed

from .inputs import validated_blocks

# Tipos admitidos para viajar a la GPU sin conversión previa en la CPU.
_FLOATS = (np.float32, np.float64)


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def predict(self, x) -> np.ndarray:
        """Estandarizar en el dispositivo. Un bloque float32 viaja sin convertirse en la CPU.

        La conversión a float64 es exacta y la resta y la división son IEEE, así que el
        resultado coincide bit a bit con la estandarización de numpy en float64.
        """
        import torch

        x = np.asarray(x)
        if x.dtype.type not in _FLOATS:
            x = x.astype(np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Las entradas de predicción no son válidas")
        device = "cuda:0"
        with torch.inference_mode():
            values = torch.as_tensor(np.ascontiguousarray(x), device=device).to(torch.float64)
            value = (values - torch.as_tensor(self.mean, device=device)) / torch.as_tensor(
                self.scale, device=device
            )
            result = value @ torch.as_tensor(self.coefficient, device=device) + self.intercept
        return result.cpu().numpy()

    def save(self, path: Path) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(
                    stream,
                    mean=self.mean,
                    scale=self.scale,
                    coefficient=self.coefficient,
                    intercept=self.intercept,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as state:
            values = [state[name].copy() for name in ("mean", "scale", "coefficient")]
            intercept = float(state["intercept"])
        if (
            any(value.ndim != 1 or not np.isfinite(value).all() for value in values)
            or not 1 <= len(values[0]) <= 4096
            or any(value.shape != values[0].shape for value in values)
            or (values[1] <= 0).any()
            or not np.isfinite(intercept)
        ):
            raise ValueError("El modelo Ridge guardado no es válido")
        return cls(*values, intercept)


class _PinnedStaging:
    """Dos búferes fijados por turno, para que la copia y la reducción no bloqueen la CPU.

    Una copia desde memoria paginable espera a todo el trabajo previo del stream, así
    que la lectura del bloque siguiente no se solapaba con el GEMM del anterior. Cada
    búfer se reutiliza solo tras el evento que confirma que su copia ha terminado.
    """

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.slots, self.events, self.turn = [None, None], [None, None], 0

    def put(self, x, y):
        torch, turn = self.torch, self.turn
        self.turn = 1 - turn
        if self.events[turn] is not None:
            self.events[turn].synchronize()
        values, target = self.slots[turn] or (None, None)
        if (
            values is None
            or values.shape[0] < len(x)
            or values.shape[1:] != x.shape[1:]
            or values.numpy().dtype != x.dtype
        ):
            kind = torch.float32 if x.dtype == np.float32 else torch.float64
            values = torch.empty(x.shape, dtype=kind, pin_memory=True)
            target = torch.empty(len(y), dtype=torch.float64, pin_memory=True)
            self.slots[turn] = values, target
        values, target = values[: len(x)], target[: len(y)]
        values.numpy()[...] = x
        target.numpy()[...] = y
        moved = (
            values.to(self.device, non_blocking=True),
            target.to(self.device, non_blocking=True),
        )
        self.events[turn] = torch.cuda.Event()
        self.events[turn].record()
        return moved


def centered_normal_equations(blocks, mean, scale, target_mean, count, *, device):
    """Acumular Z'Z y Z'y centrados por bloques, en float64 y sin concatenar filas.

    Z = (X - mean) / scale. La memoria crece con d² y con un bloque, nunca con las
    filas. El centrado final usa sumas de la misma pasada y la cuenta de la anterior.
    Un bloque float32 viaja en float32 y se convierte en el dispositivo. La conversión
    es exacta y la resta y la división son IEEE en ambos lados, así que Z coincide bit
    a bit con la estandarización de numpy en float64.
    """
    import torch

    device = torch.device(device)
    gram = torch.zeros((len(mean), len(mean)), dtype=torch.float64, device=device)
    rhs = torch.zeros(len(mean), dtype=torch.float64, device=device)
    sum_x = torch.zeros_like(rhs)
    sum_y = torch.zeros((), dtype=torch.float64, device=device)
    center = torch.as_tensor(mean, dtype=torch.float64, device=device)
    spread = torch.as_tensor(scale, dtype=torch.float64, device=device)
    staging = _PinnedStaging(torch, device) if device.type == "cuda" else None
    digest = hashlib.sha256()
    for x, y in blocks:
        x = np.ascontiguousarray(x)
        if x.dtype.type not in _FLOATS:
            raise ValueError("Las características deben ser float32 o float64")
        digest.update(memoryview(x).cast("B"))
        digest.update(y.tobytes())
        if x.shape[1] != len(mean):
            raise ValueError("Han cambiado las dimensiones del entrenamiento")
        centered = y - target_mean
        if staging is None:
            values = torch.as_tensor(x, device=device)
            target = torch.as_tensor(centered, dtype=torch.float64, device=device)
        else:
            values, target = staging.put(x, centered)
        features = (values.to(torch.float64) - center) / spread
        gram.addmm_(features.T, features)
        rhs.addmv_(features.T, target)
        sum_x += features.sum(dim=0)
        sum_y += target.sum()
    gram -= torch.outer(sum_x, sum_x) / count
    rhs -= sum_x * sum_y / count
    return gram, rhs, sum_x, digest.digest()


@dataclass(frozen=True)
class RidgeStatistics:
    """Suficientes para cualquier alpha: estandarización y sistema normal sin penalizar.

    Solo contiene reducciones de las filas de entrenamiento. Resolver un alpha no
    modifica la Gram, que puede compartirse entre las alphas de una misma ventana.
    """

    count: int
    mean: np.ndarray
    scale: np.ndarray
    target_mean: float
    gram: object
    rhs: object
    sum_x: object

    def sha256(self):
        """Huella de los valores exactos, para declarar qué estadísticas usa cada ajuste."""
        digest = hashlib.sha256()
        digest.update(np.asarray([self.count], dtype="<i8").tobytes())
        digest.update(np.asarray([self.target_mean], dtype="<f8").tobytes())
        for value in (self.mean, self.scale, self.gram, self.rhs, self.sum_x):
            array = value if isinstance(value, np.ndarray) else value.cpu().numpy()
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()


def ridge_statistics(factory, *, device: str = "cuda:0") -> RidgeStatistics:
    """Dos pasadas de reducciones: media y varianza de Chan, y después Z'Z y Z'y centrados.

    No ajusta ningún parámetro. Los bloques float32 se conservan para hashear y copiar la
    mitad de bytes, y la primera pasada los convierte a float64 sin cambiar valores.
    """
    if device != "cuda:0":
        raise ValueError("Ridge requiere cuda:0")
    require_cuda()
    count, mean, m2, target_mean = 0, None, None, 0.0
    first = hashlib.sha256()
    for x, y in validated_blocks(factory, keep_float32=True):
        x = np.ascontiguousarray(x)
        first.update(memoryview(x).cast("B"))
        first.update(y.tobytes())
        x = x.astype(np.float64, copy=False)
        n = len(x)
        block_mean = x.mean(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            block_m2 = np.square(x - block_mean).sum(axis=0)
        if mean is None:
            mean, m2 = block_mean, block_m2
        else:
            delta = block_mean - mean
            m2 += block_m2 + np.square(delta) * count * n / (count + n)
            mean += delta * n / (count + n)
        target_mean += (float(y.mean()) - target_mean) * n / (count + n)
        count += n
        if not np.isfinite(mean).all() or not np.isfinite(m2).all() or not np.isfinite(target_mean):
            raise ValueError("Las estadísticas de entrenamiento han sufrido un desbordamiento")
    if count < 2:
        raise ValueError("Ridge necesita al menos dos muestras de entrenamiento")
    variance = m2 / count
    epsilon = np.finfo(np.float64).eps
    constant = variance <= count * epsilon * variance + np.square(count * mean * epsilon)
    scale = np.sqrt(variance)
    scale[constant] = 1.0
    gram, rhs, sum_x, second = centered_normal_equations(
        validated_blocks(factory, keep_float32=True),
        mean,
        scale,
        target_mean,
        count,
        device=device,
    )
    if first.digest() != second:
        raise ValueError("Los datos de entrenamiento han cambiado entre pasadas")
    return RidgeStatistics(count, mean, scale, target_mean, gram, rhs, sum_x)


def solve_ridge(statistics: RidgeStatistics, alpha: float) -> RidgeModel:
    """Resolver un alpha sobre una copia de la Gram, que queda intacta para las demás."""
    require_learning_allowed("la solución Ridge")
    import torch

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge requiere regularización positiva")
    gram = statistics.gram.clone()
    gram.diagonal().add_(alpha)
    if not torch.isfinite(gram).all() or not torch.isfinite(statistics.rhs).all():
        raise ValueError("El sistema Ridge no contiene valores finitos")
    coefficient = torch.linalg.solve(gram, statistics.rhs).cpu().numpy()
    if not np.isfinite(coefficient).all():
        raise ValueError("El ajuste Ridge no ha producido coeficientes finitos")
    count = statistics.count
    intercept = statistics.target_mean - float(
        (statistics.sum_x / count).cpu().numpy() @ coefficient
    )
    return RidgeModel(statistics.mean, statistics.scale, coefficient, intercept)


def fit_ridge_blocks(factory, *, alpha: float = 1.0, device: str = "cuda:0") -> RidgeModel:
    require_learning_allowed("el ajuste ridge por bloques")
    if device != "cuda:0" or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge requiere cuda:0 y regularización positiva")
    return solve_ridge(ridge_statistics(factory, device=device), alpha)
