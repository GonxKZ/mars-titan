"""Ridge estandarizado por bloques con intercepto no penalizado y cálculo CUDA."""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mars_titan.data.embeddings import require_cuda

from .inputs import validated_blocks


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def predict(self, x) -> np.ndarray:
        import torch

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Las entradas de predicción no son válidas")
        device = "cuda:0"
        with torch.inference_mode():
            value = torch.as_tensor(
                (x - self.mean) / self.scale, dtype=torch.float64, device=device
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


def fit_ridge_blocks(factory, *, alpha: float = 1.0, device: str = "cuda:0") -> RidgeModel:
    import torch

    if device != "cuda:0" or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge requiere cuda:0 y regularización positiva")
    require_cuda()
    count, mean, m2, target_mean = 0, None, None, 0.0
    first = hashlib.sha256()
    for x, y in validated_blocks(factory):
        first.update(x.tobytes())
        first.update(y.tobytes())
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
    gram = torch.zeros((len(mean), len(mean)), dtype=torch.float64, device=device)
    rhs = torch.zeros(len(mean), dtype=torch.float64, device=device)
    sum_x = torch.zeros_like(rhs)
    sum_y = torch.zeros((), dtype=torch.float64, device=device)
    second = hashlib.sha256()
    for x, y in validated_blocks(factory):
        second.update(x.tobytes())
        second.update(y.tobytes())
        if x.shape[1] != len(mean):
            raise ValueError("Han cambiado las dimensiones del entrenamiento")
        features = torch.as_tensor((x - mean) / scale, dtype=torch.float64, device=device)
        target = torch.as_tensor(y - target_mean, dtype=torch.float64, device=device)
        gram.addmm_(features.T, features)
        rhs.addmv_(features.T, target)
        sum_x += features.sum(dim=0)
        sum_y += target.sum()
    if first.digest() != second.digest():
        raise ValueError("Los datos de entrenamiento han cambiado entre pasadas")
    gram -= torch.outer(sum_x, sum_x) / count
    rhs -= sum_x * sum_y / count
    gram.diagonal().add_(alpha)
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise ValueError("El sistema Ridge no contiene valores finitos")
    coefficient = torch.linalg.solve(gram, rhs).cpu().numpy()
    if not np.isfinite(coefficient).all():
        raise ValueError("El ajuste Ridge no ha producido coeficientes finitos")
    intercept = target_mean - float((sum_x / count).cpu().numpy() @ coefficient)
    return RidgeModel(mean, scale, coefficient, intercept)
