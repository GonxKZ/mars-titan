"""Acciones predictivas fijadas con entrenamiento, sin ajustar la rejilla con validación."""

import math
import re
from dataclasses import dataclass

import numpy as np


def _real(values, *, maximum=4096):
    array = np.asarray(values)
    if array.size > maximum or np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Los valores deben ser reales y respetar el presupuesto")
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = array.astype(np.float64, copy=False)
    except FloatingPointError as error:
        raise ValueError("Los valores no son representables en float64") from error
    if not np.isfinite(array).all():
        raise ValueError("Los valores necesitan ser finitos")
    return array


@dataclass(frozen=True)
class ActionGrid:
    values: np.ndarray
    scale: float
    source_sha256: str
    training_samples: int

    def __post_init__(self):
        values = _real(self.values, maximum=21)
        if (
            values.shape != (21,)
            or not np.all(np.diff(values) > 0)
            or values[10] != 0
            or type(self.scale) not in (int, float)
            or not math.isfinite(self.scale)
            or self.scale <= 0
            or not isinstance(self.source_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", self.source_sha256)
            or type(self.training_samples) is not int
            or not 1 <= self.training_samples <= 16_777_216
        ):
            raise ValueError(
                "La rejilla necesita 21 acciones, escala y procedencia de entrenamiento válidas"
            )
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @classmethod
    def fit(cls, targets, *, source_sha256, partition):
        """Usar percentiles 1 y 99 y error absoluto medio, con copia máxima de 128 MiB."""
        if partition != "train":
            raise ValueError("La rejilla solo se ajusta con etiquetas de entrenamiento")
        array = _real(targets, maximum=16_777_216)
        if array.ndim != 1 or not len(array):
            raise ValueError("Las etiquetas de entrenamiento necesitan un vector no vacío")
        scale = max(float(np.mean(np.abs(array))), 1e-8)
        low, high = np.quantile(array, (0.01, 0.99), method="linear")
        radius = max(abs(float(low)), abs(float(high)), 1e-8)
        values = np.linspace(-radius, radius, 21)
        values[10] = 0.0
        return cls(values, scale, source_sha256, len(array))

    def _probabilities(self, values):
        probabilities = _real(values, maximum=4096 * 21)
        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != 21
            or not len(probabilities)
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
        ):
            raise ValueError("Cada distribución necesita 21 probabilidades que sumen uno")
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def median(self, probabilities):
        """Elegir la mediana inferior, también cuando existe un empate exacto."""
        probabilities = self._probabilities(probabilities)
        cumulative = np.cumsum(probabilities, axis=1)
        # El margen solo selecciona sumas que necesitan compensación. No cambia el umbral.
        uncertain = np.argwhere(np.abs(cumulative - 0.5) <= 21 * np.finfo(np.float64).eps)
        for row, column in uncertain:
            cumulative[row, column] = math.fsum(probabilities[row, : column + 1])
        indexes = np.argmax(cumulative >= 0.5, axis=1)
        return self.values[indexes]

    def expected_loss(self, probabilities, targets):
        probabilities, target = self._probabilities(probabilities), _real(targets)
        if target.shape != (len(probabilities),):
            raise ValueError("Las distribuciones y etiquetas no tienen el mismo número de filas")
        with np.errstate(over="raise", invalid="raise"):
            return (probabilities * np.abs(self.values - target[:, None])).sum(axis=1) / self.scale

    def reward(self, actions, targets):
        actions, target = np.asarray(actions), _real(targets)
        if (
            actions.ndim != 1
            or not len(actions)
            or target.shape != actions.shape
            or not np.issubdtype(actions.dtype, np.integer)
            or (actions < 0).any()
            or (actions >= 21).any()
        ):
            raise ValueError("Las acciones y etiquetas no forman una decisión válida")
        with np.errstate(over="raise", invalid="raise"):
            return -np.abs(self.values[actions] - target) / self.scale

    def saturation(self, targets):
        target = _real(targets)
        if target.ndim != 1:
            raise ValueError("Las etiquetas necesitan un vector")
        return dict(
            below=int(np.sum(target < self.values[0])),
            above=int(np.sum(target > self.values[-1])),
            samples=len(target),
        )

    def to_dict(self):
        return dict(
            schema_version=1,
            values=self.values.tolist(),
            scale=self.scale,
            source_sha256=self.source_sha256,
            training_samples=self.training_samples,
        )

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value)
            != {"schema_version", "values", "scale", "source_sha256", "training_samples"}
            or value["schema_version"] != 1
        ):
            raise ValueError("La rejilla guardada no conserva su contrato")
        return cls(**{key: item for key, item in value.items() if key != "schema_version"})
