"""Boosting tabular CPU con ajuste cronológico externo y presupuesto explícito."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from mars_titan.data.storage import sha256

from .inputs import validated_blocks


@dataclass
class BoostingModel:
    estimator: HistGradientBoostingRegressor
    training_rows: int

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.estimator.n_features_in_ or not np.isfinite(x).all():
            raise ValueError("Las entradas de boosting no son válidas")
        with threadpool_limits(limits=4):
            prediction = self.estimator.predict(x)
        if not np.isfinite(prediction).all():
            raise ValueError("Boosting ha producido predicciones no finitas")
        return prediction

    def save(self, path: Path) -> str:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                joblib.dump(self, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)
        return sha256(path)

    @classmethod
    def load_local(cls, path: Path, expected_sha256: str):
        """Solo artefactos propios de confianza con una huella conservada por separado."""
        if sha256(path) != expected_sha256:
            raise ValueError("La huella del modelo de boosting no coincide")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("El archivo no contiene el modelo de boosting esperado")
        return model


def fit_boosting_batches(factory, *, max_bytes: int = 256 * 1024**2) -> BoostingModel:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("El presupuesto de boosting debe ser positivo")
    parts, labels, size = [], [], 0
    for x, y in validated_blocks(factory):
        size += x.nbytes + y.nbytes
        if size > max_bytes:
            raise ValueError("La matriz tabular supera el presupuesto de boosting")
        parts.append(x.copy())
        labels.append(y.copy())
    if not parts or sum(len(x) for x in parts) < 2:
        raise ValueError("Boosting necesita al menos dos muestras")
    x, y = np.concatenate(parts), np.concatenate(labels)
    del parts, labels
    estimator = HistGradientBoostingRegressor(
        max_iter=30,
        max_leaf_nodes=7,
        min_samples_leaf=5,
        learning_rate=0.05,
        early_stopping=False,
        random_state=42,
    )
    with threadpool_limits(limits=4):
        estimator.fit(x, y)
    return BoostingModel(estimator, len(x))
