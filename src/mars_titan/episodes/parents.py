"""Predicciones del padre ligadas a entradas, orden de activos y codificación."""

import hashlib
import json
import re
import sqlite3

import numpy as np

from mars_titan.models.baselines.inputs import MODALITIES


def input_fingerprint(raw, parent_sha256, encoding):
    ids = raw["asset_ids"]
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 4096
        or len(set(ids)) != len(ids)
        or not all(isinstance(name, str) for name in ids)
        or set(raw["inputs"]) != set(MODALITIES)
    ):
        raise ValueError("Las entradas del padre necesitan activos únicos y todas las modalidades")
    order = np.argsort(ids)
    available = np.asarray(raw["available_at"])
    if (
        type(raw["prediction_at"]) is not int
        or available.shape != (len(ids),)
        or available.dtype.kind not in "iu"
        or (available > raw["prediction_at"]).any()
    ):
        raise ValueError("Las entradas del padre proceden del futuro")
    digest = hashlib.sha256(
        json.dumps(
            [parent_sha256, encoding, raw["prediction_at"], [ids[i] for i in order]]
        ).encode()
    )
    inputs, size = {}, 0
    for name in MODALITIES:
        value = np.asarray(raw["inputs"][name])
        size += value.size * 4
        if (
            value.ndim < 2
            or len(value) != len(ids)
            or size > 64 * 1024**2
            or not np.isfinite(value).all()
        ):
            raise ValueError("Las modalidades del padre exceden el presupuesto o son inválidas")
        inputs[name] = np.asarray(value[order], dtype=np.float32)
        if not np.isfinite(inputs[name]).all():
            raise ValueError("La modalidad no se puede representar en float32")
        digest.update(name.encode())
        digest.update(str(inputs[name].shape).encode())
        digest.update(inputs[name].tobytes())
    return digest.hexdigest(), inputs, order


class ParentCache:
    """Guardar solo escalares y recalcular cuando cambien los datos sintéticos."""

    def __init__(self, path, parent_sha256, encoding, predict, *, max_bytes=64 * 1024**2):
        if (
            not re.fullmatch(r"[a-f0-9]{64}", parent_sha256)
            or not encoding
            or not callable(predict)
            or type(max_bytes) is not int
            or not 1 <= max_bytes <= 64 * 1024**2
        ):
            raise ValueError("La caché necesita identidad y un presupuesto válido")
        self.parent_sha256, self.encoding, self.predictor = parent_sha256, encoding, predict
        self.max_bytes = max_bytes
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS predictions (key TEXT PRIMARY KEY, "
            "value BLOB, sha TEXT, accessed INTEGER)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS predictions_access ON predictions(accessed)")
        self.clock = self.db.execute(
            "SELECT coalesce(max(accessed),0) FROM predictions"
        ).fetchone()[0]

    def predict(self, raw):
        key, inputs, order = input_fingerprint(raw, self.parent_sha256, self.encoding)
        self.clock += 1
        with self.db:
            cached = self.db.execute(
                "SELECT value,sha FROM predictions WHERE key=?", (key,)
            ).fetchone()
            if cached:
                blob, digest = cached
                if hashlib.sha256(blob).hexdigest() != digest:
                    raise ValueError("La predicción cacheada está corrupta")
                values = np.frombuffer(blob, dtype="<f8")
            else:
                values = np.asarray(self.predictor(inputs), dtype="<f8")
                if values.shape != (len(order),) or not np.isfinite(values).all():
                    raise ValueError("El padre no devuelve una predicción finita por activo")
                blob = values.tobytes()
                if len(blob) > self.max_bytes:
                    raise ValueError("La predicción supera el presupuesto de caché")
                self.db.execute(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?)",
                    (key, blob, hashlib.sha256(blob).hexdigest(), self.clock),
                )
            if values.shape != (len(order),) or not np.isfinite(values).all():
                raise ValueError("La caché no conserva las dimensiones de la predicción")
            self.db.execute("UPDATE predictions SET accessed=? WHERE key=?", (self.clock, key))
            size = self.db.execute(
                "SELECT coalesce(sum(length(value)),0) FROM predictions"
            ).fetchone()[0]
            entries = self.db.execute("SELECT count(*) FROM predictions").fetchone()[0]
            while size > self.max_bytes or entries > 8192:
                old_key, length = self.db.execute(
                    "SELECT key,length(value) FROM predictions ORDER BY accessed LIMIT 1"
                ).fetchone()
                self.db.execute("DELETE FROM predictions WHERE key=?", (old_key,))
                size -= length
                entries -= 1
        return values[np.argsort(order)].copy()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
