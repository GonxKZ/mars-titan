"""Recibo mínimo de una ventana walk-forward que leen los entornos de refuerzo.

El productor de predicciones escribe un recibo por ventana y mercado. Los entornos solo
leen sus límites, la última etiqueta usada por el predictor y las huellas de las
predicciones emitidas. Ningún campo se acepta si no coincide con el protocolo v2.
"""

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from mars_titan.evaluation.splits import PARTITIONS, build_folds

from .cohorts import FINAL_TEST_START_US

RECEIPT_KIND = "walk_forward_window_receipt"
FINGERPRINT = b"mars-titan-predictions-v1"
_KEYS = {"kind", "schema_version", "protocol", "fold", "parent", "labels_used_until", "predictions"}
_HEX = re.compile(r"[a-f0-9]{64}")


def _microseconds(day):
    """Medianoche UTC del límite, con la misma conversión que evaluation.splits."""
    return int(np.datetime64(day, "us").astype(np.int64))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prediction_fingerprint(prediction_at, asset_id, score):
    """Huella canónica de las predicciones de un tramo, independiente del orden de las filas.

    Ordena por instante y activo, rechaza claves repetidas y valores no finitos y resume
    instantes ``int64``, identificadores separados por saltos de línea y puntuaciones
    ``float64`` little-endian. Devuelve el número de filas y la huella.
    """
    times = np.asarray(prediction_at)
    assets = np.asarray(asset_id, dtype=str)
    values = np.asarray(score)
    if (
        times.ndim != 1
        or not len(times)
        or times.dtype.kind not in "iu"
        or assets.shape != times.shape
        or values.shape != times.shape
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
        or any("\n" in asset or not asset for asset in assets)
    ):
        raise ValueError("Las predicciones necesitan instante, activo y puntuación finita")
    order = np.lexsort((assets, times))
    times, assets = times[order].astype("<i8"), assets[order]
    if ((np.diff(times) == 0) & (assets[1:] == assets[:-1])).any():
        raise ValueError("Una predicción repite instante y activo")
    digest = hashlib.sha256(FINGERPRINT)
    digest.update(times.tobytes())
    digest.update("\n".join(assets.tolist()).encode())
    digest.update(values[order].astype("<f8").tobytes())
    return len(times), digest.hexdigest()


@dataclass(frozen=True)
class WalkForwardWindow:
    """Ventana validada de un protocolo v2, con límites en microsegundos UTC."""

    market: str
    fold: str
    bounds: tuple
    labels_used_until: int
    parent: tuple
    predictions: tuple
    sha256: str

    def segment(self, partition):
        """Intervalo [inicio, fin) de un tramo. Toda etiqueta admitida madura antes del fin."""
        bounds = dict(self.bounds)
        if partition not in bounds:
            raise ValueError("El tramo no pertenece a la ventana walk-forward")
        return bounds[partition]

    def prediction_record(self, partition):
        records = dict(self.predictions)
        if partition not in records:
            raise ValueError("El recibo no declara predicciones de ese tramo")
        return records[partition]

    def identity(self, partition):
        start, end = self.segment(partition)
        return dict(
            receipt_sha256=self.sha256,
            market=self.market,
            fold=self.fold,
            partition=partition,
            start=start,
            end=end,
        )


def read_window_receipt(receipt):
    """Validar un recibo de ventana y devolver sus límites sin abrir otras fuentes."""
    if not isinstance(receipt, dict) or set(receipt) != _KEYS:
        raise ValueError("El recibo walk-forward no conserva sus campos")
    protocol, fold = receipt["protocol"], receipt["fold"]
    if (
        receipt["kind"] != RECEIPT_KIND
        or receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or not isinstance(protocol, dict)
        or protocol.get("schema_version") != 2
        or protocol.get("final_test_start") != "2024-01-01"
    ):
        raise ValueError("El recibo necesita un protocolo walk-forward v2 con 2024 reservado")
    # build_folds valida el protocolo completo. La ventana debe ser una de las suyas.
    if not isinstance(fold, dict) or fold not in build_folds(protocol):
        raise ValueError("La ventana no pertenece al protocolo declarado")
    bounds = tuple((name, tuple(_microseconds(day) for day in fold[name])) for name in PARTITIONS)
    if bounds[-1][1][1] > FINAL_TEST_START_US:
        raise ValueError("La ventana alcanza el test reservado")
    parent, until = receipt["parent"], receipt["labels_used_until"]
    if (
        not isinstance(parent, dict)
        or set(parent) != {"id", "sha256"}
        or not isinstance(parent["id"], str)
        or not 1 <= len(parent["id"]) <= 128
        or not isinstance(parent["sha256"], str)
        or not _HEX.fullmatch(parent["sha256"])
    ):
        raise ValueError("El recibo no identifica el predictor ajustado en la ventana")
    # Ajuste, selección y calibración solo pueden usar etiquetas maduras antes de evaluar.
    if type(until) is not int or not 0 <= until < dict(bounds)["evaluation"][0]:
        raise ValueError("La última etiqueta usada debe madurar antes de la evaluación")
    predictions = receipt["predictions"]
    if (
        not isinstance(predictions, dict)
        or not set(predictions) <= set(PARTITIONS)
        or any(
            not isinstance(record, dict)
            or set(record) != {"rows", "sha256"}
            or type(record["rows"]) is not int
            or record["rows"] < 1
            or not isinstance(record["sha256"], str)
            or not _HEX.fullmatch(record["sha256"])
            for record in predictions.values()
        )
    ):
        raise ValueError("Las huellas de las predicciones no cumplen el contrato")
    return WalkForwardWindow(
        market=protocol["market"],
        fold=fold["id"],
        bounds=bounds,
        labels_used_until=until,
        parent=(parent["id"], parent["sha256"]),
        predictions=tuple(
            sorted(
                (name, (record["rows"], record["sha256"])) for name, record in predictions.items()
            )
        ),
        sha256=_digest(receipt),
    )
