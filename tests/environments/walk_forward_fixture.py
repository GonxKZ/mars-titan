"""Recibos walk-forward sintéticos sobre los protocolos v2 versionados del proyecto."""

import copy
import json
from pathlib import Path

import numpy as np

from mars_titan.environments.walk_forward_receipt import (
    RECEIPT_KIND,
    prediction_fingerprint,
    read_window_receipt,
)
from mars_titan.evaluation.splits import build_folds

ROOT = Path(__file__).parents[2]
PROTOCOLS = {
    "US": "configs/evaluation/historical-masked-us-walk-forward-v2.json",
    "CN": "configs/evaluation/historical-masked-cn-walk-forward-v2.json",
}
PARENT = {"id": "synthetic_fixture_parent", "sha256": "b" * 64}


def microseconds(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def protocol(market):
    return json.loads((ROOT / PROTOCOLS[market]).read_text())


def fold(market, index=-1):
    return copy.deepcopy(build_folds(protocol(market))[index])


def receipt(market="US", *, index=-1, until=None, predictions=None):
    """Recibo con predicciones declaradas por tramo. Por omisión, ajuste hasta la evaluación."""
    window = fold(market, index)
    return dict(
        kind=RECEIPT_KIND,
        schema_version=1,
        protocol=protocol(market),
        fold=window,
        parent=dict(PARENT),
        labels_used_until=(microseconds(window["evaluation"][0]) - 1 if until is None else until),
        predictions={} if predictions is None else predictions,
    )


def record(values):
    rows, digest = prediction_fingerprint(**values)
    return dict(rows=rows, sha256=digest)


def window(market="US", *, partition="evaluation", values=None, **options):
    """Ventana validada con la huella de ``values`` en el tramo indicado."""
    predictions = {} if values is None else {partition: record(values)}
    return read_window_receipt(receipt(market, predictions=predictions, **options))
