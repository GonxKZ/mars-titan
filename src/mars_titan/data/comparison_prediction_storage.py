"""Codificación Parquet de las predicciones nuevas de un trabajo de comparación."""

from contextlib import contextmanager
from contextvars import ContextVar

import pyarrow as pa

ENV = "MARS_TITAN_COMPARISON_PREDICTION_STORAGE"
COMPACT = "compact_v1"
_OUTPUT = ContextVar("comparison_prediction_output", default=None)
_NAMES = {f"{phase}-predictions.parquet" for phase in ("validation", "calibration", "evaluation")}


@contextmanager
def compact_predictions(folder, *, rank_rule):
    """Activar el perfil solo dentro del directorio del trabajo que está escribiendo."""
    token = _OUTPUT.set((folder.resolve(), rank_rule))
    try:
        yield
    finally:
        _OUTPUT.reset(token)


def writer_options(path, schema):
    output = _OUTPUT.get()
    if output is None or path.name not in _NAMES:
        return {}
    folder, rank_rule = output
    if not path.resolve().is_relative_to(folder):
        return {}
    # La regla de rangos reutiliza muchas puntuaciones y su diccionario ocupa menos.
    split = [
        field.name
        for field in schema
        if pa.types.is_floating(field.type) and (not rank_rule or field.name == "target")
    ]
    return dict(
        compression_level=9,
        use_byte_stream_split=split,
        use_dictionary=[name for name in schema.names if name not in split],
    )
