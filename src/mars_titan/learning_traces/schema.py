"""Formato de las trazas de aprendizaje que comparten el registrador y sus lectores."""

import pyarrow as pa

SCHEMA_VERSION = 1
KIND = "learning_traces"
STATS = ("mean", "std", "min", "max", "abs_max", "l2", "finite_fraction")
SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("phase", pa.string()),
        ("metric", pa.string()),
        ("group", pa.string()),
        ("stat", pa.string()),
        ("value", pa.float64()),
    ]
)
NAME_PATTERN = r"[a-z][a-z0-9_.:/-]{0,95}"
