"""Tablas de predicción escritas a mano para las pruebas de la campaña de comparación (#836).

Tienen el contrato de columnas de los ejecutores y valores pequeños cuyo resultado se calcula
aparte en cada prueba. Ninguna procede de un modelo.
"""

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq

# Una sesión de calibración en la que el residuo es la mitad de la predicción: λ = 0,5.
CALIBRATION = [("2019-01-02", [(0.02, 0.01), (-0.02, -0.01), (0.04, 0.02)])]
# Dos sesiones de evaluación, con (predicción, residuo) por fila.
EVALUATION = [
    ("2019-02-01", [(0.02, 0.0), (-0.04, -0.01), (0.01, 0.03)]),
    ("2019-02-04", [(0.01, 0.0), (0.03, 0.02), (-0.02, -0.02)]),
]
VALIDATION = [("2018-12-03", [(0.01, 0.02), (0.0, -0.01), (0.02, 0.0)])]
PARTITIONS = ("validation", "calibration", "evaluation")


def write_table(path, sessions, *, scale=1.0):
    """Escribir una tabla con el contrato de predicciones de los ejecutores."""
    rows = [
        (f"{day}-{position}", datetime.fromisoformat(day).replace(tzinfo=UTC), target, value)
        for day, pairs in sessions
        for position, (value, target) in enumerate(pairs)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            dict(
                sample_id=[row[0] for row in rows],
                asset_id=[row[0].rsplit("-", 1)[1] for row in rows],
                market=["US"] * len(rows),
                prediction_at=pa.array([row[1] for row in rows], pa.timestamp("us", tz="UTC")),
                target=[row[2] for row in rows],
                prediction=[scale * row[3] for row in rows],
                zero=[0.0] * len(rows),
            )
        ),
        path,
    )


def write_tables(folder, *, scale=1.0, evaluation=EVALUATION):
    """Escribir las tres tablas de un trabajo y devolver sus rutas por partición."""
    parts = dict(validation=VALIDATION, calibration=CALIBRATION, evaluation=evaluation)
    for name, sessions in parts.items():
        write_table(folder / f"{name}-predictions.parquet", sessions, scale=scale)
    return {name: folder / f"{name}-predictions.parquet" for name in parts}


def declared(folder=""):
    """Entrada `predictions` del informe de un trabajo cuyas tablas están en `folder`."""
    prefix = f"{folder}/" if folder else ""
    return {name: dict(path=f"{prefix}{name}-predictions.parquet") for name in PARTITIONS}
