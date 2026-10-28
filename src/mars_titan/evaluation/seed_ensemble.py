"""Media de semillas sobre predicciones ya emitidas (F2 de #749, parte `seed_mean_v1`).

Cada semilla es una ejecución completa del mismo modelo, receta, ventana y filas. La media
por fila de sus predicciones es la predicción del conjunto. Con la cabeza de cuantiles se
promedia cada nivel por separado, la media de funciones cuantil de Vincent (Ratcliff, 1979,
https://doi.org/10.1037/0033-2909.86.3.446). Promediar vectores ordenados conserva el orden
de los niveles.

Las tablas se ordenan por semilla antes de sumar, así que el resultado no depende del orden
en que se entreguen. La suma se hace en FP64 en ese orden fijo y se divide entre K.

El control declarado en la configuración de F2 es una sola semilla con las mismas
actualizaciones totales que todas las semillas del conjunto juntas. Esta función solo
combina predicciones y no entrena ni selecciona nada.
"""

import numpy as np
import pyarrow as pa

from mars_titan.models.quantile_head import QUANTILE_COLUMNS

SEED_MEAN = "seed_mean_v1"
MAX_SEEDS = 16
# Columnas que deben coincidir exactamente en todas las semillas.
KEYS = ("sample_id", "asset_id", "market", "prediction_at", "target", "zero")


def seed_mean(tables):
    """Tabla con la media por fila de {semilla: tabla} y su descripción reproducible."""
    if (
        type(tables) is not dict
        or not 2 <= len(tables) <= MAX_SEEDS
        or any(type(seed) is not int or seed < 0 for seed in tables)
        or any(type(table) is not pa.Table for table in tables.values())
    ):
        raise ValueError("La media necesita de 2 a 16 tablas identificadas por semilla")
    seeds = sorted(tables)
    first = tables[seeds[0]]
    names = first.schema.names
    levels = [name for name in QUANTILE_COLUMNS if name in names]
    averaged = ["prediction", *levels]
    if (
        not {*KEYS, "prediction"} <= set(names)
        or levels not in ([], list(QUANTILE_COLUMNS))
        or first.num_rows == 0
    ):
        raise ValueError("Las predicciones no tienen el esquema común")
    for seed in seeds[1:]:
        table = tables[seed]
        if table.schema != first.schema or table.num_rows != first.num_rows:
            raise ValueError("Las semillas no comparten esquema ni número de filas")
        for name in KEYS:
            if not table[name].equals(first[name]):
                raise ValueError(f"Las semillas no comparten la columna {name}")
    columns = {name: first[name] for name in names}
    for name in averaged:
        stacked = np.stack([tables[seed][name].to_numpy() for seed in seeds])
        if stacked.dtype != np.float64 or not np.isfinite(stacked).all():
            raise ValueError(f"La columna {name} debe ser FP64 finita en todas las semillas")
        total = stacked[0].copy()
        for row in stacked[1:]:
            total += row
        columns[name] = pa.array(total / len(seeds), type=pa.float64())
    combined = pa.table(columns, schema=first.schema)
    description = dict(
        schema_version=1,
        method=SEED_MEAN,
        seeds=seeds,
        rows=first.num_rows,
        averaged_columns=averaged,
        order="ascending_seed_then_sequential_fp64_sum_divided_by_count",
        quantiles="per_level_mean_of_quantile_functions" if levels else None,
    )
    return combined, description
