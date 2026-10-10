"""Filas nuevas del postentrenamiento por etapas y su disjunción con las del padre.

En la ventana k el padre es el estado elegido de la campaña base en k-1. Ajustó sus pesos
con `train_{k-1}`, eligió su época con `val_{k-1}` y se calibró con `cal_{k-1}`. El ajuste
de k solo usa las filas de `train_k` con decisión desde el final de `cal_{k-1}` hasta el
final de `train_k`, que ya purga la vista antes de `val_k`. `fit_rows_proof` lo comprueba
por identidad de fila (activo y fila de su archivo de muestras, que las vistas de una
edición comparten) contra los tres tramos del padre en su vista, guarda los recuentos y la
intersección vacía y resume las filas nuevas con su huella.

`posttraining_rows`, `asset_digest`, `combine` y `row_fingerprint` son las de
`training.campaign_chain`, con la misma huella, para que el verificador de disjunción y
los recibos de la cadena coincidan. `labels_used_until` hace de `training.label_maturity`
sobre las mismas etiquetas aceptadas.
"""

import numpy as np
import pyarrow.parquet as pq

from mars_titan.environments.walk_forward_receipt import _microseconds
from mars_titan.training.campaign_chain import (
    FINGERPRINT,
    asset_digest,
    combine,
    posttraining_rows,
    row_fingerprint,
)

__all__ = [
    "FINGERPRINT",
    "asset_digest",
    "combine",
    "fit_rows_proof",
    "labels_used_until",
    "partition_rows",
    "posttraining_rows",
    "row_fingerprint",
]

PARENT_PARTITIONS = ("train", "validation", "calibration")
PROOF_KIND = "staged_posttraining_fit_rows"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def partition_rows(dataset, partitions):
    """Filas de muestra, decisiones y madurez de las etiquetas aceptadas de cada tramo.

    Devuelve, por tramo, un diccionario `(mercado, activo) -> (filas, decisiones, madurez)`
    con los activos que tienen filas, y la huella del archivo de muestras de cada activo.
    """
    result, samples = {name: {} for name in partitions}, {}
    for asset in dataset.assets:
        key = (asset["market"], asset["symbol"])
        samples[key] = asset["samples_sha256"]
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            rows = file.metadata.num_rows
        for name in partitions:
            positions, decisions, _, maturity = dataset._labels(asset, name, rows)
            if len(positions):
                result[name][key] = (positions, decisions, maturity)
    return result, samples


def _latest(rows):
    return max((int(value[2].max()) for value in rows.values()), default=None)


def labels_used_until(dataset, partitions=PARENT_PARTITIONS):
    """Madurez de la última etiqueta aceptada de esos tramos de la vista."""
    rows, _ = partition_rows(dataset, partitions)
    latest = [value for value in (_latest(found) for found in rows.values()) if value is not None]
    _require(latest, "La vista no tiene etiquetas en los tramos pedidos")
    return max(latest)


def fit_rows_proof(parent_dataset, dataset, *, parent_fold, fold):
    """Comprobar y resumir las filas nuevas del ajuste de una ventana frente al padre.

    Una fila es un activo y su fila del archivo de muestras. Para compararlas, cada activo
    presente en las dos vistas debe conservar el mismo archivo de muestras.
    """
    start, end = posttraining_rows(parent_fold, fold)
    since, until = _microseconds(start), _microseconds(end)
    validation = _microseconds(fold["validation"][0])
    used, parent_samples = partition_rows(parent_dataset, PARENT_PARTITIONS)
    found, samples = partition_rows(dataset, PARENT_PARTITIONS)
    fit = {}
    for key, (positions, decisions, maturity) in found["train"].items():
        selected = (decisions >= since) & (decisions < until)
        if selected.any():
            fit[key] = (positions[selected], decisions[selected], maturity[selected])
    _require(fit, "La ventana no tiene filas nuevas para el ajuste")
    _require(
        all(samples[key] == parent_samples[key] for key in fit if key in parent_samples),
        "Un activo cambia de archivo de muestras entre la vista del padre y la ventana",
    )
    intersection = {
        name: sum(int(np.isin(fit[key][0], rows[key][0]).sum()) for key in fit if key in rows)
        for name, rows in used.items()
    }
    _require(not any(intersection.values()), "Una fila de ajuste ya la usó el padre en su ventana")
    parent_mature = max(_latest(rows) for rows in used.values() if rows)
    _require(
        parent_mature < since,
        "Una etiqueta del padre madura después del inicio de las filas nuevas",
    )
    decisions = np.concatenate([value[1] for value in fit.values()])
    mature = max(int(value[2].max()) for value in fit.values())
    _require(mature < validation, "Una etiqueta de ajuste madura dentro de la validación")
    rows, digest = combine({key: asset_digest(value[0]) for key, value in fit.items()})
    window_mature = max(_latest(found[name]) for name in PARENT_PARTITIONS if found[name])
    return dict(
        kind=PROOF_KIND,
        schema_version=1,
        parent_window=parent_fold["id"],
        window=fold["id"],
        parent_view_sha256=parent_dataset.identity,
        view_sha256=dataset.identity,
        start=start,
        end=end,
        start_us=since,
        end_us=until,
        validation_start_us=validation,
        parent_rows={
            name: int(sum(len(value[0]) for value in rows.values())) for name, rows in used.items()
        },
        parent_labels_mature_until=parent_mature,
        intersection=intersection,
        rows=rows,
        sha256=digest,
        first_decision=int(decisions.min()),
        last_decision=int(decisions.max()),
        labels_mature_until=mature,
        # Ajuste, selección y calibración del estado que salga de esta ventana, padre incluido.
        labels_used_until=max(parent_mature, window_mature),
    )
