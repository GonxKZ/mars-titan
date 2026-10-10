"""Última maduración de las etiquetas que una vista temporal expone al predictor.

El recibo de ventana declara `labels_used_until`, el instante en que maduró la última
etiqueta que pudo fijar los parámetros, la selección o la calibración del predictor. Se
deriva aquí de las propias etiquetas de la vista, no de las fechas del protocolo. Así, una
etiqueta que madure dentro o después de la evaluación produce un límite que
`environments.walk_forward_receipt.read_window_receipt` rechaza y la ventana no se publica.

Cada archivo de etiquetas se comprueba con la huella que declara el manifiesto de la vista
y se lee por columnas, sin cargar la vista completa.
"""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import sha256

from .corpus_inputs import _times

# Tramos cuyas etiquetas fijan parámetros (ajuste), estado elegido (selección) o la
# calibración común de las predicciones.
FIT_PARTITIONS = ("train", "validation", "calibration")
CALIBRATION_PARTITIONS = ("calibration",)
RULE = "max_target_available_at_of_accepted_labels_in_fit_view_and_calibration_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def label_maturity(manifest_path, partitions):
    """Mayor `target_available_at` de las etiquetas aceptadas de esos tramos de la vista.

    Devuelve el instante en microsegundos UTC y el número de etiquetas leídas.
    """
    manifest_path = Path(manifest_path)
    manifest, _ = read_manifest(manifest_path, 8 * 1024**2)
    partitions = tuple(partitions)
    _require(
        partitions and set(partitions) <= set(FIT_PARTITIONS),
        "Solo se leen etiquetas de ajuste, selección o calibración",
    )
    root = Path(manifest["roots"]["labels"]).resolve()
    latest, rows = None, 0
    for asset in manifest["assets"]:
        expected = sum(asset["counts"][name] for name in partitions)
        if not expected:
            continue
        path = root / asset["market"] / asset["symbol"] / "labels.parquet"
        _require(
            path.is_file()
            and not path.is_symlink()
            and os.path.commonpath((root, os.path.realpath(path))) == str(root)
            and sha256(path) == asset["labels_sha256"],
            f"Las etiquetas de {asset['market']}/{asset['symbol']} no son las de la vista",
        )
        table = pq.read_table(path, columns=["partition", "target_available_at"])
        table = table.filter(pc.is_in(table["partition"], value_set=pa.array(partitions)))
        _require(
            table.num_rows == expected,
            f"Las etiquetas de {asset['market']}/{asset['symbol']} no concilian con la vista",
        )
        maturity = int(_times(table["target_available_at"]).max())
        latest = maturity if latest is None else max(latest, maturity)
        rows += expected
    _require(latest is not None, "La vista no tiene etiquetas en los tramos pedidos")
    return latest, rows
