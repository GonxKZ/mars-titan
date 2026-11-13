"""Estadísticas de `learning_scale` leídas del tramo de ajuste de una ventana (#836).

Las escalas de aprendizaje normalizan las entradas y la salida con un centro y una escala
por columna. Este módulo los calcula con una pasada de lectura sobre el índice de ajuste de
la ventana, el mismo que recorre después el entrenador, sin construir ningún modelo. Solo
entran decisiones del tramo de ajuste: ni las entradas de su calentamiento ni ninguna fila
de validación, calibración o evaluación.

Se conserva una muestra determinista de filas (una de cada `stride` en el orden del índice),
suficiente para estimar cuantiles por columna, y todas las etiquetas que maduran en el
tramo. El resultado se guarda junto al índice de ajuste, de modo que todos los modelos y
semillas de la misma ventana leen el mismo archivo y su huella entra en su identidad.
"""

import hashlib

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json
from mars_titan.models.learning_scale import (
    MIN_ROWS,
    MODALITIES,
    LearningScale,
    fit_statistics,
    statistics_digest,
)
from mars_titan.models.titans.config import canonical

from .window_indices import locked

# Filas de entrada que se conservan como mucho. Con 32.768 filas el error típico de una
# mediana ronda el 0,7 % de la desviación de su columna y la muestra ocupa unos 235 MiB.
MAX_ROWS = 32_768
BLOCK_ROWS = 1024
SAMPLE = "every_stride_th_decision_row_in_index_order_v1"


def _decision_rows(source):
    """Decisiones del tramo de ajuste en el índice, sin decodificar ninguna entrada."""
    path = source.path.parent / source.metadata["events_path"]
    table = pq.read_table(path, columns=["event_at", "kind"], filters=[("kind", "=", 0)])
    stamps = table["event_at"].to_numpy()
    return int(np.count_nonzero(stamps >= source.phase.decision_start))


def sample_identity(config, source, max_rows=MAX_ROWS):
    """Identidad de la muestra: declaración, índice de ajuste y regla de selección."""
    if not isinstance(config, LearningScale):
        raise ValueError("Las estadísticas necesitan la declaración LearningScale")
    if type(max_rows) is not int or not MIN_ROWS <= max_rows <= 1_048_576:
        raise ValueError("La muestra de las estadísticas admite entre 64 y 1.048.576 filas")
    return dict(
        schema_version=1,
        transform=config.identity(),
        index_sha256=source.identity,
        partition=source.phase.partition,
        decision_start=source.phase.decision_start,
        decision_end=source.phase.decision_end,
        rule=SAMPLE,
        max_rows=max_rows,
    )


def read_fit_sample(source, *, max_rows=MAX_ROWS):
    """Leer la muestra de decisiones del ajuste y todas sus etiquetas maduras.

    Devuelve `(entradas, presencia, etiquetas, stride)`. Recorre el índice una vez, por
    bloques, con el lector del entrenador.
    """
    if source.phase.partition != "train":
        raise ValueError("Las estadísticas de aprendizaje solo leen el tramo de ajuste")
    total = _decision_rows(source)
    stride = max(1, -(-total // max_rows))
    kept = {name: [] for name in MODALITIES}
    presence, targets, seen = [], [], 0
    start = source.phase.decision_start
    for event in source.batched_events(block_rows=BLOCK_ROWS):
        targets.extend(value for _, _, value in event.labels)
        if event.at < start:
            continue
        for block in event.inputs:
            size = len(block["presence"])
            # Posiciones del bloque cuya cuenta global es múltiplo del paso.
            chosen = np.arange((-seen) % stride, size, stride)
            seen += size
            if not len(chosen):
                continue
            for name in MODALITIES:
                kept[name].append(np.array(block["inputs"][name][chosen], copy=True))
            presence.append(np.array(block["presence"][chosen], dtype=np.bool_, copy=True))
    if seen != total or not presence:
        raise ValueError("La pasada de estadísticas no concilia con las decisiones del índice")
    inputs = {name: np.concatenate(values) for name, values in kept.items()}
    return inputs, np.concatenate(presence), np.asarray(targets, dtype=np.float64), stride


def learning_statistics(config, source, *, max_rows=MAX_ROWS):
    """Estadísticas de la ventana, calculadas una vez y guardadas junto al índice de ajuste.

    Devuelve el documento con `statistics` y su huella. Si el archivo ya existe se comprueba
    que corresponde a la misma muestra y se reutiliza sin leer ninguna fila.
    """
    identity = sample_identity(config, source, max_rows)
    digest = hashlib.sha256(canonical(identity).encode()).hexdigest()
    path = source.path.parent / f"learning-scale-{digest[:16]}.json"
    safe_destination(path)
    # Con varios trabajos a la vez, el segundo espera y reutiliza el archivo del primero.
    with locked(path):
        if path.exists():
            document, _ = read_manifest(path, 64 * 1024**2)
            if document.get("identity") != identity or statistics_digest(
                document.get("statistics")
            ) != document.get("statistics_sha256"):
                raise ValueError("Las estadísticas guardadas no corresponden a esta ventana")
            return document
        inputs, presence, targets, stride = read_fit_sample(source, max_rows=max_rows)
        statistics = fit_statistics(config, inputs, presence, targets)
        document = dict(
            schema_version=1,
            kind="learning_scale_statistics",
            identity=identity,
            stride=stride,
            statistics=statistics,
            statistics_sha256=statistics_digest(statistics),
            final_test_opened=False,
        )
        atomic_json(path, document)
        return document
