"""Contrato de la cadena del walk-forward por etapas que publica la etapa de adaptadores.

En la ventana k ≥ 1 de un ámbito, para cada brazo base y semilla, el predictor de la
cadena es el candidato con menor puntuación de validación de esa ventana entre el padre
congelado (el estado elegido de la campaña base en k-1), cada caso de adaptadores de la
matriz y la continuación completa del padre con las filas nuevas. Solo sustituye al padre
congelado con una mejora estricta y los demás empates se resuelven por el identificador
del trabajo (`chain_validation_score_v1`). En la ventana 0 el predictor es el estado
elegido de la base.

La puntuación es el MAE medio por sesión de la mediana sobre todas las filas de `val_k`, la
misma definición que elige el estado de la campaña base. Se recalcula con
`validation_score` sobre las predicciones de validación guardadas de cada candidato, que
salen del mismo recorrido, así que dos candidatos con las mismas predicciones empatan.

Los identificadores, rutas, regla, elección y lectura vienen de `training.campaign_chain`,
que declara el diseño en el plan de la campaña. Este módulo solo añade lo propio de la
etapa: el brazo del padre congelado y la puntuación de validación.
"""

import math

import numpy as np
import pyarrow as pa

from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.training.campaign_chain import (
    CANDIDATES,
    CHAIN_SUFFIX,
    RULE,
    SELECTED,
    SELECTION,
    SELECTION_KIND,
    chain_arm,
    chain_folder,
    chain_job_id,
    choose,
    parent_jobs,
    read_selection,
    scope_windows,
)

__all__ = [
    "CANDIDATES",
    "CHAIN_SUFFIX",
    "FROZEN_SUFFIX",
    "RULE",
    "SELECTED",
    "SELECTION",
    "SELECTION_KIND",
    "chain_arm",
    "chain_folder",
    "chain_job_id",
    "choose",
    "frozen_arm",
    "parent_jobs",
    "read_selection",
    "scope_windows",
    "validation_score",
]

FROZEN_SUFFIX = "__frozen_parent"
# Filas por actualización del acumulador por sesión.
_BATCH = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_arm(base_arm):
    """Brazo de las predicciones del padre congelado de un brazo base."""
    return f"{base_arm}{FROZEN_SUFFIX}"


def validation_score(table):
    """MAE medio por sesión de la predicción (la mediana) sobre todas las filas.

    Las filas se recorren en orden de mercado, activo e instante, así que el valor solo
    depende de las predicciones y no del orden en que se escribieron.
    """
    market = table["market"].to_numpy(zero_copy_only=False).astype(str)
    asset = table["asset_id"].to_numpy(zero_copy_only=False).astype(str)
    moment = table["prediction_at"].cast(pa.int64()).to_numpy()
    error = table["prediction"].to_numpy().astype(np.float64) - table["target"].to_numpy()
    order = np.lexsort((moment, asset, market))
    errors = SessionErrors()
    for start in range(0, len(order), _BATCH):
        rows = order[start : start + _BATCH]
        errors.update(market[rows], moment[rows], error[rows])
    score = errors.summary()["session_mae"]
    _require(
        isinstance(score, float) and math.isfinite(score) and score >= 0,
        "La validación no tiene un MAE por sesión finito",
    )
    return score
