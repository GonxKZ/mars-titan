"""LightGBM con las mismas entradas que los modelos, control común de la criba (#742, #746).

Es el LightGBM de Qlib de `qlib_lightgbm`, con sus parámetros, su normalización de la etiqueta
en cada sesión y la elección de rondas con la época común. Las características son las entradas
de la criba de cada fila (`features`), no las 158 de Alpha158, y la etiqueta es el residuo de la
criba normalizado en cada sesión.
"""

import numpy as np

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS

from . import qlib_lightgbm
from .inputs import MODALITIES

IDENTITY = "same_inputs_lightgbm_v1"
FEATURES = "price_window_modalities_and_presence_v1"
LABEL = "session_zscore_of_the_residual"
VECTORS = MODALITIES[1:]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_config(config):
    """Declaración del control con los parámetros de Qlib, sus puntos de control y sus entradas."""
    return qlib_lightgbm.checked_config(config, identity=IDENTITY, features=FEATURES, label=LABEL)


def features(inputs, presence):
    """Características float32 de una tanda: ventana de precios, modalidades y presencia.

    La ventana de 64 sesiones y seis canales se aplana por sesión y canal. Siguen los vectores
    de noticias, gráficos, fundamentales y contexto macro, con sus ausencias rellenadas y sus
    marcas de observación como los recibe el modelo, y los cinco bits de presencia.
    """
    _require(
        isinstance(inputs, dict) and set(inputs) == set(MODALITIES),
        "LightGBM necesita las cuatro modalidades, el contexto macro y la ventana de precios",
    )
    prices = np.asarray(inputs["prices"], dtype=np.float32)
    rows = len(prices)
    _require(
        prices.ndim == 3 and prices.shape[1:] == (CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS)),
        "La ventana de precios no tiene 64 sesiones con sus canales",
    )
    presence = np.asarray(presence)
    _require(
        presence.shape == (rows, len(MODALITIES)) and presence.dtype == np.bool_,
        "La presencia necesita un bit por modalidad y fila",
    )
    parts = [prices.reshape(rows, -1)]
    for name in VECTORS:
        values = np.asarray(inputs[name], dtype=np.float32)
        _require(values.ndim == 2 and len(values) == rows, f"{name} no tiene una fila por decisión")
        parts.append(values)
    parts.append(presence.astype(np.float32))
    matrix = np.concatenate(parts, axis=1)
    _require(np.isfinite(matrix).all(), "Las características contienen valores no finitos")
    return matrix
