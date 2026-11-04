"""Regla de momentum y reversión sin aprendizaje, control común de la criba (#742, #746).

Lee la misma ventana de precios que los modelos: las 64 sesiones del calendario que terminan
en la decisión, con el logaritmo de cada cierre respecto al de la primera sesión presente y un
bit de presencia por paso (`price_windows`). Con t la sesión de la decisión y k las sesiones
de la reversión, salen dos componentes:

- La reversión de corto plazo, menos el rendimiento logarítmico de t − k a t (Lehmann, 1990).
- El momentum de la ventana, el rendimiento desde el cierre de anclaje hasta t − k, que salta
  las sesiones de la reversión como el momentum de Jegadeesh y Titman (1993) salta el último
  mes. La ventana solo cubre unos tres meses, así que no es su momentum de doce meses.

En cada sesión de cada mercado, cada componente se convierte en rangos centrados en
[−0,5, 0,5] y la predicción es la media de los componentes definidos. Un componente que
necesita una posición ausente no cuenta, y una fila sin ninguno vale cero. La regla no ajusta
ningún parámetro: la escala del MAE calibrado la fija el λ de cada mercado en la calibración,
como en el resto de modelos.
"""

import numpy as np
from scipy.stats import rankdata

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRESENCE_CHANNEL, PRICE_WINDOW_CHANNELS

IDENTITY = "momentum_reversal_v1"
CLOSE = PRICE_WINDOW_CHANNELS.index("log_close_to_anchor_close")
PRESENT = PRICE_WINDOW_CHANNELS.index(PRESENCE_CHANNEL)
COMBINATION = "mean_of_centered_session_ranks"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_config(config):
    """Declaración de la regla con las sesiones de la reversión."""
    _require(
        isinstance(config, dict)
        and set(config) == {"identity", "reversal_sessions", "combination"}
        and config["identity"] == IDENTITY
        and config["combination"] == COMBINATION
        and type(config["reversal_sessions"]) is int
        and 1 <= config["reversal_sessions"] < CONTEXT_SESSIONS - 1,
        f"La regla declara {IDENTITY}, sus sesiones de reversión y su combinación",
    )
    return config


def components(prices, reversal_sessions):
    """Reversión y momentum de cada ventana, con la máscara de los que están definidos."""
    prices = np.asarray(prices)
    _require(
        prices.ndim == 3 and prices.shape[1:] == (CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS)),
        "La regla lee ventanas de 64 sesiones con sus canales y la presencia",
    )
    _require(np.isfinite(prices).all(), "La ventana de precios contiene valores no finitos")
    presence = prices[:, :, PRESENT]
    _require(np.isin(presence, (0.0, 1.0)).all(), "El canal de presencia vale cero o uno")
    close, present = prices[:, :, CLOSE].astype(np.float64), presence == 1.0
    last, lag = CONTEXT_SESSIONS - 1, CONTEXT_SESSIONS - 1 - reversal_sessions
    # El cierre de anclaje vale cero por construcción, así que el momentum es el cierre en
    # t − k respecto a él.
    reversal = -(close[:, last] - close[:, lag])
    return dict(
        reversal=(reversal, present[:, last] & present[:, lag]),
        momentum=(close[:, lag], present[:, lag]),
    )


def centered_ranks(values, defined, sessions):
    """Rango medio de cada valor definido en su sesión, llevado a [−0,5, 0,5].

    Los empates reciben su rango medio. Una sesión con un solo valor definido le da cero, y
    las filas sin valor también valen cero.
    """
    result = np.zeros(len(values))
    order = np.argsort(sessions, kind="stable")
    bounds = np.flatnonzero(np.r_[True, sessions[order][1:] != sessions[order][:-1], True])
    for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
        rows = order[begin:end]
        rows = rows[defined[rows]]
        if len(rows) > 1:
            result[rows] = (rankdata(values[rows]) - 1) / (len(rows) - 1) - 0.5
    return result


def predict(prices, sessions, config):
    """Predicción de la regla para filas con su ventana y el identificador de su sesión.

    `sessions` identifica la sesión de cada fila dentro de su mercado, de modo que dos
    mercados nunca comparten sesión.
    """
    config = checked_config(config)
    sessions = np.asarray(sessions)
    parts = components(prices, config["reversal_sessions"])
    _require(sessions.shape == (len(parts["reversal"][0]),), "Cada fila necesita su sesión")
    total, count = np.zeros(len(sessions)), np.zeros(len(sessions))
    for values, defined in parts.values():
        total += centered_ranks(values, defined, sessions)
        count += defined
    return np.divide(total, count, out=np.zeros(len(sessions)), where=count > 0)
