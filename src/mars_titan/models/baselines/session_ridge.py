"""Ridge sobre la etiqueta normalizada en cada sesión, con alpha elegido por el IC de validación.

Es el modelo lineal del techo de señal (#747). Usa `Ridge` de scikit-learn (Hoerl y Kennard,
1970) con intercepto sin penalizar y la solución de Cholesky, que no depende de semillas. La
etiqueta es la de `qlib_lightgbm.session_labels`, el objetivo normalizado en cada sesión, la
misma con la que se ajusta LightGBM. Las características no se estandarizan: el control las
declara en una escala común, como los rangos en [−1, 1] y los indicadores de ausencia.

Cada alpha declarado se ajusta con las mismas filas y la validación elige el de mayor IC medio
por sesión. Un empate se resuelve con el alpha mayor, el que más contrae los coeficientes.
`fit` es el único punto de aprendizaje del módulo y comprueba la protección antes de nada.
"""

import math
from itertools import pairwise

import numpy as np

from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.training.learning_hold import require_learning_allowed

from .qlib_lightgbm import session_labels

IDENTITY = "session_ridge_v1"
SOLVER = "cholesky"
KEYS = {"identity", "alphas", "solver", "features", "label"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _declares_training(config):
    alphas = config.get("alphas") if isinstance(config, dict) else None
    return (
        isinstance(config, dict)
        and set(config) == KEYS
        and config["identity"] == IDENTITY
        and isinstance(alphas, list)
        and len(alphas) >= 1
        and all(type(alpha) is float and math.isfinite(alpha) and alpha > 0 for alpha in alphas)
        and all(left < right for left, right in pairwise(alphas))
        and config["solver"] == SOLVER
    )


def checked_config(config, *, features, label):
    """Declaración de Ridge con sus alphas crecientes, la solución de Cholesky y sus entradas."""
    _require(
        _declares_training(config) and config["features"] == features and config["label"] == label,
        f"Ridge declara {IDENTITY} con alphas positivos crecientes, la solución de Cholesky y "
        "sus entradas",
    )
    return config


def predict(model, values):
    """Predicción float64 del modelo ajustado, con un valor finito por fila."""
    prediction = np.asarray(model.predict(values), dtype=np.float64)
    _require(
        prediction.shape == (len(values),) and np.isfinite(prediction).all(),
        "Ridge no devuelve una predicción finita por fila",
    )
    return prediction


def fit(train, validation, config, *, min_assets):
    """Ajustar cada alpha y quedarse con el de mayor IC medio de validación por sesión.

    `train` y `validation` llevan su panel con el objetivo en orden canónico y sus
    características en el mismo orden. Devuelve el modelo elegido y la selección, con el IC
    medio y las sesiones definidas de cada alpha.
    """
    require_learning_allowed("el control Ridge con la etiqueta normalizada por sesión")
    _require(
        _declares_training(config),
        f"Ridge se ajusta con {IDENTITY}, alphas positivos crecientes y la solución de Cholesky",
    )
    from sklearn.linear_model import Ridge

    labels, kept = session_labels(train.panel)
    _require(kept.any(), "Ninguna sesión del ajuste tiene una sección cruzada con desviación")
    values = np.asarray(train.features, dtype=np.float64)[kept]
    candidates = []
    for alpha in config["alphas"]:
        model = Ridge(alpha=alpha, fit_intercept=True, solver=SOLVER).fit(values, labels[kept])
        correlations, defined = sd.session_correlations(
            validation.panel, predict(model, validation.features), min_assets=min_assets
        )
        sessions = int(defined.sum())
        _require(sessions, "La validación no define el IC en ninguna sesión")
        candidates.append((math.fsum(correlations[defined]) / sessions, alpha, sessions, model))
    score, alpha, _, model = max(candidates, key=lambda item: (item[0], item[1]))
    return model, dict(
        alpha=alpha,
        validation_ic=score,
        fit_rows=int(kept.sum()),
        scores=[dict(alpha=a, validation_ic=s, sessions=n) for s, a, n, _ in candidates],
    )
