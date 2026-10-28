"""LightGBM con las mismas entradas que los modelos, control común de la criba (#742, #746).

Los parámetros son los del modelo LightGBM (Ke et al., 2017) de Qlib (Yang et al., 2020) sobre
Alpha158 en el commit fijado de `QLIB_SOURCE`. `LGBModel` toma `loss: mse` como objetivo,
añade `verbosity: -1` y ajusta 1.000 rondas. El manejador Alpha158 no normaliza las
características y entrena con la etiqueta normalizada en cada sesión (`DropnaLabel` y
`CSZScoreNorm`, la media y la desviación con ddof = 1 de cada fecha). Aquí se repite esa
normalización sobre el residuo de la criba. Sin ella, las penalizaciones L1 y L2 de Qlib, pensadas
para etiquetas de varianza unidad, anularían las hojas con residuos del orden de 0,02.

Cambia lo siguiente, declarado en el protocolo:

- Las características son las entradas de la criba de cada fila (`features`), no las 158 de
  Alpha158.
- Usa 8 hilos, los núcleos físicos del equipo, con `deterministic` y `force_col_wise` para que
  un ajuste se repita con la misma semilla.
- La semilla de LightGBM es cada una de las cinco de la criba.
- El número de rondas no sale de la parada temprana de Qlib (50 rondas sin mejorar la pérdida
  de validación), sino de la regla de la época común de la criba: cada `checkpoint_rounds`
  rondas se puntúa la validación con el IC por sesión del residuo y `selection.common_epoch`
  elige el punto común a las semillas.

`subsample` no tiene efecto sin `bagging_freq`, igual que en Qlib. `fit` es el único punto de
aprendizaje del módulo y comprueba la protección antes de construir nada.
"""

import numpy as np

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.training.learning_hold import require_learning_allowed

from .inputs import MODALITIES

IDENTITY = "same_inputs_lightgbm_v1"
QLIB_SOURCE = dict(
    repository="microsoft/qlib",
    commit="54355232463878d2eebb91fe0ee5fa7fa1f5976c",
    workflow="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
    workflow_sha256="0cbe1f729f43a2b8cd735759e6244c2e9f8db967555bfe24a7143fce4605d680",
    model="qlib/contrib/model/gbdt.py",
    model_sha256="a1afddb3bb965d108a916565b57e474cfddc55ce029f781765876e825623ad39",
    handler="qlib/contrib/data/handler.py",
    handler_sha256="b621481c6009c39066c67c71390fd2bea635f56daf9f2c4e38817eff268e3232",
)
QLIB_PARAMETERS = dict(
    objective="mse",
    verbosity=-1,
    colsample_bytree=0.8879,
    learning_rate=0.2,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=20,
)
QLIB_ROUNDS = 1000
CHANGES = dict(num_threads=8, deterministic=True, force_col_wise=True)
PARAMETERS = QLIB_PARAMETERS | CHANGES
FEATURES = "price_window_modalities_and_presence_v1"
LABEL = "session_zscore_of_the_residual"
VECTORS = MODALITIES[1:]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_config(config):
    """Declaración del control con los parámetros de Qlib, sus cambios y los puntos de control."""
    _require(
        isinstance(config, dict)
        and set(config)
        == {
            "identity",
            "source",
            "parameters",
            "max_rounds",
            "checkpoint_rounds",
            "features",
            "label",
        }
        and config["identity"] == IDENTITY
        and config["source"] == QLIB_SOURCE
        and config["parameters"] == PARAMETERS
        and all(type(config["parameters"][key]) is type(value) for key, value in PARAMETERS.items())
        and config["max_rounds"] == QLIB_ROUNDS
        and type(config["checkpoint_rounds"]) is int
        and 1 <= config["checkpoint_rounds"] <= QLIB_ROUNDS
        and QLIB_ROUNDS % config["checkpoint_rounds"] == 0
        and config["features"] == FEATURES
        and config["label"] == LABEL,
        f"LightGBM declara {IDENTITY} con los parámetros de Qlib, sus cambios, sus puntos de "
        "control y sus entradas",
    )
    return config


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


def session_labels(panel):
    """Residuo normalizado en cada sesión, como `CSZScoreNorm` de Qlib, y las filas que lo tienen.

    Una sesión con menos de dos filas o con el residuo constante no tiene desviación. Qlib le
    dejaría la etiqueta vacía, así que sus filas no entran en el ajuste.
    """
    labels = np.zeros(panel.rows)
    kept = np.zeros(panel.rows, dtype=bool)
    for start, count in zip(panel.session_starts, panel.session_samples, strict=True):
        target = panel.target[start : start + count]
        if count < 2:
            continue
        spread = np.std(target, ddof=1)
        if spread > 0:
            labels[start : start + count] = (target - target.mean()) / spread
            kept[start : start + count] = True
    return labels, kept


def fit(train, validation, config, seed, *, min_assets):
    """Ajustar con una semilla y puntuar la validación en cada punto de control.

    `train` y `validation` llevan su panel con el residuo en orden canónico y sus
    características en el mismo orden. Devuelve el booster y la serie de pares (valores,
    definidas) del IC por sesión de cada punto. La predicción puntuada es la misma que emitirá
    `predict` con ese número de rondas.
    """
    require_learning_allowed("el control LightGBM de la criba de variantes")
    config = checked_config(config)
    _require(type(seed) is int and 0 <= seed < 2**31, "La semilla de LightGBM no es válida")
    import lightgbm

    labels, kept = session_labels(train.panel)
    _require(kept.any(), "Ninguna sesión del ajuste tiene una sección cruzada con desviación")
    dataset = lightgbm.Dataset(train.features[kept], label=labels[kept], free_raw_data=True)
    step, series = config["checkpoint_rounds"], []

    def score(environment):
        rounds = environment.iteration + 1
        if rounds % step == 0:
            prediction = predict(environment.model, validation.features, rounds)
            series.append(
                sd.session_correlations(validation.panel, prediction, min_assets=min_assets)
            )

    booster = lightgbm.train(
        dict(config["parameters"], seed=seed),
        dataset,
        num_boost_round=config["max_rounds"],
        callbacks=[score],
    )
    _require(
        len(series) == config["max_rounds"] // step,
        "El ajuste no puntuó todos sus puntos de control",
    )
    return booster, series


def predict(booster, values, rounds):
    """Predicción float64 con las `rounds` primeras rondas."""
    prediction = np.asarray(booster.predict(values, num_iteration=rounds), dtype=np.float64)
    _require(
        prediction.shape == (len(values),) and np.isfinite(prediction).all(),
        "LightGBM no devuelve una predicción finita por fila",
    )
    return prediction
