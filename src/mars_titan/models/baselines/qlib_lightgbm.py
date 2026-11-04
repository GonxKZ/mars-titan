"""LightGBM con los parámetros de Qlib sobre Alpha158, común a los controles tabulares.

Los parámetros son los del modelo LightGBM (Ke et al., 2017) de Qlib (Yang et al., 2020) sobre
Alpha158 en el commit fijado de `QLIB_SOURCE`. `LGBModel` toma `loss: mse` como objetivo,
añade `verbosity: -1` y ajusta 1.000 rondas. El manejador Alpha158 no normaliza las
características y entrena con la etiqueta normalizada en cada sesión (`DropnaLabel` y
`CSZScoreNorm`, la media y la desviación con ddof = 1 de cada fecha). Aquí se repite esa
normalización sobre el objetivo de cada control. Sin ella, las penalizaciones L1 y L2 de Qlib,
pensadas para etiquetas de varianza unidad, anularían las hojas con residuos del orden de 0,02.

Lo usan el control con las mismas entradas que los modelos de la criba (#746) y el techo de
señal con anomalías conocidas (#747). Cada uno declara su identidad, sus características y su
etiqueta, y `checked_config` exige esa declaración. Los dos cambian lo mismo frente a Qlib:

- Usa 8 hilos, los núcleos físicos del equipo, con `deterministic` y `force_col_wise` para que
  un ajuste se repita con la misma semilla.
- La semilla de LightGBM es cada una de las cinco de la criba.
- El número de rondas no sale de la parada temprana de Qlib (50 rondas sin mejorar la pérdida
  de validación), sino de la regla de la época común de la criba: cada `checkpoint_rounds`
  rondas se puntúa la validación con el IC por sesión del objetivo y `selection.common_epoch`
  elige el punto común a las semillas.

`subsample` no tiene efecto sin `bagging_freq`, igual que en Qlib. `fit` es el único punto de
aprendizaje del módulo y comprueba la protección antes de construir nada.
"""

import numpy as np

from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.training.learning_hold import require_learning_allowed

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
KEYS = {"identity", "source", "parameters", "max_rounds", "checkpoint_rounds", "features", "label"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _declares_training(config):
    """Los campos que usa el ajuste: fuente, parámetros con sus tipos, rondas y puntos."""
    return (
        isinstance(config, dict)
        and set(config) == KEYS
        and config["source"] == QLIB_SOURCE
        and config["parameters"] == PARAMETERS
        and all(type(config["parameters"][key]) is type(value) for key, value in PARAMETERS.items())
        and config["max_rounds"] == QLIB_ROUNDS
        and type(config["checkpoint_rounds"]) is int
        and 1 <= config["checkpoint_rounds"] <= QLIB_ROUNDS
        and QLIB_ROUNDS % config["checkpoint_rounds"] == 0
    )


def checked_config(config, *, identity, features, label):
    """Declaración de un control: los parámetros de Qlib, sus puntos de control y sus entradas."""
    _require(
        _declares_training(config)
        and config["identity"] == identity
        and config["features"] == features
        and config["label"] == label,
        f"LightGBM declara {identity} con los parámetros de Qlib, sus cambios, sus puntos de "
        "control y sus entradas",
    )
    return config


def _checked_training(config):
    """El ajuste solo exige lo que usa. Cada control comprueba antes su propia declaración."""
    _require(
        _declares_training(config),
        "LightGBM se ajusta con los parámetros de Qlib, sus cambios y sus puntos de control",
    )
    return config


def session_labels(panel):
    """Objetivo normalizado en cada sesión, como `CSZScoreNorm` de Qlib, y las filas que lo tienen.

    Una sesión con menos de dos filas o con el objetivo constante no tiene desviación. Qlib le
    dejaría la etiqueta vacía, así que sus filas no entran en el ajuste.
    """
    labels = np.zeros(panel.rows)
    kept = np.zeros(panel.rows, dtype=bool)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for start, count in zip(panel.session_starts, panel.session_samples, strict=True):
                target = panel.target[start : start + count]
                if count < 2:
                    continue
                spread = np.std(target, ddof=1)
                if spread > 0:
                    labels[start : start + count] = (target - target.mean()) / spread
                    kept[start : start + count] = True
    except FloatingPointError as error:
        raise ValueError("La normalización por sesión necesita estadísticas finitas") from error
    return labels, kept


def fit(train, validation, config, seed, *, min_assets, check_stop=None):
    """Ajustar con una semilla y puntuar la validación en cada punto de control.

    `train` y `validation` llevan su panel con el objetivo en orden canónico y sus
    características en el mismo orden. Devuelve el booster y la serie de pares (valores,
    definidas) del IC por sesión de cada punto. La predicción puntuada es la misma que emitirá
    `predict` con ese número de rondas. `check_stop` puede interrumpir antes de construir el
    conjunto y al volver de cada ronda, sin devolver un booster parcial.
    """
    require_learning_allowed("un control LightGBM con los parámetros de Qlib")
    config = _checked_training(config)
    _require(type(seed) is int and 0 <= seed < 2**31, "La semilla de LightGBM no es válida")
    if check_stop is not None:
        check_stop()
    import lightgbm

    labels, kept = session_labels(train.panel)
    _require(kept.any(), "Ninguna sesión del ajuste tiene una sección cruzada con desviación")
    dataset = lightgbm.Dataset(train.features[kept], label=labels[kept], free_raw_data=True)
    step, series = config["checkpoint_rounds"], []

    def score(environment):
        if check_stop is not None:
            check_stop()
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


def model_text(booster):
    """Texto del booster con todas sus rondas, que `load` reconstruye sin ajustar nada."""
    return booster.model_to_string(num_iteration=-1)


def load(text):
    """Booster guardado con `model_text`, para reanudar una ventana sin repetir su ajuste."""
    import lightgbm

    return lightgbm.Booster(model_str=text)


def predict(booster, values, rounds):
    """Predicción float64 con las `rounds` primeras rondas."""
    prediction = np.asarray(booster.predict(values, num_iteration=rounds), dtype=np.float64)
    _require(
        prediction.shape == (len(values),) and np.isfinite(prediction).all(),
        "LightGBM no devuelve una predicción finita por fila",
    )
    return prediction
