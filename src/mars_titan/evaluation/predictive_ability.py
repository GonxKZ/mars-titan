"""Contrastes secundarios de capacidad predictiva sobre pérdidas diarias.

La sección ``predictive_ability`` de la comparación walk-forward declara, antes de ver
ningún resultado, cinco análisis secundarios sobre la pérdida de cada brazo. Ninguno
sustituye a las comparaciones emparejadas de ``paired_comparisons``, que siguen siendo el
contraste principal y el único que decide una conclusión.

1. Diebold y Mariano (1995) con la corrección de Harvey, Leybourne y Newbold (1997) para
   cada diferencia variante menos base, con Holm dentro de la familia.
2. SPA de Hansen (2005) con la base de la familia como referencia y sus tres recentrados.
3. Reality Check de White (2000), que es el p-valor superior del mismo SPA.
4. StepM de Romano y Wolf (2005), que dice qué variantes superan a la base.
5. MCS de Hansen, Lunde y Nason (2011) sobre todos los brazos de la familia.

Además se calcula la longitud de bloque de Politis y White (2004), con la corrección de
Patton, Politis y White (2009), solo como diagnóstico. Nunca cambia la longitud declarada.

La unidad es el día UTC, la misma que remuestrea el bootstrap principal. La pérdida de un
brazo en un día es la media de sus sesiones de ese día con la ponderación entre mercados
declarada, después de promediar las semillas sesión a sesión. La media de esas pérdidas
diarias pesa igual cada día, así que puede diferir un poco de la media por sesiones del
contraste principal cuando un día tiene sesiones de los dos mercados. Los remuestreos usan
el bootstrap circular por bloques con la longitud, las réplicas y la semilla de la
comparación. Con la misma semilla, ``arch`` sortea exactamente los mismos bloques de días
que ``paired_comparisons``, y una prueba lo comprueba.

Detalles de ``arch`` 8.0.0 que cambian lo que se puede afirmar:

- Su SPA no estudentiza el estadístico aunque reciba ``studentize=True``. La corrección
  llegó después de la versión 8.0.0 (bashtage/arch#871) y su revisión sigue abierta
  (#879, #881). Aquí se pasa ``studentize=False`` para que el código haga exactamente lo
  declarado: la media de cada diferencial sin dividir por su desviación. La varianza solo
  interviene en el umbral del p-valor consistente, como en el artículo.
- ``RealityCheck`` es un alias de ``SPA``. Con el estadístico sin estudentizar y todos los
  modelos recentrados, el p-valor superior es el Reality Check de White.
- Su ``StepM`` falla si los modelos quedan todos seleccionados en pasos distintos
  (bashtage/arch#862). El descenso por pasos se hace aquí con la API pública de ``SPA`` y
  la condición de parada corregida en el repositorio de ``arch`` (#863).
- Su MCS divide por la varianza de cada par. Dos brazos con las mismas pérdidas diarias la
  anulan, así que esa familia queda sin MCS y el informe da el motivo.

Los p-valores no se corrigen entre familias, igual que en el contraste principal. Cada
familia responde a una pregunta declarada.

La declaración y sus constantes solo necesitan NumPy. ``arch`` y ``statsmodels`` pertenecen
al extra ``research`` y se importan dentro de las funciones que calculan, así que validar
o cargar una configuración de campaña no los necesita. La evaluación llama a
``library_versions`` al empezar para que un entorno sin ese extra falle antes de leer
ninguna fuente.
"""

import importlib
import math
from datetime import date
from importlib.metadata import version

import numpy as np

from mars_titan.evaluation.forecast_panel import WEIGHTINGS, SessionSeries
from mars_titan.evaluation.forecast_scores import is_loss_series

KIND = "daily_loss_predictive_ability_v1"
STATUS = "secondary_inferential"
USE = "secondary_declared_before_results_not_for_model_selection"
UNIT = "utc_day_mean_of_seed_averaged_sessions_with_declared_market_weighting"
RESAMPLING = "circular_block_bootstrap_with_comparison_block_length_replicates_and_seed"
# El objetivo es el retorno de apertura a cierre de la sesión siguiente (configs/study.toml).
HORIZON_SESSIONS = 1
DIEBOLD_MARIANO = "variant_minus_base_bartlett_hac_harvey_correction_holm_within_family"
HAC_LAGS = "max_horizon_minus_one_and_integer_ceiling_of_cube_root_of_days"
SUPERIOR_PREDICTIVE_ABILITY = "family_base_benchmark_unstudentized_lower_consistent_upper"
REALITY_CHECK = "upper_pvalue_of_unstudentized_spa"
BLOCK_LENGTH_DIAGNOSTIC = "politis_white_2004_patton_2009_diagnostic_only"
MCS_METHODS = ("R", "max")
LIBRARIES = ("arch", "statsmodels")
FIELDS = {
    "kind",
    "status",
    "declared_at",
    "use",
    "metrics",
    "unit",
    "min_days",
    "resampling",
    "horizon_sessions",
    "diebold_mariano",
    "hac_lags",
    "superior_predictive_ability",
    "reality_check",
    "stepm_size",
    "model_confidence_set",
    "block_length_diagnostic",
}
MAX_DAYS = 1_000_000
# El umbral del p-valor consistente usa log(log(días)), que solo es positivo desde tres días.
# Diebold-Mariano con la corrección de Harvey también necesita al menos tres.
MIN_DAYS = 3
# Misma tolerancia que los intervalos simultáneos para tratar una serie como constante.
_CONSTANT = 64 * np.finfo(np.float64).eps
_NUMERIC = dict(divide="raise", over="raise", invalid="raise")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _probability(value, upper):
    return isinstance(value, float) and math.isfinite(value) and 0 < value <= upper


def library_versions():
    """Importar ``arch`` y ``statsmodels`` y devolver sus versiones instaladas.

    Sin el extra ``research`` lanza ``ModuleNotFoundError`` con el nombre del paquete.
    """
    for name in LIBRARIES:
        importlib.import_module(name)
    return {name: version(name) for name in LIBRARIES}


def declaration(section, comparison):
    """Validar la sección declarada frente a la comparación, sin leer datos, y devolverla."""
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La sección predictive_ability no declara exactamente sus campos",
    )
    _require(
        section["kind"] == KIND
        and section["status"] == STATUS
        and section["use"] == USE
        and section["unit"] == UNIT
        and section["resampling"] == RESAMPLING
        and section["horizon_sessions"] == HORIZON_SESSIONS
        and section["diebold_mariano"] == DIEBOLD_MARIANO
        and section["hac_lags"] == HAC_LAGS
        and section["superior_predictive_ability"] == SUPERIOR_PREDICTIVE_ABILITY
        and section["reality_check"] == REALITY_CHECK
        and section["block_length_diagnostic"] == BLOCK_LENGTH_DIAGNOSTIC,
        "Las reglas de capacidad predictiva no son las declaradas en el protocolo",
    )
    declared_at = section["declared_at"]
    _require(isinstance(declared_at, str), "La fecha de declaración no es válida")
    try:
        date.fromisoformat(declared_at)
    except ValueError as error:
        raise ValueError("La fecha de declaración no es válida") from error
    metrics = section["metrics"]
    _require(
        isinstance(metrics, list)
        and metrics
        and metrics[0] == "mae"
        and len(set(metrics)) == len(metrics)
        and all(metric in comparison["metrics"] and is_loss_series(metric) for metric in metrics),
        "Las métricas deben empezar por el MAE y ser pérdidas de la comparación",
    )
    minimum = section["min_days"]
    _require(
        type(minimum) is int
        and max(MIN_DAYS, comparison["block_length"] + 1) <= minimum <= MAX_DAYS,
        "El mínimo de días debe ser al menos 3 y superar la longitud de bloque",
    )
    mcs = section["model_confidence_set"]
    _require(
        _probability(section["stepm_size"], 0.5)
        and isinstance(mcs, dict)
        and set(mcs) == {"size", "method"}
        and _probability(mcs["size"], 0.5)
        and mcs["method"] in MCS_METHODS,
        "Los tamaños de StepM y del MCS o el método del MCS no son válidos",
    )
    return section


def _day_means(series, periods, weighting):
    """Pérdida media de cada día UTC y si el día tiene alguna sesión definida."""
    defined = series.defined.astype(np.float64)
    values = np.where(series.defined, series.values, 0.0)
    if weighting == "session":
        sums = np.bincount(series.session_period, weights=values, minlength=periods)
        counts = np.bincount(series.session_period, weights=defined, minlength=periods)
        present = counts > 0
        return np.divide(sums, counts, out=np.zeros(periods), where=present), present
    markets = len(series.markets)
    cell = series.session_period * markets + series.session_market
    shape = (periods, markets)
    sums = np.bincount(cell, weights=values, minlength=periods * markets).reshape(shape)
    counts = np.bincount(cell, weights=defined, minlength=periods * markets).reshape(shape)
    means = np.divide(sums, counts, out=np.zeros(shape), where=counts > 0)
    used = np.count_nonzero(counts > 0, axis=1)
    present = used > 0
    return np.divide(means.sum(axis=1), used, out=np.zeros(periods), where=present), present


def daily_losses(series, weighting):
    """Matriz [días, brazos] de pérdidas diarias en el orden de ``series``.

    Un día solo entra si todos los brazos tienen alguna sesión definida en él. Devuelve
    la matriz y el número de días excluidos por ese motivo.
    """
    _require(weighting in WEIGHTINGS, "Ponderación entre mercados no declarada")
    _require(
        isinstance(series, dict)
        and len(series) >= 2
        and all(isinstance(item, SessionSeries) for item in series.values()),
        "Se necesitan al menos dos series por sesión",
    )
    first = next(iter(series.values()))
    _require(
        all(
            item.loss
            and item.metric == first.metric
            and item.cohort_sha256 == first.cohort_sha256
            and item.markets == first.markets
            and np.array_equal(item.session_period, first.session_period)
            and np.array_equal(item.session_market, first.session_market)
            for item in series.values()
        ),
        "Las pérdidas diarias necesitan la misma pérdida sobre las mismas sesiones",
    )
    periods = int(first.session_period.max()) + 1
    columns, present = [], np.ones(periods, dtype=bool)
    for item in series.values():
        means, defined = _day_means(item, periods, weighting)
        columns.append(means)
        present &= defined
    return np.column_stack(columns)[present], int(np.count_nonzero(~present))


def _varies(values):
    """Una serie es constante si su recorrido es despreciable frente a su escala."""
    return bool(np.ptp(values) > _CONSTANT * np.max(np.abs(values)))


def hac_lags(days, horizon=HORIZON_SESSIONS):
    """Retardos del estimador de Newey y West: max(h − 1, ⌈días^(1/3)⌉).

    Es la regla por defecto de ``statsmodels``. Se pasa de forma explícita para que un
    cambio de esa regla no altere el contraste declarado, y la raíz se redondea con
    aritmética entera para que el resultado no dependa de la coma flotante.
    """
    _require(type(days) is int and days >= 1, "El número de días debe ser positivo")
    root = round(days ** (1 / 3))
    while root**3 < days:
        root += 1
    while root > 1 and (root - 1) ** 3 >= days:
        root -= 1
    return max(horizon - 1, root)


def _given_loss(target, forecast):
    """Criterio de ``statsmodels`` que devuelve la pérdida ya calculada que recibe."""
    return forecast - target


def diebold_mariano(variant, base, *, horizon=HORIZON_SESSIONS):
    """Diebold-Mariano con Harvey sobre d = pérdida de la variante − pérdida de la base.

    Un estadístico positivo indica que la variante pierde más que la base. Devuelve el
    motivo en lugar del contraste si el diferencial es constante, porque su varianza de
    largo plazo es nula y el estadístico no existe.
    """
    variant, base = np.asarray(variant, dtype=np.float64), np.asarray(base, dtype=np.float64)
    _require(
        variant.ndim == 1 and variant.shape == base.shape and len(variant) >= 3,
        "Diebold-Mariano necesita dos series diarias alineadas de al menos tres días",
    )
    differential = variant - base
    row = dict(days=len(variant), mean_differential=float(np.mean(differential)))
    if not _varies(differential):
        return dict(row, reason="El diferencial diario es constante")
    from statsmodels.tsa.stattools import diebold_mariano_test

    lags = hac_lags(len(variant), horizon)
    with np.errstate(**_NUMERIC):
        result = diebold_mariano_test(
            np.zeros(len(variant)),
            variant,
            base,
            lags=lags,
            criterion=_given_loss,
            harvey_adj=True,
            horizon=horizon,
        )
    return dict(
        row,
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        lags=int(result.lags),
        harvey_factor=float(result.harvey_adj_factor),
        reference_distribution=f"student_t_{len(variant) - 1}_degrees_of_freedom",
    )


def holm(rows):
    """Añadir a cada fila contrastada su p-valor de Holm dentro de la familia."""
    from statsmodels.stats.multitest import multipletests

    tested = [row for row in rows if "pvalue" in row]
    if tested:
        adjusted = multipletests([row["pvalue"] for row in tested], method="holm")[1]
        for row, value in zip(tested, adjusted, strict=True):
            row["holm_pvalue"] = float(value)
    return rows


def _spa(benchmark, models, block_length, replicates, seed):
    from arch.bootstrap import SPA

    return SPA(
        benchmark,
        models,
        block_size=block_length,
        reps=replicates,
        bootstrap="circular",
        studentize=False,
        nested=False,
        seed=seed,
    )


def stepwise_superior(spa, size):
    """Índices de los modelos mejores que la referencia con el StepM de Romano y Wolf.

    Repite el SPA sin los modelos ya seleccionados hasta que no aparece ninguno nuevo o
    están todos. Es el bucle de ``arch.bootstrap.StepM`` con la condición de parada de
    bashtage/arch#863, que la versión 8.0.0 no tiene. Al acabar restaura todos los modelos.
    """
    count = spa.k
    spa.compute()
    better = [int(index) for index in spa.better_models(size)]
    selected = list(better)
    while better and len(selected) < count:
        selector = np.ones(count, dtype=bool)
        selector[selected] = False
        spa.subset(selector)
        spa.compute()
        better = [int(index) for index in spa.better_models(size)]
        selected.extend(better)
    spa.subset(np.ones(count, dtype=bool))
    spa.compute()
    return sorted(selected)


def superior_predictive_ability(benchmark, models, *, block_length, replicates, seed, size):
    """SPA, Reality Check y StepM con la base como referencia y los mismos remuestreos.

    ``models`` es una matriz [días, variantes]. Los p-valores son los de ``arch`` con el
    estadístico sin estudentizar. StepM reutiliza las réplicas del mismo SPA.
    """
    benchmark = np.asarray(benchmark, dtype=np.float64)
    models = np.asarray(models, dtype=np.float64)
    _require(
        benchmark.ndim == 1
        and models.ndim == 2
        and models.shape[0] == len(benchmark)
        and models.shape[1] >= 1
        and block_length < len(benchmark),
        "El SPA necesita una referencia, al menos una variante y más días que el bloque",
    )
    constant = [
        index for index in range(models.shape[1]) if not _varies(benchmark - models[:, index])
    ]
    if constant:
        return dict(reason="Alguna variante tiene un diferencial diario constante con la base")
    with np.errstate(**_NUMERIC):
        spa = _spa(benchmark, models, block_length, replicates, seed)
        spa.compute()
        pvalues = {name: float(value) for name, value in spa.pvalues.items()}
        superior = stepwise_superior(spa, size)
    return dict(
        pvalues=pvalues,
        reality_check_pvalue=pvalues["upper"],
        stepm_superior=superior,
    )


def model_confidence_set(losses, *, size, method, block_length, replicates, seed):
    """MCS de ``arch`` sobre una matriz [días, modelos], o el motivo de no calcularlo.

    Devuelve los índices incluidos, el p-valor de cada modelo y el orden de eliminación.
    """
    losses = np.asarray(losses, dtype=np.float64)
    _require(
        losses.ndim == 2 and losses.shape[1] >= 2 and block_length < losses.shape[0],
        "El MCS necesita al menos dos modelos y más días que el bloque",
    )
    columns = losses.shape[1]
    for first in range(columns):
        for second in range(first + 1, columns):
            if not _varies(losses[:, first] - losses[:, second]):
                return dict(reason="Dos brazos tienen un diferencial diario constante")
    from arch.bootstrap import MCS

    with np.errstate(**_NUMERIC):
        mcs = MCS(
            losses,
            size=size,
            reps=replicates,
            block_size=block_length,
            method=method,
            bootstrap="circular",
            seed=seed,
        )
        mcs.compute()
    table = mcs.pvalues
    order = [int(index) for index in table.index]
    values = [float(value) for value in table["Pvalue"]]
    # Un empate exacto en el estadístico hace que arch elimine dos modelos a la vez.
    if sorted(order) != list(range(columns)) or not all(0 <= v <= 1 for v in values):
        return dict(reason="El MCS no devolvió un p-valor por modelo")
    return dict(
        included=sorted(int(index) for index in mcs.included),
        pvalues=dict(zip(order, values, strict=True)),
        elimination_order=order,
    )


def block_length_diagnostic(differential, declared):
    """Longitudes de Politis y White para el bootstrap circular y el estacionario.

    Es un diagnóstico. La comparación conserva la longitud declarada antes de evaluar.
    """
    differential = np.asarray(differential, dtype=np.float64)
    _require(differential.ndim == 1, "El diagnóstico necesita una serie diaria")
    if not _varies(differential):
        return dict(reason="El diferencial diario es constante")
    from arch.bootstrap import optimal_block_length

    try:
        with np.errstate(**_NUMERIC):
            table = optimal_block_length(differential)
    except FloatingPointError:
        return dict(reason="La serie es demasiado corta para estimar las autocorrelaciones")
    circular = float(table["circular"].iloc[0])
    stationary = float(table["stationary"].iloc[0])
    if not (math.isfinite(circular) and math.isfinite(stationary)):
        return dict(reason="La estimación de la longitud no es finita")
    return dict(
        circular=circular,
        stationary=stationary,
        declared=declared,
        declared_below_circular=declared < circular,
    )


def family_structure(contrasts):
    """Clasificar una familia de ``compare_series`` para elegir sus análisis.

    Devuelve los brazos en orden de aparición, los pares variante y base de los
    contrastes con coeficientes +1 y −1, y la base común si todos esos pares la comparten.
    Las interacciones no forman pares. Una familia de niveles no tiene pares ni base.
    """
    arms, pairs = [], []
    for name, coefficients in contrasts.items():
        for arm in coefficients:
            if arm not in arms:
                arms.append(arm)
        values = {arm: float(value) for arm, value in coefficients.items()}
        if len(values) == 2 and sorted(values.values()) == [-1.0, 1.0]:
            variant = next(arm for arm, value in values.items() if value == 1.0)
            base = next(arm for arm, value in values.items() if value == -1.0)
            pairs.append((name, variant, base))
    levels = all(len(coefficients) == 1 for coefficients in contrasts.values())
    bases = {base for _, _, base in pairs}
    common = next(iter(bases)) if pairs and len(bases) == 1 else None
    return dict(arms=arms, pairs=pairs, base=common, levels=levels)


def _family(structure, losses, section, options):
    """Análisis de una familia sobre la matriz [días, brazos] en el orden de sus brazos."""
    arms = structure["arms"]
    column = {arm: losses[:, index] for index, arm in enumerate(arms)}
    block_length = options["block_length"]
    resampling = dict(
        block_length=block_length, replicates=options["replicates"], seed=options["seed"]
    )
    result = dict(
        mean_daily_loss={arm: float(np.mean(values)) for arm, values in column.items()},
        diebold_mariano=None,
        block_length_diagnostic=None,
        superior_predictive_ability=None,
        model_confidence_set=None,
    )
    if structure["pairs"]:
        rows = []
        diagnostics = {}
        for name, variant, base in structure["pairs"]:
            row = diebold_mariano(column[variant], column[base], horizon=HORIZON_SESSIONS)
            rows.append(dict(name=name, variant=variant, base=base, **row))
            diagnostics[name] = block_length_diagnostic(
                column[variant] - column[base], block_length
            )
        result["diebold_mariano"] = holm(rows)
        result["block_length_diagnostic"] = diagnostics
    base = structure["base"]
    if base is not None:
        variants = [variant for _, variant, _ in structure["pairs"]]
        spa = superior_predictive_ability(
            column[base],
            np.column_stack([column[variant] for variant in variants]),
            size=section["stepm_size"],
            **resampling,
        )
        if "reason" not in spa:
            spa = dict(
                benchmark=base,
                variants=variants,
                pvalues=spa["pvalues"],
                reality_check_pvalue=spa["reality_check_pvalue"],
                stepm_size=section["stepm_size"],
                stepm_superior=[variants[index] for index in spa["stepm_superior"]],
            )
        result["superior_predictive_ability"] = spa
    if base is not None or structure["levels"]:
        mcs = model_confidence_set(
            losses,
            size=section["model_confidence_set"]["size"],
            method=section["model_confidence_set"]["method"],
            **resampling,
        )
        if "reason" not in mcs:
            mcs = dict(
                size=section["model_confidence_set"]["size"],
                method=section["model_confidence_set"]["method"],
                included=[arms[index] for index in mcs["included"]],
                pvalues={arms[index]: value for index, value in mcs["pvalues"].items()},
                elimination_order=[arms[index] for index in mcs["elimination_order"]],
            )
        result["model_confidence_set"] = mcs
    return result


def report(section, comparison, weighting, families, views, series):
    """Calcular la sección declarada para cada vista, métrica y familia.

    ``families`` son las familias de ``compare_series`` del ámbito evaluado. ``series``
    es una función ``(brazo, vista, métrica)`` que devuelve la SessionSeries con las
    semillas ya promediadas, o None si el brazo no emite esa métrica.
    """
    options = {key: comparison[key] for key in ("block_length", "replicates", "seed")}
    result = dict(declaration=section, resampling=dict(method="circular", **options), views={})
    for view in views:
        result["views"][view] = {}
        for metric in section["metrics"]:
            rows = {}
            for name, contrasts in families.items():
                structure = family_structure(contrasts)
                collected = {arm: series(arm, view, metric) for arm in structure["arms"]}
                if len(collected) < 2:
                    rows[name] = dict(reason="La familia tiene un solo brazo")
                    continue
                if any(item is None for item in collected.values()):
                    rows[name] = dict(reason="Algún brazo de la familia no emite esta métrica")
                    continue
                losses, excluded = daily_losses(collected, weighting)
                counts = dict(arms=structure["arms"], days=len(losses), excluded_days=excluded)
                if len(losses) < section["min_days"]:
                    rows[name] = dict(counts, reason="Hay menos días que el mínimo declarado")
                    continue
                rows[name] = dict(counts, **_family(structure, losses, section, options))
            result["views"][view][metric] = rows
    return result
