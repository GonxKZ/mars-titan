"""Abstención presentada con su cobertura (#442).

Una predicción puntual cero es una abstención. La exactitud de dirección ya la cuenta como
fallo, pero las precisiones de subida y de bajada solo se calculan sobre las llamadas, así
que un modelo que casi siempre se abstiene puede mostrar una precisión alta con muy pocas
llamadas. Estas reglas quedan fijadas en el código antes de evaluar y cada informe las
publica:

- La curva riesgo-cobertura de dirección (`forecast_scores.SELECTIVE_COVERAGES`): en cada
  sesión, la exactitud entre las ceil(c · n) filas con objetivo no nulo y mayor
  |predicción|. Una predicción cero no acierta nunca, así que abstenerse no la mejora.
  `FIXED_COVERAGE` es la cobertura cuyo valor se presenta junto al resto de métricas.
- La cobertura mínima de llamadas `MIN_CALL_COVERAGE`. Un contraste de una métrica
  condicionada a las llamadas (`CALL_CONDITIONED`) con algún modelo por debajo se marca
  no elegible: se publica con su cobertura, pero no puede presentarse como ganado. La
  cobertura de llamadas de un modelo es la fracción de filas con objetivo no nulo en las
  que su predicción no es cero, sumada sobre todas las sesiones de la vista y sus semillas.

El MAE y el MSE puntúan todas las filas, así que abstenerse en todas equivale al control
cero, que ya se compara de forma explícita.
"""

from fractions import Fraction

from .forecast_scores import SELECTIVE_ACCURACY, SELECTIVE_COVERAGES, SELECTIVE_RULE

FIXED_COVERAGE = 0.5
MIN_CALL_COVERAGE = 0.5
CALL_CONDITIONED = ("up_precision", "down_precision")
FIXED_METRIC = f"{SELECTIVE_ACCURACY}{FIXED_COVERAGE:g}"
DECLARED_AT = "2026-10-13"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


_require(FIXED_COVERAGE in SELECTIVE_COVERAGES, "La cobertura fija debe estar en la curva")


def declaration():
    """Reglas de abstención que publica cada informe."""
    return dict(
        declared_at=DECLARED_AT,
        selective_rule=SELECTIVE_RULE,
        coverages=list(SELECTIVE_COVERAGES),
        fixed_coverage=FIXED_COVERAGE,
        fixed_metric=FIXED_METRIC,
        min_call_coverage=MIN_CALL_COVERAGE,
        call_conditioned_metrics=list(CALL_CONDITIONED),
    )


def call_coverage(calls, eligible):
    """Fracción de filas con objetivo no nulo en las que el modelo hace una llamada."""
    calls, eligible = int(calls), int(eligible)
    _require(0 <= calls <= eligible, "Las llamadas no pueden superar las filas juzgables")
    return None if eligible == 0 else calls / eligible


def eligible(coverage):
    """Si un modelo llama lo bastante para competir en una métrica condicionada."""
    return coverage is not None and Fraction(coverage) >= Fraction(MIN_CALL_COVERAGE)


def mark(entry, metric, coverage):
    """Marcar los contrastes de una familia según la cobertura de llamadas de sus modelos.

    `entry` es el resultado de `compare_series` y `coverage` la cobertura de cada modelo en
    la vista. Solo cambia los contrastes de las métricas condicionadas a las llamadas.
    """
    if metric not in CALL_CONDITIONED:
        return entry
    for row in entry["contrasts"]:
        arms = sorted(arm for arm, value in row["coefficients"].items() if value != 0)
        low = [arm for arm in arms if not eligible(coverage[arm])]
        row["call_coverage"] = {arm: coverage[arm] for arm in arms}
        row["eligible"] = not low
        row["ineligible_reason"] = (
            None
            if not low
            else f"Cobertura de llamadas inferior a {MIN_CALL_COVERAGE:g} en {', '.join(low)}"
        )
    return entry
