"""Cota con oráculo de un paso para el entorno financiero. Es un diagnóstico.

El entorno decide al cierre de t, ejecuta en la apertura de t+1 y valora al cierre de t+1.
El oráculo conoce el rendimiento de esa sesión, `close[t+1] / open[t+1] - 1`, y lo usa
como puntuación con la misma regla de asignación del entorno (cuartil superior positivo
con pesos iguales y exposición completa). Con ello da el rendimiento que obtendría una
predicción perfecta de la sesión siguiente con esas reglas, costes y liquidez.

No es el máximo global: ignora el hueco nocturno hasta la siguiente apertura y no optimiza
la rotación. Sirve para dos cosas:
- poner en escala el resultado de una política: qué fracción de la cota alcanza;
- detectar trampas: una política que se acerca a la cota casi seguro usa información futura.

La cinta del contrato comprueba las marcas de tiempo de las predicciones, no su contenido,
así que una cinta con puntuaciones del futuro y marcas correctas pasaría su validación.
Esta cota ataca ese hueco por el resultado. Su cinta lleva un padre propio que la
identifica, y el informe declara que queda excluida de toda comparación.
"""

import math

import numpy as np

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from mars_titan.simulation.market import MarketTape

KIND = "oracle_bound_diagnostic"
ORACLE_PARENT = "oracle_one_step_future_returns_diagnostic"
RULE = "close[t+1] / open[t+1] - 1"
FULL_EXPOSURE = 5
SUSPICIOUS_FRACTION = 0.5


def oracle_scores(tape):
    """Rendimiento de la sesión siguiente entre apertura y cierre. La última fila no tiene."""
    prices = tape.prices
    scores = np.full(prices.shape[:2], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores[:-1] = prices[1:, :, 3] / prices[1:, :, 0] - 1.0
    scores[~np.isfinite(scores)] = np.nan
    return scores


def oracle_tape(tape):
    """La misma cinta con las puntuaciones del oráculo y una identidad que no se confunde."""
    return MarketTape(
        tape.prices,
        tape.close_times,
        list(tape.assets),
        oracle_scores(tape),
        domain=tape.domain,
        currency=tape.currency,
        partition=tape.partition,
        prediction_times=tape.prediction_times,
        audit=tape.identity["audit"],
        actions=tape.actions,
        parent_id=ORACLE_PARENT,
        open_times=tape.open_times,
        source_identity=dict(oracle_of=tape.sha256, rule=RULE, excluded_from_comparisons=True),
    )


def _log_growth(result):
    value = result["financial_validation"]["net_return"]
    return None if value is None or value <= -1 else math.log1p(value)


def bound(tape, *, env_options=None, seed=42):
    """Política de la cinta con exposición completa, caja y oráculo, con las mismas reglas."""
    options = dict(env_options or {})
    full = lambda _observation, _step: FULL_EXPOSURE  # noqa: E731
    model = evaluate(FinancialEnv(tape, **options), full, seed=seed)
    cash = evaluate(FinancialEnv(tape, **options), fixed_policy("cash"), seed=seed)
    oracle = evaluate(FinancialEnv(oracle_tape(tape), **options), full, seed=seed)
    model_growth, oracle_growth = _log_growth(model), _log_growth(oracle)
    fraction = (
        model_growth / oracle_growth
        if model_growth is not None and oracle_growth is not None and oracle_growth > 0
        else None
    )
    return dict(
        schema_version=1,
        kind=KIND,
        excluded_from_comparisons=True,
        rule=RULE,
        tape_sha256=tape.sha256,
        oracle_tape_sha256=oracle_tape(tape).sha256,
        seed=seed,
        policies=dict(
            tape_scores_full_exposure=model["financial_validation"],
            cash=cash["financial_validation"],
            oracle_full_exposure=oracle["financial_validation"],
        ),
        log_growth=dict(tape_scores=model_growth, oracle=oracle_growth),
        fraction_of_oracle=fraction,
        suspicious_fraction=SUSPICIOUS_FRACTION,
        suspicious=fraction is not None and fraction >= SUSPICIOUS_FRACTION,
    )
