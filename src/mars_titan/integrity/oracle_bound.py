"""Cota con información futura de un paso para el entorno financiero. Es un diagnóstico.

El entorno decide al cierre de t y ejecuta en la apertura de t+1. La posición se mantiene
hasta la apertura de t+2, donde la decisión siguiente puede cambiarla, y la última se valora
al cierre final. La política con información futura conoce el rendimiento de esa tenencia,
`open[t+2] / open[t+1] - 1` (en la última decisión, `close[T-1] / open[T-1] - 1`), con los
desdoblamientos y dividendos que la cartera aplica en ella. Lo usa como puntuación con la
misma regla de asignación del entorno (cuartil superior positivo con pesos iguales y
exposición completa). Un activo con una baja dentro de la tenencia queda sin puntuación.

Sin costes, cada tenencia es independiente de las demás y la cota elige la mejor en cada
decisión. Solo se aparta del máximo de la regla de asignación porque el tamaño de las órdenes
se calcula al cierre de la decisión, de modo que la exposición realizada difiere algo de la
completa, y por el límite de participación cuando llega a limitar. Con costes deja de ser
un máximo, porque rota en cada sesión y no optimiza la rotación. Sirve para dos cosas:
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
RULE = "open[t+2] / open[t+1] - 1, close[T-1] / open[T-1] - 1 en la última decisión"
FULL_EXPOSURE = 5
SUSPICIOUS_FRACTION = 0.5


def full_exposure(_observation, _step):
    """Exposición completa en cada decisión: la política de la cota."""
    return FULL_EXPOSURE


def oracle_scores(tape):
    """Rendimiento de la tenencia que abre cada decisión. La última fila no decide."""
    prices = tape.prices
    # Valor al final de la tenencia de una acción comprada en la apertura de t+1.
    final = np.concatenate([prices[2:, :, 0], prices[-1:, :, 3]])
    shares, cash = np.ones_like(final), np.zeros_like(final)
    # Una acción efectiva en la apertura de t+2 cae dentro de la tenencia de la decisión t.
    rows = {int(at): index - 2 for index, at in enumerate(tape.open_times) if index >= 2}
    columns = {asset: index for index, asset in enumerate(tape.assets)}
    # Mismo orden que la cartera: la cinta ordena el dividendo antes del desdoblamiento.
    for action in tape.actions:
        row = rows.get(int(action.effective_at))
        if row is None:
            continue
        column = columns[action.asset]
        if action.kind == "split":
            shares[row, column] *= action.value
        elif action.kind == "dividend":
            cash[row, column] += shares[row, column] * action.value
        else:
            final[row, column] = np.nan
    scores = np.full(prices.shape[:2], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores[:-1] = (final * shares + cash) / prices[1:, :, 0] - 1.0
    scores[~np.isfinite(scores)] = np.nan
    return scores


def oracle_tape(tape):
    """La misma cinta puntuada con información futura y una identidad que no se confunde."""
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
    """Cinta con exposición completa, caja y política con información futura, mismas reglas."""
    options = dict(env_options or {})
    model = evaluate(FinancialEnv(tape, **options), full_exposure, seed=seed)
    cash = evaluate(FinancialEnv(tape, **options), fixed_policy("cash"), seed=seed)
    oracle = evaluate(FinancialEnv(oracle_tape(tape), **options), full_exposure, seed=seed)
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
