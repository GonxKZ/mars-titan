"""Cotas con información futura de un paso para el entorno financiero. Son diagnósticos.

El entorno decide al cierre de t y ejecuta en la apertura de t+1. La posición se mantiene
hasta la apertura de t+2, donde la decisión siguiente puede cambiarla, y la última se valora
al cierre final. El rendimiento de esa tenencia es `open[t+2] / open[t+1] - 1` (en la última
decisión, `close[T-1] / open[T-1] - 1`), con los desdoblamientos, dividendos y bajas que la
cartera aplica dentro de ella. Hay dos cotas:
- La cota exacta suma en cada decisión `log1p(max(0, max_i r_i))`. Sin apalancamiento, sin
  cortos y sin costes, el patrimonio de una tenencia crece como mucho lo que su mejor activo,
  y la caja no rinde. Relaja lotes, límites de precio, liquidez, impuestos y el tamaño de las
  órdenes, que se calcula al cierre de la decisión, así que ningún episodio sin costes puede
  superarla salvo por redondeo (`EXACT_TOLERANCE`). No es alcanzable por el entorno.
- La política con información futura usa ese rendimiento como puntuación con la misma regla
  de asignación del entorno (cuartil superior positivo con pesos iguales y exposición
  completa). No es el máximo de la regla: una política que concentra en uno o dos activos
  puede superarla sin ninguna fuga. Sirve de escala para la fracción sospechosa. Con costes
  deja de ser una referencia útil, porque rota en cada sesión y no optimiza la rotación.

Superar la cota exacta indica un fallo del entorno o de la contabilidad. Alcanzar la mitad
del crecimiento de la política con información futura obliga a revisar a mano, porque casi
seguro hay información futura, pero no lo demuestra por sí solo.

Una apertura ausente no se puede negociar y la posición pasa a la sesión siguiente. Para
repartir esa tenencia entre las dos decisiones se valora en el cierre de su misma sesión.
Cualquier valor positivo conserva la cota, porque el producto de los dos tramos es el
rendimiento real. Sin ese cierre, un episodio que mantenga el activo termina sin patrimonio
y no se compara. Una baja sin precio de salida deja la tenencia sin rendimiento por el mismo
motivo.

La cinta del contrato comprueba las marcas de tiempo de las predicciones, no su contenido,
así que una cinta con puntuaciones del futuro y marcas correctas pasaría su validación.
Estas cotas atacan ese hueco por el resultado. La cinta puntuada lleva un padre propio que
la identifica, y el informe declara que queda excluida de toda comparación.
"""

import math

import numpy as np

from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.evaluation import evaluate, fixed_policy
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.portfolio import DELISTING, UNPRICED_DELISTING

KIND = "oracle_bound_diagnostic"
ORACLE_PARENT = "oracle_one_step_future_returns_diagnostic"
RULE = "open[t+2] / open[t+1] - 1, close[T-1] / open[T-1] - 1 en la última decisión"
EXACT_RULE = "suma de log1p(max(0, max_i r_i)) por decisión, sin costes, lotes ni liquidez"
# Margen de redondeo en crecimiento logarítmico. La contabilidad suma importes con fsum.
EXACT_TOLERANCE = 1e-9
FULL_EXPOSURE = 5
SUSPICIOUS_FRACTION = 0.5


def full_exposure(_observation, _step):
    """Exposición completa en cada decisión: la política de la cota."""
    return FULL_EXPOSURE


def holding_growth(tape):
    """Valor al final de cada tenencia por unidad invertida al empezar. La última fila no decide."""
    prices = tape.prices
    opens, closes = prices[:, :, 0], prices[:, :, 3]
    # Una apertura ausente con cierre se valora en ese cierre, sin negociar.
    marks = np.where(np.isfinite(opens), opens, closes)
    final = np.concatenate([marks[2:], closes[-1:]])
    shares, cash = np.ones_like(final), np.zeros_like(final)
    # Una acción efectiva en la apertura de t+2 cae dentro de la tenencia de la decisión t.
    rows = {int(at): index - 2 for index, at in enumerate(tape.open_times) if index >= 2}
    columns = {asset: index for index, asset in enumerate(tape.assets)}
    # Mismo orden que la cartera: aplica las acciones de una apertura en el orden de la cinta.
    for action in tape.actions:
        row = rows.get(int(action.effective_at))
        if row is None:
            continue
        column = columns[action.asset]
        if action.kind == "split":
            shares[row, column] *= action.value
        elif action.kind in ("dividend", DELISTING):
            cash[row, column] += shares[row, column] * action.value
        if action.kind in ("writeoff", DELISTING):
            # La baja cambia las acciones por su cobro, que ya está en la caja de la tenencia.
            shares[row, column] = 0.0
        elif action.kind == UNPRICED_DELISTING:
            final[row, column] = np.nan
    growth = np.full(prices.shape[:2], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        held = np.where(shares > 0, final * shares, 0.0)
        growth[:-1] = (held + cash) / marks[1:]
    growth[~np.isfinite(growth)] = np.nan
    return growth


def oracle_scores(tape):
    """Rendimiento de la tenencia que abre cada decisión, solo donde se puede comprar."""
    scores = holding_growth(tape) - 1.0
    scores[:-1][~np.isfinite(tape.prices[1:, :, 0])] = np.nan
    return scores


def exact_log_growth(tape):
    """Crecimiento logarítmico máximo sin costes: el mejor activo o la caja en cada tenencia."""
    growth = holding_growth(tape)[:-1]
    best = np.max(np.where(np.isnan(growth), 1.0, growth), axis=1, initial=1.0)
    return math.fsum(np.log(best).tolist())


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
    """Cinta con exposición completa, caja, política con información futura y cota exacta."""
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
        log_growth=dict(
            tape_scores=model_growth, oracle=oracle_growth, exact=exact_log_growth(tape)
        ),
        exact_rule=EXACT_RULE,
        fraction_of_oracle=fraction,
        suspicious_fraction=SUSPICIOUS_FRACTION,
        suspicious=fraction is not None and fraction >= SUSPICIOUS_FRACTION,
    )
