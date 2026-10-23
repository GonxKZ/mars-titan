"""Cotas que ninguna política puede superar en una cinta, para detectar defectos del entorno.

Una política decide al cierre de t y sus órdenes se ejecutan en la apertura de t+1, después
de las acciones corporativas de esa apertura. Entre dos cierres, cada unidad de patrimonio
sigue uno de estos caminos: queda en efectivo, mantiene un activo de cierre a cierre, lo
vende en la apertura o compra otro en la apertura con el efectivo que tenía o que acaba de
obtener. Por eso el patrimonio de cualquier política crece en esa sesión como mucho

    max(1, mejor salto nocturno) · max(1, mejor rendimiento de apertura a cierre),

o lo que da mantener un activo sin apertura, si es mayor. El salto nocturno de un activo
incluye sus splits, dividendos y cobros de baja. Esa es la cota con información futura: la
alcanzaría una política que conociera todos los precios, rotara sin costes y no tuviera
límite de liquidez. Los costes, los impuestos, los lotes, las bandas y la participación
solo pueden reducir el resultado, así que la cota sigue siendo válida con ellos.

La rotación tiene otra cota. Cada orden ejecuta como mucho la participación declarada del
volumen del cierre anterior, escalada por los splits de la apertura, al precio de apertura.
Sumar esa capacidad en todas las sesiones y activos da el máximo importe negociable, que
dividido por el capital es la rotación máxima publicable.

Un episodio que supera cualquiera de las dos es un defecto del entorno o de la contabilidad
hasta que se demuestre lo contrario. Las cotas no dependen de las predicciones ni de la
política, solo de precios, volúmenes y acciones corporativas de la propia cinta.
"""

import math

import numpy as np

from .portfolio import DELISTING, UNPRICED_DELISTING

KIND = "environment_bounds_v1"
# Holgura relativa para el redondeo de las series publicadas. El motor nativo reconstruye
# el patrimonio con sus recompensas logarítmicas y deja errores del orden de 1e-12.
TOLERANCE = 1e-9
_OPEN, _CLOSE, _VOLUME = 0, 3, 4
# Acciones que retiran el activo en la apertura: la posición pasa a un cobro (con precio) o
# a cero. La baja sin precio deja el patrimonio sin valorar y no admite comprobación.
_RETIRING = {DELISTING, "writeoff", UNPRICED_DELISTING}


def _actions(tape):
    """Splits, dividendos y salidas de cada apertura de la cinta, por activo."""
    sessions, assets = len(tape), len(tape.assets)
    split = np.ones((sessions, assets))
    dividend = np.zeros((sessions, assets))
    exit_value = np.zeros((sessions, assets))
    retired = np.zeros((sessions, assets), dtype=bool)
    opens = {int(at): index for index, at in enumerate(tape.open_times)}
    columns = {asset: index for index, asset in enumerate(tape.assets)}
    for action in tape.actions:
        session = opens.get(int(action.effective_at))
        # El entorno solo aplica las acciones fechadas en una apertura de la cinta.
        if session is None:
            continue
        column = columns[action.asset]
        if action.kind == "split":
            split[session, column] *= action.value
        elif action.kind == "dividend":
            dividend[session, column] += action.value
        elif action.kind in _RETIRING:
            retired[session, column] = True
            if action.kind == DELISTING:
                exit_value[session, column] += action.value
        else:
            raise ValueError(f"La cota no conoce la acción corporativa {action.kind}")
    return split, dividend, exit_value, retired


def _valid(values):
    return np.isfinite(values) & (values > 0)


def session_growth(tape):
    """Crecimiento máximo [T-1] del patrimonio entre cada cierre y el siguiente."""
    prices = np.asarray(tape.prices, dtype=np.float64)
    opens, closes = prices[:, :, _OPEN], prices[:, :, _CLOSE]
    split, dividend, exit_value, retired = _actions(tape)
    growth = np.ones(len(tape) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for session in range(1, len(tape)):
            before, scale = closes[session - 1], split[session]
            held = _valid(before)
            # Un dividendo o un cobro anotado antes del split se paga por acción antigua y
            # después por acción nueva. Se toma el mayor de los dos órdenes posibles.
            credit = (dividend[session] + exit_value[session]) * np.maximum(1.0, scale)
            gone = retired[session]
            tradable = ~gone & _valid(opens[session])
            valued = ~gone & _valid(closes[session])
            overnight = np.where(
                held & gone,
                credit / before,
                np.where(held & tradable, (scale * opens[session] + credit) / before, np.nan),
            )
            hold = np.where(held & valued, (scale * closes[session] + credit) / before, np.nan)
            intraday = np.where(tradable & valued, closes[session] / opens[session], np.nan)
            rotation = np.fmax.reduce(overnight, initial=1.0) * np.fmax.reduce(
                intraday, initial=1.0
            )
            growth[session - 1] = max(rotation, np.fmax.reduce(hold, initial=1.0))
    return growth


def session_capacity(tape, *, participation):
    """Importe máximo [T-1] que pueden negociar las órdenes ejecutadas en cada apertura."""
    prices = np.asarray(tape.prices, dtype=np.float64)
    split, _, _, retired = _actions(tape)
    volume, opens = prices[:-1, :, _VOLUME], prices[1:, :, _OPEN]
    usable = np.isfinite(volume) & (volume >= 0) & _valid(opens) & ~retired[1:]
    shares = np.where(usable, participation * volume * split[1:], 0.0)
    return np.where(usable, shares * opens, 0.0).sum(axis=1)


def tape_bounds(tape, *, participation, capital):
    """Cotas de crecimiento por sesión y de rotación acumulada de una cinta."""
    if not 0 < participation <= 1 or not math.isfinite(capital) or capital <= 0:
        raise ValueError("Las cotas necesitan una participación y un capital válidos")
    growth = session_growth(tape)
    capacity = np.cumsum(session_capacity(tape, participation=participation)) / capital
    return dict(
        kind=KIND,
        close_times=[int(value) for value in tape.close_times],
        growth=growth,
        turnover=capacity,
    )


def episode_defect(record, bounds, *, tolerance=TOLERANCE):
    """Motivo por el que un episodio terminado supera alguna cota, o None si las respeta."""
    equity, steps = record["equity"], record["steps"]
    nav, times = equity["nav"], equity["close_times"]
    if times != bounds["close_times"][: len(times)] or steps != len(nav) - 1:
        return "el patrimonio no sigue los cierres de su cinta"
    for session in range(steps):
        before, after = nav[session], nav[session + 1]
        if before > 0 and after > before * bounds["growth"][session] * (1 + tolerance):
            return (
                f"el patrimonio crece {after / before:.6g} veces en la sesión {session + 1}, "
                f"más que la cota con información futura ({bounds['growth'][session]:.6g})"
            )
    limit = bounds["turnover"][steps - 1] if steps else 0.0
    if record["turnover"] > limit * (1 + tolerance):
        return (
            f"la rotación {record['turnover']:.6g} supera la capacidad de la participación "
            f"declarada ({limit:.6g})"
        )
    return None
