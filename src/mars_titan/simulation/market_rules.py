"""Reglas de acciones A de Shanghái y Shenzhen comprobadas en fuentes primarias, 2006-2023.

Las fuentes, artículos y fechas están en docs/engineering/china-market-rules.md. Sin estado
de cotización, cada activo recibe las reglas de su tablero (`cn_a_share_v1`). Con el estado
acreditado de `listing_status.py`, las bandas incluyen además la advertencia de riesgo del
tablero principal y los cinco primeros días sin límites tras una salida a bolsa
(`cn_a_share_v2`). Las ampliaciones de capital no se modelan. Fuera de los periodos
comprobados no se aplica ninguna regla.
"""

import re
from datetime import UTC, date, datetime, timedelta, timezone
from functools import cache

from .portfolio import Instrument, Period, _at

RULES = "cn_a_share_v1"
# Reglas del tablero más el estado de cotización de cada activo.
STATUS_RULES = "cn_a_share_v2"
CHINA = timezone(timedelta(hours=8))
# Vigencia comprobada. El final coincide con el periodo de desarrollo del proyecto.
VERIFIED_END = (2024, 1, 1)
BOARDS = {
    ("SS", "600"): "main",
    ("SS", "601"): "main",
    ("SS", "603"): "main",
    ("SS", "605"): "main",
    ("SS", "688"): "star",
    ("SS", "689"): "star",
    ("SZ", "000"): "main",
    ("SZ", "001"): "main",
    ("SZ", "002"): "main",
    ("SZ", "003"): "main",
    ("SZ", "300"): "chinext",
    ("SZ", "301"): "chinext",
}


def beijing_day(year, month, day):
    """Inicio de una fecha de Pekín, en microsegundos UTC."""
    moment = datetime(year, month, day, tzinfo=CHINA).astimezone(UTC)
    return int(moment.timestamp()) * 1_000_000


def _period(start, end, **values):
    return Period(beijing_day(*start), beijing_day(*end), **values)


# Impuesto de timbre sobre el efectivo negociado: 财税[2005]11号, 财税〔2007〕84号,
# comunicados del Ministerio de Hacienda de 2008, Ley del impuesto de timbre y
# 财政部 税务总局公告2023年第39号.
STAMP_DUTY = (
    _period((2005, 1, 24), (2007, 5, 30), buy=0.001, sell=0.001),
    _period((2007, 5, 30), (2008, 4, 24), buy=0.003, sell=0.003),
    _period((2008, 4, 24), (2008, 9, 19), buy=0.001, sell=0.001),
    _period((2008, 9, 19), (2023, 8, 28), sell=0.001),
    _period((2023, 8, 28), VERIFIED_END, sell=0.0005),
)
# Bandas sin estado ST. Reglas de negociación de 2006 de ambas bolsas, reforma de ChiNext
# de 2020 (深证上〔2020〕515号) y reglas de STAR de 2019 (上证发〔2019〕23号).
PRICE_LIMITS = {
    "main": (_period((2006, 7, 1), VERIFIED_END, band=0.10),),
    "chinext": (
        _period((2006, 7, 1), (2020, 8, 24), band=0.10),
        _period((2020, 8, 24), VERIFIED_END, band=0.20),
    ),
    "star": (_period((2019, 7, 22), VERIFIED_END, band=0.20),),
}


def board(asset):
    """Deducir el tablero del código de seis cifras y su bolsa."""
    match = re.fullmatch(r"(?:CN/)?(\d{6})\.(SS|SH|SZ)", asset)
    if not match:
        raise ValueError("El activo no tiene un código de acción A reconocible")
    code, exchange = match.groups()
    result = BOARDS.get(("SS" if exchange == "SH" else exchange, code[:3]))
    if result is None:
        raise ValueError("El tablero del activo no tiene reglas acreditadas")
    return result


# Banda de una acción con advertencia de riesgo (ST o *ST) en el tablero principal: artículo
# 3.4.13 de Shanghái 2006, 4.4.10 de 2023, 3.3.14 de Shenzhen 2006 y 4.5.5 de 2021 y 2023.
# ChiNext no tuvo advertencias hasta la reforma de 2020 y desde entonces aplica a todo el
# tablero el 20 %, igual que STAR, así que solo el tablero principal cambia de banda.
SPECIAL_TREATMENT_BAND = 0.05
SPECIAL_TREATMENT_BOARDS = ("main",)
# Las acciones sin reforma accionarial (prefijo S) negociaban con la misma banda del 5 %, como
# confirman sus anuncios (por ejemplo el 临2013-008 de 600733) y sus precios. El primer día
# tras la reforma, como el primero de una reanudación de cotización, no tenía límite.
DAY = 86_400_000_000
# Salidas a bolsa sin límite diario durante sus cinco primeras sesiones: STAR desde su
# inicio (上证发〔2019〕23号), ChiNext desde la reforma (深证上〔2020〕515号) y el tablero
# principal desde las primeras admisiones por registro del 10 de abril de 2023.
LIMIT_FREE_FROM = {"star": (2019, 7, 22), "chinext": (2020, 8, 24), "main": (2023, 4, 10)}
LIMIT_FREE_SESSIONS = 5


@cache
def xshg_sessions():
    """Sesiones de XSHG de la cobertura del estado de cotización, con margen en 2024."""
    from mars_titan.data.temporal import MarketClock

    return tuple(MarketClock("CN", "2010-01-01", "2024-02-29").days)


@cache
def xshg_session_set():
    return frozenset(xshg_sessions())


def limit_free_until(asset, listed_on):
    """Primera sesión con banda tras una salida a bolsa sin límites, o None si no se aplica.

    Las reglas hablan de los cinco primeros días de negociación desde la admisión, así que se
    cuentan sesiones de la bolsa, no días negociados por el valor. La admisión debe ser una
    sesión del calendario.
    """
    start = date(*LIMIT_FREE_FROM[board(asset)])
    listed = date.fromisoformat(listed_on)
    if listed < start:
        return None
    sessions = xshg_sessions()
    if listed not in sessions:
        raise ValueError("La fecha de admisión no es una sesión de XSHG")
    return sessions[sessions.index(listed) + LIMIT_FREE_SESSIONS].isoformat()


def _beijing(value, default):
    return default if value is None else beijing_day(*date.fromisoformat(value).timetuple()[:3])


def status_price_limits(kind, status):
    """Bandas del tablero con el estado acreditado del activo.

    Se cortan los periodos del tablero en cada cambio de estado. En la exención tras la
    salida a bolsa y en los días sin límite tras una reforma o una reanudación no hay banda.
    En un tramo ST o sin reforma accionarial del tablero principal la banda es del 5 % y
    fuera de los periodos comprobados no hay ninguna. Los tramos contiguos con la misma banda
    se unen, de modo que el resultado es el mismo calculado en Python y en C++.
    """
    base, end = PRICE_LIMITS[kind], beijing_day(*VERIFIED_END)
    special = [
        (_beijing(start, None), _beijing(stop, end))
        for field in ("special_treatment", "share_reform_pending")
        for start, stop in status[field]
        if kind in SPECIAL_TREATMENT_BOARDS
    ]
    free = [(_beijing(day, None), _beijing(day, None) + DAY) for day in status["limit_free_days"]]
    if status["limit_free_until"] is not None:
        free.append(
            (_beijing(status["listed_on"], None), _beijing(status["limit_free_until"], None))
        )
    cuts = {value for period in base for value in (period.start, period.end)}
    cuts |= {value for span in (*special, *free) for value in span}
    result = []
    for left, right in zip(sorted(cuts), sorted(cuts)[1:], strict=False):
        period = _at(base, left)
        if period is None or any(a <= left < b for a, b in free):
            continue
        band = SPECIAL_TREATMENT_BAND if any(a <= left < b for a, b in special) else period.band
        if result and result[-1].end == left and result[-1].band == band:
            result[-1] = Period(result[-1].start, right, band=band)
        else:
            result.append(Period(left, right, band=band))
    return tuple(result)


def china_a_share_instrument(asset, status=None):
    """Lotes, venta del resto impar, bandas diarias y timbre de un activo A.

    `status` es la entrada del activo en la tabla de estado de cotización. Sin ella se usan
    solo las reglas del tablero, como en las cintas sintéticas.
    """
    kind = board(asset)
    lot, minimum = (1, 200) if kind == "star" else (100, 0)
    return Instrument(
        "CNY",
        lot=lot,
        minimum_order=minimum,
        odd_lot_exit=True,
        price_limits=PRICE_LIMITS[kind] if status is None else status_price_limits(kind, status),
        taxes=STAMP_DUTY,
        rules=f"{RULES if status is None else STATUS_RULES}_{kind}",
    )


def tape_instruments(tape):
    """Reglas que exige una cinta: las del estado acreditado si es china y reconstruida.

    Una cinta real reconstruida lleva en su auditoría el estado de cotización de cada activo,
    y las reglas se derivan de él. Una cinta sintética china solo recibe las de su tablero.
    """
    if tape.currency != "CNY":
        return None
    audit = tape.identity.get("audit") or {}
    status = (audit.get("listing_status") or {}).get("assets")
    if tape.domain == "real" and status is not None:
        return {asset: china_a_share_instrument(asset, status[asset]) for asset in tape.assets}
    return {asset: china_a_share_instrument(asset) for asset in tape.assets}
