"""Reglas de acciones A de Shanghái y Shenzhen comprobadas en fuentes primarias, 2006-2023.

Las fuentes, artículos y fechas están en docs/engineering/china-market-rules.md. No se
modelan el estado ST, los cinco primeros días tras una salida a bolsa ni las ampliaciones
de capital. Fuera de los periodos comprobados no se aplica ninguna regla.
"""

import re
from datetime import UTC, datetime, timedelta, timezone

from .portfolio import Instrument, Period

RULES = "cn_a_share_v1"
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


def china_a_share_instrument(asset):
    """Lotes, venta del resto impar, bandas diarias y timbre de un activo A."""
    kind = board(asset)
    lot, minimum = (1, 200) if kind == "star" else (100, 0)
    return Instrument(
        "CNY",
        lot=lot,
        minimum_order=minimum,
        odd_lot_exit=True,
        price_limits=PRICE_LIMITS[kind],
        taxes=STAMP_DUTY,
        rules=f"{RULES}_{kind}",
    )
