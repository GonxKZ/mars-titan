"""Ventanas de precios sobre sesiones del calendario, con un bit de presencia por paso.

Contrato desde la edición v3.1. Una ventana cubre las 64 sesiones del calendario oficial que
terminan en la sesión de la decisión. Solo puede faltar una sesión en la que ningún archivo de
la fuente del mercado tiene una fila, como el 29 y el 30 de abril de 2019 en China. Esa posición
no es un dato. Sus cinco canales valen exactamente +0.0 y el canal de presencia vale cero. No se
interpola ni se arrastra ningún precio. Si a un activo le falta una sesión que otros activos sí
tienen, la ventana sigue excluida como en las ediciones anteriores.

Se descartó tratar esas sesiones como días sin mercado. Un paso abarcaría entonces tres sesiones,
cambiaría en silencio el horizonte de la etiqueta y el calendario de todos los activos, y la
ausencia quedaría oculta en lugar de declarada.
"""

import hashlib

import numpy as np

PRICE_FEATURES = (
    "log_open_to_anchor_close",
    "log_high_to_anchor_close",
    "log_low_to_anchor_close",
    "log_close_to_anchor_close",
    "log1p_volume_to_mean_volume",
)
PRESENCE_CHANNEL = "price_present"
PRICE_WINDOW_CHANNELS = (*PRICE_FEATURES, PRESENCE_CHANNEL)
CONTEXT_SESSIONS = 64
_FIXED = dict(
    version=1,
    slots="market_calendar_sessions_ending_at_decision",
    context_sessions=CONTEXT_SESSIONS,
    channels=list(PRICE_WINDOW_CHANNELS),
    absent_fill=0.0,
    anchor="close_of_first_present_session",
    volume_scale="mean_over_present_sessions",
    admissible_absence="session_without_rows_in_every_source_file_of_the_market",
)


def calendar_digest(clock) -> str:
    """Huella de las decisiones del calendario, igual que la de los preparados."""
    return hashlib.sha256("|".join(t.isoformat() for t in clock.decisions).encode()).hexdigest()


def price_window_contract(calendars: dict, market_absent_sessions: dict) -> dict:
    """Declarar el contrato con los calendarios y las ausencias de mercado derivadas de la fuente.

    `calendars` asigna a cada mercado `start`, `end` y `decisions_sha256` del calendario usado
    por la codificación, para que la lectura lo reconstruya y lo compruebe.
    """
    contract = dict(
        _FIXED,
        calendars={m: dict(calendars[m]) for m in sorted(calendars)},
        market_absent_sessions={
            m: sorted(market_absent_sessions[m]) for m in sorted(market_absent_sessions)
        },
    )
    return check_price_window_contract(contract)


def check_price_window_contract(contract) -> dict:
    if not isinstance(contract, dict) or set(contract) != {
        *_FIXED,
        "calendars",
        "market_absent_sessions",
    }:
        raise ValueError("El contrato de ventanas de precios no tiene sus campos exactos")
    if any(contract[key] != value for key, value in _FIXED.items()):
        raise ValueError("El contrato de ventanas de precios no corresponde a la versión 1")
    calendars, absent = contract["calendars"], contract["market_absent_sessions"]
    if (
        not isinstance(calendars, dict)
        or not isinstance(absent, dict)
        or not calendars
        or set(calendars) != set(absent)
        or not set(calendars) <= {"US", "CN"}
    ):
        raise ValueError("El contrato necesita calendario y ausencias de cada mercado")
    for market, calendar in calendars.items():
        days = absent[market]
        if (
            not isinstance(calendar, dict)
            or set(calendar) != {"start", "end", "decisions_sha256"}
            or not all(isinstance(value, str) for value in calendar.values())
            or not isinstance(days, list)
            or days != sorted(set(days))
            or any(not isinstance(day, str) or len(day) != 10 for day in days)
        ):
            raise ValueError("El calendario o las ausencias de un mercado no son válidos")
    return contract


def market_absent_sessions(clock, observed_sessions, *, last="2023-12-31") -> list[str]:
    """Sesiones del calendario sin ninguna fila en toda la fuente del mercado.

    Solo se buscan entre la primera y la última sesión observadas, sin pasar de `last`. Fuera
    de ese tramo la falta de datos es cobertura de la fuente, no una sesión ausente.
    """
    observed = {day for day in observed_sessions if isinstance(day, str)}
    if not observed:
        raise ValueError("El mercado no tiene sesiones observadas")
    first, last = min(observed), min(max(observed), last)
    return [
        day.isoformat()
        for day in clock.days
        if first <= day.isoformat() <= last and day.isoformat() not in observed
    ]


def absent_positions(clock, sessions) -> np.ndarray:
    """Posiciones en el calendario de las sesiones ausentes de todo el mercado."""
    positions = {day.isoformat(): i for i, day in enumerate(clock.days)}
    try:
        return np.array(sorted(positions[day] for day in sessions), dtype=np.int64)
    except KeyError as error:
        raise ValueError("Una ausencia declarada no es una sesión del calendario") from error


def check_row_positions(row_positions, absent) -> np.ndarray:
    """Las filas de un activo están ordenadas y ninguna cae en una ausencia de mercado."""
    row_positions = np.asarray(row_positions, dtype=np.int64)
    if row_positions.ndim != 1 or np.any(np.diff(row_positions) <= 0) or np.any(row_positions < 0):
        raise ValueError("Las posiciones de los precios deben ser únicas y crecientes")
    if np.isin(row_positions, absent).any():
        raise ValueError("Una sesión declarada ausente en todo el mercado tiene precios")
    return row_positions


def present_slots(row_positions, end_row, context, absent):
    """Presencia de cada sesión de la ventana que termina en la fila `end_row`.

    Devuelve None si la ventana empieza antes del calendario o si le falta una sesión que no
    está declarada ausente en todo el mercado. Solo mira posiciones hasta la decisión.
    """
    end = int(row_positions[end_row])
    first = end - context + 1
    if first < 0:
        return None
    start_row = int(np.searchsorted(row_positions, first))
    low, high = np.searchsorted(absent, [first, end + 1])
    if end_row - start_row + 1 + int(high - low) != context:
        return None
    slots = np.zeros(context, dtype=bool)
    slots[row_positions[start_row : end_row + 1] - first] = True
    return slots


def window_rows(row_positions, ends, context, absent) -> np.ndarray:
    """Fila de cada sesión de las ventanas, con -1 donde falta en todo el mercado.

    Rechaza la ventana que contenga cualquier otra ausencia, porque la codificación nunca la
    habría admitido.
    """
    ends = np.asarray(ends)
    if ends.ndim != 1 or ends.dtype.kind not in "iu" or (ends < 0).any():
        raise ValueError("Los índices finales de las ventanas no son válidos")
    if (ends >= len(row_positions)).any():
        raise ValueError("Una ventana termina fuera de los precios del activo")
    slots = row_positions[ends][:, None] - np.arange(context - 1, -1, -1)
    if (slots < 0).any():
        raise ValueError("La ventana empieza antes del calendario")
    rows = np.minimum(np.searchsorted(row_positions, slots), len(row_positions) - 1)
    present = row_positions[rows] == slots
    if np.any(~present & ~np.isin(slots, absent)):
        raise ValueError("La ventana contiene una sesión ausente solo para este activo")
    return np.where(present, rows, -1)


def gate_price_window(prices):
    """Dejar en +0.0 los cinco canales de cada sesión cuyo bit de presencia vale cero.

    Es la entrada común de todas las familias. Una ventana de cinco canales, de una edición
    anterior, pasa sin cambios. Acepta NumPy o Torch y no sincroniza el dispositivo.
    """
    channels = prices.shape[-1]
    if channels == len(PRICE_FEATURES):
        return prices
    if channels != len(PRICE_WINDOW_CHANNELS):
        raise ValueError("La ventana de precios no tiene cinco canales ni el bit de presencia")
    present = prices[..., -1:] > 0
    if isinstance(prices, np.ndarray):
        values = np.where(present, prices[..., :-1], np.zeros((), dtype=prices.dtype))
        return np.concatenate([values, prices[..., -1:]], axis=-1)
    import torch

    zero = torch.zeros((), dtype=prices.dtype, device=prices.device)
    return torch.cat([torch.where(present, prices[..., :-1], zero), prices[..., -1:]], dim=-1)


def validate_price_window(prices) -> None:
    """Comprobar en CPU el contrato de presencia antes de convertir una ventana en tensor.

    El bit vale cero o uno, la sesión de la decisión está presente y un hueco solo contiene
    +0.0. Las ventanas de cinco canales no tienen bit y no se comprueban aquí.
    """
    if prices.shape[-1] != len(PRICE_WINDOW_CHANNELS):
        return
    present, values = prices[..., -1], prices[..., :-1]
    if (
        not ((present == 0) | (present == 1)).all()
        or not (present[..., -1] == 1).all()
        or not (present.sum(axis=-1) >= 2).all()
        or np.any(values[present == 0].view(np.uint32 if values.dtype == np.float32 else np.uint64))
    ):
        raise ValueError("El bit de presencia o el relleno de los huecos no cumple el contrato")
