"""Lectores y contrastes de evidencia externa para la edición sin ajustar.

Las respuestas se tratan como datos. Los lectores validan estructura, fechas y valores
antes de compararlos y no completan ausencias. Los formatos corresponden a las capturas
registradas en el manifiesto privado: la serie diaria de SSE (``yunhq``), las series y
eventos de EODHD y de Alpha Vantage con sus claves demo documentadas.
"""

import json

import numpy as np

HALF_CENT = 0.005
HALF_SUBDOLLAR_TICK = 0.00005
RELATIVE_SLACK = 2e-6


def _daily(rows, label):
    if not rows:
        raise ValueError(f"{label} no contiene sesiones")
    sessions = np.array([row[0] for row in rows], dtype="datetime64[D]")
    close = np.array([float(row[1]) for row in rows], dtype=np.float64)
    order = np.argsort(sessions, kind="stable")
    sessions, close = sessions[order], close[order]
    if (np.diff(sessions) <= np.timedelta64(0, "D")).any():
        raise ValueError(f"{label} repite sesiones")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(f"{label} contiene cierres no positivos o no finitos")
    return sessions, close


def read_sse_daily(body: bytes):
    """Serie diaria de ``yunhq.sse.com.cn``: fecha, apertura, máximo, mínimo, cierre y volumen."""
    content = json.loads(body)
    kline = content.get("kline")
    if not isinstance(kline, list) or content.get("total") != len(kline):
        raise ValueError("La respuesta de SSE no contiene la serie completa declarada")
    rows = []
    for row in kline:
        if not isinstance(row, list) or len(row) < 5:
            raise ValueError("Fila de SSE con formato inesperado")
        day = str(row[0])
        rows.append((f"{day[:4]}-{day[4:6]}-{day[6:8]}", row[4]))
    return _daily(rows, "SSE")


def read_eodhd_daily(body: bytes):
    """Serie diaria de EODHD. ``close`` es el cierre negociado, sin ajustar."""
    content = json.loads(body)
    if not isinstance(content, list):
        raise ValueError("La respuesta de EODHD no es una lista de sesiones")
    return _daily([(row["date"], row["close"]) for row in content], "EODHD")


def read_alphavantage_daily(body: bytes):
    """``TIME_SERIES_DAILY`` de Alpha Vantage, con precios negociados sin ajustar."""
    content = json.loads(body)
    series = content.get("Time Series (Daily)")
    if not isinstance(series, dict):
        raise ValueError("La respuesta de Alpha Vantage no contiene la serie diaria")
    return _daily([(day, values["4. close"]) for day, values in series.items()], "Alpha Vantage")


def _events(rows, label):
    sessions = np.array([row[0] for row in rows], dtype="datetime64[D]")
    values = np.array([float(row[1]) for row in rows], dtype=np.float64)
    if len(values) and (not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError(f"{label} contiene importes o razones no positivos")
    order = np.argsort(sessions, kind="stable")
    return sessions[order], values[order]


def read_eodhd_dividends(body: bytes):
    """Dividendos de EODHD por fecha ex, con su importe sin ajustar por splits."""
    content = json.loads(body)
    if not isinstance(content, list):
        raise ValueError("La respuesta de EODHD no es una lista de dividendos")
    return _events([(row["date"], row["unadjustedValue"]) for row in content], "EODHD")


def _ratio(text: str) -> float:
    new, old = (float(part) for part in text.split("/"))
    return new / old


def read_eodhd_splits(body: bytes):
    content = json.loads(body)
    if not isinstance(content, list):
        raise ValueError("La respuesta de EODHD no es una lista de splits")
    return _events([(row["date"], _ratio(row["split"])) for row in content], "EODHD")


def read_alphavantage_dividends(body: bytes):
    content = json.loads(body)
    rows = content.get("data")
    if not isinstance(rows, list):
        raise ValueError("La respuesta de Alpha Vantage no contiene dividendos")
    return _events([(row["ex_dividend_date"], row["amount"]) for row in rows], "Alpha Vantage")


def read_alphavantage_splits(body: bytes):
    content = json.loads(body)
    rows = content.get("data")
    if not isinstance(rows, list):
        raise ValueError("La respuesta de Alpha Vantage no contiene splits")
    return _events([(row["effective_date"], row["split_factor"]) for row in rows], "Alpha Vantage")


def close_tolerance(prices) -> np.ndarray:
    """Media unidad del redondeo publicado más la holgura relativa de la reconstrucción."""
    prices = np.asarray(prices, dtype=np.float64)
    half = np.where(prices < 1, HALF_SUBDOLLAR_TICK, HALF_CENT)
    return half + RELATIVE_SLACK * prices


def compare_closes(sessions, closes, reference_sessions, reference_closes, *, start, end) -> dict:
    """Contrastar cierres reconstruidos con una fuente externa en las sesiones comunes.

    ``exact`` cuenta diferencias dentro de la holgura relativa. ``within_half_tick``
    admite además la media unidad de redondeo de la fuente, necesaria cuando publica
    con dos decimales cierres fraccionarios anteriores a la decimalización.
    """
    sessions = np.asarray(sessions, dtype="datetime64[D]")
    closes = np.asarray(closes, dtype=np.float64)
    reference_sessions = np.asarray(reference_sessions, dtype="datetime64[D]")
    reference_closes = np.asarray(reference_closes, dtype=np.float64)
    low, high = np.datetime64(start), np.datetime64(end)
    window = (sessions >= low) & (sessions <= high)
    reference_window = (reference_sessions >= low) & (reference_sessions <= high)
    common, left, right = np.intersect1d(
        sessions[window], reference_sessions[reference_window], return_indices=True
    )
    ours = closes[window][left]
    theirs = reference_closes[reference_window][right]
    defined = np.isfinite(ours)
    difference = np.abs(ours - theirs)
    relative = difference / theirs
    exact = defined & (difference <= RELATIVE_SLACK * theirs + 1e-9)
    close_enough = defined & (difference <= close_tolerance(theirs))
    return {
        "common_sessions": int(len(common)),
        "defined": int(defined.sum()),
        "exact": int(exact.sum()),
        "within_half_tick": int(close_enough.sum()),
        "rate": float(close_enough.sum() / defined.sum()) if defined.any() else None,
        "median_relative_difference": float(np.median(relative[defined]))
        if defined.any()
        else None,
        "max_relative_difference": float(relative[defined].max()) if defined.any() else None,
        "mismatch_sessions": [str(day) for day in common[defined & ~close_enough]],
        "provider_only_sessions": int(len(sessions[window]) - len(common)),
        "reference_only_sessions": int(reference_window.sum() - len(common)),
    }


def compare_events(sessions, values, reference_sessions, reference_values, *, start, end, rtol):
    """Emparejar eventos por fecha ex y comparar importes o razones con ``rtol``."""

    def window(days, amounts):
        days = np.asarray(days, dtype="datetime64[D]")
        amounts = np.asarray(amounts, dtype=np.float64)
        keep = (days >= np.datetime64(start)) & (days <= np.datetime64(end))
        return days[keep], amounts[keep]

    ours_days, ours = window(sessions, values)
    theirs_days, theirs = window(reference_sessions, reference_values)
    common, left, right = np.intersect1d(ours_days, theirs_days, return_indices=True)
    agree = np.isclose(ours[left], theirs[right], rtol=rtol, atol=0)
    return {
        "provider_events": int(len(ours_days)),
        "reference_events": int(len(theirs_days)),
        "same_date": int(len(common)),
        "same_value": int(agree.sum()),
        "value_mismatch_dates": [str(day) for day in common[~agree]],
        "provider_only_dates": [str(d) for d in np.setdiff1d(ours_days, theirs_days)],
        "reference_only_dates": [str(d) for d in np.setdiff1d(theirs_days, ours_days)],
    }
