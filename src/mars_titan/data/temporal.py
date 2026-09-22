"""Calendarios de sesión y barreras temporales comunes a todas las modalidades."""

from bisect import bisect_right
from datetime import UTC, date, datetime, timedelta

import exchange_calendars as xcals

REQUIRED_INPUTS = ("prices", "news", "fundamentals", "charts", "macro")


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("La fecha debe incluir una zona horaria explícita")
    return value.astimezone(UTC)


class MarketClock:
    """Sitúa la decisión cinco minutos después del cierre real de cada sesión."""

    def __init__(self, market: str, start: str, end: str):
        if market not in {"US", "CN"}:
            raise ValueError(f"Mercado no admitido: {market}")
        self.market = market
        calendar = xcals.get_calendar("XNYS" if market == "US" else "XSHG", start=start, end=end)
        self.days = [label.date() for label in calendar.sessions]
        self.decisions = [
            close.to_pydatetime().astimezone(UTC) + timedelta(minutes=5)
            for close in calendar.schedule["close"]
        ]
        self._by_day = dict(zip(self.days, self.decisions, strict=True))

    def decision(self, day: str | date) -> datetime:
        key = date.fromisoformat(day) if isinstance(day, str) else day
        try:
            return self._by_day[key]
        except KeyError as error:
            raise ValueError(
                f"La fecha {key} no es una sesión admitida de {self.market}"
            ) from error

    def date_available(self, day: str, lag: int = 1) -> datetime:
        if lag < 1:
            raise ValueError("El retardo de una fecha sin hora debe ser de al menos una sesión")
        index = bisect_right(self.days, date.fromisoformat(day)) + lag - 1
        if index >= len(self.days):
            raise ValueError(f"La publicación {day} queda fuera de la cobertura del calendario")
        return self.decisions[index]


def admission_errors(available: dict[str, datetime | None], cutoff: datetime) -> list[str]:
    cutoff = aware(cutoff)
    errors = []
    for modality in REQUIRED_INPUTS:
        timestamp = available.get(modality)
        if timestamp is None:
            errors.append(f"{modality}:missing_availability")
        elif aware(timestamp) > cutoff:
            errors.append(f"{modality}:future_information")
    return errors
