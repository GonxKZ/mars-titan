"""Reconstrucción por eventos de indicadores macro y sus versiones históricas.

Las unidades de entrada deben estar normalizadas al catálogo. Una fecha ALFRED
es evidencia de disponibilidad, nunca la fecha del periodo observado. Las fechas
sin hora se llevan al final del día de origen y a la siguiente sesión estricta
del mercado objetivo. Esta regla sacrifica inmediatez al cruzar zonas horarias.
"""

import math
import re
from bisect import bisect_left, bisect_right, insort
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from graphlib import CycleError, TopologicalSorter
from zoneinfo import ZoneInfo

from mars_titan.data.macro_formulas import Formula, MissingCalculation
from mars_titan.data.temporal import MarketClock


@dataclass(frozen=True)
class _Value:
    value: float | None = None
    available_at: datetime | None = None
    hashes: frozenset[str] = frozenset()
    reason: str | None = None


def _exclusion(entry: dict) -> str | None:
    if not entry.get("series_id") or entry["series_id"] == "no identifier verified":
        return "unverified_identifier"
    policy = entry.get("vintage_policy")
    if policy == "MODEL_VINTAGES_ONLY":
        return "model_vintages_required"
    if policy != "ALFRED_OR_RELEASE_ARCHIVE":
        return "vintages_not_admissible"
    return None


def _catalog(catalog):
    entries, dependencies, formulas = {}, {}, {}
    frequencies = {
        "M",
        "Q",
        "Q_END",
        "D",
        "D7",
        "W_SAT",
        "W_FRI",
        "W_WED",
        "W_WED_LEVEL",
        "W_WED_AVG",
    }
    for entry in catalog:
        identifier = entry["id"]
        if identifier in entries:
            raise ValueError(f"Duplicate catalog identifier: {identifier}")
        if entry["frequency"] not in frequencies or entry["kind"] not in {"raw", "derived"}:
            raise ValueError(f"Invalid catalog frequency or kind: {identifier}")
        entries[identifier] = entry
        dependencies[identifier] = set(filter(None, entry.get("input_ids", "").split("|")))
        if entry["kind"] == "derived":
            formulas[identifier] = Formula.parse(
                entry["formula"], dependencies[identifier], entry["unit"]
            )
        elif dependencies[identifier]:
            raise ValueError(f"Raw indicator has dependencies: {identifier}")
    if any(dep not in entries for deps in dependencies.values() for dep in deps):
        raise ValueError("Catalog contains an unknown dependency")
    try:
        order = list(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError("Cycle in macro catalog dependencies") from error
    return entries, dependencies, formulas, order


def _period(day: date, frequency: str) -> date:
    if frequency == "M":
        return day.replace(day=1)
    if frequency in {"Q", "Q_END"}:
        return date(day.year, ((day.month - 1) // 3) * 3 + 1, 1)
    return day


def _available(day: date, timezone: str, clock: MarketClock) -> datetime | None:
    source_end = datetime.combine(day, time.max, ZoneInfo(timezone))
    target_zone = ZoneInfo("America/New_York" if clock.market == "US" else "Asia/Shanghai")
    target_date = source_end.astimezone(target_zone).date()
    index = bisect_right(clock.days, target_date)
    return clock.decisions[index] if index < len(clock.days) else None


def _events(rows: Iterable[dict], entries: dict, clock: MarketClock):
    vintages = {}
    for row in rows:
        identifier = row["indicator_id"]
        if identifier not in entries or entries[identifier]["kind"] != "raw":
            raise ValueError(f"Input is not a catalog raw indicator: {identifier}")
        start = date.fromisoformat(row["realtime_start"])
        end = date.fromisoformat(row["realtime_end"])
        period = _period(date.fromisoformat(row["period_start"]), entries[identifier]["frequency"])
        timezone = row.get("source_timezone", "America/New_York")
        ZoneInfo(timezone)
        value = row["value"]
        if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
            raise ValueError("Macro values must be finite numbers or None")
        source_hash = row["source_hash"]
        if not isinstance(source_hash, str) or not re.fullmatch("[0-9a-fA-F]{64}", source_hash):
            raise ValueError("Macro source_hash must be a SHA256 digest")
        if end < start or period > start:
            raise ValueError("Invalid macro realtime interval or future reference period")
        key = identifier, period, start
        payload = end, timezone, value
        if key in vintages and vintages[key][0] != payload:
            raise ValueError(f"Conflicting macro vintage: {key}")
        if key not in vintages:
            vintages[key] = payload, set()
        vintages[key][1].add(source_hash.lower())
    events = []
    for (identifier, period, start), ((end, timezone, value), hashes) in vintages.items():
        if _exclusion(entries[identifier]):
            continue
        available = _available(start, timezone, clock)
        if available is not None:
            observation = _Value(
                value,
                available,
                frozenset(hashes),
                "missing_source_value" if value is None else None,
            )
            events.append((available, start, 1, identifier, period, observation))
        if end < date.max:
            expired = _available(end + timedelta(days=1), timezone, clock)
            if expired is not None:
                events.append(
                    (
                        expired,
                        start,
                        0,
                        identifier,
                        period,
                        _Value(None, expired, frozenset(hashes), "expired_vintage"),
                    )
                )
    # Si ambos eventos llegan al mismo corte, la retirada prevalece sobre el alta.
    events.sort(key=lambda event: (event[0], event[1], -event[2], event[3], event[4]))
    return events


def _lag(period: date, offset: int, frequency: str, periods: list[date]) -> date | None:
    if not offset:
        return period
    if frequency == "D":
        index = bisect_left(periods, period) - offset
        return periods[index] if index >= 0 else None
    if frequency in {"M", "Q", "Q_END"}:
        months = period.year * 12 + period.month - 1 - offset * (1 if frequency == "M" else 3)
        if months < 12:
            return None
        return date(months // 12, months % 12 + 1, 1)
    return period - timedelta(days=offset * (7 if frequency.startswith("W") else 1))


class _Snapshot:
    def __init__(self, entries, dependencies, formulas, order, state, raw_periods):
        self.entries, self.dependencies, self.formulas = entries, dependencies, formulas
        self.state, self.periods = state, dict(raw_periods)
        self.cache = {}
        for identifier in order:
            if identifier in formulas:
                inputs = [set(self.periods[dep]) for dep in sorted(dependencies[identifier])]
                self.periods[identifier] = sorted(set.intersection(*inputs))

    def value(self, identifier: str, period: date | None) -> _Value:
        entry = self.entries[identifier]
        if entry["kind"] == "raw":
            exclusion = _exclusion(entry)
            if exclusion:
                return _Value(reason=exclusion)
            return self.state[identifier].get(period, _Value(reason="missing_period"))
        key = identifier, period
        if key not in self.cache:
            self.cache[key] = self._derived(identifier, period)
        return self.cache[key]

    def _derived(self, identifier: str, period: date | None) -> _Value:
        formula = self.formulas[identifier]
        observations = {
            (dep, offset): self.value(
                dep, _lag(period, offset, self.entries[dep]["frequency"], self.periods[dep])
            )
            if period is not None
            else self.value(dep, None)
            for dep, offset in formula.references
        }
        timestamps = [v.available_at for v in observations.values() if v.available_at is not None]
        available = max(timestamps, default=None)
        hashes = frozenset().union(*(v.hashes for v in observations.values()))
        reason = next((v.reason for v in observations.values() if v.value is None), None)
        if reason:
            return _Value(None, available, hashes, reason)
        try:
            value = formula.calculate({key: v.value for key, v in observations.items()})
            return _Value(value, available, hashes)
        except MissingCalculation as error:
            return _Value(None, available, hashes, str(error))

    def latest(self, identifier: str) -> tuple[date | None, _Value]:
        periods = self.periods[identifier]
        if periods:
            return periods[-1], self.value(identifier, periods[-1])
        value = self.value(identifier, None)
        if value.reason == "missing_period":
            reason = "no_common_period" if identifier in self.formulas else "not_yet_available"
            value = _Value(reason=reason)
        return None, value


def calculate_macro(rows: Iterable[dict], catalog: list[dict], clock: MarketClock) -> list[dict]:
    """Emite todos los indicadores por decisión, incluidas ausencias explicadas.

    El estado conserva cada periodo conocido, incluidas sus ausencias explícitas.
    Se recorren los eventos una vez y se reutiliza la instantánea entre decisiones
    sin novedades. No se consultan redes ni se filtra el archivo por cada sesión.
    Q_END se identifica por el inicio de su trimestre, conservando su unidad de saldo.
    """
    entries, dependencies, formulas, order = _catalog(catalog)
    events = _events(rows, entries, clock)
    state = {identifier: {} for identifier in entries if identifier not in formulas}
    periods = {identifier: [] for identifier in state}
    versions, output, cached = {}, [], {}
    index = 0
    for decision in clock.decisions:
        changed = not cached
        while index < len(events) and events[index][0] <= decision:
            _, vintage, active, identifier, period, value = events[index]
            key = identifier, period
            previous = versions.get(key)
            if previous is None or vintage >= previous:
                if period not in state[identifier]:
                    insort(periods[identifier], period)
                state[identifier][period] = value
                versions[key] = vintage
                changed = True
            index += 1
        if changed:
            snapshot = _Snapshot(entries, dependencies, formulas, order, state, periods)
            cached = {identifier: snapshot.latest(identifier) for identifier in entries}
        for identifier, (period, value) in cached.items():
            output.append(
                {
                    "prediction_at": decision.astimezone(UTC),
                    "indicator_id": identifier,
                    "value": value.value,
                    "available_at": value.available_at,
                    "period_start": period.isoformat() if period else None,
                    "missing_reason": value.reason,
                    "source_hashes": sorted(value.hashes),
                    "unit": entries[identifier]["unit"],
                }
            )
    return output
