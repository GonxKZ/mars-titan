"""Reconstrucción por eventos de indicadores macro y sus versiones históricas.

Las entradas conservan sus unidades históricas nativas. Una fecha de ALFRED
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
    unit: str | None = None
    seasonal_adjustment: str | None = None


def _exclusion(entry: dict) -> str | None:
    status = entry.get("acquisition_status")
    if status is not None and status != "complete":
        return f"source_{status}:{entry.get('acquisition_reason') or 'source_not_complete'}"
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
            raise ValueError(f"Identificador duplicado en el catálogo: {identifier}")
        if entry["frequency"] not in frequencies or entry["kind"] not in {"raw", "derived"}:
            raise ValueError(f"Frecuencia o tipo de catálogo no válido: {identifier}")
        entries[identifier] = entry
        dependencies[identifier] = set(filter(None, entry.get("input_ids", "").split("|")))
        if entry["kind"] == "derived":
            formulas[identifier] = Formula.parse(
                entry["formula"], dependencies[identifier], entry["unit"]
            )
        elif dependencies[identifier]:
            raise ValueError(f"Un indicador original contiene dependencias: {identifier}")
    if any(dep not in entries for deps in dependencies.values() for dep in deps):
        raise ValueError("El catálogo contiene una dependencia desconocida")
    try:
        order = list(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError("Ciclo en las dependencias del catálogo macro") from error
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
            raise ValueError(f"La entrada no es un indicador original del catálogo: {identifier}")
        start = date.fromisoformat(row["realtime_start"])
        original_start = date.fromisoformat(
            row.get("original_realtime_start", row["realtime_start"])
        )
        end = date.fromisoformat(row["realtime_end"])
        period = _period(date.fromisoformat(row["period_start"]), entries[identifier]["frequency"])
        timezone = row.get("source_timezone", "America/New_York")
        ZoneInfo(timezone)
        value = row["value"]
        reason = row.get("missing_reason")
        unit, adjustment = row.get("native_unit"), row.get("seasonal_adjustment")
        metadata_absence = reason in {
            "missing_historical_metadata",
            "ambiguous_historical_metadata",
        }
        if (
            not isinstance(unit, str)
            or not unit.strip()
            or not isinstance(adjustment, str)
            or not adjustment.strip()
        ) and not metadata_absence:
            raise ValueError(f"Se necesita evidencia de los metadatos históricos: {identifier}")
        if metadata_absence and value is not None:
            raise ValueError("Una observación macro sin metadatos no puede tener un valor admitido")
        if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
            raise ValueError("Los valores macro deben ser números finitos o None")
        source_hash = row["source_hash"]
        if not isinstance(source_hash, str) or not re.fullmatch("[0-9a-fA-F]{64}", source_hash):
            raise ValueError("source_hash debe contener una huella SHA-256")
        if end < start or period > original_start or original_start > start:
            raise ValueError("Intervalo macro no válido o periodo de referencia futuro")
        key = identifier, period, original_start, start
        payload = end, timezone, value, unit, adjustment, reason
        if key in vintages and vintages[key][0] != payload:
            raise ValueError(f"Versión macro en conflicto: {key}")
        if key not in vintages:
            vintages[key] = payload, set()
        vintages[key][1].add(source_hash.lower())
    events = []
    for (identifier, period, original_start, start), (payload, hashes) in vintages.items():
        end, timezone, value, unit, adjustment, reason = payload
        version = original_start, start
        if _exclusion(entries[identifier]):
            continue
        available = _available(start, timezone, clock)
        if available is not None:
            observation = _Value(
                value,
                available,
                frozenset(hashes),
                reason or ("missing_source_value" if value is None else None),
                unit,
                adjustment,
            )
            events.append((available, version, 1, identifier, period, observation))
        if end < date.max:
            expired = _available(end + timedelta(days=1), timezone, clock)
            if expired is not None:
                events.append(
                    (
                        expired,
                        version,
                        0,
                        identifier,
                        period,
                        _Value(
                            None, expired, frozenset(hashes), "expired_vintage", unit, adjustment
                        ),
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
        reasons = [v.reason for v in observations.values() if v.value is None and v.reason]
        reason = next((reason for reason in reasons if reason.startswith("source_")), None)
        reason = reason or next(iter(reasons), None)
        if reason:
            return _Value(None, available, hashes, reason)
        if len({v.unit for v in observations.values()}) != 1:
            return _Value(None, available, hashes, "incompatible_historical_units")
        if len({v.seasonal_adjustment for v in observations.values()}) != 1:
            return _Value(None, available, hashes, "incompatible_seasonal_adjustment")
        try:
            value = formula.calculate({key: v.value for key, v in observations.items()})
            return _Value(value, available, hashes, unit=self.entries[identifier]["unit"])
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
                    "unit": entries[identifier]["unit"] if identifier in formulas else value.unit,
                    "seasonal_adjustment": value.seasonal_adjustment,
                }
            )
    return output
