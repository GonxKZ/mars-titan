"""Evidencia E2 por episodio, calculada solo sobre contrastes emitidos y resultados maduros.

El productor de supresiones y la retención del banco son componentes separados. Este
registro no infiere, no selecciona episodios ni modifica parámetros compartidos.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot, _digest_id
from mars_titan.models.titans.financial_inputs import FLOW_ID

DAY_US = 86_400_000_000
KIND = "spaced_utility_evidence_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, low=0, high=2**63 - 1):
    return type(value) is int and low <= value <= high


def _number(value):
    _require(type(value) in (int, float), "E2 necesita números finitos")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("El número E2 excede la precisión admitida") from error
    _require(math.isfinite(result), "E2 necesita números finitos")
    return result


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_canonical(value))


def _flow(value):
    _require(
        isinstance(value, str) and FLOW_ID.fullmatch(value) is not None,
        "El flujo E2 no corresponde a un mercado y activo conocidos",
    )
    return value.split("/", 1)[0]


@dataclass(frozen=True)
class SpacedUtilityConfig:
    """Dos confirmaciones en días separados. El margen no es una garantía de utilidad futura."""

    min_gap_days: int = 1
    positive_gain_margin: float = 0.0
    capacity: int = 1024
    max_pending_rows: int = 40960
    max_pending_sessions: int = 512
    max_references: int = 262144

    def __post_init__(self):
        for name, low, high in (
            ("min_gap_days", 1, 3650),
            ("capacity", 1, 1024),
            ("max_pending_rows", 1, 40960),
            ("max_pending_sessions", 1, 512),
            ("max_references", 1, 262144),
        ):
            _require(_integer(getattr(self, name), low, high), f"El límite E2 {name} no es válido")
        margin = _number(self.positive_gain_margin)
        _require(0 <= margin <= 1e6, "El margen E2 debe ser no negativo y acotado")
        object.__setattr__(self, "positive_gain_margin", margin)

    def identity(self):
        return dict(
            kind=KIND,
            **asdict(self),
            memory_reads=1,
            required_spaced_confirmations=2,
            contrast="selected_support_deletion_without_refill_k1",
            gain="abs_target_minus_deleted_point_minus_abs_target_minus_emitted_point",
            aggregation="mean_rows_per_episode_market_instant_then_mean_sessions",
            spacing="extrema_of_positive_decision_days_utc_not_independence",
        )


def _emission_size(rows):
    _require(
        isinstance(rows, (list, tuple)) and 1 <= len(rows) <= 8192,
        "E2 necesita una cohorte completa de hasta 8192 filas",
    )
    references = 0
    for row in rows:
        _require(
            isinstance(row, dict)
            and set(row)
            == {
                "flow_id",
                "point",
                "deleted_points",
                "label_not_before",
            },
            "La emisión E2 tiene campos incompletos o desconocidos",
        )
        pairs = row["deleted_points"]
        _require(
            isinstance(pairs, (list, tuple)) and len(pairs) <= 64,
            "E2 admite hasta 64 supresiones por fila",
        )
        references += len(pairs)
    return len(rows), references


def _rows(rows, at, eligible):
    _emission_size(rows)
    normalized, markets = {}, set()
    for row in rows:
        flow = row["flow_id"]
        markets.add(_flow(flow))
        _require(flow not in normalized, "La cohorte E2 repite un activo")
        earliest = row["label_not_before"]
        _require(
            earliest is None or (_integer(earliest) and earliest > at),
            "La maduración prevista E2 debe ser posterior a la decisión",
        )
        pairs = row["deleted_points"]
        deleted = {}
        for pair in pairs:
            _require(
                isinstance(pair, (list, tuple)) and len(pair) == 2,
                "Cada supresión E2 necesita episodio y pronóstico",
            )
            identifier, point = pair
            _require(
                _integer(identifier, 1) and identifier in eligible and identifier not in deleted,
                "La supresión E2 repite un episodio o no pertenece al snapshot elegible",
            )
            deleted[identifier] = _number(point)
        normalized[flow] = dict(
            point=_number(row["point"]),
            deleted_points=deleted,
            label_not_before=earliest,
            status=None,
        )
    _require(len(markets) == 1, "Una sesión E2 corresponde a un mercado e instante")
    return markets.pop(), normalized


def _gains(row, target):
    original_error = abs(target - row["point"])
    _require(math.isfinite(original_error), "El error E2 original no es finito")
    result = {}
    for identifier, point in row["deleted_points"].items():
        difference = abs(target - point) - original_error
        _require(math.isfinite(difference), "El contraste E2 no es finito")
        result[identifier] = difference
    return result


def _stat(maturity):
    return dict(
        maturity_at=maturity,
        sessions=0,
        mean_gain=0.0,
        positive_sessions=0,
        first_positive_day=None,
        last_positive_day=None,
    )


def _registry(previous, eligible):
    result = {}
    for episode, maturity in eligible.items():
        score = previous.get(episode, _stat(maturity))
        _require(score["maturity_at"] == maturity, "Un episodio E2 cambió de maduración")
        result[episode] = score
    return result


def _updated(previous, gain, at, margin):
    result = dict(previous)
    count = previous["sessions"] + 1
    result["sessions"] = count
    result["mean_gain"] = _number(
        math.fsum(
            (
                previous["mean_gain"] * (previous["sessions"] / count),
                gain / count,
            )
        )
    )
    if gain > margin:
        day = at // DAY_US
        first, last = previous["first_positive_day"], previous["last_positive_day"]
        result.update(
            positive_sessions=previous["positive_sessions"] + 1,
            first_positive_day=day if first is None else min(first, day),
            last_positive_day=day if last is None else max(last, day),
        )
    return result


class SpacedUtilityLedger:
    """Proponer un evento y publicar evidencia cuando se han resuelto sus sesiones completas."""

    def __init__(self, codec_id, config=None, *, scope_id=None):
        _digest_id(codec_id, "El codec E2")
        if scope_id is not None:
            _digest_id(scope_id, "El ámbito E2")
        self.codec_id = codec_id
        self.scope_id = scope_id
        self.config = SpacedUtilityConfig() if config is None else config
        _require(type(self.config) is SpacedUtilityConfig, "La configuración E2 no es válida")
        self.clock, self._phase, self._snapshot = None, None, None
        self._episodes, self._sessions, self._last_issued = {}, {}, {}
        self.issued = self.resolved = self.censored = self.completed_sessions = 0

    def begin(self, cutoff):
        _require(
            self._phase is None
            and _integer(cutoff)
            and (self.clock is None or cutoff > self.clock),
            "El evento E2 debe avanzar desde un estado publicado",
        )
        self.clock, self._phase, self._snapshot = cutoff, "emission", None

    def issue(self, snapshot, rows):
        _require(
            self._phase == "emission" and type(snapshot) is EpisodeSnapshot,
            "E2 debe registrar emisiones antes de recibir resultados",
        )
        _require(
            snapshot.codec_id == self.codec_id and snapshot.cutoff == self.clock,
            "El snapshot E2 no conserva codec y corte",
        )
        count, incoming_references = _emission_size(rows)
        total_rows = sum(len(c["rows"]) for c in self._sessions.values()) + count
        references = self._references() + incoming_references
        _require(
            total_rows <= self.config.max_pending_rows
            and len(self._sessions) < self.config.max_pending_sessions
            and references <= self.config.max_references,
            "La emisión E2 excede su presupuesto pendiente",
        )
        payload = snapshot.export_cpu()
        identifier = payload["sha256"]
        _require(
            self._snapshot is None or self._snapshot == identifier,
            "Todas las emisiones E2 del evento necesitan el mismo snapshot",
        )
        ids = payload["tensors"]["ids"].tolist()
        maturities = payload["tensors"]["maturity_at"].tolist()
        eligible = dict(zip(ids, maturities, strict=True))
        _require(len(eligible) <= self.config.capacity, "El snapshot E2 excede su capacidad")
        market, normalized = _rows(rows, self.clock, eligible)
        key = market, self.clock
        _require(
            key not in self._sessions and self.clock > self._last_issued.get(market, -1),
            "La sesión E2 ya se emitió",
        )
        current = _registry(self._episodes, eligible)
        used = {i for row in normalized.values() for i in row["deleted_points"]}
        self._episodes, self._snapshot = current, identifier
        self._sessions[key] = dict(
            snapshot_sha256=identifier,
            maturity={i: eligible[i] for i in sorted(used)},
            rows=normalized,
        )
        self._last_issued[market] = self.clock
        self.issued += len(rows)

    def _references(self):
        return sum(
            len(r["deleted_points"]) for c in self._sessions.values() for r in c["rows"].values()
        )

    def usage(self):
        """Contar las estructuras propias sin consultar etiquetas pendientes."""
        return dict(
            episodes=len(self._episodes),
            pending_sessions=len(self._sessions),
            pending_rows=sum(len(c["rows"]) for c in self._sessions.values()),
            references=self._references(),
        )

    def synchronize(self, snapshot):
        """Seguir una retención publicada sin inventar evidencia ni resolver pendientes."""
        _require(
            self._phase is None
            and self.clock is not None
            and type(snapshot) is EpisodeSnapshot
            and snapshot.codec_id == self.codec_id
            and snapshot.cutoff == self.clock
            and snapshot.count <= self.config.capacity,
            "La sincronización E2 necesita una instantánea publicada del mismo corte y capacidad",
        )
        tensors = snapshot.export_cpu()["tensors"]
        eligible = dict(zip(tensors["ids"].tolist(), tensors["maturity_at"].tolist(), strict=True))
        self._episodes = _registry(self._episodes, eligible)

    def _pending_row(self, flow, decision):
        market = _flow(flow)
        _require(_integer(decision), "La decisión E2 no es válida")
        cohort = self._sessions.get((market, decision))
        _require(cohort is not None and flow in cohort["rows"], "Falta la emisión pendiente E2")
        row = cohort["rows"][flow]
        _require(row["status"] is None, "El resultado E2 ya se procesó")
        return row

    def receive(self, outcomes):
        _require(
            self._phase in ("emission", "feedback")
            and isinstance(outcomes, (list, tuple))
            and len(outcomes) <= 8192,
            "E2 recibe una cohorte acotada dentro de su evento",
        )
        prepared, seen = [], set()
        for item in outcomes:
            _require(
                isinstance(item, (list, tuple)) and len(item) == 4,
                "El resultado E2 necesita flujo, decisión, disponibilidad y objetivo",
            )
            flow, decision, available, target = item
            row = self._pending_row(flow, decision)
            _require((flow, decision) not in seen, "El lote de resultados E2 repite una emisión")
            seen.add((flow, decision))
            earliest = row["label_not_before"]
            _require(
                earliest is not None
                and _integer(available)
                and decision < earliest <= available <= self.clock,
                "El resultado E2 no ha madurado en este corte",
            )
            target = _number(target)
            _gains(row, target)
            prepared.append((row, ("label", available, target)))
        for row, status in prepared:
            row["status"] = status
        self._phase = "feedback"
        self.resolved += len(prepared)

    def censor(self, rows, *, reason):
        _require(
            self._phase in ("emission", "feedback")
            and reason in ("excluded", "finalized")
            and isinstance(rows, (list, tuple))
            and len(rows) <= self.config.max_pending_rows,
            "La censura E2 necesita un motivo y filas pendientes",
        )
        prepared, seen = [], set()
        for item in rows:
            _require(
                isinstance(item, (list, tuple)) and len(item) == 2,
                "La censura E2 necesita identidad",
            )
            flow, decision = item
            row = self._pending_row(flow, decision)
            _require((flow, decision) not in seen, "La censura E2 repite una emisión")
            seen.add((flow, decision))
            prepared.append(row)
        for row in prepared:
            row["status"] = (reason, self.clock, None)
        self._phase = "feedback"
        self.censored += len(prepared)

    def publish(self):
        _require(self._phase is not None, "No hay evento E2 para publicar")
        ready = sorted(
            [
                key
                for key, c in self._sessions.items()
                if all(r["status"] is not None for r in c["rows"].values())
            ],
            key=lambda key: (key[1], key[0]),
        )
        staged, receipts = dict(self._episodes), []
        for market, at in ready:
            cohort = self._sessions[market, at]
            gains, resolved = {}, 0
            for flow in sorted(cohort["rows"]):
                row = cohort["rows"][flow]
                if row["status"][0] == "label":
                    resolved += 1
                    for episode, gain in _gains(row, row["status"][2]).items():
                        gains.setdefault(episode, []).append(gain)
            means = {}
            for episode, values in sorted(gains.items()):
                mean = _number(math.fsum(value / len(values) for value in values))
                means[episode] = dict(mean=mean, rows=len(values))
                if episode in staged:
                    staged[episode] = _updated(
                        staged[episode], mean, at, self.config.positive_gain_margin
                    )
            receipts.append(
                dict(
                    market=market,
                    decision_at=at,
                    published_at=self.clock,
                    gains=means,
                    resolved=resolved,
                    censored=len(cohort["rows"]) - resolved,
                )
            )
        self._episodes = staged
        for key in ready:
            del self._sessions[key]
        self.completed_sessions += len(ready)
        self._phase, self._snapshot = None, None
        return receipts

    def scores(self):
        result = {}
        for episode, score in sorted(self._episodes.items()):
            first, last = score["first_positive_day"], score["last_positive_day"]
            qualified = (
                first is not None
                and last - first >= self.config.min_gap_days
                and score["mean_gain"] > self.config.positive_gain_margin
            )
            result[episode] = dict(
                score,
                mean_gain=score["mean_gain"] if score["sessions"] else None,
                qualified=qualified,
            )
        return result

    def export(self):
        _require(self._phase is None, "E2 solo guarda estados publicados")
        identity = dict(codec_id=self.codec_id, config=self.config.identity())
        if self.scope_id is not None:
            identity["scope_id"] = self.scope_id
        state = dict(
            schema_version=1,
            identity=identity,
            clock=self.clock,
            issued=self.issued,
            resolved=self.resolved,
            censored=self.censored,
            completed_sessions=self.completed_sessions,
            last_issued=dict(self._last_issued),
            episodes=[[i, s] for i, s in sorted(self._episodes.items())],
            sessions=[],
        )
        for (market, at), c in sorted(self._sessions.items()):
            state["sessions"].append(
                dict(
                    market=market,
                    decision_at=at,
                    snapshot_sha256=c["snapshot_sha256"],
                    maturity=sorted(c["maturity"].items()),
                    rows=[
                        dict(
                            flow_id=f,
                            point=r["point"],
                            label_not_before=r["label_not_before"],
                            deleted_points=sorted(r["deleted_points"].items()),
                            status=r["status"],
                        )
                        for f, r in sorted(c["rows"].items())
                    ],
                )
            )
        state = _copy(state)
        return dict(state, sha256=hashlib.sha256(_canonical(state).encode()).hexdigest())

    @classmethod
    def restore(cls, codec_id, config, payload, *, scope_id=None):
        from .spaced_utility_state import restore

        return restore(cls(codec_id, config, scope_id=scope_id), payload)
