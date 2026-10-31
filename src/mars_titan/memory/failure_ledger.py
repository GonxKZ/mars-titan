"""Libro contextual S2, con resultados maduros y una instantánea por corte.

El manifiesto de origen declara el ajuste del predictor y sus transformaciones.
El libro comprueba su frontera temporal. No acredita por sí solo esa procedencia
ni realiza ajustes. Sus bytes lógicos no equivalen al pico de RAM del proceso.
"""

import hashlib
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from mars_titan.models.titans.config import FingerprintOnce, bounded_integer
from mars_titan.models.titans.episodic_snapshot import _digest_id
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US, FLOW_ID, HISTORICAL_START_US

from .failure_context import FailureContext, FailureContextLayout


def _clock(value):
    if type(value) is not int or not HISTORICAL_START_US <= value < FINAL_TEST_US:
        raise ValueError("El corte S2 no es válido o pertenece al test reservado")
    return value


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("S2 necesita un número real finito")
    return float(value)


def _decision(flow, at):
    if type(flow) is not str or FLOW_ID.fullmatch(flow) is None:
        raise ValueError("La decisión S2 necesita un flujo válido")
    return flow, _clock(at)


@dataclass(frozen=True)
class FailureLedgerConfig(FingerprintOnce):
    threshold: float
    threshold_manifest_id: str
    threshold_fitted_until: int
    source_manifest_id: str
    source_fitted_until: int
    capacity: int = 4096
    max_pending: int = 8192
    max_flows: int = 8192
    max_bytes: int = 128 * 1024**2
    bandwidth: float = 0.5
    strength: float = 0.5
    block_size: int = 128
    geometry: str = "associated"
    seed: int = 0
    cache_keys: bool = False

    def __post_init__(self):
        for name in ("threshold_manifest_id", "source_manifest_id"):
            _digest_id(getattr(self, name), name)
        for name in ("threshold_fitted_until", "source_fitted_until"):
            _clock(getattr(self, name))
        for name, lower, upper in (
            ("capacity", 1, 40960),
            ("max_pending", 1, 40960),
            ("max_flows", 1, 40960),
            ("max_bytes", 128, 1024**3),
            ("block_size", 1, 4096),
            ("seed", 0, 2**63 - 1),
        ):
            bounded_integer(getattr(self, name), name, lower, upper)
        if _number(self.threshold) <= 0 or not 1e-6 <= _number(self.bandwidth) <= 1e6:
            raise ValueError("El umbral debe ser positivo y el ancho del núcleo debe estar acotado")
        if not 0 <= _number(self.strength) <= 1 or self.geometry not in ("associated", "shuffled"):
            raise ValueError("La intensidad y la geometría S2 no son válidas")
        if type(self.cache_keys) is not bool:
            raise ValueError("La caché de claves S2 necesita una opción booleana explícita")

    def identity(self):
        return dict(
            schema_version=1,
            kind="mature_failure_ledger_v1",
            **asdict(self),
            risk="max_mature_severity_times_gaussian_normalized_key",
            retention="recent_available_received_decision_flow",
            publication="received_strictly_before_next_cutoff",
            bytes="unique_context_buffers_utf8_metadata_and_record_fields",
        )


@dataclass(frozen=True)
class FailureScore:
    risk: float
    confidence: float
    context_id: str | None
    eligible: int
    failure_context_id: str | None


@dataclass(frozen=True)
class _Emission:
    context: FailureContext
    prediction: float
    cache_keys: bool
    _key_cache: bytes = field(init=False, repr=False)
    has_key: bool = field(init=False)

    def __post_init__(self):
        # El contexto ya es inmutable. La caché se reconstruye desde él al recuperar.
        object.__setattr__(self, "has_key", bool(np.any(self.context.key_inputs)))
        raw = self.context.normalized_key().tobytes() if self.cache_keys and self.has_key else b""
        object.__setattr__(self, "_key_cache", raw)

    def normalized_key(self):
        if self.cache_keys and self.has_key:
            return np.frombuffer(self._key_cache, dtype=np.float64)
        return self.context.normalized_key()


@dataclass(frozen=True)
class _Failure:
    emission: _Emission
    target: float
    available_at: int
    received_at: int

    @property
    def severity(self):
        return abs(self.target - self.emission.prediction)


def _key(context):
    return context.flow_id, context.prediction_at


def _order(record):
    context = record.emission.context
    return record.available_at, record.received_at, context.prediction_at, context.flow_id


class FailureLedger:
    """Estado CPU acotado. Las operaciones por lote se publican tras validarse completas."""

    def __init__(self, layout, configuration):
        if (
            type(layout) is not FailureContextLayout
            or type(configuration) is not FailureLedgerConfig
        ):
            raise ValueError("S2 necesita diseño y configuración identificados")
        self.layout, self.configuration = layout, configuration
        self.clock = None
        self._pending, self._last = {}, {}
        self._retained, self._snapshot = (), ()
        self._issued = self._resolved = self._censored = self._admitted = 0

    def begin(self, at):
        _clock(at)
        if self.clock is not None and at < self.clock:
            raise ValueError("S2 no permite retroceder el reloj")
        if (
            max(self.configuration.threshold_fitted_until, self.configuration.source_fitted_until)
            >= at
        ):
            raise ValueError("El umbral y el origen deben haberse ajustado solo con pasado")
        if self.clock != at:
            self._snapshot = tuple(record for record in self._retained if record.received_at < at)
            self.clock = at

    def _context(self, context):
        if (
            type(context) is not FailureContext
            or self.clock is None
            or context.prediction_at != self.clock
            or context.layout_id != self.layout.fingerprint()
        ):
            raise ValueError("El contexto S2 no corresponde al corte o al diseño")
        context.verify()

    def _batch(self, rows):
        if (
            self.clock is None
            or not isinstance(rows, (list, tuple))
            or len(rows) > self.configuration.max_pending
        ):
            raise ValueError("S2 necesita un corte y un lote acotado")

    def logical_bytes(self, *, pending=None, retained=None, last=None):
        pending = self._pending if pending is None else pending
        retained = self._retained if retained is None else retained
        last = self._last if last is None else last
        emissions = {e.context.context_id: e for e in pending.values()}
        records = {r.emission.context.context_id: r for r in (*retained, *self._snapshot)}
        emissions.update({key: r.emission for key, r in records.items()})
        total = 8 * len(pending) + 32 * len(records)
        total += sum(len(flow.encode()) + 8 for flow in last)
        for emission in emissions.values():
            context = emission.context
            payload = context.export()
            total += sum(len(value) for value in payload.values() if type(value) is bytes)
            total += sum(len(value.encode()) for value in payload.values() if type(value) is str)
            total += 24
            total += len(emission._key_cache)
        return total

    def _budget(self, pending, retained, last):
        if (
            len(pending) > self.configuration.max_pending
            or len(last) > self.configuration.max_flows
        ):
            raise ValueError("S2 excede su capacidad de pendientes o flujos")
        if (
            self.logical_bytes(pending=pending, retained=retained, last=last)
            > self.configuration.max_bytes
        ):
            raise ValueError("S2 excede su presupuesto de bytes lógicos")

    def issue(self, rows):
        self._batch(rows)
        pending, last = dict(self._pending), dict(self._last)
        for context, prediction in rows:
            self._context(context)
            key = _key(context)
            if context.prediction_at <= last.get(context.flow_id, -1):
                raise ValueError("La emisión S2 repite una decisión o retrocede su flujo")
            pending[key] = _Emission(context, _number(prediction), self.configuration.cache_keys)
            last[context.flow_id] = context.prediction_at
        self._budget(pending, self._retained, last)
        self._pending, self._last = pending, last
        self._issued += len(rows)

    def resolve(self, rows):
        self._batch(rows)
        pending, retained = dict(self._pending), list(self._retained)
        admitted = 0
        for flow, prediction_at, available_at, target in rows:
            key = _decision(flow, prediction_at)
            if key not in pending or not prediction_at < _clock(available_at) <= self.clock:
                raise ValueError("El resultado S2 no enlaza un pendiente maduro")
            emission = pending.pop(key)
            target = _number(target)
            error = _number(target - emission.prediction)
            if abs(error) > self.configuration.threshold:
                retained.append(_Failure(emission, target, available_at, self.clock))
                admitted += 1
        retained = tuple(sorted(retained, key=_order)[-self.configuration.capacity :])
        self._budget(pending, retained, self._last)
        self._pending, self._retained = pending, retained
        self._resolved += len(rows)
        self._admitted += admitted

    def censor(self, rows):
        self._batch(rows)
        pending = dict(self._pending)
        for flow, at in rows:
            key = _decision(flow, at)
            if key not in pending:
                raise ValueError("La exclusión S2 no enlaza un pendiente")
            pending.pop(key)
        self._pending = pending
        self._censored += len(rows)

    def _geometry(self, context):
        market = context.flow_id.split("/", 1)[0]
        records = tuple(
            r
            for r in self._snapshot
            if r.emission.context.flow_id.split("/", 1)[0] == market
            and r.emission.context.action == context.action
            and r.emission.has_key
        )
        indices = np.arange(len(records))
        if self.configuration.geometry == "shuffled":
            raw = f"{self.configuration.seed}/{self.clock}/{market}/{context.action}".encode()
            seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
            indices = np.random.default_rng(seed).permutation(len(records))
        return records, indices

    def query(self, context):
        self._context(context)
        query = context.normalized_key()
        if not np.any(query):
            return FailureScore(0.0, 1.0, None, 0, None)
        records, indices = self._geometry(context)
        risk, match, failure = 0.0, None, None
        size, config = len(records), self.configuration
        for start in range(0, size, config.block_size):
            selected = indices[start : start + config.block_size]
            keys = np.stack([records[i].emission.normalized_key() for i in selected])
            # La resta directa evita distancias negativas por cancelación en 2−2q·c.
            distances = np.square(keys - query).sum(1)
            severity = np.array(
                [1 - config.threshold / r.severity for r in records[start : start + len(selected)]]
            )
            values = severity * np.exp(-distances / (2 * config.bandwidth**2))
            position = int(np.argmax(values))
            value = float(values[position])
            if value > risk:
                risk = value
                match = records[int(selected[position])].emission.context.context_id
                failure = records[start + position].emission.context.context_id
        return FailureScore(risk, 1 - config.strength * risk, match, size, failure)

    def counts(self):
        return dict(
            issued=self._issued,
            resolved=self._resolved,
            censored=self._censored,
            admitted=self._admitted,
            pending=len(self._pending),
        )

    def export(self):
        from .failure_ledger_state import export

        return export(self)

    @classmethod
    def restore(cls, layout, configuration, payload):
        from .failure_ledger_state import restore

        return restore(layout, configuration, payload)
