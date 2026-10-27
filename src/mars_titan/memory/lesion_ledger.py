"""Registro acotado de predicciones de B4, puntuadas solo al madurar el resultado."""

import hashlib
import json
import math

import torch

from mars_titan.evaluation.session_metrics import SessionErrors

from .associative_memory import MAX_COHORT, _check_matrix
from .lesion_protocol import LesionReadout
from .multiscale_mixture import _clock

MAX_PENDING = 40960
MAX_SESSIONS = 50000
REASONS = ("excluded", "finalized")


def _key(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or type(value[0]) is not str
        or not 1 <= len(value[0]) <= 256
        or value[0].split("/", 1)[0] not in ("US", "CN")
        or "/" not in value[0]
        or not value[0].split("/", 1)[1]
        or any(ord(char) < 32 for char in value[0])
    ):
        raise ValueError("B4 necesita identidades de flujo e instante de decisión")
    _clock(value[1])
    return tuple(value)


def _order(key):
    return key[1], key[0]


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("El resultado maduro de B4 debe ser real y finito")
    return float(value)


def _hash(value):
    def describe(item):
        if isinstance(item, torch.Tensor):
            return dict(
                shape=list(item.shape),
                dtype=str(item.dtype),
                sha256=hashlib.sha256(item.detach().contiguous().numpy().tobytes()).hexdigest(),
            )
        if isinstance(item, dict):
            return {key: describe(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [describe(part) for part in item]
        return item

    raw = json.dumps(describe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class LesionLedger:
    """Una predicción por condición y decisión, con agregación común por sesión."""

    def __init__(self, plan, *, max_pending=MAX_PENDING, max_sessions=MAX_SESSIONS):
        if (
            type(plan) is not LesionReadout
            or type(max_pending) is not int
            or not 1 <= max_pending <= MAX_PENDING
            or type(max_sessions) is not int
            or not 1 <= max_sessions <= MAX_SESSIONS
        ):
            raise ValueError("El registro B4 necesita un plan y límites explícitos")
        self.plan, self.max_pending, self.max_sessions = plan, max_pending, max_sessions
        self.clock = None
        self.issued = self.resolved = self.censored = 0
        self.pending = {}
        self._errors = tuple(SessionErrors(max_sessions=max_sessions) for _ in plan.names)
        self._event_issued, self._event_resolved, self._event_censored = {}, {}, {}

    def begin(self, at):
        _clock(at)
        if self.clock is not None and at < self.clock:
            raise ValueError("El registro B4 no permite retroceder su reloj")
        self.clock = at
        self._event_issued, self._event_resolved, self._event_censored = {}, {}, {}

    def _keys(self, rows, *, maximum=MAX_COHORT):
        if self.clock is None or not isinstance(rows, (list, tuple)) or len(rows) > maximum:
            raise ValueError("B4 necesita un corte y una cohorte acotada de decisiones")
        keys = [_key(row) for row in rows]
        if len(set(keys)) != len(keys) or any(key[1] > self.clock for key in keys):
            raise ValueError("La cohorte B4 repite una decisión o contiene futuro")
        return keys

    def issue(self, rows, predictions):
        keys = self._keys(rows)
        _check_matrix(
            predictions, "Las predicciones emitidas de B4", len(keys), len(self.plan.names)
        )
        if (
            any(key[1] != self.clock or key in self.pending for key in keys)
            or len(self.pending) + len(keys) > self.max_pending
            or len(self._event_issued) + len(keys) > MAX_COHORT
        ):
            raise ValueError("Las emisiones B4 repiten decisiones o exceden su presupuesto")
        entries = {key: predictions[i].detach().clone() for i, key in enumerate(keys)}
        self.pending.update(entries)
        self._event_issued.update(entries)
        self.issued += len(keys)

    def receive(self, records):
        if not isinstance(records, (list, tuple)) or any(
            not isinstance(row, (list, tuple)) or len(row) != 4 for row in records
        ):
            raise ValueError("B4 recibe flujo, decisión, disponibilidad y resultado maduro")
        keys = self._keys([row[:2] for row in records])
        normalized = []
        for key, (_, _, available, value) in zip(keys, records, strict=True):
            _clock(available)
            if key not in self.pending or not key[1] < available <= self.clock:
                raise ValueError("El resultado B4 no tiene emisión pendiente o no ha madurado")
            normalized.append((key, available, _number(value)))
        if not normalized:
            return {}
        normalized.sort(key=lambda row: (row[1], *_order(row[0])))
        predictions = torch.stack([self.pending[key] for key, _, _ in normalized])
        targets = torch.tensor([target for _, _, target in normalized], dtype=torch.float64)
        errors = predictions - targets[:, None]
        if not torch.isfinite(errors.square()).all():
            raise ValueError("Los errores B4 o sus cuadrados no son finitos")
        if len(self._event_resolved) + len(keys) > MAX_COHORT:
            raise ValueError("Los resultados B4 exceden la cohorte del evento")
        markets = [key[0].split("/", 1)[0] for key, _, _ in normalized]
        moments = [key[1] for key, _, _ in normalized]
        staged = []
        for case, previous in enumerate(self._errors):
            current = SessionErrors(max_sessions=self.max_sessions)
            # update crea entradas nuevas. Copiar el diccionario basta para confirmar todos
            # los casos juntos sin modificar las listas de las sesiones anteriores.
            current.sessions = previous.sessions.copy()
            for start in range(0, len(normalized), 4096):
                current.update(
                    markets[start : start + 4096],
                    moments[start : start + 4096],
                    errors[start : start + 4096, case].numpy(),
                )
            staged.append(current)
        resolved = {key: self.pending[key].clone() for key, _, _ in normalized}
        for key, available, target in normalized:
            self._event_resolved[key] = (self.pending.pop(key), available, target)
        self._errors = tuple(staged)
        self.resolved += len(keys)
        return resolved

    def discard(self, rows, *, reason):
        keys = self._keys(rows, maximum=self.max_pending)
        if (
            reason not in REASONS
            or any(key not in self.pending for key in keys)
            or len(self._event_censored) + len(keys) > self.max_pending
        ):
            raise ValueError("La censura B4 necesita pendientes y una causa declarada")
        for key in keys:
            self.pending.pop(key)
            self._event_censored[key] = reason
        self.censored += len(keys)

    def metrics(self):
        return dict(
            configuration=self.plan.identity(),
            issued=self.issued,
            resolved=self.resolved,
            censored=self.censored,
            pending=len(self.pending),
            cases={
                name: value.summary()
                for name, value in zip(self.plan.names, self._errors, strict=True)
            },
        )

    def _pack(self, rows):
        keys = sorted(rows, key=_order)
        return dict(
            keys=[list(key) for key in keys],
            predictions=torch.stack([rows[key] for key in keys])
            if keys
            else torch.empty(0, len(self.plan.names), dtype=torch.float64),
        )

    def event(self):
        resolved = self._pack({key: row[0] for key, row in self._event_resolved.items()})
        keys = [tuple(key) for key in resolved["keys"]]
        resolved["available_at"] = [self._event_resolved[key][1] for key in keys]
        resolved["target"] = [self._event_resolved[key][2] for key in keys]
        return dict(
            cutoff=self.clock,
            issued=self._pack(self._event_issued),
            resolved=resolved,
            censored=[
                [key[0], key[1], self._event_censored[key]]
                for key in sorted(self._event_censored, key=_order)
            ],
        )

    def export(self):
        keys = sorted(self._errors[0].sessions)
        counts = [self._errors[0].sessions[key][0] for key in keys]
        errors = torch.tensor(
            [[value.sessions[key][1:] for key in keys] for value in self._errors],
            dtype=torch.float64,
        ).reshape(len(self.plan.names), len(keys), 2)
        payload = dict(
            schema_version=1,
            configuration=self.plan.identity(),
            max_pending=self.max_pending,
            max_sessions=self.max_sessions,
            clock=self.clock,
            issued=self.issued,
            resolved=self.resolved,
            censored=self.censored,
            pending=self._pack(self.pending),
            sessions=dict(keys=[list(key) for key in keys], counts=counts, errors=errors),
            event=self.event(),
        )
        return dict(payload, state_sha256=_hash(payload))

    def _unpack(self, pack, maximum):
        if (
            not isinstance(pack, dict)
            or set(pack) != {"keys", "predictions"}
            or not isinstance(pack["keys"], list)
            or len(pack["keys"]) > maximum
        ):
            raise ValueError("La matriz B4 no conserva identidades y predicciones acotadas")
        keys = [_key(key) for key in pack["keys"]]
        if keys != sorted(set(keys), key=_order) or any(
            self.clock is None or key[1] > self.clock for key in keys
        ):
            raise ValueError("Las identidades B4 están repetidas, desordenadas o en el futuro")
        _check_matrix(
            pack["predictions"], "Las predicciones B4 guardadas", len(keys), len(self.plan.names)
        )
        return {key: pack["predictions"][i].detach().clone() for i, key in enumerate(keys)}

    def _restore_scores(self, sessions):
        if (
            not isinstance(sessions, dict)
            or set(sessions) != {"keys", "counts", "errors"}
            or not isinstance(sessions["keys"], list)
            or len(sessions["keys"]) > self.max_sessions
        ):
            raise ValueError("Las sesiones B4 no conservan su esquema acotado")
        keys = []
        for key in sessions["keys"]:
            if not isinstance(key, (list, tuple)) or len(key) != 2 or key[0] not in ("US", "CN"):
                raise ValueError("Las sesiones B4 necesitan mercado e instante")
            _clock(key[1])
            if self.clock is None or key[1] >= self.clock:
                raise ValueError("La sesión B4 no corresponde a un resultado maduro")
            keys.append(tuple(key))
        counts, errors = sessions["counts"], sessions["errors"]
        if (
            keys != sorted(set(keys))
            or not isinstance(counts, list)
            or len(counts) != len(keys)
            or any(type(n) is not int or n < 1 for n in counts)
        ):
            raise ValueError("Los conteos B4 no corresponden a sesiones canónicas")
        if (
            not isinstance(errors, torch.Tensor)
            or errors.device.type != "cpu"
            or errors.dtype != torch.float64
            or errors.layout != torch.strided
            or errors.shape != (len(self.plan.names), len(keys), 2)
            or not torch.isfinite(errors).all()
            or (errors < 0).any()
        ):
            raise ValueError("Las sumas de errores B4 no son tensores CPU FP64 válidos")
        for case, accumulator in enumerate(self._errors):
            accumulator.sessions = {
                key: [count, *pair]
                for key, count, pair in zip(keys, counts, errors[case].tolist(), strict=True)
            }
        if sum(counts) != self.resolved or self.issued != self.resolved + self.censored + len(
            self.pending
        ):
            raise ValueError("Los contadores B4 no concilian con sus resultados y pendientes")

    def _restore_censored(self, censored, resolved):
        if not isinstance(censored, list) or len(censored) > self.max_pending:
            raise ValueError("Las censuras B4 exceden el evento")
        for row in censored:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError("La censura B4 no conserva identidad y causa")
            key = _key(row[:2])
            if (
                key in self.pending
                or key in resolved
                or key in self._event_censored
                or row[2] not in REASONS
                or self.clock is None
                or key[1] > self.clock
            ):
                raise ValueError("La censura B4 es incoherente con el estado")
            self._event_censored[key] = row[2]

    def _restore_event(self, event):
        if (
            not isinstance(event, dict)
            or set(event) != {"cutoff", "issued", "resolved", "censored"}
            or event["cutoff"] != self.clock
        ):
            raise ValueError("El evento B4 no corresponde al corte guardado")
        self._event_issued = self._unpack(event["issued"], MAX_COHORT)
        received = event["resolved"]
        if not isinstance(received, dict) or set(received) != {
            "keys",
            "predictions",
            "available_at",
            "target",
        }:
            raise ValueError("Los resultados B4 no conservan su procedencia")
        resolved = self._unpack({key: received[key] for key in ("keys", "predictions")}, MAX_COHORT)
        available, targets = received["available_at"], received["target"]
        if (
            not isinstance(available, list)
            or not isinstance(targets, list)
            or len(available) != len(resolved)
            or len(targets) != len(resolved)
        ):
            raise ValueError("Las disponibilidades y resultados B4 no están alineados")
        for (key, prediction), at, target in zip(resolved.items(), available, targets, strict=True):
            _clock(at)
            if key in self.pending or not key[1] < at <= self.clock:
                raise ValueError("El resultado B4 guardado no ha madurado o sigue pendiente")
            self._event_resolved[key] = (prediction, at, _number(target))
        censored = event["censored"]
        self._restore_censored(censored, resolved)
        if (
            len(self._event_issued) > self.issued
            or len(resolved) > self.resolved
            or len(censored) > self.censored
        ):
            raise ValueError("El evento B4 supera sus contadores acumulados")

    @classmethod
    def restore(cls, plan, payload):
        fields = {
            "schema_version",
            "configuration",
            "max_pending",
            "max_sessions",
            "clock",
            "issued",
            "resolved",
            "censored",
            "pending",
            "sessions",
            "event",
            "state_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["configuration"] != plan.identity()
        ):
            raise ValueError("El estado B4 no conserva su versión y configuración")
        value = cls(plan, max_pending=payload["max_pending"], max_sessions=payload["max_sessions"])
        clock = payload["clock"]
        if clock is not None:
            value.begin(clock)

        value.pending = value._unpack(payload["pending"], value.max_pending)
        for name in ("issued", "resolved", "censored"):
            count = payload[name]
            if type(count) is not int or not 0 <= count < 2**63:
                raise ValueError("Los contadores B4 necesitan enteros no negativos")
            setattr(value, name, count)
        value._restore_scores(payload["sessions"])
        value._restore_event(payload["event"])
        if value.export()["state_sha256"] != payload["state_sha256"]:
            raise ValueError("La huella del estado B4 no coincide con el contenido")
        return value
