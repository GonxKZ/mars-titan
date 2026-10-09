"""M2 por error maduro, con tres índices nativos y consulta deduplicada."""

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

_ROLES = ("reservoir", "selective", "recent")
_CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_MAX_ARCHIVE = 2 * 1024**2


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _same(value[key], child) for key, child in expected.items()
        )
    return value == expected


def _ordered_ids(values, maximum):
    return (
        type(values) is list
        and len(values) <= maximum
        and all(_integer(value, 1, 2**63 - 1) for value in values)
        and values == sorted(set(values))
    )


def _record_bytes(record):
    return struct.pack(
        "<Qqqq128fdd??",
        record.id,
        record.decision_at,
        record.available_at,
        record.maturity_at,
        *record.key,
        *record.value,
        record.reward,
        record.label,
        record.reward_valid,
        record.label_valid,
    )


@dataclass(frozen=True)
class MatureErrorConfig:
    """Un único B_mem para los cupos 50/25/25, con M desactivado."""

    capacity: int = 1024
    seed: int = 73
    max_working_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if not (
            _integer(self.capacity, 4, 1024)
            and _integer(self.seed, 0, 2**64 - 1)
            and _integer(self.max_working_bytes, 1, 64 * 1024**2)
        ):
            raise ValueError("La capacidad, semilla o presupuesto de M2 no son válidos")

    @property
    def quotas(self):
        numerators = (2 * self.capacity, self.capacity, self.capacity)
        result = [value // 4 for value in numerators]
        order = sorted(range(3), key=lambda index: (-(numerators[index] % 4), index))
        for index in order[: self.capacity - sum(result)]:
            result[index] += 1
        return dict(zip(_ROLES, result, strict=True))


class MatureErrorBank:
    """Proponer copias del reservorio, top por error y recientes sin publicar."""

    def __init__(self, native, config, *, codec_id, world, partition, fold):
        if (
            type(config) is not MatureErrorConfig
            or not isinstance(codec_id, str)
            or len(codec_id) != 64
            or any(char not in "0123456789abcdef" for char in codec_id)
            or any(
                not isinstance(value, str) or not 1 <= len(value) <= 128
                for value in (world, partition, fold)
            )
        ):
            raise ValueError("M2 necesita configuración y ámbito identificados")
        self._native, self.config, self._codec_id = native, config, codec_id
        self._scope = dict(world=world, partition=partition, fold=fold)
        self._config_values = asdict(config)
        self._check_budget(0)
        identity = dict(
            schema_version=1,
            recipe="episodic_m2_three_index_v1",
            config=asdict(config),
            scope=self._scope,
            codec_id=codec_id,
            quotas=config.quotas,
            allocation="integer_largest_remainders_2_1_1_role_order",
            native_sha256=native.binary_sha256,
            native_torch_version=native.torch_version,
            native_schema_version=2,
            code_sha256=_CODE,
            reservoir="native_v2_mt19937_64_seed_only",
            score="absolute_emitted_mature_error_fp64",
            selection="highest_score_then_lowest_mature_ordinal",
            recent="latest_mature_ordinals",
            cm_m="disabled",
            key_width=64,
            value_width=64,
            duplicate_storage="physical_slots_counted_union_checked_bitwise",
            budget_version=1,
        )
        self._identity_json = _canonical(identity)
        self._indices = {}
        for role, capacity in config.quotas.items():
            scope = native.MemoryScope()
            scope.world, scope.partition, scope.fold = world, partition, fold
            scope.representation = self.fingerprint() + "/" + role
            self._indices[role] = native.EpisodicMemory(scope, config.seed, capacity, 2)
        self._scores, self._receipt = {}, None

    def identity(self):
        return json.loads(self._identity_json)

    def fingerprint(self):
        return hashlib.sha256(self._identity_json.encode()).hexdigest()

    def estimated_bytes(self, incoming):
        if not _integer(incoming, 0, 8192):
            raise ValueError("El lote M2 debe contener como máximo 8192 candidatos")
        # Reserva conjunta para estados viejo/nuevo, archivos, normalización y objetos.
        return 16 * 1024**2 + 4096 * (incoming + 2 * self.config.capacity)

    def _check_budget(self, incoming):
        if type(self.config) is not MatureErrorConfig or not _same(
            {key: getattr(self.config, key) for key in self._config_values}, self._config_values
        ):
            raise ValueError("La configuración M2 cambió fuera de su identidad")
        if self.estimated_bytes(incoming) > self.config.max_working_bytes:
            raise ValueError("La propuesta M2 supera el presupuesto conjunto de memoria")

    def _new(self):
        return type(self)(self._native, self.config, codec_id=self._codec_id, **self._scope)

    def _records(self):
        by_role, unique = {}, {}
        for role in _ROLES:
            by_role[role] = self._indices[role].retained_records()
            for record in by_role[role]:
                if not record.label_valid or record.reward_valid:
                    raise ValueError("M2 conserva etiquetas maduras y no recompensas")
                previous = unique.get(record.id)
                if previous is not None and _record_bytes(previous) != _record_bytes(record):
                    raise ValueError("Las copias de un episodio no conservan los mismos bits")
                unique[record.id] = record
        return by_role, unique

    @property
    def seen(self):
        return self._indices["reservoir"].seen

    @property
    def size(self):
        return len(self.records())

    @property
    def receipt(self):
        if self._receipt is None:
            return None
        self._check_receipt_shape(self._receipt)
        return json.loads(_canonical(self._receipt))

    @property
    def selective_scores(self):
        return dict(self._scores)

    def index_ids(self):
        by_role, _ = self._records()
        return {role: tuple(sorted(row.id for row in rows)) for role, rows in by_role.items()}

    def records(self):
        self._check_budget(0)
        _, unique = self._records()
        return [unique[key] for key in sorted(unique)]

    def _verify(self):
        self._check_budget(0)
        headers = [self._indices[role].snapshot_metadata() for role in _ROLES]
        if any(value != headers[0] for value in headers[1:]):
            raise ValueError("Los tres índices no conservan el mismo cursor maduro")
        header = headers[0]
        by_role, unique = self._records()
        if any(
            len(by_role[role]) != min(header["seen"], quota)
            for role, quota in self.config.quotas.items()
        ):
            raise ValueError("La ocupación de M2 no corresponde a sus cupos")
        if (
            not isinstance(self._scores, dict)
            or set(self._scores) != {row.id for row in by_role["selective"]}
            or any(
                type(key) is not int
                or type(value) is not float
                or not math.isfinite(value)
                or value < 0
                for key, value in self._scores.items()
            )
            or (header["seen"] and max(row.id for row in by_role["recent"]) != header["last_id"])
        ):
            raise ValueError("Las puntuaciones o la recencia de M2 no son coherentes")
        if not header["seen"]:
            if self._receipt is not None:
                raise ValueError("Un banco vacío no admite un recibo de ofertas")
        else:
            self._check_receipt(header, by_role, unique)
        return header, by_role, unique

    def _check_receipt_shape(self, receipt):
        fields = {
            "before_seen",
            "after_seen",
            "last_id",
            "confirmed_at",
            "offered_ids",
            "retained_ids",
            "index_ids",
            "physical_slots",
            "unique_episodes",
            "duplicate_slots",
            "index_offers",
            "score_evaluations",
            "estimated_peak_bytes",
            "before_ids",
            "new_unique_ids",
            "evicted_ids",
            "native_archive_bytes",
        }
        if type(receipt) is not dict or set(receipt) != fields:
            raise ValueError("El recibo de M2 no conserva su esquema")
        integer_fields = fields - {
            "offered_ids",
            "retained_ids",
            "index_ids",
            "before_ids",
            "new_unique_ids",
            "evicted_ids",
            "native_archive_bytes",
        }
        if any(not _integer(receipt[name], 0, 2**63 - 1) for name in integer_fields):
            raise ValueError("El recibo de M2 necesita contadores enteros")
        for name in ("offered_ids", "retained_ids", "before_ids", "new_unique_ids", "evicted_ids"):
            maximum = 8192 if name == "offered_ids" else self.config.capacity
            if not _ordered_ids(receipt[name], maximum):
                raise ValueError("El recibo de M2 contiene IDs inválidos")
        indices, sizes = receipt["index_ids"], receipt["native_archive_bytes"]
        if (
            type(indices) is not dict
            or set(indices) != set(_ROLES)
            or any(
                not _ordered_ids(indices[role], quota) for role, quota in self.config.quotas.items()
            )
            or type(sizes) is not dict
            or set(sizes) != set(_ROLES)
            or any(not _integer(size, 1, _MAX_ARCHIVE) for size in sizes.values())
        ):
            raise ValueError("Los índices o tamaños de archivo de M2 no son válidos")

    def _check_receipt(self, header, by_role, unique):
        receipt = self._receipt
        self._check_receipt_shape(receipt)
        offered, retained, before = map(
            set,
            (
                receipt["offered_ids"],
                receipt["retained_ids"],
                receipt["before_ids"],
            ),
        )
        index_ids = {role: sorted(row.id for row in rows) for role, rows in by_role.items()}
        slots = sum(map(len, by_role.values()))
        if (
            receipt["after_seen"] != header["seen"]
            or receipt["last_id"] != header["last_id"]
            or receipt["confirmed_at"] != header["confirmed_at"]
            or not offered
            or receipt["offered_ids"][-1] != header["last_id"]
            or receipt["after_seen"] - receipt["before_seen"] != len(offered)
            or not max(min(receipt["before_seen"], q) for q in self.config.quotas.values())
            <= len(before)
            <= min(receipt["before_seen"], self.config.capacity)
            or (before and max(before) >= receipt["offered_ids"][0])
            or not retained <= before | offered
            or retained != set(unique)
            or _canonical(receipt["index_ids"]) != _canonical(index_ids)
            or index_ids["recent"] != sorted(before | offered)[-self.config.quotas["recent"] :]
            or receipt["physical_slots"] != slots
            or receipt["unique_episodes"] != len(unique)
            or receipt["duplicate_slots"] != slots - len(unique)
            or receipt["index_offers"] != 3 * len(offered)
            or receipt["score_evaluations"] != len(offered)
            or receipt["new_unique_ids"] != sorted(retained - before)
            or receipt["evicted_ids"] != sorted(before - retained)
            or receipt["estimated_peak_bytes"] != self.estimated_bytes(len(offered))
            or receipt["estimated_peak_bytes"] > self.config.max_working_bytes
        ):
            raise ValueError("El recibo de M2 no concilia con sus índices y ofertas")

    def snapshot(self):
        self._verify()
        indices = {role: self._indices[role].snapshot_bytes() for role in _ROLES}
        if self._receipt is not None and self._receipt["native_archive_bytes"] != {
            role: len(value) for role, value in indices.items()
        }:
            raise ValueError("El recibo no conserva los bytes de sus tres archivos")
        return dict(
            identity=self.identity(),
            indices=indices,
            scores=dict(self._scores),
            receipt=self.receipt,
        )

    def restore(self, payload):
        if (
            not isinstance(payload, dict)
            or set(payload) != {"identity", "indices", "scores", "receipt"}
            or not _same(payload["identity"], self.identity())
            or not isinstance(payload["indices"], dict)
            or set(payload["indices"]) != set(_ROLES)
            or any(
                not isinstance(value, bytes) or not 1 <= len(value) <= _MAX_ARCHIVE
                for value in payload["indices"].values()
            )
            or not isinstance(payload["scores"], dict)
            or len(payload["scores"]) > self.config.quotas["selective"]
        ):
            raise ValueError("El snapshot no corresponde al contrato compuesto de M2")
        if payload["receipt"] is not None:
            self._check_receipt_shape(payload["receipt"])
        candidate = self._new()
        for role in _ROLES:
            candidate._indices[role].restore_bytes(payload["indices"][role])
        candidate._scores = dict(payload["scores"])
        candidate._receipt = json.loads(_canonical(payload["receipt"]))
        candidate._verify()
        if candidate._receipt is not None and candidate._receipt["native_archive_bytes"] != {
            role: len(payload["indices"][role]) for role in _ROLES
        }:
            raise ValueError("El recibo no conserva los bytes de sus tres archivos")
        return candidate

    def propose(self, incoming, *, errors, confirmed_at):
        if not isinstance(incoming, (list, tuple)) or not 1 <= len(incoming) <= 8192:
            raise ValueError("La propuesta M2 necesita entre 1 y 8192 candidatos")
        self._check_budget(len(incoming))
        header, old_indices, before = self._verify()
        identifiers = [row.id for row in incoming]
        if (
            not _integer(confirmed_at, 0, 2**63 - 1)
            or not isinstance(errors, dict)
            or set(errors) != set(identifiers)
            or any(
                type(key) is not int or type(value) is not float or not math.isfinite(value)
                for key, value in errors.items()
            )
            or any(not row.label_valid or row.reward_valid for row in incoming)
        ):
            raise ValueError("Falta el error financiero finito de cada candidato maduro")
        self._indices["reservoir"].validate_batch(incoming, confirmed_at)
        scores = {**self._scores, **{key: abs(value) for key, value in errors.items()}}
        selective = sorted(scores, key=lambda key: (-scores[key], key))[
            : self.config.quotas["selective"]
        ]
        recent = sorted([row.id for row in old_indices["recent"]] + identifiers)[
            -self.config.quotas["recent"] :
        ]
        candidate = self.restore(self.snapshot())
        for row in incoming:
            candidate._indices["reservoir"].write(row, confirmed_at)
        candidate._indices["selective"].retain_batch(incoming, sorted(selective), confirmed_at)
        candidate._indices["recent"].retain_batch(incoming, recent, confirmed_at)
        candidate._scores = {key: scores[key] for key in sorted(selective)}
        by_role, retained = candidate._records()
        slots = sum(map(len, by_role.values()))
        candidate._receipt = dict(
            before_seen=header["seen"],
            after_seen=header["seen"] + len(incoming),
            last_id=identifiers[-1],
            confirmed_at=confirmed_at,
            offered_ids=identifiers,
            before_ids=sorted(before),
            retained_ids=sorted(retained),
            index_ids={role: sorted(row.id for row in rows) for role, rows in by_role.items()},
            physical_slots=slots,
            unique_episodes=len(retained),
            duplicate_slots=slots - len(retained),
            index_offers=3 * len(incoming),
            score_evaluations=len(incoming),
            new_unique_ids=sorted(retained.keys() - before.keys()),
            evicted_ids=sorted(before.keys() - retained.keys()),
            estimated_peak_bytes=self.estimated_bytes(len(incoming)),
            native_archive_bytes={
                role: len(candidate._indices[role].snapshot_bytes()) for role in _ROLES
            },
        )
        candidate._verify()
        return candidate
