"""M2 por error maduro y M3 compuesta, con tres índices nativos y consulta deduplicada.

M2 y M3 comparten capacidad, cupos 50/25/25, ofertas, reservorio y recientes. Solo cambia la
puntuación del índice selectivo: el error absoluto en M2 y la combinación congelada de
`write_scores` en M3.
"""

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

from . import native_memory_archive, write_scores

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


def _quotas(capacity):
    numerators = (2 * capacity, capacity, capacity)
    result = [value // 4 for value in numerators]
    order = sorted(range(3), key=lambda index: (-(numerators[index] % 4), index))
    for index in order[: capacity - sum(result)]:
        result[index] += 1
    return dict(zip(_ROLES, result, strict=True))


def _budget(config):
    return (
        _integer(config.capacity, 4, 1024)
        and _integer(config.seed, 0, 2**64 - 1)
        and _integer(config.max_working_bytes, 1, 64 * 1024**2)
    )


@dataclass(frozen=True)
class MatureErrorConfig:
    """Un único B_mem para los cupos 50/25/25, con M desactivado."""

    capacity: int = 1024
    seed: int = 73
    max_working_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if not _budget(self):
            raise ValueError("La capacidad, semilla o presupuesto de M2 no son válidos")

    @property
    def quotas(self):
        return _quotas(self.capacity)


@dataclass(frozen=True)
class CompositeScoreConfig:
    """M3 con el mismo B_mem, cupos y semilla que M2 y escalas congeladas de entrenamiento."""

    scalers: write_scores.WriteScalers
    capacity: int = 1024
    seed: int = 73
    max_working_bytes: int = 64 * 1024**2
    weights: tuple = write_scores.WEIGHTS

    def __post_init__(self):
        weights = self.weights
        if (
            not _budget(self)
            or type(self.scalers) is not write_scores.WriteScalers
            or type(weights) is not tuple
            or len(weights) != 3
            or any(type(w) is not float or not math.isfinite(w) or w < 0 for w in weights)
            or abs(sum(weights) - 1.0) > 1e-12
        ):
            raise ValueError("M3 necesita capacidad, semilla, escalas y pesos válidos")

    @property
    def quotas(self):
        return _quotas(self.capacity)


_POLICIES = {
    MatureErrorConfig: ("episodic_m2_three_index_v1", "absolute_emitted_mature_error_fp64"),
    CompositeScoreConfig: (
        "episodic_m3_three_index_v1",
        "weighted_normalized_mature_error_anomaly_relevance_fp64",
    ),
}
# Campos del recibo que solo registra M3: componentes ofrecidos y cambios del selectivo.
_M3_RECEIPT = {
    "components",
    "selective_new_ids",
    "selective_rejected_ids",
    "selective_evicted_ids",
}


class MatureErrorBank:
    """Proponer copias del reservorio, top por puntuación y recientes sin publicar.

    Con `MatureErrorConfig` es M2. Con `CompositeScoreConfig` es M3 y conserva además los
    componentes sin normalizar de cada episodio retenido para volver a comprobar su selección.
    """

    def __init__(self, native, config, *, codec_id, world, partition, fold):
        if (
            type(config) not in _POLICIES
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
        self._composite = type(config) is CompositeScoreConfig
        self._check_budget(0)
        recipe, score = _POLICIES[type(config)]
        identity = dict(
            schema_version=1,
            recipe=recipe,
            config=asdict(config),
            scope=self._scope,
            codec_id=codec_id,
            quotas=config.quotas,
            allocation="integer_largest_remainders_2_1_1_role_order",
            native_sha256=native.binary_sha256,
            native_torch_version=native.torch_version,
            native_schema_version=2,
            code_sha256=_CODE,
            archive_validation_sha256=hashlib.sha256(
                Path(native_memory_archive.__file__).read_bytes()
            ).hexdigest(),
            reservoir="native_v2_mt19937_64_seed_only",
            score=score,
            selection="highest_score_then_lowest_mature_ordinal",
            recent="latest_mature_ordinals",
            cm_m="disabled",
            key_width=64,
            value_width=64,
            duplicate_storage="physical_slots_counted_union_checked_bitwise",
            budget_version=2,
        )
        if self._composite:
            identity.update(
                write_score=write_scores.declaration(),
                write_scores_sha256=hashlib.sha256(
                    Path(write_scores.__file__).read_bytes()
                ).hexdigest(),
                scalers_sha256=config.scalers.fingerprint(),
            )
        self._identity_json = _canonical(identity)
        self._indices = {}
        for role, capacity in config.quotas.items():
            scope = native.MemoryScope()
            scope.world, scope.partition, scope.fold = world, partition, fold
            scope.representation = self.fingerprint() + "/" + role
            self._indices[role] = native.EpisodicMemory(scope, config.seed, capacity, 2)
        # En M3, componentes sin normalizar de cada episodio retenido en U, S o R.
        self._scores, self._receipt, self._features = {}, None, {}

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
        if type(self.config) not in _POLICIES or not _same(
            asdict(self.config), self._config_values
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

    @property
    def write_features(self):
        """Componentes M3 sin normalizar por episodio retenido. Vacío en M2."""
        return {key: list(value) for key, value in self._features.items()}

    def _score(self, raw):
        e, a, r, _ = write_scores.normalized(self.config.scalers, raw)
        return write_scores.score(self.config.weights, e, a, r)

    def _check_features(self, by_role, unique):
        """M3: componentes de cada retenido y selectivo igual al mejor de los retenidos."""
        features = self._features
        if not isinstance(features, dict) or set(features) != set(unique):
            raise ValueError("M3 no conserva los componentes de cada episodio retenido")
        for raw in features.values():
            if (
                type(raw) is not list
                or len(raw) != 4
                or any(type(v) is not float or not math.isfinite(v) or v < 0 for v in raw[:2])
                or (
                    raw[2] is not None
                    and (type(raw[2]) is not float or not math.isfinite(raw[2]) or raw[2] < 0)
                )
                or type(raw[3]) is not bool
            ):
                raise ValueError("Los componentes M3 de un episodio no son válidos")
        scores = {key: self._score(raw) for key, raw in features.items()}
        top = sorted(scores, key=lambda key: (-scores[key], key))[: self.config.quotas["selective"]]
        if sorted(top) != sorted(row.id for row in by_role["selective"]) or self._scores != {
            key: scores[key] for key in sorted(top)
        }:
            raise ValueError("El índice selectivo M3 no conserva las mayores puntuaciones")

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
        if self._composite:
            self._check_features(by_role, unique)
        elif self._features:
            raise ValueError("M2 no conserva componentes de M3")
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
        if type(receipt) is not dict or set(receipt) != fields | (
            _M3_RECEIPT if self._composite else set()
        ):
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
        if self._composite:
            self._check_components_shape(receipt)

    def _check_components_shape(self, receipt):
        quota = self.config.quotas["selective"]
        rows = receipt["components"]
        if (
            type(rows) is not list
            or not 1 <= len(rows) <= 8192
            or any(
                type(row) is not list
                or len(row) != 6
                or not _integer(row[0], 1, 2**63 - 1)
                or any(type(v) is not float or not 0 <= v <= 1 for v in row[1:5])
                or not _integer(row[5], 0, 3)
                for row in rows
            )
            or [row[0] for row in rows] != receipt["offered_ids"]
            or not _ordered_ids(receipt["selective_new_ids"], quota)
            or not _ordered_ids(receipt["selective_evicted_ids"], quota)
            or not _ordered_ids(receipt["selective_rejected_ids"], 8192)
        ):
            raise ValueError("Los componentes del recibo M3 no son válidos")

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
        if self._composite:
            self._check_selective_changes(receipt, offered, index_ids["selective"])

    def _check_selective_changes(self, receipt, offered, selective):
        """Admitidos, rechazados y expulsados del selectivo, y puntuaciones de las ofertas."""
        selective, quota = set(selective), self.config.quotas["selective"]
        new, evicted = set(receipt["selective_new_ids"]), set(receipt["selective_evicted_ids"])
        before = min(receipt["before_seen"], quota)
        weights = self.config.weights
        if (
            new != offered & selective
            or set(receipt["selective_rejected_ids"]) != offered - selective
            or evicted & (selective | offered)
            or not evicted <= set(receipt["before_ids"])
            or len(selective) != before + len(new) - len(evicted)
            or any(
                row[4] != write_scores.score(weights, *row[1:4])
                or (row[0] in self._features and row[1:6] != self._components(row[0]))
                for row in receipt["components"]
            )
        ):
            raise ValueError("El recibo M3 no concilia sus admisiones, rechazos y expulsiones")

    def _components(self, key):
        e, a, r, mask = write_scores.normalized(self.config.scalers, self._features[key])
        return [e, a, r, write_scores.score(self.config.weights, e, a, r), mask]

    def snapshot(self):
        self._verify()
        indices = {role: self._indices[role].snapshot_bytes() for role in _ROLES}
        if self._receipt is not None and self._receipt["native_archive_bytes"] != {
            role: len(value) for role, value in indices.items()
        }:
            raise ValueError("El recibo no conserva los bytes de sus tres archivos")
        result = dict(
            identity=self.identity(),
            indices=indices,
            scores=dict(self._scores),
            receipt=self.receipt,
        )
        if self._composite:
            result["features"] = self.write_features
        return result

    def restore(self, payload):
        fields = {"identity", "indices", "scores", "receipt"}
        if self._composite:
            fields.add("features")
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or not _same(payload["identity"], self.identity())
            or not isinstance(payload["indices"], dict)
            or set(payload["indices"]) != set(_ROLES)
            or any(
                not isinstance(value, bytes) or not 1 <= len(value) <= _MAX_ARCHIVE
                for value in payload["indices"].values()
            )
            or not isinstance(payload["scores"], dict)
            or len(payload["scores"]) > self.config.quotas["selective"]
            or (
                self._composite
                and (
                    not isinstance(payload["features"], dict)
                    or len(payload["features"]) > self.config.capacity
                )
            )
        ):
            raise ValueError("El snapshot no corresponde al contrato compuesto de M2")
        if payload["receipt"] is not None:
            self._check_receipt_shape(payload["receipt"])
        # Reservar las copias codificadas y dos materializaciones de los storages.
        available = (
            self.config.max_working_bytes
            - self.estimated_bytes(0)
            - sum(map(len, payload["indices"].values()))
        )
        native_memory_archive.verify_memory_archives(
            payload["indices"], self.config.quotas, max_expanded_bytes=max(available // 2, 0)
        )
        candidate = self._new()
        for role in _ROLES:
            candidate._indices[role].restore_bytes(payload["indices"][role])
        candidate._scores = dict(payload["scores"])
        if self._composite:
            candidate._features = {
                key: list(value) if isinstance(value, (list, tuple)) else value
                for key, value in payload["features"].items()
            }
        candidate._receipt = json.loads(_canonical(payload["receipt"]))
        candidate._verify()
        if candidate._receipt is not None and candidate._receipt["native_archive_bytes"] != {
            role: len(payload["indices"][role]) for role in _ROLES
        }:
            raise ValueError("El recibo no conserva los bytes de sus tres archivos")
        return candidate

    def propose(self, incoming, *, errors, confirmed_at, features=None):
        """Ofrecer cada candidato maduro a los tres índices sobre una copia del banco.

        `errors` es el error firmado `etiqueta - predicción emitida`. M3 recibe además los
        `DecisionFeatures` de cada candidato, conocidos en el corte de su decisión.
        """
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
        if self._composite != (features is not None) or (
            features is not None
            and (not isinstance(features, dict) or set(features) != set(identifiers))
        ):
            raise ValueError("M3 necesita los rasgos de cada candidato y M2 no los admite")
        self._indices["reservoir"].validate_batch(incoming, confirmed_at)
        if self._composite:
            raw = {key: write_scores.raw_components(errors[key], features[key]) for key in errors}
            offered = {key: self._score(value) for key, value in raw.items()}
        else:
            offered = {key: abs(value) for key, value in errors.items()}
        scores = {**self._scores, **offered}
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
        if self._composite:
            known = {**self._features, **raw}
            candidate._features = {key: known[key] for key in sorted(retained)}
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
        if self._composite:
            chosen, previous = set(selective), set(self._scores)
            components = []
            for key in identifiers:
                e, a, r, mask = write_scores.normalized(self.config.scalers, raw[key])
                components.append([key, e, a, r, offered[key], mask])
            candidate._receipt.update(
                components=components,
                selective_new_ids=sorted(chosen & set(identifiers)),
                selective_rejected_ids=sorted(set(identifiers) - chosen),
                selective_evicted_ids=sorted(previous - chosen),
            )
        candidate._verify()
        return candidate
