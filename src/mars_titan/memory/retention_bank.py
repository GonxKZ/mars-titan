"""Políticas de retención sobre episodios maduros del banco nativo de 64 coordenadas."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mars_titan.cm import anchored_medoids, medoids
from mars_titan.cm.anchored_medoids import select_anchored_medoids
from mars_titan.cm.medoids import MedoidBudgetExceeded

_CODE = {
    "retention_bank": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "anchored_medoids": hashlib.sha256(Path(anchored_medoids.__file__).read_bytes()).hexdigest(),
    "medoids": hashlib.sha256(Path(medoids.__file__).read_bytes()).hexdigest(),
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_canonical(value))


def _check_receipt_selection(receipt, config):
    clients, fixed, candidates, retained = (
        set(receipt[key]) for key in ("client_ids", "fixed_ids", "candidate_ids", "retained_ids")
    )
    variable_count = len(retained) - len(fixed)
    restricted = config.policy == "anchored" and len(clients) > config.capacity
    fixed_count = config.capacity - config.frontier if restricted else len(retained)
    if (
        not fixed <= retained <= clients
        or not candidates <= clients
        or fixed & candidates
        or not retained <= fixed | candidates
        or len(fixed) != fixed_count
        or (not restricted and candidates)
        or (
            restricted
            and not config.frontier <= len(candidates) <= config.frontier + config.new_candidates
        )
    ):
        raise ValueError("Los centros del recibo no concuerdan con E y la capacidad")
    expected_status = (
        "fixed_only"
        if not candidates
        else "all_candidates"
        if variable_count == len(candidates)
        else "one_swap_local_restricted"
    )
    if receipt["status"] != expected_status:
        raise ValueError("El estado del selector no corresponde a una retención completa")
    if (
        receipt["background_pairs"] != (len(clients) - len(fixed)) * len(fixed)
        or receipt["distance_pairs"] != receipt["background_pairs"] + receipt["variable_pairs"]
        or receipt["distance_pairs"] > config.max_distance_pairs
        or (not candidates and receipt["variable_pairs"] != 0)
        or (candidates and receipt["variable_pairs"] == 0)
        or not 4096 * len(clients) + 8 * 1024**2
        <= receipt["estimated_peak_bytes"]
        <= config.max_working_bytes
        or (len(clients) == len(retained) and receipt["objective"] != 0.0)
    ):
        raise ValueError("El coste del recibo no concuerda con la selección y sus presupuestos")


def _check_receipt(receipt, bank, config):
    if receipt is None:
        if bank.seen != 0:
            raise ValueError("Falta el recibo de un banco no vacío")
        return
    if not isinstance(receipt, dict):
        raise ValueError("El recibo de retención necesita un objeto")
    counters = (
        "before_seen",
        "after_seen",
        "confirmed_at",
        "distance_pairs",
        "background_pairs",
        "variable_pairs",
        "estimated_peak_bytes",
    )
    id_fields = ("client_ids", "fixed_ids", "candidate_ids", "retained_ids")
    fields = {
        *counters,
        *id_fields,
        "policy",
        "objective",
        "status",
        "coordinate_dtype",
        "distance_dtype",
        "background_backend",
        "client_geometry_sha256",
    }
    if set(receipt) != fields:
        raise ValueError(
            "El esquema del recibo de retención está incompleto o contiene campos ajenos"
        )
    if any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in counters):
        raise ValueError("El recibo necesita contadores enteros no negativos")
    for key in id_fields:
        ids = receipt.get(key)
        if (
            not isinstance(ids, list)
            or len(ids) > config.capacity + 8192
            or any(type(i) is not int or not 0 < i < 2**63 for i in ids)
            or ids != sorted(set(ids))
        ):
            raise ValueError("El recibo contiene IDs inválidos o desordenados")
    state = bank.snapshot_metadata()
    admitted = receipt["after_seen"] - receipt["before_seen"]
    expected_clients = min(receipt["before_seen"], config.capacity) + admitted
    if (
        receipt["after_seen"] != state["seen"]
        or receipt["confirmed_at"] != state["confirmed_at"]
        or not 1 <= admitted <= 8192
        or len(receipt["client_ids"]) != expected_clients
        or not receipt["client_ids"]
        or receipt["client_ids"][-1] != state["last_id"]
        or receipt["retained_ids"] != sorted(r.id for r in bank.retained_records())
        or len(receipt["retained_ids"]) != min(state["seen"], config.capacity)
        or receipt.get("policy") != config.policy
        or type(receipt.get("objective")) is not float
        or not math.isfinite(receipt["objective"])
        or receipt["objective"] < 0
    ):
        raise ValueError("El recibo no concuerda con el banco retenido")
    geometry = receipt["client_geometry_sha256"]
    if (
        receipt["coordinate_dtype"] != "float32"
        or receipt["distance_dtype"] != "float64"
        or receipt["background_backend"] != "numpy"
        or not isinstance(geometry, str)
        or len(geometry) != 64
        or any(value not in "0123456789abcdef" for value in geometry)
    ):
        raise ValueError("La representación o la huella del recibo no pertenece al contrato")
    _check_receipt_selection(receipt, config)


@dataclass(frozen=True)
class RetentionConfig:
    """Capacidad nativa y búsqueda restringida, sin garantía de óptimo global."""

    policy: str = "reservoir"
    capacity: int = 1024
    seed: int = 73
    frontier: int = 8
    new_candidates: int = 8
    max_swaps: int = 8
    max_distance_pairs: int = 50_000_000
    max_working_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if self.policy not in ("reservoir", "uniform", "recent", "anchored"):
            raise ValueError("La política episódica no está admitida")
        bounds = (
            (self.capacity, 1, 1024),
            (self.seed, 0, 2**64 - 1),
            (self.frontier, 1, min(self.capacity, 32)),
            (self.new_candidates, self.frontier, 32),
            (self.max_swaps, 1, 100),
            (self.max_distance_pairs, 1, 50_000_000),
            (self.max_working_bytes, 16 * 1024**2, 64 * 1024**2),
        )
        if any(type(value) is not int or not low <= value <= high for value, low, high in bounds):
            raise ValueError("La configuración de retención excede sus límites")


class RetentionBank:
    """Proponer una copia del banco. El coordinador publica la transición completa."""

    def __init__(self, native, config, *, codec_id, world, partition, fold):
        if (
            not isinstance(config, RetentionConfig)
            or not isinstance(codec_id, str)
            or len(codec_id) != 64
            or any(value not in "0123456789abcdef" for value in codec_id)
        ):
            raise ValueError("Falta la identidad de la retención o del codec")
        self.config = config
        self._native, self._codec_id = native, codec_id
        self._scope = dict(world=world, partition=partition, fold=fold)
        identity = dict(
            schema_version=1,
            implementation=_CODE,
            recipe="native_episodic_retention_v1",
            codec_id=codec_id,
            config=asdict(config),
            scope=self._scope,
            key_width=64,
            value_width=64,
            coordinate_dtype="float32",
            metric="euclidean_on_native_normalized_keys",
            background_backend="numpy",
            numpy_version=np.__version__,
            native_sha256=native.binary_sha256,
            native_torch_version=native.torch_version,
            uniform_rng="PCG64",
            anchored_frontier="oldest_retained_then_earliest_new_fixed_fill",
            anchored_candidates="old_frontier_plus_seeded_hash_rank_of_remaining_new",
            partial_selection="reject",
        )
        self._identity_json = _canonical(identity)
        scope = native.MemoryScope()
        scope.world, scope.partition, scope.fold = world, partition, fold
        scope.representation = self.fingerprint()
        self._memory = native.EpisodicMemory(scope, config.seed, config.capacity)
        self._rng = np.random.Generator(np.random.PCG64(config.seed))
        self._receipt = None

    def identity(self):
        return json.loads(self._identity_json)

    def fingerprint(self):
        return hashlib.sha256(self._identity_json.encode()).hexdigest()

    @property
    def seen(self):
        return self._memory.seen

    @property
    def receipt(self):
        return _copy(self._receipt)

    def records(self):
        return self._memory.retained_records()

    def _new(self):
        return type(self)(self._native, self.config, codec_id=self._codec_id, **self._scope)

    def snapshot(self):
        return dict(
            identity=self.identity(),
            memory=self._memory.snapshot_bytes(),
            retention_rng=_copy(self._rng.bit_generator.state),
            receipt=self.receipt,
        )

    def restore(self, payload):
        if (
            not isinstance(payload, dict)
            or set(payload) != {"identity", "memory", "retention_rng", "receipt"}
            or _canonical(payload["identity"]) != self._identity_json
            or not isinstance(payload["memory"], bytes)
            or len(payload["memory"]) > 2 * 1024**2
        ):
            raise ValueError("El snapshot pertenece a otro contrato de retención")
        candidate = self._new()
        candidate._memory.restore_bytes(payload["memory"])
        try:
            candidate._rng.bit_generator.state = _copy(payload["retention_rng"])
        except (TypeError, KeyError) as error:
            raise ValueError("El estado del generador de retención no es válido") from error
        if _canonical(candidate._rng.bit_generator.state) != _canonical(payload["retention_rng"]):
            raise ValueError("Los tipos del generador de retención han cambiado")
        receipt = _copy(payload["receipt"])
        _check_receipt(receipt, candidate._memory, self.config)
        candidate._receipt = receipt
        return candidate

    def query(self, key, cutoff, *, exclude_id=None):
        if (
            not isinstance(key, np.ndarray)
            or key.dtype != np.float32
            or key.shape != (64,)
            or not key.flags.c_contiguous
            or not np.isfinite(key).all()
            or type(cutoff) is not int
        ):
            raise ValueError("La consulta necesita una clave FP32 de 64 coordenadas y corte entero")
        return self._memory.query(key.tolist(), cutoff, exclude_id)

    def _centers(self, old, incoming):
        frontier = min(self.config.frontier, self.config.capacity)
        fixed = [r.id for r in sorted(old, key=lambda row: row.id)[frontier:]]
        for record in incoming:
            if len(fixed) == self.config.capacity - frontier:
                break
            fixed.append(record.id)
        fixed_set = set(fixed)
        candidates = [r.id for r in old if r.id not in fixed_set]
        remaining = [r.id for r in incoming if r.id not in fixed_set]
        remaining.sort(
            key=lambda identifier: (
                hashlib.sha256(f"{self.config.seed}/{self.seen}/{identifier}".encode()).digest(),
                identifier,
            )
        )
        candidates.extend(remaining[: self.config.new_candidates])
        return sorted(fixed), sorted(candidates)

    def propose(self, incoming, *, confirmed_at):
        if (
            not isinstance(incoming, (list, tuple))
            or not 1 <= len(incoming) <= 8192
            or type(confirmed_at) is not int
        ):
            raise ValueError("La propuesta necesita entre 1 y 8192 episodios y un corte entero")
        size = self._memory.size + len(incoming)
        # Incluye copias de registros y claves, archivos transitorios y objetos Python.
        own_bytes = 4096 * size + 8 * 1024**2
        remaining = self.config.max_working_bytes - own_bytes
        if remaining <= 0:
            raise MedoidBudgetExceeded("Los registros del banco agotaron el presupuesto")
        normalized = self._memory.validate_batch(incoming, confirmed_at)
        old = self.records()
        eligible = sorted((*old, *normalized), key=lambda row: row.id)
        ids = [r.id for r in eligible]
        names = [f"{identifier:020d}" for identifier in ids]
        points = np.asarray([r.key for r in eligible], dtype=np.float32)
        candidate = self.restore(self.snapshot())
        if self.config.policy == "reservoir":
            for record in incoming:
                candidate._memory.write(record, confirmed_at)
            retained = sorted(r.id for r in candidate.records())
            fixed, candidates = retained, []
        elif self.config.policy == "uniform":
            retained = [r.id for r in old]
            for record in incoming:
                if len(retained) < self.config.capacity:
                    retained.append(record.id)
                else:
                    retained[int(candidate._rng.integers(self.config.capacity))] = record.id
            fixed, candidates = sorted(retained), []
        elif self.config.policy == "recent":
            fixed, candidates = ids[-self.config.capacity :], []
        elif len(eligible) <= self.config.capacity:
            fixed, candidates = ids, []
        else:
            fixed, candidates = self._centers(old, incoming)
        selection = select_anchored_medoids(
            points,
            names,
            [f"{identifier:020d}" for identifier in fixed],
            [f"{identifier:020d}" for identifier in candidates],
            min(self.config.capacity, len(eligible)),
            background_backend="numpy",
            max_working_bytes=remaining,
            max_distance_pairs=self.config.max_distance_pairs,
            max_swaps=self.config.max_swaps,
        )
        if selection.status == "swap_limit":
            raise MedoidBudgetExceeded("La retención no completó los intercambios permitidos")
        retained = sorted(int(identifier) for identifier in selection.retained_ids)
        if self.config.policy != "reservoir":
            candidate._memory.retain_batch(incoming, retained, confirmed_at)
        candidate._receipt = dict(
            policy=self.config.policy,
            before_seen=self.seen,
            after_seen=candidate.seen,
            confirmed_at=confirmed_at,
            client_ids=ids,
            fixed_ids=fixed,
            candidate_ids=candidates,
            retained_ids=retained,
            objective=float(selection.objective),
            status=selection.status,
            distance_pairs=selection.distance_pairs,
            background_pairs=selection.background_pairs,
            variable_pairs=selection.variable_pairs,
            estimated_peak_bytes=own_bytes + selection.estimated_peak_bytes,
            coordinate_dtype="float32",
            distance_dtype="float64",
            background_backend="numpy",
            client_geometry_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        )
        return candidate
