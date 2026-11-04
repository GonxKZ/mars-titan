"""Retención E2 de registros nativos maduros, con propuestas de banco y evidencia separadas."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from mars_titan.models.titans import episodic_snapshot

from . import financial_consumers, spaced_utility, spaced_utility_state
from .financial_consumers import bank_snapshot
from .spaced_utility import (
    SpacedUtilityConfig,
    SpacedUtilityLedger,
    _canonical,
    _copy,
    _integer,
    _require,
)

POLICIES = ("utility", "permuted_utility", "uniform_frontier", "recent", "reservoir")
_CODE = {
    "spaced_retention": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "spaced_retention_state": hashlib.sha256(
        Path(__file__).with_name("spaced_retention_state.py").read_bytes()
    ).hexdigest(),
    "spaced_utility": hashlib.sha256(Path(spaced_utility.__file__).read_bytes()).hexdigest(),
    "spaced_utility_state": hashlib.sha256(
        Path(spaced_utility_state.__file__).read_bytes()
    ).hexdigest(),
    "financial_consumers": hashlib.sha256(
        Path(financial_consumers.__file__).read_bytes()
    ).hexdigest(),
    "episodic_snapshot": hashlib.sha256(Path(episodic_snapshot.__file__).read_bytes()).hexdigest(),
}


@dataclass(frozen=True)
class SpacedRetentionConfig:
    policy: str = "utility"
    capacity: int = 1024
    frontier: int = 8
    seed: int = 73
    max_working_bytes: int = 512 * 1024**2

    def __post_init__(self):
        _require(
            type(self.policy) is str and self.policy in POLICIES, "La política E2 no es válida"
        )
        _require(_integer(self.capacity, 1, 1024), "La capacidad E2 debe pertenecer a [1, 1024]")
        _require(_integer(self.frontier, 1, min(self.capacity, 32)), "La frontera E2 no es válida")
        _require(_integer(self.seed, 0, 2**64 - 1), "La semilla E2 no es válida")
        _require(
            _integer(self.max_working_bytes, 16 * 1024**2, 512 * 1024**2),
            "El presupuesto E2 debe pertenecer a [16, 512] MiB",
        )


@dataclass(frozen=True)
class SpacedRetentionProposal:
    bank: "SpacedUtilityBank"
    evidence: SpacedUtilityLedger


def working_bytes(clients, usage):
    """Estimar copias nativas y del ledger. No es una cota certificada de RAM."""
    return (
        8 * 1024**2
        + 4096 * (clients + usage["pending_rows"])
        + 512 * usage["references"]
        + 2048 * usage["episodes"]
    )


class SpacedUtilityBank:
    """Seleccionar tras la maduración. El coordinador debe publicar conjuntamente la propuesta."""

    def __init__(
        self, native, config, *, codec_id, model_id, world, partition, fold, evidence_config=None
    ):
        from mars_titan.models.titans.episodic_snapshot import _digest_id

        _require(type(config) is SpacedRetentionConfig, "El banco E2 necesita su configuración")
        _digest_id(codec_id, "El codec del banco E2")
        _digest_id(model_id, "El predictor del banco E2")
        self._config, self._native, self._codec_id = config, native, codec_id
        self._model_id = model_id
        self._scope = dict(world=world, partition=partition, fold=fold)
        _require(
            all(type(v) is str and 1 <= len(v) <= 128 for v in self._scope.values()),
            "El banco E2 necesita mundo, partición y ventana identificados",
        )
        self.evidence_config = (
            SpacedUtilityConfig(capacity=config.capacity)
            if evidence_config is None
            else evidence_config
        )
        _require(
            type(self.evidence_config) is SpacedUtilityConfig
            and self.evidence_config.capacity == config.capacity,
            "La evidencia y el banco E2 necesitan la misma capacidad",
        )
        self._identity_json = _canonical(
            dict(
                schema_version=1,
                recipe="spaced_utility_retention_v1",
                implementation=_CODE,
                codec_id=codec_id,
                model_id=model_id,
                config=asdict(config),
                evidence_config=self.evidence_config.identity(),
                scope=self._scope,
                native_sha256=native.binary_sha256,
                native_torch_version=native.torch_version,
                numpy_version=np.__version__,
                memory_contract="causal_v2",
                frontier="latest_admitted_ids_except_original_reservoir",
                utility="qualified_old_mean_then_low_id_fill_latest_admission",
                uniform="conditional_current_candidates_without_replacement",
                permutation="old_score_assignment_only_not_evidence_rewrite",
                rng="PCG64_SeedSequence_low_high_e2_domain_policy_tag",
            )
        )
        scope = native.MemoryScope()
        scope.world, scope.partition, scope.fold = world, partition, fold
        scope.representation = self.fingerprint()
        self._memory = native.EpisodicMemory(scope, config.seed, config.capacity, 2)
        seed = np.random.SeedSequence(
            [config.seed & 0xFFFFFFFF, config.seed >> 32, 0xE2000001, POLICIES.index(config.policy)]
        )
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._receipt = None

    @property
    def config(self):
        return self._config

    @property
    def seen(self):
        return self._memory.seen

    @property
    def receipt(self):
        return _copy(self._receipt)

    def identity(self):
        return json.loads(self._identity_json)

    def fingerprint(self):
        return hashlib.sha256(self._identity_json.encode()).hexdigest()

    def records(self):
        return sorted(self._memory.retained_records(), key=lambda record: record.id)

    def retained_arrays(self):
        export = getattr(self._memory, "retained_arrays", None)
        return None if export is None else export()

    def _new(self):
        return type(self)(
            self._native,
            self.config,
            codec_id=self._codec_id,
            model_id=self._model_id,
            evidence_config=self.evidence_config,
            **self._scope,
        )

    def query(self, key, cutoff, *, exclude_id=None):
        _require(
            isinstance(key, np.ndarray)
            and key.dtype == np.float32
            and key.shape == (64,)
            and key.flags.c_contiguous
            and np.isfinite(key).all()
            and _integer(cutoff),
            "La consulta E2 necesita una clave FP32 finita de 64 coordenadas y corte entero",
        )
        return self._memory.query(key.tolist(), cutoff, exclude_id)

    def _evidence_header(self, evidence, cutoff):
        _require(
            type(evidence) is SpacedUtilityLedger
            and evidence.codec_id == self._codec_id
            and evidence.scope_id == self.fingerprint()
            and evidence._phase is None
            and evidence.config.identity() == self.evidence_config.identity(),
            "La evidencia no pertenece a este banco E2",
        )
        metadata = self._memory.snapshot_metadata()
        _require(
            _integer(evidence.clock)
            and evidence.clock >= metadata["confirmed_at"]
            and (cutoff is None or evidence.clock == cutoff),
            "La evidencia E2 no tiene el corte publicado requerido",
        )

    def _evidence(self, evidence, *, cutoff=None):
        self._evidence_header(evidence, cutoff)
        payload = evidence.export()
        registry = {i: score["maturity_at"] for i, score in payload["episodes"]}
        _require(
            registry == {r.id: r.maturity_at for r in self.records()},
            "La evidencia E2 no conserva los episodios y maduraciones del banco",
        )
        return payload

    def validate_evidence(self, evidence):
        """Conciliar la pareja recuperada. Esta comprobación no autentica su historia."""
        self._evidence(evidence)

    def _selection(self, old, incoming, evidence):
        clients = sorted([r.id for r in old] + [r.id for r in incoming])
        if len(clients) <= self.config.capacity:
            return clients, [], [], []
        frontier = [r.id for r in incoming][-self.config.frontier :]
        slots = self.config.capacity - len(frontier)
        ranking = []
        if self.config.policy == "uniform_frontier":
            reserved = set(frontier)
            remaining = [i for i in clients if i not in reserved]
            chosen = self._rng.choice(remaining, size=slots, replace=False).tolist()
            return sorted(frontier + chosen), [], frontier, ranking
        if self.config.policy in ("utility", "permuted_utility"):
            ids = sorted(r.id for r in old)
            sources = (
                self._rng.permutation(ids).tolist()
                if self.config.policy == "permuted_utility"
                else ids
            )
            scores = evidence.scores()
            ranking = [
                [i, source, scores[source]["mean_gain"]]
                for i, source in zip(ids, sources, strict=True)
                if scores[source]["qualified"]
            ]
            ranking.sort(key=lambda item: (-item[2], item[0]))
        protected = sorted(item[0] for item in ranking[:slots])
        anchors = set(frontier) | set(protected)
        fill = sorted((i for i in clients if i not in anchors), reverse=True)
        return (
            sorted([*anchors, *fill[: self.config.capacity - len(anchors)]]),
            protected,
            frontier,
            ranking,
        )

    def propose(self, incoming, *, confirmed_at, evidence):
        _require(
            isinstance(incoming, (list, tuple))
            and 1 <= len(incoming) <= 8192
            and _integer(confirmed_at)
            and type(evidence) is SpacedUtilityLedger,
            "La propuesta E2 necesita un lote maduro acotado, corte y evidencia",
        )
        self._evidence_header(evidence, confirmed_at)
        usage = evidence.usage()
        estimate = working_bytes(self._memory.size + len(incoming), usage)
        _require(estimate <= self.config.max_working_bytes, "La propuesta E2 supera su presupuesto")
        payload = self._evidence(evidence, cutoff=confirmed_at)
        _require(
            all(type(record) is self._native.MemoryRecord for record in incoming),
            "La propuesta E2 necesita registros del enlace nativo declarado",
        )
        _require(all(_integer(r.id, 1) for r in incoming), "Los IDs E2 deben caber en int64")
        normalized = self._memory.validate_batch(incoming, confirmed_at)
        old, before = self.records(), dict(self._memory.snapshot_metadata())
        candidate = self._new()
        candidate._memory.restore_from(self._memory)
        candidate._rng.bit_generator.state = self._rng.bit_generator.state
        if self.config.policy == "reservoir":
            for record in incoming:
                candidate._memory.write(record, confirmed_at)
            retained = [r.id for r in candidate.records()]
            protected, frontier, ranking = [], [], []
        else:
            retained, protected, frontier, ranking = candidate._selection(old, normalized, evidence)
            # El enlace normaliza los originales una vez. No escribir los ya normalizados.
            candidate._memory.retain_batch(incoming, retained, confirmed_at)
        following = SpacedUtilityLedger.restore(
            self._codec_id, self.evidence_config, payload, scope_id=self.fingerprint()
        )
        following.synchronize(
            bank_snapshot(
                candidate,
                codec_id=self._codec_id,
                context_id=self.fingerprint(),
                cutoff=confirmed_at,
                dtype=torch.float64,
                device="cpu",
            )
        )
        candidate._receipt = dict(
            before_seen=before["seen"],
            after_seen=candidate.seen,
            before_last_id=before["last_id"],
            confirmed_at=confirmed_at,
            old_ids=[r.id for r in old],
            new_ids=[r.id for r in normalized],
            retained_ids=retained,
            protected_ids=protected,
            frontier_ids=frontier,
            ranking=ranking,
            evidence_sha256=payload["sha256"],
            ledger_usage=usage,
            estimated_working_bytes=estimate,
        )
        return SpacedRetentionProposal(candidate, following)

    def snapshot(self):
        from .spaced_retention_state import checksum

        state = dict(
            schema_version=1,
            identity=self.identity(),
            memory=self._memory.snapshot_bytes(),
            rng=_copy(self._rng.bit_generator.state),
            receipt=self.receipt,
        )
        return dict(state, sha256=checksum(state))

    def restore(self, payload):
        from .spaced_retention_state import restore

        return restore(self._new(), payload)
