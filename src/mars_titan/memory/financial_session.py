"""Ciclo financiero congelado sobre el ejecutor, el banco y los bloques existentes."""

import hashlib
import inspect
import json
import math
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.input_policy import MODALITIES
from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_inputs import (
    FINAL_TEST_US,
    DecisionBatch,
    validated_cpu_batch,
)
from mars_titan.training import prefix_eligibility
from mars_titan.training.prefix_eligibility import PrefixEvidence, PrefixTargetVerifier

from . import episodic_session, financial_consumers, financial_state_artifacts
from .episodic_session import (
    EpisodicSession,
    _canonical,
    _check_pending,
    _digest,
    _empty_pending,
    _row_key,
)
from .financial_consumers import ADMISSION_ERROR, SOURCE_ERROR, bind_consumer
from .session_artifacts import SessionArtifacts


def _callback_signature(instance):
    result = []
    for name, function in inspect.getmembers(FinancialSession, inspect.isfunction):
        if name in vars(instance) or function.__module__ not in {
            __name__,
            episodic_session.__name__,
        }:
            raise ValueError("La sesión no admite métodos o callbacks sustituidos")
        result.append((name, id(function), id(function.__code__)))
    return tuple(result)


def _binding_signature(binding):
    kind = type(binding)
    methods = inspect.getmembers(kind, inspect.isfunction)
    if kind is not financial_consumers.TitansBinding or any(
        name in vars(binding) for name, _ in methods
    ):
        raise ValueError("El enlace del consumidor no admite tipos o métodos sustituidos")
    return kind, tuple((name, id(function.__code__)) for name, function in methods)


def _check_callbacks(instance):
    if (
        _callback_signature(instance) != instance._callback_versions
        or _binding_signature(instance._binding) != instance._binding_versions
    ):
        raise ValueError("Los callbacks cambiaron respecto de la sesión identificada")
    current = (
        instance.contract_id,
        instance.model_id,
        instance.consumer.model_id,
        instance.prefixes.policy_id,
        asdict(instance.phase),
        instance.block_rows,
        instance.admission,
        instance.max_input_blocks,
    )
    if current != instance._runtime_contract:
        raise ValueError("La configuración efectiva cambió durante la sesión")


def _invoke(instance, method, arguments):
    _check_callbacks(instance)
    return getattr(FinancialSession, method)(instance, *arguments)


@dataclass(frozen=True)
class FinancialPhase:
    partition: str
    warmup_start: int
    decision_start: int
    decision_end: int
    close_at: int

    def __post_init__(self):
        if (
            self.partition not in {"train", "validation", "calibration", "evaluation"}
            or any(
                type(value) is not int
                for value in (
                    self.warmup_start,
                    self.decision_start,
                    self.decision_end,
                    self.close_at,
                )
            )
            or not 0
            < self.warmup_start
            <= self.decision_start
            < self.decision_end
            <= self.close_at
            <= FINAL_TEST_US
        ):
            raise ValueError("La fase necesita intervalos explícitos sin abrir los datos de 2024")


class FinancialSession(EpisodicSession):
    """Reutilizar almacenamiento, codec y banco. Executor publica la única generación.

    La preparación queda fijada a FrozenFinancialConsumer mediante su enlace
    cerrado. No se aceptan callbacks predictivos arbitrarios ni reglas M3
    incompletas. Los helpers heredados conservan la validación de inputs
    y procedencia de la sesión v1.
    """

    def __init__(
        self,
        output,
        *,
        native,
        consumer,
        codec,
        prefixes,
        retention,
        phase,
        world,
        fold,
        admission="m0",
        block_rows=64,
        max_input_blocks=128,
        max_log_bytes=16 * 1024**3,
        resume=False,
    ):
        if (
            type(self) is not FinancialSession
            or type(prefixes) is not PrefixTargetVerifier
            or type(phase) is not FinancialPhase
            or type(resume) is not bool
        ):
            raise ValueError(ADMISSION_ERROR)
        binding = bind_consumer(consumer, codec, retention, admission)
        bounded_integer(block_rows, "lote físico", 1, binding.max_batch)
        bounded_integer(max_input_blocks, "bloques de inputs pendientes", 1, 256)
        bounded_integer(max_log_bytes, "registro de la fase", 1, 16 * 1024**3)
        binding.check()
        specification = binding.input_spec
        if prefixes.source_id != specification.source_sha256:
            raise ValueError(SOURCE_ERROR)
        consumer.verify()
        self._binding, self._binding_versions = binding, _binding_signature(binding)
        self.output, self.native, self.codec = Path(output), native, codec
        self.consumer, self.prefixes, self.phase = consumer, prefixes, phase
        self.admission, self.block_rows, self.max_input_blocks = (
            admission,
            block_rows,
            max_input_blocks,
        )
        self.task, self.horizon, self._input_spec = "residual", 1, specification
        self._inflight, self._proposal = None, None
        code = {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (
                ("financial_session", inspect.getmodule(FinancialSession)),
                ("financial_consumers", financial_consumers),
                ("financial_state_artifacts", financial_state_artifacts),
                ("prefix_eligibility", prefix_eligibility),
            )
        }
        code.update(binding.code())
        self.model_id = _digest(dict(consumer=consumer.identity(), integration=code))
        contract = dict(
            schema_version=2,
            recipe="frozen_financial_session_v2",
            implementation={**episodic_session._CODE, **code},
            model_id=self.model_id,
            codec=codec.identity(),
            prefix_policy=prefixes.identity(),
            retention=asdict(retention),
            admission=admission,
            phase=asdict(phase),
            block_rows=block_rows,
            world=world,
            fold=fold,
            task=self.task,
            horizon=self.horizon,
            native_sha256=native.binary_sha256,
            consumer_kind=binding.kind,
            pending_widths=list(binding.widths),
            pending_dtype=str(binding.pending_dtype),
            feature_width=binding.feature_width,
            max_assets=8192,
            max_pending=32768,
            max_transient_pending=40960,
            max_input_blocks=max_input_blocks,
            max_observation_bytes=64 * 1024**2,
            max_artifact_files=1024,
            max_artifact_total_bytes=2 * 1024**3,
            max_log_bytes=max_log_bytes,
            input_compaction="canonical_live_rows_256_when_block_budget_requires",
            fast_compaction="canonical_live_rows_when_block_or_byte_budget_requires",
        )
        self.contract_id = _digest(contract)
        self._callback_versions = _callback_signature(self)
        self._runtime_contract = (
            self.contract_id,
            self.model_id,
            consumer.model_id,
            prefixes.policy_id,
            asdict(phase),
            block_rows,
            admission,
            max_input_blocks,
        )
        self._prototype = binding.bank(
            native,
            retention,
            codec_id=codec.fingerprint(),
            world=_digest([world, self.task, self.horizon]),
            partition=phase.partition,
            fold=fold,
        )
        self.artifacts = SessionArtifacts(
            native, self.output / "artifacts", max_files=1024, max_total_bytes=2 * 1024**3
        )
        self._fast_store = binding.state_store(self.artifacts, self.contract_id)
        definition = native.Definition()
        identity = native.Identity()
        identity.source_sha256, identity.view_sha256 = (
            specification.source_sha256,
            specification.view_sha256,
        )
        identity.representation_sha256, identity.model_sha256 = self.contract_id, self.model_id
        definition.identity, definition.tasks = identity, [native.Task(self.task, self.horizon)]
        definition.prediction_mode = native.PredictionMode.financial
        definition.phase = native.PhaseContract(
            phase.partition,
            phase.warmup_start,
            phase.decision_start,
            phase.decision_end,
            phase.close_at,
            prefixes.policy_id,
        )
        definition.initial_state_json = _canonical(self._state(0, None))
        limits = native.Limits()
        limits.feature_width, limits.max_assets = binding.feature_width, 8192
        limits.max_record_bytes = limits.max_checkpoint_bytes = 64 * 1024**2
        limits.max_log_bytes = max_log_bytes
        definition.limits = limits
        owner = weakref.proxy(self)
        self._executor = native.Executor(
            str(self.output),
            definition,
            lambda *args: _invoke(owner, "_prepare_event", args),
            lambda *args: _invoke(owner, "_resolve", args),
            resume,
        )
        try:
            self._verify_confirmed()
            self._prune()
        except Exception:
            self.close()
            raise

    def _bundle(self, state):
        self._check_state(state)
        if state["bundle"] is None:
            return None
        value = self._read(state["bundle"], "bundle")
        fields = {
            "generation",
            "cutoff",
            "bank",
            "pending",
            "fast",
            "inputs",
            "current_inputs",
            "prefix",
            "control",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or type(value["generation"]) is not int
            or value["generation"] != state["generation"]
            or type(value["cutoff"]) is not int
            or not 0 < value["cutoff"] <= self.phase.close_at
        ):
            raise ValueError("El bundle financiero no conserva sus campos, generación y corte")
        return value

    def _input_rows(self, reference, **_):
        return super()._input_rows(reference, allow_mixed_cutoffs=True)

    def _empty_queue(self):
        return _empty_pending(self._binding.widths, self._binding.pending_dtype)

    def _check_queue(self, pending, *, maximum=32768):
        _check_pending(
            pending,
            maximum=maximum,
            widths=self._binding.widths,
            dtype=self._binding.pending_dtype,
        )

    def _bank(self, reference):
        bank, episodes = super()._bank(reference)
        self._binding.check_bank(bank, episodes)
        return bank, episodes

    def diagnostics(self):
        if self.admission != "m2":
            return super().diagnostics()
        bundle = self._bundle(self.snapshot()["state"])
        bank, _ = self._bank(bundle["bank"] if bundle else None)
        indices = bank.index_ids()
        slots = sum(map(len, indices.values()))
        unique = len({identifier for values in indices.values() for identifier in values})
        return dict(
            cursor=self._executor.cursor,
            admitted=bank.seen,
            retained=unique,
            pending=len(self._executor.pending()),
            B_mem=bank.config.capacity,
            quotas=bank.config.quotas,
            physical_slots=slots,
            duplicate_slots=slots - unique,
        )

    def _fast(self, bundle):
        value = self._fast_store.empty() if bundle is None else self._read(bundle["fast"], "fast")
        self._fast_store.verify(value)
        return value

    def fast_state(self):
        return self._fast(self._bundle(self.snapshot()["state"]))

    def step(
        self,
        batches,
        feedback,
        *,
        kind="decision",
        cutoff=None,
        close_phase=False,
        batch_rows=None,
        fault=None,
    ):
        _check_callbacks(self)
        if (
            self._inflight is not None
            or kind not in {"warmup", "decision", "settlement"}
            or type(close_phase) is not bool
            or not isinstance(feedback, (list, tuple))
            or len(feedback) > 8192
            or (
                batch_rows is not None
                and (type(batch_rows) is not int or batch_rows != self.block_rows)
            )
        ):
            raise ValueError("El evento, lote físico o número de resoluciones no es válido")
        self.consumer.verify()
        if kind == "settlement":
            if batches != [] and batches != ():
                raise ValueError("Un settlement no admite observaciones")
            if type(cutoff) is not int:
                raise ValueError("El settlement necesita su reloj explícito")
            rows, proofs = (), ()
        else:
            rows = self._rows(batches)
            if cutoff is not None and cutoff != rows[0].prediction_at:
                raise ValueError("El corte declarado no corresponde a los inputs")
            cutoff = rows[0].prediction_at
            proofs = (
                tuple(self.prefixes.evidence(row.flow_id, row.prediction_at) for row in rows)
                if kind == "decision"
                else ()
            )
        cohort = self.native.Cohort()
        cohort.cursor, cohort.cutoff = self._executor.cursor, cutoff
        cohort.kind, cohort.close_phase = getattr(self.native.EventKind, kind), close_phase
        cohort.observations = [
            self.native.Observation(
                row.flow_id, row.input_available_at, self._binding.features(row)
            )
            for row in rows
        ]
        cohort.prefix_exclusions = [
            self.native.PrefixExclusion(
                proof.flow_id,
                self.native.Task(self.task, self.horizon),
                proof.decision_at,
                getattr(self.native.PrefixReason, proof.reason),
                proof.history_pairs,
                proof.market_variance,
                proof.fingerprint(),
            )
            for proof in proofs
            if proof.reason is not None
        ]
        self._inflight = dict(
            rows=rows,
            kind=kind,
            cutoff=cutoff,
            proofs=proofs,
            before_observed=self.snapshot()["observed"],
        )

        def guarded(boundary):
            if fault is not None:
                fault(boundary)
            if boundary == self.native.Boundary.before_commit:
                self._verify_bundle(self._proposal)
                self._check_observed(self._proposal, self._inflight["before_observed"] + len(rows))
                self.consumer.verify()
                for proof in proofs:
                    self.prefixes.verify(
                        proof, flow_id=proof.flow_id, decision_at=proof.decision_at
                    )

        try:
            result = self._executor.step(cohort, feedback, self.block_rows, guarded)
            self._verify_confirmed()
            self._prune()
            return result
        except Exception:
            self.close()
            raise
        finally:
            self._inflight = self._proposal = None

    def _decision_batch(self, rows):
        raw = dict(
            inputs={name: np.stack([r.inputs[name] for r in rows]) for name in MODALITIES},
            presence=np.stack([r.presence for r in rows]),
            sample_ids=[r.sample_id for r in rows],
            prediction_at=np.asarray([r.prediction_at for r in rows], dtype="datetime64[us]"),
            input_available_at=np.asarray(
                [r.input_available_at for r in rows], dtype="datetime64[us]"
            ),
        )
        if not self._binding.masked:
            raw.pop("presence")
        return DecisionBatch.from_validated(
            validated_cpu_batch(raw, self._input_spec),
            device=self._binding.device,
            dtype=self._binding.dtype,
        )

    def _prepare_event(self, kind, observations, tasks, cutoff, state_json, batch_rows):
        state = json.loads(state_json)
        rows = self._inflight["rows"]
        if (
            kind != getattr(self.native.EventKind, self._inflight["kind"])
            or cutoff != self._inflight["cutoff"]
            or batch_rows != self.block_rows
            or [r.flow_id for r in rows] != [r.asset for r in observations]
            or [(t.name, t.horizon) for t in tasks] != [(self.task, self.horizon)]
        ):
            raise ValueError("La preparación no corresponde al evento admitido")
        bundle = self._bundle(state)
        input_refs = self._stage_inputs(rows)
        rows = tuple(row for ref in input_refs for row in self._input_rows(ref))
        fast = self._fast(bundle)
        if self._fast_store.needs_compaction(fast, math.ceil(len(rows) / batch_rows)):
            fast = self._fast_store.compact(fast)
        bank, _ = self._bank(bundle["bank"] if bundle else None)
        head = json.loads(self.native.read_blob(str(self.output / "latest.json"), 65536))
        context_id = _digest(
            dict(
                checkpoint=head["checkpoint_sha256"],
                contract=self.contract_id,
                kind=self._inflight["kind"],
                group=[r.metadata() for r in rows],
            )
        )
        warmup = kind == self.native.EventKind.warmup
        snapshot = None if warmup else self._binding.snapshot(bank, context_id, cutoff)
        values, fast, control = self._binding.prepare_event(
            self,
            rows,
            fast,
            snapshot,
            context_id=context_id,
            warmup=warmup,
            batch_rows=batch_rows,
        )
        self.consumer.verify()
        pending = self._read(bundle["pending"], "pending") if bundle else self._empty_queue()
        self._check_queue(pending)
        if not warmup:
            pending = self._append_pending(pending, rows)
        proposal = dict(
            generation=state["generation"] + 1,
            cutoff=cutoff,
            bank=bundle["bank"] if bundle else None,
            fast=self._stage(fast, "fast"),
            pending=self._stage(pending, "pending"),
            inputs=[*(bundle["inputs"] if bundle else []), *(input_refs if not warmup else [])],
            current_inputs=input_refs,
            prefix=self._stage([asdict(p) for p in self._inflight["proofs"]], "prefix"),
            control=self._stage(control, "control"),
        )
        self._proposal = proposal
        return values, _canonical(proposal)

    def _append_pending(self, pending, rows):
        metadata = [*pending["rows"], *(r.metadata() for r in rows)]
        keys = torch.cat(
            (
                pending["key_inputs"],
                torch.tensor(np.stack([r.key_inputs for r in rows]), device="cpu"),
            )
        )
        values = torch.cat(
            (pending["values"], torch.tensor(np.stack([r.value for r in rows]), device="cpu"))
        )
        order = sorted(range(len(metadata)), key=lambda i: _row_key(metadata[i]))
        positions = torch.tensor(order, dtype=torch.int64, device="cpu")
        result = dict(
            rows=[metadata[i] for i in order],
            key_inputs=keys.index_select(0, positions),
            values=values.index_select(0, positions),
        )
        self._check_queue(result, maximum=40960)
        return result

    def _settlement_proposal(self, state):
        bundle = self._bundle(state)
        if bundle is None:
            fast = self._stage(self._fast_store.empty(), "fast")
            pending = self._stage(self._empty_queue(), "pending")
        else:
            fast, pending = bundle["fast"], bundle["pending"]
        return dict(
            generation=state["generation"] + 1,
            cutoff=self._inflight["cutoff"],
            bank=bundle["bank"] if bundle else None,
            fast=fast,
            pending=pending,
            inputs=bundle["inputs"] if bundle else [],
            current_inputs=[],
            prefix=self._stage([], "prefix"),
            control=self._stage(
                dict(
                    observations=0,
                    mac_updates=0,
                    refinements=0,
                    reevaluations=0,
                    group_estimated_bytes=0,
                ),
                "control",
            ),
        )

    def _resolve(self, proposed_json, outcomes, excluded, finalized):
        proposed = json.loads(proposed_json)
        if self._inflight["kind"] == "settlement":
            proposed = self._settlement_proposal(proposed)
        bank, episodes = self._bank(proposed["bank"])
        pending = self._read(proposed["pending"], "pending")
        self._check_queue(pending, maximum=40960)
        positions = {_row_key(row): i for i, row in enumerate(pending["rows"])}
        admitted, removed = [], set()
        for item in outcomes:
            prediction = item.prediction
            key = prediction.asset, prediction.decision_at
            if key not in positions or key in removed:
                raise ValueError("El resultado no enlaza una predicción pendiente")
            index = positions[key]
            if self.admission in {"m1", "m2"}:
                metadata = pending["rows"][index]
                identifier = bank.seen + len(admitted) + 1
                admitted.append((identifier, index, item))
                episodes[identifier] = dict(
                    metadata,
                    model_id=self.model_id,
                    task=self.task,
                    horizon=self.horizon,
                    prediction_id=prediction.id,
                    issued_prediction=prediction.value,
                    label=item.label.value,
                    error=item.label.value - prediction.value,
                    maturity_at=item.label.available_at,
                )
            removed.add(key)
        proofs = {(p.flow_id, p.decision_at): p for p in self._inflight["proofs"]}
        for item in excluded:
            key = item.prediction.asset, item.prediction.decision_at
            proof = proofs.get(key)
            if (
                proof is None
                or self.prefixes.verify(proof, flow_id=key[0], decision_at=key[1]) is None
                or proof.fingerprint() != item.evidence.evidence_sha256
            ):
                raise ValueError("La exclusión nativa no conserva su evidencia del prefijo")
            if key not in positions or key in removed:
                raise ValueError("La exclusión ya se resolvió o no tiene inputs")
            removed.add(key)
        for item in finalized:
            key = item.prediction.asset, item.prediction.decision_at
            if key not in positions or key in removed or item.closed_at != self.phase.close_at:
                raise ValueError("La finalización no corresponde al pendiente y cierre declarados")
            removed.add(key)
        if admitted:
            bank = self._binding.propose(
                self.native, bank, pending, admitted, episodes, confirmed_at=proposed["cutoff"]
            )
        selected = {r.id for r in bank.records()}
        bank_ref = self._stage(
            dict(snapshot=bank.snapshot(), episodes={i: episodes[i] for i in sorted(selected)}),
            "bank",
        )
        keep = [i for i, row in enumerate(pending["rows"]) if _row_key(row) not in removed]
        indexes = torch.tensor(keep, dtype=torch.int64, device="cpu")
        pending = dict(
            rows=[pending["rows"][i] for i in keep],
            key_inputs=pending["key_inputs"].index_select(0, indexes),
            values=pending["values"].index_select(0, indexes),
        )
        self._check_queue(pending)
        sources = self._check_sources(proposed["inputs"], pending)
        if len(sources) > self.max_input_blocks:
            sources = self._compact_inputs(sources, pending)
        bundle = dict(
            proposed, bank=bank_ref, pending=self._stage(pending, "pending"), inputs=sources
        )
        self._verify_bundle(bundle)
        self.consumer.verify()
        self._proposal = bundle
        return _canonical(self._state(bundle["generation"], self._stage(bundle, "bundle")))

    def _compact_inputs(self, references, pending):
        total = sum(r["bytes"] for r in references)
        files = self.artifacts._inventory()
        blocks = math.ceil(len(pending["rows"]) / 256)
        if (
            total > 512 * 1024**2
            or len(files) + blocks > self.artifacts.max_files
            or sum(p.stat().st_size for p in files) + total + blocks * 65536
            > self.artifacts.max_total_bytes
        ):
            raise ValueError("La compactación de inputs no cabe junto a las generaciones vivas")
        needed = {_row_key(row) for row in pending["rows"]}
        rows = [
            row
            for ref in references
            for row in self._input_rows(ref)
            if (row.flow_id, row.prediction_at) in needed
        ]
        rows.sort(key=lambda r: (r.flow_id, r.prediction_at))
        result = self._stage_inputs(rows)
        if self._check_sources(result, pending) != result:
            raise ValueError("La compactación no conserva los inputs pendientes")
        return result

    def _verify_bundle(self, bundle):
        self._bank(bundle["bank"])
        fast = self._read(bundle["fast"], "fast")
        self._fast_store.verify(fast)
        if any(
            row["observed_steps"] < 1 or row["last_prediction_at"] > bundle["cutoff"]
            for row in fast["references"].values()
        ):
            raise ValueError("El estado rápido contiene flujos sin observar o información futura")
        pending = self._read(bundle["pending"], "pending")
        self._check_queue(pending)
        if self._check_sources(bundle["inputs"], pending) != bundle["inputs"]:
            raise ValueError("La generación conserva inputs pendientes huérfanos")
        if not isinstance(bundle["current_inputs"], list) or len(bundle["current_inputs"]) > 32:
            raise ValueError("Las observaciones de la generación superan su presupuesto")
        observed, input_bytes = [], 0
        for reference in bundle["current_inputs"]:
            rows = self._input_rows(reference)
            if any(row.prediction_at != bundle["cutoff"] for row in rows):
                raise ValueError("La observación no pertenece al corte confirmado")
            for row in rows:
                input_bytes += (
                    sum(value.nbytes for value in row.inputs.values()) + row.presence.nbytes
                )
                cursor = fast["references"].get(row.flow_id)
                if (
                    input_bytes > 64 * 1024**2
                    or cursor is None
                    or cursor["last_prediction_at"] != row.prediction_at
                    or cursor["last_sample_id"] != row.sample_id
                ):
                    raise ValueError(
                        "Las observaciones no corresponden al cursor rápido o a su presupuesto"
                    )
                observed.append(row.flow_id)
        if observed != sorted(set(observed)):
            raise ValueError("Las observaciones confirmadas repiten o desordenan un flujo")
        proofs = self._read(bundle["prefix"], "prefix")
        if not isinstance(proofs, list) or len(proofs) > 8192:
            raise ValueError("Las evidencias del prefijo superan el grupo lógico")
        seen = set()
        for raw in proofs:
            if not isinstance(raw, dict) or set(raw) != set(PrefixEvidence.__dataclass_fields__):
                raise ValueError("La evidencia serializada no conserva su esquema")
            proof = PrefixEvidence(**raw)
            if proof.decision_at != bundle["cutoff"] or proof.flow_id in seen:
                raise ValueError("La evidencia repite un flujo o corresponde a otro corte")
            self.prefixes.verify(proof, flow_id=proof.flow_id, decision_at=proof.decision_at)
            seen.add(proof.flow_id)
        control = self._read(bundle["control"], "control")
        if (
            not isinstance(control, dict)
            or type(control.get("observations")) is not int
            or control["observations"] != len(observed)
            or (seen and seen != set(observed))
        ):
            raise ValueError("Las evidencias y el control no concilian con las observaciones")

    def _check_observed(self, bundle, observed):
        fast = self._read(bundle["fast"], "fast")
        if sum(row["observed_steps"] for row in fast["references"].values()) != observed:
            raise ValueError(
                "Los flujos rápidos no concilian con todas las observaciones confirmadas"
            )

    def _verify_confirmed(self):
        snapshot = self.snapshot()
        self._check_state(snapshot["state"], self._executor.cursor)
        bundle = self._bundle(snapshot["state"])
        if bundle is None:
            return
        self._verify_bundle(bundle)
        self._check_observed(bundle, snapshot["observed"])
        record = json.loads(self._executor.record_json(snapshot["cursor"]))
        control = self._read(bundle["control"], "control")
        proofs = [PrefixEvidence(**raw) for raw in self._read(bundle["prefix"], "prefix")]
        if record["observations_count"] != control["observations"] or len(proofs) != (
            control["observations"] if record["event_kind"] == "decision" else 0
        ):
            raise ValueError("El registro nativo no corresponde al grupo de inputs y evidencias")
        exclusions = {proof.flow_id: proof for proof in proofs if proof.reason is not None}
        if len(exclusions) != len(record["excluded"]):
            raise ValueError("El registro nativo no conserva las exclusiones acreditadas")
        for row in record["excluded"]:
            proof = exclusions.get(row["prediction"]["asset"])
            if proof is None or proof.fingerprint() != row["evidence"]["evidence_sha256"]:
                raise ValueError("La resolución nativa no conserva la evidencia verificada")
        bank, _ = self._bank(bundle["bank"])
        if bank.seen != (snapshot["applied"] if self.admission in {"m1", "m2"} else 0):
            raise ValueError("El banco no concilia con los labels maduros y la regla de admisión")
        pending = self._read(bundle["pending"], "pending")
        expected = {(p.asset, p.decision_at) for p in self._executor.pending()}
        if expected != {_row_key(row) for row in pending["rows"]}:
            raise ValueError("La cola no coincide con las predicciones confirmadas")

    def _prune(self):
        head = json.loads(self.native.read_blob(str(self.output / "latest.json"), 65536))
        references = {}
        for generation, digest in (
            (head["generation"], head["checkpoint_sha256"]),
            (head["generation"] - 1, head["previous_sha256"]),
        ):
            if generation < 0:
                continue
            payload = self.native.read_blob(
                str(self.output / f"checkpoint-{generation}.json"), 64 * 1024**2
            )
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("La poda no confirma la huella del checkpoint")
            state = json.loads(payload)["state"]
            self._check_state(state, generation)
            bundle = self._bundle(state)
            if bundle is None:
                continue
            self._verify_bundle(bundle)
            live = [
                state["bundle"],
                *(bundle[name] for name in ("bank", "pending", "fast", "prefix", "control")),
                *bundle["inputs"],
                *bundle["current_inputs"],
                *self._fast_store.live_references(self._read(bundle["fast"], "fast")),
            ]
            for reference in live:
                previous = references.get(reference["name"])
                if previous is not None and previous != reference:
                    raise ValueError("Las generaciones mezclan referencias al mismo archivo")
                references[reference["name"]] = reference
        self.artifacts.prune_unreferenced(list(references.values()))
