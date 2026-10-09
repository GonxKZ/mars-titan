"""Enlace cerrado del coordinador financiero con sus dos consumidores conocidos.

`FinancialSession` controla una única cronología: fases, prefijos, pendientes y
publicación. Cada enlace aporta solo lo que difiere entre Titans-MAC y la GRU:
geometría de la cola, banco, estado por flujo, instantánea y preparación.
"""

import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from mars_titan.data.input_policy import masked_inputs
from mars_titan.models.candidate.episode_codec import FrozenCandidateCodec
from mars_titan.models.candidate.frozen_consumer import FrozenCandidateConsumer
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

from . import candidate_bank, flow_cursors, write_policy
from .candidate_bank import CandidateBankConfig, CandidateEpisodeBank, CandidateEpisodes
from .episodic_codec import FrozenEpisodeCodec
from .financial_state_artifacts import FinancialStateArtifacts
from .flow_cursors import FlowCursors
from .retention_bank import RetentionBank, RetentionConfig
from .write_policy import MatureErrorBank, MatureErrorConfig

ADMISSION_ERROR = "La admisión requiere M0/M1 con retención original o M2 con sus tres índices"
SOURCE_ERROR = "El predictor, codec y prefijo no comparten la fuente de inputs"


def _file_digest(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _words(digest):
    return [int(digest[i : i + 8], 16) for i in range(0, 64, 8)]


def bank_retention(admission, *, capacity, seed, policy="reservoir", **options):
    """Retención de la admisión: M2 con sus tres índices y M0/M1 con la política declarada."""
    if admission == "m2":
        if policy != "reservoir" or options:
            raise ValueError("M2 conserva sus tres índices y no admite otra retención")
        return MatureErrorConfig(capacity=capacity, seed=seed)
    if admission not in ("m0", "m1"):
        raise ValueError("M3 no tiene definición acreditada. Se admiten M0, M1 y M2")
    return RetentionConfig(policy=policy, capacity=capacity, seed=seed, **options)


def episodic_bank(native, retention, admission, **scope):
    """Banco nativo 64×64 de la admisión declarada, con el contrato causal v2 en M0/M1."""
    if admission == "m2":
        return MatureErrorBank(native, retention, **scope)
    return RetentionBank(native, retention, memory_contract="causal_v2", **scope)


def bank_snapshot(bank, *, codec_id, context_id, cutoff, dtype, device):
    """Instantánea por ID de los episodios retenidos, común a todos los bloques del evento."""
    records = sorted(bank.records(), key=lambda r: r.id)
    size = len(records)
    return EpisodeSnapshot.create(
        keys=torch.tensor([r.key for r in records], dtype=torch.float32, device="cpu").reshape(
            size, 64
        ),
        values=torch.tensor([r.value for r in records], dtype=torch.float32, device="cpu").reshape(
            size, 64
        ),
        labels=torch.tensor([r.label for r in records], dtype=torch.float64, device="cpu"),
        ids=torch.tensor([r.id for r in records], dtype=torch.int64, device="cpu"),
        decision_at=torch.tensor([r.decision_at for r in records], dtype=torch.int64, device="cpu"),
        available_at=torch.tensor(
            [r.available_at for r in records], dtype=torch.int64, device="cpu"
        ),
        maturity_at=torch.tensor([r.maturity_at for r in records], dtype=torch.int64, device="cpu"),
        cutoff=cutoff,
        codec_id=codec_id,
        context_id=context_id,
        dtype=dtype,
        device=device,
    )


def mature_record(native, *, identifier, decision_at, available_at, maturity_at, key, value, label):
    """Episodio con etiqueta madura, la clave sin normalizar del codec y su valor fijo."""
    record = native.MemoryRecord()
    record.id, record.decision_at = identifier, decision_at
    record.available_at, record.maturity_at = available_at, maturity_at
    record.key, record.value = key, value
    record.label, record.label_valid = label, True
    return record


class TitansBinding:
    """Predictor Titans-MAC, codec 64×64 y bancos nativos M0/M1/M2."""

    kind = "titans_mac"
    widths = (64, 64)
    pending_dtype = torch.float32
    feature_width = 136

    def __init__(self, consumer, codec, retention, admission):
        if type(codec) is not FrozenEpisodeCodec or not (
            (admission in {"m0", "m1"} and type(retention) is RetentionConfig)
            or (admission == "m2" and type(retention) is MatureErrorConfig)
        ):
            raise ValueError(ADMISSION_ERROR)
        self.consumer, self.codec, self.admission = consumer, codec, admission
        predictor = consumer.predictor
        self.input_spec, self.max_batch = predictor.config.inputs, predictor.config.max_batch
        self.device, self.dtype = predictor.head.weight.device, predictor.head.weight.dtype
        self.masked = predictor.masked

    def check(self):
        consumer = self.consumer
        if (
            self.admission == "m0"
            and consumer.readout is not None
            and consumer.readout.config.mode != "no_bank"
        ):
            raise ValueError("M0 exige readout=None o el control explícito no_bank")
        if self.codec.identity()["input_specification"] != self.input_spec.identity() or (
            consumer.readout is not None
            and consumer.readout.config.codec_id != self.codec.fingerprint()
        ):
            raise ValueError(SOURCE_ERROR)

    def code(self):
        return {"write_policy": _file_digest(write_policy)} if self.admission == "m2" else {}

    def bank(self, native, retention, **scope):
        return episodic_bank(native, retention, self.admission, **scope)

    def state_store(self, artifacts, identity):
        return FinancialStateArtifacts(self.consumer, artifacts, identity=identity)

    @staticmethod
    def features(row):
        return [*row.key_inputs.tolist(), *row.value.tolist(), *_words(row.input_sha256)]

    def check_bank(self, bank, episodes):
        if self.admission == "m2":
            scores = {key: abs(row["error"]) for key, row in episodes.items()}
            selected = sorted(scores, key=lambda key: (-scores[key], key))[
                : bank.config.quotas["selective"]
            ]
            if bank.selective_scores != {key: scores[key] for key in selected}:
                raise ValueError("El índice selectivo M2 contradice los errores de las emisiones")

    def snapshot(self, bank, context_id, cutoff):
        extension = self.consumer.readout
        if extension is None or extension.config.mode == "no_bank":
            return None
        return bank_snapshot(
            bank,
            codec_id=self.codec.fingerprint(),
            context_id=context_id,
            cutoff=cutoff,
            dtype=self.dtype,
            device=self.device,
        )

    def prepare_event(self, session, rows, fast, snapshot, *, context_id, warmup, batch_rows):
        model = self.consumer.predictor
        selection = None
        if model.local_control is not None and model.local_control.config.mode != "disabled":
            steps = torch.tensor(
                [fast["references"].get(r.flow_id, {}).get("observed_steps", 0) for r in rows],
                dtype=torch.int64,
                device="cpu",
            )
            selection = model.local_control.select_flows(
                tuple(r.flow_id for r in rows), steps, context_id=context_id
            )
        values, measurements = [], []
        for start in range(0, len(rows), batch_rows):
            block = rows[start : start + batch_rows]
            batch = session._decision_batch(block)
            unknown = tuple(flow for flow in batch.flow_ids if flow not in fast["references"])
            initial = model.initial_state(unknown) if unknown else None
            previous = session._fast_store.gather(fast, batch.flow_ids, initial_state=initial)
            result = self.consumer.prepare(
                batch,
                previous,
                context_id=context_id,
                snapshot=snapshot,
                warmup=warmup,
                selection=selection,
            )
            fast = session._fast_store.replace(fast, previous, result.next_state)
            if not warmup:
                values.extend(result.point_predictions.detach().cpu().tolist())
            if result.local_control is not None:
                measured = result.local_control
                measurements.append(
                    dict(
                        flow_ids=measured.flow_ids,
                        reevaluations=measured.reevaluations,
                        angular_corrected_estimate=measured.estimates.angular_corrected_estimate.detach().cpu(),
                        penalty=measured.penalty.detach().cpu()
                        if measured.penalty is not None
                        else None,
                    )
                )
        control = dict(
            context_id=context_id,
            selection=asdict(selection) if selection else None,
            measurements=measurements,
            observations=len(rows),
            mac_updates=len(rows) if model.config.variant == "mac_online" else 0,
            refinements=0
            if warmup or self.consumer.readout is None
            else self.consumer.readout.config.refinements,
            snapshot_bytes=sum(t.numel() * t.element_size() for t in snapshot._values)
            if snapshot
            else 0,
            reevaluations=sum(m["reevaluations"] for m in measurements),
            group_estimated_bytes=selection.estimated_bytes if selection else 0,
        )
        return values, fast, control

    def propose(self, native, bank, pending, admitted, episodes, *, confirmed_at):
        incoming = [
            mature_record(
                native,
                identifier=identifier,
                decision_at=item.prediction.decision_at,
                available_at=pending["rows"][index]["input_available_at"],
                maturity_at=item.label.available_at,
                key=pending["key_inputs"][index].tolist(),
                value=pending["values"][index].tolist(),
                label=item.label.value,
            )
            for identifier, index, item in admitted
        ]
        options = dict(confirmed_at=confirmed_at)
        if self.admission == "m2":
            options["errors"] = {record.id: episodes[record.id]["error"] for record in incoming}
        return bank.propose(incoming, **options)


class CandidateBinding:
    """GRU nativa con codec CPU 128×256, banco tensorial M0/M1 y cursores sin estado neural."""

    kind = "candidate_gru"
    widths = (128, 256)
    feature_width = 16

    def __init__(self, consumer, codec, retention, admission):
        if (
            type(codec) is not FrozenCandidateCodec
            or type(retention) is not CandidateBankConfig
            or admission not in {"m0", "m1"}
        ):
            raise ValueError(ADMISSION_ERROR)
        self.consumer, self.codec, self.admission = consumer, codec, admission
        self.input_spec, self.max_batch = consumer.input_spec, consumer.max_batch
        self.device, self.dtype = consumer.device, consumer.dtype
        self.pending_dtype = consumer.dtype
        self.masked = masked_inputs(self.input_spec.input_policy)

    def check(self):
        identity = self.codec.identity()
        if (
            identity["input_specification"] != self.input_spec.identity()
            or identity["projection_id"] != self.consumer.model.representation_id()
            or identity["dtype"] != str(self.dtype).removeprefix("torch.")
        ):
            raise ValueError(SOURCE_ERROR)

    @staticmethod
    def code():
        return {
            "candidate_bank": _file_digest(candidate_bank),
            "flow_cursors": _file_digest(flow_cursors),
        }

    def bank(self, native, retention, **scope):
        return CandidateEpisodeBank(
            native,
            retention,
            representation_id=self.consumer.model.representation_id(),
            dtype=self.dtype,
            **scope,
        )

    def state_store(self, artifacts, identity):
        return FlowCursors(self.consumer)

    @staticmethod
    def features(row):
        encoding = hashlib.sha256(row.key_inputs.tobytes() + row.value.tobytes()).hexdigest()
        return [*_words(row.input_sha256), *_words(encoding)]

    @staticmethod
    def check_bank(bank, episodes):
        decisions = {(row["flow_id"], row["prediction_at"]) for row in episodes.values()}
        if len(decisions) != len(episodes):
            raise ValueError("El banco GRU repite la decisión de un mismo flujo")

    def snapshot(self, bank, context_id, cutoff):
        return self.consumer.memory(bank.read_view() if self.admission == "m1" else None)

    def prepare_event(self, session, rows, fast, snapshot, *, context_id, warmup, batch_rows):
        values = []
        for start in range(0, len(rows), batch_rows):
            batch = session._decision_batch(rows[start : start + batch_rows])
            fast = session._fast_store.advance(fast, batch)
            if not warmup:
                result = self.consumer.prepare(batch, snapshot)
                values.extend(result.point_predictions.detach().cpu().tolist())
        itemsize = self.dtype.itemsize
        control = dict(
            context_id=context_id,
            selection=None,
            measurements=[],
            observations=len(rows),
            mac_updates=0,
            refinements=0 if warmup else self.consumer.refinements,
            snapshot_bytes=0
            if snapshot is None
            else snapshot.size * ((sum(self.widths) + 1) * itemsize + 8),
            reevaluations=0,
            group_estimated_bytes=0,
        )
        return values, fast, control

    @staticmethod
    def propose(native, bank, pending, admitted, episodes, *, confirmed_at):
        positions = torch.tensor([index for _, index, _ in admitted], dtype=torch.int64)
        incoming = CandidateEpisodes(
            ids=torch.tensor([identifier for identifier, _, _ in admitted], dtype=torch.int64),
            keys=pending["key_inputs"].index_select(0, positions),
            values=pending["values"].index_select(0, positions),
            times=torch.tensor(
                [
                    [
                        item.prediction.decision_at,
                        pending["rows"][index]["input_available_at"],
                        item.label.available_at,
                    ]
                    for _, index, item in admitted
                ],
                dtype=torch.int64,
            ),
            labels=torch.tensor([item.label.value for _, _, item in admitted], dtype=torch.float64),
        )
        return bank.propose(incoming, confirmed_at=confirmed_at)


def bind_consumer(consumer, codec, retention, admission):
    if type(consumer) is FrozenFinancialConsumer:
        return TitansBinding(consumer, codec, retention, admission)
    if type(consumer) is FrozenCandidateConsumer:
        return CandidateBinding(consumer, codec, retention, admission)
    raise ValueError(ADMISSION_ERROR)
