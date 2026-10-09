"""Ciclo de una tarea con banco nativo, rasgos pendientes y una publicación de sesión."""

import hashlib
import json
import math
import re
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from mars_titan.data.input_policy import MODALITIES, masked_inputs
from mars_titan.memory import episodic_codec, native_backend, retention_bank, session_artifacts
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.retention_bank import RetentionBank, RetentionConfig
from mars_titan.memory.session_artifacts import SessionArtifacts
from mars_titan.models.titans.financial_inputs import (
    CPUDecisionBatch,
    FinancialInputSpec,
    validated_cpu_batch,
)

_CODE = {
    "episodic_session": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    **{
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in (
            ("episodic_codec", episodic_codec),
            ("native_backend", native_backend),
            ("retention_bank", retention_bank),
            ("session_artifacts", session_artifacts),
        )
    },
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class DecisionRow:
    flow_id: str
    sample_id: str
    prediction_at: int
    input_available_at: int
    input_sha256: str
    inputs: object
    presence: np.ndarray
    key_inputs: np.ndarray
    value: np.ndarray

    def metadata(self):
        return {
            name: getattr(self, name)
            for name in (
                "flow_id",
                "sample_id",
                "prediction_at",
                "input_available_at",
                "input_sha256",
            )
        }


@dataclass(frozen=True)
class SessionPreparation:
    predictions: dict[str, float]
    fast_state: dict


class _BankView:
    def __init__(self, bank, episodes, cutoff):
        self._bank, self._episodes, self._cutoff = bank, episodes, cutoff

    def query(self, row):
        if not isinstance(row, DecisionRow) or row.prediction_at != self._cutoff:
            raise ValueError("La consulta no corresponde a la sesión preparada")
        result = []
        for neighbor in self._bank.query(row.key_inputs, self._cutoff):
            record = neighbor.record
            value = np.asarray(record.value, dtype=np.float32)
            value = np.frombuffer(value.tobytes(), dtype=np.float32)
            result.append(
                dict(self._episodes[record.id], value=value, similarity=neighbor.similarity)
            )
        return tuple(result)


def _empty_pending(widths=(64, 64), dtype=torch.float32):
    return dict(
        rows=[],
        key_inputs=torch.empty((0, widths[0]), dtype=dtype, device="cpu"),
        values=torch.empty((0, widths[1]), dtype=dtype, device="cpu"),
    )


def _row_key(row):
    return row["flow_id"], row["prediction_at"]


def _check_metadata(row, *, episode=False):
    fields = {"flow_id", "sample_id", "prediction_at", "input_available_at", "input_sha256"}
    if episode:
        fields |= {
            "model_id",
            "task",
            "horizon",
            "prediction_id",
            "issued_prediction",
            "label",
            "error",
            "maturity_at",
        }
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError("La procedencia no conserva sus campos")
    if (
        any(
            not isinstance(row[name], str) or not row[name]
            for name in ("flow_id", "sample_id", "input_sha256")
        )
        or row["sample_id"] != f"{row['flow_id']}/{row['prediction_at']}"
        or not re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"])
        or type(row["prediction_at"]) is not int
        or type(row["input_available_at"]) is not int
        or not 0 <= row["input_available_at"] <= row["prediction_at"]
    ):
        raise ValueError("La procedencia de los rasgos no es válida")
    if episode and (
        any(type(row[name]) is not int for name in ("horizon", "maturity_at"))
        or row["horizon"] <= 0
        or row["maturity_at"] <= row["prediction_at"]
        or any(
            type(row[name]) is not float or not math.isfinite(row[name])
            for name in ("issued_prediction", "label", "error")
        )
        or any(
            not isinstance(row[name], str) or not re.fullmatch(r"[0-9a-f]{64}", row[name])
            for name in ("model_id", "prediction_id")
        )
        or not isinstance(row["task"], str)
    ):
        raise ValueError("El episodio no conserva los tipos de su predicción y etiqueta")


def _check_pending(pending, *, maximum=32768, widths=(64, 64), dtype=torch.float32):
    if not isinstance(pending, dict) or set(pending) != {"rows", "key_inputs", "values"}:
        raise ValueError("La cola de rasgos no conserva su formato")
    rows = pending["rows"]
    if not isinstance(rows, list) or len(rows) > maximum:
        raise ValueError("La cola de rasgos supera el presupuesto")
    for value, width in zip((pending["key_inputs"], pending["values"]), widths, strict=True):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != dtype
            or value.shape != (len(rows), width)
        ):
            raise ValueError(
                "La cola necesita claves y valores con la geometría y precisión del banco"
            )
    for row in rows:
        _check_metadata(row)
    keys = [_row_key(row) for row in rows]
    if keys != sorted(set(keys)):
        raise ValueError("Los rasgos pendientes repiten o desordenan una decisión")


class EpisodicSession:
    """Preparar inputs y admitir feedback mediante Executor, sin ejecutar otra cronología.

    El callback recibe filas, estado rápido anterior, vista del banco, identidad
    del grupo y tamaño físico. Esta API no implementa una cabeza ni el refinamiento
    posterior a MAC. El adaptador del predictor debe aportar ese comportamiento.
    """

    def __init__(
        self,
        output,
        *,
        native,
        codec,
        retention,
        model_id,
        task,
        horizon,
        world,
        partition,
        fold,
        prepare,
        resume=False,
    ):
        if (
            not isinstance(codec, FrozenEpisodeCodec)
            or not isinstance(retention, RetentionConfig)
            or not callable(prepare)
        ):
            raise ValueError("La sesión necesita codec, retención y preparación explícitos")
        if type(horizon) is not int or type(resume) is not bool:
            raise ValueError("El horizonte y la recuperación tienen tipos incompatibles")
        self.output, self.native, self.codec = Path(output), native, codec
        self.task, self.horizon, self.model_id = task, horizon, model_id
        self._prepare_callback, self._inflight = prepare, None
        contract = dict(
            schema_version=1,
            recipe="single_task_episodic_session_v1",
            implementation=_CODE,
            codec=codec.identity(),
            retention=asdict(retention),
            model_id=model_id,
            task=task,
            horizon=horizon,
            world=world,
            partition=partition,
            fold=fold,
            native_sha256=native.binary_sha256,
            input_digest="canonical_row_bytes_v1",
            input_artifacts="complete_modalities_canonical_blocks_256_v1",
            max_assets=8192,
            max_pending=32768,
            max_mature=8192,
            max_observation_bytes=64 * 1024**2,
            max_artifact_files=512,
            max_artifact_total_bytes=2 * 1024**3,
        )
        self.contract_id = _digest(contract)
        self._prototype = RetentionBank(
            native,
            retention,
            codec_id=codec.fingerprint(),
            world=_digest([world, task, horizon]),
            partition=partition,
            fold=fold,
        )
        self.artifacts = SessionArtifacts(
            native, self.output / "artifacts", max_files=512, max_total_bytes=2 * 1024**3
        )
        specification = codec.identity()["input_specification"]
        self._input_spec = FinancialInputSpec(
            **{
                key: specification[key]
                for key in (
                    "source_sha256",
                    "view_sha256",
                    "representation",
                    "dimensions",
                    "input_policy",
                )
            }
        )
        identity = native.Identity()
        identity.source_sha256, identity.view_sha256 = (
            specification["source_sha256"],
            specification["view_sha256"],
        )
        identity.representation_sha256, identity.model_sha256 = self.contract_id, model_id
        definition = native.Definition()
        definition.identity, definition.tasks = identity, [native.Task(task, horizon)]
        definition.prediction_mode = native.PredictionMode.prepared
        definition.initial_state_json = _canonical(self._state(0, None))
        limits = native.Limits()
        limits.feature_width = 136
        limits.max_assets = 8192
        limits.max_record_bytes = 64 * 1024**2
        definition.limits = limits
        owner = weakref.proxy(self)
        self._executor = native.Executor(
            str(self.output),
            definition,
            lambda *args: owner._prepare(*args),
            lambda *args: owner._update(*args),
            resume,
        )
        try:
            self._verify_confirmed()
            self._prune()
        except Exception:
            self.close()
            raise

    def _state(self, generation, bundle):
        return dict(
            schema_version=1, contract_id=self.contract_id, generation=generation, bundle=bundle
        )

    def _check_state(self, state, generation=None):
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "contract_id", "generation", "bundle"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["contract_id"] != self.contract_id
            or type(state["generation"]) is not int
            or state["generation"] < 0
            or (generation is not None and state["generation"] != generation)
            or (state["bundle"] is None) != (state["generation"] == 0)
        ):
            raise ValueError("El estado de sesión pertenece a otro contrato o generación")

    def _read(self, reference, kind):
        return self.artifacts.read(reference, identity=self.contract_id, kind=kind)

    def _stage(self, payload, kind):
        return self.artifacts.stage(payload, identity=self.contract_id, kind=kind)

    def _bank(self, reference):
        if reference is None:
            return self._prototype, {}
        value = self._read(reference, "bank")
        if not isinstance(value, dict) or set(value) != {"snapshot", "episodes"}:
            raise ValueError("El banco no conserva sus episodios y snapshot")
        bank = self._prototype.restore(value["snapshot"])
        episodes = value["episodes"]
        records = bank.records()
        if not isinstance(episodes, dict) or set(episodes) != {r.id for r in records}:
            raise ValueError("Los registros no concuerdan con su procedencia")
        for record in records:
            episode = episodes[record.id]
            _check_metadata(episode, episode=True)
            if (
                episode["model_id"] != self.model_id
                or episode["task"] != self.task
                or episode["horizon"] != self.horizon
                or episode["label"] != record.label
                or episode["prediction_at"] != record.decision_at
                or episode["input_available_at"] != record.available_at
                or episode["maturity_at"] != record.maturity_at
                or not record.label_valid
                or record.reward_valid
                or episode["error"] != record.label - episode["issued_prediction"]
            ):
                raise ValueError("Un episodio no corresponde a su predicción emitida")
        return bank, episodes

    def _bundle(self, state):
        self._check_state(state)
        if state["bundle"] is None:
            return None
        bundle = self._read(state["bundle"], "bundle")
        if (
            not isinstance(bundle, dict)
            or set(bundle) != {"generation", "cutoff", "bank", "pending", "fast", "inputs"}
            or type(bundle["generation"]) is not int
            or bundle["generation"] != state["generation"]
            or type(bundle["cutoff"]) is not int
            or bundle["cutoff"] <= 0
        ):
            raise ValueError("El bundle no conserva su generación o formato")
        return bundle

    def _rows(self, batches, *, allow_mixed_cutoffs=False):
        if not isinstance(batches, (list, tuple)) or not 1 <= len(batches) <= 8192:
            raise ValueError("Falta el grupo lógico de entradas")
        rows, total_bytes = [], 0
        for batch in batches:
            if not isinstance(batch, CPUDecisionBatch):
                raise ValueError("La sesión necesita vistas CPU verificadas")
            total_bytes += (
                sum(value.nbytes for value in batch.inputs.values()) + batch.presence.nbytes
            )
            if total_bytes > 64 * 1024**2:
                raise ValueError("Las modalidades de la cohorte superan 64 MiB")
            encoded = self.codec.encode(batch)
            for index, flow in enumerate(batch.flow_ids):
                metadata = dict(
                    flow_id=flow,
                    sample_id=batch.sample_ids[index],
                    prediction_at=batch.prediction_at[index],
                    input_available_at=batch.input_available_at[index],
                    input_contract_id=batch.input_contract_id,
                )
                digest = hashlib.sha256(_canonical(metadata).encode())
                for name in MODALITIES:
                    digest.update(batch.inputs[name][index].tobytes())
                digest.update(batch.presence[index].tobytes())
                rows.append(
                    DecisionRow(
                        flow,
                        metadata["sample_id"],
                        metadata["prediction_at"],
                        metadata["input_available_at"],
                        digest.hexdigest(),
                        MappingProxyType({name: batch.inputs[name][index] for name in MODALITIES}),
                        batch.presence[index],
                        encoded.key_inputs[index],
                        encoded.values[index],
                    )
                )
            if len(rows) > 8192:
                raise ValueError("La cohorte supera 8192 flujos")
        rows.sort(key=lambda row: (row.flow_id, row.prediction_at))
        if len({(row.flow_id, row.prediction_at) for row in rows}) != len(rows) or (
            not allow_mixed_cutoffs and len({row.prediction_at for row in rows}) != 1
        ):
            raise ValueError("Los flujos se repiten o mezclan cortes")
        return tuple(rows)

    def _stage_inputs(self, rows):
        references = []
        for start in range(0, len(rows), 256):
            block = rows[start : start + 256]
            payload = dict(
                rows=[row.metadata() for row in block],
                inputs={
                    name: torch.tensor(np.stack([row.inputs[name] for row in block]), device="cpu")
                    for name in MODALITIES
                },
                presence=torch.tensor(np.stack([row.presence for row in block]), device="cpu"),
            )
            references.append(self._stage(payload, "inputs"))
        return references

    def _input_rows(self, reference, *, allow_mixed_cutoffs=False):
        value = self._read(reference, "inputs")
        if (
            not isinstance(value, dict)
            or set(value) != {"rows", "inputs", "presence"}
            or not isinstance(value["rows"], list)
            or not 1 <= len(value["rows"]) <= 256
            or not isinstance(value["inputs"], dict)
            or set(value["inputs"]) != set(MODALITIES)
            or any(not isinstance(array, torch.Tensor) for array in value["inputs"].values())
            or not isinstance(value["presence"], torch.Tensor)
        ):
            raise ValueError("El artefacto de inputs no conserva sus modalidades completas")
        metadata = value["rows"]
        for row in metadata:
            _check_metadata(row)
        raw = dict(
            inputs={name: tensor.numpy() for name, tensor in value["inputs"].items()},
            presence=value["presence"].numpy(),
            sample_ids=[row["sample_id"] for row in metadata],
            prediction_at=np.asarray(
                [row["prediction_at"] for row in metadata], dtype="datetime64[us]"
            ),
            input_available_at=np.asarray(
                [row["input_available_at"] for row in metadata], dtype="datetime64[us]"
            ),
        )
        if not masked_inputs(self._input_spec.input_policy):
            presence = raw.pop("presence")
            if (
                presence.dtype != np.bool_
                or presence.shape != (len(metadata), len(MODALITIES))
                or not presence.all()
            ):
                raise ValueError("El artefacto estricto necesita todas sus modalidades presentes")
        blocks = [raw]
        if allow_mixed_cutoffs:
            blocks = []
            for moment in np.unique(raw["prediction_at"]):
                positions = np.flatnonzero(raw["prediction_at"] == moment)
                blocks.append(
                    dict(
                        inputs={name: values[positions] for name, values in raw["inputs"].items()},
                        **({"presence": raw["presence"][positions]} if "presence" in raw else {}),
                        sample_ids=[raw["sample_ids"][i] for i in positions],
                        prediction_at=raw["prediction_at"][positions],
                        input_available_at=raw["input_available_at"][positions],
                    )
                )
        rows = self._rows(
            [validated_cpu_batch(block, self._input_spec) for block in blocks],
            allow_mixed_cutoffs=allow_mixed_cutoffs,
        )
        if [row.metadata() for row in rows] != metadata:
            raise ValueError("Los inputs completos no conservan la huella de sus observaciones")
        return rows

    def _check_sources(self, references, pending):
        if not isinstance(references, list) or len(references) > self.artifacts.max_files:
            raise ValueError("Las referencias de inputs exceden su presupuesto")
        positions = {_row_key(row): i for i, row in enumerate(pending["rows"])}
        found, used, digests = set(), [], set()
        for reference in references:
            rows = self._input_rows(reference)
            if reference["sha256"] in digests:
                raise ValueError("El bundle repite un artefacto de inputs")
            digests.add(reference["sha256"])
            active = False
            for row in rows:
                key = row.flow_id, row.prediction_at
                if key not in positions:
                    continue
                index = positions[key]
                if (
                    key in found
                    or row.metadata() != pending["rows"][index]
                    or row.key_inputs.tobytes() != pending["key_inputs"][index].numpy().tobytes()
                    or row.value.tobytes() != pending["values"][index].numpy().tobytes()
                ):
                    raise ValueError(
                        "Los rasgos pendientes no proceden de los inputs referenciados"
                    )
                found.add(key)
                active = True
            if active:
                used.append(reference)
        if found != set(positions):
            raise ValueError("Faltan inputs completos de una predicción pendiente")
        return used

    def step(self, batches, feedback, *, batch_rows=64, fault=None):
        if (
            self._inflight is not None
            or not isinstance(feedback, (list, tuple))
            or len(feedback) > 8192
        ):
            raise ValueError("La sesión está ocupada o excede el lote de maduración")
        rows = self._rows(batches)
        cohort = self.native.Cohort()
        cohort.cursor, cohort.cutoff = self._executor.cursor, rows[0].prediction_at
        cohort.observations = [
            self.native.Observation(
                row.flow_id,
                row.input_available_at,
                [
                    *row.key_inputs.tolist(),
                    *row.value.tolist(),
                    *(int(row.input_sha256[i : i + 8], 16) for i in range(0, 64, 8)),
                ],
            )
            for row in rows
        ]
        self._inflight = rows
        try:
            result = self._executor.step(cohort, feedback, batch_rows, fault)
            self._verify_confirmed()
            self._prune()
            return result
        except Exception:
            self.close()
            raise
        finally:
            self._inflight = None

    def _prepare(self, observations, tasks, cutoff, state_json, batch_rows):
        state = json.loads(state_json)
        bundle = self._bundle(state)
        rows = self._inflight
        if (
            rows is None
            or [row.flow_id for row in rows] != [row.asset for row in observations]
            or [(task.name, task.horizon) for task in tasks] != [(self.task, self.horizon)]
        ):
            raise ValueError("La preparación no corresponde a las observaciones del ejecutor")
        input_refs = self._stage_inputs(rows)
        recovered_rows = tuple(
            row for reference in input_refs for row in self._input_rows(reference)
        )
        if [row.metadata() for row in recovered_rows] != [row.metadata() for row in rows]:
            raise ValueError("La preparación no conserva el grupo de inputs sellados")
        rows = recovered_rows
        bank, episodes = self._bank(bundle["bank"] if bundle else None)
        pending = self._read(bundle["pending"], "pending") if bundle else _empty_pending()
        _check_pending(pending)
        fast = self._read(bundle["fast"], "fast") if bundle else None
        head = json.loads(self.native.read_blob(str(self.output / "latest.json"), 65536))
        context_id = _digest(
            dict(
                checkpoint=head["checkpoint_sha256"],
                contract=self.contract_id,
                group=[row.metadata() for row in rows],
            )
        )
        prepared = self._prepare_callback(
            rows, fast, _BankView(bank, episodes, cutoff), context_id, batch_rows
        )
        if (
            not isinstance(prepared, SessionPreparation)
            or not isinstance(prepared.predictions, dict)
            or set(prepared.predictions) != {row.flow_id for row in rows}
            or not isinstance(prepared.fast_state, dict)
        ):
            raise ValueError("La preparación no conserva salidas y estado explícitos")
        values = [prepared.predictions[row.flow_id] for row in rows]
        if any(
            type(value) not in (float, int, np.float32, np.float64) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("La preparación necesita una predicción finita por flujo")
        combined_rows = [*pending["rows"], *(row.metadata() for row in rows)]
        keys = torch.cat(
            (
                pending["key_inputs"],
                torch.tensor(np.stack([r.key_inputs for r in rows]), device="cpu"),
            )
        )
        data = torch.cat(
            (pending["values"], torch.tensor(np.stack([r.value for r in rows]), device="cpu"))
        )
        order = sorted(range(len(combined_rows)), key=lambda index: _row_key(combined_rows[index]))
        index = torch.tensor(order, dtype=torch.int64, device="cpu")
        next_pending = dict(
            rows=[combined_rows[i] for i in order],
            key_inputs=keys.index_select(0, index),
            values=data.index_select(0, index),
        )
        _check_pending(next_pending)
        proposal = dict(
            generation=state["generation"] + 1,
            cutoff=cutoff,
            bank=bundle["bank"] if bundle else None,
            fast=self._stage(prepared.fast_state, "fast"),
            pending=self._stage(next_pending, "pending"),
            inputs=[*(bundle["inputs"] if bundle else []), *input_refs],
        )
        return [float(value) for value in values], _canonical(proposal)

    def _update(self, proposed_json, outcomes):
        proposed = json.loads(proposed_json)
        bank, episodes = self._bank(proposed["bank"])
        pending = self._read(proposed["pending"], "pending")
        _check_pending(pending)
        positions = {_row_key(row): index for index, row in enumerate(pending["rows"])}
        incoming, removed = [], set()
        for offset, item in enumerate(outcomes, 1):
            prediction = item.prediction
            key = prediction.asset, prediction.decision_at
            if (
                key not in positions
                or key in removed
                or (prediction.task.name, prediction.task.horizon) != (self.task, self.horizon)
            ):
                raise ValueError("El resultado no enlaza una predicción pendiente de esta tarea")
            index = positions[key]
            metadata = pending["rows"][index]
            record = self.native.MemoryRecord()
            record.id, record.decision_at = bank.seen + offset, prediction.decision_at
            record.available_at, record.maturity_at = (
                metadata["input_available_at"],
                item.label.available_at,
            )
            record.key, record.value = (
                pending["key_inputs"][index].tolist(),
                pending["values"][index].tolist(),
            )
            record.label, record.label_valid = item.label.value, True
            incoming.append(record)
            episodes[record.id] = dict(
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
        if incoming:
            bank = bank.propose(incoming, confirmed_at=proposed["cutoff"])
        selected = {record.id for record in bank.records()}
        bank_ref = self._stage(
            dict(
                snapshot=bank.snapshot(), episodes={key: episodes[key] for key in sorted(selected)}
            ),
            "bank",
        )
        keep = [i for i, row in enumerate(pending["rows"]) if _row_key(row) not in removed]
        index = torch.tensor(keep, dtype=torch.int64, device="cpu")
        next_pending = dict(
            rows=[pending["rows"][i] for i in keep],
            key_inputs=pending["key_inputs"].index_select(0, index),
            values=pending["values"].index_select(0, index),
        )
        pending_ref = self._stage(next_pending, "pending")
        input_refs = self._check_sources(proposed["inputs"], next_pending)
        self._read(proposed["fast"], "fast")
        bundle = dict(
            generation=proposed["generation"],
            cutoff=proposed["cutoff"],
            bank=bank_ref,
            pending=pending_ref,
            fast=proposed["fast"],
            inputs=input_refs,
        )
        return _canonical(self._state(bundle["generation"], self._stage(bundle, "bundle")))

    def _verify_confirmed(self):
        snapshot = self.snapshot()
        state = snapshot["state"]
        self._check_state(state, self._executor.cursor)
        bundle = self._bundle(state)
        if bundle is None:
            return
        self._bank(bundle["bank"])
        self._read(bundle["fast"], "fast")
        pending = self._read(bundle["pending"], "pending")
        _check_pending(pending)
        if self._check_sources(bundle["inputs"], pending) != bundle["inputs"]:
            raise ValueError("El bundle conserva inputs sin decisiones pendientes")
        expected = {
            (prediction.asset, prediction.decision_at) for prediction in self._executor.pending()
        }
        if expected != {_row_key(row) for row in pending["rows"]}:
            raise ValueError("La cola de rasgos no coincide con las predicciones pendientes")

    def _prune(self):
        head = json.loads(self.native.read_blob(str(self.output / "latest.json"), 65536))
        references = []
        for generation, digest in (
            (head["generation"], head["checkpoint_sha256"]),
            (head["generation"] - 1, head["previous_sha256"]),
        ):
            if generation < 0:
                continue
            content = self.native.read_blob(
                str(self.output / f"checkpoint-{generation}.json"), 64 * 1024**2
            )
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("La poda no confirma la identidad del checkpoint")
            state = json.loads(content)["state"]
            self._check_state(state, generation)
            bundle = self._bundle(state)
            if bundle:
                self._bank(bundle["bank"])
                pending = self._read(bundle["pending"], "pending")
                _check_pending(pending)
                if self._check_sources(bundle["inputs"], pending) != bundle["inputs"]:
                    raise ValueError("El bundle anterior conserva inputs huérfanos")
                self._read(bundle["fast"], "fast")
                references.extend(
                    [state["bundle"], bundle["bank"], bundle["pending"], bundle["fast"]]
                )
                references.extend(bundle["inputs"])
        self.artifacts.prune_unreferenced(references)

    def snapshot(self):
        return json.loads(self._executor.snapshot_json())

    def retained_episodes(self):
        bundle = self._bundle(self.snapshot()["state"])
        _, episodes = self._bank(bundle["bank"] if bundle else None)
        return [episodes[key] for key in sorted(episodes)]

    def diagnostics(self):
        bundle = self._bundle(self.snapshot()["state"])
        bank, _ = self._bank(bundle["bank"] if bundle else None)
        return dict(
            cursor=self._executor.cursor,
            admitted=bank.seen,
            retained=bank._memory.size,
            pending=len(self._executor.pending()),
        )

    def close(self):
        self._executor.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
