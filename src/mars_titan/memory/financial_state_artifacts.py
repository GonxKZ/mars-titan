"""Referencias de estado rápido y compactación CPU, sin publicar generaciones."""

import math

import torch

from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_blocks import (
    _copy_to,
    _cpu_payload,
    _decode,
    _export_state_cpu,
    _gather_cpu,
    _replace_state_references,
    _validate_references,
)


class FinancialStateArtifacts:
    def __init__(
        self,
        consumer,
        artifacts,
        *,
        identity,
        max_source_bytes=512 * 1024**2,
        compact_after_blocks=128,
        compact_after_bytes=512 * 1024**2,
    ):
        bounded_integer(max_source_bytes, "bytes de fuentes rápidas", 1, 2 * 1024**3)
        bounded_integer(compact_after_blocks, "bloques antes de compactar", 1, 1024)
        bounded_integer(compact_after_bytes, "bytes antes de compactar", 1, 2 * 1024**3)
        self.consumer, self.predictor = consumer, consumer.predictor
        self.artifacts, self.identity = artifacts, identity
        self.max_source_bytes = max_source_bytes
        self.compact_after_blocks, self.compact_after_bytes = (
            compact_after_blocks,
            compact_after_bytes,
        )

    def empty(self):
        return dict(schema_version=1, model_id=self.consumer.model_id, references={}, blocks={})

    def _header(self, manifest):
        self.consumer.verify(strong=False)
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "model_id", "references", "blocks"}
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["model_id"] != self.consumer.model_id
            or not isinstance(manifest["blocks"], dict)
        ):
            raise ValueError("El manifiesto rápido no conserva el modelo y sus campos")
        _validate_references(self.predictor, manifest["references"])
        needed = {row["block_id"] for row in manifest["references"].values()}
        if needed != set(manifest["blocks"]):
            raise ValueError("Los bloques rápidos no corresponden exactamente a las filas vivas")
        total = 0
        for identity, reference in manifest["blocks"].items():
            if (
                not isinstance(reference, dict)
                or reference.get("sha256") != identity
                or type(reference.get("bytes")) is not int
                or not 0 < reference["bytes"] <= self.artifacts.max_bytes
            ):
                raise ValueError("El bloque rápido no conserva su ID y tamaño")
            total += reference["bytes"]
        if len(needed) > self.artifacts.max_files or total > self.artifacts.max_total_bytes:
            raise ValueError("Las fuentes rápidas superan el presupuesto de artefactos")

    def _read(self, reference):
        payload = self.artifacts.read(reference, identity=self.identity, kind="fast_block")
        state = _decode(self.predictor, payload, device=torch.device("cpu"))
        return payload, state

    @staticmethod
    def _match(flow, record, state):
        row = record["row"]
        if row >= len(state.flow_ids) or state.flow_ids[row] != flow:
            raise ValueError("La referencia rápida apunta a otro flujo")
        expected = dict(
            config_id=state.config_id,
            parameter_id=state.parameter_id,
            observed_steps=int(state.observed_steps[row]),
            last_sample_id=state.last_sample_ids[row],
            last_prediction_at=state.last_prediction_at[row],
        )
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError("Los cursores de la referencia no coinciden con el payload confirmado")

    def verify(self, manifest):
        self._header(manifest)
        by_block = {}
        for flow, record in manifest["references"].items():
            by_block.setdefault(record["block_id"], []).append((flow, record))
        for identity, reference in manifest["blocks"].items():
            _, state = self._read(reference)
            for flow, record in by_block[identity]:
                self._match(flow, record, state)
        return dict(
            flows=len(manifest["references"]),
            blocks=len(manifest["blocks"]),
            source_bytes=sum(r["bytes"] for r in manifest["blocks"].values()),
        )

    def live_references(self, manifest):
        self.verify(manifest)
        return [dict(manifest["blocks"][key]) for key in sorted(manifest["blocks"])]

    def _gather(self, manifest, flow_ids, initial_state=None):
        self._header(manifest)
        self.predictor._check_flows(flow_ids)
        requested = tuple(flow_ids)
        references = manifest["references"]
        unknown = tuple(flow for flow in requested if flow not in references)
        if unknown:
            if initial_state is None or initial_state.flow_ids != unknown:
                raise ValueError("Los flujos nuevos requieren initial_state explícito y exacto")
            self.predictor._validate_state(initial_state)
            if torch.count_nonzero(initial_state.observed_steps).item():
                raise ValueError("El estado inicial nuevo ya contiene observaciones")
        elif initial_state is not None:
            raise ValueError("No hay flujos nuevos para el estado inicial proporcionado")
        needed = {references[flow]["block_id"] for flow in requested if flow in references}
        total = sum(manifest["blocks"][identity]["bytes"] for identity in needed)
        if initial_state is not None:
            total += self.predictor.state_usage(initial_state)["total_bytes"]
        if total > self.max_source_bytes:
            raise ValueError("Los bloques únicos pedidos superan su presupuesto de bytes")
        sources, loaded = {}, {}
        for identity in sorted(needed):
            loaded[identity] = self._read(manifest["blocks"][identity])
        for flow in requested:
            if flow in references:
                record = references[flow]
                payload, state = loaded[record["block_id"]]
                self._match(flow, record, state)
                sources[flow] = payload, record["row"]
        if unknown:
            payload = _export_state_cpu(self.predictor, initial_state)
            sources.update({flow: (payload, row) for row, flow in enumerate(unknown)})
        return _gather_cpu(
            self.predictor, sources, requested, max_source_bytes=self.max_source_bytes
        )

    def gather(self, manifest, flow_ids, *, initial_state=None):
        state = self._gather(manifest, flow_ids, initial_state)
        if self.predictor.head.weight.device.type != "cpu":
            state = _copy_to(state, self.predictor.head.weight.device)
        self.predictor._validate_state(state)
        return state

    def replace(self, manifest, previous, following):
        self._header(manifest)
        # Se valida el reemplazo antes de escribir cualquier propuesta.
        _replace_state_references(
            self.predictor, manifest["references"], previous, following, "0" * 64
        )
        payload = _export_state_cpu(self.predictor, following)
        reference = self.artifacts.stage(payload, identity=self.identity, kind="fast_block")
        identity = reference["sha256"]
        references = _replace_state_references(
            self.predictor, manifest["references"], previous, following, identity
        )
        blocks = {**manifest["blocks"], identity: reference}
        needed = {row["block_id"] for row in references.values()}
        result = dict(
            self.empty(), references=references, blocks={key: blocks[key] for key in sorted(needed)}
        )
        self._header(result)
        return result

    def needs_compaction(self, manifest, incoming_blocks=0):
        self._header(manifest)
        bounded_integer(incoming_blocks, "bloques entrantes", 0, 8192)
        return (
            len(manifest["blocks"]) + incoming_blocks > self.compact_after_blocks
            or sum(r["bytes"] for r in manifest["blocks"].values()) > self.compact_after_bytes
        )

    def compact(self, manifest):
        self.verify(manifest)
        flows = tuple(sorted(manifest["references"]))
        if not flows:
            return self.empty()
        per_flow = self.predictor._tensor_bytes_per_flow()
        rows = min(
            self.predictor.config.max_batch,
            max(
                1,
                (min(self.artifacts.max_bytes, self.predictor.config.max_state_bytes) - 1024**2)
                // per_flow,
            ),
        )
        outputs = math.ceil(len(flows) / rows)
        # Incluye todas las generaciones todavía presentes y un margen por bloque.
        files = self.artifacts._inventory()
        estimate = len(flows) * self.predictor._tensor_bytes_per_flow() + outputs * 1024**2
        if (
            len(files) + outputs > self.artifacts.max_files
            or sum(path.stat().st_size for path in files) + estimate
            > self.artifacts.max_total_bytes
        ):
            raise ValueError("La compactación no cabe junto a las generaciones conservadas")
        references, blocks = {}, {}
        for start in range(0, len(flows), rows):
            selected = flows[start : start + rows]
            state = self._gather(manifest, selected)
            payload = _cpu_payload(self.predictor, state)
            reference = self.artifacts.stage(payload, identity=self.identity, kind="fast_block")
            identity = reference["sha256"]
            blocks[identity] = reference
            for row, flow in enumerate(selected):
                references[flow] = dict(manifest["references"][flow], block_id=identity, row=row)
        result = dict(self.empty(), references=references, blocks=blocks)
        self.verify(result)
        return result
