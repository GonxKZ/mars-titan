"""Cursores por flujo sin estado neural, para consumidores que reinician su estado por ventana.

Conservan los campos que el coordinador comprueba en el estado rápido de Titans:
observaciones, último corte y última muestra. No hay bloques ni pesos rápidos.
"""

import re

MAX_FLOWS = 8192
_FLOW = re.compile(r"(?:US|CN)/[A-Z0-9.^_=\-]{1,64}")
_FIELDS = {"observed_steps", "last_prediction_at", "last_sample_id"}


class FlowCursors:
    def __init__(self, consumer):
        self.consumer = consumer

    def empty(self):
        return dict(schema_version=1, model_id=self.consumer.model_id, references={}, blocks={})

    def verify(self, manifest):
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "model_id", "references", "blocks"}
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["model_id"] != self.consumer.model_id
            or manifest["blocks"] != {}
            or not isinstance(manifest["references"], dict)
            or len(manifest["references"]) > MAX_FLOWS
        ):
            raise ValueError("Los cursores no conservan el modelo, sus campos o su límite")
        for flow, record in manifest["references"].items():
            if (
                not isinstance(flow, str)
                or _FLOW.fullmatch(flow) is None
                or not isinstance(record, dict)
                or set(record) != _FIELDS
                or any(type(record[key]) is not int for key in _FIELDS - {"last_sample_id"})
                or record["observed_steps"] < 1
                or record["last_prediction_at"] < 0
                or record["last_sample_id"] != f"{flow}/{record['last_prediction_at']}"
            ):
                raise ValueError("Un cursor no identifica su flujo, observaciones y corte")
        return dict(flows=len(manifest["references"]), blocks=0, source_bytes=0)

    def needs_compaction(self, manifest, incoming_blocks=0):
        self.verify(manifest)
        return False

    def live_references(self, manifest):
        self.verify(manifest)
        return []

    def advance(self, manifest, batch):
        """Avanzar una observación por flujo. Los flujos ausentes conservan su cursor."""
        self.verify(manifest)
        references = {flow: dict(record) for flow, record in manifest["references"].items()}
        for flow, sample, moment in zip(
            batch.flow_ids, batch.sample_ids, batch.prediction_at, strict=True
        ):
            previous = references.get(flow)
            if previous is not None and moment <= previous["last_prediction_at"]:
                raise ValueError("El cursor GRU repite o retrocede una observación")
            references[flow] = dict(
                observed_steps=1 + (previous["observed_steps"] if previous else 0),
                last_prediction_at=moment,
                last_sample_id=sample,
            )
        result = dict(self.empty(), references=references)
        self.verify(result)
        return result
