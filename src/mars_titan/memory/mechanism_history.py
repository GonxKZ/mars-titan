"""Historial sellado de mecanismos, referenciado por la generación del ejecutor.

Este módulo comprueba identidad, secuencia, tamaños e integridad del archivo. El enlace
financiero valida la semántica de las observaciones, acciones y resultados del payload.
No publica una cabeza propia ni decide qué propuestas confirma la sesión.
"""

import json
import math
import re
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass

from mars_titan.models.titans.config import FingerprintOnce, bounded_integer, canonical

from .session_artifacts import SessionArtifacts

KIND = "mechanism_history_v1"
CHUNK_KIND = "mechanism_trace_chunk"
ENCODING = "canonical_json_utf8_v1"
FINAL_TEST_US = 1_704_067_200_000_000
EVENT_FIELDS = {
    "schema_version",
    "model_id",
    "generation",
    "cutoff",
    "kind",
    "native_record_sha256",
    "payload",
}


class HistoryCapacityError(ValueError):
    """La evidencia válida no cabe. Su agotamiento no permite eliminarla ni omitirla."""


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("El historial necesita una identidad SHA-256")


def _json_only(value, budget, depth=0):
    budget[0] -= 1
    if depth > 16:
        raise ValueError("La traza supera la profundidad admitida")
    if budget[0] < 0:
        raise HistoryCapacityError("La traza supera el número de elementos")
    kind = type(value)
    if value is None or kind is bool:
        return
    if kind is int and -(2**63) <= value < 2**63:
        return
    if kind is float and math.isfinite(value):
        return
    if kind is str:
        if len(value) > budget[1]:
            raise HistoryCapacityError("Una cadena de la traza supera su presupuesto")
        return
    if kind is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            _json_only(key, budget, depth + 1)
            _json_only(item, budget, depth + 1)
        return
    if kind in (list, tuple):
        for item in value:
            _json_only(item, budget, depth + 1)
        return
    raise ValueError("La traza necesita valores JSON finitos y tipos ordinarios")


def _json_bytes(value, maximum):
    _json_only(value, [1_000_000, maximum])
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    size = 0
    for part in encoder.iterencode(value):
        size += len(part)
        if size > maximum:
            raise HistoryCapacityError("La traza supera su presupuesto JSON")
    return size


@dataclass(frozen=True)
class MechanismHistoryConfig(FingerprintOnce):
    max_events_per_chunk: int = 32
    target_chunk_bytes: int = 4 * 1024**2
    max_event_bytes: int = 8 * 1024**2
    max_chunks: int = 512
    compression: str = "none"

    def __post_init__(self):
        bounded_integer(self.max_events_per_chunk, "eventos por bloque histórico", 1, 64)
        bounded_integer(self.target_chunk_bytes, "objetivo de bytes históricos", 1024, 16 * 1024**2)
        bounded_integer(
            self.max_event_bytes,
            "bytes por evento histórico",
            self.target_chunk_bytes,
            16 * 1024**2,
        )
        bounded_integer(self.max_chunks, "bloques históricos", 1, 512)
        if type(self.compression) is not str or self.compression not in ("none", "zlib1"):
            raise ValueError("La compresión histórica debe ser none o zlib1")

    def identity(self):
        return dict(
            kind=KIND,
            encoding=ENCODING,
            **asdict(self),
            zlib_version=zlib.ZLIB_RUNTIME_VERSION if self.compression == "zlib1" else None,
        )


class MechanismHistory:
    """Bloques inmutables y cursor externo, con caché de verificación acotada."""

    def __init__(self, artifacts, *, identity, model_id, config=None):
        config = MechanismHistoryConfig() if config is None else config
        if type(artifacts) is not SessionArtifacts or type(config) is not MechanismHistoryConfig:
            raise ValueError("El historial necesita los artefactos y límites identificados")
        _digest(identity)
        _digest(model_id)
        if config.max_event_bytes + 65536 > artifacts.max_bytes:
            raise ValueError("El evento histórico no cabe en el presupuesto de artefactos")
        self.artifacts, self.identity, self.model_id, self.config = (
            artifacts,
            identity,
            model_id,
            config,
        )
        self._verified = OrderedDict()

    def empty(self):
        return dict(
            schema_version=1,
            kind=KIND,
            model_id=self.model_id,
            config_id=self.config.fingerprint(),
            events=0,
            cutoff=None,
            bytes=0,
            chunks=[],
        )

    def _event(self, value, generation):
        if (
            type(value) is not dict
            or len(value) != len(EVENT_FIELDS)
            or any(type(key) is not str for key in value)
            or set(value) != EVENT_FIELDS
        ):
            raise ValueError("El evento histórico no conserva su envoltura")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or type(value["model_id"]) is not str
            or value["model_id"] != self.model_id
        ):
            raise ValueError("El evento pertenece a otro modelo o esquema")
        bounded_integer(value["generation"], "generación histórica", generation, generation)
        bounded_integer(value["cutoff"], "corte histórico", 1, FINAL_TEST_US)
        if (
            type(value["kind"]) is not str
            or value["kind"] not in ("warmup", "decision", "settlement")
            or type(value["payload"]) is not dict
        ):
            raise ValueError("El evento histórico no conserva su tipo y payload")
        if value["cutoff"] == FINAL_TEST_US and value["kind"] != "settlement":
            raise ValueError("El límite reservado solo admite cierre administrativo")
        _digest(value["native_record_sha256"])
        _json_bytes(value, self.config.max_event_bytes)

    def _stamp(self, reference):
        header = self.artifacts._header(self.identity, CHUNK_KIND)
        if (
            type(reference) is not dict
            or len(reference) != len(header) + 3
            or any(type(key) is not str for key in reference)
            or set(reference) != set(header) | {"name", "sha256", "bytes"}
            or any(
                type(reference[key]) is not type(value) or reference[key] != value
                for key, value in header.items()
            )
            or type(reference.get("name")) is not str
            or re.fullmatch(r"[0-9a-f]{64}\.bin", reference["name"]) is None
        ):
            raise ValueError("La referencia histórica no tiene un nombre admitido")
        _digest(reference["sha256"])
        if reference["name"] != reference["sha256"] + ".bin":
            raise ValueError("La referencia histórica no conserva su huella")
        bounded_integer(
            reference["bytes"], "bytes del bloque histórico", 1, self.artifacts.max_bytes
        )
        path = self.artifacts.directory / reference["name"]
        self.artifacts.native.require_safe_path(str(path))
        if not path.is_file():
            raise ValueError("Falta un bloque histórico confirmado")
        stamp = path.stat()
        return stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns

    def _remember(self, reference, stamp, summary):
        key = canonical(reference)
        self._verified[key] = stamp, summary
        self._verified.move_to_end(key)
        while len(self._verified) > self.config.max_chunks + 2:
            self._verified.popitem(last=False)

    def _decode(self, content, generation=None):
        if type(content) is not bytes or not 0 < len(content) <= self.artifacts.max_bytes:
            raise ValueError("El evento histórico necesita bytes acotados")
        if self.config.compression == "zlib1":
            decoder = zlib.decompressobj()
            try:
                plain = decoder.decompress(content, self.config.max_event_bytes + 1)
            except zlib.error as error:
                raise ValueError("El evento comprimido no conserva su formato") from error
            if len(plain) > self.config.max_event_bytes:
                raise HistoryCapacityError("El evento descomprimido supera su presupuesto")
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("El evento comprimido tiene datos sobrantes o incompletos")
        else:
            plain = content
            if len(plain) > self.config.max_event_bytes:
                raise HistoryCapacityError("El evento JSON supera su presupuesto")
        try:
            value = json.loads(plain)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
            raise ValueError("El evento histórico no contiene JSON válido") from error
        first = value.get("generation") if type(value) is dict else None
        bounded_integer(
            first,
            "generación del evento codificado",
            1,
            self.config.max_chunks * self.config.max_events_per_chunk,
        )
        self._event(value, first if generation is None else generation)
        if canonical(value).encode("ascii") != plain:
            raise ValueError("El evento no conserva una representación JSON canónica única")
        return value

    def _chunk(self, reference, *, strong=False):
        before = self._stamp(reference)
        data = self.artifacts.read(reference, identity=self.identity, kind=CHUNK_KIND)
        if (
            type(data) is not dict
            or set(data) != {"schema_version", "model_id", "events", "encoding", "compression"}
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["model_id"] != self.model_id
            or data["encoding"] != ENCODING
            or data["compression"] != self.config.compression
            or type(data["events"]) is not list
        ):
            raise ValueError("El bloque histórico no conserva su modelo y formato")
        events = data["events"]
        bounded_integer(
            len(events), "eventos históricos del bloque", 1, self.config.max_events_per_chunk
        )
        known = self._verified.get(canonical(reference))
        if not strong and known is not None and before == known[0]:
            if before != self._stamp(reference):
                raise ValueError("El bloque histórico cambió durante su lectura")
            self._remember(reference, before, known[1])
            return data, known[1]
        first_event = self._decode(events[0])
        first = first_event["generation"]
        previous = 0
        for offset, content in enumerate(events):
            event = first_event if offset == 0 else self._decode(content, first + offset)
            if event["cutoff"] < previous:
                raise ValueError("El reloj histórico retrocede dentro del bloque")
            previous = event["cutoff"]
        if before != self._stamp(reference):
            raise ValueError("El bloque histórico cambió durante su lectura")
        summary = (first, first + len(events) - 1, first_event["cutoff"], previous)
        self._remember(reference, before, summary)
        return data, summary

    def verify(self, manifest, *, strong=False):
        empty = self.empty()
        if (
            type(strong) is not bool
            or type(manifest) is not dict
            or len(manifest) != len(empty)
            or any(type(key) is not str for key in manifest)
            or set(manifest) != set(empty)
            or any(
                type(manifest[key]) is not str or manifest[key] != empty[key]
                for key in ("kind", "model_id", "config_id")
            )
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or type(manifest["chunks"]) is not list
        ):
            raise ValueError("El manifiesto histórico pertenece a otro contrato")
        bounded_integer(
            manifest["events"],
            "eventos históricos",
            0,
            self.config.max_chunks * self.config.max_events_per_chunk,
        )
        bounded_integer(manifest["bytes"], "bytes históricos", 0, self.artifacts.max_total_bytes)
        bounded_integer(len(manifest["chunks"]), "bloques históricos", 0, self.config.max_chunks)
        if manifest["events"]:
            bounded_integer(manifest["cutoff"], "corte del historial", 1, FINAL_TEST_US)
        elif manifest["cutoff"] is not None:
            raise ValueError("El historial vacío no tiene un corte confirmado")
        expected, cutoff, total = 1, 0, 0
        for chunk in manifest["chunks"]:
            if type(chunk) is not dict or set(chunk) != {"reference", "first", "last", "cutoff"}:
                raise ValueError("El índice histórico no conserva su formato")
            reference = chunk["reference"]
            stamp = self._stamp(reference)
            key = canonical(reference)
            known = self._verified.get(key)
            if strong or known is None or stamp != known[0]:
                _, summary = self._chunk(reference, strong=strong)
            else:
                summary = known[1]
                self._verified.move_to_end(key)
            first, last, first_cutoff, last_cutoff = summary
            if (
                any(type(chunk[name]) is not int for name in ("first", "last", "cutoff"))
                or (chunk["first"], chunk["last"], chunk["cutoff"]) != (first, last, last_cutoff)
                or first != expected
                or first_cutoff < cutoff
            ):
                raise ValueError("El índice histórico contradice las generaciones o sus relojes")
            expected, cutoff = last + 1, last_cutoff
            total += reference["bytes"]
        if (
            manifest["events"] != expected - 1
            or manifest["cutoff"] != (cutoff if expected > 1 else None)
            or manifest["bytes"] != total
        ):
            raise ValueError("El manifiesto no concilia con el archivo histórico")
        return dict(events=expected - 1, chunks=len(manifest["chunks"]), bytes=total)

    def append(self, manifest, event):
        self.verify(manifest)
        self._event(event, manifest["events"] + 1)
        event = dict(event)
        encoded = canonical(event).encode("ascii")
        if self.config.compression == "zlib1":
            encoded = zlib.compress(encoded, level=1)
        if manifest["cutoff"] is not None and event["cutoff"] < manifest["cutoff"]:
            raise ValueError("El evento histórico no puede retroceder el reloj")
        chunks = [dict(chunk) for chunk in manifest["chunks"]]
        events = []
        first, first_cutoff = event["generation"], event["cutoff"]
        if (
            chunks
            and chunks[-1]["last"] - chunks[-1]["first"] + 1 < self.config.max_events_per_chunk
        ):
            data, summary = self._chunk(chunks[-1]["reference"])
            if sum(map(len, data["events"])) + len(encoded) <= self.config.target_chunk_bytes:
                events = data["events"]
                first, _, first_cutoff, _ = summary
        if events:
            chunks.pop()
        elif len(chunks) == self.config.max_chunks:
            raise HistoryCapacityError(
                "El historial agota su presupuesto de bloques sin borrar evidencia"
            )
        data = dict(
            schema_version=1,
            model_id=self.model_id,
            encoding=ENCODING,
            compression=self.config.compression,
            events=events + [encoded],
        )
        reference = self.artifacts.stage(data, identity=self.identity, kind=CHUNK_KIND)
        # stage ya ha validado, serializado y sellado estos bytes. Solo se reutilizan sus
        # metadatos mientras conserve la misma referencia y firma de archivo.
        self._remember(
            reference,
            self._stamp(reference),
            (first, event["generation"], first_cutoff, event["cutoff"]),
        )
        chunks.append(
            dict(
                reference=reference,
                first=first,
                last=event["generation"],
                cutoff=event["cutoff"],
            )
        )
        result = dict(
            self.empty(),
            chunks=chunks,
            events=event["generation"],
            cutoff=event["cutoff"],
            bytes=sum(chunk["reference"]["bytes"] for chunk in chunks),
        )
        self.verify(result)
        return result

    def live_references(self, manifest):
        self.verify(manifest)
        return [dict(chunk["reference"]) for chunk in manifest["chunks"]]

    def read_event(self, manifest, generation):
        self.verify(manifest)
        bounded_integer(generation, "generación histórica solicitada", 1, manifest["events"])
        for chunk in manifest["chunks"]:
            if chunk["first"] <= generation <= chunk["last"]:
                data, _ = self._chunk(chunk["reference"])
                return self._decode(data["events"][generation - chunk["first"]], generation)
        raise ValueError("No se encuentra la generación en el índice histórico")

    def iter_events(self, manifest):
        self.verify(manifest)
        for chunk in manifest["chunks"]:
            data, _ = self._chunk(chunk["reference"])
            for offset, content in enumerate(data["events"]):
                yield self._decode(content, chunk["first"] + offset)
