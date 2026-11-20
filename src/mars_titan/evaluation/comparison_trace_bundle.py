"""Comprobaciones de los recibos y las partes de traza de un trabajo de comparación."""

import hashlib
import json
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.learning_traces import schema
from mars_titan.training.comparison_traces import MAX_TRACE_BYTES

_HASH = re.compile(r"[a-f0-9]{64}")
_EVENT = ("step", "phase", "metric", "group", "stat")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def policy(record, runner):
    """Recuperar la política compartida y el paso de reanudación del último intento."""
    policies = [
        attempt["learning_traces"]
        for attempt in record.get("attempts", [])
        if attempt.get("learning_traces") is not None
    ]
    if not policies:
        return None
    first = policies[0]
    _require(
        runner in ("fit", "reader")
        and isinstance(first, dict)
        and _integer(first.get("every"), 1)
        and _integer(first.get("max_bytes"), 1024)
        and first["max_bytes"] <= MAX_TRACE_BYTES
        and _integer(first.get("start_step"))
        and all(
            isinstance(first.get(k), str) and _HASH.fullmatch(first[k])
            for k in ("policy_sha256", "qualification_sha256")
        ),
        "La política de trazas del intento no es válida",
    )
    fields = ("every", "max_bytes", "start_step", "policy_sha256", "qualification_sha256")
    _require(
        all(
            isinstance(p, dict)
            and _integer(p.get("resume_step"), first["start_step"])
            and all(type(p.get(k)) is type(first[k]) and p[k] == first[k] for k in fields)
            for p in policies
        ),
        "Los intentos no comparten la cobertura y procedencia de las trazas",
    )
    return policies[-1]


def confirmed(record):
    """Solo el último intento cerrado puede confirmar el cursor de recuperación."""
    flag = record["attempts"][-1].get("trace_checkpoint_confirmed")
    _require(flag is None or type(flag) is bool, "La confirmación de trazas debe ser booleana")
    _require(
        record["status"] != "completed" or flag is not None,
        "Falta la confirmación de trazas del trabajo terminado",
    )
    return record["status"] in ("completed", "paused") and flag is True


def check_header(job, window, fit, declared, receipt, manifest, *, max_parts):
    """Vincular el cursor al checkpoint, el informe de ajuste y su configuración."""
    _require(
        isinstance(fit, dict) and isinstance(fit.get("identity"), dict),
        "Falta la identidad del ajuste con trazas",
    )
    _require(
        isinstance(receipt, dict)
        and type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and receipt.get("kind") == "comparison_learning_trace_checkpoint"
        and all(
            isinstance(receipt.get(k), str) and _HASH.fullmatch(receipt[k])
            for k in ("fit_run_id", "checkpoint_sha256")
        )
        and receipt["fit_run_id"] == fit.get("run_id")
        and receipt["fit_run_id"] == window.get("identity", {}).get("fit_run_id")
        and window.get("final_test_opened") is False
        and fit.get("final_test_opened") is False
        and receipt["checkpoint_sha256"] == (fit.get("recovery_checkpoint") or {}).get("sha256")
        and _integer(receipt.get("start_step"))
        and _integer(receipt.get("global_step"), declared["resume_step"])
        and type(fit.get("global_step")) is int
        and receipt["global_step"] == fit.get("global_step")
        and all(
            receipt.get(k) == declared[k]
            for k in ("start_step", "policy_sha256", "qualification_sha256")
        ),
        "El recibo de trazas no corresponde al checkpoint y al paso del ajuste",
    )
    cursor = receipt.get("cursor")
    _require(
        isinstance(cursor, dict)
        and set(cursor) == {"sequence", "bytes", "exhausted_at"}
        and _integer(cursor["sequence"])
        and cursor["sequence"] < max_parts
        and _integer(cursor["bytes"])
        and cursor["bytes"] <= declared["max_bytes"]
        and (
            cursor["exhausted_at"] is None
            or (
                _integer(cursor["exhausted_at"], receipt["start_step"])
                and cursor["exhausted_at"] <= receipt["global_step"]
            )
        ),
        "El cursor confirmado de trazas no es válido",
    )
    encoded = json.dumps(fit["identity"], sort_keys=True, default=str, allow_nan=False)
    identity_digest = hashlib.sha256(encoded.encode()).hexdigest()
    config = manifest.get("config", {})
    identity = manifest.get("identity", {})
    family = "mars_titan_readout" if job.runner == "reader" else job.model
    _require(
        isinstance(identity, dict)
        and identity.get("family") == family
        and re.fullmatch(schema.NAME_PATTERN, family),
        "La familia de las trazas no corresponde al trabajo",
    )
    _require(
        manifest.get("kind") == schema.KIND
        and type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == schema.SCHEMA_VERSION
        and manifest.get("columns") == schema.SCHEMA.names
        and manifest.get("stats") == list(schema.STATS)
        and manifest.get("step_unit") == "optimizer_step"
        and "exhausted_at_step" in manifest
        and type(manifest["exhausted_at_step"]) is type(cursor["exhausted_at"])
        and manifest["exhausted_at_step"] == cursor["exhausted_at"]
        and identity.get("run_identity_sha256") == identity_digest
        and config.get("enabled") is True
        and type(config.get("every")) is int
        and config["every"] == declared["every"]
        and type(config.get("max_bytes")) is int
        and config["max_bytes"] == declared["max_bytes"],
        "El manifiesto de trazas no corresponde al ajuste, su cobertura o su presupuesto",
    )


def read_part(source, *, max_rows, max_bytes):
    """Comprobar el tamaño y la longitud de las claves antes de expandir sus diccionarios."""
    metadata = pq.read_metadata(source)
    _require(
        metadata.schema.to_arrow_schema().equals(schema.SCHEMA)
        and 0 < metadata.num_rows <= max_rows
        and sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        <= max_bytes,
        "Una parte de trazas supera el límite o tiene otro esquema",
    )
    names = _EVENT[1:]
    with pq.ParquetFile(source, metadata=metadata, read_dictionary=list(names)) as parquet:
        table = parquet.read(use_threads=False)
    for name in names:
        for chunk in table[name].chunks:
            values = chunk.dictionary if pa.types.is_dictionary(chunk.type) else chunk
            _require(
                (pc.max(pc.binary_length(values)).as_py() or 0) <= 96,
                "Un nombre de traza supera el límite de longitud",
            )
        table = table.set_column(
            table.schema.get_field_index(name), name, table[name].cast(pa.string())
        )
    return table


def check_parts(sources, receipt, every, *, max_rows, max_bytes):
    """Validar una parte cada vez. Solo las filas del último paso pueden cruzar la frontera."""
    rows, first, last, boundary = 0, None, None, None
    upper_step = receipt["cursor"]["exhausted_at"]
    if upper_step is None:
        upper_step = receipt["global_step"]
    for source in sources:
        table = read_part(source, max_rows=max_rows, max_bytes=max_bytes)
        _require(all(c.null_count == 0 for c in table.columns), "Hay celdas de traza ausentes")
        phase, steps = table["phase"], table["step"]
        bounds = pc.min_max(steps).as_py()
        low, high = bounds["min"], bounds["max"]
        _require(
            receipt["start_step"] <= low <= high <= upper_step and (last is None or low >= last),
            "Una parte de trazas queda fuera de los pasos confirmados o retrocede",
        )
        _require(
            pc.all(pc.is_in(phase, value_set=pa.array(["train", "validation"]))).as_py()
            and pc.all(
                pc.is_in(table["stat"], value_set=pa.array([*schema.STATS, "value"]))
            ).as_py()
            and all(
                pc.all(
                    pc.match_substring_regex(pc.unique(table[k]), f"^{schema.NAME_PATTERN}$")
                ).as_py()
                for k in ("metric", "group")
            ),
            "Una traza tiene una fase, un nombre o una estadística desconocidos",
        )
        trained = pc.unique(pc.filter(steps, pc.equal(phase, "train"))).to_pylist()
        _require(
            all(s % every == 0 for s in trained),
            "Un paso de ajuste no respeta la cadencia de trazas",
        )
        keys = table.select(_EVENT)
        if low == last:
            keys = pa.concat_tables([boundary, keys])
        _require(
            keys.group_by(list(_EVENT), use_threads=False).aggregate([]).num_rows == keys.num_rows,
            "Una observación de traza está duplicada",
        )
        boundary = keys.filter(pc.equal(keys["step"], high))
        _require(boundary.num_rows <= max_rows, "Un paso de trazas supera el límite de filas")
        rows += table.num_rows
        first = low if first is None else first
        last = high
    return dict(rows=rows, first_observed_step=first, last_observed_step=last)
