"""Describir decisiones confirmadas a partir de trazas, sin volver a ejecutar la política."""

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import safe_destination

MAX_INDEX = 4 * 1024**2
MAX_SHARD = 1024**2


def _read(path, maximum):
    safe_destination(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("El archivo de trazas no es regular o supera su límite")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("El archivo de trazas creció por encima de su límite")
    return data


def _json(path):
    def object_hook(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("El documento de trazas contiene campos repetidos")
            result[name] = value
        return result

    return json.loads(_read(path, MAX_INDEX), object_pairs_hook=object_hook)


def _count(value):
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ValueError("La traza contiene un contador no admitido")
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("La identidad de la traza no es una huella SHA256")
    return value


def _time(value):
    try:
        return (
            datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=_count(value))
        ).isoformat()
    except OverflowError as error:
        raise ValueError("La fecha de la traza excede el calendario admitido") from error


def _describe(row):
    action = _count(row["action"])
    probabilities = [row[f"probability_{index}"] for index in range(6)]
    if (
        action >= 6
        or any(
            type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1
            for value in probabilities
        )
        or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)
    ):
        raise ValueError("La decisión contiene acciones o probabilidades inválidas")
    for name in ("critic", "reward", "costs_delta"):
        if type(row[name]) not in (float, int) or not math.isfinite(row[name]):
            raise ValueError("La decisión contiene valores no finitos")
    for name in ("learning_allowed", "reward_valid", "terminated", "truncated", "costs_present"):
        if type(row[name]) is not bool:
            raise ValueError("La decisión no conserva sus máscaras booleanas")
    if not row["reward_valid"] and row["reward"] != 0:
        raise ValueError("Una recompensa sin valoración necesita el marcador cero")
    when = _time(row["decision_at"])
    if _count(row["outcome_at"]) <= row["decision_at"]:
        raise ValueError("El resultado no es posterior al momento de la decisión")
    outcome_time = _time(row["outcome_at"])
    mode = row["mode"]
    if mode == "warmup":
        if action != 1 or row["learning_allowed"]:
            raise ValueError("La decisión de calentamiento no respeta la exclusión del aprendizaje")
        reason = (
            "La acción se forzó a efectivo durante el calentamiento y no entró en el aprendizaje."
        )
    elif mode == "sampled":
        reason = "La acción se eligió mediante muestreo de la distribución registrada."
    elif mode == "greedy":
        reason = "La acción se eligió por el máximo de la política durante la evaluación."
    else:
        raise ValueError("La traza contiene un modo de decisión desconocido")
    memories = _count(row["retrieved_count"])
    if memories > 4:
        raise ValueError("La decisión excede cuatro recuerdos")
    evidence = []
    for index in range(memories):
        available = _count(row[f"matured_at_{index}"])
        if available > row["decision_at"]:
            raise ValueError("La decisión contiene un recuerdo no maduro")
        score = row[f"similarity_{index}"]
        if type(score) not in (float, int) or not math.isfinite(score):
            raise ValueError("La similitud del recuerdo no es finita")
        identifier = _count(row[f"memory_id_{index}"])
        evidence.append(
            f"recuerdo {identifier}, similitud {score:.6g}, maduro desde {_time(available)}"
        )
    source = _digest(row["world_sha256"])
    context = _digest(row["context_sha256"])
    text = (
        f"Decisión {_count(row['decision_id'])}, entorno {_count(row['lane'])}, "
        f"episodio {_count(row['episode'])}, sesión {_count(row['cursor'])}, {when}. "
        f"Acción {action}. {reason} Probabilidad registrada de esa acción: "
        f"{probabilities[action]:.6g}. Valor interno estimado: {row['critic']:.6g}. "
        f"Actualización {_count(row['optimizer_step'])}. "
        f"Fuente {source}. Contexto {context}."
    )
    text += (
        " Se consultaron " + ", ".join(evidence) + "."
        if evidence
        else " No se consultaron recuerdos."
    )
    text += (
        f" Recompensa posterior observada en {outcome_time}: {row['reward']:.6g}."
        if row["reward_valid"]
        else " La transición no dispone de una recompensa valorable."
    )
    if row["costs_present"]:
        if row["costs_delta"] < 0:
            raise ValueError("Los costes de la decisión no pueden ser negativos")
        text += f" Costes de ejecución registrados: {row['costs_delta']:.6g}."
    if type(row["memory_sensitivity_present"]) is not bool:
        raise ValueError("Falta la máscara de disponibilidad de la sensibilidad")
    if row["memory_sensitivity_present"]:
        masked = [row[f"memory_masked_probability_{index}"] for index in range(6)]
        changed_action = _count(row["memory_masked_action"])
        distance = row["memory_probability_l1"]
        if (
            changed_action >= 6
            or type(row["memory_action_changed"]) is not bool
            or row["memory_action_changed"] != (changed_action != action)
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in masked
            )
            or not math.isclose(sum(masked), 1, abs_tol=1e-5)
            or type(distance) not in (int, float)
            or not math.isfinite(distance)
            or not math.isclose(
                distance,
                sum(abs(a - b) for a, b in zip(masked, probabilities, strict=True)),
                abs_tol=1e-5,
            )
        ):
            raise ValueError("La sensibilidad no coincide con las distribuciones registradas")
        text += (
            f" Al ocultar los recuerdos con la misma observación y estado interno, "
            f"la acción por máximo fue {changed_action} y la distancia L1 entre distribuciones "
            f"fue {distance:.6g}. Es una sensibilidad local, "
            "sin identificación de causas económicas."
        )
    else:
        text += " Sensibilidad no registrada."
    return text


def explain(run_path, *, limit=20, after=0):
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("La explicación admite entre una y 200 decisiones")
    after = _count(after)
    run_path = Path(run_path)
    report = _json(run_path)
    trace = report["trace"]
    confirmed = _count(trace["confirmed_cursor"])
    if after > confirmed:
        raise ValueError("La consulta comienza después del cursor confirmado")
    relative = Path(trace["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("La ruta de trazas debe permanecer dentro de la ejecución")
    directory = run_path.parent / relative
    envelope = _json(directory / "trace-index.json")
    payload = envelope["payload"]
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != _digest(envelope["sha256"]):
        raise ValueError("El índice de trazas perdió su integridad")
    if payload["schema_version"] != 1 or payload["identity_sha256"] != _digest(
        trace["identity_sha256"]
    ):
        raise ValueError("La identidad de la traza no corresponde a la ejecución")
    if confirmed > _count(payload["cursor"]) or not isinstance(payload["shards"], list):
        raise ValueError("El cursor confirmado no existe en el índice de trazas")
    selected, last = [], 0
    for shard in payload["shards"]:
        first, end, rows = (_count(shard[name]) for name in ("first", "last", "rows"))
        if first != last + 1 or end < first or rows != end - first + 1 or rows > 128:
            raise ValueError("Los bloques de trazas contienen huecos o recuentos inválidos")
        last = end
        if end <= after or first > confirmed or len(selected) >= limit:
            continue
        digest = _digest(shard["sha256"])
        if shard["path"] != f"trace-{digest}.parquet":
            raise ValueError("La ruta del bloque no corresponde a su huella")
        data = _read(directory / shard["path"], MAX_SHARD)
        if len(data) != _count(shard["bytes"]) or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("El bloque de trazas perdió su integridad")
        parquet = pq.ParquetFile(
            pa.BufferReader(data),
            thrift_string_size_limit=MAX_SHARD,
            thrift_container_size_limit=8192,
            pre_buffer=False,
            page_checksum_verification=True,
        )
        if (
            parquet.metadata.num_rows != rows
            or parquet.num_row_groups != 1
            or parquet.metadata.num_columns != 48
            or not 0 <= parquet.metadata.row_group(0).total_byte_size <= 8 * MAX_SHARD
        ):
            raise ValueError("El bloque de trazas supera su presupuesto decodificado")
        metadata = parquet.schema_arrow.metadata or {}
        if (
            metadata.get(b"trace_identity_sha256") != trace["identity_sha256"].encode()
            or metadata.get(b"schema_version") != b"1"
            or any(parquet.metadata.row_group(0).column(index).file_path for index in range(48))
        ):
            raise ValueError("El bloque de trazas contiene otra identidad o una ruta externa")
        table = parquet.read(use_threads=False)
        if any(column.null_count for column in table.columns):
            raise ValueError("El bloque de trazas contiene valores NULL no admitidos")
        for offset, row in enumerate(table.to_pylist()):
            if row["decision_id"] != first + offset:
                raise ValueError("Los IDs de las decisiones no corresponden al índice")
            if after < row["decision_id"] <= confirmed and len(selected) < limit:
                selected.append(_describe(row))
    if last != payload["cursor"]:
        raise ValueError("El cursor del índice no corresponde al último bloque")
    intro = (
        f"Decisiones confirmadas hasta el ID {confirmed}. "
        "Las probabilidades describen la distribución de acciones. "
        "No estiman la probabilidad de rentabilidad."
    )
    return "\n\n".join([intro, *selected]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--after", type=int, default=0)
    args = parser.parse_args()
    result = explain(args.run, limit=args.limit, after=args.after)
    safe_destination(args.output)
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(result)


if __name__ == "__main__":
    main()
