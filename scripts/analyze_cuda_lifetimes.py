"""Cruzar snapshots y pilas de Compute Sanitizer sin deducir propiedad por tamaños."""

import argparse
import hashlib
import json
import re
from pathlib import Path

LEAK = re.compile(r"Leaked (\d+) bytes at (0x[0-9a-fA-F]+)")
TOTAL = re.compile(r"LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocations")
ERRORS = re.compile(r"ERROR SUMMARY: (\d+) errors")
MAX_LOG_BYTES = 16 * 1024**2


def _address(value):
    if not isinstance(value, str) or not re.fullmatch(r"0[xX][0-9a-fA-F]+|0", value):
        raise ValueError("La dirección del diagnóstico no es hexadecimal")
    return int(value, 0)


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Los bytes del diagnóstico deben ser enteros no negativos")
    return value


def _segments(snapshot):
    segments, blocks, workspace_addresses = {}, {}, set()
    totals = dict(allocated_bytes=0, active_bytes=0, reserved_bytes=0)
    for segment in snapshot["segments"]:
        start, size = _address(segment["address"]), _count(segment["bytes"])
        if any(start < end and begin < start + size for begin, end in segments):
            raise ValueError("Los segmentos se solapan")
        segments[start, start + size] = segment
        offset = allocated = active = 0
        for block in segment["blocks"]:
            address, length = _address(block["address"]), _count(block["bytes"])
            if (
                address != start + offset
                or length == 0
                or type(block["allocated"]) is not bool
                or type(block["active"]) is not bool
                or (block["allocated"] and not block["active"])
            ):
                raise ValueError("Los bloques no forman un segmento coherente")
            blocks[address] = block
            offset += length
            allocated += length if block["allocated"] else 0
            active += length if block["active"] else 0
        if (offset, allocated, active) != (
            size,
            _count(segment["allocated_bytes"]),
            _count(segment["active_bytes"]),
        ):
            raise ValueError("Los bytes de los bloques y del segmento difieren")
        totals["reserved_bytes"] += size
        totals["allocated_bytes"] += allocated
        totals["active_bytes"] += active
    if any(value != _count(snapshot[key]) for key, value in totals.items()):
        raise ValueError("Los segmentos y las estadísticas del asignador difieren")
    for workspace in snapshot["cublas_workspaces"] + snapshot["cublaslt_workspaces"]:
        address, size = _address(workspace["address"]), _count(workspace["bytes"])
        if not address and not size:
            continue
        block = blocks.get(address)
        if block is None or not block["allocated"] or size > block["bytes"]:
            raise ValueError("El espacio de trabajo no corresponde a un bloque asignado")
        workspace_addresses.add(address)
    unattributed = sum(
        block["bytes"]
        for address, block in blocks.items()
        if block["active"] and address not in workspace_addresses
    )
    return segments, workspace_addresses, unattributed


def _ownership(record, segments, workspace_addresses, host_segments):
    frames = record.pop("frames")
    symbols = {
        line.split("Host Frame:", 1)[1].strip().split(" [", 1)[0].split(" in ", 1)[0]
        for line in frames
    }
    host = bool(symbols & {"cuMemHostAlloc", "cudaMallocHost", "cudaHostAlloc"})
    device = bool(symbols & {"cuMemAlloc_v2", "cuMemAllocAsync", "cudaMalloc", "cudaMallocAsync"})
    record["memory_space"] = "host_pinned" if host else "device" if device else "unspecified"
    start = _address(record["address"])
    segment = None if host else segments.get((start, start + record["bytes"]))
    host_segment = host_segments.get((start, start + record["bytes"])) if host else None
    if host_segment is not None:
        category = (
            "torch_host_active_unattributed" if host_segment["active"] else "torch_host_cache"
        )
        record["evidence"] = "same_process_host_segment_address_and_size"
    elif segment is not None:
        record["memory_space"] = "device"
        record["evidence"] = "same_process_segment_address_and_size"
        if not segment["active_bytes"]:
            category = "torch_inactive_cache"
        elif all(
            not block["active"] or _address(block["address"]) in workspace_addresses
            for block in segment["blocks"]
        ):
            category = "torch_cublas_workspace"
        else:
            category = "torch_active_unattributed"
    elif symbols & {"cublasCreate", "cublasCreate_v2", "cublasLtCreate"}:
        category = "cublas_handle_creation"
        record["evidence"] = "allocation_backtrace"
    elif "cudnnCreate" in symbols:
        category = "cudnn_handle_creation"
        record["evidence"] = "allocation_backtrace"
    else:
        category = "unattributed"
        record["evidence"] = "insufficient"
    record["category"] = category
    return record


def _host_segments(snapshot):
    segments = {}
    for segment in snapshot.get("host_segments", []):
        address, size = _address(segment["address"]), _count(segment["bytes"])
        if (
            size == 0
            or any(address < end and begin < address + size for begin, end in segments)
            or type(segment["allocated"]) is not bool
            or type(segment["active"]) is not bool
            or (segment["allocated"] and not segment["active"])
        ):
            raise ValueError("El snapshot de memoria host no es coherente")
        segments[address, address + size] = segment
    return segments


def _growth(repeated):
    columns = {
        name: [] for name in ("allocated", "active", "reserved", "host_active", "host_reserved")
    }
    for state in repeated:
        for name in ("allocated", "active", "reserved"):
            columns[name].append(state[name + "_bytes"])
        host = _host_segments(state).values()
        columns["host_reserved"].append(sum(item["bytes"] for item in host))
        columns["host_active"].append(sum(item["bytes"] for item in host if item["active"]))
    return dict(
        **{
            name + "_growth_bytes": values[-1] - values[0] if values else None
            for name, values in columns.items()
        },
        stable_after_first=all(len(set(values)) <= 1 for values in columns.values())
        if repeated
        else None,
    )


def _validate_completion(configurations, repeated, completed):
    if len(configurations) > 1:
        raise ValueError("Se necesita un recorrido de un único proceso")
    if not configurations:
        return  # Logs anteriores a la introducción del registro de configuración.
    config = configurations[0]
    count = config["repetitions"]
    if config.get("schema_version") != 1 or type(count) is not int or not 1 <= count <= 300:
        raise ValueError("La versión o el número de repeticiones del diagnóstico no son válidos")
    if completed != [True] or [state["iteration"] for state in repeated] != list(
        range(1, count + 1)
    ):
        raise ValueError("El diagnóstico no confirmó todas sus repeticiones y su final")


def analyze(text):
    if len(text.encode("utf-8")) > MAX_LOG_BYTES:
        raise ValueError("El log supera el límite de 16 MiB")
    snapshots, allocations, summaries, errors, configurations = [], [], [], [], []
    completed = []
    for line in text.splitlines():
        if line.startswith("{"):
            entry = json.loads(line)
            kind = entry.get("record")
            if kind == "allocation_snapshot":
                _segments(entry)
                _host_segments(entry)
                snapshots.append(entry)
            elif kind == "configuration":
                configurations.append(entry)
            elif kind == "completed":
                completed.append(entry.get("rng_unchanged") is True)
        elif match := LEAK.search(line):
            allocations.append(dict(bytes=int(match[1]), address=match[2], frames=[]))
        elif match := TOTAL.search(line):
            summaries.append(tuple(map(int, match.groups())))
        elif match := ERRORS.search(line):
            errors.append(int(match[1]))
        elif "Host Frame:" in line and allocations:
            allocations[-1]["frames"].append(line)
    if not snapshots:
        raise ValueError("Se necesita un recorrido de un único proceso con snapshots")
    repeated = [state for state in snapshots if state["phase"] == "after_scope"]
    _validate_completion(configurations, repeated, completed)
    observed = bool(allocations or summaries or errors)
    retained = sum(item["bytes"] for item in allocations)
    if observed and (
        summaries != [(retained, len(allocations))]
        or len(errors) != 1
        or errors[0] < len(allocations)
    ):
        raise ValueError("El informe de Compute Sanitizer está truncado o no concilia")
    segments, references, unattributed = _segments(snapshots[-1])
    host_segments = _host_segments(snapshots[-1])
    return dict(
        schema_version=1,
        kind="cuda_lifetime_attribution",
        memcheck_observed=observed,
        memcheck_errors=errors[0] if observed else None,
        non_leak_errors=errors[0] - len(allocations) if observed else None,
        retained_bytes=retained if observed else None,
        allocations=[_ownership(item, segments, references, host_segments) for item in allocations],
        final_phase=snapshots[-1]["phase"],
        unattributed_active_bytes=unattributed,
        host_active_bytes=sum(item["bytes"] for item in host_segments.values() if item["active"]),
        repetitions=len(repeated),
        **_growth(repeated),
        snapshot_count=len(snapshots),
        completion_record_observed=completed == [True],
        limits="La pila atribuye la creación en el proveedor. No acredita la liberación de todos "
        "sus recursos ni sustituye la comprobación de errores de acceso.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.log.stat().st_size > MAX_LOG_BYTES:
        raise ValueError("El log supera el límite de 16 MiB")
    content = args.log.read_bytes()
    report = analyze(content.decode("utf-8"))
    report["log_sha256"] = hashlib.sha256(content).hexdigest()
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
