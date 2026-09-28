"""Medir procesos PPO de diagnóstico con fuentes sintéticas fijas y sin CUDA."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def files(directory):
    result = subprocess.run(
        ["rg", "--files", "--hidden", "--no-ignore", str(directory)],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return [Path(line) for line in result.stdout.splitlines()]


def synthetic_source(directory, partition, seed, sessions):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.simulation.market import MarketTape
    from mars_titan.simulation.storage import write_tape

    rng = np.random.default_rng(seed)
    assets = 8
    returns = np.clip(rng.normal(0, 0.004, (sessions, assets)), -0.01, 0.01)
    prices = np.empty((sessions, assets, 5), dtype=np.float64)
    previous = np.linspace(80, 120, assets)
    for at in range(sessions):
        opening = previous
        closing = opening * (1 + returns[at])
        prices[at, :, 0] = opening
        prices[at, :, 1] = np.maximum(opening, closing) * 1.001
        prices[at, :, 2] = np.minimum(opening, closing) * 0.999
        prices[at, :, 3] = closing
        prices[at, :, 4] = 100000
        previous = closing
    origin = 946684800000000 if partition == "train" else 1672531200000000
    times = origin + np.arange(sessions, dtype=np.int64) * 86400000000 + 57600000000
    scores = rng.normal(0.003, 0.002, (sessions, assets))
    tape = MarketTape(
        prices,
        times,
        [f"FIC{index:02d}" for index in range(assets)],
        scores,
        domain="synthetic",
        currency="USD",
        partition=partition,
        parent_id="diagnostic-score-tape-v1",
        source_identity={"generator": {"seed": seed, "name": "diagnostic-finite-tape-v1"}},
    )
    write_tape(tape, directory)
    context = rng.normal(0, 1, (sessions, 2)).astype(np.float32)
    table = pa.table(
        {
            "session": pa.array(np.repeat(np.arange(sessions), 2), type=pa.int32()),
            "feature": pa.array(np.tile(np.arange(2), sessions), type=pa.int32()),
            "value": pa.array(context.reshape(-1), type=pa.float32()),
            "present": pa.array(np.ones(sessions * 2, dtype=bool)),
            "available_at": pa.array(np.repeat(times, 2), type=pa.int64()),
        }
    )
    parquet = directory / "context.parquet"
    pq.write_table(table, parquet, row_group_size=16)
    save(
        directory / "context.json",
        {
            "schema_version": 1,
            "domain": "synthetic",
            "market_manifest_sha256": digest(directory / "manifest.json"),
            "fields": [
                {"name": "factor", "unit": "normalizado"},
                {"name": "event", "unit": "normalizado"},
            ],
            "file": {
                "path": "context.parquet",
                "sha256": digest(parquet),
                "bytes": parquet.stat().st_size,
            },
        },
    )
    return {
        "partition": partition,
        "generator_seed": seed,
        "sessions": sessions,
        "assets": assets,
        "context_fields": 2,
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(files(directory))
        ],
    }


def concurrent_snapshot():
    process = subprocess.run(
        ["ps", "-eo", "comm=,pcpu=,rss=", "--sort=-pcpu"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    top = []
    for line in process.stdout.splitlines()[:8]:
        name, cpu, rss = line.rsplit(maxsplit=2)
        if name.startswith(("clang-tidy", "clang-check")):
            category = "análisis estático"
        elif name.startswith(("clang", "cc1", "gcc", "g++", "ld", "cmake", "ninja")):
            category = "compilación nativa"
        elif name.startswith("python"):
            category = "procesos Python"
        elif name.startswith(("firefox", "chrome", "chromium")):
            category = "navegador"
        elif name.startswith(("kwin", "plasmashell", "Xorg")):
            category = "escritorio gráfico"
        else:
            category = "otros procesos"
        top.append(
            {"category": category, "lifetime_cpu_percent": float(cpu), "rss_bytes": int(rss) * 1024}
        )
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "load_average": list(os.getloadavg()),
        "top_processes": top,
    }


def gpu_snapshot():
    from mars_titan.training.gpu_supervisor import read_gpu

    try:
        snapshot = read_gpu()
        return {
            "observed_at": datetime.now(UTC).isoformat(),
            "total_mib": snapshot.total_mib,
            "free_mib": snapshot.free_mib,
            "compute_process_count": len(snapshot.compute_pids),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"observed_at": datetime.now(UTC).isoformat(), "unavailable": type(error).__name__}


def checkpoint_bytes(output):
    index = read(output / "ppo-index.json")["payload"]
    records = {record["bundle"]: record for record in index["recent"]}
    if index["best"] is not None:
        records[index["best"]["bundle"]] = index["best"]
    payload = dict(metadata=0, policy=0, rollout=0)
    for name, record in records.items():
        folder = output / name
        manifest_file = folder / "manifest.json"
        require(
            digest(manifest_file) == record["sha256"],
            "El manifiesto retenido no conserva su SHA256",
        )
        manifest = read(manifest_file)
        for key, filename in (
            ("metadata", "metadata.json"),
            ("policy", "policy.pt"),
            ("rollout", "rollout.pt"),
        ):
            item = folder / filename
            expected = manifest["files"][filename]
            require(
                item.stat().st_size == expected["bytes"] and digest(item) == expected["sha256"],
                "Un archivo retenido no conserva tamaño y SHA256",
            )
            payload[key] += item.stat().st_size
    actual = files(output)
    return {
        "retained_bundles": len(records),
        "recent_bundles": len(index["recent"]),
        "selected_best_present": index["best"] is not None,
        "retained_payload_bytes": payload,
        "output_file_bytes": sum(path.stat().st_size for path in actual),
        "output_allocated_bytes": sum(path.stat().st_blocks * 512 for path in actual),
        "output_file_count": len(actual),
    }


def summary(values):
    return {
        "count": len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "stdev_sample": statistics.stdev(values),
    }


def perf_probe(log_path):
    policy = Path("/proc/sys/kernel/perf_event_paranoid")
    result = {
        "returncode": None,
        "status": "unavailable",
        "events_requested": ["task-clock", "cycles", "instructions"],
        "perf_event_paranoid": policy.read_text().strip() if policy.is_file() else None,
        "reason": "perf no está instalado.",
    }
    executable = shutil.which("perf")
    if executable is None:
        log_path.write_text(result["reason"] + "\n")
        return result
    try:
        probe = subprocess.run(
            [executable, "stat", "-e", "task-clock,cycles,instructions", "--", "true"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        log_path.write_text(str(error) + "\n")
        result["reason"] = "No se pudo completar la consulta de permisos de perf."
        return result
    log_path.write_text(probe.stderr)
    result.update(
        returncode=probe.returncode,
        status="blocked" if probe.returncode else "available",
        reason="perf no pudo medir los eventos solicitados." if probe.returncode else None,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("La salida ya existe. Elige otro archivo para conservar sus resultados.")
    if args.output.resolve().is_relative_to(args.private.resolve()):
        parser.error("La salida debe quedar fuera del directorio privado del benchmark.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["LC_ALL"] = "C"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    deadline = time.monotonic() + 240
    args.private.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    binary_hash = digest(binary)
    backend = binary.with_name("libmars_titan_simulation.so")
    backend_hash = digest(backend)
    cache = {}
    for line in binary.with_name("CMakeCache.txt").read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("#", "//")):
            name, value = line.split("=", 1)
            cache[name.split(":", 1)[0]] = value
    require(
        cache["CMAKE_BUILD_TYPE"] == "Release" and cache["MARS_TITAN_SANITIZER"] == "none",
        "La medición requiere Release sin sanitizadores",
    )
    require(
        cache["MARS_TITAN_ENABLE_COVERAGE"] == "OFF" and cache["MARS_TITAN_PGO"] == "off",
        "La medición requiere cobertura y PGO desactivados",
    )
    train, validation = args.private / "train", args.private / "validation"
    sources = [
        synthetic_source(train, "train", 1701, 65),
        synthetic_source(validation, "validation", 2701, 9),
    ]
    base = read(args.root / "configs/simulation/native-ppo-diagnostic.json")
    require(
        base["training"]["total_transitions"] == 32
        and base["training"]["rollout_transitions"] == 16
        and base["training"]["workers"] == 1,
        "El protocolo requiere 32 transiciones, recorridos de 16 y un trabajador",
    )
    base["selection"]["early_stopping"] = False
    hardware = json.loads(subprocess.check_output(["lscpu", "-J"], text=True, timeout=5))["lscpu"]
    wanted = {
        "Architecture:",
        "CPU(s):",
        "Model name:",
        "Thread(s) per core:",
        "Core(s) per socket:",
        "Socket(s):",
        "NUMA node(s):",
    }
    hardware = {
        item["field"].removesuffix(":"): item["data"]
        for item in hardware
        if item["field"] in wanted
    }
    perf = perf_probe(args.private / "perf-admission.log")
    report = {
        "schema_version": 1,
        "status": "running",
        "measured_at": datetime.now(UTC).isoformat(),
        "domain": "technical",
        "device": "cpu",
        "diagnostic": True,
        "binary_sha256": binary_hash,
        "backend_sha256": backend_hash,
        "build": {
            "type": cache["CMAKE_BUILD_TYPE"],
            "sanitizer": cache["MARS_TITAN_SANITIZER"],
            "coverage": cache["MARS_TITAN_ENABLE_COVERAGE"],
            "pgo": cache["MARS_TITAN_PGO"],
        },
        "harness_sha256": digest(Path(__file__)),
        "platform": platform.platform(),
        "hardware": hardware,
        "software": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "pyarrow")
        },
        "protocol": {
            "environments": [1, 2, 4, 8],
            "policy_seeds": [42, 43, 44],
            "training_transitions": 32,
            "rollout_transitions": 16,
            "workers": 1,
            "torch_threads": 1,
            "repetitions_per_case": 5,
            "warmup_runs_per_case": 1,
            "base_configuration": base,
            "order_seed": 20260928,
            "validation_fixed": True,
            "environment": {
                "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        },
        "sources": sources,
        "source_definition": {
            "kind": "fixture_synthetic",
            "market_rows": (
                "OHLC diarios de ocho activos con retornos normales de desviación 0,004, "
                "recortados a ±0,01. Volumen fijo de 100000."
            ),
            "scores": (
                "Valores normales de media 0,003 y desviación 0,002, congelados en la cinta. "
                "No proceden de un modelo entrenado."
            ),
            "context": "Dos valores normales por sesión, disponibles al cierre y sin ausencias.",
        },
        "measurement": {
            "process_seconds": (
                "perf_counter alrededor del lanzamiento y communicate con pipes de stdout "
                "y stderr. Los logs se escriben después de terminar el intervalo medido."
            ),
            "internal_seconds": (
                "invocation_seconds del recibo nativo, desde run_ppo_experiment "
                "hasta antes de publicar el recibo final."
            ),
            "rss": "VmHWM de procfs leído por el ejecutable nativo.",
            "warmup": (
                "Un proceso completo descartado por cada combinación N y semilla. "
                "Calienta la caché de archivos, cada repetición sigue arrancando un proceso nuevo."
            ),
            "bytes": (
                "Tamaños reales de archivos retenidos. output_allocated_bytes suma "
                "st_blocks multiplicado por 512."
            ),
        },
        "perf": perf,
        "gpu_observations": [],
        "warmups": [],
        "records": [],
        "summaries": [],
        "limits": [
            "Son ejecuciones técnicas de 32 transiciones de entrenamiento. "
            "No miden calidad de aprendizaje ni rendimiento científico o CUDA.",
            "Cambiar N cambia el horizonte temporal del recorrido y las trayectorias, "
            "aunque se conserven fuentes, muestras globales y pasos de Adam. "
            "No se interpreta como aceleración de un trabajo numéricamente equivalente.",
            "La validación realiza ocho transiciones por evaluación y permanece fija. "
            "Sus transiciones se registran aparte del presupuesto de entrenamiento.",
            "Cinco repeticiones por caso describen dispersión local, "
            "no latencias de cola estables.",
            "No se vaciaron cachés, fijaron afinidades, alteraron permisos "
            "o detuvieron aplicaciones ajenas.",
            "La CPU compartió el equipo con los procesos observados. Los porcentajes de ps "
            "son medias de vida del proceso, no uso durante la ejecución medida.",
            "No se midieron energía, coste monetario, asignaciones, bytes de E/S físicos "
            "ni transferencias CPU/GPU.",
        ],
    }
    report["software"]["python"] = platform.python_version()
    report["software"]["compiler"] = subprocess.check_output(
        [cache["CMAKE_CXX_COMPILER"], "--version"], text=True, timeout=5
    ).splitlines()[0]
    start = time.monotonic()
    order = random.Random(20260928)
    ordinal = 0
    for seed in (42, 43, 44):
        report["gpu_observations"].append(gpu_snapshot())
        for repetition in range(6):
            environments = [1, 2, 4, 8]
            order.shuffle(environments)
            for lanes in environments:
                require(
                    time.monotonic() < deadline,
                    "El benchmark excede su límite total de cuatro minutos",
                )
                require(
                    digest(binary) == binary_hash and digest(backend) == backend_hash,
                    "Cambió el ejecutable durante la medición",
                )
                config = json.loads(json.dumps(base))
                config["training"]["seed"] = seed
                config["environments"] = lanes
                name = f"n{lanes}-s{seed}-r{repetition}"
                config_path = args.private / f"{name}.json"
                save(config_path, config)
                output = args.private / name
                command = [
                    str(binary),
                    "--config",
                    str(config_path),
                    "--train-tape",
                    str(train),
                    "--validation-tape",
                    str(validation),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                    "--diagnostic",
                ]
                concurrent = concurrent_snapshot()
                remaining = deadline - time.monotonic()
                require(remaining > 0, "El benchmark agotó su presupuesto de cuatro minutos")
                measured = time.perf_counter()
                result = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=min(20, remaining),
                    env={**os.environ, "LC_ALL": "C"},
                )
                elapsed = time.perf_counter() - measured
                require(
                    len(result.stdout) + len(result.stderr) <= 65536,
                    "El proceso excedió el presupuesto de logs",
                )
                (args.private / f"{name}.stdout").write_bytes(result.stdout)
                (args.private / f"{name}.stderr").write_bytes(result.stderr)
                require(result.returncode == 0, "El proceso de diagnóstico falló: " + name)
                record = read(output / "run.json")
                identity = read(output / "identity.json")["identity"]
                if "native_source_sha256" not in report["build"]:
                    report["build"].update(
                        {
                            key: identity[key]
                            for key in (
                                "native_source_sha256",
                                "native_build_sha256",
                                "torch_version",
                            )
                        }
                    )
                require(
                    record["status"] == "completed" and record["transitions"] == 32,
                    "El proceso no completó el presupuesto de diagnóstico",
                )
                require(
                    record["optimizer_steps"] == 8 and record["invalid_transitions"] == 0,
                    "El proceso no conserva los pasos de Adam o contiene transiciones inválidas",
                )
                require(
                    record["evaluations"] == 3 and record["partial_ticks"] == 0,
                    "El proceso no completó las evaluaciones y los recorridos previstos",
                )
                require(
                    record["resources"]["ram_peak_method"] == "procfs_VmHWM"
                    and record["resources"]["ram_peak_bytes"] > 0,
                    "Falta la medición de memoria VmHWM",
                )
                require(
                    digest(binary) == binary_hash and digest(backend) == backend_hash,
                    "Cambió el ejecutable durante la medición",
                )
                measured_record = {
                    "ordinal": ordinal,
                    "environments": lanes,
                    "seed": seed,
                    "repetition": repetition,
                    "warmup": repetition == 0,
                    "process_seconds": elapsed,
                    "internal_seconds": record["invocation_seconds"],
                    "training_transitions_per_process_second": record["transitions"] / elapsed,
                    "peak_rss_bytes": record["resources"]["ram_peak_bytes"],
                    "rss_method": record["resources"]["ram_peak_method"],
                    "training_transitions": record["transitions"],
                    "optimizer_steps": record["optimizer_steps"],
                    "evaluations": record["evaluations"],
                    "validation_transitions": 8 * record["evaluations"],
                    "invalid_transitions": record["invalid_transitions"],
                    "identity_sha256": record["identity_sha256"],
                    "concurrent_workload": concurrent,
                    **checkpoint_bytes(output),
                }
                require(
                    measured_record["output_file_bytes"] < 5 * 1024**2,
                    "Los archivos de una ejecución superan el presupuesto de 5 MiB",
                )
                report["warmups" if repetition == 0 else "records"].append(measured_record)
                ordinal += 1
                save(args.private / "progress.json", report)
                print(
                    f"N={lanes} semilla={seed} repetición={repetition} "
                    f"proceso={elapsed:.6f}s interno={record['invocation_seconds']:.6f}s",
                    flush=True,
                )
    report["gpu_observations"].append(gpu_snapshot())
    for seed in (42, 43, 44):
        for lanes in (1, 2, 4, 8):
            records = [
                record
                for record in report["records"]
                if record["seed"] == seed and record["environments"] == lanes
            ]
            report["summaries"].append(
                {
                    "environments": lanes,
                    "seed": seed,
                    **{
                        key: summary([record[key] for record in records])
                        for key in (
                            "process_seconds",
                            "internal_seconds",
                            "training_transitions_per_process_second",
                            "peak_rss_bytes",
                            "output_file_bytes",
                            "output_allocated_bytes",
                        )
                    },
                    "retained_bundle_counts": sorted(
                        {record["retained_bundles"] for record in records}
                    ),
                }
            )
    report["benchmark_wall_seconds"] = time.monotonic() - start
    report["status"] = "completed"
    report["finished_at"] = datetime.now(UTC).isoformat()
    save(args.private / "result.json", report)
    save(args.output, report)
    print(
        json.dumps(
            {
                "records": len(report["records"]),
                "warmups": len(report["warmups"]),
                "seconds": report["benchmark_wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
