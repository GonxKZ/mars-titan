"""Coste CPU del historial con escritura sellada y poda de fixtures, sin publicar sesiones."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import resource
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.memory.mechanism_history import MechanismHistory, MechanismHistoryConfig
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.session_artifacts import SessionArtifacts


def event(generation, rows, *, payload="varied"):
    generator = np.random.default_rng(7000 + generation)
    centres = generator.normal(0, 0.01, rows)
    widths = generator.uniform(0.001, 0.03, (rows, 4))
    records = []
    for index in range(rows):
        centre = float(centres[index]) if payload == "varied" else 0.0
        a, b, c, d = widths[index] if payload == "varied" else (1.0, 1.0, 1.0, 1.0)
        raw = [centre - a - b, centre - a, centre, centre + c, centre + c + d]
        values = [float(value) for value in raw]
        calibrated = (
            [
                float(value)
                for value in (raw[0] - 0.01, raw[1] - 0.01, centre, raw[3] + 0.01, raw[4] + 0.01)
            ]
            if payload == "varied"
            else [-3.0, -2.0, 0.0, 2.0, 3.0]
        )
        record = dict(
            flow_id=f"US/A{index:03d}",
            sample_id=f"US/A{index:03d}/{generation}",
            point=centre,
            raw_quantiles=values,
            quantiles=calibrated,
            requested="parent/k4cr/online",
            executed="parent/k4cr/online",
            reader_steps=4,
            searches=2,
            discarded_steps=0,
            reasons=[],
        )
        if payload == "varied":
            record["input_sha256"] = generator.bytes(32).hex()
        records.append(record)
    return dict(
        schema_version=1,
        model_id="a" * 64,
        generation=generation,
        cutoff=1_609_459_200_000_000 + generation,
        kind="decision",
        native_record_sha256="c" * 64,
        payload=dict(rows=records),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--implementation", choices=("current", "dictionary"), default="current")
    parser.add_argument("--compression", choices=("none", "zlib1"))
    parser.add_argument("--payload", choices=("constant", "varied"), default="varied")
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 2 or args.warmup < 1:
        raise ValueError("La salida debe ser nueva y las repeticiones deben ser válidas")
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        raise RuntimeError("Falta el enlace nativo para medir la escritura real")
    native = load_native(path)
    if args.implementation == "dictionary":
        if args.compression is not None:
            raise ValueError("La referencia de diccionarios no dispone de compresión por evento")
        specification = importlib.util.spec_from_file_location(
            "mars_titan.memory.history_dictionary_reference",
            Path(__file__).with_name("dictionary_reference.py"),
        )
        reference = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = reference
        specification.loader.exec_module(reference)
        archive_type, config_type = reference.MechanismHistory, reference.MechanismHistoryConfig
    else:
        archive_type, config_type = MechanismHistory, MechanismHistoryConfig
    config = (
        config_type() if args.compression is None else config_type(compression=args.compression)
    )
    started = time.perf_counter()
    result = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        device="cpu",
        real_data_used=False,
        optimizer_steps=0,
        scientific_study=False,
        native_sha256=native.binary_sha256,
        python=platform.python_version(),
        numpy=np.__version__,
        warmup=args.warmup,
        repetitions=args.repetitions,
        implementation=args.implementation,
        payload=args.payload,
        cases=[],
    )
    with tempfile.TemporaryDirectory(prefix="mars-mechanism-history-fixture-") as folder:
        for count in (8, 64):
            for rows in (64, 256):
                archive = archive_type(
                    SessionArtifacts(native, Path(folder) / f"{count}-{rows}"),
                    identity="b" * 64,
                    model_id="a" * 64,
                    config=config,
                )
                manifest, previous = archive.empty(), archive.empty()
                samples = []
                for index in range(1, count + args.warmup + args.repetitions + 1):
                    current = event(index, rows, payload=args.payload)
                    before = time.perf_counter_ns()
                    proposal = archive.append(manifest, current)
                    archive.artifacts.prune_unreferenced(
                        archive.live_references(proposal) + archive.live_references(manifest)
                    )
                    elapsed = (time.perf_counter_ns() - before) / 1e6
                    previous, manifest = manifest, proposal
                    if index > count + args.warmup:
                        samples.append(elapsed)
                assert archive.read_event(manifest, 1) == event(1, rows, payload=args.payload)
                audit = time.perf_counter_ns()
                archive.verify(manifest, strong=True)
                audit_ms = (time.perf_counter_ns() - audit) / 1e6
                digest = hashlib.sha256()
                for item in archive.iter_events(manifest):
                    digest.update(
                        json.dumps(
                            item, sort_keys=True, separators=(",", ":"), allow_nan=False
                        ).encode()
                    )
                size = sum(p.stat().st_size for p in archive.artifacts._inventory())
                result["cases"].append(
                    dict(
                        prior_events=count,
                        rows_per_event=rows,
                        final_events=manifest["events"],
                        chunks=len(manifest["chunks"]),
                        history_bytes=manifest["bytes"],
                        retained_artifact_bytes=size,
                        raw_samples_ms=samples,
                        mean_ms=float(np.mean(samples)),
                        std_ms=float(np.std(samples, ddof=1)),
                        **{f"p{p}_ms": float(np.percentile(samples, p)) for p in (50, 95, 99)},
                        rows_per_second=rows * 1000 / float(np.mean(samples)),
                        strong_recovery_ms=audit_ms,
                        semantic_sha256=digest.hexdigest(),
                        configuration=archive.config.identity(),
                    )
                )
                assert previous["events"] + 1 == manifest["events"]
    result["internal_seconds"] = time.perf_counter() - started
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["scope"] = (
        "Incluye empaquetado, verificación incremental, serialización, sello nativo y poda "
        "de propuestas "
        "de dos generaciones. Cada paso añade un evento distinto y escribe contenido nuevo. "
        "El historial crece durante las repeticiones. La recuperación fuerte y el digest semántico "
        "quedan fuera de cada latencia y dentro del proceso. No hay modelo, GPU, publicación de "
        "FinancialSession ni métricas financieras. Datos y carpetas son fixtures eliminados "
        "al terminar. RSS corresponde a todo el proceso. Sin energía medida y con percentiles "
        "empíricos de pocas muestras."
    )
    result["sources"] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            str(Path(inspect.getfile(archive_type)).resolve().relative_to(Path.cwd())),
            "src/mars_titan/memory/session_artifacts.py",
            str(Path(__file__).relative_to(Path.cwd())),
        )
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(result["cases"]), seconds=result["internal_seconds"])))


if __name__ == "__main__":
    main()
