"""Comparar vistas nativas individuales y conjuntas con fixtures, sin aprender."""

import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import tempfile
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]

from test_financial_mechanisms import engines  # noqa: E402
from test_financial_session import moment  # noqa: E402
from test_financial_session_controls import flow_source  # noqa: E402

from mars_titan.memory.financial_session import (  # noqa: E402
    FinancialPhase,
    FinancialSession,
    _check_callbacks,
)
from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig  # noqa: E402
from mars_titan.memory.retention_bank import RetentionConfig  # noqa: E402

REFERENCE_COMMIT = "ba6395cedaa7117bffb9cef395a3ea695032e820"
REFERENCE_SHA256 = "7709cd573ad327c8579674704a06164d1ff365d6d9ecafc33fa035b5a69103dd"
MODES = ("legacy_individual", "individual", "window")


def read(run, mode, generations, through, reference):
    options = dict(through_generation=through, before=moment(134))
    if mode == "window":
        return run.mechanism_collections_at(generations, **options)
    if mode == "individual":
        return [run.mechanism_collection_at(generation, **options) for generation in generations]
    results = []
    for generation in generations:
        # El envoltorio anterior comprobaba el consumidor y la cabeza en cada llamada.
        _check_callbacks(run)
        run.consumer.verify()
        before = run.snapshot()
        result = reference.collection_at(
            run,
            run._bundle(before["state"]),
            generation,
            max_records=256,
            max_bytes=64 * 1024**2,
            **options,
        )
        if run.snapshot() != before:
            raise RuntimeError("La sesión cambió durante la referencia histórica")
        results.append(result)
    return results


def content(result, mode):
    rows = result["collections"] if mode == "window" else result
    return [{key: row[key] for key in ("collection", "supervision", "censored")} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("La salida necesita una ruta nueva en un directorio existente")
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("La referencia no conserva la fuente histórica fijada")
    spec = importlib.util.spec_from_file_location(
        "mars_titan.memory._asof_reference", args.reference
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0, sin alternativa CPU implícita")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = load_native()
    report = dict(
        status="running_not_complete",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        reference_commit=REFERENCE_COMMIT,
        reference_source_sha256=REFERENCE_SHA256,
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        numpy=np.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        device="cuda:0",
        producer_dtype="torch.float64",
        reader_compute="cpu_json_and_native_io",
        cpu_threads=torch.get_num_threads(),
        real_corpus=False,
        optimizer_steps=0,
        warmup=3,
        repetitions=20,
        cases=[],
        native_binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "src/mars_titan/memory/mechanism_collection_asof.py",
                ROOT / "src/mars_titan/memory/financial_session.py",
                ROOT / "tests/memory/test_financial_session_controls.py",
                ROOT / "tests/memory/test_financial_mechanisms.py",
            )
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mars-window-cost-") as folder:
        root = Path(folder)
        for count in (4, 64, 256):
            source = flow_source(
                root / f"source-{count}", flows=tuple(f"US/T{i:04d}" for i in range(count))
            )
            engine = engines(source, "cuda:0", online=True)[3]
            for collections in (1, 2, 4):
                setup = time.perf_counter()
                with FinancialSession(
                    root / f"session-{count}-{collections}",
                    native=native,
                    consumer=engine,
                    codec=source["codec"],
                    prefixes=source["prefixes"],
                    retention=RetentionConfig(capacity=256, frontier=8, new_candidates=8),
                    phase=FinancialPhase(
                        "validation", moment(125), moment(126), moment(132), moment(133)
                    ),
                    admission="m1",
                    block_rows=64,
                    world="window_cost_fixture",
                    fold="0",
                    mechanism_collection=MechanismCollectionConfig(),
                ) as run:
                    pending = []
                    for day in (126, 127, 128, 130)[:collections]:
                        issued = run.step(source["batches"][day], pending)
                        pending = [
                            native.Feedback(p.id, 0, moment(day + 1), 0.125)
                            for p in issued.predictions
                        ]
                    run.step([], pending, kind="settlement", cutoff=moment(133), close_phase=True)
                    setup_seconds = time.perf_counter() - setup
                    generations, through = tuple(range(1, collections + 1)), collections + 1
                    expected = read(run, "legacy_individual", generations, through, reference)
                    expected_content = content(expected, "legacy_individual")
                    before = run.snapshot()
                    rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state(0)
                    samples = {mode: [] for mode in MODES}
                    peak = {mode: 0 for mode in MODES}
                    source_bytes = {}
                    for repetition in range(23):
                        order = MODES[repetition % 3 :] + MODES[: repetition % 3]
                        for mode in order:
                            torch.cuda.synchronize(0)
                            torch.cuda.reset_peak_memory_stats(0)
                            tick = time.perf_counter_ns()
                            result = read(run, mode, generations, through, reference)
                            torch.cuda.synchronize(0)
                            elapsed = (time.perf_counter_ns() - tick) / 1e6
                            if content(result, mode) != expected_content:
                                raise RuntimeError("La ventana cambió las filas o sus resultados")
                            if mode == "individual" and result != expected:
                                raise RuntimeError(
                                    "La API individual cambió el formato o su huella"
                                )
                            peak[mode] = max(peak[mode], torch.cuda.max_memory_allocated(0))
                            source_bytes[mode] = (
                                result["as_of"]["source_bytes"]
                                if mode == "window"
                                else sum(row["as_of"]["source_bytes"] for row in result)
                            )
                            if repetition >= 3:
                                samples[mode].append(elapsed)
                    memory = {}
                    for mode in MODES:
                        # Asignaciones medidas aparte, sin instrumentar las latencias anteriores.
                        tracemalloc.start()
                        trace_base = tracemalloc.get_traced_memory()[0]
                        result = read(run, mode, generations, through, reference)
                        live, trace_peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        memory[mode] = dict(
                            python_retained_bytes=live - trace_base,
                            python_peak_bytes=trace_peak - trace_base,
                        )
                        del result
                    if (
                        run.snapshot() != before
                        or not torch.equal(torch.random.get_rng_state(), rng)
                        or not torch.equal(torch.cuda.get_rng_state(0), gpu_rng)
                    ):
                        raise RuntimeError("La lectura cambió el estado confirmado o el RNG")
                    for mode, values in samples.items():
                        report["cases"].append(
                            dict(
                                rows=count,
                                collections=collections,
                                mode=mode,
                                samples_ms=values,
                                p50_ms=float(np.percentile(values, 50)),
                                p95_ms=float(np.percentile(values, 95)),
                                p99_ms=float(np.percentile(values, 99)),
                                collections_per_second=collections * 1000 / float(np.mean(values)),
                                packet_and_native_bytes=source_bytes[mode],
                                setup_seconds=setup_seconds,
                                allocated_resident_bytes=torch.cuda.memory_allocated(0),
                                allocated_peak_bytes=peak[mode],
                                **memory[mode],
                                provenance=dict(
                                    contract_id=run.contract_id,
                                    model_id=run.model_id,
                                    consumer_id=engine.model_id,
                                    parameter_id=engine.predictor._parameter_id,
                                    input_spec_id=source["spec"].fingerprint(),
                                    collection_sha256=[
                                        row["collection"]["sha256"] for row in expected
                                    ],
                                    native_records=expected[0]["as_of"]["native_records"],
                                    supervision_origin=source["label_origin"],
                                ),
                            )
                        )
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                    print(
                        json.dumps(dict(rows=count, collections=collections, complete=True)),
                        flush=True,
                    )
    report.update(
        status="complete",
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
