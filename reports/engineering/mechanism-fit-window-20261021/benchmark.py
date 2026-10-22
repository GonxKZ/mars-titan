"""Coste de auditar ventanas, sin preparar matrices de ajuste ni resolver regresiones."""

import argparse
import hashlib
import json
import os
import platform
import resource
import runpy
import subprocess
import time
import tracemalloc
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "reports/engineering/mechanism-window-20261021/benchmark.py"
helper = runpy.run_path(str(HELPER))

from mars_titan.training.mechanism_fit_contract import MechanismFitConfig  # noqa: E402
from mars_titan.training.mechanism_fit_window import (  # noqa: E402
    MechanismFitWindow,
    audit_fit_windows,
    iter_fit_windows,
)

MODES = ("combined", "separate")


def summary(samples):
    return dict(
        samples_ms=samples,
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=float(np.percentile(samples, 99)),
        calls_per_second=1000 / float(np.mean(samples)),
    )


def content(requests, protocol):
    return [
        (packet.sha256, supervision.record(), metadata)
        for collections, _ in iter_fit_windows(requests, config=protocol)
        for packet, supervision, metadata in collections
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("Se necesita una salida nueva en un directorio existente")
    if not args.storage_root.is_dir():
        raise ValueError("Falta el directorio externo para los fixtures")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0 para producir el fixture, sin alternativa CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = helper["load_native"](os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    moment = helper["moment"]
    report = dict(
        status="running_not_complete",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        scope="read_only_fit_window_audit",
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        cpu_threads=torch.get_num_threads(),
        producer_device="cuda:0",
        producer_dtype="torch.float64",
        audit_device="CPU_JSON_and_native_files",
        tf32=False,
        autocast=False,
        real_corpus=False,
        shared_optimizer_steps=0,
        fit_policy_executed=False,
        protected_cohort_reader_executed=False,
        normalization_executed=False,
        native_sha256=native.binary_sha256,
        energy_measured=False,
        cases=[],
    )

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    save()
    started = time.perf_counter()
    for rows in (4, 64, 256):
        source = helper["flow_source"](
            args.storage_root / f"source-{rows}", flows=tuple(f"US/T{i:04d}" for i in range(rows))
        )
        engine = helper["engines"](source, "cuda:0", online=True)[3]
        setup = time.perf_counter()
        with helper["FinancialSession"](
            args.storage_root / f"session-{rows}",
            native=native,
            consumer=engine,
            codec=source["codec"],
            prefixes=source["prefixes"],
            retention=helper["RetentionConfig"](capacity=256, frontier=8, new_candidates=8),
            phase=helper["FinancialPhase"](
                "validation", moment(125), moment(126), moment(132), moment(133)
            ),
            admission="m1",
            block_rows=64,
            world="fit_window_audit_fixture",
            fold="0",
            mechanism_collection=helper["MechanismCollectionConfig"](),
        ) as run:
            pending = []
            for day in (126, 127, 128, 130):
                emitted = run.step(source["batches"][day], pending)
                pending = [
                    native.Feedback(p.id, 0, moment(day + 1), 0.125) for p in emitted.predictions
                ]
            run.step([], pending, kind="settlement", cutoff=moment(133), close_phase=True)
            setup_seconds = time.perf_counter() - setup
            for count, through, before in ((1, 2, 128), (2, 3, 129), (4, 5, 134)):
                generations = tuple(range(1, count + 1))
                protocol = MechanismFitConfig(
                    parent_consumer_id=engine.model_id,
                    parent_selected_before=moment(125),
                    fit_start=moment(126),
                    fit_end=moment(before),
                    ridge=0.01,
                    interval_weight=0.0,
                    min_gain=0.0,
                    market_weights=(("US", 1.0),),
                    costs=(0.0,) * 36,
                    min_cohorts=2,
                    source_mode="native_windows",
                )
                requests = dict(
                    combined=(MechanismFitWindow(run, generations, through_generation=through),),
                    separate=tuple(
                        MechanismFitWindow(run, (g,), through_generation=through)
                        for g in generations
                    ),
                )
                assert content(requests["combined"], protocol) == content(
                    requests["separate"], protocol
                )
                head = run.snapshot()
                rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state(0)
                calls = {
                    mode: partial(audit_fit_windows, requests[mode], config=protocol)
                    for mode in MODES
                }
                checks = {mode: call() for mode, call in calls.items()}
                for key in (
                    "observed_cohorts",
                    "mature_rows",
                    "pending_rows",
                    "excluded_rows",
                    "censored_rows",
                ):
                    assert checks["combined"][key] == checks["separate"][key]
                assert checks["combined"]["mature_rows"] == rows * count
                samples, peaks = ({mode: [] for mode in MODES} for _ in range(2))
                for repetition in range(23):
                    for mode in MODES if repetition % 2 == 0 else MODES[::-1]:
                        torch.cuda.synchronize(0)
                        torch.cuda.reset_peak_memory_stats(0)
                        tick = time.perf_counter_ns()
                        result = calls[mode]()
                        torch.cuda.synchronize(0)
                        elapsed = (time.perf_counter_ns() - tick) / 1e6
                        assert result == checks[mode]
                        if repetition >= 3:
                            samples[mode].append(elapsed)
                            peaks[mode].append(torch.cuda.max_memory_allocated(0))
                        del result
                python_peaks = {}
                for mode in MODES:
                    tracemalloc.start()
                    resident = tracemalloc.get_traced_memory()[0]
                    result = calls[mode]()
                    python_peaks[mode] = tracemalloc.get_traced_memory()[1] - resident
                    tracemalloc.stop()
                    del result
                assert run.snapshot() == head
                assert torch.equal(torch.random.get_rng_state(), rng) and torch.equal(
                    torch.cuda.get_rng_state(0), gpu_rng
                )
                report["cases"].append(
                    dict(
                        rows=rows,
                        collections=count,
                        through_generation=through,
                        before=moment(before),
                        protocol=protocol.identity(),
                        protocol_sha256=protocol.fingerprint(),
                        input_spec_id=source["spec"].fingerprint(),
                        parameter_id=engine.predictor._parameter_id,
                        supervision_origin=source["label_origin"],
                        setup_seconds=setup_seconds,
                        warmup=3,
                        repetitions=20,
                        cohort_counts={
                            k: v for k, v in checks["combined"].items() if k != "sources"
                        },
                        versions={
                            mode: dict(
                                **summary(samples[mode]),
                                source_bytes=checks[mode]["source_bytes"],
                                python_additional_peak_bytes=python_peaks[mode],
                                allocated_peak_bytes=max(peaks[mode]),
                                allocated_resident_bytes=torch.cuda.memory_allocated(0),
                                observed_cohorts_per_second=count
                                * 1000
                                / float(np.mean(samples[mode])),
                            )
                            for mode in MODES
                        },
                    )
                )
                save()
                del requests, calls, checks, samples
        del run, engine, source
        torch.cuda.empty_cache()
    paths = [
        Path(__file__),
        HELPER,
        ROOT / "uv.lock",
        ROOT / "tests/memory/test_financial_session_controls.py",
        ROOT / "tests/memory/test_financial_mechanisms.py",
    ]
    paths.extend(
        ROOT / ("src/mars_titan/training/" + name + ".py")
        for name in ("mechanism_fit_contract", "mechanism_fit_window", "mechanism_policy_fit")
    )
    paths.extend(
        ROOT / ("src/mars_titan/memory/" + name + ".py")
        for name in ("mechanism_collection_asof", "financial_session")
    )
    report.update(
        status="complete",
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    )
    save()


if __name__ == "__main__":
    main()
