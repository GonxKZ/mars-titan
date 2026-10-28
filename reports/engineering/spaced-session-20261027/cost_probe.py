"""Coste durable E2 con fuentes sintéticas, sin ajustar ni evaluar un modelo financiero."""

import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

import mars_titan
from mars_titan.memory.native_backend import load_native

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]
from test_financial_session import moment  # noqa: E402
from test_financial_session_controls import (  # noqa: E402
    flow_source,
    no_target_estimation,
    paired_consumers,
    tensor_leaves,
)
from test_spaced_session import feedback, open_session  # noqa: E402


def signature(run, points):
    records = [
        {k: v for k, v in r.items() if k not in ("model_id", "prediction_id")}
        for r in run.retained_episodes()
    ]
    tensors = {
        key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
        for key, value in tensor_leaves(run.fast_state()).items()
    }
    state = run.snapshot()
    counters = {k: state[k] for k in ("observed", "issued", "applied", "excluded", "finalized")}
    return hashlib.sha256(
        json.dumps([points, records, tensors, counters], sort_keys=True).encode()
    ).hexdigest()


def trial(native, source, consumer, directory, policy):
    run = open_session(
        native,
        source,
        consumer,
        directory,
        policy=policy,
        block_rows=64,
        capacity=min(1024, 4 * len(source["batches"][125])),
    )
    points, previous = [], []
    started = time.perf_counter()
    try:
        for at in (125, 126, 127, 128):
            result = run.step(source["batches"][at], feedback(native, previous, at))
            previous = result.predictions
            points.extend((p.asset, p.decision_at, p.value) for p in previous)
        snapshot = run.snapshot()
        ledger = run.spaced_state() if policy else None
        run.close()
        run = open_session(
            native,
            source,
            consumer,
            directory,
            policy=policy,
            block_rows=64,
            capacity=min(1024, 4 * len(source["batches"][125])),
            resume=True,
        )
        assert run.snapshot() == snapshot
        if policy:
            assert run.spaced_state() == ledger
        run.step([], feedback(native, previous, 129), kind="settlement", cutoff=moment(129))
        following = run.step(source["batches"][130], [])
        points.extend((p.asset, p.decision_at, p.value) for p in following.predictions)
        elapsed = time.perf_counter() - started
        common = signature(run, points)
        retained_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        return dict(
            seconds=elapsed,
            common_state_sha256=common,
            retained_bytes=retained_bytes,
            recovery_equal=True,
        )
    finally:
        run.close()


def main():
    assert Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    assert not torch.cuda.is_initialized()
    native = load_native()
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    guard = no_target_estimation.__wrapped__()
    next(guard)
    started, rows = time.perf_counter(), []
    try:
        for assets in (64, 256):
            source = flow_source(
                out / f"fixture-{assets}", flows=tuple(f"US/F{i:04d}" for i in range(assets))
            )
            consumer = paired_consumers(source)["disabled"]
            for repetition in range(4):
                pair = []
                for active in (False, True) if repetition % 2 == 0 else (True, False):
                    result = trial(
                        native,
                        source,
                        consumer,
                        out / f"run-{assets}-{repetition}-{active}",
                        "reservoir" if active else None,
                    )
                    result.update(
                        assets=assets, active=active, repetition=repetition, warmup=repetition == 0
                    )
                    pair.append(result)
                    rows.append(result)
                assert pair[0]["common_state_sha256"] == pair[1]["common_state_sha256"]
                report = dict(
                    kind="technical_durable_session_cost",
                    native_sha256=native.binary_sha256,
                    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cpu_threads=2,
                    dtype="torch.float64",
                    cases=rows,
                    process_seconds_without_imports=time.perf_counter() - started,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    limits=[
                        "Fixtures compactos, etiquetas manuales y ningún ajuste.",
                        "Incluye cuatro decisiones, cierre, recuperación, "
                        "settlement y siguiente decisión.",
                        "Fuentes fuera del cronómetro. Incluye almacenamiento y comprobaciones.",
                        "Tres muestras por modo no estiman colas p95/p99 con precisión.",
                        "Sin GPU. Energía no medida.",
                        "RSS absoluto del proceso completo. No representa el incremento de E2.",
                    ],
                )
                (out / "cost.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                )
    finally:
        guard.close()
    print(
        json.dumps(
            [
                {
                    "assets": assets,
                    "active": active,
                    "median_seconds": statistics.median(
                        r["seconds"]
                        for r in rows
                        if r["assets"] == assets and r["active"] == active and not r["warmup"]
                    ),
                }
                for assets in (64, 256)
                for active in (False, True)
            ]
        )
    )


if __name__ == "__main__":
    main()
