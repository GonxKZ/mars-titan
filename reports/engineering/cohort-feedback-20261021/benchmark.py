"""Coste de propuestas PT2 en CPU con cohortes de prueba, sin aprendizaje."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.calibration.cohort_feedback import CohortFeedback, CohortOutcome
from mars_titan.calibration.cohort_feedback_state import packed_calibrator
from mars_titan.calibration.online_conformal import OnlineConformal
from mars_titan.models.quantile_head import LEVELS


def calibrator():
    y = np.tile(np.arange(-9.0, 10.0), 2)
    return OnlineConformal.start(
        y,
        np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], (38, 1)),
        np.repeat(np.array(["US", "CN"]), 19),
        levels=LEVELS,
        nominals=(0.8, 0.95),
        min_rows=1,
        rate_fraction=0.25,
    )


def fixture(rows, backlog):
    base = calibrator()
    state = CohortFeedback.start(base)
    names = tuple(f"asset{i:04d}" for i in range(rows))
    q = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], (rows, 1))
    for i in range(backlog):
        state = state.emit("US", 10 + i, names, q, (100 + i,) * rows).state
        base.emit("US", 10 + i, q, 100 + i)
    return state, base, names, q, np.linspace(-20.0, 20.0, rows)


def trajectory(fixture_, mode):
    state, base, names, q, y = fixture_
    times = {}
    started = time.perf_counter_ns()
    before = started
    emitted = state.emit("US", 20, names, q, (110,) * len(names))
    times["emit_ms"] = (time.perf_counter_ns() - before) / 1e6
    before = time.perf_counter_ns()
    if mode == "cancel":
        events = [
            CohortOutcome("US", 10, names[0], 100, status="excluded", evidence_sha256="a" * 64)
        ]
    else:
        events = [
            CohortOutcome("US", 10, name, 100, float(value))
            for name, value in zip(names, y, strict=True)
        ]
    stop = len(events) // 2 if mode == "split" else len(events)
    received = emitted.state.feedback(events[:stop], at=100)
    times["first_feedback_ms"] = (time.perf_counter_ns() - before) / 1e6
    before = time.perf_counter_ns()
    if mode == "split":
        received = received.state.feedback(events[stop:], at=101)
    times["second_feedback_ms"] = (time.perf_counter_ns() - before) / 1e6
    before = time.perf_counter_ns()
    payload = received.state.export()
    times["export_ms"] = (time.perf_counter_ns() - before) / 1e6
    times["total_ms"] = (time.perf_counter_ns() - started) / 1e6
    return times, payload, received.settlements


def check_reference(base, names, q, y, mode, payload, settlements):
    reference = OnlineConformal.restore(base.export())
    reference.emit("US", 20, q, 110)
    if mode == "cancel":
        reference.cancel("US", 10, at=100)
    else:
        reference.mature("US", 10, y, at=101 if mode == "split" else 100)
    assert payload["calibrator"] == packed_calibrator(reference)
    assert settlements[0]["status"] == ("cancelled" if mode == "cancel" else "matured")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 2 or args.warmup < 0:
        raise ValueError("Salida nueva y repeticiones válidas obligatorias")
    started = time.perf_counter()
    definitions = [
        (n, b, m) for n in (64, 256) for b in (1, 4) for m in ("complete", "split", "cancel")
    ]
    fixtures = {(n, b): fixture(n, b) for n, b, _ in definitions}
    digests = {key: value[0].state_sha256() for key, value in fixtures.items()}
    samples = {key: [] for key in definitions}
    sizes = {}
    for repetition in range(args.warmup + args.repetitions):
        offset = repetition % len(definitions)
        for key in definitions[offset:] + definitions[:offset]:
            n, b, mode = key
            current = fixtures[n, b]
            times, payload, settlements = trajectory(current, mode)
            check_reference(
                current[1], current[2], current[3], current[4], mode, payload, settlements
            )
            assert CohortFeedback.restore(payload).state_sha256() == payload["state_sha256"]
            if repetition >= args.warmup:
                samples[key].append(times)
                sizes[key] = len(
                    json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                )
    assert all(value[0].state_sha256() == digests[key] for key, value in fixtures.items())
    cases = []
    for key, records in samples.items():
        latency = {}
        for name in records[0]:
            values = np.array([item[name] for item in records])
            latency[name] = dict(
                mean=float(values.mean()),
                std=float(values.std(ddof=1)),
                **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
            )
        cases.append(
            dict(
                rows=key[0],
                pending_cohorts_before_emission=key[1],
                mode=key[2],
                latencies_ms=latency,
                serialized_state_bytes=sizes[key],
                raw_samples=records,
            )
        )
    code = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path("src/mars_titan/calibration/cohort_feedback.py"),
            Path("src/mars_titan/calibration/cohort_feedback_state.py"),
            Path("src/mars_titan/calibration/online_conformal.py"),
        ]
    }
    result = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        kind="cohort_feedback_cpu_fixture_cost_v1",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources=code,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tracked_worktree_clean=subprocess.run(["git", "diff", "--quiet"], check=False).returncode
        == 0,
        python=platform.python_version(),
        numpy=np.__version__,
        processor=platform.processor(),
        machine=platform.machine(),
        device="cpu",
        warmup=args.warmup,
        repetitions=args.repetitions,
        trajectory_count=len(definitions) * (args.warmup + args.repetitions),
        cases=cases,
        reference_parity=True,
        initial_snapshots_unchanged=True,
        optimizer_steps=0,
        real_data_used=False,
        final_test_opened=False,
        internal_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        scope=(
            "Incluye emisión propuesta, recepción con registros tipados y exportación. "
            "Los tiempos parciales están dentro del total. La referencia, restauración de "
            "auditoría e inicialización quedan fuera de cada trayectoria y dentro del tiempo "
            "de proceso. RSS incluye imports y todas las condiciones. Los percentiles son "
            "empíricos, sin garantía de cola. No mide predictor, CUDA ni publicación "
            "en disco."
        ),
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                cases=len(cases),
                seconds=result["internal_seconds"],
                rss=result["process_peak_rss_bytes"],
            )
        )
    )


if __name__ == "__main__":
    main()
