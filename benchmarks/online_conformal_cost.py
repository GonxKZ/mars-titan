"""Medir el coste en CPU de la calibración conformal en línea (PT2) frente a la CQR estática.

Los cuantiles y objetivos son sintéticos con semilla fija y solo fijan las formas: un año
de evaluación de 252 sesiones, dos mercados, tres meses de calibración y el número de
activos de cada caso. No se ajusta, selecciona ni evalúa nada con estos valores. Se mide
el tiempo de arranque, la réplica completa, la latencia de cada emisión y maduración, la
memoria reservada por NumPy durante la réplica y el tamaño del estado exportado.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.calibration.conformal_quantiles import (
    apply_conformal_quantiles,
    fit_conformal_quantiles,
)
from mars_titan.calibration.online_conformal import OnlineConformal, replay_online_conformal
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.quantile_head import LEVELS

NOMINALS = (0.8, 0.95)
MARKETS = ("CN", "US")
SESSIONS = 252
CALIBRATION_SESSIONS = 63
# Activos por mercado y horizonte de madurez en sesiones. 128 es la comparación principal
# propuesta y 4.200 la mayor población US de una sesión en la edición desde 2000.
CASES = {"main_h1": (128, 1), "main_h5": (128, 5), "population_h1": (4200, 1)}
SECOND_US = 1_000_000
DAY_US = 86_400 * SECOND_US
START_US = 1_546_300_800_000_000  # 2019-01-01, dentro del periodo de desarrollo


def synthetic(assets, sessions, horizon, seed, offset=0):
    rng = np.random.default_rng(seed)
    rows = assets * sessions * len(MARKETS)
    groups = np.repeat(np.array(MARKETS), assets * sessions)
    session = np.tile(np.repeat(np.arange(sessions), assets), len(MARKETS))
    at = START_US + (offset + session) * DAY_US
    center = rng.normal(0, 0.01, rows)
    spread = np.abs(rng.normal(0.02, 0.005, rows))[:, None]
    quantiles = center[:, None] + spread * np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
    target = center + rng.standard_t(4, rows) * 0.02
    return quantiles, groups, at.astype(np.int64), target, (at + horizon * DAY_US)


def timed(function, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - started)
    return result, dict(
        median_s=statistics.median(samples),
        min_s=min(samples),
        max_s=max(samples),
        repeats=repeats,
    )


def percentiles(samples):
    values = np.array(samples)
    return {f"p{q}_us": float(np.percentile(values, q) * 1e6) for q in (50, 95, 99)} | dict(
        count=len(samples)
    )


def instrumented(calibrator):
    """Envolver emit y mature para medir cada llamada sin cambiar su resultado."""
    latencies = dict(emit=[], mature=[])
    for name in latencies:
        original = getattr(calibrator, name)

        def wrapper(*args, _original=original, _name=name, **kwargs):
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            latencies[_name].append(time.perf_counter() - started)
            return result

        setattr(calibrator, name, wrapper)
    return latencies


def measure_case(assets, horizon, repeats, rate_fraction):
    calibration = synthetic(assets, CALIBRATION_SESSIONS, horizon, seed=1, offset=-100)
    panel = synthetic(assets, SESSIONS, horizon, seed=2)
    quantiles, groups, at, target, matures = panel
    options = dict(levels=LEVELS, nominals=NOMINALS, min_rows=1000)

    def start():
        return OnlineConformal.start(
            calibration[3], calibration[0], calibration[1], rate_fraction=rate_fraction, **options
        )

    _, start_cost = timed(start, repeats)
    record, static_fit_cost = timed(
        lambda: fit_conformal_quantiles(calibration[3], calibration[0], calibration[1], **options),
        repeats,
    )
    _, static_apply_cost = timed(
        lambda: apply_conformal_quantiles(record, quantiles, groups), repeats
    )
    _, replay_cost = timed(
        lambda: replay_online_conformal(start(), quantiles, groups, at, target, matures), repeats
    )
    calibrator = start()
    latencies = instrumented(calibrator)
    tracemalloc.start()
    emitted, emissions, updates = replay_online_conformal(
        calibrator, quantiles, groups, at, target, matures
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload, export_cost = timed(calibrator.export, repeats)
    _, restore_cost = timed(lambda: OnlineConformal.restore(payload), repeats)
    pending_bytes = sum(
        cohort["quantiles"].nbytes
        for market in payload["markets"].values()
        for cohort in market["pending"]
    )
    return dict(
        assets_per_market=assets,
        markets=len(MARKETS),
        sessions=SESSIONS,
        calibration_rows=len(calibration[1]),
        evaluation_rows=len(groups),
        horizon_sessions=horizon,
        cohorts_emitted=len(emissions),
        cohorts_matured=len(updates),
        start=start_cost,
        static_fit=static_fit_cost,
        static_apply=static_apply_cost,
        online_replay=replay_cost,
        online_over_static_apply=replay_cost["median_s"] / static_apply_cost["median_s"],
        emit_latency=percentiles(latencies["emit"]),
        mature_latency=percentiles(latencies["mature"]),
        replay_numpy_peak_bytes=peak,
        export=export_cost,
        restore=restore_cost,
        pending_quantile_bytes_at_end=pending_bytes,
        median_unchanged=bool(np.array_equal(emitted[:, 2], quantiles[:, 2])),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rate-fraction", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    cpu = cpu_name()
    cases = {
        name: measure_case(assets, horizon, args.repeats, args.rate_fraction)
        for name, (assets, horizon) in CASES.items()
    }
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/calibration/online_conformal.py",
            "src/mars_titan/calibration/conformal_quantiles.py",
            "benchmarks/online_conformal_cost.py",
        )
    }
    receipt = dict(
        schema_version=1,
        kind="online_conformal_cost",
        issue=454,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        rate_fraction=args.rate_fraction,
        environment=dict(
            cpu=cpu,
            cpu_count=os.cpu_count(),
            threads=dict(
                omp=os.environ.get("OMP_NUM_THREADS"),
                mkl=os.environ.get("MKL_NUM_THREADS"),
                openblas=os.environ.get("OPENBLAS_NUM_THREADS"),
            ),
            python=platform.python_version(),
            numpy=np.__version__,
            load_average_at_start=os.getloadavg(),
            concurrent_load="otras sesiones comparten la máquina, ver load_average_at_start",
        ),
        not_measured=[
            "GPU, porque el componente no tiene tensores ni parámetros entrenables",
            "energía, sin instrumento adecuado",
            "integración con la comparación por ventanas, todavía pendiente",
        ],
        cases=cases,
    )
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({name: case["online_replay"] for name, case in cases.items()}, indent=1))


if __name__ == "__main__":
    main()
