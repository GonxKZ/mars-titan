"""Medir el contrato de comparación por sesión con un panel sintético de tamaño real.

Los datos son sintéticos y no representan resultados de mercado. Solo sirven para
medir tiempo y memoria de las métricas y del remuestreo con formas realistas.
"""

import argparse
import json
import os
import platform
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import forecast_scores
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions, selective_risk
from mars_titan.evaluation.paired_comparisons import compare_series, delta, interaction

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)


def synthetic(rows, days, seed):
    rng = np.random.default_rng(seed)
    per_session = rows // (2 * days)
    day = np.repeat(np.arange(days), 2 * per_session)
    market = np.tile(np.repeat(np.array(["US", "CN"]), per_session), days)
    hours = np.where(market == "US", 20, 7).astype("timedelta64[h]")
    start = np.datetime64("2000-01-03", "us")
    times = start + day.astype("timedelta64[D]") + hours
    count = len(day)
    identities = pa.array([f"{name}/{i:07d}" for i, name in enumerate(market)])
    target = rng.standard_t(4, count) * 0.02
    models = {}
    for name, strength in (("B", 0.10), ("C", 0.12), ("M", 0.11), ("CM", 0.13)):
        point = strength * target + rng.normal(0, 0.02, count)
        width = np.abs(rng.normal(0.02, 0.005, count))
        quantiles = point[:, None] + width[:, None] * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        models[name] = (point, quantiles)
    shuffle = rng.permutation(count)
    return identities, market, times, target, models, shuffle


def measure(function, repetitions):
    elapsed, result = [], None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        elapsed.append(time.perf_counter() - start)
    return result, dict(median_s=statistics.median(elapsed), min_s=min(elapsed), runs_s=elapsed)


def traced(function):
    tracemalloc.start()
    try:
        function()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--days", type=int, default=2_500)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    identities, market, times, target, models, shuffle = synthetic(args.rows, args.days, 7)
    point, quantiles = models["B"]

    def panel():
        return ForecastPanel.from_columns(
            identities.take(pa.array(shuffle)),
            market[shuffle],
            times[shuffle],
            target[shuffle],
            point[shuffle],
            quantiles=quantiles[shuffle],
            levels=LEVELS,
        )

    stages = {}
    built, stages["panel"] = measure(panel, args.repetitions)

    def digest():
        built.__dict__.pop("cohort_sha256", None)
        return built.cohort_sha256

    _, stages["cohort_digest"] = measure(digest, args.repetitions)
    # Alternativa descartada y ruta elegida para ordenar dentro de cada sesión.
    _, stages["order_lexsort"] = measure(
        lambda: np.lexsort((built.target, built.session)), args.repetitions
    )
    _, stages["order_two_stable_passes"] = measure(
        lambda: forecast_scores._within_session_order(built, built.target), args.repetitions
    )
    scores, stages["session_scores"] = measure(lambda: score_sessions(built), args.repetitions)
    _, stages["summary"] = measure(lambda: scores.summary(), args.repetitions)
    _, stages["selective_risk"] = measure(lambda: selective_risk(built).summary(), args.repetitions)
    series = {"B": scores.series("mae")}
    for name in ("C", "M", "CM"):
        other = ForecastPanel.from_columns(identities, market, times, target, models[name][0])
        series[name] = score_sessions(other).series("mae")
    contrasts = {
        "C": delta("B", "C"),
        "M": delta("B", "M"),
        "CM": delta("B", "CM"),
        "I": interaction("B", "C", "M", "CM"),
    }
    block = round(len(np.unique(series["B"].session_period)) ** (1 / 3))
    _, stages["paired_contrasts"] = measure(
        lambda: compare_series(
            series,
            contrasts,
            block_length=block,
            replicates=args.replicates,
            seed=42,
            sensitivity_block_lengths=(5, 10, 40),
        ),
        args.repetitions,
    )
    memory = dict(
        panel_peak_bytes=traced(panel),
        session_scores_peak_bytes=traced(lambda: score_sessions(built)),
        selective_risk_peak_bytes=traced(lambda: selective_risk(built)),
    )
    report = dict(
        schema_version=1,
        kind="forecast_metrics_benchmark",
        measured_at=datetime.now(UTC).isoformat(),
        synthetic=True,
        shape=dict(
            rows=built.rows,
            sessions=built.sessions,
            days=built.periods,
            levels=list(LEVELS),
            models=4,
            contrasts=len(contrasts),
            replicates=args.replicates,
            block_length=block,
            sensitivity_block_lengths=[5, 10, 40],
        ),
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pyarrow=pa.__version__,
            machine=platform.machine(),
            cpu_count=os.cpu_count(),
            load_average=os.getloadavg(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        ),
        stages=stages,
        traced_memory=memory,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        atomic_json(args.output, report)
    print(text)


if __name__ == "__main__":
    main()
