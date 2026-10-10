"""Medir el coste en CPU de los contrastes de capacidad predictiva a escala de la campaña.

Las pérdidas son sintéticas con semilla fija y solo fijan las formas: los días de decisión
reales de US (2005 a 2023) y de CN (2011 a 2023), los brazos y familias de la comparación
conjunta declarada y la longitud de bloque, réplicas y semilla de esa comparación. No se
ajusta, selecciona ni evalúa ningún modelo con estos valores. Se mide la sección completa
de las tres vistas del ámbito conjunto, el tiempo de cada análisis y la memoria reservada
por NumPy, y se comprueba que dos repeticiones dan el mismo informe.
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
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation import predictive_ability as pa
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import SessionSeries

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/historical-masked-2000-joint-comparison.json"
# Años de evaluación de las ventanas declaradas: 19 en US y 13 en CN.
SPANS = {"US": ("2005-01-01", "2023-12-31"), "CN": ("2011-01-01", "2023-12-31")}
TIMED = (
    "diebold_mariano",
    "superior_predictive_ability",
    "model_confidence_set",
    "block_length_diagnostic",
)


def sessions():
    """Mercado y día UTC de cada sesión con los calendarios reales, en orden canónico."""
    markets, days = [], []
    for code, market in enumerate(("US", "CN")):
        clock = MarketClock(market, *SPANS[market])
        markets.append(np.full(len(clock.decisions), code, dtype=np.int64))
        days.append(np.array([moment.date().toordinal() for moment in clock.decisions]))
    market, day = np.concatenate(markets), np.concatenate(days)
    _, period = np.unique(day, return_inverse=True)
    return market, period.astype(np.int64)


def arm_series(arms, market, period, seed):
    """Una SessionSeries de MAE por brazo con un desplazamiento propio y ruido común."""
    rng = np.random.default_rng(seed)
    common = rng.gamma(2.0, 0.004, len(market))
    result = {}
    for index, arm in enumerate(arms):
        noise = rng.normal(0.0, 0.0015, len(market))
        values = np.abs(common + noise + 0.0001 * index)
        result[arm] = SessionSeries(
            "mae",
            True,
            "synthetic",
            ("US", "CN"),
            market,
            period,
            values,
            np.ones(len(market), dtype=bool),
        )
    return result


def views(series, market):
    """Vista conjunta y una por mercado, con los días renumerados como en la comparación."""
    result = {"US+CN": series}
    for code, name in enumerate(("US", "CN")):
        mask = market == code
        _, period = np.unique(next(iter(series.values())).session_period[mask], return_inverse=True)
        result[name] = {
            arm: SessionSeries(
                item.metric,
                item.loss,
                f"synthetic-{name}",
                item.markets,
                item.session_market[mask],
                period.astype(np.int64),
                item.values[mask],
                item.defined[mask],
            )
            for arm, item in series.items()
        }
    return result


@contextmanager
def timers(totals):
    """Acumular el tiempo de cada análisis sin cambiar su resultado."""
    originals = {name: getattr(pa, name) for name in TIMED}

    def wrap(name, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                totals[name] = totals.get(name, 0.0) + time.perf_counter() - started

        return timed

    for name, function in originals.items():
        setattr(pa, name, wrap(name, function))
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(pa, name, function)


def measure(repeats):
    config = walk.load_config(CONFIG)
    scope = config["resolved_scopes"]["US+CN"]
    market, period = sessions()
    by_view = views(arm_series(list(scope["arms"]), market, period, 20261010), market)
    section, comparison = config[walk.PREDICTIVE_FIELD], config["comparison"]

    def run():
        return pa.report(
            section,
            comparison,
            config["metrics"]["market_weighting"],
            scope["families"],
            list(by_view),
            lambda arm, view, metric: by_view[view][arm],
        )

    runs = []
    for _ in range(repeats):
        totals = {}
        started = time.perf_counter()
        with timers(totals):
            report = run()
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        runs.append(dict(elapsed_s=elapsed, by_analysis_s=totals, report_sha256=digest))
    # La memoria se mide en una ejecución aparte, porque tracemalloc ralentiza NumPy.
    tracemalloc.start()
    run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(
        families=len(scope["families"]),
        arms=len(scope["arms"]),
        days={view: rows["mae"]["levels"]["days"] for view, rows in report["views"].items()},
        sessions=int(len(market)),
        block_length=comparison["block_length"],
        replicates=comparison["replicates"],
        runs=runs,
        median_elapsed_s=statistics.median(run["elapsed_s"] for run in runs),
        numpy_peak_bytes=peak,
        identical_reports=len({run["report_sha256"] for run in runs}) == 1,
    )


def cpu_name():
    """Modelo de la CPU de /proc/cpuinfo y, si no aparece, la arquitectura."""
    info = Path("/proc/cpuinfo")
    lines = info.read_text().splitlines() if info.is_file() else []
    names = (line.split(":", 1)[1].strip() for line in lines if line.startswith("model name"))
    return next(names, platform.processor() or platform.machine())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    load = os.getloadavg()
    result = measure(args.repeats)
    sources = (
        "src/mars_titan/evaluation/predictive_ability.py",
        "benchmarks/predictive_ability_cost.py",
        "configs/evaluation/historical-masked-2000-joint-comparison.json",
    )
    record = dict(
        schema_version=1,
        kind="predictive_ability_cost",
        issue=493,
        measured_at=started,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        sources_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        data="pérdidas sintéticas con semilla fija sobre calendarios reales, solo fijan formas",
        environment=dict(
            cpu=cpu_name(),
            cpu_count=os.cpu_count(),
            threads={
                name: os.environ.get(f"{name.upper()}_NUM_THREADS")
                for name in ("omp", "mkl", "openblas")
            },
            python=platform.python_version(),
            numpy=np.__version__,
            arch=version("arch"),
            statsmodels=version("statsmodels"),
            load_average_at_start=list(load),
            concurrent_load="otras sesiones comparten la máquina, ver load_average_at_start",
        ),
        not_measured=[
            "GPU, porque los contrastes solo usan NumPy en CPU",
            "energía, sin instrumento adecuado",
            "lectura y puntuación de predicciones, que mide benchmark_evaluation_scale",
        ],
        **result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(median_elapsed_s=record["median_elapsed_s"], days=record["days"])))


if __name__ == "__main__":
    main()
