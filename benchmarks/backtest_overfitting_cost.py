"""Medir en CPU el PBO por CSCV y el Sharpe deflactado de la cartera larga y corta (#794).

Los rendimientos son normales con semilla fija y solo fijan las formas. La CSCV se mide con
1.248 sesiones en 16 bloques de 78 y con 8, 30, 128 y 1.024 estrategias, el máximo que
admite el módulo. Cada caso calienta una vez, repite la CSCV y compara el Sharpe del elegido
dentro y fuera en 40 mitades con el cálculo directo sobre las filas, para comprobar que la
combinación de momentos por bloques conserva la referencia. El análisis completo se mide
sobre una carpeta sintética de la cartera con dos mercados, dos costes y tres semillas, que
incluye leer las sesiones y agruparlas. Nada se ajusta ni se selecciona.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy

from benchmarks.output_stage_cost import peak_bytes, timed
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import backtest_overfitting as bo
from mars_titan.hardware.platform_identity import cpu_name

DECLARATION = Path("configs/evaluation/backtest-overfitting-v1.json")
SESSIONS = 1248
BLOCKS = 16
CHECKED_HALVES = 40
# Estrategias y repeticiones de cada caso de la CSCV.
CSCV_CASES = {
    "strategies_8": (8, 5),
    "strategies_30": (30, 5),
    "strategies_128": (128, 3),
    "strategies_1024": (1024, 3),
}
# Modelos del análisis completo, con cinco sesiones antiguas que el recorte descarta. 23 son
# los de la comparación de la campaña A.
ANALYSIS_CASES = {"models_8": 8, "models_23": 23}
SEEDS = (1, 2, 3)
COSTS = (0, 10)


def returns(sessions, strategies, seed):
    return np.random.default_rng(seed).standard_normal((sessions, strategies)) * 0.01 + 2e-4


def direct_sharpe(values, inside):
    """Sharpe de una serie en los bloques marcados, recorriendo sus filas."""
    part = values.reshape(len(inside), -1)[inside].ravel()
    return part.mean() / part.std(ddof=1)


def reference_error(values, result):
    """Máxima diferencia relativa frente al cálculo directo en mitades repartidas."""
    combinations = bo.halves(BLOCKS)
    worst = 0.0
    for index in np.linspace(0, len(combinations) - 1, CHECKED_HALVES).astype(int):
        inside = np.zeros(BLOCKS, dtype=bool)
        inside[combinations[index]] = True
        column = values[:, result["chosen"][index]]
        for estimate, mask in (
            (result["in_sample"][index], inside),
            (result["out_of_sample"][index], ~inside),
        ):
            expected = direct_sharpe(column, mask)
            worst = max(worst, abs(estimate / expected - 1))
    return worst


def measure_cscv(strategies, repeats):
    values = returns(SESSIONS, strategies, seed=strategies)
    bo.pbo_cscv(values, blocks=BLOCKS)
    result, cost = timed(lambda: bo.pbo_cscv(values, blocks=BLOCKS), repeats)
    return dict(
        sessions=SESSIONS,
        blocks=BLOCKS,
        strategies=strategies,
        combinations=result["combinations"],
        cscv=cost,
        traced_peak_bytes=peak_bytes(lambda: bo.pbo_cscv(values, blocks=BLOCKS)),
        checked_halves=CHECKED_HALVES,
        max_relative_error_vs_rows=reference_error(values, result),
    )


def portfolio(folder, models):
    """Carpeta con el informe y las sesiones de una cartera terminada, sin modelos reales."""
    sessions = SESSIONS + 5
    times = pa.array(
        [1_546_380_000_000_000 + day * 86_400_000_000 for day in range(sessions)],
        pa.timestamp("us", tz="UTC"),
    )
    arms = [f"model_{index:02d}" for index in range(models)]
    tables = []
    for code, market in enumerate(("US", "CN")):
        for index, arm in enumerate(arms):
            for seed in SEEDS:
                net = returns(sessions, len(COSTS), seed=10_000 * code + 10 * index + seed)
                columns = dict(
                    arm=[arm] * sessions,
                    seed=pa.array([seed] * sessions, pa.int64()),
                    market=[market] * sessions,
                    prediction_at=times,
                )
                columns |= {f"net_return_{cost}bps": net[:, i] for i, cost in enumerate(COSTS)}
                tables.append(pa.table(columns))
    folder.mkdir()
    pq.write_table(pa.concat_tables(tables), folder / "sessions.parquet", compression="zstd")
    report = dict(
        kind=bo.SOURCE_KIND,
        status="completed",
        final_test_opened=False,
        markets=["US", "CN"],
        declaration=dict(cost_bps_per_side=list(COSTS)),
        views={market: dict(arms={arm: {} for arm in arms}) for market in ("US", "CN")},
        artifacts={"sessions.parquet": sha256(folder / "sessions.parquet")},
    )
    atomic_json(folder / "long_short.json", report)
    return len(arms) * len(SEEDS) * 2 * sessions


def measure_analysis(models, scratch):
    folder = Path(scratch) / f"long-short-{models}"
    rows = portfolio(folder, models)
    bo.analyze_long_short(folder, DECLARATION)
    report, cost = timed(lambda: bo.analyze_long_short(folder, DECLARATION), 3)
    first = report["markets"]["US"]["0"]
    return dict(
        models=models,
        seeds=len(SEEDS),
        markets=2,
        costs=list(COSTS),
        session_rows=rows,
        sessions_per_market=first["sessions"],
        trimmed_earliest_sessions=first["trimmed_earliest_sessions"],
        analysis=cost,
        traced_peak_bytes=peak_bytes(lambda: bo.analyze_long_short(folder, DECLARATION)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("La salida debe ser nueva")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    load = os.getloadavg()
    cscv = {name: measure_cscv(*shape) for name, shape in CSCV_CASES.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Las carpetas sintéticas van junto a la salida y no en /tmp, que es memoria.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        analysis = {
            name: measure_analysis(models, scratch) for name, models in ANALYSIS_CASES.items()
        }
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/evaluation/backtest_overfitting.py",
            "benchmarks/backtest_overfitting_cost.py",
            str(DECLARATION),
        )
    }
    receipt = dict(
        schema_version=1,
        kind="backtest_overfitting_cost",
        issue=794,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="rendimientos normales con semilla fija, solo para fijar formas. Nada se ajusta",
        environment=dict(
            cpu=cpu_name(),
            cpu_count=os.cpu_count(),
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            pyarrow=pa.__version__,
            threads={
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            load_average_at_start=load,
            load_average_at_end=os.getloadavg(),
            concurrent_load="otras sesiones comparten la máquina, ver las cargas medias",
        ),
        not_measured=[
            "GPU, porque el análisis es de CPU y no tiene tensores",
            "informes reales de la cartera, bloqueados con la evaluación científica",
            "energía, sin instrumento adecuado",
        ],
        cscv=cscv,
        analysis=analysis,
    )
    json.dumps(receipt, allow_nan=False)
    atomic_json(args.output, receipt)
    print(json.dumps({name: case["cscv"]["median_s"] for name, case in cscv.items()}))


if __name__ == "__main__":
    main()
