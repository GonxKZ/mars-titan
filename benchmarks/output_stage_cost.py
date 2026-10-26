"""Medir en CPU el coste de las etapas de salida del estudio de variantes (#748).

Las etapas trabajan sobre predicciones guardadas, sin tensores ni parámetros entrenables, así
que no hay forward ni backward que medir. Los cuantiles y objetivos son sintéticos con semilla
fija y solo fijan las formas: bloques S de tres meses y de un año con 256 activos de EE. UU.,
la población de la criba, y un caso de 4.200 activos, la mayor población US de una sesión en
la edición desde 2000. Se miden el ajuste en G, la emisión, las puntuaciones de los controles
y la memoria reservada por NumPy. Nada se ajusta, selecciona ni evalúa con estos valores.
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

from mars_titan.evaluation import neutral_abstention as na
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.quantile_head import LEVELS

DAY_US = 86_400_000_000
START_US = 1_546_300_800_000_000  # 2019-01-01, dentro del periodo de desarrollo
CALIBRATION_SESSIONS = 63
SEEDS = 5
# Activos y sesiones del tramo evaluado.
CASES = {"s_block_256": (256, 63), "year_256": (256, 252), "population_4200": (4200, 252)}


def synthetic(assets, sessions, offset, seed):
    """Panel con mediana libre y colas simétricas, más los activos y la madurez por fila."""
    rng = np.random.default_rng(seed)
    rows = assets * sessions
    names = np.array([f"US/A{index:05d}" for index in range(assets)])
    asset = np.tile(names, sessions)
    at = START_US + (offset + np.repeat(np.arange(sessions), assets)) * DAY_US
    median = rng.normal(0, 0.01, rows)
    spread = np.abs(rng.normal(0.02, 0.005, rows))[:, None]
    quantiles = median[:, None] + spread * np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
    target = 0.05 * median + rng.standard_t(4, rows) * 0.02
    panel = ForecastPanel.from_columns(
        np.char.add(np.char.add(asset, "/"), at.astype(str)),
        np.full(rows, "US"),
        at,
        target,
        median,
        quantiles=quantiles,
        levels=LEVELS,
    )
    # El panel ordena por instante e identidad. Los activos se leen en ese mismo orden.
    order = np.lexsort((np.char.add(np.char.add(asset, "/"), at.astype(str)), at))
    return panel, asset[order], panel.prediction_at + DAY_US


def timed(function, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - started)
    return result, dict(median_s=statistics.median(samples), min_s=min(samples), repeats=repeats)


def peak_bytes(function):
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def measure_case(assets, sessions, repeats):
    calibration, calibration_assets, calibration_matures = synthetic(
        assets, CALIBRATION_SESSIONS, -CALIBRATION_SESSIONS - 2, seed=1
    )
    seeds = [synthetic(assets, sessions, 0, seed=10 + seed)[0] for seed in range(SEEDS)]
    # Las semillas comparten filas y objetivos y solo cambian la predicción.
    seeds = [
        ForecastPanel.from_columns(
            seeds[0].row_id,
            np.full(seeds[0].rows, "US"),
            seeds[0].prediction_at,
            seeds[0].target,
            item.prediction,
            quantiles=item.quantiles,
            levels=LEVELS,
        )
        for item in seeds
    ]
    evaluation, evaluation_assets, evaluation_matures = synthetic(assets, sessions, 0, seed=10)
    history = dict(
        asset_id=np.r_[calibration_assets, evaluation_assets],
        target=np.r_[calibration.target, evaluation.target],
        matures_at=np.r_[calibration_matures, evaluation_matures],
    )
    record, fit_cost = timed(lambda: na.fit(calibration, calibration_matures), repeats)
    _, apply_cost = timed(lambda: na.apply(record, seeds[0]), repeats)
    volatility, volatility_cost = timed(
        lambda: na.past_volatility(
            evaluation_assets,
            evaluation.prediction_at,
            history,
            window=na.DEFAULT["volatility_window"],
            min_labels=na.DEFAULT["volatility_min_labels"],
        ),
        repeats,
    )
    disagreement, disagreement_cost = timed(lambda: na.seed_disagreement(seeds), repeats)

    def emit():
        return na.emit(record, seeds[0], stream=0, volatility=volatility, disagreement=disagreement)

    (_, summary), emit_cost = timed(emit, repeats)
    return dict(
        assets=assets,
        sessions=sessions,
        rows=evaluation.rows,
        calibration_rows=calibration.rows,
        history_rows=len(history["target"]),
        seeds=SEEDS,
        fit=fit_cost,
        apply=apply_cost,
        past_volatility=volatility_cost,
        seed_disagreement=disagreement_cost,
        emit_with_four_controls=emit_cost,
        emit_numpy_peak_bytes=peak_bytes(emit),
        volatility_numpy_peak_bytes=peak_bytes(
            lambda: na.past_volatility(
                evaluation_assets, evaluation.prediction_at, history, window=60, min_labels=20
            )
        ),
        abstained_fraction=summary["abstained_rows"] / summary["rows"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    cases = {
        name: measure_case(assets, sessions, args.repeats)
        for name, (assets, sessions) in CASES.items()
    }
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/evaluation/neutral_abstention.py",
            "src/mars_titan/calibration/conformal_quantiles.py",
            "benchmarks/output_stage_cost.py",
        )
    }
    receipt = dict(
        schema_version=1,
        kind="output_stage_cost",
        issue=748,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        environment=dict(
            cpu=cpu_name(),
            cpu_count=os.cpu_count(),
            threads=dict(
                omp=os.environ.get("OMP_NUM_THREADS"),
                openblas=os.environ.get("OPENBLAS_NUM_THREADS"),
            ),
            python=platform.python_version(),
            numpy=np.__version__,
            load_average_at_start=os.getloadavg(),
            concurrent_load="otras sesiones comparten la máquina, ver load_average_at_start",
        ),
        not_measured=[
            "forward y backward, porque la etapa no tiene tensores ni parámetros entrenables",
            "GPU, por el mismo motivo",
            "lectura de Parquet y alineación con la vista, que hace la criba",
            "energía, sin instrumento adecuado",
        ],
        cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({name: case["emit_with_four_controls"] for name, case in cases.items()}))


if __name__ == "__main__":
    main()
