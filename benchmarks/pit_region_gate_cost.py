"""Medir en CPU el coste de la puerta por regiones en fecha C4 (#751).

La etapa trabaja sobre predicciones guardadas y residuos ya maduros, sin tensores ni
parámetros entrenables. Los paneles son los sintéticos de `output_stage_cost` y el historial
tiene ocho factores latentes con semilla fija, solo para fijar formas. Se miden la lectura de
activos, las regiones de la correlación pasada, el ajuste completo de la variante y del
control permutado, la emisión y la memoria reservada por NumPy. Nada se ajusta, selecciona ni
evalúa con estos valores.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from benchmarks.output_stage_cost import (
    CALIBRATION_SESSIONS,
    CASES,
    DAY_US,
    START_US,
    peak_bytes,
    synthetic,
    timed,
)
from mars_titan.evaluation import pit_region_gate as gate
from mars_titan.hardware.platform_identity import cpu_name

FACTORS = 8


def residual_history(names, window, end, seed):
    """`window` madureces por activo, la última justo en `end`, con ocho factores latentes."""
    rng = np.random.default_rng(seed)
    loadings = np.eye(FACTORS)[np.arange(len(names)) % FACTORS]
    values = loadings @ rng.normal(size=(FACTORS, window)) + rng.normal(size=(len(names), window))
    matures = end - (window - 1 - np.arange(window)) * DAY_US
    return dict(
        asset_id=np.repeat(names, window),
        target=0.02 * values.ravel(),
        matures_at=np.tile(matures, len(names)),
    )


def measure_case(assets, sessions, repeats):
    offset = -CALIBRATION_SESSIONS - 2
    calibration, _, calibration_matures = synthetic(assets, CALIBRATION_SESSIONS, offset, seed=1)
    evaluation, _, _ = synthetic(assets, sessions, 0, seed=10)
    # Los activos se leen de las identidades `mercado/activo/instante`, como en la criba.
    calibration_assets = gate.panel_assets(calibration)
    names = np.unique(calibration_assets)
    cutoff = START_US + offset * DAY_US
    past = residual_history(names, gate.DEFAULT["window"], cutoff, seed=2)
    # El historial también lleva las etiquetas de G, que maduran después del corte.
    history = dict(
        asset_id=np.r_[past["asset_id"], calibration_assets],
        target=np.r_[past["target"], calibration.target],
        matures_at=np.r_[past["matures_at"], calibration_matures],
    )
    options = {key: gate.DEFAULT[key] for key in ("clusters", "window", "min_labels")}
    evaluation_assets, assets_cost = timed(lambda: gate.panel_assets(evaluation), repeats)
    regions, regions_cost = timed(
        lambda: gate.residual_regions(names, history, cutoff, **options), repeats
    )

    def fit(kind):
        config = gate.DEFAULT | dict(regions=kind)
        return gate.fit(calibration, calibration_matures, calibration_assets, history, config)

    record, fit_cost = timed(lambda: fit("residual_correlation"), repeats)
    _, permuted_cost = timed(lambda: fit("permuted"), repeats)
    _, apply_cost = timed(lambda: gate.apply(record, evaluation, evaluation_assets), repeats)
    sizes = np.unique(list(regions.values()), return_counts=True)
    return dict(
        assets=assets,
        sessions=sessions,
        rows=evaluation.rows,
        calibration_rows=calibration.rows,
        history_rows=len(history["target"]),
        regions=dict(zip(sizes[0].tolist(), sizes[1].tolist(), strict=True)),
        panel_assets=assets_cost,
        residual_regions=regions_cost,
        fit=fit_cost,
        fit_permuted=permuted_cost,
        apply=apply_cost,
        fit_numpy_peak_bytes=peak_bytes(lambda: fit("residual_correlation")),
        apply_numpy_peak_bytes=peak_bytes(
            lambda: gate.apply(record, evaluation, evaluation_assets)
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    load = os.getloadavg()
    cases = {
        name: measure_case(assets, sessions, args.repeats)
        for name, (assets, sessions) in CASES.items()
    }
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/data/correlation_regions.py",
            "src/mars_titan/evaluation/pit_region_gate.py",
            "src/mars_titan/evaluation/scale_diagnostics.py",
            "benchmarks/output_stage_cost.py",
            "benchmarks/pit_region_gate_cost.py",
        )
    }
    receipt = dict(
        schema_version=1,
        kind="pit_region_gate_cost",
        issue=751,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        config=gate.DEFAULT,
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
            scipy=scipy.__version__,
            load_average_at_start=load,
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
    print(json.dumps({name: case["fit"] for name, case in cases.items()}))


if __name__ == "__main__":
    main()
