"""Comparar el recorrido de salida con los mismos paneles sintéticos y sin aprendizaje."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmarks.output_stage_cost import CALIBRATION_SESSIONS, SEEDS, synthetic
from mars_titan.evaluation import neutral_abstention as corrected
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.hardware.platform_identity import cpu_name

REFERENCE_SHA256 = "5c920072506ad605520264f25a9d469ded9d7b5ac1153ba345f7130253764204"


def load_reference(path):
    """Cargar solo la fuente anterior cuya huella se registró antes de la corrección."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("La referencia debe ser un archivo regular")
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("La referencia no conserva la huella anterior a la corrección")
    spec = importlib.util.spec_from_file_location("abstention_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


def measure(reference, assets, sessions, repeats=5):
    """Medir las dos etapas completas sobre las mismas entradas, con orden alterno."""
    calibration, calibration_assets, calibration_matures = synthetic(
        assets, CALIBRATION_SESSIONS, -CALIBRATION_SESSIONS - 2, seed=1
    )
    panels = [synthetic(assets, sessions, 0, seed=10 + seed)[0] for seed in range(SEEDS)]
    panels = [
        ForecastPanel.from_columns(
            panels[0].row_id,
            np.full(panels[0].rows, "US"),
            panels[0].prediction_at,
            panels[0].target,
            p.prediction,
            quantiles=p.quantiles,
            levels=p.levels,
        )
        for p in panels
    ]
    evaluation, names, matures = synthetic(assets, sessions, 0, seed=10)
    history = dict(
        asset_id=np.r_[calibration_assets, names],
        target=np.r_[calibration.target, evaluation.target],
        matures_at=np.r_[calibration_matures, matures],
    )

    def pipeline(module):
        record = module.fit(calibration, calibration_matures)
        volatility = module.past_volatility(
            names, evaluation.prediction_at, history, window=60, min_labels=20
        )
        disagreement = module.seed_disagreement(panels)
        outputs, _ = module.emit(
            record, evaluation, stream=0, volatility=volatility, disagreement=disagreement
        )
        return outputs, volatility

    # Cada implementación se calienta con el recorrido completo.
    old, new = pipeline(reference), pipeline(corrected)
    np.testing.assert_allclose(new[1], old[1], rtol=2e-12, atol=1e-15, equal_nan=True)
    identical = all(np.array_equal(old[0][key], new[0][key]) for key in old[0])
    if not identical:
        raise AssertionError("Las salidas ordinarias no conservan paridad exacta")
    cost = dict(reference=[], corrected=[])
    for repeat in range(repeats):
        order = [("reference", reference), ("corrected", corrected)]
        if repeat % 2:
            order.reverse()
        for key, module in order:
            start = time.perf_counter()
            pipeline(module)
            cost[key].append(time.perf_counter() - start)
    memory = {}
    for key, module in [("reference", reference), ("corrected", corrected)]:
        tracemalloc.start()
        pipeline(module)
        memory[key] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return dict(
        assets=assets,
        sessions=sessions,
        rows=evaluation.rows,
        calibration_rows=calibration.rows,
        seeds=SEEDS,
        history_rows=len(history["target"]),
        elapsed_seconds=cost,
        median_seconds={key: float(np.median(value)) for key, value in cost.items()},
        traced_peak_bytes=memory,
        outputs_exact=identical,
        volatility_max_abs_difference=float(np.nanmax(np.abs(new[1] - old[1]))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5, choices=range(1, 21))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("La medida necesita un archivo nuevo")
    reference = load_reference(args.reference)
    started = time.perf_counter()
    report = dict(
        measured_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        numpy=np.__version__,
        cpu=cpu_name(),
        device="cpu",
        gpu_used=False,
        thread_environment={
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        load_average_start=os.getloadavg(),
        repetitions=args.repeats,
        warmup=1,
        timing="perf_counter_complete_output_pipeline",
        input_preparation_in_timing=False,
        memory="tracemalloc_peak_in_pipeline",
        optimizer_steps=0,
        scientific_data=False,
        reference_sha256=REFERENCE_SHA256,
        source_sha256=hashlib.sha256(Path(corrected.__file__).read_bytes()).hexdigest(),
        cases=[],
    )
    for assets, sessions in [(256, 63), (256, 252), (4200, 252)]:
        result = measure(reference, assets, sessions, args.repeats)
        report["cases"].append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("assets", "sessions", "median_seconds", "outputs_exact")
                }
            ),
            flush=True,
        )
    report.update(
        process_elapsed_seconds=time.perf_counter() - started,
        process_peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        load_average_end=os.getloadavg(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
