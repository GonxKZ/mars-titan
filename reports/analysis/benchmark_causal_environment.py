"""Medir el entorno CPU con cohortes sintéticas, sin atribuirle el coste del predictor."""

import argparse
import platform
import resource
import time
from pathlib import Path

import gymnasium
import numpy as np

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.actions import ActionGrid
from mars_titan.environments.prediction import CausalPredictionEnv


def measure(assets=2500, steps=64, warmup=8, repeats=3):
    shapes = dict(prices=(64, 5), news=(384,), charts=(512,), fundamentals=(45,), macro=(420,))
    values = {name: np.ones((assets, *shape), dtype=np.float32) for name, shape in shapes.items()}
    ids = [f"US/A{index:04}" for index in range(assets)]
    origin, day = 1_546_300_800_000_000, 86_400_000_000
    identity = "a" * 64
    grid = ActionGrid.fit(np.linspace(-0.1, 0.1, 101), source_sha256=identity, partition="train")

    def source(index):
        if index >= steps + warmup + 1:
            return None
        at = origin + index * day
        return dict(
            prediction_at=at,
            asset_ids=ids,
            available_at=np.full(assets, at),
            target_available_at=np.full(assets, at + 2 * day),
            target=np.full(assets, 0.02),
            inputs=values,
        )

    trials = []
    for _ in range(repeats):
        environment = CausalPredictionEnv(
            source, source_sha256=identity, grid=grid, shapes=shapes, max_assets=assets
        )
        environment.reset(seed=42)
        actions = np.full(assets, 10)
        for _ in range(warmup):
            environment.step(actions)
        durations = []
        for _ in range(steps):
            started = time.perf_counter()
            _, _, done, _, info = environment.step(actions)
            durations.append(time.perf_counter() - started)
            if done or info["pending"] != assets:
                raise ValueError("La medida no corresponde al tramo estable previsto")
        trials.append(
            dict(
                step_seconds=durations,
                p50_seconds=float(np.quantile(durations, 0.5)),
                p95_seconds=float(np.quantile(durations, 0.95)),
                p99_seconds=float(np.quantile(durations, 0.99)),
                samples_per_second=assets * steps / sum(durations),
            )
        )
        environment.close()
    root = Path(__file__).resolve().parents[2]
    return dict(
        kind="synthetic_causal_environment_benchmark",
        device="cpu",
        assets_per_cohort=assets,
        shapes={name: list(shape) for name, shape in shapes.items()},
        warmup_steps=warmup,
        measured_steps_per_trial=steps,
        repetitions=repeats,
        source_array_bytes=sum(v.nbytes for v in values.values()),
        trials=trials,
        python=platform.python_version(),
        numpy=np.__version__,
        gymnasium=gymnasium.__version__,
        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        code={
            name: sha256(root / name)
            for name in (
                "src/mars_titan/environments/actions.py",
                "src/mars_titan/environments/cohorts.py",
                "src/mars_titan/environments/prediction.py",
                "reports/analysis/benchmark_causal_environment.py",
            )
        },
        limits=[
            "Cohortes sintéticas en RAM, sin lectura Parquet, red neuronal ni entrenamiento.",
            "No es una comparación de aceleración frente a otra implementación.",
            "El RSS incluye todo el proceso, los datos sintéticos y el calentamiento.",
            "La máquina puede compartir CPU y memoria con otras aplicaciones.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrent-workload", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("La salida ya existe. Conserva las medidas anteriores.")
    result = measure()
    result["concurrent_workloads"] = args.concurrent_workload or [
        "No se ha certificado la ausencia de otras cargas."
    ]
    atomic_json(args.output, result)
    print(
        {
            "p50_seconds": [trial["p50_seconds"] for trial in result["trials"]],
            "peak_rss_bytes": result["process_lifetime_peak_rss_bytes"],
        }
    )


if __name__ == "__main__":
    main()
