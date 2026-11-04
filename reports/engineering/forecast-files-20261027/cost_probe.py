"""Contrastar lectura B4 anterior y común sobre Parquet de fixtures, sin aprendizaje."""

import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import forecast_files, lesion_curves
from mars_titan.hardware.platform_identity import cpu_name

BASE = "917c6fdad747d7c788bc11172f612099717309c7"
ROOT = Path(__file__).resolve().parents[3]
SOURCE = "src/mars_titan/evaluation/lesion_curves.py"
NAMES = ("intact", "zero", "area_a", "area_b")


def fixture(path, assets):
    rows = []
    for day in range(16):
        for market in ("US", "CN"):
            for asset in range(assets):
                target = 0.01 * np.sin(asset + day)
                prediction = target + (0.004 if market == "CN" else 0.002)
                rows.append(
                    dict(
                        sample_id=f"{market}/{day}/{asset}",
                        market=market,
                        prediction_at=1_600_000_000_000_000 + day * 86_400_000_000,
                        target=target,
                        prediction=prediction,
                        lesion_intact=prediction,
                        lesion_zero=target + 0.003,
                        lesion_area_a=target - 0.001,
                        lesion_area_b=target + 0.001,
                    )
                )
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        2, "prediction_at", table["prediction_at"].cast(pa.timestamp("us", "UTC"))
    )
    order = np.random.Generator(np.random.PCG64(73)).permutation(len(table))
    pq.write_table(table.take(order), path, row_group_size=37, compression="zstd")
    return len(table)


def main():
    assert Path(forecast_files.__file__).resolve().is_relative_to(ROOT / "src")
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    original = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT)
    previous = types.ModuleType("mars_titan.evaluation._legacy_lesion_files")
    previous.__file__ = str(ROOT / SOURCE)
    previous.__package__ = "mars_titan.evaluation"
    exec(compile(original, str(ROOT / SOURCE), "exec"), previous.__dict__)
    optimizer_attempts = 0

    def reject_optimizer(*_):
        nonlocal optimizer_attempts
        optimizer_attempts += 1
        raise RuntimeError("La medida de archivos no admite optimizadores")

    guard = register_optimizer_step_pre_hook(reject_optimizer)
    started = time.perf_counter()
    samples, inputs = [], []
    try:
        for assets in (64, 256, 1024):
            path = output / f"fixture-{assets}.parquet"
            count = fixture(path, assets)
            inputs.append(
                dict(assets=assets, rows=count, bytes=path.stat().st_size, sha256=sha256(path))
            )
            columns = (
                "sample_id",
                "market",
                "prediction_at",
                "target",
                "prediction",
                *(f"lesion_{n}" for n in NAMES),
            )
            for repetition in range(4):
                results = {}
                order = ("previous", "shared") if repetition % 2 == 0 else ("shared", "previous")
                for name in order:
                    assert time.perf_counter() - started < 300, (
                        "Presupuesto de 300 segundos agotado"
                    )
                    module = previous if name == "previous" else lesion_curves
                    before = time.perf_counter()
                    scores, resources = module._scores(
                        path, columns, NAMES, dict(rank_ic_min_assets=3), output, 4096
                    )
                    seconds = time.perf_counter() - before
                    results[name] = scores
                    assert resources["rows"] == count and resources["sessions"] == 32
                    samples.append(
                        dict(
                            assets=assets,
                            implementation=name,
                            repetition=repetition,
                            warmup=repetition == 0,
                            seconds=seconds,
                        )
                    )
                for name in results["previous"]:
                    first, second = (results[mode][name] for mode in ("previous", "shared"))
                    assert first.summary() == second.summary()
                    assert first.to_table().equals(second.to_table())
        summaries = [
            dict(
                assets=assets,
                implementation=name,
                repetitions=len(values),
                median_seconds=statistics.median(values),
                min_seconds=min(values),
                max_seconds=max(values),
            )
            for assets in (64, 256, 1024)
            for name in ("previous", "shared")
            for values in [
                [
                    s["seconds"]
                    for s in samples
                    if s["assets"] == assets and s["implementation"] == name and not s["warmup"]
                ]
            ]
        ]
        result = dict(
            kind="technical_prediction_file_scoring_parity_and_cost",
            status="completed",
            base_commit=BASE,
            previous_source_sha256=hashlib.sha256(original).hexdigest(),
            sources={
                p: sha256(ROOT / p)
                for p in (
                    SOURCE,
                    "src/mars_titan/evaluation/forecast_files.py",
                    "src/mars_titan/evaluation/forecast_panel.py",
                    "src/mars_titan/evaluation/forecast_scores.py",
                )
            },
            probe_sha256=sha256(Path(__file__)),
            runtime=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                pyarrow=pa.__version__,
                duckdb=duckdb.__version__,
                torch=str(torch.__version__),
                device="cpu",
                dtype="float64",
                threads=2,
            ),
            hardware=dict(cpu=cpu_name(), platform=platform.platform()),
            fixtures=inputs,
            samples=samples,
            summaries=summaries,
            paired_checks=12,
            conditions_per_pair=5,
            all_session_tables_and_summaries_exact=True,
            optimizer_attempts=optimizer_attempts,
            cuda_initialized=torch.cuda.is_initialized(),
            seconds_without_imports=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            timing_scope=(
                "Lectura, ordenación y puntuación B4. Excluye crear fixtures, "
                "comparación posterior, importaciones y carga de la referencia. "
                "El tiempo total incluye fixtures y comparación."
            ),
            limits=[
                "Fixtures de 32 sesiones con 64/256/1024 activos, cinco condiciones "
                "y grupos Parquet de 37 filas.",
                "Una muestra de calentamiento y tres medidas por implementación y tamaño. "
                "No caracteriza p95 o p99.",
                "CPU FP64. Sin GPU, transferencias, energía, padre, inferencia "
                "ni evaluación científica.",
                "El pico RSS corresponde a todo el proceso y no atribuye memoria "
                "por implementación.",
            ],
        )
        atomic_json(output / "cost.json", result)
        print(json.dumps(dict(status=result["status"], summaries=summaries)))
    finally:
        guard.remove()


if __name__ == "__main__":
    main()
