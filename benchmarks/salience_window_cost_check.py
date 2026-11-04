"""Coste CPU de ventanas E3 completas con corpus y pesos de fixture, sin aprendizaje."""

import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.salience_budget import SalienceBudgetConfig
from tests.memory.spaced_window_fixture import study_fixture
from tests.training.test_salience_correction_window import frozen_backend as frozen_backend
from tests.training.test_salience_correction_window import (
    only_manual_frozen_parameters as only_manual_frozen_parameters,
)
from tests.training.test_salience_correction_window import run


def population(path):
    table = pq.read_table(path, columns=["sample_id", "prediction_at", "target"]).sort_by(
        "sample_id"
    )
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, table.schema) as writer:
        writer.write_table(table)
    return table.num_rows, hashlib.sha256(buffer.getvalue()).hexdigest()


def measure_window_cost(window, output, *, repeats=3, warmup=1):
    """Medir lectura, MAC, codec, escritura, métricas y publicación de las tres fases."""
    if type(repeats) is not int or not 1 <= repeats <= 3 or type(warmup) is not int or warmup != 1:
        raise ValueError("La medida necesita un calentamiento y de una a tres repeticiones")
    if output.exists():
        raise ValueError(
            "La medida de latencia necesita una salida nueva, sin ventanas reutilizadas"
        )
    if window["dataset"].manifest.get("technical_fixture") is not True:
        raise ValueError("Esta medida solo admite el corpus manual de fixture")
    parent = window["anchor"].parent / "parent"
    parent_report = json.loads((parent / "run.json").read_text())
    parent_files = (
        parent / "fit/run.json",
        parent / "run.json",
        parent / parent_report["checkpoint"]["path"],
    )
    before = tuple(sha256(path) for path in parent_files)
    baseline = None
    cases = []
    arms = (
        ("baseline", None),
        ("standardized_error", SalienceBudgetConfig()),
        ("absolute_error", SalienceBudgetConfig(rule="absolute_error")),
        ("uniform", SalienceBudgetConfig(rule="uniform")),
    )
    started = time.perf_counter()
    for repetition in range(warmup + repeats):
        for name, config in arms if repetition % 2 == 0 else arms[::-1]:
            destination = output / f"{repetition}-{name}"
            begin = time.perf_counter()
            report = run(window, destination, config)
            elapsed = time.perf_counter() - begin
            if report["status"] != "completed":
                raise RuntimeError("La ventana de fixture no se completó")
            populations = {
                phase: population(destination / f"{phase}-predictions.parquet")
                for phase in report["predictions"]
            }
            if baseline is None:
                baseline = populations
            if populations != baseline:
                raise RuntimeError("Las variantes no conservan filas y etiquetas")
            selected = {
                phase: values["metrics"]
                .get("salience", {})
                .get("selected", values["metrics"]["associative_writes"])
                for phase, values in report["predictions"].items()
            }
            cases.append(
                dict(
                    arm=name,
                    repetition=repetition,
                    warmup=repetition < warmup,
                    seconds=elapsed,
                    selected=selected,
                    rows={phase: count for phase, (count, _) in populations.items()},
                    mature_rows_per_second=sum(count for count, _ in populations.values())
                    / elapsed,
                    output_bytes=sum(path.stat().st_size for path in destination.iterdir()),
                    population_sha256={phase: digest for phase, (_, digest) in populations.items()},
                )
            )
            atomic_json(
                output / "cost-progress.json",
                dict(
                    schema_version=1,
                    measurement_state="partial",
                    cases=cases,
                    repeats=repeats,
                    warmup=warmup,
                ),
            )
            if time.perf_counter() - started > 900:
                raise TimeoutError("La medida supera su presupuesto de quince minutos por tamaño")
    if before != tuple(sha256(path) for path in parent_files):
        raise RuntimeError("El padre del fixture cambió durante la medida")
    summaries = []
    for arm, _ in arms:
        values = [case["seconds"] for case in cases if case["arm"] == arm and not case["warmup"]]
        summaries.append(
            dict(
                arm=arm,
                samples=len(values),
                median_seconds=statistics.median(values),
                minimum_seconds=min(values),
                maximum_seconds=max(values),
            )
        )
    atomic_json(
        output / "cost-progress.json",
        dict(
            schema_version=1,
            measurement_state="completed",
            cases=cases,
            repeats=repeats,
            warmup=warmup,
            parent_unchanged=True,
        ),
    )
    return dict(
        price_context=window["dataset"].context,
        input_widths=list(window["source"]["spec"].widths),
        configuration=dict(
            writes_per_market=64, min_history=4, initial_scale=0.01, scale_floor=1e-8
        ),
        cases=cases,
        summaries=summaries,
        warmup=warmup,
        repeats=repeats,
        total_seconds=time.perf_counter() - started,
        parent_unchanged=True,
    )


def test_complete_fixture_windows_cost(tmp_path, record_testsuite_property):
    began = time.perf_counter()
    rows = []
    for assets in (64, 256):
        fixture = study_fixture(
            tmp_path / f"fixture-{assets}", assets=assets, rich=True, price_row_group_size=8192
        )
        row = measure_window_cost(fixture, tmp_path / f"runs-{assets}")
        row["assets"] = assets
        row["price_row_group_size"] = 8192
        rows.append(row)
    root = Path(__file__).resolve().parents[1]
    files = (
        "benchmarks/salience_window_cost_check.py",
        "src/mars_titan/training/salience_inference.py",
        "src/mars_titan/training/mars_titan_correction.py",
    )
    report = dict(
        schema_version=1,
        scientific_data=False,
        device="cpu",
        python=platform.python_version(),
        platform=platform.platform(),
        measured=rows,
        total_seconds=time.perf_counter() - began,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sources={name: sha256(root / name) for name in files},
        limits=[
            "Corpus manual y padres sin entrenar. No mide calidad predictiva ni CUDA.",
            "El tiempo interno de cada ventana incluye sus tres fases y su publicación.",
            "La preparación del corpus y las comparaciones de población están "
            "en el tiempo de proceso.",
            "Los tamaños, máscaras y calendarios de fixture no representan "
            "toda la edición histórica.",
            "El RSS es absoluto y tres muestras no acreditan colas de latencia estables.",
        ],
    )
    record_testsuite_property("salience_window_cost", json.dumps(report, allow_nan=False))
