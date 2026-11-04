"""Medir en CPU un ciclo completo de autoedición de la salida (G1, #748).

El ciclo no tiene tensores ni parámetros entrenables, así que no hay forward ni backward. Los
paneles son sintéticos con semilla fija (`output_stage_cost.synthetic`) y solo fijan formas:
256 activos de EE. UU., cinco semillas y cuatro tramos consecutivos (validación y calibración
de G y dos bloques de S) de tres o seis meses. Se miden el generador sobre toda la gramática,
la selección de la vigente y las 2N candidatas, el contraste con 2.000 réplicas y la memoria
reservada por NumPy. Nada se ajusta, selecciona ni evalúa con estos valores.
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

from benchmarks.output_stage_cost import SEEDS, synthetic
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import output_program
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.output_program import SeedPanels
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.proposals import output_self_edit as edit
from mars_titan.proposals.self_edit_cycles import CANDIDATES, SelfEditLedger

ASSETS = 256
# Sesiones de cada tramo: tres y seis meses.
CASES = {"quarter_blocks_256": 63, "half_year_blocks_256": 126}
PARTITIONS = ("validation", "calibration", "select", "assess")
PLAN = dict(
    candidates=CANDIDATES,
    placebo_seed=20261027,
    decision=dict(block_length=16, replicates=2000, seed=20261023, confidence=0.95),
)


def seed_panels(sessions, offset, seed):
    """Cinco semillas con las mismas filas y objetivos, y la madurez de cada etiqueta."""
    panels = [synthetic(ASSETS, sessions, offset, seed=seed + k)[0] for k in range(SEEDS)]
    first = panels[0]
    shared = tuple(
        ForecastPanel.from_columns(
            first.row_id,
            np.full(first.rows, "US"),
            first.prediction_at,
            first.target,
            panel.prediction,
            quantiles=panel.quantiles,
            levels=panel.levels,
        )
        for panel in panels
    )
    return SeedPanels(shared, 0, shared[0].prediction_at + 86_400_000_000)


def timed(function, repeats):
    samples, result = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - started)
    return result, dict(median_s=statistics.median(samples), min_s=min(samples), repeats=repeats)


def measure_case(sessions, repeats):
    parts = {
        name: seed_panels(sessions, (sessions + 2) * position, seed=100 * (position + 1))
        for position, name in enumerate(PARTITIONS)
    }

    def cycle():
        ledger = SelfEditLedger(PLAN, edit.GRAMMARS)
        index = edit.declare_output_cycle(
            ledger, inner_fit=parts["validation"], inner_score=parts["calibration"]
        )
        edit.select_output_cycle(ledger, index, fit_on=parts["calibration"], select=parts["select"])
        edit.assess_output_cycle(ledger, index, fit_on=parts["calibration"], assess=parts["assess"])
        return ledger

    _, generate_cost = timed(
        lambda: edit.generate(
            output_program.REFERENCE,
            parts["validation"],
            parts["calibration"],
            count=CANDIDATES,
        ),
        repeats,
    )
    ledger, cycle_cost = timed(cycle, repeats)
    tracemalloc.start()
    cycle()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    state = ledger.export()
    return dict(
        assets=ASSETS,
        seeds=SEEDS,
        sessions_per_partition=sessions,
        rows_per_partition=parts["select"].reference.rows,
        programs_in_grammar=len(output_program.programs()),
        programs_scored_by_generator=len(output_program.programs()) - 1,
        programs_scored_in_selection=2 * CANDIDATES + 1,
        generate=generate_cost,
        full_cycle=cycle_cost,
        full_cycle_numpy_peak_bytes=peak,
        exported_state_bytes=len(json.dumps(state).encode()),
    )


def run(args):
    """Mide los casos con los argumentos de la orden y devuelve el recibo."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    cases = {name: measure_case(sessions, args.repeats) for name, sessions in CASES.items()}
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/evaluation/output_program.py",
            "src/mars_titan/proposals/self_edit_cycles.py",
            "src/mars_titan/proposals/output_self_edit.py",
            "benchmarks/output_self_edit_cost.py",
        )
    }
    receipt = dict(
        schema_version=1,
        kind="output_self_edit_cost",
        issue=748,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        plan=PLAN,
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
            "forward y backward, porque el ciclo no tiene tensores ni parámetros entrenables",
            "GPU, por el mismo motivo",
            "lectura de Parquet y alineación con la vista, que hará la criba",
            "energía, sin instrumento adecuado",
        ],
        cases=cases,
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("La medida necesita un destino nuevo")
    receipt = run(args)
    # atomic_json crea la carpeta, escribe en un temporal y lo confirma con os.replace.
    atomic_json(args.output, receipt)
    print(json.dumps({name: case["full_cycle"] for name, case in receipt["cases"].items()}))


if __name__ == "__main__":
    main()
