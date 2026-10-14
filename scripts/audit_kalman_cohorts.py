"""Composición de las cohortes de escritura de B6 kalman en las vistas reales (#455).

B6 escribe juntas las etiquetas que maduran en el mismo instante, y la regla kalman trata cada
cohorte con la ϱ de una sesión, que se estima por par de mercado e instante de decisión en los
últimos meses del ajuste. El script lee solo las etiquetas de cada vista walk-forward y, con las
mismas fases que la ventana B6 (tramo de estimación y los tres tramos medidos) y el mismo filtro
del índice de observaciones (decisión dentro del tramo y madurez antes de su final), cuenta
cuántas cohortes mezclan sesiones o mercados y cuántas sesiones se reparten entre cohortes. El
calentamiento de las fases solo observa entradas, así que no cambia ninguna cohorte. No lee
entradas ni carga modelos, y la reserva de 2024 queda sin abrir.

    uv run python scripts/audit_kalman_cohorts.py --views <vistas A v2>/US+CN \\
        --recipe configs/titans/kalman-correction-historical-masked.json \\
        --output reports/engineering/kalman-cohort-composition-20261014.json
"""

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.splits import build_folds
from mars_titan.memory.cohort_noise import cohort_composition
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.mars_titan_correction import estimation_phase
from mars_titan.training.titans_walk_forward import view_protocol
from mars_titan.training.walk_forward_phases import PREDICTED, window_phases

# Las cohortes no dependen del calentamiento, que solo observa entradas.
WARMUP_MONTHS = 12
COUNTS = (
    "rows",
    "cohorts",
    "sessions",
    "cohorts_mixing_sessions",
    "rows_in_cohorts_mixing_sessions",
    "cohorts_mixing_markets",
    "rows_in_cohorts_mixing_markets",
    "sessions_split_across_cohorts",
)


def phase_labels(dataset, phase):
    """Madurez, mercado e instante de decisión de las etiquetas que el índice da a una fase."""
    maturity, markets, sessions = [], [], []
    for asset in dataset.assets:
        if not asset["counts"].get(phase.partition):
            continue
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            rows = file.metadata.num_rows
        _, stamps, _, matures = dataset._labels(asset, phase.partition, rows)
        keep = (
            (stamps >= phase.decision_start)
            & (stamps < phase.decision_end)
            & (matures < phase.decision_end)
        )
        maturity.append(matures[keep])
        sessions.append(stamps[keep])
        markets += [asset["market"]] * int(keep.sum())
    return (
        np.concatenate(maturity).astype(np.int64),
        np.array(markets),
        np.concatenate(sessions).astype(np.int64),
    )


def audit_window(manifest, estimation_months):
    """Composición de las cohortes de cada fase de B6 en la vista de una ventana."""
    window = Path(manifest).parent.name
    protocol = view_protocol(manifest)
    fold = {item["id"]: item for item in build_folds(protocol)}[window]
    dataset = CorpusDataset(Path(manifest), cache_bytes=0, input_policy=HISTORICAL_MASKED)
    measured = window_phases(fold, WARMUP_MONTHS)
    phases = dict(
        estimation=estimation_phase(fold, estimation_months, WARMUP_MONTHS),
        **{name: measured[name] for name in PREDICTED},
    )
    return dict(
        view_sha256=dataset.identity,
        phases={
            name: cohort_composition(*phase_labels(dataset, phase))
            for name, phase in phases.items()
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started, clock = time.perf_counter(), time.process_time()
    months = json.loads(args.recipe.read_text())["recipe"]["estimation_months"]
    windows = {
        path.parent.name: audit_window(path, months)
        for path in sorted(args.views.glob("fold-*/manifest.json"))
    }
    totals = {
        phase: {key: sum(item["phases"][phase][key] for item in windows.values()) for key in COUNTS}
        for phase in ("estimation", *PREDICTED)
    }
    root = Path(__file__).resolve().parents[1]
    atomic_json(
        args.output,
        dict(
            schema_version=1,
            kind="kalman_cohort_composition",
            created_at_utc=datetime.now(UTC).isoformat(),
            issue=455,
            scope=args.views.name,
            recipe=dict(path=str(args.recipe), sha256=sha256(args.recipe)),
            estimation_months=months,
            cohort="labels_maturing_at_the_same_instant_within_a_b6_phase",
            session="market_and_decision_instant",
            windows=windows,
            totals=totals,
            code={
                name: sha256(root / name)
                for name in (
                    "scripts/audit_kalman_cohorts.py",
                    "src/mars_titan/memory/cohort_noise.py",
                    "src/mars_titan/training/corpus_inputs.py",
                )
            },
            seconds=time.perf_counter() - started,
            cpu_seconds=time.process_time() - clock,
            max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            final_test_opened=False,
        ),
    )
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
