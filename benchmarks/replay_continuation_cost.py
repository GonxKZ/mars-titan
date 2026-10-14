"""Medir en CPU lo que el repaso de #55 añade a una ventana real del ajuste por etapas.

Recibe la vista del padre (ventana k-1) y la de la ventana k, publicadas por
`prepare_temporal_corpus`, y repite la preparación de `posttraining.matrix_runs`: índice de
cohortes, filas nuevas desde el final de la calibración del padre y, solo para el repaso,
las sesiones anteriores que el padre ajustó completas con su régimen. Después recorre una
época de lotes de la continuación completa y de cada control de repaso con el lote de la
campaña y el presupuesto de bloque de la etapa A.

El padre es un predictor nulo de coste despreciable: la medida separa la lectura y la
preparación de los lotes del coste del padre, que la caché paga una vez por fila y ya mide
`campaign_throughput`. No se crea ningún modelo ni optimizador, no se registran pérdidas ni
errores y la reserva de 2024 no se abre.
"""

import argparse
import json
import platform
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.walk_forward_receipt import _microseconds
from mars_titan.posttraining import matrix_runs
from mars_titan.posttraining import replay_continuation as replay
from mars_titan.posttraining.inputs import PairedInputs
from mars_titan.posttraining.parent_cache import ParentCache
from mars_titan.posttraining.replay_inputs import ReplayInputs
from mars_titan.training.campaign_chain import posttraining_rows
from mars_titan.training.campaign_throughput import _view_fold
from mars_titan.training.corpus_inputs import CorpusDataset

STAGE = Path("configs/posttraining/historical-masked-adapter-stage-a.json")
CAMPAIGN = Path("configs/baselines/historical-masked-campaign-a.json")
MATRIX = Path("configs/posttraining/adapter-matrix-v3.json")
SEED = 42


def _commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _zero(inputs, presence=None):
    return np.zeros(len(next(iter(inputs.values()))), dtype=np.float32)


def _epoch(data, batch_size):
    """Una época de la semilla de la matriz: segundos, lotes y filas por origen."""
    began, batches, rows = time.perf_counter(), 0, {}
    for batch in data.batches(
        partition="train", condition="real", batch_size=batch_size, epoch=0, seed=SEED
    ):
        batches += 1
        rows[batch["origin"]] = rows.get(batch["origin"], 0) + len(batch["target"])
    return dict(seconds=time.perf_counter() - began, batches=batches, rows=rows)


def _statistics(source):
    return {key: value for key, value in source.statistics.items() if not isinstance(value, dict)}


def measure(parent_view, view, work, output):
    stage = json.loads(STAGE.read_text())
    campaign = json.loads(CAMPAIGN.read_text())
    ratio = json.loads(MATRIX.read_text())[replay.SECTION_NAME]["ratio"]
    budget = stage["cohort_reading"]["max_block_bytes"]
    batch_size = campaign["neural"]["batch_size"]
    folds = [
        _view_fold(CorpusDataset(Path(path), input_policy=HISTORICAL_MASKED))
        for path in (parent_view, view)
    ]
    start, _ = posttraining_rows(*folds)
    since = _microseconds(start)
    timings, report = {}, {}
    began = time.perf_counter()
    window = matrix_runs.MatrixWindow(
        view,
        Path(work) / "window",
        encoding=sha256(view),
        input_policy=HISTORICAL_MASKED,
        batch_size=batch_size,
        stop=SimpleNamespace(requested=False),
        max_block_bytes=budget,
        since=since,
    )
    timings["window"] = time.perf_counter() - began
    with window, tempfile.TemporaryDirectory(dir=work) as folder:
        cache = ParentCache(Path(folder) / "parent.sqlite", "0" * 64, "cost", _zero)
        began = time.perf_counter()
        sessions = window.earlier(parent_view)
        timings["earlier_sessions_and_routes"] = time.perf_counter() - began
        plain = PairedInputs(window.train, window.validation, cache)
        report["full_continuation"] = dict(
            epoch=_epoch(plain, batch_size), source=_statistics(window.train)
        )
        for control in replay.CONTROLS:
            began = time.perf_counter()
            earlier = replay.SOURCES[control] == replay.EARLIER
            with ReplayInputs(
                window.train,
                window.validation,
                cache,
                spec=replay.spec(control, ratio),
                batch_size=batch_size,
                seed=SEED,
                candidates=sessions if earlier else None,
                open_earlier=window.open_sessions if earlier else None,
            ) as data:
                planned = time.perf_counter() - began
                first = _epoch(data, batch_size)
                again = _epoch(data, batch_size)
                report[control] = dict(
                    plan_seconds=planned,
                    budget=data.budget("real", batch_size),
                    planned_sessions=len(data.plan),
                    first_epoch=first,
                    second_epoch=again,
                    earlier_source=None if data.earlier is None else _statistics(data.earlier),
                    resident=None if data.earlier is None else bool(data.earlier.resident),
                )
        cache.close()
    routes = sessions["routes"]
    receipt = dict(
        schema_version=1,
        kind="replay_continuation_cost",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Preparación y lectura en CPU de una época con un padre nulo, sin modelos, "
            "optimizadores, pérdidas ni errores."
        ),
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        parent_view=dict(path=str(parent_view), sha256=sha256(parent_view)),
        view=dict(path=str(view), sha256=sha256(view)),
        windows=[fold["id"] for fold in folds],
        since=start,
        batch_size=batch_size,
        max_block_bytes=budget,
        ratio=ratio,
        seed=SEED,
        new_rows=plain.counts["train"],
        earlier=dict(
            admitted=len(sessions["moments"]),
            rows=int(sum(sessions["rows"])),
            not_fitted=sessions["not_fitted"],
            late=sessions["late"],
            routes={str(route): routes.count(route) for route in sorted(set(routes))},
        ),
        seconds=timings,
        controls=report,
        optimizer_steps=0,
        final_test_opened=False,
    )
    atomic_json(Path(output), receipt)
    print(json.dumps(dict(seconds=timings, earlier=receipt["earlier"]), indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-view", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measure(args.parent_view, args.view, args.work, args.output)


if __name__ == "__main__":
    main()
