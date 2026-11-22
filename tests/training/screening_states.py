"""Estados sintéticos de campaña para las pruebas del informe de la criba y de su promoción."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SESSIONS = dict(validation=40, calibration=20, evaluation=60)
FIRST = dict(validation=0, calibration=100, evaluation=200)


def attempt(site, gpu, *, machine="x86_64", commit="a" * 40, seconds=3600.0):
    """Un intento con la telemetría del equipo en el que corrió."""
    return dict(
        seconds=seconds,
        execution=dict(code_commit=commit),
        resources=dict(
            site=site,
            gpu_name=gpu,
            platform=dict(system="Linux", machine=machine),
            versions=dict(torch="2.14.0+cu130"),
        ),
    )


def write_model(
    state, name, year, signal, *, seed, markets=("US",), assets=40, tier="core", attempts=None
):
    """Un modelo terminado cuyo acierto es `signal` veces el objetivo más un ruido propio."""
    run = state / "runs" / name / str(year) / "seed-42"
    run.mkdir(parents=True)
    noise = np.random.default_rng(seed)
    for split, count in SESSIONS.items():
        rows = dict(sample_id=[], market=[], prediction_at=[], target=[], prediction=[])
        for market in markets:
            # El objetivo depende del mercado, el origen y el tramo, no del modelo.
            targets = np.random.default_rng([year, FIRST[split], len(market), ord(market[0])])
            for session in range(count):
                instant = datetime(year - 1, 1, 1, 20, 5, tzinfo=UTC) + timedelta(
                    days=FIRST[split] + session
                )
                target = targets.normal(size=assets)
                prediction = signal * target + noise.normal(size=assets)
                for asset in range(assets):
                    rows["sample_id"].append(f"{market}/A{asset:03d}/{instant.timestamp():.0f}")
                    rows["market"].append(market)
                    rows["prediction_at"].append(instant)
                    rows["target"].append(float(target[asset]))
                    rows["prediction"].append(float(prediction[asset]))
        table = pa.table(rows).cast(
            pa.schema(
                [
                    ("sample_id", pa.string()),
                    ("market", pa.string()),
                    ("prediction_at", pa.timestamp("us", tz="UTC")),
                    ("target", pa.float64()),
                    ("prediction", pa.float64()),
                ]
            )
        )
        pq.write_table(table, run / f"{split}-predictions.parquet")
    (run / "fit").mkdir()
    history = [
        dict(epoch=epoch, validation=dict(session_rank_ic=0.01 * epoch)) for epoch in range(1, 5)
    ]
    (run / "fit/run.json").write_text(json.dumps(dict(history=history)))
    queue = state / "queue"
    queue.mkdir(exist_ok=True)
    record = dict(
        status="completed",
        model=name,
        tier=tier,
        year=year,
        seed=42,
        attempts=[dict(seconds=3600.0)] if attempts is None else attempts,
        result=dict(best_epoch=2, best_score=0.48),
    )
    (queue / f"{name}--{year}--seed-42.json").write_text(json.dumps(record))
