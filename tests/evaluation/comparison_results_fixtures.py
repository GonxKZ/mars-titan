"""Estados de campaña con predicciones manuales, sin dataset ni modelos reales."""

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.comparison_protocol import declaration as analysis_declaration
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.comparison_cohorts import write_cohort
from tests.training.test_screening_cohorts import reconcile, source

COMMIT = "a" * 40


def prediction_table(path, symbols, year, partition, scale):
    starts = dict(
        validation=datetime(year - 1, 9, 1, tzinfo=UTC),
        calibration=datetime(year - 1, 10, 1, tzinfo=UTC),
        evaluation=datetime(year, 1, 2, tzinfo=UTC),
    )
    days = dict(validation=2, calibration=3, evaluation=40)[partition]
    rows = []
    for day in range(days):
        at = starts[partition] + timedelta(days=day)
        for position, symbol in enumerate(symbols):
            value = (position - 1.2) * 0.02 + (day % 5) * 0.001
            target = value * (0.25 if partition == "evaluation" else 0.5)
            rows.append((f"{symbol}/{at.isoformat()}", symbol, at, target, scale * value))
    columns = dict(
        sample_id=[r[0] for r in rows],
        asset_id=[r[1] for r in rows],
        market=["US"] * len(rows),
        prediction_at=pa.array([r[2] for r in rows], pa.timestamp("us", tz="UTC")),
        target=[r[3] for r in rows],
        prediction=[r[4] for r in rows],
        zero=[0.0] * len(rows),
    )
    for name, offset in (
        ("0025", -0.02),
        ("0100", -0.01),
        ("0500", 0),
        ("0900", 0.01),
        ("0975", 0.02),
    ):
        columns[f"quantile_{name}"] = [r[4] + offset for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return dict(path=path.name, sha256=sha256(path), rows=len(rows), bytes=path.stat().st_size)


def campaign_state(tmp_path, *, years=(2019,), seeds=(42, 43), reader=False, blocks=False):
    document, _ = campaign.load_declaration()
    document = copy.deepcopy(document)
    document["execution"].pop("neural_blocks", None)
    names = ["transformer", "titans_mac", *(["mars_titan_m3"] if reader else [])]
    document["models"] = {name: document["models"][name] for name in names}
    document["cohort"] = dict(size=3)
    document["analysis"] = analysis_declaration(document["models"])
    document["analysis"]["bootstrap"]["replicates"] = 64
    document["origins"] = [
        dict(year=year, fold=f"fold-{year - 2005:03}", role="development") for year in years
    ]
    document["seeds"] = list(seeds)
    if blocks:
        document["execution"]["neural_blocks"] = {
            f"{year}/seed-{seed}": campaign.MACHINES[i % 2]
            for year in years
            for i, seed in enumerate(seeds)
        }
    document["stopping"]["fit"]["max_epochs"] = 1
    declaration_path = tmp_path / "declaration.json"
    atomic_json(declaration_path, document)
    digest = sha256(declaration_path)
    layout = campaign.Layout(tmp_path / "state", tmp_path / "views")
    for origin in document["origins"]:
        view = source(origin["year"])
        for asset in view["assets"]:
            asset["counts"].update(validation=2, calibration=3, evaluation=40)
        atomic_json(layout.source(origin["fold"]), reconcile(view))
        write_cohort(layout.source(origin["fold"]), layout.cohort(origin["year"]), size=3)
    campaign.authorize(layout, digest, commit=COMMIT, note="Estado de prueba, sin entrenamiento")

    for machine in campaign.MACHINES:
        for job in campaign.plan(document, machine):
            view_path = layout.cohort(job.year) / "manifest.json"
            view = json.loads(view_path.read_text())
            symbols = [asset["symbol"] for asset in view["assets"]]
            directory = layout.output(job.key)
            scale = (1 if job.model == "transformer" else 0.8) * (1 + 0.1 * (job.seed - 42))
            predictions = {
                partition: prediction_table(
                    directory / f"{partition}-predictions.parquet",
                    symbols,
                    job.year,
                    partition,
                    scale,
                )
                for partition in ("validation", "calibration", "evaluation")
            }
            run_id = hashlib.sha256(job.key.encode()).hexdigest()
            weights = directory / "fit/checkpoints/model.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"NO INCLUIR ESTOS PESOS")
            atomic_json(directory / "model.json", dict(weights=[1, 2, 3]))
            checkpoint = dict(path="fit/checkpoints/model.pt", sha256=sha256(weights))
            fit = dict(
                schema_version=1,
                kind="fixture_fit",
                run_id=run_id,
                status="completed",
                final_test_opened=False,
                history=[
                    dict(epoch=0, global_step=0, train=None, validation=dict(session_mae=0.02)),
                    dict(
                        epoch=1,
                        global_step=1,
                        train=dict(session_mae=0.02, mean_loss=0.01, updates=1, labels=150),
                        validation=dict(session_mae=0.01),
                    ),
                ],
                best_epoch=1,
                best_score=0.01,
                best_checkpoint=checkpoint,
            )
            atomic_json(directory / "fit/run.json", fit)
            report = dict(
                schema_version=1,
                kind="titans_walk_forward_window",
                run_id=run_id,
                status="completed",
                final_test_opened=False,
                request=dict(seed=job.seed, window=job.fold, view_sha256=sha256(view_path)),
                identity=dict(window=view["temporal_view"]["fold"]),
                predictions=predictions,
                fit=dict(path="fit/run.json", sha256=sha256(directory / "fit/run.json")),
                checkpoint=checkpoint,
            )
            if job.needs:
                parent = layout.output(job.needs) / "run.json"
                parent_report = json.loads(parent.read_text())
                report["request"]["parent"] = dict(
                    run_sha256=sha256(parent),
                    checkpoint_sha256=parent_report["checkpoint"]["sha256"],
                )
            atomic_json(directory / "run.json", report)
            atomic_json(
                layout.record(job.key),
                dict(
                    status="completed",
                    model=job.model,
                    year=job.year,
                    seed=job.seed,
                    attempts=[dict(status="completed", seconds=4.0, process_wall_seconds=5.0)],
                    result=dict(run_id=run_id),
                ),
            )
    return document, digest, layout
