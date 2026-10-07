"""Corpus técnico completo generado desde cero, sin datos ni pesos científicos."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.macro_coverage import assess_macro_completeness
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.data.test_macro_coverage import SCHEMA
from tests.training.test_corpus_inputs import corpus


def temporal_fixture(root, market="CN"):
    """Crear 19 decisiones mensuales y diez ventanas. No escribir recibos de modelos."""
    root = Path(root)
    parent = corpus(root / "parent", assets=1, rows=19, markets=(market,))
    meta = json.loads(parent.read_text())
    clock = MarketClock(market, "2022-01-01", "2024-01-05")
    months = [(2022, month) for month in range(6, 13)] + [(2023, month) for month in range(1, 13)]
    positions = [
        next(
            i
            for i, day in enumerate(clock.days)
            if (day.year, day.month) == month and day.day >= 15
        )
        for month in months
    ]
    times = [clock.decisions[i] for i in positions]
    maturity = [clock.decisions[i + 1] for i in positions]
    asset = meta["assets"][0]
    relative = Path(market) / asset["symbol"]
    price_path = Path(meta["roots"]["prepared"]) / relative / "prices.parquet"
    count = positions[-1] + 2
    values = 10 + np.arange(count, dtype=np.float64) / 1000
    pq.write_table(
        pa.table(
            dict(
                open=values,
                high=values + 0.2,
                low=values - 0.2,
                close=values + 0.1,
                volume=100 + np.arange(count, dtype=np.float64),
                available_at=clock.decisions[:count],
            )
        ),
        price_path,
    )
    sample_path = Path(meta["roots"]["samples"]) / relative / "samples.parquet"
    pq.write_table(
        pa.table(
            dict(
                prediction_at=times,
                price_end_index=positions,
                news=[[i / 100] * 384 for i in range(19)],
                charts=[[i / 200] * 512 for i in range(19)],
                fundamentals=[[1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * 19,
                macro=[[0.0] * 420] * 19,
                input_availability=[
                    dict.fromkeys(("prices", "news", "charts", "fundamentals", "macro"), moment)
                    for moment in times
                ],
            )
        ),
        sample_path,
        row_group_size=4,
    )
    label_path = Path(meta["roots"]["labels"]) / relative / "labels.parquet"
    pq.write_table(
        pa.table(
            dict(
                sample_row=np.arange(19),
                prediction_at=times,
                target_available_at=maturity,
                target=[(i % 3 - 1) / 100 for i in range(19)],
                partition=["train"] * 7 + ["validation"] * 12,
                reason=["accepted"] * 19,
            )
        ),
        label_path,
    )
    encoded = root / "encoded/manifest.json"
    atomic_json(
        encoded,
        dict(
            schema_version=1,
            technical_fixture=True,
            context_sessions=64,
            configuration=dict(encoders={"kind": "generated_test_vectors", "seed": 42}),
            final_test_opened=False,
        ),
    )
    counts = dict(train=7, validation=12)
    asset.update(
        prices_sha256=sha256(price_path),
        samples_sha256=sha256(sample_path),
        labels_sha256=sha256(label_path),
        counts=counts,
    )
    meta.update(
        technical_fixture=True,
        context_sessions=64,
        scope="full_corpus",
        cohort_complete=True,
        markets=[market],
        counts=counts,
        configuration=dict(source_manifest_sha256=sha256(encoded)),
        final_test_opened=False,
    )
    atomic_json(parent, meta)
    catalog = root / "macro/catalog.csv"
    catalog.parent.mkdir()
    catalog.write_bytes(Path("data/catalogs/macro-indicators.csv").read_bytes())
    with catalog.open() as stream:
        indicators = list(csv.DictReader(stream))
    macro = catalog.parent / "macro.parquet"
    fixture_hash = sha256(encoded)
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    prediction_at=moment,
                    indicator_id=entry["id"],
                    value=float(i + 1),
                    available_at=moment,
                    period_start=moment.date().replace(day=1).isoformat(),
                    missing_reason=None,
                    source_hashes=[fixture_hash],
                    unit=entry["unit"],
                    seasonal_adjustment="technical_fixture",
                )
                for moment in times
                for i, entry in enumerate(indicators)
            ],
            schema=SCHEMA,
        ),
        macro,
        row_group_size=140,
    )
    admission = root / "admission"
    assess_macro_completeness(
        macro, catalog, admission, market=market, start="2022-06-01", end="2023-12-31"
    )
    protocol = json.loads(Path("configs/evaluation/chinese-real-walk-forward.json").read_text())
    protocol.update(market=market, seeds=[42])
    protocol_path = root / "protocol.json"
    atomic_json(protocol_path, protocol)
    views = root / "views"
    prepare_temporal_corpus(parent, protocol_path, macro, admission / "report.json", views)
    neural = json.loads(Path("configs/baselines/strict-temporal-search-us.json").read_text())
    neural.update(
        arms=[market],
        models=["gru"],
        case_indices=[0],
        finalist_seeds=[42],
        max_epochs=2,
        patience=1,
        batch_size=8,
        posttraining_epochs=1,
        continuation_selection=dict(metric="session_mae", patience=1, min_delta=0.0),
    )
    files = {"neural_config": neural}
    for name, filename in (
        ("tabular_config", "tabular-convergence-us.json"),
        ("post_config", "real-matched-posttraining.json"),
    ):
        plan = json.loads((Path("configs/baselines") / filename).read_text())
        plan["seeds" if name == "post_config" else "finalist_seeds"] = [42]
        files[name] = plan
    paths = {}
    for name, value in files.items():
        path = root / f"{name}.json"
        atomic_json(path, value)
        paths[name] = path
    return SimpleNamespace(
        parent=parent,
        encoded=encoded,
        views=views,
        protocol=protocol_path,
        admission=admission / "report.json",
        reference=root / "reference",
        **paths,
    )


def confirmed_references(fixture):
    """Escribir recibos técnicos verificables, con artefactos que nunca se cargan como pesos."""
    plan = json.loads(fixture.neural_config.read_text())
    arm = plan["arms"][0]
    temporal = json.loads((fixture.views / "report.json").read_text())
    folds = []
    for fold in temporal["folds"]:
        root = fixture.reference / fold["id"]
        source = fixture.views / fold["id"] / "manifest.json"
        source_hash = sha256(source)
        population = json.loads(source.read_text())
        population["source_manifest_sha256"] = source_hash
        view = root / "views" / f"{arm}.json"
        atomic_json(view, population)
        runs = []
        for index in range(3):
            folder = root / f"runs/case-{index}"
            folder.mkdir(parents=True)
            artifacts = {}
            for name in ("model.pt", "train.parquet", "validation.parquet"):
                path = folder / name
                path.write_bytes(b"technical receipt fixture, not model weights or predictions")
                artifacts[name] = dict(path=name, sha256=sha256(path))
            report = dict(
                technical_fixture=True,
                status="completed",
                scope="full_corpus",
                cohort_complete=True,
                samples=population["counts"],
                identity=dict(manifest_sha256=sha256(view), weighting="natural"),
                checkpoint=artifacts["model.pt"],
                predictions={p: artifacts[f"{p}.parquet"] for p in ("train", "validation")},
                final_test_opened=False,
            )
            atomic_json(folder / "run.json", report)
            runs.append(
                dict(
                    path=f"runs/case-{index}",
                    status="completed",
                    arm=arm,
                    weighting="natural",
                    report_sha256=sha256(folder / "run.json"),
                )
            )
        atomic_json(
            root / "summary.json",
            dict(
                technical_fixture=True,
                kind="reference_search",
                status="completed",
                planned_runs=3,
                completed_runs=3,
                scope="full_corpus",
                cohort_complete=True,
                final_test_opened=False,
                identity=dict(manifest_sha256=source_hash, configuration=plan),
                runs=runs,
            ),
        )
        folds.append(dict(id=fold["id"], status="completed", planned_runs=3, completed_runs=3))
    atomic_json(
        fixture.reference / "summary.json",
        dict(
            technical_fixture=True,
            kind="temporal_reference_search",
            status="completed",
            planned_runs=30,
            completed_runs=30,
            final_test_opened=False,
            folds=folds,
        ),
    )
