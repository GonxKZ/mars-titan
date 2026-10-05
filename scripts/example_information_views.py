"""Recorrido sintético de vistas y consumidores sobre CorpusDataset, sin entrenar."""

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.information_views import ROLES, ViewConsumer, attach_parent
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.information_inputs import ViewDataset, corpus_view


def control(output, assets, rows):
    roots = {name: output / name for name in ("prepared", "samples", "labels")}
    entries = []
    times = [datetime(2022, 1, 3, tzinfo=UTC) + timedelta(days=i) for i in range(rows)]
    for index in range(assets):
        symbol = f"CONTROL{index:03}"
        folders = {name: root / "US" / symbol for name, root in roots.items()}
        for folder in folders.values():
            folder.mkdir(parents=True)
        prices = pa.table(
            dict(
                open=[10.0, 11.0, 12.0],
                high=[12.0, 13.0, 14.0],
                low=[9.0, 10.0, 11.0],
                close=[11.0, 12.0, 13.0],
                volume=[100.0, 120.0, 80.0],
                available_at=[times[0] - timedelta(days=n) for n in (3, 2, 1)],
            )
        )
        samples = pa.table(
            dict(
                prediction_at=times,
                price_end_index=[2] * rows,
                news=[[float(i + 1)] * 384 for i in range(rows)],
                charts=[[2.0] * 512] * rows,
                fundamentals=[[1.0, 2.0, 1.0, 1.0, 0.5, 0.75]] * rows,
                macro=[[3.0, 4.0, 1.0, 1.0, 0.25, 0.5]] * rows,
                input_availability=[dict.fromkeys(MODALITIES, stamp) for stamp in times],
            )
        )
        labels = pa.table(
            dict(
                sample_row=list(range(rows)),
                prediction_at=times,
                target_available_at=[stamp + timedelta(seconds=1) for stamp in times],
                target=[0.0] * rows,
                partition=["train"] * rows,
            )
        )
        paths = {
            "prices": folders["prepared"] / "prices.parquet",
            "samples": folders["samples"] / "samples.parquet",
            "labels": folders["labels"] / "labels.parquet",
        }
        for name, table in (("prices", prices), ("samples", samples), ("labels", labels)):
            pq.write_table(table, paths[name], row_group_size=64)
        entries.append(
            dict(
                market="US",
                symbol=symbol,
                counts=dict(train=rows, validation=0),
                **{name + "_sha256": sha256(path) for name, path in paths.items()},
            )
        )
    path = output / "manifest.json"
    atomic_json(
        path,
        dict(
            schema_version=1,
            kind="corpus_supervision",
            context_sessions=2,
            roots={name: str(root.resolve()) for name, root in roots.items()},
            assets=entries,
            scope="development_snapshot",
            cohort_complete=False,
            counts=dict(train=assets * rows, validation=0),
            historical_evidence=False,
            macro_catalog_sha256=sha256(
                Path(__file__).parents[1] / "data/catalogs/macro-indicators.csv"
            ),
            representation=dict(
                fundamental_concepts=[
                    "us-gaap:AssetsCurrent:USD",
                    "us-gaap:LiabilitiesCurrent:USD",
                ],
                macro_indicators=["us_cpi", "us_cpi_yoy"],
                encoders={"kind": "synthetic_control"},
                representation_code={"control": sha256(Path(__file__))},
                text_aggregation="mean",
                context_sessions=2,
                news_lookback_sessions=5,
            ),
        ),
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=int, default=8)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.assets <= 128 or not 2 <= args.rows <= 256 or not 1 <= args.repeats <= 100:
        parser.error("El tamaño o las repeticiones del control exceden su presupuesto")
    if args.output.exists():
        parser.error("El control necesita un directorio nuevo")
    started = time.perf_counter()
    manifest = control(args.output, args.assets, args.rows)
    source = CorpusDataset(manifest, cache_bytes=16 * 1024**2)
    full = corpus_view(
        source, macro_catalog=Path(__file__).parents[1] / "data/catalogs/macro-indicators.csv"
    )
    view = full.without(sources=["news", "prices"], variables=["macro/us_cpi"])
    full.save(args.output / "full-view.json")
    view.save(args.output / "reduced-view.json")
    artifact_hash = sha256(Path(__file__))

    def excluded_probe(inputs):
        return sum(
            inputs[name].reshape(len(inputs[name]), -1).sum(1, dtype=np.float64)
            for name in ("prices", "news", "charts", "macro")
        )

    contracts = {role: view.artifact(role, artifact_hash) for role in sorted(ROLES)}
    consumers = {
        role: ViewConsumer(view, contract, excluded_probe, artifact_path=Path(__file__))
        for role, contract in contracts.items()
    }
    rejected = False
    try:
        ViewConsumer(view, full.artifact("parent", artifact_hash), excluded_probe)
    except ValueError:
        rejected = True
    atomic_json(args.output / "consumer-contracts.json", contracts)
    kwargs = dict(partition="train", batch_size=32, epoch=0, seed=42)
    reference = hashlib.sha256()
    for batch in source.batches(**kwargs):
        reference.update(json.dumps(batch["sample_ids"]).encode())
    durations, input_bytes = [], 0
    checks = dict.fromkeys(consumers, 0.0)
    for repetition in range(args.repeats + 1):
        begin, count, observed = time.perf_counter(), 0, hashlib.sha256()
        for batch in ViewDataset(source, view).batches(**kwargs):
            observed.update(json.dumps(batch["sample_ids"]).encode())
            count += len(batch["sample_ids"])
            input_bytes += sum(value.nbytes for value in batch["inputs"].values())
            for role, consumer in consumers.items():
                checks[role] = max(checks[role], float(np.max(np.abs(consumer(batch)))))
            inherited = attach_parent(batch, consumers["parent"])
            if inherited["features"][:, -1].any():
                raise ValueError("La predicción del control restauró una entrada retirada")
        if observed.digest() != reference.digest() or count != args.assets * args.rows:
            raise ValueError("La vista cambió la población del control")
        if repetition:
            durations.append(time.perf_counter() - begin)
    result = dict(
        historical_evidence=False,
        trained_model=False,
        consumer_algorithm="suma de entradas retiradas",
        rows=count,
        same_population=True,
        other_view_rejected=rejected,
        consumer_checks=checks,
        full_view_sha256=full.identity,
        reduced_view_sha256=view.identity,
        measurement=dict(
            repetitions=args.repeats,
            warmup=1,
            iteration_seconds=durations,
            p50_seconds=float(np.quantile(durations, 0.5)),
            p95_seconds=float(np.quantile(durations, 0.95)),
            rows_per_second=count / float(np.mean(durations)),
            input_bytes_per_iteration=input_bytes // (args.repeats + 1),
            wall_seconds=time.perf_counter() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            gpu_used=False,
            energy_joules=None,
            total_cost=None,
        ),
        environment=dict(
            python=platform.python_version(), numpy=np.__version__, pyarrow=pa.__version__
        ),
    )
    atomic_json(args.output / "comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
