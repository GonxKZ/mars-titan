"""Corpus temporal técnico con entradas ausentes y objetivos declarados, sin modelos."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.cohort_contract import representation_hash
from tests.training.test_corpus_inputs import corpus


def historical_temporal_fixture(root, markets=("US",)):
    root = Path(root)
    parent = corpus(root / "parent", assets=1, rows=1, markets=markets)
    meta = json.loads(parent.read_text())
    policy = policy_identity(HISTORICAL_MASKED)
    representation = dict(
        **policy,
        fundamental_concepts=["fixture:Assets"],
        macro_indicators=["macro_a", "macro_b"],
        encoders={"purpose": "technical_fixture"},
        representation_code={"fixture": "0" * 64},
        text_aggregation="fixture_mean",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    coverage, protocols, clocks = [], {}, {}
    counts = dict(train=0, validation=0)
    for asset in meta["assets"]:
        market, symbol = asset["market"], asset["symbol"]
        clock = clocks[market] = MarketClock(market, "1999-01-01", "2024-01-05")
        requested = [
            "2000-01-04",
            "2000-06-30",
            "2009-01-15",
            "2022-11-15",
            "2022-12-15",
            "2022-12-30",
            "2023-01-17",
            "2023-02-15",
            "2023-03-15",
            "2023-09-28",
            "2023-10-02",
            "2023-12-28",
            "2023-12-29",
        ]
        positions = [
            clock.days.index(date.fromisoformat(day))
            for day in requested
            if date.fromisoformat(day) in clock.days
        ]
        price_count = positions[-1] + 1
        values = 10 + np.arange(price_count, dtype=np.float64) / 1000
        prices = pa.table(
            dict(
                open=values,
                high=values + 0.2,
                low=values - 0.2,
                close=values + 0.1,
                volume=np.full(price_count, 100.0),
                available_at=clock.decisions[:price_count],
            )
        )
        samples, labels = [], []
        for row_number, position in enumerate(positions):
            moment = clock.decisions[position]
            day = moment.date().isoformat()
            observed_macro = row_number % 3 == 0
            presence = [True, False, True, False, observed_macro]
            samples.append(
                dict(
                    cohort_id="original_audited",
                    prediction_at=moment,
                    price_end_index=position,
                    news=[0.0] * 384,
                    charts=[0.25] * 512,
                    fundamentals=[0.0] * 3,
                    macro=[0.0, 0.0, float(observed_macro), 0.0, 0.0, 0.0],
                    presence=presence,
                    news_count=0,
                    input_availability={
                        name: moment if presence[index] else None
                        for index, name in enumerate(MODALITIES)
                    },
                )
            )
            reason = (
                "insufficient_history"
                if day == "2000-01-04"
                else "target_after_cutoff"
                if day == "2023-12-29"
                else "target_crosses_partition_boundary"
                if day == "2022-12-30"
                else "accepted"
            )
            labels.append(
                dict(
                    sample_row=row_number,
                    prediction_at=moment,
                    target_available_at=clock.decisions[position + 1]
                    if reason in {"accepted", "target_crosses_partition_boundary"}
                    else None,
                    target=(row_number + 1) / 100
                    if reason in {"accepted", "target_crosses_partition_boundary"}
                    else None,
                    partition=("train" if moment.year <= 2022 else "validation")
                    if reason == "accepted"
                    else None,
                    reason=reason,
                    cohort_id="original_audited",
                )
            )
        stamp = pa.timestamp("us", tz="UTC")
        sample_schema = pa.schema(
            [
                ("cohort_id", pa.string()),
                ("prediction_at", stamp),
                ("price_end_index", pa.int64()),
                *[
                    (name, pa.list_(pa.float32()))
                    for name in ("news", "charts", "fundamentals", "macro")
                ],
                ("presence", pa.list_(pa.bool_(), 5)),
                ("news_count", pa.int64()),
                ("input_availability", pa.struct([(name, stamp) for name in MODALITIES])),
            ]
        )
        label_schema = pa.schema(
            [
                ("sample_row", pa.int64()),
                ("prediction_at", stamp),
                ("target_available_at", stamp),
                ("target", pa.float64()),
                ("partition", pa.string()),
                ("reason", pa.string()),
                ("cohort_id", pa.string()),
            ]
        )
        for kind, table in (
            ("prices", prices),
            ("samples", pa.Table.from_pylist(samples, schema=sample_schema)),
            ("labels", pa.Table.from_pylist(labels, schema=label_schema)),
        ):
            key = "prepared" if kind == "prices" else kind
            path = Path(meta["roots"][key]) / market / symbol / f"{kind}.parquet"
            pq.write_table(table, path, row_group_size=3)
            asset[kind + "_sha256"] = sha256(path)
        asset.update(
            cohort_id="original_audited",
            samples=len(samples),
            representation_sha256=representation_hash(
                representation, input_policy=HISTORICAL_MASKED
            ),
            counts={part: sum(row["partition"] == part for row in labels) for part in counts},
        )
        for part in counts:
            counts[part] += asset["counts"][part]
        coverage.extend(
            [
                dict(market=market, symbol=symbol, state="encoded", samples=len(samples)),
                dict(market=market, symbol="NO_PRICES", state="missing_required_prices"),
            ]
        )
        config = json.loads(Path("configs/evaluation/real-expanded-walk-forward.json").read_text())
        config.update(market=market, train_start="2000-01-01")
        protocols[market] = root / f"protocol-{market}.json"
        atomic_json(protocols[market], config)
    meta.update(
        **policy,
        schema_version=3,
        cohort_id="original_audited",
        news_content_policy="source_audited_not_external",
        technical_fixture=True,
        context_sessions=64,
        scope="full_corpus",
        cohort_complete=True,
        training_ready=False,
        final_test_opened=False,
        representation=representation,
        markets=list(markets),
        coverage=coverage,
        candidate_count=len(coverage),
        failed_assets=0,
        samples=sum(item.get("samples", 0) for item in coverage),
        counts=counts,
    )
    atomic_json(parent, meta)
    return SimpleNamespace(parent=parent, protocols=protocols, clocks=clocks, metadata=meta)
