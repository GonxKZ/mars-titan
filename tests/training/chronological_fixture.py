"""Corpus técnico con varios activos asíncronos, ausencias y etiquetas de la sesión siguiente."""

import json
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.training.cohort_contract import representation_hash

US_2023 = 1_672_531_200_000_000
US_2024 = 1_704_067_200_000_000
STAMP = pa.timestamp("us", tz="UTC")


@cache
def clock():
    return MarketClock("US", "1999-01-01", "2024-01-05")


def micros(moment):
    return int(np.datetime64(moment.replace(tzinfo=None), "us").astype(np.int64))


def decision(day):
    return micros(clock().decision(day))


def _schemas():
    samples = pa.schema(
        [
            ("cohort_id", pa.string()),
            ("prediction_at", STAMP),
            ("price_end_index", pa.int64()),
            ("news", pa.list_(pa.float32())),
            ("charts", pa.list_(pa.float32())),
            ("fundamentals", pa.list_(pa.float32())),
            ("macro", pa.list_(pa.float32())),
            ("presence", pa.list_(pa.bool_(), 5)),
            ("news_count", pa.int64()),
            ("input_availability", pa.struct([(name, STAMP) for name in MODALITIES])),
        ]
    )
    labels = pa.schema(
        [
            ("sample_row", pa.int64()),
            ("prediction_at", STAMP),
            ("target_available_at", STAMP),
            ("target", pa.float64()),
            ("partition", pa.string()),
            ("reason", pa.string()),
            ("cohort_id", pa.string()),
        ]
    )
    return samples, labels


def _numeric(width, present, value):
    """Valores, máscaras y edades con el primer concepto observado cuando hay presencia."""
    result = [0.0] * (3 * width)
    if present:
        result[0], result[width], result[2 * width] = float(value), 1.0, 0.25
    return result


def _observed(asset, index):
    """Activos con huecos, alta tardía y presencia variable de modalidades."""
    if asset == 1 and index % 4 == 3:
        return False
    return not (asset == 2 and index < 6)


def chronological_corpus(
    root,
    *,
    assets=3,
    first="2022-11-01",
    last="2023-02-28",
    group_size=8,
    perturb_after=None,
    widths=(4, 3, 1, 2),
):
    """Escribir una supervisión histórica con máscaras sin pasar por modelos ni objetivos reales.

    `perturb_after` altera entradas, precios posteriores y etiquetas que maduran después
    de ese instante, para comprobar que el pasado del recorrido no cambia.
    """
    root = Path(root)
    news, charts, concepts, indicators = widths
    roots = {name: root / name for name in ("prepared", "samples", "labels")}
    days = [d for d in clock().days if date.fromisoformat(first) <= d <= date.fromisoformat(last)]
    positions = [clock().days.index(day) for day in days]
    sample_schema, label_schema = _schemas()
    representation = dict(
        **policy_identity(HISTORICAL_MASKED),
        fundamental_concepts=[f"fixture:C{i}" for i in range(concepts)],
        macro_indicators=[f"macro_{i}" for i in range(indicators)],
        encoders={"purpose": "technical_fixture"},
        representation_code={"fixture": "0" * 64},
        text_aggregation="fixture_mean",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    entries, coverage, counts = [], [], dict(train=0, validation=0)
    for asset in range(assets):
        symbol = f"A{asset:04}"
        generator = np.random.default_rng(1000 + asset)
        price_count = positions[-1] + 2
        base = 20 + asset + np.cumsum(generator.normal(0, 0.05, price_count))
        if perturb_after is not None:
            limit = np.searchsorted(
                np.asarray([micros(m) for m in clock().decisions[:price_count]]),
                perturb_after,
                side="right",
            )
            base[limit:] *= 1.5
        prices = pa.table(
            dict(
                open=base,
                high=base + 0.3,
                low=base - 0.3,
                close=base + 0.1,
                volume=np.full(price_count, 100.0) + asset,
                available_at=clock().decisions[:price_count],
            )
        )
        samples, labels = [], []
        observed_rows = [i for i in range(len(days)) if _observed(asset, i)]
        for row_number, index in enumerate(observed_rows):
            position = positions[index]
            moment = clock().decisions[position]
            at = micros(moment)
            future = perturb_after is not None and at > perturb_after
            news_present = (index + asset) % 3 == 0
            fundamentals_present = (index + asset) % 2 == 0
            macro_present = index % 5 != 4
            noise = generator.normal(0, 1, news + charts + 2).astype(np.float32)
            if future:
                noise = noise * 3 + 1
            presence = [True, news_present, True, fundamentals_present, macro_present]
            samples.append(
                dict(
                    cohort_id="original_audited",
                    prediction_at=moment,
                    price_end_index=position,
                    news=noise[:news].tolist() if news_present else [0.0] * news,
                    charts=noise[news : news + charts].tolist(),
                    fundamentals=_numeric(concepts, fundamentals_present, noise[-2]),
                    macro=_numeric(indicators, macro_present, noise[-1]),
                    presence=presence,
                    news_count=2 if news_present else 0,
                    input_availability={
                        name: moment if presence[k] else None for k, name in enumerate(MODALITIES)
                    },
                )
            )
            maturity = clock().decisions[position + 1]
            year, mature_year = moment.year, maturity.year
            reason = (
                "insufficient_history"
                if row_number < 2
                else "zero_market_variance"
                if (asset, index) == (0, 18)
                else "missing_next_session"
                if index == len(days) - 1
                else "target_crosses_partition_boundary"
                if year != mature_year
                else "accepted"
            )
            target = round(float(generator.normal(0, 0.01)), 8)
            if perturb_after is not None and micros(maturity) > perturb_after:
                target = target - 1.0
            accepted = reason == "accepted"
            labels.append(
                dict(
                    sample_row=row_number,
                    prediction_at=moment,
                    target_available_at=maturity if accepted else None,
                    target=target if accepted else None,
                    partition=("train" if year <= 2022 else "validation") if accepted else None,
                    reason=reason,
                    cohort_id="original_audited",
                )
            )
        item = dict(market="US", symbol=symbol, cohort_id="original_audited")
        for kind, table in (
            ("prices", prices),
            ("samples", pa.Table.from_pylist(samples, schema=sample_schema)),
            ("labels", pa.Table.from_pylist(labels, schema=label_schema)),
        ):
            key = "prepared" if kind == "prices" else kind
            folder = roots[key] / "US" / symbol
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{kind}.parquet"
            pq.write_table(table, path, row_group_size=group_size)
            item[kind + "_sha256"] = sha256(path)
        item.update(
            samples=len(samples),
            representation_sha256=representation_hash(
                representation, input_policy=HISTORICAL_MASKED
            ),
            counts={part: sum(row["partition"] == part for row in labels) for part in counts},
        )
        for part in counts:
            counts[part] += item["counts"][part]
        entries.append(item)
        coverage.append(dict(market="US", symbol=symbol, state="encoded", samples=len(samples)))
    coverage.append(dict(market="US", symbol="NO_PRICES", state="missing_required_prices"))
    meta = dict(
        **policy_identity(HISTORICAL_MASKED),
        schema_version=3,
        kind="corpus_supervision",
        context_sessions=64,
        roots={name: str(path) for name, path in roots.items()},
        assets=entries,
        scope="full_corpus",
        cohort_complete=True,
        cohort_id="original_audited",
        news_content_policy="source_audited_not_external",
        technical_fixture=True,
        training_ready=False,
        final_test_opened=False,
        representation=representation,
        markets=["US"],
        coverage=coverage,
        candidate_count=len(coverage),
        failed_assets=0,
        samples=sum(item["samples"] for item in entries),
        counts=counts,
    )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(meta))
    return manifest


def phases(*, warmup="2022-11-01", train_decisions="2022-11-15", validation_warmup="2022-12-01"):
    """Fases de ajuste y validación con calentamiento explícito, sin abrir 2024."""
    train = FinancialPhase("train", decision(warmup), decision(train_decisions), US_2023, US_2023)
    validation = FinancialPhase(
        "validation", decision(validation_warmup), US_2023, US_2024, US_2024
    )
    return train, validation
