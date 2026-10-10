"""Corpus temporal técnico con entradas ausentes y objetivos declarados, sin modelos."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.price_windows import calendar_digest, price_window_contract
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.cohort_contract import representation_hash
from tests.training.test_corpus_inputs import corpus

DEFAULT_DAYS = (
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
)


def historical_temporal_fixture(
    root,
    markets=("US",),
    days=None,
    *,
    assets=1,
    presence=None,
    market_absent=None,
    price_start=None,
):
    """Crear el corpus técnico. `days` permite fijar las sesiones de cada mercado.

    `presence(market, symbol, row)` puede devolver noticias, fundamentales y macro de cada
    muestra, con vectores, recuento de noticias y disponibilidad coherentes. Sin ella, las
    muestras no tienen noticias ni fundamentales y el macro aparece en una de cada tres.
    `market_absent` declara sesiones sin filas en todo el mercado, que desaparecen de los
    precios de cada activo y entran en el contrato de ventanas de la v3.1. `price_start` fija la
    primera sesión con precio de cada activo.
    """
    root = Path(root)
    clocks = {market: MarketClock(market, "1999-01-01", "2024-01-05") for market in markets}
    parent = corpus(root / "parent", assets=assets, rows=1, markets=markets)
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
    if market_absent is not None:
        representation["price_window"] = price_window_contract(
            {
                market: dict(
                    start=clock.days[0].isoformat(),
                    end=clock.days[-1].isoformat(),
                    decisions_sha256=calendar_digest(clock),
                )
                for market, clock in clocks.items()
            },
            {market: list(market_absent.get(market, [])) for market in markets},
        )
    coverage, protocols = [], {}
    counts = dict(train=0, validation=0)
    for asset in meta["assets"]:
        market, symbol = asset["market"], asset["symbol"]
        clock = clocks[market]
        requested = DEFAULT_DAYS if days is None else days[market]
        positions = [
            clock.days.index(date.fromisoformat(day))
            for day in requested
            if date.fromisoformat(day) in clock.days
        ]
        price_count = positions[-1] + 1
        # Cada fila conserva el valor de su sesión, así que quitar una sesión no cambia el resto.
        absent = [
            clock.days.index(date.fromisoformat(day))
            for day in (market_absent or {}).get(market, [])
        ]
        start = clock.days.index(date.fromisoformat(price_start)) if price_start else 0
        kept = np.setdiff1d(np.arange(start, price_count), absent)
        values = 10 + kept.astype(np.float64) / 1000
        prices = pa.table(
            dict(
                open=values,
                high=values + 0.2,
                low=values - 0.2,
                close=values + 0.1,
                volume=np.full(len(kept), 100.0),
                available_at=[clock.decisions[i] for i in kept],
            )
        )
        samples, labels = [], []
        for row_number, position in enumerate(positions):
            moment = clock.decisions[position]
            day = moment.date().isoformat()
            news, fundamentals, observed_macro = False, False, row_number % 3 == 0
            if presence is not None:
                news, fundamentals, observed_macro = presence(market, symbol, row_number)
            flags = [True, news, True, fundamentals, observed_macro]
            samples.append(
                dict(
                    cohort_id="original_audited",
                    prediction_at=moment,
                    price_end_index=int(np.searchsorted(kept, position)),
                    news=[0.5 if news else 0.0] * 384,
                    charts=[0.25] * 512,
                    # Valor, máscara y edad del único concepto fundamental del fixture.
                    fundamentals=[1.0, 1.0, 0.0] if fundamentals else [0.0] * 3,
                    macro=[0.0, 0.0, float(observed_macro), 0.0, 0.0, 0.0],
                    presence=flags,
                    news_count=int(news),
                    input_availability={
                        name: moment if flags[index] else None
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
        coverage.append(dict(market=market, symbol=symbol, state="encoded", samples=len(samples)))
        if not any(row["symbol"] == "NO_PRICES" and row["market"] == market for row in coverage):
            coverage.append(
                dict(market=market, symbol="NO_PRICES", state="missing_required_prices")
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
