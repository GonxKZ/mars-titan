"""Objetivos residuales pequeños y completos para probar el diagnóstico de MT-016.

Las etiquetas se generan con el mismo código que `targets-v3` (`residual_targets_array` y la
asignación de particiones de `corpus_targets`), sobre precios sintéticos de dos activos y un
factor con huecos colocados a propósito:

- el factor empieza 150 sesiones después que el activo AAA, así que hay muestras con historia
  propia suficiente y sin pares con el factor;
- al factor le falta una sesión suelta, al activo AAA otra, y otra tiene una apertura nula;
- el activo BBB empieza más tarde y tiene muestras sin historia suficiente;
- AAA tiene una muestra en una hora que no es una decisión del calendario.
"""

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.batches import atomic_parquet_batches, read_bounded_table
from mars_titan.data.residual_arrays import residual_targets_array
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.corpus_targets import _label_batches

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs" / "targets" / "residual-diagnostics-v1.json"
START = "2020-06-01"
CUTOFF = date(2023, 12, 31)
FACTOR_START = 150
FACTOR_GAP = 400
STOCK_GAP = 450
INVALID_OPEN = 500
BBB_START = 200
OFF_CALENDAR = pd.Timestamp("2022-07-04 12:00", tz="UTC")


def clock():
    return MarketClock("US", START, "2024-01-05")


def _write_prices(path, sessions, decisions, returns, *, drop=(), invalid=(), late=0):
    rows = []
    level = 100.0
    for index, (session, decision, change) in enumerate(
        zip(sessions, decisions, returns, strict=True)
    ):
        level *= 1.0 + 0.3 * change
        opened = 0.0 if index in invalid else level
        rows.append(
            dict(
                open=opened,
                high=max(level, level * (1 + change)),
                low=min(level, level * (1 + change)),
                close=level * (1.0 + change),
                volume=1000 + index,
                session=session,
                available_at=decision - timedelta(minutes=5),
            )
        )
    rows = [row for index, row in enumerate(rows) if index not in drop]
    # Sesiones de 2024 con valores absurdos: no deben llegar al diagnóstico.
    day = date(2024, 1, 2)
    for offset in range(late):
        moment = pd.Timestamp(day + timedelta(days=offset), tz="UTC") + pd.Timedelta(hours=21)
        rows.append(
            dict(
                open=1.0,
                high=1e6,
                low=1.0,
                close=1e6,
                volume=1,
                session=(day + timedelta(days=offset)).isoformat(),
                available_at=moment,
            )
        )
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        table.schema.get_field_index("session"), "session", table["session"].dictionary_encode()
    )
    table = table.set_column(
        table.schema.get_field_index("available_at"),
        "available_at",
        table["available_at"].cast(pa.timestamp("us", tz="UTC")),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_samples(path, moments):
    presence = [[True, i % 3 == 0, True, i % 5 == 0, True] for i in range(len(moments))]
    table = pa.table(
        dict(
            prediction_at=pa.array(moments, type=pa.timestamp("us", tz="UTC")),
            presence=pa.array(presence, type=pa.list_(pa.bool_(), 5)),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def build(root: Path, *, late=0, seed=16):
    """Objetivos, edición y declaración de prueba. Devuelve sus rutas."""
    market = clock()
    sessions = [d.isoformat() for d in market.days if d <= CUTOFF]
    decisions = market.decisions[: len(sessions)]
    rng = np.random.default_rng(seed)
    factor_returns = rng.normal(0.0, 0.01, len(sessions))
    prepared, samples, labels = root / "prepared", root / "samples", root / "labels"
    factor_path = root / "factor" / "prices.parquet"
    _write_prices(
        factor_path,
        sessions,
        decisions,
        factor_returns,
        drop=set(range(FACTOR_START)) | {FACTOR_GAP},
        late=late,
    )
    factor = read_bounded_table(factor_path, max_rows=200_000).to_pandas()
    assets = []
    specs = dict(
        AAA=dict(first=0, drop={STOCK_GAP}, invalid={INVALID_OPEN}, beta=1.2),
        BBB=dict(first=BBB_START, drop=set(), invalid=set(), beta=0.7),
    )
    for symbol, spec in specs.items():
        stock_returns = 0.0002 + spec["beta"] * factor_returns + rng.normal(0, 0.015, len(sessions))
        price_path = prepared / "US" / symbol / "prices.parquet"
        _write_prices(
            price_path,
            sessions,
            decisions,
            stock_returns,
            drop=set(range(spec["first"])) | spec["drop"],
            invalid=spec["invalid"],
            late=late,
        )
        moments = [pd.Timestamp(d) for d in decisions[spec["first"] + 64 :]]
        if symbol == "AAA":
            moments.insert(100, OFF_CALENDAR)
        sample_path = samples / "US" / symbol / "samples.parquet"
        _write_samples(sample_path, moments)
        frame = read_bounded_table(price_path, max_rows=200_000).to_pandas()
        calculated = residual_targets_array(frame, factor, market, cutoff="2023-12-31")
        audit = Counter()
        label_path = labels / "US" / symbol / "labels.parquet"
        total = atomic_parquet_batches(label_path, _label_batches(sample_path, calculated, audit))
        assets.append(
            dict(
                market="US",
                symbol=symbol,
                prices_sha256=sha256(price_path),
                samples_sha256=sha256(sample_path),
                samples=total,
                labels_sha256=sha256(label_path),
                counts={p: audit[p] for p in ("train", "validation")},
                excluded_reasons={
                    k: v for k, v in sorted(audit.items()) if k not in {"train", "validation"}
                },
            )
        )
    edition = root / "edition.json"
    edition.write_text(
        json.dumps(
            dict(
                calendar_start={"US": START},
                candidate_count=3,
                coverage=[
                    *(
                        dict(market="US", symbol=a["symbol"], state="encoded", samples=a["samples"])
                        for a in assets
                    ),
                    dict(market="US", symbol="ZZZ", state="missing_required_prices", samples=0),
                ],
            )
        )
    )
    manifest = root / "targets.json"
    manifest.write_text(
        json.dumps(
            dict(
                kind="corpus_supervision",
                final_test_opened=False,
                roots=dict(prepared=str(prepared), samples=str(samples), labels=str(labels)),
                assets=assets,
                samples=sum(a["samples"] for a in assets),
                counts={p: sum(a["counts"][p] for a in assets) for p in ("train", "validation")},
                market_factors=dict(
                    US=dict(
                        market="US",
                        symbol="SPY",
                        prices_path=str(factor_path),
                        prices_sha256=sha256(factor_path),
                        point_in_time_verified=False,
                    )
                ),
                configuration=dict(source_manifest_sha256=sha256(edition)),
            )
        )
    )
    declaration = root / "declaration.json"
    document = json.loads(DECLARATION.read_text())
    document.update(minimum_assets_per_session=2, extremes=3)
    declaration.write_text(json.dumps(document))
    return dict(targets=manifest, edition=edition, declaration=declaration, root=root)


def rewrite_labels(paths, symbol, change):
    """Aplicar `change` a las etiquetas de un activo y actualizar su recibo."""
    manifest = json.loads(paths["targets"].read_text())
    path = paths["root"] / "labels" / "US" / symbol / "labels.parquet"
    table = pq.read_table(path).to_pandas()
    table = change(table)
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path)
    for receipt in manifest["assets"]:
        if receipt["symbol"] == symbol:
            receipt["labels_sha256"] = sha256(path)
            receipt["samples"] = len(table)
    manifest["samples"] = sum(a["samples"] for a in manifest["assets"])
    paths["targets"].write_text(json.dumps(manifest))
