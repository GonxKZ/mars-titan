"""Paneles de medición y normalización reanudable por activo."""

import csv
import hashlib
import json
import os
import re
import resource
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fundamentals import read_fundamentals
from .inventory import entries
from .news import read_news
from .prices import read_prices
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock


def candidates(source: Path, database: Path, market: str) -> list[dict]:
    file = (
        "sp500stock_data_description.csv" if market == "US" else "hs300stock_data_description.csv"
    )
    with (source / file).open(encoding="utf-8-sig" if market == "US" else "gb18030") as stream:
        catalog = list(csv.DictReader(stream))
    sectors = {
        r["stock_name" if market == "US" else "证券代码"].upper().replace(".SH", ".SS"): r[
            "Sector" if market == "US" else "申万一级行业"
        ]
        for r in catalog
    }
    grouped = {}
    for row in entries(database):
        if row["market"] != market or row["state"] == "error":
            continue
        symbol = row["symbol"]
        asset = grouped.setdefault(
            symbol,
            {
                "symbol": symbol,
                "sector": sectors.get(symbol, ""),
                "bytes": 0,
                "paths": defaultdict(list),
            },
        )
        asset["paths"][row["modality"]].append(row["path"])
        if row["modality"] != "charts":
            asset["bytes"] += row["bytes"]
    return sorted(
        [
            x
            for x in grouped.values()
            if x["sector"] not in {"", "N/A", "Client Error"}
            and all(x["paths"].get(m) for m in ("prices", "news", "fundamentals"))
        ],
        key=lambda x: x["symbol"],
    )


def select_panel(assets: list[dict], *, per_sector: int = 2, seed: int = 42) -> list[dict]:
    if per_sector < 1:
        raise ValueError("per_sector debe ser positivo")
    groups = defaultdict(list)
    for asset in assets:
        groups[asset["sector"]].append(asset)
    selected = []
    for sector in sorted(groups):
        group = sorted(groups[sector], key=lambda x: (x["bytes"], x["symbol"]))
        median = group[len(group) // 2]
        high = group[min(len(group) - 1, int(len(group) * 0.9))]
        chosen = [median]
        if high != median:
            chosen.append(high)
        rest = sorted(
            [x for x in group if x not in chosen],
            key=lambda x: hashlib.sha256(f"{seed}:{x['symbol']}".encode()).hexdigest(),
        )
        selected.extend((chosen + rest)[:per_sector])
    return selected


def atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, name, compression="zstd", row_group_size=8192)
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def prepare_asset(
    source: Path,
    destination: Path,
    asset: dict,
    clock: MarketClock,
    *,
    news_reviews: dict | None = None,
) -> dict:
    outside_source(source, destination)
    symbol = asset["symbol"]
    if not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol) or symbol in {".", ".."}:
        raise ValueError("El símbolo del activo no es seguro")
    sources = {}
    for paths in asset["paths"].values():
        for relative in paths:
            path = source / relative
            if not path.resolve().is_relative_to(source.resolve()) or path.is_symlink():
                raise ValueError("La ruta de entrada sale del directorio de origen")
    for modality in ("prices", "news", "fundamentals"):
        paths = sorted(asset["paths"].get(modality, []))
        if not paths or (modality == "prices" and len(paths) != 1):
            raise ValueError(
                f"Faltan archivos de {modality} para {symbol} o su selección es ambigua"
            )
        for relative in paths:
            sources[relative] = sha256(source / relative)
    started = time.perf_counter()
    folder = destination / clock.market / symbol
    manifest_path = folder / "manifest.json"
    policy = {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "preparation.py",
            "prices.py",
            "news.py",
            "news_reviews.py",
            "fundamentals.py",
            "temporal.py",
        )
    }
    policy["calendar"] = hashlib.sha256(
        "|".join(x.isoformat() for x in clock.decisions).encode()
    ).hexdigest()
    policy["pyarrow"] = pa.__version__
    policy["news_reviews"] = (
        hashlib.sha256(json.dumps(news_reviews, sort_keys=True).encode()).hexdigest()
        if news_reviews is not None
        else None
    )
    fingerprint = hashlib.sha256(json.dumps([sources, policy], sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        valid = old["fingerprint"] == fingerprint and all(
            (folder / name).is_file() and sha256(folder / name) == digest
            for name, digest in old["artifacts"].items()
        )
        if valid:
            return {**old, "reused": True}
    prices, price_audit = read_prices(source / asset["paths"]["prices"][0], clock)
    news, rejected_news = [], []
    for relative in sorted(asset["paths"]["news"]):
        accepted, rejected = read_news(source / relative, symbol, clock, reviews=news_reviews)
        news.extend(accepted)
        rejected_news.extend(rejected)
    facts, fact_audit = read_fundamentals(
        [source / p for p in sorted(asset["paths"]["fundamentals"])], clock.market, clock
    )
    tables = {
        "prices": pa.Table.from_pandas(prices, preserve_index=False),
        "news": pa.Table.from_pylist(news),
        "fundamentals": pa.Table.from_pylist(facts),
    }
    artifacts = {}
    for name, table in tables.items():
        path = folder / f"{name}.parquet"
        atomic_parquet(path, table)
        artifacts[path.name] = sha256(path)
    for relative, digest in sources.items():
        if sha256(source / relative) != digest:
            raise ValueError(f"La fuente ha cambiado durante la preparación de {symbol}")
    result = {
        "schema_version": 1,
        "market": clock.market,
        "symbol": symbol,
        "sector_current_metadata": asset.get("sector"),
        "purpose": "engineering_profile_only",
        "fingerprint": fingerprint,
        "sources": sources,
        "policy": policy,
        "artifacts": artifacts,
        "counts": {k: len(v) for k, v in tables.items()},
        "price_audit": price_audit,
        "fundamentals_audit": fact_audit,
        "news_rejections": rejected_news,
        "reused": False,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "training_ready": False,
        "news_content_policy": "verified_full_articles"
        if news_reviews is not None
        else "not_reviewed",
    }
    atomic_json(manifest_path, result)
    return result
