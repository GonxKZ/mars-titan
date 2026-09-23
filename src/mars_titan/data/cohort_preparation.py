"""Preparación por activo con cohorte, fuentes y publicación contable explícitas."""

import fcntl
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .cohort_news import COHORT_POLICIES, write_cohort_news
from .corpus_catalog import _source_path
from .fundamentals import read_fundamentals
from .preparation import atomic_parquet
from .prices import read_prices
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

FACT_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "concept",
            "unit",
            "period_start",
            "period_end",
            "filed",
            "accession",
            "source_file",
            "availability_rule",
        )
    ]
    + [("value", pa.float64()), ("available_at", pa.timestamp("us", tz="UTC"))]
)


def prepare_cohort_asset(
    source: Path,
    destination: Path,
    asset: dict,
    clock: MarketClock,
    *,
    cohort: str,
    reviews: dict,
    cutoff: str = "2023-12-31",
) -> dict:
    """Normalizar una empresa completa o conservar un fallo visible, sin cambiar fuentes."""
    outside_source(source, destination)
    if cohort not in COHORT_POLICIES:
        raise ValueError("La cohorte no está admitida")
    if asset.get("market") != clock.market:
        raise ValueError("El activo y el calendario pertenecen a otro mercado")
    symbol = asset.get("symbol")
    if (
        not isinstance(symbol, str)
        or symbol in {".", ".."}
        or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
    ):
        raise ValueError("El símbolo no es válido")
    if any(not asset["paths"].get(name) for name in ("prices", "news", "fundamentals", "charts")):
        raise ValueError("Falta una modalidad del activo")
    if len(asset["paths"]["prices"]) != 1 or asset.get("source_errors"):
        raise ValueError("El inventario contiene fuentes ambiguas o con errores")
    end = date.fromisoformat(cutoff)
    relative_paths = sorted(
        {path for kind in ("prices", "news", "fundamentals") for path in asset["paths"][kind]}
    )
    paths = {name: _source_path(source, name) for name in relative_paths}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if any(asset["hashes"].get(name) != digest for name, digest in hashes.items()):
        raise ValueError("Una huella de fuente no coincide con el inventario")
    policy = dict(
        cohort_id=cohort,
        cutoff=cutoff,
        pyarrow=pa.__version__,
        calendar=hashlib.sha256(
            "|".join(t.isoformat() for t in clock.decisions).encode()
        ).hexdigest(),
        reviews=hashlib.sha256(json.dumps(reviews, sort_keys=True).encode()).hexdigest(),
        code={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "cohort_preparation.py",
                "cohort_news.py",
                "corpus_catalog.py",
                "news.py",
                "news_reviews.py",
                "news_audit.py",
                "prices.py",
                "fundamentals.py",
                "temporal.py",
                "batches.py",
                "storage.py",
                "preparation.py",
            )
        },
    )
    fingerprint = hashlib.sha256(
        json.dumps([clock.market, symbol, hashes, policy], sort_keys=True).encode()
    ).hexdigest()
    folder = destination / clock.market / symbol
    if folder.is_symlink() or not folder.resolve().is_relative_to(destination.resolve()):
        raise ValueError("La ruta de salida no pertenece al destino")
    for name in (
        ".asset.lock",
        "configuration.json",
        "manifest.json",
        "prices.parquet",
        "fundamentals.parquet",
        "news",
    ):
        if (folder / name).is_symlink():
            raise ValueError("Un artefacto de salida es un enlace")
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (folder / ".asset.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configuration = folder / "configuration.json"
        identity = dict(fingerprint=fingerprint, policy=policy, sources=hashes)
        if configuration.exists():
            if json.loads(configuration.read_text()) != identity:
                raise ValueError("La configuración corresponde a otra edición del activo")
        elif any(p.name != ".asset.lock" for p in folder.iterdir()):
            raise ValueError("La carpeta contiene una edición no identificada")
        else:
            atomic_json(configuration, identity)
        manifest_path = folder / "manifest.json"
        expected = {
            "prices.parquet",
            "fundamentals.parquet",
            "news/news.parquet",
            "news/excluded.parquet",
            "news/manifest.json",
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if (
                previous.get("fingerprint") != fingerprint
                or set(previous.get("artifacts", {})) != expected
                or previous.get("schema_version") != 3
                or previous.get("cohort_id") != cohort
                or previous.get("news_content_policy") != COHORT_POLICIES[cohort]
                or previous.get("market") != clock.market
                or previous.get("symbol") != symbol
                or previous.get("training_ready") is not False
                or previous.get("policy") != policy
                or previous.get("sources") != hashes
                or set(previous.get("counts", {})) != {"prices", "news", "fundamentals"}
                or any(type(n) is not int or n < 0 for n in previous["counts"].values())
            ):
                raise ValueError("El recibo no corresponde a esta edición")
            if all(
                (folder / name).is_file()
                and not (folder / name).is_symlink()
                and sha256(folder / name) == digest
                for name, digest in previous["artifacts"].items()
            ):
                for kind, relative in (
                    ("prices", "prices.parquet"),
                    ("fundamentals", "fundamentals.parquet"),
                    ("news", "news/news.parquet"),
                ):
                    with pq.ParquetFile(folder / relative) as table:
                        if table.metadata.num_rows != previous["counts"][kind]:
                            raise ValueError("El recuento del recibo difiere de los Parquet")
                return {**previous, "reused": True}
        news = write_cohort_news(
            source,
            asset["paths"]["news"],
            folder / "news",
            symbol=symbol,
            clock=clock,
            cohort=cohort,
            reviews=reviews,
            cutoff=cutoff,
        )
        prices, price_audit = read_prices(paths[asset["paths"]["prices"][0]], clock)
        reserved_prices = int((prices.session > cutoff).sum())
        prices = prices.loc[prices.session <= cutoff].reset_index(drop=True)
        facts, fact_audit = read_fundamentals(
            [paths[p] for p in asset["paths"]["fundamentals"]], clock.market, clock
        )
        reserved_facts = sum(row["available_at"].date() > end for row in facts)
        facts = [row for row in facts if row["available_at"].date() <= end]
        atomic_parquet(
            folder / "prices.parquet", pa.Table.from_pandas(prices, preserve_index=False)
        )
        atomic_parquet(
            folder / "fundamentals.parquet", pa.Table.from_pylist(facts, schema=FACT_SCHEMA)
        )
        if any(sha256(path) != hashes[name] for name, path in paths.items()):
            raise ValueError("Las fuentes cambiaron durante la preparación")
        report = dict(
            schema_version=3,
            market=clock.market,
            symbol=symbol,
            cohort_id=cohort,
            news_content_policy=COHORT_POLICIES[cohort],
            fingerprint=fingerprint,
            policy=policy,
            sources=hashes,
            artifacts={name: sha256(folder / name) for name in sorted(expected)},
            counts=dict(
                prices=len(prices), news=news["counts"]["accepted"], fundamentals=len(facts)
            ),
            reserved_counts=dict(prices=reserved_prices, fundamentals=reserved_facts),
            price_audit=price_audit,
            fundamentals_audit=fact_audit,
            news_counts=news["counts"],
            news_reasons=news["reasons"],
            chart_policy="regenerate_from_past_prices",
            training_ready=False,
            elapsed_seconds=time.perf_counter() - started,
            reused=False,
        )
        atomic_json(manifest_path, report)
        return report
