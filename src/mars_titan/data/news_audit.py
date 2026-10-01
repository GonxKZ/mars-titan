"""Auditoría de noticias con derivados separados y recuentos reconciliables."""

import argparse
import hashlib
import json
import re
import resource
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from .news import read_news
from .news_reviews import load_reviews
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

NEWS_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "source_file",
            "source_record_hash",
            "source_date",
            "event_id",
            "text",
            "content_hash",
            "url",
            "language",
            "availability_rule",
            "symbol",
            "association_evidence",
            "content_review",
            "reviewed_body_sha256",
            "review_evidence_url",
            "review_checked_at",
        )
    ]
    + [("line", pa.int64())]
    + [("historical_body_version_verified", pa.bool_())]
    + [
        (name, pa.timestamp("us", tz="UTC"))
        for name in ("event_at", "published_at", "available_at")
    ]
)
EXCLUSION_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "source_file",
            "source_record_hash",
            "source_date",
            "reason",
            "detail",
        )
    ]
    + [("line", pa.int64())]
)


def audit_news_panel(
    source: Path, panel: dict, output: Path, clock: MarketClock, *, reviews: dict | None = None
) -> dict:
    outside_source(source, output)
    if output.exists():
        raise ValueError("Usa un directorio nuevo para conservar las auditorías anteriores")
    if panel["market"] != clock.market or not panel["assets"]:
        raise ValueError("El panel y su calendario deben pertenecer al mismo mercado")
    symbols = set()
    for asset in panel["assets"]:
        symbol = asset["symbol"]
        if not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol) or symbol in {".", ".."}:
            raise ValueError("El símbolo del activo no es seguro")
        if symbol in symbols or not asset["paths"].get("news"):
            raise ValueError("El activo está duplicado o no tiene fuentes de noticias")
        symbols.add(symbol)
        for relative in asset["paths"]["news"]:
            path = source / relative
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                raise ValueError("La ruta de noticias sale del directorio de origen")
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "market": clock.market,
        "calendar": {
            "first_session": clock.days[0].isoformat(),
            "last_session": clock.days[-1].isoformat(),
            "decisions_sha256": hashlib.sha256(
                "|".join(value.isoformat() for value in clock.decisions).encode()
            ).hexdigest(),
        },
        "grain": "asset_source_record",
        "admission_scope": "verified_full_editorial_content"
        if reviews is not None
        else "source_binding_and_publication_fields",
        "admitted_content_reviewed": reviews is not None,
        "reviews_sha256": hashlib.sha256(json.dumps(reviews, sort_keys=True).encode()).hexdigest()
        if reviews is not None
        else None,
        "semantic_validation_complete": False,
        "historical_body_version_verified": False,
        "raw_records": 0,
        "accepted": 0,
        "excluded": 0,
        "reasons": Counter(),
        "assets": [],
        "source_hashes": {},
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("news.py", "news_audit.py", "news_reviews.py")
        },
    }
    shifts = []
    for asset in panel["assets"]:
        admitted, rejected, seen = [], [], set()
        raw_records = 0
        for relative in sorted(asset["paths"]["news"]):
            path = source / relative
            before = sha256(path)
            rows, errors = read_news(path, asset["symbol"], clock, reviews=reviews)
            if sha256(path) != before:
                raise ValueError("La fuente ha cambiado durante la auditoría de noticias")
            report["source_hashes"][relative] = before
            raw_records += len(rows) + len(errors)
            rejected.extend(errors)
            for row in rows:
                if row["event_id"] in seen:
                    rejected.append({**row, "reason": "duplicate_across_files"})
                else:
                    seen.add(row["event_id"])
                    admitted.append(row)
        admitted.sort(key=lambda row: (row["available_at"], row["event_id"]))
        lag2_unavailable = 0
        for row in admitted:
            if row["event_at"] is None:
                try:
                    later = clock.date_available(row["source_date"], lag=2)
                    shifts.append((later - row["available_at"]).total_seconds() / 3600)
                except ValueError:
                    lag2_unavailable += 1
        folder = output / clock.market / asset["symbol"]
        outside_source(source, folder)
        atomic_parquet(folder / "news.parquet", pa.Table.from_pylist(admitted, schema=NEWS_SCHEMA))
        atomic_parquet(
            folder / "exclusions.parquet", pa.Table.from_pylist(rejected, schema=EXCLUSION_SCHEMA)
        )
        reasons = Counter(row["reason"] for row in rejected)
        summary = {
            "symbol": asset["symbol"],
            "raw_records": raw_records,
            "accepted": len(admitted),
            "excluded": len(rejected),
            "reasons": dict(reasons),
            "date_only": sum(row["event_at"] is None for row in admitted),
            "language_unknown": sum(row["language"] is None for row in admitted),
            "lag2_outside_calendar": lag2_unavailable,
            "first_available_at": admitted[0]["available_at"].isoformat() if admitted else None,
            "last_available_at": admitted[-1]["available_at"].isoformat() if admitted else None,
            "artifacts": {
                name: sha256(folder / name) for name in ("news.parquet", "exclusions.parquet")
            },
        }
        assert raw_records == len(admitted) + len(rejected)
        report["assets"].append(summary)
        for name in ("raw_records", "accepted", "excluded"):
            report[name] += summary[name]
        report["reasons"].update(reasons)
    report["reasons"] = dict(report["reasons"])
    report["lag2_shift_hours"] = {"min": min(shifts), "max": max(shifts)} if shifts else None
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    atomic_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/manifests/news-reviews.json"),
        help="Admitir solo cuerpos completos con revisión acreditada",
    )
    parser.add_argument(
        "--fields-only", action="store_true", help="Diagnóstico de campos, sin certificar contenido"
    )
    args = parser.parse_args()
    outside_source(args.source, args.report)
    outside_source(args.output, args.report)
    if args.report.exists():
        raise ValueError("Usa un informe nuevo para conservar los archivos existentes")
    panel = json.loads(args.panel.read_text())
    reviews = None if args.fields_only else load_reviews(args.reviews)
    start = "1990-01-01" if panel["market"] == "US" else "2000-01-01"
    result = audit_news_panel(
        args.source,
        panel,
        args.output,
        MarketClock(panel["market"], start, "2026-01-01"),
        reviews=reviews,
    )
    atomic_json(args.report, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "raw_records",
                    "accepted",
                    "excluded",
                    "elapsed_seconds",
                    "peak_rss_mib",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
