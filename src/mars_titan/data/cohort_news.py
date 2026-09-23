"""Noticias por cohorte, ordenadas en disco y sin acumular sus cuerpos en RAM."""

import fcntl
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa

from .batches import atomic_parquet_batches
from .corpus_catalog import _read_source, _source_path, _unique_object
from .news import _publication, _text
from .news_audit import EXCLUSION_SCHEMA, NEWS_SCHEMA
from .news_reviews import reviewed_body
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

COHORT_POLICIES = {
    "original_audited": "source_audited_not_external",
    "externally_verified": "verified_full_articles",
}
SCHEMA = (
    NEWS_SCHEMA.append(pa.field("cohort_id", pa.string()))
    .append(pa.field("content_kind", pa.string()))
    .append(pa.field("quality_flags", pa.list_(pa.string())))
)
TIMESTAMPS = ("event_at", "published_at", "available_at")


def _normalise(raw, provenance, symbol, clock, cohort, reviews, lag):
    declared = _text(raw.get("Stock_symbol")).upper().replace(".SH", ".SS")
    if declared and declared != symbol:
        return None, "symbol_mismatch"
    if not declared and cohort == "externally_verified":
        return None, "missing_symbol_evidence"
    body = _text(raw.get("Article"))
    content_kind = "article_candidate" if body else "summary"
    body = body or _text(raw.get("summary"))
    if not body:
        return None, "missing_content"
    title = _text(raw.get("Article_title", raw.get("title")))
    source_date = _text(raw.get("Date", raw.get("datetime")))
    if not source_date:
        return None, "missing_publication"
    review = reviews.get(provenance["source_record_hash"])
    if review and review["status"] == "rejected":
        return None, "content_rejected"
    metadata = dict(
        content_review="source_audited_not_external", historical_body_version_verified=False
    )
    flags = [] if title else ["missing_title"]
    if cohort == "externally_verified":
        body, metadata, reason = reviewed_body(raw, provenance["source_record_hash"], reviews)
        if reason:
            return None, reason
        content_kind = "verified_full_article"
    published, available, reason = _publication(source_date, clock, lag)
    rule = "next_session_close" if published is None else "source_timestamp"
    if reason == "unverified_timezone" and cohort == "original_audited":
        parsed = datetime.fromisoformat(source_date)
        if parsed.tzinfo is None:
            available = clock.date_available(parsed.date().isoformat(), lag=lag)
            rule, reason = "declared_date_without_verified_timezone", None
            flags.append("unverified_timezone")
    if reason:
        return None, reason
    text = "\n".join(filter(None, [title, body]))
    digest = hashlib.sha256(text.encode()).hexdigest()
    url = _text(raw.get("Url", raw.get("url")))
    identity = (digest, url, published.isoformat() if published else source_date)
    return {
        **provenance,
        **metadata,
        "source_date": source_date,
        "event_id": hashlib.sha256(
            json.dumps([symbol, *identity], separators=(",", ":")).encode()
        ).hexdigest(),
        "text": text,
        "content_hash": digest,
        "url": url,
        "language": _text(raw.get("language", raw.get("lang"))) or None,
        "event_at": published,
        "published_at": published,
        "available_at": available,
        "availability_rule": rule,
        "symbol": symbol,
        "association_evidence": "source_field" if declared else "inventory_file_assignment",
        "cohort_id": cohort,
        "content_kind": content_kind,
        "quality_flags": flags,
    }, None


def _tables(db, table, schema, batch_rows, max_bytes):
    order = "available_at,content_hash,event_id" if table == "accepted" else "rowid"
    rows, size = [], 0
    for (payload,) in db.execute(f"SELECT payload FROM {table} ORDER BY {order}"):
        width = len(payload.encode()) + 1024
        if rows and (len(rows) == batch_rows or size + width > max_bytes):
            yield pa.Table.from_pylist(rows, schema=schema)
            rows, size = [], 0
        row = json.loads(payload)
        for name in TIMESTAMPS:
            if row.get(name) is not None:
                row[name] = datetime.fromisoformat(row[name])
        rows.append(row)
        size += width
    if rows:
        yield pa.Table.from_pylist(rows, schema=schema)
    else:
        yield pa.Table.from_pylist([], schema=schema)


def write_cohort_news(
    source: Path,
    paths: list[str],
    output: Path,
    *,
    symbol: str,
    clock: MarketClock,
    cohort: str,
    reviews: dict | None = None,
    cutoff: str = "2023-12-31",
    date_only_lag: int = 1,
    batch_rows: int = 512,
    max_record_bytes: int = 1024**2,
    max_batch_bytes: int = 8 * 1024**2,
    spool_bytes: int = 2 * 1024**3,
) -> dict:
    """Confirmar ambos Parquet y su recibo después de contrastar las fuentes."""
    if cohort not in COHORT_POLICIES:
        raise ValueError("La cohorte editorial no está admitida")
    if cohort == "externally_verified" and reviews is None:
        raise ValueError("La cohorte estricta necesita un registro de revisiones explícito")
    if (
        not isinstance(symbol, str)
        or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
        or symbol in {".", ".."}
        or not paths
        or len(set(paths)) != len(paths)
        or any(
            type(n) is not int or n < 1
            for n in (date_only_lag, batch_rows, max_record_bytes, max_batch_bytes, spool_bytes)
        )
        or max_record_bytes > 1024**2
        or max_batch_bytes < 4 * max_record_bytes
        or spool_bytes < 1024**2
    ):
        raise ValueError("La identidad o los presupuestos editoriales no son válidos")
    end = date.fromisoformat(cutoff)
    outside_source(source, output)
    if output.is_symlink() or any(
        (output / name).is_symlink()
        for name in (
            ".preparation.lock",
            "configuration.json",
            "manifest.json",
            "news.parquet",
            "excluded.parquet",
        )
    ):
        raise ValueError("La salida no puede ser un enlace")
    inputs = {relative: _source_path(source, relative) for relative in sorted(paths)}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    configuration = dict(
        cohort_id=cohort,
        symbol=symbol,
        market=clock.market,
        cutoff=cutoff,
        sources=hashes,
        source_root=str(source.resolve()),
        date_only_lag=date_only_lag,
        batch_rows=batch_rows,
        max_record_bytes=max_record_bytes,
        max_batch_bytes=max_batch_bytes,
        spool_bytes=spool_bytes,
        pyarrow=pa.__version__,
        calendar=hashlib.sha256(
            "|".join(t.isoformat() for t in clock.decisions).encode()
        ).hexdigest(),
        reviews=hashlib.sha256(json.dumps(reviews or {}, sort_keys=True).encode()).hexdigest(),
        code={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "cohort_news.py",
                "corpus_catalog.py",
                "news.py",
                "news_reviews.py",
                "temporal.py",
                "batches.py",
            )
        },
    )
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".preparation.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path, receipt_path = output / "configuration.json", output / "manifest.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != configuration:
                raise ValueError("La configuración pertenece a otra edición editorial")
        elif any(p.name != ".preparation.lock" for p in output.iterdir()):
            raise ValueError("La salida contiene archivos de otra edición")
        else:
            atomic_json(config_path, configuration)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                set(receipt.get("artifacts", {})) != {"news.parquet", "excluded.parquet"}
                or receipt.get("cohort_id") != cohort
                or receipt.get("news_content_policy") != COHORT_POLICIES[cohort]
                or set(receipt.get("counts", {})) != {"records", "accepted", "excluded"}
                or any(type(n) is not int or n < 0 for n in receipt["counts"].values())
                or receipt["counts"]["records"]
                != receipt["counts"]["accepted"] + receipt["counts"]["excluded"]
                or sum(receipt.get("reasons", {}).values()) != receipt["counts"]["excluded"]
            ):
                raise ValueError("El recibo editorial está incompleto o no concilia")
            if receipt["fingerprint"] != identity:
                raise ValueError("La identidad editorial ha cambiado")
            if all(
                (output / name).is_file() and sha256(output / name) == digest
                for name, digest in receipt["artifacts"].items()
            ):
                return {**receipt, "reused": True}
        counts, reasons = Counter(records=0, accepted=0, excluded=0), Counter()
        with tempfile.TemporaryDirectory(prefix=".news-sort-", dir=output) as temporary:
            with closing(sqlite3.connect(Path(temporary) / "news.sqlite")) as db:
                db.execute("PRAGMA trusted_schema=OFF")
                db.execute("PRAGMA cache_size=-8192")
                db.execute("PRAGMA temp_store=FILE")
                page_size = db.execute("PRAGMA page_size").fetchone()[0]
                db.execute(f"PRAGMA max_page_count={spool_bytes // page_size}")
                db.executescript("""
                    CREATE TABLE accepted(event_id TEXT PRIMARY KEY, available_at TEXT NOT NULL,
                        content_hash TEXT NOT NULL, payload TEXT NOT NULL);
                    CREATE INDEX ordering ON accepted(available_at,content_hash,event_id);
                    CREATE TABLE excluded(payload TEXT NOT NULL);
                """)
                for relative, path in inputs.items():
                    with path.open("rb") as raw_stream:
                        for item in _read_source(
                            path, symbol, cutoff, max_record_bytes, hashes[relative]
                        ):
                            (
                                ordinal,
                                offset,
                                length,
                                raw_hash,
                                record_hash,
                                source_date,
                                *_,
                                state,
                            ) = item
                            counts["records"] += 1
                            provenance = dict(
                                source_file=relative,
                                line=ordinal,
                                source_record_hash=record_hash.hex() if record_hash else None,
                                source_date=source_date,
                            )
                            row, reason = None, None
                            if state not in {"pending_verification", "needs_provenance"}:
                                reason = state
                            else:
                                raw_stream.seek(offset)
                                payload = raw_stream.read(length)
                                if hashlib.sha256(payload).digest() != raw_hash:
                                    raise ValueError(
                                        "La huella del registro difiere entre lecturas"
                                    )
                                raw = json.loads(
                                    payload.decode("utf-8-sig"),
                                    parse_constant=lambda _: None,
                                    object_pairs_hook=_unique_object,
                                )
                                try:
                                    row, reason = _normalise(
                                        raw,
                                        provenance,
                                        symbol,
                                        clock,
                                        cohort,
                                        reviews or {},
                                        date_only_lag,
                                    )
                                    if row and row["available_at"].date() > end:
                                        row, reason = None, "availability_after_cutoff"
                                except (ValueError, TypeError, OverflowError) as error:
                                    reason = "invalid_record"
                                    provenance["detail"] = str(error)
                            if row is not None:
                                payload = json.dumps(
                                    row,
                                    default=lambda t: t.isoformat(),
                                    ensure_ascii=False,
                                    allow_nan=False,
                                )
                                inserted = db.execute(
                                    "INSERT OR IGNORE INTO accepted VALUES (?,?,?,?)",
                                    (
                                        row["event_id"],
                                        row["available_at"].isoformat(timespec="microseconds"),
                                        row["content_hash"],
                                        payload,
                                    ),
                                ).rowcount
                                if inserted:
                                    counts["accepted"] += 1
                                else:
                                    reason = "duplicate"
                            if reason:
                                db.execute(
                                    "INSERT INTO excluded VALUES (?)",
                                    (
                                        json.dumps(
                                            {**provenance, "reason": reason}, ensure_ascii=False
                                        ),
                                    ),
                                )
                                counts["excluded"] += 1
                                reasons[reason] += 1
                            if counts["records"] % 128 == 0:
                                db.commit()
                db.commit()
                for table, filename, schema in (
                    ("accepted", "news.parquet", SCHEMA),
                    ("excluded", "excluded.parquet", EXCLUSION_SCHEMA),
                ):
                    atomic_parquet_batches(
                        output / filename, _tables(db, table, schema, batch_rows, max_batch_bytes)
                    )
        if any(sha256(path) != hashes[name] for name, path in inputs.items()):
            raise ValueError("Una fuente cambió durante la preparación editorial")
        receipt = dict(
            schema_version=1,
            fingerprint=identity,
            cohort_id=cohort,
            news_content_policy=COHORT_POLICIES[cohort],
            market=clock.market,
            symbol=symbol,
            counts=dict(counts),
            reasons=dict(reasons),
            reused=False,
            artifacts={
                name: sha256(output / name) for name in ("news.parquet", "excluded.parquet")
            },
        )
        atomic_json(receipt_path, receipt)
        return receipt
