"""Cola editorial completa, independiente del índice y de los originales."""

import fcntl
import hashlib
import json
import os
import re
import resource
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from .corpus_catalog import _readonly, _source_path
from .news_fetch import ArticleFetcher, SourceUnavailable, _address
from .news_reviews import _validate_review, load_reviews, reviewed_body
from .news_sources import parse_fool
from .news_verification import compare_article
from .storage import outside_source, sha256

_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE work (
    file_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, market TEXT NOT NULL,
    symbol TEXT NOT NULL, record_hash BLOB, state TEXT NOT NULL, reason TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
    review TEXT, PRIMARY KEY(file_id, ordinal)
) WITHOUT ROWID;
CREATE INDEX work_asset ON work(symbol, market);
CREATE INDEX work_hash ON work(record_hash);
"""


def _policy_hash():
    names = ("news_sources.py", "news_verification.py", "news_registry.py", "news_fetch.py")
    return hashlib.sha256(
        "".join(sha256(Path(__file__).with_name(n)) for n in names).encode()
    ).hexdigest()


def _connect(database):
    db = sqlite3.connect(database, uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA trusted_schema=OFF")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    return db


def _bind(db, metadata):
    catalog = Path(metadata["catalog_path"])
    if _catalog_digest(catalog) != metadata["catalog_sha256"]:
        raise ValueError("El índice cambió después de confirmar la cola editorial")
    if _policy_hash() != metadata["policy_sha256"]:
        raise ValueError("La política editorial cambió y necesita un registro nuevo")
    db.execute("ATTACH DATABASE ? AS corpus", (catalog.as_uri() + "?mode=ro",))


def _catalog_digest(catalog):
    journal = Path(str(catalog) + "-wal")
    if journal.exists() and journal.stat().st_size:
        raise ValueError("El índice contiene cambios WAL sin consolidar")
    return sha256(catalog)


def _read_record(db, metadata, file_id, ordinal):
    row = db.execute(
        "SELECT r.*,f.path FROM corpus.records r JOIN corpus.source_files f "
        "ON f.id=r.file_id WHERE r.file_id=? AND r.ordinal=?",
        (file_id, ordinal),
    ).fetchone()
    source = Path(metadata["source_root"])
    path = _source_path(source, os.fsdecode(row["path"]))
    if row["byte_length"] > 1024**2:
        raise ValueError("El registro supera el límite de lectura editorial")
    with path.open("rb") as stream:
        stream.seek(row["byte_offset"])
        payload = stream.read(row["byte_length"])
    if hashlib.sha256(payload).digest() != row["raw_sha256"]:
        raise ValueError("El registro original ha cambiado desde su indexación")
    text = payload.decode("utf-8-sig").rstrip("\r\n")
    if hashlib.sha256(text.encode()).digest() != row["source_record_hash"]:
        raise ValueError("La identidad textual no coincide con el original")
    return json.loads(text, parse_constant=lambda _: None), row


def initialize_verification(
    catalog: Path, database: Path, *, manual_reviews: Path | None = None
) -> dict:
    """Copiar todos los localizadores y conservar íntegramente las revisiones previas."""
    catalog = catalog.resolve()
    if database.is_symlink() or database.resolve() == catalog:
        raise ValueError("El registro editorial no puede sustituir el índice")
    with closing(_readonly(catalog)) as source:
        config = dict(source.execute("SELECT key,value FROM metadata"))
        if source.execute("SELECT count(*) FROM source_files WHERE completed=0").fetchone()[0]:
            raise ValueError("El índice debe estar completo antes de crear la cola editorial")
    outside_source(Path(config["source_root"]), database)
    metadata = {
        "schema_version": "1",
        "catalog_path": str(catalog),
        "catalog_sha256": _catalog_digest(catalog),
        "source_root": config["source_root"],
        "policy_sha256": _policy_hash(),
        "manual_sha256": sha256(manual_reviews) if manual_reviews else "",
    }
    manual = load_reviews(manual_reviews) if manual_reviews else {}
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(database)) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables:
            if (
                tables != {"metadata", "work"}
                or dict(db.execute("SELECT * FROM metadata")) != metadata
            ):
                raise ValueError("El registro existente pertenece a otra configuración")
            return verification_status(database)
        _bind(db, metadata)
        db.executescript("BEGIN IMMEDIATE;" + _SCHEMA)
        try:
            db.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
            db.execute("""
                INSERT INTO work(file_id,ordinal,market,symbol,record_hash,state)
                SELECT r.file_id,r.ordinal,a.market,a.symbol,r.source_record_hash,
                       CASE r.state WHEN 'pending_verification' THEN 'pending' ELSE r.state END
                FROM corpus.records r JOIN corpus.source_files f ON r.file_id=f.id
                JOIN corpus.assets a ON f.asset_id=a.asset_id
            """)
            for key, review in manual.items():
                matches = db.execute(
                    "SELECT w.* FROM work w JOIN corpus.source_files f ON f.id=w.file_id "
                    "WHERE w.record_hash=? AND w.symbol=? AND f.path=? AND w.ordinal=?",
                    (
                        bytes.fromhex(key),
                        review["symbol"],
                        os.fsencode(review["source_file"]),
                        review["source_row"],
                    ),
                ).fetchall()
                if not matches:
                    raise ValueError("Una revisión manual no pertenece al índice confirmado")
                for item in matches:
                    raw, row = _read_record(db, metadata, item["file_id"], item["ordinal"])
                    if (
                        raw.get("Stock_symbol", "").strip().upper().replace(".SH", ".SS")
                        != review["symbol"]
                        or raw.get("Date") != review["source_date"]
                        or raw.get("Url") != review["source_url"]
                        or (
                            review["status"] == "verified_full_article"
                            and reviewed_body(raw, key, manual)[2]
                        )
                    ):
                        raise ValueError("La revisión manual no corresponde al contenido original")
                    state = item["state"] if item["state"] == "reserved" else review["status"]
                    db.execute(
                        "UPDATE work SET state=?,review=? WHERE file_id=? AND ordinal=?",
                        (
                            state,
                            json.dumps(review),
                            row["file_id"],
                            row["ordinal"],
                        ),
                    )
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return verification_status(database)


def candidate_urls(raw):
    """Proponer direcciones del editor, sin dar su existencia o contenido por comprobados."""
    address = urlsplit(raw.get("Url", ""))
    if address.scheme != "https" or address.username or address.password:
        return ()
    if address.netloc == "www.fool.com":
        return (raw["Url"],)
    body = raw.get("Article", "")
    if (
        address.netloc != "www.nasdaq.com"
        or not address.path.startswith("/articles/")
        or not isinstance(body, str)
        or "The Motley Fool" not in body
    ):
        return ()
    day = raw.get("Date", "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        return ()
    slug = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", address.path.removeprefix("/articles/"))
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        return ()
    base = f"https://www.fool.com/investing/{day.replace('-', '/')}/{slug}"
    return (base + ".aspx", base + "/") if day < "2020" else (base + "/", base + ".aspx")


def _verify(raw, fetcher, network):
    urls = candidate_urls(raw)
    if not urls:
        return {
            "state": "needs_provenance",
            "reason": "unresolved_publisher",
            "review": None,
            "retry_at": 0,
        }
    failures = []
    for url in urls:
        try:
            _address(url)
        except ValueError:
            return {
                "state": "unverifiable",
                "reason": "unsupported_url",
                "review": None,
                "retry_at": 0,
            }
        capture = fetcher.cached(url)
        if capture is None and not network:
            failures.append(SourceUnavailable("missing_capture", time.time() + 3600))
            continue
        try:
            capture = capture or fetcher.fetch(url)
            evidence = parse_fool(capture.body, capture.url)
        except SourceUnavailable as error:
            failures.append(error)
            continue
        except ValueError:
            return {
                "state": "unverifiable",
                "reason": "unsupported_capture",
                "review": None,
                "retry_at": 0,
            }
        review = compare_article(raw, evidence)
        review["checked_at"] = capture.fetched_at
        return {
            "state": review["status"],
            "reason": review.get("reason"),
            "review": review,
            "retry_at": 0,
        }
    failure = min(failures, key=lambda error: error.retry_at)
    return {
        "state": "retry",
        "reason": failure.reason,
        "review": None,
        "retry_at": failure.retry_at,
    }


def verify_pending(
    database: Path,
    evidence_root: Path,
    *,
    stop_after: int | None = None,
    network=False,
    quota_bytes=2 * 1024**3,
) -> dict:
    """Recorrer la cola en bloques, con confirmación por registro y sin abrir el test final."""
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("El corte operativo debe ser un entero positivo")
    if not database.is_file() or database.is_symlink():
        raise ValueError("Falta un registro editorial existente")
    started, processed, cursor, now = time.perf_counter(), 0, (0, 0), time.time()
    with database.open("rb") as lock, closing(_connect(database)) as db:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = dict(db.execute("SELECT * FROM metadata"))
        _bind(db, metadata)
        outside_source(Path(metadata["source_root"]), evidence_root)
        fetcher = ArticleFetcher(evidence_root, quota_bytes=quota_bytes)
        while stop_after is None or processed < stop_after:
            limit = min(128, stop_after - processed) if stop_after else 128
            batch = db.execute(
                "SELECT file_id,ordinal FROM work WHERE (file_id,ordinal)>(?,?) "
                "AND (state='pending' OR (state='retry' AND retry_at<=?)) "
                "ORDER BY file_id,ordinal LIMIT ?",
                (*cursor, now, limit),
            ).fetchall()
            if not batch:
                break
            for item in batch:
                cursor = tuple(item)
                raw, row = _read_record(db, metadata, *cursor)
                result = _verify(raw, fetcher, network)
                review = result["review"]
                if review is not None:
                    review.update(
                        {
                            "source_record_hash": row["source_record_hash"].hex(),
                            "source_file": os.fsdecode(row["path"]),
                            "source_row": row["ordinal"],
                            "symbol": raw.get("Stock_symbol", "")
                            .strip()
                            .upper()
                            .replace(".SH", ".SS"),
                            "source_date": raw.get("Date"),
                            "source_url": raw.get("Url"),
                        }
                    )
                    _validate_review(review)
                db.execute(
                    "UPDATE work SET state=?,reason=?,review=?,retry_at=?,attempts=attempts+1 "
                    "WHERE file_id=? AND ordinal=?",
                    (
                        result["state"],
                        result["reason"],
                        json.dumps(review) if review else None,
                        result["retry_at"],
                        *cursor,
                    ),
                )
                db.commit()
                processed += 1
    return {
        **verification_status(database),
        "processed": processed,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def reviews_for_asset(database: Path, symbol: str, *, market: str = "US") -> dict[str, dict]:
    """Leer únicamente las revisiones de un activo, hasta 128 MiB de JSON."""
    with closing(_readonly(database)) as db:
        where = "symbol=? AND market=? AND review IS NOT NULL AND state!='reserved'"
        size = db.execute(
            f"SELECT coalesce(sum(length(review)),0) FROM work WHERE {where}", (symbol, market)
        ).fetchone()[0]
        if size > 128 * 1024**2:
            raise ValueError("Las revisiones del activo superan el límite de lectura")
        result = {}
        for (payload,) in db.execute(f"SELECT review FROM work WHERE {where}", (symbol, market)):
            review = json.loads(payload)
            _validate_review(review)
            result[review["source_record_hash"]] = review
    return result


def verification_status(database: Path) -> dict:
    with closing(_readonly(database)) as db:
        states = dict(db.execute("SELECT state,count(*) FROM work GROUP BY state"))
        reasons = dict(
            db.execute("SELECT reason,count(*) FROM work WHERE reason IS NOT NULL GROUP BY reason")
        )
        markets = {
            row[0]: row[1] for row in db.execute("SELECT market,count(*) FROM work GROUP BY market")
        }
    return {
        "schema_version": 1,
        "records": sum(states.values()),
        "states": states,
        "reasons": reasons,
        "markets": markets,
        "training_ready": False,
    }
