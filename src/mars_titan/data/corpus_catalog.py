"""Censo completo e índice textual recuperable, sin admisión editorial implícita."""

import hashlib
import json
import os
import re
import resource
import sqlite3
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .inventory import classify
from .storage import outside_source, sha256

MODALITIES = ("prices", "news", "fundamentals", "charts")
_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE assets (
    asset_id TEXT PRIMARY KEY, market TEXT NOT NULL, symbol TEXT NOT NULL,
    has_all_sources INTEGER NOT NULL, missing_modalities TEXT NOT NULL
);
CREATE TABLE source_errors (
    asset_id TEXT NOT NULL REFERENCES assets(asset_id), path BLOB PRIMARY KEY,
    modality TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE source_files (
    id INTEGER PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    path BLOB UNIQUE NOT NULL, expected_sha256 TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0, records INTEGER NOT NULL DEFAULT 0,
    bytes INTEGER NOT NULL DEFAULT 0, states TEXT NOT NULL DEFAULT '{}',
    confirmed_at TEXT
);
CREATE TABLE records (
    file_id INTEGER NOT NULL REFERENCES source_files(id), ordinal INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL,
    raw_sha256 BLOB NOT NULL, source_record_hash BLOB, source_date TEXT,
    declared_day TEXT, title TEXT, url TEXT, body_sha256 BLOB,
    content_kind TEXT, state TEXT NOT NULL,
    PRIMARY KEY(file_id, ordinal)
) WITHOUT ROWID;
CREATE INDEX record_evidence_identity ON records(source_record_hash);
CREATE INDEX record_url_body ON records(url, body_sha256);
"""


def _source_path(source: Path, relative: str) -> Path:
    path = source / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.is_symlink()
        or path.resolve() != path
        or not path.is_file()
    ):
        raise ValueError("La ruta de una fuente no es un archivo regular dentro del origen")
    return path


def _readonly(database: Path):
    if not database.is_file() or database.is_symlink():
        raise ValueError("La base de datos debe ser un archivo existente y no un enlace")
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA trusted_schema=OFF")
    return db


def corpus_candidates(source: Path, inventory: Path, market: str) -> list[dict]:
    """Conservar todos los instrumentos del inventario, incluso los incompletos."""
    source = source.resolve()
    if market not in {"US", "CN"} or not source.is_dir():
        raise ValueError("El mercado o el directorio de origen no son válidos")
    assets = {}
    with closing(_readonly(inventory)) as db:
        root = db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()
        if root is None or root[0] != str(source):
            raise ValueError("El inventario pertenece a otro origen")
        for (payload,) in db.execute("SELECT record FROM files WHERE present=1 ORDER BY path"):
            row = json.loads(payload)
            if row.get("market") != market:
                continue
            relative, symbol, modality = row["path"], row["symbol"], row["modality"]
            if classify(Path(relative)) != (market, modality, symbol):
                raise ValueError("La identidad del inventario no coincide con su ruta")
            if (
                modality not in MODALITIES
                or symbol in {".", ".."}
                or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
            ):
                raise ValueError("La identidad del instrumento no es válida")
            asset = assets.setdefault(
                symbol,
                {
                    "market": market,
                    "symbol": symbol,
                    "paths": {name: [] for name in MODALITIES},
                    "hashes": {},
                    "source_errors": [],
                },
            )
            if row.get("state") == "error":
                asset["source_errors"].append(
                    dict(
                        path=relative,
                        modality=modality,
                        reason=str(row.get("error") or "No se pudo inspeccionar la fuente"),
                    )
                )
                continue
            _source_path(source, relative)
            if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
                raise ValueError("La fuente no tiene una huella válida")
            asset["paths"][modality].append(relative)
            asset["hashes"][relative] = row["sha256"]
    result = []
    for symbol in sorted(assets):
        asset = assets[symbol]
        missing = [name for name in MODALITIES if not asset["paths"][name]]
        result.append({**asset, "missing_modalities": missing, "has_all_sources": not missing})
    return result


def _catalog(source, assets):
    if not assets:
        raise ValueError("El catálogo debe contener instrumentos")
    identities, paths, normalized = set(), set(), []
    for asset in sorted(assets, key=lambda row: (row["market"], row["symbol"])):
        market, symbol = asset["market"], asset["symbol"]
        key = f"{market}/{symbol}"
        if market not in {"US", "CN"} or key in identities:
            raise ValueError("Hay mercados inválidos o identidades duplicadas")
        identities.add(key)
        sources = {name: sorted(asset["paths"].get(name, [])) for name in MODALITIES}
        hashes = {}
        for modality, names in sources.items():
            for relative in names:
                _source_path(source, relative)
                if relative in paths or classify(Path(relative)) != (market, modality, symbol):
                    raise ValueError("Hay fuentes duplicadas o con identidad incompatible")
                paths.add(relative)
                digest = asset["hashes"][relative]
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("La fuente no tiene una huella válida")
                hashes[relative] = digest
        missing = [name for name in MODALITIES if not sources[name]]
        errors = sorted(asset.get("source_errors", []), key=lambda row: row["path"])
        for error in errors:
            relative = error["path"]
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in paths
                or error["modality"] not in MODALITIES
                or classify(Path(relative)) != (market, error["modality"], symbol)
                or not isinstance(error["reason"], str)
                or not error["reason"].strip()
            ):
                raise ValueError("La fuente fallida no tiene una identidad y un motivo válidos")
            paths.add(relative)
        normalized.append(
            dict(
                asset_id=key,
                market=market,
                symbol=symbol,
                paths=sources,
                hashes=hashes,
                missing_modalities=missing,
                has_all_sources=not missing,
                source_errors=errors,
            )
        )
    return normalized


def _initialize(db, config, assets):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables:
        if tables != {"metadata", "assets", "source_files", "records", "source_errors"}:
            raise ValueError("La base de datos contiene otro formato y se conserva sin cambios")
        existing = dict(db.execute("SELECT key,value FROM metadata"))
        if existing != config:
            raise ValueError("La configuración del índice no coincide con la confirmada")
        return
    db.executescript("BEGIN IMMEDIATE;" + _SCHEMA)
    try:
        db.executemany("INSERT INTO metadata VALUES (?,?)", config.items())
        for asset in assets:
            db.execute(
                "INSERT INTO assets VALUES (?,?,?,?,?)",
                (
                    asset["asset_id"],
                    asset["market"],
                    asset["symbol"],
                    asset["has_all_sources"],
                    json.dumps(asset["missing_modalities"]),
                ),
            )
            db.executemany(
                "INSERT INTO source_errors VALUES (?,?,?,?)",
                (
                    (
                        asset["asset_id"],
                        os.fsencode(error["path"]),
                        error["modality"],
                        error["reason"],
                    )
                    for error in asset["source_errors"]
                ),
            )
            db.executemany(
                "INSERT INTO source_files(asset_id,path,expected_sha256) VALUES (?,?,?)",
                (
                    (asset["asset_id"], os.fsencode(name), asset["hashes"][name])
                    for name in asset["paths"]["news"]
                ),
            )
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("El objeto JSON contiene claves duplicadas")
        result[key] = value
    return result


def _fields(payload, symbol, cutoff):
    values = dict(
        source_record_hash=None,
        source_date=None,
        declared_day=None,
        title=None,
        url=None,
        body_sha256=None,
        content_kind=None,
        state="pending_verification",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {**values, "state": "invalid_utf8"}
    values["source_record_hash"] = hashlib.sha256(text.rstrip("\r\n").encode()).digest()
    try:
        raw = json.loads(text, parse_constant=lambda _: None, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        return {**values, "state": "invalid_json"}
    if not isinstance(raw, dict):
        return {**values, "state": "invalid_record"}

    def clean(value):
        return value.strip() if isinstance(value, str) else ""

    title = clean(raw.get("Article_title", raw.get("title")))
    published = clean(raw.get("Date", raw.get("datetime")))
    url = clean(raw.get("Url", raw.get("url")))
    body = clean(raw.get("Article"))
    kind = "article_candidate" if body else "summary"
    body = body or clean(raw.get("summary"))
    if len(title) > 8192 or len(url) > 8192 or len(published) > 64:
        return {**values, "state": "metadata_too_large"}
    try:
        for value in (title, published, url, body):
            value.encode("utf-8")
    except UnicodeEncodeError:
        return {**values, "state": "invalid_unicode"}
    try:
        parsed = urlsplit(url)
        valid_url = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )
    except ValueError:
        valid_url = False
    try:
        day = datetime.fromisoformat(published).date().isoformat()
    except ValueError:
        day = None
    values.update(
        source_date=published or None,
        declared_day=day,
        title=title or None,
        url=url if valid_url else None,
        body_sha256=hashlib.sha256(body.encode()).digest() if body else None,
        content_kind=kind if body else None,
    )
    declared = clean(raw.get("Stock_symbol")).upper().replace(".SH", ".SS")
    if day is not None and day > cutoff:
        values["state"] = "reserved"
    elif not body:
        values["state"] = "missing_content"
    elif declared and declared != symbol:
        values["state"] = "symbol_mismatch"
    elif not day or not valid_url or not title or not declared or kind == "summary":
        values["state"] = "needs_provenance"
    return values


def _read_source(path, symbol, cutoff, max_bytes, expected_hash):
    digest, offset, ordinal = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while block := stream.readline(max_bytes + 1):
            ordinal += 1
            length, raw_hash = len(block), hashlib.sha256(block)
            digest.update(block)
            payload = block if length <= max_bytes else None
            while not block.endswith(b"\n") and len(block) == max_bytes + 1:
                block = stream.readline(max_bytes + 1)
                digest.update(block)
                raw_hash.update(block)
                length += len(block)
                payload = None
            if payload is None:
                fields = dict(
                    source_record_hash=None,
                    source_date=None,
                    declared_day=None,
                    title=None,
                    url=None,
                    body_sha256=None,
                    content_kind=None,
                    state="record_too_large",
                )
            else:
                if ordinal == 1 and payload.startswith(b"\xef\xbb\xbf"):
                    payload = payload[3:]
                fields = _fields(payload, symbol, cutoff)
            yield (ordinal, offset, length, raw_hash.digest(), *fields.values())
            offset += length
        after = os.fstat(stream.fileno())
    if digest.hexdigest() != expected_hash or (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"La fuente ha cambiado o no coincide con el inventario: {path.name}")


def index_news(
    source: Path,
    assets: list[dict],
    database: Path,
    *,
    cutoff: str,
    max_record_bytes: int = 1024**2,
) -> dict:
    """Confirmar cada archivo completo. Una interrupción revierte solo ese archivo."""
    source = source.resolve()
    outside_source(source, database)
    if database.is_symlink():
        raise ValueError("La base de datos no puede ser un enlace")
    if type(max_record_bytes) is not int or not 1 <= max_record_bytes <= 16 * 1024**2:
        raise ValueError("El presupuesto de un registro debe estar entre uno y 16 MiB")
    cutoff = date.fromisoformat(cutoff).isoformat()
    catalog = _catalog(source, assets)
    config = {
        "schema_version": "1",
        "source_root": str(source),
        "cutoff": cutoff,
        "max_record_bytes": str(max_record_bytes),
        "indexer_sha256": sha256(Path(__file__)),
        "catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
    }
    started, reused = time.perf_counter(), 0
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database, timeout=5)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA cache_size=-16384")
        db.execute("PRAGMA temp_store=FILE")
        _initialize(db, config, catalog)
        db.execute("PRAGMA journal_mode=WAL")
        files = db.execute(
            "SELECT f.id,f.path,f.expected_sha256,a.symbol FROM source_files f "
            "JOIN assets a ON a.asset_id=f.asset_id ORDER BY f.id"
        ).fetchall()
        for file_id, relative, expected, symbol in files:
            path = _source_path(source, os.fsdecode(relative))
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute(
                    "SELECT completed FROM source_files WHERE id=?", (file_id,)
                ).fetchone()[0]:
                    if sha256(path) != expected:
                        raise ValueError("Una fuente ya confirmada ha cambiado")
                    db.commit()
                    reused += 1
                    continue
                states, count, size = Counter(), 0, 0
                for row in _read_source(path, symbol, cutoff, max_record_bytes, expected):
                    db.execute(
                        "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (file_id, *row)
                    )
                    states[row[-1]] += 1
                    count += 1
                    size += row[2]
                db.execute(
                    "UPDATE source_files SET completed=1,records=?,bytes=?,states=?,confirmed_at=? "
                    "WHERE id=?",
                    (count, size, json.dumps(states), datetime.now(UTC).isoformat(), file_id),
                )
                db.commit()
            except BaseException:
                db.rollback()
                raise
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {
        **corpus_status(database),
        "reused_files": reused,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def corpus_status(database: Path) -> dict:
    """Leer recibos confirmados, sin recorrer todos los registros ni admitir noticias."""
    markets, states, files, completed, records, size = {}, Counter(), 0, 0, 0, 0
    source_errors = Counter()
    with closing(_readonly(database)) as db:
        config = dict(db.execute("SELECT key,value FROM metadata"))
        if config.get("schema_version") != "1":
            raise ValueError("La versión del catálogo no es compatible")
        for market, total, complete in db.execute(
            "SELECT market,count(*),sum(has_all_sources) FROM assets GROUP BY market"
        ):
            markets[market] = dict(
                assets=total, with_four_sources=complete, records=0, candidate_records=0
            )
        for market, eligible, done, count, length, counts in db.execute(
            "SELECT a.market,a.has_all_sources,f.completed,f.records,f.bytes,f.states "
            "FROM source_files f JOIN assets a ON a.asset_id=f.asset_id"
        ):
            files += 1
            completed += done
            records += count
            size += length
            states.update(json.loads(counts))
            markets[market]["records"] += count
            markets[market]["candidate_records"] += count if eligible else 0
        for modality, count in db.execute(
            "SELECT modality,count(*) FROM source_errors GROUP BY modality"
        ):
            source_errors[modality] = count
        files += source_errors["news"]
    return {
        "schema_version": 1,
        "scope": "source_index_not_training_admission",
        "index_complete": files == completed,
        "training_ready": False,
        "source_files": files,
        "completed_files": completed,
        "records": records,
        "confirmed_source_bytes": size,
        "records_by_state": dict(states),
        "errors": sum(source_errors.values()),
        "source_errors_by_modality": dict(source_errors),
        "markets": markets,
        "catalog_sha256": config["catalog_sha256"],
        "indexer_sha256": config["indexer_sha256"],
        "cutoff": config["cutoff"],
        "database_bytes": database.stat().st_size,
    }
