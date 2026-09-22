"""Inventario completo con lectura acotada y recuperación por archivo."""

import codecs
import csv
import hashlib
import json
import os
import re
import resource
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .storage import outside_source, sha256

CLASSIFICATION_VERSION = 1


def _chart_identity(path: Path) -> tuple[str, str, str]:
    """Contrastar la colección, la carpeta del activo y el nombre del gráfico."""
    markets = {
        market
        for segment in path.parts[1:-2]
        for prefix, market in (("hs300_", "CN"), ("s&p500_", "US"))
        if segment.lower().startswith(prefix)
    }
    match = re.fullmatch(
        r"([A-Z0-9.^_=\-]{1,64})_([0-9]{4})_H([12])_candlestick\.png",
        path.name,
        flags=re.IGNORECASE,
    )
    if len(markets) != 1 or match is None:
        raise ValueError("El mercado o el formato del gráfico no queda identificado")
    market = markets.pop()
    names = [path.parent.name.upper(), match[1].upper()]
    if market == "CN":
        names = [name[:-3] + ".SS" if name.endswith(".SH") else name for name in names]
    if names[0] != names[1] or names[0] in {".", ".."}:
        raise ValueError("La carpeta y el nombre del gráfico identifican activos distintos")
    return market, "charts", names[0]


def classify(path: Path) -> tuple[str, str, str | None]:
    parts = path.parts
    if parts and parts[0].lower() == "image":
        if path.suffix.lower() == ".md":
            return "catalog", "metadata", None
        return _chart_identity(path)
    if len(parts) < 3:
        return "catalog", "metadata", None
    market = "CN" if "hs300" in parts[1].lower() else "US"
    modality = {
        "time_series": "prices",
        "table": "fundamentals",
        "text": "news",
        "image": "charts",
    }.get(parts[0], "metadata")
    symbol = parts[2] if modality in {"fundamentals", "charts"} else path.stem
    if market == "CN":
        symbol = symbol.split("_")[0].replace(".SH", ".SS")
    return market, modality, symbol.upper()


def inspect_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(65536)
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return {
            "format": "png",
            "width": int.from_bytes(prefix[16:20], "big"),
            "height": int.from_bytes(prefix[20:24], "big"),
        }
    encoding = "utf-8-sig"
    try:
        text = codecs.getincrementaldecoder(encoding)().decode(prefix, final=False)
    except UnicodeDecodeError:
        encoding = "gb18030"
        text = prefix.decode(encoding, errors="replace")
    if path.suffix.lower() == ".csv":
        return {
            "format": "csv",
            "encoding": encoding,
            "columns": next(csv.reader(text.splitlines()), []),
        }
    return {
        "format": "json_array" if text.lstrip().startswith("[") else path.suffix[1:],
        "encoding": encoding,
        "inspection": "bounded_header_only",
    }


def entries(database: Path):
    with sqlite3.connect(database) as db:
        for row in db.execute("SELECT record FROM files WHERE present=1 ORDER BY path"):
            yield json.loads(row[0])


def inventory(source: Path, database: Path, *, verify: bool = False) -> dict:
    source, database = source.resolve(), database.resolve()
    if not source.is_dir():
        raise ValueError(f"El directorio de origen no existe: {source}")
    outside_source(source, database)
    database.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    inspected_at = datetime.now(UTC).isoformat()
    listing = subprocess.run(
        ["rg", "--files", "--hidden", "--no-ignore", "--null", "."],
        cwd=source,
        capture_output=True,
        check=False,
    )
    if listing.returncode not in (0, 1):
        raise OSError(listing.stderr.decode("utf-8", errors="replace"))
    names = listing.stdout.split(b"\0")
    hashed = reclassified = identity_changes = 0
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, "
            "mtime INTEGER, ctime INTEGER, present INTEGER, record TEXT)"
        )
        previous_root = db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()
        if previous_root and previous_root[0] != str(source):
            raise ValueError("El inventario pertenece a otro directorio de origen")
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('root', ?)", (str(source),))
        db.execute("UPDATE files SET path=CAST(path AS BLOB) WHERE typeof(path)='text'")
        db.execute("UPDATE files SET present=0")
        for index, raw in enumerate(filter(None, names), 1):
            relative = Path(raw.decode("utf-8", errors="surrogateescape")).as_posix()
            path_key = os.fsencode(relative)
            path = source / relative
            before = path.lstat()
            signature = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            previous = db.execute(
                "SELECT size,mtime,ctime,record FROM files WHERE path=?", (path_key,)
            ).fetchone()
            cached = json.loads(previous[3]) if previous else None
            classification_error = None
            try:
                market, modality, symbol = classify(Path(relative))
            except ValueError as error:
                market, modality, symbol = "unknown", "charts", None
                classification_error = str(error)
            if (
                previous
                and previous[:3] == signature
                and not verify
                and not path.is_symlink()
                and cached.get("sha256")
                and "schema" in cached
                and (cached["state"] != "error" or cached.get("classification_error"))
                and cached.get("inspection_version") == 2
            ):
                changed = (cached["market"], cached["modality"], cached["symbol"]) != (
                    market,
                    modality,
                    symbol,
                )
                if (
                    cached.get("classification_version") != CLASSIFICATION_VERSION
                    or changed
                    or cached.get("classification_error") != classification_error
                ):
                    cached.update(
                        market=market,
                        modality=modality,
                        symbol=symbol,
                        classification_version=CLASSIFICATION_VERSION,
                        classified_at=inspected_at,
                    )
                    if classification_error is not None:
                        cached.update(
                            state="error",
                            error=classification_error,
                            classification_error=classification_error,
                        )
                    else:
                        cached["state"] = "inspected"
                        cached.pop("error", None)
                        cached.pop("classification_error", None)
                    db.execute(
                        "UPDATE files SET present=1,record=? WHERE path=?",
                        (json.dumps(cached, ensure_ascii=True), path_key),
                    )
                    reclassified += 1
                    identity_changes += changed
                else:
                    db.execute("UPDATE files SET present=1 WHERE path=?", (path_key,))
                continue
            record = {
                "path": relative,
                "bytes": before.st_size,
                "market": market,
                "modality": modality,
                "symbol": symbol,
                "sha256": None,
                "inspected_at": inspected_at,
                "state": "inspected",
                "inspection_version": 2,
                "classification_version": CLASSIFICATION_VERSION,
                "classified_at": inspected_at,
                "acquisition_revision": "unknown",
                "redistribution": "not_granted",
            }
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(source):
                    raise ValueError("No se admiten enlaces simbólicos como archivos de origen")
                record["sha256"] = sha256(path)
                record["schema"] = inspect_header(path)
                after = path.stat()
                if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("La fuente ha cambiado durante la inspección")
                hashed += 1
            except (OSError, ValueError) as error:
                record.update(state="error", error=str(error), sha256=None)
            else:
                if classification_error is not None:
                    record.update(
                        state="error",
                        error=classification_error,
                        classification_error=classification_error,
                    )
            db.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,1,?)",
                (path_key, *signature, json.dumps(record, ensure_ascii=True)),
            )
            if index % 250 == 0:
                db.commit()
                print(f"Inventario: {index}/{len(names) - 1} archivos", file=sys.stderr, flush=True)
        db.commit()
        groups = [
            dict(market=market, modality=modality, files=count, bytes=size)
            for market, modality, count, size in db.execute(
                "SELECT json_extract(record,'$.market'), json_extract(record,'$.modality'),"
                "count(*),sum(size) FROM files WHERE present=1 GROUP BY 1,2 ORDER BY 1,2"
            )
        ]
        errors = db.execute(
            "SELECT count(*) FROM files WHERE present=1 AND json_extract(record,'$.state')='error'"
        ).fetchone()[0]
        known_hashes, unique = db.execute(
            "SELECT count(json_extract(record,'$.sha256')),"
            "count(DISTINCT json_extract(record,'$.sha256')) FROM files WHERE present=1"
        ).fetchone()
        count = sum(group["files"] for group in groups)
        snapshot = hashlib.sha256()
        for (record,) in db.execute("SELECT record FROM files WHERE present=1 ORDER BY path"):
            item = json.loads(record)
            snapshot.update(
                json.dumps(
                    [item["path"], item["sha256"], item["bytes"]],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
            )
            snapshot.update(b"\n")
    return {
        "schema_version": 1,
        "inspected_at": inspected_at,
        "files": count,
        "snapshot_sha256": snapshot.hexdigest(),
        "bytes": sum(group["bytes"] for group in groups),
        "groups": groups,
        "hashed_files": hashed,
        "reclassified_files": reclassified,
        "changed_identities": identity_changes,
        "classification_version": CLASSIFICATION_VERSION,
        "code_sha256": sha256(Path(__file__)),
        "database_sha256": sha256(database),
        "errors": errors,
        "duplicate_content_files": known_hashes - unique,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "validation": "inventory_and_bounded_headers_only",
    }
