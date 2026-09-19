"""Inventario completo con lectura acotada y recuperación por archivo."""

import codecs
import csv
import hashlib
import json
import os
import resource
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .storage import outside_source, sha256


def classify(path: Path) -> tuple[str, str, str | None]:
    parts = path.parts
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
        raise ValueError(f"Source directory does not exist: {source}")
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
    hashed = 0
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, "
            "mtime INTEGER, ctime INTEGER, present INTEGER, record TEXT)"
        )
        previous_root = db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()
        if previous_root and previous_root[0] != str(source):
            raise ValueError("Inventory belongs to a different source root")
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
            if (
                previous
                and previous[:3] == signature
                and not verify
                and not path.is_symlink()
                and json.loads(previous[3])["state"] != "error"
                and json.loads(previous[3]).get("inspection_version") == 2
            ):
                db.execute("UPDATE files SET present=1 WHERE path=?", (path_key,))
                continue
            market, modality, symbol = classify(Path(relative))
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
                "acquisition_revision": "unknown",
                "redistribution": "not_granted",
            }
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(source):
                    raise ValueError("Symbolic links are not admitted as source files")
                record["sha256"] = sha256(path)
                record["schema"] = inspect_header(path)
                after = path.stat()
                if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("Source changed during inspection")
                hashed += 1
            except (OSError, ValueError) as error:
                record.update(state="error", error=str(error), sha256=None)
            db.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,1,?)",
                (path_key, *signature, json.dumps(record, ensure_ascii=True)),
            )
            if index % 250 == 0:
                db.commit()
                print(f"inventory: {index}/{len(names) - 1} files", file=sys.stderr, flush=True)
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
        unique = db.execute(
            "SELECT count(DISTINCT json_extract(record,'$.sha256')) FROM files WHERE present=1"
        ).fetchone()[0]
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
        "errors": errors,
        "duplicate_content_files": count - errors - unique,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "validation": "inventory_and_bounded_headers_only",
    }
