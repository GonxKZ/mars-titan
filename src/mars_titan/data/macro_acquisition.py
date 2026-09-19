"""Adquisición acotada y reanudable de vintages macro desde ALFRED."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

_ALFRED_HOST = "alfred.stlouisfed.org"
_FORM_URL = f"https://{_ALFRED_HOST}/series/downloaddata?seid={{series_id}}"
_QUERY_VERSION = "alfred-real-time-period-v1"
_DATABASE = "macro.sqlite3"
_BATCH_SIZE = 350
_MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_MEMBERS = 4
_HTTP_TIMEOUT_SECONDS = 90
_HTTP_RETRIES = 2


class _VintagePageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside = False
        self.dates: list[str] = []
        self.observation_start: str | None = None
        self.observation_end: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "select" and attributes.get("id") == "form_selected_vintage_dates":
            self._inside = True
        elif self._inside and tag == "option" and attributes.get("value"):
            self.dates.append(attributes["value"])
        elif tag == "input" and attributes.get("value"):
            if attributes.get("id") == "form_obs_start_date":
                self.observation_start = attributes["value"]
            elif attributes.get("id") == "form_obs_end_date":
                self.observation_end = attributes["value"]

    def handle_endtag(self, tag: str) -> None:
        if tag == "select" and self._inside:
            self._inside = False


def _iso_day(value: date | str, name: str) -> str:
    if isinstance(value, datetime):
        raise TypeError(f"{name} must be a date, not datetime")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return date.fromisoformat(value).isoformat()
    raise TypeError(f"{name} must be a date or ISO date string")


def _official_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != _ALFRED_HOST:
        raise ValueError("ALFRED requests must use the official HTTPS host")


def _request(url: str, *, fields: dict[str, str | list[str]] | None = None):
    """Return body, content type, effective URL and HTTP status using bounded curl."""
    _official_url(url)
    with tempfile.TemporaryDirectory(prefix="mars-titan-alfred-") as temporary:
        output = Path(temporary) / "response"
        command = [
            "curl",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--connect-timeout",
            "15",
            "--max-time",
            str(_HTTP_TIMEOUT_SECONDS),
            "--max-filesize",
            str(_MAX_COMPRESSED_BYTES),
            "--output",
            str(output),
            "--write-out",
            "%{http_code}\n%{content_type}\n%{url_effective}",
            url,
        ]
        if fields:
            for key, value in fields.items():
                values = value if isinstance(value, list) else [value]
                for item in values:
                    command.extend(["--data-urlencode", f"{key}={item}"])
        for attempt in range(_HTTP_RETRIES + 1):
            result = subprocess.run(command, capture_output=True, check=False)
            if result.returncode:
                raise OSError(result.stderr.decode("utf-8", errors="replace").strip())
            metadata = result.stdout.decode("utf-8", errors="strict").splitlines()
            if len(metadata) != 3:
                raise ValueError("Unexpected curl metadata")
            status = int(metadata[0])
            content_type, effective_url = metadata[1], metadata[2]
            _official_url(effective_url)
            body = output.read_bytes()
            if len(body) > _MAX_COMPRESSED_BYTES:
                raise ValueError("ALFRED response exceeds the compressed size limit")
            if status != 429 and not 500 <= status <= 599:
                return body, content_type, effective_url, status
            if attempt < _HTTP_RETRIES:
                time.sleep(0.5 * 2**attempt)
        return body, content_type, effective_url, status


def _vintage_metadata(page: bytes) -> tuple[list[str], str | None, str | None]:
    try:
        text = page.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("ALFRED vintage page is not UTF-8") from error
    parser = _VintagePageParser()
    parser.feed(text)
    if not parser.dates:
        raise ValueError("ALFRED vintage page has no selectable vintage dates")
    dates = []
    for value in parser.dates:
        dates.append(date.fromisoformat(value).isoformat())
    if dates != sorted(set(dates)):
        raise ValueError("ALFRED vintage dates are duplicated or unordered")
    observation_start = (
        date.fromisoformat(parser.observation_start).isoformat()
        if parser.observation_start
        else None
    )
    observation_end = (
        date.fromisoformat(parser.observation_end).isoformat() if parser.observation_end else None
    )
    return dates, observation_start, observation_end


def _vintage_dates(page: bytes) -> list[str]:
    return _vintage_metadata(page)[0]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _readme_dates(readme: str) -> set[str]:
    marker = "Vintage Dates Specified:"
    if marker not in readme:
        raise ValueError("ALFRED README does not record requested vintage dates")
    values = set()
    for line in readme.split(marker, 1)[1].splitlines():
        candidate = line.strip()
        try:
            values.add(date.fromisoformat(candidate).isoformat())
        except ValueError:
            continue
    return values


def _parse_zip(content: bytes, series_id: str, requested_dates: set[str]) -> list[dict]:
    if len(content) > _MAX_COMPRESSED_BYTES:
        raise ValueError("ALFRED ZIP exceeds the compressed size limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise ValueError("ALFRED response is not a valid ZIP archive") from error
    with archive:
        members = archive.infolist()
        if not 1 < len(members) <= _MAX_MEMBERS:
            raise ValueError("ALFRED ZIP has an unexpected number of members")
        if sum(member.file_size for member in members) > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError("ALFRED ZIP exceeds the uncompressed size limit")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
                raise ValueError("ALFRED ZIP contains an unsafe member path")
        names = {member.filename for member in members}
        csv_names = sorted(name for name in names if name.lower().endswith(".csv"))
        if "README.txt" not in names or len(csv_names) != 1:
            raise ValueError("ALFRED ZIP must contain one CSV and README.txt")
        try:
            readme = archive.read("README.txt").decode("utf-8")
            csv_text = archive.read(csv_names[0]).decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("ALFRED ZIP text is not UTF-8") from error
    recorded_dates = _readme_dates(readme)
    if recorded_dates != requested_dates:
        raise ValueError("ALFRED README vintage dates do not match the request")
    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    expected = [
        "period_start_date",
        series_id,
        "realtime_start_date",
        "realtime_end_date",
    ]
    if reader.fieldnames != expected:
        raise ValueError("ALFRED CSV has an unexpected header")
    result: list[dict] = []
    seen: dict[tuple[str, str], tuple[str, float | None]] = {}
    for row in reader:
        period = date.fromisoformat(row["period_start_date"]).isoformat()
        realtime_start = date.fromisoformat(row["realtime_start_date"]).isoformat()
        raw_end = row["realtime_end_date"].strip()
        realtime_end = (
            "9999-12-31" if raw_end in {"", "."} else date.fromisoformat(raw_end).isoformat()
        )
        if realtime_end < realtime_start or period > realtime_start:
            raise ValueError("ALFRED CSV contains an invalid temporal interval")
        raw_value = row[series_id].strip()
        if raw_value in {"", "."}:
            value = None
        else:
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("ALFRED CSV contains a non-finite value")
        key = period, realtime_start
        payload = realtime_end, value
        if key in seen:
            if seen[key] != payload:
                raise ValueError("ALFRED CSV contains conflicting vintages")
            continue
        seen[key] = payload
        result.append(
            {
                "period_start": period,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
                "value": value,
            }
        )
    return result


def _connect(destination: Path) -> sqlite3.Connection:
    database = destination / _DATABASE
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS configuration (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS series (
            indicator_id TEXT PRIMARY KEY,
            series_id TEXT,
            status TEXT NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS batches (
            indicator_id TEXT NOT NULL,
            batch_key TEXT NOT NULL,
            status TEXT NOT NULL,
            raw_path TEXT,
            source_hash TEXT,
            requested_dates TEXT NOT NULL,
            query TEXT NOT NULL,
            retrieved_at TEXT,
            row_count INTEGER,
            PRIMARY KEY (indicator_id, batch_key)
        );
        CREATE TABLE IF NOT EXISTS vintages (
            indicator_id TEXT NOT NULL,
            period_start TEXT NOT NULL,
            realtime_start TEXT NOT NULL,
            realtime_end TEXT NOT NULL,
            value REAL,
            source_hash TEXT NOT NULL,
            source_timezone TEXT NOT NULL,
            PRIMARY KEY (indicator_id, period_start, realtime_start)
        );
        """
    )
    return connection


def _configuration(
    observation_start: str, observation_end: str, realtime_start: str, realtime_end: str
) -> str:
    return json.dumps(
        {
            "query_version": _QUERY_VERSION,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _initialize(destination: Path, configuration: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with _connect(destination) as connection:
        previous = connection.execute(
            "SELECT value FROM configuration WHERE key='acquisition'"
        ).fetchone()
        if previous and previous[0] != configuration:
            raise ValueError("Destination contains a different macro acquisition")
        connection.execute(
            "INSERT OR IGNORE INTO configuration VALUES ('acquisition', ?)", (configuration,)
        )


def _exclusion(entry: dict) -> str | None:
    if entry.get("kind") != "raw":
        return "derived_not_downloaded"
    if entry.get("id") == "global_supply_pressure" or entry.get("series_id") == "GSCPI":
        return "gscpi_release_timestamp_unverified"
    if entry.get("vintage_policy") == "MODEL_VINTAGES_ONLY":
        return "model_vintages_only"
    if entry.get("vintage_policy") == "NO_VINTAGES_EXCLUDE":
        return "no_vintages_exclude"
    if entry.get("verification_status") != "verified_metadata_not_ingested":
        return "metadata_not_verified"
    series_id = entry.get("series_id")
    if not series_id or series_id == "no identifier verified":
        return "unverified_identifier"
    source = urlsplit(entry.get("source_url", ""))
    if "FRED" not in entry.get("provider", "") or source.hostname != "fred.stlouisfed.org":
        return "unsupported_provider"
    if entry.get("vintage_policy") != "ALFRED_OR_RELEASE_ARCHIVE":
        return "vintages_not_admissible"
    return None


def _batches(values: list[str]) -> list[list[str]]:
    return [
        values[index : index + _BATCH_SIZE] for index in range(0, len(values), _BATCH_SIZE)
    ] or [[]]


def _batch_key(entry: dict, selected: list[str], entered: list[str], configuration: str) -> str:
    payload = json.dumps(
        {
            "configuration": configuration,
            "indicator_id": entry["id"],
            "series_id": entry["series_id"],
            "selected": selected,
            "entered": entered,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _completed_batch(destination: Path, indicator_id: str, key: str) -> tuple[str, int] | None:
    with _connect(destination) as connection:
        row = connection.execute(
            "SELECT raw_path,source_hash,row_count FROM batches "
            "WHERE indicator_id=? AND batch_key=? AND status='complete'",
            (indicator_id, key),
        ).fetchone()
    if not row:
        return None
    raw_path, digest, count = row
    path = destination / raw_path
    if not path.is_file() or _sha256(path.read_bytes()) != digest:
        return None
    return digest, count


def _store_batch(
    destination: Path,
    entry: dict,
    key: str,
    content: bytes,
    requested_dates: set[str],
    query: dict,
    rows: list[dict],
    realtime_start: str,
    realtime_end: str,
) -> tuple[str, int]:
    digest = _sha256(content)
    relative = Path("raw") / entry["id"] / f"{key}.zip"
    _atomic_bytes(destination / relative, content)
    admitted = [
        row
        for row in rows
        if row["realtime_start"] <= realtime_end and row["realtime_end"] >= realtime_start
    ]
    retrieved_at = datetime.now(UTC).isoformat()
    with _connect(destination) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for row in admitted:
            previous = connection.execute(
                "SELECT realtime_end,value FROM vintages "
                "WHERE indicator_id=? AND period_start=? AND realtime_start=?",
                (entry["id"], row["period_start"], row["realtime_start"]),
            ).fetchone()
            payload = row["realtime_end"], row["value"]
            if previous is not None and previous != payload:
                raise ValueError(
                    "Conflicting macro vintage: "
                    f"{entry['id']}/{row['period_start']}/{row['realtime_start']}"
                )
            connection.execute(
                "INSERT INTO vintages VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(indicator_id,period_start,realtime_start) DO UPDATE SET "
                "source_hash=excluded.source_hash,source_timezone=excluded.source_timezone",
                (
                    entry["id"],
                    row["period_start"],
                    row["realtime_start"],
                    row["realtime_end"],
                    row["value"],
                    digest,
                    "America/New_York",
                ),
            )
        connection.execute(
            "INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?,?,?,?,?)",
            (
                entry["id"],
                key,
                "complete",
                relative.as_posix(),
                digest,
                json.dumps(sorted(requested_dates)),
                json.dumps(query, sort_keys=True),
                retrieved_at,
                len(admitted),
            ),
        )
    return digest, len(admitted)


def _acquire_series(
    entry: dict,
    destination: Path,
    observation_start: str,
    observation_end: str,
    realtime_start: str,
    realtime_end: str,
    configuration: str,
) -> dict:
    url = _FORM_URL.format(series_id=quote(entry["series_id"], safe=""))
    page, content_type, effective_url, status = _request(url)
    _official_url(effective_url)
    if status != 200 or "html" not in content_type.lower():
        raise ValueError(f"ALFRED vintage page returned HTTP {status} as {content_type}")
    dates, source_observation_start, source_observation_end = _vintage_metadata(page)
    page_digest = _sha256(page)
    _atomic_bytes(destination / "raw" / entry["id"] / f"vintage-list-{page_digest}.html", page)
    if realtime_end < dates[0]:
        raise ValueError("ALFRED series starts after the requested real-time interval")
    effective_realtime_start = max(realtime_start, dates[0])
    effective_observation_start = max(
        observation_start, source_observation_start or observation_start
    )
    effective_observation_end = min(observation_end, source_observation_end or observation_end)
    if effective_observation_start > effective_observation_end:
        raise ValueError("ALFRED series has no observations in the requested interval")
    selected = [value for value in dates if realtime_start <= value <= realtime_end]
    batches = _batches(selected)
    source_hashes: list[str] = []
    rows = 0
    resumed = 0
    for index, batch in enumerate(batches):
        entered = (
            list(dict.fromkeys([effective_realtime_start, realtime_end])) if index == 0 else []
        )
        requested_dates = set(batch) | set(entered)
        batch_configuration = configuration
        if (
            effective_observation_start != observation_start
            or effective_observation_end != observation_end
        ):
            batch_configuration += effective_observation_start + effective_observation_end
        key = _batch_key(entry, batch, entered, batch_configuration)
        completed = _completed_batch(destination, entry["id"], key)
        if completed:
            digest, count = completed
            source_hashes.append(digest)
            rows += count
            resumed += 1
            continue
        fields: dict[str, str | list[str]] = {
            "form[units]": "lin",
            "form[obs_start_date]": effective_observation_start,
            "form[obs_end_date]": effective_observation_end,
            "form[entered_vintage_dates]": " ".join(entered),
            "form[selected_vintage_dates][]": batch,
            "form[file_type]": "1",
            "form[file_format]": "csv",
            "form[download_data]": "",
        }
        content, response_type, response_url, response_status = _request(url, fields=fields)
        _official_url(response_url)
        if response_status != 200 or "zip" not in response_type.lower():
            detail = "html_error" if b"<html" in content[:4096].lower() else response_type
            raise ValueError(f"ALFRED download returned HTTP {response_status}: {detail}")
        parsed = _parse_zip(content, entry["series_id"], requested_dates)
        query = {
            "query_version": _QUERY_VERSION,
            "url": url,
            "observation_start": effective_observation_start,
            "observation_end": effective_observation_end,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "selected_vintage_dates": batch,
            "entered_vintage_dates": entered,
            "output_type": 1,
        }
        digest, count = _store_batch(
            destination,
            entry,
            key,
            content,
            requested_dates,
            query,
            parsed,
            realtime_start,
            realtime_end,
        )
        source_hashes.append(digest)
        rows += count
    if rows == 0:
        raise ValueError("ALFRED series returned no rows in the requested intervals")
    updated_at = datetime.now(UTC).isoformat()
    with _connect(destination) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO series VALUES (?,?,?,?,?)",
            (entry["id"], entry["series_id"], "complete", None, updated_at),
        )
    return {
        "indicator_id": entry["id"],
        "series_id": entry["series_id"],
        "status": "complete",
        "vintage_dates_available": len(dates),
        "vintage_dates_selected": len(selected),
        "vintage_date_first": selected[0] if selected else None,
        "vintage_date_last": selected[-1] if selected else None,
        "effective_realtime_start": effective_realtime_start,
        "coverage_limited_at_start": effective_realtime_start != realtime_start,
        "effective_observation_start": effective_observation_start,
        "effective_observation_end": effective_observation_end,
        "observation_coverage_limited": (
            effective_observation_start != observation_start
            or effective_observation_end != observation_end
        ),
        "batches": len(batches),
        "resumed_batches": resumed,
        "rows": rows,
        "source_hashes": sorted(set(source_hashes)),
        "vintage_page_hash": page_digest,
    }


def acquire_catalog(
    catalog,
    destination: Path,
    *,
    observation_start,
    observation_end,
    realtime_start,
    realtime_end,
    workers=2,
) -> dict:
    """Audit a catalog and acquire every admissible ALFRED raw series."""
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("workers must be one or two")
    destination = Path(destination)
    observation_start = _iso_day(observation_start, "observation_start")
    observation_end = _iso_day(observation_end, "observation_end")
    realtime_start = _iso_day(realtime_start, "realtime_start")
    realtime_end = _iso_day(realtime_end, "realtime_end")
    if observation_start > observation_end or realtime_start > realtime_end:
        raise ValueError("Acquisition date ranges must be ordered")
    configuration = _configuration(observation_start, observation_end, realtime_start, realtime_end)
    _initialize(destination, configuration)
    entries = list(catalog)
    excluded = []
    eligible = []
    for entry in entries:
        reason = _exclusion(entry)
        if reason:
            excluded.append({"indicator_id": entry.get("id"), "reason": reason})
            with _connect(destination) as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO series VALUES (?,?,?,?,?)",
                    (
                        entry.get("id"),
                        entry.get("series_id"),
                        "excluded",
                        reason,
                        datetime.now(UTC).isoformat(),
                    ),
                )
        else:
            eligible.append(entry)
    results = []
    failures = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _acquire_series,
                entry,
                destination,
                observation_start,
                observation_end,
                realtime_start,
                realtime_end,
                configuration,
            ): entry
            for entry in eligible
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # Cada fuente queda auditada sin ocultar fallos parciales.
                failure = {
                    "indicator_id": entry["id"],
                    "series_id": entry["series_id"],
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                with _connect(destination) as connection:
                    connection.execute(
                        "INSERT OR REPLACE INTO series VALUES (?,?,?,?,?)",
                        (
                            entry["id"],
                            entry["series_id"],
                            "error",
                            failure["error"],
                            datetime.now(UTC).isoformat(),
                        ),
                    )
    series = sorted(results + failures, key=lambda item: item["indicator_id"])
    return {
        "schema_version": 1,
        "query_version": _QUERY_VERSION,
        "destination": str(destination.resolve()),
        "catalog_entries": len(entries),
        "eligible_series": len(eligible),
        "completed_series": len(results),
        "failed_series": len(failures),
        "excluded_entries": len(excluded),
        "complete": not failures and len(results) == len(eligible),
        "observation_start": observation_start,
        "observation_end": observation_end,
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "series": series,
        "exclusions": excluded,
    }


def iter_vintages(destination: Path):
    """Yield normalized vintages from SQLite without loading the full panel."""
    database = Path(destination) / _DATABASE
    if not database.is_file():
        raise FileNotFoundError(f"Macro acquisition database not found: {database}")
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(
            "SELECT indicator_id,period_start,realtime_start,realtime_end,value,"
            "source_hash,source_timezone FROM vintages "
            "ORDER BY indicator_id,period_start,realtime_start"
        )
        while rows := cursor.fetchmany(1000):
            for row in rows:
                yield {
                    "indicator_id": row[0],
                    "period_start": row[1],
                    "realtime_start": row[2],
                    "realtime_end": row[3],
                    "value": row[4],
                    "source_hash": row[5],
                    "source_timezone": row[6],
                }
