"""Capturas HTTPS acotadas, con caché inmutable y plazos de consulta persistentes."""

import fcntl
import gzip
import hashlib
import http.client
import ipaddress
import socket
import sqlite3
import ssl
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from protego import Protego

from .news_sources import MAX_CAPTURE_BYTES

USER_AGENT = "MARS-TITAN-Research/0.1 (+https://github.com/GonxKZ/mars-titan)"
_ROBOTS = "https://www.fool.com/robots.txt"


class SourceUnavailable(Exception):
    """Indicar un impedimento de adquisición, no la falsedad del contenido."""

    def __init__(self, reason: str, retry_at: float = 0):
        super().__init__(reason)
        self.reason = reason
        self.retry_at = retry_at


@dataclass(frozen=True)
class Capture:
    url: str
    body: bytes
    sha256: str
    fetched_at: str


def _address(url):
    address = urlsplit(url)
    if (
        address.scheme != "https"
        or address.netloc != "www.fool.com"
        or address.fragment
        or any(ord(character) < 33 for character in url)
    ):
        raise ValueError("La URL no pertenece al editor HTTPS autorizado")
    return address


def _request(url, *, timeout, max_bytes):
    """Fijar una IP pública y conservar TLS con SNI, sin proxies ni redirecciones."""
    address = _address(url)
    addresses = socket.getaddrinfo(address.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise ValueError("El editor no resuelve exclusivamente a una dirección pública")
    family, kind, protocol, _, destination = addresses[0]
    started = time.monotonic()
    with socket.socket(family, kind, protocol) as raw:
        raw.settimeout(timeout)
        raw.connect(destination)
        context = ssl.create_default_context()
        with context.wrap_socket(raw, server_hostname=address.hostname) as encrypted:
            with closing(http.client.HTTPSConnection(address.hostname, timeout=timeout)) as client:
                client.sock = encrypted
                target = address.path or "/"
                if address.query:
                    target += "?" + address.query
                client.request(
                    "GET",
                    target,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept-Encoding": "identity",
                        "Accept": "text/html,text/plain;q=0.9",
                    },
                )
                response = client.getresponse()
                headers = {key.lower(): value for key, value in response.getheaders()}
                if headers.get("content-encoding", "identity").lower() != "identity":
                    raise SourceUnavailable("unexpected_encoding")
                if int(headers.get("content-length", "0")) > max_bytes:
                    raise SourceUnavailable("response_too_large")
                body = bytearray()
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("La captura superó el plazo total")
                    encrypted.settimeout(remaining)
                    block = response.read1(min(65536, max_bytes + 1 - len(body)))
                    if not block:
                        break
                    body.extend(block)
                    if len(body) > max_bytes:
                        raise SourceUnavailable("response_too_large")
                if response.length:
                    raise SourceUnavailable("truncated_response")
                return response.status, headers, bytes(body)


def _retry_time(value, now):
    try:
        return now + max(0, int(value))
    except (ValueError, TypeError):
        try:
            return max(now, parsedate_to_datetime(value).timestamp())
        except (ValueError, TypeError, AttributeError, OverflowError):
            return now + 3600


class ArticleFetcher:
    """Serializar las consultas al editor y limitar el archivo de caché en disco."""

    def __init__(self, root: Path, *, quota_bytes=2 * 1024**3, min_interval=5, timeout=30):
        if quota_bytes < 65536 or min_interval < 0 or not 0 < timeout <= 60:
            raise ValueError("La cuota, el intervalo o el plazo de captura no son válidos")
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "captures.sqlite"
        if root.is_symlink() or self.path.is_symlink():
            raise ValueError("La caché no puede ser un enlace simbólico")
        self.interval, self.timeout, self.quota_bytes = min_interval, timeout, quota_bytes
        with closing(self._connect()) as db:
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            pages = quota_bytes // page_size
            if db.execute("PRAGMA page_count").fetchone()[0] > pages:
                raise ValueError("La caché existente supera la cuota solicitada")
            db.execute(f"PRAGMA max_page_count={pages}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS captures (
                    url TEXT PRIMARY KEY, body BLOB NOT NULL, sha256 TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    url TEXT PRIMARY KEY, reason TEXT NOT NULL, retry_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA trusted_schema=OFF")
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        db.execute(f"PRAGMA max_page_count={self.quota_bytes // page_size}")
        return db

    def cached(self, url: str) -> Capture | None:
        _address(url)
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT body,sha256,fetched_at FROM captures WHERE url=?", (url,)
            ).fetchone()
        if row is None:
            return None
        import io

        try:
            with gzip.GzipFile(fileobj=io.BytesIO(row[0])) as stream:
                body = stream.read(MAX_CAPTURE_BYTES + 1)
        except (OSError, EOFError) as error:
            raise ValueError("La captura comprimida está dañada") from error
        if len(body) > MAX_CAPTURE_BYTES or hashlib.sha256(body).hexdigest() != row[1]:
            raise ValueError("La huella o el tamaño de la captura no coincide")
        return Capture(url, body, row[1], row[2])

    @staticmethod
    def _state(db, key, value):
        db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, str(value)))
        db.commit()

    def _get(self, db, url, delay, limit):
        now = time.time()
        previous = db.execute("SELECT reason,retry_at FROM attempts WHERE url=?", (url,)).fetchone()
        if previous and previous[1] > now:
            raise SourceUnavailable(*previous)
        row = db.execute("SELECT value FROM state WHERE key='next_request'").fetchone()
        if row and float(row[0]) > now:
            raise SourceUnavailable("rate_limit", float(row[0]))
        self._state(db, "next_request", now + max(delay, self.interval))
        try:
            status, headers, body = _request(url, timeout=self.timeout, max_bytes=limit)
            if status != 200:
                if status in {301, 302, 303, 307, 308}:
                    _address(urljoin(url, headers.get("location", "")))
                retry = _retry_time(headers.get("retry-after"), now)
                if status in {401, 403, 404, 410}:
                    retry = max(retry, now + 86400)
                if status in {429, 503}:
                    self._state(db, "next_request", max(retry, now + delay))
                raise SourceUnavailable(f"http_{status}", retry)
            return body, headers
        except (OSError, http.client.HTTPException) as error:
            failure = SourceUnavailable("network_error", now + 3600)
            self._remember_failure(db, url, failure)
            raise failure from error
        except SourceUnavailable as error:
            if not error.retry_at:
                error.retry_at = now + 86400
            self._remember_failure(db, url, error)
            raise

    @staticmethod
    def _remember_failure(db, url, error):
        db.execute(
            "INSERT OR REPLACE INTO attempts VALUES (?,?,?)", (url, error.reason, error.retry_at)
        )
        db.commit()

    def _robots(self, db, url):
        stored = dict(db.execute("SELECT key,value FROM state WHERE key LIKE 'robots_%'"))
        now = time.time()
        if float(stored.get("robots_expires", "0")) <= now:
            body, _ = self._get(db, _ROBOTS, self.interval, 256 * 1024)
            stored["robots_text"] = body.decode("utf-8", errors="strict")
            db.executemany(
                "INSERT OR REPLACE INTO state VALUES (?,?)",
                (
                    ("robots_text", stored["robots_text"]),
                    ("robots_expires", str(now + 86400)),
                    ("robots_fetched", str(now)),
                ),
            )
            db.commit()
            stored["robots_fetched"] = str(now)
        policy = Protego.parse(stored["robots_text"])
        if not policy.can_fetch(url, USER_AGENT):
            raise SourceUnavailable("robots_disallow", now + 86400)
        rate = policy.request_rate(USER_AGENT)
        delay = max(
            self.interval,
            policy.crawl_delay(USER_AGENT) or 0,
            rate.seconds / rate.requests if rate and rate.requests else 0,
        )
        next_request = db.execute("SELECT value FROM state WHERE key='next_request'").fetchone()
        self._state(
            db,
            "next_request",
            max(
                float(next_request[0]) if next_request else 0,
                float(stored.get("robots_fetched", "0")) + delay,
            ),
        )
        return delay

    def fetch(self, url: str) -> Capture:
        _address(url)
        if capture := self.cached(url):
            return capture
        with self.path.open("rb") as lock, closing(self._connect()) as db:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if capture := self.cached(url):
                return capture
            delay = self._robots(db, url)
            body, headers = self._get(db, url, delay, MAX_CAPTURE_BYTES)
            if headers.get("content-type", "").split(";")[0].strip().lower() != "text/html":
                error = SourceUnavailable("content_type", time.time() + 86400)
                self._remember_failure(db, url, error)
                raise error
            capture = Capture(
                url, body, hashlib.sha256(body).hexdigest(), datetime.now(UTC).isoformat()
            )
            try:
                db.execute(
                    "INSERT INTO captures VALUES (?,?,?,?)",
                    (
                        url,
                        gzip.compress(body, mtime=0),
                        capture.sha256,
                        capture.fetched_at,
                    ),
                )
                db.commit()
            except sqlite3.OperationalError as error:
                db.rollback()
                raise SourceUnavailable("cache_write_error", time.time() + 3600) from error
            return capture
