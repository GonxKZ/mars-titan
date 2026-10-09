"""Servidor local de solo lectura para seguir el observatorio en directo.

Sirve el sitio estático y los JSON del recolector y empuja los cambios mediante
Server-Sent Events. No escribe en las fuentes, no toma sus bloqueos y no cambia
nada del sistema. Está pensado para un único equipo: escucha en 127.0.0.1, rechaza
cabeceras Host ajenas para evitar el reenlace de DNS y acota clientes, conexiones,
eventos por segundo y tamaño de lo que lee.
"""

import json
import math
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import formatdate, parsedate_to_datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .telemetry import FIELDS, SystemProbe, TelemetryRing

VERSION = "observatory-live/1"
SITE_PATH = re.compile(
    r"(index\.html|styles\.css|[a-z][a-z0-9-]*\.m?js|vendor/uplot/(uPlot\.esm\.js|uPlot\.min\.css|LICENSE)"
    r"|fonts/[a-z0-9-]+\.(woff2|txt))"
)
DATA_PATH = re.compile(r"data/(observatory\.json|deployment\.json|pages/[a-f0-9]{64}\.json)")
TRACE_PATH = re.compile(r"data/traces/(index\.json|[A-Za-z0-9][\w.-]{0,95}\.(json|bin))")
LABEL = re.compile(r"[A-Za-z0-9][\w.-]{0,95}")
JOB_PART = re.compile(r"[A-Za-z0-9][\w.+-]{0,95}")
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".bin": "application/octet-stream",
    "": "text/plain; charset=utf-8",
}
# La misma política que declara la página. El servidor la repite como cabecera para
# que también cubra las respuestas JSON y binarias.
POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)


@dataclass(frozen=True)
class Limits:
    """Presupuesto del servidor. Los valores por defecto bastan para una o dos pestañas."""

    max_clients: int = 4
    max_connections: int = 24
    events_per_second: float = 2.0
    heartbeat_seconds: float = 15.0
    telemetry_seconds: float = 5.0
    idle_telemetry_seconds: float = 30.0
    source_seconds: float = 2.0
    telemetry_capacity: int = 17_280
    backlog_samples: int = 4_320
    max_file_bytes: int = 8 * 1024**2
    max_trace_bytes: int = 256 * 1024**2
    max_event_bytes: int = 4 * 1024**2
    max_campaign_jobs: int = 20_000
    max_active_jobs: int = 8

    def __post_init__(self):
        checks = (
            1 <= self.max_clients <= 64,
            self.max_clients < self.max_connections <= 256,
            0.1 <= self.events_per_second <= 20,
            1 <= self.heartbeat_seconds <= 60,
            1 <= self.telemetry_seconds <= self.idle_telemetry_seconds <= 600,
            0.5 <= self.source_seconds <= 60,
            1 <= self.backlog_samples <= self.telemetry_capacity <= 1_000_000,
            1024 <= self.max_file_bytes <= 64 * 1024**2,
            1024 <= self.max_trace_bytes <= 1024**3,
            1024 <= self.max_event_bytes <= 16 * 1024**2,
            1 <= self.max_campaign_jobs <= 100_000,
            0 <= self.max_active_jobs <= 64,
        )
        if not all(checks):
            raise ValueError("Límites del servidor en directo fuera de su rango admitido")


def utc_text(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def open_regular(root, relative, maximum):
    """Abrir un archivo regular dentro de `root` sin seguir enlaces y con tamaño acotado."""
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise PermissionError(relative)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise PermissionError(relative)
        return os.fdopen(fd, "rb"), info
    except BaseException:
        os.close(fd)
        raise


def read_json(root, relative, maximum):
    stream, info = open_regular(root, relative, maximum)
    with stream:
        body = stream.read(maximum + 1)
    if len(body) > maximum:
        raise ValueError("El archivo ha crecido durante la lectura")
    return json.loads(body), info


def signature(path):
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return info.st_ino, info.st_size, info.st_mtime_ns


def entity_tag(info):
    return f'"{info.st_size:x}-{info.st_mtime_ns:x}-{info.st_ino:x}"'


class Hub:
    """Último estado de cada tema. Solo se conserva la versión más reciente de cada uno,
    de modo que un cliente lento recibe el estado actual y no una cola de cambios."""

    def __init__(self, limits):
        self.limits = limits
        self.condition = threading.Condition()
        self.topics = {}
        self.clients = 0
        self.closed = False

    def publish(self, name, payload):
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        if len(body) > self.limits.max_event_bytes:
            raise ValueError(f"El evento {name} supera el tamaño máximo")
        with self.condition:
            version, previous = self.topics.get(name, (0, None))
            if previous == body:
                return False
            self.topics[name] = (version + 1, body)
            self.condition.notify_all()
            return True

    def acquire(self):
        with self.condition:
            if self.closed or self.clients >= self.limits.max_clients:
                return False
            self.clients += 1
            self.condition.notify_all()
            return True

    def release(self):
        with self.condition:
            self.clients -= 1

    def changes(self, seen, timeout):
        """Esperar a que algún tema cambie respecto a `seen` o a que venza el latido."""
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.closed:
                pending = [
                    (name, version, body)
                    for name, (version, body) in self.topics.items()
                    if seen.get(name) != version
                ]
                remaining = deadline - time.monotonic()
                if pending or remaining <= 0:
                    return pending
                self.condition.wait(remaining)
            return []

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def summarize_index(document, info):
    """Resumen pequeño del índice: el navegador descarga el índice completo con su ETag."""
    pagination = document.get("pagination") or {}
    return dict(
        etag=entity_tag(info),
        generated_at=document.get("generated_at"),
        modified_at=utc_text(info.st_mtime_ns / 1e9),
        total_runs=pagination.get("total_runs", len(document.get("runs", []))),
        pages=len(pagination.get("pages", [])),
        running=sum(1 for run in document.get("runs", []) if run.get("status") == "running"),
    )


def campaign_state(label, folder, limits):
    """Normalizar el resumen de una campaña por ventanas sin tomar su bloqueo.

    `summary.json` declara qué trabajos están confirmados. Un trabajo sin recibo con una
    carpeta de intento se marca como «intento sin confirmar», que no equivale a que el
    proceso siga vivo. La fecha de modificación de cada recibo aproxima su confirmación
    y sirve al navegador para estimar el ritmo, siempre rotulado como estimación.
    """
    folder = Path(folder)
    summary, info = read_json(folder, "summary.json", limits.max_file_bytes)
    jobs = summary.get("jobs")
    if not isinstance(jobs, dict) or len(jobs) > limits.max_campaign_jobs:
        raise ValueError("El resumen de la campaña no declara sus trabajos dentro del límite")
    vocabulary = {key: {} for key in ("scopes", "windows", "arms", "names")}

    def code(kind, value):
        return vocabulary[kind].setdefault(value, len(vocabulary[kind]))

    cells, active = [], []
    for job_id, done in jobs.items():
        parts = job_id.split("/")
        if len(parts) != 4 or not all(JOB_PART.fullmatch(part) for part in parts):
            raise ValueError("Identificador de trabajo fuera del contrato de la campaña")
        job_folder = folder / "jobs" / job_id
        confirmed = None
        state = "done" if done is True else "pending"
        if state == "done":
            receipt = signature(job_folder / "receipt.json")
            confirmed = utc_text(receipt[2] / 1e9) if receipt else None
        elif job_folder.is_dir():
            attempts = sorted(p for p in job_folder.glob("attempt-*") if p.is_dir())
            if attempts:
                state = "attempt"
                active.append((job_id, attempts[-1]))
        scope, window, arm, name = parts
        cells.append(
            [
                code("scopes", scope),
                code("windows", window),
                code("arms", arm),
                code("names", name),
                state,
                confirmed,
            ]
        )
    runs = []
    for job_id, attempt in sorted(
        active, key=lambda item: -(signature(item[1] / "run.json") or (0, 0, 0))[2]
    )[: limits.max_active_jobs]:
        try:
            report, report_info = read_json(attempt, "run.json", limits.max_file_bytes)
        except (OSError, ValueError):
            runs.append(dict(job=job_id, attempt=attempt.name, updated_at=None, epochs=[]))
            continue
        runs.append(
            dict(
                job=job_id,
                attempt=attempt.name,
                updated_at=utc_text(report_info.st_mtime_ns / 1e9),
                global_step=report.get("global_step")
                if type(report.get("global_step")) is int
                else None,
                epochs=[
                    epoch_point(epoch)
                    for epoch in (report.get("epochs") or [])[:2000]
                    if isinstance(epoch, dict)
                ],
            )
        )
    return dict(
        id=label,
        kind=summary.get("kind"),
        status=summary.get("status"),
        updated_at=summary.get("updated_at_utc"),
        summary_modified_at=utc_text(info.st_mtime_ns / 1e9),
        planned=summary.get("planned"),
        completed=summary.get("completed"),
        final_test_opened=summary.get("final_test_opened"),
        vocabulary={kind: list(values) for kind, values in vocabulary.items()},
        cells=cells,
        active=runs,
    )


def _finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def epoch_point(epoch):
    """Medidas observadas de una época en curso. No se calcula ninguna medida nueva."""
    train = epoch.get("train") if isinstance(epoch.get("train"), dict) else {}
    validation = epoch.get("validation") if isinstance(epoch.get("validation"), dict) else {}
    return dict(
        epoch=epoch.get("epoch") if type(epoch.get("epoch")) is int else None,
        train_mae=_finite(train.get("mae")),
        mae=_finite(validation.get("mae")),
        session_mae=_finite(validation.get("session_mae")),
        train_samples_per_second=_finite(train.get("samples_per_second")),
        train_seconds=_finite(train.get("elapsed_seconds")),
    )


class Sampler(threading.Thread):
    """Hilo único que observa las fuentes. Sin clientes solo toma telemetría de forma
    espaciada, para que la página encuentre historia al abrirse sin gastar CPU."""

    def __init__(self, server):
        super().__init__(name="observatory-sampler", daemon=True)
        self.server = server
        self.hub = server.hub
        self.limits = server.limits
        self.signatures = {}
        self.next = {}
        self.ring = TelemetryRing(self.limits.telemetry_capacity)
        self.errors = {}

    def due(self, name, interval, now):
        if now < self.next.get(name, 0):
            return False
        self.next[name] = now + interval
        return True

    def run(self):
        watched_before = False
        while not self.hub.closed:
            now = time.monotonic()
            watched = self.hub.clients > 0
            if watched and not watched_before:
                # Un cliente nuevo debe ver el estado actual sin esperar al siguiente turno.
                self.next.clear()
            watched_before = watched
            interval = (
                self.limits.telemetry_seconds if watched else self.limits.idle_telemetry_seconds
            )
            if self.due("telemetry", interval, now):
                self.telemetry()
            if watched and self.due("sources", self.limits.source_seconds, now):
                self.sources()
            # Se duerme hasta la siguiente tarea. La llegada de un cliente despierta el hilo.
            wake = min(self.next.values(), default=now + 1) - time.monotonic()
            with self.hub.condition:
                if not self.hub.closed and (self.hub.clients > 0) == watched:
                    self.hub.condition.wait(min(max(wake, 0.05), 5.0))

    def telemetry(self):
        values = self.server.probe.sample()
        timestamp = time.time()
        self.ring.append(timestamp, values)
        self.hub.publish(
            "telemetry",
            dict(
                t=round(timestamp, 3),
                values={k: (None if math.isnan(v) else round(v, 4)) for k, v in values.items()},
                gpu=self.server.probe.nvml.name,
                disk_path=self.server.probe.disk_path,
                interval_seconds=self.limits.telemetry_seconds,
            ),
        )

    def watch(self, name, path, build):
        """Volver a leer una fuente solo cuando cambia su identidad de archivo."""
        current = signature(path)
        if current is not None and current == self.signatures.get(name):
            return
        self.signatures[name] = current
        try:
            payload = build() if current is not None else dict(available=False)
            self.errors.pop(name, None)
        except (OSError, ValueError, TypeError, KeyError) as error:
            payload = dict(available=False, error=type(error).__name__)
            self.errors[name] = type(error).__name__
        self.hub.publish(name, payload)

    def sources(self):
        server = self.server
        if server.public_dir is not None:

            def index():
                document, info = read_json(
                    server.public_dir, "observatory.json", self.limits.max_file_bytes
                )
                return dict(available=True, **summarize_index(document, info))

            self.watch("index", server.public_dir / "observatory.json", index)
        for label, folder in server.campaigns.items():
            self.watch(
                f"campaign:{label}",
                folder / "summary.json",
                lambda label=label, folder=folder: dict(
                    available=True, **campaign_state(label, folder, self.limits)
                ),
            )
        if server.traces_dir is not None:

            def traces():
                document, info = read_json(
                    server.traces_dir, "index.json", self.limits.max_file_bytes
                )
                return dict(
                    available=True, etag=entity_tag(info), bundles=len(document.get("bundles", []))
                )

            self.watch("traces", server.traces_dir / "index.json", traces)


class ObservatoryHandler(BaseHTTPRequestHandler):
    server_version = VERSION
    sys_version = ""
    protocol_version = "HTTP/1.1"
    # Una conexión persistente inactiva libera su hilo y su plaza a los 30 segundos.
    timeout = 30

    def log_message(self, format, *args):
        # Los accesos no se registran: la consola queda para errores y para la URL.
        return

    def allowed_host(self):
        return self.headers.get("Host", "") in self.server.hosts

    def send_plain(self, status, text="", headers=()):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        self.send_plain(HTTPStatus.METHOD_NOT_ALLOWED, "Solo lectura", [("Allow", "GET, HEAD")])

    do_PUT = do_DELETE = do_PATCH = do_POST

    def do_GET(self):
        if not self.allowed_host():
            self.send_plain(HTTPStatus.MISDIRECTED_REQUEST, "Host no admitido")
            return
        path = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        if path == "api/events" and self.command == "GET":
            self.events()
        elif path == "api/status":
            self.json_response(self.server.status())
        elif SITE_PATH.fullmatch(path):
            self.file(self.server.site_dir, path, "no-cache")
        elif DATA_PATH.fullmatch(path) and self.server.public_dir is not None:
            immutable = path.startswith("data/pages/")
            cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
            self.file(self.server.public_dir, path.removeprefix("data/"), cache)
        elif TRACE_PATH.fullmatch(path) and self.server.traces_dir is not None:
            maximum = self.server.limits.max_trace_bytes if path.endswith(".bin") else None
            self.file(
                self.server.traces_dir, path.removeprefix("data/traces/"), "no-cache", maximum
            )
        else:
            self.send_plain(HTTPStatus.NOT_FOUND, "No encontrado")

    def json_response(self, payload):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def file(self, root, relative, cache, maximum=None):
        try:
            stream, info = open_regular(
                root, relative, maximum or self.server.limits.max_file_bytes
            )
        except (FileNotFoundError, NotADirectoryError):
            self.send_plain(HTTPStatus.NOT_FOUND, "No encontrado")
            return
        except (PermissionError, OSError):
            self.send_plain(HTTPStatus.FORBIDDEN, "Archivo no admitido")
            return
        with stream:
            tag = entity_tag(info)
            modified = formatdate(info.st_mtime, usegmt=True)
            if self.not_modified(tag, info.st_mtime):
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", tag)
                self.send_header("Last-Modified", modified)
                self.send_header("Cache-Control", cache)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", TYPES[Path(relative).suffix])
            self.send_header("Content-Length", str(info.st_size))
            self.send_header("ETag", tag)
            self.send_header("Last-Modified", modified)
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if relative.endswith(".html"):
                self.send_header("Content-Security-Policy", POLICY)
            self.end_headers()
            if self.command == "HEAD":
                return
            remaining = info.st_size
            while remaining > 0:
                chunk = stream.read(min(remaining, 256 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def not_modified(self, tag, mtime):
        match = self.headers.get("If-None-Match")
        if match is not None:
            return tag in [value.strip() for value in match.split(",")] or match.strip() == "*"
        since = self.headers.get("If-Modified-Since")
        if since is None:
            return False
        try:
            return int(mtime) <= parsedate_to_datetime(since).timestamp()
        except (TypeError, ValueError, OverflowError):
            return False

    def events(self):
        hub, limits = self.server.hub, self.server.limits
        if not hub.acquire():
            self.send_plain(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Demasiados clientes en directo",
                [("Retry-After", "30")],
            )
            return
        # El hilo de la conexión no debe quedar bloqueado si el cliente deja de leer.
        self.connection.settimeout(limits.heartbeat_seconds * 2)
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.close_connection = True
            seen, gap, last = {}, 1 / limits.events_per_second, 0.0
            sync = dict(
                status=self.server.status(),
                backlog=self.server.sampler.ring.latest(limits.backlog_samples),
                topics={},
            )
            with hub.condition:
                for name, (version, body) in hub.topics.items():
                    sync["topics"][name] = json.loads(body)
                    seen[name] = version
            self.write_event("sync", json.dumps(sync, separators=(",", ":"), allow_nan=False))
            while not hub.closed:
                pending = hub.changes(seen, limits.heartbeat_seconds)
                if not pending:
                    self.wfile.write(b": latido\n\n")
                    self.wfile.flush()
                    continue
                for name, version, body in pending:
                    # Límite de frecuencia por cliente. Tras esperar se envía la versión
                    # más reciente del tema, de modo que los cambios intermedios se funden.
                    wait = last + gap - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    with hub.condition:
                        version, body = hub.topics[name]
                    self.write_event(name, body, version)
                    seen[name] = version
                    last = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            hub.release()

    def write_event(self, name, body, version=None):
        head = f"event: {name.split(':', 1)[0]}\n"
        if version is not None:
            head += f"id: {version}\n"
        self.wfile.write(f"{head}data: {body}\n\n".encode())
        self.wfile.flush()
        self.server.events_sent += 1


class LiveObservatory(ThreadingHTTPServer):
    """Servidor HTTP con conexiones acotadas. Cada conexión usa un hilo propio."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address,
        *,
        site_dir,
        public_dir=None,
        campaigns=None,
        traces_dir=None,
        limits=None,
        probe=None,
        allow_remote=False,
    ):
        host = address[0]
        if host not in {"127.0.0.1", "::1", "localhost"} and not allow_remote:
            raise ValueError("El servidor en directo solo escucha en la interfaz local por defecto")
        self.limits = limits or Limits()
        self.site_dir = Path(site_dir).resolve()
        self.public_dir = Path(public_dir).resolve() if public_dir else None
        self.traces_dir = Path(traces_dir).resolve() if traces_dir else None
        self.campaigns = {}
        for label, folder in (campaigns or {}).items():
            if not LABEL.fullmatch(label):
                raise ValueError("Etiqueta de campaña no admitida")
            self.campaigns[label] = Path(folder).resolve()
        if len(self.campaigns) > 8:
            raise ValueError("Se admiten como máximo ocho campañas")
        self.connections = threading.BoundedSemaphore(self.limits.max_connections)
        self.hub = Hub(self.limits)
        self.probe = probe or SystemProbe(self.public_dir or self.site_dir)
        self.started_at = utc_text(time.time())
        self.events_sent = 0
        super().__init__(address, ObservatoryHandler)
        port = self.server_address[1]
        literal = f"[{host}]" if ":" in host else host
        names = {literal, "127.0.0.1", "localhost", "[::1]"} if not allow_remote else {literal}
        self.hosts = {f"{name}:{port}" for name in names}
        self.sampler = Sampler(self)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nContent-Length: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def status(self):
        return dict(
            mode="live",
            version=VERSION,
            started_at=self.started_at,
            clients=self.hub.clients,
            events_sent=self.events_sent,
            limits=dict(
                max_clients=self.limits.max_clients,
                events_per_second=self.limits.events_per_second,
                telemetry_seconds=self.limits.telemetry_seconds,
                heartbeat_seconds=self.limits.heartbeat_seconds,
            ),
            sources=dict(
                public=self.public_dir is not None,
                campaigns=sorted(self.campaigns),
                traces=self.traces_dir is not None,
                gpu=self.probe.nvml.name,
            ),
            fields=list(FIELDS),
            errors=dict(self.sampler.errors),
        )

    def serve(self):
        self.sampler.start()
        try:
            self.serve_forever(poll_interval=0.5)
        finally:
            self.hub.close()

    def server_close(self):
        self.hub.close()
        super().server_close()
