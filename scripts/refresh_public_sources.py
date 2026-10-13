"""Descargar instantáneas públicas nuevas, sin modificar la referencia ni capturas previas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from mars_titan.data import public_snapshots

if __package__:
    from .public_source_formats import SUPPORTED_VALIDATORS, validate
else:
    from public_source_formats import SUPPORTED_VALIDATORS, validate

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/catalogs/public-sources.json"
OUTPUT_ROOT = ROOT / "data/external"
CAPS = {
    "max_total_bytes": 100 * 1024 * 1024,
    "max_source_bytes": 10 * 1024 * 1024,
    "timeout_seconds": 45,
    # Techo de disco del almacén: archivos confirmados, ejecuciones interrumpidas y nuevos.
    "max_store_bytes": 512 * 1024 * 1024,
}
EXTENSIONS = {
    "csv": ".csv",
    "xml": ".xml",
    "rss_xml": ".xml",
    "zip_csv": ".zip",
    "xlsx": ".xlsx",
    "pdf": ".pdf",
}
USER_AGENT = "MARS-TITAN public-source research (+https://github.com/GonxKZ/mars-titan)"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def require_https(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("La URL debe usar HTTPS sin credenciales")
    if parsed.fragment or any(ord(c) < 32 for c in url):
        raise ValueError("URL con fragmento o caracteres de control")
    return url


def eligible(source: dict) -> bool:
    return (
        source.get("status") == "downloaded_validated"
        and source.get("auth") == "none"
        and source.get("validator") in SUPPORTED_VALIDATORS
        and source.get("format") in EXTENSIONS
    )


def select_sources(catalog: dict, source_ids: list[str] | None = None) -> list[dict]:
    sources = catalog.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Catálogo sin lista de fuentes")
    identifiers = []
    for source in sources:
        identifier = source.get("id", "")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier
        ):
            raise ValueError("Identificador de fuente no válido")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Identificador de fuente duplicado")
    wanted = set(source_ids or [])
    if wanted - set(identifiers):
        raise ValueError(f"Fuentes desconocidas: {sorted(wanted - set(identifiers))}")
    if wanted:
        selected = [source for source in sources if source["id"] in wanted]
        if any(not eligible(source) for source in selected):
            raise ValueError("Fuente no validada, con autenticación o formato no admitido")
        return selected
    return [
        source for source in sources if eligible(source) and source["validator"] != "financial_pdf"
    ]


class CurrentLinks(HTMLParser):
    """Leer enlaces etiquetados current.csv, sin ejecutar contenido de la página."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.href = None
        self.label = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = []

    def handle_data(self, data):
        if self.href is not None:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a":
            if self.href and "".join(self.label).strip().lower() == "current.csv":
                self.links.append(self.href)
            self.href = None


def resolve_url(source: dict, as_of: date, *, page_html: bytes | None = None) -> str:
    identifier = source["id"]
    if identifier in {"fred_md", "fred_qd"}:
        if page_html is None:
            raise ValueError("Falta la página oficial con current.csv")
        page = require_https(source["url"])
        parser = CurrentLinks()
        parser.feed(page_html.decode("utf-8-sig"))
        section = "/monthly/" if identifier == "fred_md" else "/quarterly/"
        candidates = set()
        for href in parser.links:
            link = require_https(urljoin(page, href))
            parsed = urlsplit(link)
            if parsed.hostname not in {"stlouisfed.org", "www.stlouisfed.org"}:
                raise ValueError("Enlace current.csv fuera del proveedor oficial")
            if section in parsed.path and parsed.path.lower().endswith(".csv"):
                candidates.add(link)
        if len(candidates) != 1:
            raise ValueError("No hay un único enlace oficial current.csv para la frecuencia")
        return candidates.pop()
    original = require_https(source["download_url"])
    parsed = urlsplit(original)
    updates = {}
    if identifier == "ecb_usd_eur":
        updates = {
            "startPeriod": (as_of - timedelta(days=89)).isoformat(),
            "endPeriod": as_of.isoformat(),
        }
    elif identifier == "treasury_yields":
        updates = {"field_tdr_date_value_month": as_of.strftime("%Y%m")}
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in updates
    ]
    query.extend(updates.items())
    return urlunsplit(parsed._replace(query=urlencode(query))) if updates else original


class FetchError(RuntimeError):
    def __init__(self, message: str, metadata: dict):
        super().__init__(message)
        self.metadata = metadata


def response_validators(raw: str) -> dict:
    """ETag y Last-Modified de la última respuesta, tras las redirecciones."""
    blocks = [block for block in raw.replace("\r\n", "\n").split("\n\n") if block.strip()]
    found = {}
    for line in blocks[-1].splitlines()[1:] if blocks else []:
        name, separator, value = line.partition(":")
        key = {"etag": "etag", "last-modified": "last_modified"}.get(name.strip().lower())
        if separator and key:
            found[key] = public_snapshots.header_value(value.strip())
    return {"etag": found.get("etag"), "last_modified": found.get("last_modified")}


def conditional_headers(conditions: dict | None) -> list[str]:
    """Cabeceras If-None-Match e If-Modified-Since de una captura anterior."""
    headers = []
    for header, key in (("If-None-Match", "etag"), ("If-Modified-Since", "last_modified")):
        value = (conditions or {}).get(key)
        if value is None:
            continue
        if public_snapshots.header_value(value) is None:
            raise ValueError(f"Validador {key} no admitido como cabecera")
        headers.extend(["--header", f"{header}: {value}"])
    return headers


def fetch_url(
    url: str,
    max_bytes: int,
    timeout_seconds: float,
    user_agent: str,
    conditions: dict | None = None,
) -> tuple[bytes, dict]:
    require_https(url)
    if max_bytes <= 0 or timeout_seconds <= 0:
        raise ValueError("Presupuesto de descarga agotado")
    with tempfile.TemporaryDirectory(prefix="mars-public-source-") as directory:
        payload = Path(directory) / "response.bin"
        headers = Path(directory) / "headers.txt"
        command = [
            "curl",
            "--disable",
            "--silent",
            "--show-error",
            "--location",
            "--max-redirs",
            "5",
            "--fail-with-body",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            str(min(10, timeout_seconds)),
            "--max-time",
            str(timeout_seconds),
            "--max-filesize",
            str(max_bytes),
            "--user-agent",
            user_agent,
            *conditional_headers(conditions),
            "--dump-header",
            str(headers),
            "--output",
            str(payload),
            "--write-out",
            "%{json}",
            url,
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds + 2, check=False
        )
        info = json.loads(result.stdout) if result.stdout.strip() else {}
        received = max(
            int(info.get("size_download", 0)), payload.stat().st_size if payload.exists() else 0
        )
        metadata = {
            "http_status": info.get("http_code") or None,
            "effective_url": info.get("url_effective"),
            "content_type": info.get("content_type"),
            "received_bytes": received,
            "elapsed_seconds": info.get("time_total"),
            "curl_exit_code": result.returncode,
            "acquired_at_utc": utc_now(),
            "conditional_request": bool(conditions),
            **response_validators(
                headers.read_text(encoding="latin-1") if headers.is_file() else ""
            ),
        }
        if result.returncode:
            raise FetchError(result.stderr.strip() or info.get("errormsg", "Fallo HTTP"), metadata)
        if metadata["http_status"] == 304 and conditions:
            # Sin cuerpo: el contenido es el de la captura que aportó los validadores.
            return b"", metadata
        if not payload.is_file() or received > max_bytes:
            raise FetchError("Descarga ausente o superior al presupuesto", metadata)
        return payload.read_bytes(), metadata


class Session:
    """Contar también transferencias fallidas y la página de descubrimiento."""

    def __init__(self, limits: dict, user_agent: str):
        self.limits = limits
        self.user_agent = user_agent
        self.transferred = 0
        self.source_bytes = 0
        self.last_finished = None
        self.deadline = 0.0
        self.blocked_hosts = {}

    def begin_source(self):
        self.source_bytes = 0
        self.deadline = time.monotonic() + self.limits["timeout_seconds"]

    def request(self, url: str, conditions: dict | None = None) -> tuple[bytes, dict]:
        require_https(url)
        hostname = urlsplit(url).hostname
        if hostname in self.blocked_hosts:
            raise FetchError(
                "Host suspendido en esta ejecución tras HTTP 403 o 429",
                {
                    "http_status": None,
                    "prior_http_status": self.blocked_hosts[hostname],
                    "blocked_host": hostname,
                    "request_sent": False,
                    "received_bytes": 0,
                },
            )
        if self.last_finished is not None:
            time.sleep(max(0, 1 - (time.monotonic() - self.last_finished)))
        maximum = min(
            self.limits["max_total_bytes"] - self.transferred,
            self.limits["max_source_bytes"] - self.source_bytes,
        )
        remaining_time = self.deadline - time.monotonic()
        if maximum <= 0 or remaining_time <= 0:
            raise ValueError("Presupuesto de bytes o tiempo agotado")
        try:
            body, metadata = fetch_url(url, maximum, remaining_time, self.user_agent, conditions)
            status = int(metadata.get("http_status") or 0)
            if not (200 <= status < 300 or (status == 304 and conditions)):
                raise FetchError("Estado HTTP no satisfactorio", metadata)
        except FetchError as exc:
            received = int(exc.metadata.get("received_bytes", 0))
            self.transferred += received
            self.source_bytes += received
            status = exc.metadata.get("http_status")
            if status in {403, 429}:
                self.blocked_hosts[hostname] = status
                effective_host = urlsplit(exc.metadata.get("effective_url") or url).hostname
                if effective_host:
                    self.blocked_hosts[effective_host] = status
            raise
        finally:
            self.last_finished = time.monotonic()
        received = max(len(body), int(metadata.get("received_bytes", len(body))))
        self.transferred += received
        self.source_bytes += received
        if received > maximum:
            raise FetchError("La respuesta supera el presupuesto de bytes", metadata)
        require_https(metadata["effective_url"])
        return body, metadata


def write_new_file(destination: Path, content: bytes) -> None:
    """Publicar un archivo completo sin reemplazar ningún nombre existente."""
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".pending-") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, destination)


def maturity(source: dict) -> str:
    return {
        "fed_rss": "current_feed_requires_publication_audit",
        "gscpi_csv": "vintage_matrix_release_timestamps_pending",
        "stooq_csv": "latest_price_history_availability_unverified",
        "financial_pdf": "fixed_document_publication_history_unverified",
    }.get(source["validator"], "latest_revised_not_point_in_time")


def run_refresh(
    catalog_path: Path,
    output_root: Path,
    source_ids: list[str] | None = None,
    *,
    run_at: datetime | None = None,
) -> dict:
    catalog_path = catalog_path.resolve()
    output_root = output_root.resolve()
    catalog_reference = catalog_path.relative_to(ROOT.resolve()).as_posix()
    output_root.relative_to(ROOT.resolve())
    raw_catalog = catalog_path.read_bytes()
    catalog = json.loads(raw_catalog)
    sources = select_sources(catalog, source_ids)
    if not sources:
        raise ValueError("No hay fuentes públicas validadas seleccionadas")
    policy = catalog.get("download_policy", {})
    limits = {}
    for name, cap in CAPS.items():
        value = policy.get(name, cap)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Límite no válido: {name}")
        limits[name] = min(value, cap)
    when = run_at or datetime.now(UTC)
    if when.tzinfo is None:
        raise ValueError("La fecha de captura debe tener zona horaria")
    when = when.astimezone(UTC)
    run_id = when.strftime("%Y%m%dT%H%M%S.%fZ")
    with public_snapshots.exclusive_update(output_root):
        return capture_sources(
            sources, limits, policy, when, run_id, output_root, catalog_reference, raw_catalog
        )


def capture_sources(
    sources: list[dict],
    limits: dict,
    policy: dict,
    when: datetime,
    run_id: str,
    output_root: Path,
    catalog_reference: str,
    raw_catalog: bytes,
) -> dict:
    # El índice previo sale de los manifiestos confirmados. Un manifiesto corrupto detiene
    # la ejecución antes de descargar nada.
    index = public_snapshots.build_index(ROOT, output_root)
    directory = output_root / run_id
    directory.mkdir(exist_ok=False)
    written = {}
    store_bytes = index["stored_bytes"] + index["interrupted_bytes"]
    session = Session(limits, policy.get("user_agent", USER_AGENT))
    pages = {}
    records = []
    for source in sources:
        session.begin_source()
        record = {
            "source_id": source["id"],
            "provider": source.get("provider"),
            "source_metadata": source,
            "catalog_download_url": source.get("download_url"),
            "requested_url": None,
            "effective_url": None,
            "http_status": None,
            "content_type": None,
            "started_at_utc": utc_now(),
            "acquired_at_utc": None,
            "status": "failed",
            "sha256": None,
            "bytes": 0,
            "local_path": None,
            "expected_format": source["format"],
            "format_validation": None,
            "maturity": maturity(source),
            "benchmark_eligible": False,
            "refresh_mode": "fixed_document"
            if source["validator"] == "financial_pdf"
            else "current_source",
            "conditional_request": False,
            "etag": None,
            "last_modified": None,
            "content_reused": False,
            "reused_from_run_id": None,
            # La hora de adquisición no acredita cuándo se publicó cada dato. Sin evidencia
            # de publicación queda desconocida y `maturity` explica el motivo.
            "available_at_utc": None,
        }
        try:
            html = None
            if source["id"] in {"fred_md", "fred_qd"}:
                page_url = require_https(source["url"])
                if page_url not in pages:
                    try:
                        pages[page_url] = session.request(page_url)
                    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                        pages[page_url] = str(exc)
                        raise
                cached = pages[page_url]
                if isinstance(cached, str):
                    raise ValueError(f"Descubrimiento FRED fallido, sin reintento: {cached}")
                html, receipt = cached
                record["url_resolution"] = {
                    "page_url": page_url,
                    "effective_page_url": receipt["effective_url"],
                    "page_sha256": hashlib.sha256(html).hexdigest(),
                    "page_bytes": len(html),
                    "anchor_label": "current.csv",
                }
            url = resolve_url(source, when.date(), page_html=html)
            record["requested_url"] = url
            previous = public_snapshots.validators(index, source["id"], url)
            body, receipt = session.request(url, previous)
            record.update(receipt)
            status = "downloaded_validated"
            if receipt.get("http_status") == 304:
                # El proveedor confirma que no ha cambiado: se valida de nuevo el contenido
                # anterior y se conservan sus validadores si la respuesta no trae otros.
                body, _ = public_snapshots.read_content(ROOT, index, previous["sha256"])
                status = "not_modified"
                record["etag"] = record["etag"] or previous["etag"]
                record["last_modified"] = record["last_modified"] or previous["last_modified"]
            checked = validate(body, source["validator"], as_of=when.date())
            sha256 = hashlib.sha256(body).hexdigest()
            # Unos bytes ya confirmados no se vuelven a guardar: el registro remite a ellos.
            confirmed = public_snapshots.reusable_path(ROOT, index, sha256)
            if sha256 in written:
                local_path, origin = written[sha256], run_id
            elif confirmed is not None:
                local_path, origin = confirmed, index["contents"][sha256]["run_id"]
            else:
                if store_bytes + len(body) > limits["max_store_bytes"]:
                    raise ValueError("Límite de disco del almacén de capturas alcanzado")
                destination = directory / (source["id"] + EXTENSIONS[source["format"]])
                write_new_file(destination, body)
                store_bytes += len(body)
                local_path, origin = destination.relative_to(ROOT).as_posix(), None
                written[sha256] = local_path
            record.update(
                status=status,
                sha256=sha256,
                bytes=len(body),
                local_path=local_path,
                content_reused=origin is not None,
                reused_from_run_id=origin,
                format_validation=checked,
            )
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, FetchError):
                record.update(exc.metadata)
            record["failure_reason"] = str(exc)[:1000]
        record["finished_at_utc"] = utc_now()
        records.append(record)
    manifest = {
        "schema_version": 1,
        "project": "MARS-TITAN",
        "run_id": run_id,
        "started_at_utc": when.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "finished_at_utc": utc_now(),
        "source_catalog": catalog_reference,
        "source_catalog_sha256": hashlib.sha256(raw_catalog).hexdigest(),
        "limits": limits | {"serial_requests": True, "minimum_pause_seconds": 1, "retries": 0},
        "transferred_bytes": session.transferred,
        # Bytes nuevos guardados en esta ejecución. Los reutilizados ya estaban confirmados.
        "retained_bytes": sum(
            record["bytes"] for record in records if not record["content_reused"]
        ),
        "reused_bytes": sum(record["bytes"] for record in records if record["content_reused"]),
        "previous_interrupted_runs": index["interrupted_runs"],
        "benchmark_eligible": False,
        "benchmark_integration": "none",
        "records": records,
    }
    path = directory / "manifest.json"
    write_new_file(
        path, (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    )
    # El índice se escribe después del manifiesto. Si falla, la captura sigue confirmada y
    # la siguiente ejecución lo reconstruye.
    index_path = public_snapshots.write_index(ROOT, public_snapshots.build_index(ROOT, output_root))
    return {"manifest_path": path, "manifest": manifest, "index_path": index_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="Mostrar el catálogo sin acceder a la red"
    )
    parser.add_argument("--source", action="append", help="Identificador de una fuente validada")
    args = parser.parse_args(argv)
    try:
        if args.list:
            catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
            defaults = {source["id"] for source in select_sources(catalog)}
            for source in catalog["sources"]:
                print(
                    json.dumps(
                        {
                            "id": source["id"],
                            "provider": source.get("provider"),
                            "status": source.get("status"),
                            "default_enabled": source["id"] in defaults,
                            "explicit_selection_allowed": eligible(source),
                        },
                        ensure_ascii=False,
                    )
                )
            return 0
        result = run_refresh(CATALOG, OUTPUT_ROOT, args.source)
        for record in result["manifest"]["records"]:
            print(
                json.dumps(
                    {
                        key: record.get(key)
                        for key in ("source_id", "status", "http_status", "bytes", "failure_reason")
                    },
                    ensure_ascii=False,
                )
            )
        print(
            json.dumps(
                {
                    "manifest": result["manifest_path"].relative_to(ROOT).as_posix(),
                    "index": result["index_path"].relative_to(ROOT).as_posix(),
                    "retained_bytes": result["manifest"]["retained_bytes"],
                    "reused_bytes": result["manifest"]["reused_bytes"],
                },
                ensure_ascii=False,
            )
        )
        # Una respuesta 304 con el contenido anterior validado también es un éxito.
        return int(
            any(
                record["status"] not in public_snapshots.VALID
                for record in result["manifest"]["records"]
            )
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
