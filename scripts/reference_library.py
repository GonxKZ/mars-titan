"""Mantenimiento de la biblioteca documental, separado del modelo científico."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_PDF_BYTES = 128 * 1024 * 1024


def read_catalogs(paths: list[Path]) -> list[dict]:
    """Leer catálogos sin permitir colisiones ni rutas derivadas de identificadores."""
    entries = []
    seen = set()
    for path in paths:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"El catálogo debe ser una lista: {path}")
        for entry in values:
            identifier = entry.get("id", "")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
                raise ValueError(f"identificador no válido en {path}: {identifier!r}")
            if identifier in seen:
                raise ValueError(f"Identificador duplicado: {identifier}")
            seen.add(identifier)
            entries.append(entry)
    return entries


def pdf_metadata(path: Path) -> dict:
    """Rechazar respuestas HTML y obtener tamaño y SHA-256, sin cargar el PDF en RAM."""
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError(f"La respuesta no tiene cabecera PDF: {path.name}")
        stream.seek(max(0, path.stat().st_size - 1024))
        if b"%%EOF" not in stream.read():
            raise ValueError(f"PDF incompleto, sin marcador final: {path.name}")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def download_reference(entry: dict, output: Path, timeout: float = 45) -> dict:
    """Conservar descargas previas y registrar fallos sin alterar el archivo final."""
    result = {"id": entry["id"], "requested_url": entry.get("pdf_url"), "url": None}
    url = entry.get("pdf_url")
    if not url:
        return result | {"status": "reference_only"}

    temporary = None
    try:
        identifier = entry["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
            raise ValueError("identificador no válido")
        if urlparse(url).scheme != "https":
            raise ValueError("La descarga requiere HTTPS")
        output.mkdir(parents=True, exist_ok=True)
        destination = output / f"{identifier}.pdf"
        receipt_path = output / f"{identifier}.source.json"
        if destination.is_symlink():
            raise ValueError("El destino no puede ser un enlace simbólico")
        if destination.exists():
            metadata = pdf_metadata(destination)
            if not receipt_path.exists():
                return result | {"status": "cached_unverified", **metadata}
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["url"] != url or receipt["sha256"] != metadata["sha256"]:
                raise ValueError("La procedencia o el hash difieren del catálogo o archivo actual")
            return result | receipt | {"status": "cached", **metadata}

        request = Request(url, headers={"User-Agent": "MARS-TITAN-research-library/0.1"})
        with urlopen(request, timeout=timeout) as response:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("La redirección no usa HTTPS")
            result["final_url"] = response.geturl()
            expected_size = response.headers.get("Content-Length")
            with tempfile.NamedTemporaryFile(dir=output, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise ValueError("El PDF supera el límite de 128 MiB")
                    stream.write(chunk)
            if expected_size is not None and total != int(expected_size):
                raise ValueError("Descarga incompleta respecto a Content-Length")
        metadata = pdf_metadata(temporary)
        # Publicación atómica que no sobrescribe un destino creado por otro proceso.
        os.link(temporary, destination)
        result.update(
            status="downloaded", url=url, downloaded_at=datetime.now(UTC).isoformat(), **metadata
        )
        with receipt_path.open("x", encoding="utf-8") as receipt:
            receipt.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    except Exception as error:
        return result | {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
