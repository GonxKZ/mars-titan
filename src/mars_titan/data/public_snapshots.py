"""Índice de las capturas públicas confirmadas y reutilización de su contenido.

Una captura confirmada es la inicial de `data/manifests/public-snapshots.json` o un
directorio `data/external/<run_id>/` con su `manifest.json`. El índice es una función pura
de esos manifiestos y de los archivos que nombran: no guarda fechas propias y se
reconstruye al empezar y al terminar cada ejecución. Así una interrupción entre el
manifiesto y el índice no deja nada incoherente, y una ejecución sin manifiesto figura como
interrumpida sin que sus archivos se reutilicen.

El contenido se identifica por su SHA-256. Un archivo solo cuenta como disponible si
existe, es regular y sus bytes tienen la huella del manifiesto. Las capturas siguen fuera
del benchmark: el índice no cambia `benchmark_eligible` ni la decisión de incorporarlas.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path

from mars_titan.data.storage import atomic_json

INITIAL = "data/manifests/public-snapshots.json"
INDEX = "data/manifests/public-snapshots/index.json"
LOCK = ".refresh.lock"
INITIAL_RUN = "initial"
RUN_ID = re.compile(r"\d{8}T\d{6}\.\d{6}Z")
SHA256 = re.compile(r"[a-f0-9]{64}")
# Estados con contenido válido: descarga validada o respuesta 304 que remite a la anterior.
VALID = frozenset({"downloaded_validated", "not_modified"})
# Valores de ETag y Last-Modified que se pueden reenviar como cabecera: ASCII visible,
# sin saltos de línea que permitan inyectar otras cabeceras.
HEADER_VALUE = re.compile(r"[\x21-\x7e][\x20-\x7e]{0,1023}")
RECORD_FIELDS = (
    "source_id",
    "status",
    "http_status",
    "sha256",
    "bytes",
    "local_path",
    "requested_url",
    "etag",
    "last_modified",
    "content_reused",
)


def header_value(value):
    """Devuelve un validador HTTP reenviable o None."""
    return value if isinstance(value, str) and HEADER_VALUE.fullmatch(value) else None


def file_sha256(path):
    """Huella de un archivo regular sin seguir enlaces, o None si no existe."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _local(root, relative):
    """Ruta local de un manifiesto, que debe quedar dentro de la raíz del proyecto."""
    if not isinstance(relative, str):
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Ruta local fuera del proyecto: {relative}")
    return Path(root) / candidate


def _records(document, run_id):
    records = document.get("records")
    if document.get("schema_version") != 1 or not isinstance(records, list):
        raise ValueError(f"Manifiesto de captura no válido: {run_id}")
    result = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("source_id"), str):
            raise ValueError(f"Registro sin fuente en la captura {run_id}")
        entry = {field: record.get(field) for field in RECORD_FIELDS}
        entry["content_reused"] = record.get("content_reused") is True
        valid = record.get("status") in VALID
        if valid and (
            not isinstance(record.get("sha256"), str)
            or not SHA256.fullmatch(record["sha256"])
            or not isinstance(record.get("local_path"), str)
            or type(record.get("bytes")) is not int
            or record["bytes"] < 0
        ):
            raise ValueError(f"Contenido válido sin huella ni ruta en la captura {run_id}")
        for name in ("etag", "last_modified"):
            entry[name] = header_value(record.get(name))
        result.append(entry)
    return result


def _diagnostics(document):
    """Peticiones de diagnóstico sin contenido conservado, como el 429 de GDELT."""
    found = document.get("diagnostics")
    return [
        dict(source_id=item.get("source_id"), http_status=item.get("http_status"))
        for item in (found if isinstance(found, list) else [])
        if isinstance(item, dict) and isinstance(item.get("source_id"), str)
    ]


def _folder_bytes(folder):
    """Bytes de los archivos regulares de una carpeta de ejecución, sin seguir enlaces."""
    total = 0
    with os.scandir(folder) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            total += info.st_size if stat.S_ISREG(info.st_mode) else 0
    return total


def confirmed_runs(output_root):
    """Ejecuciones confirmadas, en orden, y las que no llegaron a escribir su manifiesto.

    Las interrumpidas se devuelven con los bytes que ocupan, porque cuentan para el límite
    de disco aunque su contenido no se reutilice.
    """
    output_root = Path(output_root)
    runs, interrupted = [], []
    if not output_root.is_dir():
        return runs, interrupted
    for folder in sorted(output_root.iterdir()):
        if not RUN_ID.fullmatch(folder.name) or folder.is_symlink() or not folder.is_dir():
            continue
        manifest = folder / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            interrupted.append(dict(run_id=folder.name, bytes=_folder_bytes(folder)))
            continue
        raw = manifest.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or document.get("run_id") != folder.name:
            raise ValueError(f"El manifiesto de {folder.name} no corresponde a su ejecución")
        runs.append((folder.name, manifest, raw, document))
    return runs, interrupted


def build_index(root, output_root, *, initial=INITIAL):
    """Reconstruir el índice desde la captura inicial y los manifiestos confirmados."""
    root = Path(root)
    entries = []
    initial_path = root / initial
    if initial_path.is_file():
        raw = initial_path.read_bytes()
        entries.append((INITIAL_RUN, initial_path, raw, json.loads(raw)))
    has_initial = bool(entries)
    runs, interrupted = confirmed_runs(output_root)
    entries.extend(runs)
    captures, contents, hashes, stored = [], {}, {}, {}

    def available(relative, expected):
        if relative not in hashes:
            hashes[relative] = file_sha256(_local(root, relative))
        return hashes[relative] == expected

    for run_id, path, raw, document in entries:
        records = _records(document, run_id)
        captures.append(
            dict(
                run_id=run_id,
                manifest=_relative(root, path),
                manifest_sha256=hashlib.sha256(raw).hexdigest(),
                started_at_utc=document.get("started_at_utc"),
                finished_at_utc=document.get("finished_at_utc"),
                records=records,
                diagnostics=_diagnostics(document),
            )
        )
        for record in records:
            if record["status"] not in VALID:
                continue
            sha, relative = record["sha256"], record["local_path"]
            found = available(relative, sha)
            if found:
                # Archivos distintos con los mismos bytes ocupan disco igualmente: las
                # capturas anteriores a la reutilización guardaron copias repetidas.
                stored[relative] = record["bytes"]
            known = contents.get(sha)
            if known is not None and known["available"]:
                continue
            if known is None or found:
                contents[sha] = dict(
                    local_path=relative,
                    run_id=run_id,
                    bytes=record["bytes"],
                    first_run_id=known["first_run_id"] if known else run_id,
                    available=found,
                )
    valid = [r for capture in captures for r in capture["records"] if r["status"] in VALID]
    return dict(
        schema_version=1,
        project="MARS-TITAN",
        kind="public_snapshot_index",
        initial_manifest=captures[0]["manifest"] if has_initial else None,
        output_root=_relative(root, output_root),
        captures=captures,
        contents=dict(sorted(contents.items())),
        interrupted_runs=[item["run_id"] for item in interrupted],
        # Bytes que citan los registros válidos, bytes únicos por huella y bytes que ocupan
        # en disco los archivos confirmados y las ejecuciones interrumpidas.
        referenced_bytes=sum(record["bytes"] for record in valid),
        distinct_bytes=sum(item["bytes"] for item in contents.values() if item["available"]),
        stored_bytes=sum(stored.values()),
        interrupted_bytes=sum(item["bytes"] for item in interrupted),
        benchmark_eligible=False,
    )


@contextmanager
def exclusive_update(output_root):
    """Impedir dos actualizaciones simultáneas sobre el mismo almacén de capturas.

    Sin este bloqueo, dos ejecuciones podrían guardar dos copias de los mismos bytes o
    escribir el índice con un estado que ya no incluye la otra captura.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(output_root / LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Hay otra actualización de fuentes públicas en curso") from None
        yield
    finally:
        os.close(fd)


def write_index(root, index):
    path = Path(root) / INDEX
    atomic_json(path, index)
    return path


def reusable_path(root, index, sha256):
    """Ruta de un contenido confirmado con esos bytes, comprobada de nuevo en disco."""
    known = index["contents"].get(sha256)
    if known is None or not known["available"]:
        return None
    path = _local(root, known["local_path"])
    return known["local_path"] if file_sha256(path) == sha256 else None


def read_content(root, index, sha256):
    """Bytes de un contenido confirmado, con su huella comprobada al leerlos."""
    relative = reusable_path(root, index, sha256)
    if relative is None:
        raise ValueError("El contenido confirmado ya no está disponible con su huella")
    body = _local(root, relative).read_bytes()
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError("El contenido confirmado cambió durante la lectura")
    return body, relative


def validators(index, source_id, requested_url):
    """ETag y Last-Modified de la última captura válida de la misma fuente y URL.

    Solo cuenta la más reciente: si ya no trajo validadores, o su contenido no está
    disponible, la petición vuelve a ser completa en lugar de compararse con una versión
    anterior.
    """
    for capture in reversed(index["captures"]):
        for record in reversed(capture["records"]):
            if (
                record["source_id"] != source_id
                or record["status"] not in VALID
                or record["requested_url"] != requested_url
            ):
                continue
            usable = (record["etag"] or record["last_modified"]) and index["contents"].get(
                record["sha256"], {}
            ).get("available")
            if not usable:
                return None
            return dict(
                etag=record["etag"],
                last_modified=record["last_modified"],
                sha256=record["sha256"],
                run_id=capture["run_id"],
            )
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruir el índice de capturas públicas.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-root", default="data/external")
    args = parser.parse_args(argv)
    index = build_index(args.root, args.root / args.output_root)
    path = write_index(args.root, index)
    print(json.dumps(dict(index=path.as_posix(), captures=len(index["captures"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
