"""Paquete de datos de la campaña de comparación para llevarla a otro equipo (#836).

El segundo equipo lee los mismos archivos que este: por cada origen declarado, la vista de su
ventana y, para los activos de su cohorte, precios preparados, muestras codificadas y
etiquetas. `export_bundle` los copia a uno o varios archivos tar sin compresión (Parquet ya
está comprimido) junto a `bundle.json`, que lista cada archivo con su tamaño y su huella.
`import_bundle` los restaura en el otro equipo y comprueba cada huella mientras escribe.

Las vistas y los manifiestos guardan rutas absolutas dentro de la carpeta personal, así que
el paquete se restaura en las mismas rutas. El equipo de destino necesita la misma carpeta
personal que el de origen: `import_bundle` lo comprueba antes de escribir nada.

El paquete solo contiene derivados locales de la edición. No se publica ni se versiona.
"""

import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json

KIND = "comparison_campaign_data_v1"
MANIFEST = "bundle.json"
PART = "mars-titan-comparison-data-{:03d}.tar"
ARTIFACTS = ("prices", "samples", "labels")
_VIEW_BYTES = 64 * 1024**2
_CHUNK = 4 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _regular(path):
    return path.is_file() and not path.is_symlink()


def campaign_files(declaration, layout, *, native=None):
    """Archivos que leen los trabajos de la campaña, con la huella que ya declara su vista.

    Devuelve `{ruta absoluta: sha256 o None}` en orden estable. Las cohortes deben estar
    preparadas, porque deciden qué activos viajan. De cada activo viaja su carpeta entera de
    datos preparados y de muestras, y sus etiquetas de cada ventana. `native` es el enlace
    episódico compilado: si se indica, viajan también las bibliotecas de su carpeta.
    """
    files = {}

    def add(path, digest=None):
        path = Path(path)
        _require(path.is_absolute() and _regular(path), f"Falta el archivo {path}")
        previous = files.get(path)
        _require(
            previous is None or digest is None or previous == digest,
            f"Dos huellas declaradas para {path}",
        )
        files[path] = previous or digest

    for origin in declaration["origins"]:
        source = layout.source(origin["fold"])
        view, _ = read_manifest(source, _VIEW_BYTES)
        cohort_path = layout.cohort(origin["year"]) / "manifest.json"
        _require(cohort_path.is_file(), f"Falta la cohorte de {origin['year']}. Prepárala antes")
        cohort, _ = read_manifest(cohort_path, _VIEW_BYTES)
        add(source)
        # Manifiestos de procedencia que cita la vista. Viajan si existen en este equipo.
        for path in (
            (view.get("parent_preparation") or {}).get("path"),
            (view.get("temporal_view") or {}).get("parent_manifest"),
        ):
            if path and _regular(Path(path)):
                add(path)
        for market in view["markets"]:
            factor = view["market_factors"][market]
            add(factor["prices_path"], factor["prices_sha256"])
        roots = {name: Path(path) for name, path in cohort["roots"].items()}
        for asset in cohort["assets"]:
            folder = Path(asset["market"]) / asset["symbol"]
            for kind in ARTIFACTS:
                root = roots["prepared" if kind == "prices" else kind]
                add(root / folder / f"{kind}.parquet", asset[f"{kind}_sha256"])
            for root in (roots["prepared"], roots["samples"]):
                for path in sorted((root / folder).rglob("*")):
                    if _regular(path) and not path.name.startswith("."):
                        add(path)
    if native is not None:
        native = Path(native)
        add(native)
        for path in sorted(native.parent.glob("*.so*")):
            if _regular(path):
                add(path)
    return dict(sorted(files.items()))


class _Hashing:
    """Lector que calcula la huella de lo que entrega, para copiar y comprobar en una pasada."""

    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def _tar_size(written):
    """Tamaño del tar cerrado, incluidos sus dos bloques finales y su relleno."""
    record = tarfile.RECORDSIZE
    return ((written + 2 * tarfile.BLOCKSIZE + record - 1) // record) * record


def export_bundle(files, destination, *, home=None, part_bytes=None, identity=None):
    """Escribir el paquete en `destination` y devolver su manifiesto.

    `files` es el resultado de `campaign_files`. Cada archivo se lee una vez: su huella se
    calcula mientras se copia y debe coincidir con la que declara su vista. `part_bytes`
    reparte el paquete en varios tar, para discos que no admiten archivos grandes.
    """
    home = Path(home or Path.home())
    destination = Path(destination)
    _require(destination.is_dir(), f"El destino {destination} debe ser una carpeta existente")
    _require(
        not (destination / MANIFEST).exists() and not list(destination.glob("*.tar")),
        "El destino ya contiene un paquete. Usa una carpeta vacía",
    )
    _require(
        part_bytes is None or (type(part_bytes) is int and part_bytes >= 1024**2),
        "El tamaño de cada parte es un entero de al menos 1 MiB",
    )
    _require(files, "No hay archivos que exportar")
    for path in files:
        _require(path.is_relative_to(home), f"{path} queda fuera de la carpeta personal {home}")
    # Margen de metadatos y relleno por archivo, además de sus bytes de contenido.
    estimated = sum(path.stat().st_size + 16 * 1024 for path in files) + 1024**2
    _require(
        shutil.disk_usage(destination).free >= estimated,
        f"El paquete necesita aproximadamente {estimated / 2**30:.2f} GiB de espacio libre",
    )
    records, parts = [], []
    archive, written = None, 0
    try:
        for path, declared in files.items():
            _require(_regular(path), f"Falta el archivo regular {path}")
            metadata = path.stat()
            size = metadata.st_size
            name = path.relative_to(home).as_posix()
            information = tarfile.TarInfo(name)
            information.size = size
            information.mode = metadata.st_mode & 0o777
            information.mtime = metadata.st_mtime
            blocks = -(-size // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
            entry_size = len(information.tobuf(format=tarfile.PAX_FORMAT)) + blocks
            _require(
                part_bytes is None or _tar_size(entry_size) <= part_bytes,
                f"{path} no cabe en una parte de {part_bytes} bytes",
            )
            if archive is None or (part_bytes and _tar_size(written + entry_size) > part_bytes):
                if archive is not None:
                    archive.close()
                parts.append(PART.format(len(parts) + 1))
                archive = tarfile.open(destination / parts[-1], "w", format=tarfile.PAX_FORMAT)
                written = 0
            with path.open("rb") as stream:
                reader = _Hashing(stream)
                archive.addfile(information, reader)
            digest = reader.digest.hexdigest()
            _require(
                declared in (None, digest),
                f"{path} no conserva la huella que declara su vista",
            )
            _require(path.stat().st_size == size, f"{path} cambió mientras se exportaba")
            records.append(dict(path=name, bytes=size, sha256=digest, part=parts[-1]))
            written += entry_size
    finally:
        if archive is not None:
            archive.close()
    manifest = dict(
        schema_version=1,
        kind=KIND,
        home=str(home),
        parts=parts,
        files=records,
        total_bytes=sum(record["bytes"] for record in records),
        identity=identity or {},
        final_test_opened=False,
    )
    atomic_json(destination / MANIFEST, manifest)
    return manifest


def _checked_manifest(source):
    manifest, _ = read_manifest(Path(source) / MANIFEST, 256 * 1024**2)
    _require(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 1
        and manifest.get("kind") == KIND
        and manifest.get("final_test_opened") is False
        and isinstance(manifest.get("files"), list)
        and isinstance(manifest.get("parts"), list)
        and isinstance(manifest.get("home"), str),
        "La carpeta no contiene un paquete de datos de la campaña de comparación",
    )
    parts = manifest["parts"]
    _require(
        bool(parts)
        and all(
            isinstance(part, str) and part.endswith(".tar") and Path(part).name == part
            for part in parts
        )
        and len(set(parts)) == len(parts),
        "El manifiesto del paquete declara partes no válidas",
    )
    names = set()
    for record in manifest["files"]:
        name = record.get("path") if isinstance(record, dict) else None
        _require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and name not in names
            and type(record.get("bytes")) is int
            and isinstance(record.get("sha256"), str)
            and record.get("part") in manifest["parts"],
            "El manifiesto del paquete declara un archivo no válido",
        )
        names.add(name)
    return manifest


def _same(path, record):
    """El archivo ya existe con el tamaño y la huella del paquete."""
    if not _regular(path) or path.stat().st_size != record["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(_CHUNK):
            digest.update(data)
    return digest.hexdigest() == record["sha256"]


def import_bundle(source, *, home=None):
    """Restaurar el paquete en la carpeta personal y comprobar cada huella al escribir.

    Un archivo que ya existe con la misma huella se conserva. Uno distinto detiene la
    importación sin sobrescribirlo. Devuelve el manifiesto con los recuentos de lo hecho.
    """
    home = Path(home or Path.home())
    source = Path(source)
    manifest = _checked_manifest(source)
    _require(
        str(home) == manifest["home"],
        f"El paquete se exportó desde {manifest['home']} y sus manifiestos guardan esas rutas. "
        f"Este equipo necesita la misma carpeta personal, no {home}",
    )
    expected = {record["path"]: record for record in manifest["files"]}
    restored, kept, seen = 0, 0, set()
    for part in manifest["parts"]:
        path = source / part
        _require(path.is_file(), f"Falta la parte {part} del paquete")
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                _require(
                    member.name not in seen,
                    f"El paquete contiene un archivo repetido: {member.name}",
                )
                record = expected.get(member.name)
                _require(
                    record is not None
                    and record["part"] == part
                    and member.isreg()
                    and member.size == record["bytes"],
                    f"La parte {part} contiene {member.name}, que el manifiesto no declara así",
                )
                target = home / member.name
                safe_destination(target)
                seen.add(member.name)
                if target.exists():
                    _require(
                        _same(target, record),
                        f"{target} ya existe con otro contenido. No se sobrescribe",
                    )
                    kept += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".importing", dir=target.parent
                )
                temporary = Path(name)
                try:
                    digest = hashlib.sha256()
                    with (
                        os.fdopen(descriptor, "wb") as output,
                        archive.extractfile(member) as stream,
                    ):
                        while data := stream.read(_CHUNK):
                            digest.update(data)
                            output.write(data)
                        output.flush()
                        os.fsync(output.fileno())
                    _require(
                        digest.hexdigest() == record["sha256"],
                        f"{member.name} llegó dañado: su huella no coincide",
                    )
                    os.chmod(temporary, member.mode & 0o777 or 0o600)
                    # Publicar sin sobrescribir un archivo creado durante la copia.
                    safe_destination(target)
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        _require(
                            _same(target, record),
                            f"{target} ya existe con otro contenido. No se sobrescribe",
                        )
                        kept += 1
                    else:
                        restored += 1
                finally:
                    temporary.unlink()
    missing = sorted(expected.keys() - seen)
    _require(not missing, f"El paquete no trae {len(missing)} archivos, el primero {missing[:1]}")
    return dict(
        files=len(expected),
        restored=restored,
        kept=kept,
        total_bytes=manifest["total_bytes"],
        identity=manifest["identity"],
    )
