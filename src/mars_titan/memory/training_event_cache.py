"""Archivos de trabajo acotados para reutilizar las entradas verificadas de cada tramo.

Cada tramo de la ventana (ajuste, validación, calibración y evaluación) guarda una copia en su
propia carpeta. La copia se construye una vez desde el lector Parquet con el bloque máximo, se
verifica fila a fila contra él y después se reproduce con los mismos eventos, etiquetas y
cursor. Con el bloque máximo ningún instante ocupa más de una fila guardada, así que la misma
copia sirve a cualquier tamaño de bloque, aunque el ajuste y el lector lean con bloques
distintos. Sin copia válida, el lector vuelve a la ruta original.
"""

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import threading
import time
from contextlib import closing
from pathlib import Path

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.titans.config import MAX_BLOCK_ROWS, canonical
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.training.input_pipeline import background
from mars_titan.training.window_indices import locked

from . import event_archive
from .financial_observations import FinancialObservationSource

ENV = "MARS_TITAN_TRAIN_ARCHIVE"
RESERVE = 4 * 1024**3
KIND = "verified_training_event_archive_v2"
# Bloque con el que se guarda cada copia. Un instante del lector original admite como mucho
# 8.192 entradas, el mismo bloque máximo, así que ninguno ocupa más de una fila guardada.
# `split_events` lo comprueba en cada copia. Con un bloque por consumidor, el ajuste (8.192) y el
# lector (3.584) reconstruían la copia del otro en cada cambio.
STORED_BLOCK = MAX_BLOCK_ROWS
# Tramos que admiten copia. La reserva final no está entre ellos: nunca se empaqueta.
PARTITIONS = ("train", "validation", "calibration", "evaluation")
SPECIFICATION = "input-specification.json"
SPECIFICATION_KIND = "observation_input_specification_v1"
_lock = threading.Lock()
_usage = {}


class PackedPhaseSource(FinancialObservationSource):
    """Fuente de un tramo con transporte verificable e índice canónico independiente."""

    def __init__(self, dataset, manifest, directory):
        super().__init__(dataset, manifest)
        if self.phase.partition not in PARTITIONS:
            raise ValueError("La copia preparada solo admite los tramos de la ventana")
        self.directory = directory
        self._specification = None

    def specification(self):
        """Especificación de entradas del tramo, guardada una vez junto a su índice.

        La primera consulta decodifica el primer evento con entradas, como la fuente original,
        y guarda sus anchuras con la identidad del corpus, del índice, de la política y de la
        representación. Las siguientes, en este proceso o en otro trabajo sobre el mismo
        índice, la reconstruyen sin leer Parquet. Una anchura que no correspondiera a los datos
        la rechazaría la validación del primer bloque y la huella de la copia empaquetada.
        """
        if self._specification is None:
            manifest = self.dataset.manifest
            identity = dict(
                source_sha256=self.dataset.identity,
                view_sha256=self.identity,
                input_policy=manifest["input_policy"],
                representation_sha256=sha256_text(canonical(manifest["representation"])),
            )
            path = self.path.parent / SPECIFICATION
            with locked(path):
                dimensions = _stored_dimensions(path, identity)
                if dimensions is None:
                    specification = super().specification()
                    atomic_json(
                        path,
                        dict(
                            schema_version=1,
                            kind=SPECIFICATION_KIND,
                            identity=identity,
                            dimensions=specification.dimensions,
                        ),
                    )
                    _note("specification_reads")
                else:
                    specification = FinancialInputSpec(
                        source_sha256=self.dataset.identity,
                        view_sha256=self.identity,
                        representation=manifest["representation"],
                        dimensions=dimensions,
                        input_policy=manifest["input_policy"],
                    )
            self._specification = specification
        return self._specification

    def reference_batched_events(self, *, start_cursor=0, block_rows=256, max_cached_bytes=None):
        return super().batched_events(
            start_cursor=start_cursor, block_rows=block_rows, max_cached_bytes=max_cached_bytes
        )

    def batched_events(self, *, start_cursor=0, block_rows=256, max_cached_bytes=None):
        if max_cached_bytes is not None:
            return self.reference_batched_events(
                start_cursor=start_cursor, block_rows=block_rows, max_cached_bytes=max_cached_bytes
            )
        return stream(self, self.directory, start_cursor, block_rows)


def source_for_phase(dataset, manifest, phase):
    """Fuente empaquetada en la carpeta de su tramo si la campaña declara la caché."""
    directory = os.environ.get(ENV)
    if directory and phase.partition in PARTITIONS:
        return PackedPhaseSource(dataset, manifest, Path(directory) / phase.partition)
    return FinancialObservationSource(dataset, manifest)


def supported(source):
    return type(source) in (FinancialObservationSource, PackedPhaseSource)


def source_implementation(source):
    return implementation() if type(source) is PackedPhaseSource else {}


def reset_usage():
    with _lock:
        _usage.clear()


def receipt():
    with _lock:
        return dict(_usage)


def _note(name, amount=1):
    with _lock:
        _usage[name] = _usage.get(name, 0) + amount


def implementation():
    return {
        m: sha256(Path(file))
        for m, file in (
            (event_archive.__name__, event_archive.__file__),
            ("mars_titan.memory.training_event_cache", __file__),
        )
    }


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _stored_dimensions(path, identity):
    """Anchuras guardadas si el archivo corresponde a esta identidad, o None."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
        return None
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if (
        not isinstance(stamp, dict)
        or stamp.get("schema_version") != 1
        or stamp.get("kind") != SPECIFICATION_KIND
        or stamp.get("identity") != identity
        or not isinstance(stamp.get("dimensions"), dict)
    ):
        return None
    return stamp["dimensions"]


def _budget(folder):
    return min(event_archive.MAX_BYTES, max(0, shutil.disk_usage(folder).free - RESERVE))


def _try_lock(handle, mode):
    try:
        fcntl.flock(handle, mode | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _match(path, manifest, expected):
    if not manifest.is_file() or not path.is_file() or manifest.stat().st_size > 64 * 1024:
        return None
    try:
        stamp = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(stamp, dict) or any(stamp.get(k) != v for k, v in expected.items()):
        return None
    verified = stamp.get("verified")
    if (
        not isinstance(verified, dict)
        or not isinstance(verified.get("digest"), str)
        or len(verified["digest"]) != 64
    ):
        return None
    recorded = stamp.get("signature")
    if (
        type(stamp.get("bytes")) is not int
        or not isinstance(stamp.get("sha256"), str)
        or not isinstance(recorded, list)
        or len(recorded) != 4
        or any(type(value) is not int for value in recorded)
    ):
        return None
    try:
        signature = event_archive.verified_signature(path, stamp)
    except event_archive.InvalidArchive:
        _note("invalid_copies")
        return None
    if list(signature) != recorded:
        # Mismo contenido con otra firma, por ejemplo tras cambiar los permisos: cada apertura
        # vuelve a leer la copia entera, y el recibo lo cuenta.
        _note("rehashed_copies")
    return stamp, signature


def _build(source, spec, path, manifest, expected):
    size = _budget(path.parent)
    if size < 1024**2:
        _note("space_fallbacks")
        return None
    stage = path.with_suffix(".pending")
    safe_destination(stage)
    # El cerrojo exclusivo garantiza que este resto procede de una preparación ya cerrada.
    stage.unlink(missing_ok=True)
    start = time.perf_counter()
    try:
        stamp = event_archive.write(
            source,
            spec,
            stage,
            start=0,
            count=expected["events"],
            block=expected["block"],
            max_bytes=size,
            verify_rows=True,
            reserve=RESERVE,
        )
        stamp.update(expected)
        os.replace(stage, path)
        # El renombrado mueve el ctime. La huella se comprueba sobre el archivo ya en su sitio
        # y el sello publica la firma de ese momento para que las pasadas no la repitan.
        signature = event_archive.verified_signature(path, stamp)
        stamp["signature"] = list(signature)
        atomic_json(manifest, stamp)
    except event_archive.ArchiveBudgetError:
        _note("space_fallbacks")
        return None
    except OSError as error:
        if error.errno not in (errno.ENOSPC, errno.EDQUOT):
            raise
        _note("space_fallbacks")
        return None
    finally:
        stage.unlink(missing_ok=True)
    _note("build_seconds", time.perf_counter() - start)
    _note("builds")
    _note("verified_rows", stamp["verified"]["rows"])
    _note("built_bytes", stamp["bytes"])
    return stamp, signature


def stream(source, root, start, block):
    total = len(source.metadata["groups"])
    if type(start) is not int or not 0 <= start <= total:
        raise ValueError("El cursor no pertenece al índice")
    if type(block) is not int or not 1 <= block <= MAX_BLOCK_ROWS:
        raise ValueError("El tamaño del bloque no es válido")
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("La carpeta de entradas preparadas debe ser absoluta")
    safe_destination(root)
    for protected in (
        *source.dataset.roots.values(),
        source.path.parent,
        source.dataset.path.parent,
    ):
        outside_source(protected, root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root_stat = root.stat()
    if root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o022:
        raise ValueError(
            "La carpeta de caché debe pertenecer a la cuenta y solo ella puede escribir"
        )
    path, manifest, lockfile = (
        root / f"events.{extension}" for extension in ("arrow", "json", "lock")
    )
    for item in (path, manifest, lockfile):
        safe_destination(item)
    spec = source.specification()
    expected = dict(
        kind=KIND,
        dataset=source.dataset.identity,
        index=source.identity,
        spec=spec.fingerprint(),
        start=0,
        events=total,
        block=STORED_BLOCK,
        implementation=implementation(),
    )
    original = getattr(source, "reference_batched_events", source.batched_events)
    with lockfile.open("a") as handle:
        if not _try_lock(handle, fcntl.LOCK_SH):
            _note("busy_fallbacks")
            yield from original(start_cursor=start, block_rows=block)
            return
        hit = _match(path, manifest, expected)
        if hit is None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            if start or not _try_lock(handle, fcntl.LOCK_EX):
                _note("cursor_fallbacks" if start else "busy_fallbacks")
                yield from original(start_cursor=start, block_rows=block)
                return
            hit = _match(path, manifest, expected) or _build(source, spec, path, manifest, expected)
            if hit is None:
                fcntl.flock(handle, fcntl.LOCK_UN)
                yield from original(start_cursor=start, block_rows=block)
                return
            # Se conserva el cerrojo exclusivo durante esta primera lectura: cambiar de
            # modo no es atómico en todos los sistemas de archivos.
        stamp, signature = hit
        if not event_archive.compatible(stamp, block):
            # Una copia con instantes divididos solo sirve a los bloques que dividen el suyo.
            # Reconstruirla daría la misma copia, así que este consumidor lee el original.
            fcntl.flock(handle, fcntl.LOCK_UN)
            _note("block_fallbacks")
            yield from original(start_cursor=start, block_rows=block)
            return
        _note("packed_passes")
        with _lock:
            _usage.update(
                archive_sha256=stamp["sha256"],
                index_sha256=stamp["index"],
                input_sha256=stamp["verified"]["digest"],
                block_rows=block,
            )
        events = event_archive.replay(
            source, spec, path, stamp, cursor=start, block=block, signature=signature
        )
        depth = source.dataset.pipeline.prefetch_batches
        yield from background(events, depth) if depth else events


def main():
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.memory.financial_observations import FinancialObservationSource
    from mars_titan.training.corpus_inputs import CorpusDataset

    parser = argparse.ArgumentParser(
        description="Preparar y verificar en CPU la copia cronológica del índice de un tramo"
    )
    parser.add_argument("view", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--block-rows", type=int, default=1024)
    args = parser.parse_args()
    dataset = CorpusDataset(args.view, input_policy=HISTORICAL_MASKED)
    source = FinancialObservationSource(dataset, args.index)
    if source.phase.partition not in PARTITIONS:
        raise ValueError("La preparación solo admite el índice de un tramo de la ventana")
    with closing(stream(source, args.directory, 0, args.block_rows)) as events:
        next(events)
    result = receipt()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("packed_passes") else 2


if __name__ == "__main__":
    raise SystemExit(main())
