"""Dos archivos de trabajo acotados para reutilizar entradas de ajuste verificadas."""

import argparse
import errno
import fcntl
import json
import os
import shutil
import threading
import time
from contextlib import closing
from pathlib import Path

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.titans.config import MAX_BLOCK_ROWS
from mars_titan.training.input_pipeline import background

from . import event_archive
from .financial_observations import FinancialObservationSource

ENV = "MARS_TITAN_TRAIN_ARCHIVE"
RESERVE = 4 * 1024**3
KIND = "verified_training_event_archive_v1"
_lock = threading.Lock()
_usage = {}


class PackedTrainingSource(FinancialObservationSource):
    """Fuente de ajuste con transporte verificable e índice canónico independiente."""

    def __init__(self, dataset, manifest, directory):
        super().__init__(dataset, manifest)
        if self.phase.partition != "train":
            raise ValueError("La copia preparada solo admite el tramo de ajuste")
        self.directory = directory

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
    directory = os.environ.get(ENV)
    if directory and phase.partition == "train":
        return PackedTrainingSource(dataset, manifest, directory)
    return FinancialObservationSource(dataset, manifest)


def supported(source):
    return type(source) in (FinancialObservationSource, PackedTrainingSource)


def source_implementation(source):
    return implementation() if type(source) is PackedTrainingSource else {}


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
    if type(stamp.get("bytes")) is not int or not isinstance(stamp.get("sha256"), str):
        return None
    try:
        return stamp, event_archive.verified_signature(path, stamp)
    except event_archive.InvalidArchive:
        _note("invalid_copies")
        return None


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
    return stamp, event_archive.verified_signature(path, stamp)


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
    slot = "wide" if block >= 1024 else "small"
    path, manifest, lockfile = (
        root / f"{slot}.{extension}" for extension in ("arrow", "json", "lock")
    )
    for item in (path, manifest, lockfile):
        safe_destination(item)
    spec = getattr(source, "_archive_spec", None)
    if spec is None:
        spec = source._archive_spec = source.specification()
    expected = dict(
        kind=KIND,
        dataset=source.dataset.identity,
        index=source.identity,
        spec=spec.fingerprint(),
        start=0,
        events=total,
        block=block,
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
        description="Preparar y verificar en CPU la copia cronológica de un índice de ajuste"
    )
    parser.add_argument("view", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--block-rows", type=int, default=1024)
    args = parser.parse_args()
    dataset = CorpusDataset(args.view, input_policy=HISTORICAL_MASKED)
    source = FinancialObservationSource(dataset, args.index)
    if source.phase.partition != "train":
        raise ValueError("La preparación solo admite un índice de ajuste")
    with closing(stream(source, args.directory, 0, args.block_rows)) as events:
        next(events)
    result = receipt()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("packed_passes") else 2


if __name__ == "__main__":
    raise SystemExit(main())
