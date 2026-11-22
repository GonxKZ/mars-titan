#!/usr/bin/env python3
"""Copia coherente del estado de una campaña a una carpeta persistente, y su vuelta.

Una sesión de Colab pierde su disco al terminar. Este guion copia el estado de la campaña
(cola, resultados, puntos de recuperación y registros) a una carpeta que sobrevive, como la de
Drive, mientras los trabajos siguen escribiendo, y lo devuelve a una máquina nueva.

La copia es coherente sin parar nada porque sigue un orden: primero lee los índices (los
registros de la cola, los informes de ejecución y los índices de los puntos de recuperación),
después copia los datos y solo al final publica los índices que leyó. Un índice así publicado
solo nombra archivos que ya existían cuando se leyó. Los estados de recuperación llevan su
huella en el nombre y no cambian; si uno desaparece por rotación antes de copiarse, su índice
no se publica y la copia conserva el anterior.

Uso:
    colab_sync.py push --state <estado local> --mirror <carpeta persistente>
    colab_sync.py pull --state <estado local> --mirror <carpeta persistente>

Las copias empaquetadas de entrada (packed-inputs) no viajan: se regeneran de los datos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

EXCLUDED_DIRECTORIES = ("packed-inputs",)
EXCLUDED_SUFFIXES = (".lock", ".tmp", ".part")
TEMPORARY = ".mars-sync-"
CHECKPOINT_INDEX = "latest.json"
RUN_REPORT = "run.json"
CHUNK = 4 * 1024 * 1024


def is_index(relative):
    """Archivos que nombran a otros y cambian mientras la campaña corre."""
    parts = relative.parts
    return relative.name in (CHECKPOINT_INDEX, RUN_REPORT) or (
        len(parts) == 2 and parts[0] == "queue" and relative.suffix == ".json"
    )


def excluded(relative):
    return (
        any(part in EXCLUDED_DIRECTORIES for part in relative.parts)
        or relative.name.endswith(EXCLUDED_SUFFIXES)
        or ".pending" in relative.name
        or relative.name.startswith(TEMPORARY)
    )


def walk(root):
    """Rutas relativas de los archivos de un árbol, sin seguir ni admitir enlaces."""
    files = []
    for directory, names, leaves in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *leaves]:
            if (base / name).is_symlink():
                raise ValueError(f"Hay un enlace simbólico en {base / name}")
        names[:] = sorted(name for name in names if name not in EXCLUDED_DIRECTORIES)
        for leaf in sorted(leaves):
            relative = (base / leaf).relative_to(root)
            if not excluded(relative):
                files.append(relative)
    return files


def same(source, destination):
    """Mismo tamaño y misma marca de tiempo: la copia conserva la marca del origen."""
    try:
        left, right = source.stat(), destination.stat()
    except FileNotFoundError:
        return False
    return left.st_size == right.st_size and left.st_mtime_ns == right.st_mtime_ns


def write_atomic(destination, data, mtime_ns=None):
    """Escribir en un temporal de la misma carpeta, sincronizar y sustituir de una vez."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f"{TEMPORARY}{os.getpid()}-{destination.name}")
    try:
        with stage.open("wb") as writer:
            if isinstance(data, bytes):
                writer.write(data)
            else:
                shutil.copyfileobj(data, writer, CHUNK)
            writer.flush()
            os.fsync(writer.fileno())
        if mtime_ns is not None:
            os.utime(stage, ns=(mtime_ns, mtime_ns))
        os.replace(stage, destination)
    finally:
        stage.unlink(missing_ok=True)


def copy_file(source, destination):
    """Copiar un archivo. Devuelve sus bytes, o None si desapareció o cambió mientras se leía."""
    # Se escribe primero en un subdirectorio temporal del mismo sistema de ficheros para
    # conservar intacta cualquier copia previa válida en `destination` si el origen cambia.
    stage_dir = destination.parent / f"{TEMPORARY}stage.{os.getpid()}.{id(source)}"
    staged = stage_dir / destination.name
    try:
        before = source.stat()
        with source.open("rb") as reader:
            write_atomic(staged, reader, before.st_mtime_ns)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            # Alguien lo reescribió durante la copia: se descarta el temporal y el destino queda.
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, destination)
        return before.st_size
    except FileNotFoundError:
        return None
    finally:
        staged.unlink(missing_ok=True)
        try:
            stage_dir.rmdir()
        except OSError:
            pass


def referenced_states(index_bytes):
    """Nombres y tamaños de los estados que nombra un índice de puntos de recuperación."""
    document = json.loads(index_bytes)
    records = [*document.get("latest", []), *document.get("pinned", [])]
    if document.get("best") is not None:
        records.append(document["best"])
    return {record["name"]: record.get("size") for record in records}


def complete(index_relative, index_bytes, mirror):
    """Si la copia tiene todos los estados que nombra el índice, con su tamaño."""
    try:
        states = referenced_states(index_bytes)
    except (ValueError, KeyError, TypeError, AttributeError):
        return False
    folder = mirror / index_relative.parent
    for name, size in states.items():
        path = folder / name
        if not path.is_file() or (size is not None and path.stat().st_size != size):
            return False
    return True


def prune_states(index_relative, mirror):
    """Quitar de la copia los estados que su índice publicado ya no nombra."""
    folder = mirror / index_relative.parent
    try:
        keep = set(referenced_states((mirror / index_relative).read_bytes()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 0
    removed = 0
    for path in folder.glob("state-*.pt"):
        if path.name not in keep:
            path.unlink()
            removed += 1
    return removed


def push(state, mirror):
    """Copiar el estado a la carpeta persistente con los índices al final."""
    state, mirror = Path(state), Path(mirror)
    if not state.is_dir():
        raise ValueError(f"No existe el estado {state}")
    mirror.mkdir(parents=True, exist_ok=True)
    files = walk(state)
    indexes = {}
    for relative in files:
        if is_index(relative):
            try:
                path = state / relative
                indexes[relative] = (path.read_bytes(), path.stat().st_mtime_ns)
            except FileNotFoundError:
                continue
    report = dict(copied=0, bytes=0, unchanged=0, retried_later=[], indexes=0, held=[], pruned=0)
    for relative in files:
        if is_index(relative):
            continue
        source, destination = state / relative, mirror / relative
        if same(source, destination):
            report["unchanged"] += 1
            continue
        size = copy_file(source, destination)
        if size is None:
            report["retried_later"].append(relative.as_posix())
        else:
            report["copied"] += 1
            report["bytes"] += size
    # Publicar primero los índices de checkpoints para detectar cuáles se retienen; un `run.json`
    # del mismo directorio `fit/` no debe publicarse si su `checkpoints/latest.json` fue retenido.
    held_fit_dirs = set()
    ordered_indexes = sorted(
        indexes.items(), key=lambda item: (0 if item[0].name == CHECKPOINT_INDEX else 1, item[0])
    )
    for relative, (data, mtime_ns) in ordered_indexes:
        destination = mirror / relative
        if relative.name == CHECKPOINT_INDEX and not complete(relative, data, mirror):
            report["held"].append(relative.as_posix())
            if len(relative.parents) >= 2:
                held_fit_dirs.add(relative.parents[1])
            continue
        if relative.name == "run.json" and relative.parent in held_fit_dirs:
            report["held"].append(relative.as_posix())
            continue
        if not (destination.is_file() and destination.read_bytes() == data):
            write_atomic(destination, data, mtime_ns)
            report["indexes"] += 1
        if relative.name == CHECKPOINT_INDEX:
            report["pruned"] += prune_states(relative, mirror)
    return report


def pull(state, mirror):
    """Devolver la copia a un estado local sin pisar nada distinto que ya exista."""
    state, mirror = Path(state), Path(mirror)
    if not mirror.is_dir():
        raise ValueError(f"No existe la copia {mirror}")
    files = walk(mirror)
    for relative in files:
        if relative.name == CHECKPOINT_INDEX and not complete(
            relative, (mirror / relative).read_bytes(), mirror
        ):
            raise ValueError(f"La copia no tiene todos los estados de {relative}")
    report = dict(copied=0, bytes=0, unchanged=0)
    ordered = [r for r in files if not is_index(r)] + [r for r in files if is_index(r)]
    for relative in ordered:
        source, destination = mirror / relative, state / relative
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise ValueError(f"Ya existe otra cosa en {destination}")
            if destination.stat().st_size != source.stat().st_size or (
                destination.read_bytes() != source.read_bytes()
            ):
                raise ValueError(f"El estado local ya tiene otro contenido en {relative}")
            report["unchanged"] += 1
            continue
        size = copy_file(source, destination)
        if size is None:
            raise ValueError(f"La copia cambió mientras se leía {relative}")
        report["copied"] += 1
        report["bytes"] += size
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("push", "pull"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = (push if args.command == "push" else pull)(args.state, args.mirror)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
