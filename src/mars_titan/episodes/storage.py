"""Parquet por bloques recuperables y cohortes bajo demanda, con caché acotada."""

import fcntl
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.cohorts import (
    FINAL_TEST_START_US,
    VALIDATION_START_US,
    read_cohort,
    shapes_contract,
)
from mars_titan.models.baselines.inputs import MODALITIES

from .worlds import recipe_fingerprints

MAX_BYTES = 64 * 1024**2


def _table(world, position):
    raw = world(position)
    world = world.raw_world
    at = position + world.config.context - 1
    rows = len(raw["asset_ids"])
    values = dict(
        asset_id=raw["asset_ids"],
        prediction_at=[raw["prediction_at"]] * rows,
        available_at=raw["available_at"],
        target_available_at=raw["target_available_at"],
        target=raw["target"],
        event=world.events(at),
        fundamental_period_end=[int(world.fundamental_period_end[at])] * rows,
        fundamental_available_at=[int(world.fundamental_available_at[at])] * rows,
    )
    for name, inputs in raw["inputs"].items():
        values[name] = pa.FixedSizeListArray.from_arrays(
            pa.array(inputs.reshape(-1)), int(np.prod(inputs.shape[1:]))
        )
    for col, name in enumerate(("open", "high", "low", "close", "volume")):
        values[name] = world.prices[at, :rows, col]
    period = world.periods[at]
    for name in ("assets", "liabilities", "equity"):
        values[name] = getattr(world, name)[period, :rows]
    values["market_return"] = np.full(rows, world.market_returns[at])
    table = pa.table(values)
    if table.nbytes > MAX_BYTES:
        raise ValueError("La cohorte excede el presupuesto de escritura")
    return table


def _verified(root, record):
    path = root / record["path"]
    if Path(record["path"]).name != record["path"]:
        raise ValueError("El bloque debe permanecer junto a su manifiesto")
    safe_destination(path)
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or sha256(path) != record["sha256"]
    ):
        raise ValueError("El bloque está incompleto o su huella ha cambiado")
    return path


def write_world(world, output, *, resume=False, stop=None, block_sessions=16, check_resources=None):
    """Confirmar cada bloque antes de su cursor y reanudar desde el último confirmado."""
    if type(block_sessions) is not int or not 1 <= block_sessions <= 64:
        raise ValueError("El bloque necesita entre una y 64 sesiones")
    output = Path(output)
    safe_destination(output)
    if output.exists() and not resume or resume and not output.is_dir():
        raise ValueError("Usa una salida nueva o solicita su recuperación")
    output.mkdir(parents=True, exist_ok=resume)
    descriptor = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = json.loads(
            json.dumps(
                dict(
                    world=world.identity,
                    block_sessions=block_sessions,
                    world_content_sha256=world.raw_world.content_sha256(),
                    writer_sha256=sha256(Path(__file__)),
                    transformations=recipe_fingerprints(),
                    pyarrow=pa.__version__,
                )
            )
        )
        progress = output / "progress.json"
        if resume:
            report = json.loads(progress.read_text())
            if report["identity"] != identity:
                raise ValueError("La recuperación pertenece a otro generador o configuración")
            cursor = 0
            if report["index"] != world.index or report["shapes"] != {
                key: list(shape) for key, shape in world.shapes.items()
            }:
                raise ValueError("El índice o las dimensiones de recuperación han cambiado")
            for block in report["blocks"]:
                path = _verified(output, block)
                if (
                    type(block["start"]) is not int
                    or type(block["cohorts"]) is not int
                    or block["start"] != cursor
                    or not 1 <= block["cohorts"] <= block_sessions
                ):
                    raise ValueError("El cursor contiene huecos o bloques inválidos")
                with pq.ParquetFile(path) as file:
                    if file.num_row_groups != block["cohorts"] or cursor + block["cohorts"] > len(
                        world
                    ):
                        raise ValueError("El cursor no coincide con los grupos Parquet")
                    for group in range(file.num_row_groups):
                        actual = file.read_row_group(
                            group, columns=["prediction_at", "asset_id"], use_threads=False
                        )
                        at, rows = world.index[cursor + group]
                        if actual.num_rows != rows or set(actual["prediction_at"].to_pylist()) != {
                            at
                        }:
                            raise ValueError("El bloque no conserva sus cohortes confirmadas")
                cursor += block["cohorts"]
        else:
            report = dict(
                schema_version=1,
                kind="experimental_episodes",
                status="running",
                identity=identity,
                source_sha256=world.source_sha256,
                partition=world.partition,
                domain="synthetic",
                shapes=world.shapes,
                max_assets=world.max_assets,
                encoding=world.identity["encoding"],
                index=world.index,
                blocks=[],
                final_test_opened=False,
            )
            atomic_json(progress, report)
        cursor = sum(block["cohorts"] for block in report["blocks"])
        for start in range(cursor, len(world), block_sessions):
            if check_resources:
                check_resources()
            if recipe_fingerprints() != identity["transformations"]:
                raise ValueError("Las transformaciones han cambiado durante la escritura")
            if stop is not None and stop():
                report["status"] = "paused"
                atomic_json(progress, report)
                return report
            temporary = output / "pending.parquet"
            end = min(start + block_sessions, len(world))
            with pq.ParquetWriter(
                temporary, _table(world, start).schema, compression="zstd"
            ) as writer:
                for position in range(start, end):
                    writer.write_table(_table(world, position))
                    if check_resources:
                        check_resources()
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            digest = sha256(temporary)
            destination = output / f"block-{start:06d}-{digest}.parquet"
            os.replace(temporary, destination)
            report["blocks"].append(
                dict(
                    path=destination.name,
                    sha256=digest,
                    size_bytes=destination.stat().st_size,
                    start=start,
                    cohorts=end - start,
                )
            )
            report["status"] = "running"
            atomic_json(progress, report)
        report["status"] = "completed"
        atomic_json(progress, report)
        atomic_json(output / "manifest.json", report)
        return report
    finally:
        os.close(descriptor)


class EpisodeSource:
    """Leer cohortes con un solo bloque abierto y un máximo de dos grupos en RAM."""

    def __init__(self, manifest, *, max_cache_bytes=MAX_BYTES):
        self.path = Path(manifest)
        safe_destination(self.path)
        if not self.path.is_file() or self.path.stat().st_size > 8 * 1024**2:
            raise ValueError("El manifiesto de episodios no es regular o supera el límite")
        self.metadata = json.loads(self.path.read_text())
        meta = self.metadata
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != "experimental_episodes"
            or meta.get("status") != "completed"
            or meta.get("final_test_opened") is not False
            or meta.get("partition") not in {"train", "validation"}
            or type(max_cache_bytes) is not int
            or not 1 <= max_cache_bytes <= MAX_BYTES
        ):
            raise ValueError("El manifiesto no admite lectura de desarrollo")
        self.max_assets = meta["max_assets"]
        self.shapes = shapes_contract(meta["shapes"], self.max_assets, MAX_BYTES)
        self.partition, self.source_sha256 = meta["partition"], meta["source_sha256"]
        self.encoding, self.index = meta["encoding"], meta["index"]
        self.blocks = meta["blocks"]
        if not 1 <= len(self.index) <= 100_000 or not self.blocks:
            raise ValueError("El índice está vacío o excede el presupuesto")
        cursor = 0
        for block in self.blocks:
            if (
                block["start"] != cursor
                or type(block["cohorts"]) is not int
                or not 1 <= block["cohorts"] <= 64
            ):
                raise ValueError("El índice de bloques tiene huecos o solapamientos")
            cursor += block["cohorts"]
        if cursor != len(self.index):
            raise ValueError("Los bloques no concilian con las cohortes")
        self.manifest_sha256 = sha256(self.path)
        self.manifest_stat = self.path.stat()
        self.cache, self.signatures = OrderedDict(), {}
        self.max_cache_bytes, self.cache_bytes, self.closed = max_cache_bytes, 0, False
        self._file, self._file_block = None, None
        self.starts = np.array([block["start"] for block in self.blocks])

    def __len__(self):
        return len(self.index)

    def _group(self, position, columns):
        block_index = int(np.searchsorted(self.starts, position, side="right") - 1)
        block = self.blocks[block_index]
        path = self.path.parent / block["path"]
        safe_destination(path)
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if block_index in self.signatures:
            if signature != self.signatures[block_index]:
                raise ValueError("El archivo cambió durante la lectura")
        else:
            _verified(self.path.parent, block)
            self.signatures[block_index] = signature
        key = (position, tuple(columns))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self._file_block != block_index:
            if self._file is not None:
                self._file.close()
            self._file, self._file_block = None, None
            self._file = pq.ParquetFile(path)
            self._file_block = block_index
        group = position - block["start"]
        if self._file.metadata.row_group(group).total_byte_size > self.max_cache_bytes:
            raise ValueError("El bloque supera el presupuesto de lectura")
        table = self._file.read_row_group(group, columns=columns, use_threads=False)
        if table.nbytes > self.max_cache_bytes:
            raise ValueError("Las columnas decodificadas superan el presupuesto")
        while self.cache and (
            len(self.cache) >= 2 or self.cache_bytes + table.nbytes > self.max_cache_bytes
        ):
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= old.nbytes
        self.cache[key] = table
        self.cache_bytes += table.nbytes
        return table

    def read(self, position, *, modalities=MODALITIES):
        if self.closed or type(position) is not int or not 0 <= position < len(self):
            raise ValueError("El cursor no pertenece a una fuente abierta")
        current = self.path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
            self.manifest_stat.st_ino,
            self.manifest_stat.st_size,
            self.manifest_stat.st_mtime_ns,
            self.manifest_stat.st_ctime_ns,
        ):
            raise ValueError("El manifiesto cambió durante la lectura")
        if (
            not modalities
            or len(set(modalities)) != len(modalities)
            or not set(modalities) <= set(MODALITIES)
        ):
            raise ValueError("La selección de modalidades no es válida")
        columns = [
            "asset_id",
            "prediction_at",
            "available_at",
            "target_available_at",
            "target",
            *modalities,
        ]
        table = self._group(position, columns).combine_chunks()
        at, rows = self.index[position]
        if table.num_rows != rows or not np.all(table["prediction_at"].to_numpy() == at):
            raise ValueError("El bloque no conserva su cohorte")
        inputs = {
            name: table[name]
            .combine_chunks()
            .flatten()
            .to_numpy()
            .reshape((rows, *self.shapes[name]))
            for name in modalities
        }
        raw = dict(
            prediction_at=at,
            asset_ids=table["asset_id"].to_pylist(),
            inputs=inputs,
            available_at=table["available_at"].to_numpy(),
            target_available_at=table["target_available_at"].to_numpy(),
            target=table["target"].to_numpy(),
        )
        low, high = (
            (0, VALIDATION_START_US)
            if self.partition == "train"
            else (VALIDATION_START_US, FINAL_TEST_START_US)
        )
        if not low <= at < high or (raw["target_available_at"] >= high).any():
            raise ValueError("La cohorte cruza la partición temporal")
        if set(modalities) == set(MODALITIES):
            checked = read_cohort(raw, self.shapes, self.max_assets, MAX_BYTES)
            return {key: value for key, value in checked.items() if key != "sha256"}
        return raw

    def __call__(self, position):
        if type(position) is not int:
            raise ValueError("El cursor debe ser entero")
        if position == len(self) and not self.closed:
            return None
        return self.read(position)

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file, self._file_block = None, None
        self.cache.clear()
        self.cache_bytes = 0
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
