"""Ordenación externa y lectura acotada de cohortes predictivas desde Parquet."""

import argparse
import fcntl
import os
import re
import resource
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.cohort_news import COHORT_POLICIES
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.run_receipts import initialize_receipt

from .actions import ActionGrid
from .cohorts import FINAL_TEST_START_US, VALIDATION_START_US, read_cohort, shapes_contract

MAX_BLOCK_BYTES = 64 * 1024**2


def _code():
    root = Path(__file__).parents[1]
    return {
        name: sha256(root / name)
        for name in (
            "environments/corpus_source.py",
            "environments/cohorts.py",
            "environments/actions.py",
            "training/corpus_inputs.py",
            "training/cohort_contract.py",
            "training/run_receipts.py",
            "data/streaming.py",
            "data/batches.py",
            "data/cohort_files.py",
            "data/cohort_news.py",
            "data/storage.py",
        )
    }


def _tables(dataset, partition, batch_size, shapes, stop):
    for batch in dataset.batches(partition=partition, batch_size=batch_size, epoch=0, seed=0):
        if stop.requested:
            raise InterruptedError("Preparación interrumpida antes de confirmar la partición")
        if np.isnat(batch["input_available_at"]).any():
            raise ValueError("Falta evidencia de disponibilidad para el entorno causal")
        columns = dict(
            cohort_id=pa.array([dataset.cohort] * len(batch["target"]), type=pa.string()),
            sample_id=batch["sample_ids"],
            asset_id=[key.rsplit("/", 1)[0] for key in batch["sample_ids"]],
            prediction_at=batch["prediction_at"].astype(np.int64),
            available_at=batch["input_available_at"].astype(np.int64),
            target_available_at=batch["target_available_at"].astype(np.int64),
            target=batch["target"],
        )
        for name in MODALITIES:
            values = batch["inputs"][name]
            if list(values.shape[1:]) != shapes[name]:
                raise ValueError("Las dimensiones de las modalidades cambian entre bloques")
            columns[name] = pa.FixedSizeListArray.from_arrays(
                pa.array(values.reshape(-1), type=pa.float32()),
                int(np.prod(shapes[name])),
            )
        table = pa.table(columns)
        if table.nbytes > MAX_BLOCK_BYTES:
            raise ValueError("El lote supera el presupuesto de 64 MiB")
        yield table


def _bounded_groups(path, shapes):
    """Reagrupar solo si el mínimo de 2048 filas de DuckDB supera el presupuesto."""
    row_bytes = 4 * sum(int(np.prod(shape)) for shape in shapes.values()) + 512
    with pq.ParquetFile(path) as file:
        oversized = any(
            max(
                file.metadata.row_group(i).total_byte_size,
                file.metadata.row_group(i).num_rows * row_bytes,
            )
            > MAX_BLOCK_BYTES
            for i in range(file.num_row_groups)
        )
        if not oversized:
            return path
        rows = max(1, min(2048, (MAX_BLOCK_BYTES - MAX_BLOCK_BYTES // 8) // row_bytes))
        regrouped = path.with_name("regrouped.parquet")
        atomic_parquet_batches(
            regrouped,
            (
                pa.Table.from_batches([batch])
                for batch in file.iter_batches(batch_size=rows, use_threads=False)
            ),
        )
    with pq.ParquetFile(regrouped) as file:
        if any(
            max(
                file.metadata.row_group(i).total_byte_size,
                file.metadata.row_group(i).num_rows * row_bytes,
            )
            > MAX_BLOCK_BYTES
            for i in range(file.num_row_groups)
        ):
            raise ValueError("El formato Parquet no cabe en el presupuesto del lector")
    return regrouped


def _partition(dataset, output, partition, report, stop):
    with tempfile.TemporaryDirectory(prefix=f"{partition}-pending-", dir=output) as directory:
        temporary = Path(directory)
        raw, ordered = temporary / "input.parquet", temporary / "ordered.parquet"
        count = atomic_parquet_batches(
            raw,
            _tables(dataset, partition, report["identity"]["batch_size"], report["shapes"], stop),
        )
        if count != report["counts"][partition]:
            raise ValueError("La partición no conserva el número de muestras")
        with duckdb.connect(
            config={
                "threads": 4,
                "memory_limit": "4GiB",
                "temp_directory": str(temporary / "spill"),
                "max_temp_directory_size": "32GiB",
                "preserve_insertion_order": True,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            }
        ) as connection:
            connection.execute(
                "COPY (SELECT * FROM read_parquet($source_path) ORDER BY prediction_at, asset_id) "
                "TO $destination (FORMAT PARQUET, ROW_GROUP_SIZE 2048, COMPRESSION ZSTD)",
                {"source_path": str(raw), "destination": str(ordered)},
            )
            index = connection.execute(
                "SELECT prediction_at, count(*), count(DISTINCT asset_id), min(available_at), "
                "max(available_at), min(target_available_at), max(target_available_at) "
                "FROM read_parquet(?) GROUP BY prediction_at ORDER BY prediction_at",
                [str(ordered)],
            ).fetchmany(100_001)
            markets = dict(
                connection.execute(
                    "SELECT split_part(asset_id, '/', 1), count(*) FROM read_parquet(?) GROUP BY 1",
                    [str(ordered)],
                ).fetchall()
            )
        lower, upper = (
            (0, VALIDATION_START_US)
            if partition == "train"
            else (VALIDATION_START_US, FINAL_TEST_START_US)
        )
        if not 1 <= len(index) <= 100_000 or sum(row[1] for row in index) != count:
            raise ValueError("El índice temporal no concilia o supera el presupuesto de cohortes")
        for at, rows, distinct, available_min, available_max, mature_min, mature_max in index:
            if not (
                1 <= rows == distinct <= 4096
                and 0 <= available_min <= available_max <= at < mature_min <= mature_max < upper
                and lower <= at
            ):
                raise ValueError("Las claves, disponibilidad o fechas de una cohorte son inválidas")
        maximum = max(row[1] for row in index)
        shapes_contract(report["shapes"], maximum, MAX_BLOCK_BYTES)
        ordered = _bounded_groups(ordered, report["shapes"])
        digest = sha256(ordered)
        destination = output / f"{partition}-{digest}.parquet"
        with ordered.open("rb") as stream:
            os.fsync(stream.fileno())
        if destination.exists():
            safe_destination(destination)
            if sha256(destination) != digest:
                raise ValueError("Un archivo de contenido confirmado tiene otra huella")
        else:
            os.link(ordered, destination)
        descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return dict(
            path=destination.name,
            sha256=digest,
            size_bytes=destination.stat().st_size,
            rows=count,
            cohorts=[[row[0], row[1]] for row in index],
            max_assets=maximum,
            market_rows=markets,
        )


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file(output, partition, record):
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])
        or type(record.get("size_bytes")) is not int
        or record["size_bytes"] <= 0
    ):
        raise ValueError("La huella o el tamaño de la partición no son válidos")
    expected = f"{partition}-{record['sha256']}.parquet"
    if record.get("path") != expected:
        raise ValueError("La ruta de la partición no corresponde a su huella")
    path = output / expected
    safe_destination(path)
    signature = _signature(path)
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or sha256(path) != record["sha256"]
        or _signature(path) != signature
    ):
        raise ValueError("El archivo confirmado ha cambiado o no conserva su huella")
    return path, signature


def _fit_grid(output, record, source_sha256):
    count = record["rows"]
    if not 1 <= count <= 16_777_216:
        raise ValueError("Las etiquetas de la rejilla exceden su presupuesto de 128 MiB")
    path, signature = _file(output, "train", record)
    targets = np.empty(count, dtype=np.float64)
    position = 0
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(batch_size=4096, columns=["target"], use_threads=False):
            values = batch.column(0).to_numpy()
            if position + len(values) > count:
                raise ValueError("Hay más etiquetas que las declaradas")
            targets[position : position + len(values)] = values
            position += len(values)
    if _signature(path) != signature:
        raise ValueError("Las etiquetas han cambiado durante la lectura")
    if position != count:
        raise ValueError("Faltan etiquetas en la rejilla de entrenamiento")
    return ActionGrid.fit(targets, source_sha256=source_sha256, partition="train").to_dict()


def prepare_causal_corpus(manifest, output, *, batch_size=256, resume=False, stop=None):
    """Confirmar cada partición completa y reutilizarla tras una interrupción."""
    if type(batch_size) is not int or not 1 <= batch_size <= 4096 or type(resume) is not bool:
        raise ValueError("El tamaño del lote o la recuperación no son válidos")
    started = time.perf_counter()
    dataset, output, stop = CorpusDataset(Path(manifest)), Path(output), stop or StopRequest()
    if min(dataset.manifest["counts"].values()) < 1:
        raise ValueError("Se necesitan entrenamiento y validación no vacíos")
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    safe_destination(output)
    if output.exists() and not resume:
        raise ValueError("La preparación necesita una salida nueva o recuperación explícita")
    if resume and not output.exists():
        raise ValueError("No existe una preparación recuperable")
    identity = dict(
        source_sha256=dataset.identity,
        batch_size=batch_size,
        code=_code(),
        duckdb=duckdb.__version__,
        pyarrow=pa.__version__,
        numpy=np.__version__,
        cohort_id=dataset.cohort,
        news_content_policy=dataset.manifest.get("news_content_policy"),
    )
    output.mkdir(parents=True, exist_ok=resume)
    lock = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(output, identity, record="progress.json", lock=".lock")
        if confirmed:
            report = read_manifest(output / "progress.json")[0]
            if report["identity"] != identity:
                raise ValueError("La identidad de datos, versiones o código ha cambiado")
            if report["status"] == "completed":
                final = output / "manifest.json"
                if final.exists() and read_manifest(final)[0] != report:
                    raise ValueError("La preparación final no concuerda con su progreso confirmado")
                for partition in ("train", "validation"):
                    _file(output, partition, report["partitions"][partition])
                ActionGrid.from_dict(report["grid"])
                if not final.exists():
                    atomic_json(final, report)
                return report
        else:
            first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
            report = dict(
                schema_version=1,
                kind="causal_prediction_corpus",
                status="running",
                identity=identity,
                source_sha256=dataset.identity,
                counts=dataset.manifest["counts"],
                cohort_id=dataset.manifest.get("cohort_id"),
                news_content_policy=dataset.manifest.get("news_content_policy"),
                scope=dataset.manifest["scope"],
                cohort_complete=dataset.manifest["cohort_complete"],
                final_test_opened=False,
                shapes={name: list(value.shape[1:]) for name, value in first["inputs"].items()},
                partitions={},
            )
        report["status"] = "running"
        atomic_json(output / "progress.json", report)
        try:
            for partition in ("train", "validation"):
                if stop.requested:
                    raise InterruptedError("Preparación interrumpida entre particiones")
                if partition in report["partitions"]:
                    _file(output, partition, report["partitions"][partition])
                else:
                    report["partitions"][partition] = _partition(
                        dataset, output, partition, report, stop
                    )
                if _code() != identity["code"] or sha256(dataset.path) != dataset.identity:
                    raise ValueError(
                        "El código o manifiesto de origen ha cambiado durante la preparación"
                    )
                atomic_json(output / "progress.json", report)
            report["grid"] = _fit_grid(output, report["partitions"]["train"], dataset.identity)
            report["status"] = "completed"
        except InterruptedError:
            report["status"] = "paused"
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report.update(
                attempt_seconds=time.perf_counter() - started,
                process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            )
            atomic_json(output / "progress.json", report)
        if report["status"] == "completed":
            atomic_json(output / "manifest.json", report)
        return report
    finally:
        os.close(lock)


class ParquetCohortSource:
    """Leer una cohorte por índice, con dos grupos como máximo en una caché acotada."""

    def __init__(self, manifest, *, partition, max_cache_bytes=MAX_BLOCK_BYTES):
        self.manifest_path = Path(manifest)
        meta, self.manifest_sha256 = read_manifest(self.manifest_path)
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != "causal_prediction_corpus"
            or meta.get("status") != "completed"
            or meta.get("final_test_opened") is not False
            or partition not in {"train", "validation"}
            or type(max_cache_bytes) is not int
            or not 1 <= max_cache_bytes <= 128 * 1024**2
        ):
            raise ValueError("El manifiesto o el presupuesto de lectura no es válido")
        record = meta["partitions"][partition]
        self.cohort_id = meta.get("cohort_id")
        self.news_content_policy = meta.get("news_content_policy")
        if (
            meta["identity"].get("cohort_id") != self.cohort_id
            or meta["identity"].get("news_content_policy") != self.news_content_policy
            or (self.cohort_id is None and self.news_content_policy is not None)
            or (
                self.cohort_id is not None
                and COHORT_POLICIES.get(self.cohort_id) != self.news_content_policy
            )
        ):
            raise ValueError("La cohorte no conserva su procedencia y política editorial")
        self.path, verified_signature = _file(self.manifest_path.parent, partition, record)
        self.shapes = shapes_contract(meta["shapes"], record["max_assets"], MAX_BLOCK_BYTES)
        self.max_assets, self.max_cache_bytes = record["max_assets"], max_cache_bytes
        self.source_sha256, self.partition = meta["source_sha256"], partition
        self.index = record["cohorts"]
        low, high = (
            (0, VALIDATION_START_US)
            if partition == "train"
            else (VALIDATION_START_US, FINAL_TEST_START_US)
        )
        if (
            not isinstance(self.index, list)
            or not 1 <= len(self.index) <= 100_000
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(type(v) is not int for v in row)
                or not low <= row[0] < high
                or not 1 <= row[1] <= self.max_assets
                for row in self.index
            )
            or any(a[0] >= b[0] for a, b in zip(self.index, self.index[1:], strict=False))
            or sum(row[1] for row in self.index) != record["rows"]
            or record["rows"] != meta["counts"][partition]
        ):
            raise ValueError("El índice de cohortes no concilia con la población")
        with self.path.open("rb") as stream:
            stream.seek(-8, 2)
            footer = stream.read(8)
        if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 64 * 1024**2:
            raise ValueError("Los metadatos Parquet exceden el presupuesto o no son válidos")
        self.file = pq.ParquetFile(self.path)
        if self.file.metadata.num_rows != record["rows"]:
            self.file.close()
            raise ValueError("Las filas Parquet no concilian con el recibo")
        self.signature = self._signature()
        if self.signature != verified_signature:
            self.file.close()
            raise ValueError("El archivo ha cambiado después de verificar su huella")
        self.offsets = np.cumsum([0] + [row[1] for row in self.index])
        self.groups = np.cumsum(
            [0]
            + [self.file.metadata.row_group(i).num_rows for i in range(self.file.num_row_groups)]
        )
        self.cache, self.cache_bytes, self.closed = OrderedDict(), 0, False

    def _signature(self):
        return _signature(self.path)

    def __len__(self):
        return len(self.index)

    def _group(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        if self.file.metadata.row_group(index).total_byte_size > self.max_cache_bytes:
            raise ValueError("El grupo Parquet supera el presupuesto")
        table = self.file.read_row_group(index, use_threads=False)
        if table.nbytes > self.max_cache_bytes:
            raise ValueError("El grupo decodificado supera el presupuesto")
        while self.cache and (
            len(self.cache) >= 2 or self.cache_bytes + table.nbytes > self.max_cache_bytes
        ):
            _, previous = self.cache.popitem(last=False)
            self.cache_bytes -= previous.nbytes
        self.cache[index] = table
        self.cache_bytes += table.nbytes
        return table

    def __call__(self, position):
        if self.closed or type(position) is not int or not 0 <= position <= len(self):
            raise ValueError("El cursor no pertenece a una fuente abierta")
        safe_destination(self.path)
        if self._signature() != self.signature:
            raise ValueError("El archivo Parquet ha cambiado durante la lectura")
        if position == len(self):
            return None
        start, end = map(int, self.offsets[position : position + 2])
        pieces = []
        first = int(np.searchsorted(self.groups, start, side="right") - 1)
        for group in range(first, self.file.num_row_groups):
            if self.groups[group] >= end:
                break
            begin, stop = max(start, int(self.groups[group])), min(end, int(self.groups[group + 1]))
            pieces.append(self._group(group).slice(begin - int(self.groups[group]), stop - begin))
        table = pa.concat_tables(pieces).combine_chunks()
        if "cohort_id" not in table.column_names or any(
            value != self.cohort_id for value in table["cohort_id"].to_pylist()
        ):
            raise ValueError("Las filas no pertenecen a la cohorte declarada")
        at = self.index[position][0]
        if table.num_rows != end - start or not np.all(table["prediction_at"].to_numpy() == at):
            raise ValueError("Las posiciones no corresponden a la cohorte declarada")
        inputs = {}
        for name, shape in self.shapes.items():
            column = table[name].combine_chunks()
            lengths = pa.compute.list_value_length(column).to_numpy()
            if column.null_count or not np.all(lengths == int(np.prod(shape))):
                raise ValueError("La modalidad no conserva sus dimensiones")
            inputs[name] = column.flatten().to_numpy().reshape((len(table), *shape))
        raw = dict(
            prediction_at=at,
            asset_ids=table["asset_id"].to_pylist(),
            inputs=inputs,
            available_at=table["available_at"].to_numpy(),
            target_available_at=table["target_available_at"].to_numpy(),
            target=table["target"].to_numpy(),
        )
        checked = read_cohort(raw, self.shapes, self.max_assets, MAX_BLOCK_BYTES)
        upper = VALIDATION_START_US if self.partition == "train" else FINAL_TEST_START_US
        if np.any(checked["target_available_at"] >= upper):
            raise ValueError("La etiqueta cruza la partición declarada")
        if self._signature() != self.signature:
            raise ValueError("El archivo ha cambiado durante la lectura de una cohorte")
        return {key: value for key, value in checked.items() if key != "sha256"}

    def close(self):
        if not self.closed:
            self.file.close()
            self.cache.clear()
            self.cache_bytes = 0
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    with StopRequest() as stop:
        report = prepare_causal_corpus(**vars(parser.parse_args()), stop=stop)
    print(f"Estado: {report['status']}. Filas previstas: {report['counts']}")


if __name__ == "__main__":
    main()
