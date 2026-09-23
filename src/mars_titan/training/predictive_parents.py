"""Predicciones padre alineadas por clave y procedencia, sin replicar las modalidades."""

import fcntl
import os
import re
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.corpus_source import ParquetCohortSource


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _verified_file(root, record, *, maximum_bytes=16 * 1024**3):
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])
    ):
        raise ValueError("El artefacto padre necesita una ruta y una huella válidas")
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("El artefacto debe permanecer dentro de su ejecución")
    path = root / relative
    safe_destination(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("El artefacto falta o excede su presupuesto")
    signature = _signature(path)
    if sha256(path) != record["sha256"] or _signature(path) != signature:
        raise ValueError("El artefacto ha cambiado o no coincide con su huella")
    return path, signature


def _parent(ordered, parent):
    source, source_hash = read_manifest(ordered)
    report, report_hash = read_manifest(parent, 8 * 1024**2)
    identity = report.get("identity", {})
    manifest_hash = identity.get("manifest_sha256", report.get("manifest_sha256"))
    kind = identity.get("case", {}).get("kind", report.get("model"))
    if (
        source.get("kind") != "causal_prediction_corpus"
        or source.get("status") != "completed"
        or report.get("status") != "completed"
        or report.get("final_test_opened") is not False
        or source.get("final_test_opened") is not False
        or manifest_hash != source["source_sha256"]
        or kind
        not in {"rnn", "lstm", "gru", "dlinear", "ridge", "boosting", "xgboost_external_cuda"}
        or report.get("samples") != source["counts"]
        or report.get("scope") != source["scope"]
        or report.get("cohort_complete") != source["cohort_complete"]
        or set(report.get("predictions", {})) != {"train", "validation"}
    ):
        raise ValueError("El padre no es una referencia terminada de la misma población y objetivo")
    checkpoint, signature = _verified_file(
        parent.parent, report["checkpoint"], maximum_bytes=512 * 1024**2
    )
    weighting = identity.get("weighting", "natural")
    markets = source["partitions"]["train"]["market_rows"]
    if weighting not in {"natural", "balanced_markets"}:
        raise ValueError("La ponderación del padre no está definida")
    weights = {
        market: 1.0 if weighting == "natural" else sum(markets.values()) / (len(markets) * count)
        for market, count in markets.items()
    }
    if identity.get("market_weights", weights) != weights:
        raise ValueError("La ponderación del padre no coincide con la población")
    contract = dict(
        ordered_manifest_sha256=source_hash,
        source_sha256=manifest_hash,
        parent_report_sha256=report_hash,
        checkpoint_sha256=report["checkpoint"]["sha256"],
        model=kind,
        weighting=weighting,
        market_weights=weights,
        counts=source["counts"],
        cohort_id=source.get("cohort_id"),
        news_content_policy=source.get("news_content_policy"),
        horizon_binding="source_manifest_sha256",
        final_test_opened=False,
        code_sha256=sha256(Path(__file__)),
        duckdb=duckdb.__version__,
        numpy=np.__version__,
    )
    return source, report, contract, (checkpoint, signature)


def _prediction_schema(path):
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        footer = stream.read(8)
    if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 64 * 1024**2:
        raise ValueError("Los metadatos de predicción no son válidos o exceden 64 MiB")
    with pq.ParquetFile(path) as file:
        schema = file.schema_arrow
    for name in ("sample_id", "asset_id", "market"):
        if name not in schema.names or not pa.types.is_string(schema.field(name).type):
            raise ValueError("Faltan identidades de texto en las predicciones")
    for name in ("target", "prediction"):
        if name not in schema.names or not pa.types.is_floating(schema.field(name).type):
            raise ValueError("La predicción y el objetivo necesitan columnas numéricas")
    field = schema.field("prediction_at").type
    if not pa.types.is_timestamp(field) or not field.tz or field.unit != "us":
        raise ValueError("La predicción necesita microsegundos y zona horaria")


def _align(source, parent, record, output):
    prediction, signature = _verified_file(parent, record)
    _prediction_schema(prediction)
    count = int(source.offsets[-1])
    with tempfile.TemporaryDirectory(prefix="parent-pending-", dir=output) as name:
        temporary = Path(name)
        mapped = np.lib.format.open_memmap(
            temporary / "values.npy", mode="w+", dtype="<f8", shape=(count,)
        )
        position = 0
        try:
            with duckdb.connect(
                config={
                    "threads": 4,
                    "memory_limit": "512MiB",
                    "temp_directory": str(temporary / "spill"),
                    "max_temp_directory_size": "32GiB",
                    "autoinstall_known_extensions": False,
                    "autoload_known_extensions": False,
                }
            ) as connection:
                total, unique = connection.execute(
                    "SELECT count(*),count(DISTINCT sample_id) FROM read_parquet(?)",
                    [str(prediction)],
                ).fetchone()
                if total != unique or total != count:
                    raise ValueError("El padre tiene claves duplicadas o una cobertura diferente")
                reader = connection.execute(
                    "SELECT b.sample_id,b.asset_id,b.prediction_at,b.target,p.prediction, "
                    "p.target AS parent_target,p.asset_id AS parent_asset, p.market, "
                    "epoch_us(p.prediction_at) AS parent_time "
                    "FROM read_parquet($source_path) b "
                    "LEFT JOIN read_parquet($predictions) p USING(sample_id) "
                    "ORDER BY b.prediction_at,b.asset_id",
                    {"source_path": str(source.path), "predictions": str(prediction)},
                ).to_arrow_reader(batch_size=4096)
                for batch in reader:
                    values = batch.column("prediction").to_numpy(zero_copy_only=False)
                    target = batch.column("target").to_numpy(zero_copy_only=False)
                    parent_target = batch.column("parent_target").to_numpy(zero_copy_only=False)
                    assets = batch.column("asset_id").to_pylist()
                    if (
                        position + len(batch) > count
                        or not np.isfinite(values).all()
                        or not np.isfinite(target).all()
                        or not np.array_equal(target, parent_target)
                        or assets != batch.column("parent_asset").to_pylist()
                        or [asset.split("/")[0] for asset in assets]
                        != batch.column("market").to_pylist()
                        or not np.array_equal(
                            batch.column("prediction_at").to_numpy(),
                            batch.column("parent_time").to_numpy(),
                        )
                    ):
                        raise ValueError(
                            "El join del padre cambia claves, fechas, objetivos o finitud"
                        )
                    mapped[position : position + len(batch)] = values
                    position += len(batch)
            if (
                position != count
                or _signature(prediction) != signature
                or source._signature() != source.signature
            ):
                raise ValueError("La población o los archivos han cambiado durante la alineación")
            mapped.flush()
        finally:
            mapped._mmap.close()
        path = temporary / "values.npy"
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        digest = sha256(path)
        destination = output / f"{source.partition}-{digest}.npy"
        if destination.exists():
            _verified_file(output, dict(path=destination.name, sha256=digest))
        else:
            os.link(path, destination)
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return dict(
            path=destination.name,
            sha256=digest,
            rows=count,
            dtype="float64",
            payload_bytes=count * 8,
            parent_predictions_sha256=record["sha256"],
        )


def prepare_parent_cache(ordered, parent, output, *, resume=False):
    ordered, parent, output = Path(ordered), Path(parent), Path(output)
    source, report, identity, checkpoint = _parent(ordered, parent)
    for protected in (ordered.parent, parent.parent):
        outside_source(protected, output)
        outside_source(output, protected)
    safe_destination(output)
    if output.exists() and not resume:
        raise ValueError("La caché necesita una salida nueva o recuperación explícita")
    if resume and not any((output / name).is_file() for name in ("manifest.json", "progress.json")):
        raise ValueError("No existe una caché confirmada para recuperar")
    output.mkdir(parents=True, exist_ok=resume)
    descriptor = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume and (output / "manifest.json").exists():
            cached = read_manifest(output / "manifest.json")[0]
            if cached["identity"] != identity or cached["status"] != "completed":
                raise ValueError("La caché pertenece a otro padre, código o corpus")
            for partition in ("train", "validation"):
                _verified_file(output, cached["partitions"][partition])
            return cached
        result = (
            read_manifest(output / "progress.json")[0]
            if resume
            else dict(
                schema_version=1,
                kind="aligned_parent_predictions",
                status="running",
                identity=identity,
                model=identity["model"],
                counts=source["counts"],
                checkpoint_sha256=identity["checkpoint_sha256"],
                partitions={},
            )
        )
        if result["identity"] != identity:
            raise ValueError("La recuperación pertenece a otro padre, código o corpus")
        result["status"] = "running"
        atomic_json(output / "progress.json", result)
        try:
            for partition in ("train", "validation"):
                if partition in result["partitions"]:
                    _verified_file(output, result["partitions"][partition])
                else:
                    with ParquetCohortSource(ordered, partition=partition) as cohort:
                        result["partitions"][partition] = _align(
                            cohort, parent.parent, report["predictions"][partition], output
                        )
                atomic_json(output / "progress.json", result)
            if (
                _parent(ordered, parent)[2] != identity
                or _signature(checkpoint[0]) != checkpoint[1]
            ):
                raise ValueError("El contrato del padre o del corpus ha cambiado")
            result["status"] = "completed"
            atomic_json(output / "manifest.json", result)
        except BaseException as error:
            result.update(
                status="failed", last_failure=dict(type=type(error).__name__, message=str(error))
            )
            raise
        finally:
            atomic_json(output / "progress.json", result)
        return result
    finally:
        os.close(descriptor)


class ParentPredictions:
    """Acceder a copias pequeñas de una caché escalar inmutable, sin prestar el mapa."""

    def __init__(self, manifest, source):
        path = Path(manifest)
        metadata, self.manifest_sha256 = read_manifest(path)
        if (
            metadata.get("kind") != "aligned_parent_predictions"
            or metadata.get("status") != "completed"
            or metadata["identity"]["ordered_manifest_sha256"] != source.manifest_sha256
            or metadata["identity"]["source_sha256"] != source.source_sha256
            or metadata["identity"].get("cohort_id") != source.cohort_id
            or metadata["counts"][source.partition] != int(source.offsets[-1])
        ):
            raise ValueError("La caché no pertenece a la fuente causal")
        record = metadata["partitions"][source.partition]
        self.path, self.signature = _verified_file(path.parent, record)
        self._array = np.load(
            self.path, allow_pickle=False, mmap_mode="r", max_header_size=16 * 1024
        )
        if (
            self._array.dtype != np.dtype("float64")
            or self._array.shape != (int(source.offsets[-1]),)
            or _signature(self.path) != self.signature
        ):
            self._array._mmap.close()
            raise ValueError("La caché no conserva su tipo, filas o huella")
        self.dtype, self.payload_bytes, self.closed = self._array.dtype, self._array.nbytes, False

    def values(self, start, size):
        if (
            self.closed
            or type(start) is not int
            or type(size) is not int
            or not 1 <= size <= 4096
            or not 0 <= start <= len(self._array) - size
        ):
            raise ValueError("El tramo solicitado no pertenece a la caché abierta")
        if _signature(self.path) != self.signature:
            raise ValueError("La caché ha cambiado durante la lectura")
        result = self._array[start : start + size].copy()
        if _signature(self.path) != self.signature or not np.isfinite(result).all():
            raise ValueError("La caché ha cambiado o contiene valores no finitos")
        result.setflags(write=False)
        return result

    def close(self):
        if not self.closed:
            self._array._mmap.close()
            self.closed = True
