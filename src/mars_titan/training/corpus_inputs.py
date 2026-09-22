"""Lotes supervisados por activo y grupo Parquet, sin acumular el corpus en RAM."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.batches import read_bounded_table
from mars_titan.data.storage import sha256
from mars_titan.data.streaming import price_features

VECTORS = ("news", "charts", "fundamentals", "macro")
MAX_TABLE_BYTES = 64 * 1024**2


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("El manifiesto contiene una clave duplicada")
        result[key] = value
    return result


def _times(column):
    if not pa.types.is_timestamp(column.type) or not column.type.tz or column.null_count:
        raise ValueError("Las fechas necesitan una zona y no pueden contener valores ausentes")
    return column.cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy()


def _random(seed, epoch, key):
    number = int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{key}".encode()).digest()[:8], "big")
    return np.random.default_rng(number)


class CorpusDataset:
    """Validar una edición y reutilizar sus huellas mientras no cambien los archivos."""

    def __init__(self, manifest: Path):
        self.path = Path(manifest)
        if self.path.is_symlink() or self.path.stat().st_size > 8 * 1024**2:
            raise ValueError("El manifiesto no es regular o supera 8 MiB")
        self.identity = sha256(self.path)
        self.manifest = json.loads(self.path.read_text(), object_pairs_hook=_unique)
        meta = self.manifest
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != "corpus_supervision"
            or meta.get("scope") not in {"development_snapshot", "full_corpus"}
            or type(meta.get("cohort_complete")) is not bool
            or (meta["scope"] == "full_corpus" and not meta["cohort_complete"])
            or type(meta.get("context_sessions")) is not int
            or not 2 <= meta["context_sessions"] <= 512
            or set(meta.get("roots", {})) != {"prepared", "samples", "labels"}
            or not isinstance(meta.get("assets"), list)
            or not meta["assets"]
        ):
            raise ValueError("El manifiesto supervisado no cumple su contrato")
        self.context = meta["context_sessions"]
        self.roots = {key: Path(value).resolve() for key, value in meta["roots"].items()}
        self.assets = meta["assets"]
        self.verified = {}
        identities, counts = set(), {"train": 0, "validation": 0}
        for asset in self.assets:
            symbol, market = asset["symbol"], asset["market"]
            if (
                not isinstance(symbol, str)
                or symbol in {".", ".."}
                or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
                or market not in {"US", "CN"}
                or (market, symbol) in identities
                or set(asset["counts"]) != set(counts)
                or any(type(n) is not int or n < 0 for n in asset["counts"].values())
            ):
                raise ValueError(
                    "El activo está duplicado o tiene una identidad o recuento inválido"
                )
            identities.add((market, symbol))
            for partition in counts:
                counts[partition] += asset["counts"][partition]
            for kind in ("prices", "samples", "labels"):
                self._file(asset, kind)
        if counts != meta.get("counts"):
            raise ValueError("Los recuentos del corpus no concilian con los activos")

    def _file(self, asset, kind):
        root = self.roots["prepared" if kind == "prices" else kind]
        path = root / asset["market"] / asset["symbol"] / f"{kind}.parquet"
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Falta un artefacto regular dentro del origen declarado")
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.verified.get(path) != signature:
            if sha256(path) != asset[kind + "_sha256"]:
                raise ValueError("Un artefacto supervisado ha cambiado desde su confirmación")
            after = path.stat()
            if signature != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("Un artefacto ha cambiado durante la comprobación")
            self.verified[path] = signature
        with path.open("rb") as stream:
            stream.seek(-8, 2)
            footer = stream.read(8)
        if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 8 * 1024**2:
            raise ValueError("La cabecera final de Parquet no es válida o excede su presupuesto")
        return path

    def _labels(self, asset, partition, sample_rows):
        table = read_bounded_table(
            self._file(asset, "labels"), max_rows=1_000_000, max_bytes=MAX_TABLE_BYTES
        )
        if not pa.types.is_integer(table["sample_row"].type) or table["sample_row"].null_count:
            raise ValueError("La etiqueta necesita una posición entera de muestra")
        positions = table["sample_row"].to_numpy()
        if np.any(positions < 0) or np.any(positions >= sample_rows):
            raise ValueError("Una etiqueta no corresponde a una posición del archivo de muestras")
        prediction = _times(table["prediction_at"])
        if len(np.unique(positions)) != len(positions) or len(np.unique(prediction)) != len(
            prediction
        ):
            raise ValueError("Hay etiquetas duplicadas para la misma muestra o decisión")
        if len(positions) != sample_rows:
            raise ValueError("Falta una etiqueta o una exclusión explícita de una muestra")
        reasons = (
            table["reason"].to_pylist()
            if "reason" in table.column_names
            else ["accepted"] * len(table)
        )
        allowed = {
            "accepted",
            "insufficient_history",
            "zero_market_variance",
            "missing_next_session",
            "target_after_cutoff",
            "target_crosses_partition_boundary",
            "outside_label_calendar",
            "final_test_reserved",
        }
        if not set(reasons) <= allowed:
            raise ValueError("Hay un motivo de exclusión de etiqueta desconocido")
        partition_values = table["partition"].to_pylist()
        if any(
            part is not None
            for part, reason in zip(partition_values, reasons, strict=True)
            if reason != "accepted"
        ):
            raise ValueError("Una etiqueta excluida no puede asignarse a una partición")
        accepted = np.flatnonzero(np.asarray(reasons) == "accepted")
        table = table.take(pa.array(accepted, type=pa.int64()))
        positions, prediction = positions[accepted], prediction[accepted]
        if not len(table):
            if any(asset["counts"].values()):
                raise ValueError("La partición no concilia con sus etiquetas")
            return positions, prediction, np.empty(0, dtype=np.float64)
        maturity = _times(table["target_available_at"])
        if np.any(maturity <= prediction):
            raise ValueError("La etiqueta debe madurar después de su predicción")
        values = table["target"].to_numpy()
        if table["target"].null_count or not np.isfinite(values).all():
            raise ValueError("Hay etiquetas ausentes o no finitas")
        partitions = np.asarray(table["partition"].to_pylist())
        years = prediction.astype("datetime64[us]").astype("datetime64[Y]").astype(int) + 1970
        mature_years = maturity.astype("datetime64[us]").astype("datetime64[Y]").astype(int) + 1970
        valid = ((partitions == "train") & (years <= 2022) & (mature_years <= 2022)) | (
            (partitions == "validation") & (years == 2023) & (mature_years == 2023)
        )
        if not valid.all() or any(
            int(np.sum(partitions == p)) != asset["counts"][p] for p in ("train", "validation")
        ):
            raise ValueError("Una etiqueta cruza la partición o sus recuentos no concilian")
        selected = np.flatnonzero(partitions == partition)
        selected = selected[np.argsort(positions[selected])]
        return positions[selected], prediction[selected], values[selected]

    def _rows(self, partition, epoch, seed, cursor):
        order = _random(seed, epoch, "assets").permutation(len(self.assets))
        consumed = sum(self.assets[int(i)]["counts"][partition] for i in order[: cursor["asset"]])
        dimensions = None
        for asset_position in range(cursor["asset"], len(order)):
            asset = self.assets[int(order[asset_position])]
            key = f"{asset['market']}/{asset['symbol']}"
            with pq.ParquetFile(self._file(asset, "samples")) as file:
                positions, prediction, target = self._labels(
                    asset, partition, file.metadata.num_rows
                )
                price_table = read_bounded_table(
                    self._file(asset, "prices"), max_rows=200_000, max_bytes=MAX_TABLE_BYTES
                )
                prices = np.column_stack(
                    [price_table[c].to_numpy() for c in ("open", "high", "low", "close", "volume")]
                )
                available = _times(price_table["available_at"])
                if np.any(np.diff(available) <= 0):
                    raise ValueError("Los precios necesitan un orden temporal único")
                groups = _random(seed, epoch, key + "/groups").permutation(file.num_row_groups)
                offsets = np.cumsum(
                    [0] + [file.metadata.row_group(g).num_rows for g in range(file.num_row_groups)]
                )
                start_group = cursor["group"] if asset_position == cursor["asset"] else 0
                if not 0 <= start_group < max(1, len(groups)):
                    raise ValueError("El cursor señala un grupo inexistente")
                for group_position, group in enumerate(groups):
                    first, end = np.searchsorted(positions, [offsets[group], offsets[group + 1]])
                    if group_position < start_group:
                        consumed += end - first
                        continue
                    indexes = np.arange(first, end)
                    if partition == "train":
                        indexes = _random(seed, epoch, key + f"/{group}").permutation(indexes)
                    start = (
                        cursor["offset"]
                        if (asset_position, group_position) == (cursor["asset"], cursor["group"])
                        else 0
                    )
                    if not 0 <= start <= len(indexes):
                        raise ValueError("El cursor señala una posición inexistente")
                    if (asset_position, group_position) == (cursor["asset"], cursor["group"]):
                        consumed += start
                        if consumed != cursor["consumed"]:
                            raise ValueError("El consumo confirmado del cursor no concilia")
                    if not len(indexes) or start == len(indexes):
                        continue
                    columns = ["prediction_at", "price_end_index", *VECTORS]
                    metadata = file.metadata.row_group(int(group))
                    size = sum(
                        metadata.column(c).total_uncompressed_size
                        for c in range(metadata.num_columns)
                        if metadata.column(c).path_in_schema.split(".")[0] in columns
                    )
                    if size > MAX_TABLE_BYTES:
                        raise ValueError("El grupo de características supera 64 MiB")
                    table = file.read_row_group(int(group), columns=columns, use_threads=False)
                    if table.nbytes > MAX_TABLE_BYTES:
                        raise ValueError("El grupo decodificado supera el presupuesto")
                    timestamps = _times(table["prediction_at"])
                    ends = table["price_end_index"].to_numpy()
                    vectors = {}
                    for name in VECTORS:
                        column = table[name].combine_chunks()
                        if not (
                            pa.types.is_list(column.type)
                            or pa.types.is_fixed_size_list(column.type)
                        ):
                            raise ValueError("Cada modalidad necesita un vector explícito")
                        lengths = pa.compute.list_value_length(column).to_numpy()
                        if (
                            column.null_count
                            or not len(lengths)
                            or not 1 <= lengths[0] <= 2048
                            or not (lengths == lengths[0]).all()
                        ):
                            raise ValueError("Las dimensiones de una modalidad no son válidas")
                        vectors[name] = np.asarray(
                            column.flatten().to_numpy(), dtype=np.float32
                        ).reshape(len(table), int(lengths[0]))
                    shape = {name: values.shape[1] for name, values in vectors.items()}
                    if dimensions is not None and dimensions != shape:
                        raise ValueError("Las dimensiones cambian entre activos")
                    dimensions = shape
                    for offset in range(start, len(indexes)):
                        label = indexes[offset]
                        row = positions[label] - offsets[group]
                        price_end = ends[row]
                        if (
                            not np.issubdtype(ends.dtype, np.integer)
                            or not self.context - 1 <= price_end < len(prices)
                            or timestamps[row] != prediction[label]
                            or available[price_end] > prediction[label]
                        ):
                            raise ValueError(
                                "Las modalidades y la etiqueta no coinciden temporalmente"
                            )
                        inputs = {name: values[row].copy() for name, values in vectors.items()}
                        if any(not np.isfinite(v).all() for v in inputs.values()):
                            raise ValueError("Una modalidad contiene valores no finitos")
                        inputs["prices"] = price_features(
                            prices[price_end - self.context + 1 : price_end + 1]
                        )
                        consumed += 1
                        yield (
                            inputs,
                            float(target[label]),
                            key,
                            int(prediction[label]),
                            {
                                "asset": asset_position,
                                "group": group_position,
                                "offset": offset + 1,
                                "consumed": int(consumed),
                            },
                        )
            for kind in ("prices", "samples", "labels"):
                self._file(asset, kind)
        if consumed != self.manifest["counts"][partition]:
            raise ValueError("El recorrido no visita exactamente la población declarada")

    def batches(self, *, partition, batch_size, epoch, seed, cursor=None):
        if (
            partition not in {"train", "validation"}
            or type(batch_size) is not int
            or not 1 <= batch_size <= 4096
            or type(epoch) is not int
            or not 0 <= epoch < 2**32
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise ValueError("La partición, el lote, la época o la semilla no son válidos")
        if sha256(self.path) != self.identity:
            raise ValueError("El manifiesto ha cambiado desde su confirmación")
        identity = {
            "manifest_sha256": self.identity,
            "partition": partition,
            "epoch": epoch,
            "seed": seed,
        }
        point = {"asset": 0, "group": 0, "offset": 0, "consumed": 0}
        if cursor is not None:
            if (
                set(cursor) != set(identity) | set(point)
                or any(cursor[k] != v for k, v in identity.items())
                or any(type(cursor[k]) is not int or cursor[k] < 0 for k in point)
                or cursor["asset"] >= len(self.assets)
            ):
                raise ValueError("El cursor no corresponde al corpus, época o partición")
            point = {key: cursor[key] for key in point}
        records = []
        for inputs, target, key, timestamp, next_point in self._rows(partition, epoch, seed, point):
            records.append((inputs, target, key, timestamp))
            if len(records) == batch_size:
                yield _batch(records, {**identity, **next_point})
                records = []
        if records:
            yield _batch(records, {**identity, **next_point})


def _batch(records, cursor):
    return {
        "inputs": {name: np.stack([r[0][name] for r in records]) for name in (*VECTORS, "prices")},
        "target": np.asarray([r[1] for r in records], dtype=np.float64),
        "sample_ids": [f"{r[2]}/{r[3]}" for r in records],
        "market": [r[2].split("/")[0] for r in records],
        "prediction_at": np.asarray([r[3] for r in records], dtype="datetime64[us]"),
        "weight": np.ones(len(records), dtype=np.float64),
        "confirmed_cursor": cursor,
    }


def supervised_batches(
    manifest: Path,
    *,
    partition: str,
    batch_size: int,
    epoch: int,
    seed: int,
    cursor: dict | None = None,
):
    """Crear un lector para una pasada. La campaña reutiliza CorpusDataset entre épocas."""
    yield from CorpusDataset(manifest).batches(
        partition=partition, batch_size=batch_size, epoch=epoch, seed=seed, cursor=cursor
    )


def prepare_corpus_targets(manifest: Path, prepared: Path, output: Path) -> dict:
    """Preparar las etiquetas mediante el mismo contrato que consume este lector."""
    from .corpus_targets import prepare_corpus_targets as prepare

    return prepare(manifest, prepared, output)
