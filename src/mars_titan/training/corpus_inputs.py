"""Lotes supervisados por activo y grupo Parquet, sin acumular el corpus en RAM."""

import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.batches import read_bounded_table
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import (
    MODALITIES,
    STRICT_INPUTS,
    masked_inputs,
    validate_historical_vectors,
)
from mars_titan.data.storage import sha256

from .cohort_contract import cohort_identity, representation_identity, validate_cohort_rows

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


def _historical_times(table):
    timestamps = _times(table["prediction_at"])
    if (timestamps < np.datetime64("2000-01-01", "us").astype(np.int64)).any() or (
        timestamps >= np.datetime64("2024-01-01", "us").astype(np.int64)
    ).any():
        raise ValueError("Una muestra histórica queda fuera del corte 2000–2023")
    return timestamps


def _availability(table, *, macro_override=None, presence=None):
    if "input_availability" not in table.column_names:
        if presence is not None:
            raise ValueError("Las máscaras necesitan disponibilidad explícita por bloque")
        return None, None
    column = table["input_availability"].combine_chunks()
    names = {"prices", "news", "charts", "fundamentals", "macro"}
    if not pa.types.is_struct(column.type):
        raise ValueError("La disponibilidad no identifica las cuatro modalidades y macro")
    fields = {field.name for field in column.type}
    if fields != names and not (
        presence is None
        and fields == names - {"macro"}
        and "macro_available_at" in table.column_names
    ):
        raise ValueError("La disponibilidad no identifica las cuatro modalidades y macro")
    valid = ~column.is_null().to_numpy(zero_copy_only=False)
    bounds = np.zeros(len(column), dtype=np.int64)
    for name in sorted(names):
        if name == "macro" and macro_override is not None:
            available, observed = macro_override
            bounds = np.maximum(bounds, available)
            valid &= observed
            continue
        field = (
            column.field(name) if name in fields else table["macro_available_at"].combine_chunks()
        )
        known = ~field.is_null().to_numpy(zero_copy_only=False)
        if presence is not None:
            valid &= known == presence[:, MODALITIES.index(name)]
            if not pa.types.is_timestamp(field.type) or not field.type.tz:
                raise ValueError("La disponibilidad histórica necesita fechas con zona")
        else:
            valid &= known
        if field.null_count == len(field):
            continue
        if not pa.types.is_timestamp(field.type) or not field.type.tz:
            raise ValueError("La disponibilidad necesita marcas temporales con zona horaria")
        filled = field.fill_null(pa.scalar(0, type=field.type))
        bounds = np.maximum(bounds, _times(filled))
    return bounds, valid


def _presence(table, vectors, representation):
    """Contrastar bloques y conceptos antes de seleccionar filas de un grupo."""
    if "presence" not in table.column_names:
        raise ValueError("Faltan las máscaras de presencia de la edición histórica")
    column = table["presence"].combine_chunks()
    if (
        not (pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type))
        or not pa.types.is_boolean(column.type.value_type)
        or column.null_count
        or column.flatten().null_count
        or not (pa.compute.list_value_length(column).to_numpy() == len(MODALITIES)).all()
    ):
        raise ValueError("La presencia necesita cinco booleanos por muestra")
    presence = column.flatten().to_numpy(zero_copy_only=False).reshape(len(table), len(MODALITIES))
    if not presence[:, [0, 2]].all():
        raise ValueError("Los precios y gráficos causales son obligatorios")
    counts = table["news_count"]
    if not pa.types.is_integer(counts.type) or counts.null_count:
        raise ValueError("El recuento de noticias debe ser un entero conocido")
    events = counts.to_numpy()
    if (events < 0).any() or not np.array_equal(events > 0, presence[:, 1]):
        raise ValueError("La presencia de noticias no coincide con sus eventos admitidos")
    validate_historical_vectors(vectors, presence, representation)
    return presence


def _vectors(table, *, macro=None, historical=False):
    """Leer las formas y tipos del corpus antes de validar sus máscaras."""
    vectors = {}
    for name in VECTORS:
        if name == "macro" and macro is not None:
            vectors[name] = macro
            continue
        column = table[name].combine_chunks()
        if not (pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type)):
            raise ValueError("Cada modalidad necesita un vector explícito")
        if historical and column.type.value_type != pa.float32():
            raise ValueError("Los vectores históricos deben conservar el tipo float32")
        lengths = pa.compute.list_value_length(column).to_numpy()
        if (
            column.null_count
            or not len(lengths)
            or not 1 <= lengths[0] <= 2048
            or not (lengths == lengths[0]).all()
        ):
            raise ValueError("Las dimensiones de una modalidad no son válidas")
        vectors[name] = np.asarray(column.flatten().to_numpy(), dtype=np.float32).reshape(
            len(table), int(lengths[0])
        )
    return vectors


def _price_contexts(prices, ends, context):
    """Transformar hasta 256 ventanas, con las mismas operaciones y precisión por fila."""
    if (
        prices.ndim != 2
        or prices.shape[1] != 5
        or prices.dtype.kind not in "fiu"
        or ends.ndim != 1
        or ends.dtype.kind not in "iu"
        or not 1 <= len(ends) <= 256
        or type(context) is not int
        or not 2 <= context <= 512
        or (ends < context - 1).any()
        or (ends >= len(prices)).any()
    ):
        raise ValueError("Las ventanas del bloque no tienen índices o dimensiones válidos")
    windows = prices[ends.astype(np.intp, copy=False)[:, None] - np.arange(context - 1, -1, -1)]
    if (
        not np.isfinite(windows).all()
        or (windows[:, :, :4] <= 0).any()
        or (windows[:, :, 4] < 0).any()
    ):
        raise ValueError("Los valores OHLCV del bloque no son válidos")
    volume = windows[:, :, 4]
    mean = volume.mean(axis=1, keepdims=True)
    relative = np.zeros(volume.shape, dtype=np.result_type(volume.dtype, mean.dtype))
    np.divide(volume, mean, out=relative, where=mean > 0)
    np.log1p(relative, out=relative)
    result = np.empty(windows.shape, dtype=np.float32)
    result[:, :, :4] = np.log(windows[:, :, :4] / windows[:, 0:1, 3:4])
    result[:, :, 4] = relative
    return result


class CorpusDataset:
    """Validar una edición y reutilizar sus huellas mientras no cambien los archivos."""

    def __init__(
        self,
        manifest: Path,
        *,
        cache_bytes: int = 1024**3,
        cache_sample_tables: bool = False,
        input_policy: str = STRICT_INPUTS,
    ):
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= 4 * 1024**3:
            raise ValueError("La caché de entrada debe estar entre cero y cuatro GiB")
        if type(cache_sample_tables) is not bool:
            raise ValueError("La caché de tablas necesita una opción booleana explícita")
        self.cache_limit = cache_bytes
        self.cache_sample_tables = cache_sample_tables
        self.cache_entry_limit = 16384 if cache_sample_tables else 8192
        self.cached_bytes = 0
        self._cache = OrderedDict()
        self.path = Path(manifest)
        if self.path.is_symlink() or self.path.stat().st_size > 8 * 1024**2:
            raise ValueError("El manifiesto no es regular o supera 8 MiB")
        self.manifest, self.identity = read_manifest(self.path, 8 * 1024**2)
        meta = self.manifest
        self.masked = masked_inputs(input_policy)
        self.cohort = cohort_identity(meta, input_policy=input_policy)
        from .temporal_contract import temporal_contracts

        contracts = temporal_contracts(meta, input_policy=input_policy)
        self.temporals = {}
        if contracts:
            from .temporal_corpus import TemporalInputs

            self.temporals = {
                market: TemporalInputs(view, input_policy=input_policy)
                for market, view in contracts.items()
            }
            if self.masked and any(
                temporal.representation
                != representation_identity(meta["representation"], input_policy=input_policy)
                for temporal in self.temporals.values()
            ):
                raise ValueError("La vista no conserva la representación del corpus de origen")
        self.temporal = (
            next(iter(self.temporals.values()))
            if len(self.temporals) == 1
            else self.temporals or None
        )
        self.partitions = (
            ("train", "validation", "calibration", "evaluation")
            if self.temporal
            else ("train", "validation")
        )
        if (
            meta.get("schema_version") not in ({3} if self.masked else {1, 2})
            or meta.get("kind") != "corpus_supervision"
            or meta.get("scope") not in {"development_snapshot", "full_corpus"}
            or type(meta.get("cohort_complete")) is not bool
            or (meta["scope"] == "full_corpus" and not meta["cohort_complete"])
            or type(meta.get("context_sessions")) is not int
            or not 2 <= meta["context_sessions"] <= 512
            or (self.masked and meta["context_sessions"] != 64)
            or set(meta.get("roots", {})) != {"prepared", "samples", "labels"}
            or not isinstance(meta.get("assets"), list)
            or not meta["assets"]
        ):
            raise ValueError("El manifiesto supervisado no cumple su contrato")
        self.context = meta["context_sessions"]
        self.roots = {key: Path(value).resolve() for key, value in meta["roots"].items()}
        self.assets = meta["assets"]
        self.verified = {}
        self._artifact_paths = {}
        self._artifact_path_limit = 3 * len(self.assets)
        identities, counts = set(), dict.fromkeys(self.partitions, 0)
        for asset in self.assets:
            symbol, market = asset["symbol"], asset["market"]
            if (
                not isinstance(symbol, str)
                or symbol in {".", ".."}
                or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
                or market not in {"US", "CN"}
                or (self.temporals and market not in self.temporals)
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
        key = (root, asset["market"], asset["symbol"], kind)
        path = self._artifact_paths.get(key)
        if path is None:
            path = root.joinpath(asset["market"], asset["symbol"], f"{kind}.parquet")
            if len(self._artifact_paths) < self._artifact_path_limit:
                self._artifact_paths[key] = path
        # Solo se reutiliza la ruta. El archivo se comprueba en cada acceso.
        try:
            valid = (
                path.is_file()
                and not path.is_symlink()
                and os.path.commonpath((root, os.path.realpath(path))) == str(root)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Falta un artefacto regular dentro del origen declarado")
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.verified.get(path) != signature:
            if sha256(path) != asset[kind + "_sha256"]:
                raise ValueError("Un artefacto supervisado ha cambiado desde su confirmación")
            with path.open("rb") as stream:
                stream.seek(-8, 2)
                footer = stream.read(8)
            if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 8 * 1024**2:
                raise ValueError(
                    "La cabecera final de Parquet no es válida o excede su presupuesto"
                )
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
        return path

    def _cached(self, key, signature):
        if key not in self._cache:
            return None
        previous, value, size = self._cache[key]
        if previous != signature:
            self._cache.pop(key)
            self.cached_bytes -= size
            return None
        self._cache.move_to_end(key)
        return value

    def _remember(self, key, signature, value):
        size = (
            value.get_total_buffer_size()
            if isinstance(value, pa.Table)
            else sum(array.nbytes for array in value)
        )
        if self.cache_limit == 0 or size > self.cache_limit:
            return
        if key in self._cache:
            _, _, previous_size = self._cache.pop(key)
            self.cached_bytes -= previous_size
        while self._cache and (
            self.cached_bytes + size > self.cache_limit
            or len(self._cache) >= self.cache_entry_limit
        ):
            _, (_, _, previous_size) = self._cache.popitem(last=False)
            self.cached_bytes -= previous_size
        if not isinstance(value, pa.Table):
            for array in value:
                array.flags.writeable = False
        self._cache[key] = signature, value, size
        self.cached_bytes += size

    @staticmethod
    def _partition_labels(arrays, partition):
        positions, prediction, values, maturity, partitions = arrays
        selected = np.flatnonzero(partitions == partition)
        selected = selected[np.argsort(positions[selected])]
        return positions[selected], prediction[selected], values[selected], maturity[selected]

    def _labels(self, asset, partition, sample_rows):
        path = self._file(asset, "labels")
        signature = self.verified[path], sample_rows, tuple(sorted(asset["counts"].items()))
        key = "labels", path
        cached = self._cached(key, signature)
        if cached is not None:
            return self._partition_labels(cached, partition)
        table = read_bounded_table(path, max_rows=1_000_000, max_bytes=MAX_TABLE_BYTES)
        validate_cohort_rows(table, self.cohort)
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
        if self.temporal:
            from .temporal_corpus import VIEW_REASONS

            allowed |= VIEW_REASONS
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
            arrays = (
                positions,
                prediction,
                np.empty(0, dtype=np.float64),
                prediction.copy(),
                np.empty(0, dtype="U11"),
            )
            self._remember(key, signature, arrays)
            return self._partition_labels(arrays, partition)
        maturity = _times(table["target_available_at"])
        if np.any(maturity <= prediction):
            raise ValueError("La etiqueta debe madurar después de su predicción")
        values = table["target"].to_numpy()
        if table["target"].null_count or not np.isfinite(values).all():
            raise ValueError("Hay etiquetas ausentes o no finitas")
        partitions = np.asarray(table["partition"].to_pylist())
        years = prediction.astype("datetime64[us]").astype("datetime64[Y]").astype(int) + 1970
        mature_years = maturity.astype("datetime64[us]").astype("datetime64[Y]").astype(int) + 1970
        if self.temporal:
            temporal = self.temporals[asset["market"]]
            _, available, eligible = temporal.lookup(prediction)
            assigned = temporal.assign(prediction, available, maturity, eligible)
            valid = (assigned["partition"] == partitions) & (assigned["reason"] == "accepted")
        else:
            valid = ((partitions == "train") & (years <= 2022) & (mature_years <= 2022)) | (
                (partitions == "validation") & (years == 2023) & (mature_years == 2023)
            )
        if not valid.all() or any(
            int(np.sum(partitions == p)) != asset["counts"][p] for p in self.partitions
        ):
            raise ValueError("Una etiqueta cruza la partición o sus recuentos no concilian")
        arrays = positions, prediction, values, maturity, partitions
        self._remember(key, signature, arrays)
        return self._partition_labels(arrays, partition)

    def _prices(self, asset):
        path = self._file(asset, "prices")
        signature = self.verified[path]
        key = "prices", path
        cached = self._cached(key, signature)
        if cached is not None:
            return cached
        table = read_bounded_table(path, max_rows=200_000, max_bytes=MAX_TABLE_BYTES)
        prices = np.column_stack(
            [table[c].to_numpy() for c in ("open", "high", "low", "close", "volume")]
        )
        available = _times(table["available_at"])
        if np.any(np.diff(available) <= 0):
            raise ValueError("Los precios necesitan un orden temporal único")
        self._remember(key, signature, (prices, available))
        return prices, available

    def _blocks(self, partition, epoch, seed, cursor):
        order = _random(seed, epoch, "assets").permutation(len(self.assets))
        consumed = sum(self.assets[int(i)]["counts"][partition] for i in order[: cursor["asset"]])
        dimensions = None
        for asset_position in range(cursor["asset"], len(order)):
            asset = self.assets[int(order[asset_position])]
            temporal = self.temporals.get(asset["market"])
            key = f"{asset['market']}/{asset['symbol']}"
            path = self._file(asset, "samples")
            with pq.ParquetFile(path) as file:
                positions, prediction, target, maturity = self._labels(
                    asset, partition, file.metadata.num_rows
                )
                prices, available = self._prices(asset)
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
                    columns = ["prediction_at", "price_end_index", *VECTORS] + (
                        ["cohort_id"] if self.cohort else []
                    )
                    if self.masked:
                        if not {"presence", "news_count"} <= set(file.schema_arrow.names):
                            raise ValueError("Faltan las máscaras o los recuentos históricos")
                        columns.extend(("presence", "news_count"))
                    if self.temporal and not self.masked:
                        columns.remove("macro")
                    if "input_availability" in file.schema_arrow.names:
                        columns.append("input_availability")
                        if "macro_available_at" in file.schema_arrow.names:
                            columns.append("macro_available_at")
                    metadata = file.metadata.row_group(int(group))
                    size = sum(
                        metadata.column(c).total_uncompressed_size
                        for c in range(metadata.num_columns)
                        if metadata.column(c).path_in_schema.split(".")[0] in columns
                    )
                    if size > MAX_TABLE_BYTES:
                        raise ValueError("El grupo de características supera 64 MiB")
                    cache_key = "samples", path, int(group), tuple(columns)
                    signature = self.verified[path]
                    table = self._cached(cache_key, signature) if self.cache_sample_tables else None
                    cache_miss = table is None
                    if cache_miss:
                        if self.masked:
                            _historical_times(
                                file.read_row_group(
                                    int(group), columns=["prediction_at"], use_threads=False
                                )
                            )
                        table = file.read_row_group(int(group), columns=columns, use_threads=False)
                    validate_cohort_rows(table, self.cohort)
                    if table.nbytes > MAX_TABLE_BYTES:
                        raise ValueError("El grupo decodificado supera el presupuesto")
                    timestamps = (
                        _historical_times(table) if self.masked else _times(table["prediction_at"])
                    )
                    macro = temporal.lookup(timestamps) if temporal and not self.masked else None
                    ends = table["price_end_index"].to_numpy()
                    vectors = _vectors(
                        table, macro=macro[0] if macro is not None else None, historical=self.masked
                    )
                    shape = {name: values.shape[1] for name, values in vectors.items()}
                    if dimensions is not None and dimensions != shape:
                        raise ValueError("Las dimensiones cambian entre activos")
                    dimensions = shape
                    presence = (
                        _presence(table, vectors, self.manifest["representation"])
                        if self.masked
                        else None
                    )
                    availability, availability_valid = _availability(
                        table, macro_override=macro[1:] if macro else None, presence=presence
                    )
                    if self.cache_sample_tables and cache_miss:
                        self._remember(cache_key, signature, table)
                    for offset in range(start, len(indexes), 256):
                        labels = indexes[offset : offset + 256]
                        block_rows = positions[labels] - offsets[group]
                        if (block_rows < 0).any() or (block_rows >= len(table)).any():
                            raise ValueError("La etiqueta queda fuera de su grupo de muestras")
                        contexts = _price_contexts(prices, ends[block_rows], self.context)
                        yield dict(
                            vectors=vectors,
                            rows=block_rows,
                            prices=contexts,
                            target=target[labels],
                            key=key,
                            prediction_at=prediction[labels],
                            target_available_at=maturity[labels],
                            sample_at=timestamps[block_rows],
                            price_available_at=available[ends[block_rows]],
                            input_available_at=(
                                availability[block_rows] if availability is not None else None
                            ),
                            availability_valid=(
                                availability_valid[block_rows] if availability is not None else None
                            ),
                            presence=presence[block_rows] if presence is not None else None,
                            cursor={
                                "asset": asset_position,
                                "group": group_position,
                                "offset": offset,
                                "consumed": int(consumed),
                            },
                        )
                        consumed += len(labels)
            for kind in ("prices", "samples", "labels"):
                self._file(asset, kind)
        if consumed != self.manifest["counts"][partition]:
            raise ValueError("El recorrido no visita exactamente la población declarada")

    def batches(self, *, partition, batch_size, epoch, seed, cursor=None):
        if (
            partition not in self.partitions
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
        for temporal in self.temporals.values():
            temporal.verify()
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
        batch, filled, consumed = None, 0, point["consumed"]
        total = self.manifest["counts"][partition]
        for block in self._blocks(partition, epoch, seed, point):
            offset = 0
            while offset < len(block["rows"]):
                if batch is None:
                    batch = _new_batch(
                        block["vectors"],
                        self.context,
                        min(batch_size, total - consumed),
                        masked=self.masked,
                    )
                    if self.cohort:
                        batch["cohort_id"] = self.cohort
                count = min(len(batch["target"]) - filled, len(block["rows"]) - offset)
                if count <= 0:
                    raise ValueError("El recorrido excede la población declarada")
                _fill_batch(batch, filled, block, offset, offset + count)
                offset += count
                filled += count
                consumed += count
                batch["confirmed_cursor"] = {
                    **identity,
                    **block["cursor"],
                    "offset": block["cursor"]["offset"] + offset,
                    "consumed": consumed,
                }
                if filled == batch_size:
                    yield batch
                    batch, filled = None, 0
            # El grupo consumido puede liberarse antes de que el lector abra el siguiente.
            block = None
        # La última tanda parcial se publica después de comprobar los archivos y el recuento.
        if batch is not None:
            if filled != len(batch["target"]):
                raise ValueError("El recorrido no completa la población declarada")
            yield batch


def _new_batch(vectors, context, size, *, masked=False):
    return {
        **({"presence": np.empty((size, len(MODALITIES)), dtype=np.bool_)} if masked else {}),
        "inputs": {
            **{
                name: np.empty((size, values.shape[1]), dtype=np.float32)
                for name, values in vectors.items()
            },
            "prices": np.empty((size, context, 5), dtype=np.float32),
        },
        "target": np.empty(size, dtype=np.float64),
        "sample_ids": [],
        "market": [],
        "prediction_at": np.empty(size, dtype="datetime64[us]"),
        "target_available_at": np.empty(size, dtype="datetime64[us]"),
        "input_available_at": np.full(size, np.datetime64("NaT", "us")),
        "weight": np.ones(size, dtype=np.float64),
    }


def _fill_batch(batch, filled, block, start, stop):
    source, destination = slice(start, stop), slice(filled, filled + stop - start)
    at = block["prediction_at"][source]
    if (block["sample_at"][source] != at).any() or (block["price_available_at"][source] > at).any():
        raise ValueError("Las modalidades y la etiqueta no coinciden temporalmente")
    available = block["input_available_at"]
    if available is not None and (
        not block["availability_valid"][source].all() or (available[source] > at).any()
    ):
        raise ValueError("La disponibilidad de una modalidad es ausente o futura")
    for name, values in block["vectors"].items():
        selected = batch["inputs"][name][destination]
        # Los índices se han validado antes. clip evita el buffer adicional del modo raise.
        np.take(values, block["rows"][source], axis=0, out=selected, mode="clip")
        if not np.isfinite(selected).all():
            raise ValueError("Una modalidad contiene valores no finitos")
    batch["inputs"]["prices"][destination] = block["prices"][source]
    for name in ("target", "prediction_at", "target_available_at"):
        batch[name][destination] = block[name][source]
    if available is not None:
        batch["input_available_at"][destination] = available[source]
    if block.get("presence") is not None:
        batch["presence"][destination] = block["presence"][source]
    batch["sample_ids"].extend(f"{block['key']}/{moment}" for moment in at)
    batch["market"].extend([block["key"].split("/", 1)[0]] * (stop - start))


def supervised_batches(
    manifest: Path,
    *,
    partition: str,
    batch_size: int,
    epoch: int,
    seed: int,
    cursor: dict | None = None,
    input_policy: str = STRICT_INPUTS,
):
    """Crear un lector para una pasada. La campaña reutiliza CorpusDataset entre épocas."""
    yield from CorpusDataset(manifest, input_policy=input_policy).batches(
        partition=partition, batch_size=batch_size, epoch=epoch, seed=seed, cursor=cursor
    )


def prepare_corpus_targets(
    manifest: Path, prepared: Path, output: Path, *, input_policy: str = STRICT_INPUTS
) -> dict:
    """Preparar las etiquetas mediante el mismo contrato que consume este lector."""
    from .corpus_targets import prepare_corpus_targets as prepare

    return prepare(manifest, prepared, output, input_policy=input_policy)
