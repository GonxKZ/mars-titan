"""Lotes supervisados por activo y grupo Parquet, sin acumular el corpus en RAM."""

import fcntl
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
from mars_titan.data.modality_ablation import ablate_samples, ablated_modalities
from mars_titan.data.storage import atomic_json, sha256

from .cohort_contract import cohort_identity, representation_identity, validate_cohort_rows
from .input_pipeline import PipelineOptions, background, ordered_map

VECTORS = ("news", "charts", "fundamentals", "macro")
MAX_TABLE_BYTES = 64 * 1024**2
# Hilos que calculan las huellas de los artefactos al abrir una edición.
HASH_WORKERS = 4
# Huellas ya calculadas en este proceso por ruta y firma de stat. La firma incluye ctime, que
# cambia con cualquier escritura, así que un archivo modificado se vuelve a leer completo.
_DIGESTS = {}
_DIGEST_LIMIT = 1 << 18
_DIGEST_LOCK = threading.Lock()


# Archivo de huellas compartido entre procesos, por ejemplo los de las ranuras de una campaña.
DIGEST_CACHE_ENV = "MARS_TITAN_DIGEST_CACHE"
_DIGEST_KIND = "mars_titan_file_digests"
_DIGEST_FILE_BYTES = 64 * 1024**2


def _signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _digest_file():
    raw = os.environ.get(DIGEST_CACHE_ENV)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{DIGEST_CACHE_ENV} debe ser una ruta absoluta y regular")
    return path


def _read_digests(path):
    """Entradas válidas del archivo compartido. Un archivo ilegible no aporta ninguna."""
    try:
        if not path.is_file() or path.stat().st_size > _DIGEST_FILE_BYTES:
            return {}
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict) or document.get("kind") != _DIGEST_KIND:
        return {}
    entries = {}
    for entry in document.get("entries", ()):
        if (
            isinstance(entry, list)
            and len(entry) == 7
            and isinstance(entry[0], str)
            and all(type(value) is int for value in entry[1:6])
            and isinstance(entry[6], str)
            and re.fullmatch(r"[0-9a-f]{64}", entry[6])
        ):
            entries[entry[0], tuple(entry[1:6])] = entry[6]
    return entries


def load_shared_digests():
    """Incorporar las huellas que otros procesos guardaron con la misma firma de stat.

    Una entrada solo vale para la ruta y la firma exactas con que se calculó. La firma
    incluye el ctime, que cambia con cualquier escritura y que un usuario no puede fijar,
    así que un archivo modificado se vuelve a leer completo, igual que con la memoria del
    proceso.
    """
    path = _digest_file()
    if path is None:
        return
    entries = _read_digests(path)
    with _DIGEST_LOCK:
        for key, value in entries.items():
            _DIGESTS.setdefault(key, value)


def store_shared_digests(keys):
    """Añadir al archivo compartido las huellas de `keys`, con cerrojo y escritura atómica."""
    path = _digest_file()
    if path is None or not keys:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = _read_digests(path)
        with _DIGEST_LOCK:
            entries.update({key: _DIGESTS[key] for key in keys if key in _DIGESTS})
        rows = [[name, *signature, value] for (name, signature), value in sorted(entries.items())]
        atomic_json(path, dict(kind=_DIGEST_KIND, entries=rows[-_DIGEST_LIMIT:]))


def _digest(path, signature):
    """Huella del archivo con esa firma, calculada como mucho una vez por proceso."""
    key = str(path), signature
    with _DIGEST_LOCK:
        known = _DIGESTS.get(key)
    if known is not None:
        return known
    value = sha256(path)
    with _DIGEST_LOCK:
        if len(_DIGESTS) >= _DIGEST_LIMIT:
            _DIGESTS.clear()
        _DIGESTS[key] = value
    return value


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


def _populated_groups(file):
    """Índices físicos de los grupos con filas.

    Un grupo vacío no contiene muestras ni desplaza posiciones, así que se omite sin
    renumerar los demás. Los vectores de un grupo con filas siguen validándose completos.
    """
    return [g for g in range(file.num_row_groups) if file.metadata.row_group(g).num_rows]


def _group_bytes(file, group, columns):
    """Bytes sin comprimir de las columnas leídas de un grupo, según sus metadatos."""
    metadata = file.metadata.row_group(int(group))
    return sum(
        metadata.column(c).total_uncompressed_size
        for c in range(metadata.num_columns)
        if metadata.column(c).path_in_schema.split(".")[0] in columns
    )


def _row_range(decoded, start, stop):
    """Filas `start:stop` de unas muestras decodificadas, como vistas sin copia."""
    timestamps, ends, vectors, presence, availability, availability_valid = decoded
    return (
        timestamps[start:stop],
        ends[start:stop],
        {name: values[start:stop] for name, values in vectors.items()},
        presence[start:stop],
        availability[start:stop],
        availability_valid[start:stop],
    )


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
    return _window_features(windows)


def _window_contexts(prices, ends, context):
    """Ventanas de filas de activos distintos, con las comprobaciones de `_price_contexts`.

    `prices[i]` es la tabla OHLCV del activo de la fila `i`. Las ventanas se copian en un
    bloque contiguo y se transforman juntas, así que cada fila coincide con la de
    `_price_contexts` sobre su propio activo. Con tipos distintos se transforma fila a fila.
    """
    ends = np.asarray(ends)
    if (
        len(prices) != len(ends)
        or any(
            table.ndim != 2 or table.shape[1] != 5 or table.dtype.kind not in "fiu"
            for table in prices
        )
        or ends.ndim != 1
        or ends.dtype.kind not in "iu"
        or not 1 <= len(ends) <= 256
        or type(context) is not int
        or not 2 <= context <= 512
        or (ends < context - 1).any()
        or any(end >= len(table) for table, end in zip(prices, ends, strict=True))
    ):
        raise ValueError("Las ventanas del bloque no tienen índices o dimensiones válidos")
    if len({table.dtype for table in prices}) != 1:
        return np.concatenate(
            [_price_contexts(table, ends[i : i + 1], context) for i, table in enumerate(prices)]
        )
    windows = np.empty((len(ends), context, 5), dtype=prices[0].dtype)
    for row, (table, end) in enumerate(zip(prices, ends.astype(np.intp), strict=True)):
        windows[row] = table[end - context + 1 : end + 1]
    return _window_features(windows)


def _window_features(windows):
    """Transformar ventanas `[filas, contexto, 5]` con las mismas operaciones por fila."""
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
    """Validar una edición y reutilizar sus huellas mientras no cambien los archivos.

    `modality_ablation` nombra una variante de `data.modality_ablation`. Solo existe con la
    edición con máscaras: las modalidades de la variante se leen como una ausencia real en
    todas las filas. Sin ella, la lectura no cambia.

    `pipeline` declara los hilos de decodificación y los lotes adelantados. Sin ella se
    leen de `PipelineOptions.from_environment`, que por defecto es la ruta secuencial. La
    tubería entrega los mismos lotes, cursores y errores en el mismo orden.
    """

    def __init__(
        self,
        manifest: Path,
        *,
        cache_bytes: int = 1024**3,
        cache_sample_tables: bool = False,
        input_policy: str = STRICT_INPUTS,
        modality_ablation: str | None = None,
        pipeline: PipelineOptions | None = None,
    ):
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= 4 * 1024**3:
            raise ValueError("La caché de entrada debe estar entre cero y cuatro GiB")
        if type(cache_sample_tables) is not bool:
            raise ValueError("La caché de tablas necesita una opción booleana explícita")
        pipeline = PipelineOptions.from_environment() if pipeline is None else pipeline
        if type(pipeline) is not PipelineOptions:
            raise ValueError("La tubería de lectura se declara con PipelineOptions")
        self.pipeline = pipeline
        self._decoder = None
        self._lock = threading.RLock()
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
        if modality_ablation is not None:
            if not self.masked:
                raise ValueError("La ablación de modalidades necesita la edición con máscaras")
            ablated_modalities(modality_ablation)
        self.modality_ablation = modality_ablation
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
        from .target_factors import revision_sources

        self._target_factor_sources = revision_sources(meta, input_policy=input_policy)
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
        self._verify_files()
        if counts != meta.get("counts"):
            raise ValueError("Los recuentos del corpus no concilian con los activos")

    def _verify_files(self):
        """Comprobar todos los artefactos, con las huellas pendientes calculadas en paralelo.

        Las huellas se calculan antes en hilos y la comprobación recorre después los activos
        en su orden, así que el primer artefacto inválido es el mismo que en serie.
        """
        load_shared_digests()
        pending = []
        for asset in self.assets:
            for kind in ("prices", "samples", "labels"):
                path = self._path(asset, kind)
                try:
                    if path.is_file() and not path.is_symlink():
                        pending.append((path, _signature(path.stat())))
                except OSError:
                    continue
        with _DIGEST_LOCK:
            missing = [item for item in pending if (str(item[0]), item[1]) not in _DIGESTS]
        if missing:
            with ThreadPoolExecutor(min(HASH_WORKERS, len(missing))) as pool:
                list(pool.map(lambda item: _digest(*item), missing))
        store_shared_digests([(str(path), signature) for path, signature in missing])
        for asset in self.assets:
            for kind in ("prices", "samples", "labels"):
                self._file(asset, kind)

    def _executor(self):
        """Hilos de decodificación de esta edición, creados al primer uso."""
        with self._lock:
            if self._decoder is None:
                self._decoder = ThreadPoolExecutor(
                    self.pipeline.decode_workers, thread_name_prefix="mars-titan-decode"
                )
            return self._decoder

    def _path(self, asset, kind):
        root = self.roots["prepared" if kind == "prices" else kind]
        key = (root, asset["market"], asset["symbol"], kind)
        path = self._artifact_paths.get(key)
        if path is None:
            path = root.joinpath(asset["market"], asset["symbol"], f"{kind}.parquet")
            if len(self._artifact_paths) < self._artifact_path_limit:
                self._artifact_paths[key] = path
        return path

    def _file(self, asset, kind):
        root = self.roots["prepared" if kind == "prices" else kind]
        path = self._path(asset, kind)
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
        signature = _signature(path.stat())
        if self.verified.get(path) != signature:
            if _digest(path, signature) != asset[kind + "_sha256"]:
                raise ValueError("Un artefacto supervisado ha cambiado desde su confirmación")
            with path.open("rb") as stream:
                stream.seek(-8, 2)
                footer = stream.read(8)
            if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 8 * 1024**2:
                raise ValueError(
                    "La cabecera final de Parquet no es válida o excede su presupuesto"
                )
            if signature != _signature(path.stat()):
                raise ValueError("Un artefacto ha cambiado durante la comprobación")
            self.verified[path] = signature
        return path

    def _cached(self, key, signature):
        with self._lock:
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
        if not isinstance(value, pa.Table):
            for array in value:
                array.flags.writeable = False
        with self._lock:
            if key in self._cache:
                _, _, previous_size = self._cache.pop(key)
                self.cached_bytes -= previous_size
            while self._cache and (
                self.cached_bytes + size > self.cache_limit
                or len(self._cache) >= self.cache_entry_limit
            ):
                _, (_, _, previous_size) = self._cache.popitem(last=False)
                self.cached_bytes -= previous_size
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

    def _sample_columns(self, file):
        """Columnas de muestras que lee la edición, comprobadas contra el esquema."""
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
        return columns

    def _sample_group(self, asset, file, group):
        """Decodificar las modalidades una vez con las mismas reglas en ambos recorridos."""
        path = self._file(asset, "samples")
        columns = self._sample_columns(file)
        if _group_bytes(file, group, columns) > MAX_TABLE_BYTES:
            raise ValueError("El grupo de características supera 64 MiB")
        cache_key = "samples", path, int(group), tuple(columns)
        signature = self.verified[path]
        table = self._cached(cache_key, signature) if self.cache_sample_tables else None
        cache_miss = table is None
        if cache_miss:
            if self.masked:
                _historical_times(
                    file.read_row_group(int(group), columns=["prediction_at"], use_threads=False)
                )
            table = file.read_row_group(int(group), columns=columns, use_threads=False)
        validate_cohort_rows(table, self.cohort)
        if table.nbytes > MAX_TABLE_BYTES:
            raise ValueError("El grupo decodificado supera el presupuesto")
        decoded = self._converted(table, asset)
        if self.cache_sample_tables and cache_miss:
            self._remember(cache_key, signature, table)
        if self.modality_ablation is not None:
            table, decoded = self._ablated(table, decoded)
        return table, *decoded

    def _converted(self, table, asset):
        """Fechas, ventanas, vectores, presencia y disponibilidad, comprobados fila a fila."""
        temporal = self.temporals.get(asset["market"])
        timestamps = _historical_times(table) if self.masked else _times(table["prediction_at"])
        macro = temporal.lookup(timestamps) if temporal and not self.masked else None
        ends = table["price_end_index"].to_numpy()
        vectors = _vectors(
            table, macro=macro[0] if macro is not None else None, historical=self.masked
        )
        presence = (
            _presence(table, vectors, self.manifest["representation"]) if self.masked else None
        )
        availability, availability_valid = _availability(
            table, macro_override=macro[1:] if macro else None, presence=presence
        )
        return timestamps, ends, vectors, presence, availability, availability_valid

    def _ablated(self, table, decoded):
        # La lectura original ya pasó sus comprobaciones. La tabla ablacionada vuelve a
        # pasarlas igual que una muestra con la modalidad ausente.
        timestamps, ends = decoded[:2]
        table = ablate_samples(table, self.modality_ablation)
        vectors = _vectors(table, historical=True)
        presence = _presence(table, vectors, self.manifest["representation"])
        availability, availability_valid = _availability(table, presence=presence)
        return table, (timestamps, ends, vectors, presence, availability, availability_valid)

    def _asset_samples(self, works):
        """Decodificar juntos los grupos de un activo, o `None` si hay que leerlos de uno en uno.

        Una sola lectura de Arrow, con sus hilos si la tubería los tiene, sustituye a una
        lectura por grupo. Las comprobaciones de `_sample_group` son por fila, así que la
        tabla del activo las supera si y solo si las supera cada grupo, y cada grupo es un
        tramo con los mismos valores que su lectura aislada. Con cualquier fallo, con la
        caché de tablas o fuera de la edición con máscaras se devuelve `None` y el activo se
        lee grupo a grupo, que lanza el mismo error en la misma posición.
        """
        if not works or not self._reads_assets():
            return None
        first = works[0]
        groups = sorted(work["group"] for work in works)
        try:
            self._file(first["asset"], "samples")
            with pq.ParquetFile(first["path"], metadata=first["metadata"], pre_buffer=True) as file:
                columns = self._sample_columns(file)
                if any(_group_bytes(file, group, columns) > MAX_TABLE_BYTES for group in groups):
                    return None
                table = file.read_row_groups(
                    groups, columns=columns, use_threads=bool(self.pipeline.decode_workers)
                )
                sizes = [file.metadata.row_group(group).num_rows for group in groups]
            # Cada grupo ocupa como mucho lo que ocupa la tabla de su activo.
            if table.nbytes > MAX_TABLE_BYTES:
                return None
            validate_cohort_rows(table, self.cohort)
            decoded = self._converted(table, first["asset"])
            if self.modality_ablation is not None:
                _, decoded = self._ablated(table, decoded)
        except Exception:  # noqa: BLE001 - la lectura por grupo reproduce el error
            return None
        starts = dict(zip(groups, np.cumsum([0, *sizes[:-1]]).tolist(), strict=True))
        sizes = dict(zip(groups, sizes, strict=True))
        result = []
        for work in works:
            start, size = starts[work["group"]], sizes[work["group"]]
            result.append((size, _row_range(decoded, start, start + size)))
        return result

    def _group_plan(self, partition, epoch, seed, cursor):
        """Recorrer activos y grupos en el orden del lector sin decodificar vectores.

        Entrega `("group", trabajo)` por cada grupo con filas pendientes, con sus bloques de
        hasta 256 etiquetas y el consumo confirmado antes de cada uno, y `("asset", activo)`
        al terminar cada activo. Los cursores y los errores del recorrido son los de la ruta
        secuencial porque se calculan aquí, en el mismo orden.
        """
        order = _random(seed, epoch, "assets").permutation(len(self.assets))
        consumed = sum(self.assets[int(i)]["counts"][partition] for i in order[: cursor["asset"]])
        for asset_position in range(cursor["asset"], len(order)):
            asset = self.assets[int(order[asset_position])]
            key = f"{asset['market']}/{asset['symbol']}"
            path = self._file(asset, "samples")
            with pq.ParquetFile(path) as file:
                metadata = file.metadata
                positions, prediction, target, maturity = self._labels(
                    asset, partition, metadata.num_rows
                )
                prices, available = self._prices(asset)
                # El orden permuta solo los grupos con filas y usa su rango entre ellos.
                # Sin grupos vacíos, rango e índice coinciden y el orden no cambia. Con ellos,
                # el recorrido y el cursor son los del mismo archivo sin esos grupos.
                populated = _populated_groups(file)
                groups = _random(seed, epoch, key + "/groups").permutation(len(populated))
                offsets = np.cumsum(
                    [0] + [metadata.row_group(g).num_rows for g in range(file.num_row_groups)]
                )
                start_group = cursor["group"] if asset_position == cursor["asset"] else 0
                if not 0 <= start_group < max(1, len(groups)):
                    raise ValueError("El cursor señala un grupo inexistente")
                for group_position, rank in enumerate(groups):
                    group = populated[rank]
                    first, end = np.searchsorted(positions, [offsets[group], offsets[group + 1]])
                    if group_position < start_group:
                        consumed += end - first
                        continue
                    indexes = np.arange(first, end)
                    if partition == "train":
                        indexes = _random(seed, epoch, key + f"/{rank}").permutation(indexes)
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
                    blocks = []
                    for offset in range(start, len(indexes), 256):
                        labels = indexes[offset : offset + 256]
                        blocks.append((offset, labels, int(consumed)))
                        consumed += len(labels)
                    yield (
                        "group",
                        dict(
                            asset=asset,
                            key=key,
                            path=path,
                            metadata=metadata,
                            group=group,
                            first_row=offsets[group],
                            cursor=(asset_position, group_position),
                            blocks=blocks,
                            labels=(positions, prediction, target, maturity),
                            prices=(prices, available),
                        ),
                    )
            yield "asset", asset
        if consumed != self.manifest["counts"][partition]:
            raise ValueError("El recorrido no visita exactamente la población declarada")

    def _group_blocks(self, work):
        """Decodificar un grupo del plan por separado y formar sus bloques."""
        with pq.ParquetFile(work["path"], metadata=work["metadata"]) as file:
            table, *decoded = self._sample_group(work["asset"], file, work["group"])
        return self._group_result(work, len(table), decoded)

    def _group_result(self, work, size, decoded):
        """Bloques de un grupo con `size` filas decodificadas.

        Si un bloque no es válido, se devuelven los anteriores y el error, que el lector
        lanza después de entregarlos, como en la ruta secuencial.
        """
        timestamps, ends, vectors, presence, availability, availability_valid = decoded
        shape = {name: values.shape[1] for name, values in vectors.items()}
        positions, prediction, target, maturity = work["labels"]
        prices, available = work["prices"]
        asset_position, group_position = work["cursor"]
        blocks, error = [], None
        try:
            for offset, labels, consumed in work["blocks"]:
                block_rows = positions[labels] - work["first_row"]
                if (block_rows < 0).any() or (block_rows >= size).any():
                    raise ValueError("La etiqueta queda fuera de su grupo de muestras")
                contexts = _price_contexts(prices, ends[block_rows], self.context)
                blocks.append(
                    dict(
                        vectors=vectors,
                        rows=block_rows,
                        prices=contexts,
                        target=target[labels],
                        key=work["key"],
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
                            "consumed": consumed,
                        },
                    )
                )
        except Exception as failure:  # noqa: BLE001 - se lanza tras los bloques anteriores
            error = failure
        return shape, blocks, error

    def _asset_blocks(self, item):
        """Bloques de los grupos de un activo en el orden del plan. Puede ir en un hilo.

        Devuelve los resultados por grupo, el error que interrumpió el activo y el activo si
        el plan lo completó. Sin lectura conjunta, cada grupo se lee aparte y se detiene en
        el primer error, que el lector lanza después de los bloques anteriores.
        """
        works, asset = item
        samples = self._asset_samples(works)
        results = []
        if samples is not None:
            for work, (size, decoded) in zip(works, samples, strict=True):
                results.append(self._group_result(work, size, decoded))
                if results[-1][2] is not None:
                    break
            return results, None, asset
        for work in works:
            try:
                results.append(self._group_blocks(work))
            except Exception as error:  # noqa: BLE001 - se lanza tras los grupos anteriores
                return results, error, asset
            if results[-1][2] is not None:
                break
        return results, None, asset

    def _reads_assets(self):
        """La lectura conjunta por activo solo se usa en la edición con máscaras sin caché."""
        return self.masked and not self.cache_sample_tables

    @staticmethod
    def _group_items(plan):
        """El plan grupo a grupo, como activos de un grupo, para la lectura por grupo."""
        for kind, value in plan:
            yield ([value], None) if kind == "group" else ([], value)

    @staticmethod
    def _asset_plan(plan):
        """Agrupar los trabajos del plan por activo.

        Un error del plan se lanza después de entregar los grupos que lo preceden, que
        forman un activo incompleto sin comprobación final de sus archivos.
        """
        works = []
        try:
            for kind, value in plan:
                if kind == "group":
                    works.append(value)
                    continue
                yield works, value
                works = []
        except Exception:
            if works:
                yield works, None
            raise

    def _blocks(self, partition, epoch, seed, cursor):
        plan = self._group_plan(partition, epoch, seed, cursor)
        # Por activo, cada unidad retiene un activo decodificado. Por grupo, un grupo.
        if self._reads_assets():
            plan, lookahead = self._asset_plan(plan), self.pipeline.decode_workers + 1
        else:
            plan, lookahead = self._group_items(plan), self.pipeline.lookahead
        if self.pipeline.decode_workers:
            stream = ordered_map(self._asset_blocks, plan, self._executor(), lookahead)
        else:
            stream = map(self._asset_blocks, plan)
        dimensions = None
        for results, error, asset in stream:
            for shape, blocks, failure in results:
                if dimensions is not None and dimensions != shape:
                    raise ValueError("Las dimensiones cambian entre activos")
                dimensions = shape
                yield from blocks
                if failure is not None:
                    raise failure
            if error is not None:
                raise error
            if asset is not None:
                for name in ("prices", "samples", "labels"):
                    self._file(asset, name)

    def observation_batches(self, *, start, end, batch_size=256):
        """Leer todas las filas históricas del intervalo, por activo y sin labels.

        La ordenación temporal entre activos corresponde al índice de cohortes.
        La ausencia de objetivo no elimina una observación del recorrido.
        """
        if (
            not self.masked
            or type(start) is not int
            or type(end) is not int
            or not 946_684_800_000_000 <= start < end <= 1_704_067_200_000_000
            or type(batch_size) is not int
            or not 1 <= batch_size <= 256
        ):
            raise ValueError(
                "La lectura de observaciones necesita adhesión histórica y límites explícitos"
            )
        if sha256(self.path) != self.identity:
            raise ValueError("El manifiesto cambió desde su confirmación")
        dimensions = None
        for asset in sorted(self.assets, key=lambda a: (a["market"], a["symbol"])):
            path = self._file(asset, "samples")
            prices, available = self._prices(asset)
            key, previous = f"{asset['market']}/{asset['symbol']}", None
            with pq.ParquetFile(path) as file:
                if file.metadata.num_rows > 1_000_000:
                    raise ValueError("El activo supera el presupuesto de muestras")
                for group in _populated_groups(file):
                    table, moments, ends, vectors, presence, availability, valid = (
                        self._sample_group(asset, file, group)
                    )
                    if len(moments) and (
                        np.any(np.diff(moments) <= 0)
                        or (previous is not None and moments[0] <= previous)
                    ):
                        raise ValueError("Las observaciones necesitan un orden único por activo")
                    if len(moments):
                        previous = moments[-1]
                    shape = {name: values.shape[1] for name, values in vectors.items()}
                    if dimensions is not None and dimensions != shape:
                        raise ValueError("Las dimensiones cambian entre activos")
                    dimensions = shape
                    selected = np.flatnonzero((moments >= start) & (moments < end))
                    for offset in range(0, len(selected), batch_size):
                        positions = selected[offset : offset + batch_size]
                        block = dict(
                            vectors=vectors,
                            rows=positions,
                            prices=_price_contexts(prices, ends[positions], self.context),
                            key=key,
                            prediction_at=moments[positions],
                            sample_at=moments[positions],
                            price_available_at=available[ends[positions]],
                            input_available_at=availability[positions]
                            if availability is not None
                            else None,
                            availability_valid=valid[positions] if valid is not None else None,
                            presence=presence[positions],
                        )
                        batch = _new_batch(
                            vectors, self.context, len(positions), masked=True, supervised=False
                        )
                        _fill_batch(batch, 0, block, 0, len(positions))
                        batch["source_positions"] = np.asarray(positions, dtype=np.int64)
                        batch["source_group"] = group
                        yield batch
            for kind in ("prices", "samples"):
                self._file(asset, kind)
        if sha256(self.path) != self.identity:
            raise ValueError("El manifiesto cambió durante la lectura de observaciones")

    def batches(self, *, partition, batch_size, epoch, seed, cursor=None):
        from .target_factors import confirm_sources

        confirm_sources(self._target_factor_sources)
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
        stream = self._batch_stream(partition, batch_size, epoch, seed, identity, point)
        if self.pipeline.prefetch_batches:
            stream = background(stream, self.pipeline.prefetch_batches)
        yield from stream

    def _batch_stream(self, partition, batch_size, epoch, seed, identity, point):
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


def _new_batch(vectors, context, size, *, masked=False, supervised=True):
    result = {
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
    if not supervised:
        for name in ("target", "target_available_at", "weight"):
            result.pop(name)
    return result


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
        if name in batch:
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
    manifest: Path,
    prepared: Path,
    output: Path,
    *,
    input_policy: str = STRICT_INPUTS,
    target_factors: Path | None = None,
) -> dict:
    """Preparar las etiquetas mediante el mismo contrato que consume este lector."""
    from .corpus_targets import prepare_corpus_targets as prepare

    return prepare(
        manifest, prepared, output, input_policy=input_policy, target_factors=target_factors
    )
