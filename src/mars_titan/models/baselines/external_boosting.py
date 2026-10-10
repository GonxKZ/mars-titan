"""XGBoost CUDA con páginas externas y entradas verificadas por bloques."""

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import sha256
from mars_titan.training.learning_hold import require_learning_allowed

from .boosting_selection import BoostingSelection, selection_callback, session_validation

MAX_MODEL_BYTES = 128 * 1024**2
MAX_DISK_CACHE_BYTES = 4 * 1024**4


def free_disk_bytes(path):
    """Espacio libre del sistema de archivos que alojará la ruta, aunque aún no exista."""
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def available_ram_bytes():
    """MemAvailable de Linux, que incluye la caché de páginas recuperable."""
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("No se puede comprobar la memoria disponible del equipo")


def directory_bytes(path):
    """Bytes de los archivos regulares bajo la ruta, sin seguir enlaces."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            status = os.lstat(os.path.join(root, name))
            if stat.S_ISREG(status.st_mode):
                total += status.st_size
    return total


def external_cache_plan(
    *,
    rows,
    features,
    max_bin,
    on_host,
    max_host_cache_bytes,
    max_disk_cache_bytes,
    available_ram,
    free_disk,
    other_disk_bytes=0,
    resident_bytes=0,
):
    """Comprobar antes de recorrer el corpus que la caché cabe en sus presupuestos reales.

    La cota host repite la fórmula float32 que el ajuste ya aplica. Para disco se
    estiman páginas ELLPACK densas con índice de bin local por característica,
    ceil(log2(max_bin)) bits por valor, sin símbolo de ausente porque las entradas son
    finitas. Con XGBoost 3.3 se midieron exactamente 6, 7 y 8 bits con 64, 128 y 256
    bins sobre filas reales, y la construcción vuelve a contrastar los bytes escritos.
    También se informa de la cota con índices globales, ceil(log2(features * max_bin + 1))
    bits. La validación residente ocupa RAM y se contrasta con la memoria disponible.
    """
    for value, low, high in (
        (rows, 1, 2**63 - 1),
        (features, 1, 65536),
        (max_bin, 2, 512),
        (max_host_cache_bytes, 1, 24 * 1024**3),
        (available_ram, 0, 2**63 - 1),
        (free_disk, 0, 2**63 - 1),
        (other_disk_bytes, 0, 2**63 - 1),
        (resident_bytes, 0, 2**63 - 1),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("El plan de memoria externa necesita enteros acotados")
    if type(on_host) is not bool or (
        max_disk_cache_bytes is not None
        and (
            on_host
            or type(max_disk_cache_bytes) is not int
            or not 1 <= max_disk_cache_bytes <= MAX_DISK_CACHE_BYTES
        )
    ):
        raise ValueError("El presupuesto de disco solo se declara para la caché en disco")
    host = rows * (features * 4 + 16)
    local_bits, global_bits = (max_bin - 1).bit_length(), (features * max_bin).bit_length()
    dense = math.ceil(rows * features * local_bits / 8)
    upper = math.ceil(rows * features * global_bits / 8)
    disk = 0 if on_host else dense
    plan = dict(
        schema_version=1,
        rows=rows,
        features=features,
        max_bin=max_bin,
        cache_location="host" if on_host else "disk",
        host_cache_bytes_estimate=host if on_host else 0,
        max_host_cache_bytes=max_host_cache_bytes,
        available_ram_bytes=available_ram,
        disk_cache_bytes_estimate=disk,
        disk_cache_estimate_basis="ellpack_dense_feature_local_bins_measured_xgboost_3_3",
        disk_cache_bytes_global_bins_bound=0 if on_host else upper,
        max_disk_cache_bytes=max_disk_cache_bytes,
        other_disk_bytes=other_disk_bytes,
        free_disk_bytes=free_disk,
        resident_validation_bytes=resident_bytes,
    )
    if on_host and host > min(max_host_cache_bytes, available_ram):
        raise ValueError(
            "La caché host estimada supera el presupuesto declarado o la RAM disponible"
        )
    if resident_bytes + (host if on_host else 0) > available_ram:
        raise ValueError("La validación residente y la caché host no caben en la RAM disponible")
    if max_disk_cache_bytes is not None and disk > max_disk_cache_bytes:
        raise ValueError("Las páginas estimadas superan el presupuesto de disco declarado")
    if disk + other_disk_bytes > free_disk:
        raise ValueError("Las páginas y cachés estimadas no caben en el disco libre")
    return plan


def _libraries():
    # PyTorch carga primero sus bibliotecas CUDA, antes de la detección de CuPy.
    require_cuda()
    try:
        import cupy as cp
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "La ruta externa necesita el extra boosting con XGBoost y CuPy"
        ) from error
    if (
        xgb.__version__ != "3.3.0"
        or cp.__version__ != "14.2.0"
        or not xgb.build_info().get("USE_CUDA")
    ):
        raise RuntimeError("Las versiones o el soporte CUDA no coinciden con la ruta comprobada")
    cp.cuda.Device(0).use()
    return cp, xgb


def _device(booster):
    value = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    if value != "cuda:0":
        raise RuntimeError("XGBoost no está utilizando cuda:0. No se admite sustitución por CPU.")
    return value


def _blocks(factory, expected_rows, max_bytes):
    """Normalizar cortes físicos a bloques repetibles sin concatenar todo el corpus."""
    output_x, output_y, position, features, capacity = None, None, 0, None, None
    for raw_x, raw_y in factory():
        x, y = np.asarray(raw_x), np.asarray(raw_y)
        if (
            x.ndim != 2
            or not len(x)
            or not 1 <= x.shape[1] <= 65536
            or y.shape != (len(x),)
            or np.iscomplexobj(x)
            or np.iscomplexobj(y)
        ):
            raise ValueError("El bloque no contiene una matriz y sus etiquetas")
        if x.nbytes + y.nbytes > max_bytes:
            raise ValueError("El bloque supera el presupuesto antes de su copia a CUDA")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("El bloque necesita valores finitos, sin sustituirlos por ausencias")
        if features is not None and features != x.shape[1]:
            raise ValueError("Las dimensiones cambian entre bloques")
        features = x.shape[1]
        capacity = min(expected_rows, max_bytes // ((features + 1) * 4))
        if capacity < 1:
            raise ValueError("El presupuesto no permite una fila tabular")
        offset = 0
        while offset < len(x):
            if output_x is None:
                output_x = np.empty((capacity, features), dtype=np.float32)
                output_y = np.empty(capacity, dtype=np.float32)
            size = min(capacity - position, len(x) - offset)
            try:
                with np.errstate(over="raise", invalid="raise"):
                    output_x[position : position + size] = x[offset : offset + size]
                    output_y[position : position + size] = y[offset : offset + size]
            except FloatingPointError as error:
                raise ValueError("El bloque no conserva valores finitos en float32") from error
            position, offset = position + size, offset + size
            if position == capacity:
                yield output_x, output_y
                output_x, output_y, position = None, None, 0
    if position:
        yield output_x[:position], output_y[:position]


@dataclass
class ExternalBoostingModel:
    booster: object
    training_rows: int
    features: int
    audit: dict
    max_batch_bytes: int = 256 * 1024**2

    def predict(self, values):
        values = np.asarray(values)
        if (
            values.ndim != 2
            or values.shape[1] != self.features
            or not len(values)
            or values.nbytes > self.max_batch_bytes
            or values.size * np.dtype(np.float32).itemsize > self.max_batch_bytes
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Las entradas de boosting no tienen forma, presupuesto o valores finitos válidos"
            )
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = np.ascontiguousarray(values, dtype=np.float32)
        except (TypeError, ValueError, FloatingPointError) as error:
            raise ValueError("Las entradas no conservan valores finitos en float32") from error
        import cupy as cp

        with cp.cuda.Device(0):
            return self.predict_device(cp.asarray(values, dtype=cp.float32))

    def predict_device(self, values):
        """Predecir un bloque float32 que ya reside en cuda:0 y se comprobó al cargarlo."""
        import cupy as cp

        if (
            not isinstance(values, cp.ndarray)
            or values.dtype != cp.float32
            or values.ndim != 2
            or values.shape[1] != self.features
            or not values.flags.c_contiguous
        ):
            raise ValueError("El bloque residente no es una matriz float32 contigua en CUDA")
        _device(self.booster)
        with cp.cuda.Device(0):
            result = self.booster.inplace_predict(values)
            if not isinstance(result, cp.ndarray):
                raise RuntimeError("La predicción no se ha ejecutado en CUDA")
            result = result.get()
        if result.shape != (len(values),) or not np.isfinite(result).all():
            raise ValueError("La predicción de boosting no es finita o tiene otra forma")
        return result

    def predict_matrix(self, matrix):
        """Predecir en orden todas las filas de la matriz cuantizada de entrenamiento.

        Un árbol hist divide en cortes de esa matriz y el predictor de ELLPACK compara el
        límite inferior de cada bin, así que cada fila sigue las mismas ramas que con sus
        valores float32. La matriz no vuelve a recorrer su factoría.
        """
        if (
            not isinstance(matrix, ExternalMatrix)
            or matrix.data is None
            or matrix.features != self.features
            or matrix.rows != self.training_rows
        ):
            raise ValueError("La matriz no corresponde a las filas y dimensiones del modelo")
        import cupy as cp

        _device(self.booster)
        passes = list(matrix.iterator.completed)
        with cp.cuda.Device(0):
            result = np.asarray(self.booster.predict(matrix.data))
        if matrix.iterator.completed != passes:
            raise ValueError("La predicción ha vuelto a recorrer la factoría de la matriz")
        if result.shape != (matrix.rows,) or not np.isfinite(result).all():
            raise ValueError("La predicción de boosting no es finita o tiene otra forma")
        return result

    def save(self, path):
        path = Path(path)
        safe_destination(path)
        if path.exists() or path.suffix != ".ubj":
            raise ValueError("El modelo necesita un archivo .ubj nuevo, que no exista")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".boosting-", suffix=".ubj", dir=path.parent
        )
        os.close(descriptor)
        try:
            self.booster.set_attr(
                mars_external_schema="2",
                mars_external_contract=json.dumps(
                    dict(
                        training_rows=self.training_rows,
                        features=self.features,
                        max_batch_bytes=self.max_batch_bytes,
                        audit=self.audit,
                    ),
                    allow_nan=False,
                    sort_keys=True,
                ),
            )
            self.booster.save_model(temporary)
            if os.path.getsize(temporary) > MAX_MODEL_BYTES:
                raise ValueError("El tamaño del modelo supera el presupuesto de recuperación")
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(temporary)
        return sha256(path)

    @classmethod
    def load(cls, path, expected_sha256, *, training_rows=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("El modelo no es un archivo regular")
        with path.open("rb") as stream:
            payload = stream.read(MAX_MODEL_BYTES + 1)
        if len(payload) > MAX_MODEL_BYTES or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("La huella o el tamaño del modelo no coinciden")
        _, xgb = _libraries()
        booster = xgb.Booster()
        booster.load_model(bytearray(payload))
        if booster.attr("mars_external_schema") != "2":
            raise ValueError("El modelo no conserva el contrato externo del proyecto")
        meta = json.loads(booster.attr("mars_external_contract"))
        count, budget = meta.get("training_rows"), meta.get("max_batch_bytes")
        if (
            type(count) is not int
            or count < 1
            or (training_rows is not None and training_rows != count)
            or type(budget) is not int
            or not 1 <= budget <= 512 * 1024**2
            or meta.get("features") != booster.num_features()
            or not isinstance(meta.get("audit"), dict)
            or meta["audit"].get("rows") != count
            or meta["audit"].get("rounds") != booster.num_boosted_rounds()
        ):
            raise ValueError("Las filas, dimensiones o presupuestos del modelo no coinciden")
        booster.set_param(dict(device="cuda:0", nthread=4))
        _device(booster)
        return cls(booster, count, booster.num_features(), meta["audit"], budget)


@dataclass
class ExternalMatrix:
    """Matriz cuantizada viva con sus páginas, su construcción y la auditoría de sus pasadas.

    Los cortes y las páginas solo dependen de las filas, su orden, el reparto en lotes y
    `max_bin`. Dos construcciones con la misma `construction` producen los mismos bytes,
    así que una matriz viva puede servir a varias configuraciones de árboles.
    """

    data: object
    iterator: object
    directory: Path
    rows: int
    features: int
    construction: dict
    audit: dict

    def disk_bytes(self):
        return directory_bytes(self.directory) if self.directory.exists() else 0

    def close(self):
        """Liberar la matriz. XGBoost borra sus páginas al destruirla y después el directorio.

        El manejador se libera aunque otro objeto conserve la referencia de Python, para
        que las páginas desaparezcan antes de borrar el directorio y no al recolectarla.
        """
        iterator, data = self.iterator, self.data
        self.data = self.iterator = None
        if iterator is not None and iterator.iterator is not None:
            if hasattr(iterator.iterator, "close"):
                iterator.iterator.close()
        if data is not None:
            data.__del__()
        del iterator, data
        if self.directory.exists():
            safe_destination(self.directory)
            shutil.rmtree(self.directory)


def _construction(
    *, expected_rows, max_bin, max_batch_bytes, max_host_cache_bytes, on_host, max_disk_cache_bytes
):
    for value, low, high in (
        (expected_rows, 1, 2**63 - 1),
        (max_bin, 2, 512),
        (max_batch_bytes, 1, 512 * 1024**2),
        (max_host_cache_bytes, 1, 24 * 1024**3),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Los presupuestos y parámetros de boosting no son válidos")
    if type(on_host) is not bool:
        raise ValueError("La ubicación de la caché debe ser booleana")
    if max_disk_cache_bytes is not None and (
        on_host
        or type(max_disk_cache_bytes) is not int
        or not 1 <= max_disk_cache_bytes <= MAX_DISK_CACHE_BYTES
    ):
        raise ValueError("El presupuesto de disco solo se declara para la caché en disco")
    return dict(
        expected_rows=expected_rows,
        max_bin=max_bin,
        max_batch_bytes=max_batch_bytes,
        max_host_cache_bytes=max_host_cache_bytes,
        on_host=on_host,
        max_disk_cache_bytes=max_disk_cache_bytes,
    )


def build_external_matrix(
    factory,
    cache_directory,
    *,
    expected_rows,
    max_bin=128,
    max_batch_bytes=256 * 1024**2,
    max_host_cache_bytes=16 * 1024**3,
    on_host=True,
    max_disk_cache_bytes=None,
):
    """Cuantizar todas las filas en páginas externas, sin ajustar ningún árbol.

    El iterador recorre la factoría dos veces (bosquejo de cuantiles y páginas) y exige
    las mismas filas y valores en ambas. Con un presupuesto de disco declarado, cada lote
    comprueba los bytes ya escritos y la construcción falla si los supera.
    """
    construction = _construction(
        expected_rows=expected_rows,
        max_bin=max_bin,
        max_batch_bytes=max_batch_bytes,
        max_host_cache_bytes=max_host_cache_bytes,
        on_host=on_host,
        max_disk_cache_bytes=max_disk_cache_bytes,
    )
    if not callable(factory):
        raise ValueError("La factoría no es válida")
    cache_directory = Path(cache_directory)
    safe_destination(cache_directory)
    if cache_directory.exists():
        raise ValueError("La caché necesita un directorio nuevo")
    cp, xgb = _libraries()
    cache_directory.mkdir(parents=True)

    class Iterator(xgb.DataIter):
        def __init__(self):
            self.iterator = None
            self.rows, self.features, self.max_bytes = 0, None, 0
            self.completed, self.fingerprint, self.confirmed_digest = [], None, None
            self.exhausted = False
            super().__init__(
                cache_prefix=str(cache_directory / "pages"),
                on_host=on_host,
                min_cache_page_bytes=min(max_batch_bytes, 64 * 1024**2),
            )

        def reset(self):
            if self.iterator is not None and hasattr(self.iterator, "close"):
                self.iterator.close()
            self.iterator = iter(_blocks(factory, expected_rows, max_batch_bytes))
            self.rows, self.fingerprint, self.exhausted = 0, hashlib.sha256(), False

        def next(self, input_data):
            if self.exhausted:
                return False
            try:
                x, y = next(self.iterator)
            except StopIteration:
                if self.rows != expected_rows:
                    raise ValueError(
                        "El recorrido no coincide con las filas de la población"
                    ) from None
                digest = self.fingerprint.hexdigest()
                if self.confirmed_digest is not None and digest != self.confirmed_digest:
                    raise ValueError("Los valores o el orden cambian entre pasadas") from None
                self.confirmed_digest = digest
                self.completed.append(self.rows)
                self.exhausted = True
                return False
            x, y = np.asarray(x), np.asarray(y)
            if (
                x.ndim != 2
                or not len(x)
                or not 1 <= x.shape[1] <= 65536
                or y.shape != (len(x),)
                or np.iscomplexobj(x)
                or np.iscomplexobj(y)
            ):
                raise ValueError("El bloque no contiene una matriz y sus etiquetas")
            size = x.nbytes + y.nbytes
            if size > max_batch_bytes:
                raise ValueError("El bloque supera el presupuesto antes de su copia a CUDA")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(
                    "El bloque necesita valores finitos, sin sustituirlos por ausencias"
                )
            if self.features is not None and self.features != x.shape[1]:
                raise ValueError("Las dimensiones cambian entre bloques")
            self.features = x.shape[1]
            estimate = expected_rows * (self.features * 4 + 16)
            if on_host and estimate > max_host_cache_bytes:
                raise ValueError(
                    "La estimación conservadora de la caché supera el presupuesto host"
                )
            with np.errstate(over="raise", invalid="raise"):
                x, y = (
                    np.ascontiguousarray(x, dtype=np.float32),
                    np.ascontiguousarray(y, dtype=np.float32),
                )
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("El bloque no conserva valores finitos al convertir a float32")
            if (
                max_disk_cache_bytes is not None
                and directory_bytes(cache_directory) > max_disk_cache_bytes
            ):
                raise ValueError("Las páginas escritas superan el presupuesto de disco")
            self.rows += len(x)
            if self.rows > expected_rows:
                raise ValueError("El iterador entrega más filas que la población declarada")
            self.max_bytes = max(self.max_bytes, size)
            self.fingerprint.update(memoryview(x).cast("B"))
            self.fingerprint.update(memoryview(y).cast("B"))
            self.values, self.target = cp.asarray(x), cp.asarray(y)
            input_data(data=self.values, label=self.target)
            return True

    pool = cp.cuda.MemoryAsyncPool("default")
    data, iterator = None, None
    try:
        with (
            cp.cuda.Device(0),
            cp.cuda.using_allocator(pool.malloc),
            xgb.config_context(use_cuda_async_pool=True),
        ):
            iterator = Iterator()
            data = xgb.ExtMemQuantileDMatrix(
                iterator, max_bin=max_bin, cache_host_ratio=1.0, nthread=4
            )
            if (
                data.num_row() != expected_rows
                or data.num_col() != iterator.features
                or not iterator.completed
            ):
                raise ValueError("La matriz externa no conserva la población y sus dimensiones")
            disk_audit = {}
            if max_disk_cache_bytes is not None:
                observed = directory_bytes(cache_directory)
                if observed > max_disk_cache_bytes:
                    raise ValueError("Las páginas escritas superan el presupuesto de disco")
                disk_audit = dict(
                    disk_cache=dict(max_bytes=max_disk_cache_bytes, observed_bytes=observed)
                )
        # El último lote ya está en las páginas. No se retiene en la GPU mientras viva la matriz.
        iterator.values = iterator.target = None
        matrix = ExternalMatrix(
            data,
            iterator,
            cache_directory,
            expected_rows,
            iterator.features,
            construction,
            dict(
                cache_location="host" if on_host else "disk",
                completed_pass_rows=list(iterator.completed),
                data_sha256=iterator.confirmed_digest,
                max_input_batch_bytes=iterator.max_bytes,
                **disk_audit,
            ),
        )
        data = iterator = None
        return matrix
    finally:
        if (
            iterator is not None
            and iterator.iterator is not None
            and hasattr(iterator.iterator, "close")
        ):
            iterator.iterator.close()
        del data, iterator


def fit_external_boosting(
    factory,
    cache_directory,
    *,
    expected_rows,
    rounds=100,
    max_depth=4,
    max_bin=128,
    learning_rate=0.05,
    seed=42,
    max_batch_bytes=256 * 1024**2,
    max_host_cache_bytes=16 * 1024**3,
    on_host=True,
    max_disk_cache_bytes=None,
    resume=None,
    checkpoint=None,
    checkpoint_interval=10,
    selection=None,
    validation_factory=None,
    validation_rows=None,
    stop_requested=None,
    matrix=None,
):
    """Ajustar todas las filas mediante ExtMemQuantileDMatrix, sin concatenación global.

    Sin `matrix` se construye una matriz propia en `cache_directory` y se libera al
    terminar. Una `matrix` ya construida con la misma construcción declarada se usa sin
    recorrer de nuevo la factoría y la libera quien la creó.
    """
    require_learning_allowed("el ajuste XGBoost con páginas externas")
    construction = _construction(
        expected_rows=expected_rows,
        max_bin=max_bin,
        max_batch_bytes=max_batch_bytes,
        max_host_cache_bytes=max_host_cache_bytes,
        on_host=on_host,
        max_disk_cache_bytes=max_disk_cache_bytes,
    )
    for value, low, high in (
        (rounds, 1, 2000 if selection is not None else 1000),
        (max_depth, 1, 12),
        (seed, 0, 2**31 - 1),
        (checkpoint_interval, 1, 1000),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Los presupuestos y parámetros de boosting no son válidos")
    if (
        not callable(factory)
        or type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 1
        or (resume is not None and not isinstance(resume, ExternalBoostingModel))
        or (checkpoint is not None and not callable(checkpoint))
        or (stop_requested is not None and not callable(stop_requested))
        or (matrix is not None and not isinstance(matrix, ExternalMatrix))
    ):
        raise ValueError("La factoría o la tasa de aprendizaje no son válidas")
    selector = None
    if selection is not None:
        selector = BoostingSelection(
            selection, rounds, state=resume.audit.get("selection") if resume else None
        )
        if (
            not callable(validation_factory)
            or type(validation_rows) is not int
            or validation_rows < 1
            or resume is not None
            and (
                resume.audit.get("selection") is None
                or resume.audit.get("selection_policy") != selection
            )
        ):
            raise ValueError("La selección necesita validación completa y un estado recuperable")
    elif validation_factory is not None or validation_rows is not None:
        raise ValueError("La validación durante el ajuste necesita una edición de selección")
    owned = matrix is None
    if not owned and (matrix.construction != construction or matrix.data is None):
        raise ValueError("La matriz compartida no corresponde a esta construcción")
    if owned:
        matrix = build_external_matrix(factory, cache_directory, **construction)
    cp, xgb = _libraries()
    pool = cp.cuda.MemoryAsyncPool("default")
    try:
        with (
            cp.cuda.Device(0),
            cp.cuda.using_allocator(pool.malloc),
            xgb.config_context(use_cuda_async_pool=True),
        ):
            params = dict(
                device="cuda:0",
                tree_method="hist",
                objective="reg:squarederror",
                max_bin=max_bin,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=1.0,
                colsample_bytree=1.0,
                seed=seed,
                nthread=4,
            )
            built = matrix.audit
            audit = dict(
                device="cuda:0",
                external_memory=True,
                cache_location=built["cache_location"],
                completed_pass_rows=list(built["completed_pass_rows"]),
                data_sha256=built["data_sha256"],
                max_input_batch_bytes=built["max_input_batch_bytes"],
                rows=expected_rows,
                xgboost=xgb.__version__,
                cupy=cp.__version__,
                params=params,
                precision="float32_features_labels",
                cuda_async_pool=True,
                **({"disk_cache": dict(built["disk_cache"])} if "disk_cache" in built else {}),
            )
            completed = 0
            if resume is not None:
                completed = resume.booster.num_boosted_rounds()
                if (
                    resume.training_rows != expected_rows
                    or resume.features != matrix.features
                    or resume.max_batch_bytes != max_batch_bytes
                    or resume.audit.get("data_sha256") != built["data_sha256"]
                    or resume.audit.get("params") != params
                    or resume.audit.get("xgboost") != xgb.__version__
                    or resume.audit.get("cupy") != cp.__version__
                    or resume.audit.get("rounds") != completed
                    or not 1 <= completed <= rounds
                    or selector is not None
                    and selector.state["completed_rounds"] != completed
                ):
                    raise ValueError("La continuación cambió de datos, parámetros o presupuesto")

            features = matrix.features
            recovery_callback = None

            def wrap(booster):
                selection_audit = (
                    dict(selection=dict(selector.state), selection_policy=dict(selector.policy))
                    if selector
                    else {}
                )
                if selector:
                    selection_audit["replayed_rounds"] = (
                        recovery_callback.replayed_rounds if recovery_callback else 0
                    )
                    selection_audit["replay_seconds"] = (
                        recovery_callback.replay_seconds if recovery_callback else 0.0
                    )
                return ExternalBoostingModel(
                    booster,
                    expected_rows,
                    features,
                    dict(
                        audit,
                        device=_device(booster),
                        rounds=booster.num_boosted_rounds(),
                        **selection_audit,
                    ),
                    max_batch_bytes,
                )

            class SaveRound(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    count = model.num_boosted_rounds()
                    if checkpoint is not None and (
                        count % checkpoint_interval == 0 or count == rounds
                    ):
                        checkpoint(wrap(model))
                    return False

            callbacks = [SaveRound()]
            if selector:
                recovery_callback = selection_callback(
                    xgb,
                    selector,
                    lambda booster: session_validation(
                        wrap(booster), validation_factory, expected_rows=validation_rows
                    ),
                    lambda booster, state: checkpoint(wrap(booster)) if checkpoint else None,
                    replay_model=resume.booster if resume is not None else None,
                    stop_requested=stop_requested,
                )
                callbacks = [recovery_callback]
            passes = list(matrix.iterator.completed)
            booster = (
                resume.booster
                if completed == rounds or selector and selector.state["stop_reason"]
                else xgb.train(
                    params,
                    matrix.data,
                    num_boost_round=rounds if selector else rounds - completed,
                    xgb_model=resume.booster if resume is not None and not selector else None,
                    callbacks=callbacks,
                )
            )
            if matrix.iterator.completed != passes:
                raise ValueError("El ajuste ha vuelto a recorrer la factoría de la matriz")
            if selector:
                if booster.num_boosted_rounds() < completed:
                    if not stop_requested or not stop_requested():
                        raise ValueError("La reconstrucción no alcanza el prefijo confirmado")
                    booster = resume.booster
                booster = booster[: selector.state["selected_round"]]
            return wrap(booster)
    finally:
        if owned:
            matrix.close()
