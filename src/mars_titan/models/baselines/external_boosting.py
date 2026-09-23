"""XGBoost CUDA con páginas externas y entradas verificadas por bloques."""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import sha256

MAX_MODEL_BYTES = 128 * 1024**2


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

        _device(self.booster)
        with cp.cuda.Device(0):
            result = self.booster.inplace_predict(cp.asarray(values, dtype=cp.float32))
            if not isinstance(result, cp.ndarray):
                raise RuntimeError("La predicción no se ha ejecutado en CUDA")
            result = result.get()
        if result.shape != (len(values),) or not np.isfinite(result).all():
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
    resume=None,
    checkpoint=None,
    checkpoint_interval=10,
):
    """Ajustar todas las filas mediante ExtMemQuantileDMatrix, sin concatenación global."""
    for value, low, high in (
        (expected_rows, 1, 2**63 - 1),
        (rounds, 1, 1000),
        (max_depth, 1, 12),
        (max_bin, 2, 512),
        (max_batch_bytes, 1, 512 * 1024**2),
        (max_host_cache_bytes, 1, 24 * 1024**3),
        (seed, 0, 2**31 - 1),
        (checkpoint_interval, 1, 1000),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Los presupuestos y parámetros de boosting no son válidos")
    if (
        not callable(factory)
        or type(on_host) is not bool
        or type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 1
        or (resume is not None and not isinstance(resume, ExternalBoostingModel))
        or (checkpoint is not None and not callable(checkpoint))
    ):
        raise ValueError("La factoría o la tasa de aprendizaje no son válidas")
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
            audit = dict(
                device="cuda:0",
                external_memory=True,
                cache_location="host" if on_host else "disk",
                completed_pass_rows=iterator.completed,
                data_sha256=iterator.confirmed_digest,
                max_input_batch_bytes=iterator.max_bytes,
                rows=expected_rows,
                xgboost=xgb.__version__,
                cupy=cp.__version__,
                params=params,
                precision="float32_features_labels",
                cuda_async_pool=True,
            )
            completed = 0
            if resume is not None:
                completed = resume.booster.num_boosted_rounds()
                if (
                    resume.training_rows != expected_rows
                    or resume.features != iterator.features
                    or resume.max_batch_bytes != max_batch_bytes
                    or resume.audit.get("data_sha256") != iterator.confirmed_digest
                    or resume.audit.get("params") != params
                    or resume.audit.get("xgboost") != xgb.__version__
                    or resume.audit.get("cupy") != cp.__version__
                    or resume.audit.get("rounds") != completed
                    or not 1 <= completed <= rounds
                ):
                    raise ValueError("La continuación cambió de datos, parámetros o presupuesto")

            features = iterator.features

            def wrap(booster):
                return ExternalBoostingModel(
                    booster,
                    expected_rows,
                    features,
                    dict(audit, device=_device(booster), rounds=booster.num_boosted_rounds()),
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

            booster = (
                resume.booster
                if completed == rounds
                else xgb.train(
                    params,
                    data,
                    num_boost_round=rounds - completed,
                    xgb_model=resume.booster if resume is not None else None,
                    callbacks=[SaveRound()],
                )
            )
            return wrap(booster)
    finally:
        if (
            iterator is not None
            and iterator.iterator is not None
            and hasattr(iterator.iterator, "close")
        ):
            iterator.iterator.close()
        del data, iterator
