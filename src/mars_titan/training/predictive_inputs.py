"""Lotes offline del adaptador con lecturas Parquet y un padre escalar alineado."""

from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import sha256
from mars_titan.environments.corpus_source import ParquetCohortSource
from mars_titan.models.baselines.inputs import MODALITIES

from .corpus_inputs import _random
from .predictive_parents import ParentPredictions


class PredictiveDataset:
    """Compartir el origen multimodal y leer solo la caché escalar propia de cada padre."""

    def __init__(self, ordered, parent_cache):
        self.ordered_path, self.cache_path = Path(ordered), Path(parent_cache)
        self.metadata, self.ordered_hash = read_manifest(self.ordered_path)
        self.cache_metadata, self.cache_hash = read_manifest(self.cache_path)
        self.sources, self.parents, self.closed = {}, {}, False
        try:
            for partition in ("train", "validation"):
                source = ParquetCohortSource(self.ordered_path, partition=partition)
                self.sources[partition] = source
                self.parents[partition] = ParentPredictions(self.cache_path, source)
            self.shapes = self.sources["train"].shapes
            self.features = 1 + sum(int(np.prod(shape)) for shape in self.shapes.values())
            if not 1 <= self.features <= 16384:
                raise ValueError("La dimensión del adaptador excede su presupuesto")
            self.counts = self.metadata["counts"]
            self.weights = self.cache_metadata["identity"]["market_weights"]
        except BaseException:
            self.close()
            raise

    def _buffer(self, size):
        return dict(
            features=np.empty((size, self.features), dtype=np.float32),
            parent=np.empty(size, dtype=np.float64),
            target=np.empty(size, dtype=np.float64),
            weight=np.empty(size, dtype=np.float64),
            prediction_at=np.empty(size, dtype="datetime64[us]"),
            target_available_at=np.empty(size, dtype="datetime64[us]"),
            sample_ids=[],
            market=[],
        )

    def batches(self, *, partition, batch_size, epoch, seed, cursor=None):
        if (
            self.closed
            or partition not in {"train", "validation"}
            or type(batch_size) is not int
            or not 1 <= batch_size <= 4096
            or type(epoch) is not int
            or not 0 <= epoch < 2**32
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise ValueError("El lector está cerrado o el lote, época o semilla no son válidos")
        if batch_size * (self.features * 4 + 512) > 64 * 1024**2:
            raise ValueError("El lote del adaptador supera el presupuesto de 64 MiB")
        if (
            sha256(self.ordered_path) != self.ordered_hash
            or sha256(self.cache_path) != self.cache_hash
        ):
            raise ValueError("Un manifiesto ha cambiado desde su confirmación")
        source, parent = self.sources[partition], self.parents[partition]
        identity = dict(
            ordered_sha256=self.ordered_hash,
            parent_cache_sha256=self.cache_hash,
            partition=partition,
            epoch=epoch,
            seed=seed,
            batch_size=batch_size,
        )
        point = dict(cohort=0, offset=0, consumed=0)
        order = (
            _random(seed, epoch, "predictive_cohorts").permutation(len(source))
            if partition == "train"
            else np.arange(len(source))
        )
        if cursor is not None:
            if (
                not isinstance(cursor, dict)
                or set(cursor) != set(identity) | set(point)
                or any(cursor[key] != value for key, value in identity.items())
                or any(type(cursor[key]) is not int or cursor[key] < 0 for key in point)
                or not cursor["cohort"] < len(source)
            ):
                raise ValueError("El cursor no corresponde al corpus y la configuración")
            point = {key: cursor[key] for key in point}
        consumed = (
            sum(source.index[int(index)][1] for index in order[: point["cohort"]]) + point["offset"]
        )
        if consumed != point["consumed"]:
            raise ValueError("El cursor no conserva el número de filas confirmado")
        buffer, position, next_point = self._buffer(batch_size), 0, None
        for cohort_position in range(point["cohort"], len(order)):
            if self.closed:
                raise ValueError("El lector se ha cerrado durante el recorrido")
            index = int(order[cohort_position])
            block = source(index)
            count = len(block["asset_ids"])
            inherited = parent.values(int(source.offsets[index]), count)
            rows = (
                _random(seed, epoch, f"predictive_rows/{index}").permutation(count)
                if partition == "train"
                else np.arange(count)
            )
            offset = point["offset"] if cohort_position == point["cohort"] else 0
            if not 0 <= offset <= count:
                raise ValueError("El cursor señala una fila inexistente")
            while offset < count:
                if self.closed:
                    raise ValueError("El lector se ha cerrado durante el recorrido")
                size = min(batch_size - position, count - offset)
                chosen = rows[offset : offset + size]
                target = slice(position, position + size)
                column = 0
                for name in MODALITIES:
                    values = block["inputs"][name][chosen].reshape(size, -1)
                    buffer["features"][target, column : column + values.shape[1]] = values
                    column += values.shape[1]
                with np.errstate(over="raise", invalid="raise"):
                    buffer["features"][target, -1] = inherited[chosen]
                buffer["parent"][target] = inherited[chosen]
                buffer["target"][target] = block["target"][chosen]
                buffer["prediction_at"][target] = block["prediction_at"]
                buffer["target_available_at"][target] = block["target_available_at"][chosen]
                ids = [block["asset_ids"][i] for i in chosen]
                markets = [asset.split("/")[0] for asset in ids]
                buffer["weight"][target] = [self.weights[market] for market in markets]
                buffer["sample_ids"].extend(f"{asset}/{block['prediction_at']}" for asset in ids)
                buffer["market"].extend(markets)
                position, offset, consumed = position + size, offset + size, consumed + size
                next_point = dict(cohort=cohort_position, offset=offset, consumed=consumed)
                if position == batch_size:
                    yield dict(buffer, confirmed_cursor=identity | next_point)
                    buffer, position = self._buffer(batch_size), 0
        if consumed != self.counts[partition]:
            raise ValueError("La época no conserva todas las filas de la población")
        if position:
            yield {
                **{key: value[:position] for key, value in buffer.items()},
                "confirmed_cursor": identity | next_point,
            }

    def close(self):
        for parent in self.parents.values():
            parent.close()
        for source in self.sources.values():
            source.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@threadpool_limits.wrap(limits=4)
def fit_standardizer(dataset, *, batch_size=256):
    """Ajustar solo con train, por bloques y con la misma ponderación del padre."""
    scaler, count, weight_sum = StandardScaler(), 0, 0.0
    for batch in dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=0):
        values = batch["features"].astype(np.float64)
        weights = batch["weight"]
        if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("La normalización requiere valores finitos y pesos positivos")
        scaler.partial_fit(values, sample_weight=weights)
        count += len(values)
        weight_sum += float(weights.sum())
    if (
        count != dataset.counts["train"]
        or not count
        or not np.isclose(weight_sum, count, rtol=1e-10)
    ):
        raise ValueError("La normalización no recorre exactamente la población ponderada")
    if (
        not np.isfinite(scaler.mean_).all()
        or not np.isfinite(scaler.scale_).all()
        or (scaler.scale_ <= 0).any()
    ):
        raise ValueError("Las estadísticas del entrenamiento no son finitas")
    return dict(
        ordered_manifest_sha256=dataset.ordered_hash,
        parent_cache_sha256=dataset.cache_hash,
        fit_batch_size=batch_size,
        code_sha256=sha256(Path(__file__)),
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        variance=scaler.var_.tolist(),
        samples=count,
        weight_sum=weight_sum,
        fit_partition="train",
        sklearn=sklearn.__version__,
        feature_order=[*MODALITIES, "parent_prediction"],
        input_dtype="float32",
        statistics_dtype="float64",
    )
