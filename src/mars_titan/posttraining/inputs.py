"""Lectura acotada por cohortes con presupuesto y recuperación comunes."""

import hashlib
import json
import math
from dataclasses import asdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mars_titan.environments.cohorts import (
    FINAL_TEST_START_US,
    VALIDATION_START_US,
    read_cohort,
    shapes_contract,
)
from mars_titan.episodes.augmentation import training_visits
from mars_titan.episodes.windows import EpisodeView
from mars_titan.models.baselines.inputs import MODALITIES

CONDITIONS = ("real", "real_resampled", "real_synthetic")
MAX_BYTES = 64 * 1024**2


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _resume_point(cursor, identity, sizes, offsets, batch_size):
    point = dict(visit=0, offset=0, consumed=0)
    if cursor is not None:
        if (
            not isinstance(cursor, dict)
            or set(cursor) != set(identity) | set(point)
            or any(cursor.get(k) != v for k, v in identity.items())
            or any(type(cursor.get(k)) is not int or cursor[k] < 0 for k in point)
            or cursor["visit"] > len(sizes)
            or cursor["offset"] >= (sizes[cursor["visit"]] if cursor["visit"] < len(sizes) else 1)
            or cursor["offset"] % batch_size
            or cursor["consumed"] != int(offsets[cursor["visit"]]) + cursor["offset"]
        ):
            raise ValueError("El cursor no conserva las visitas y filas confirmadas")
        point = {key: cursor[key] for key in point}
    return point


class PairedInputs:
    """Compartir la fuente real y cargar un único episodio adicional cada vez."""

    def __init__(
        self, train, validation, parent, *, windows=(), synthetic=None, synthetic_identity=None
    ):
        if train.partition != "train" or validation.partition != "validation":
            raise ValueError("Las fuentes deben separar entrenamiento y validación real")
        self.shapes = shapes_contract(train.shapes, train.max_assets, MAX_BYTES)
        if self.shapes != validation.shapes or not len(train) or not len(validation):
            raise ValueError("Las fuentes no comparten dimensiones o están vacías")
        if len(windows) > 100_000:
            raise ValueError("El número de episodios excede el presupuesto")
        for window in windows:
            EpisodeView(train, window)
        self.train, self.validation, self.parent = train, validation, parent
        self.windows, self.synthetic = tuple(windows), synthetic
        self.features = 1 + sum(math.prod(shape) for shape in self.shapes.values())
        if self.features > 16384:
            raise ValueError("Las dimensiones exceden el presupuesto del adaptador")
        self.identity = dict(
            train_sha256=train.manifest_sha256,
            validation_sha256=validation.manifest_sha256,
            parent_sha256=parent.parent_sha256,
            encoding=parent.encoding,
            windows=[asdict(window) for window in windows],
            synthetic=synthetic_identity,
            shapes={name: list(shape) for name, shape in self.shapes.items()},
        )
        self.sha256 = fingerprint(self.identity)
        self.counts = {
            source.partition: sum(row[1] for row in source.index) for source in (train, validation)
        }

    def _visits(self, partition, condition, epoch, seed):
        if partition not in {"train", "validation"} or condition not in CONDITIONS:
            raise ValueError("La partición o la condición no pertenece al diseño")
        if partition == "validation":
            if condition != "real":
                raise ValueError("La validación principal solo admite datos reales")
            from mars_titan.episodes.augmentation import Visit

            return [Visit("real", -1, i, True) for i in range(len(self.validation))]
        if condition != "real" and not self.windows:
            raise ValueError("La condición requiere un aumento confirmado")
        return training_visits(
            self.train, () if condition == "real" else self.windows, epoch=epoch, seed=seed
        )

    def budget(self, condition, batch_size):
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("El lote debe contener entre una y 4096 filas")
        visits = self._visits("train", condition, 0, 0)
        extra = sum(self.train.index[v.cohort][1] for v in visits if v.arm == "extra")
        return dict(
            real_rows=self.counts["train"],
            extra_rows=extra,
            rows=self.counts["train"] + extra,
            updates=sum(math.ceil(self.train.index[v.cohort][1] / batch_size) for v in visits),
        )

    def _episode(self, source, episode, condition):
        window = self.windows[episode]
        if condition == "real_resampled":
            view = EpisodeView(source, window)
        else:
            if self.synthetic is None:
                raise ValueError("Faltan episodios sintéticos emparejados")
            view = self.synthetic(episode)
            if (self.identity["synthetic"] or {}).get(str(episode)) != view.source.manifest_sha256:
                raise ValueError("El episodio sintético no conserva su identidad")
            counts = [source.index[i][1] for i in range(window.decision_start, window.stop)]
            other = view.window
            observed = [view.source.index[i][1] for i in range(other.decision_start, other.stop)]
            if (
                view.shapes != self.shapes
                or view.partition != "train"
                or counts != observed
                or view.window.origin != "synthetic"
            ):
                raise ValueError("El episodio sintético no conserva el contrato emparejado")
        return view

    def batches(self, *, partition, condition, batch_size, epoch=0, seed=0, cursor=None):
        if (
            type(batch_size) is not int
            or not 1 <= batch_size <= 4096
            or any(type(v) is not int or not 0 <= v < 2**32 for v in (epoch, seed))
            or batch_size * (self.features * 8 + 512) > MAX_BYTES
        ):
            raise ValueError("El lote, la semilla o la época exceden su presupuesto")
        visits = self._visits(partition, condition, epoch, seed)
        source = self.train if partition == "train" else self.validation
        sizes = [source.index[v.cohort][1] for v in visits]
        offsets = np.cumsum([0, *sizes])
        identity = dict(
            data_sha256=self.sha256,
            partition=partition,
            condition=condition,
            epoch=epoch,
            seed=seed,
            batch_size=batch_size,
        )
        point = _resume_point(cursor, identity, sizes, offsets, batch_size)
        current_episode, view = None, None
        for position in range(point["visit"], len(visits)):
            visit = visits[position]
            if visit.arm == "real":
                raw = source(visit.cohort)
            else:
                if current_episode != visit.episode:
                    view = self._episode(source, visit.episode, condition)
                    current_episode = visit.episode
                raw = view(visit.cohort - self.windows[visit.episode].decision_start)
            raw = read_cohort(raw, self.shapes, source.max_assets, MAX_BYTES)
            low, high = (
                (0, VALIDATION_START_US)
                if partition == "train"
                else (VALIDATION_START_US, FINAL_TEST_START_US)
            )
            if not low <= raw["prediction_at"] < high or (raw["target_available_at"] >= high).any():
                raise ValueError("Una etiqueta u observación cruza la partición")
            if len(raw["target"]) != sizes[position]:
                raise ValueError("La cohorte ha cambiado su número de filas")
            inherited = self.parent.predict(raw)
            start = point["offset"] if position == point["visit"] else 0
            for offset in range(start, len(inherited), batch_size):
                end = min(offset + batch_size, len(inherited))
                inputs = {k: v[offset:end].copy() for k, v in raw["inputs"].items()}
                prediction = inherited[offset:end].copy()
                with np.errstate(over="raise", invalid="raise"):
                    features = np.concatenate(
                        [inputs[k].reshape(end - offset, -1) for k in MODALITIES]
                        + [prediction[:, None].astype(np.float32)],
                        axis=1,
                    )
                final = end == len(inherited)
                confirmed = identity | dict(
                    visit=position + int(final),
                    offset=0 if final else end,
                    consumed=int(offsets[position]) + end,
                )
                ids = raw["asset_ids"][offset:end]
                yield dict(
                    inputs=inputs,
                    features=features,
                    parent=prediction,
                    target=raw["target"][offset:end].copy(),
                    sample_ids=[f"{asset}/{raw['prediction_at']}" for asset in ids],
                    market=[asset.split("/")[0] for asset in ids],
                    prediction_at=np.full(end - offset, raw["prediction_at"], dtype=np.int64),
                    origin="real" if visit.arm == "real" else condition.removeprefix("real_"),
                    confirmed_cursor=confirmed,
                )


@threadpool_limits.wrap(limits=2)
def fit_normalization(dataset, *, batch_size=256, stop=None):
    """Fijar la escala con todas las filas reales de train, antes de cualquier aumento."""
    scaler, count = StandardScaler(), 0
    for batch in dataset.batches(partition="train", condition="real", batch_size=batch_size):
        if stop is not None and stop.requested:
            raise InterruptedError("Normalización interrumpida antes de confirmar sus estadísticas")
        scaler.partial_fit(batch["features"].astype(np.float64))
        count += len(batch["target"])
    if count != dataset.counts["train"] or not count:
        raise ValueError("La normalización no concilia todas las filas de entrenamiento")
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("La normalización contiene estadísticas no finitas")
    return dict(
        train_sha256=dataset.train.manifest_sha256,
        parent_sha256=dataset.parent.parent_sha256,
        encoding=dataset.parent.encoding,
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        samples=count,
        fit_partition="train",
        feature_order=[*MODALITIES, "parent_prediction"],
    )
