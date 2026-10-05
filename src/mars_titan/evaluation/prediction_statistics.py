"""Revisar predicciones congeladas y comparar pérdidas por sesión observada."""

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.special import log_softmax
from scipy.stats import rankdata

from mars_titan.data.storage import sha256
from mars_titan.environments.actions import ActionGrid

KEYS = ("sample_id", "asset_id", "market", "prediction_at", "target")
PREDICTIONS = ("prediction", "parent", "zero", "center")
MAX_ROWS = 1_000_000
MAX_FILE_BYTES = 512 * 1024**2
ABS_TOL, REL_TOL = 1e-12, 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bounds(source):
    _require(source.get("partition") in {"calibration", "evaluation"}, "Partición no admitida")
    try:
        start, end = (np.datetime64(value, "us") for value in source["bounds"])
    except (TypeError, ValueError) as error:
        raise ValueError("Los límites temporales no son válidos") from error
    _require(
        not np.isnat(start) and not np.isnat(end) and start < end <= np.datetime64("2024-01-01"),
        "Los límites invaden la reserva final o no forman un intervalo",
    )
    return start, end


def initial_policy(parent, grid):
    """Reconstruir la política inicial con la rejilla ya ajustada, sin leer etiquetas."""
    action_grid = ActionGrid.from_dict(grid)
    values = np.asarray(parent, dtype=np.float64)
    _require(
        values.ndim == 1 and 0 < values.size <= MAX_ROWS and np.isfinite(values).all(),
        "El padre necesita predicciones finitas y acotadas",
    )
    result = np.empty_like(values)
    z = action_grid.values / action_grid.scale
    for offset in range(0, len(values), 4096):
        centers = values[offset : offset + 4096] / action_grid.scale
        with np.errstate(over="raise", invalid="raise"):
            logits = centers[:, None] * z[None, :] - 0.5 * np.square(z)[None, :]
            probabilities = np.exp(log_softmax(logits, axis=1))
        result[offset : offset + len(centers)] = action_grid.median(probabilities)
    return result


class InitialPolicyCache:
    """Reutilizar proyecciones idénticas con un límite de bytes y de entradas."""

    def __init__(self, max_bytes=8 * 1024**2):
        _require(type(max_bytes) is int and 0 <= max_bytes <= 64 * 1024**2, "Caché no admitida")
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.entries = OrderedDict()

    def get(self, parent, grid):
        values = np.ascontiguousarray(parent, dtype=np.float64)
        digest = hashlib.sha256(values.view(np.uint8))
        digest.update(json.dumps(grid, sort_keys=True, allow_nan=False).encode())
        key = digest.digest()
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        result = initial_policy(values, grid)
        result.setflags(write=False)
        if result.nbytes <= self.max_bytes:
            while self.entries and (
                self.bytes_used + result.nbytes > self.max_bytes or len(self.entries) >= 32
            ):
                _, expired = self.entries.popitem(last=False)
                self.bytes_used -= expired.nbytes
            self.entries[key] = result
            self.bytes_used += result.nbytes
        return result


def _correlation(left, right, *, rank=False):
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    if rank:
        left, right = rankdata(left, method="average"), rankdata(right, method="average")
    left, right = left - left[0], right - right[0]
    left = left / np.max(np.abs(left))
    right = right / np.max(np.abs(right))
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else None


def _mean_defined(values):
    valid = [value for value in values if value is not None]
    return math.fsum(valid) / len(valid) if valid else None


def _declared(metrics, expected):
    if expected is None:
        return
    _require(isinstance(expected, dict) and "prediction" in expected, "Faltan métricas declaradas")
    for name, record in expected.items():
        _require(name in metrics, "Falta una columna de predicciones declarada")
        for field in ("samples", "session_count", "session_mae", "session_mse"):
            _require(field in record, "Falta una métrica primaria declarada")
        for field in ("samples", "session_count", "session_mae", "session_mse", "mae", "mse"):
            if field not in record:
                continue
            actual, target = metrics[name][field], record[field]
            _require(
                type(target) in (int, float)
                and math.isfinite(target)
                and (
                    actual == target
                    if field in {"samples", "session_count"}
                    else math.isclose(actual, target, rel_tol=REL_TOL, abs_tol=ABS_TOL)
                ),
                f"No coincide la métrica declarada {name}.{field}",
            )


def review_predictions(source, *, cohort=None, initial_cache=None):
    """Leer solo una predicción, comprobar su cohorte y devolver pérdidas por sesión."""
    start, end = _bounds(source)
    path = Path(source["path"])
    _require(
        not path.is_symlink() and 0 < path.stat().st_size <= MAX_FILE_BYTES, "Archivo no admitido"
    )
    before = path.stat()
    _require(sha256(path) == source["sha256"], "La huella de predicciones no coincide")
    parquet = pq.ParquetFile(path, pre_buffer=False)
    try:
        schema, metadata = parquet.schema_arrow, parquet.metadata
        _require(0 < metadata.num_rows <= MAX_ROWS, "La predicción supera el presupuesto de filas")
        _require(
            sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
            <= MAX_FILE_BYTES,
            "El Parquet supera el presupuesto descomprimido",
        )
        required = {*KEYS, "prediction", "parent", "zero"}
        _require(required <= set(schema.names), "Faltan columnas obligatorias")
        _require(
            schema.field("prediction_at").type == pa.timestamp("us", tz="UTC"),
            "Fechas sin UTC en microsegundos",
        )
        # Se comprueban los tiempos antes de solicitar cualquier columna de etiquetas.
        stamp = parquet.read(columns=["prediction_at"], use_threads=False)["prediction_at"]
        _require(stamp.null_count == 0, "Hay fechas ausentes")
        times = stamp.to_numpy()
        _require(
            np.all((times >= start) & (times < end)),
            "Las fechas no pertenecen al bloque autorizado",
        )
        names = [name for name in PREDICTIONS if name in schema.names]
        table = parquet.read(columns=[*KEYS, *names], use_threads=False).combine_chunks()
    finally:
        parquet.close()
    after = path.stat()
    _require(
        (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "Las predicciones cambiaron durante la lectura",
    )
    _require(all(column.null_count == 0 for column in table.columns), "Hay valores ausentes")
    for name in ("sample_id", "asset_id", "market"):
        _require(pa.types.is_string(table[name].type), "Una identidad no es texto")
        lengths = pc.utf8_length(table[name]).to_numpy()
        _require(np.all((lengths > 0) & (lengths <= 256)), "Una identidad está vacía o es excesiva")
    for name in ("target", *names):
        _require(table[name].type == pa.float64(), "Los valores deben conservar float64")
    table = table.sort_by([("sample_id", "ascending")])
    identifiers = table["sample_id"].to_numpy()
    _require(np.all(identifiers[1:] != identifiers[:-1]), "Hay muestras duplicadas")
    target = table["target"].to_numpy()
    values = {name: table[name].to_numpy() for name in names}
    _require(
        np.isfinite(target).all() and all(np.isfinite(v).all() for v in values.values()),
        "Hay valores no finitos",
    )
    _require(np.all(values["zero"] == 0), "El control cero no es nulo")
    current_cohort = table.select(KEYS)
    _require(
        cohort is None or current_cohort.equals(cohort), "La población o sus etiquetas han cambiado"
    )
    grid = source.get("metadata", {}).get("grid")
    if grid is not None:
        project = initial_policy if initial_cache is None else initial_cache.get
        values["initial_policy"] = project(values["parent"], grid)
    markets = table["market"].to_numpy()
    _require(np.isin(markets, ("US", "CN")).all(), "Mercado no admitido")
    moments = table["prediction_at"].to_numpy().astype(np.int64)
    assets = table["asset_id"].to_numpy()
    order = np.lexsort((assets, moments, markets))
    duplicate = (
        (markets[order][1:] == markets[order][:-1])
        & (moments[order][1:] == moments[order][:-1])
        & (assets[order][1:] == assets[order][:-1])
    )
    _require(not np.any(duplicate), "Hay un activo duplicado dentro de una sesión")
    changes = (markets[order][1:] != markets[order][:-1]) | (
        moments[order][1:] != moments[order][:-1]
    )
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    daily = dict(
        market=markets[order][starts],
        prediction_at=pa.array(moments[order][starts], type=pa.timestamp("us", tz="UTC")),
        samples=counts,
    )
    metrics = {}
    for name, predicted in values.items():
        with np.errstate(over="raise", invalid="raise"):
            errors = predicted - target
            absolute, squared = np.abs(errors), np.square(errors)
            means = np.add.reduceat(absolute[order], starts) / counts
            means_squared = np.add.reduceat(squared[order], starts) / counts
        _require(
            np.isfinite(means).all() and np.isfinite(means_squared).all(),
            "El error desborda float64",
        )
        daily[f"mae_{name}"] = means
        daily[f"mse_{name}"] = means_squared
        metrics[name] = dict(
            samples=len(target),
            session_count=len(starts),
            session_mae=math.fsum(means) / len(starts),
            session_mse=math.fsum(means_squared) / len(starts),
            mae=float(np.mean(absolute)),
            mse=float(np.mean(squared)),
        )
    _declared(metrics, source.get("declared_metrics"))
    rank_ic, pearson_ic, direction, nonzero = [], [], [], []
    for begin, count in zip(starts, counts, strict=True):
        selected = order[begin : begin + count]
        observed, predicted = target[selected], values["prediction"][selected]
        rank_ic.append(_correlation(observed, predicted, rank=True))
        pearson_ic.append(_correlation(observed, predicted))
        eligible = observed != 0
        direction.append(
            float(np.mean(np.sign(observed[eligible]) == np.sign(predicted[eligible])))
            if eligible.any()
            else None
        )
        nonzero.append(float(np.mean(predicted != 0)))
    daily.update(
        rank_ic=pa.array(rank_ic, pa.float64()),
        pearson_ic=pa.array(pearson_ic, pa.float64()),
        direction_accuracy=pa.array(direction, pa.float64()),
        nonzero_prediction_fraction=nonzero,
    )
    return dict(
        metrics=metrics,
        sessions=pa.table(daily),
        cohort=current_cohort,
        prediction=values["prediction"],
        parent=values["parent"],
        diagnostics=dict(
            assets=len(set(assets)),
            rank_ic_valid_sessions=sum(value is not None for value in rank_ic),
            rank_ic_undefined_sessions=sum(value is None for value in rank_ic),
            mean_session_rank_ic=_mean_defined(rank_ic),
            mean_session_pearson_ic=_mean_defined(pearson_ic),
            mean_session_direction_accuracy=_mean_defined(direction),
            mean_session_nonzero_prediction_fraction=_mean_defined(nonzero),
            prediction_min=float(np.min(values["prediction"])),
            prediction_max=float(np.max(values["prediction"])),
            primary_equals_initial_policy=(
                bool(np.array_equal(values["prediction"], values["initial_policy"]))
                if "initial_policy" in values
                else None
            ),
            center_parent_max_absolute_difference=(
                float(np.max(np.abs(values["center"] - values["parent"])))
                if "center" in values
                else None
            ),
        ),
    )


def _effects(effects):
    raw = np.asarray(effects)
    _require(
        raw.ndim == 2 and min(raw.shape) > 0 and raw.size <= 4_000_000 and raw.dtype.kind in "fiu",
        "Los efectos deben ser una matriz real y acotada",
    )
    if raw.dtype.kind in "iu":
        _require(
            np.all((raw >= -(2**53)) & (raw <= 2**53)), "Los enteros pierden precisión en float64"
        )
    values = raw.astype(np.float64, copy=False)
    _require(np.isfinite(values).all(), "Los efectos deben ser finitos")
    return values


def resampled_means(effects, starts, block_length):
    """Aplicar los mismos bloques a todas las comparaciones mediante una matriz de pesos."""
    values, begins = _effects(effects), np.asarray(starts)
    _require(
        values.ndim == 2 and min(values.shape) > 0 and np.isfinite(values).all(),
        "Efectos inválidos",
    )
    n, comparisons = values.shape
    _require(type(block_length) is int and 1 <= block_length <= n, "Longitud de bloque inválida")
    _require(
        begins.ndim == 2
        and np.issubdtype(begins.dtype, np.integer)
        and 0 < len(begins) <= 10_000
        and begins.shape[1] == math.ceil(n / block_length)
        and np.all((begins >= 0) & (begins <= n - block_length))
        and len(begins) * (n + comparisons) <= 10_000_000,
        "El remuestreo no forma bloques válidos o supera su presupuesto",
    )
    indexes = (begins[:, :, None] + np.arange(block_length)).reshape(len(begins), -1)[:, :n]
    combined = indexes + np.arange(len(begins))[:, None] * n
    weights = np.bincount(combined.ravel(), minlength=len(begins) * n).reshape(len(begins), n)
    with np.errstate(over="raise", invalid="raise"):
        result = (weights.astype(np.float64) / n) @ values
    _require(np.isfinite(result).all(), "El remuestreo ha desbordado float64")
    return result


def block_intervals(effects, *, block_length=5, repetitions=2000, seed=42):
    """Intervalos exploratorios por sesión, sin corrección de búsqueda o multiplicidad."""
    values = _effects(effects)
    _require(
        values.ndim == 2
        and min(values.shape) > 0
        and values.size <= 4_000_000
        and np.isfinite(values).all(),
        "Los efectos deben ser una matriz finita y acotada",
    )
    for name, value, lower, upper in (
        ("bloque", block_length, 1, 100_000),
        ("repeticiones", repetitions, 2, 10_000),
        ("semilla", seed, 0, 2**63 - 1),
    ):
        _require(type(value) is int and lower <= value <= upper, f"Parámetro {name} inválido")
    n, comparisons = values.shape
    _require(repetitions * (n + comparisons) <= 10_000_000, "El bootstrap supera el presupuesto")
    try:
        with np.errstate(over="raise", invalid="raise"):
            estimate = np.mean(values, axis=0)
    except FloatingPointError as error:
        raise ValueError("La estimación no cabe en float64") from error
    _require(np.isfinite(estimate).all(), "La estimación no es finita")
    result = dict(
        estimate=estimate.tolist(),
        lower=None,
        upper=None,
        reason=None,
        sessions=n,
        repetitions=repetitions,
        seed=seed,
        block_length=block_length,
    )
    if block_length >= n:
        result["reason"] = "Se necesitan más sesiones que la longitud del bloque"
        return result
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, comparisons), dtype=np.float64)
    for offset in range(0, repetitions, 256):
        size = min(256, repetitions - offset)
        starts = rng.integers(0, n - block_length + 1, size=(size, math.ceil(n / block_length)))
        draws[offset : offset + size] = resampled_means(values, starts, block_length)
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    result.update(lower=lower.tolist(), upper=upper.tolist())
    return result
