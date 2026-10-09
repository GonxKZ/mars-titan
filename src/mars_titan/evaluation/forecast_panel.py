"""Contratos por fila y por sesión para comparar predicciones sobre la misma población.

Una fila es una predicción emitida para un activo en un instante de decisión.
Una sesión es el par (mercado, instante de decisión). El periodo de remuestreo
es el día natural UTC de ese instante, de modo que las sesiones de ambos
mercados del mismo día se remuestrean juntas. Las filas se guardan en orden
canónico (mercado declarado, instante, identidad), así que el resultado no
depende del orden de entrada.
"""

import hashlib
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

MAX_ROWS = 5_000_000
MAX_LEVELS = 32
MARKETS = ("US", "CN")
WEIGHTINGS = ("session", "market")
DAY_MICROSECONDS = 86_400_000_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real(values, name, shape):
    raw = np.asarray(values)
    _require(
        raw.shape == shape and raw.dtype.kind in "fiu",
        f"{name} debe ser real con forma {shape}",
    )
    if raw.dtype.kind in "iu":
        _require(
            raw.size == 0
            or (raw.max() <= 2**53 and (raw.dtype.kind == "u" or raw.min() >= -(2**53))),
            f"{name} contiene enteros que pierden precisión en float64",
        )
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = raw.astype(np.float64)
    except FloatingPointError as error:
        raise ValueError(f"{name} desborda float64") from error
    _require(np.isfinite(array).all(), f"{name} contiene NaN o infinitos")
    return array


def _microseconds(values, rows):
    raw = np.asarray(values)
    _require(raw.shape == (rows,), "Los instantes no tienen una fila por predicción")
    if np.issubdtype(raw.dtype, np.datetime64):
        _require(not np.isnat(raw).any(), "Hay instantes de decisión ausentes")
        converted = raw.astype("datetime64[us]")
        _require(
            np.array_equal(converted.astype(raw.dtype), raw),
            "Los instantes pierden precisión al expresarse en microsegundos",
        )
        return converted.astype(np.int64)
    _require(
        raw.dtype.kind == "i" or (raw.dtype.kind == "u" and (raw.size == 0 or raw.max() < 2**63)),
        "Los instantes deben ser datetime64 o enteros de microsegundos UTC",
    )
    return raw.astype(np.int64)


def _market_codes(values, rows, markets):
    raw = np.asarray(values)
    _require(raw.shape == (rows,), "Los mercados no tienen una fila por predicción")
    codes = np.full(rows, -1, dtype=np.int8)
    for code, name in enumerate(markets):
        codes[raw == name] = code
    _require(np.all(codes >= 0), "Hay mercados no declarados")
    return codes


def _identities(values, rows):
    try:
        array = values if isinstance(values, pa.Array | pa.ChunkedArray) else pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
        raise ValueError("Las identidades de fila deben ser textos o enteros") from error
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    _require(len(array) == rows and array.null_count == 0, "Faltan identidades de fila")
    if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        array = array.cast(pa.large_string())
        _require(rows == 0 or pc.min(pc.utf8_length(array)).as_py() > 0, "Una identidad está vacía")
    else:
        _require(pa.types.is_integer(array.type), "Las identidades deben ser textos o enteros")
        try:
            array = array.cast(pa.int64())
        except pa.ArrowInvalid as error:
            raise ValueError("Una identidad entera no cabe en int64") from error
    ranks = pc.rank(array, sort_keys="ascending", tiebreaker="dense").to_numpy()
    _require(rows == 0 or int(ranks.max()) == rows, "Hay identidades de fila duplicadas")
    return array, ranks


def _levels(levels):
    _require(
        isinstance(levels, tuple | list) and 1 <= len(levels) <= MAX_LEVELS,
        "Los niveles de cuantiles deben ser una secuencia acotada",
    )
    values = tuple(float(level) for level in levels)
    _require(
        all(0 < level < 1 for level in values)
        and all(left < right for left, right in zip(values, values[1:], strict=False)),
        "Los niveles deben estar en (0, 1) y ser estrictamente crecientes",
    )
    return values


@dataclass(frozen=True, eq=False)
class ForecastPanel:
    """Predicciones de un modelo, fold y semilla en orden canónico y ya validadas."""

    markets: tuple
    row_id: pa.Array
    market: np.ndarray
    prediction_at: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    levels: tuple | None
    quantiles: np.ndarray | None
    session: np.ndarray
    session_starts: np.ndarray
    session_samples: np.ndarray
    session_market: np.ndarray
    session_time: np.ndarray
    session_period: np.ndarray
    periods: int

    @classmethod
    def from_columns(
        cls,
        row_id,
        market,
        prediction_at,
        target,
        prediction,
        *,
        quantiles=None,
        levels=None,
        markets=MARKETS,
    ):
        """Validar sin corregir, ordenar y localizar sesiones con operaciones vectoriales.

        ``prediction_at`` es datetime64 UTC sin zona o enteros de microsegundos UTC.
        Los cuantiles tienen forma [filas, niveles] y no pueden cruzarse. Un cruce
        se rechaza en lugar de reordenarse, porque ocultaría un fallo del modelo.
        """
        _require(
            isinstance(markets, tuple)
            and 1 <= len(markets) <= 8
            and len(set(markets)) == len(markets)
            and all(isinstance(name, str) and name for name in markets),
            "Los mercados declarados deben ser textos únicos",
        )
        rows = len(target) if np.ndim(target) == 1 else 0
        _require(
            1 <= rows <= MAX_ROWS, "El panel debe ser un vector de filas dentro del presupuesto"
        )
        target_values = _real(target, "El objetivo", (rows,))
        point = _real(prediction, "La predicción puntual", (rows,))
        times = _microseconds(prediction_at, rows)
        codes = _market_codes(market, rows, markets)
        identities, ranks = _identities(row_id, rows)
        if (quantiles is None) != (levels is None):
            raise ValueError("Los cuantiles y sus niveles se declaran juntos")
        if quantiles is not None:
            levels = _levels(levels)
            quantiles = _real(quantiles, "Los cuantiles", (rows, len(levels)))
            crossed = int(np.count_nonzero(np.any(np.diff(quantiles, axis=1) < 0, axis=1)))
            _require(crossed == 0, f"Hay {crossed} filas con cuantiles cruzados")
        order = np.lexsort((ranks, times, codes))
        codes, times = codes[order], times[order]
        change = np.ones(rows, dtype=bool)
        change[1:] = (codes[1:] != codes[:-1]) | (times[1:] != times[:-1])
        starts = np.flatnonzero(change)
        days = np.floor_divide(times[starts], DAY_MICROSECONDS)
        unique_days, session_period = np.unique(days, return_inverse=True)
        arrays = dict(
            market=codes,
            prediction_at=times,
            target=target_values[order],
            prediction=point[order],
            quantiles=None if quantiles is None else np.ascontiguousarray(quantiles[order]),
            session=np.cumsum(change) - 1,
            session_starts=starts,
            session_samples=np.diff(np.r_[starts, rows]),
            session_market=codes[starts],
            session_time=times[starts],
            session_period=session_period.astype(np.int64),
        )
        for value in arrays.values():
            if value is not None:
                value.setflags(write=False)
        return cls(
            markets=markets,
            row_id=identities.take(pa.array(order)),
            levels=levels,
            periods=len(unique_days),
            **arrays,
        )

    @classmethod
    def from_arrow(
        cls, table, *, prediction="prediction", quantile_columns=None, levels=None, markets=MARKETS
    ):
        """Leer el contrato de predicciones: sample_id, market, prediction_at y target."""
        required = {"sample_id", "market", "prediction_at", "target", prediction}
        if quantile_columns is not None:
            required |= set(quantile_columns)
        _require(required <= set(table.column_names), "Faltan columnas del contrato")
        _require(
            table.schema.field("prediction_at").type == pa.timestamp("us", tz="UTC"),
            "Los instantes deben ser timestamp UTC en microsegundos",
        )
        _require(all(table[name].null_count == 0 for name in required), "Hay valores ausentes")
        quantiles = None
        if quantile_columns is not None:
            quantiles = np.column_stack([table[name].to_numpy() for name in quantile_columns])
        return cls.from_columns(
            table["sample_id"],
            table["market"].to_numpy(),
            table["prediction_at"].to_numpy(),
            table["target"].to_numpy(),
            table[prediction].to_numpy(),
            quantiles=quantiles,
            levels=levels,
            markets=markets,
        )

    @property
    def rows(self):
        return len(self.target)

    @property
    def sessions(self):
        return len(self.session_starts)

    @cached_property
    def cohort_sha256(self):
        """Huella de identidades, mercados, instantes y objetivos, sin las predicciones."""
        digest = hashlib.sha256(b"mars-titan-forecast-panel-v1\0")
        digest.update("\0".join(self.markets).encode())
        for array in (self.market, self.prediction_at, self.target):
            digest.update(np.ascontiguousarray(array).tobytes())
        if pa.types.is_integer(self.row_id.type):
            digest.update(self.row_id.to_numpy().tobytes())
        else:
            _, offsets, data = self.row_id.buffers()
            bounds = np.frombuffer(
                offsets, dtype=np.int64, count=self.rows + 1, offset=8 * self.row_id.offset
            )
            digest.update((bounds - bounds[0]).tobytes())
            digest.update(memoryview(data)[bounds[0] : bounds[-1]])
        return digest.hexdigest()

    def interval(self, nominal):
        """Índices (inferior, superior) del intervalo central de nivel nominal declarado."""
        for lower, upper in central_intervals(self.levels or ()):
            if math.isclose(self.levels[upper] - self.levels[lower], nominal, abs_tol=1e-12):
                return lower, upper
        raise ValueError("El modelo no emite el intervalo central solicitado")


def central_intervals(levels):
    """Pares de niveles simétricos (tau, 1 - tau), del más estrecho al más ancho."""
    pairs = []
    for lower, low in enumerate(levels):
        for upper in range(len(levels) - 1, lower, -1):
            if low < 0.5 and math.isclose(low + levels[upper], 1.0, abs_tol=1e-12):
                pairs.append((lower, upper))
    return sorted(pairs, key=lambda pair: levels[pair[1]] - levels[pair[0]])


def weighted_session_mean(values, defined, session_market, markets, weighting):
    """Media entre sesiones con la ponderación declarada y las sesiones no definidas fuera.

    ``session`` da el mismo peso a cada par mercado-instante definido. ``market``
    promedia primero dentro de cada mercado y después da el mismo peso a cada
    mercado con al menos una sesión definida. Devuelve None si no hay ninguna.
    """
    _require(weighting in WEIGHTINGS, "Ponderación entre mercados no declarada")
    if weighting == "session":
        selected = values[defined]
        return math.fsum(selected) / len(selected) if len(selected) else None
    means = []
    for code in range(markets):
        selected = values[defined & (session_market == code)]
        if len(selected):
            means.append(math.fsum(selected) / len(selected))
    return math.fsum(means) / len(means) if means else None


@dataclass(frozen=True, eq=False)
class SessionSeries:
    """Una métrica por sesión con su máscara, lista para comparaciones emparejadas.

    ``loss`` indica que el valor es no negativo y que menor es mejor. Solo en ese
    caso tiene sentido una mejora relativa respecto a una base.
    """

    metric: str
    loss: bool
    cohort_sha256: str
    markets: tuple
    session_market: np.ndarray
    session_period: np.ndarray
    values: np.ndarray
    defined: np.ndarray

    def __post_init__(self):
        sessions = len(self.session_market)
        _require(sessions >= 1, "La serie no contiene sesiones")
        for array, dtype in (
            (self.session_market, "i"),
            (self.session_period, "i"),
            (self.values, "f"),
            (self.defined, "b"),
        ):
            _require(
                isinstance(array, np.ndarray)
                and array.shape == (sessions,)
                and array.dtype.kind == dtype,
                "La serie por sesión no es coherente",
            )
        _require(np.isfinite(self.values).all(), "La serie por sesión contiene NaN o infinitos")
        _require(
            np.all(self.values[~self.defined] == 0), "Las sesiones no definidas deben valer cero"
        )
        _require(not self.loss or np.all(self.values >= 0), "Una pérdida no puede ser negativa")

    @classmethod
    def average(cls, items):
        """Promediar semillas de un mismo modelo sesión a sesión, sin añadir sesiones."""
        _require(len(items) >= 1, "No hay series que promediar")
        first = items[0]
        _require(
            all(
                item.metric == first.metric
                and item.loss == first.loss
                and item.cohort_sha256 == first.cohort_sha256
                for item in items
            ),
            "Solo se promedian semillas de la misma métrica y población",
        )
        defined = np.logical_and.reduce([item.defined for item in items])
        values = np.mean([item.values for item in items], axis=0)
        return cls(
            first.metric,
            first.loss,
            first.cohort_sha256,
            first.markets,
            first.session_market,
            first.session_period,
            np.where(defined, values, 0.0),
            defined,
        )
