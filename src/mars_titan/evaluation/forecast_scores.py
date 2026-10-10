"""Error, dirección, orden transversal, cuantiles y abstención calculados por sesión.

Todas las métricas se calculan primero dentro de cada sesión (mercado e instante)
y después se promedian entre sesiones con la ponderación declarada. Un valor no
definido se cuenta con su motivo y nunca se sustituye por cero en el promedio.

Con cuantiles se añaden la puntuación de cada intervalo central y la probabilidad
implícita de que el residuo sea positivo, con su Brier por sesión y los recuentos
por intervalo de probabilidad que permiten la curva de fiabilidad y el ECE.
"""

import hashlib
import math
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import pyarrow as pa

from mars_titan.evaluation.forecast_panel import (
    DAY_MICROSECONDS,
    ForecastPanel,
    SessionSeries,
    central_intervals,
    weighted_session_mean,
)

DIRECTION_CONVENTION = "zero_target_excluded_zero_prediction_is_abstention_and_miss"
RANK_IC_STATUS = ("defined", "insufficient_assets", "constant_target", "constant_prediction")
DEFAULT_COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_SERIES = {
    "mae": True,
    "mse": True,
    "direction_accuracy": False,
    "rank_ic": False,
    "pinball": True,
    "up_precision": False,
    "down_precision": False,
    "sign_brier": True,
}
COVERAGE_ERROR = "coverage_error@"
INTERVAL_SCORE = "interval_score@"
# Métricas que solo existen si el modelo emite cuantiles.
QUANTILE_SERIES = ("pinball", "sign_brier")
SIGN_PROBABILITY_RULE = "piecewise_linear_cdf_at_zero_flat_beyond_extreme_levels_v1"
SIGN_BINS = 10
# Estadísticos con una fila por sesión. El resto describe el panel completo.
_PER_SESSION = (
    "session_market",
    "session_time",
    "samples",
    "mae",
    "mse",
    "direction_eligible",
    "direction_calls",
    "direction_hits",
    "zero_targets",
    "zero_predictions",
    "positive_targets",
    "negative_targets",
    "up_calls",
    "down_calls",
    "up_hits",
    "down_hits",
    "rank_ic",
    "rank_ic_status",
    "pinball",
    "below",
    "coverage",
    "width",
    "commitments",
    "committed_judged",
    "committed_wrong",
    "interval_score",
    "sign_brier",
    "sign_bin_rows",
    "sign_bin_probability",
    "sign_bin_up",
)


def is_loss_series(metric):
    """Indica si la serie por sesión de ``metric`` es una pérdida, no negativa y mejor si baja."""
    if not isinstance(metric, str):
        return False
    if metric.startswith(INTERVAL_SCORE):
        return True
    return _SERIES.get(metric, False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(panel, mask):
    return np.bincount(panel.session[mask], minlength=panel.sessions)


def _mean(panel, values):
    return np.bincount(panel.session, weights=values, minlength=panel.sessions) / (
        panel.session_samples
    )


def _ratio(numerator, denominator):
    defined = denominator > 0
    return np.where(defined, numerator / np.maximum(denominator, 1), np.nan), defined


def _percent(value):
    return None if value is None else 100 * value


def _day_index(times):
    _, period = np.unique(np.floor_divide(times, DAY_MICROSECONDS), return_inverse=True)
    return period.astype(np.int64)


def _within_session_order(panel, values):
    """Ordenar cada sesión por valor con desempate por la posición canónica.

    Equivale a ``np.lexsort((values, panel.session))``. Dos ordenaciones estables
    permiten que la segunda use radix sort sobre uint16, medida un 30 % más rápida
    con 2 millones de filas y 5.000 sesiones.
    """
    first = np.argsort(values, kind="stable")
    sessions = panel.session[first]
    if panel.sessions <= 1 << 16:
        sessions = sessions.astype(np.uint16)
    order = first[np.argsort(sessions, kind="stable")]
    position = np.arange(panel.rows) - panel.session_starts[panel.session]
    return order, position


def _average_ranks(panel, values):
    """Rangos medios dentro de cada sesión, como scipy.stats.rankdata(method="average")."""
    order, position = _within_session_order(panel, values)
    ordered = values[order]
    run_start = np.ones(panel.rows, dtype=bool)
    run_start[1:] = (panel.session[1:] != panel.session[:-1]) | (ordered[1:] != ordered[:-1])
    run = np.cumsum(run_start) - 1
    average = position[run_start] + (np.bincount(run) + 1) / 2
    ranks = np.empty(panel.rows, dtype=np.float64)
    ranks[order] = average[run]
    return ranks


def _rank_ic(panel, min_assets):
    center = (panel.session_samples[panel.session] + 1) / 2
    target = _average_ranks(panel, panel.target) - center
    predicted = _average_ranks(panel, panel.prediction) - center
    covariance = np.bincount(panel.session, weights=target * predicted, minlength=panel.sessions)
    target_variance = np.bincount(panel.session, weights=target**2, minlength=panel.sessions)
    predicted_variance = np.bincount(panel.session, weights=predicted**2, minlength=panel.sessions)
    status = np.select(
        [panel.session_samples < min_assets, target_variance == 0, predicted_variance == 0],
        [1, 2, 3],
        0,
    ).astype(np.int8)
    defined = status == 0
    denominator = np.sqrt(np.where(defined, target_variance * predicted_variance, 1.0))
    values = np.where(defined, np.clip(covariance / denominator, -1.0, 1.0), np.nan)
    return values, status


@dataclass(frozen=True, eq=False)
class SessionScores:
    """Estadísticos por sesión de un panel, suficientes para resumir y comparar."""

    cohort_sha256: str
    markets: tuple
    session_market: np.ndarray
    session_time: np.ndarray
    session_period: np.ndarray
    samples: np.ndarray
    mae: np.ndarray
    mse: np.ndarray
    direction_eligible: np.ndarray
    direction_calls: np.ndarray
    direction_hits: np.ndarray
    zero_targets: np.ndarray
    zero_predictions: np.ndarray
    positive_targets: np.ndarray
    negative_targets: np.ndarray
    up_calls: np.ndarray
    down_calls: np.ndarray
    up_hits: np.ndarray
    down_hits: np.ndarray
    rank_ic: np.ndarray
    rank_ic_status: np.ndarray
    rank_ic_min_assets: int
    levels: tuple | None
    pinball: np.ndarray | None
    below: np.ndarray | None
    intervals: tuple
    coverage: np.ndarray | None
    width: np.ndarray | None
    commitments: np.ndarray | None
    committed_judged: np.ndarray | None
    committed_wrong: np.ndarray | None
    point_equals_median: bool | None
    interval_score: np.ndarray | None = None
    sign_brier: np.ndarray | None = None
    sign_bin_rows: np.ndarray | None = None
    sign_bin_probability: np.ndarray | None = None
    sign_bin_up: np.ndarray | None = None

    def _values(self, metric):
        if metric == "mae":
            return self.mae, np.ones(len(self.mae), dtype=bool)
        if metric == "mse":
            return self.mse, np.ones(len(self.mse), dtype=bool)
        if metric == "direction_accuracy":
            return _ratio(self.direction_hits, self.direction_eligible)
        if metric == "rank_ic":
            return self.rank_ic, self.rank_ic_status == 0
        if metric == "up_precision":
            return _ratio(self.up_hits, self.up_calls)
        if metric == "down_precision":
            return _ratio(self.down_hits, self.down_calls)
        if metric == "sign_brier":
            _require(self.sign_brier is not None, "El modelo no emite cuantiles para el Brier")
            return self.sign_brier, self.direction_eligible > 0
        _require(self.pinball is not None, "El modelo no emite cuantiles para la pérdida pinball")
        return self.pinball.mean(axis=1), np.ones(len(self.mae), dtype=bool)

    def _interval_index(self, nominal):
        for index, (declared, _, _) in enumerate(self.intervals):
            if math.isclose(declared, nominal, abs_tol=1e-12):
                return index
        raise ValueError("El modelo no emite el intervalo central solicitado")

    def _nominal(self, metric, prefix):
        try:
            return float(metric.removeprefix(prefix))
        except ValueError as error:
            raise ValueError("La cobertura nominal no es un número") from error

    def _coverage_error(self, metric):
        """Cobertura observada menos nominal por sesión. Negativa indica sobreconfianza."""
        _require(self.coverage is not None, "El modelo no emite intervalos para su cobertura")
        index = self._interval_index(self._nominal(metric, COVERAGE_ERROR))
        return self.coverage[:, index] - self.intervals[index][0]

    def series(self, metric):
        """Serie por sesión para comparaciones emparejadas con bloques temporales.

        ``coverage_error@0.8`` da la cobertura del intervalo central del 80 % menos
        0,8 en cada sesión. Su media con ``level`` mide la sobreconfianza.
        ``interval_score@0.8`` da la puntuación media del mismo intervalo, una pérdida.
        """
        prefixed = {COVERAGE_ERROR: False, INTERVAL_SCORE: True}
        prefix = next(
            (key for key in prefixed if isinstance(metric, str) and metric.startswith(key)), None
        )
        if prefix is not None:
            if prefix == COVERAGE_ERROR:
                values = self._coverage_error(metric)
            else:
                _require(self.interval_score is not None, "El modelo no emite intervalos")
                values = self.interval_score[:, self._interval_index(self._nominal(metric, prefix))]
            return SessionSeries(
                metric,
                prefixed[prefix],
                self.cohort_sha256,
                self.markets,
                self.session_market,
                self.session_period,
                values,
                np.ones(len(values), dtype=bool),
            )
        _require(metric in _SERIES, "Métrica por sesión no admitida")
        values, defined = self._values(metric)
        return SessionSeries(
            metric,
            _SERIES[metric],
            self.cohort_sha256,
            self.markets,
            self.session_market,
            self.session_period,
            np.where(defined, values, 0.0),
            defined,
        )

    def _mean(self, values, defined, weighting, market=None):
        selected = defined if market is None else defined & (self.session_market == market)
        return weighted_session_mean(
            values, selected, self.session_market, len(self.markets), weighting
        )

    def _point(self, weighting, market=None):
        everywhere = np.ones(len(self.mae), dtype=bool)
        mae = self._mean(self.mae, everywhere, weighting, market)
        mse = self._mean(self.mse, everywhere, weighting, market)
        accuracy, eligible = _ratio(self.direction_hits, self.direction_eligible)
        conditional, called = _ratio(self.direction_hits, self.direction_calls)
        defined = self.rank_ic_status == 0
        selected = defined if market is None else defined & (self.session_market == market)
        direction = self._mean(accuracy, eligible, weighting, market)
        conditional = self._mean(conditional, called, weighting, market)
        signs = {}
        for side, calls, hits, targets in (
            ("up", self.up_calls, self.up_hits, self.positive_targets),
            ("down", self.down_calls, self.down_hits, self.negative_targets),
        ):
            precision, claimed = _ratio(hits, calls)
            recall, present = _ratio(hits, targets)
            signs[f"{side}_precision"] = self._mean(precision, claimed, weighting, market)
            signs[f"{side}_recall"] = self._mean(recall, present, weighting, market)
        return dict(
            sessions=int(
                len(self.mae) if market is None else np.sum(self.session_market == market)
            ),
            mae=mae,
            mse=mse,
            rmse=None if mse is None else float(np.sqrt(mse)),
            direction_accuracy=direction,
            conditional_direction_accuracy=conditional,
            direction_accuracy_percent=_percent(direction),
            conditional_direction_accuracy_percent=_percent(conditional),
            **signs,
            rank_ic=self._mean(self.rank_ic, defined, weighting, market),
            rank_ic_sessions=int(np.sum(selected)),
        )

    def _quantiles(self, weighting):
        everywhere = np.ones(len(self.mae), dtype=bool)
        pinball = [
            self._mean(self.pinball[:, j], everywhere, weighting) for j in range(len(self.levels))
        ]
        below = [
            self._mean(self.below[:, j], everywhere, weighting) for j in range(len(self.levels))
        ]
        intervals = []
        for index, (nominal, lower, upper) in enumerate(self.intervals):
            commitment = self.commitments[:, index] / self.samples
            error, judged = _ratio(self.committed_wrong[:, index], self.committed_judged[:, index])
            coverage = self._mean(self.coverage[:, index], everywhere, weighting)
            intervals.append(
                dict(
                    nominal=nominal,
                    lower_level=lower,
                    upper_level=upper,
                    coverage=coverage,
                    coverage_gap=coverage - nominal,
                    width=self._mean(self.width[:, index], everywhere, weighting),
                    interval_score=self._mean(self.interval_score[:, index], everywhere, weighting),
                    sign_commitment=self._mean(commitment, everywhere, weighting),
                    committed_sign_error=self._mean(error, judged, weighting),
                    committed_sign_error_sessions=int(np.sum(judged)),
                    committed_rows=int(np.sum(self.commitments[:, index])),
                    committed_wrong_rows=int(np.sum(self.committed_wrong[:, index])),
                )
            )
        level_errors = [abs(value - level) for level, value in zip(self.levels, below, strict=True)]
        interval_errors = [abs(row["coverage_gap"]) for row in intervals]
        return dict(
            levels=list(self.levels),
            pinball={str(level): value for level, value in zip(self.levels, pinball, strict=True)},
            mean_pinball=float(np.mean(pinball)),
            level_frequency={
                str(level): value for level, value in zip(self.levels, below, strict=True)
            },
            level_frequency_gap={
                str(level): value - level for level, value in zip(self.levels, below, strict=True)
            },
            intervals=intervals,
            calibration=dict(
                level_mean_absolute_error=math.fsum(level_errors) / len(level_errors),
                level_max_absolute_error=max(level_errors),
                interval_mean_absolute_error=(
                    math.fsum(interval_errors) / len(interval_errors) if intervals else None
                ),
                undercovered_intervals=[
                    row["nominal"] for row in intervals if row["coverage"] < row["nominal"]
                ],
            ),
            point_equals_median=self.point_equals_median,
            sign_probability=self._sign_probability(weighting),
        )

    def _sign_probability(self, weighting):
        """Brier por sesión, curva de fiabilidad y ECE de la probabilidad implícita de subida.

        El ECE suma, por intervalo de probabilidad, la diferencia absoluta entre subidas
        observadas y probabilidad emitida, y divide por las filas con objetivo no nulo de
        todas las sesiones. Cada fila pesa lo mismo, también en el ECE de varias sesiones.
        """
        rows, probability, up = (
            values.sum(axis=0)
            for values in (self.sign_bin_rows, self.sign_bin_probability, self.sign_bin_up)
        )
        total = int(rows.sum())
        curve = []
        for index in range(SIGN_BINS):
            count = int(rows[index])
            curve.append(
                dict(
                    lower=index / SIGN_BINS,
                    upper=(index + 1) / SIGN_BINS,
                    rows=count,
                    mean_probability=float(probability[index] / count) if count else None,
                    observed_up_frequency=float(up[index] / count) if count else None,
                )
            )
        return dict(
            rule=SIGN_PROBABILITY_RULE,
            event="residual_target_above_zero",
            eligible_rows=total,
            brier=self._mean(self.sign_brier, self.direction_eligible > 0, weighting),
            ece=expected_calibration_error(rows, probability, up),
            ece_weighting="eligible_rows_pooled_over_sessions",
            reliability=curve,
        )

    def summary(self, *, market_weighting="session"):
        """Resumen declarado, con denominadores y métricas no definidas explícitas."""
        rows = int(np.sum(self.samples))
        status = {
            name: int(np.sum(self.rank_ic_status == code))
            for code, name in enumerate(RANK_IC_STATUS)
        }
        defined_ic = self.rank_ic[self.rank_ic_status == 0]
        eligible = int(np.sum(self.direction_eligible))
        calls = int(np.sum(self.direction_calls))
        return dict(
            schema_version=1,
            kind="forecast_session_scores",
            cohort_sha256=self.cohort_sha256,
            market_weighting=market_weighting,
            rows=rows,
            sessions=len(self.mae),
            periods=int(len(np.unique(self.session_period))),
            point=self._point(market_weighting),
            by_market={
                name: self._point(market_weighting, code)
                for code, name in enumerate(self.markets)
                if np.any(self.session_market == code)
            },
            row_weighted=dict(
                mae=float(np.dot(self.samples, self.mae) / rows),
                mse=float(np.dot(self.samples, self.mse) / rows),
            ),
            direction=dict(
                convention=DIRECTION_CONVENTION,
                eligible_rows=eligible,
                zero_target_rows=int(np.sum(self.zero_targets)),
                zero_prediction_rows=int(np.sum(self.zero_predictions)),
                calls=calls,
                hits=int(np.sum(self.direction_hits)),
                call_coverage=calls / eligible if eligible else None,
                undefined_sessions=int(np.sum(self.direction_eligible == 0)),
                positive_target_rows=int(np.sum(self.positive_targets)),
                negative_target_rows=int(np.sum(self.negative_targets)),
                up_calls=int(np.sum(self.up_calls)),
                down_calls=int(np.sum(self.down_calls)),
                up_hits=int(np.sum(self.up_hits)),
                down_hits=int(np.sum(self.down_hits)),
            ),
            rank_ic=dict(
                method="spearman_average_ranks_within_session",
                min_assets=self.rank_ic_min_assets,
                sessions=status,
                session_std=float(np.std(defined_ic, ddof=1)) if len(defined_ic) > 1 else None,
                positive_session_fraction=(
                    float(np.mean(defined_ic > 0)) if len(defined_ic) else None
                ),
            ),
            quantiles=None if self.levels is None else self._quantiles(market_weighting),
            quantiles_reason=None if self.levels is not None else "El modelo no emite cuantiles",
        )

    def to_table(self):
        """Tabla por sesión para conservar estadísticos suficientes con valores nulos explícitos."""
        accuracy, eligible = _ratio(self.direction_hits, self.direction_eligible)
        columns = dict(
            market=pa.array(np.asarray(self.markets)[self.session_market]),
            prediction_at=pa.array(self.session_time, type=pa.timestamp("us", tz="UTC")),
            samples=self.samples,
            mae=self.mae,
            mse=self.mse,
            direction_eligible=self.direction_eligible,
            direction_calls=self.direction_calls,
            direction_hits=self.direction_hits,
            direction_accuracy=pa.array(accuracy, mask=~eligible),
            positive_targets=self.positive_targets,
            negative_targets=self.negative_targets,
            up_calls=self.up_calls,
            down_calls=self.down_calls,
            up_hits=self.up_hits,
            down_hits=self.down_hits,
            rank_ic=pa.array(self.rank_ic, mask=self.rank_ic_status != 0),
            rank_ic_status=pa.array(np.asarray(RANK_IC_STATUS)[self.rank_ic_status]),
        )
        for j, level in enumerate(self.levels or ()):
            columns[f"pinball_{level}"] = self.pinball[:, j]
            columns[f"below_{level}"] = self.below[:, j]
        for index, (nominal, _, _) in enumerate(self.intervals):
            columns[f"coverage_{nominal:g}"] = self.coverage[:, index]
            columns[f"width_{nominal:g}"] = self.width[:, index]
            columns[f"interval_score_{nominal:g}"] = self.interval_score[:, index]
        if self.sign_brier is not None:
            columns["sign_brier"] = pa.array(self.sign_brier, mask=self.direction_eligible == 0)
            for name in ("sign_bin_rows", "sign_bin_probability", "sign_bin_up"):
                values = getattr(self, name)
                offsets = pa.array(np.arange(0, values.size + 1, SIGN_BINS), pa.int32())
                columns[name] = pa.ListArray.from_arrays(offsets, values.ravel())
        return pa.table(columns)

    def _take(self, index, cohort_sha256):
        """Copiar las sesiones indicadas y renumerar sus días UTC para el remuestreo."""
        arrays = {
            name: None if getattr(self, name) is None else getattr(self, name)[index]
            for name in _PER_SESSION
        }
        return replace(
            self,
            cohort_sha256=cohort_sha256,
            session_period=_day_index(arrays["session_time"]),
            **arrays,
        )

    def select_sessions(self, mask, *, label):
        """Conservar un subconjunto de sesiones, por ejemplo un mercado, sin recalcular filas.

        Las métricas son medias de estadísticos por sesión, así que el subconjunto da el
        mismo valor que puntuar solo las filas de esas sesiones.
        """
        mask = np.asarray(mask)
        _require(
            mask.dtype == np.bool_ and mask.shape == self.session_market.shape and mask.any(),
            "La selección de sesiones debe ser una máscara alineada y no vacía",
        )
        _require(isinstance(label, str) and label, "La selección necesita una etiqueta")
        digest = hashlib.sha256(b"mars-titan-session-selection-v1\0")
        digest.update(f"{self.cohort_sha256}\0{label}\0".encode())
        digest.update(np.packbits(mask).tobytes())
        return self._take(np.flatnonzero(mask), digest.hexdigest())

    @classmethod
    def concatenate(cls, items):
        """Unir ventanas disjuntas en orden canónico (mercado, instante).

        La huella combina las de cada parte sin depender de su orden. Una sesión que
        aparezca en dos partes se rechaza, porque contaría dos veces la misma decisión.
        """
        _require(len(items) >= 1, "No hay puntuaciones que unir")
        first = items[0]
        _require(
            all(isinstance(item, cls) for item in items)
            and all(
                item.markets == first.markets
                and item.rank_ic_min_assets == first.rank_ic_min_assets
                and item.levels == first.levels
                and item.intervals == first.intervals
                for item in items
            ),
            "Solo se unen puntuaciones con mercados, mínimo de activos y cuantiles comunes",
        )
        joined = {}
        for name in _PER_SESSION:
            values = [getattr(item, name) for item in items]
            joined[name] = None if values[0] is None else np.concatenate(values)
        order = np.lexsort((joined["session_time"], joined["session_market"]))
        market, time = joined["session_market"][order], joined["session_time"][order]
        repeated = (market[1:] == market[:-1]) & (time[1:] == time[:-1])
        _require(not repeated.any(), "Una sesión aparece en dos ventanas")
        digest = hashlib.sha256(b"mars-titan-session-concatenation-v1\0")
        for value in sorted(item.cohort_sha256 for item in items):
            digest.update(value.encode())
        medians = [item.point_equals_median for item in items]
        whole = replace(
            first,
            point_equals_median=None if None in medians else all(medians),
            **joined,
        )
        return whole._take(order, digest.hexdigest())


def score_sessions(panel, *, rank_ic_min_assets=3):
    """Calcular todas las métricas por sesión de un panel con operaciones vectoriales."""
    _require(isinstance(panel, ForecastPanel), "Se necesita un ForecastPanel validado")
    _require(
        type(rank_ic_min_assets) is int and 3 <= rank_ic_min_assets <= 100_000,
        "El mínimo de activos para Rank IC debe ser un entero de al menos 3",
    )
    error = panel.prediction - panel.target
    eligible = panel.target != 0
    calls = eligible & (panel.prediction != 0)
    hits = calls & (np.sign(panel.prediction) == np.sign(panel.target))
    up_calls, down_calls = eligible & (panel.prediction > 0), eligible & (panel.prediction < 0)
    rank_ic, status = _rank_ic(panel, rank_ic_min_assets)
    quantile_fields = dict(
        pinball=None,
        below=None,
        intervals=(),
        coverage=None,
        width=None,
        commitments=None,
        committed_judged=None,
        committed_wrong=None,
        point_equals_median=None,
        interval_score=None,
        sign_brier=None,
        sign_bin_rows=None,
        sign_bin_probability=None,
        sign_bin_up=None,
    )
    if panel.levels is not None:
        quantile_fields.update(_quantile_scores(panel))
    return SessionScores(
        cohort_sha256=panel.cohort_sha256,
        markets=panel.markets,
        session_market=panel.session_market,
        session_time=panel.session_time,
        session_period=panel.session_period,
        samples=panel.session_samples,
        mae=_mean(panel, np.abs(error)),
        mse=_mean(panel, np.square(error)),
        direction_eligible=_count(panel, eligible),
        direction_calls=_count(panel, calls),
        direction_hits=_count(panel, hits),
        zero_targets=_count(panel, ~eligible),
        zero_predictions=_count(panel, panel.prediction == 0),
        positive_targets=_count(panel, panel.target > 0),
        negative_targets=_count(panel, panel.target < 0),
        up_calls=_count(panel, up_calls),
        down_calls=_count(panel, down_calls),
        up_hits=_count(panel, up_calls & (panel.target > 0)),
        down_hits=_count(panel, down_calls & (panel.target < 0)),
        rank_ic=rank_ic,
        rank_ic_status=status,
        rank_ic_min_assets=rank_ic_min_assets,
        levels=panel.levels,
        **quantile_fields,
    )


def _quantile_scores(panel):
    quantiles, target = panel.quantiles, panel.target
    pinball = np.empty((panel.sessions, len(panel.levels)))
    below = np.empty_like(pinball)
    for j, level in enumerate(panel.levels):
        residual = target - quantiles[:, j]
        pinball[:, j] = _mean(panel, np.maximum(level * residual, (level - 1) * residual))
        below[:, j] = _count(panel, target <= quantiles[:, j]) / panel.session_samples
    pairs = central_intervals(panel.levels)
    shape = (panel.sessions, len(pairs))
    fields = dict(
        coverage=np.empty(shape),
        width=np.empty(shape),
        interval_score=np.empty(shape),
        commitments=np.empty(shape, dtype=np.int64),
        committed_judged=np.empty(shape, dtype=np.int64),
        committed_wrong=np.empty(shape, dtype=np.int64),
    )
    intervals = []
    for index, (lower, upper) in enumerate(pairs):
        low, high = quantiles[:, lower], quantiles[:, upper]
        intervals.append(
            (
                round(panel.levels[upper] - panel.levels[lower], 12),
                panel.levels[lower],
                panel.levels[upper],
            )
        )
        fields["coverage"][:, index] = _count(panel, (low <= target) & (target <= high)) / (
            panel.session_samples
        )
        fields["width"][:, index] = _mean(panel, high - low)
        # Gneiting y Raftery (2007): anchura más 2/alfa por la distancia al extremo superado.
        alpha = 2 * panel.levels[lower]
        excess = np.maximum(low - target, 0.0) + np.maximum(target - high, 0.0)
        fields["interval_score"][:, index] = _mean(panel, high - low + (2 / alpha) * excess)
        # Un intervalo que excluye el cero afirma un signo. Se cuenta si esa afirmación falla.
        claim = np.where(low > 0, 1.0, np.where(high < 0, -1.0, 0.0))
        judged = (claim != 0) & (target != 0)
        fields["commitments"][:, index] = _count(panel, claim != 0)
        fields["committed_judged"][:, index] = _count(panel, judged)
        fields["committed_wrong"][:, index] = _count(panel, judged & (np.sign(target) != claim))
    median = panel.levels.index(0.5) if 0.5 in panel.levels else None
    return dict(
        pinball=pinball,
        below=below,
        intervals=tuple(intervals),
        point_equals_median=(
            None if median is None else bool(np.array_equal(panel.prediction, quantiles[:, median]))
        ),
        **fields,
        **_sign_scores(panel),
    )


def implied_up_probability(quantiles, levels):
    """Probabilidad implícita de un residuo positivo, 1 − F(0), con la regla declarada.

    F es lineal a trozos entre los puntos (q_j, tau_j) de cada fila. Por debajo del
    primer cuantil vale tau_1 y por encima del último tau_K, es decir, la probabilidad
    menos segura que permiten los niveles emitidos. Con cuantiles iguales a cero se
    toma el último nivel que no supera el cero, como una función de distribución
    continua por la derecha. Los cuantiles deben estar ordenados en cada fila.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    below = np.count_nonzero(quantiles <= 0, axis=1)
    inner = (below > 0) & (below < len(levels))
    left = np.clip(below - 1, 0, len(levels) - 2)
    rows = np.arange(len(quantiles))
    low, high = quantiles[rows, left], quantiles[rows, left + 1]
    step = np.where(inner, high - low, 1.0)
    cdf = levels[left] + (levels[left + 1] - levels[left]) * np.where(inner, -low / step, 0.0)
    cdf = np.where(below == 0, levels[0], np.where(below == len(levels), levels[-1], cdf))
    return 1.0 - cdf


def expected_calibration_error(rows, probability, up):
    """ECE con intervalos fijos: sum_b |subidas_b − probabilidad_b| / filas, por el último eje.

    Los argumentos son recuentos de filas, sumas de probabilidad y subidas por intervalo.
    Devuelve None (escalar) o NaN (matriz) si no hay filas.
    """
    total = np.sum(rows, axis=-1)
    gap = np.sum(np.abs(np.asarray(up, dtype=np.float64) - probability), axis=-1)
    if np.ndim(total) == 0:
        return float(gap / total) if total else None
    return np.where(total > 0, gap / np.maximum(total, 1), np.nan)


def _sign_scores(panel):
    """Brier por sesión y recuentos por intervalo de probabilidad en filas con objetivo no nulo."""
    probability = implied_up_probability(panel.quantiles, panel.levels)
    eligible = panel.target != 0
    up = (panel.target > 0).astype(np.float64)
    weights = np.where(eligible, 1.0, 0.0)
    count = np.bincount(panel.session, weights=weights, minlength=panel.sessions)
    squared = np.bincount(
        panel.session, weights=weights * (probability - up) ** 2, minlength=panel.sessions
    )
    brier = np.where(count > 0, squared / np.maximum(count, 1), 0.0)
    bins = np.minimum((probability * SIGN_BINS).astype(np.int64), SIGN_BINS - 1)
    cell = panel.session * SIGN_BINS + bins
    size = panel.sessions * SIGN_BINS
    shape = (panel.sessions, SIGN_BINS)
    return dict(
        sign_brier=brier,
        sign_bin_rows=np.bincount(cell, weights=weights, minlength=size)
        .astype(np.int64)
        .reshape(shape),
        sign_bin_probability=np.bincount(
            cell, weights=weights * probability, minlength=size
        ).reshape(shape),
        sign_bin_up=np.bincount(cell, weights=weights * up, minlength=size)
        .astype(np.int64)
        .reshape(shape),
    )


@dataclass(frozen=True, eq=False)
class SelectiveRisk:
    """MAE de las filas retenidas al abstenerse en las de mayor anchura de intervalo."""

    cohort_sha256: str
    markets: tuple
    session_market: np.ndarray
    session_period: np.ndarray
    samples: np.ndarray
    nominal: float
    coverages: tuple
    retained: np.ndarray
    mae: np.ndarray
    oracle_mae: np.ndarray
    full_mae: np.ndarray

    def _index(self, coverage):
        _require(coverage in self.coverages, "Cobertura selectiva no calculada")
        return self.coverages.index(coverage)

    def series(self, coverage):
        index = self._index(coverage)
        return SessionSeries(
            f"selective_mae@{coverage:g}",
            True,
            self.cohort_sha256,
            self.markets,
            self.session_market,
            self.session_period,
            self.mae[:, index].copy(),
            np.ones(len(self.samples), dtype=bool),
        )

    def summary(self, *, market_weighting="session"):
        everywhere = np.ones(len(self.samples), dtype=bool)

        def mean(values):
            return weighted_session_mean(
                values, everywhere, self.session_market, len(self.markets), market_weighting
            )

        return dict(
            schema_version=1,
            kind="selective_risk_by_interval_width",
            cohort_sha256=self.cohort_sha256,
            market_weighting=market_weighting,
            score=f"central_interval_width_{self.nominal:g}",
            selection="lowest_width_within_each_session_ties_by_canonical_row",
            full_mae=mean(self.full_mae),
            curve=[
                dict(
                    coverage=coverage,
                    retained_rows=int(np.sum(self.retained[:, index])),
                    retained_fraction=float(np.sum(self.retained[:, index]) / np.sum(self.samples)),
                    mae=mean(self.mae[:, index]),
                    oracle_mae=mean(self.oracle_mae[:, index]),
                )
                for index, coverage in enumerate(self.coverages)
            ],
        )


def _retained(samples, coverage):
    fraction = Fraction(coverage).limit_denominator(1_000_000)
    return np.maximum(1, -(-(fraction.numerator * samples) // fraction.denominator))


def selective_risk(panel, *, nominal=0.8, coverages=DEFAULT_COVERAGES):
    """Curva riesgo-cobertura usando solo la anchura de un intervalo emitido por el modelo.

    En cada sesión se conservan las ceil(c · n) filas de menor anchura. La selección
    usa información disponible en la decisión y conserva el peso de cada sesión.
    El oráculo ordena por el error real y solo sirve como cota inferior descriptiva.
    """
    _require(isinstance(panel, ForecastPanel), "Se necesita un ForecastPanel validado")
    _require(panel.levels is not None, "El modelo no emite cuantiles para abstenerse")
    lower, upper = panel.interval(nominal)
    coverages = tuple(float(value) for value in coverages)
    _require(
        1 <= len(coverages) <= 100
        and all(0 < value <= 1 for value in coverages)
        and all(left < right for left, right in zip(coverages, coverages[1:], strict=False)),
        "Las coberturas deben ser crecientes y estar en (0, 1]",
    )
    absolute = np.abs(panel.prediction - panel.target)
    width = panel.quantiles[:, upper] - panel.quantiles[:, lower]
    by_width, position = _within_session_order(panel, width)
    by_error, _ = _within_session_order(panel, absolute)
    shape = (panel.sessions, len(coverages))
    retained = np.empty(shape, dtype=np.int64)
    mae, oracle = np.empty(shape), np.empty(shape)
    for index, coverage in enumerate(coverages):
        retained[:, index] = _retained(panel.session_samples, coverage)
        keep = position < retained[panel.session, index]
        for target, order in ((mae, by_width), (oracle, by_error)):
            kept = np.bincount(
                panel.session,
                weights=np.where(keep, absolute[order], 0.0),
                minlength=panel.sessions,
            )
            target[:, index] = kept / retained[:, index]
    return SelectiveRisk(
        cohort_sha256=panel.cohort_sha256,
        markets=panel.markets,
        session_market=panel.session_market,
        session_period=panel.session_period,
        samples=panel.session_samples,
        nominal=float(panel.levels[upper] - panel.levels[lower]),
        coverages=coverages,
        retained=retained,
        mae=mae,
        oracle_mae=oracle,
        full_mae=_mean(panel, absolute),
    )
