"""Error, dirección, orden transversal, cuantiles y abstención calculados por sesión.

Todas las métricas se calculan primero dentro de cada sesión (mercado e instante)
y después se promedian entre sesiones con la ponderación declarada. Un valor no
definido se cuenta con su motivo y nunca se sustituye por cero en el promedio.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pyarrow as pa

from mars_titan.evaluation.forecast_panel import (
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
}


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

    def _values(self, metric):
        if metric == "mae":
            return self.mae, np.ones(len(self.mae), dtype=bool)
        if metric == "mse":
            return self.mse, np.ones(len(self.mse), dtype=bool)
        if metric == "direction_accuracy":
            return _ratio(self.direction_hits, self.direction_eligible)
        if metric == "rank_ic":
            return self.rank_ic, self.rank_ic_status == 0
        _require(self.pinball is not None, "El modelo no emite cuantiles para la pérdida pinball")
        return self.pinball.mean(axis=1), np.ones(len(self.mae), dtype=bool)

    def series(self, metric):
        """Serie por sesión para comparaciones emparejadas con bloques temporales."""
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
        return dict(
            sessions=int(
                len(self.mae) if market is None else np.sum(self.session_market == market)
            ),
            mae=mae,
            mse=mse,
            rmse=None if mse is None else float(np.sqrt(mse)),
            direction_accuracy=self._mean(accuracy, eligible, weighting, market),
            conditional_direction_accuracy=self._mean(conditional, called, weighting, market),
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
                    sign_commitment=self._mean(commitment, everywhere, weighting),
                    committed_sign_error=self._mean(error, judged, weighting),
                    committed_sign_error_sessions=int(np.sum(judged)),
                    committed_rows=int(np.sum(self.commitments[:, index])),
                    committed_wrong_rows=int(np.sum(self.committed_wrong[:, index])),
                )
            )
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
            point_equals_median=self.point_equals_median,
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
            rank_ic=pa.array(self.rank_ic, mask=self.rank_ic_status != 0),
            rank_ic_status=pa.array(np.asarray(RANK_IC_STATUS)[self.rank_ic_status]),
        )
        for j, level in enumerate(self.levels or ()):
            columns[f"pinball_{level}"] = self.pinball[:, j]
            columns[f"below_{level}"] = self.below[:, j]
        for index, (nominal, _, _) in enumerate(self.intervals):
            columns[f"coverage_{nominal:g}"] = self.coverage[:, index]
            columns[f"width_{nominal:g}"] = self.width[:, index]
        return pa.table(columns)


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
