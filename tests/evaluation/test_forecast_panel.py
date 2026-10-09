"""Contrato canónico de filas, sesiones, días de remuestreo y series por sesión."""

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.evaluation.forecast_panel import (
    ForecastPanel,
    SessionSeries,
    central_intervals,
    weighted_session_mean,
)

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
DAY = np.datetime64("2023-03-01T00:00:00", "us")


def at(day, hour):
    return DAY + np.timedelta64(day, "D") + np.timedelta64(hour, "h")


def columns(**changes):
    values = dict(
        row_id=np.array(["a", "b", "c", "d", "e"]),
        market=np.array(["US", "US", "CN", "US", "CN"]),
        prediction_at=np.array([at(0, 20), at(0, 20), at(0, 7), at(1, 20), at(1, 7)]),
        target=np.array([0.1, -0.2, 0.3, 0.0, 0.5]),
        prediction=np.array([0.2, -0.1, 0.1, 0.1, 0.4]),
    )
    values.update(changes)
    return values


def panel(**changes):
    return ForecastPanel.from_columns(**columns(**changes))


def test_sessions_separate_markets_and_group_the_same_utc_day_for_resampling():
    result = panel()
    assert result.rows == 5 and result.sessions == 4 and result.periods == 2
    assert result.markets == ("US", "CN")
    # Orden canónico: mercado declarado, instante y luego identidad.
    assert result.row_id.to_pylist() == ["a", "b", "d", "c", "e"]
    assert result.session.tolist() == [0, 0, 1, 2, 3]
    assert result.session_samples.tolist() == [2, 1, 1, 1]
    assert result.session_market.tolist() == [0, 0, 1, 1]
    # US y CN del mismo día UTC comparten periodo aunque sus instantes difieran.
    assert result.session_period.tolist() == [0, 1, 0, 1]


def test_row_order_does_not_change_arrays_or_cohort_fingerprint():
    reference = panel()
    order = np.array([4, 2, 0, 3, 1])
    shuffled = ForecastPanel.from_columns(**{k: v[order] for k, v in columns().items()})
    for name in ("market", "prediction_at", "target", "prediction", "session", "session_period"):
        assert np.array_equal(getattr(reference, name), getattr(shuffled, name))
    assert reference.row_id.equals(shuffled.row_id)
    assert reference.cohort_sha256 == shuffled.cohort_sha256


def test_fingerprint_depends_on_targets_and_identities_but_not_predictions():
    reference = panel().cohort_sha256
    assert panel(prediction=np.zeros(5)).cohort_sha256 == reference
    assert panel(target=np.array([0.1, -0.2, 0.3, 0.0, 0.6])).cohort_sha256 != reference
    assert panel(row_id=np.array(["a", "b", "c", "d", "f"])).cohort_sha256 != reference
    integers = panel(row_id=np.arange(5)).cohort_sha256
    assert integers != reference and integers == panel(row_id=np.arange(5)).cohort_sha256


def test_integer_microseconds_are_equivalent_to_datetime64():
    stamps = columns()["prediction_at"]
    assert (
        panel(prediction_at=stamps.astype(np.int64)).cohort_sha256
        == panel(prediction_at=stamps).cohort_sha256
    )


@pytest.mark.parametrize(
    "changes,message",
    [
        (dict(target=np.array([0.1, np.nan, 0.3, 0.0, 0.5])), "NaN o infinitos"),
        (dict(prediction=np.array([0.1, np.inf, 0.3, 0.0, 0.5])), "NaN o infinitos"),
        (dict(target=np.array([True, False, True, True, False])), "real"),
        (dict(prediction=np.zeros(4)), "real"),
        (dict(row_id=np.array(["a", "b", "c", "d", "a"])), "duplicadas"),
        (dict(row_id=np.array(["a", "b", "", "d", "e"])), "vacía"),
        (dict(row_id=np.array([0.5, 1.5, 2.5, 3.5, 4.5])), "textos o enteros"),
        (dict(row_id=np.array(["a", None, "c", "d", "e"], dtype=object)), "Faltan"),
        (dict(market=np.array(["US", "US", "HK", "US", "CN"])), "no declarados"),
        (
            dict(prediction_at=np.array([at(0, 20)] * 4 + [np.datetime64("NaT", "us")])),
            "ausentes",
        ),
        (
            dict(
                prediction_at=np.array([at(0, 20)] * 4 + [at(0, 20)]).astype("datetime64[ns]")
                + np.array([0, 0, 0, 0, 1]).astype("timedelta64[ns]")
            ),
            "microsegundos",
        ),
        (dict(prediction_at=np.ones(5)), "datetime64 o enteros"),
        (dict(target=np.array([2**60, 0, 0, 0, 0])), "pierden precisión"),
    ],
)
def test_invalid_inputs_fail_instead_of_being_repaired(changes, message):
    with pytest.raises(ValueError, match=message):
        panel(**changes)


def test_quantiles_are_validated_and_crossings_are_rejected_with_their_count():
    base = columns()["prediction"][:, None] + np.array([-2, -1, 0, 1, 2]) * 0.01
    result = panel(quantiles=base, levels=LEVELS)
    assert result.levels == LEVELS and result.quantiles.shape == (5, 5)
    assert np.array_equal(result.quantiles[:, 2], result.prediction)
    crossed = base.copy()
    crossed[[0, 3], 1] = crossed[[0, 3], 2] + 1
    with pytest.raises(ValueError, match="Hay 2 filas con cuantiles cruzados"):
        panel(quantiles=crossed, levels=LEVELS)
    with pytest.raises(ValueError, match="juntos"):
        panel(quantiles=base)
    with pytest.raises(ValueError, match="estrictamente crecientes"):
        panel(quantiles=base, levels=(0.025, 0.5, 0.1, 0.9, 0.975))
    with pytest.raises(ValueError, match="estrictamente crecientes"):
        panel(quantiles=base, levels=(0.0, 0.1, 0.5, 0.9, 1.0))
    with pytest.raises(ValueError, match="forma"):
        panel(quantiles=base[:, :4], levels=LEVELS)
    nan = base.copy()
    nan[2, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        panel(quantiles=nan, levels=LEVELS)


def test_central_intervals_pair_symmetric_levels_from_narrowest():
    assert central_intervals(LEVELS) == [(1, 3), (0, 4)]
    assert central_intervals((0.1, 0.5, 0.8)) == []
    result = panel(quantiles=np.zeros((5, 5)) + np.arange(5), levels=LEVELS)
    assert result.interval(0.8) == (1, 3) and result.interval(0.95) == (0, 4)
    with pytest.raises(ValueError, match="intervalo central"):
        result.interval(0.5)
    with pytest.raises(ValueError, match="intervalo central"):
        panel().interval(0.8)


def test_arrow_contract_reads_sample_ids_and_requires_utc_microseconds():
    values = columns()
    table = pa.table(
        dict(
            sample_id=values["row_id"],
            market=values["market"],
            prediction_at=pa.array(values["prediction_at"], type=pa.timestamp("us", tz="UTC")),
            target=values["target"],
            prediction=values["prediction"],
            q10=values["prediction"] - 1,
            q90=values["prediction"] + 1,
        )
    )
    result = ForecastPanel.from_arrow(table, quantile_columns=("q10", "q90"), levels=(0.1, 0.9))
    assert result.cohort_sha256 == panel().cohort_sha256
    assert result.interval(0.8) == (0, 1)
    naive = table.set_column(
        2, "prediction_at", pa.array(values["prediction_at"], type=pa.timestamp("us"))
    )
    with pytest.raises(ValueError, match="UTC"):
        ForecastPanel.from_arrow(naive)
    with pytest.raises(ValueError, match="Faltan columnas"):
        ForecastPanel.from_arrow(table.drop_columns(["target"]))


def test_declared_market_weighting_by_hand():
    # US: sesiones 1 y 3. CN: sesión 8. Una sesión US no definida.
    values = np.array([1.0, 3.0, 8.0, 0.0])
    defined = np.array([True, True, True, False])
    market = np.array([0, 0, 1, 0])
    assert weighted_session_mean(values, defined, market, 2, "session") == 4.0
    assert weighted_session_mean(values, defined, market, 2, "market") == 5.0
    assert weighted_session_mean(values, ~defined, market, 2, "session") == 0.0
    assert weighted_session_mean(values, np.zeros(4, bool), market, 2, "market") is None
    with pytest.raises(ValueError, match="no declarada"):
        weighted_session_mean(values, defined, market, 2, "rows")


def series(values, defined=None, *, loss=True, cohort="x", metric="mae"):
    values = np.asarray(values, dtype=np.float64)
    defined = np.ones(len(values), bool) if defined is None else np.asarray(defined)
    return SessionSeries(
        metric,
        loss,
        cohort,
        ("US", "CN"),
        np.zeros(len(values), dtype=np.int64),
        np.arange(len(values), dtype=np.int64),
        values,
        defined,
    )


def test_session_series_rejects_hidden_or_incoherent_values():
    with pytest.raises(ValueError, match="NaN"):
        series([1.0, np.nan])
    with pytest.raises(ValueError, match="deben valer cero"):
        series([1.0, 2.0], [True, False])
    with pytest.raises(ValueError, match="negativa"):
        series([1.0, -2.0])
    assert series([1.0, -2.0], loss=False).values.tolist() == [1.0, -2.0]


def test_seed_average_keeps_sessions_and_requires_the_same_population():
    averaged = SessionSeries.average(
        [series([1.0, 2.0, 0.0], [True, True, False]), series([3.0, 0.0, 5.0], [True, False, True])]
    )
    assert averaged.values.tolist() == [2.0, 0.0, 0.0]
    assert averaged.defined.tolist() == [True, False, False]
    with pytest.raises(ValueError, match="misma métrica y población"):
        SessionSeries.average([series([1.0]), series([1.0], cohort="y")])
