"""Anomalías conocidas de #747: fórmulas, ventanas mínimas, calendario y causalidad.

Las series son técnicas. Cada fórmula se compara con un cálculo directo de la sesión, y la
causalidad se comprueba alterando todo lo posterior a una sesión.
"""

import math

import numpy as np
import pytest

from mars_titan.data import anomaly_features as af

COLUMN = {name: k for k, name in enumerate(af.FEATURES)}


def series(length=400, *, seed=0):
    generator = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(generator.normal(0, 0.01, length)))
    volume = generator.integers(1_000, 5_000, length).astype(np.float64)
    market = 300 * np.exp(np.cumsum(generator.normal(0, 0.008, length)))
    return close, volume, market


def feature(values, name, session):
    return values[session, COLUMN[name]]


def test_the_families_name_every_feature_once():
    names = [name for family in af.FAMILIES.values() for name in family]
    assert tuple(names) == af.FEATURES and len(set(names)) == len(names) == 10


def test_each_return_and_level_feature_follows_its_formula():
    close, volume, market = series()
    values = af.asset_features(close, volume, market, [])
    t = 350
    assert feature(values, "reversal_1", t) == pytest.approx(close[t] / close[t - 1] - 1)
    assert feature(values, "reversal_5", t) == pytest.approx(close[t] / close[t - 5] - 1)
    assert feature(values, "momentum_12_1", t) == pytest.approx(close[t - 21] / close[t - 252] - 1)
    assert feature(values, "distance_to_52_week_high", t) == pytest.approx(
        close[t] / close[t - 251 : t + 1].max() - 1
    )
    logs = np.log1p(volume)
    assert feature(values, "volume_shock_20", t) == pytest.approx(logs[t] - logs[t - 20 : t].mean())
    assert feature(values, "relative_turnover_20_252", t) == pytest.approx(
        logs[t - 19 : t + 1].mean() - logs[t - 251 : t + 1].mean()
    )


@pytest.mark.parametrize("length", [20, 60])
def test_the_idiosyncratic_volatility_is_the_residual_deviation_of_the_market_regression(length):
    close, volume, market = series()
    values = af.asset_features(close, volume, market, [])
    t = 300
    asset = close[t - length + 1 : t + 1] / close[t - length : t] - 1
    factor = market[t - length + 1 : t + 1] / market[t - length : t] - 1
    slope, intercept = np.polyfit(factor, asset, 1)
    residual = asset - (intercept + slope * factor)
    expected = math.sqrt(residual @ residual / (length - 2))
    name = f"idiosyncratic_volatility_{length}"
    assert feature(values, name, t) == pytest.approx(expected, rel=1e-10)


def test_windows_need_three_quarters_of_their_observations():
    close, volume, market = series()
    t = 300
    # Un cierre ausente anula los dos retornos que lo usan, salvo en el borde de la ventana. Con
    # los cierres de t − 20, t − 18 y t − 16 quedan 15 pares de 20, el mínimo, y sin el de t, 14.
    for gaps_at, defined in (
        ([t - 20, t - 18, t - 16], True),
        ([t - 20, t - 18, t - 16, t], False),
    ):
        gaps = close.copy()
        gaps[gaps_at] = np.nan
        values = af.asset_features(gaps, volume, market, [])
        assert np.isfinite(feature(values, "idiosyncratic_volatility_20", t)) == defined
    # El máximo anual necesita 189 cierres de 252.
    for absent, defined in ((63, True), (64, False)):
        gaps = close.copy()
        gaps[t - 251 : t - 251 + absent] = np.nan
        values = af.asset_features(gaps, volume, market, [])
        assert np.isfinite(feature(values, "distance_to_52_week_high", t)) == defined
    # Antes de tener un año de cierres no hay máximo anual ni momentum.
    values = af.asset_features(close, volume, market, [])
    assert np.isnan(values[187, COLUMN["distance_to_52_week_high"]])
    assert np.isfinite(values[188, COLUMN["distance_to_52_week_high"]])
    assert np.isnan(values[251, COLUMN["momentum_12_1"]])
    assert np.isfinite(values[252, COLUMN["momentum_12_1"]])


def test_each_momentum_end_takes_the_last_close_up_to_five_sessions_back():
    close, volume, market = series()
    t = 350
    gaps = close.copy()
    gaps[t - 252 - 4 : t - 252 + 1] = np.nan
    gaps[t - 21 - 2 : t - 21 + 1] = np.nan
    values = af.asset_features(gaps, volume, market, [])
    # El extremo antiguo queda a cinco sesiones del último cierre y el reciente a tres.
    expected = close[t - 21 - 3] / close[t - 252 - 5] - 1
    assert feature(values, "momentum_12_1", t) == pytest.approx(expected)
    gaps[t - 252 - 5] = np.nan
    values = af.asset_features(gaps, volume, market, [])
    assert np.isnan(feature(values, "momentum_12_1", t))


def test_a_return_needs_both_closes():
    close, volume, market = series()
    gaps = close.copy()
    gaps[200] = np.nan
    values = af.asset_features(gaps, volume, market, [])
    assert np.isnan(values[200, COLUMN["reversal_1"]]) and np.isnan(
        values[201, COLUMN["reversal_1"]]
    )
    assert np.isnan(values[205, COLUMN["reversal_5"]])
    assert np.isfinite(values[206, COLUMN["reversal_5"]])


def test_the_filing_age_counts_sessions_since_the_last_known_filing():
    close, volume, market = series(40)
    values = af.asset_features(close, volume, market, [5, 20])
    age = values[:, COLUMN["sessions_since_filing"]]
    recent = values[:, COLUMN["recent_filing_5"]]
    assert np.isnan(age[:5]).all() and np.isnan(recent[:5]).all()
    assert age[5:10].tolist() == [0, 1, 2, 3, 4] and recent[5:10].tolist() == [1] * 5
    # Con cinco sesiones la presentación ya no es reciente.
    assert age[10] == 5 and recent[10] == 0
    assert age[19] == 14 and recent[19] == 0
    # Una presentación nueva reinicia la cuenta en la sesión en que se conoce.
    assert age[20] == 0 and age[39] == 19
    # Un fondo sin presentaciones no tiene antigüedad en ninguna sesión.
    values = af.asset_features(close, volume, market, np.empty(0, dtype=np.int64))
    columns = [COLUMN["sessions_since_filing"], COLUMN["recent_filing_5"]]
    assert np.isnan(values[:, columns]).all()


def test_no_feature_reads_a_session_after_its_own():
    close, volume, market = series(500, seed=3)
    filings = np.array([30, 95, 160, 290])
    reference = af.asset_features(close, volume, market, filings)
    generator = np.random.default_rng(9)
    for cut in (260, 300, 420):
        later = slice(cut + 1, None)
        changed = [value.copy() for value in (close, volume, market)]
        for value in changed:
            value[later] *= np.exp(generator.normal(0, 0.3, len(value[later])))
        changed[0][cut + 3] = np.nan
        future = np.unique(np.r_[filings[filings <= cut], cut + 1, cut + 7])
        values = af.asset_features(*changed, future)
        np.testing.assert_array_equal(values[: cut + 1], reference[: cut + 1])
        assert not np.array_equal(values[cut + 1 :], reference[cut + 1 :], equal_nan=True)


def test_market_sessions_skip_the_sessions_absent_in_the_whole_market():
    absent = np.array([3, 4, 9])
    assert af.market_sessions([0, 2, 5, 8, 10], absent).tolist() == [0, 2, 3, 6, 7]
    # Lo que llega en una sesión ausente se conoce en la siguiente con mercado.
    assert af.market_sessions([3, 4, 9], absent).tolist() == [3, 3, 7]
    with pytest.raises(ValueError, match="ausencias ordenadas"):
        af.market_sessions([1], [4, 3])


def test_each_filing_is_known_from_its_first_fact_and_at_or_after_a_decision():
    decisions = np.array([10, 20, 30, 40, 50, 60])
    absent = np.array([2])
    accessions = ["a", "a", "b", "c", "d", "e"]
    # «a» se conoce con su primer hecho, «b» en una decisión exacta, «c» entre dos decisiones,
    # «d» en la sesión ausente y «e» después del calendario.
    available = [35, 15, 40, 41, 30, 61]
    sessions = af.filing_sessions(accessions, available, decisions, absent)
    # Sin la sesión ausente, «a» cae en la sesión 1, «b» en la 2 y «c» en la 3. «d» se conoce en
    # la siguiente sesión con mercado, la misma que «b», y «e» no se conoce nunca.
    assert sessions.tolist() == [1, 2, 3]
    # Sola, una presentación que llega justo en una decisión se conoce en ella, y una que llega
    # un instante después espera a la siguiente.
    for moment, session in ((39, 2), (40, 2), (41, 3)):
        assert af.filing_sessions(["b"], [moment], decisions, absent).tolist() == [session]
    assert af.filing_sessions([], [], decisions, absent).size == 0
    with pytest.raises(ValueError, match="no tiene acceso"):
        af.filing_sessions([""], [15], decisions, absent)
    with pytest.raises(ValueError, match="un acceso y una disponibilidad"):
        af.filing_sessions(["a", "b"], [15], decisions, absent)


def test_a_dense_series_places_each_value_in_its_session():
    dense = af.dense_series([1.0, 2.0, 3.0], [0, 2, 5], 6)
    np.testing.assert_array_equal(dense, [1.0, np.nan, 2.0, np.nan, np.nan, 3.0])
    assert af.dense_series([], np.empty(0, dtype=np.int64), 3).shape == (3,)
    for sessions in ([0, 2, 2], [2, 0, 1], [0, 2, 6], [-1, 0, 1]):
        with pytest.raises(ValueError, match="únicas, crecientes y del calendario"):
            af.dense_series([1.0, 2.0, 3.0], sessions, 6)


def test_invalid_series_are_refused():
    close, volume, market = series(30)
    with pytest.raises(ValueError, match="mismas sesiones"):
        af.asset_features(close[:-1], volume, market, [])
    with pytest.raises(ValueError, match="positivos"):
        af.asset_features(np.where(np.arange(30) == 3, -1.0, close), volume, market, [])
    with pytest.raises(ValueError, match="no negativos"):
        af.asset_features(close, np.where(np.arange(30) == 3, -1.0, volume), market, [])
    with pytest.raises(ValueError, match="positivos"):
        af.asset_features(close, volume, np.where(np.arange(30) == 3, 0.0, market), [])
    for filings in ([4, 2], [3, 3], [-1, 2]):
        with pytest.raises(ValueError, match="únicas y crecientes"):
            af.asset_features(close, volume, market, filings)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_inputs_are_refused_instead_of_becoming_missing_features(position, value):
    values = list(series(300))
    values[position][-1] = value
    with pytest.raises(ValueError, match="infinitos"):
        af.asset_features(*values, [])


def test_all_missing_series_preserve_explicit_missing_features():
    missing = np.full(300, np.nan)
    features = af.asset_features(missing, missing, missing, [])
    assert features.shape == (300, len(af.FEATURES))
    assert np.isnan(features).all()


def test_the_log1p_volume_signal_is_not_invariant_to_a_common_adjustment():
    close, volume, market = series(300)
    volume[:] = 1
    volume[-1] = 3
    before = af.asset_features(close, volume, market, [])
    after = af.asset_features(close, 2 * volume, market, [])
    original = feature(before, "volume_shock_20", -1)
    adjusted = feature(after, "volume_shock_20", -1)
    assert original == pytest.approx(math.log(2))
    assert adjusted == pytest.approx(math.log(7 / 3))
    assert abs(original - adjusted) > 0.1


def test_session_ranks_map_each_cross_section_to_minus_one_one():
    values = np.full((4, 3, 2), np.nan)
    present = np.ones((4, 3), dtype=bool)
    values[:, 0, 0] = [3.0, 1.0, 2.0, 2.0]
    values[:, 0, 1] = [5.0, np.nan, 7.0, 6.0]
    values[:, 1, 0] = [1.0, 9.0, 4.0, 8.0]
    present[3, 1] = False
    values[2, 2, 0] = 4.0
    ranks, missing = af.session_ranks(values, present)
    # Empates con el rango medio: 1, 2,5, 2,5 y 4 entre cuatro activos.
    np.testing.assert_allclose(ranks[:, 0, 0], [1.0, -1.0, 0.0, 0.0])
    np.testing.assert_allclose(ranks[:, 0, 1], [-1.0, 0.0, 1.0, 0.0])
    # Un activo sin precio en la sesión queda fuera de su sección cruzada.
    np.testing.assert_allclose(ranks[:, 1, 0], [-1.0, 1.0, 0.0, 0.0])
    assert missing[3, 1, 0] and not missing[:3, 1, 0].any()
    # Un único valor válido no tiene orden y vale cero.
    assert ranks[2, 2, 0] == 0.0 and missing[:, 2, 0].tolist() == [True, True, False, True]
    assert missing[1, 0, 1] and ranks[1, 0, 1] == 0.0


def test_a_session_rank_depends_only_on_that_session():
    generator = np.random.default_rng(4)
    values = generator.normal(size=(6, 5, 3))
    present = generator.random((6, 5)) > 0.2
    ranks, missing = af.session_ranks(values, present)
    changed = values.copy()
    changed[:, 3:] = generator.normal(size=(6, 2, 3)) * 10
    other = present.copy()
    other[:, 3:] = True
    again, absent = af.session_ranks(changed, other)
    np.testing.assert_array_equal(again[:, :3], ranks[:, :3])
    np.testing.assert_array_equal(absent[:, :3], missing[:, :3])


def test_session_ranks_refuse_a_presence_of_another_shape():
    with pytest.raises(ValueError, match="su presencia"):
        af.session_ranks(np.zeros((2, 3, 1)), np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="su presencia"):
        af.session_ranks(np.zeros((2, 3, 1)), np.ones((2, 3)))
