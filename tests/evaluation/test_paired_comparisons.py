"""Contrastes emparejados, bootstrap circular por días y familias de afirmaciones."""

import json

import numpy as np
import pytest

from mars_titan.evaluation.forecast_panel import ForecastPanel, SessionSeries
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.evaluation.paired_comparisons import (
    circular_block_counts,
    compare_series,
    delta,
    interaction,
    level,
)

OPTIONS = dict(block_length=3, replicates=400, seed=42)


def series(values, *, periods=None, markets=None, defined=None, loss=True, cohort="cohort"):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    defined = np.ones(count, dtype=bool) if defined is None else np.asarray(defined)
    return SessionSeries(
        "mae" if loss else "rank_ic",
        loss,
        cohort,
        ("US", "CN"),
        np.zeros(count, dtype=np.int64) if markets is None else np.asarray(markets),
        np.arange(count, dtype=np.int64) if periods is None else np.asarray(periods),
        np.where(defined, values, 0.0),
        defined,
    )


def by_name(result):
    return {row["name"]: row for row in result["contrasts"]}


def test_contrast_helpers_fix_the_sign_conventions():
    assert delta("B", "V") == {"V": 1.0, "B": -1.0}
    assert interaction("B", "C", "M", "CM") == {"CM": 1.0, "C": -1.0, "M": -1.0, "B": 1.0}
    assert level("B") == {"B": 1.0}


def test_delta_and_relative_improvement_by_hand():
    base = series([0.4, 0.2, 0.6, 0.4])
    variant = series([0.3, 0.2, 0.3, 0.2])
    rows = by_name(compare_series({"B": base, "V": variant}, {"d": delta("B", "V")}, **OPTIONS))
    row = rows["d"]
    # MAE_B = 0,4 y MAE_V = 0,25: Delta = -0,15 y mejora relativa del 37,5 %.
    assert row["estimate"] == pytest.approx(-0.15)
    assert row["base_estimate"] == pytest.approx(0.4)
    assert row["variant_estimate"] == pytest.approx(0.25)
    assert row["relative_improvement_percent"] == pytest.approx(37.5)
    lower, upper = row["relative_improvement_interval"]
    assert lower <= 37.5 <= upper
    assert row["sessions"] == 4 and row["excluded_sessions"] == 0


def test_relative_improvement_is_undefined_when_the_base_is_zero():
    zero = series([0.0, 0.0, 0.0, 0.0])
    variant = series([0.1, 0.2, 0.0, 0.1])
    row = compare_series({"B": zero, "V": variant}, {"d": delta("B", "V")}, **OPTIONS)
    row = row["contrasts"][0]
    assert row["estimate"] == pytest.approx(0.1)
    assert row["relative_improvement_percent"] is None
    assert row["relative_improvement_reason"] == "La métrica de la base es cero"
    partly_zero = series([0.0, 0.0, 0.0, 0.3])
    result = compare_series({"B": partly_zero, "V": variant}, {"d": delta("B", "V")}, **OPTIONS)[
        "contrasts"
    ][0]
    assert result["relative_improvement_percent"] == pytest.approx(-100 * (0.1 - 0.075) / 0.075)
    assert result["relative_improvement_interval"] is None
    assert result["relative_improvement_reason"] == "Alguna réplica tiene una base igual a cero"


def test_relative_improvement_is_not_reported_for_non_loss_metrics_or_non_pairwise_contrasts():
    ic = {"A": series([0.1, 0.2, 0.3], loss=False), "B": series([0.2, 0.1, 0.0], loss=False)}
    row = compare_series(ic, {"d": delta("A", "B")}, **OPTIONS)["contrasts"][0]
    assert "relative_improvement_percent" not in row
    losses = {name: series([1.0, 2.0, 3.0]) for name in ("B", "C", "M", "CM")}
    row = compare_series(losses, {"I": interaction("B", "C", "M", "CM")}, **OPTIONS)
    assert "relative_improvement_percent" not in row["contrasts"][0]


def test_interaction_by_hand_and_its_sign():
    values = dict(
        B=[1.0, 1.0, 1.0, 1.0],
        C=[0.9, 0.8, 0.9, 0.8],
        M=[0.9, 0.9, 1.0, 1.0],
        CM=[0.7, 0.6, 0.7, 0.6],
    )
    family = {name: series(value) for name, value in values.items()}
    row = compare_series(family, {"I": interaction("B", "C", "M", "CM")}, **OPTIONS)
    # I = 0,65 − 0,85 − 0,95 + 1 = −0,15: la combinación reduce más que la suma de efectos.
    assert row["contrasts"][0]["estimate"] == pytest.approx(-0.15)


def test_identical_models_and_constant_shifts_collapse_their_intervals():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.5, 1.5, 60)
    family = {"B": series(base), "same": series(base), "shift": series(base + 0.1)}
    result = compare_series(
        family, {"same": delta("B", "same"), "shift": delta("B", "shift")}, **OPTIONS
    )
    rows = by_name(result)
    assert rows["same"]["estimate"] == 0.0
    assert rows["same"]["interval"] == [0.0, 0.0]
    assert rows["same"]["simultaneous_interval"] == [0.0, 0.0]
    assert rows["same"]["simultaneous_excludes_zero"] is False
    # Solo un remuestreo con los mismos días para ambos modelos conserva la diferencia exacta.
    for bound in (*rows["shift"]["interval"], *rows["shift"]["simultaneous_interval"]):
        assert bound == pytest.approx(0.1, abs=1e-12)
    assert rows["shift"]["simultaneous_excludes_zero"] is True


def test_all_sessions_of_a_utc_day_are_resampled_together_across_markets():
    rng = np.random.default_rng(2)
    days = 40
    effect = rng.normal(size=days)
    # US y CN tienen diferencias opuestas cada día: solo se cancelan si se remuestrean juntas.
    periods = np.repeat(np.arange(days), 2)
    markets = np.tile([0, 1], days)
    difference = np.ravel(np.column_stack([effect, -effect]))
    base = np.full(2 * days, 5.0)
    family = {
        "B": series(base, periods=periods, markets=markets),
        "V": series(base + difference, periods=periods, markets=markets),
    }
    row = compare_series(family, {"d": delta("B", "V")}, **OPTIONS)["contrasts"][0]
    assert row["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert row["interval"][0] == pytest.approx(0.0, abs=1e-12)
    assert row["interval"][1] == pytest.approx(0.0, abs=1e-12)


def test_resampling_is_reproducible_and_records_its_configuration():
    rng = np.random.default_rng(3)
    family = {"B": series(rng.uniform(size=50)), "V": series(rng.uniform(size=50))}
    contrasts = {"d": delta("B", "V")}
    first = compare_series(family, contrasts, **OPTIONS)
    assert first == compare_series(family, contrasts, **OPTIONS)
    other = compare_series(family, contrasts, block_length=3, replicates=400, seed=43)
    assert other["contrasts"][0]["interval"] != first["contrasts"][0]["interval"]
    assert other["contrasts"][0]["estimate"] == first["contrasts"][0]["estimate"]
    resampling = first["resampling"]
    assert resampling["method"] == "circular_block_bootstrap"
    assert resampling["unit"] == "utc_calendar_day_with_all_sessions_and_assets"
    assert (resampling["block_length"], resampling["replicates"], resampling["seed"]) == (
        3,
        400,
        42,
    )
    assert resampling["bit_generator"] == "PCG64" and resampling["periods"] == 50


def test_block_length_sensitivity_uses_the_same_estimate_and_reports_each_length():
    rng = np.random.default_rng(4)
    family = {"B": series(rng.uniform(size=80)), "V": series(rng.uniform(size=80))}
    result = compare_series(
        family, {"d": delta("B", "V")}, **OPTIONS, sensitivity_block_lengths=(1, 10, 80)
    )
    lengths = [entry["block_length"] for entry in result["sensitivity"]]
    assert lengths == [1, 10, 80]
    intervals = [entry["contrasts"][0]["interval"] for entry in result["sensitivity"]]
    assert intervals[0] != intervals[1] and intervals[2] is None
    assert result["sensitivity"][2]["reason"] == "Se necesitan más días que la longitud del bloque"


def test_block_longer_than_the_sample_gives_no_interval_but_keeps_the_estimate():
    family = {"B": series([1.0, 2.0, 3.0]), "V": series([1.0, 1.0, 1.0])}
    result = compare_series(family, {"d": delta("B", "V")}, block_length=3, replicates=50, seed=1)
    row = result["contrasts"][0]
    assert row["estimate"] == pytest.approx(-1.0)
    assert row["interval"] is None and row["simultaneous_interval"] is None
    assert result["resampling"]["reason"] == "Se necesitan más días que la longitud del bloque"
    assert row["relative_improvement_percent"] == pytest.approx(50.0)
    assert row["relative_improvement_reason"] == "No hay remuestreo disponible"


def test_simultaneous_family_widens_the_critical_value_with_more_claims():
    rng = np.random.default_rng(5)
    family = {name: series(rng.uniform(size=120)) for name in ("B", "C", "M", "CM")}
    single = compare_series(family, {"C": delta("B", "C")}, **OPTIONS)
    joint = compare_series(
        family,
        {"C": delta("B", "C"), "M": delta("B", "M"), "CM": delta("B", "CM")},
        **OPTIONS,
    )
    assert joint["multiplicity"]["family_size"] == 3
    assert joint["multiplicity"]["critical_value"] > single["multiplicity"]["critical_value"]
    alone, together = single["contrasts"][0], joint["contrasts"][0]
    # Las réplicas son las mismas. Solo cambia el redondeo del producto matricial.
    assert alone["interval"] == pytest.approx(together["interval"], rel=1e-12)
    assert np.ptp(together["simultaneous_interval"]) > np.ptp(alone["simultaneous_interval"])


def test_auxiliary_base_columns_do_not_enter_the_multiplicity_family():
    rng = np.random.default_rng(6)
    family = {"B": series(rng.uniform(size=90)), "V": series(rng.uniform(size=90))}
    paired = compare_series(family, {"d": delta("B", "V")}, **OPTIONS)
    scaled = compare_series(family, {"d": {"V": 2.0, "B": -2.0}}, **OPTIONS)
    assert paired["multiplicity"]["critical_value"] == pytest.approx(
        scaled["multiplicity"]["critical_value"]
    )
    assert "relative_improvement_percent" in paired["contrasts"][0]
    assert "relative_improvement_percent" not in scaled["contrasts"][0]


def test_undefined_sessions_are_excluded_jointly_and_reported():
    first = series([0.1, 0.5, 0.0, 0.3], defined=[True, True, False, True], loss=False)
    second = series([0.2, 0.0, 0.4, 0.1], defined=[True, False, True, True], loss=False)
    result = compare_series(
        {"A": first, "B": second},
        {"A": level("A"), "d": delta("A", "B")},
        **OPTIONS,
    )
    rows = by_name(result)
    assert rows["A"]["estimate"] == pytest.approx(0.3) and rows["A"]["sessions"] == 3
    # Solo las sesiones 0 y 3 están definidas en ambos modelos.
    assert rows["d"]["estimate"] == pytest.approx(((0.2 - 0.1) + (0.1 - 0.3)) / 2)
    assert rows["d"]["sessions"] == 2 and rows["d"]["excluded_sessions"] == 2


def build_panel(prediction, *, target, market, days):
    hours = np.where(market == "US", 20, 7)
    times = np.datetime64("2023-01-02T00", "us") + days * np.timedelta64(1, "D")
    return ForecastPanel.from_columns(
        np.arange(len(target)),
        market,
        times + hours * np.timedelta64(1, "h"),
        target,
        prediction,
    )


def test_point_estimates_match_the_session_score_summary_under_both_weightings():
    rng = np.random.default_rng(7)
    rows = 800
    target = rng.normal(size=rows)
    market = np.where(rng.random(rows) < 0.3, "CN", "US")
    days = rng.integers(0, 30, rows)
    scores = {
        name: score_sessions(
            build_panel(target * k + rng.normal(size=rows), target=target, market=market, days=days)
        )
        for name, k in (("B", 0.1), ("V", 0.5))
    }
    for weighting in ("session", "market"):
        result = compare_series(
            {name: score.series("mae") for name, score in scores.items()},
            {"B": level("B"), "d": delta("B", "V")},
            **OPTIONS,
            market_weighting=weighting,
        )
        rows_by_name = by_name(result)
        summary = {name: s.summary(market_weighting=weighting) for name, s in scores.items()}
        assert rows_by_name["B"]["estimate"] == summary["B"]["point"]["mae"]
        expected = summary["V"]["point"]["mae"] - summary["B"]["point"]["mae"]
        assert rows_by_name["d"]["estimate"] == pytest.approx(expected, abs=1e-14)
        assert result["market_weighting"] == weighting


def test_circular_counts_keep_the_sample_size_and_give_equal_weight_to_extremes():
    rng = np.random.default_rng(8)
    counts = circular_block_counts(rng, 4000, 30, 7)
    assert counts.shape == (4000, 30) and np.all(counts.sum(axis=1) == 30)
    # Sin recorrido circular los primeros y últimos días aparecerían menos.
    assert np.allclose(counts.mean(axis=0), 1.0, atol=0.06)
    one = circular_block_counts(np.random.default_rng(0), 1, 10, 4)[0]
    assert one.sum() == 10


@pytest.mark.parametrize(
    "family,contrasts,message",
    [
        ({"B": series([1.0, 2.0]), "V": series([1.0, 2.0], cohort="other")}, None, "población"),
        ({"B": series([1.0, 2.0]), "V": series([1.0, 2.0], loss=False)}, None, "métrica"),
        ({"B": series([1.0, 2.0])}, {"d": delta("B", "X")}, "no usa modelos"),
        ({"B": series([1.0, 2.0])}, {"d": {"B": 0.0}}, "no usa modelos"),
        ({"B": series([1.0, 2.0])}, {"d": {"B": float("nan")}}, "no usa modelos"),
        ({"B": series([1.0, 2.0])}, {}, "familia"),
    ],
)
def test_invalid_families_are_rejected(family, contrasts, message):
    contrasts = {"d": delta("B", "V")} if contrasts is None else contrasts
    with pytest.raises(ValueError, match=message):
        compare_series(family, contrasts, **OPTIONS)


@pytest.mark.parametrize(
    "changes,message",
    [
        (dict(block_length=0), "bloque"),
        (dict(replicates=5), "réplicas"),
        (dict(seed=-1), "semilla"),
        (dict(confidence=1.0), "confianza"),
        (dict(confidence=95), "confianza"),
        (dict(market_weighting="rows"), "no declarada"),
        (dict(sensitivity_block_lengths=(0,)), "sensibilidad"),
    ],
)
def test_invalid_resampling_options_are_rejected(changes, message):
    family = {"B": series([1.0, 2.0, 3.0]), "V": series([1.0, 2.0, 2.0])}
    options = dict(OPTIONS, **changes)
    with pytest.raises(ValueError, match=message):
        compare_series(family, {"d": delta("B", "V")}, **options)


def coverage(rng, *, block_length, phi, simulations, periods):
    hits = 0
    for simulation in range(simulations):
        noise = rng.normal(size=periods)
        values = np.empty(periods)
        values[0] = noise[0]
        for t in range(1, periods):
            values[t] = phi * values[t - 1] + noise[t]
        family = {"B": series(np.zeros(periods), loss=False), "V": series(values, loss=False)}
        row = compare_series(
            family,
            {"d": delta("B", "V")},
            block_length=block_length,
            replicates=300,
            seed=simulation,
        )["contrasts"][0]
        hits += row["interval"][0] <= 0 <= row["interval"][1]
    return hits / simulations


def test_block_intervals_cover_the_true_mean_and_beat_iid_resampling_under_dependence():
    # Media real cero. Con independencia la cobertura nominal es 95 %.
    independent = coverage(
        np.random.default_rng(10), block_length=1, phi=0.0, simulations=200, periods=120
    )
    assert 0.88 <= independent <= 0.99
    # Con AR(1) de coeficiente 0,8, el remuestreo de días aislados infracubre.
    iid = coverage(np.random.default_rng(11), block_length=1, phi=0.8, simulations=150, periods=240)
    blocks = coverage(
        np.random.default_rng(11), block_length=20, phi=0.8, simulations=150, periods=240
    )
    assert iid < 0.75 and blocks > iid + 0.1


def test_market_weighting_is_applied_inside_every_replicate():
    rng = np.random.default_rng(12)
    days = 50
    periods = np.r_[np.arange(days), np.arange(0, days, 2)]
    markets = np.r_[np.zeros(days, np.int64), np.ones(days // 2, np.int64)]
    base = rng.uniform(1, 2, len(periods))
    # Desplazamiento constante por mercado: 0,1 en US y 0,3 en CN.
    shifted = base + np.where(markets == 0, 0.1, 0.3)
    family = {
        "B": series(base, periods=periods, markets=markets),
        "V": series(shifted, periods=periods, markets=markets),
    }
    by_market = compare_series(
        family, {"d": delta("B", "V")}, **OPTIONS, market_weighting="market"
    )["contrasts"][0]
    assert by_market["estimate"] == pytest.approx(0.2)
    assert by_market["interval"] == pytest.approx([0.2, 0.2], abs=1e-12)
    by_session = compare_series(family, {"d": delta("B", "V")}, **OPTIONS)["contrasts"][0]
    assert by_session["estimate"] == pytest.approx((0.1 * 50 + 0.3 * 25) / 75)
    assert np.ptp(by_session["interval"]) > 1e-3


def test_contrasts_without_defined_sessions_have_no_estimate_or_interval():
    empty = series(np.zeros(6), defined=np.zeros(6, dtype=bool), loss=False)
    full = series([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], loss=False)
    result = compare_series({"A": empty, "B": full}, {"A": level("A")}, **OPTIONS)
    row = result["contrasts"][0]
    assert row["estimate"] is None and row["interval"] is None and row["sessions"] == 0
    assert result["resampling"]["reason"] == "Ningún contraste tiene sesiones definidas"
    mixed = compare_series({"A": empty, "B": full}, {"A": level("A"), "B": level("B")}, **OPTIONS)
    rows = by_name(mixed)
    assert rows["A"]["interval"] is None and rows["B"]["interval"] is not None
    json.dumps(mixed, allow_nan=False)


def test_numpy_coefficients_and_confidence_are_accepted():
    family = {"B": series([1.0, 2.0, 3.0, 4.0]), "V": series([1.0, 1.0, 2.0, 2.0])}
    contrasts = {"d": {"V": np.float64(1), "B": np.int64(-1)}}
    row = compare_series(family, contrasts, **OPTIONS, confidence=np.float64(0.9))["contrasts"][0]
    assert row["estimate"] == pytest.approx(-1.0) and row["relative_improvement_percent"] == 40.0
    with pytest.raises(ValueError, match="no usa modelos"):
        compare_series(family, {"d": {"V": True, "B": -1}}, **OPTIONS)
