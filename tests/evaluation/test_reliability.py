"""F1 y F2: la declaración, las piezas puras con casos hechos a mano y la orden de un origen."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.stats import binom, spearmanr

from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import reliability as rel
from tests.evaluation import test_agent_core as core

DECLARATION = core.ROOT / "configs" / "experiments" / "posttraining-reliability.json"


def declaration():
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def test_the_declaration_is_the_registered_one():
    document, digest = rel.checked_declaration(declaration())
    assert digest == rel.REGISTERED_SHA256
    assert document["directional"]["target_precision"] == 0.9
    assert all(isinstance(c, float) for c in document["reliability"]["coverages"])


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["directional"].update(target_precision=0.8),
        lambda d: d["directional"].update(delta=0.1),
        lambda d: d["directional"]["thresholds"].pop(0),
        lambda d: d["reliability"].update(alpha_cap_quantile=0.999),
        lambda d: d["reliability"]["coverages"].pop(),
        lambda d: d.update(parent="titans_mac"),
    ],
)
def test_any_change_of_the_declared_rules_is_refused(change):
    document = declaration()
    change(document)
    with pytest.raises(ValueError, match="no es la registrada para F"):
        rel.checked_declaration(document)


def test_wilson_matches_the_textbook_interval():
    entry = rel.wilson(8, 10)
    assert entry["rate"] == 0.8
    assert entry["low"] == pytest.approx(0.4902, abs=1e-4)
    assert entry["high"] == pytest.approx(0.9433, abs=1e-4)
    assert rel.wilson(0, 0) is None
    assert rel.wilson(0, 5)["low"] == pytest.approx(0.0, abs=1e-12)


def test_the_liquid_third_is_the_top_of_each_session_among_known_values():
    liquidity = np.array([5, 1, np.nan, 7, 3, 2, 8, 4, 6, 10, 30, 20], dtype=float)
    keep = rel.liquid_rows(liquidity, [(0, 9), (9, 12)])
    # Ocho valores conocidos en la primera sesión: su tercio son los dos mayores (8 y 7).
    assert np.flatnonzero(keep[:9]).tolist() == [3, 6]
    assert np.flatnonzero(keep[9:]).tolist() == [1]


def _two_markets(sessions=12, assets=40, ic_us=1.0, ic_cn=-0.2, seed=3):
    """Sesiones de dos mercados intercaladas en el tiempo; la señal de cada una tiene el IC dado."""
    rng = np.random.default_rng(seed)
    score, target, bounds, keys = [], [], [], []
    for t in range(sessions):
        for market, ic in (("US", ic_us), ("CN", ic_cn)):
            y = rng.standard_normal(assets)
            s = y if ic == 1.0 else ic * y + np.sqrt(1 - ic * ic) * rng.standard_normal(assets)
            start = len(score)
            score += list(s)
            target += list(y)
            bounds.append((start, start + assets))
            keys.append((market, 1000 + t))
    return np.array(score), np.array(target), bounds, keys


def test_windows_are_consecutive_sessions_of_each_market_without_mixing_markets():
    tool = og._screening()
    score, target, bounds, keys = _two_markets(ic_cn=-1.0)
    measured = rel.session_measures(tool, score, target, bounds, min_assets=30)
    assert len(measured) == 24
    weekly = rel.window_rates(measured, keys, 5)
    # Dos ventanas completas por mercado: las de US con IC 1, las de CN con IC −1.
    assert weekly["windows"] == 4 and weekly["ic"]["hits"] == 2
    assert rel.window_rates(measured, keys, 21)["windows"] == 0
    assert rel.window_rates(measured, keys, 21)["ic"] is None


def test_mixed_markets_would_change_the_windows():
    tool = og._screening()
    score, target, bounds, keys = _two_markets(ic_cn=-0.2, seed=5)
    measured = rel.session_measures(tool, score, target, bounds, min_assets=30)
    us = [v for i, v, _ in measured if keys[i][0] == "US"]
    assert np.allclose(us, 1.0)
    # Por mercado: las dos de US positivas y las dos de CN negativas. Mezclando los mercados en el
    # orden de las filas saldrían las cuatro positivas.
    assert rel.window_rates(measured, keys, 5)["ic"]["hits"] == 2


def test_selective_precision_counts_the_tails_by_hand():
    score = np.arange(10, dtype=float)
    target = np.array([0.1, -1, -1, -1, 0.2, 0.3, 1, 1, 1, 0.5])
    flagged = np.zeros(10, dtype=bool)
    flagged[0] = True
    curve = rel.selective_precision(
        score, target, [(0, 10)], [1.0, 0.2], min_assets=1, flagged=flagged
    )
    # Al 20 %: un largo (puntuación 9, objetivo 0,5, acierto) y un corto (0, objetivo 0,1, fallo).
    assert curve["0.2"]["sign"]["hits"] == 1 and curve["0.2"]["names"] == 2
    assert curve["0.2"]["flagged"] == 0.5
    # Mediana 0,25: el largo la supera y el corto la queda por debajo, así que acierta los dos.
    assert curve["0.2"]["relative"]["hits"] == 2
    assert curve["0.2"]["spread_bps"] == pytest.approx(1e4 * 0.4)
    # Al 100 %, cinco largos (0,3; 1; 1; 1; 0,5) y cinco cortos (0,1; −1; −1; −1; 0,2).
    assert curve["1.0"]["names"] == 10 and curve["1.0"]["sign"]["hits"] == 8
    few = rel.selective_precision(score, target, [(0, 10)], [1.0], min_assets=11)
    assert few["1.0"]["sessions"] == 0 and few["1.0"]["sign"] is None


def test_zero_excluding_intervals_act_long_above_and_short_below():
    low = np.array([0.1, -0.2, -0.3, 0.05])
    high = np.array([0.4, -0.1, 0.3, 0.2])
    target = np.array([0.2, 0.1, 0.5, -0.1])
    entry = rel.zero_excluding(low, high, target)
    assert (entry["long"], entry["short"], entry["rows"]) == (2, 1, 3)
    assert entry["sign"]["hits"] == 1
    kept = rel.zero_excluding(low, high, target, keep=np.array([True, True, True, False]))
    assert kept["rows"] == 2 and kept["share"] == pytest.approx(2 / 3)


def test_the_isotonic_fit_matches_scikit_learn_and_ignores_the_row_order():
    isotonic = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(11)
    x = np.round(rng.standard_normal(400), 1)
    y = (x + rng.standard_normal(400) > 0).astype(float)
    model = rel.isotonic_fit(x, y)
    expected = isotonic.IsotonicRegression(increasing=True).fit(x, y).predict(x)
    assert np.allclose(rel.isotonic_predict(model, x), expected, rtol=0, atol=1e-12)
    assert np.all(np.diff(model[1]) > 0)
    order = rng.permutation(400)
    again = rel.isotonic_fit(x[order], y[order])
    assert np.array_equal(again[0], model[0]) and np.allclose(again[1], model[1])
    assert rel.isotonic_predict(model, [-99.0, 99.0]).tolist() == [model[1][0], model[1][-1]]


def test_a_null_target_is_an_error_in_both_directions():
    right = rel.correct(np.array([0.0, 0.0, 1.0, -1.0, -1.0]), np.array([1, -1, 1, -1, 1]))
    assert right.tolist() == [False, False, True, True, False]


def test_minimum_blocks_keep_the_fit_increasing_and_the_totals():
    rng = np.random.default_rng(12)
    x = rng.standard_normal(1000)
    y = (x + rng.standard_normal(1000) > 0).astype(float)
    edges, values = rel.isotonic_fit(x, y, min_rows=100)
    counts = np.bincount(np.searchsorted(edges, x, side="left"), minlength=len(edges))
    assert counts.min() >= 100 and len(edges) <= 10
    assert np.all(np.diff(values) > 0)
    assert float((counts * values).sum()) == pytest.approx(float(y.sum()))
    # Sin señal, un extremo de tres filas que acierta da P = 1 en la isótona sola; con bloques de
    # 100 filas, el escalón de arriba es la media de al menos 100 filas.
    noise = (rng.random(1000) < 0.5).astype(float)
    x2, y2 = np.r_[x, [9, 9.1, 9.2]], np.r_[noise, [1, 1, 1]]
    assert rel.isotonic_predict(rel.isotonic_fit(x2, y2), [9.5])[0] == 1.0
    edges2, values2 = rel.isotonic_fit(x2, y2, min_rows=100)
    top = x2 > edges2[-2]
    assert top.sum() >= 100 and values2[-1] == pytest.approx(float(y2[top].mean()))
    assert values2[-1] < 0.75


def test_calibration_error_by_hand():
    entry = rel.calibration_error(np.array([0.05, 0.15, 0.15]), np.array([0.0, 1.0, 0.0]), 10)
    assert entry["ece"] == pytest.approx((0.05 + 2 * 0.35) / 3)
    assert entry["mce"] == pytest.approx(0.35)
    assert entry["table"][1]["rows"] == 2 and entry["table"][9]["rows"] == 0
    seen = rel.probability_range(np.array([0.2, 0.5, 0.95]), np.array([-1.0, 1.0, -1.0]), 0.9)
    assert seen["at_target_rows"] == 1 and seen["at_target_precision"]["hits"] == 0
    assert seen["max_confidence"] == pytest.approx(0.95)


def _bands(rng, sessions=60, rows=200, scale=(1.0,), over=1.0):
    """Bandas ±1 alrededor de cero; el objetivo tiene la escala de su grupo (ruido normal)."""
    groups = np.tile(np.arange(rows) % len(scale), sessions)
    target = rng.standard_normal(sessions * rows) * np.asarray(scale)[groups]
    lower, upper = np.full(len(target), -over), np.full(len(target), over)
    spans = [("US", 1000 + t, t * rows, (t + 1) * rows) for t in range(sessions)]
    return lower, upper, target, spans, groups


def test_static_adaptive_bands_are_the_split_cqr():
    rng = np.random.default_rng(13)
    lower, upper, target, spans, _ = _bands(rng)
    scores = ac_conformity(lower[:4000], upper[:4000], rng.standard_normal(4000))
    low, high, _ = rel.adaptive_bands(
        lower, upper, target, spans, np.zeros(len(spans)), scores, 0.8, 0.0
    )
    correction = rel.ac.conformal_correction(scores, 0.8)
    assert np.allclose(low, lower - correction) and np.allclose(high, upper + correction)


def ac_conformity(lower, upper, target):
    return rel.ac.conformity(lower, upper, target)


def test_aci_only_learns_from_mature_sessions_and_tracks_the_nominal():
    rng = np.random.default_rng(14)
    # Calibración con ruido 1 y evaluación con ruido 1,5: la CQR estática se queda corta.
    lower, upper, target, spans, _ = _bands(rng, scale=(1.5,))
    scores = rel.ac.conformity(np.full(5000, -1.0), np.full(5000, 1.0), rng.standard_normal(5000))
    never = np.full(len(spans), np.iinfo(np.int64).max)
    static = rel.adaptive_bands(lower, upper, target, spans, never, scores, 0.8, 0.0)
    frozen = rel.adaptive_bands(lower, upper, target, spans, never, scores, 0.8, 0.05)
    assert np.array_equal(static[0], frozen[0])
    soon = np.array([instant + 1 for _, instant, _, _ in spans])
    low, high, path = rel.adaptive_bands(lower, upper, target, spans, soon, scores, 0.8, 0.05)
    covered = lambda lo, hi: float(((target >= lo) & (target <= hi)).mean())  # noqa: E731
    assert covered(*static[:2]) < 0.7
    assert abs(covered(low, high) - 0.8) < abs(covered(*static[:2]) - 0.8) / 2
    assert path["alpha_min"] < 0.2


def test_mondrian_bands_cover_each_group():
    rng = np.random.default_rng(15)
    lower, upper, target, spans, groups = _bands(rng, scale=(0.5, 2.0))
    cal_groups = np.arange(8000) % 2
    cal_target = rng.standard_normal(8000) * np.where(cal_groups == 0, 0.5, 2.0)
    scores = rel.ac.conformity(np.full(8000, -1.0), np.full(8000, 1.0), cal_target)
    maturity = np.zeros(len(spans))
    pooled = rel.adaptive_bands(lower, upper, target, spans, maturity, scores, 0.8, 0.0)
    mondrian = rel.adaptive_bands(
        lower,
        upper,
        target,
        spans,
        maturity,
        scores,
        0.8,
        0.0,
        groups=groups,
        score_groups=cal_groups,
    )
    for band, close in ((pooled, False), (mondrian, True)):
        inside = (target >= band[0]) & (target <= band[1])
        rates = [float(inside[groups == g].mean()) for g in (0, 1)]
        assert all(abs(r - 0.8) < 0.03 for r in rates) is close


def test_the_interval_rule_of_r1():
    rule = declaration()["adaptive_intervals"]["rule"]
    good = dict(
        coverage=0.81, volatility={"0": dict(coverage=0.76)}, market={"US": dict(coverage=0.81)}
    )
    assert rel.interval_rule(good, 0.8, rule) == []
    total = dict(good, coverage=0.831)
    assert len(rel.interval_rule(total, 0.8, rule)) == 1
    decile = dict(good, volatility={"0": dict(coverage=0.74), "1": dict(coverage=0.9)})
    assert rel.interval_rule(decile, 0.8, rule) == ["decil de volatilidad con 0.740"]


def test_the_block_interval_widens_with_dependence_and_is_reproducible():
    settings = declaration()["reliability"]["block_interval"]
    rng = np.random.default_rng(16)
    keys = [("US", t) for t in range(250)]
    independent = (rng.random(250) < 0.7).astype(float)
    first = rel.block_interval(independent, keys, settings)
    assert first == rel.block_interval(independent, keys, settings)
    wilson = rel.wilson(int(independent.sum()), 250)
    ratio = (first["high"] - first["low"]) / (wilson["high"] - wilson["low"])
    assert 0.7 < ratio < 1.4
    runs = np.repeat(np.arange(10) % 2, 25).astype(float)
    block, flat = rel.block_interval(runs, keys, settings), rel.wilson(int(runs.sum()), 250)
    assert block["high"] - block["low"] > 1.5 * (flat["high"] - flat["low"])
    two = rel.block_interval(
        np.r_[np.ones(10), np.zeros(10)],
        [("US", t) for t in range(10)] + [("CN", t) for t in range(10)],
        settings,
    )
    assert two["rate"] == 0.5 and two["low"] <= 0.5 <= two["high"]
    assert rel._claim(dict(rate=0.95, low=0.91), 0.9, dict(low=0.88)) == "compatible"
    assert rel._claim(dict(rate=0.95, low=0.91), 0.9, dict(low=0.92)) == "demostrado"


def test_clustered_precision_by_hand():
    entry = rel.clustered_precision([9, 1], [10, 10])
    assert entry["rate"] == 0.5
    # Residuos 4 y −4: se = √(2 · 32) / 20 = 0,4.
    assert entry["se"] == pytest.approx(0.4)
    assert rel.clustered_precision([0, 0], [0, 0]) is None
    assert rel.clustered_precision([3], [4])["se"] is None


def _risk_panel(rng, sessions=40, rows=50):
    """Probabilidades con dos zonas: confianza alta y casi siempre acertada, y baja al azar."""
    probability, target, session = [], [], []
    for s in range(sessions):
        p = np.where(rng.random(rows) < 0.3, 0.97, 0.6)
        right = rng.random(rows) < np.where(p > 0.9, 0.97, 0.6)
        probability += list(p)
        target += list(np.where(right, 1.0, -1.0))
        session += [s] * rows
    return np.array(probability), np.array(target), np.array(session)


def test_the_risk_control_picks_the_lowest_valid_threshold_with_both_tests():
    rng = np.random.default_rng(2)
    probability, target, session = _risk_panel(rng)
    thresholds = [0.55, 0.9, 0.95]
    chosen, tests = rel.risk_tests(
        probability, target, session, 40, thresholds, precision=0.9, delta=0.05
    )
    assert chosen == 0.9
    assert [t["valid"] for t in tests] == [False, True, True]
    high = tests[1]
    errors = high["acted"] - round(high["precision"] * high["acted"])
    assert high["binomial_p"] == pytest.approx(binom.cdf(errors, high["acted"], 0.1))
    # Con pocas filas, aunque todas acierten, la binomial no llega al nivel de Bonferroni.
    few, few_tests = rel.risk_tests(
        np.full(10, 0.99), np.ones(10), np.arange(10), 10, thresholds, precision=0.9, delta=0.05
    )
    assert few is None and few_tests[0]["binomial_p"] == pytest.approx(0.9**10)


def test_a_precision_made_of_a_few_whole_sessions_fails_the_clustered_test():
    # 47 sesiones aciertan todo y 3 lo fallan todo: precisión 0,94 en 5.000 filas. La binomial,
    # que trata las filas como independientes, pasa; por conglomerados de sesión, no.
    session = np.repeat(np.arange(50), 100)
    target = np.where(session < 47, 1.0, -1.0)
    chosen, tests = rel.risk_tests(
        np.full(5000, 0.99), target, session, 50, [0.9], precision=0.9, delta=0.05
    )
    assert tests[0]["binomial_p"] < 1e-6 and tests[0]["clustered_p"] > 0.05
    assert chosen is None


def _view(rng, sessions, rows, strength, *, base=0, markets=("US",)):
    score, target, bounds, keys = [], [], [], []
    # Como en las tablas alineadas, las filas van por mercado y, dentro de él, por instante.
    for market in markets:
        for t in range(sessions):
            s = rng.standard_normal(rows)
            y = strength * s + rng.standard_normal(rows)
            start = len(score)
            score += list(s)
            target += list(y)
            bounds.append((start, start + rows))
            keys.append((market, base + t))
    n = len(score)
    return dict(
        score=np.array(score),
        target=np.array(target),
        bounds=bounds,
        keys=keys,
        liquidity=np.ones(n),
        alpha=np.zeros(n),
        outlier=np.zeros(n, dtype=bool),
    )


def test_nothing_in_f2_is_chosen_with_the_evaluation():
    rng = np.random.default_rng(4)
    settings = declaration()["directional"]
    cal = _view(rng, 30, 60, 3.0, markets=("US", "CN"))
    ev = _view(rng, 20, 60, 3.0, base=100)
    first = rel.directional(cal, ev, settings)
    shuffled = dict(ev, target=rng.permutation(ev["target"]))
    second = rel.directional(cal, shuffled, settings)
    assert first["chosen_threshold"] == second["chosen_threshold"]
    assert first["risk_tests"] == second["risk_tests"]
    assert first["calibrator"] == second["calibrator"]
    assert first["chosen_threshold"] is not None
    at = first["evaluation"]["at_threshold"]
    assert at["coverage"] > 0 and at["precision"]["rate"] > 0.85


def test_the_calibrator_sees_only_the_first_half_in_time():
    rng = np.random.default_rng(6)
    settings = declaration()["directional"]
    cal = _view(rng, 20, 50, 1.0, markets=("US", "CN"))
    ev = _view(rng, 5, 50, 1.0, base=100)
    reference = rel.directional(cal, ev, settings)
    late = cal["target"].copy()
    for (start, end), (_, instant) in zip(cal["bounds"], cal["keys"], strict=True):
        if instant >= 10:  # segunda mitad en el tiempo, que en el orden de filas va intercalada
            late[start:end] = -late[start:end]
    changed = rel.directional(dict(cal, target=late), ev, settings)
    assert changed["calibrator"] == reference["calibrator"]
    assert np.array_equal(
        [
            c["precision"]["rate"] if c["precision"] else None
            for c in changed["evaluation"]["curve"]
        ],
        [
            c["precision"]["rate"] if c["precision"] else None
            for c in reference["evaluation"]["curve"]
        ],
    )
    assert changed["risk_tests"] != reference["risk_tests"]


def test_without_signal_no_threshold_passes_and_the_coverage_is_zero():
    rng = np.random.default_rng(9)
    settings = declaration()["directional"]
    result = rel.directional(_view(rng, 30, 40, 0.0), _view(rng, 10, 40, 0.0, base=50), settings)
    assert result["chosen_threshold"] is None and result["evaluation"]["at_threshold"] is None


def test_restricted_views_drop_rows_and_empty_sessions():
    rng = np.random.default_rng(1)
    view = _view(rng, 3, 4, 1.0)
    keep = np.array([True, False, True, False] + [False] * 4 + [False, True, True, True])
    small = rel.restricted(view, keep)
    assert small["bounds"] == [(0, 2), (2, 5)] and small["keys"] == [
        view["keys"][0],
        view["keys"][2],
    ]
    assert np.array_equal(small["score"], view["score"][keep])


def test_the_alpha_cap_uses_calibration_alone():
    settings = declaration()["reliability"]
    alpha = dict(calibration=np.r_[np.linspace(-1, 1, 101), np.nan], evaluation=np.full(5, 1e6))
    cap = rel.alpha_cap(alpha, settings)
    assert cap == pytest.approx(np.quantile(np.abs(np.linspace(-1, 1, 101)), 0.99))
    assert rel.alpha_cap(dict(alpha, evaluation=np.zeros(5)), settings) == cap
    with pytest.raises(ValueError, match="no hay tope"):
        rel.alpha_cap(dict(calibration=np.full(3, np.nan)), settings)


def test_the_improvement_pairs_the_sessions_that_both_score():
    tool = og._screening()
    keys = [("US", 1), ("US", 2), ("US", 3), ("US", 4)]
    measured = [(0, 0.1, 0.01), (1, 0.2, 0.02), (2, 0.05, 0.0), (3, 0.3, 0.01)]
    baseline = [(0, 0.05, 0.0), (2, 0.1, 0.0), (3, 0.1, 0.02)]
    block = declaration()["reliability"]["block_interval"]
    item = rel.improvement(tool, measured, baseline, keys, [2], block)
    assert item["sessions"] == 3
    assert item["mean"] == pytest.approx((0.05 - 0.05 + 0.2) / 3)
    assert item["positive"]["hits"] == 2
    assert item["spread_mean_bps"] == pytest.approx(1e4 * (0.01 + 0.0 - 0.01) / 3)
    assert item["windows"]["2"]["windows"] == 1
    assert item["positive_block"]["rate"] == pytest.approx(2 / 3)
    assert rel.improvement(tool, measured, [], keys, [2], block)["sessions"] == 0


def _summary_for_verdicts():
    def f2(chosen, coverage, precision, low, sessions=40):
        at = None
        if chosen is not None:
            at = dict(
                coverage=coverage,
                sessions_with_action=sessions,
                precision=dict(rate=precision, low=precision - 0.02, high=precision + 0.02),
                clustered=dict(rate=precision, low=low, high=precision + 0.03),
            )
        seen = dict(rows=100, min=0.4, max=0.6, max_confidence=0.6, at_target_rows=0)
        part = dict(range=seen, calibration_error=dict(ece=0.01))
        return dict(
            chosen_threshold=chosen,
            evaluation=dict(at_threshold=at),
            parts=dict(fit=part, risk=part, evaluation=part),
        )

    return dict(
        f1={
            "operable/s2/all": dict(
                ic_positive=dict(rate=0.95, low=0.91, high=0.97),
                windows={"5": dict(ic=dict(rate=0.92, low=0.85, high=0.97)), "21": dict(ic=None)},
            ),
            "label/s2/all": dict(ic_positive=dict(rate=0.80, low=0.75, high=0.85), windows={}),
        },
        versus_alpha={"s2/all": dict(mean=0.01, t_nw=2.5)},
        f2={"s2/all": f2(0.9, 0.01, 0.93, 0.88), "alpha/all": f2(None, None, None, None)},
        intervals=dict(
            adaptive={
                "0.8": dict(
                    delivered="core_mondrian_aci",
                    confirmed=False,
                    candidates={
                        "core_mondrian_aci": dict(reasons=["decil de volatilidad con 0.740"])
                    },
                )
            }
        ),
    )


def test_the_delivery_verdicts_follow_the_declared_rules():
    rules = declaration()["delivery"]
    summary = _summary_for_verdicts()
    verdict = rel.verdicts(summary, rules)
    reading = verdict["f1"]["operable/s2/all"]
    assert reading == dict(
        sessions="demostrado", windows_5="compatible", windows_21="sin datos", over_alpha=True
    )
    assert verdict["f1"]["label/s2/all"] == dict(sessions="no")
    assert verdict["f2"]["enters"] and verdict["f2"]["reasons"] == []
    assert verdict["f2"]["confidence"]["statement"] == "no hay casos al 90 %"
    assert verdict["intervals"] == {
        "0.8": dict(
            delivered="core_mondrian_aci",
            confirmed=False,
            reasons=["decil de volatilidad con 0.740"],
        )
    }
    # Con el intervalo por bloques por debajo de 0,90, la fracción por sesión ya no se demuestra.
    blocked = _summary_for_verdicts()
    blocked["f1"]["operable/s2/all"]["ic_positive_block"] = dict(rate=0.95, low=0.88, high=0.99)
    assert rel.verdicts(blocked, rules)["f1"]["operable/s2/all"]["sessions"] == "compatible"
    summary["versus_alpha"]["s2/all"]["t_nw"] = 1.9
    assert rel.verdicts(summary, rules)["f1"]["operable/s2/all"]["over_alpha"] is False
    summary["f2"]["alpha/all"] = summary["f2"]["s2/all"]
    assert rel.verdicts(summary, rules)["f2"]["reasons"] == [
        "no cubre más que el alfa solo con su propio umbral"
    ]
    summary = _summary_for_verdicts()
    summary["f2"]["s2/all"]["evaluation"]["at_threshold"]["precision"]["rate"] = 0.89
    summary["f2"]["s2/all"]["evaluation"]["at_threshold"]["clustered"]["low"] = 0.8
    summary["f2"]["s2/all"]["evaluation"]["at_threshold"]["coverage"] = 0.0005
    reasons = rel.verdicts(summary, rules)["f2"]["reasons"]
    assert len(reasons) == 3 and "cobertura prácticamente cero" in reasons
    summary["f2"]["s2/all"] = dict(
        summary["f2"]["s2/all"], chosen_threshold=None, evaluation=dict(at_threshold=None)
    )
    assert rel.verdicts(summary, rules)["f2"]["reasons"] == [
        "ningún umbral pasa el control de riesgo en calibración"
    ]


def _arguments(paths, output, *extra):
    arguments = core._arguments(paths, output, *extra)
    arguments[arguments.index("--declaration") + 1] = str(DECLARATION)
    return arguments


def _session_ics(score, target, instants, min_assets=30):
    values = []
    for instant in np.unique(instants):
        rows = instants == instant
        if rows.sum() >= min_assets:
            values.append(spearmanr(score[rows], target[rows])[0])
    return np.array(values)


def test_f1_and_f2_of_an_origin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(core, "ASSETS", 96)
    paths = core.campaign(tmp_path)
    monkeypatch.setattr(rel.ac, "known_rows", core._known(paths))
    output = tmp_path / "fiabilidad"
    assert rel.main(_arguments(paths, output)) == 0
    folder = output / str(core.YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["markets"] == ["US"]
    # Cuatro predictores en la etiqueta y cinco en lo operable (con el alfa solo), por dos alcances.
    assert len(summary["f1"]) == 18 and len(summary["f2"]) == 10
    assert "label/alpha/all" not in summary["f1"] and "alpha/liquid" in summary["f2"]
    # El IC de la etiqueta del padre, recalculado aparte desde su emisión.
    table = pq.read_table(
        paths["state"]
        / "runs"
        / "titans_mac_nogain"
        / str(core.YEAR)
        / "seed-42"
        / "evaluation-predictions.parquet"
    )
    instants = table.column("prediction_at").cast("int64").to_numpy()
    prediction = table.column("prediction").to_numpy()
    target = table.column("target").to_numpy()
    expected = _session_ics(prediction, target, instants)
    label = summary["f1"]["label/parent/all"]
    assert label["sessions"] == len(expected)
    assert label["ic_mean"] == pytest.approx(float(expected.mean()), abs=1e-9)
    # Y el de lo operable, con el factor elegido y el alfa conocido.
    factor = np.float32(summary["operable_factors"]["titans_mac_nogain"])
    known = paths["known"]["alpha"]["evaluation"]
    alpha = np.array([known[s] for s in table.column("sample_id").to_pylist()]).astype(np.float32)
    operable = _session_ics(
        (factor * prediction + alpha).astype(np.float32),
        (target + alpha).astype(np.float32),
        instants,
    )
    assert summary["f1"]["operable/parent/all"]["ic_mean"] == pytest.approx(
        float(operable.mean()), abs=1e-9
    )
    alone = _session_ics(alpha.astype(np.float64), (target + alpha).astype(np.float32), instants)
    assert summary["f1"]["operable/alpha/all"]["ic_mean"] == pytest.approx(
        float(alone.mean()), abs=1e-9
    )
    versus = summary["versus_alpha"]["parent/all"]
    assert versus["sessions"] == len(alone)
    assert versus["mean"] == pytest.approx(float((operable - alone).mean()), abs=1e-9)
    assert set(summary["versus_alpha"]) == {f"{p}/{s}" for p in rel.PREDICTORS for s in rel.SCOPES}
    assert set(summary["verdicts"]["f2"]) == {"system", "enters", "reasons", "confidence"}
    assert summary["verdicts"]["f2"]["system"] == "s2/all"
    liquid = summary["f1"]["operable/s2/liquid"]
    assert liquid["sessions"] == len(expected) and liquid["selective"]["1.0"]["names"] < 96 * 60
    assert summary["alpha_cap"] > 0
    calibration = np.abs(np.array(list(paths["known"]["alpha"]["calibration"].values())))
    used = summary["f2"]["s2/all"]["calibrator"]
    assert used["fit_rows"] + used["risk_rows"] == int((calibration <= summary["alpha_cap"]).sum())
    assert used["fit_rows"] + used["risk_rows"] < len(calibration)
    for item in summary["f2"].values():
        assert (item["chosen_threshold"] is None) == (item["evaluation"]["at_threshold"] is None)
        assert item["calibrator"]["fit_rows"] + item["calibrator"]["risk_rows"] > 0
    levels = summary["intervals"]["levels"]
    assert abs(levels["0.8"]["systems"]["core_cqr"]["coverage"] - 0.8) < 0.05
    assert len(levels["0.8"]["systems"]["core_raw"]["volatility"]) == 10
    assert set(summary["intervals"]["zero_excluding"]) == {"0.8", "0.95"}
    assert (folder / "f1-sesiones.parquet").is_file()
    for split, sessions in (("calibration", 40), ("evaluation", 60)):
        emitted = pq.read_table(folder / f"s2-confianza-{split}.parquet")
        assert emitted.num_rows == sessions * 96
        probability = emitted.column("probability_up").to_numpy()
        assert probability.min() >= 0 and probability.max() <= 1
    adaptive = summary["intervals"]["adaptive"]
    assert set(adaptive) == {"0.8", "0.9", "0.95"}
    for level, entry in adaptive.items():
        assert list(entry["candidates"]) == declaration()["adaptive_intervals"]["candidates"]
        if level in levels:
            static = entry["candidates"]["core_cqr"]["coverage"]
            assert static == pytest.approx(
                levels[level]["systems"]["core_cqr"]["coverage"], abs=1e-12
            )
        assert entry["delivered"] == "core_mondrian_aci"
        assert entry["confirmed"] == entry["candidates"]["core_mondrian_aci"]["passes"]
    # El 90 % sale de la misma banda que el 80 %, más ancha.
    assert (
        adaptive["0.9"]["candidates"]["core_cqr"]["width"]
        > adaptive["0.8"]["candidates"]["core_cqr"]["width"]
    )
    assert set(summary["verdicts"]["intervals"]) == {"0.8", "0.9", "0.95"}
    assert summary["f1"]["label/s2/all"]["ic_positive_block"]["low"] <= 1.0
    assert "casos al 90" in summary["verdicts"]["f2"]["confidence"]["statement"]
    report = (folder / "resumen.md").read_text(encoding="utf-8")
    assert "F2, confianza direccional" in report
    # Lo primero del informe es lo que se lee como «90 % de fiabilidad».
    assert report.split("\n## ")[1].startswith("Lo que se lee como «90 % de fiabilidad»")
    assert "| 90 % |" in report and "D10" in report
    (folder / "resumen.md").unlink()
    assert rel.main(["informe", "--summary", str(folder / "resumen.json")]) == 0
    assert (folder / "resumen.md").read_text(encoding="utf-8") == report
    assert [p.name for p in output.iterdir()] == [str(core.YEAR)]
    capsys.readouterr()
    assert rel.main(_arguments(paths, output)) == 0
    assert "ya calculado" in capsys.readouterr().out


def test_a_joint_state_needs_its_market_and_measures_only_that_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(core, "ASSETS", 96)
    paths = core.campaign(tmp_path, markets=("US", "CN"))
    monkeypatch.setattr(rel.ac, "known_rows", core._known(paths))
    output = tmp_path / "fiabilidad"
    assert rel.main(_arguments(paths, output)) == 1
    assert "mercado" in capsys.readouterr().err
    assert rel.main(_arguments(paths, output, "--market", "CN")) == 0
    summary = json.loads((output / f"{core.YEAR}-CN" / "resumen.json").read_text(encoding="utf-8"))
    assert summary["markets"] == ["CN"]


def test_without_the_trees_it_waits_and_exits_with_code_75(tmp_path, capsys):
    paths = core.campaign(tmp_path, tree=False)
    assert rel.main(_arguments(paths, tmp_path / "f", "--wait-hours", "0")) == og.MISSING_EXIT
    assert "lightgbm_technical" in capsys.readouterr().err
