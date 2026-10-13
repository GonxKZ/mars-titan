"""Contrastes secundarios de capacidad predictiva frente a ``arch`` y ``statsmodels``.

Las pérdidas son series sintéticas escritas en cada prueba. No proceden de ningún modelo
ajustado y no se ejecuta ningún paso de optimizador. Cada envoltorio se compara con la
llamada directa a la biblioteca o con la fórmula escrita a mano.
"""

import copy
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
from arch.bootstrap import MCS, SPA, RealityCheck, StepM, optimal_block_length
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import diebold_mariano_test

from mars_titan.evaluation import predictive_ability as pa
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.paired_comparisons import delta, interaction, level

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
JOINT = ROOT / "configs/evaluation/historical-masked-2000-joint-comparison.json"
RESAMPLING = dict(block_length=5, replicates=300, seed=1)


def campaign():
    document = json.loads(CAMPAIGN.read_text())
    return document["predictive_ability"], document["comparison"]


def gamma_losses(seed, days, shifts, scale=0.3):
    """Pérdidas diarias positivas con un desplazamiento fijo por modelo."""
    rng = np.random.default_rng(seed)
    common = rng.gamma(2.0, 0.5, days)
    noise = rng.normal(0.0, scale, (days, len(shifts)))
    return np.abs(common[:, None] + noise + np.asarray(shifts))


# Declaración


def test_the_horizon_is_the_one_of_the_study_target():
    target = tomllib.loads((ROOT / "configs/study.toml").read_text())["target"]
    assert target["horizon_sessions"] == pa.HORIZON_SESSIONS == 1


def test_both_campaign_comparisons_declare_the_same_section():
    joint = json.loads(JOINT.read_text())
    section, comparison = campaign()
    assert joint["predictive_ability"] == section
    assert pa.declaration(section, comparison) is section
    assert section["metrics"] == ["mae"] and section["min_days"] == 250
    assert section["declared_at"] == "2026-10-10"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s.update(extra=1), "exactamente"),
        (lambda s: s.pop("hac_lags"), "exactamente"),
        (lambda s: s.update(kind="other"), "protocolo"),
        (lambda s: s.update(superior_predictive_ability="studentized"), "protocolo"),
        (lambda s: s.update(horizon_sessions=5), "protocolo"),
        (lambda s: s.update(declared_at="2026-13-01"), "fecha"),
        (lambda s: s.update(metrics=["mse", "mae"]), "MAE"),
        (lambda s: s.update(metrics=["mae", "direction_accuracy"]), "pérdidas"),
        (lambda s: s.update(metrics=["mae", "interval_score@0.5"]), "pérdidas"),
        (lambda s: s.update(min_days=15), "bloque"),
        (lambda s: s.update(min_days=250.0), "bloque"),
        (lambda s: s.update(stepm_size=0.6), "StepM"),
        (lambda s: s.update(stepm_size=0), "StepM"),
        (lambda s: s.update(model_confidence_set=dict(size=0.1, method="Q")), "MCS"),
        (lambda s: s.update(model_confidence_set=dict(size=1.0, method="R")), "MCS"),
    ],
)
def test_the_declaration_rejects_any_other_rule(change, message):
    section, comparison = campaign()
    section = copy.deepcopy(section)
    change(section)
    with pytest.raises(ValueError, match=message):
        pa.declaration(section, comparison)


def test_the_minimum_needs_three_days_even_with_one_day_blocks():
    """El umbral consistente usa log(log(días)), que no es positivo con dos días o menos."""
    section, comparison = campaign()
    comparison = dict(comparison, block_length=1)
    with pytest.raises(ValueError, match="al menos 3"):
        pa.declaration(dict(section, min_days=2), comparison)
    assert pa.declaration(dict(section, min_days=3), comparison)["min_days"] == 3
    assert math.log(math.log(2)) < 0 < math.log(math.log(3))


def test_the_minimum_must_exceed_the_block_length():
    section, comparison = campaign()
    comparison = dict(comparison, block_length=300)
    with pytest.raises(ValueError, match="bloque"):
        pa.declaration(section, comparison)
    assert pa.declaration(dict(section, min_days=301), comparison)["min_days"] == 301


# Pérdidas diarias


def series(values, *, defined=None, metric="mae"):
    """Cinco sesiones en tres días UTC: el día 1 tiene dos de US y una de CN."""
    values = np.asarray(values, dtype=np.float64)
    defined = np.ones(len(values), dtype=bool) if defined is None else np.asarray(defined)
    return SessionSeries(
        metric,
        True,
        "cohort",
        ("US", "CN"),
        np.array([0, 0, 0, 1, 1], dtype=np.int64),
        np.array([0, 1, 1, 1, 2], dtype=np.int64),
        np.where(defined, values, 0.0),
        defined,
    )


def test_session_weighting_averages_the_sessions_of_each_day():
    losses, excluded = pa.daily_losses(
        {"a": series([1, 2, 4, 6, 3]), "b": series([0, 1, 1, 1, 9])}, "session"
    )
    assert excluded == 0
    np.testing.assert_allclose(losses, [[1, 0], [4, 1], [3, 9]], rtol=0, atol=1e-15)


def test_market_weighting_averages_markets_present_in_each_day():
    losses, _ = pa.daily_losses(
        {"a": series([1, 2, 4, 6, 3]), "b": series([0, 1, 1, 1, 9])}, "market"
    )
    # Día 1: US = (2 + 4) / 2 = 3 y CN = 6, así que (3 + 6) / 2 = 4,5.
    np.testing.assert_allclose(losses[:, 0], [1, 4.5, 3], rtol=0, atol=1e-15)


def test_a_day_without_defined_sessions_in_one_arm_leaves_the_comparison():
    defined = [True, True, True, True, False]
    losses, excluded = pa.daily_losses(
        {"a": series([1, 2, 4, 6, 3]), "b": series([0, 1, 1, 1, 9], defined=defined)}, "session"
    )
    assert excluded == 1 and losses.shape == (2, 2)
    # Una sesión no definida dentro de un día con otras no lo excluye.
    defined = [True, False, True, True, True]
    losses, excluded = pa.daily_losses(
        {"a": series([1, 2, 4, 6, 3]), "b": series([0, 1, 1, 1, 9], defined=defined)}, "session"
    )
    assert excluded == 0 and losses[1, 1] == 1.0


def test_daily_losses_need_the_same_loss_on_the_same_sessions():
    with pytest.raises(ValueError, match="mismas sesiones"):
        pa.daily_losses({"a": series([1] * 5), "b": series([1] * 5, metric="mse")}, "session")
    with pytest.raises(ValueError, match="al menos dos"):
        pa.daily_losses({"a": series([1] * 5)}, "session")
    with pytest.raises(ValueError, match="Ponderación"):
        pa.daily_losses({"a": series([1] * 5), "b": series([2] * 5)}, "asset")


# Diebold-Mariano


def test_hac_lags_use_the_integer_ceiling_of_the_cube_root():
    assert [pa.hac_lags(days) for days in (1, 2, 8, 9, 26, 27, 28, 64, 65, 1000)] == [
        1,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        5,
        10,
    ]
    assert pa.hac_lags(5, horizon=9) == 8


def test_hac_lags_equal_the_statsmodels_default_rule():
    rng = np.random.default_rng(14)
    for days in (3, 7, 8, 26, 27, 28, 125, 250, 251, 4800):
        differential = rng.normal(size=days)
        default = diebold_mariano_test(np.zeros(days), differential, np.zeros(days))
        assert default.lags == pa.hac_lags(days)


def test_diebold_mariano_matches_newey_west_with_harvey_by_hand():
    losses = gamma_losses(4, 61, [0.0, 0.08])
    variant, base = losses[:, 1], losses[:, 0]
    result = pa.diebold_mariano(variant, base)
    days, lags = 61, pa.hac_lags(61)
    centred = (variant - base) - np.mean(variant - base)
    gamma = [centred[j:] @ centred[: days - j] / days for j in range(lags + 1)]
    long_run = gamma[0] + 2 * sum((1 - j / (lags + 1)) * gamma[j] for j in range(1, lags + 1))
    harvey = math.sqrt((days + 1 - 2) / days)
    statistic = harvey * np.mean(variant - base) / math.sqrt(long_run / days)
    assert result["lags"] == lags == 4
    assert result["statistic"] == pytest.approx(statistic, rel=1e-12)
    assert result["harvey_factor"] == pytest.approx(harvey, rel=1e-15)
    assert result["statistic"] > 0  # la variante pierde 0,08 más cada día
    reference = diebold_mariano_test(
        np.zeros(days), variant, base, lags=lags, criterion="mae", harvey_adj=True
    )
    assert result["pvalue"] == pytest.approx(float(reference.pvalue), rel=1e-12)
    assert result["reference_distribution"] == "student_t_60_degrees_of_freedom"


def test_diebold_mariano_reports_a_constant_differential_instead_of_dividing_by_zero():
    base = gamma_losses(5, 40, [0.0])[:, 0]
    for variant in (base.copy(), base + 0.25):
        result = pa.diebold_mariano(variant, base)
        assert "statistic" not in result and "constante" in result["reason"]
    with pytest.raises(ValueError, match="alineadas"):
        pa.diebold_mariano(base[:2], base[:2])


def test_holm_adjusts_only_the_tested_rows_within_the_family():
    rows = [dict(pvalue=0.01), dict(reason="constante"), dict(pvalue=0.04), dict(pvalue=0.03)]
    pa.holm(rows)
    expected = multipletests([0.01, 0.04, 0.03], method="holm")[1]
    assert [row.get("holm_pvalue") for row in rows] == [expected[0], None, expected[1], expected[2]]


# SPA, Reality Check y StepM


def test_spa_and_reality_check_are_the_arch_pvalues_without_studentizing():
    losses = gamma_losses(6, 400, [0.0, -0.05, 0.02, -0.01])
    base, models = losses[:, 0], losses[:, 1:]
    result = pa.superior_predictive_ability(base, models, size=0.05, **RESAMPLING)
    spa = SPA(base, models, block_size=5, reps=300, bootstrap="circular", studentize=False, seed=1)
    spa.compute()
    assert result["pvalues"] == {key: float(value) for key, value in spa.pvalues.items()}
    check = RealityCheck(base, models, block_size=5, reps=300, bootstrap="circular", seed=1)
    check.compute()
    assert result["reality_check_pvalue"] == float(check.pvalues["upper"])
    assert result["pvalues"]["lower"] <= result["pvalues"]["consistent"]
    assert result["pvalues"]["consistent"] <= result["pvalues"]["upper"]


def test_arch_8_spa_ignores_studentize_which_is_why_the_section_declares_it_off():
    """Si una versión nueva de arch estudentiza, esta prueba falla y obliga a revisar."""
    losses = gamma_losses(7, 300, [0.0, -0.03, 0.01])
    base, models = losses[:, 0], losses[:, 1:]
    pvalues = []
    for studentize in (True, False):
        spa = SPA(
            base,
            models,
            block_size=5,
            reps=200,
            bootstrap="circular",
            studentize=studentize,
            seed=2,
        )
        spa.compute()
        pvalues.append(spa.pvalues.to_dict())
    assert pvalues[0] == pvalues[1]


def test_stepm_matches_arch_when_arch_does_not_fail():
    losses = gamma_losses(8, 400, [0.0, -0.3, -0.25, 0.05])
    base, models = losses[:, 0], losses[:, 1:]
    result = pa.superior_predictive_ability(base, models, size=0.05, **RESAMPLING)
    step = StepM(
        base,
        models,
        size=0.05,
        block_size=5,
        reps=300,
        bootstrap="circular",
        studentize=False,
        seed=1,
    )
    step.compute()
    assert result["stepm_superior"] == [int(i) for i in step.superior_models] == [0, 1]


def stepped_case(trial):
    """Una mejora grande y muy variable y otra pequeña y estable sobre la misma base."""
    rng = np.random.default_rng(trial)
    base = 1.0 + rng.normal(0, 0.05, 300)
    large = base - 0.5 + rng.normal(0, 3.0, 300)
    small = base - 0.06 + rng.normal(0, 0.05, 300)
    return base, np.column_stack([large, small])


def test_stepm_selects_every_model_when_arch_8_fails_across_steps():
    base, models = stepped_case(0)
    spa = SPA(base, models, block_size=5, reps=300, bootstrap="circular", studentize=False, seed=1)
    spa.compute()
    assert list(spa.better_models(0.05)) == [0]  # el primer paso solo ve la mejora grande
    step = StepM(
        base,
        models,
        size=0.05,
        block_size=5,
        reps=300,
        bootstrap="circular",
        studentize=False,
        seed=1,
    )
    with pytest.raises(ValueError, match="zero-size"):
        step.compute()  # bashtage/arch#862, corregido después de la versión 8.0.0
    result = pa.superior_predictive_ability(base, models, size=0.05, **RESAMPLING)
    assert result["stepm_superior"] == [0, 1]


def test_stepm_restores_every_model_after_the_steps():
    base, models = stepped_case(1)
    spa = SPA(base, models, block_size=5, reps=300, bootstrap="circular", studentize=False, seed=1)
    spa.compute()
    before = spa.pvalues.to_dict()
    assert pa.stepwise_superior(spa, 0.05) == [0, 1]
    assert spa.pvalues.to_dict() == before
    assert list(spa.better_models(0.05)) == [0]


def test_spa_reports_a_variant_identical_to_the_base():
    losses = gamma_losses(9, 100, [0.0, -0.1])
    models = np.column_stack([losses[:, 1], losses[:, 0]])
    result = pa.superior_predictive_ability(losses[:, 0], models, size=0.05, **RESAMPLING)
    assert set(result) == {"reason"} and "constante" in result["reason"]
    with pytest.raises(ValueError, match="más días que el bloque"):
        pa.superior_predictive_ability(losses[:5, 0], losses[:5, 1:], size=0.05, **RESAMPLING)


# MCS


@pytest.mark.parametrize("method", ["R", "max"])
def test_the_model_confidence_set_is_the_arch_one(method):
    losses = gamma_losses(10, 500, [0.0, 0.01, 0.2, 0.25, 0.0])
    result = pa.model_confidence_set(losses, size=0.1, method=method, **RESAMPLING)
    mcs = MCS(losses, size=0.1, reps=300, block_size=5, method=method, bootstrap="circular", seed=1)
    mcs.compute()
    assert result["included"] == sorted(int(i) for i in mcs.included)
    table = mcs.pvalues
    assert result["elimination_order"] == [int(i) for i in table.index]
    assert result["pvalues"] == dict(zip(map(int, table.index), table["Pvalue"], strict=True))
    assert {2, 3}.isdisjoint(result["included"]) and {0, 4} <= set(result["included"])
    # Los p-valores del MCS no bajan a lo largo del orden de eliminación.
    ordered = [result["pvalues"][i] for i in result["elimination_order"]]
    assert ordered == sorted(ordered)


def test_the_model_confidence_set_refuses_two_identical_arms():
    losses = gamma_losses(11, 80, [0.0, 0.1])
    losses = np.column_stack([losses, losses[:, 0]])
    result = pa.model_confidence_set(losses, size=0.1, method="R", **RESAMPLING)
    assert set(result) == {"reason"}


# Longitud de bloque


def test_the_block_length_diagnostic_is_the_arch_estimate_and_changes_nothing():
    rng = np.random.default_rng(12)
    noise = rng.normal(size=600)
    differential = np.convolve(noise, np.ones(8) / 8, mode="same")  # autocorrelación de 8 días
    result = pa.block_length_diagnostic(differential, 16)
    table = optimal_block_length(differential)
    assert result["circular"] == float(table["circular"].iloc[0])
    assert result["stationary"] == float(table["stationary"].iloc[0])
    assert result["declared"] == 16
    assert result["declared_below_circular"] is (16 < result["circular"])
    assert "reason" in pa.block_length_diagnostic(np.full(50, 0.3), 16)
    assert "corta" in pa.block_length_diagnostic(rng.normal(size=10), 16)["reason"]


# Familias e informe


def test_family_structure_follows_the_declared_kinds():
    single = pa.family_structure({"b-a": delta("a", "b"), "c-a": delta("a", "c")})
    assert single == dict(
        arms=["b", "a", "c"], pairs=[("b-a", "b", "a"), ("c-a", "c", "a")], base="a", levels=False
    )
    factorial = pa.family_structure(
        {
            "c-b": delta("b", "c"),
            "m-b": delta("b", "m"),
            "cm-b": delta("b", "cm"),
            "interaction": interaction("b", "c", "m", "cm"),
        }
    )
    assert factorial["base"] == "b" and len(factorial["pairs"]) == 3
    assert set(factorial["arms"]) == {"b", "c", "m", "cm"}
    levels = pa.family_structure({arm: level(arm) for arm in ("x", "y", "z")})
    assert levels == dict(arms=["x", "y", "z"], pairs=[], base=None, levels=True)
    paired = pa.family_structure({"aj-a": delta("a", "aj"), "bj-b": delta("b", "bj")})
    assert paired["base"] is None and not paired["levels"] and len(paired["pairs"]) == 2


def day_series(values, metric="mae"):
    """Una sesión por día, de modo que la pérdida diaria es el propio valor."""
    days = len(values)
    return SessionSeries(
        metric,
        True,
        "cohort",
        ("US",),
        np.zeros(days, dtype=np.int64),
        np.arange(days, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        np.ones(days, dtype=bool),
    )


def report_case():
    section, comparison = campaign()
    section = dict(section, min_days=40)
    comparison = dict(comparison, block_length=5, replicates=200, seed=3)
    losses = gamma_losses(13, 60, [0.0, -0.1, 0.05, 0.02])
    arms = {name: losses[:, i] for i, name in enumerate(("zero", "ridge", "gru", "titans"))}
    families = {
        "references": {"ridge-zero": delta("zero", "ridge"), "gru-zero": delta("zero", "gru")},
        "levels": {arm: level(arm) for arm in ("ridge", "gru", "titans")},
        "pairs": {"gru-ridge": delta("ridge", "gru"), "titans-zero": delta("zero", "titans")},
    }
    return section, comparison, arms, families


def test_the_report_runs_each_analysis_where_it_applies():
    section, comparison, arms, families = report_case()

    def provider(arm, view, metric):
        assert (view, metric) == ("US", "mae")
        return day_series(arms[arm])

    result = pa.report(section, comparison, "session", families, ["US"], provider)
    assert result["resampling"] == dict(method="circular", block_length=5, replicates=200, seed=3)
    rows = result["views"]["US"]["mae"]
    references = rows["references"]
    assert references["days"] == 60 and references["excluded_days"] == 0
    assert references["arms"] == ["ridge", "zero", "gru"]
    assert [row["name"] for row in references["diebold_mariano"]] == ["ridge-zero", "gru-zero"]
    assert all("holm_pvalue" in row for row in references["diebold_mariano"])
    spa = references["superior_predictive_ability"]
    assert spa["benchmark"] == "zero" and spa["variants"] == ["ridge", "gru"]
    direct = pa.superior_predictive_ability(
        arms["zero"],
        np.column_stack([arms["ridge"], arms["gru"]]),
        size=0.05,
        block_length=5,
        replicates=200,
        seed=3,
    )
    assert spa["pvalues"] == direct["pvalues"]
    assert spa["stepm_superior"] == [["ridge", "gru"][i] for i in direct["stepm_superior"]]
    mcs = references["model_confidence_set"]
    assert set(mcs["included"]) <= {"ridge", "zero", "gru"} and mcs["size"] == 0.1
    assert rows["levels"]["diebold_mariano"] is None
    assert rows["levels"]["superior_predictive_ability"] is None
    assert rows["levels"]["model_confidence_set"]["method"] == "R"
    assert rows["pairs"]["superior_predictive_ability"] is None
    assert rows["pairs"]["model_confidence_set"] is None
    assert len(rows["pairs"]["diebold_mariano"]) == 2
    assert set(rows["pairs"]["block_length_diagnostic"]) == {"gru-ridge", "titans-zero"}
    json.dumps(result, allow_nan=False)


def test_the_report_gives_the_reason_for_short_or_missing_series():
    section, comparison, arms, families = report_case()

    def provider(arm, view, metric):
        return None if arm == "titans" else day_series(arms[arm])

    short = pa.report(dict(section, min_days=61), comparison, "session", families, ["US"], provider)
    rows = short["views"]["US"]["mae"]
    assert "métrica" in rows["levels"]["reason"] and "métrica" in rows["pairs"]["reason"]
    assert "mínimo" in rows["references"]["reason"] and rows["references"]["days"] == 60


def test_a_family_with_exactly_the_minimum_of_days_is_analysed():
    section, comparison, arms, families = report_case()

    def provider(arm, view, metric):
        return day_series(arms[arm])

    exact = pa.report(dict(section, min_days=60), comparison, "session", families, ["US"], provider)
    rows = exact["views"]["US"]["mae"]
    assert all("reason" not in row and row["days"] == 60 for row in rows.values())
