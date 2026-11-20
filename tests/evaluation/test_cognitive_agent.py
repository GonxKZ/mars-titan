"""Agente cognitivo con herramientas: declaración, mezcla causal, controles y familias."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.evaluation import cognitive_agent as agent
from mars_titan.evaluation import variant_screening_analysis as analysis
from mars_titan.evaluation.arbiter_analysis import joined, stage_metric
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.evaluation.online_arbiters import standardized

START = 1_104_786_300_000_000  # 2005-01-03 21:05 UTC


def _panel(prediction, target, sessions, assets, offset):
    session = np.repeat(np.arange(sessions), assets)
    asset = np.tile(np.arange(assets), sessions)
    times = START + (offset + session) * DAY_MICROSECONDS
    ids = pa.array([f"US/A{a:03d}/{t}" for a, t in zip(asset, times, strict=True)])
    return ForecastPanel.from_columns(
        ids, np.array(["US"] * len(times)), times, target, prediction, markets=("US",)
    )


def world(seed=0, calibration=60, evaluation=80, assets=40, target_changes=None):
    """Un núcleo con señal débil, una herramienta con señal fuerte y otra sin señal."""
    rng = np.random.default_rng(seed)
    total = calibration + evaluation
    signal = rng.standard_normal((total, assets))
    target = 0.01 * (0.3 * signal + rng.standard_normal((total, assets)))
    if target_changes is not None:
        target = target_changes(target)

    def option(weight, noise_seed):
        noise = np.random.default_rng(noise_seed).standard_normal((total, assets))
        return weight * signal + noise

    emissions = {}
    for name, values in (
        ("core", option(0.3, 1)),
        ("strong", option(1.0, 2)),
        ("noise", option(0.0, 3)),
    ):
        c = _panel(
            values[:calibration].ravel(), target[:calibration].ravel(), calibration, assets, 0
        )
        s = _panel(
            values[calibration:].ravel(),
            target[calibration:].ravel(),
            evaluation,
            assets,
            calibration,
        )
        emissions[name] = analysis.Emission(s, s.prediction, c)
    core = emissions["core"]
    maturity = dict(
        calibration=core.calibration.prediction_at + DAY_MICROSECONDS,
        evaluation=core.panel.prediction_at + DAY_MICROSECONDS,
    )
    return emissions, maturity


def declaration(**changes):
    document = dict(
        schema_version=1,
        kind=agent.KIND,
        final_test_access=False,
        core="core",
        tools={"strong_tool": ["strong"], "noise_tool": ["noise"]},
        feedback=dict(window=20, min_assets=10, margin=None, shuffled_seed=5),
        placebo=dict(seed=9),
        context=None,
        shapley=False,
        metric="session_rank_ic",
        inference=dict(block_days=5, replicates=200, seed=1, confidence=0.95),
        origins=[2005],
        seeds=[42],
    )
    document.update(changes)
    return agent.checked_declaration(document)


def _mean(outcome, metric="session_rank_ic"):
    values, defined = stage_metric(outcome["emission"], outcome["abstained"], metric, min_assets=10)
    return values[defined].mean()


@pytest.mark.parametrize(
    "changes",
    [
        dict(final_test_access=True),
        dict(tools={"core_again": ["core"]}),
        dict(tools={"a": ["strong"], "b": ["strong"]}),
        dict(tools={"placebo": ["strong"]}),
        dict(tools={}),
        dict(metric="session_mae_raw"),
        dict(feedback=dict(window=0, min_assets=10, margin=None, shuffled_seed=5)),
        dict(feedback=dict(window=20, min_assets=10, margin=1, shuffled_seed=5)),
        dict(inference=dict(block_days=5, replicates=10, seed=1, confidence=0.95)),
        dict(origins=[2024]),
        dict(origins=[2006, 2005]),
        dict(seeds=[42, 42]),
        dict(placebo=dict(seed=9, kind="other")),
        dict(context=dict(window=1)),
        dict(context=dict(window=10, axes=["dispersion"])),
        dict(shapley=None),
        dict(shapley=True, tools={f"t{i}": [f"m{i}"] for i in range(agent.MAX_SHAPLEY_TOOLS + 1)}),
    ],
)
def test_declaration_rejects_designs_outside_the_contract(changes):
    with pytest.raises(ValueError):
        declaration(**changes)


def test_the_loop_learns_to_trust_the_informative_tool_and_ignores_the_placebo():
    emissions, maturity = world()
    outcomes = agent.variants(declaration(), emissions, maturity)
    assert _mean(outcomes[agent.AGENT]) > _mean(outcomes[agent.CORE]) + 0.02
    assert _mean(outcomes[agent.AGENT]) > _mean(outcomes[agent.WITHOUT + "strong_tool"]) + 0.02
    assert _mean(outcomes[agent.WITH + "strong_tool"]) > _mean(outcomes[agent.CORE])
    weights = agent.tool_weights(declaration(), outcomes[agent.PLACEBO])["overall"]
    assert weights["strong_tool"] > 0.5
    assert weights["placebo"] < weights["strong_tool"] / 3
    assert set(outcomes) == {
        agent.CORE,
        agent.AGENT,
        agent.UNIFORM,
        agent.SHUFFLED,
        agent.PLACEBO,
        agent.WITHOUT + "strong_tool",
        agent.WITHOUT + "noise_tool",
        agent.WITH + "strong_tool",
        agent.WITH + "noise_tool",
    }


def test_each_session_decides_only_with_labels_that_already_matured():
    emissions, maturity = world()
    calibration, changed = 60, 50

    def future(target):
        target = target.copy()
        target[calibration + changed :] = -target[calibration + changed :]
        return target

    altered, _ = world(target_changes=future)
    base = agent.variants(declaration(), emissions, maturity)[agent.AGENT]
    other = agent.variants(declaration(), altered, maturity)[agent.AGENT]
    rows = emissions["core"].panel.session_starts[changed + 1]
    np.testing.assert_array_equal(
        base["emission"].prediction[:rows], other["emission"].prediction[:rows]
    )
    np.testing.assert_array_equal(base["weights"][: changed + 1], other["weights"][: changed + 1])
    assert not np.array_equal(base["weights"][changed + 2 :], other["weights"][changed + 2 :])


def test_uniform_control_is_the_mean_of_options_tipified_within_each_session():
    emissions, maturity = world()
    options = [emissions["core"], emissions["strong"], emissions["noise"]]
    emission, abstained, weights = agent.mixture(
        options, maturity, declaration()["feedback"], "uniform"
    )
    panel = emissions["core"].panel
    expected = np.mean([standardized(panel, option.prediction) for option in options], axis=0)
    np.testing.assert_allclose(emission.prediction, expected, rtol=0, atol=1e-12)
    assert not abstained.any() and np.allclose(weights, 1 / 3)


def test_a_single_option_is_emitted_unchanged_and_never_abstains():
    emissions, maturity = world()
    emission, abstained, weights = agent.mixture(
        [emissions["core"]], maturity, declaration()["feedback"], "online"
    )
    np.testing.assert_array_equal(emission.prediction, emissions["core"].prediction)
    assert not abstained.any() and weights.shape == (emissions["core"].panel.sessions, 1)


def test_placebo_keeps_each_session_distribution_and_destroys_its_order():
    emissions, _ = world()
    core = emissions["core"]
    placebo = agent.permuted(core, 9)
    for panel, original, shuffled in (
        (core.panel, core.prediction, placebo.prediction),
        (core.calibration, core.calibration.prediction, placebo.calibration.prediction),
    ):
        for start, count in zip(panel.session_starts, panel.session_samples, strict=True):
            np.testing.assert_array_equal(
                np.sort(original[start : start + count]), np.sort(shuffled[start : start + count])
            )
        assert not np.array_equal(original, shuffled)
    assert placebo.panel.cohort_sha256 == core.panel.cohort_sha256


def test_abstention_scores_zero_and_its_variant_is_reported():
    emissions, maturity = world()
    document = declaration(feedback=dict(window=20, min_assets=10, margin=0.99, shuffled_seed=5))
    outcomes = agent.variants(document, emissions, maturity)
    agent_outcome = outcomes[agent.AGENT]
    assert agent_outcome["abstained"].all()
    values, defined = stage_metric(
        agent_outcome["emission"], agent_outcome["abstained"], "session_rank_ic", min_assets=10
    )
    assert defined.any() and np.all(values[defined] == 0.0)
    assert not outcomes[agent.NO_ABSTENTION]["abstained"].any()


def test_families_attribute_the_gain_to_the_informative_tool():
    emissions, maturity = world()
    document = declaration()
    outcomes = agent.variants(document, emissions, maturity)
    series = {
        name: stage_metric(o["emission"], o["abstained"], "session_rank_ic", min_assets=10)
        for name, o in outcomes.items()
    }
    day = np.floor_divide(emissions["core"].panel.session_time, DAY_MICROSECONDS)
    result = agent.families(document, series, day)
    marginal = {row["name"]: row for row in result["marginal_tools"]["members"]}
    assert marginal["agent − agent_without_strong_tool"]["estimate"] > 0
    assert marginal["agent − agent_without_strong_tool"]["passes"]
    system = {row["name"]: row for row in result["system"]["members"]}
    assert system["agent − core"]["passes"]
    assert set(result) == {"marginal_tools", "isolated_tools", "system", "placebo"}


def test_maturity_is_aligned_to_the_canonical_rows_and_checks_the_targets():
    emissions, maturity = world(calibration=5, evaluation=6, assets=12)
    core = emissions["core"]
    tables = {}
    for name, panel in (("calibration", core.calibration), ("evaluation", core.panel)):
        order = np.random.default_rng(3).permutation(panel.rows)
        ids = np.array(panel.row_id.to_pylist())[order]
        tables[name] = pa.table(
            dict(
                asset_id=pa.array(["/".join(i.split("/")[:2]) for i in ids]),
                market=pa.array(["US"] * panel.rows),
                prediction_at=pa.array(panel.prediction_at[order]).cast(
                    pa.timestamp("us", tz="UTC")
                ),
                target=panel.target[order],
                matures_at=maturity[name][order],
            )
        )
    aligned = agent.aligned_maturity(tables, core)
    for name in agent.PARTITIONS:
        np.testing.assert_array_equal(aligned[name], maturity[name])
    broken = dict(tables)
    broken["evaluation"] = tables["evaluation"].set_column(
        3, "target", pa.array(tables["evaluation"]["target"].to_numpy() + 1e-9)
    )
    with pytest.raises(ValueError, match="objetivos"):
        agent.aligned_maturity(broken, core)


def regime_world(seed=0, calibration=120, evaluation=160, assets=40, block=12):
    """Dos herramientas, cada una informativa en un régimen de dispersión distinto."""
    rng = np.random.default_rng(seed)
    total = calibration + evaluation
    high = (np.arange(total) // block) % 2 == 1
    scale = np.where(high, 4.0, 1.0)[:, None]
    signal = rng.standard_normal((total, assets))
    target = 0.01 * scale * (0.5 * signal + rng.standard_normal((total, assets)))

    def option(mask, noise_seed):
        noise = np.random.default_rng(noise_seed).standard_normal((total, assets))
        return np.where(mask[:, None], signal, 0.0) + noise

    emissions = {}
    sources = (
        ("core", option(np.zeros(total, dtype=bool), 1)),
        ("hi", option(high, 2)),
        ("lo", option(~high, 3)),
    )
    for name, values in sources:
        c = _panel(
            values[:calibration].ravel(), target[:calibration].ravel(), calibration, assets, 0
        )
        s = _panel(
            values[calibration:].ravel(),
            target[calibration:].ravel(),
            evaluation,
            assets,
            calibration,
        )
        emissions[name] = analysis.Emission(s, s.prediction, c)
    core = emissions["core"]
    maturity = dict(
        calibration=core.calibration.prediction_at + DAY_MICROSECONDS,
        evaluation=core.panel.prediction_at + DAY_MICROSECONDS,
    )
    return emissions, maturity, high[calibration:]


def test_contextual_routing_uses_each_tool_in_its_own_regime():
    emissions, maturity, high = regime_world()
    document = declaration(
        tools={"hi_tool": ["hi"], "lo_tool": ["lo"]},
        placebo=None,
        context=dict(window=3),
        feedback=dict(window=20, min_assets=10, margin=None, shuffled_seed=5),
    )
    outcomes = agent.variants(document, emissions, maturity)
    contextual = outcomes[agent.CONTEXTUAL]
    assert _mean(contextual) > _mean(outcomes[agent.AGENT]) + 0.01
    weights = agent.tool_weights(document, contextual)["by_regime"]
    high_regimes = [name for name in agent.REGIMES if name.startswith("high")]
    low_regimes = [name for name in agent.REGIMES if name.startswith("low")]
    hi_share = np.mean(
        [weights[n]["shares"]["hi_tool"] for n in high_regimes if weights[n]["shares"]]
    )
    lo_share = np.mean(
        [weights[n]["shares"]["lo_tool"] for n in low_regimes if weights[n]["shares"]]
    )
    assert hi_share > 0.5 and lo_share > 0.5
    np.testing.assert_allclose(contextual["weights"].sum(axis=1), 1.0, rtol=0, atol=1e-12)
    known = contextual["regime"] != agent.UNKNOWN_REGIME
    detected = np.isin(contextual["regime"][known], [2, 3])
    assert np.mean(detected == high[known]) > 0.75


def test_regimes_only_read_matured_sessions():
    emissions, maturity, _ = regime_world()
    core = emissions["core"]
    panel, positions = joined(core.calibration, core.panel)
    matures = np.concatenate([maturity["calibration"], maturity["evaluation"]])[positions]
    session_matures = np.maximum.reduceat(matures, panel.session_starts)
    base = agent.regimes(panel, session_matures, 5)
    assert np.all(base[:5] == agent.UNKNOWN_REGIME) and np.all(base[6:] != agent.UNKNOWN_REGIME)
    cut = 150
    changed = np.array(panel.target)
    changed[panel.session_starts[cut] :] *= -7.0
    altered = agent.regimes(replace(panel, target=changed), session_matures, 5)
    np.testing.assert_array_equal(base[: cut + 1], altered[: cut + 1])


def test_shapley_is_efficient_symmetric_and_null_for_a_useless_tool():
    emissions, maturity = world()
    emissions["twin"] = emissions["strong"]
    document = declaration(
        tools={"strong_tool": ["strong"], "twin_tool": ["twin"], "noise_tool": ["noise"]},
        shapley=True,
        placebo=None,
    )
    outcomes = agent.variants(document, emissions, maturity)
    series = {
        name: stage_metric(o["emission"], o["abstained"], "session_rank_ic", min_assets=10)
        for name, o in outcomes.items()
    }
    values = agent.shapley_values(list(document["tools"]), series)
    full, empty = series[agent.subset_name(document["tools"])], series[agent.subset_name(())]
    defined = values["strong_tool"][1]
    total = sum(values[tool][0] for tool in document["tools"])
    np.testing.assert_allclose(total[defined], (full[0] - empty[0])[defined], rtol=0, atol=1e-12)
    strong, twin = values["strong_tool"][0][defined].mean(), values["twin_tool"][0][defined].mean()
    noise = values["noise_tool"][0][defined].mean()
    assert abs(strong - twin) < 0.25 * abs(strong)
    assert strong > 0.02 and abs(noise) < strong / 4
    assert agent.subset_name(("b", "a")) == agent.subset_name(("a", "b"))


def test_shared_mixtures_are_computed_once_per_options_mode_and_margin(monkeypatch):
    emissions, maturity = world()
    calls = []
    original = agent.mixture

    def counted(options, maturity, feedback, mode):
        calls.append((len(options), mode, feedback["margin"]))
        return original(options, maturity, feedback, mode)

    monkeypatch.setattr(agent, "mixture", counted)
    agent.variants(declaration(shapley=True, placebo=None), emissions, maturity)
    # Núcleo, agente, uniforme, desordenado y una variante por herramienta: los subconjuntos
    # de Shapley sin margen coinciden con ellas y no se recalculan.
    assert len(calls) == 6
