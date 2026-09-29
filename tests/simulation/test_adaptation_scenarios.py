"""Mundos técnicos separados, contexto disponible y cambios de mecanismo comprobables."""

import copy
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.simulation.adaptation_scenarios import (
    FAMILIES,
    cases,
    fit_hmm,
    generate_scenario,
    prepare_adaptation_scenarios,
)
from mars_titan.simulation.portfolio import Instrument, Portfolio
from mars_titan.simulation.storage import read_tape


def settings():
    return dict(
        schema_version=1,
        families=list(FAMILIES),
        generator=dict(assets=4, sessions=32, warmup_sessions=8, regime_sessions=8),
        worlds=dict(train=1, validation=1, audit=1),
        seed_base=20260929,
        final_test_opened=False,
    )


def world(family, **changes):
    config = settings()
    config["families"] = [family]
    config["generator"].update(changes)
    return generate_scenario(cases(config)[0])


def test_catalog_allocates_distinct_seeds_and_keeps_audit_separate():
    catalog = cases(settings())
    assert len(catalog) == 24
    assert len({case["seed"] for case in catalog}) == 24
    assert {case["split"] for case in catalog} == {"train", "validation", "audit"}
    assert all(case["partition"] == "validation" for case in catalog if case["split"] == "audit")


@pytest.mark.parametrize("bad", [True, -1, 2**32, 0.5])
def test_invalid_seed_is_rejected_before_generation(bad):
    config = settings()
    config["seed_base"] = bad
    with pytest.raises(ValueError):
        cases(config)


def test_context_schema_is_shared_and_truth_never_enters_observations():
    generated = [world(family) for family in FAMILIES]
    fields = generated[0]["fields"]
    assert fields[-1] == dict(name="trading_enabled", unit="boolean")
    assert all(item["fields"] == fields for item in generated)
    assert not {"latent_regime", "expected_return", "target"} & {f["name"] for f in fields}
    for item in generated:
        assert np.all(item["context"][:8, -1] == 0)
        assert np.all(item["context"][8:, -1] == 1)
        assert np.all(item["tape"].scores == 0.01)
        assert np.all(item["available_at"] <= item["tape"].close_times[:, None])


@pytest.mark.parametrize("family", FAMILIES)
def test_future_extension_does_not_change_generated_prefix(family):
    before = world(family, sessions=24)
    after = world(family, sessions=32)
    np.testing.assert_array_equal(before["tape"].prices, after["tape"].prices[:24])
    np.testing.assert_array_equal(before["context"], after["context"][:24])


def test_regime_return_relation_reverses_and_recurs():
    item = world("regime_recurrence")
    truth = item["truth"]
    assert truth["relation"][8:16].tolist() == [1] * 8
    assert truth["relation"][16:24].tolist() == [-1] * 8
    assert truth["relation"][24:].tolist() == [1] * 8
    np.testing.assert_array_equal(
        truth["expected_return"][8:],
        0.006 * truth["latent_signal"][8:] * truth["relation"][8:],
    )
    assert np.all(world("no_signal")["truth"]["expected_return"] == 0)


def test_delayed_cue_appears_once_before_trading_without_price_leakage():
    item = world("delayed_cue")
    cue = item["context"][:, 0]
    assert np.flatnonzero(cue).tolist() == [4]
    assert np.all(item["truth"]["expected_return"][:8] == 0)
    assert np.all(cue[8:] == 0)
    assert np.all(item["context"][:8, -1] == 0)


def test_labels_mature_after_decision_and_final_label_is_absent():
    item = world("known_signal")
    tape, truth = item["tape"], item["truth"]
    assert np.all(truth["target_available_at"][:-1] == tape.close_times[1:])
    assert np.isnan(truth["target"][-1])
    np.testing.assert_allclose(
        truth["target"][:-1],
        (tape.prices[1:, :, 3] / tape.prices[1:, :, 0] - 1).mean(axis=1),
    )


def test_signal_affects_the_following_return_not_the_contemporaneous_price():
    config = settings()
    config["families"] = ["known_signal"]
    case = cases(config)[0]
    signal = generate_scenario(case)
    noise = generate_scenario(dict(case, family="no_signal"))
    np.testing.assert_allclose(
        signal["truth"]["target"][:-1] - noise["truth"]["target"][:-1],
        signal["truth"]["expected_return"][:-1],
        rtol=1e-10,
        atol=1e-14,
    )


def test_preparation_round_trips_and_links_every_artifact(tmp_path):
    config = settings()
    config["families"] = ["delayed_cue"]
    output = tmp_path / "prepared"
    index = prepare_adaptation_scenarios(config, output)
    assert len(index["records"]) == 3 and index["analysis_domain"] == "technical"
    assert index["macro_coverage"]["simulated_concepts"] == 3
    assert index["real_corpus_compatible"] is False
    for record in index["records"]:
        folder = output / record["path"]
        tape = read_tape(folder)
        assert record["manifest_sha256"] == sha256(folder / "manifest.json")
        context = json.loads((folder / "context.json").read_text())
        assert context["market_manifest_sha256"] == record["manifest_sha256"]
        assert context["file"]["sha256"] == sha256(folder / "context.parquet")
        table = pq.read_table(folder / "context.parquet")
        assert table.num_rows == 32 * len(context["fields"])
        assert all(column.null_count == 0 for column in table.columns)
        assert tape.identity["source"]["generator"]["seed"] == record["seed"]
        assert record["truth_sha256"] == sha256(folder / "evaluator/truth.parquet")
    previous = (output / "index.json").read_bytes()
    with pytest.raises(ValueError):
        prepare_adaptation_scenarios(config, output)
    assert (output / "index.json").read_bytes() == previous


def test_bad_counts_and_unknown_families_fail_without_output(tmp_path):
    for changed in (
        {"families": ["unknown"]},
        {"worlds": {"train": 0, "validation": 1, "audit": 1}},
    ):
        config = copy.deepcopy(settings())
        config.update(changed)
        with pytest.raises(ValueError):
            prepare_adaptation_scenarios(config, tmp_path / "bad")
        assert not (tmp_path / "bad").exists()


def test_default_catalog_has_the_declared_population_and_no_reused_seeds():
    from pathlib import Path

    config = json.loads(Path("configs/simulation/adaptation-scenarios.json").read_text())
    catalog = cases(config)
    assert len(catalog) == 896 and len({case["seed"] for case in catalog}) == 896
    for family in FAMILIES:
        for split, count in (("train", 32), ("validation", 16), ("audit", 64)):
            assert (
                sum(case["family"] == family and case["split"] == split for case in catalog)
                == count
            )


def test_source_reliability_is_declared_and_conflicting_values_are_observable():
    item = world("source_conflict", sessions=256)
    latent = item["truth"]["latent_signal"]
    assert (item["context"][:, 0] == latent).mean() > 0.8
    assert (item["context"][:, 1] == latent).mean() < 0.2
    np.testing.assert_allclose(item["context"][:, 2:4], np.tile([0.9, 0.1], (256, 1)))
    assert all(
        event["available_at"] == item["tape"].close_times[event["session"]]
        for event in item["events"]
    )
    assert len({event["event_id"] for event in item["events"]}) == len(item["events"])


def test_execution_friction_changes_volume_and_preserves_price_bounds():
    prices = world("execution_friction")["tape"].prices
    assert np.all(prices[:16, :, 4] == 100000)
    assert np.all(prices[16:, :, 4] == 10)
    assert np.all(prices[:, :, 1] >= np.maximum(prices[:, :, 0], prices[:, :, 3]))
    assert np.all(prices[:, :, 2] <= np.minimum(prices[:, :, 0], prices[:, :, 3]))


def test_volatility_shift_changes_correlation_and_contains_reserved_jump_truth():
    config = settings()
    config["families"] = ["volatility_shift"]
    config["generator"].update(assets=16, sessions=256, warmup_sessions=64, regime_sessions=64)
    config["worlds"]["train"] = 8
    first, second, jumps = [], [], 0
    for case in cases(config):
        if case["split"] != "train":
            continue
        item = generate_scenario(case)
        prices, truth = item["tape"].prices, item["truth"]
        returns = prices[:, :, 3] / prices[:, :, 0] - 1
        expected = np.concatenate(([0], truth["expected_return"][:-1]))
        residual = returns - expected[:, None]
        first.append(residual[:128])
        second.append(residual[128:])
        jumps += np.count_nonzero(truth["common_jump"])
        assert np.all(truth["common_jump"][:128] == 0)
        assert set(np.unique(truth["common_jump"])) <= {-0.04, 0, 0.04}
        assert "common_jump" not in {field["name"] for field in item["fields"]}
        assert np.isfinite(prices).all() and np.all(prices[:, :, :4] > 0)
        np.testing.assert_allclose(item["context"][:, 4], returns.mean(axis=1), atol=1e-8)
    earlier, later = np.concatenate(first), np.concatenate(second)
    upper = np.triu_indices(16, 1)
    assert np.corrcoef(earlier, rowvar=False)[upper].mean() > 0.9
    assert np.corrcoef(later, rowvar=False)[upper].mean() < 0.6
    assert later.std() > 4 * earlier.std()
    assert 8 <= jumps <= 40


def test_execution_friction_actions_preserve_quantities_and_dividend_bookkeeping(tmp_path):
    item = world("execution_friction", assets=1)
    tape = item["tape"]
    assert len(tape.actions) == 2
    split, dividend = tape.actions
    assert split.kind == "split" and split.value == 2 and split.verified
    assert dividend.kind == "dividend" and dividend.verified
    assert split.effective_at == tape.open_times[16]
    assert dividend.effective_at == tape.open_times[17]
    assert dividend.pay_at == tape.close_times[19]
    assert split.effective_at > tape.close_times[15]
    assert dividend.effective_at > tape.close_times[16]
    assert tape.prices[16, 0, 0] == tape.prices[15, 0, 3] / 2
    assert tape.prices[17, 0, 0] == tape.prices[16, 0, 3] - dividend.value
    book = Portfolio({"FIC000": Instrument("USD")}, {"USD": 10000}, cost_bps=0, participation=1)
    book.start(int(tape.close_times[14]), tape.quotes(14))
    book.submit({"FIC000": 2}, decision_at=int(tape.close_times[14]))
    for at in range(15, 20):
        cash_before = book.cash["USD"]
        result = book.advance(
            int(tape.open_times[at]),
            int(tape.close_times[at]),
            tape.quotes(at),
            actions=[
                action for action in tape.actions if action.effective_at == tape.open_times[at]
            ],
        )
        quantity = 2 if at == 15 else 4
        assert book.positions["FIC000"] == quantity
        entitlement = quantity * dividend.value if 17 <= at < 19 else 0
        assert result["nav"]["USD"] == pytest.approx(
            book.cash["USD"] + quantity * tape.prices[at, 0, 3] + entitlement
        )
        if at in (16, 17, 18):
            assert book.cash["USD"] == cash_before
        if at == 19:
            assert book.cash["USD"] == pytest.approx(cash_before + quantity * dividend.value)
    config = settings()
    config["families"] = ["execution_friction"]
    index = prepare_adaptation_scenarios(config, tmp_path / "corporate")
    restored = read_tape(tmp_path / "corporate" / index["records"][0]["path"])
    assert len(restored.actions) == 8 and all(action.verified for action in restored.actions)


@pytest.mark.parametrize("split", ["validation", "audit"])
def test_hmm_rejects_reserved_sources_before_importing_optional_dependency(split):
    with pytest.raises(ValueError, match="entrenamiento"):
        fit_hmm([(split, np.ones((12, 3)))])


def test_hmm_exports_finite_native_parameters_with_training_provenance():
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(91)
    first = rng.normal([0, 0.002, 0.001], [0.001, 0.0001, 0.0001], (32, 3))
    second = rng.normal([0, 0.009, 0.004], [0.003, 0.0003, 0.0003], (32, 3))
    result = fit_hmm([("train", np.concatenate([first, second, first]))])
    parameters = result["parameters"]
    assert result["implementation"] == "hmmlearn==0.3.3"
    assert parameters["states"] == 2 and parameters["dimensions"] == 3
    np.testing.assert_allclose(sum(parameters["prior"]), 1)
    np.testing.assert_allclose(np.asarray(parameters["transitions"]).reshape(2, 2).sum(1), 1)
    assert np.all(np.asarray(parameters["variances"]) > 0)
    means = np.asarray(parameters["means"]).reshape(2, 3)
    assert means[0, 1] < means[1, 1]
    assert result["training_context_point_in_time"] is False


def test_hmm_rejects_degenerate_emissions():
    with pytest.raises(ValueError, match="variación"):
        fit_hmm([("train", np.ones((16, 3)))])
