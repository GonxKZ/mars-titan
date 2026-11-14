"""Remuestreo emparejado por ventanas y correcciones de familias."""

import numpy as np

from mars_titan.evaluation import comparison_inference as inference

SETTINGS = dict(replicates=128, seed=73, confidence=0.95)


def member(values, name="variant/base"):
    values = np.asarray(values, dtype=np.float64)
    return dict(
        name=name, values=values, defined=np.ones(values.shape, dtype=bool), direction="less"
    )


def test_bootstrap_never_moves_observations_between_origins():
    values = np.r_[np.ones(20), np.full(20, 100.0)][:, None]
    days = np.r_[np.arange(20), np.arange(20) + 100]
    origins = np.r_[np.full(20, 2019), np.full(20, 2020)]

    draws = inference.stratified_draws(
        values,
        days,
        origins,
        block_length=4,
        **{k: v for k, v in SETTINGS.items() if k != "confidence"},
    )

    np.testing.assert_array_equal(draws, np.full((128, 1), 50.5))


def test_paired_favourable_effect_is_reported_with_both_corrections():
    days = np.arange(80)
    values = -0.1 - 0.01 * np.sin(days)
    result = inference.compare_family(
        [member(values)], days, np.full(80, 2019), block_length=8, settings=SETTINGS
    )

    assert result["available"] is True
    row = result["members"][0]
    assert row["estimate"] == np.mean(values)
    assert row["simultaneous_interval"][1] < 0
    assert row["romano_wolf_pvalue"] <= 0.05 and row["holm_pvalue"] <= 0.05


def test_short_or_undefined_families_are_explicitly_unavailable():
    short = inference.compare_family(
        [member([1, 2, 3])], np.arange(3), np.full(3, 2019), block_length=4, settings=SETTINGS
    )
    assert short["available"] is False and short["reason"]
    item = member(np.ones(40))
    item["defined"][:] = False
    empty = inference.compare_family(
        [item], np.arange(40), np.full(40, 2019), block_length=4, settings=SETTINGS
    )
    assert empty["available"] is False and empty["sessions"] == 0


def test_inference_is_reproducible_and_does_not_consume_global_numpy_rng():
    state = np.random.get_state()
    days = np.arange(40)
    arguments = ([member(np.sin(days))], days, np.full(40, 2019))
    first = inference.compare_family(*arguments, block_length=4, settings=SETTINGS)
    second = inference.compare_family(*arguments, block_length=4, settings=SETTINGS)
    assert first == second
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
