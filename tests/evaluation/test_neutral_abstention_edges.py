"""Casos límite de tiempo, cobertura y escala de la abstención neutral."""

import numpy as np
import pytest

from mars_titan.evaluation import neutral_abstention as na
from tests.evaluation.test_neutral_abstention import START, fitted, panel


def test_unsigned_maturity_cannot_wrap_into_an_eligible_past_label():
    history = dict(asset_id=["US/A"], target=[7.0], matures_at=np.array([2**64 - 1], np.uint64))
    with pytest.raises(ValueError, match="rango"):
        na.past_volatility(["US/A"], [START], history, window=2, min_labels=1)


def test_control_calls_match_even_when_some_medians_are_exactly_zero():
    record, _ = fitted()
    evaluation = panel([0.0, 2.0, 3.0, -4.0], np.full(4, 5.0), [0, 0, 0, 0])
    outputs, _ = na.emit(
        record,
        evaluation,
        stream=0,
        volatility=np.array([0.0, 10.0, 3.0, 1.0]),
        disagreement=np.array([0.0, 1.0, 10.0, 3.0]),
    )
    expected = np.count_nonzero(outputs[na.IDENTITY])
    assert expected == 3
    assert {name: np.count_nonzero(value) for name, value in outputs.items()} == dict.fromkeys(
        outputs, expected
    )


def test_old_extreme_labels_do_not_erase_the_current_window_volatility():
    history = dict(
        asset_id=["US/A"] * 4,
        target=[1e10, 3e-6, 4e-6, 1e-6],
        matures_at=np.arange(START - 4, START),
    )
    actual = na.past_volatility(["US/A"], [START - 2], history, window=2, min_labels=2)
    expected = np.hypot(3e-6, 4e-6) / np.sqrt(2)
    assert actual[0] == pytest.approx(expected, rel=2e-14, abs=0.0)


def test_finite_large_labels_have_finite_rms():
    history = dict(
        asset_id=["US/A"] * 3,
        target=[1e308, -1e308, 1e308],
        matures_at=np.arange(START - 3, START),
    )
    actual = na.past_volatility(["US/A"], [START], history, window=3, min_labels=3)
    assert np.isfinite(actual).all()
    assert actual[0] == pytest.approx(1e308, rel=2e-14)


def test_empty_queries_return_an_empty_vector():
    history = dict(asset_id=[], target=[], matures_at=np.array([], dtype=np.int64))
    actual = na.past_volatility([], np.array([], np.int64), history, window=2, min_labels=1)
    assert actual.shape == (0,)


@pytest.mark.parametrize("window, minimum", [(0, 1), (-2, 1), (True, 1), (2, 0), (2, 3)])
def test_direct_volatility_call_validates_its_window(window, minimum):
    history = dict(asset_id=["US/A"], target=[1.0], matures_at=[START - 1])
    with pytest.raises(ValueError, match="ventana"):
        na.past_volatility(["US/A"], [START], history, window=window, min_labels=minimum)


@pytest.mark.parametrize("window", [1, 2, 7, 60, 10_000])
@pytest.mark.parametrize("rows", [0, 1, 2, 19, 101])
def test_block_rms_matches_direct_norms_at_every_boundary(window, rows):
    rng = np.random.default_rng(17)
    values = rng.choice([-1.0, 1.0], rows) * 10.0 ** rng.uniform(-300, 300, rows)
    actual = na._rolling_rms(values, window)
    expected = []
    for end in range(1, rows + 1):
        part = values[max(0, end - window) : end]
        expected.append(np.hypot.reduce(np.abs(part) / np.sqrt(len(part))))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)


def test_eligible_control_ranks_ignore_zero_medians_and_keep_undefined_scores_last():
    evaluation = panel([0.0, 2.0, 3.0, 0.0, 0.0, -2.0], np.ones(6), [0, 0, 0, 0, 1, 1])
    original = np.array([True, False, True, True, True, False])
    scores = np.array([100.0, np.nan, 3.0, -100.0, np.nan, np.nan])
    selected = na.matched_abstention(
        evaluation, original, scores, eligible=evaluation.prediction != 0
    )
    assert selected.tolist() == [True, False, True, True, True, False]
    with pytest.raises(ValueError, match="elegibilidad"):
        na.matched_abstention(evaluation, original, scores, eligible=np.ones(6))
