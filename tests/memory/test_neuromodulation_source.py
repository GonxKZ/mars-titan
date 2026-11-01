"""Fuente N1 con resultados manuales, sin ajustar ni entrenar modelos."""

import importlib
import math

import numpy as np
import pytest

from mars_titan.memory.salience_budget import MatureWrite


def api():
    return importlib.import_module("mars_titan.memory.neuromodulation_source")


def config(**changes):
    return api().NeuromodulationSourceConfig(
        **(
            dict(
                window=2,
                min_events=2,
                expected_scale=1,
                scale_floor=0.1,
                cusum_reference=1,
                cusum_slack=0,
                cusum_limit=2,
            )
            | changes
        )
    )


def outcome(at, target=1.0, issued=0.0, flow="US/AAA", available=None):
    return MatureWrite(flow, at - 1, at if available is None else available, issued, target)


def advanced(state, rows, at):
    return state.advance(
        rows,
        cutoff=at,
        generation=state.generation + 1,
        applied_total=state.applied_total + len(rows),
    )


def prefix(**options):
    state = api().NeuromodulationSource(config(**options))
    state = advanced(state, [outcome(10)], 10)
    return advanced(state, [outcome(20)], 20)


def values(state, at, flows=("US/AAA",)):
    return state.snapshot(flows, prediction_at=at).array()


def test_manual_residual_scale_and_unexpected_error_have_separate_updates():
    state = prefix()
    np.testing.assert_array_equal(values(state, 21), [[0.5, 0]])
    following = advanced(state, [outcome(30, target=3)], 30)
    # La publicación del instante no modifica las señales de sus predicciones.
    np.testing.assert_array_equal(values(following, 30), [[0.5, 0]])
    expected = math.sqrt(5) / (1 + math.sqrt(5))
    np.testing.assert_allclose(values(following, 31), [[expected, 1]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(values(state, 31), [[0.5, 0]])


def test_same_instant_generations_and_row_order_preserve_the_following_signals():
    state = prefix()
    rows = [outcome(30, target=3), outcome(30, target=1, flow="US/BBB")]
    whole = advanced(state, rows, 30)
    reversed_rows = advanced(state, list(reversed(rows)), 30)
    first = advanced(state, rows[:1], 30)
    split = advanced(first, rows[1:], 30)
    for result in (whole, reversed_rows, first, split):
        np.testing.assert_array_equal(values(result, 30), [[0.5, 0]])
    for result in (whole, reversed_rows, split):
        np.testing.assert_allclose(
            values(result, 31), [[math.sqrt(3) / (1 + math.sqrt(3)), 0.5]], rtol=0, atol=1e-15
        )


def test_perfect_issued_points_do_not_turn_residual_volatility_into_error():
    state = api().NeuromodulationSource(config())
    state = advanced(state, [outcome(10, target=2, issued=2)], 10)
    state = advanced(state, [outcome(20, target=-2, issued=-2)], 20)
    state = advanced(state, [outcome(30, target=20, issued=20)], 30)
    expected = math.sqrt(202) / (1 + math.sqrt(202))
    np.testing.assert_allclose(values(state, 31), [[expected, 0]], rtol=0, atol=1e-15)


def test_future_availability_is_rejected_before_reading_the_target():
    state = prefix()
    row = MatureWrite("US/AAA", 29, 40, object(), object())
    with pytest.raises(ValueError, match="disponibilidad"):
        advanced(state, [row], 30)
    np.testing.assert_array_equal(values(state, 31), [[0.5, 0]])


def test_restore_reproduces_the_next_signal_and_does_not_share_exported_values():
    state = advanced(prefix(), [outcome(30, target=3)], 30)
    exported = state.export()
    restored = api().NeuromodulationSource.restore(config(), exported)
    np.testing.assert_array_equal(values(state, 31), values(restored, 31))
    left = advanced(state, [outcome(40, target=0.25)], 40)
    right = advanced(restored, [outcome(40, target=0.25)], 40)
    assert left.export() == right.export()
    array = values(state, 31)
    array[:] = 0
    assert values(state, 31)[0, 0] > 0


@pytest.mark.parametrize(
    "kind", ["duplicate", "generation", "counter", "retrograde", "future_snapshot"]
)
def test_invalid_publications_and_queries_cannot_advance_the_state(kind):
    state = prefix()
    before = state.export()
    with pytest.raises(ValueError):
        if kind == "duplicate":
            advanced(state, [outcome(30), outcome(30)], 30)
        elif kind == "generation":
            state.advance(
                [], cutoff=30, generation=state.generation, applied_total=state.applied_total
            )
        elif kind == "counter":
            state.advance([], cutoff=30, generation=state.generation + 1, applied_total=999)
        elif kind == "retrograde":
            advanced(state, [], 19)
        else:
            values(state, 19)
    assert state.export() == before


def test_empty_publications_and_same_instant_retries_do_not_add_statistical_events():
    state = advanced(prefix(), [outcome(30, target=3)], 30)
    repeated = advanced(state, [], 30)
    np.testing.assert_array_equal(values(repeated, 31), values(state, 31))
    np.testing.assert_array_equal(values(repeated, 30), [[0.5, 0]])
    following = advanced(repeated, [], 40)
    np.testing.assert_array_equal(values(following, 40), values(state, 31))
    assert following.export()["base"]["US"]["events"] == 3
    with pytest.raises(ValueError):
        advanced(state, [outcome(30, target=3)], 30)


def test_markets_are_separate_and_unknown_history_has_a_zero_prior():
    state = prefix()
    result = advanced(state, [outcome(30, target=3, flow="CN/600000")], 30)
    np.testing.assert_array_equal(values(result, 31, ("US/BBB", "CN/600000")), [[0.5, 0], [0, 0]])
    assert result.snapshot(("US/BBB",), prediction_at=30).selection == "before_instant"


def test_shuffled_control_uses_only_prior_observed_pairs_and_preserves_rng_and_resume():
    original_rng = np.random.get_state()
    observed = api().NeuromodulationSource(config())
    shuffled = api().NeuromodulationSource(config(source="shuffled_past_prefix", seed=71))
    history = []
    for at, target in ((10, 1.0), (20, 1.0), (30, 3.0), (40, 0.25), (50, 2.0), (60, 0.5)):
        observed = advanced(observed, [outcome(at, target=target)], at)
        shuffled = advanced(shuffled, [outcome(at, target=target)], at)
        selected = values(shuffled, at + 1)[0].tolist()
        assert selected in (history[-2:] or [[0.0, 0.0]])
        history.append(values(observed, at + 1)[0].tolist())
        shuffled = api().NeuromodulationSource.restore(shuffled.config, shuffled.export())
    current = np.random.get_state()
    assert original_rng[0] == current[0] and np.array_equal(original_rng[1], current[1])
    assert original_rng[2:] == current[2:]


def test_windows_and_source_configuration_remain_bounded():
    state = prefix()
    for at in range(30, 140, 10):
        state = advanced(state, [outcome(at)], at)
    record = state.export()["base"]["US"]
    assert len(record["window"]) == 2 and len(record["history"]) == 3
    assert len(state.export()["rows"]) == 1
    with pytest.raises(AttributeError):
        state.config = config(source="shuffled_past_prefix")


@pytest.mark.parametrize(
    "changes",
    [
        dict(window=1),
        dict(window=True),
        dict(window=257),
        dict(min_events=3),
        dict(expected_scale=0),
        dict(scale_floor=float("inf")),
        dict(cusum_limit=0),
        dict(source="unknown"),
        dict(seed=-1),
        dict(max_rows=0),
        dict(expected_scale=1 << 2048),
    ],
)
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_cumulative_instant_budget_and_snapshot_shapes_are_checked():
    state = prefix(max_rows=1)
    state = advanced(state, [outcome(30)], 30)
    with pytest.raises(ValueError):
        advanced(state, [outcome(30, flow="US/BBB")], 30)
    for flows in ((), ("US/AAA", "US/AAA"), ("unknown",)):
        with pytest.raises(ValueError):
            values(state, 31, flows)


def test_large_finite_values_do_not_overflow_the_rms_or_normalization():
    state = prefix()
    result = advanced(state, [outcome(30, target=1e308)], 30)
    assert np.isfinite(values(result, 31)).all()
    np.testing.assert_array_equal(values(result, 31), [[1, 1]])
    assert (
        api().NeuromodulationSource.restore(config(), result.export()).export() == result.export()
    )
    before = state.export()
    with pytest.raises(ValueError):
        advanced(state, [outcome(30, target=1e308, issued=-1e308)], 30)
    assert state.export() == before


def test_cusum_saturates_instead_of_creating_a_signal_above_one():
    state = advanced(prefix(), [outcome(30, target=3)], 30)
    state = advanced(state, [outcome(40, target=30)], 40)
    assert values(state, 41)[0, 1] == 1
    assert np.isfinite(values(state, 41)).all()


def test_shuffled_source_is_seeded_and_does_not_replace_the_observed_statistics():
    states = [
        api().NeuromodulationSource(config(source="shuffled_past_prefix", seed=seed))
        for seed in (0, 71)
    ]
    pairs = [[], []]
    for at in range(10, 110, 10):
        for i in (0, 1):
            states[i] = advanced(states[i], [outcome(at, target=float(at / 10))], at)
            pairs[i].append(values(states[i], at + 1)[0].tolist())
    assert pairs[0] != pairs[1]
    for field in ("events", "window", "history", "cusum"):
        assert states[0].export()["base"]["US"][field] == states[1].export()["base"]["US"][field]


@pytest.mark.parametrize(
    "kind",
    ["schema", "checksum", "configuration", "window", "history", "signal", "counter", "future"],
)
def test_restore_rejects_incompatible_or_inconsistent_states_even_after_rehashing(kind):
    import copy

    state = advanced(prefix(), [outcome(30, target=3)], 30)
    payload = copy.deepcopy(state.export())
    if kind == "schema":
        payload["schema_version"] = True
    elif kind == "checksum":
        payload["sha256"] = "f" * 64
    elif kind == "configuration":
        payload["configuration"]["configuration"]["expected_scale"] = 2
    elif kind == "window":
        payload["base"]["US"]["window"] = payload["base"]["US"]["window"][:1]
    elif kind == "history":
        payload["base"]["US"]["history"][-1] = [0.9, 0.9]
    elif kind == "signal":
        payload["base"]["US"]["signal"] = [0.9, 0.9]
    elif kind == "counter":
        payload["applied_total"] = 0
    else:
        payload["rows"][0]["available_at"] = 31
    if kind != "checksum":
        payload["sha256"] = api()._digest({k: v for k, v in payload.items() if k != "sha256"})
    with pytest.raises(ValueError):
        api().NeuromodulationSource.restore(config(), payload)
