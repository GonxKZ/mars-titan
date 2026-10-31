"""Contexto N2 fechado, prefijos pasados y controles, sin estimar regímenes."""

import numpy as np
import pytest
import torch

from mars_titan.memory.regimes import REGIME_RULE, RegimeRule
from mars_titan.models.titans.dendritic_regime import (
    DendriticRegime,
    parameter_seed,
    parse_dendritic_regime,
    regime_context,
)
from tests.models.titans.test_financial_dendrites import OPTIONS, configured, raw


def test_observable_context_matches_the_existing_regime_and_records_its_sources():
    _, batch = configured()
    config = DendriticRegime(**OPTIONS)
    result = regime_context(config, batch, seed=42)
    expected, _ = RegimeRule(REGIME_RULE, min_assets=1).routes(
        batch.inputs["prices"].numpy(), batch.flow_ids, batch.prediction_at[0]
    )
    assert result.routes == tuple(expected.tolist())
    assert result.source_slots == (63, 63)
    assert result.prediction_at == batch.prediction_at[0]
    assert result.input_available_at == batch.input_available_at
    assert torch.equal(result.values, torch.eye(5, dtype=torch.float64)[expected, 1:])


def test_unknown_regime_is_zero_and_keeps_the_operator_neutral():
    _, batch = configured()
    result = regime_context(DendriticRegime(), batch, seed=42)
    assert result.values.count_nonzero() == 0
    assert result.routes == (0, 0) and result.source_slots == (-1, -1)


def test_past_prefix_control_does_not_read_the_suffix_after_its_selected_slot():
    model, batch = configured("shuffled_past_prefix")
    config = model.config.active_dendrites
    before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    result = regime_context(config, batch, seed=42)
    slot = result.source_slots[0]
    assert 42 <= slot < 63 and result.source_slots == (slot, slot)
    changed = {name: tensor.clone() for name, tensor in batch.inputs.items()}
    changed["prices"][:, slot + 1 :, 3] = -8
    other = batch.derived(changed, "fixture_changes_only_after_selected_prefix")
    actual = regime_context(config, other, seed=42)
    assert torch.equal(actual.values, result.values)
    assert actual.source_slots == result.source_slots
    assert torch.equal(before, torch.get_rng_state())
    numpy_after = np.random.get_state()
    assert (
        numpy_before[0] == numpy_after[0]
        and np.array_equal(numpy_before[1], numpy_after[1])
        and numpy_before[2:] == numpy_after[2:]
    )


def test_control_has_no_candidate_when_only_the_current_slot_has_enough_returns():
    _, batch = configured()
    config = DendriticRegime(**{**OPTIONS, "mode": "shuffled_past_prefix", "min_returns": 63})
    result = regime_context(config, batch, seed=42)
    assert result.values.count_nonzero() == 0 and result.source_slots == (-1, -1)


def test_constant_control_preserves_the_unknown_mask_and_parameter_seed_is_separate():
    _, batch = configured()
    result = regime_context(DendriticRegime(**{**OPTIONS, "mode": "constant"}), batch, seed=42)
    assert result.routes == (1, 1)
    assert parameter_seed(42) != 42 and parameter_seed(42) != parameter_seed(43)
    assert parse_dendritic_regime(None) is None
    config = DendriticRegime()
    assert parse_dendritic_regime(config) is config
    assert config.identity()["state"] == "none_recomputed_from_the_received_window"


def test_future_availability_is_rejected_by_the_decision_boundary():
    from tests.models.titans.test_financial_adapter import api

    model, _ = configured()
    values = raw()
    values["input_available_at"][:] = values["prediction_at"] + np.timedelta64(1, "us")
    with pytest.raises(ValueError, match="futuro"):
        api().DecisionBatch.from_corpus(values, model.config.inputs, dtype=torch.float64)


def test_context_builder_rejects_a_corrupted_future_date_and_nonfinite_derived_statistics():
    _, batch = configured()
    original = batch.input_available_at
    object.__setattr__(batch, "input_available_at", (batch.prediction_at[0] + 1,) * 2)
    with pytest.raises(ValueError, match="posteriores"):
        regime_context(DendriticRegime(**OPTIONS), batch, seed=42)
    object.__setattr__(batch, "input_available_at", original)
    changed = {name: tensor.clone() for name, tensor in batch.inputs.items()}
    changed["prices"][:, ::2, 3] = 1e200
    other = batch.derived(changed, "fixture_overflow")
    with pytest.raises(ValueError, match="aritmética"):
        regime_context(DendriticRegime(**OPTIONS), other, seed=42)


@pytest.mark.parametrize(
    "value", [False, "observable", {"mode": "observable"}, {**OPTIONS, "mode": "future"}]
)
def test_parser_rejects_implicit_or_unknown_context_contracts(value):
    with pytest.raises(ValueError):
        parse_dendritic_regime(value)


def test_builder_requires_its_real_configuration_and_verified_batch():
    with pytest.raises(ValueError):
        regime_context(None, None, seed=42)


def test_array_boundary_rejects_missing_configuration_and_incompatible_geometry():
    from mars_titan.models.titans.dendritic_regime import context_arrays

    model, batch = configured()
    arguments = (
        batch.inputs["prices"].numpy(),
        batch.flow_ids,
        batch.prediction_at,
        batch.input_available_at,
    )
    with pytest.raises(ValueError):
        context_arrays(None, *arguments, seed=42)
    with pytest.raises(ValueError):
        context_arrays(
            model.config.active_dendrites, arguments[0][:, :, :5], *arguments[1:], seed=42
        )


def test_markets_use_their_own_regime_and_latest_input_availability():
    from tests.models.titans.test_financial_adapter import api

    model, _ = configured()
    values = raw()
    at = int(values["prediction_at"].astype(np.int64)[0])
    values["inputs"] = {
        name: np.concatenate([array, array.copy()]) for name, array in values["inputs"].items()
    }
    values["inputs"]["prices"][2:, :, 3] *= -1
    values["presence"] = np.concatenate([values["presence"]] * 2)
    values["sample_ids"] = [f"{flow}/{at}" for flow in ("US/AAA", "US/BBB", "CN/AAA", "CN/BBB")]
    values["prediction_at"] = np.full(4, at, dtype="datetime64[us]")
    values["input_available_at"] = np.array(
        [at - 2, at - 1, at - 4, at - 3], dtype="datetime64[us]"
    )
    values["market"] = ["US", "US", "CN", "CN"]
    batch = api().DecisionBatch.from_corpus(values, model.config.inputs, dtype=torch.float64)
    result = regime_context(DendriticRegime(**OPTIONS), batch, seed=42)
    assert result.routes[0] == result.routes[1]
    assert result.routes[2] == result.routes[3]
    assert result.routes[0] != result.routes[2]
    assert result.input_available_at == (at - 1, at - 1, at - 3, at - 3)
