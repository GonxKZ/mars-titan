"""Lesiones de lectura de B4 frente a operadores restringidos explícitos."""

import pytest
import torch

from mars_titan.memory.area_memory import AreaLayout, AreaMemory
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig, AreaQueryBridgeState
from mars_titan.memory.associative_memory import AssociativeMemory, AssociativeMemoryConfig


def memory(*, bridge=False, steps=1, overflow=False):
    config = AssociativeMemoryConfig("delta", key_size=8, value_size=1)
    layout = AreaLayout(areas=3, seed=7)
    width = layout.feature_size(config.key_size)
    weights = tuple(
        AssociativeMemory(
            AssociativeMemoryConfig("delta", key_size=width, value_size=1),
            torch.arange(width, dtype=torch.float64)[:, None] / 20 + area / 10,
        )
        for area in range(layout.areas)
    )
    if not bridge:
        return AreaMemory(config, layout, weights)
    rule = AssociativeMemoryConfig("delta", key_size=width, value_size=width)
    maps = []
    for area in range(layout.areas):
        matrix = torch.eye(width, dtype=torch.float64).roll(area + 1, dims=1) * 0.5
        if overflow:
            matrix[:, 0] = 1e308
        maps.append(AssociativeMemory(rule, matrix))
    settings = AreaQueryBridgeConfig(gain=0.25, steps=steps)
    state = AreaQueryBridgeState(rule, layout.areas, settings, maps=tuple(maps))
    return AreaMemory(config, layout, weights, query_bridge=settings, bridge_state=state)


def reference_features(state, keys, active, coordinates):
    features = state.features(keys) * active[:, None, None] * coordinates[:, None, :]
    if state._bridge is not None:
        for _ in range(state.query_bridge.steps):
            messages = []
            for source, operator in enumerate(state._bridge.matrices):
                target = (source + 1) % state.layout.areas
                restricted = (
                    operator
                    * coordinates[source, :, None]
                    * coordinates[target, None, :]
                    * active[source]
                    * active[target]
                )
                messages.append(features[source] @ restricted)
            translated = torch.stack(messages).roll(1, dims=0)
            mixed = (1 - state.query_bridge.gain) * features + state.query_bridge.gain * translated
            norm = torch.linalg.vector_norm(mixed, dim=-1, keepdim=True)
            features = mixed / torch.where(norm > 0, norm, 1.0)
    return features


def reference(state, keys, active, coordinates, reduction):
    features = reference_features(state, keys, active, coordinates)
    values = torch.stack(
        [part @ area.matrix for part, area in zip(features, state._areas, strict=True)]
    )
    denominator = int(active.sum()) if reduction == "active_mean" else state.layout.areas
    return values.sum(dim=0) / max(denominator, 1)


def test_bridge_applies_coordinate_mask_to_unmasked_input_queries():
    state = memory(bridge=True, steps=2)
    keys = torch.arange(1, 25, dtype=torch.float64).reshape(3, 8) / 100
    active = torch.ones(3, dtype=torch.bool)
    coordinates = torch.ones(3, 16, dtype=torch.bool)
    coordinates[:, ::3] = False
    actual = state._bridge.apply(state.features(keys), coordinates=coordinates)
    expected = reference_features(state, keys, active, coordinates)
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-16)


@pytest.mark.parametrize("steps", [0, 1, 2, 4])
@pytest.mark.parametrize("reduction", ["active_mean", "fixed_mean"])
def test_lesions_match_restricted_operators_at_every_bridge_step(steps, reduction):
    state = memory(bridge=steps > 0, steps=max(steps, 1))
    keys = torch.arange(1, 25, dtype=torch.float64).reshape(3, 8) / 100
    active = torch.tensor([True, False, True])
    coordinates = torch.ones(3, 16, dtype=torch.bool)
    coordinates[0, ::2] = False
    coordinates[2, 1::2] = False
    before = state.export()
    actual = state.read_diagnostic(
        keys, active=active, coordinates=coordinates, reduction=reduction
    )
    expected = reference(state, keys, active, coordinates, reduction)
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-16)
    assert state.export()["matrix_sha256"] == before["matrix_sha256"]
    assert state.writes == 0 and state.cursor is None


def test_no_lesion_is_bitwise_identical_and_all_areas_can_be_removed():
    state = memory(bridge=True)
    keys = torch.ones(2, 8, dtype=torch.float64) / 8
    assert torch.equal(state.read_diagnostic(keys), state.read(keys))
    result = state.read_diagnostic(keys, active=[False] * 3)
    assert torch.equal(result, torch.zeros(2, 1, dtype=torch.float64))
    with pytest.raises(ValueError):
        state.read(keys, active=[False] * 3)


def test_removed_bridge_outputs_are_not_computed_before_the_mask():
    state = memory(bridge=True, overflow=True)
    keys = torch.ones(2, 8, dtype=torch.float64) / 8
    with pytest.raises(ValueError, match="no finitos"):
        state.read(keys)
    coordinates = torch.ones(3, 16, dtype=torch.bool)
    coordinates[:, 0] = False
    result = state.read_diagnostic(keys, coordinates=coordinates)
    assert torch.isfinite(result).all()


def test_coordinate_lesions_preserve_query_gradients():
    state = memory(bridge=True, steps=2)
    keys = torch.arange(1, 17, dtype=torch.float64).reshape(2, 8).div(100).requires_grad_()
    coordinates = torch.ones(3, 16, dtype=torch.bool)
    coordinates[:, ::3] = False
    assert torch.autograd.gradcheck(
        lambda value: state.read_diagnostic(value, coordinates=coordinates), (keys,)
    )


def test_zero_gain_keeps_the_declared_normalization_after_coordinate_removal():
    from dataclasses import replace

    state = memory(bridge=True)
    settings = replace(state.query_bridge, gain=0.0)
    bridge = AreaQueryBridgeState(state._bridge.rule, 3, settings, maps=state._bridge._maps)
    state = AreaMemory(
        state.config, state.layout, state._areas, query_bridge=settings, bridge_state=bridge
    )
    keys = torch.arange(1, 25, dtype=torch.float64).reshape(3, 8) / 100
    coordinates = torch.ones(3, 16, dtype=torch.bool)
    coordinates[:, ::3] = False
    expected = reference(state, keys, torch.ones(3, dtype=torch.bool), coordinates, "active_mean")
    actual = state.read_diagnostic(keys, coordinates=coordinates)
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-16)


def test_diagnostic_read_rejects_nonfinite_corrections():
    state = memory()
    weights = tuple(
        AssociativeMemory(area.config, torch.full_like(area.matrix, 1e308)) for area in state._areas
    )
    state = AreaMemory(state.config, state.layout, weights)
    with pytest.raises(ValueError, match="no finitos"):
        state.read_diagnostic(torch.ones(2, 8, dtype=torch.float64) / 8)


@pytest.mark.parametrize(
    "options",
    [
        {"active": [1, 0, 1]},
        {"active": [True]},
        {"coordinates": torch.ones(3, 16)},
        {"coordinates": torch.ones(16, dtype=torch.bool)},
        {"reduction": "unknown"},
    ],
)
def test_invalid_lesions_are_rejected(options):
    with pytest.raises(ValueError):
        memory().read_diagnostic(torch.ones(2, 8, dtype=torch.float64) / 8, **options)
