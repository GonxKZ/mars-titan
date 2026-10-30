"""Ligadura bipolar de B3 y controles frente a ecuaciones independientes."""

import numpy as np
import pytest
import torch

from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.role_binding import SOURCES, RoleBindingConfig


def sample():
    values = torch.linspace(-2, 2, 5 * 64, dtype=torch.float64).reshape(5, 64)
    return values, torch.arange(5, dtype=torch.int64)


def test_bipolar_binding_matches_sign_products_and_is_exactly_self_inverse():
    config = RoleBindingConfig(seed=13)
    values, roles = sample()
    signs = 2 * np.random.Generator(np.random.PCG64([0xB3, 0, 13])).integers(0, 2, (5, 64)) - 1
    expected = values * torch.from_numpy(signs)
    bound = config.apply(values, roles)
    assert torch.equal(bound, expected)
    assert torch.equal(config.unbind(bound, roles), values)
    assert torch.equal(config.apply(values, roles), bound)


def test_roles_vary_for_the_same_content_and_constant_control_ignores_them():
    values, roles = sample()
    values = values[:1].expand(5, -1)
    bound = RoleBindingConfig().apply(values, roles)
    assert len({tuple(row.tolist()) for row in bound}) == 5
    constant = RoleBindingConfig(mode="constant_role").apply(values, roles)
    assert torch.equal(constant, constant[:1].expand(5, -1))


@pytest.mark.parametrize("mode", ["bipolar", "constant_role", "projected_concat"])
def test_query_gradients_are_finite_and_inputs_are_unchanged(mode):
    values, roles = sample()
    before = values.clone()
    values.requires_grad_(True)
    result = RoleBindingConfig(mode=mode).apply(values, roles)
    assert result.shape == values.shape
    result.square().sum().backward()
    assert torch.isfinite(values.grad).all()
    assert torch.equal(values, before)


def test_private_randomness_does_not_consume_numpy_or_torch_global_rng():
    values, roles = sample()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    for mode in ("bipolar", "constant_role", "projected_concat"):
        config = RoleBindingConfig(mode=mode, seed=9)
        config.apply(values, roles)
        config.identity()
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before_torch, torch.get_rng_state())


@pytest.mark.parametrize("source", list(SOURCES))
@pytest.mark.parametrize(
    "areas,bridge", [(None, None), (AreaLayout(), None), (AreaLayout(), AreaQueryBridgeConfig())]
)
def test_bound_correction_roundtrip_keeps_its_role_identity(source, areas, bridge):
    binding = RoleBindingConfig(source=source, seed=7)
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal", rate=0.2, forgetting=0.01),
        role_binding=binding,
        areas=areas,
        query_bridge=bridge,
    )
    inputs, roles = sample()
    inputs = inputs.to(torch.float32)
    plain = MatureCorrection(correction.memory).keys(inputs)
    keys = correction.keys(inputs, roles)
    assert torch.equal(keys, binding.apply(plain, roles))
    state = correction.empty()
    feedback = correction.feedback(
        ids=[1, 2, 3, 4, 5],
        decision_at=[1] * 5,
        available_at=[2] * 5,
        keys=keys,
        values=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    written = state.write(feedback, cutoff=2)
    restored = correction.restore(written.export())
    assert torch.equal(written.read(keys), restored.read(keys))
    assert written.writes == 5 and state.writes == 0
    other = MatureCorrection(
        correction.memory,
        role_binding=RoleBindingConfig(source=source, seed=8),
        areas=areas,
        query_bridge=bridge,
    )
    with pytest.raises(ValueError, match="roles"):
        other.restore(written.export())


def test_role_binding_requires_codec_and_a_delta_or_proximal_memory():
    with pytest.raises(ValueError, match="roles"):
        MatureCorrection(
            AssociativeMemoryConfig("proximal", key_size=1),
            key="constant",
            role_binding=RoleBindingConfig(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(source="current_sector"),
        dict(mode="FHRR"),
        dict(seed=True),
        dict(seed=-1),
        dict(seed=2**32),
        dict(seed=1.0),
    ],
)
def test_invalid_configurations_are_rejected(changes):
    with pytest.raises(ValueError):
        RoleBindingConfig(**changes)


@pytest.mark.parametrize(
    "values,roles",
    [
        (torch.zeros(2, 64), torch.zeros(2, dtype=torch.int64)),
        (torch.zeros(2, 63, dtype=torch.float64), torch.zeros(2, dtype=torch.int64)),
        (torch.zeros(2, 64, dtype=torch.float64), torch.zeros(2)),
        (torch.zeros(2, 64, dtype=torch.float64), torch.zeros(1, dtype=torch.int64)),
        (torch.zeros(2, 64, dtype=torch.float64), torch.tensor([-1, 0])),
        (torch.zeros(2, 64, dtype=torch.float64), torch.tensor([0, 5])),
        (torch.full((2, 64), float("nan"), dtype=torch.float64), torch.zeros(2, dtype=torch.int64)),
        (torch.full((2, 64), float("inf"), dtype=torch.float64), torch.zeros(2, dtype=torch.int64)),
        (torch.empty(8193, 64, dtype=torch.float64), torch.zeros(8193, dtype=torch.int64)),
    ],
)
def test_invalid_keys_or_roles_are_rejected(values, roles):
    with pytest.raises(ValueError):
        RoleBindingConfig().apply(values, roles)


def test_inverse_restores_extreme_values_and_signed_zero_bit_patterns():
    row = torch.tensor(
        [0.0, -0.0, 5e-324, -5e-324, 1e308, -1e308, 1e-300, -1e-300], dtype=torch.float64
    ).repeat(8)
    roles = torch.arange(5)
    keys = row.expand(5, -1)
    config = RoleBindingConfig()
    restored = config.unbind(config.apply(keys, roles), roles)
    assert torch.equal(keys.view(torch.int64), restored.view(torch.int64))


def test_projected_concat_matches_independent_projection_and_has_no_inverse():
    keys, roles = sample()
    keys /= torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    rng = np.random.Generator(np.random.PCG64([0xB3, 0, 4]))
    codes = torch.from_numpy(2 * rng.integers(0, 2, (5, 64)) - 1).to(torch.float64)
    rng = np.random.Generator(np.random.PCG64([0xB3, 1, 4]))
    projection = torch.from_numpy(2 * rng.integers(0, 2, (128, 64)) - 1).to(torch.float64) / 8
    expected = torch.cat((keys, codes / 8), dim=1) @ projection
    expected /= torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    config = RoleBindingConfig(mode="projected_concat", seed=4)
    torch.testing.assert_close(config.apply(keys, roles), expected, rtol=2e-15, atol=2e-16)
    assert config.numeric_state_bytes == (5 * 64 + 128 * 64) * 8
    with pytest.raises(ValueError, match="inversa"):
        config.unbind(keys, roles)


@pytest.mark.parametrize("mode", ["bipolar", "constant_role", "projected_concat"])
def test_empty_keys_and_permuted_rows_are_supported(mode):
    keys, roles = sample()
    config = RoleBindingConfig(mode=mode)
    assert config.apply(keys[:0], roles[:0]).shape == (0, 64)
    order = torch.tensor([4, 2, 0, 3, 1])
    torch.testing.assert_close(
        config.apply(keys[order], roles[order]), config.apply(keys, roles)[order], rtol=0, atol=0
    )


def test_nonfinite_projection_is_reported_without_silent_correction():
    keys = torch.full((1, 64), 1e308, dtype=torch.float64)
    with pytest.raises(ValueError, match="no finitos"):
        RoleBindingConfig(mode="projected_concat").apply(keys, torch.tensor([0]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("writes", True),
        ("writes", 99),
        ("cursor", [1, 2, 3]),
        ("matrix_sha256", "wrong"),
        ("configuration", {}),
    ],
)
def test_recovery_rejects_incompatible_or_corrupted_metadata(field, value):
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal"), role_binding=RoleBindingConfig()
    )
    payload = correction.empty().export()
    payload[field] = value
    with pytest.raises(ValueError, match="roles"):
        correction.restore(payload)


def test_distinct_roles_do_not_guarantee_orthogonality_for_sparse_content():
    keys = torch.zeros(5, 64, dtype=torch.float64)
    keys[:, 0] = 1
    bound = RoleBindingConfig().apply(keys, torch.arange(5))
    # Los cinco códigos pueden ser distintos y las claves ligadas seguir colineales.
    assert torch.equal((bound @ bound.T).abs(), torch.ones(5, 5, dtype=torch.float64))


def test_empty_maturity_does_not_apply_forgetting_or_allocate_another_state():
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal", forgetting=0.2), role_binding=RoleBindingConfig()
    )
    memory = correction.empty()
    feedback = correction.feedback(
        ids=[],
        decision_at=[],
        available_at=[],
        keys=torch.empty(0, 64, dtype=torch.float64),
        values=[],
    )
    assert memory.write(feedback, cutoff=10) is memory
    assert memory.numeric_state_bytes == 64 * 8 + 5 * 64 * 8
    assert memory.config == correction.memory
    assert memory.writes == 0 and memory.cursor is None


def test_projection_and_role_streams_do_not_alias_across_seeds():
    first = RoleBindingConfig(seed=0, mode="projected_concat")
    second = RoleBindingConfig(seed=1)
    assert not torch.equal(first._projection[:5] * 8, second._codes)
