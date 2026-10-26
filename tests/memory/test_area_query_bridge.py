"""Traducción de consultas entre áreas, frente a sus ecuaciones y controles."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_area_memory import cohort, reference_dpfp, unit_keys

from mars_titan.memory.area_memory import AreaLayout, AreaMemory
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig, AreaQueryBridgeState
from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureCorrection,
)

RULE = AssociativeMemoryConfig("proximal", key_size=8, value_size=8, rate=0.25, forgetting=0.01)


def feedback(rows=5):
    keys = unit_keys(rows, size=4, seed=12)
    return cohort(keys, torch.linspace(-0.1, 0.2, rows, dtype=torch.float64))


def features(rows=5):
    return torch.stack([unit_keys(rows, size=8, seed=i) for i in range(3)])


def solve_map(source, target):
    rows, size = source.shape
    system = torch.eye(size, dtype=torch.float64) + RULE.rate * source.T @ source / rows
    return torch.linalg.solve(system, RULE.rate * source.T @ target / rows)


def unit(values):
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / torch.where(norm > 0, norm, 1)


def test_bridge_maps_follow_the_cohort_proximal_equation_without_financial_values():
    basis = features()
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig())
    written = state.write(basis, feedback(), cutoff=10)
    expected = torch.stack([solve_map(basis[i], basis[(i + 1) % 3]) for i in range(3)])
    torch.testing.assert_close(written.matrices, expected, rtol=1e-13, atol=1e-15)
    changed = replace(feedback(), values=torch.full((5, 1), 123.0, dtype=torch.float64))
    torch.testing.assert_close(
        state.write(basis, changed, cutoff=10).matrices, written.matrices, rtol=0, atol=0
    )
    assert state.writes == 0 and torch.count_nonzero(state.matrices) == 0
    assert written.writes == 5 and written.cursor == (10, 5, 5)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_query_messages_are_synchronous_and_use_the_previous_step(steps):
    config = AreaQueryBridgeConfig(gain=0.25, steps=steps)
    state = AreaQueryBridgeState(RULE, 3, config).write(features(), feedback(), cutoff=10)
    original = features(2)
    expected = original
    for _ in range(steps):
        expected = torch.stack(
            [
                unit(
                    0.75 * expected[b]
                    + 0.25 * (expected[(b - 1) % 3] @ state.matrices[(b - 1) % 3])
                )
                for b in range(3)
            ]
        )
    actual = state.apply(original)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)
    assert not torch.allclose(actual, original)
    assert torch.equal(original, features(2))


def test_zero_gain_returns_the_same_queries_and_still_accounts_for_maps():
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig(gain=0)).write(
        features(), feedback(), cutoff=10
    )
    query = features(2)
    assert state.apply(query) is query
    assert state.numeric_state_bytes == 3 * 8 * 8 * 8
    assert state.writes == 5 and torch.count_nonzero(state.matrices) > 0


def test_shuffled_correspondence_preserves_targets_and_is_independent_of_delivery_order():
    config = AreaQueryBridgeConfig(targets="shuffled_within_cohort", seed=7)
    basis, data = features(), feedback()
    state = AreaQueryBridgeState(RULE, 3, config)
    written = state.write(basis, data, cutoff=10)
    expected = []
    for source in range(3):
        order = np.random.Generator(np.random.PCG64([178, 7, 1, source])).permutation(5)
        expected.append(solve_map(basis[source], basis[(source + 1) % 3, order]))
    torch.testing.assert_close(written.matrices, torch.stack(expected), rtol=1e-13, atol=1e-15)
    reverse = torch.arange(4, -1, -1)
    reversed_data = replace(
        data,
        **{
            name: getattr(data, name)[reverse]
            for name in ("ids", "keys", "values", "decision_at", "available_at", "weights")
        },
    )
    torch.testing.assert_close(
        state.write(basis[:, reverse], reversed_data, cutoff=10).matrices,
        written.matrices,
        rtol=0,
        atol=0,
    )


def test_bridge_shuffle_does_not_reuse_the_area_label_shuffle_stream():
    config = AreaQueryBridgeConfig(targets="shuffled_within_cohort", seed=7)
    basis, data = features(7), feedback(7)
    state = AreaQueryBridgeState(RULE, 3, config).write(basis, data, cutoff=10)
    label_shuffle = np.random.Generator(np.random.PCG64([7, 1])).permutation(7)
    reused = solve_map(basis[0], basis[1, label_shuffle])
    assert not torch.allclose(state.matrices[0], reused, rtol=1e-13, atol=1e-15)


def test_bridge_rejects_future_and_repeated_cohorts_without_mutating_state():
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig())
    with pytest.raises(ValueError, match="antes del corte"):
        state.write(features(), feedback(), cutoff=9)
    written = state.write(features(), feedback(), cutoff=10)
    with pytest.raises(ValueError, match="orden canónico"):
        written.write(features(), feedback(), cutoff=10)
    assert state.writes == 0


def test_bridge_keeps_query_gradients_and_detaches_the_written_state():
    source = features().requires_grad_()
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig()).write(
        source, feedback(), cutoff=10
    )
    query = (features(2) * 0.8).requires_grad_()
    assert torch.autograd.gradcheck(state.apply, (query,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert not state.matrices.requires_grad


def test_restore_preserves_maps_and_queries_and_rejects_configuration_or_digest_changes():
    config = AreaQueryBridgeConfig()
    state = AreaQueryBridgeState(RULE, 3, config).write(features(), feedback(), cutoff=10)
    payload = state.export()
    restored = AreaQueryBridgeState.restore(RULE, 3, config, payload)
    torch.testing.assert_close(
        restored.apply(features(2)), state.apply(features(2)), rtol=0, atol=0
    )
    with pytest.raises(ValueError):
        AreaQueryBridgeState.restore(RULE, 3, replace(config, gain=0.5), payload)
    payload["maps"][0]["matrix"][0, 0] += 1
    with pytest.raises(ValueError):
        AreaQueryBridgeState.restore(RULE, 3, config, payload)


def test_exported_maps_have_owned_contiguous_storage_for_the_session():
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig()).write(
        features(), feedback(), cutoff=10
    )
    for item in state.export()["maps"]:
        matrix = item["matrix"]
        assert matrix.is_contiguous()
        assert matrix.storage_offset() == 0
        assert matrix.untyped_storage().nbytes() == matrix.nbytes


def test_overflow_in_a_finite_map_is_rejected_before_normalizing_the_query():
    maps = tuple(
        AssociativeMemory(RULE, torch.full((8, 8), 1e308, dtype=torch.float64)) for _ in range(3)
    )
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig(), maps=maps)
    with pytest.raises(ValueError, match="no finitos"):
        state.apply(torch.full((3, 1, 8), 8**-0.5, dtype=torch.float64))


@pytest.mark.parametrize(
    "options",
    [
        dict(gain=True),
        dict(gain=float("nan")),
        dict(gain=1.1),
        dict(steps=0),
        dict(steps=1.0),
        dict(targets="future"),
        dict(seed=-1),
    ],
)
def test_invalid_bridge_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        AreaQueryBridgeConfig(**options)


def test_constructing_and_writing_the_bridge_does_not_touch_global_random_generators():
    torch_before, numpy_before = torch.get_rng_state(), np.random.get_state()
    AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig(targets="shuffled_within_cohort")).write(
        features(), feedback(), cutoff=10
    )
    assert torch.equal(torch_before, torch.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_area_memory_without_a_bridge_preserves_its_export_and_bitwise_reads():
    rule, layout = AssociativeMemoryConfig("proximal"), AreaLayout(areas=3, seed=3)
    plain = AreaMemory(rule, layout)
    explicit = AreaMemory(rule, layout, query_bridge=None)
    data = cohort(unit_keys(5, seed=4), torch.linspace(-0.1, 0.1, 5, dtype=torch.float64))
    plain, explicit = plain.write(data, cutoff=10), explicit.write(data, cutoff=10)
    assert plain.identity() == explicit.identity() and "query_bridge" not in plain.identity()
    assert plain.export()["schema_version"] == 1 and "query_bridge" not in plain.export()
    assert torch.equal(plain.read(unit_keys(3)), explicit.read(unit_keys(3)))


def test_area_bridge_is_used_before_reading_corrections_and_is_part_of_the_checkpoint():
    rule, layout = AssociativeMemoryConfig("proximal", rate=0.25), AreaLayout(areas=3, seed=3)
    bridge = AreaQueryBridgeConfig()
    correction = MatureCorrection(rule, areas=layout, query_bridge=bridge)
    memory = correction.empty()
    data = cohort(unit_keys(5, seed=4), torch.linspace(-0.1, 0.1, 5, dtype=torch.float64))
    plain = AreaMemory(rule, layout).write(data, cutoff=10)
    written = memory.write(data, cutoff=10)
    query = unit_keys(3)
    assert not torch.allclose(written.read(query), plain.read(query), rtol=1e-6, atol=1e-10)
    restored = correction.restore(written.export())
    assert torch.equal(restored.matrix, written.matrix)
    assert torch.equal(restored.read(query), written.read(query))
    assert written.numeric_state_bytes == plain.numeric_state_bytes + 3 * 128 * 128 * 8
    assert written.export()["matrix_sha256"] != plain.export()["matrix_sha256"]
    with pytest.raises(ValueError):
        MatureCorrection(rule, areas=layout).restore(written.export())


def test_zero_gain_bridge_preserves_area_predictions_with_the_same_written_maps():
    rule, layout = AssociativeMemoryConfig("proximal", rate=0.25), AreaLayout(areas=3, seed=3)
    zero = AreaMemory(rule, layout, query_bridge=AreaQueryBridgeConfig(gain=0))
    plain = AreaMemory(rule, layout)
    data = cohort(unit_keys(5, seed=4), torch.linspace(-0.1, 0.1, 5, dtype=torch.float64))
    zero, plain = zero.write(data, cutoff=10), plain.write(data, cutoff=10)
    assert torch.equal(zero.read(unit_keys(3)), plain.read(unit_keys(3)))


def test_bridge_requires_areas_and_rejects_the_single_area_control():
    with pytest.raises(ValueError):
        MatureCorrection(AssociativeMemoryConfig("proximal"), query_bridge=AreaQueryBridgeConfig())
    with pytest.raises(ValueError):
        AreaMemory(
            AssociativeMemoryConfig("proximal"),
            AreaLayout(areas=1),
            query_bridge=AreaQueryBridgeConfig(),
        )


def test_inactive_areas_cannot_send_or_receive_query_messages():
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig()).write(
        features(), feedback(), cutoff=10
    )
    query, active = features(2), torch.tensor([True, False, True])
    actual = state.apply(query, active=active)
    altered = query.clone()
    altered[1] *= -1
    assert torch.equal(actual, state.apply(altered, active=active))
    assert torch.count_nonzero(actual[1]) == 0
    torch.testing.assert_close(actual[2], query[2], rtol=1e-14, atol=1e-15)
    expected_zero = unit(0.75 * query[0] + 0.25 * query[2] @ state.matrices[2])
    torch.testing.assert_close(actual[0], expected_zero, rtol=1e-14, atol=1e-15)


def test_sparse_features_are_rejected_before_attempting_to_flatten_them():
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig())
    with pytest.raises(ValueError):
        state.apply(features().to_sparse())


def test_bridge_checkpoint_schema_requires_an_integer_version():
    config = AreaQueryBridgeConfig()
    state = AreaQueryBridgeState(RULE, 3, config)
    payload = state.export()
    payload["schema_version"] = True
    with pytest.raises(ValueError):
        AreaQueryBridgeState.restore(RULE, 3, config, payload)


def test_area_checkpoint_rejects_float_versions_and_a_bridge_with_another_cursor_count():
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal"),
        areas=AreaLayout(areas=2),
        query_bridge=AreaQueryBridgeConfig(),
    )
    memory = correction.empty().write(cohort(unit_keys(3), [0.1, 0.2, 0.3]), cutoff=10)
    payload = memory.export()
    payload["schema_version"] = 2.0
    with pytest.raises(ValueError):
        correction.restore(payload)
    payload = memory.export()
    for item in payload["query_bridge"]["maps"]:
        item["writes"] += 1
    with pytest.raises(ValueError, match="no corresponden"):
        correction.restore(payload)


def test_delta_bridge_follows_the_canonical_sequential_update():
    rule = replace(RULE, rule="delta", rate=0.5, forgetting=0.1)
    basis, data = features(), replace(feedback(), weights=None)
    state = AreaQueryBridgeState(rule, 3, AreaQueryBridgeConfig()).write(basis, data, cutoff=10)
    expected = []
    for area in range(3):
        matrix = torch.zeros((8, 8), dtype=torch.float64)
        for key, value in zip(basis[area], basis[(area + 1) % 3], strict=True):
            matrix = 0.9 * matrix + 0.5 * torch.outer(key, value - key @ matrix)
        expected.append(matrix)
    torch.testing.assert_close(state.matrices, torch.stack(expected), rtol=1e-13, atol=1e-15)


def test_empty_cohort_does_not_apply_forgetting_or_change_the_cursor():
    state = AreaQueryBridgeState(RULE, 3, AreaQueryBridgeConfig()).write(
        features(), feedback(), cutoff=10
    )
    data = feedback()
    empty = replace(
        data,
        **{
            name: getattr(data, name)[:0]
            for name in ("ids", "keys", "values", "decision_at", "available_at")
        },
        weights=None,
    )
    assert state.write(features()[:, :0], empty, cutoff=11) is state


def test_dpfp_can_erase_information_needed_by_another_area_so_no_exact_translation_is_claimed():
    key = [2**-0.5, 0, 2**-0.5, 0]
    source = torch.tensor(reference_dpfp(key, 1), dtype=torch.float64)
    target = torch.tensor(reference_dpfp([key[i] for i in (0, 2, 1, 3)], 1), dtype=torch.float64)
    assert torch.count_nonzero(source) == 0 and torch.count_nonzero(target) > 0
    assert torch.equal(source @ torch.eye(8, dtype=torch.float64), source)


def test_area_read_excludes_both_messages_and_outputs_of_inactive_areas():
    rule, layout, config = (
        AssociativeMemoryConfig("proximal"),
        AreaLayout(areas=3),
        AreaQueryBridgeConfig(),
    )
    correction = MatureCorrection(rule, areas=layout, query_bridge=config)
    memory = correction.empty().write(cohort(unit_keys(5), [0.1, -0.2, 0.3, -0.1, 0.2]), cutoff=10)
    payload, query, active = (
        memory.export(),
        unit_keys(2, seed=19),
        torch.tensor([True, False, True]),
    )
    bridge_rule = replace(rule, key_size=128, value_size=128)
    bridge = AreaQueryBridgeState.restore(bridge_rule, 3, config, payload["query_bridge"])
    translated = bridge.apply(memory.features(query), active=active)
    expected = torch.stack([translated[i] @ payload["areas"][i]["matrix"] for i in (0, 2)]).mean(0)
    torch.testing.assert_close(memory.read(query, active=active), expected, rtol=0, atol=0)
