"""Memoria asociativa delta y proximal con resultados manuales, sin ajustar parámetros.

Cada prueba compara una escritura con su ecuación o con una cota derivada. Ninguna
recorre muchas escrituras para comprobar que la memoria aprende una señal sintética.
"""

import math

import pytest
import torch

from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureFeedback,
)


def unit_keys(rows, size, seed):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn((rows, size), dtype=torch.float64, generator=generator)
    return keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)


def feedback(keys, values, *, start=1, available=None, weights=None):
    rows = keys.shape[0]
    ids = torch.arange(start, start + rows, dtype=torch.int64)
    available = ids * 10 if available is None else torch.tensor(available, dtype=torch.int64)
    return MatureFeedback(
        ids=ids,
        decision_at=available - 5,
        available_at=available,
        keys=keys,
        values=values,
        weights=weights,
    )


def uniform(rows):
    return torch.full((rows,), 1.0 / rows, dtype=torch.float64)


def state(config, seed):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(
        (config.key_size, config.value_size), dtype=torch.float64, generator=generator
    )
    return AssociativeMemory(config, matrix)


def test_delta_write_matches_the_rank_one_equation():
    config = AssociativeMemoryConfig("delta", key_size=8, value_size=3, rate=0.7, forgetting=0.2)
    memory = state(config, 1)
    key = unit_keys(1, 8, 2)
    value = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float64)
    written = memory.write(feedback(key, value), cutoff=10)
    previous = memory.matrix
    expected = 0.8 * previous + 0.7 * torch.outer(key[0], value[0] - key[0] @ previous)
    torch.testing.assert_close(written.matrix, expected, rtol=0, atol=1e-15)
    assert written.writes == 1 and written.cursor == (10, 5, 1)
    torch.testing.assert_close(memory.matrix, previous, rtol=0, atol=0)


@pytest.mark.parametrize(("forgetting", "rate"), [(0.0, 2.0), (0.1, 1.2), (0.5, 0.25), (1.0, 1.0)])
def test_delta_difference_follows_its_operator_and_never_expands(forgetting, rate):
    config = AssociativeMemoryConfig(
        "delta", key_size=6, value_size=2, rate=rate, forgetting=forgetting
    )
    left, right = state(config, 3), state(config, 4)
    key = unit_keys(1, 6, 5) * 0.9
    value = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    change = feedback(key, value)
    difference = left.write(change, cutoff=10).matrix - right.write(change, cutoff=10).matrix
    operator = (1 - forgetting) * torch.eye(6, dtype=torch.float64) - rate * torch.outer(
        key[0], key[0]
    )
    torch.testing.assert_close(
        difference, operator @ (left.matrix - right.matrix), atol=1e-14, rtol=0
    )
    spectral = torch.linalg.matrix_norm(operator, ord=2)
    squared = float(key[0] @ key[0])
    bound = max(abs(1 - forgetting), abs(1 - forgetting - rate * squared))
    assert math.isclose(float(spectral), bound, rel_tol=0, abs_tol=1e-14)
    assert bound <= config.contraction_bound() + 1e-15 <= 1 + 1e-15


def test_documented_counterexample_with_forgetting_is_rejected():
    # λ = 0,1 y η = 1,95 con clave unitaria dan q = 1,05 en la derivación.
    with pytest.raises(ValueError, match="2 − λ"):
        AssociativeMemoryConfig("delta", rate=1.95, forgetting=0.1)
    AssociativeMemoryConfig("delta", rate=1.9, forgetting=0.1)


def test_write_order_is_canonical_and_decided_by_label_availability():
    config = AssociativeMemoryConfig("delta", key_size=1, value_size=1, rate=0.5)
    keys = torch.ones((2, 1), dtype=torch.float64)
    values = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
    empty = AssociativeMemory(config)
    # Las escrituras no conmutan: aplicar 2 y después 4 da 2,5, al revés da 2.
    first = empty.write(feedback(keys, values, available=[10, 20]), cutoff=20)
    second = empty.write(feedback(keys, values, available=[20, 10]), cutoff=20)
    assert float(first.matrix) == 2.5 and float(second.matrix) == 2.0
    swapped = feedback(keys.flip(0), values.flip(0), available=[20, 10])
    swapped = MatureFeedback(
        ids=torch.tensor([2, 1]),
        decision_at=swapped.decision_at,
        available_at=swapped.available_at,
        keys=swapped.keys,
        values=swapped.values,
    )
    assert float(empty.write(swapped, cutoff=20).matrix) == 2.5


def test_ties_in_availability_are_resolved_by_decision_and_then_id():
    config = AssociativeMemoryConfig("delta", key_size=1, value_size=1, rate=0.5)
    keys = torch.ones((3, 1), dtype=torch.float64)
    values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    base = MatureFeedback(
        ids=torch.tensor([7, 3, 5]),
        decision_at=torch.tensor([4, 4, 2]),
        available_at=torch.tensor([10, 10, 10]),
        keys=keys,
        values=values,
    )
    written = AssociativeMemory(config).write(base, cutoff=10)
    # Orden canónico: id 5 (decisión 2), id 3 y después id 7.
    expected = 0.0
    for value in (3.0, 2.0, 1.0):
        expected = 0.5 * expected + 0.5 * value
    assert float(written.matrix) == expected and written.cursor == (10, 4, 7)


def test_a_repeated_or_out_of_order_delivery_is_rejected():
    config = AssociativeMemoryConfig("delta", key_size=4, value_size=1, rate=0.5)
    keys = unit_keys(2, 4, 6)
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    written = AssociativeMemory(config).write(feedback(keys, values), cutoff=20)
    for repeated in (feedback(keys[1:], values[1:], start=2), feedback(keys[:1], values[:1])):
        with pytest.raises(ValueError, match="orden canónico"):
            written.write(repeated, cutoff=30)
    later = written.write(feedback(keys[:1], values[:1], start=3), cutoff=30)
    assert later.writes == 3 and later.cursor == (30, 25, 3)


@pytest.mark.parametrize(
    "fault",
    ["immature", "decision_after_label", "duplicate_id", "zero_id", "key_norm", "nonfinite"],
)
def test_invalid_feedback_fails_before_changing_the_state(fault):
    config = AssociativeMemoryConfig("delta", key_size=3, value_size=1, rate=0.5)
    keys = unit_keys(2, 3, 7)
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    item = feedback(keys, values)
    ids, decisions, available = item.ids, item.decision_at, item.available_at
    cutoff = 20
    if fault == "immature":
        cutoff = 19
    elif fault == "decision_after_label":
        decisions = available.clone()
    elif fault == "duplicate_id":
        ids = torch.tensor([1, 1])
    elif fault == "zero_id":
        ids = torch.tensor([0, 1])
    elif fault == "key_norm":
        keys = keys * 1.001
    else:
        values = torch.tensor([[1.0], [math.nan]], dtype=torch.float64)
    memory = state(config, 8)
    before = memory.matrix
    with pytest.raises(ValueError):
        memory.write(MatureFeedback(ids, decisions, available, keys, values), cutoff=cutoff)
    torch.testing.assert_close(memory.matrix, before, rtol=0, atol=0)
    assert memory.writes == 0 and memory.cursor is None


@pytest.mark.parametrize("rule", ["delta", "proximal"])
def test_empty_cohort_keeps_the_state_without_forgetting(rule):
    config = AssociativeMemoryConfig(rule, key_size=3, value_size=2, rate=0.5, forgetting=0.5)
    memory = state(config, 9)
    empty = MatureFeedback(
        ids=torch.empty(0, dtype=torch.int64),
        decision_at=torch.empty(0, dtype=torch.int64),
        available_at=torch.empty(0, dtype=torch.int64),
        keys=torch.empty((0, 3), dtype=torch.float64),
        values=torch.empty((0, 2), dtype=torch.float64),
    )
    assert memory.write(empty, cutoff=10) is memory


def test_proximal_write_solves_its_normal_equations():
    config = AssociativeMemoryConfig(
        "proximal", key_size=5, value_size=2, rate=3.0, forgetting=0.25
    )
    memory = state(config, 10)
    keys = unit_keys(4, 5, 11)
    values = torch.tensor([[1.0, 0.0], [0.5, -1.0], [2.0, 3.0], [-1.0, 0.25]], dtype=torch.float64)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    written = memory.write(feedback(keys, values, weights=weights), cutoff=40)
    gram = (keys * weights.unsqueeze(1)).T @ keys
    cross = (keys * weights.unsqueeze(1)).T @ values
    system = torch.eye(5, dtype=torch.float64) + 3.0 * gram
    torch.testing.assert_close(
        system @ written.matrix, 0.75 * memory.matrix + 3.0 * cross, rtol=0, atol=1e-13
    )
    assert written.writes == 4 and written.cursor == (40, 35, 4)


def test_proximal_write_is_invariant_to_the_load_order():
    config = AssociativeMemoryConfig("proximal", key_size=4, value_size=1, rate=2.0, forgetting=0.1)
    memory = state(config, 12)
    keys = unit_keys(3, 4, 13)
    values = torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    ordered = memory.write(feedback(keys, values, weights=weights), cutoff=30)
    permutation = torch.tensor([2, 0, 1])
    original = feedback(keys, values, weights=weights)
    shuffled = MatureFeedback(
        *(
            getattr(original, name).index_select(0, permutation)
            for name in ("ids", "decision_at", "available_at", "keys", "values", "weights")
        )
    )
    torch.testing.assert_close(
        memory.write(shuffled, cutoff=30).matrix, ordered.matrix, rtol=0, atol=0
    )


def test_proximal_difference_contracts_with_the_smallest_gram_eigenvalue():
    config = AssociativeMemoryConfig("proximal", key_size=3, value_size=2, rate=4.0, forgetting=0.2)
    left, right = state(config, 14), state(config, 15)
    keys = torch.eye(3, dtype=torch.float64)
    values = torch.ones((3, 2), dtype=torch.float64)
    change = feedback(keys, values, weights=uniform(3))
    after = left.write(change, cutoff=30).matrix - right.write(change, cutoff=30).matrix
    before = left.matrix - right.matrix
    smallest = 1.0 / 3.0
    ratio = float(torch.linalg.matrix_norm(after) / torch.linalg.matrix_norm(before))
    assert ratio <= 0.8 / (1 + 4.0 * smallest) + 1e-14
    # Con claves ortonormales y pesos iguales, la cota se alcanza.
    assert math.isclose(ratio, 0.8 / (1 + 4.0 * smallest), rel_tol=1e-12)


def test_proximal_write_accepts_rank_deficient_keys_without_contraction_beyond_rho():
    config = AssociativeMemoryConfig("proximal", key_size=4, value_size=1, rate=10.0)
    memory = state(config, 16)
    keys = unit_keys(1, 4, 17).repeat(3, 1)
    values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    written = memory.write(feedback(keys, values, weights=uniform(3)), cutoff=30)
    # Con ρ = 1 la dirección ortogonal a la clave no cambia.
    orthogonal = torch.linalg.svd(keys[:1])[2][1:]
    torch.testing.assert_close(
        orthogonal @ written.matrix, orthogonal @ memory.matrix, rtol=0, atol=1e-13
    )


def test_proximal_rule_differs_from_the_explicit_rule_at_second_order():
    config = AssociativeMemoryConfig("proximal", key_size=3, value_size=1, rate=1.0, forgetting=0.3)
    memory = state(config, 18)
    keys = unit_keys(2, 3, 19)
    values = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    weights = uniform(2)
    gram = (keys * weights.unsqueeze(1)).T @ keys
    cross = (keys * weights.unsqueeze(1)).T @ values
    gaps = []
    for rate in (1e-2, 5e-3):
        current = AssociativeMemory(
            AssociativeMemoryConfig("proximal", 3, 1, rate, 0.3), memory.matrix
        )
        written = current.write(feedback(keys, values, weights=weights), cutoff=20).matrix
        first_order = 0.7 * memory.matrix + rate * (cross - 0.7 * gram @ memory.matrix)
        gaps.append(float(torch.linalg.matrix_norm(written - first_order)))
    # El resto es O(η²): reducir η a la mitad divide el resto por cerca de cuatro.
    assert 3.8 < gaps[0] / gaps[1] < 4.2
    assert config.contraction_bound() == pytest.approx(0.7)


@pytest.mark.parametrize("weights", [[0.5, 0.6], [1.5, -0.5], [math.nan, 1.0]])
def test_proximal_weights_must_be_declared_nonnegative_and_sum_one(weights):
    config = AssociativeMemoryConfig("proximal", key_size=2, value_size=1, rate=1.0)
    keys = unit_keys(2, 2, 20)
    values = torch.ones((2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="pesos"):
        AssociativeMemory(config).write(
            feedback(keys, values, weights=torch.tensor(weights, dtype=torch.float64)), cutoff=20
        )
    with pytest.raises(ValueError, match="pesos"):
        AssociativeMemory(config).write(feedback(keys, values), cutoff=20)
    delta = AssociativeMemoryConfig("delta", key_size=2, value_size=1, rate=1.0)
    with pytest.raises(ValueError, match="pesos"):
        AssociativeMemory(delta).write(feedback(keys, values, weights=uniform(2)), cutoff=20)


def test_read_is_pure_and_an_empty_memory_reads_zero():
    config = AssociativeMemoryConfig("delta", key_size=4, value_size=2, rate=0.5)
    keys = unit_keys(3, 4, 21)
    assert torch.equal(
        AssociativeMemory(config).read(keys), torch.zeros((3, 2), dtype=torch.float64)
    )
    memory = state(config, 22)
    before = memory.matrix
    torch.testing.assert_close(memory.read(keys), keys @ before, rtol=0, atol=0)
    assert torch.equal(memory.matrix, before) and memory.writes == 0
    with pytest.raises(ValueError):
        memory.read(keys * 2)
    with pytest.raises(ValueError):
        memory.read(keys.float())


def test_export_restore_roundtrip_and_tampering():
    config = AssociativeMemoryConfig("delta", key_size=3, value_size=1, rate=0.5, forgetting=0.1)
    memory = AssociativeMemory(config).write(
        feedback(unit_keys(2, 3, 23), torch.tensor([[1.0], [2.0]], dtype=torch.float64)), cutoff=20
    )
    payload = memory.export()
    restored = AssociativeMemory.restore(config, payload)
    assert torch.equal(restored.matrix, memory.matrix)
    assert (restored.writes, restored.cursor) == (memory.writes, memory.cursor)
    tampered = dict(payload, matrix=payload["matrix"] + 1e-12)
    with pytest.raises(ValueError, match="huella"):
        AssociativeMemory.restore(config, tampered)
    other = AssociativeMemoryConfig("delta", key_size=3, value_size=1, rate=0.4, forgetting=0.1)
    with pytest.raises(ValueError, match="otra configuración"):
        AssociativeMemory.restore(other, payload)
    with pytest.raises(ValueError, match="coherentes"):
        AssociativeMemory.restore(config, dict(payload, cursor=None))


def test_identities_separate_rules_and_rates():
    delta = AssociativeMemoryConfig("delta", rate=0.5)
    proximal = AssociativeMemoryConfig("proximal", rate=0.5)
    assert delta.fingerprint() != proximal.fingerprint()
    assert delta.fingerprint() != AssociativeMemoryConfig("delta", rate=0.25).fingerprint()
    assert AssociativeMemoryConfig("delta", rate=1).identity()["rate"] == 1.0
    for invalid in (
        dict(rule="rls"),
        dict(rule="delta", rate=-1.0),
        dict(rule="delta", key_size=0),
    ):
        with pytest.raises(ValueError):
            AssociativeMemoryConfig(**invalid)
