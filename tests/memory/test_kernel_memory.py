"""Control de Nadaraya-Watson de B1 frente a su fórmula escrita episodio a episodio."""

import math

import pytest
import torch
from test_area_memory import cohort, unit_keys

from mars_titan.memory.associative_memory import MatureCorrection
from mars_titan.memory.kernel_memory import KernelMemory, KernelMemoryConfig

CONFIG = KernelMemoryConfig(bandwidth=0.2, forgetting=0.1, capacity=64)


def reference(query, episodes, writes, config):
    """ŷ(q) = Σ_i w_i v_i con w_i ∝ exp(qᵀk_i/τ) ρ^(c − c_i), con un bucle por episodio."""
    result = []
    for row in query.tolist():
        weights = [
            math.exp(sum(a * b for a, b in zip(row, key, strict=True)) / config.bandwidth)
            * (1 - config.forgetting) ** (writes - written)
            for key, _, written in episodes
        ]
        total = sum(weights)
        result.append(sum(w * value for w, (_, value, _) in zip(weights, episodes, strict=True)))
        result[-1] /= total
    return torch.tensor(result, dtype=torch.float64).reshape(-1, 1)


def test_read_is_the_weighted_mean_of_the_mature_episodes():
    first, second, query = unit_keys(4, seed=1), unit_keys(3, seed=2), unit_keys(5, seed=3)
    values = [0.4, -0.2, 0.1, 0.3]
    later = [-0.5, 0.25, 0.0]
    memory = KernelMemory(CONFIG)
    assert torch.equal(memory.read(query), torch.zeros((5, 1), dtype=torch.float64))
    memory = memory.write(cohort(first, values, rule="nadaraya_watson"), cutoff=10)
    memory = memory.write(cohort(second, later, start=5, at=20, rule="nw"), cutoff=20)
    episodes = [(key, value, 1) for key, value in zip(first.tolist(), values, strict=True)]
    episodes += [(key, value, 2) for key, value in zip(second.tolist(), later, strict=True)]
    torch.testing.assert_close(
        memory.read(query), reference(query, episodes, 2, CONFIG), rtol=1e-13, atol=1e-15
    )
    assert (memory.writes, memory.cohorts) == (7, 2)
    assert memory.cursor == (20, 15, 7) and memory.size == 7


def test_capacity_drops_the_oldest_episodes_in_canonical_order():
    config = KernelMemoryConfig(bandwidth=0.5, forgetting=0.0, capacity=3)
    keys, query = unit_keys(5, seed=4), unit_keys(2, seed=5)
    memory = KernelMemory(config).write(cohort(keys[:2], [1.0, 2.0], rule="nw"), cutoff=10)
    memory = memory.write(cohort(keys[2:], [3.0, 4.0, 5.0], start=3, at=20, rule="nw"), cutoff=20)
    episodes = [
        (key, value, 2) for key, value in zip(keys[2:].tolist(), [3.0, 4.0, 5.0], strict=True)
    ]
    torch.testing.assert_close(
        memory.read(query), reference(query, episodes, 2, config), rtol=1e-13, atol=1e-15
    )
    assert memory.size == 3 and (memory.writes, memory.cohorts) == (5, 2)


def test_blocks_of_queries_only_change_the_rounding(monkeypatch):
    # Otra forma del producto cambia el orden de las sumas de BLAS, no el resultado.
    from mars_titan.memory import kernel_memory

    keys, query = unit_keys(9, seed=6), unit_keys(7, seed=7)
    memory = KernelMemory(CONFIG).write(
        cohort(keys, torch.linspace(-1, 1, 9), rule="nw"), cutoff=10
    )
    whole = memory.read(query)
    monkeypatch.setattr(kernel_memory, "READ_BLOCK", 9 * 2)
    torch.testing.assert_close(memory.read(query), whole, rtol=1e-14, atol=1e-16)


def test_export_and_restore_reproduce_the_episodes_and_reject_changes():
    keys = unit_keys(4, seed=8)
    memory = KernelMemory(CONFIG).write(cohort(keys, [0.1, 0.2, 0.3, 0.4], rule="nw"), cutoff=10)
    payload = memory.export()
    restored = MatureCorrection(CONFIG).restore(payload)
    query = unit_keys(3, seed=9)
    assert torch.equal(restored.read(query), memory.read(query))
    assert (restored.writes, restored.cohorts, restored.cursor) == (4, 1, (10, 5, 4))
    changed = dict(payload, values=payload["values"] + 1.0)
    with pytest.raises(ValueError, match="huella"):
        KernelMemory.restore(CONFIG, changed)
    with pytest.raises(ValueError, match="otra configuración"):
        KernelMemory.restore(
            KernelMemoryConfig(bandwidth=0.3, forgetting=0.1, capacity=64), payload
        )


def test_writes_reject_labels_that_are_not_mature_or_already_applied():
    keys = unit_keys(2, seed=10)
    memory = KernelMemory(CONFIG)
    with pytest.raises(ValueError, match="antes del corte"):
        memory.write(cohort(keys, [0.1, 0.2], at=11, rule="nw"), cutoff=10)
    written = memory.write(cohort(keys, [0.1, 0.2], rule="nw"), cutoff=10)
    with pytest.raises(ValueError, match="orden canónico"):
        written.write(cohort(keys, [0.1, 0.2], rule="nw"), cutoff=10)


@pytest.mark.parametrize(
    "options",
    [
        dict(bandwidth=0.0),
        dict(bandwidth=0.1, forgetting=1.0),
        dict(bandwidth=0.1, capacity=0),
        dict(bandwidth=0.1, capacity=65537),
        dict(bandwidth=float("nan")),
        dict(bandwidth=0.1, key_size=32),
    ],
)
def test_configuration_rejects_values_outside_the_control(options):
    with pytest.raises(ValueError):
        KernelMemoryConfig(**options)
