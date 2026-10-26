"""Memoria redundante por áreas de B1 frente a sus ecuaciones y sus controles.

Las referencias se construyen aparte: DPFP con un bucle sobre la ecuación 37 de Schlag et al.
(2021) y la escritura proximal con su sistema lineal resuelto por `torch.linalg.solve`. Ninguna
prueba recorre muchas escrituras para comprobar que la memoria aprende una señal sintética.
"""

import numpy as np
import pytest
import torch

from mars_titan.memory.area_memory import AreaLayout, AreaMemory, dpfp
from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    KalmanNoise,
    MatureCorrection,
    MatureFeedback,
)

PROXIMAL = AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01)
DELTA = AssociativeMemoryConfig("delta", rate=0.5, forgetting=0.1)


def unit_keys(rows, size=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn((rows, size), dtype=torch.float64, generator=generator)
    return keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)


def cohort(keys, values, *, start=1, at=10, rule="proximal"):
    rows = keys.shape[0]
    ids = torch.arange(start, start + rows, dtype=torch.int64)
    return MatureFeedback(
        ids=ids,
        decision_at=torch.full((rows,), at - 5, dtype=torch.int64),
        available_at=torch.full((rows,), at, dtype=torch.int64),
        keys=keys,
        values=torch.as_tensor(values, dtype=torch.float64).reshape(rows, 1),
        weights=torch.full((rows,), 1.0 / rows, dtype=torch.float64)
        if rule == "proximal"
        else None,
    )


def reference_dpfp(vector, shifts):
    """Ecuación 37 con índices explícitos: r_i · r_{(i − j) mod 2d} en el bloque j."""
    rectified = [max(value, 0.0) for value in vector] + [max(-value, 0.0) for value in vector]
    width = len(rectified)
    return [
        rectified[i] * rectified[(i - j) % width]
        for j in range(1, shifts + 1)
        for i in range(width)
    ]


def reference_features(keys, layout):
    """Proyección de cada área sin pasar por `AreaMemory`: permutar, DPFP y norma uno."""
    result = []
    for permutation in layout.permutations(keys.shape[1]):
        rows = []
        for key in keys.tolist():
            projected = torch.tensor(
                reference_dpfp([key[i] for i in permutation], layout.shifts), dtype=torch.float64
            )
            norm = torch.linalg.vector_norm(projected)
            rows.append(projected / norm if norm > 0 else projected)
        result.append(torch.stack(rows))
    return result


def proximal_step(matrix, features, values, config):
    """(I + η KᵀWK) A⁺ = ρA + η KᵀWv con pesos uniformes, resuelto sin factorizar a mano."""
    rows, size = features.shape
    weighted = features / rows
    system = torch.eye(size, dtype=torch.float64) + config.rate * weighted.T @ features
    right = config.retention * matrix + config.rate * weighted.T @ values.reshape(rows, 1)
    return torch.linalg.solve(system, right)


@pytest.mark.parametrize("shifts", [1, 2, 4])
def test_dpfp_matches_equation_37_and_the_listing_order(shifts):
    keys = unit_keys(5, 6, seed=shifts)
    projected = dpfp(keys, shifts)
    assert projected.shape == (5, 2 * 6 * shifts) and bool((projected >= 0).all())
    expected = torch.tensor(
        [reference_dpfp(key, shifts) for key in keys.tolist()], dtype=torch.float64
    )
    torch.testing.assert_close(projected, expected, rtol=0, atol=0)


def test_seeded_permutations_are_fixed_distinct_and_identity_is_the_control():
    layout = AreaLayout(areas=4, seed=11)
    first, second = layout.permutations(64), AreaLayout(areas=4, seed=11).permutations(64)
    assert np.array_equal(first, second)
    assert len({tuple(row) for row in first}) == 4
    assert all(sorted(row) == list(range(64)) for row in first.tolist())
    identity = AreaLayout(areas=4, permutation="identity").permutations(64)
    assert np.array_equal(identity, np.tile(np.arange(64), (4, 1)))
    assert not np.array_equal(first, AreaLayout(areas=4, seed=12).permutations(64))


@pytest.mark.parametrize("config", [PROXIMAL, DELTA], ids=["proximal", "delta"])
def test_permuting_a_linear_key_leaves_every_read_unchanged(config):
    """Contraejemplo que motiva DPFP: sin una proyección no lineal, las áreas son copias.

    Si Q es ortogonal, la memoria escrita con Qk es Q por la escrita con k, en las dos reglas,
    y la lectura (Qq)ᵀ(QA) coincide. Por eso permutar la clave de un estado matricial completo
    no reduce ningún ruido.
    """
    keys, query = unit_keys(16, seed=1), unit_keys(3, seed=2)
    values = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    plain = AssociativeMemory(config)
    plain = plain.write(cohort(keys[:8], values[:8], rule=config.rule), cutoff=10)
    plain = plain.write(cohort(keys[8:], values[8:], start=9, at=20, rule=config.rule), cutoff=20)
    for permutation in AreaLayout(areas=3, seed=5).permutations(64):
        index = torch.from_numpy(permutation)
        area = AssociativeMemory(config)
        area = area.write(cohort(keys[:8, index], values[:8], rule=config.rule), cutoff=10)
        area = area.write(
            cohort(keys[8:, index], values[8:], start=9, at=20, rule=config.rule), cutoff=20
        )
        torch.testing.assert_close(
            area.read(query[:, index]), plain.read(query), rtol=0, atol=1e-13
        )


def test_area_write_and_read_follow_the_proximal_equation_per_area():
    layout = AreaLayout(areas=3, shifts=2, seed=7)
    memory = AreaMemory(PROXIMAL, layout)
    keys, query = unit_keys(6, seed=3), unit_keys(4, seed=4)
    values = torch.tensor([0.3, -0.1, 0.25, 0.0, -0.4, 0.2], dtype=torch.float64)
    later = torch.tensor([0.05, -0.2, 0.1], dtype=torch.float64)
    written = memory.write(cohort(keys, values), cutoff=10)
    written = written.write(cohort(keys[:3], later, start=7, at=20), cutoff=20)
    matrices = []
    for first, second in zip(
        reference_features(keys, layout), reference_features(keys[:3], layout), strict=True
    ):
        matrix = torch.zeros((first.shape[1], 1), dtype=torch.float64)
        matrix = proximal_step(matrix, first, values, PROXIMAL)
        matrices.append(proximal_step(matrix, second, later, PROXIMAL))
    expected = torch.stack(
        [
            part @ matrix
            for part, matrix in zip(reference_features(query, layout), matrices, strict=True)
        ]
    )
    torch.testing.assert_close(written.read_areas(query), expected, rtol=0, atol=1e-13)
    torch.testing.assert_close(written.read(query), expected.mean(dim=0), rtol=0, atol=1e-13)
    assert written.writes == 9 and written.cursor == (20, 15, 9)
    # Escribir devuelve otra memoria y deja intacta la anterior.
    assert torch.count_nonzero(memory.matrix) == 0 and memory.writes == 0


def test_seeded_areas_disagree_where_identical_areas_reduce_to_one():
    keys, query = unit_keys(12, seed=5), unit_keys(5, seed=6)
    values = torch.linspace(-0.5, 0.5, 12, dtype=torch.float64)
    seeded = AreaMemory(PROXIMAL, AreaLayout(areas=4, seed=3)).write(
        cohort(keys, values), cutoff=10
    )
    reads = seeded.read_areas(query)
    assert all(not torch.allclose(reads[0], reads[s], atol=1e-6) for s in range(1, 4))
    identical = AreaMemory(PROXIMAL, AreaLayout(areas=4, permutation="identity"))
    single = AreaMemory(PROXIMAL, AreaLayout(areas=1, permutation="identity"))
    identical = identical.write(cohort(keys, values), cutoff=10)
    single = single.write(cohort(keys, values), cutoff=10)
    # Con cuatro copias iguales la media en coma flotante es exacta: x + x + x + x = 4x.
    assert torch.equal(identical.read(query), single.read(query))


def test_single_area_control_has_the_same_state_size_and_another_read():
    keys, query = unit_keys(12, seed=7), unit_keys(5, seed=8)
    values = torch.linspace(-0.5, 0.5, 12, dtype=torch.float64)
    areas = AreaMemory(PROXIMAL, AreaLayout(areas=4, shifts=1, seed=1))
    joint = AreaMemory(PROXIMAL, AreaLayout(areas=1, shifts=4, permutation="identity"))
    assert areas.matrix.numel() == joint.matrix.numel() == 4 * 2 * 64
    areas = areas.write(cohort(keys, values), cutoff=10)
    joint = joint.write(cohort(keys, values), cutoff=10)
    assert not torch.allclose(areas.read(query), joint.read(query), atol=1e-6)


def test_shuffled_labels_write_a_fixed_permutation_of_the_cohort_values():
    layout = AreaLayout(areas=2, seed=9, labels="shuffled_within_cohort")
    keys, query = unit_keys(7, seed=9), unit_keys(3, seed=10)
    values = torch.tensor([0.7, -0.2, 0.1, 0.4, -0.6, 0.3, 0.05], dtype=torch.float64)
    shuffled = AreaMemory(PROXIMAL, layout).write(cohort(keys, values, start=5), cutoff=10)
    permutation = np.random.Generator(np.random.PCG64([9, 5])).permutation(7)
    assert sorted(values[permutation].tolist()) == sorted(values.tolist())
    mature = AreaMemory(PROXIMAL, AreaLayout(areas=2, seed=9))
    expected = mature.write(cohort(keys, values[permutation], start=5), cutoff=10)
    torch.testing.assert_close(shuffled.read(query), expected.read(query), rtol=0, atol=0)
    # El barajado sigue el orden canónico, así que entregar las filas al revés no lo cambia.
    reverse = torch.arange(6, -1, -1)
    feedback = cohort(keys, values, start=5)
    reversed_feedback = MatureFeedback(
        ids=feedback.ids[reverse],
        decision_at=feedback.decision_at[reverse],
        available_at=feedback.available_at[reverse],
        keys=keys[reverse],
        values=feedback.values[reverse],
        weights=feedback.weights[reverse],
    )
    again = AreaMemory(PROXIMAL, layout).write(reversed_feedback, cutoff=10)
    torch.testing.assert_close(again.read(query), shuffled.read(query), rtol=0, atol=1e-15)


def test_lesion_reads_the_renormalized_mean_of_the_active_areas():
    keys, query = unit_keys(10, seed=11), unit_keys(4, seed=12)
    memory = AreaMemory(PROXIMAL, AreaLayout(areas=4, seed=2)).write(
        cohort(keys, torch.linspace(-1, 1, 10, dtype=torch.float64)), cutoff=10
    )
    reads = memory.read_areas(query)
    active = torch.tensor([True, False, True, False])
    torch.testing.assert_close(
        memory.read(query, active), (reads[0] + reads[2]) / 2, rtol=0, atol=1e-16
    )
    for invalid in (
        torch.zeros(4, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        [1, 0, 1, 0],
    ):
        with pytest.raises(ValueError, match="máscara booleana"):
            memory.read(query, invalid)


def test_a_key_without_neighbour_pairs_neither_reads_nor_writes():
    lonely = torch.zeros((1, 64), dtype=torch.float64)
    lonely[0, 5] = 1.0
    memory = AreaMemory(PROXIMAL, AreaLayout(areas=2, seed=4))
    assert torch.count_nonzero(memory.features(lonely)) == 0
    keys = torch.cat([lonely, unit_keys(3, seed=13)])
    written = memory.write(cohort(keys, [5.0, 0.1, 0.2, 0.3]), cutoff=10)
    assert torch.count_nonzero(written.read(lonely)) == 0


def test_export_and_restore_reproduce_the_memory_and_reject_changes():
    layout = AreaLayout(areas=3, seed=6)
    keys = unit_keys(5, seed=14)
    memory = AreaMemory(PROXIMAL, layout).write(cohort(keys, [0.1, 0.2, -0.3, 0.4, 0.0]), cutoff=10)
    payload = memory.export()
    restored = AreaMemory.restore(PROXIMAL, layout, payload)
    assert torch.equal(restored.matrix, memory.matrix)
    assert (restored.writes, restored.cursor) == (memory.writes, memory.cursor)
    correction = MatureCorrection(PROXIMAL, areas=layout)
    assert torch.equal(correction.restore(payload).matrix, memory.matrix)
    with pytest.raises(ValueError, match="otra memoria por áreas"):
        AreaMemory.restore(PROXIMAL, AreaLayout(areas=3, seed=7), payload)
    tampered = dict(payload, matrix_sha256="0" * 64)
    with pytest.raises(ValueError, match="huella"):
        AreaMemory.restore(PROXIMAL, layout, tampered)
    with pytest.raises(ValueError, match="campos"):
        AreaMemory.restore(PROXIMAL, layout, {k: v for k, v in payload.items() if k != "cursor"})


def test_writes_reject_labels_that_are_not_mature_or_already_applied():
    memory = AreaMemory(PROXIMAL, AreaLayout(areas=2))
    keys = unit_keys(3, seed=15)
    with pytest.raises(ValueError, match="antes del corte"):
        memory.write(cohort(keys, [0.1, 0.2, 0.3], at=11), cutoff=10)
    written = memory.write(cohort(keys, [0.1, 0.2, 0.3]), cutoff=10)
    with pytest.raises(ValueError, match="orden canónico"):
        written.write(cohort(keys, [0.1, 0.2, 0.3]), cutoff=10)
    with pytest.raises(ValueError, match="norma"):
        memory.write(cohort(2 * keys, [0.1, 0.2, 0.3]), cutoff=10)


def test_configuration_rejects_what_the_variant_does_not_define():
    with pytest.raises(ValueError):
        AreaLayout(areas=0)
    with pytest.raises(ValueError):
        AreaLayout(shifts=5)
    with pytest.raises(ValueError):
        AreaLayout(permutation="random")
    with pytest.raises(ValueError):
        AreaLayout(labels="shuffled")
    with pytest.raises(ValueError, match="regla delta o proximal"):
        AreaMemory(
            AssociativeMemoryConfig("kalman", kalman=KalmanNoise(1e-4, 0.5, 0.2, 0.5)),
            AreaLayout(),
        )
    # La regla delta conserva su cota η ≤ 2 − λ en cada área, porque ‖φ‖ = 1.
    with pytest.raises(ValueError, match="2 − λ"):
        AreaMemory(AssociativeMemoryConfig("delta", rate=1.95, forgetting=0.1), AreaLayout())
