"""Estados incompatibles y precisión de las memorias por áreas, sin entrenamiento."""

import pytest
import torch
from test_area_memory import unit_keys

from mars_titan.memory.area_memory import AreaLayout, AreaMemory
from mars_titan.memory.associative_memory import AssociativeMemoryConfig
from mars_titan.memory.kernel_memory import KernelMemory, KernelMemoryConfig


@pytest.mark.parametrize(
    "options", [dict(key_size=64.0), dict(value_size=True), dict(value_size=1.0)]
)
def test_kernel_dimensions_require_integer_types(options):
    with pytest.raises(ValueError):
        KernelMemoryConfig(bandwidth=0.2, **options)


@pytest.mark.parametrize("which", ["values", "written"])
def test_kernel_rejects_state_tensors_outside_cpu_before_reading_them(which):
    values = torch.ones(1, 1, dtype=torch.float64)
    written = torch.ones(1, dtype=torch.int64)
    if which == "values":
        values = torch.empty(1, 1, dtype=torch.float64, device="meta")
    else:
        written = torch.empty(1, dtype=torch.int64, device="meta")
    with pytest.raises(ValueError):
        KernelMemory(
            KernelMemoryConfig(0.2),
            unit_keys(1),
            values,
            written,
            writes=1,
            cohorts=1,
            cursor=(10, 5, 1),
        )


@pytest.mark.parametrize("cursor", [(10, 11, 1), (10, 5, 0)])
def test_kernel_rejects_a_cursor_that_could_not_come_from_mature_feedback(cursor):
    with pytest.raises(ValueError, match="cursor"):
        KernelMemory(
            KernelMemoryConfig(0.2),
            unit_keys(1),
            torch.ones(1, 1, dtype=torch.float64),
            torch.ones(1, dtype=torch.int64),
            writes=1,
            cohorts=1,
            cursor=cursor,
        )


def test_kernel_state_reconciles_episode_count_and_cohort_order():
    with pytest.raises(ValueError):
        KernelMemory(
            KernelMemoryConfig(0.2),
            unit_keys(1),
            torch.ones(1, 1, dtype=torch.float64),
            torch.ones(1, dtype=torch.int64),
            writes=2,
            cohorts=1,
            cursor=(10, 5, 2),
        )
    with pytest.raises(ValueError):
        KernelMemory(
            KernelMemoryConfig(0.2),
            unit_keys(2),
            torch.ones(2, 1, dtype=torch.float64),
            torch.tensor([2, 1]),
            writes=2,
            cohorts=2,
            cursor=(10, 5, 2),
        )


def test_kernel_counters_fit_the_integer_arithmetic_used_for_recency():
    with pytest.raises(ValueError):
        KernelMemory(
            KernelMemoryConfig(0.2, capacity=1),
            unit_keys(1),
            torch.ones(1, 1, dtype=torch.float64),
            torch.ones(1, dtype=torch.int64),
            writes=2**63,
            cohorts=2**63,
            cursor=(10, 5, 1),
        )


@pytest.mark.parametrize("tiny", [1e-200, 1e-300])
def test_dpfp_does_not_turn_a_finite_nonzero_projection_into_a_zero_key(tiny):
    memory = AreaMemory(
        AssociativeMemoryConfig("proximal"), AreaLayout(areas=1, permutation="identity")
    )
    keys = torch.zeros(1, 64, dtype=torch.float64)
    keys[0, :2] = torch.tensor([1.0, tiny], dtype=torch.float64)
    values = memory.features(keys)
    expected = torch.zeros(1, 1, 128, dtype=torch.float64)
    expected[0, 0, 1] = 1
    torch.testing.assert_close(values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tiny", [1e-160, 1e-200, 1e-300])
def test_dpfp_preserves_ratios_when_all_valid_products_are_tiny(tiny):
    memory = AreaMemory(
        AssociativeMemoryConfig("proximal"), AreaLayout(areas=1, permutation="identity")
    )
    keys = torch.zeros(1, 64, dtype=torch.float64)
    # La coordenada dominante está aislada. Solo sobreviven t² y 2t² en los rasgos.
    keys[0, 0] = 1.0
    keys[0, 3:6] = torch.tensor([tiny, tiny, 2 * tiny], dtype=torch.float64)
    expected = torch.zeros(1, 1, 128, dtype=torch.float64)
    expected[0, 0, 4:6] = torch.tensor([1.0, 2.0], dtype=torch.float64) / 5**0.5
    torch.testing.assert_close(memory.features(keys), expected, rtol=1e-15, atol=0)
