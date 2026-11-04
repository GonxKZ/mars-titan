"""Rechazo de estados incoherentes y contabilidad física de E1."""

from dataclasses import replace

import pytest
import torch

from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.multiscale_mixture import MultiscaleMemory, MultiscaleMixtureConfig
from mars_titan.memory.role_binding import RoleBindingConfig

BASE = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.5))
MIX = MultiscaleMixtureConfig()


@pytest.mark.parametrize(
    "changes",
    [
        {"clock": True},
        {"clock": 2**63},
        {"clock": 10},
        {"cohorts": True},
        {"cohorts": -1},
        {"experts": [None] * 3},
        {"log_weights": torch.tensor([0.0, float("nan"), 0.0], dtype=torch.float64)},
        {"log_weights": torch.ones(3, dtype=torch.float64)},
        {"log_weights": torch.tensor([0.0, -1.0, -1.0], dtype=torch.float64)},
    ],
)
def test_invalid_initial_state_is_rejected(changes):
    with pytest.raises(ValueError):
        MultiscaleMemory(BASE, MIX, **changes)


@pytest.mark.parametrize(
    "correction,mixture",
    [
        (None, MIX),
        (BASE, None),
        (replace(BASE, memory=replace(BASE.memory, forgetting=0.1)), MIX),
        (replace(BASE, multiscale=MIX), MIX),
    ],
)
def test_invalid_backend_is_rejected(correction, mixture):
    with pytest.raises(ValueError):
        MultiscaleMemory(correction, mixture)


@pytest.mark.parametrize("rows", [0, 8193])
def test_empty_or_oversized_read_is_rejected(rows):
    with pytest.raises(ValueError, match="cohorte"):
        MultiscaleMemory(BASE, MIX).read(torch.zeros(rows, 64, dtype=torch.float64), at=0)


def test_nonfinite_prediction_is_not_silently_repaired():
    from mars_titan.memory.associative_memory import AssociativeMemory

    experts = [
        AssociativeMemory(BASE.memory, torch.full((64, 1), 1e308, dtype=torch.float64))
        for _ in range(3)
    ]
    memory = MultiscaleMemory(BASE, MIX, experts=experts)
    keys = torch.full((1, 64), 0.125, dtype=torch.float64)
    with pytest.raises(ValueError, match="no finitos"):
        memory.read(keys, at=0)


@pytest.mark.parametrize("bridge,role", [(False, False), (True, False), (True, True)])
def test_state_bytes_count_expert_indices_and_shared_role_tables_once(bridge, role):
    correction = replace(
        BASE,
        areas=AreaLayout(areas=3),
        query_bridge=AreaQueryBridgeConfig() if bridge else None,
        role_binding=RoleBindingConfig(mode="projected_concat") if role else None,
    )
    memory = MultiscaleMemory(correction, MIX)

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.nbytes
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(item) for item in value)
        return 0

    # Las permutaciones se reconstruyen. Las tablas de roles comparten una sola instancia.
    derived_indices = 3 * 3 * 64 * 8
    role_bytes = correction.role_binding.numeric_state_bytes if role else 0
    assert (
        memory.numeric_state_bytes == tensor_bytes(memory.export()) + derived_indices + role_bytes
    )
