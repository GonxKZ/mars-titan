"""Decaimiento puro de coeficientes, sin escribir etiquetas ni avanzar cursores."""

import pytest
import torch

from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.role_binding import RoleBindingConfig


@pytest.mark.parametrize(
    "areas,bridge,role",
    [
        (None, None, None),
        (AreaLayout(), None, None),
        (AreaLayout(), AreaQueryBridgeConfig(), None),
        (AreaLayout(), AreaQueryBridgeConfig(), RoleBindingConfig()),
    ],
)
def test_decay_preserves_metadata_and_scales_all_coefficients(areas, bridge, role):
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal", rate=0.2),
        areas=areas,
        query_bridge=bridge,
        role_binding=role,
    )
    keys = torch.randn(4, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    keys /= torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    feedback = correction.feedback(
        ids=[1, 2, 3, 4],
        decision_at=[1] * 4,
        available_at=[2] * 4,
        keys=keys,
        values=[0.1, 0.2, 0.3, 0.4],
    )
    memory = correction.empty().write(feedback, cutoff=2)
    before = memory.matrix.clone()
    decayed = memory.decay_coefficients(0.5)
    assert torch.equal(decayed.matrix, before * 0.5)
    assert torch.equal(memory.matrix, before)
    assert decayed.writes == memory.writes and decayed.cursor == memory.cursor
    assert memory.decay_coefficients(1.0) is memory
    assert torch.equal(correction.restore(decayed.export()).matrix, decayed.matrix)


@pytest.mark.parametrize("factor", [-1.0, 1.1, float("nan"), float("inf"), True])
def test_invalid_decay_is_rejected(factor):
    with pytest.raises(ValueError):
        MatureCorrection(AssociativeMemoryConfig("proximal")).empty().decay_coefficients(factor)
