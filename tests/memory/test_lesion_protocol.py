"""Máscaras predefinidas y lecturas diagnósticas sin cambiar B1/B2/B3/E1."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.multiscale_mixture import DAY_US, MultiscaleMixtureConfig
from mars_titan.memory.role_binding import RoleBindingConfig


def protocol(**changes):
    from mars_titan.memory.lesion_protocol import LesionProtocol

    return LesionProtocol(**changes)


def correction(*, bridge=False, roles=False, mixture=False, monolithic=False):
    return MatureCorrection(
        AssociativeMemoryConfig("proximal", rate=0.25),
        areas=AreaLayout(areas=1 if monolithic else 4, shifts=4 if monolithic else 1),
        query_bridge=AreaQueryBridgeConfig(steps=2) if bridge else None,
        role_binding=RoleBindingConfig() if roles else None,
        multiscale=MultiscaleMixtureConfig() if mixture else None,
    )


def written(config):
    raw = np.random.Generator(np.random.PCG64(73)).normal(size=(3, 64)).astype(np.float32)
    keys = config.keys(raw, torch.tensor([0, 1, 2]) if config.routing else None)
    feedback = config.feedback(
        ids=[1, 2, 3],
        decision_at=[0] * 3,
        available_at=[1] * 3,
        keys=keys,
        values=[0.02, -0.01, 0.03],
    )
    memory = config.empty()
    extra = (
        dict(issued_predictions=torch.zeros(3, 3, dtype=torch.float64)) if config.multiscale else {}
    )
    return memory.write(feedback, cutoff=1, **extra), keys


def test_default_protocol_has_twenty_cases_and_nested_masks_with_private_rng():
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    plan = protocol().bind(correction())
    manifest = plan.identity()
    assert len(plan.names) == 20 and plan.names[:2] == ("intact", "zero")
    assert len(set(plan.names)) == 20
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert all(
        np.array_equal(a, b) for a, b in zip(numpy_before, np.random.get_state(), strict=True)
    )
    assert manifest == protocol().bind(correction()).identity()
    masks = plan.masks()
    for axis in ("area", "coordinate"):
        suffix = "_active_mean" if axis == "area" else ""
        indices = [
            plan.names.index(f"{axis}_s0_{part}{suffix}") for part in ("1of4", "1of2", "3of4")
        ]
        for left, right in zip(indices[:-1], indices[1:], strict=True):
            earlier, later = masks[left][axis], masks[right][axis]
            assert torch.all(later <= earlier)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"bridge": True},
        {"bridge": True, "roles": True},
        {"bridge": True, "roles": True, "mixture": True},
        {"monolithic": True},
    ],
)
def test_readings_are_finite_and_leave_the_original_memory_unchanged(options):
    config = correction(**options)
    state, keys = written(config)
    plan = protocol(seeds=(0,), fractions=((1, 2),)).bind(config)
    before = state.export()
    at = DAY_US + 1
    baseline = state.read(keys, **(dict(at=at) if config.multiscale else {}))
    result = plan.read(state, keys, at=at, baseline=baseline)
    assert result.shape == (3, 5) and torch.isfinite(result).all()
    assert torch.equal(result[:, :1], baseline)
    assert torch.count_nonzero(result[:, 1]) == 0
    assert state.export()["matrix_sha256"] == before["matrix_sha256"]
    if config.multiscale:
        assert state.export()["state_sha256"] == before["state_sha256"]
    assert state.writes == 3 and state.cursor == (1, 0, 3)


def test_returned_masks_do_not_modify_the_plan():
    plan = protocol().bind(correction())
    before = plan.identity()
    masks = plan.masks()
    masks[2]["area"].zero_()
    assert plan.identity() == before
    assert plan.masks()[2]["area"].any()


@pytest.mark.parametrize(
    "changes",
    [
        dict(seeds=()),
        dict(seeds=(0, 0)),
        dict(seeds=(True,)),
        dict(fractions=((1, 0),)),
        dict(fractions=((1, 1),)),
        dict(fractions=((1, 2), (2, 4))),
        dict(fractions=((3, 4), (1, 4))),
        dict(seeds=(0, 1, 2, 3), fractions=((1, 4), (1, 2), (3, 4))),
    ],
)
def test_invalid_or_unbounded_protocols_are_rejected(changes):
    with pytest.raises(ValueError):
        protocol(**changes)


def test_protocol_requires_areas_and_the_same_configuration_for_reading():
    with pytest.raises(ValueError):
        protocol().bind(MatureCorrection(AssociativeMemoryConfig("proximal")))
    config = correction()
    state, keys = written(config)
    other = replace(config, memory=replace(config.memory, rate=0.5))
    with pytest.raises(ValueError):
        protocol().bind(other).read(state, keys, at=2)
