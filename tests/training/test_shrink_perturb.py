"""Contracción y ruido de F3 con valores literales, sin optimizador."""

import pytest
import torch

from mars_titan.training.shrink_perturb import SHRINK_PERTURB, ShrinkPerturb, shrink_perturb


def parameters(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(3, 2, generator=generator, dtype=torch.float64),
        "b": torch.randn(2, generator=generator, dtype=torch.float64),
    }


def test_the_update_is_shrink_times_parent_plus_noise_times_fresh():
    parent, fresh = parameters(0), parameters(1)
    named = {name: value.clone() for name, value in parent.items()}
    shrink_perturb(named, fresh, ShrinkPerturb(shrink=0.6, noise=0.01, seed=1))
    for name, value in named.items():
        torch.testing.assert_close(
            value, 0.6 * parent[name] + 0.01 * fresh[name], rtol=1e-15, atol=0
        )


def test_the_control_values_leave_the_parent_bit_for_bit():
    parent = parameters(0)
    named = {name: value.clone() for name, value in parent.items()}
    shrink_perturb(named, parameters(1), ShrinkPerturb(shrink=1.0, noise=0.0))
    assert all(torch.equal(named[name], parent[name]) for name in parent)


@pytest.mark.parametrize(
    "fresh",
    [
        {"a": torch.zeros(3, 2, dtype=torch.float64)},
        {"a": torch.zeros(2, 3, dtype=torch.float64), "b": torch.zeros(2, dtype=torch.float64)},
        {"a": torch.zeros(3, 2), "b": torch.zeros(2)},
    ],
)
def test_missing_or_mismatched_fresh_parameters_are_rejected(fresh):
    named = parameters(0)
    with pytest.raises(ValueError):
        shrink_perturb(named, fresh, ShrinkPerturb())


@pytest.mark.parametrize(
    "options",
    [
        dict(shrink=0.0),
        dict(shrink=1.5),
        dict(noise=-0.1),
        dict(noise=True),
        dict(seed=-1),
        dict(seed=1.0),
    ],
)
def test_invalid_configurations_are_rejected(options):
    with pytest.raises(ValueError):
        ShrinkPerturb(**options)


def test_identity_names_the_noise_source():
    identity = ShrinkPerturb().identity()
    assert identity["name"] == SHRINK_PERTURB and identity["shrink"] == 0.6
    assert identity["noise_source"].startswith("same_parameter_of_a_fresh")
