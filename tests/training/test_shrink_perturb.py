"""Contracción y ruido de F3 con valores literales, sin optimizador."""

import numpy as np
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
    assert identity["fresh_seed"] == "seed_sequence_of_the_declared_seed_and_the_fit_seed"


def test_each_fit_seed_receives_its_own_reproducible_fresh_seed():
    config = ShrinkPerturb(seed=1)
    seeds = [config.fresh_seed(seed) for seed in (42, 43, 44, 45, 46)]
    assert len(set(seeds)) == 5 and seeds == [config.fresh_seed(s) for s in (42, 43, 44, 45, 46)]
    assert all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
    # Es la primera palabra de la SeedSequence de la semilla declarada y la del ajuste.
    expected = int(np.random.SeedSequence([1, 42]).generate_state(1)[0])
    assert seeds[0] == expected
    assert ShrinkPerturb(seed=2).fresh_seed(42) != expected


@pytest.mark.parametrize("fit_seed", [-1, 2**32, 1.0, True])
def test_an_invalid_fit_seed_is_rejected(fit_seed):
    with pytest.raises(ValueError, match="semilla del ajuste"):
        ShrinkPerturb().fresh_seed(fit_seed)
