"""Regularizar el lector sin cambiar al padre, el banco ni las actualizaciones."""

import copy
from pathlib import Path

import pytest
import torch

from mars_titan.training.anchored_decay import recipe_factory
from mars_titan.training.mars_titan_run import case_recipe, load_recipe

BASE = Path("configs/titans/episodic-readout-historical-masked.json")
PROFILES = [
    (0.0, BASE),
    (0.001, BASE.with_name("episodic-readout-weight-decay-1e-3.json")),
    (0.01, BASE.with_name("episodic-readout-weight-decay-1e-2.json")),
]


@pytest.mark.parametrize("decay,path", PROFILES)
@pytest.mark.parametrize("case", ["lr1e-4", "lr1e-3"])
def test_readout_profile_reaches_its_optimizer_without_updates(decay, path, case):
    document = load_recipe(path)
    recipe = case_recipe(document, case)
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = recipe_factory(recipe)([parameter])
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["weight_decay"] == decay
    assert optimizer.param_groups[0]["lr"] == recipe.learning_rate
    assert recipe.identity()["weight_decay"] == decay
    assert recipe.identity()["trained_parameters"] == "episodic_readout_only_parent_frozen"
    assert recipe.weight_decay_anchor is None
    assert torch.equal(parameter, torch.ones_like(parameter)) and not optimizer.state


@pytest.mark.parametrize("decay,path", PROFILES[1:])
def test_readout_regularization_keeps_retention_budget_and_selection(decay, path):
    expected = copy.deepcopy(load_recipe(BASE))
    actual = load_recipe(path)
    expected["recipe"]["weight_decay"] = decay
    for document in (expected, actual):
        document.pop("pending")
    assert actual == expected
    assert actual["recipe"]["selection"]["stopping"] == "fixed_budget"
    assert actual["recipe"]["bank_capacity"] == 1024


def test_readout_decay_and_learning_rate_pairs_have_distinct_identities():
    identities = set()
    for _, path in PROFILES:
        document = load_recipe(path)
        for case in document["walk_forward"]["search_cases"]:
            identity = case_recipe(document, case).identity()
            identities.add(tuple(sorted((key, repr(value)) for key, value in identity.items())))
    assert len(identities) == 6
