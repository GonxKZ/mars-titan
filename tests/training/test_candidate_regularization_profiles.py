"""Regularización de la GRU con recetas reales y sin pasos de optimizador."""

import copy
import json
from pathlib import Path

import pytest
import torch

from mars_titan.training.anchored_decay import recipe_factory
from mars_titan.training.candidate_run import load_recipe

BASE = Path("configs/candidate/chronological-training.json")
PROFILES = [
    (0.0, BASE),
    (0.001, BASE.with_name("chronological-training-weight-decay-1e-3.json")),
    (0.01, BASE.with_name("chronological-training-weight-decay-1e-2.json")),
]
VARIANTS = ["m1_k1", "m0_k1", "m1_k2", "m1_k4", "m1_k1_l1_median"]
CASES = ["lr1e-4", "lr1e-3"]


@pytest.mark.parametrize("decay,path", PROFILES)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASES)
def test_declared_decay_reaches_adamw_without_changing_parameters(decay, path, variant, case):
    recipe, document = load_recipe(path, variant=variant, search_case=case)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64, device="cpu"))
    before = parameter.detach().clone()
    optimizer = recipe_factory(recipe)([parameter])
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["weight_decay"] == decay
    assert (
        optimizer.param_groups[0]["lr"]
        == document["walk_forward"]["search_cases"][case]["learning_rate"]
    )
    assert recipe.identity()["weight_decay"] == decay
    assert recipe.weight_decay_anchor is None
    assert not optimizer.state and torch.equal(parameter, before)


@pytest.mark.parametrize("decay,path", PROFILES[1:])
def test_profile_preserves_the_historical_protocol_except_decay_and_notes(decay, path):
    _, baseline = load_recipe(BASE, variant="m1_k1", search_case="lr1e-4")
    _, profile = load_recipe(path, variant="m1_k1", search_case="lr1e-4")
    expected = copy.deepcopy(baseline)
    expected["recipe"]["weight_decay"] = decay
    for value in (expected, profile):
        value.pop("pending")
    assert profile == expected
    assert profile["recipe"]["selection"]["stopping"] == "fixed_budget"
    assert profile["recipe"]["epochs"] == 30
    assert profile["recipe"]["precision"] == "fp32_strict"
    assert profile["principal"] == "m1_k1"


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_decay_and_rate_has_its_own_training_identity(variant):
    identities = set()
    for _, path in PROFILES:
        for case in CASES:
            recipe, _ = load_recipe(path, variant=variant, search_case=case)
            identities.add(json.dumps(recipe.identity(), sort_keys=True, allow_nan=False))
    assert len(identities) == 6
