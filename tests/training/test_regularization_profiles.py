"""Perfiles de regularización opcionales, sin ajustar pesos ni abrir datos."""

import copy
from pathlib import Path

import pytest
import torch

from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.training.anchored_decay import recipe_factory
from mars_titan.training.financial_run import load_recipe
from mars_titan.training.titans_walk_forward import case_recipe

BASE = Path("configs/titans/chronological-training-historical-masked.json")
PROFILES = [
    (0.0, BASE),
    (0.001, BASE.with_name("chronological-training-weight-decay-1e-3.json")),
    (0.01, BASE.with_name("chronological-training-weight-decay-1e-2.json")),
]


@pytest.mark.parametrize("decay,path", PROFILES)
@pytest.mark.parametrize("case", ["lr1e-4", "lr1e-3"])
def test_profile_reaches_adamw_and_has_its_own_identity_without_updates(decay, path, case):
    _, document = load_recipe(path)
    recipe = case_recipe(document, case)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
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
def test_regularization_profiles_change_only_weight_decay(decay, path):
    _, baseline = load_recipe(BASE)
    _, profile = load_recipe(path)
    expected = copy.deepcopy(baseline)
    expected["recipe"]["weight_decay"] = decay
    for value in (expected, profile):
        value.pop("pending")
    assert profile == expected
    assert profile["recipe"]["selection"]["stopping"] == "fixed_budget"
    assert profile["recipe"]["epochs"] == 30
    assert profile["recipe"]["precision"] == "fp32_strict"
    assert profile["predictor"]["refinements"] == 1


def test_each_decay_and_learning_rate_pair_has_a_distinct_training_identity():
    identities = set()
    for _, path in PROFILES:
        _, document = load_recipe(path)
        for case in document["walk_forward"]["search_cases"]:
            recipe = case_recipe(document, case)
            identity = tuple(sorted((key, repr(value)) for key, value in recipe.identity().items()))
            identities.add(identity)
    assert len(identities) == 6


@pytest.mark.parametrize("probability", [0.0, 0.1, 0.2])
def test_reference_dropout_stays_in_fusion_and_is_disabled_for_evaluation(probability):
    # La construcción restaura el generador CPU y no cambia las semillas de CUDA.
    with torch.random.fork_rng(devices=[]):
        model = MultimodalReference(
            "gru",
            {name: 4 for name in ("prices", "news", "charts", "fundamentals", "macro")},
            context=4,
            hidden_size=32,
            layers=2,
            dropout=probability,
        )
    layers = [layer for layer in model.fusion.modules() if isinstance(layer, torch.nn.Dropout)]
    assert len(layers) == 2 and all(layer.p == probability for layer in layers)
    assert model.price_encoder.dropout == 0.0
    model.train()
    assert all(layer.training for layer in layers)
    model.eval()
    value = torch.ones(8, 32)
    rng = torch.get_rng_state().clone()
    assert all(not layer.training and torch.equal(layer(value), value) for layer in layers)
    assert torch.equal(torch.get_rng_state(), rng)
