"""Receta de Titans-MAC para la campaña con máscaras: identidad, emparejamiento y casos.

La receta de campaña activa la memoria con residual y LayerNorm junto a `gate_bias` y
declara dos casos de búsqueda del optimizador. Las recetas v1 no cambian. Estas pruebas
construyen predictores y recetas, sin datos ni pasos de optimizador.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from mars_titan.evaluation.splits import stopping_rule
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import VARIANTS, FinancialConfig
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.financial_run import load_recipe
from tests.models.titans.test_financial_adapter import specification
from tests.training.test_financial_run_accumulation import IDENTITY_RECIPES

CAMPAIGN = Path("configs/titans/chronological-training-historical-masked.json")
V1_SCALAR = Path("configs/titans/chronological-training.json")
V1_QUANTILE = Path("configs/titans/chronological-training-quantile.json")
# Huellas de las recetas v1 en develop fec4591c. La receta de campaña es un archivo aparte.
V1_FILES = {
    V1_SCALAR: "b030fd3c6149561d1a722502181ee7b380a1a9278d8ee9a6df563fd04bdd9802",
    V1_QUANTILE: "e72fd925b86914785646b9edaa9898322efb6fd0b0ff8fb10e87e52b1bf03ed8",
}
PROTOCOLS = Path("configs/evaluation")


def document():
    return json.loads(CAMPAIGN.read_text())


def digest(recipe):
    return hashlib.sha256(canonical(recipe.identity()).encode()).hexdigest()


def options(value):
    return {key: item for key, item in value["predictor"].items() if key != "dtype"}


def test_v1_recipes_keep_their_files_and_identities():
    for path, expected in V1_FILES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        assert digest(load_recipe(path)[0]) == IDENTITY_RECIPES[str(path)]


def test_campaign_recipe_with_everything_disabled_gives_the_v1_identities():
    campaign, (v1_recipe, v1) = document(), load_recipe(V1_QUANTILE)
    # El caso con la tasa v1 reproduce la identidad de la receta v1, sin acumulación.
    assert wf.case_recipe(campaign, "lr1e-3").identity() == v1_recipe.identity()
    assert digest(wf.case_recipe(campaign, "lr1e-3")) == IDENTITY_RECIPES[str(V1_QUANTILE)]
    assert digest(wf.case_recipe(campaign, "lr1e-4")) != IDENTITY_RECIPES[str(V1_QUANTILE)]
    disabled = options(campaign) | dict(memory_residual_layer_norm=False)
    assert options(v1) == {k: v for k, v in disabled.items() if k != "memory_residual_layer_norm"}
    for variant in VARIANTS:
        old = FinancialConfig(specification(), variant=variant, seed=42, **options(v1))
        same = FinancialConfig(specification(), variant=variant, seed=42, **disabled)
        new = FinancialConfig(specification(), variant=variant, seed=42, **options(campaign))
        assert same.identity() == old.identity()
        assert new.identity() == old.identity() | dict(memory_residual_layer_norm=True)


def test_requests_only_name_the_search_case_when_the_recipe_declares_cases(tmp_path):
    view = tmp_path / "manifest.json"
    view.write_text("{}")
    arguments = (view, "0" * 64, "fold-000", V1_QUANTILE, "mac_online", 42, "cpu")
    request = wf._request(*arguments)
    assert set(request) == {
        "view_sha256",
        "protocol_sha256",
        "window",
        "recipe_sha256",
        "variant",
        "seed",
        "device",
        "code",
    }
    assert wf._request(*arguments, "lr1e-4") == request | dict(search_case="lr1e-4")


def test_four_controls_pair_from_mac_online_with_the_residual_memory():
    campaign = document()
    source, receipt = wf._predictor(campaign, specification(), "mac_online", 42, "cpu")
    assert receipt is None
    assert source.mac.memory.config.residual_layer_norm is True
    reference = dict(source.named_parameters())
    for variant in ("transformer_direct", "mac_disabled", "mac_frozen"):
        target, receipt = wf._predictor(campaign, specification(), variant, 42, "cpu")
        identity = target.config.identity()
        assert identity["memory_residual_layer_norm"] is True
        assert identity["memory_gate_bias"] == campaign["predictor"]["gate_bias"]
        assert receipt["initialized_only"] == [] and receipt["runtime_state_transferred"] is False
        copied = dict(target.named_parameters())
        assert set(receipt["copied_parameters"]) == set(copied)
        for name, value in copied.items():
            torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
        if target.mac is not None:
            assert target.mac.memory.config.residual_layer_norm is True


@pytest.mark.parametrize(
    "protocol",
    [
        "historical-masked-us-walk-forward-v2.json",
        "historical-masked-cn-walk-forward-v2.json",
        "historical-masked-joint-us-walk-forward-v2.json",
    ],
)
def test_each_search_case_applies_the_protocol_rule_and_changes_only_its_rate(protocol):
    rule = stopping_rule(json.loads((PROTOCOLS / protocol).read_text()))
    recipes = {}
    for name, rate in (("lr1e-4", 1e-4), ("lr1e-3", 1e-3)):
        recipe, value, walk = wf._recipe(CAMPAIGN, rule, name)
        assert recipe.learning_rate == rate and recipe.max_grad_norm == 1.0
        assert recipe.epochs == rule["max_epochs"] == 30
        assert recipe.accumulation_rows is None
        assert walk["warmup_months"] == 12 and value["status"] == "declared_not_executed"
        recipes[name] = recipe.identity()
    assert {k for k in recipes["lr1e-4"] if recipes["lr1e-4"][k] != recipes["lr1e-3"][k]} == {
        "learning_rate"
    }
    with pytest.raises(ValueError, match="Elige uno"):
        wf._recipe(CAMPAIGN, rule)
    with pytest.raises(ValueError, match="Elige uno"):
        wf._recipe(CAMPAIGN, rule, "lr3e-4")
    with pytest.raises(ValueError, match="no declara casos"):
        wf._recipe(V1_QUANTILE, rule, "lr1e-4")


@pytest.mark.parametrize(
    "cases",
    [
        {},
        {f"lr{i}": dict(learning_rate=rate) for i, rate in enumerate((1e-4, 3e-4, 1e-3, 3e-3))},
        {"A": dict(learning_rate=1e-4)},
        {"lr": dict(dropout=0.1)},
        {"lr": dict(learning_rate=1e-4), "clip": dict(max_grad_norm=0.5)},
        {"lr": dict(learning_rate=1e-4), "same": dict(learning_rate=1e-4)},
        {"lr": dict(learning_rate=0.0)},
        {"lr": {}},
        {"lr": 1e-4},
        ["lr1e-4"],
    ],
    ids=[
        "empty",
        "four",
        "name",
        "other_hyperparameter",
        "mixed_keys",
        "repeated",
        "invalid_rate",
        "empty_case",
        "not_a_case",
        "not_a_mapping",
    ],
)
def test_search_cases_must_be_few_distinct_optimizer_changes(cases):
    value = document()
    value["walk_forward"]["search_cases"] = cases
    with pytest.raises(ValueError):
        wf.walk_forward_options(value)


def test_searched_hyperparameters_cannot_keep_a_base_value():
    value = document()
    value["recipe"]["learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="ausentes de la receta base"):
        wf.walk_forward_options(value)
    value = document()
    value["walk_forward"]["search_cases"] = {
        "a": dict(learning_rate=1e-4, max_grad_norm=1.0),
        "b": dict(learning_rate=1e-3, max_grad_norm=1.0),
    }
    with pytest.raises(ValueError, match="ausentes de la receta base"):
        wf.walk_forward_options(value)
    value["recipe"].pop("max_grad_norm")
    assert wf.case_recipe(value, "b").max_grad_norm == 1.0


def test_cases_must_change_the_same_hyperparameters():
    value = document()
    value["recipe"].pop("max_grad_norm")
    value["walk_forward"]["search_cases"] = {
        "lr": dict(learning_rate=1e-4),
        "clip": dict(max_grad_norm=0.5),
    }
    with pytest.raises(ValueError, match="mismos hiperparámetros"):
        wf.walk_forward_options(value)
