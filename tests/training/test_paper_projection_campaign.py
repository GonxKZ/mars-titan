"""Campañas adicionales con layout común y proyecciones identificadas."""

import json
from pathlib import Path

import pytest

from mars_titan.data.storage import sha256
from mars_titan.models.titans.config import PAPER_PROJECTIONS
from mars_titan.models.titans.financial import VARIANTS
from mars_titan.training import campaign_plan as plan
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.cm_v1_factorial import load_declaration
from tests.models.titans.test_financial_adapter import raw_batch, specification

ROOT = Path("configs")
HISTORICAL = ROOT / "baselines/historical-masked-campaign-a-v2.json"
OLD_RECIPE = ROOT / "titans/chronological-training-historical-masked.json"
OLD_CM = ROOT / "titans/cm-v1-factorial.json"
RECIPES = {
    "sequence": ROOT / "titans/chronological-training-historical-sequence-v1.json",
    "paper": ROOT / "titans/chronological-training-historical-paper-v1.json",
}
CAMPAIGNS = {key: ROOT / f"baselines/historical-masked-campaign-a-{key}-v1.json" for key in RECIPES}
FACTORIALS = {key: ROOT / f"titans/cm-v1-{key}-factorial.json" for key in RECIPES}
HISTORICAL_SHA = {
    OLD_RECIPE: "4a56ec8d5c42c43885fbf4c4f4270750b2013bc784a4f98bb5664f9a416c76bf",
    HISTORICAL: "cb8ce969532b46ac9beb0bd466a6283e36b9dadc9199e4cd1de5e2d6da3e0873",
    OLD_CM: "796d9aaee3c2dd5421dc0b9cdde87e8089cd86312d837b412b02cf83ea6e821e",
}


def read(path):
    return json.loads(path.read_text())


def test_historical_declarations_keep_their_exact_bytes():
    assert {path: sha256(path) for path in HISTORICAL_SHA} == HISTORICAL_SHA


def test_new_recipes_keep_the_historical_budget_and_isolate_projections():
    old, sequence, paper = read(OLD_RECIPE), read(RECIPES["sequence"]), read(RECIPES["paper"])
    assert (
        "memory_projections" not in old["predictor"] and "attention_layout" not in old["predictor"]
    )
    assert sequence["predictor"] == old["predictor"] | dict(
        attention_layout="sequence_contiguous_v1"
    )
    assert paper["predictor"] == sequence["predictor"] | dict(memory_projections=PAPER_PROJECTIONS)
    for recipe in (sequence, paper):
        assert {k: v for k, v in recipe.items() if k not in ["predictor", "pending"]} == {
            k: v for k, v in old.items() if k not in ["predictor", "pending"]
        }
        assert recipe["recipe"]["epochs"] == 30
        assert recipe["recipe"]["accumulation_rows"] == 1024
        assert recipe["status"] == "declared_not_executed"


@pytest.mark.parametrize("name", ["sequence", "paper"])
def test_campaign_and_factorial_share_the_new_recipe_without_changing_other_models(name):
    old, new = read(HISTORICAL), read(CAMPAIGNS[name])
    assert new["name"] != old["name"]
    assert (CAMPAIGNS[name].parent / new["titans_mac"]["recipe"]).resolve() == RECIPES[
        name
    ].resolve()
    assert (CAMPAIGNS[name].parent / new["cm_v1"]["declaration"]).resolve() == FACTORIALS[
        name
    ].resolve()
    new["name"] = old["name"]
    new["titans_mac"]["recipe"] = old["titans_mac"]["recipe"]
    new["cm_v1"]["declaration"] = old["cm_v1"]["declaration"]
    assert new == old
    declaration = load_declaration(FACTORIALS[name])
    assert declaration["recipes"]["core_recipe"] == str(RECIPES[name].resolve())
    assert declaration["sha256"] != sha256(OLD_CM)
    current = read(FACTORIALS[name])
    baseline = read(OLD_CM)
    current["base"]["core_recipe"] = baseline["base"]["core_recipe"]
    current["pending"] = baseline["pending"]
    assert current == baseline


def test_alternative_plans_keep_jobs_and_temporal_contracts_with_distinct_sources():
    original = {job["id"]: job for job in plan.plan_campaign(plan.load_campaign(HISTORICAL))}
    for name in RECIPES:
        campaign = plan.load_campaign(CAMPAIGNS[name])
        jobs = {job["id"]: job for job in plan.plan_campaign(campaign)}
        assert jobs.keys() == original.keys()
        changed = [job for key, job in jobs.items() if job != original[key]]
        assert changed
        assert all(job["family"] in {plan.TITANS, plan.CM} for job in changed)
        for key, job in jobs.items():
            assert job["depends"] == original[key]["depends"]
            assert job["scope"] == original[key]["scope"]
            assert job["window"] == original[key]["window"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_each_recipe_constructs_the_four_controls_with_a_common_identity(device):
    for name, path in RECIPES.items():
        for variant in VARIANTS:
            model, _ = wf._predictor(read(path), specification(), variant, 42, device)
            assert model.config.attention_layout == "sequence_contiguous_v1"
            expected = PAPER_PROJECTIONS if name == "paper" else "linear_v1"
            assert model.config.memory_projections == expected
            assert model.config.quantile_routing is None


@pytest.mark.parametrize("name", ["sequence", "paper"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_new_online_recipes_support_finite_gradients_without_changing_shared_weights(name, device):
    import torch

    from mars_titan.models.titans.financial_inputs import DecisionBatch

    with wf.unfused_attention():
        model, _ = wf._predictor(read(RECIPES[name]), specification(), "mac_online", 42, device)
        before = [p.detach().clone() for p in model.parameters()]
        batch = DecisionBatch.from_corpus(raw_batch(), model.config.inputs, device=device)
        output = model.prepare(
            batch, model.initial_state(batch.flow_ids, differentiable=True), differentiable=True
        )
        gradients = torch.autograd.grad(
            output.quantiles.square().mean(), tuple(model.parameters()), allow_unused=True
        )
        assert any(g is not None and g.abs().max().item() > 0 for g in gradients)
        assert all(g is None or torch.isfinite(g).all().item() for g in gradients)
        assert all(
            torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
        )


@pytest.mark.parametrize("name", ["sequence", "paper"])
def test_a_historical_checkpoint_is_not_accepted_as_a_new_reference(name):
    old, _ = wf._predictor(read(OLD_RECIPE), specification(), "mac_online", 42, "cpu")
    new, _ = wf._predictor(read(RECIPES[name]), specification(), "mac_online", 42, "cpu")
    assert old.config.identity() != new.config.identity()
    with pytest.raises((ValueError, RuntimeError)):
        new.load_state_dict(old.state_dict())
