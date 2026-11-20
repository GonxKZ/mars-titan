"""Campaña DGX v2: enmienda de la selección, numérica declarada y variantes de una sola variable."""

import json
from pathlib import Path

import pytest

from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.determinism import TORCH_DETERMINISTIC_UNFILLED
from mars_titan.training.kernel_policy import FP32_IEEE_ATTENTION
from mars_titan.training.selection import SESSION_MAE, SESSION_RANK_IC

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/experiments/dgx-us-campaign-v2.json"
PREVIOUS = ROOT / "configs/experiments/dgx-us-campaign-v1.json"
# Cada variante y el control del que solo la separa el campo declarado a su derecha.
CONTROLS = dict(
    titans_mac_mae=("titans_mac", "stopping"),
    titans_mac_autograd=("titans_mac", "predictor.memory_inner_gradient"),
    titans_mac_cascade=("titans_mac_autograd", "predictor.memory_multiscale"),
    titans_mac_cascade_film=("titans_mac_cascade", "predictor.presence_modulation"),
    titans_mac_huber=("titans_mac_autograd", "predictor.memory_huber"),
    titans_mac_eta015=("titans_mac_autograd", "predictor.gate_bias"),
    titans_mac_stability=("titans_mac_eta015", "predictor.memory_stability"),
    titans_mac_film=("titans_mac", "predictor.presence_modulation"),
    titans_mac_wide=("titans_mac", "predictor.presence_modulation"),
    titans_mac_timescales=("titans_mac", "predictor.memory_time_scales"),
    titans_mac_diffattn=("titans_mac", "predictor.differential_attention"),
    titans_mac_l2=("titans_mac", "predictor.layers"),
    titans_mac_p16=("titans_mac", "predictor.persistent_tokens"),
    titans_mac_nols=("titans_mac", "predictor.learning_scale"),
    # La cabeza escalar se ajusta con su propia pérdida: los dos campos son una sola variable.
    titans_mac_scalar=("titans_mac_nols", ("predictor.head", "recipe.loss")),
    titans_mac_ic=("titans_mac_scalar", "predictor.session_ic_objective"),
    titans_mac_rank=("titans_mac_nols", "predictor.session_rank_inputs"),
    titans_mac_ic_rank=("titans_mac_ic", "predictor.session_rank_inputs"),
    titans_mac_aux=("titans_mac", "predictor.auxiliary_horizons"),
    titans_mac_ema=("titans_mac", "recipe.weight_average"),
    titans_mac_sf=("titans_mac", "recipe.schedule_free"),
    titans_mac_mdrop=("titans_mac", "predictor.modality_dropout"),
    titans_mac_fdrop=("titans_mac", "predictor.fusion_dropout"),
    titans_mac_wd=("titans_mac", "recipe.weight_decay"),
    **{"titans_mac_lr3e-4": ("titans_mac", "recipe.learning_rate")},
    **{"titans_mac_lr3e-3": ("titans_mac", "recipe.learning_rate")},
)


@pytest.fixture(scope="module")
def document():
    return campaign.load_declaration(DECLARATION)[0]


def specification(document, name):
    """Todo lo que declara un modelo: receta, cambios del predictor y regla de parada."""
    model = document["models"][name]
    recipe = json.loads((ROOT / document["recipes"][model["recipe"]]).read_text())
    predictor = {**recipe["predictor"], **model.get("predictor_changes", {})}
    flat = {f"predictor.{key}": value for key, value in predictor.items()}
    flat.update({f"recipe.{key}": value for key, value in recipe["recipe"].items()})
    flat.update({f"walk_forward.{key}": value for key, value in recipe["walk_forward"].items()})
    flat["stopping"] = campaign.job_rule(document, model)
    flat["variant"] = model["variant"]
    return flat


def test_the_campaign_keeps_every_previous_model_and_adds_one_tier_per_variant(document):
    previous = json.loads(PREVIOUS.read_text())
    assert document["final_test_access"] is False and document["seeds"] == [42, 43, 44]
    assert document["origins"] == previous["origins"] and document["cohort"] == dict(size=4198)
    for name, model in previous["models"].items():
        assert document["models"][name] == model
    added = set(document["models"]) - set(previous["models"])
    assert added == set(CONTROLS)
    assert document["tiers"][: len(previous["tiers"])] == previous["tiers"]
    for name in added:
        tier = "v_" + name.removeprefix("titans_mac_")
        assert document["models"][name]["tier"] == tier and tier in document["tiers"]
    assert len(campaign.plan(document, "mars_titan")) == 48 * 19 * 3


def test_every_runner_selects_by_the_session_rank_ic_and_the_control_by_the_protocol(document):
    for runner, epochs in (("fit", 16), ("reader", 12)):
        rule = document["stopping"][runner]
        assert rule["metric"] == SESSION_RANK_IC and rule["min_assets"] == 30
        assert (rule["patience"], rule["min_delta"], rule["max_epochs"]) == (3, 0.0, epochs)
    for name, model in document["models"].items():
        rule = campaign.job_rule(document, model)
        expected = SESSION_MAE if name == "titans_mac_mae" else SESSION_RANK_IC
        assert rule["metric"] == expected
    previous = json.loads(PREVIOUS.read_text())["stopping"]["fit"]
    assert document["models"]["titans_mac_mae"]["stopping"] == previous


def test_every_recipe_declares_the_ieee_attention_and_the_deterministic_algorithms(document):
    assert document["execution"]["threads"] == dict(intra=20, interop=20)
    assert document["execution"]["order"] == "seed_tier"
    assert document["budget"]["min_free_gib"] == 60
    # Cada trabajo dispone de 16 GiB de GPU: la ventana ya no los rebaja a los 6 del portátil.
    assert document["budget"]["cuda_gib"] == dict(mars_titan=16)
    assert document["budget"]["memory_gib"] == dict(mars_titan=24)
    assert document["execution"]["workers"]["mars_titan"] == 4
    for path in document["recipes"].values():
        recipe = json.loads((ROOT / path).read_text())["recipe"]
        assert recipe["precision"] == FP32_IEEE_ATTENTION
        assert recipe["determinism"] == TORCH_DETERMINISTIC_UNFILLED


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_each_variant_differs_from_its_control_in_exactly_one_declared_field(document, name):
    control, field = CONTROLS[name]
    variant, reference = specification(document, name), specification(document, control)
    different = {
        key for key in variant.keys() | reference.keys() if variant.get(key) != reference.get(key)
    }
    assert different == ({field} if isinstance(field, str) else set(field))


def test_the_memory_variants_the_fused_write_does_not_cover_use_autograd(document):
    for name in ("cascade", "cascade_film", "huber", "eta015", "stability", "autograd"):
        spec = specification(document, f"titans_mac_{name}")
        assert spec["predictor.memory_inner_gradient"] == "autograd_v1"
    for name in ("titans_mac", "titans_mac_film", "titans_mac_timescales", "titans_mac_l2"):
        assert specification(document, name)["predictor.memory_inner_gradient"] == "fused_v1"
