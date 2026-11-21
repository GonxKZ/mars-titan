"""Campaña DGX v3: niveles para dejar fuera modelos redundantes y modelos de la segunda ronda."""

import hashlib
import json
from pathlib import Path

import pytest

from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.comparison_cohorts import FULL_VIEW_RULE

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/experiments/dgx-us-campaign-v3.json"
PREVIOUS = ROOT / "configs/experiments/dgx-us-campaign-v2.json"
JOINT = ROOT / "configs/experiments/dgx-us-cn-campaign-v2.json"
PREVIOUS_JOINT = ROOT / "configs/experiments/dgx-us-cn-campaign-v1.json"
MOVED = dict(
    mars_titan_m1="ladder_m1",
    mars_titan_m2="ladder",
    mars_titan_m3_k4="cognitive_controls",
    mars_titan_cognitive_constant_gate="cognitive_controls",
    area_network="networks_area",
    area_network_asset_groups="networks_area",
    area_network_hidden_only="networks_area",
    area_network_membership_only="networks_area",
    asset_groups="networks_area",
    asset_groups_shuffled="networks_area",
)
# Cada modelo nuevo y el control del que solo lo separan los campos declarados a su derecha.
CONTROLS = dict(
    cross_asset_attention_disabled=("titans_mac_disabled", "predictor.cross_asset_attention"),
    cross_asset_attention_frozen=("titans_mac_frozen", "predictor.cross_asset_attention"),
    titans_mac_disabled_nols=("titans_mac_disabled", "predictor.learning_scale"),
    titans_mac_disabled_scalar=("titans_mac_disabled_nols", ("predictor.head", "recipe.loss")),
    titans_mac_disabled_ic=("titans_mac_disabled_scalar", "predictor.session_ic_objective"),
    titans_mac_disabled_rank=("titans_mac_disabled_nols", "predictor.session_rank_inputs"),
    titans_mac_disabled_macro=("titans_mac_disabled", "predictor.presence_modulation"),
    titans_mac_disabled_ema=("titans_mac_disabled", "recipe.weight_average"),
    titans_mac_disabled_autograd=("titans_mac_disabled", "predictor.memory_inner_gradient"),
    titans_mac_disabled_h128=("titans_mac_disabled_autograd", "predictor.hidden_size"),
    titans_mac_disabled_h32=("titans_mac_disabled_autograd", "predictor.hidden_size"),
    titans_mac_w2009_o2016=("titans_mac", "walk_forward.fit_months"),
    titans_mac_frozen_w2009_o2016=("titans_mac_frozen", "walk_forward.fit_months"),
    titans_mac_disabled_w2009_o2016=("titans_mac_disabled", "walk_forward.fit_months"),
    titans_mac_w2009_o2023=("titans_mac", "walk_forward.fit_months"),
    titans_mac_frozen_w2009_o2023=("titans_mac_frozen", "walk_forward.fit_months"),
    titans_mac_disabled_w2009_o2023=("titans_mac_disabled", "walk_forward.fit_months"),
)
# Los mismos cambios sobre la memoria en línea ya están en la criba: solo cambia la variante.
ONLINE = dict(
    cross_asset_attention_disabled="cross_asset_attention",
    cross_asset_attention_frozen="cross_asset_attention",
    titans_mac_disabled_nols="titans_mac_nols",
    titans_mac_disabled_scalar="titans_mac_scalar",
    titans_mac_disabled_ic="titans_mac_ic",
    titans_mac_disabled_rank="titans_mac_rank",
    titans_mac_disabled_macro="titans_mac_macro",
    titans_mac_disabled_ema="titans_mac_ema",
    titans_mac_disabled_autograd="titans_mac_autograd",
)


@pytest.fixture(scope="module")
def document():
    return campaign.load_declaration(DECLARATION)[0]


def specification(document, name):
    model = document["models"][name]
    recipe = json.loads((ROOT / document["recipes"][model["recipe"]]).read_text())
    predictor = {**recipe["predictor"], **model.get("predictor_changes", {})}
    flat = {f"predictor.{key}": value for key, value in predictor.items()}
    flat.update({f"recipe.{key}": value for key, value in recipe["recipe"].items()})
    flat.update({f"walk_forward.{key}": value for key, value in recipe["walk_forward"].items()})
    flat["stopping"] = campaign.job_rule(document, model)
    flat["variant"] = model["variant"]
    return flat


def difference(document, left, right):
    one, other = specification(document, left), specification(document, right)
    return {key for key in one.keys() | other.keys() if one.get(key) != other.get(key)}


def test_the_campaign_can_inherit_the_previous_one(document):
    previous = json.loads(PREVIOUS.read_text())
    # La herencia exige el mismo contrato y solo traslada los modelos que no cambian.
    for field in ("origins", "seeds", "cohort", "stopping", "analysis"):
        assert document.get(field) == previous.get(field)
    assert (
        document["execution"] == previous["execution"] and document["budget"] == previous["budget"]
    )
    for name, model in previous["models"].items():
        if name in MOVED:
            assert document["models"][name] == dict(model, tier=MOVED[name])
        else:
            assert document["models"][name] == model
    for name, path in previous["recipes"].items():
        assert document["recipes"][name] == path
    digest = hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()
    assert (
        document["source_declaration_sha256"] == digest and document["final_test_access"] is False
    )


def test_the_added_models_are_the_second_round_and_the_plan_counts_them(document):
    previous = json.loads(PREVIOUS.read_text())
    added = set(document["models"]) - set(previous["models"])
    assert added == set(CONTROLS) | {"mars_titan_m0_xattn"}
    tiers = document["tiers"]
    assert tiers[:4] == ["core", "cognitive", "controls", "networks"]
    assert {document["models"][name]["tier"] for name in added} == {
        "r2_xattn",
        "r2_reader_xattn",
        "r2_disabled",
        "r2_capacity",
        "ventana_2016",
        "ventana_2023",
    }
    # Los niveles principales quedan con lo que no es redundante en 2013.
    by_tier = {
        tier: sorted(n for n, m in document["models"].items() if m["tier"] == tier)
        for tier in tiers
    }
    assert by_tier["core"] == ["mars_titan_m0", "mars_titan_m3", "titans_mac"]
    assert by_tier["cognitive"] == ["mars_titan_cognitive"]
    assert by_tier["networks"] == ["cross_asset_attention", "cross_asset_attention_self"]
    assert by_tier["ladder_m1"] == ["mars_titan_m1"]
    assert len(document["models"]) == 71
    # La herencia admite como mucho 4.096 trabajos en la declaración de origen.
    assert len(campaign.plan(document, "mars_titan")) == 71 * 19 * 3 <= 4096


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_each_added_fit_differs_from_its_control_in_exactly_one_declared_field(document, name):
    control, field = CONTROLS[name]
    expected = {field} if isinstance(field, str) else set(field)
    assert difference(document, name, control) == expected


@pytest.mark.parametrize("name", sorted(ONLINE))
def test_each_crossing_is_a_screened_change_with_another_memory_variant(document, name):
    assert difference(document, name, ONLINE[name]) == {"variant"}
    assert document["models"][name]["variant"] in ("mac_disabled", "mac_frozen")


def test_the_bounded_fits_start_in_april_2009_and_only_differ_in_their_window(document):
    from mars_titan.training.walk_forward_phases import months_before

    # El tramo de ajuste de cada origen termina el 1 de abril del año anterior.
    for year in (2016, 2023):
        months = specification(document, f"titans_mac_w2009_o{year}")["walk_forward.fit_months"]
        assert months_before(f"{year - 1}-04-01", months) == "2009-04-01"
        names = [n for n, m in document["models"].items() if m["tier"] == f"ventana_{year}"]
        assert sorted(document["models"][name]["variant"] for name in names) == [
            "mac_disabled",
            "mac_frozen",
            "mac_online",
        ]
        assert len({document["models"][name]["recipe"] for name in names}) == 1


def test_the_reader_over_cross_asset_attention_is_the_bankless_reader_with_another_parent(document):
    models = document["models"]
    reader, reference = models["mars_titan_m0_xattn"], models["mars_titan_m0"]
    different = {
        key for key in reader.keys() | reference.keys() if reader.get(key) != reference.get(key)
    }
    assert different == {"parent", "tier"} and reader["parent"] == "cross_asset_attention"
    assert models[reader["parent"]]["variant"] == "mac_online"


def test_the_joint_campaign_mirrors_the_us_one_and_can_inherit_its_previous_version():
    joint = campaign.load_declaration(JOINT)[0]
    single = json.loads(DECLARATION.read_text())
    previous = json.loads(PREVIOUS_JOINT.read_text())
    assert joint["market"] == "US+CN" and joint["cohort"] == dict(
        size=4198 + 810, rule=FULL_VIEW_RULE
    )
    own = {"market", "cohort", "description", "source_declaration_sha256"}
    assert set(joint) == set(single)
    for key in set(single) - own:
        assert joint[key] == single[key], key
    assert (
        joint["source_declaration_sha256"] == hashlib.sha256(DECLARATION.read_bytes()).hexdigest()
    )
    for field in ("origins", "seeds", "cohort", "stopping", "analysis"):
        assert joint.get(field) == previous.get(field)
    for name, model in previous["models"].items():
        expected = dict(model, tier=MOVED[name]) if name in MOVED else model
        assert joint["models"][name] == expected
    pairs = [(job.key, job.runner, job.needs) for job in campaign.plan(joint, "mars_titan")]
    alone = [
        (j.key, j.runner, j.needs)
        for j in campaign.plan(campaign.load_declaration(DECLARATION)[0], "mars_titan")
    ]
    assert pairs == alone


def test_the_contrasts_give_every_model_a_control_that_ends_in_the_core(document):
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v3-contrasts.json").read_text()
    )
    models, controls = document["models"], declared["controls"]
    # Todos los modelos menos el núcleo tienen un control declarado, que es otro modelo.
    assert set(controls) == set(models) - {"titans_mac"}
    for name in controls:
        chain, current = [name], controls[name]
        while current != "titans_mac":
            assert current in controls and current not in chain, name
            chain.append(current)
            current = controls[current]
    # Los contrastes de un solo campo de las pruebas son los mismos que usa la criba.
    for name, (control, _) in CONTROLS.items():
        assert controls[name] == control
    assert set(declared["always_kept"]) <= set(models)
    assert all(models[name]["tier"].startswith("v_") for name in declared["always_kept"])


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("dgx-us-campaign-v4.json", "dgx-us-campaign-v3.json"),
        ("dgx-us-cn-campaign-v3.json", "dgx-us-cn-campaign-v2.json"),
    ],
)
def test_the_campaigns_with_technical_inputs_only_add_the_declared_extension(name, origin):
    path, source = ROOT / "configs/experiments" / name, ROOT / "configs/experiments" / origin
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    assert document["inputs"] == dict(technical="long_horizon_v1") and "inputs" not in previous
    own = {"inputs", "description", "source_declaration_sha256"}
    assert set(document) == set(previous) | {"inputs"}
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["source_declaration_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert document["final_test_access"] is False
    wrong = dict(json.loads(path.read_text()), inputs=dict(technical="otra"))
    with pytest.raises(ValueError, match="extensión de entradas"):
        campaign.checked_declaration(wrong, root=ROOT)
