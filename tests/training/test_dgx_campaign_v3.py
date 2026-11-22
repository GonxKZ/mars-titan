"""Campaña DGX v3: niveles para dejar fuera modelos redundantes y modelos de la segunda ronda."""

import hashlib
import json
from pathlib import Path

import pytest

from mars_titan.data import technical_features
from mars_titan.models.titans.cross_asset_attention import parse_cross_asset_attention
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training import mars_titan_run as readout
from mars_titan.training.comparison_cohorts import FULL_VIEW_RULE
from mars_titan.training.financial_run import ChronologicalRecipe

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
    assert document["inputs"] == dict(technical="long_horizon_v2") and "inputs" not in previous
    own = {"inputs", "description", "source_declaration_sha256", "cohort"}
    assert set(document) == set(previous) | {"inputs"}
    # La extensión la declara la cohorte, que es la vista completa de cada origen.
    assert document["cohort"] == dict(size=previous["cohort"]["size"], rule=FULL_VIEW_RULE)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["source_declaration_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert document["final_test_access"] is False
    wrong = dict(json.loads(path.read_text()), inputs=dict(technical="otra"))
    with pytest.raises(ValueError, match="extensión de entradas"):
        campaign.checked_declaration(wrong, root=ROOT)


BRANCH_CONTROLS = dict(
    titans_mac_branch="titans_mac",
    titans_mac_frozen_branch="titans_mac_frozen",
    titans_mac_disabled_branch="titans_mac_disabled",
    cross_asset_attention_branch="cross_asset_attention",
    cross_asset_attention_disabled_branch="cross_asset_attention_disabled",
)
BRANCH_READERS = dict(mars_titan_m0_branch="mars_titan_m0", mars_titan_m3_branch="mars_titan_m3")


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("dgx-us-campaign-v5.json", "dgx-us-campaign-v4.json"),
        ("dgx-us-cn-campaign-v4.json", "dgx-us-cn-campaign-v3.json"),
    ],
)
def test_the_campaigns_with_the_technical_branch_only_add_its_models(name, origin):
    from mars_titan.data.technical_features import VERSIONS

    path, source = ROOT / "configs/experiments" / name, ROOT / "configs/experiments" / origin
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["tiers"] == [*previous["tiers"], "r3_branch", "r3_branch_more"]
    added = set(document["models"]) - set(previous["models"])
    assert added == set(BRANCH_CONTROLS) | set(BRANCH_READERS)
    for model, definition in previous["models"].items():
        assert document["models"][model] == definition
    columns = len(VERSIONS[document["inputs"]["technical"]])
    # Cada ajuste con rama difiere de su control en ese único campo del predictor.
    for model, control in BRANCH_CONTROLS.items():
        assert difference(document, model, control) == {"predictor.technical_branch"}
        branch = specification(document, model)["predictor.technical_branch"]
        assert branch == dict(columns=columns, bins=16, embedding=8, hidden=128)
    # Y cada lector, en el núcleo con rama sobre el que se ajusta.
    for model, control in BRANCH_READERS.items():
        reader, reference = document["models"][model], document["models"][control]
        changed = {
            key for key in reader.keys() | reference.keys() if reader.get(key) != reference.get(key)
        }
        assert changed == {"parent", "tier"} and reader["parent"] == "titans_mac_branch"
    assert document["final_test_access"] is False
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v5-contrasts.json").read_text()
    )
    assert set(declared["controls"]) == set(document["models"]) - {"titans_mac"}
    for model, control in {**BRANCH_CONTROLS, **BRANCH_READERS}.items():
        assert declared["controls"][model] == control


STEP_CONTROLS = dict(
    titans_mac_disabled_t1=("titans_mac_disabled", 1),
    titans_mac_disabled_t2=("titans_mac_disabled", 2),
    titans_mac_disabled_t4=("titans_mac_disabled", 4),
    titans_mac_frozen_t2=("titans_mac_frozen", 2),
    titans_mac_t2=("titans_mac", 2),
    titans_mac_t4=("titans_mac", 4),
    titans_mac_disabled_branch_t2=("titans_mac_disabled_branch", 2),
    titans_mac_frozen_branch_t2=("titans_mac_frozen_branch", 2),
)


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("dgx-us-campaign-v6.json", "dgx-us-campaign-v5.json"),
        ("dgx-us-cn-campaign-v5.json", "dgx-us-cn-campaign-v4.json"),
    ],
)
def test_the_campaigns_with_more_updates_only_change_the_instants_of_each_update(name, origin):
    path, source = ROOT / "configs/experiments" / name, ROOT / "configs/experiments" / origin
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["tiers"] == [*previous["tiers"], "r3_steps", "r3_combo"]
    assert set(document["models"]) - set(previous["models"]) == set(STEP_CONTROLS)
    for model, definition in previous["models"].items():
        assert document["models"][model] == definition
    for recipe, file in previous["recipes"].items():
        assert document["recipes"][recipe] == file
    for model, (control, instants) in STEP_CONTROLS.items():
        assert difference(document, model, control) == {"recipe.truncation"}
        assert specification(document, model)["recipe.truncation"] == instants
        assert specification(document, control)["recipe.truncation"] == 8
        assert document["models"][model]["tier"] == (
            "r3_combo" if "branch" in model else "r3_steps"
        )
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v6-contrasts.json").read_text()
    )
    assert set(declared["controls"]) == set(document["models"]) - {"titans_mac"}
    for model, (control, _) in STEP_CONTROLS.items():
        assert declared["controls"][model] == control


def flat_specification(document, name):
    """Especificación de un modelo, también si es un lector (sin predictor propio en su receta)."""
    model = document["models"][name]
    recipe = json.loads((ROOT / document["recipes"][model["recipe"]]).read_text())
    flat = {f"recipe.{key}": value for key, value in recipe["recipe"].items()}
    flat.update({f"predictor.{k}": v for k, v in (recipe.get("predictor") or {}).items()})
    flat.update({f"predictor.{k}": v for k, v in (model.get("predictor_changes") or {}).items()})
    flat.update({key: value for key, value in model.items() if key not in ("recipe", "tier")})
    return flat


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("dgx-us-campaign-v7.json", "dgx-us-campaign-v6.json"),
        ("dgx-us-cn-campaign-v6.json", "dgx-us-cn-campaign-v5.json"),
    ],
)
def test_the_campaigns_on_the_faster_recipe_only_change_the_instants_of_each_update(name, origin):
    path, source = ROOT / "configs/experiments" / name, ROOT / "configs/experiments" / origin
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    kept = [tier for tier in previous["tiers"] if tier not in ("r3_steps", "r3_combo")]
    assert document["tiers"] == [*kept, "r3_t8", "r3_steps"]
    carried = {n for n, m in previous["models"].items() if m["tier"] in kept}
    added = set(document["models"]) - carried
    assert added == {
        "titans_mac_t8",
        "titans_mac_frozen_t8",
        "titans_mac_disabled_t8",
        "titans_mac_t1",
        "titans_mac_t4",
        "titans_mac_disabled_t1",
        "titans_mac_disabled_t4",
        "mars_titan_m0_u2",
    }
    # Cada modelo conserva su definición: solo cambia, en los ajustes, la cadencia de su receta.
    for model in sorted(carried):
        assert document["models"][model] == previous["models"][model], model
        now, before = flat_specification(document, model), flat_specification(previous, model)
        changed = {key for key in now.keys() | before.keys() if now.get(key) != before.get(key)}
        if document["models"][model]["runner"] == "fit":
            assert changed == {"recipe.truncation"}, model
            assert now["recipe.truncation"] == 2 and before["recipe.truncation"] == 8
        else:
            assert changed == set(), model
    # Los controles de la cadencia: la receta anterior y otros valores, con ese único cambio.
    for model in sorted(added - {"mars_titan_m0_u2"}):
        control, instants = model.rsplit("_t", 1)
        assert difference(document, model, control) == {"recipe.truncation"}
        assert specification(document, model)["recipe.truncation"] == int(instants)
        tier = "r3_t8" if instants == "8" else "r3_steps"
        assert document["models"][model]["tier"] == tier
    for model in ("titans_mac_t8", "titans_mac_frozen_t8", "titans_mac_disabled_t8"):
        before = flat_specification(previous, model.removesuffix("_t8"))
        assert flat_specification(document, model) == before
    reader, quick = (
        flat_specification(document, "mars_titan_m0"),
        flat_specification(document, "mars_titan_m0_u2"),
    )
    changed = {key for key in reader.keys() | quick.keys() if reader.get(key) != quick.get(key)}
    assert changed == {"recipe.update_instants"} and quick["recipe.update_instants"] == 2
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v7-contrasts.json").read_text()
    )
    assert set(declared["controls"]) == set(document["models"]) - {"titans_mac"}
    assert declared["controls"]["titans_mac_disabled_t8"] == "titans_mac_disabled"
    assert declared["controls"]["mars_titan_m0_u2"] == "mars_titan_m0"


def test_the_pilot_of_the_session_ic_term_changes_one_field_per_model():
    path = ROOT / "configs/experiments/dgx-us-pilot-v8.json"
    source = ROOT / "configs/experiments/dgx-us-campaign-v7.json"
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["tiers"] == ["core", "controls", "p8_a", "p8_b", "p8_c"]
    assert document["recipes"]["core"] == previous["recipes"]["core"]
    for model in ("titans_mac", "titans_mac_disabled", "titans_mac_frozen"):
        assert document["models"][model] == previous["models"][model]
    term, branch = "recipe.session_ic_term", "predictor.technical_branch"
    dense = {"columns": 46, "hidden": 128, "depth": 2}
    # Cada modelo y su control, con el único campo que los separa.
    contrasts = {
        ("titans_mac_disabled_ic3", "titans_mac_disabled"): {term},
        ("titans_mac_disabled_dense", "titans_mac_disabled"): {branch},
        ("titans_mac_disabled_dense_ic3", "titans_mac_disabled_ic3"): {branch},
        ("titans_mac_disabled_dense_ic3", "titans_mac_disabled_dense"): {term},
        ("titans_mac_disabled_branch_ic3", "titans_mac_disabled_ic3"): {branch},
        ("titans_mac_disabled_branch_ic3", "titans_mac_disabled_dense_ic3"): {branch},
        ("titans_mac_frozen_dense_ic3", "titans_mac_disabled_dense_ic3"): {"variant"},
        ("titans_mac_dense_ic3", "titans_mac_disabled_dense_ic3"): {"variant"},
        ("titans_mac_disabled_dense_ic1", "titans_mac_disabled_dense_ic3"): {term},
        ("titans_mac_disabled_dense_ic10", "titans_mac_disabled_dense_ic3"): {term},
        ("titans_mac_disabled_dense_ic3_t1", "titans_mac_disabled_dense_ic3"): {
            "recipe.truncation"
        },
    }
    for (model, control), fields in contrasts.items():
        assert difference(document, model, control) == fields, (model, control)
    assert {model for model, _ in contrasts} == set(document["models"]) - {
        "titans_mac",
        "titans_mac_disabled",
        "titans_mac_frozen",
    }
    for model, weight in (("ic1", 1.0), ("ic3", 3.0), ("ic10", 10.0)):
        declared = specification(document, f"titans_mac_disabled_dense_{model}")
        assert declared[term] == {"weight": weight, "minimum_rows": 8}
        assert declared[branch] == dense and declared["recipe.loss"] == "pinball"
    piecewise = specification(document, "titans_mac_disabled_branch_ic3")[branch]
    assert piecewise == {"columns": 46, "bins": 16, "embedding": 8, "hidden": 128}
    assert specification(document, "titans_mac_disabled_dense_ic3_t1")["recipe.truncation"] == 1
    # Las recetas se cargan con su término y los predictores con su rama.
    for name in ("core_ic1", "core_ic3", "core_ic10", "core_ic3_t1"):
        recipe = json.loads((ROOT / document["recipes"][name]).read_text())
        assert ChronologicalRecipe(**recipe["recipe"]).session_ic() is not None


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("dgx-us-campaign-v8.json", "dgx-us-campaign-v7.json"),
        ("dgx-us-cn-campaign-v7.json", "dgx-us-cn-campaign-v6.json"),
    ],
)
def test_the_campaigns_on_the_session_ic_term_and_the_dense_branch_keep_each_contrast(name, origin):
    path, source = ROOT / "configs/experiments" / name, ROOT / "configs/experiments" / origin
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    dropped = {"r3_branch", "r3_branch_more", "r3_t8", "r3_steps"}
    kept = [tier for tier in previous["tiers"] if tier not in dropped]
    added = [
        "r4_noic",
        "r4_nobranch",
        "r4_piecewise",
        "r4_weight",
        "r4_steps",
        "r4_original",
        "r4_dense",
        "r4_readers",
        "r4_ranks",
    ]
    assert [tier for tier in document["tiers"] if tier != "v_scalar_noic"] == kept + added
    assert document["tiers"].index("v_scalar_noic") == document["tiers"].index("v_scalar") + 1
    term, branch, steps = (
        "recipe.session_ic_term",
        "predictor.technical_branch",
        "recipe.truncation",
    )
    weight = {"weight": 3.0, "minimum_rows": 8}
    dense = {"columns": 46, "hidden": 128, "depth": 2}

    def changed(left, right, other=None):
        one, two = flat_specification(document, left), flat_specification(other or document, right)
        # Los cambios del predictor ya están desplegados campo a campo.
        keys = (one.keys() | two.keys()) - {"predictor_changes"}
        return {key for key in keys if one.get(key) != two.get(key)}

    # Cada modelo conserva su definición: solo gana el término y, si normaliza, la rama.
    carried = {model for model, entry in previous["models"].items() if entry["tier"] not in dropped}
    assert carried < set(document["models"])
    for model in sorted(carried):
        entry, before = document["models"][model], previous["models"][model]
        assert {k: v for k, v in entry.items() if k != "recipe"} == {
            k: v for k, v in before.items() if k != "recipe"
        }, model
        now = flat_specification(document, model)
        objective = "session_ic_objective" in (entry.get("predictor_changes") or {})
        if entry["runner"] != "fit" or objective:
            expected = set()
        elif now.get("predictor.learning_scale") is None or model == "titans_mac_no_charts":
            expected = {term}
        else:
            expected = {term, branch}
        assert changed(model, model, previous) == expected, model
        if term in expected:
            assert now[term] == weight and now["recipe.truncation"] == 2
        if branch in expected:
            assert now[branch] == dense
    # Los niveles r4_ quitan o cambian una pieza de la base.
    for base in ("titans_mac", "titans_mac_frozen", "titans_mac_disabled"):
        assert changed(f"{base}_noic", base) == {term}
        assert changed(f"{base}_nobranch", base) == {branch}
        assert changed(f"{base}_t8", base) == {steps}
        assert changed(f"{base}_original", base) == {term, branch, steps}
        # La receta original es la de la campaña sin rasgos, y la base anterior, la v7.
        assert changed(f"{base}_original", f"{base}_t8", previous) == set()
        assert changed(f"{base}_xrank", base) == {branch}
        assert flat_specification(document, f"{base}_xrank")[branch] == {
            **dense,
            "session_ranks": True,
        }
        assert flat_specification(document, f"{base}_noic").get(term) is None
        assert flat_specification(document, f"{base}_nobranch").get(branch) is None
    for base in ("titans_mac", "titans_mac_disabled"):
        assert changed(f"{base}_piecewise", base) == {branch}
        assert changed(f"{base}_dense256", base) == {branch}
        assert changed(f"{base}_icw1", base) == changed(f"{base}_icw10", base) == {term}
        assert changed(f"{base}_t1", base) == changed(f"{base}_t4", base) == {steps}
        assert flat_specification(document, f"{base}_icw10")[term]["weight"] == 10.0
        assert flat_specification(document, f"{base}_piecewise")[branch] == {
            "columns": 46,
            "bins": 16,
            "embedding": 8,
            "hidden": 128,
        }
    for head in ("titans_mac", "titans_mac_disabled"):
        assert changed(f"{head}_scalar_noic", f"{head}_scalar") == {term}
        assert changed(f"{head}_ic", f"{head}_scalar_noic") == {"predictor.session_ic_objective"}
    assert changed("titans_mac_no_charts", "titans_mac_nobranch") == {
        "predictor.removed_modalities"
    }
    assert changed("mars_titan_m0_noic", "mars_titan_m0") == {"parent"}
    assert changed("mars_titan_m3_noic", "mars_titan_m3") == {"parent"}
    assert document["models"]["mars_titan_m0_noic"]["parent"] == "titans_mac_noic"
    assert changed("mars_titan_m0_u2", "mars_titan_m0") == {"recipe.update_instants"}
    # Las recetas se cargan con su término y el predictor admite su rama.
    for recipe_name in (
        "core",
        "core_scalar",
        "core_nols",
        "core_ema",
        "core_t8",
        "core_piecewise",
    ):
        recipe = json.loads((ROOT / document["recipes"][recipe_name]).read_text())
        assert ChronologicalRecipe(**recipe["recipe"]).session_ic() is not None, recipe_name
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v8-contrasts.json").read_text()
    )["controls"]
    assert set(declared) == set(document["models"]) - {"titans_mac"}
    assert set(declared.values()) <= set(document["models"])
    assert declared["titans_mac_ic"] == "titans_mac_scalar_noic"
    assert declared["titans_mac_no_charts"] == "titans_mac_nobranch"
    assert declared["titans_mac_disabled_noic"] == "titans_mac_disabled"


def test_the_extension_of_the_campaign_changes_one_field_per_model():
    path = ROOT / "configs/experiments/dgx-us-campaign-v8-ext1.json"
    source = ROOT / "configs/experiments/dgx-us-campaign-v8.json"
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["tiers"] == [
        "core",
        "controls",
        "x1_liquid",
        "x1_weight",
        "x1_readers",
        "x1_reader_ic",
    ]
    # El núcleo y los lectores de la base son los de la campaña: mismos campos y recetas.
    for model in ("titans_mac", "titans_mac_disabled"):
        assert document["models"][model] == previous["models"][model]
    for model in ("mars_titan_m0", "mars_titan_m3", "mars_titan_cognitive"):
        entry, before = document["models"][model], previous["models"][model]
        assert {k: v for k, v in entry.items() if k != "tier"} == {
            k: v for k, v in before.items() if k != "tier"
        }
        assert entry["tier"] == "x1_readers"
    for name in ("core", "reader", "reader_cognitive"):
        assert document["recipes"][name] == previous["recipes"][name]

    def changed(left, right):
        one, two = flat_specification(document, left), flat_specification(document, right)
        keys = (one.keys() | two.keys()) - {"predictor_changes"}
        return {key for key in keys if one.get(key) != two.get(key)}

    term = "recipe.session_ic_term"
    base = flat_specification(document, "titans_mac")[term]
    assert base == {"weight": 3.0, "minimum_rows": 8}
    # La columna declarada es el volumen medio en dólares de 21 sesiones de los rasgos.
    names = technical_features.NAMES_V2
    column = names.index("log_dollar_volume_21") - len(names)
    for control in ("titans_mac", "titans_mac_disabled"):
        for suffix, power in (("liq05", 0.5), ("liq1", 1.0)):
            assert changed(f"{control}_{suffix}", control) == {term}
            declared = flat_specification(document, f"{control}_{suffix}")[term]
            assert declared == {**base, "liquidity_column": column, "liquidity_power": power}
            recipe = ChronologicalRecipe(**_recipe(document, f"{control}_{suffix}"))
            assert recipe.session_ic().weighted
    for control in ("titans_mac", "titans_mac_disabled"):
        for suffix, weight in (("icw03", 0.3), ("icw01", 0.1)):
            assert changed(f"{control}_{suffix}", control) == {term}
            declared = flat_specification(document, f"{control}_{suffix}")[term]
            assert declared == {**base, "weight": weight}
    for control in ("mars_titan_m0", "mars_titan_m3", "mars_titan_cognitive"):
        assert changed(f"{control}_ic", control) == {term}
        assert flat_specification(document, f"{control}_ic")[term] == base
        assert document["models"][f"{control}_ic"]["parent"] == "titans_mac"
        reader = readout.ReadoutRecipe(**_recipe(document, f"{control}_ic"))
        assert reader.session_ic() is not None and not reader.session_ic().weighted
        assert readout.ReadoutRecipe(**_recipe(document, control)).session_ic() is None
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v8-ext1-contrasts.json").read_text()
    )["controls"]
    assert set(declared) == set(document["models"]) - {"titans_mac"}
    assert set(declared.values()) <= set(document["models"])
    assert declared["mars_titan_m3_ic"] == "mars_titan_m3"
    assert declared["titans_mac_disabled_liq1"] == "titans_mac_disabled"


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("dgx-us-campaign-v9.json", "dgx-us-campaign-v8.json"),
        ("dgx-us-cn-campaign-v8.json", "dgx-us-cn-campaign-v7.json"),
    ],
)
def test_the_campaigns_with_the_median_gain_only_add_it_and_their_controls(name, origin):
    path, source = ROOT / "configs/experiments" / name, ROOT / "configs/experiments" / origin
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["source_declaration_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert document["tiers"] == previous["tiers"] + ["r4_nogain", "r4_liquid", "r4_reader_ic"]
    gain, term = "predictor.median_gain", "recipe.session_ic_term"

    def changed(left, right, other=None):
        one, two = flat_specification(document, left), flat_specification(other or document, right)
        keys = (one.keys() | two.keys()) - {"predictor_changes"}
        return {key for key in keys if one.get(key) != two.get(key)}

    # Cada modelo conserva su entrada. Solo gana la ganancia si ajusta una cabeza de cuantiles.
    assert set(previous["models"]) < set(document["models"])
    gained = 0
    for model, before in previous["models"].items():
        assert document["models"][model] == before, model
        now = flat_specification(document, model)
        quantiles = now.get("predictor.head") == "quantile_head_v1"
        kept = before.get("recipe") in ("core_original", "core_scalar_noic", "core_scalar")
        if before["runner"] == "fit" and quantiles and not kept:
            assert changed(model, model, previous) == {gain}, model
            assert now[gain] == "learned_median_gain_v1"
            gained += 1
        else:
            assert changed(model, model, previous) == set(), model
            assert now.get(gain) is None, model
    assert gained > 60
    # La ganancia solo se declara con la cabeza de cuantiles, que es la única que la admite.
    for model in document["models"]:
        now = flat_specification(document, model)
        assert now.get(gain) is None or now.get("predictor.head") == "quantile_head_v1", model
    base = flat_specification(document, "titans_mac")[term]
    assert base == {"weight": 3.0, "minimum_rows": 8}
    for control in ("titans_mac", "titans_mac_frozen", "titans_mac_disabled"):
        # Sin la ganancia, el núcleo es el de la campaña anterior, campo a campo.
        assert changed(f"{control}_nogain", control) == {gain}
        assert changed(f"{control}_nogain", control, previous) == set()
        assert document["models"][f"{control}_nogain"]["tier"] == "r4_nogain"
    names = technical_features.NAMES_V2
    column = names.index("log_dollar_volume_21") - len(names)
    for control in ("titans_mac", "titans_mac_disabled"):
        for suffix, power in (("liq05", 0.5), ("liq1", 1.0)):
            assert changed(f"{control}_{suffix}", control) == {term}
            declared = flat_specification(document, f"{control}_{suffix}")[term]
            assert declared == {**base, "liquidity_column": column, "liquidity_power": power}
            assert (
                ChronologicalRecipe(**_recipe(document, f"{control}_{suffix}"))
                .session_ic()
                .weighted
            )
            assert document["models"][f"{control}_{suffix}"]["tier"] == "r4_liquid"
        for suffix, weight in (("icw03", 0.3), ("icw01", 0.1), ("icw1", 1.0), ("icw10", 10.0)):
            assert changed(f"{control}_{suffix}", control) == {term}
            declared = flat_specification(document, f"{control}_{suffix}")[term]
            assert declared == {**base, "weight": weight}
            assert document["models"][f"{control}_{suffix}"]["tier"] == "r4_weight"
    for control in ("mars_titan_m0", "mars_titan_m3", "mars_titan_cognitive"):
        assert changed(f"{control}_ic", control) == {term}
        assert flat_specification(document, f"{control}_ic")[term] == base
        assert document["models"][f"{control}_ic"]["parent"] == "titans_mac"
        assert document["models"][f"{control}_ic"]["tier"] == "r4_reader_ic"
        assert readout.ReadoutRecipe(**_recipe(document, f"{control}_ic")).session_ic() is not None
    # Las piezas de la base siguen separadas de ella por un solo campo.
    for control in ("titans_mac", "titans_mac_frozen", "titans_mac_disabled"):
        assert changed(f"{control}_noic", control) == {term}
        assert changed(f"{control}_nobranch", control) == {"predictor.technical_branch"}
        assert changed(f"{control}_xrank", control) == {"predictor.technical_branch"}
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v9-contrasts.json").read_text()
    )["controls"]
    assert set(declared) == set(document["models"]) - {"titans_mac"}
    assert set(declared.values()) <= set(document["models"])
    assert declared["titans_mac_frozen_nogain"] == "titans_mac_frozen"
    assert declared["mars_titan_cognitive_ic"] == "mars_titan_cognitive"
    assert declared["titans_mac_disabled_icw01"] == "titans_mac_disabled"
    # El plan cuenta los modelos nuevos en los tres orígenes de desarrollo de sus niveles.
    jobs = campaign.plan(document, "mars_titan")
    added = {job.model for job in jobs if job.tier in ("r4_nogain", "r4_liquid", "r4_reader_ic")}
    assert len(added) == 3 + 4 + 3


@pytest.mark.parametrize(
    ("name", "origin"),
    [
        ("dgx-us-campaign-v10.json", "dgx-us-campaign-v9.json"),
        ("dgx-us-cn-campaign-v9.json", "dgx-us-cn-campaign-v8.json"),
    ],
)
def test_the_campaigns_with_session_ranks_in_the_base_keep_each_contrast(name, origin):
    path, source = ROOT / "configs/experiments" / name, ROOT / "configs/experiments" / origin
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes", "stopping"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    # La regla de parada es la misma meseta de validación con otro tope: en el origen 2013 de
    # la v8 el de 16 épocas cortó dos ajustes que seguían mejorando. El tope queda como
    # salvaguarda que no llega a decidir.
    caps = dict(fit=(16, 48), reader=(12, 24))
    assert set(document["stopping"]) == set(previous["stopping"]) == set(caps)
    for stage, (before, after) in caps.items():
        rule, old = document["stopping"][stage], previous["stopping"][stage]
        assert (old["max_epochs"], rule["max_epochs"]) == (before, after)
        assert {**rule, "max_epochs": before} == old
        assert rule["stopping"] == "validation_plateau" and rule["patience"] == 3
    assert document["source_declaration_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert document["tiers"] == [
        "r4_noranks" if tier == "r4_ranks" else tier for tier in previous["tiers"]
    ]
    branch, gain, term = (
        "predictor.technical_branch",
        "predictor.median_gain",
        "recipe.session_ic_term",
    )
    dense = {"columns", "hidden", "depth"}
    cores = ("titans_mac", "titans_mac_frozen", "titans_mac_disabled")

    def changed(left, right, other=None):
        one, two = flat_specification(document, left), flat_specification(other or document, right)
        keys = (one.keys() | two.keys()) - {"predictor_changes"}
        return {key for key in keys if one.get(key) != two.get(key)}

    carried = {model for model in previous["models"] if not model.endswith("_xrank")}
    assert carried | {f"{core}_noranks" for core in cores} == set(document["models"])
    ranked = 0
    for model in sorted(carried):
        entry, before = document["models"][model], previous["models"][model]
        assert entry == before, model
        was, now = flat_specification(previous, model), flat_specification(document, model)
        if before["runner"] == "reader":
            # Los lectores solo cambian el bloque, para que cada sesión llegue entera al padre.
            assert changed(model, model, previous) == {
                "recipe.block_rows",
                "recipe.max_working_bytes",
            }, model
            assert now["recipe.block_rows"] == 8192
            assert now["recipe.max_working_bytes"] == 2 * 1024**3
        elif isinstance(was.get(branch), dict) and set(was[branch]) == dense:
            assert changed(model, model, previous) == {branch}, model
            assert now[branch] == {**was[branch], "session_ranks": True}
            ranked += 1
        else:
            assert changed(model, model, previous) == set(), model
    assert ranked > 60
    for core in cores:
        # Sin los rangos, el núcleo es el de la campaña anterior, campo a campo.
        assert changed(f"{core}_noranks", core) == {branch}
        assert changed(f"{core}_noranks", core, previous) == set()
        assert document["models"][f"{core}_noranks"]["tier"] == "r4_noranks"
        assert changed(f"{core}_nogain", core) == {gain}
        assert changed(f"{core}_noic", core) == {term}
        assert changed(f"{core}_nobranch", core) == {branch}
    # Todo lector tiene un padre que necesita la sesión entera, y su bloque la contiene.
    for model, entry in document["models"].items():
        if entry["runner"] != "reader":
            continue
        parent = flat_specification(document, entry["parent"])
        whole = (parent.get(branch) or {}).get("session_ranks") is True
        assert whole or parent.get("predictor.cross_asset_attention") is not None, model
        assert (
            flat_specification(document, model)["recipe.block_rows"] >= document["cohort"]["size"]
        )
        reader = readout.ReadoutRecipe(**_recipe(document, model))
        assert reader.block_rows == 8192
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v10-contrasts.json").read_text()
    )["controls"]
    assert set(declared) == set(document["models"]) - {"titans_mac"}
    assert set(declared.values()) <= set(document["models"])
    assert declared["titans_mac_disabled_noranks"] == "titans_mac_disabled"
    assert not any(model.endswith("_xrank") for model in declared)
    jobs = campaign.plan(document, "mars_titan")
    assert {job.model for job in jobs if job.tier == "r4_noranks"} == {
        f"{core}_noranks" for core in cores
    }


def _recipe(document, model):
    name = document["models"][model]["recipe"]
    return json.loads((ROOT / document["recipes"][name]).read_text())["recipe"]


def test_the_extension_of_v10_changes_one_recipe_field_per_model():
    path = ROOT / "configs/experiments/dgx-us-campaign-v10-ext1.json"
    source = ROOT / "configs/experiments/dgx-us-campaign-v10.json"
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["source_declaration_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert document["tiers"] == ["core", "controls", "x1_ema", "x1_routing", "x1_median", "x1_norm"]
    # El núcleo y sus controles son los de la campaña, para heredar sus ajustes.
    cores = ("titans_mac", "titans_mac_frozen", "titans_mac_disabled")
    for model in cores:
        assert document["models"][model] == previous["models"][model]
    assert document["recipes"]["core"] == previous["recipes"]["core"]

    def changed(left, right):
        one, two = flat_specification(document, left), flat_specification(document, right)
        keys = (one.keys() | two.keys()) - {"predictor_changes"}
        return {key for key in keys if one.get(key) != two.get(key)}

    scale = flat_specification(document, "titans_mac")["predictor.learning_scale"]
    assert scale["zero_median"] is True
    expected = {
        "titans_mac_ema999": ("x1_ema", "recipe.weight_average", {"decay": 0.999}),
        "titans_mac_ema9998": ("x1_ema", "recipe.weight_average", {"decay": 0.9998}),
        "titans_mac_qroute": (
            "x1_routing",
            "predictor.quantile_routing",
            {"median_from_tails": False, "encoder_from_spreads": False},
        ),
        "titans_mac_qroute_median": (
            "x1_routing",
            "predictor.quantile_routing",
            {"median_from_tails": False, "encoder_from_spreads": True},
        ),
        "titans_mac_qroute_encoder": (
            "x1_routing",
            "predictor.quantile_routing",
            {"median_from_tails": True, "encoder_from_spreads": False},
        ),
        "titans_mac_nozm": (
            "x1_median",
            "predictor.learning_scale",
            {**scale, "zero_median": False},
        ),
        "titans_mac_nonorm": ("x1_norm", "predictor.memory_residual_layer_norm", False),
    }
    assert set(document["models"]) == set(cores) | set(expected)
    for model, (tier, field, value) in expected.items():
        entry = document["models"][model]
        assert entry == {
            **previous["models"]["titans_mac"],
            "recipe": entry["recipe"],
            "tier": tier,
        }
        assert changed(model, "titans_mac") == {field}, model
        assert flat_specification(document, model)[field] == value
    # La v10 ya mide el promedio de pesos con otra retención: aquí solo cambia ese número.
    for model, decay in (("titans_mac_ema999", 0.999), ("titans_mac_ema9998", 0.9998)):
        assert ChronologicalRecipe(**_recipe(document, model)).weight_average == {"decay": decay}
    assert flat_specification(previous, "titans_mac_ema")["recipe.weight_average"] == {
        "decay": 0.995
    }
    declared = json.loads(
        (ROOT / "configs/experiments/dgx-us-campaign-v10-ext1-contrasts.json").read_text()
    )["controls"]
    assert set(declared) == set(document["models"]) - {"titans_mac"}
    assert set(declared.values()) == {"titans_mac"}
    jobs = campaign.plan(document, "mars_titan")
    assert len({job.model for job in jobs if job.tier.startswith("x1_")}) == 7


def test_the_first_screen_of_the_new_base_changes_one_recipe_field_per_model():
    path = ROOT / "configs/experiments/us-campaign-v11-screen1.json"
    source = ROOT / "configs/experiments/dgx-us-campaign-v10.json"
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["source_declaration_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    tiers = ["core", "controls", "y1_norm", "y1_clip", "y1_recency", "y1_scale", "y1_speed"]
    assert document["tiers"] == [
        *tiers[:2],
        "r4_nogain",
        *tiers[2:],
        "y1_operable",
        "y1_operable_nogain",
        "y1_xattn",
        "y1_xattn_late",
        "y1_xattn_nogain",
        "y1_xattn_nogain_controls",
    ]
    # El núcleo y sus controles son los de la v10, campo a campo: la base nueva los reproduce.
    cores = ("titans_mac", "titans_mac_frozen", "titans_mac_disabled")
    # El trío sin ganancia de la mediana entra también tal como está en la v10.
    nogain = ("titans_mac_nogain", "titans_mac_frozen_nogain", "titans_mac_disabled_nogain")
    for model in (*cores, *nogain):
        assert document["models"][model] == previous["models"][model]
    for key in ("core", "core_nogain"):
        assert document["recipes"][key] == previous["recipes"][key]

    def changed(left, right):
        one, two = flat_specification(document, left), flat_specification(document, right)
        keys = (one.keys() | two.keys()) - {"predictor_changes"}
        return {key for key in keys if one.get(key) != two.get(key)}

    base = flat_specification(document, "titans_mac")
    branch, scale = base["predictor.technical_branch"], base["predictor.learning_scale"]
    assert base["recipe.max_grad_norm"] == 1.0 and base["recipe.accumulation_rows"] == 8192
    expected = {
        "titans_mac_ln": (
            "titans_mac",
            "y1_norm",
            "predictor.technical_branch",
            {**branch, "layer_norm": True},
        ),
        "titans_mac_cliprole": ("titans_mac", "y1_clip", "recipe.grad_clip_by_role", True),
        "titans_mac_clip10": ("titans_mac", "y1_clip", "recipe.max_grad_norm", 10.0),
        "titans_mac_rec5": ("titans_mac", "y1_recency", "recipe.recency_half_life_years", 5.0),
        "titans_mac_rec10": ("titans_mac", "y1_recency", "recipe.recency_half_life_years", 10.0),
        "titans_mac_smooth": (
            "titans_mac",
            "y1_scale",
            "predictor.learning_scale",
            {**scale, "smooth_clip": True},
        ),
        "titans_mac_fscale": (
            "titans_mac",
            "y1_scale",
            "predictor.technical_branch",
            {**branch, "feature_scaling": True},
        ),
        "titans_mac_nograph": ("titans_mac", "y1_speed", "recipe.observe_without_graph", True),
        "titans_mac_frozen_nograph": (
            "titans_mac_frozen",
            "y1_speed",
            "recipe.observe_without_graph",
            True,
        ),
        "titans_mac_disabled_nograph": (
            "titans_mac_disabled",
            "y1_speed",
            "recipe.observe_without_graph",
            True,
        ),
    }
    # El término de IC sobre lo operable, en los tres modos de memoria: el alfa conocido es la
    # primera de las 46 columnas técnicas del bloque del gráfico, en tanto por ciento.
    assert branch["columns"] == 46 and technical_features.NAMES[0] == "alpha_percent"
    operable = {**base["recipe.session_ic_term"], "offset_column": -46, "offset_scale": 0.01}
    assert set(base["recipe.session_ic_term"]) == {"weight", "minimum_rows"}
    for parent in cores:
        expected[f"{parent}_icop"] = (parent, "y1_operable", "recipe.session_ic_term", operable)
    # Y sobre el trío sin ganancia: allí el término ajusta la escala de la mediana él solo.
    for parent in nogain:
        assert (
            flat_specification(document, parent)["recipe.session_ic_term"]
            == (base["recipe.session_ic_term"])
        )
        assert "predictor.median_gain" not in flat_specification(document, parent)
        expected[f"{parent}_icop"] = (
            parent,
            "y1_operable_nogain",
            "recipe.session_ic_term",
            operable,
        )
    # La atención entre activos en sus dos posiciones, cada una con su control self_only. Las
    # dos de la v10 se declaran como allí; las tardías solo añaden la posición.
    late = dict(position="after_technical")
    mixed, alone = dict(heads=4, mode="same_session"), dict(heads=4, mode="self_only")
    # Por modelo: la atención declarada, su control, el modelo del que toma la receta y su nivel.
    mixers = {
        "cross_asset_attention": (mixed, "titans_mac", "titans_mac", "y1_xattn"),
        "cross_asset_attention_self": (alone, "cross_asset_attention", "titans_mac", "y1_xattn"),
        "cross_asset_attention_late": (
            {**mixed, **late},
            "cross_asset_attention",
            "titans_mac",
            "y1_xattn_late",
        ),
        "cross_asset_attention_late_self": (
            {**alone, **late},
            "cross_asset_attention_late",
            "titans_mac",
            "y1_xattn_late",
        ),
        # La atención sobre el trío sin ganancia de la mediana, el mejor núcleo medido en 2013:
        # la mezcla y su control self_only, y los dos modos de memoria que la acompañan.
        "cross_asset_attention_nogain": (
            mixed,
            "titans_mac_nogain",
            "titans_mac_nogain",
            "y1_xattn_nogain",
        ),
        "cross_asset_attention_nogain_self": (
            alone,
            "cross_asset_attention_nogain",
            "titans_mac_nogain",
            "y1_xattn_nogain",
        ),
        "cross_asset_attention_frozen_nogain": (
            mixed,
            "cross_asset_attention_nogain",
            "titans_mac_frozen_nogain",
            "y1_xattn_nogain_controls",
        ),
        "cross_asset_attention_disabled_nogain": (
            mixed,
            "cross_asset_attention_nogain",
            "titans_mac_disabled_nogain",
            "y1_xattn_nogain_controls",
        ),
    }
    positions = {}
    for model, (declared, _control, parent, tier) in mixers.items():
        entry = document["models"][model]
        assert entry == {
            **previous["models"][parent],
            "predictor_changes": dict(cross_asset_attention=declared),
            "tier": tier,
        }
        if model in previous["models"]:
            assert entry == {**previous["models"][model], "tier": "y1_xattn"}
        # Fuera de la atención, cada uno es su modelo de origen campo a campo.
        assert changed(model, parent) <= {"predictor.cross_asset_attention"}, model
        positions[model] = parse_cross_asset_attention(declared).identity()["position"]
    assert set(previous["models"]) & set(mixers) == {
        "cross_asset_attention",
        "cross_asset_attention_self",
    }
    assert positions == {
        **{model: "after_mac_before_head" for model in mixers if "_late" not in model},
        "cross_asset_attention_late": "after_technical_branch_before_head",
        "cross_asset_attention_late_self": "after_technical_branch_before_head",
    }
    assert set(document["models"]) == set(cores) | set(nogain) | set(expected) | set(mixers)
    for model, (parent, tier, field, value) in expected.items():
        entry = document["models"][model]
        assert entry == {**previous["models"][parent], "recipe": entry["recipe"], "tier": tier}
        assert changed(model, parent) == {field}, model
        assert flat_specification(document, model)[field] == value
        # La receta de cada variante se construye con el código de esta rama.
        ChronologicalRecipe(**_recipe(document, model))
    declared = json.loads(
        (ROOT / "configs/experiments/us-campaign-v11-screen1-contrasts.json").read_text()
    )["controls"]
    controls = {model: parent for model, (parent, *_rest) in expected.items()}
    controls.update(titans_mac_frozen="titans_mac", titans_mac_disabled="titans_mac")
    controls.update(zip(nogain, cores, strict=True))
    controls.update({model: control for model, (_declared, control, *_rest) in mixers.items()})
    assert declared == controls
    jobs = campaign.plan(document, "mars_titan")
    assert len({job.model for job in jobs if job.tier.startswith("y1_")}) == 24
    # Los modos de memoria de la atención sin ganancia tienen su nivel: se ajustan si esta gana.
    assert {job.model for job in jobs if job.tier == "y1_xattn_nogain_controls"} == {
        "cross_asset_attention_frozen_nogain",
        "cross_asset_attention_disabled_nogain",
    }


def test_the_second_screen_of_the_new_base_adds_the_two_terms_and_the_weight_decay():
    path = ROOT / "configs/experiments/us-campaign-v11-screen2.json"
    source = ROOT / "configs/experiments/dgx-us-campaign-v10.json"
    document, previous = campaign.load_declaration(path)[0], json.loads(source.read_text())
    own = {"description", "source_declaration_sha256", "models", "tiers", "recipes"}
    assert set(document) == set(previous)
    for key in set(previous) - own:
        assert document[key] == previous[key], key
    assert document["source_declaration_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert document["tiers"] == [
        "core",
        "controls",
        "r4_nogain",
        "y2_both",
        "y2_both_controls",
        "y2_wd",
        "y2_wd_controls",
        "y2_combo",
    ]
    cores = ("titans_mac", "titans_mac_frozen", "titans_mac_disabled")
    nogain = ("titans_mac_nogain", "titans_mac_frozen_nogain", "titans_mac_disabled_nogain")
    # Los dos tríos entran como en la v10: son el control de cada variante con el mismo código.
    for model in (*cores, *nogain):
        assert document["models"][model] == previous["models"][model]
    for key in ("core", "core_nogain"):
        assert document["recipes"][key] == previous["recipes"][key]

    def changed(left, right):
        one, two = flat_specification(document, left), flat_specification(document, right)
        keys = (one.keys() | two.keys()) - {"predictor_changes"}
        return {key for key in keys if one.get(key) != two.get(key)}

    base = flat_specification(document, "titans_mac")
    assert set(base["recipe.session_ic_term"]) == {"weight", "minimum_rows"}
    assert base["recipe.weight_decay"] == 0 and technical_features.NAMES[0] == "alpha_percent"
    # Los dos términos de IC: el de lo operable, con el alfa conocido, y el de la etiqueta con
    # el mismo peso.
    both = {
        **base["recipe.session_ic_term"],
        "offset_column": -46,
        "offset_scale": 0.01,
        "plain_weight": base["recipe.session_ic_term"]["weight"],
    }
    expected = {}
    for parent in (*cores, *nogain):
        online = parent in ("titans_mac", "titans_mac_nogain")
        tier = "y2_both" if online else "y2_both_controls"
        expected[f"{parent}_icboth"] = (parent, tier, "recipe.session_ic_term", both)
    # El decaimiento de pesos de 0,01: con la ganancia en los tres modos, y sin ella en línea.
    for parent in (*cores, "titans_mac_nogain"):
        online = parent in ("titans_mac", "titans_mac_nogain")
        tier = "y2_wd" if online else "y2_wd_controls"
        expected[f"{parent}_wd1e2"] = (parent, tier, "recipe.weight_decay", 0.01)
    # La candidata a receta final de un solo modelo: sin ganancia, con los dos términos y con la
    # atención entre activos de la v10. De su control solo la separa la atención.
    combo, mixed = "cross_asset_attention_nogain_icboth", dict(heads=4, mode="same_session")
    assert set(document["models"]) == set(cores) | set(nogain) | set(expected) | {combo}
    assert document["models"][combo] == {
        **document["models"]["titans_mac_nogain_icboth"],
        "predictor_changes": dict(cross_asset_attention=mixed),
        "tier": "y2_combo",
    }
    assert changed(combo, "titans_mac_nogain_icboth") == {"predictor.cross_asset_attention"}
    for model, (parent, tier, field, value) in expected.items():
        entry = document["models"][model]
        assert entry == {**previous["models"][parent], "recipe": entry["recipe"], "tier": tier}
        assert changed(model, parent) == {field}, model
        assert flat_specification(document, model)[field] == value
        ChronologicalRecipe(**_recipe(document, model))
    declared = json.loads(
        (ROOT / "configs/experiments/us-campaign-v11-screen2-contrasts.json").read_text()
    )["controls"]
    controls = {model: parent for model, (parent, *_rest) in expected.items()}
    controls[combo] = "titans_mac_nogain_icboth"
    controls.update(titans_mac_frozen="titans_mac", titans_mac_disabled="titans_mac")
    controls.update(zip(nogain, cores, strict=True))
    assert declared == controls
    jobs = campaign.plan(document, "mars_titan")
    by_tier = {}
    for job in jobs:
        by_tier.setdefault(job.tier, set()).add(job.model)
    assert by_tier["y2_both"] == {"titans_mac_icboth", "titans_mac_nogain_icboth"}
    assert by_tier["y2_wd"] == {"titans_mac_wd1e2", "titans_mac_nogain_wd1e2"}
    assert len(by_tier["y2_both_controls"]) == 4 and len(by_tier["y2_wd_controls"]) == 2
    assert by_tier["y2_combo"] == {combo}
