"""Campaña DGX v3: niveles para dejar fuera modelos redundantes y modelos de la segunda ronda."""

import hashlib
import json
from pathlib import Path

import pytest

from mars_titan.training import comparison_campaign as campaign
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
