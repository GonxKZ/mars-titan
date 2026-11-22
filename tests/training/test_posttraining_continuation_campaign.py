"""Continuaciones de post-entrenamiento sobre los padres de la v10 (fase 4 del programa).

`dgx-us-campaign-v10-post1.json` vuelve a ajustar el padre de la v10 en el origen anterior y
continúa desde su estado elegido con cuatro recetas que solo cambian el ajuste: la misma
receta, decaimiento hacia cero, decaimiento anclado al padre (L2-SP) y contracción con ruido.
Ninguna prueba ajusta un modelo: comprueban la declaración, su plan y sus recetas.
"""

import hashlib
import json

import pytest

from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.financial_run import ChronologicalRecipe

DECLARATION = "configs/experiments/dgx-us-campaign-v10-post1.json"
CORE = "configs/titans/dgx-us-core-u2-ic3-dense-gr-v9.json"
# Huella de la receta del núcleo de la v10 en 3d10a30f, la versión que ejecuta la campaña (con
# accumulation_rows de 8192): el padre que se vuelve a ajustar es el mismo modelo.
CORE_SHA256 = "3b88760f86b922f7fec97c8f0a9af41a8bee3148b252def45ca43468d68a61e6"
# Lo único que cambia cada receta de continuación respecto del núcleo de la v10.
CHANGES = {
    "cont_wd": dict(weight_decay=0.1),
    "cont_l2sp": dict(weight_decay=0.1, weight_decay_anchor="initial_parameters"),
    "cont_sp": dict(shrink_perturb=dict(shrink=0.6, noise=0.01, seed=1)),
}
ARMS = {
    "titans_mac_cont": "core",
    "titans_mac_cont_wd": "cont_wd",
    "titans_mac_cont_l2sp": "cont_l2sp",
    "titans_mac_cont_sp": "cont_sp",
}
SEEDS = (42, 43, 44)


def load(path):
    return json.loads((campaign.ROOT / path).read_text(encoding="utf-8"))


def test_the_parent_recipe_is_the_core_of_campaign_v10():
    digest = hashlib.sha256((campaign.ROOT / CORE).read_bytes()).hexdigest()
    assert digest == CORE_SHA256
    document = load(DECLARATION)
    assert document["recipes"]["core"] == CORE
    assert document["models"]["titans_mac"] == dict(
        machine="mars_titan", recipe="core", runner="fit", tier="core", variant="mac_online"
    )


def test_every_arm_continues_from_the_parent_of_the_previous_origin_with_its_seed():
    document = campaign.checked_declaration(load(DECLARATION))
    assert document["final_test_access"] is False and document["market"] == "US"
    assert document["cohort"] == dict(size=4198, rule="entire_source_view")
    assert document["inputs"] == dict(technical="long_horizon_v2")
    assert [origin["year"] for origin in document["origins"]] == [2012, 2013]
    assert document["stopping"]["fit"] == dict(
        metric="session_rank_ic",
        min_assets=30,
        stopping="validation_plateau",
        patience=3,
        min_delta=0,
        max_epochs=48,
        initial_state="reported",
    )
    jobs = campaign.plan(document, "mars_titan")
    assert not campaign.plan(document, "references")
    parents = sorted(job.key for job in jobs if job.model == "titans_mac")
    assert parents == sorted(
        f"titans_mac/{year}/seed-{seed}" for year in (2012, 2013) for seed in SEEDS
    )
    continued = sorted((job.key, job.needs) for job in jobs if job.model != "titans_mac")
    # Sin trabajo en 2012: el primer origen no tiene un padre anterior del que partir.
    assert continued == sorted(
        (f"{arm}/2013/seed-{seed}", f"titans_mac/2012/seed-{seed}")
        for arm in ARMS
        for seed in SEEDS
    )
    for arm, recipe in ARMS.items():
        assert document["models"][arm] == dict(
            machine="mars_titan",
            recipe=recipe,
            runner="fit",
            tier="post",
            variant="mac_online",
            continuation=campaign.CONTINUATION,
            continues="titans_mac",
        )


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_each_continuation_recipe_changes_only_its_declared_fit_option(name):
    core, recipe = load(CORE), load(load(DECLARATION)["recipes"][name])
    # La misma arquitectura: el estado del padre se carga entero en la continuación.
    assert recipe["predictor"] == core["predictor"]
    for field in ("pairing_source", "variants", "walk_forward", "recipe_name"):
        assert recipe[field] == core[field]
    assert recipe["recipe"] == dict(core["recipe"], **CHANGES[name])
    options = ChronologicalRecipe(**recipe["recipe"]).weight_options()
    assert ChronologicalRecipe(**core["recipe"]).weight_options() == dict.fromkeys(options)
    if name == "cont_sp":
        assert options["shrink_perturb"] is not None
    else:
        assert all(value is None for value in options.values())
    if name == "cont_l2sp":
        # Con λ = 0 el ancla no haría nada: el decaimiento lleva cada peso hacia el padre.
        assert recipe["recipe"]["weight_decay"] > 0
