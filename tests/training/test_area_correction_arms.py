"""Modelos de B1 y sus controles declarados, validados como los modelos B6 de la ventana.

Cada modelo de `configs/experiments/redundant-area-memory-v1.json` se completa con los casos
de su receta y pasa por la misma validación que `run_correction_window` aplica antes de abrir
ninguna fuente. Así la criba puede combinarlos con el padre Titans-MAC sin código nuevo.
"""

import json
from pathlib import Path

import pytest

from mars_titan.memory import mars_titan_variant
from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.kernel_memory import KernelMemoryConfig
from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.models.titans.config import canonical
from mars_titan.training.mars_titan_correction import (
    arm_components,
    case_values,
    load_correction_recipe,
)

DECLARED = Path("configs/experiments/redundant-area-memory-v1.json")
B6_RECIPE = Path("configs/titans/mature-correction-historical-masked.json")
KERNEL_RECIPE = Path("configs/titans/kernel-correction-historical-masked.json")
AREAS = dict(areas=4, shifts=1, permutation="seeded_random", seed=0, labels="mature")


@pytest.fixture(scope="module")
def declaration():
    return load_declaration()


@pytest.fixture(scope="module")
def study():
    return json.loads(DECLARED.read_text(encoding="utf-8"))


def completed(declaration, arm, case=None):
    document = load_correction_recipe(arm["recipe"])
    case = case or next(iter(document["walk_forward"]["search_cases"]))
    combination = arm_components(arm["components"], case_values(document, case))
    return check_components(declaration, combination)


def test_every_declared_arm_is_a_valid_window_model_with_its_own_identity(declaration, study):
    assert study["status"] == "declared_not_executed" and study["final_test_access"] is False
    identities = {}
    for name, arm in study["arms"].items():
        document = load_correction_recipe(arm["recipe"])
        for case in document["walk_forward"]["search_cases"]:
            checked, correction = completed(declaration, arm, case)
            assert set(checked) == {"associative_memory"} and correction is not None
            identities[name, case] = canonical(correction.identity())
    assert len(set(identities.values())) == len(identities) == 12
    named = set(study["arms"]) | set(study["references"])
    assert all({pair["variant"], pair["control"]} <= named for pair in study["comparisons"])
    assert {pair["control"] for pair in study["comparisons"]} == named - {"area_memory"}


def test_the_controls_differ_from_the_variant_in_one_declared_way(declaration, study):
    arms = {name: arm["components"]["associative_memory"] for name, arm in study["arms"].items()}
    variant = arms["area_memory"]["areas"]
    assert arms["identical_areas"]["areas"] == dict(variant, permutation="identity")
    assert arms["shuffled_labels"]["areas"] == dict(variant, labels="shuffled_within_cohort")
    single = AreaLayout(**arms["single_area_same_coordinates"]["areas"])
    layout = AreaLayout(**variant)
    assert single.areas * single.feature_size(64) == layout.areas * layout.feature_size(64)
    assert arms["linear_codec"] == {"rule": "proximal", "key": "codec"}
    assert arms["nadaraya_watson"] == {"rule": "nadaraya_watson", "key": "codec"}


def test_b6_models_keep_their_components_and_identity(declaration):
    values = case_values(load_correction_recipe(B6_RECIPE), "eta5e-2")
    checked, correction = check_components(
        declaration,
        arm_components({"associative_memory": {"rule": "proximal", "key": "codec"}}, values),
    )
    assert checked["associative_memory"] == dict(
        rule="proximal", key="codec", rate=0.05, forgetting=0.01
    )
    assert "areas" not in correction.identity() and correction.areas is None
    assert type(correction.empty()).__name__ == "AssociativeMemory"


@pytest.mark.parametrize(
    ("arm", "recipe"),
    [
        ({"rule": "nadaraya_watson", "key": "codec"}, B6_RECIPE),
        ({"rule": "proximal", "key": "codec", "areas": AREAS}, KERNEL_RECIPE),
        ({"rule": "nadaraya_watson", "key": "codec", "areas": AREAS}, KERNEL_RECIPE),
        (
            {"rule": "kalman", "key": "codec", "cohort_correlation": "zero", "areas": AREAS},
            B6_RECIPE,
        ),
    ],
)
def test_arms_with_another_recipe_or_areas_on_another_rule_are_rejected(arm, recipe):
    document = load_correction_recipe(recipe)
    values = case_values(document, next(iter(document["walk_forward"]["search_cases"])))
    with pytest.raises(ValueError):
        arm_components({"associative_memory": arm}, values)


@pytest.mark.parametrize(
    "value",
    [
        dict(rule="proximal", key="constant", rate=0.05, forgetting=0.01, areas=AREAS),
        dict(rule="proximal", key="codec_by_regime", rate=0.05, forgetting=0.01, areas=AREAS),
        dict(rule="proximal", key="codec", rate=0.05, forgetting=0.01, areas={"areas": 4}),
        dict(rule="proximal", key="codec", rate=0.05, forgetting=0.01, areas=dict(AREAS, shifts=5)),
        dict(rule="nadaraya_watson", key="constant", bandwidth=0.1, forgetting=0.0, capacity=8),
        dict(rule="nadaraya_watson", key="codec", bandwidth=0.1, forgetting=0.0),
    ],
)
def test_components_outside_the_declaration_are_rejected(declaration, value):
    with pytest.raises(ValueError):
        check_components(declaration, {"associative_memory": value})


def test_areas_are_rejected_with_the_episodic_bank(declaration):
    value = dict(rule="proximal", key="codec", rate=0.05, forgetting=0.01, areas=AREAS)
    with pytest.raises(ValueError, match="sin banco episódico"):
        check_components(declaration, {"associative_memory": value, "episodic_bank": "m1"})


def test_corrections_reject_areas_and_kernels_outside_the_codec_key():
    with pytest.raises(ValueError, match="clave del codec"):
        MatureCorrection(
            AssociativeMemoryConfig("proximal", key_size=1), key="constant", areas=AreaLayout()
        )
    with pytest.raises(ValueError, match="clave del codec"):
        MatureCorrection(KernelMemoryConfig(bandwidth=0.1), areas=AreaLayout())
    with pytest.raises(ValueError, match="clave del codec"):
        MatureCorrection(AssociativeMemoryConfig("proximal"), areas={"areas": 4})


def test_declaration_drift_in_the_area_block_fails_the_load(tmp_path):
    document = json.loads(mars_titan_variant.DECLARATION.read_text(encoding="utf-8"))
    document["components"]["associative_memory"]["area_value"]["areas"]["shifts"] = [1, 8]
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="B1"):
        load_declaration(path)
