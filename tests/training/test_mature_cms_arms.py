"""R2 (#786) se declara en la ventana de correcciones con el control replicado de E1."""

import json
from pathlib import Path

from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.training.mars_titan_correction import (
    arm_components,
    case_values,
    load_correction_recipe,
)

STUDY = Path("configs/experiments/mature-cms-v1.json")
E1 = Path("configs/experiments/multiscale-mature-mixture-v1.json")


def test_the_control_is_the_replicated_quarter_of_e1_and_the_same_screening_job():
    study, e1 = json.loads(STUDY.read_text()), json.loads(E1.read_text())
    assert study["issue"] == 786 and study["epic"] == 782
    assert study["default_enabled"] is False and study["final_test_access"] is False
    assert study["arms"]["same_frequency"] == e1["arms"]["replicated_quarter"]
    assert study["comparisons"] == [{"variant": "mature_cms", "control": "same_frequency"}]
    variant = study["arms"]["mature_cms"]["components"]["associative_memory"]["multiscale"]
    control = study["arms"]["same_frequency"]["components"]["associative_memory"]["multiscale"]
    # El modelo y su control solo difieren en los periodos de escritura.
    assert {key: value for key, value in variant.items() if key != "write_periods"} == control


def test_declared_models_are_executable_and_only_r2_names_its_periods():
    study = json.loads(STUDY.read_text())
    identities = {}
    for name, arm in study["arms"].items():
        recipe = load_correction_recipe(arm["recipe"])
        for case in recipe["walk_forward"]["search_cases"]:
            checked, correction = check_components(
                load_declaration(), arm_components(arm["components"], case_values(recipe, case))
            )
            identities.setdefault(name, []).append(
                json.dumps(correction.identity(), sort_keys=True)
            )
            multiscale = checked["associative_memory"]["multiscale"]
            if name == "mature_cms":
                assert correction.multiscale.write_periods == ("session", "week", "month")
                assert list(multiscale["write_periods"]) == ["session", "week", "month"]
            else:
                assert correction.multiscale.write_periods is None
                assert "write_periods" not in multiscale
    every = identities["mature_cms"] + identities["same_frequency"]
    assert len(every) == len(set(every)) == 2 * len(identities["mature_cms"])
