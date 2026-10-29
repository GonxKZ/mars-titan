"""E1 se valida con la misma receta de ventanas y olvido discreto explícitamente cero."""

import json
from dataclasses import asdict
from pathlib import Path

from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.memory.multiscale_mixture import MultiscaleMixtureConfig
from mars_titan.training.mars_titan_correction import (
    arm_components,
    case_values,
    load_correction_recipe,
)


def test_the_factor_is_normalized_by_the_common_window_builder():
    # Sin los periodos de R2 (#786) la identidad normalizada de E1 no los nombra.
    factor = asdict(MultiscaleMixtureConfig())
    assert factor.pop("write_periods") is None
    components = dict(associative_memory=dict(rule="proximal", key="codec", multiscale=factor))
    checked, correction = check_components(
        load_declaration(), arm_components(components, dict(rate=0.05, forgetting=0.0))
    )
    assert correction.multiscale == MultiscaleMixtureConfig()
    assert checked["associative_memory"]["multiscale"] == factor


def test_declared_models_and_controls_are_executable_with_their_own_recipes():
    study = json.loads(Path("configs/experiments/multiscale-mature-mixture-v1.json").read_text())
    assert study["default_enabled"] is False and study["final_test_access"] is False
    identities = []
    for arm in (study["arms"] | study["references"]).values():
        recipe = load_correction_recipe(arm["recipe"])
        for case in recipe["walk_forward"]["search_cases"]:
            _, correction = check_components(
                load_declaration(), arm_components(arm["components"], case_values(recipe, case))
            )
            identities.append(json.dumps(correction.identity(), sort_keys=True))
    assert len(identities) == len(set(identities)) == 24
