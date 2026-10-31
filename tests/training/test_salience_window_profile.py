"""El perfil E3 conserva el padre, la receta y sus casos y solo añade el factor opcional."""

import inspect
import json
from pathlib import Path

from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.memory.salience_budget import RULES, SalienceBudgetConfig
from mars_titan.training.mars_titan_correction import (
    arm_components,
    case_values,
    load_correction_recipe,
    run_correction_window,
)


def test_declared_window_profiles_use_real_arguments_and_match_their_write_budgets():
    path = Path("configs/experiments/salience-correction-window-v1.json")
    document = json.loads(path.read_text())
    assert document["default_enabled"] is False and document["final_test_access"] is False
    assert document["status"] == "declared_not_executed"
    assert document["entrypoint"] == f"{run_correction_window.__module__}.run_correction_window"
    assert set(document["arguments"]) <= set(inspect.signature(run_correction_window).parameters)
    assert document["arms"]["baseline"] is None
    configurations = [
        SalienceBudgetConfig(**value) for value in document["arms"].values() if value is not None
    ]
    assert {value.rule for value in configurations} == set(RULES)
    identities = [
        {key: value for key, value in config.identity().items() if key != "rule"}
        for config in configurations
    ]
    assert identities[0] == identities[1] == identities[2]
    recipe = load_correction_recipe(document["arguments"]["recipe"])
    for case in recipe["walk_forward"]["search_cases"]:
        combined = arm_components(document["arguments"]["components"], case_values(recipe, case))
        _, correction = check_components(load_declaration(), combined)
        assert correction.memory.rule == "proximal" and correction.memory.value_size == 1
