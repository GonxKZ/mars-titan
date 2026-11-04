"""El perfil E4 conserva el padre, la receta y sus casos y solo añade la puerta opcional."""

import inspect
import json
from pathlib import Path

from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.memory.regime_evidence_gate import RegimeGateConfig
from mars_titan.training import regime_gate_inference
from mars_titan.training.mars_titan_correction import (
    arm_components,
    case_values,
    load_correction_recipe,
    run_correction_window,
)
from mars_titan.training.regime_gate_inference import checked_configuration


def test_declared_window_profiles_use_real_arguments_and_a_matched_control():
    path = Path("configs/experiments/regime-evidence-gate-window-v1.json")
    document = json.loads(path.read_text())
    assert document["default_enabled"] is False and document["final_test_access"] is False
    assert document["status"] == "declared_not_executed"
    assert document["entrypoint"] == f"{run_correction_window.__module__}.run_correction_window"
    assert set(document["arguments"]) <= set(inspect.signature(run_correction_window).parameters)
    arms = document["arms"]
    assert set(arms) == {"baseline", "cusum", "matched_constant"} and arms["baseline"] is None
    variant = RegimeGateConfig(**arms["cusum"])
    assert variant.identity() == {**RegimeGateConfig().identity(), **arms["cusum"]}
    matched = arms["matched_constant"]
    assert matched["control_of"] == "cusum"
    module, name = matched["builder"].rsplit(".", 1)
    assert module == regime_gate_inference.__name__ and callable(
        getattr(regime_gate_inference, name)
    )
    recipe = load_correction_recipe(document["arguments"]["recipe"])
    for case in recipe["walk_forward"]["search_cases"]:
        combined = arm_components(document["arguments"]["components"], case_values(recipe, case))
        _, correction = check_components(load_declaration(), combined)
        assert checked_configuration(variant, correction) is variant
