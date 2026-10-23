"""La medición rechazada no habilita una nueva variante financiera."""

import json
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.mark.parametrize("boundary", ["predictor", "window"])
def test_rejected_vector_precomputation_is_not_admitted_by_financial_factories(boundary):
    if boundary == "predictor":
        from tests.models.titans.test_financial_adapter import setup

        model, _ = setup("transformer_direct")
        with pytest.raises(ValueError):
            replace(model.config, temporal_encoder="vector_delta_precomputed_v1")
    else:
        from tests.training.test_window_variants import recipe

        with pytest.raises(ValueError):
            recipe(temporal_encoder="vector_delta_precomputed_v1")


def test_optional_month_recipe_changes_only_the_execution_and_keeps_the_original_control():
    from mars_titan.training.window_variant_run import WindowVariantRecipe

    root = Path(__file__).resolve().parents[1] / "configs/experiments"
    reference = json.loads(
        (root / "window-variants-pilot-month-v3/kronlstm-bp-adamw.json").read_text()
    )
    candidate = json.loads((root / "kronlstm-precomputed/window-month.json").read_text())
    assert candidate == dict(reference, temporal_encoder="kronlstm_precomputed_v1")
    assert WindowVariantRecipe(**candidate).final_test_access is False
