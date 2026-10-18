"""Opciones declaradas de los núcleos combinados, sin convertirlas en búsqueda implícita."""

import pytest

from mars_titan.training.mars_titan_run import ReadoutRecipe
from mars_titan.training.search_cases import checked_predictor_changes


def test_declared_multiscale_and_presence_options_can_be_combined():
    changes = dict(
        memory_multiscale=dict(form="cascade", coupling=0.0625),
        presence_modulation="film_v1",
        attention_layout="sequence_contiguous_v1",
        memory_stability=dict(alpha_floor=0.002, eta_ceiling=0.3, gradient_clip=4.0, gate_box=True),
    )
    assert checked_predictor_changes(changes) == changes


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"head": "other"},
        {"memory_multiscale": "cascade"},
        {"presence_modulation": {}},
        {"presence_modulation": False},
    ],
)
def test_undeclared_fields_or_wrong_shapes_are_rejected(changes):
    with pytest.raises(ValueError):
        checked_predictor_changes(changes)


def test_readout_recipe_keeps_legacy_identity_unless_initialization_is_enabled():
    assert "refinement_initialization" not in ReadoutRecipe().identity()
    enabled = ReadoutRecipe(refinement_initialization="zero_refinement")
    assert enabled.identity()["refinement_initialization"] == "zero_refinement"
    with pytest.raises(ValueError):
        ReadoutRecipe(refinement_initialization="unknown")
