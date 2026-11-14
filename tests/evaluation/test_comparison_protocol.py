"""La inferencia y los contrastes se fijan antes de leer las predicciones."""

import copy

import pytest

from mars_titan.evaluation import comparison_protocol as protocol
from mars_titan.training.comparison_campaign import load_declaration


def test_protocol_preserves_primary_metric_pairs_and_independent_bootstrap_seed():
    models = load_declaration()[0]["models"]
    options = protocol.checked(protocol.declaration(models), models)
    assert options["primary_metric"] == "mae_raw"
    assert options["seed_aggregation"] == "mean_of_session_metrics"
    assert options["bootstrap"]["stratification"] == "origin"
    assert options["bootstrap"]["seed"] not in (42, 43, 44)
    assert ["mars_titan_m3", "mars_titan_m0"] in options["families"]["episodic_memory"]
    assert len(options["families"]["primary"]) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("replicates", True),
        ("replicates", 100_001),
        ("seed", -1),
        ("confidence", float("nan")),
        ("block_days", 0),
        ("stratification", "all_years_together"),
    ],
)
def test_bootstrap_settings_must_be_bounded_and_declared(field, value):
    models = load_declaration()[0]["models"]
    options = protocol.declaration(models)
    options["bootstrap"][field] = value
    with pytest.raises(ValueError):
        protocol.checked(options, models)


@pytest.mark.parametrize(
    "pairs",
    [
        [["unknown", "zero"]],
        [["transformer", "transformer"]],
        [["transformer", "zero"], ["transformer", "zero"]],
    ],
)
def test_invalid_or_repeated_contrasts_are_refused(pairs):
    models = load_declaration()[0]["models"]
    options = protocol.declaration(models)
    options["families"]["references"] = pairs
    with pytest.raises(ValueError, match="contrastes"):
        protocol.checked(options, models)


def test_validation_does_not_mutate_the_declared_protocol():
    models = load_declaration()[0]["models"]
    original = protocol.declaration(models)
    copy_before = copy.deepcopy(original)
    checked = protocol.checked(original, models)
    checked["bootstrap"]["seed"] += 1
    assert original == copy_before
