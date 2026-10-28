"""Declaración independiente del diagnóstico regional congelado."""

import json
from pathlib import Path

import pytest

from mars_titan.memory.spaced_study import load_study

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/experiments/spaced-regional-utility-study-v1.json"
)


def test_regional_study_is_explicit_disabled_and_has_its_own_identity():
    assert CONFIG.is_file(), "Falta la configuración independiente E2 regional"
    study, digest = load_study(CONFIG)
    assert len(digest) == 64 and study.reader_contract == "regional"
    assert study.identity()["kind"] == "spaced_regional_utility_frozen_study_v1"
    assert study.identity()["reader_contract"] == "regional"
    assert study.arms == (
        "baseline",
        "reservoir",
        "utility",
        "permuted_utility",
        "uniform_frontier",
        "recent",
    )
    assert json.loads(CONFIG.read_text())["default_enabled"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("reader_contract", "standard"),
        ("schema_version", 1),
        ("kind", "spaced_utility_frozen_study_v1"),
        ("variant_id", "spaced_utility_consolidation_v1"),
    ],
)
def test_regional_profile_cannot_masquerade_as_the_standard_study(tmp_path, field, value):
    assert CONFIG.is_file(), "Falta la configuración independiente E2 regional"
    document = json.loads(CONFIG.read_text())
    document[field] = value
    path = tmp_path / "study.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_study(path)
