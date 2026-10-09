"""Declaración de MARS-TITAN con ampliaciones: apagada, sin ejecutar y coherente con el código."""

import json
import re
from pathlib import Path

import pytest

from mars_titan.memory.associative_memory import RULES
from mars_titan.models.titans.episodic_readout import EpisodicReadoutConfig
from mars_titan.models.titans.financial import VARIANTS

CONFIG = Path("configs/titans/mars-titan-extensions.json")
FIELDS = {
    "map_ids",
    "enabled",
    "disabled_value",
    "allowed",
    "phases",
    "state",
    "insertion",
    "level",
    "issues",
    "control",
    "pending",
}


@pytest.fixture(scope="module")
def declaration():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _values(change):
    return change if isinstance(change, list) else [change]


def test_declaration_is_not_executed_and_every_component_is_disabled(declaration):
    assert declaration["status"] == "declared_not_executed"
    assert declaration["final_test_opened"] is False
    assert declaration["identity_rule"]["all_disabled"] == "base"
    for name, component in declaration["components"].items():
        assert FIELDS <= component.keys(), name
        assert component["enabled"] is False, name
        assert component["level"] in {"N1", "N2", "N3", "N4"}, name
        assert component["issues"] and component["control"], name


def test_base_is_the_titans_arm_without_any_extension(declaration):
    base = declaration["base"]
    assert base["architecture"] == "titans_mac" and base["variant"] in VARIANTS
    assert base["consumer"] == {"readout": None, "admission": "m0", "local_control": None}
    recipe = json.loads(Path(base["recipe"]).read_text(encoding="utf-8"))
    assert recipe["predictor"]["bank_policy"] == "disabled"
    assert recipe["predictor"]["refinements"] == 1


def test_insertion_points_name_existing_files_and_symbols(declaration):
    for name, component in declaration["components"].items():
        path, _, symbol = component["insertion"].partition(":")
        assert Path(path).is_file(), name
        if symbol:
            last = symbol.rsplit(".", 1)[-1]
            text = Path(path).read_text(encoding="utf-8")
            assert re.search(rf"^\s*(?:def|class) {re.escape(last)}\b", text, re.M), name


def test_refinements_match_the_readout_and_do_not_count_memory_updates(declaration):
    refinements = declaration["components"]["refinements"]
    assert refinements["counts_memory_updates"] is False
    assert refinements["requires"] == "episodic_bank"
    for k in refinements["allowed"]:
        EpisodicReadoutConfig("a" * 64, refinements=k)
    with pytest.raises(ValueError):
        EpisodicReadoutConfig("a" * 64, refinements=3)
    assert "m3" not in declaration["components"]["episodic_bank"]["allowed"]


def test_associative_rules_match_the_implemented_component(declaration):
    assert declaration["components"]["associative_memory"]["allowed"] == list(RULES)


def test_each_ablation_changes_one_declared_component(declaration):
    components = declaration["components"]
    known = {"base", "fitted_arm"}
    for ablation in declaration["ablations"]:
        assert ablation["reference"] in known, ablation["id"]
        (name, change), *rest = ablation["change"].items()
        assert not rest and name in components, ablation["id"]
        assert set(_values(change)) <= set(components[name]["allowed"]), ablation["id"]
        known.add(ablation["id"])


def test_specification_lists_every_mapped_modification(declaration):
    text = Path(declaration["specification"]).read_text(encoding="utf-8")
    assert "configs/titans/mars-titan-extensions.json" in text
    for component in declaration["components"].values():
        for identifier in component["map_ids"]:
            assert f"| {identifier} |" in text, identifier
