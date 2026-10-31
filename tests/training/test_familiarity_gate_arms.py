"""R3 (#786) se declara como cambio de la receta del lector y llega a su ventana y a la criba."""

import json
from pathlib import Path

import pytest

from benchmarks import variant_screening_study as study
from mars_titan.data.storage import atomic_json
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.mars_titan_variant import (
    check_components,
    core_identity,
    load_declaration,
    select_variant,
)
from mars_titan.training import mars_titan_run as mt
from mars_titan.training import mars_titan_walk_forward, variant_screening
from tests.training.test_financial_run import shared as shared
from tests.training.test_mars_titan_run import build, parent, recipe
from tests.training.test_mars_titan_run import explicit_fastpath as explicit_fastpath
from tests.training.test_mars_titan_run import native as native

STUDY = Path("configs/experiments/familiarity-gate-v1.json")
READER = "configs/titans/episodic-readout-historical-masked.json"


def test_the_recipe_names_the_gate_only_when_declared():
    plain = mt.ReadoutRecipe()
    assert plain.familiarity_gate is None and "familiarity_gate" not in plain.identity()
    for gate in ("familiarity", "constant"):
        identity = mt.ReadoutRecipe(familiarity_gate=gate).identity()
        assert identity["familiarity_gate"] == gate
        assert {key: value for key, value in identity.items() if key != "familiarity_gate"} == (
            plain.identity()
        )


class Name(str):
    """Un texto que solo se parece al nombre de la puerta."""


@pytest.mark.parametrize("value", ["mean", True, 1, "", Name("familiarity")])
def test_the_recipe_rejects_an_unknown_gate(value):
    with pytest.raises(ValueError, match="receta del lector"):
        mt.ReadoutRecipe(familiarity_gate=value)


def test_the_window_builds_the_reader_with_the_gate_of_its_recipe(shared):
    _, streams = shared
    predictor = parent(streams)
    codec = FrozenEpisodeCodec(streams["train"].specification())
    base = core_identity(predictor, dict(recipe_sha256="0" * 64, search_case=None, window="w"))
    variant = select_variant(load_declaration(), {"episodic_bank": "m1"}, base=base)
    for gate in (None, "familiarity", "constant"):
        plan = recipe(familiarity_gate=gate)
        readout = mars_titan_walk_forward._readout(variant, codec, predictor, plan, 42)
        assert readout.config.familiarity_gate == gate


def test_the_reader_run_needs_the_gate_of_its_recipe(shared, tmp_path, native):
    _, streams = shared
    plan = recipe(familiarity_gate="familiarity")
    trainer = build(streams, tmp_path / "gated", native, plan=plan, familiarity_gate="familiarity")
    assert trainer.readout.config.familiarity_gate == "familiarity"
    for gate in (None, "constant"):
        with pytest.raises(ValueError, match="receta declarada"):
            build(streams, tmp_path / str(gate), native, plan=plan, familiarity_gate=gate)


def test_the_declared_models_differ_only_in_the_gate_and_reach_the_screening_recipe(tmp_path):
    document = json.loads(STUDY.read_text())
    assert document["issue"] == 786 and document["epic"] == 782
    assert document["default_enabled"] is False and document["final_test_access"] is False
    assert document["recipe"] == READER
    declaration = load_declaration()
    for arm in document["arms"].values():
        components, correction = check_components(declaration, arm["components"])
        assert components == {"episodic_bank": "m1"} and correction is None
    protocol = variant_screening.load_protocol()
    entry = dict(
        identity=document["variant_id"],
        runner="reader",
        declaration=str(STUDY),
        parent=document["parent"],
        recipe=READER,
    )
    arms = variant_screening.entry_arms(dict(protocol, catalog={"R3": entry}), "R3")
    gates = {}
    for name, arm in arms.items():
        path = tmp_path / f"{name}.json"
        atomic_json(path, study.reader_recipe(protocol, arm.definition))
        derived = mt.load_recipe(path)
        cases = derived["walk_forward"]["search_cases"] or {None: {}}
        gates[name] = {mt.case_recipe(derived, case).familiarity_gate for case in cases}
    assert gates == {"familiarity_gate": {"familiarity"}, "constant_gate": {"constant"}}
    # El control del lector de referencia es el de la criba anterior: M1 con K = 1 y sin puerta.
    reference = protocol["references"]["reader_reference"]
    assert reference["components"] == {"episodic_bank": "m1"} and reference["recipe"] == READER
    assert reference["recipe_changes"] == {}
