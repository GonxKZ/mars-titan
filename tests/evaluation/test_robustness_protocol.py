"""Declaración de robustez comprobada contra el código, sin evaluar nada."""

import copy
import json

import pytest

from mars_titan.data import input_perturbations
from mars_titan.evaluation import robustness_protocol

DECLARATION = "configs/evaluation/historical-masked-2000-robustness.json"


@pytest.fixture
def document():
    return json.loads(open(DECLARATION, encoding="utf-8").read())


def written(tmp_path, document):
    path = tmp_path / "robustness.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_repository_declaration_matches_the_code():
    declaration, digest = robustness_protocol.load_declaration(DECLARATION)
    assert declaration["status"] == "declared_not_executed" and len(digest) == 64
    variants = declaration["input_perturbations"]["variants"]
    assert set(variants) == set(input_perturbations.PERTURBATIONS)


def change(document, dotted, value):
    changed = copy.deepcopy(document)
    target = changed
    *parents, last = dotted.split(".")
    for part in parents:
        target = target[part]
    if value is KeyError:
        del target[last]
    else:
        target[last] = value
    return changed


@pytest.mark.parametrize(
    "dotted, value, message",
    [
        ("final_test_opened", True, "reserva"),
        ("executions", 1, "reserva"),
        ("partition", "validation", "reserva"),
        ("input_perturbations.seed", 1, "no son las del código"),
        ("input_perturbations.random", "pcg64", "no son las del código"),
        ("input_perturbations.variants.noise_10pct.scale", 0.2, "no son las del código"),
        ("input_perturbations.variants.burst_absence_21d_10pct", KeyError, "no son las del código"),
        ("registered.periods", KeyError, "citar costes"),
        ("registered.costs.values_bps", [0, 10], "no son los declarados"),
        ("registered.costs.sources", ["configs/evaluation/no-existe.json"], "no existe"),
        ("registered.modality_absence.fields", ["modality_masks"], "no está en sus fuentes"),
        ("registered.abstention.pending", "", "qué falta"),
        (
            "reader_guards.duplicated_session.test",
            "tests/training/test_input_perturbations.py::test_missing",
            "no existe",
        ),
        ("reader_guards.availability_after_decision.test", "sin referencia", "no cita"),
        ("reader_guards.availability_falsified_earlier.reason", "", "no se detecta"),
        ("reader_guards", {}, "Faltan las guardas"),
    ],
)
def test_a_declaration_that_differs_from_the_repository_is_rejected(
    tmp_path, document, dotted, value, message
):
    path = written(tmp_path, change(document, dotted, value))
    with pytest.raises(ValueError, match=message):
        robustness_protocol.load_declaration(path)


def test_a_new_perturbation_in_the_code_needs_its_declaration(monkeypatch):
    spec = dict(kind="multiplicative_noise", modalities=("news",), scale=0.2)
    monkeypatch.setitem(input_perturbations.PERTURBATIONS, "noise_news_20pct", spec)
    with pytest.raises(ValueError, match="no son las del código"):
        robustness_protocol.load_declaration(DECLARATION)
