"""Todo sitio que ajusta parámetros pasa por una guarda inventariada y con prueba de rechazo.

El grafo de llamadas recorre `src/mars_titan` y `scripts`. Un punto de entrada nuevo que
llegue a un paso de optimizador, a un `fit`, a `xgboost.train` o a una solución cerrada sin
pasar por `require_learning_allowed` hace fallar la primera prueba. Una guarda nueva,
retirada o duplicada cambia el inventario y hace fallar la tercera.
"""

import ast
from functools import cache
from pathlib import Path

import pytest

from tests.training.learning_hold_graph import CallGraph, label
from tests.training.learning_hold_inventory import EXEMPT_SITES, GUARDS, NO_STEP_HOOKS
from tests.training.test_learning_hold_guards import ENTRY_POINTS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def graph():
    return CallGraph(ROOT)


@cache
def _tests(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _readable(found):
    return {name: " <- ".join(chain) for name, (_, chain) in found.items()}


def test_every_fitting_site_is_reached_only_through_a_guard(graph):
    assert graph.sites, "El grafo no encuentra ningún sitio de ajuste"
    assert _readable(graph.unprotected(exempt_sites=EXEMPT_SITES)) == {}


def test_no_function_can_fit_before_its_first_guard(graph):
    assert graph.late_guards(exempt_sites=EXEMPT_SITES) == []


def test_each_guard_call_has_its_refusal_test_in_the_inventory(graph):
    found = {label(node): len(lines) for node, lines in graph.guards.items()}
    assert {name: len(references) for name, references in GUARDS.items()} == found


def test_inventory_references_name_existing_refusal_tests():
    references = [reference for group in GUARDS.values() for reference in group]
    # Cada caso de ENTRY_POINTS demuestra exactamente una guarda y todos figuran.
    assert sorted(r for r in references if "::" not in r) == sorted(ENTRY_POINTS)
    for reference in {r for r in references if "::" in r}:
        path, name = reference.split("::")
        test = _tests(path).get(name)
        assert test is not None, f"No existe la prueba {reference}"
        source = ast.unparse(test)
        assert "learning_hold" in source or "LearningHoldError" in source, reference


def test_declared_hooks_and_exemptions_are_the_ones_still_needed(graph):
    assert {label(node) for node in graph.hooks} == set(NO_STEP_HOOKS)
    # Una exención solo se mantiene si sin ella el sitio quedaría expuesto.
    assert set(EXEMPT_SITES) <= set(graph.unprotected())
    covered = {label(node) for node in graph.without_steps}
    assert covered, "Ningún medidor instala el gancho sin pasos"
