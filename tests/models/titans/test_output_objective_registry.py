"""Cada objetivo de salida registrado se puede declarar y entra en las huellas de código."""

from mars_titan.models.titans.frozen_financial import _SOURCE_MODULES
from mars_titan.models.titans.output_objectives import OBJECTIVES
from mars_titan.training.search_cases import PREDICTOR_CHANGES
from mars_titan.training.titans_walk_forward import _CODE


def test_every_output_objective_is_declarable_and_enters_the_code_identities():
    modules = {kind.__module__ for kind in OBJECTIVES.values()}
    assert len(modules) == len(OBJECTIVES)
    # Sin su módulo en las huellas, cambiar la pérdida no cambiaría la identidad del recorrido.
    assert modules <= set(_CODE)
    assert modules <= set(_SOURCE_MODULES)
    assert set(OBJECTIVES) <= set(PREDICTOR_CHANGES)
