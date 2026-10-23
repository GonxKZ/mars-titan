"""Disponibilidad del generador finito, sin construir redes o ajustar parámetros."""

import importlib
import importlib.util


def test_error_proposer_entry_points_are_available_in_call():
    for name, function in (
        ("error_rule", "ErrorProposalPlan"),
        ("error_source", "prepare_error_review"),
        ("error_proposer", "propose_from_mature_errors"),
    ):
        path = f"mars_titan.proposals.{name}"
        assert importlib.util.find_spec(path) is not None, f"Falta el módulo {path}"
        assert callable(getattr(importlib.import_module(path), function, None))
