"""Disponibilidad explícita de la ejecución opcional, sin alterar la referencia."""

import importlib
import importlib.util


def test_precomputed_scan_has_an_explicit_optional_entry_point():
    name = "mars_titan.models.kronlstm.precomputed"
    assert importlib.util.find_spec(name) is not None, "Falta la ejecución precomputada opcional"
    module = importlib.import_module(name)
    package = importlib.import_module("mars_titan.models.kronlstm")
    assert callable(module.kronlstm_scan_precomputed)
    assert package.kronlstm_scan_precomputed is module.kronlstm_scan_precomputed
