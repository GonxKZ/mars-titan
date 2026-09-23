"""Presupuesto de búsqueda fijo, sin elegir combinaciones después de ver resultados."""

import importlib
import json
from collections import Counter

import pytest


def design(models):
    try:
        module = importlib.import_module("mars_titan.training.reference_design")
    except ModuleNotFoundError:
        pytest.fail("Falta el diseño determinista de configuraciones")
    return module.design_cases(models)


def test_twelve_unique_cases_per_family_cover_every_declared_level():
    cases = design(["rnn", "lstm", "gru", "dlinear"])
    assert len(cases) == 48
    for kind in ("rnn", "lstm", "gru", "dlinear"):
        rows = [r["case"] for r in cases if r["case"]["kind"] == kind]
        assert len(rows) == len({json.dumps(row, sort_keys=True) for row in rows}) == 12
        assert Counter(r["architecture"]["hidden_size"] for r in rows) == {32: 4, 64: 4, 128: 4}
        assert Counter(r["architecture"]["layers"] for r in rows) == {1: 6, 2: 6}
        assert Counter(r["architecture"]["dropout"] for r in rows) == {0.0: 4, 0.1: 4, 0.2: 4}
        assert Counter(r["loss"] for r in rows) == {"mae": 4, "mse": 4, "huber": 4}
        assert Counter(r["learning_rate"] for r in rows) == {0.0001: 4, 0.0003: 4, 0.001: 4}
        assert all(r["seed"] == 42 and r["epochs"] == 30 for r in rows)
        assert all(
            r["selection"] == dict(metric="session_mae", patience=5, min_delta=0.0) for r in rows
        )


def test_every_family_receives_the_same_deterministic_hyperparameter_design():
    first = design(["rnn", "gru"])
    assert first == design(["rnn", "gru"])
    by_kind = [
        [
            {k: v for k, v in row["case"].items() if k != "kind"}
            for row in first
            if row["case"]["kind"] == kind
        ]
        for kind in ("rnn", "gru")
    ]
    assert by_kind[0] == by_kind[1]


@pytest.mark.parametrize("models", [[], ["mars-titan"], ["rnn", "rnn"]])
def test_unplanned_or_repeated_families_are_rejected(models):
    with pytest.raises(ValueError, match="familias|modelos"):
        design(models)
