import importlib
from datetime import UTC, datetime

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.universe")
    except ModuleNotFoundError:
        pytest.fail("La selección del piloto sin información futura todavía no existe")


def test_selection_order_does_not_use_sector_future_volume_or_survival():
    rows = [
        {"symbol": symbol, "bytes": size, "sector": "current"}
        for symbol, size in [("A", 1), ("B", 100), ("C", 10000)]
    ]
    order = [x["symbol"] for x in module().selection_order(rows, seed=42)]
    changed = [{**x, "bytes": 0, "sector": "new", "survives_to_test": False} for x in rows[::-1]]
    assert [x["symbol"] for x in module().selection_order(changed, seed=42)] == order


def test_training_coverage_ignores_appended_future_samples():
    cutoff = datetime(2018, 12, 31, 23, 59, tzinfo=UTC)
    past = datetime(2018, 12, 28, 21, 5, tzinfo=UTC)
    future = datetime(2025, 3, 28, 20, 5, tzinfo=UTC)
    rows = [{"prediction_at": past}, {"prediction_at": future}]
    assert module().training_coverage(rows, {past, future}, cutoff) == 1
    assert module().training_coverage(rows, {future}, cutoff) == 0
