"""Los controles E3 cargan la misma política acotada y solo cambian su puntuación."""

import json
from pathlib import Path

from mars_titan.memory.salience_budget import RULES, SalienceBudgetConfig


def test_declared_controls_are_optional_and_load_the_same_bounded_configuration():
    data = json.loads(Path("configs/experiments/salience-write-budget-v1.json").read_text())
    assert data["default_enabled"] is False and data["final_test_access"] is False
    assert data["status"] == "technical_core_only"
    loaded = [SalienceBudgetConfig(**record).identity() for record in data["arms"].values()]
    assert {record["rule"] for record in loaded} == set(RULES)
    for record in loaded:
        record.pop("rule")
    assert loaded[0] == loaded[1] == loaded[2]
