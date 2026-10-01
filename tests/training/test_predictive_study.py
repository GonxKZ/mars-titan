"""Orquestación de controles emparejados sin repetir preparaciones ni ajustes terminados."""

import importlib
import json

import pytest

from mars_titan.training.checkpoints import StopRequest
from tests.training.test_predictive_parents import setup

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("duckdb", "gymnasium")),
    reason="Requiere los extras data y reinforcement",
)


def test_study_resumes_paired_controls_and_reuses_parent_preparation(tmp_path, monkeypatch):
    engine = importlib.import_module("mars_titan.training.predictive_study")
    ordered, parent, _ = setup(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                schema_version=1,
                modes=["reinforce", "expected", "mae"],
                seeds=[42],
                epochs=1,
                learning_rate=0.001,
                weight_decay=0.01,
                clip_norm=1.0,
                batch_size=5,
                checkpoint_seconds=60,
                final_test_opened=False,
            )
        )
    )
    stop = StopRequest()
    real = engine.run_predictive_case
    new_runs = []

    def controlled(*args, **kwargs):
        if not kwargs.get("resume"):
            new_runs.append(args[3]["mode"])
        result = real(*args, **kwargs)
        stop.request_stop()
        return result

    monkeypatch.setattr(engine, "run_predictive_case", controlled)
    first = engine.run_predictive_study(config, ordered, parent, tmp_path / "study", stop=stop)
    assert first["status"] == "paused" and first["completed_runs"] == 1
    monkeypatch.setattr(engine, "run_predictive_case", real)
    result = engine.run_predictive_study(config, ordered, parent, tmp_path / "study", resume=True)
    assert result["status"] == "completed" and result["completed_runs"] == 3
    assert {item["case"]["mode"] for item in result["runs"]} == {"reinforce", "expected", "mae"}
    assert {item["global_step"] for item in result["runs"]} == {3}
    assert new_runs == ["reinforce"]
    assert result["normalization_sha256"] == first["normalization_sha256"]
