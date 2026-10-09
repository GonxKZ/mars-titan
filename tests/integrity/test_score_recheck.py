"""Recálculo de una comparación walk-forward publicada sobre predicciones de juguete.

Reutiliza el estudio de `tests/evaluation/test_walk_forward_comparison.py`, cuyas
predicciones se escriben con valores conocidos. No hay modelos ajustados ni pasos de
optimizador.
"""

import json

import numpy as np
import pytest

from mars_titan.evaluation import forecast_scores
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.integrity import score_recheck
from tests.evaluation.test_walk_forward_comparison import Study


def publish(tmp_path, scope="US+CN"):
    study = Study(tmp_path / "study", scope=scope)
    output = tmp_path / "result"
    walk.write_walk_forward(study.config_path, study.sources_path, scope, output)
    return study, output


@pytest.mark.parametrize("scope", ["US", "US+CN"])
def test_a_faithful_comparison_passes_the_recheck(tmp_path, scope):
    study, output = publish(tmp_path, scope)
    result = score_recheck.recheck(study.config_path, study.sources_path, scope, output)
    assert result["passed"], json.dumps(result, indent=1)[:2000]
    # Brazos ridge (1 semilla), gru y titans (2 semillas) en 2 ventanas.
    assert len(result["windows"]) == 2 * (1 + 2 + 2)
    assert all(check["sessions"] > 0 for check in result["windows"])
    views = {entry["view"] for entry in result["aggregates"]}
    assert views == ({"US"} if scope == "US" else {"US+CN", "US", "CN"})


def test_a_bug_in_the_primary_mae_is_detected(tmp_path, monkeypatch):
    original = forecast_scores._mean
    calls = {"n": 0}

    def biased(panel, values):
        # Simula un error de agregación: la primera media de cada panel se divide por n+1.
        calls["n"] += 1
        result = original(panel, values)
        if calls["n"] % 2 == 1:
            result = result * panel.session_samples / (panel.session_samples + 1)
        return result

    monkeypatch.setattr(forecast_scores, "_mean", biased)
    study, output = publish(tmp_path, "US")
    monkeypatch.setattr(forecast_scores, "_mean", original)
    result = score_recheck.recheck(study.config_path, study.sources_path, "US", output)
    assert not result["passed"]
    assert any(
        check["columns"].get("mae", {}).get("mismatched_sessions", 0) > 0
        for check in result["windows"]
    )


def test_a_bug_in_the_interval_score_is_detected(tmp_path, monkeypatch):
    original = forecast_scores._quantile_scores

    def halved(panel):
        fields = original(panel)
        fields["interval_score"] = fields["interval_score"] * 0.5
        return fields

    monkeypatch.setattr(forecast_scores, "_quantile_scores", halved)
    study, output = publish(tmp_path, "US")
    monkeypatch.setattr(forecast_scores, "_quantile_scores", original)
    result = score_recheck.recheck(study.config_path, study.sources_path, "US", output)
    assert not result["passed"]
    failing = {
        name
        for check in result["windows"]
        for name, entry in check["columns"].items()
        if entry.get("mismatched_sessions")
    }
    assert failing == {"interval_score_0.8", "interval_score_0.95"}


def test_a_changed_session_table_is_rejected_by_its_digest(tmp_path):
    study, output = publish(tmp_path, "US")
    path = output / "sessions.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="huella"):
        score_recheck.recheck(study.config_path, study.sources_path, "US", output)


def test_the_cli_writes_the_recheck_and_signals_failure(tmp_path, monkeypatch):
    study, output = publish(tmp_path, "US")
    target = tmp_path / "recheck.json"
    arguments = [str(study.config_path), str(study.sources_path), "US", str(output)]
    assert score_recheck.main([*arguments, "--output", str(target)]) == 0
    written = json.loads(target.read_text())
    assert written["passed"] and written["coverage"]["calibrated_quantiles"] is False
    assert np.isfinite(written["tolerance"]["rtol"])
