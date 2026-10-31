"""CLI del techo de señal (#747), con la declaración real y los orígenes sustituidos."""

import json
import runpy
from pathlib import Path

import pytest

from mars_titan.evaluation import signal_ceiling_report
from mars_titan.training import screening_cohorts, signal_ceiling
from mars_titan.training.learning_hold import LearningHoldError
from tests.evaluation.test_signal_ceiling_report import origin

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_signal_ceiling.py"
DECLARATION = json.loads((ROOT / "configs/experiments/signal-ceiling-v1.json").read_text())


def cli():
    return runpy.run_path(str(SCRIPT), run_name="signal_ceiling_cli_test")["main"]


def test_without_run_the_cli_checks_the_declaration_and_prints_its_plan(capsys, tmp_path):
    assert cli()([]) == 0
    assert json.loads(capsys.readouterr().out) == dict(
        origins=DECLARATION["screening"]["origins"],
        models=["lightgbm", "ridge"],
        targets=["raw_open_close", "residual"],
    )
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(dict(DECLARATION, final_test_access=True)))
    with pytest.raises(ValueError, match="test final cerrado"):
        cli()(["--declaration", str(path)])
    assert capsys.readouterr().out == ""


def test_a_run_needs_cohorts_and_output_and_the_hold_stops_it_before_reading(tmp_path):
    with pytest.raises(SystemExit, match="--cohorts y --output"):
        cli()(["--run", "--output", str(tmp_path / "out")])
    # La declaración no existe: la protección se comprueba antes de leerla.
    arguments = ["--run", "--declaration", str(tmp_path / "absent.json")]
    arguments += ["--cohorts", str(tmp_path / "cohorts"), "--output", str(tmp_path / "out")]
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        cli()(arguments)
    assert not (tmp_path / "out").exists()


def test_a_run_verifies_the_cohorts_measures_each_origin_and_writes_the_report(
    tmp_path, monkeypatch, learning_doubles, capsys
):
    calls = []
    monkeypatch.setattr(screening_cohorts, "verify", lambda work: calls.append(("verify", work)))

    def run_origin(view, chosen, declaration, output):
        assert declaration == DECLARATION
        calls.append(("origin", view, chosen, output))
        return origin(chosen["year"])

    monkeypatch.setattr(signal_ceiling, "run_origin", run_origin)
    cohorts, output = tmp_path / "cohorts", tmp_path / "out"
    assert cli()(["--run", "--cohorts", str(cohorts), "--output", str(output)]) == 0
    origins = DECLARATION["screening"]["origins"]
    assert calls == [("verify", cohorts)] + [
        ("origin", cohorts / str(item["year"]) / "manifest.json", item, output / "origins")
        for item in origins
    ]
    expected = signal_ceiling_report.report(
        [origin(item["year"]) for item in origins], DECLARATION["screening"]["inference"]
    )
    assert json.loads((output / "report.json").read_text()) == expected
    assert json.loads(capsys.readouterr().out) == dict(status="completed", output=str(output))
