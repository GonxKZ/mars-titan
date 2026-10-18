"""El lanzador separa contextos y conserva resultados sin necesitar una GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_cuda_tests as runner

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_cuda_tests.py"


def launch(tmp_path, code, *options, environment=None):
    project = tmp_path / "project with spaces"
    project.mkdir()
    (project / "pytest.ini").write_text("[pytest]\nmarkers = cuda: fixture del lanzador\n")
    (project / "test_fixture.py").write_text(code)
    output = tmp_path / "receipts with spaces"
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="-1")
    env.update(environment or {})
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), *options, "test_fixture.py"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    path = output / "summary.json"
    return result, json.loads(path.read_text()) if path.is_file() else None, output


def test_collection_and_each_parameter_have_distinct_processes_and_keep_the_hold(tmp_path):
    hold = tmp_path / "hold.json"
    hold.write_text('{"training_allowed":false}')
    result, report, output = launch(
        tmp_path,
        "import os, pathlib, pytest, sys\n"
        "@pytest.mark.cuda\n"
        "@pytest.mark.parametrize('value', [1, 2], ids=['a value/one', 'two'])\n"
        "def test_process(value):\n"
        "    assert 'left_by_previous_case' not in sys.modules\n"
        "    sys.modules['left_by_previous_case'] = object()\n"
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'\n"
        "    assert pathlib.Path(os.environ['MARS_TITAN_TRAINING_HOLD']).read_text() == "
        "'{\"training_allowed\":false}'\n"
        "def test_not_selected():\n"
        "    pytest.fail('No lleva la marca cuda')\n",
        environment={
            "MARS_TITAN_TRAINING_HOLD": str(hold),
            "PYTEST_ADDOPTS": "--invalid-outer-option",
            "PYTEST_PLUGINS": "missing_outer_plugin",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["exit_code"] == 0
    assert len(report["selected"]) == 2
    assert "a value/one" in report["selected"][0]
    assert [case["state"] for case in report["cases"]] == ["passed", "passed"]
    pids = [report["supervisor_pid"], report["collection"]["pid"]]
    pids += [case["pid"] for case in report["cases"]]
    assert len(set(pids)) == 4
    assert report["supervisor_imported_torch"] is False
    assert report["pending"] == []
    assert hold.read_text() == '{"training_allowed":false}'
    for index in range(2):
        directory = output / f"{index:05d}"
        assert (directory / "junit.xml").is_file()
        case = json.loads((directory / "receipt.json").read_text())
        assert case["selected"] == [report["selected"][index]]
        assert [phase["outcome"] for phase in case["phases"]] == ["passed"] * 3


def test_failure_does_not_hide_skip_or_prevent_following_cases(tmp_path):
    result, report, _ = launch(
        tmp_path,
        "import pytest\npytestmark = pytest.mark.cuda\n"
        "def test_failure():\n    assert False, 'fallo esperado'\n"
        "def test_skipped():\n    pytest.skip('motivo explícito')\n"
        "def test_success():\n    assert True\n",
    )
    assert result.returncode == report["exit_code"] == 1
    assert [case["state"] for case in report["cases"]] == ["failed", "skipped", "passed"]
    assert [case["exit_code"] for case in report["cases"]] == [1, 0, 0]
    assert report["pending"] == []


@pytest.mark.parametrize(
    ("code", "exit_code"),
    [("raise RuntimeError('colección rechazada')\n", 2), ("def test_cpu(): pass\n", 5)],
)
def test_collection_error_and_empty_selection_do_not_launch_cases(tmp_path, code, exit_code):
    result, report, _ = launch(tmp_path, code)
    assert result.returncode == report["exit_code"] == exit_code
    assert report["collection"]["exit_code"] == exit_code
    assert report["cases"] == []


def test_timeout_stops_child_and_records_the_unstarted_case(tmp_path):
    result, report, _ = launch(
        tmp_path,
        "import time, pytest\npytestmark = pytest.mark.cuda\n"
        "def test_timeout():\n    time.sleep(20)\n"
        "def test_after():\n    assert True\n",
        "--timeout-seconds",
        "3",
    )
    assert result.returncode == report["exit_code"] == 124
    assert report["cases"][0]["state"] == "timed_out"
    assert report["pending"] == [report["selected"][1]]
    with pytest.raises(ProcessLookupError):
        os.kill(report["cases"][0]["pid"], 0)


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "receipts with spaces"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text('{"existing":true}')
    result, _, _ = launch(tmp_path, "")
    assert result.returncode != 0
    assert marker.read_text() == '{"existing":true}'


def test_signal_to_the_supervisor_stops_its_child_and_keeps_pending_nodes(tmp_path):
    result, report, _ = launch(
        tmp_path,
        "import os, signal, time, pytest\npytestmark = pytest.mark.cuda\n"
        "def test_signal():\n"
        "    os.kill(os.getppid(), signal.SIGTERM)\n    time.sleep(20)\n"
        "def test_after():\n    assert True\n",
    )
    assert result.returncode == report["exit_code"] == 2
    assert report["cases"][0]["state"] == "interrupted"
    assert report["pending"] == [report["selected"][1]]
    with pytest.raises(ProcessLookupError):
        os.kill(report["cases"][0]["pid"], 0)


def test_interrupted_pytest_is_not_relabelled_as_a_test_failure(tmp_path):
    result, report, _ = launch(
        tmp_path,
        "import pytest\npytestmark = pytest.mark.cuda\n"
        "def test_stop():\n    raise KeyboardInterrupt\n"
        "def test_after():\n    assert True\n",
    )
    assert result.returncode == report["exit_code"] == 2
    assert report["cases"][0]["state"] == "interrupted"
    assert report["pending"] == [report["selected"][1]]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "3601"])
def test_invalid_time_budget_does_not_start_collection(tmp_path, value):
    result, report, output = launch(tmp_path, "", "--timeout-seconds", value)
    assert result.returncode == 2
    assert report is None
    assert not output.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"pid": 8},
        {"exit_code": 1},
        {"selected": ["a", "a"]},
        {"selected": [""]},
        {"selected": [1]},
        {"selected": "a"},
    ],
)
def test_receipt_rejects_wrong_process_outcome_or_selection(tmp_path, change):
    receipt = dict(pid=7, exit_code=0, selected=["a"], phases=[])
    receipt.update(change)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        runner._receipt(tmp_path, dict(pid=7, exit_code=0))


def test_missing_teardown_cannot_count_as_passed():
    phases = [dict(nodeid="a", when=when, outcome="passed") for when in ("setup", "call")]
    assert runner._case_state(dict(selected=["a"], phases=phases), "a", 0) == "incomplete"


@pytest.mark.parametrize("fault", ["selection", "phase", "duplicate"])
def test_unexpected_or_duplicate_case_phases_are_rejected(fault):
    phases = [
        dict(nodeid="a", when=when, outcome="passed") for when in ("setup", "call", "teardown")
    ]
    receipt = dict(selected=["a"], phases=phases)
    if fault == "selection":
        receipt["selected"].append("b")
    elif fault == "phase":
        phases[1]["nodeid"] = "b"
    else:
        phases.append(dict(phases[1]))
    with pytest.raises(ValueError):
        runner._case_state(receipt, "a", 0)


def test_an_unresponsive_child_is_killed_after_the_grace_period(monkeypatch):
    signals, waits = [], []

    def wait(**options):
        waits.append(options)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("fixture", 5)

    monkeypatch.setattr(runner.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    runner._stop(SimpleNamespace(pid=123, poll=lambda: None, wait=wait))
    assert signals == [(123, runner.signal.SIGINT), (123, runner.signal.SIGKILL)]
    assert len(waits) == 2


def test_invalid_child_receipt_stops_the_queue_with_an_explicit_error(tmp_path):
    result, report, _ = launch(
        tmp_path,
        "import sys, pytest\npytestmark = pytest.mark.cuda\n"
        "def test_changes_receipt():\n"
        "    sys.modules['scripts.pytest_case_receipt']._receipt['selected'] = ['other']\n"
        "def test_after():\n    assert True\n",
    )
    assert result.returncode == report["exit_code"] == 3
    assert report["cases"][0]["state"] == "invalid_receipt"
    assert "exclusivamente" in report["error"]
    assert report["pending"] == [report["selected"][1]]


def test_interrupt_before_a_child_starts_still_finishes_the_summary(tmp_path, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_suite", interrupt)
    output = tmp_path / "receipt"
    assert runner.main(["--output", str(output)]) == 2
    report = json.loads((output / "summary.json").read_text())
    assert report["exit_code"] == 2
    assert report["cases"] == []
    assert report["finished_at_utc"]


def test_empty_collection_receipt_cannot_report_a_passed_suite(tmp_path, monkeypatch):
    def collect(directory, arguments, timeout):
        directory.mkdir()
        (directory / "receipt.json").write_text(
            json.dumps(dict(pid=7, exit_code=0, selected=[], phases=[]))
        )
        return dict(pid=7, exit_code=0)

    monkeypatch.setattr(runner, "_process", collect)
    summary = dict(cases=[], pending=[])
    assert runner._suite(tmp_path, ["tests"], 600, summary) == 5
    assert summary["cases"] == []


def test_receipt_write_failure_after_spawn_also_stops_the_child(tmp_path, monkeypatch):
    process, stopped = SimpleNamespace(pid=123), []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_stop", stopped.append)

    def fail(*args):
        raise OSError("Fallo al guardar el recibo")

    monkeypatch.setattr(runner, "_write", fail)
    with pytest.raises(OSError, match="recibo"):
        runner._process(tmp_path / "case", ["test_fixture.py"], 600)
    assert stopped == [process]


@pytest.mark.parametrize("phase", ["interrupt", "kill"])
def test_child_exit_between_poll_and_signal_is_reaped(monkeypatch, phase):
    waited = []

    def disappeared(pid, signum):
        if signum == (runner.signal.SIGINT if phase == "interrupt" else runner.signal.SIGKILL):
            raise ProcessLookupError

    def wait(**options):
        waited.append(options)
        if phase == "kill" and len(waited) == 1:
            raise subprocess.TimeoutExpired("fixture", 5)

    monkeypatch.setattr(runner.os, "killpg", disappeared)
    runner._stop(SimpleNamespace(pid=123, poll=lambda: None, wait=wait))
    assert waited
