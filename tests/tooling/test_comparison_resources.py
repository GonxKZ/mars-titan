"""Admisión por trabajo y cierre de sus procesos sin cambiar el experimento."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_comparison_resources.py"


@pytest.fixture
def engine():
    path = Path(os.environ.get("COMPARISON_RESOURCES_TEST_SOURCE", SCRIPT))
    if not path.is_file():
        pytest.fail("Falta el lanzador que reserva RAM por trabajo")
    spec = importlib.util.spec_from_file_location("comparison_resources_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("runner", ["fit", "reader", "ridge", "rule"])
def test_neural_and_small_controls_do_not_reserve_lightgbm_memory(engine, runner):
    assert engine.memory_limit(runner) == 8


def test_lightgbm_keeps_its_measured_memory_margin(engine):
    assert engine.memory_limit("tabular") == 12
    with pytest.raises(ValueError):
        engine.memory_limit("unknown")


def test_available_memory_uses_reclaimable_memory_and_rejects_missing_data(engine, tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemFree: 123 kB\nMemAvailable: 12582912 kB\n")
    assert engine.available_memory(path) == 12 * 1024**3
    path.write_text("MemFree: 123 kB\n")
    with pytest.raises(ValueError):
        engine.available_memory(path)


def test_admission_keeps_four_gib_for_the_system(engine):
    assert not engine.memory_ready(8, 12 * 1024**3 - 1)
    assert engine.memory_ready(8, 12 * 1024**3)
    assert not engine.memory_ready(12, 15 * 1024**3)
    assert engine.memory_ready(12, 16 * 1024**3)


@pytest.mark.parametrize("limit,available", [(0, 10), (8, -1), (8, 1.5)])
def test_invalid_admission_values_fail_closed(engine, limit, available):
    with pytest.raises(ValueError):
        engine.memory_ready(limit, available)


def test_running_status_requires_an_owned_process(engine, tmp_path):
    path = tmp_path / "record.json"
    assert engine.running_pid(path, {}) is None
    path.write_text(json.dumps(dict(status="running", attempts=[dict(pid=123)])))
    assert engine.running_pid(path, {123: 1}) == 123
    assert engine.running_pid(path, {124: 1}) is None
    path.write_text(json.dumps(dict(status="completed", attempts=[dict(pid=123)])))
    assert engine.running_pid(path, {123: 1}) is None
    with path.open("wb") as stream:
        stream.truncate(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match="límite"):
        engine.running_pid(path, {})


@pytest.fixture
def launcher(tmp_path):
    path = tmp_path / "resources"
    path.write_text('#!/bin/sh\nshift 4\nexec "$@"\n')
    path.chmod(0o700)
    return path


def run_worker(engine, tmp_path, launcher, code, *, stop=None, probe=None):
    from mars_titan.training.gpu_supervisor import StopFlag

    return engine.run_worker(
        launcher,
        8,
        [sys.executable, "-c", code],
        status=tmp_path / "status.json",
        record=tmp_path / "record.json",
        job="fixture/2021/seed-42",
        stop=stop or StopFlag(),
        probe=probe or (lambda: 20 * 1024**3),
        poll_seconds=0.02,
        stop_timeout=3,
    )


def test_child_failure_is_not_reported_as_success(engine, tmp_path, launcher):
    assert run_worker(engine, tmp_path, launcher, "raise SystemExit(7)") == 7
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase"] == "failed" and status["returncode"] == 7


@pytest.mark.parametrize("choice", ["default", "custom", "disabled"])
def test_worker_shares_file_digests_within_the_state(
    engine, tmp_path, launcher, monkeypatch, choice
):
    name = "MARS_TITAN_DIGEST_CACHE"
    if choice == "default":
        monkeypatch.delenv(name, raising=False)
        expected = str(tmp_path / "file-digests.json")
    else:
        expected = "" if choice == "disabled" else str(tmp_path / "custom.json")
        monkeypatch.setenv(name, expected)
    output = tmp_path / "child-environment.json"
    code = (
        "import json, os, pathlib\n"
        f"pathlib.Path({str(output)!r}).write_text(json.dumps(os.environ.get({name!r})))"
    )
    assert run_worker(engine, tmp_path, launcher, code) == 0
    assert json.loads(output.read_text()) == expected


def test_worker_runs_after_memory_becomes_available(engine, tmp_path, launcher):
    readings = iter([11 * 1024**3, 12 * 1024**3])
    output = tmp_path / "executed"
    code = f"from pathlib import Path\nPath({str(output)!r}).write_text('yes')"
    result = run_worker(
        engine, tmp_path, launcher, code, probe=lambda: next(readings, 12 * 1024**3)
    )
    assert result == 0 and output.read_text() == "yes"
    assert json.loads((tmp_path / "status.json").read_text())["limit_gib"] == 8


def test_stop_while_waiting_never_launches_a_job(engine, tmp_path, launcher, monkeypatch):
    from mars_titan.training.gpu_supervisor import StopFlag

    stop = StopFlag()
    output = tmp_path / "unexpected"
    launched = []
    original = engine.subprocess.Popen

    def observe_launch(*args, **kwargs):
        child = original(*args, **kwargs)
        launched.append(child.pid)
        return child

    monkeypatch.setattr(engine.subprocess, "Popen", observe_launch)

    def probe():
        stop.request()
        return 2 * 1024**3

    code = f"from pathlib import Path\nPath({str(output)!r}).touch()"
    assert run_worker(engine, tmp_path, launcher, code, stop=stop, probe=probe) == 3
    assert launched == []
    assert not output.exists()
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "paused"


def test_stop_confirms_checkpoint_before_returning(engine, tmp_path, launcher):
    from mars_titan.training.gpu_supervisor import StopFlag

    ready, checkpoint = tmp_path / "ready", tmp_path / "checkpoint"
    code = f"""
import signal,time
from pathlib import Path
def stop(*_):
    Path({str(checkpoint)!r}).write_text('confirmed')
    raise SystemExit(3)
signal.signal(signal.SIGTERM, stop)
Path({str(ready)!r}).touch()
while True: time.sleep(.01)
"""
    stop = StopFlag()

    def request():
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.request()

    thread = threading.Thread(target=request)
    thread.start()
    try:
        assert run_worker(engine, tmp_path, launcher, code, stop=stop) == 3
    finally:
        thread.join(timeout=6)
    assert checkpoint.read_text() == "confirmed"


def test_parent_exit_stops_the_job_and_confirms_its_state(engine, tmp_path, launcher, monkeypatch):
    ready, checkpoint = tmp_path / "ready", tmp_path / "checkpoint"
    code = f"""
import signal,time
from pathlib import Path
def stop(*_):
    Path({str(checkpoint)!r}).write_text('confirmed')
    raise SystemExit(3)
signal.signal(signal.SIGTERM, stop)
Path({str(ready)!r}).touch()
while True: time.sleep(.01)
"""
    monkeypatch.setattr(engine.os, "getppid", lambda: 20 if ready.exists() else 10)
    assert run_worker(engine, tmp_path, launcher, code) == 3
    assert checkpoint.read_text() == "confirmed"


@pytest.mark.parametrize("runner,limit", [("fit", "8"), ("reader", "8"), ("tabular", "12")])
def test_queue_wraps_the_original_command_without_changing_its_arguments(
    engine, tmp_path, runner, limit
):
    original = [sys.executable, "-m", "mars_titan.training.comparison_campaign", "job"]

    class Queue:
        def __init__(self, *args, **kwargs):
            assert kwargs["workers"] == 1

        def _command(self, job, *, preflight=False):
            return original + (["--preflight"] if preflight else [])

    layout = SimpleNamespace(state=tmp_path, record=lambda key: tmp_path / "record.json")
    campaign = SimpleNamespace(
        load_declaration=lambda path: ({}, "digest"),
        Layout=lambda *args: layout,
        code_commit=lambda: "commit",
        code_changes=lambda: None,
        authorized=lambda *args, **kwargs: True,
        Queue=Queue,
    )
    args = SimpleNamespace(
        declaration=None,
        state=None,
        views=None,
        machine="references",
        runtime=tmp_path / "runtime",
        resource_launcher=tmp_path / "resources",
    )
    queue = engine.make_queue(campaign, args)
    command = queue._command(SimpleNamespace(runner=runner, key="job"), preflight=True)
    assert command[command.index("--limit-gib") + 1] == limit
    assert command[command.index("--") + 1 :] == [*original, "--preflight"]
    assert command[command.index("--runtime") + 1] == str(args.runtime)


def test_authorization_is_checked_before_constructing_the_queue(engine, tmp_path):
    calls = []
    campaign = SimpleNamespace(
        load_declaration=lambda path: ({}, "digest"),
        Layout=lambda *args: SimpleNamespace(state=tmp_path),
        code_commit=lambda: "commit",
        code_changes=lambda: None,
        authorized=lambda *args, **kwargs: False,
        Queue=lambda *args, **kwargs: calls.append(True),
    )
    args = SimpleNamespace(declaration=None, state=None, views=None, machine="references")
    with pytest.raises(ValueError, match="permiso"):
        engine.make_queue(campaign, args)
    assert calls == []


def test_missing_runtime_fails_before_importing_training_code(engine, tmp_path):
    with pytest.raises(ValueError, match="runtime"):
        engine.load_campaign(tmp_path)


def test_non_executable_resource_launcher_is_rejected_before_the_queue(
    engine, tmp_path, monkeypatch
):
    launcher = tmp_path / "not-executable"
    launcher.write_text("contenido")
    monkeypatch.setattr(engine, "load_campaign", lambda runtime: SimpleNamespace())
    monkeypatch.setattr(
        engine, "make_queue", lambda *args: pytest.fail("Se construyó la cola sin ejecutable")
    )
    with pytest.raises(ValueError, match="ejecutable"):
        engine.main(
            [
                "--runtime",
                str(tmp_path),
                "--resource-launcher",
                str(launcher),
                "--machine",
                "references",
            ]
        )


def test_imported_controller_must_belong_to_the_requested_runtime(engine, tmp_path, monkeypatch):
    path = tmp_path / "src/mars_titan/training/comparison_campaign.py"
    path.parent.mkdir(parents=True)
    path.write_text('"""Runtime de la prueba."""\n')
    monkeypatch.setattr(engine.sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(
        engine.importlib, "import_module", lambda name: SimpleNamespace(__file__=__file__)
    )
    with pytest.raises(ValueError, match="distinto"):
        engine.load_campaign(tmp_path)


def test_cli_help_does_not_import_a_scientific_runtime():
    if not SCRIPT.is_file():
        pytest.fail("Falta el lanzador que reserva RAM por trabajo")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
    assert "--runtime" in result.stdout


@pytest.mark.parametrize("completed", [True, False])
def test_cli_only_exports_a_complete_queue_and_resolves_the_destination(
    engine, tmp_path, launcher, monkeypatch, completed
):
    exports = []
    queue = SimpleNamespace(
        run=lambda: {},
        request_stop=lambda *args: None,
        layout=SimpleNamespace(state=tmp_path / "state"),
        jobs=["job"],
    )
    campaign = SimpleNamespace(
        _print_summary=lambda result: None,
        job_state=lambda layout, job: "completed" if completed else "paused",
        DONE="completed",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "load_campaign", lambda runtime: campaign)
    monkeypatch.setattr(engine, "make_queue", lambda campaign, args: queue)
    monkeypatch.setattr(engine.signal, "signal", lambda *args: None)

    def export(command, **kwargs):
        exports.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(engine.subprocess, "run", export)
    result = engine.main(
        [
            "--runtime",
            "runtime",
            "--resource-launcher",
            str(launcher),
            "--machine",
            "references",
            "--results",
            "result.zip",
        ]
    )
    assert result == (0 if completed else 3)
    assert len(exports) == int(completed)
    if completed:
        command, kwargs = exports[0]
        assert command[command.index("--destination") + 1] == str(tmp_path / "result.zip")
        assert command[command.index("--state") + 1] == str(tmp_path / "state")
        assert kwargs["cwd"] == tmp_path / "runtime"
