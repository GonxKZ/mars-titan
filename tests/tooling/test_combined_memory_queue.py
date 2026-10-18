"""Reanudar intentos pausados sin repetir resultados ni ampliar el límite temporal."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from benchmarks import combined_memory_queue as queue
from mars_titan.training.learning_hold import LearningHoldError


def setup(work, monkeypatch):
    jobs = [dict(id="core", kind="core"), dict(id="reader", kind="readout", parent="core")]
    receipt = dict(jobs=jobs)
    (work / "prepared.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(queue, "verify_prepared", lambda _: receipt)
    monkeypatch.setattr(queue, "remaining_seconds", lambda _: 100)
    return jobs


def save(work, job, status, step):
    folder = work / "runs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(json.dumps(dict(status=status)))
    checkpoint = folder / "fit/checkpoints/latest.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(dict(latest=[dict(sha256=str(step))])))


def test_paused_job_resumes_before_its_dependents(tmp_path, monkeypatch, learning_doubles):
    setup(tmp_path, monkeypatch)
    calls = []

    def launch(work, runtime, job, timeout, log):
        calls.append(job["id"])
        status = "paused" if len(calls) == 1 else "completed"
        save(work, job, status, len(calls))
        return dict(exit_code=0, wall_seconds=0.001, timeout=False)

    result = queue.run_queue(
        tmp_path, tmp_path, datetime.now(UTC) + timedelta(seconds=100), launch=launch
    )
    assert calls == ["core", "core", "reader"]
    assert result["status"] == "completed"
    queue.run_queue(tmp_path, tmp_path, datetime.now(UTC) + timedelta(seconds=100), launch=launch)
    assert calls == ["core", "core", "reader"]


def test_no_progress_does_not_retry_forever(tmp_path, monkeypatch, learning_doubles):
    jobs = setup(tmp_path, monkeypatch)
    save(tmp_path, jobs[0], "paused", 0)
    calls = []

    def launch(*args):
        calls.append(1)
        assert len(calls) == 1
        return dict(exit_code=0, wall_seconds=0.001, timeout=False)

    result = queue.run_queue(
        tmp_path, tmp_path, datetime.now(UTC) + timedelta(seconds=100), launch=launch
    )
    assert calls == [1] and result["status"] == "paused_without_progress"


def test_expired_deadline_never_launches_a_job(tmp_path, monkeypatch, learning_doubles):
    setup(tmp_path, monkeypatch)

    def forbidden(*args):
        raise AssertionError("El plazo ya terminó")

    result = queue.run_queue(
        tmp_path, tmp_path, datetime.now(UTC) - timedelta(seconds=1), launch=forbidden
    )
    assert result["status"] == "budget_exhausted"


def test_queue_obeys_the_hold_before_reading_sources(tmp_path, monkeypatch):
    hold = tmp_path / "hold.json"
    hold.write_text('{"training_allowed":false}')
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(hold))
    with pytest.raises(LearningHoldError):
        queue.run_queue(tmp_path / "absent", tmp_path, datetime.now(UTC))


def test_second_coordinator_cannot_enter_the_same_work(tmp_path, monkeypatch, learning_doubles):
    import fcntl

    setup(tmp_path, monkeypatch)
    with (tmp_path / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="otra|Ya hay"):
            queue.run_queue(tmp_path, tmp_path, datetime.now(UTC) + timedelta(seconds=10))


def test_interrupted_attempt_is_counted_and_releases_the_lock(
    tmp_path, monkeypatch, learning_doubles
):
    setup(tmp_path, monkeypatch)

    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        queue.run_queue(
            tmp_path, tmp_path, datetime.now(UTC) + timedelta(seconds=10), launch=interrupted
        )
    receipts = list((tmp_path / "measurements").glob("*.json"))
    assert len(receipts) == 1
    record = json.loads(receipts[0].read_text())
    assert record["status"] == "interrupted" and record["process_seconds"] >= 0

    def forbidden(*args):
        raise AssertionError("El plazo expirado no puede iniciar otra ejecución")

    result = queue.run_queue(
        tmp_path, tmp_path, datetime.now(UTC) - timedelta(seconds=1), launch=forbidden
    )
    assert result["status"] == "budget_exhausted"


@pytest.mark.parametrize("mode", ["completed", "terminate", "kill"])
def test_child_process_is_reaped_after_deadline(tmp_path, monkeypatch, mode):
    import signal
    import subprocess

    calls, signals = [], []

    class Child:
        pid = 1234567
        returncode = None
        attempts = 0

        def wait(self, timeout=None):
            self.attempts += 1
            if mode != "completed" and self.attempts == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            if mode == "kill" and self.attempts == 2:
                raise subprocess.TimeoutExpired("fixture", timeout)
            if mode == "completed":
                self.returncode = 0
            return self.returncode

        def poll(self):
            return self.returncode

    child = Child()

    def launch(command, **options):
        calls.append((command, options))
        return child

    def kill(pid, value):
        assert pid == child.pid
        signals.append(value)
        if mode == "terminate" or value == signal.SIGKILL:
            child.returncode = -int(value)

    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    monkeypatch.setattr(queue.os, "killpg", kill)
    result = queue.launch_job(tmp_path, tmp_path, dict(id="fixture"), 0.1, tmp_path / "log.txt")
    assert result["timeout"] == (mode != "completed")
    assert result["exit_code"] == (0 if mode == "completed" else -15 if mode == "terminate" else -9)
    assert signals == (
        []
        if mode == "completed"
        else [signal.SIGTERM]
        if mode == "terminate"
        else [signal.SIGTERM, signal.SIGKILL]
    )
    assert calls[0][0][:4] == ["uv", "run", "--no-sync", "python"]
    assert calls[0][1]["start_new_session"] is True


def test_queue_cli_keeps_the_absolute_deadline(tmp_path, monkeypatch, capsys):
    import sys

    calls = []

    def run(work, runtime, deadline):
        calls.append((work, runtime, deadline))
        return dict(status="completed", events=[])

    monkeypatch.setattr(queue, "run_queue", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "--work",
            str(tmp_path),
            "--runtime",
            str(tmp_path),
            "--deadline-utc",
            "2026-10-18T16:14:49+00:00",
        ],
    )
    queue.main()
    assert calls == [(tmp_path, tmp_path, datetime(2026, 10, 18, 16, 14, 49, tzinfo=UTC))]
    assert json.loads(capsys.readouterr().out) == dict(status="completed")
