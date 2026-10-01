"""Cadencia de publicación y reintentos independientes de la recolección."""

import json
import subprocess

import pytest

from mars_titan.observatory.collector import write_pages
from mars_titan.observatory.publication import GitPublisher, PublicationSchedule, PublicationWorker


def test_terminal_change_is_prioritized_and_failed_delivery_is_not_acknowledged():
    schedule = PublicationSchedule()
    assert schedule.due("a", "running", 0)
    schedule.succeeded("a", "running", 0)
    assert not schedule.due("b", "running", 15)
    assert schedule.due("b", "completed", 15)
    schedule.failed(15)
    assert not schedule.due("b", "completed", 16)
    assert schedule.due("b", "completed", 30)
    schedule.succeeded("b", "completed", 30)
    assert not schedule.due("b", "completed", 500)
    assert schedule.due("c", "completed", 500)


def test_retry_delay_is_bounded():
    schedule = PublicationSchedule()
    for n in range(20):
        schedule.failed(n * 1000)
        assert schedule.retry_at <= n * 1000 + 300


def test_publication_uses_only_data_branch_and_recovers_failed_push(tmp_path):
    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)

    remote, checkout, output = (tmp_path / name for name in ("remote.git", "checkout", "public"))
    checkout.mkdir()
    git(tmp_path, "init", "--bare", str(remote))
    git(checkout, "init", "-b", "observatory-data")
    git(checkout, "config", "user.name", "Prueba local")
    git(checkout, "config", "user.email", "test@example.invalid")
    git(checkout, "remote", "add", "origin", str(remote))
    snapshot = {"schema_version": 2, "project": "MARS-TITAN", "runs": []}
    write_pages(snapshot, output)
    publisher = GitPublisher(checkout, dispatch=False)
    revision = publisher.publish(output)
    assert (
        git(tmp_path, "--git-dir", str(remote), "rev-parse", "observatory-data").stdout.strip()
        == revision
    )
    assert json.loads((checkout / "observatory.json").read_text())["runs"] == []
    write_pages({**snapshot, "notes": ["Cambio público de prueba"]}, output)
    git(checkout, "remote", "set-url", "origin", str(tmp_path / "unavailable.git"))
    with pytest.raises(subprocess.CalledProcessError):
        publisher.publish(output)
    assert (
        git(tmp_path, "--git-dir", str(remote), "rev-parse", "observatory-data").stdout.strip()
        == revision
    )
    git(checkout, "remote", "set-url", "origin", str(remote))
    assert publisher.publish(output) != revision
    git(checkout, "switch", "-c", "develop")
    with pytest.raises(ValueError, match="exclusiva"):
        GitPublisher(checkout, dispatch=False)


def test_slow_network_never_queues_or_blocks_collection(tmp_path):
    import threading
    import time

    release, started = threading.Event(), threading.Event()
    calls = []

    class Publisher:
        def publish(self, output, index):
            calls.append(index)
            started.set()
            assert release.wait(2)
            return "a" * 40

    worker = PublicationWorker(Publisher())
    try:
        snapshot = {"runs": []}
        worker.tick(snapshot, {"value": "first"}, tmp_path, 0)
        assert started.wait(1)
        before = time.monotonic()
        for n in range(100):
            worker.tick(snapshot, {"value": n}, tmp_path, n)
        assert time.monotonic() - before < 0.5
        assert calls == [{"value": "first"}]
        release.set()
        assert worker.future.result(timeout=1) == "a" * 40
        assert worker.tick(snapshot, {}, tmp_path, 101)[0] == "published"
    finally:
        release.set()
        worker.close()


def test_single_publication_failure_is_reported_when_worker_closes(tmp_path):
    class Publisher:
        def publish(self, *_args):
            raise OSError("Fallo de red de prueba")

    worker = PublicationWorker(Publisher())
    worker.tick({"runs": []}, {}, tmp_path, 0)
    with pytest.raises(OSError):
        worker.close()
