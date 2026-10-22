"""Exclusión entre publicadores sobre checkouts temporales sin remoto."""

import json
import select
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from mars_titan.observatory import publication
from mars_titan.observatory.collector import write_pages
from mars_titan.observatory.publication import GitPublisher


def git(checkout, *args):
    return subprocess.run(
        ["git", *args], cwd=checkout, capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()


def data_checkout(path):
    path.mkdir()
    git(path, "init", "-b", "observatory-data")
    git(path, "config", "user.name", "Prueba local")
    git(path, "config", "user.email", "test@example.invalid")
    return path


@pytest.fixture
def checkout(tmp_path):
    return data_checkout(tmp_path / "checkout")


@pytest.fixture
def outputs(tmp_path):
    paths = tuple(tmp_path / name for name in ("first-output", "second-output"))
    for number, path in enumerate(paths):
        write_pages(
            dict(schema_version=2, project="MARS-TITAN", runs=[], notes=[f"Fixture {number}"]),
            path,
        )
    return paths


@pytest.fixture
def publisher(monkeypatch):
    def create(path, *, dispatch=False):
        instance = GitPublisher(path, dispatch=dispatch)
        original = instance.git

        def local_git(*args):
            # La única frontera sustituida es el envío. El repositorio no tiene remoto.
            return "" if args[0] == "push" else original(*args)

        monkeypatch.setattr(instance, "git", local_git)
        return instance

    return create


@pytest.mark.parametrize("alias", ["same", "symlink", "dotdot", "same-instance"])
@pytest.mark.parametrize("phase", ["write", "push", "dispatch"])
def test_competing_publication_is_rejected_before_writes(
    checkout, outputs, publisher, monkeypatch, tmp_path, alias, phase
):
    other_path = checkout
    if alias == "symlink":
        other_path = tmp_path / "checkout-alias"
        other_path.symlink_to(checkout, target_is_directory=True)
    elif alias == "dotdot":
        (tmp_path / "nested").mkdir()
        other_path = tmp_path / "nested" / ".." / "checkout"
    first = publisher(checkout, dispatch=phase == "dispatch")
    second = first if alias == "same-instance" else publisher(other_path)
    started, release = Event(), Event()

    def pause():
        if not started.is_set():
            started.set()
            assert release.wait(10)

    atomic_json, run, first_git = publication.atomic_json, subprocess.run, first.git

    def write(path, body):
        if phase == "write":
            pause()
        return atomic_json(path, body)

    def publish_git(*args):
        if args[0] == "push" and phase == "push":
            pause()
        return first_git(*args)

    def dispatch(command, **kwargs):
        if command[0] == "gh":
            pause()
            return subprocess.CompletedProcess(command, 0, "", "")
        return run(command, **kwargs)

    monkeypatch.setattr(publication, "atomic_json", write)
    monkeypatch.setattr(first, "git", publish_git)
    monkeypatch.setattr(subprocess, "run", dispatch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first.publish, outputs[0])
        try:
            assert started.wait(5)
            index_path = checkout / "observatory.json"
            before = index_path.read_bytes() if index_path.exists() else None
            with pytest.raises(BlockingIOError, match="publicador"):
                second.publish(outputs[1])
            assert (index_path.read_bytes() if index_path.exists() else None) == before
        finally:
            release.set()
        first_revision = future.result(timeout=5)
    assert second.publish(outputs[1]) != first_revision
    assert json.loads((checkout / "observatory.json").read_text())["notes"] == ["Fixture 1"]
    assert git(checkout, "ls-files").splitlines() == ["observatory.json"]
    assert git(checkout, "status", "--porcelain") == ""


def test_independent_process_cannot_enter_the_same_checkout(
    checkout, outputs, publisher, source_environment
):
    script = """
import sys
from mars_titan.observatory.publication import GitPublisher
p = GitPublisher(sys.argv[1], dispatch=False)
original = p.git
def git(*args):
    if args[0] == "push":
        print("ready", flush=True)
        assert sys.stdin.readline() == "release\\n"
        return ""
    return original(*args)
p.git = git
p.publish(sys.argv[2])
"""
    second = publisher(checkout)
    with subprocess.Popen(
        [sys.executable, "-c", script, str(checkout), str(outputs[0])],
        env=source_environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            assert select.select([process.stdout], [], [], 5)[0]
            assert process.stdout.readline() == "ready\n"
            with pytest.raises(BlockingIOError, match="publicador"):
                second.publish(outputs[1])
        finally:
            _, errors = process.communicate(input="release\n", timeout=5)
        assert process.returncode == 0, errors
    assert second.publish(outputs[1]) == git(checkout, "rev-parse", "HEAD")


@pytest.mark.parametrize("phase", ["validation", "write", "commit", "push", "dispatch"])
def test_failure_releases_checkout_for_a_later_publisher(
    checkout, outputs, publisher, monkeypatch, phase
):
    first = publisher(checkout, dispatch=phase == "dispatch")
    with monkeypatch.context() as patch:
        original_git = first.git

        def fail_git(*args):
            if args[0] == phase:
                raise subprocess.CalledProcessError(1, ["git", *args])
            return original_git(*args)

        def fail_write(*_args):
            raise OSError("Fallo de escritura de prueba")

        run = subprocess.run

        def fail_dispatch(command, **kwargs):
            if command[0] == "gh":
                raise subprocess.CalledProcessError(1, command)
            return run(command, **kwargs)

        patch.setattr(first, "git", fail_git)
        if phase == "write":
            patch.setattr(publication, "atomic_json", fail_write)
        patch.setattr(subprocess, "run", fail_dispatch)
        invalid = {"schema_version": 1} if phase == "validation" else None
        error = ValueError if phase == "validation" else OSError
        if phase in {"commit", "push", "dispatch"}:
            error = subprocess.CalledProcessError
        with pytest.raises(error):
            first.publish(outputs[0], invalid)
    second = publisher(checkout)
    assert second.publish(outputs[1]) == git(checkout, "rev-parse", "HEAD")
    assert json.loads((checkout / "observatory.json").read_text())["notes"] == ["Fixture 1"]
    assert git(checkout, "ls-files").splitlines() == ["observatory.json"]


def test_unrelated_checkout_can_publish_while_another_is_reserved(
    checkout, outputs, publisher, monkeypatch, tmp_path
):
    independent = data_checkout(tmp_path / "independent")
    first, second = publisher(checkout), publisher(independent)
    started, release = Event(), Event()
    original_git = first.git

    def pause_push(*args):
        if args[0] == "push":
            started.set()
            assert release.wait(10)
        return original_git(*args)

    monkeypatch.setattr(first, "git", pause_push)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first.publish, outputs[0])
        try:
            assert started.wait(5)
            second_revision = second.publish(outputs[1])
            assert json.loads((independent / "observatory.json").read_text())["notes"] == [
                "Fixture 1"
            ]
        finally:
            release.set()
        assert future.result(timeout=5) != second_revision


def test_linked_worktree_keeps_its_lock_outside_the_versioned_tree(
    checkout, outputs, publisher, tmp_path
):
    publisher(checkout).publish(outputs[0])
    git(checkout, "switch", "-c", "fixture-source")
    linked = tmp_path / "data-worktree"
    git(checkout, "worktree", "add", str(linked), "observatory-data")
    assert (linked / ".git").is_file()
    first = publisher(linked)
    first.publish(outputs[1])
    lock_path = Path(git(linked, "rev-parse", "--absolute-git-dir")) / "observatory-publisher.lock"
    identity = lock_path.stat().st_ino
    with pytest.raises(FileNotFoundError):
        first.publish(tmp_path / "absent-output")
    publisher(linked).publish(outputs[0])
    assert lock_path.stat().st_ino == identity
    assert git(linked, "ls-files").splitlines() == ["observatory.json"]
    assert git(linked, "status", "--porcelain") == ""
