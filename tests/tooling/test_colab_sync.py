"""La copia del estado publica cada índice después de los archivos que nombra."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("colab_sync", ROOT / "scripts" / "colab_sync.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)

CHECKPOINTS = Path("runs/m/2013/seed-42/fit/checkpoints")


def record(name, size, step):
    return dict(name=name, size=size, global_step=step)


def index(latest, best=None, pinned=()):
    return json.dumps(
        dict(schema_version=1, latest=list(latest), best=best, pinned=list(pinned))
    ).encode()


def make_state(root):
    """Un estado con un trabajo en curso, su punto de recuperación y una entrada empaquetada."""
    (root / "queue").mkdir(parents=True)
    (root / "queue" / "m--2013--seed-42.json").write_text('{"status": "running"}')
    (root / "queue" / "m--2013--seed-42.lock").write_text("")
    folder = root / CHECKPOINTS
    folder.mkdir(parents=True)
    (folder / "state-aaa.pt").write_bytes(b"A" * 10)
    (folder / "state-bbb.pt").write_bytes(b"B" * 12)
    (folder / "state-ccc.pending.pt").write_bytes(b"medio escrito")
    first, second = record("state-aaa.pt", 10, 1), record("state-bbb.pt", 12, 2)
    (folder / "latest.json").write_bytes(index([second, first], best=first))
    (root / "packed-inputs" / "2013").mkdir(parents=True)
    (root / "packed-inputs" / "2013" / "train.arrow").write_bytes(b"x" * 100)
    (root / "authorization.json").write_text("{}")
    return root


def test_push_copies_data_and_indexes_and_leaves_out_what_does_not_travel(tmp_path):
    state, mirror = make_state(tmp_path / "state"), tmp_path / "mirror"
    report = sync.push(state, mirror)
    assert report["copied"] == 3 and report["indexes"] == 2 and report["held"] == []
    assert (mirror / CHECKPOINTS / "latest.json").read_bytes() == (
        state / CHECKPOINTS / "latest.json"
    ).read_bytes()
    assert (mirror / "queue" / "m--2013--seed-42.json").is_file()
    assert not (mirror / "packed-inputs").exists()
    assert not (mirror / "queue" / "m--2013--seed-42.lock").exists()
    assert not (mirror / CHECKPOINTS / "state-ccc.pending.pt").exists()
    assert not [p for p in mirror.rglob("*") if p.name.startswith(sync.TEMPORARY)]
    again = sync.push(state, mirror)
    assert again["copied"] == 0 and again["indexes"] == 0 and again["unchanged"] == 3


def test_an_index_naming_a_state_that_rotated_away_is_held_and_the_old_one_stays(tmp_path):
    state, mirror = make_state(tmp_path / "state"), tmp_path / "mirror"
    sync.push(state, mirror)
    published = (mirror / CHECKPOINTS / "latest.json").read_bytes()
    # El índice nuevo nombra un estado que la rotación borró antes de poder copiarlo.
    third = record("state-ddd.pt", 7, 3)
    (state / CHECKPOINTS / "latest.json").write_bytes(index([third], best=third))
    report = sync.push(state, mirror)
    assert report["held"] == [(CHECKPOINTS / "latest.json").as_posix()]
    assert (mirror / CHECKPOINTS / "latest.json").read_bytes() == published
    assert (mirror / CHECKPOINTS / "state-aaa.pt").is_file()
    # Cuando el estado existe, el índice se publica y los que ya no nombra se retiran.
    (state / CHECKPOINTS / "state-ddd.pt").write_bytes(b"D" * 7)
    report = sync.push(state, mirror)
    assert report["held"] == [] and report["pruned"] == 2
    assert sorted(p.name for p in (mirror / CHECKPOINTS).glob("state-*.pt")) == ["state-ddd.pt"]


def test_a_state_with_another_size_does_not_count_as_copied(tmp_path):
    state, mirror = make_state(tmp_path / "state"), tmp_path / "mirror"
    wrong = record("state-bbb.pt", 99, 2)
    assert not sync.complete(CHECKPOINTS / "latest.json", index([wrong]), state)
    assert sync.complete(CHECKPOINTS / "latest.json", index([record("state-bbb.pt", 12, 2)]), state)
    assert not sync.complete(CHECKPOINTS / "latest.json", b"no es json", state)
    assert not mirror.exists()


def test_pull_restores_an_empty_machine_and_refuses_to_overwrite_other_content(tmp_path):
    state, mirror = make_state(tmp_path / "state"), tmp_path / "mirror"
    sync.push(state, mirror)
    fresh = tmp_path / "fresh"
    report = sync.pull(fresh, mirror)
    assert report["copied"] == 5
    for relative in sync.walk(mirror):
        assert (fresh / relative).read_bytes() == (mirror / relative).read_bytes()
    assert sync.pull(fresh, mirror) == dict(copied=0, bytes=0, unchanged=5)
    (fresh / "authorization.json").write_text('{"otra": 1}')
    with pytest.raises(ValueError, match="otro contenido"):
        sync.pull(fresh, mirror)


def test_pull_rejects_a_mirror_whose_index_names_a_missing_state(tmp_path):
    state, mirror = make_state(tmp_path / "state"), tmp_path / "mirror"
    sync.push(state, mirror)
    (mirror / CHECKPOINTS / "state-bbb.pt").unlink()
    with pytest.raises(ValueError, match="no tiene todos los estados"):
        sync.pull(tmp_path / "fresh", mirror)


def test_symbolic_links_are_rejected(tmp_path):
    state = make_state(tmp_path / "state")
    os.symlink(state / "authorization.json", state / "enlace.json")
    with pytest.raises(ValueError, match="enlace simbólico"):
        sync.push(state, tmp_path / "mirror")


def test_a_file_rewritten_during_the_copy_is_not_kept(tmp_path, monkeypatch):
    state, mirror = make_state(tmp_path / "state"), tmp_path / "mirror"
    target = state / "authorization.json"
    original = sync.write_atomic

    def rewriting(destination, data, mtime_ns=None):
        original(destination, data, mtime_ns)
        if destination.name == "authorization.json":
            target.write_text('{"cambiado": true}')
            os.utime(target, ns=(1, 1))

    monkeypatch.setattr(sync, "write_atomic", rewriting)
    report = sync.push(state, mirror)
    assert report["retried_later"] == ["authorization.json"]
    assert not (mirror / "authorization.json").exists()


def test_the_command_line_reports_and_fails_for_a_missing_state(tmp_path, capsys):
    assert (
        sync.main(["push", "--state", str(tmp_path / "no"), "--mirror", str(tmp_path / "m")]) == 1
    )
    assert "No existe el estado" in capsys.readouterr().err
    state = make_state(tmp_path / "state")
    assert sync.main(["push", "--state", str(state), "--mirror", str(tmp_path / "m")]) == 0
    assert json.loads(capsys.readouterr().out)["copied"] == 3
