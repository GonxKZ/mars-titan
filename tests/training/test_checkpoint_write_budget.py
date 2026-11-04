"""El archivo temporal respeta los límites aunque los metadatos superen la estimación."""

import errno
import io
from types import SimpleNamespace

import pytest
import torch

from mars_titan.training import checkpoints

MIB = 1024**2
IDENTITY = {"fixture": "metadata-write-budget"}


def metadata_state(step):
    empty = torch.empty(0)
    return {"global_step": step, "views": [empty[:0] for _ in range(40_000)]}


def observed_sizes(monkeypatch, directory):
    original = torch.save
    sizes = []

    def save(payload, stream):
        try:
            return original(payload, stream)
        finally:
            stream.flush()
            sizes.append((directory / ".pending.pt").stat().st_size)

    monkeypatch.setattr(torch, "save", save)
    return sizes


def test_metadata_cannot_consume_the_disk_reserve(tmp_path, monkeypatch):
    previous = checkpoints.save_training_state(tmp_path, {"global_step": 0}, identity=IDENTITY)
    index = (tmp_path / "latest.json").read_bytes()
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda _: SimpleNamespace(free=18 * MIB))
    sizes = observed_sizes(monkeypatch, tmp_path)
    with pytest.raises(OSError) as error:
        checkpoints.save_training_state(tmp_path, metadata_state(1), identity=IDENTITY)
    assert error.value.errno == errno.ENOSPC
    assert sizes and max(sizes) <= 2 * MIB
    assert previous.is_file() and (tmp_path / "latest.json").read_bytes() == index
    assert not (tmp_path / ".pending.pt").exists()
    assert checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["global_step"] == 0


def test_serialized_limit_is_enforced_before_the_pending_file_grows_past_it(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "MAX_STATE_BYTES", 2 * MIB)
    sizes = observed_sizes(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="presupuesto"):
        checkpoints.save_training_state(tmp_path, metadata_state(1), identity=IDENTITY)
    assert sizes and max(sizes) <= 2 * MIB
    assert not (tmp_path / ".pending.pt").exists() and not (tmp_path / "latest.json").exists()


def test_metadata_larger_than_the_estimate_can_be_saved_when_it_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints.shutil, "disk_usage", lambda _: SimpleNamespace(free=24 * MIB))
    path = checkpoints.save_training_state(tmp_path, metadata_state(1), identity=IDENTITY)
    assert 2 * MIB < path.stat().st_size < 3 * MIB
    state = checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)
    assert state["global_step"] == 1 and len(state["views"]) == 40_000
    assert all(value.numel() == 0 for value in state["views"])


def test_bounded_writer_preserves_the_existing_torch_archive(tmp_path):
    shared = torch.arange(8, dtype=torch.float32)
    state = {"global_step": 1, "views": [shared[1::2], shared[-1:]], "metadata": (None, False, 3.5)}
    expected = io.BytesIO()
    torch.save({"schema_version": 1, "identity": IDENTITY, "state": state}, expected)
    path = checkpoints.save_training_state(tmp_path, state, identity=IDENTITY)
    assert path.read_bytes() == expected.getvalue()
