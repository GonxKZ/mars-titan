"""Pruebas TDD de retención acotada y promoción atómica en `checkpoints.py`."""

import json

import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.training.checkpoints import load_training_state, save_training_state

IDENTITY = {"data": "edition-a", "optimizer": "AdamW", "precision": "float64"}


def save_step(directory, step, **kwargs):
    return save_training_state(
        directory,
        {"global_step": step, "model": {"weight": torch.tensor([float(step)])}},
        identity=IDENTITY,
        **kwargs,
    )


def test_keep_pinned_prunes_obsolete_pinned_checkpoints_atomically(tmp_path):
    p1 = save_step(tmp_path, 1, best=True, pin=True)
    p2 = save_step(tmp_path, 2, pin=True)
    p3 = save_step(tmp_path, 3, pin=True)
    # Al guardar paso 4 solo retenemos p1 y p3 (y el propio p4). p2 debe eliminarse.
    p4 = save_step(tmp_path, 4, pin=True, keep_pinned={sha256(p1), sha256(p3)})
    # Paso 5 y 6 empujan p2 y p3 fuera de `latest` (que conserva 2).
    p5 = save_step(tmp_path, 5, keep_pinned={sha256(p1), sha256(p3), sha256(p4)})
    p6 = save_step(tmp_path, 6, keep_pinned={sha256(p1), sha256(p3), sha256(p4)})

    assert p1.is_file() and p3.is_file() and p4.is_file() and p5.is_file() and p6.is_file()
    assert not p2.exists()
    index = json.loads((tmp_path / "latest.json").read_text())
    pinned_digests = {item["sha256"] for item in index["pinned"]}
    assert pinned_digests == {sha256(p1), sha256(p3), sha256(p4)}


def test_bounded_keep_pinned_survives_more_than_64_pinned_saves(tmp_path):
    kept = []
    for step in range(1, 75):
        keep_set = {sha256(path) for path in kept[-3:]}
        path = save_step(tmp_path, step, pin=True, keep_pinned=keep_set)
        kept.append(path)
    index = json.loads((tmp_path / "latest.json").read_text())
    assert len(index["pinned"]) == 4
    files = list(tmp_path.glob("state-*.pt"))
    assert len(files) <= 6


def test_best_sha256_atomically_promotes_an_existing_pinned_checkpoint(tmp_path):
    p1 = save_step(tmp_path, 1, best=True, pin=True)
    p2 = save_step(tmp_path, 2, best=True, pin=True)
    d1, d2 = sha256(p1), sha256(p2)
    # En paso 3 el suavizado retroactivo vuelve a elegir p1 como mejor estado.
    p3 = save_step(tmp_path, 3, pin=True, keep_pinned={d1, d2}, best_sha256=d1)
    assert p3.is_file()
    index = json.loads((tmp_path / "latest.json").read_text())
    assert index["best"]["sha256"] == d1
    best_state = load_training_state(tmp_path, expected_identity=IDENTITY, selection="best")
    assert best_state["global_step"] == 1
    assert best_state["model"]["weight"].item() == 1.0


def test_best_sha256_rejects_unknown_or_corrupt_digest_without_modifying_index(tmp_path):
    p1 = save_step(tmp_path, 1, best=True, pin=True)
    d1 = sha256(p1)
    before = (tmp_path / "latest.json").read_text()
    with pytest.raises(ValueError, match="fijado|identidad"):
        save_step(tmp_path, 2, best_sha256="0" * 64)
    assert (tmp_path / "latest.json").read_text() == before
    with pytest.raises(ValueError, match="mutuamente|best"):
        save_step(tmp_path, 2, best=True, best_sha256=d1)
    assert (tmp_path / "latest.json").read_text() == before
