"""Artefactos de tensores sin grafo y referencias previas a la publicación de sesión."""

import os

import pytest
import torch

from mars_titan.memory.native_backend import load_native
from mars_titan.memory.session_artifacts import SessionArtifacts


@pytest.fixture
def store(tmp_path):
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip("Falta el enlace nativo CPU compilado")
    return SessionArtifacts(load_native(path), tmp_path / "artifacts", max_bytes=512 * 1024)


def test_staging_is_deterministic_and_does_not_publish_a_head(store):
    payload = dict(weights=torch.arange(16, dtype=torch.float32).reshape(4, 4), count=3)
    first = store.stage(payload, identity="a" * 64, kind="fast")
    assert store.stage(payload, identity="a" * 64, kind="fast") == first
    recovered = store.read(first, identity="a" * 64, kind="fast")
    assert torch.equal(recovered["weights"], payload["weights"])
    assert not recovered["weights"].requires_grad
    assert not (store.directory / "latest.json").exists()
    assert len(tuple(store.directory.iterdir())) == 1


def test_corrupt_bytes_and_incompatible_reference_are_rejected(store):
    reference = store.stage(dict(value=torch.ones(2)), identity="a" * 64, kind="pending")
    with pytest.raises(ValueError):
        store.read(reference, identity="b" * 64, kind="pending")
    with pytest.raises(ValueError):
        store.read(dict(reference, schema_version=True), identity="a" * 64, kind="pending")
    with pytest.raises(ValueError):
        store.read(
            dict(reference, bytes=float(reference["bytes"])), identity="a" * 64, kind="pending"
        )
    (store.directory / reference["name"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        store.read(reference, identity="a" * 64, kind="pending")


def test_graphs_oversized_storage_and_nonfinite_payloads_fail_before_writing(store):
    for value in (
        torch.ones(2, requires_grad=True),
        torch.tensor([float("inf")]),
        torch.zeros(200_000),
        torch.arange(10, dtype=torch.float32)[2:4],
    ):
        with pytest.raises(ValueError):
            store.stage(dict(value=value), identity="a" * 64, kind="fast")
    assert not store.directory.exists()


def test_pruning_keeps_all_declared_live_references_and_rejects_missing_ones(store):
    refs = [store.stage(dict(value=i), identity="a" * 64, kind="fixture") for i in range(3)]
    store.prune_unreferenced(refs[:2])
    assert sorted(p.name for p in store.directory.iterdir()) == sorted(r["name"] for r in refs[:2])
    with pytest.raises(ValueError):
        store.prune_unreferenced(refs)
    assert all((store.directory / ref["name"]).exists() for ref in refs[:2])


def test_another_valid_archive_cannot_replace_the_referenced_bytes(store):
    first = store.stage(dict(value=torch.ones(8)), identity="a" * 64, kind="fast")
    other = store.stage(dict(value=torch.zeros(8)), identity="a" * 64, kind="fast")
    assert first["bytes"] == other["bytes"]
    (store.directory / first["name"]).write_bytes((store.directory / other["name"]).read_bytes())
    with pytest.raises(ValueError, match="cambiado"):
        store.read(first, identity="a" * 64, kind="fast")
