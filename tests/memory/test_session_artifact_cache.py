"""Reutilizar el control validado sin compartir estado mutable ni omitir el disco."""

import os

import pytest
import torch

from mars_titan.memory.native_backend import load_native
from mars_titan.memory.session_artifacts import SessionArtifacts
from tests.suite_support import missing_native

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
IDENTITY = "a" * 64


@pytest.fixture
def store(tmp_path):
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip(missing_native("MARS_TITAN_EPISODIC_NATIVE"))
    result = SessionArtifacts(
        load_native(path),
        tmp_path / "artifacts",
        max_bytes=512 * 1024,
    )
    result.max_cached_bytes = 65536
    return result


@pytest.fixture
def loads(monkeypatch):
    calls = []
    original = torch.load

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted)
    return calls


def read(store, reference):
    return store.read(reference, identity=IDENTITY, kind="control")


def test_cached_control_keeps_binary64_types_and_independent_mutable_values(store, loads):
    payload = {3: (b"bytes", -0.0), "rows": [[1.0000000000000002, 7]], "tensor": torch.ones(2)}
    reference = store.stage(payload, identity=IDENTITY, kind="control")
    first = read(store, reference)
    first["rows"][0][0] = 99.0
    first["tensor"][0] = 20
    payload["rows"][0][1] = 30
    second = read(store, reference)
    assert second[3] == (b"bytes", -0.0)
    assert type(second[3]) is tuple
    assert second[3][1].hex() == "-0x0.0p+0"
    assert second["rows"] == [[1.0000000000000002, 7]]
    assert torch.equal(second["tensor"], torch.ones(2))
    second["tensor"][1] = 50
    assert torch.equal(read(store, reference)["tensor"], torch.ones(2))
    assert len(loads) == 1


def test_cache_checks_file_content_after_each_hit_even_with_restored_mtime(store, loads):
    first = store.stage({"value": 1.0}, identity=IDENTITY, kind="control")
    other = store.stage({"value": 2.0}, identity=IDENTITY, kind="control")
    assert read(store, first) == {"value": 1.0}
    assert read(store, first) == {"value": 1.0}
    path = store.directory / first["name"]
    stamp = path.stat()
    replacement = (store.directory / other["name"]).read_bytes()
    assert len(replacement) == stamp.st_size
    path.write_bytes(replacement)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="cambiado"):
        read(store, first)
    assert len(loads) == 1


@pytest.mark.parametrize("changes", [{"identity": "b" * 64}, {"kind": "bank"}, {"bytes": True}])
def test_a_cached_payload_does_not_authorize_an_incompatible_reference(store, changes):
    reference = store.stage({"value": 3}, identity=IDENTITY, kind="control")
    read(store, reference)
    with pytest.raises(ValueError):
        read(store, dict(reference, **changes))


def test_reducing_artifact_budget_revalidates_a_previously_cached_control(store):
    reference = store.stage({"rows": [0.0] * 3000}, identity=IDENTITY, kind="control")
    store.max_cached_bytes = 256 * 1024
    read(store, reference)
    store.max_bytes = 65536
    with pytest.raises(ValueError, match="presupuesto"):
        read(store, reference)


def test_oversized_payload_is_not_cached_and_keeps_the_previous_small_control(store, loads):
    small = store.stage({"value": 1}, identity=IDENTITY, kind="control")
    large = store.stage({"rows": [0.0] * 3000}, identity=IDENTITY, kind="control")
    assert read(store, small) == {"value": 1}
    assert len(read(store, large)["rows"]) == 3000
    assert len(read(store, large)["rows"]) == 3000
    assert read(store, small) == {"value": 1}
    assert len(loads) == 3


def test_cache_retains_only_one_control_and_never_reuses_other_artifact_kinds(store, loads):
    first = store.stage({"value": 1}, identity=IDENTITY, kind="control")
    second = store.stage({"value": 2}, identity=IDENTITY, kind="control")
    bank = store.stage({"value": 3}, identity=IDENTITY, kind="bank")
    assert read(store, first)["value"] == 1
    assert len(loads) == 1
    assert read(store, second)["value"] == 2
    assert len(loads) == 2
    assert read(store, second)["value"] == 2
    assert len(loads) == 2
    assert read(store, first)["value"] == 1
    assert len(loads) == 3
    for _ in range(2):
        assert store.read(bank, identity=IDENTITY, kind="bank") == {"value": 3}
    assert len(loads) == 5
    assert read(store, first)["value"] == 1
    assert len(loads) == 5


def test_disabling_cache_does_not_reuse_a_previously_decoded_control(store, loads):
    reference = store.stage({"value": 1}, identity=IDENTITY, kind="control")
    read(store, reference)
    store.max_cached_bytes = 0
    assert read(store, reference) == {"value": 1}
    assert read(store, reference) == {"value": 1}
    assert len(loads) == 3


def test_default_store_remains_uncached_and_staging_preserves_the_original_bytes(store, loads):
    uncached = SessionArtifacts(store.native, store.directory)
    cached = SessionArtifacts(store.native, store.directory, max_cached_bytes=65536)
    payload = {"value": 1.0000000000000002}
    reference = uncached.stage(payload, identity=IDENTITY, kind="control")
    assert cached.stage(payload, identity=IDENTITY, kind="control") == reference
    assert read(uncached, reference) == payload
    assert read(uncached, reference) == payload
    assert len(loads) == 2


def test_cache_budget_includes_validated_payload_archive_and_metadata_margin(store, loads):
    reference = store.stage({"value": 1}, identity=IDENTITY, kind="control")
    # Dict, clave UTF-8 e int, más archivo y margen del contrato de caché.
    charge = 32 + 37 + 32 + reference["bytes"] + 512
    store.max_cached_bytes = charge
    assert read(store, reference) == {"value": 1}
    assert read(store, reference) == {"value": 1}
    assert len(loads) == 1
    store.max_cached_bytes = charge - 1
    assert read(store, reference) == {"value": 1}
    assert read(store, reference) == {"value": 1}
    assert len(loads) == 3


def test_pruning_an_orphan_releases_its_cached_control(store, loads):
    reference = store.stage({"value": 1}, identity=IDENTITY, kind="control")
    read(store, reference)
    store.prune_unreferenced([])
    assert not (store.directory / reference["name"]).exists()
    # Una referencia idéntica vuelve a ser una propuesta nueva, aún sin cabeza.
    assert store.stage({"value": 1}, identity=IDENTITY, kind="control") == reference
    assert read(store, reference) == {"value": 1}
    assert len(loads) == 2


@pytest.mark.parametrize("limit", [-1, True, 64 * 1024**2 + 1])
def test_cache_rejects_invalid_limits_before_creating_files(store, tmp_path, limit):
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError):
        SessionArtifacts(store.native, directory, max_cached_bytes=limit)
    assert not directory.exists()
