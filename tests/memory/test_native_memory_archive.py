"""Validar archivos episódicos sin ejecutar código ni materializar tensores."""

import importlib
import io
import pickletools
import zipfile

import pytest
import torch
from test_native_episode_backend import native as native
from test_native_episode_backend import record
from test_write_policy import bank


def api():
    return importlib.import_module("mars_titan.memory.native_memory_archive")


def rewrite(content, *, member=None, replacement=None, added_bytes=0, extra=None):
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(content)) as source,
        zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            data = source.read(info)
            if info.filename == member and replacement is not None:
                data = replacement(data)
            with target.open(info.filename, "w") as destination:
                destination.write(data)
                if info.filename == member:
                    chunk = bytes(1024**2)
                    remaining = added_bytes
                    while remaining:
                        size = min(remaining, len(chunk))
                        destination.write(chunk[:size])
                        remaining -= size
        if extra:
            target.writestr(extra, b"extra")
    return output.getvalue()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("count", [0, 1, 8])
def test_valid_native_versions_remain_unchanged(native, version, count):
    scope = native.MemoryScope()
    scope.world, scope.partition, scope.fold, scope.representation = "fixture", "train", "0", "64"
    original = native.EpisodicMemory(scope, 73, 8, version)
    for index in range(1, count + 1):
        original.write(record(native, index), 17)
    encoded = original.snapshot_bytes()
    archives = {"one": encoded}
    expected = sum(info.file_size for info in zipfile.ZipFile(io.BytesIO(encoded)).infolist())
    assert (
        api().verify_memory_archives(archives, {"one": 8}, max_expanded_bytes=expected) == expected
    )
    restored = native.EpisodicMemory(scope, 73, 8, version)
    restored.restore_bytes(encoded)
    assert restored.snapshot_bytes() == encoded


def test_expansion_budget_is_exact_and_sums_all_indices(native):
    payload = bank(native).snapshot()["indices"]
    expanded = {
        role: sum(info.file_size for info in zipfile.ZipFile(io.BytesIO(content)).infolist())
        for role, content in payload.items()
    }
    total = sum(expanded.values())
    assert (
        api().verify_memory_archives(
            payload, {"reservoir": 2, "selective": 1, "recent": 1}, max_expanded_bytes=total
        )
        == total
    )
    for limit in (total - 1, max(expanded.values())):
        with pytest.raises(ValueError, match="presupuesto"):
            api().verify_memory_archives(
                payload, {"reservoir": 2, "selective": 1, "recent": 1}, max_expanded_bytes=limit
            )


def test_restore_counts_coded_bytes_and_two_expanded_copies_at_the_exact_limit(native):
    first = bank(native)
    archives = first.snapshot()["indices"]
    coded = sum(map(len, archives.values()))
    expanded = sum(
        info.file_size
        for content in archives.values()
        for info in zipfile.ZipFile(io.BytesIO(content)).infolist()
    )
    required = first.estimated_bytes(0) + coded + 2 * expanded
    for maximum, accepted in ((required, True), (required - 1, False)):
        value = bank(native, max_working_bytes=maximum)
        payload = value.snapshot()
        assert sum(map(len, payload["indices"].values())) == coded
        if accepted:
            assert value.restore(payload).snapshot() == payload
        else:
            with pytest.raises(ValueError, match="presupuesto"):
                value.restore(payload)


def test_metadata_inspection_never_calls_the_torch_tensor_rebuilder(native, monkeypatch):
    payload = bank(native).snapshot()["indices"]
    module = api()

    def forbidden(*args, **kwargs):
        pytest.fail("La inspección no puede reconstruir tensores Torch")

    monkeypatch.setattr(torch._utils, "_rebuild_tensor_v2", forbidden)
    assert module.verify_memory_archives(payload, {"reservoir": 2, "selective": 1, "recent": 1}) > 0


def test_compressed_80_mib_is_rejected_before_creating_a_native_restore(native, monkeypatch):
    value = bank(native, max_working_bytes=17 * 1024**2).propose(
        [record(native, 1)], errors={1: 0.5}, confirmed_at=3
    )
    payload = value.snapshot()
    payload["indices"]["reservoir"] = rewrite(
        payload["indices"]["reservoir"], member="archive/data/0", added_bytes=80 * 1024**2
    )
    assert len(payload["indices"]["reservoir"]) < 100_000
    payload["receipt"]["native_archive_bytes"]["reservoir"] = len(payload["indices"]["reservoir"])

    def forbidden():
        pytest.fail("El archivo expandido no debe alcanzar el restaurador nativo")

    monkeypatch.setattr(value, "_new", forbidden)
    with pytest.raises(ValueError, match="presupuesto|tamaño|forma"):
        value.restore(payload)


@pytest.mark.parametrize("member", ["archive/data/0", "archive/data/2", "archive/data.pkl"])
def test_incompatible_storage_or_metadata_sizes_fail_before_native_load(native, member):
    payload = bank(native).snapshot()["indices"]
    payload["reservoir"] = rewrite(payload["reservoir"], member=member, added_bytes=1)
    with pytest.raises(ValueError):
        api().verify_memory_archives(payload, {"reservoir": 2, "selective": 1, "recent": 1})


@pytest.mark.parametrize("kind", ["global", "reduce", "alias", "offset", "stride", "shape", "memo"])
def test_closed_metadata_grammar_rejects_executable_or_incompatible_descriptors(native, kind):
    value = bank(native).propose([record(native, 1)], errors={1: 0.5}, confirmed_at=3)
    payload = value.snapshot()["indices"]

    def change(data):
        if kind == "global":
            return b"\x80\x02cbuiltins\neval\n."
        if kind == "reduce":
            return b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n)R."
        if kind == "memo":
            return b"\x80\x02}r\xff\xff\xff\x7f."
        if kind == "alias":
            assert data.count(b"X\x01\x00\x00\x001") == 1
            return data.replace(b"X\x01\x00\x00\x001", b"X\x01\x00\x00\x000")
        operations = list(pickletools.genops(data))
        first = next(i for i, (op, _, _) in enumerate(operations) if op.name == "BINPERSID")
        integers = [(arg, at) for op, arg, at in operations[first:] if op.name == "BININT1"]
        # Primer tensor: offset, filas, columnas, stride de fila y de columna.
        chosen = {"offset": 0, "shape": 2, "stride": 3}[kind]
        _, at = integers[chosen]
        return data[: at + 1] + bytes([1 if kind == "offset" else 63]) + data[at + 2 :]

    payload["reservoir"] = rewrite(
        payload["reservoir"], member="archive/data.pkl", replacement=change
    )
    with pytest.raises(ValueError):
        api().verify_memory_archives(payload, {"reservoir": 2, "selective": 1, "recent": 1})


def test_extra_archive_members_are_rejected(native):
    payload = bank(native).snapshot()["indices"]
    payload["reservoir"] = rewrite(payload["reservoir"], extra="archive/data/4")
    with pytest.raises(ValueError):
        api().verify_memory_archives(payload, {"reservoir": 2, "selective": 1, "recent": 1})
