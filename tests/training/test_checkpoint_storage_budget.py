"""El presupuesto de un checkpoint incluye los almacenamientos que realmente se guardan."""

import errno
from collections import namedtuple

import pytest
import torch

from mars_titan.training import checkpoints

IDENTITY = {"data": "storage-budget-fixture", "precision": "float32"}
MIB = 1024**2
DiskUsage = namedtuple("DiskUsage", "total used free")


def constrained_disk(monkeypatch, free):
    monkeypatch.setattr(
        checkpoints.shutil, "disk_usage", lambda _: DiskUsage(100 * MIB, 100 * MIB - free, free)
    )


def forbid_serialization(monkeypatch):
    def unexpected_save(*args, **kwargs):
        raise AssertionError("El almacenamiento que no cabe debe rechazarse antes de serializar")

    monkeypatch.setattr(checkpoints.torch, "save", unexpected_save)


@pytest.mark.parametrize("view", ["offset", "empty", "strided"])
def test_large_backing_storage_is_rejected_before_writing_a_small_view(tmp_path, monkeypatch, view):
    previous = checkpoints.save_training_state(tmp_path, {"global_step": 1}, identity=IDENTITY)
    index = (tmp_path / "latest.json").read_bytes()
    backing = torch.zeros(8 * MIB, dtype=torch.float32)
    views = {"offset": backing[-1:], "empty": backing[:0], "strided": backing[::MIB]}
    constrained_disk(monkeypatch, 20 * MIB)
    writes = []

    def record_write(*args, **kwargs):
        writes.append(True)
        raise AssertionError("No debe empezar a serializar un estado que no cabe")

    monkeypatch.setattr(checkpoints.torch, "save", record_write)
    with pytest.raises(OSError) as error:
        checkpoints.save_training_state(
            tmp_path, {"global_step": 2, "view": views[view]}, identity=IDENTITY
        )
    assert error.value.errno == errno.ENOSPC
    assert not writes and not (tmp_path / ".pending.pt").exists()
    assert previous.is_file() and (tmp_path / "latest.json").read_bytes() == index
    assert checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["global_step"] == 1


def test_shared_storage_is_reserved_once_and_aliases_survive_recovery(tmp_path, monkeypatch):
    backing = torch.arange(MIB // 2, dtype=torch.float32)
    constrained_disk(monkeypatch, 20 * MIB)
    views = [backing, backing[1:], backing[::2]] * 10
    path = checkpoints.save_training_state(
        tmp_path, {"global_step": 1, "views": views}, identity=IDENTITY
    )
    assert 2 * MIB < path.stat().st_size < 3 * MIB
    restored = checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["views"]
    for actual, expected in zip(restored, views, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.untyped_storage().data_ptr() == restored[0].untyped_storage().data_ptr()
    restored[0][1] = -7
    assert restored[1][0] == -7


def test_distinct_storages_are_not_deduplicated_by_shape_or_content(tmp_path, monkeypatch):
    backing = torch.zeros(2 * MIB, dtype=torch.float32)
    constrained_disk(monkeypatch, 28 * MIB)
    forbid_serialization(monkeypatch)
    with pytest.raises(OSError) as error:
        checkpoints.save_training_state(
            tmp_path,
            {"global_step": 1, "views": [backing[:1], backing.clone()[:1]]},
            identity=IDENTITY,
        )
    assert error.value.errno == errno.ENOSPC
    assert not (tmp_path / "latest.json").exists()


def test_sparse_storage_views_cannot_escape_the_reservation(tmp_path, monkeypatch):
    backing = torch.zeros(4 * MIB, dtype=torch.int64)
    indices = backing[:2].reshape(2, 1)
    value = torch.sparse_coo_tensor(indices, torch.ones(1), (2, 2))
    constrained_disk(monkeypatch, 20 * MIB)
    forbid_serialization(monkeypatch)
    with pytest.raises(OSError) as error:
        checkpoints.save_training_state(
            tmp_path, {"global_step": 1, "sparse": value}, identity=IDENTITY
        )
    assert error.value.errno == errno.ENOSPC


def test_small_sparse_tensor_round_trips_without_dense_materialization(tmp_path):
    value = torch.sparse_coo_tensor([[1, 1], [0, 0]], [2.0, 3.0], (10_000, 10_000))
    checkpoints.save_training_state(
        tmp_path, {"global_step": 1, "sparse": value}, identity=IDENTITY
    )
    restored = checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["sparse"]
    assert restored.layout == torch.sparse_coo and restored.shape == value.shape
    assert not restored.is_coalesced()
    torch.testing.assert_close(restored._indices(), value._indices())
    torch.testing.assert_close(restored._values(), value._values())


@pytest.mark.parametrize("layout", ["csr", "csc", "bsr", "bsc"])
def test_compressed_indices_include_their_entire_storage(tmp_path, monkeypatch, layout):
    backing = torch.zeros(4 * MIB, dtype=torch.int64)
    backing[:3] = torch.tensor([0, 1, 1])
    values = torch.ones(1, 1, 1) if layout in {"bsr", "bsc"} else torch.ones(1)
    constructor = getattr(torch, f"sparse_{layout}_tensor")
    value = constructor(backing[:3], torch.tensor([0]), values, size=(2, 2))
    constrained_disk(monkeypatch, 20 * MIB)
    forbid_serialization(monkeypatch)
    with pytest.raises(OSError) as error:
        checkpoints.save_training_state(
            tmp_path, {"global_step": 1, "value": value}, identity=IDENTITY
        )
    assert error.value.errno == errno.ENOSPC


@pytest.mark.parametrize("layout", ["csr", "csc", "bsr", "bsc"])
def test_compressed_tensor_retains_its_layout_and_values(tmp_path, layout):
    dense = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    converter = getattr(dense, f"to_sparse_{layout}")
    value = converter((1, 1)) if layout in {"bsr", "bsc"} else converter()
    checkpoints.save_training_state(tmp_path, {"global_step": 1, "value": value}, identity=IDENTITY)
    restored = checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["value"]
    assert restored.layout == value.layout
    torch.testing.assert_close(restored.to_dense(), dense, rtol=0, atol=0)


@pytest.mark.parametrize("per_channel", [False, True])
def test_quantized_tensor_preserves_its_storage_and_quantization(tmp_path, per_channel):
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    quantized = (
        torch.quantize_per_channel(
            value, torch.tensor([0.1, 0.2]), torch.tensor([0, 1]), 0, torch.qint8
        )
        if per_channel
        else torch.quantize_per_tensor(value, 0.1, 0, torch.qint8)
    )
    checkpoints.save_training_state(
        tmp_path, {"global_step": 1, "value": quantized}, identity=IDENTITY
    )
    restored = checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["value"]
    assert restored.qscheme() == quantized.qscheme()
    torch.testing.assert_close(restored.dequantize(), quantized.dequantize(), rtol=0, atol=0)


def test_meta_tensor_does_not_reserve_nonexistent_dense_data(tmp_path, monkeypatch):
    value = torch.empty(200 * MIB, device="meta")
    constrained_disk(monkeypatch, 20 * MIB)
    path = checkpoints.save_training_state(
        tmp_path, {"global_step": 1, "value": value}, identity=IDENTITY
    )
    assert path.stat().st_size < MIB
    restored = checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["value"]
    assert restored.device.type == "meta" and restored.shape == value.shape


@pytest.mark.cuda
def test_cuda_storage_is_reserved_before_copying_and_aliases_survive_loading(tmp_path, monkeypatch):
    backing = torch.arange(MIB, dtype=torch.float32, device="cuda:0")
    state = {"global_step": 1, "views": [backing[1::2], backing[100:102]]}
    constrained_disk(monkeypatch, 20 * MIB)
    with pytest.raises(OSError) as error:
        checkpoints.save_training_state(tmp_path, state, identity=IDENTITY)
    assert error.value.errno == errno.ENOSPC and not (tmp_path / ".pending.pt").exists()
    constrained_disk(monkeypatch, 24 * MIB)
    path = checkpoints.save_training_state(tmp_path, state, identity=IDENTITY)
    assert 4 * MIB < path.stat().st_size < 5 * MIB
    first, second = checkpoints.load_training_state(tmp_path, expected_identity=IDENTITY)["views"]
    assert first.device.type == second.device.type == "cpu"
    assert first.stride() == (2,) and first.storage_offset() == 1
    assert second.storage_offset() == 100
    torch.testing.assert_close(first, torch.arange(1, MIB, 2, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(second, torch.tensor([100.0, 101.0]), rtol=0, atol=0)
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
