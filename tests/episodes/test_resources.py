"""La admisión limita también a los consumidores existentes de CUDA."""

from types import SimpleNamespace

import pytest
import torch

from mars_titan.training import experiment_resources as resources


@pytest.fixture
def cuda_fixture(monkeypatch, tmp_path):
    limits = []
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: (3 * resources.GIB, 8 * resources.GIB)
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda value, device=None: limits.append(value),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "RTX 4070")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(resources.resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=1))
    return limits


def test_legacy_calls_cannot_expand_the_granted_budget(cuda_fixture):
    original = torch.cuda.set_per_process_memory_fraction
    with resources.GpuLease() as lease:
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        torch.cuda.set_per_process_memory_fraction(0.1, 0)
        assert lease.record["max_vram_bytes"] == 2 * resources.GIB
        assert cuda_fixture[-2:] == [0.25, 0.1]
    assert torch.cuda.set_per_process_memory_fraction is original


def test_native_compute_is_excluded_and_lock_is_released(cuda_fixture, monkeypatch):
    monkeypatch.setattr(
        resources.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="123456, /opt/train_native\n"),
    )
    with pytest.raises(RuntimeError):
        with resources.GpuLease():
            pass
