"""La medición opcional de otra GPU no cambia la admisión del entrenador."""

import pytest
import torch

from benchmarks import titans_cuda_updates as study


def test_exploratory_dispatch_is_explicit_and_restored_on_error(monkeypatch):
    device = torch.device("cuda:0")

    def unsupported(_):
        return False

    monkeypatch.setattr(study.cuda_kernels, "_supported_device", unsupported)
    monkeypatch.setattr(study, "find_spec", lambda _: object())
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    with pytest.raises(ValueError, match="probe-device"):
        study.benchmark_dispatch(device, False)
    with pytest.raises(RuntimeError, match="compilación"):
        with study.benchmark_dispatch(device, True):
            assert study.cuda_kernels._supported_device(device)
            assert not study.cuda_kernels._supported_device(torch.device("cuda:1"))
            raise RuntimeError("compilación")
    assert study.cuda_kernels._supported_device is unsupported


def test_probe_rejects_a_cpu_even_if_requested():
    with pytest.raises(ValueError, match="CUDA"):
        study.benchmark_dispatch(torch.device("cpu"), True)
