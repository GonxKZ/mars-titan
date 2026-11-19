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


def test_probe_detects_wrong_ties_and_checks_higher_derivatives(monkeypatch):
    def ordinary(weights, previous, gradients, gates, *, differentiable):
        assert differentiable
        following, velocity = study.reference_write(weights[0], previous[0], gradients[0], *gates)
        return (following,), (velocity,)

    def stable(scores, count):
        return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]

    monkeypatch.setattr(study.cuda_kernels, "try_memory_write", ordinary)
    monkeypatch.setattr(study.cuda_kernels, "stable_neighbors", stable)
    summary = study.check_probe(torch.device("cpu"))
    assert summary["first_derivatives"] == summary["second_derivatives"] == 6
    monkeypatch.setattr(study.cuda_kernels, "stable_neighbors", lambda x, k: stable(x, k).flip([1]))
    with pytest.raises(ValueError, match="orden de los vecinos"):
        study.check_probe(torch.device("cpu"))
