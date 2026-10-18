"""Comprobaciones del benchmark de representación, sin CUDA ni aprendizaje."""

import importlib
import json

import numpy as np
import pytest


def benchmark():
    try:
        return importlib.import_module("benchmarks.candle_geometry")
    except ModuleNotFoundError:
        pytest.fail("Falta el benchmark acotado de geometría de velas")


def test_cpu_benchmark_identifies_equal_inputs_and_real_timings(tmp_path):
    result = benchmark().measure(samples=4, repeats=2, warmup=1, device="cpu")
    assert result["workload"]["samples"] == 4
    assert result["workload"]["context_sessions"] == 64
    assert result["optimizer_steps"] == 0
    assert result["cuda"] is None
    assert set(result["results"]) == {"geometry", "ink"}
    assert result["output_bytes_per_row"] == 512 * 4
    for measured in result["results"].values():
        assert len(measured["seconds"]) == 2 and min(measured["seconds"]) > 0
        assert measured["rows_per_second"] > 0
        assert measured["reproducible"] is True
        assert len(measured["output_sha256"]) == 64
    path = tmp_path / "cost.json"
    assert (
        benchmark().main(
            ["measure", "--samples", "2", "--repeats", "2", "--warmup", "0", "--output", str(path)]
        )
        == 0
    )
    assert json.loads(path.read_text())["workload"]["samples"] == 2


@pytest.mark.parametrize(
    "options",
    [dict(samples=0), dict(samples=257), dict(repeats=0), dict(warmup=-1), dict(device="other")],
)
def test_benchmark_rejects_unbounded_or_ambiguous_requests(options):
    with pytest.raises(ValueError):
        benchmark().measure(**options)


def test_saved_windows_preserve_masks_and_are_hashed(tmp_path):
    path = tmp_path / "windows.npz"
    windows = np.tile([10.0, 12.0, 9.0, 11.0], (3, 64, 1))
    presence = np.ones((3, 64), dtype=bool)
    presence[:, 4] = False
    windows[:, 4] = 0
    np.savez(path, windows=windows, presence=presence)
    result = benchmark().measure(windows_path=path, repeats=2, warmup=0)
    assert result["workload"]["samples"] == 3
    assert result["workload"]["absent_slots"] == 3
    assert len(result["workload"]["source_sha256"]) == 64
    assert result["workload"]["kind"] == "provided_windows"
    windows[:, 4] = 10
    np.savez(path, windows=windows, presence=presence)
    with pytest.raises(ValueError, match="ausentes"):
        benchmark().measure(windows_path=path, repeats=2, warmup=0)


def test_cuda_request_cannot_fall_back_when_gpu_admission_fails(monkeypatch):
    from mars_titan.training.experiment_resources import GpuLease

    def unavailable(self):
        raise RuntimeError("GPU no preparada para esta comprobación")

    monkeypatch.setattr(GpuLease, "__enter__", unavailable)
    with pytest.raises(RuntimeError, match="GPU no preparada"):
        benchmark().measure(samples=1, repeats=1, warmup=0, device="cuda")
