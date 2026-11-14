"""Medidas de un intento, con reloj y CUDA dobles y sin ejecutar modelos."""

import sys
from types import SimpleNamespace

import pytest

from mars_titan.training import comparison_telemetry as telemetry


def usage(cpu, rss):
    return SimpleNamespace(ru_utime=cpu, ru_stime=0.5, ru_maxrss=rss)


def test_attempt_reports_elapsed_cpu_and_process_peak_rss(monkeypatch):
    clocks = iter((10.0, 13.5))
    usages = iter((usage(1, 100), usage(3, 900)))
    monkeypatch.setattr(telemetry.time, "perf_counter", lambda: next(clocks))
    monkeypatch.setattr(telemetry.resource, "getrusage", lambda *_: next(usages))
    measurement = telemetry.AttemptMetrics()

    result = measurement.finish("cpu")

    assert result["wall_seconds"] == 3.5
    assert result["cpu_seconds"] == 2
    assert result["peak_rss_bytes"] == 900 * 1024
    assert result["rss_scope"] == "process_lifetime"
    assert result["cuda_allocated_peak_bytes"] is None
    assert result["energy_joules"] is None
    assert result["energy_reason"]


def test_cuda_metrics_do_not_initialize_a_device(monkeypatch):
    cuda = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    result = telemetry.AttemptMetrics().finish("cuda:0")

    assert result["cuda_allocated_peak_bytes"] is None
    assert result["cuda_reason"] == "CUDA no se inició en este proceso"


def test_cuda_measurement_does_not_import_torch_when_it_is_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    result = telemetry.AttemptMetrics().finish("cuda:0")
    assert result["cuda_allocated_peak_bytes"] is None
    assert "torch" not in sys.modules


def test_missing_package_versions_remain_unknown(monkeypatch):
    def missing(name):
        raise telemetry.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(telemetry.importlib.metadata, "version", missing)
    result = telemetry.AttemptMetrics().finish("cpu")
    assert result["versions"]["torch"] is None
    assert result["versions"]["lightgbm"] is None


def test_cuda_records_allocator_peaks_without_claiming_total_vram(monkeypatch):
    devices = []

    def allocated(device):
        devices.append(device)
        return 1234

    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        max_memory_allocated=allocated,
        max_memory_reserved=lambda device: 2048,
        get_device_name=lambda device: "GPU de prueba",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    result = telemetry.AttemptMetrics().finish("cuda:0")

    assert devices == ["cuda:0"]
    assert result["cuda_allocated_peak_bytes"] == 1234
    assert result["cuda_reserved_peak_bytes"] == 2048
    assert result["cuda_scope"] == "pytorch_allocator"
    assert result["gpu_name"] == "GPU de prueba"
    assert result["cuda_reason"] is None


def test_a_cuda_measurement_failure_is_reported_as_unavailable(monkeypatch):
    def failed(*_):
        raise RuntimeError("dispositivo no disponible")

    cuda = SimpleNamespace(is_initialized=lambda: True, max_memory_allocated=failed)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    result = telemetry.AttemptMetrics().finish("cuda:0")

    assert result["cuda_allocated_peak_bytes"] is None
    assert "dispositivo no disponible" in result["cuda_reason"]


def test_attempt_rejects_an_implicit_or_unknown_device():
    with pytest.raises(ValueError, match="dispositivo"):
        telemetry.AttemptMetrics().finish("auto")
