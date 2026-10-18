"""Comprobaciones del benchmark de representación, sin CUDA ni aprendizaje."""

import importlib
import json

import numpy as np
import pytest


def test_derivation_benchmark_records_distinct_stages_without_models(tmp_path):
    from benchmarks.candle_geometry_derivation import measure_derivation

    result = measure_derivation(tmp_path, samples=5, repeats=2, batch_rows=2)
    assert result["fixture"]["samples"] == 5
    assert len(result["runs"]) == 2
    for run in result["runs"]:
        assert run["timing_seconds"]["read"] > 0
        assert run["timing_seconds"]["geometry"] > 0
        assert run["timing_seconds"]["write"] > 0
        assert run["samples"] == 5
    assert result["device"] == "cpu" and result["learned_models_loaded"] == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [("samples", 513), ("repeats", 0), ("batch_rows", True)])
def test_derivation_benchmark_rejects_unbounded_work_before_creating_files(tmp_path, field, value):
    from benchmarks.candle_geometry_derivation import measure_derivation

    with pytest.raises(ValueError, match="presupuesto"):
        measure_derivation(tmp_path, **{field: value})
    assert list(tmp_path.iterdir()) == []


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
    assert result["encoder"] is None
    assert "mars_titan.data.embeddings" in result["source_sha256"]
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


def test_benchmark_preserves_the_declared_ohlc_tolerance_in_every_representation(tmp_path):
    path = tmp_path / "rounded.npz"
    values = np.tile([10.0, 10.0, 9.0, 10.0 + 1e-10], (1, 64, 1))
    np.savez(path, windows=values, presence=np.ones((1, 64), dtype=bool))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="tolerancia"):
        benchmark().measure(windows_path=path, repeats=2, warmup=0)
    result = benchmark().measure(windows_path=path, repeats=2, warmup=0, ordering_rtol=1e-9)
    assert result["workload"]["ordering_rtol"] == 1e-9
    assert set(result["results"]) == {"geometry", "ink"}
    assert path.read_bytes() == before


@pytest.mark.parametrize("tolerance", [True, -1.0, 1e-5, float("nan")])
def test_benchmark_rejects_invalid_ordering_tolerance_before_loading(tolerance, monkeypatch):
    monkeypatch.setattr(benchmark(), "windows", lambda *_: pytest.fail("No debe abrir ventanas"))
    with pytest.raises(ValueError):
        benchmark().measure(ordering_rtol=tolerance)


def test_cuda_cli_sets_strict_fp32_before_the_measurement(tmp_path, monkeypatch):
    import torch

    from mars_titan.data.embeddings import _strict_precision

    before = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )

    def measured(**kwargs):
        assert kwargs["device"] == "cuda"
        return {"runtime_precision": _strict_precision()}

    monkeypatch.setattr(benchmark(), "measure", measured)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        destination = tmp_path / "cuda.json"
        benchmark().main(["measure", "--device", "cuda", "--output", str(destination)])
        assert json.loads(destination.read_text())["runtime_precision"]["cudnn_tf32"] is False
    finally:
        torch.set_float32_matmul_precision(before[0])
        torch.backends.cuda.matmul.allow_tf32 = before[1]
        torch.backends.cudnn.allow_tf32 = before[2]


def test_changed_implementation_invalidates_the_benchmark(monkeypatch):
    values = iter(({"source": "a" * 64}, {"source": "b" * 64}))
    monkeypatch.setattr(benchmark(), "code_identity", lambda: next(values))
    with pytest.raises(ValueError, match="código cambió"):
        benchmark().measure(samples=1, repeats=1, warmup=0)


@pytest.mark.parametrize("command", ["measure", "compare"])
@pytest.mark.parametrize("kind", ["file", "dangling_link"])
def test_cli_rejects_existing_artifacts_before_computing(tmp_path, monkeypatch, command, kind):
    output = tmp_path / "existing.json"
    if kind == "file":
        output.write_text("resultado anterior")
    else:
        output.symlink_to(tmp_path / "missing.json")
    arguments = [command, "--output", str(output)]
    if command == "compare":
        arguments += ["--previous", str(tmp_path / "old"), "--current", str(tmp_path / "new")]
    monkeypatch.setattr(benchmark(), "measure", lambda **_: pytest.fail("No debe medir"))
    monkeypatch.setattr(
        benchmark(), "compare_representations", lambda *_: pytest.fail("No debe leer")
    )
    with pytest.raises(FileExistsError):
        benchmark().main(arguments)
    if kind == "file":
        assert output.read_text() == "resultado anterior"
    else:
        assert output.is_symlink()
