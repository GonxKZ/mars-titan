"""Medidas del componente con el mismo cálculo y límites explícitos."""

import json
import sys
from contextlib import contextmanager

import pytest
import torch

from benchmarks import refinement_routes as benchmark


@pytest.mark.parametrize("checks", ["immediate", "deferred"])
def test_small_cpu_measure_preserves_rng_and_reports_scope_and_parity(checks):
    before = torch.random.get_rng_state()
    result = benchmark.measure(
        device="cpu", batches=(2,), episodes=(8,), warmup=0, repeats=2, checks=checks
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert result["optimizer_steps"] == 0
    assert result["scope"] == "reader_only_without_mac_data_loading_or_training"
    assert result["checks"] == checks
    assert result["cases"][0]["parity"] == "bitwise_cached_k4cc_and_legacy_first_read_k4"
    values = result["cases"][0]["variants"]
    assert len(values) == 7
    assert all(len(item["seconds"]) == 2 and item["p50_ms"] > 0 for item in values.values())
    assert all(item["peak_allocated_bytes"] is None for item in values.values())


@pytest.mark.parametrize(
    "changes",
    [
        dict(batches=()),
        dict(episodes=(-1,)),
        dict(batches=(257,)),
        dict(repeats=1),
        dict(device="automatic"),
        dict(checks="disabled"),
    ],
)
def test_measure_rejects_invalid_work_before_building_models(changes, monkeypatch):
    monkeypatch.setattr(benchmark, "fixture", lambda *_: pytest.fail("No debe construir modelos"))
    with pytest.raises(ValueError):
        benchmark.measure(**(dict(device="cpu") | changes))


def test_strict_precision_is_restored_when_the_measure_fails():
    before = torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError), benchmark.strict_numerics():
        assert torch.get_float32_matmul_precision() == "highest"
        raise RuntimeError("interrupción técnica")
    assert torch.get_float32_matmul_precision() == before


def test_deferred_mode_wraps_each_complete_call_and_closes_its_checks(monkeypatch):
    active, completed = [], []
    original = benchmark.calls

    @contextmanager
    def checks():
        active.append(True)
        try:
            yield
        finally:
            active.pop()
            completed.append(True)

    def guarded(function):
        def run():
            assert active == [True]
            return function()

        return run

    def calls(*args):
        return {name: guarded(function) for name, function in original(*args).items()}

    monkeypatch.setattr(benchmark, "deferred_checks", checks)
    monkeypatch.setattr(benchmark, "calls", calls)
    benchmark.measure(
        device="cpu", batches=(1,), episodes=(0,), warmup=0, repeats=2, checks="deferred"
    )
    assert completed and not active


@pytest.mark.parametrize("fault", ["parity", "repeatability"])
def test_a_measurement_rejects_different_cached_results_and_drifting_repetitions(
    fault, monkeypatch
):
    original = benchmark.calls

    def calls(*args):
        result = original(*args)
        name = "cached_k4cc" if fault == "parity" else "legacy_k1_with_head"
        function, count = result[name], []

        def changed():
            count.append(True)
            delta = 1.0 if fault == "parity" or len(count) > 1 else 0.0
            return function() + delta

        result[name] = changed
        return result

    monkeypatch.setattr(benchmark, "calls", calls)
    with pytest.raises(ValueError, match="bits"):
        benchmark.measure(device="cpu", batches=(1,), episodes=(0,), warmup=0, repeats=2)


def test_a_source_change_during_measurement_invalidates_the_result(monkeypatch):
    identities = iter(({"source": "a" * 64}, {"source": "b" * 64}))
    monkeypatch.setattr(benchmark, "source_identity", lambda: next(identities))
    with pytest.raises(ValueError, match="código ha cambiado"):
        benchmark.measure(device="cpu", batches=(1,), episodes=(0,), warmup=0, repeats=2)


def test_cli_records_check_mode_and_never_overwrites_a_measurement(tmp_path, monkeypatch):
    output = tmp_path / "cost.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refinement_routes",
            "--device",
            "cpu",
            "--batches",
            "1",
            "--episodes",
            "0",
            "--warmup",
            "0",
            "--repeats",
            "2",
            "--checks",
            "deferred",
            "--output",
            str(output),
        ],
    )
    benchmark.main()
    before = output.read_bytes()
    assert json.loads(before)["checks"] == "deferred"
    with pytest.raises(FileExistsError):
        benchmark.main()
    assert output.read_bytes() == before


def test_unavailable_cuda_fails_before_models_without_cpu_fallback(monkeypatch):
    from mars_titan.training.experiment_resources import GpuLease

    monkeypatch.setattr(GpuLease, "__enter__", lambda _: None)
    monkeypatch.setattr(GpuLease, "__exit__", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "fixture", lambda *_: pytest.fail("No debe construir modelos"))
    with pytest.raises(RuntimeError, match="No se cambia a CPU"):
        benchmark.measure(device="cuda:0", batches=(1,), episodes=(0,), warmup=0, repeats=2)
