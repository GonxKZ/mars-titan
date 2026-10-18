"""Medidas del componente con el mismo cálculo y límites explícitos."""

import pytest
import torch

from benchmarks import refinement_routes as benchmark


def test_small_cpu_measure_preserves_rng_and_reports_scope_and_parity():
    before = torch.random.get_rng_state()
    result = benchmark.measure(device="cpu", batches=(2,), episodes=(8,), warmup=0, repeats=2)
    assert torch.equal(before, torch.random.get_rng_state())
    assert result["optimizer_steps"] == 0
    assert result["scope"] == "reader_only_without_mac_data_loading_or_training"
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
