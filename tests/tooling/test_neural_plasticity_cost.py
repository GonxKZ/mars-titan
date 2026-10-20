"""Comprobar unidades, denominadores y selección de entradas de la medida."""

import importlib
import json

import numpy as np
import pytest


def api():
    return importlib.import_module("benchmarks.neural_plasticity_cost")


def test_latency_quantiles_and_throughput_use_all_observations():
    result = api().summary([1.0, 2.0, 3.0], rows=4)
    assert result["p50_ms"] == 2000
    assert result["p95_ms"] == 2900
    assert result["p99_ms"] == 2980
    assert result["rows_per_second"] == 2
    assert result["observations"] == 12


@pytest.mark.parametrize("samples", [[], [0.0], [-1.0], [float("inf")], [float("nan")]])
def test_invalid_measurements_cannot_be_reported_as_speedups(samples):
    with pytest.raises(ValueError):
        api().summary(samples, rows=4)


def test_subsetting_preserves_identity_availability_and_does_not_modify_source():
    raw = dict(
        inputs=dict(prices=np.arange(24).reshape(3, 2, 4)),
        presence=np.ones((3, 5), dtype=bool),
        sample_ids=["US/A/10", "US/B/10", "US/C/10"],
        market=["US"] * 3,
        prediction_at=np.full(3, 10),
        input_available_at=np.full(3, 9),
    )
    selected = api().subset(raw, 2)
    assert selected["sample_ids"] == raw["sample_ids"][:2]
    assert selected["input_available_at"].tolist() == [9, 9]
    selected["inputs"]["prices"].fill(0)
    assert raw["inputs"]["prices"].sum() == sum(range(24))
    with pytest.raises(ValueError):
        api().subset(raw, 4)


@pytest.mark.parametrize(
    "metadata",
    [
        dict(targets_read=True, dates_us=[1600000000000000]),
        dict(targets_read=False, dates_us=[1704067200000000]),
        dict(targets_read=False, dates_us=[]),
    ],
)
def test_protected_materializations_are_rejected_before_opening_arrays(tmp_path, metadata):
    data = tmp_path / "absent.npz"
    data.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        api().load_inputs(data)
