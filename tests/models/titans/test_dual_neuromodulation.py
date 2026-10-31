"""Signos, neutralidad y contrato de las dos señales externas de N1."""

import importlib
from dataclasses import replace

import pytest
import torch


def api():
    return importlib.import_module("mars_titan.models.titans.dual_neuromodulation")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_expected_and_unexpected_channels_have_separate_restricted_signs(dtype):
    module = api().DualNeuromodulation(api().DualNeuromodulationConfig())
    signal = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=dtype)
    result = module(signal)
    assert set(result) == {"write", "forget"}
    assert result["write"].shape == result["forget"].shape == (4, 1)
    assert result["write"][0] == result["forget"][0] == 0
    assert result["write"][1] < 0 and result["forget"][1] == 0
    assert result["write"][2] > 0 and result["forget"][2] > 0
    assert result["write"][3] == 0 and result["forget"][3] > 0
    for value in result.values():
        assert value.dtype == dtype and torch.isfinite(value).all() and value.abs().max() <= 1


def test_zero_gains_are_an_exact_neutral_control_without_parameters_or_rng_use():
    state = torch.get_rng_state().clone()
    config = api().DualNeuromodulationConfig(
        write_expected=0, write_unexpected=0, forget_unexpected=0
    )
    module = api().DualNeuromodulation(config)
    result = module(torch.ones((3, 2), dtype=torch.float64))
    assert all(
        torch.equal(value, torch.zeros((3, 1), dtype=torch.float64)) for value in result.values()
    )
    assert not list(module.parameters()) and not list(module.buffers())
    assert torch.equal(state, torch.get_rng_state())
    assert config.identity()["kind"] == "dual_neuromodulation_v1"


def test_large_coefficients_still_respect_the_declared_shift_bound():
    module = api().DualNeuromodulation(
        api().DualNeuromodulationConfig(
            write_expected=4, write_unexpected=4, forget_unexpected=4, bound=4
        )
    )
    result = module(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
    assert all(value.abs().max() <= 4 for value in result.values())
    assert result["write"][1] > 3 and result["forget"][1] > 3


def test_signal_rows_are_bounded_before_the_arithmetic():
    module = api().DualNeuromodulation(api().DualNeuromodulationConfig())
    with pytest.raises(ValueError):
        module(torch.zeros((8193, 2)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("write_expected", -1),
        ("write_unexpected", True),
        ("forget_unexpected", float("nan")),
        ("bound", 0),
        ("bound", 5),
        ("bound", float("inf")),
        ("write_expected", 5),
    ],
)
def test_configuration_rejects_invalid_or_unbounded_gains(field, value):
    with pytest.raises(ValueError):
        replace(api().DualNeuromodulationConfig(), **{field: value})


@pytest.mark.parametrize(
    "kind", ["shape", "empty", "range", "negative", "nan", "inf", "integer", "gradient"]
)
def test_invalid_signals_are_rejected_even_with_zero_gains(kind):
    module = api().DualNeuromodulation(
        api().DualNeuromodulationConfig(write_expected=0, write_unexpected=0, forget_unexpected=0)
    )
    signal = torch.zeros((2, 2), dtype=torch.float64)
    if kind == "shape":
        signal = torch.zeros((2, 3), dtype=torch.float64)
    elif kind == "empty":
        signal = torch.zeros((0, 2), dtype=torch.float64)
    elif kind in ("range", "negative", "nan", "inf"):
        signal[0, 0] = dict(range=1.1, negative=-0.1, nan=float("nan"), inf=float("inf"))[kind]
    elif kind == "integer":
        signal = signal.to(torch.int64)
    else:
        signal.requires_grad_(True)
    with pytest.raises(ValueError):
        module(signal)


def test_validation_binds_rows_precision_and_device_to_the_predictor():
    module = api().DualNeuromodulation(api().DualNeuromodulationConfig())
    signal = torch.zeros((2, 2), dtype=torch.float64)
    reference = torch.ones(1, dtype=torch.float64)
    module.validate(signal, 2, reference=reference)
    with pytest.raises(ValueError):
        module.validate(signal, 1, reference=reference)
    with pytest.raises(ValueError):
        module.validate(signal, 2, reference=reference.float())
    with pytest.raises(ValueError):
        module.validate(signal, 2, reference=torch.empty(1, device="meta", dtype=torch.float64))


def test_autocast_is_not_a_silent_precision_change():
    module = api().DualNeuromodulation(api().DualNeuromodulationConfig())
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError):
        module(torch.zeros((2, 2)))


@pytest.mark.parametrize("signal", [None, [], torch.zeros((2, 1, 2))])
def test_forward_does_not_guess_the_layout_of_nonstandard_inputs(signal):
    module = api().DualNeuromodulation(api().DualNeuromodulationConfig())
    with pytest.raises(ValueError):
        module(signal)


@pytest.mark.parametrize("config", [None, {}, 1])
def test_constructor_requires_the_typed_configuration(config):
    with pytest.raises(ValueError):
        api().DualNeuromodulation(config)


@pytest.mark.cuda
def test_cuda_fp32_shifts_match_the_cpu_reference():
    module = api().DualNeuromodulation(api().DualNeuromodulationConfig())
    signal = torch.tensor([[0.2, 0.7], [0.8, 0.1]], dtype=torch.float32)
    expected = module(signal)
    actual = module(signal.to("cuda:0"))
    for name in expected:
        torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=1e-6, atol=1e-7)
