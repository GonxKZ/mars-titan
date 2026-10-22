"""Objetivos puntuales y de intervalos con ejemplos calculados a mano."""

import importlib

import pytest
import torch


def api():
    return importlib.import_module("mars_titan.models.titans.mechanism_losses")


def forecasts(dtype=torch.float64, device="cpu"):
    values = torch.tensor([-2, -1, 0, 1, 2], dtype=dtype, device=device).repeat(2, 36, 1)
    values[:, 4] = torch.tensor([-4, -2, 0, 2, 4], dtype=dtype, device=device)
    return values


def test_equal_medians_keep_the_same_point_error_but_different_interval_scores():
    q = forecasts()
    result = api().mechanism_losses(q, torch.tensor([0, 3], dtype=q.dtype))
    assert result.point_errors.shape == (2, 12)
    assert result.point_errors[0].tolist() == [0] * 12
    assert result.point_errors[1].tolist() == [3] * 12
    torch.testing.assert_close(
        result.interval_scores[:, [3, 4]], torch.tensor([[3, 6], [33, 11]], dtype=q.dtype)
    )
    assert result.coverage[1, 4].tolist() == [False, True]
    assert result.widths[1, 4].tolist() == [4, 8]
    assert result.coverage[1, 3].tolist() == [False, False]


def test_negative_gains_remain_negative_and_the_baseline_has_zero_gain():
    q = forecasts()
    result = api().mechanism_losses(q, torch.tensor([0, 3], dtype=q.dtype))
    point, interval = result.gains()
    assert not point.any()
    torch.testing.assert_close(interval[:, 4], torch.tensor([-3, 22], dtype=q.dtype))
    assert interval[:, 0].tolist() == [0, 0]


def test_interval_boundaries_are_included_without_an_exceedance_penalty():
    q = forecasts()
    result = api().mechanism_losses(q, torch.tensor([-1, 2], dtype=q.dtype))
    assert result.coverage[0, 0].tolist() == [True, True]
    assert result.coverage[1, 0].tolist() == [False, True]
    torch.testing.assert_close(result.interval_scores[:, 0], torch.tensor([3, 8], dtype=q.dtype))


@pytest.mark.parametrize("fault", ["crossed", "median", "nan", "missing", "dtype", "target"])
def test_invalid_or_incomplete_alternatives_do_not_turn_into_losses(fault):
    q = forecasts()
    y = torch.zeros(2, dtype=q.dtype)
    if fault == "crossed":
        q[0, 0, 4] = -8
    elif fault == "median":
        q[0, 4, 2] = 0.5
    elif fault == "nan":
        q[0, 1, 0] = float("nan")
    elif fault == "missing":
        q = q[:, :35]
    elif fault == "dtype":
        y = y.float()
    else:
        y[0] = float("inf")
    with pytest.raises(ValueError):
        api().mechanism_losses(q, y)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_loss_autocast_cannot_change_the_precision_of_the_scores(device):
    if device == "cuda:0":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    q = forecasts(torch.float32, device)
    y = torch.zeros(2, dtype=q.dtype, device=device)
    backend = "cuda" if device == "cuda:0" else "cpu"
    with torch.autocast(backend), pytest.raises(ValueError, match="autocast"):
        api().mechanism_losses(q, y)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_losses_match_the_cpu_reference(dtype):
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    y = torch.tensor([0, 3], dtype=dtype)
    expected = api().mechanism_losses(forecasts(dtype), y)
    actual = api().mechanism_losses(forecasts(dtype, "cuda:0"), y.to("cuda:0"))
    for name in ("point_errors", "interval_scores", "coverage", "widths"):
        torch.testing.assert_close(getattr(actual, name).cpu(), getattr(expected, name))
