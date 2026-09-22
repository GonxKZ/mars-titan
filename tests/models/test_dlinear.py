"""Referencia aritmética independiente para la adaptación de DLinear."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.dlinear")
    except ModuleNotFoundError:
        pytest.fail("Falta la referencia DLinear")


def test_decomposition_replicates_endpoints_and_reconstructs_observed_window():
    torch = pytest.importorskip("torch")
    model = module().DLinear(context=5, kernel_size=3)
    values = torch.arange(1.0, 6.0).reshape(1, 5, 1)
    seasonal, trend = model.decompose(values)
    torch.testing.assert_close(trend.flatten(), torch.tensor([4 / 3, 2, 3, 4, 14 / 3]))
    torch.testing.assert_close(seasonal + trend, values)
    with torch.no_grad():
        model.seasonal.weight.fill_(0.2)
        model.trend.weight.fill_(0.2)
        model.seasonal.bias.zero_()
        model.trend.bias.zero_()
    torch.testing.assert_close(model(values), torch.tensor([[3.0]]))


def test_channels_and_batches_do_not_mix_during_decomposition():
    torch = pytest.importorskip("torch")
    model = module().DLinear(context=5, kernel_size=3)
    value = torch.arange(10.0).reshape(1, 5, 2)
    batch = torch.cat([value, value + 100])
    alone = model.decompose(value)
    together = model.decompose(batch)
    for index in (0, 1):
        torch.testing.assert_close(alone[index], together[index][:1])
        torch.testing.assert_close(alone[index][:, :, :1], model.decompose(value[:, :, :1])[index])
    assert model(batch).shape == (2, 2)


def test_seasonal_and_trend_use_their_own_weights_and_biases():
    torch = pytest.importorskip("torch")
    model = module().DLinear(context=5, kernel_size=3)
    with torch.no_grad():
        model.seasonal.weight.copy_(torch.tensor([[1.0, 2, 3, 4, 5]]))
        model.seasonal.bias.fill_(2.0)
        model.trend.weight.copy_(torch.tensor([[-1.0, 1, -1, 1, -1]]))
        model.trend.bias.fill_(-1.0)
    # Componente estacional 10/3 y componente de tendencia -4.
    value = torch.arange(1.0, 6.0).reshape(1, 5, 1)
    torch.testing.assert_close(model(value), torch.tensor([[-2 / 3]]))


@pytest.mark.parametrize("context,kernel_size", [(0, 3), (5, 0), (5, 2), (5, 7)])
def test_invalid_context_or_moving_average_fails_early(context, kernel_size):
    with pytest.raises(ValueError):
        module().DLinear(context=context, kernel_size=kernel_size)


def test_wrong_window_shape_is_rejected():
    torch = pytest.importorskip("torch")
    model = module().DLinear(context=5, kernel_size=3)
    for values in (torch.ones(5), torch.ones(1, 4, 2), torch.ones(1, 5, 0)):
        with pytest.raises(ValueError, match="ventana"):
            model(values)
