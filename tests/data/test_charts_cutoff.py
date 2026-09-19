import importlib

import numpy as np
import pytest


def charts_module():
    try:
        return importlib.import_module("mars_titan.data.charts")
    except ModuleNotFoundError:
        pytest.fail("La generación causal de gráficos todavía no existe")


def test_chart_only_depends_on_past_window_and_is_reproducible():
    prices = np.array([[10, 12, 9, 11], [11, 14, 10, 12], [12, 13, 10, 11]], dtype=float)
    module = charts_module()
    one = module.chart_png(prices, end_index=1, context=2)
    prices[2] = [100, 200, 80, 190]
    assert module.chart_png(prices, end_index=1, context=2) == one
    assert one.startswith(b"\x89PNG")
    assert module.chart_png(prices, end_index=2, context=2) != one


def test_chart_refuses_short_invalid_and_nonfinite_windows():
    module = charts_module()
    with pytest.raises(ValueError):
        module.chart_png(np.ones((2, 4)), end_index=0, context=2)
    with pytest.raises(ValueError):
        module.chart_png(np.full((2, 4), np.nan), end_index=1, context=2)
