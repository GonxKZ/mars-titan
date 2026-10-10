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


# Fila real de 000001.SZ (2007-01-26). El cierre supera al máximo en una unidad de redondeo.
ROUNDED = [3.976320878595475, 3.9916150569915767, 3.845233163135132, 3.991615056991577]


def test_chart_relaxes_only_the_ordering_check_with_the_declared_tolerance():
    module = charts_module()
    ordered = np.array([[10, 12, 9, 11], [11, 14, 10, 12]], dtype=float)
    # Una ventana ordenada produce el mismo PNG con o sin tolerancia.
    assert module.chart_png(ordered, end_index=1, context=2) == module.chart_png(
        ordered, end_index=1, context=2, ordering_rtol=1e-9
    )
    rounded = np.array([[10, 12, 9, 11], ROUNDED])
    before = rounded.copy()
    with pytest.raises(ValueError, match="incoherentes"):
        module.chart_png(rounded, end_index=1, context=2)
    assert module.chart_png(rounded, end_index=1, context=2, ordering_rtol=1e-9).startswith(
        b"\x89PNG"
    )
    assert np.array_equal(rounded, before)
    # Un desorden mayor que la tolerancia sigue siendo incoherente.
    broken = np.array([[10, 12, 9, 11], [10, 12, 9, 12 * (1 + 2e-9)]])
    with pytest.raises(ValueError, match="incoherentes"):
        module.chart_png(broken, end_index=1, context=2, ordering_rtol=1e-9)
    low_broken = np.array([[10, 12, 9, 11], [10, 12, 10 * (1 + 2e-9), 11]])
    with pytest.raises(ValueError, match="incoherentes"):
        module.chart_png(low_broken, end_index=1, context=2, ordering_rtol=1e-9)
    with pytest.raises(ValueError, match="tolerancia"):
        module.chart_png(ordered, end_index=1, context=2, ordering_rtol=1e-3)
