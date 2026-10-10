"""Ventanas por sesión del calendario con bit de presencia, sin inventar precios."""

import numpy as np
import pytest
import torch

from mars_titan.data.charts import chart_png
from mars_titan.data.price_windows import (
    PRICE_WINDOW_CHANNELS,
    absent_positions,
    calendar_digest,
    check_price_window_contract,
    check_row_positions,
    gate_price_window,
    market_absent_sessions,
    present_slots,
    price_window_contract,
    validate_price_window,
    window_rows,
)
from mars_titan.data.temporal import MarketClock


@pytest.fixture(scope="module")
def clock():
    return MarketClock("CN", "2019-01-01", "2019-12-31")


def contract(clock, absent=("2019-04-29", "2019-04-30")):
    calendar = dict(
        start=clock.days[0].isoformat(),
        end=clock.days[-1].isoformat(),
        decisions_sha256=calendar_digest(clock),
    )
    return price_window_contract({"CN": calendar}, {"CN": list(absent)})


def test_contract_declares_every_field_and_rejects_any_change(clock):
    declared = contract(clock)
    assert declared["channels"] == list(PRICE_WINDOW_CHANNELS)
    assert declared["absent_fill"] == 0.0
    assert declared["market_absent_sessions"] == {"CN": ["2019-04-29", "2019-04-30"]}
    assert check_price_window_contract(declared) is declared
    for key, value in [
        ("absent_fill", float("nan")),
        ("context_sessions", 32),
        ("anchor", "close_of_first_session"),
        ("volume_scale", "mean_over_all_sessions"),
        ("version", 2),
    ]:
        with pytest.raises(ValueError):
            check_price_window_contract({**declared, key: value})
    with pytest.raises(ValueError):
        check_price_window_contract({**declared, "extra": 1})
    with pytest.raises(ValueError):
        check_price_window_contract(
            {**declared, "market_absent_sessions": {"CN": ["2019-04-30", "2019-04-29"]}}
        )
    with pytest.raises(ValueError):
        check_price_window_contract({**declared, "market_absent_sessions": {"US": []}})


def test_market_absences_are_sessions_without_rows_after_the_first_observed_one(clock):
    days = [d.isoformat() for d in clock.days]
    observed = set(days[5:]) - {"2019-04-29", "2019-04-30"}
    # Antes de la primera fila del mercado no hay ausencias, sino falta de cobertura.
    assert market_absent_sessions(clock, observed) == ["2019-04-29", "2019-04-30"]
    assert market_absent_sessions(clock, observed, last="2019-04-29") == ["2019-04-29"]
    with pytest.raises(ValueError):
        market_absent_sessions(clock, set())


def test_rows_cannot_fall_on_a_declared_market_absence(clock):
    absent = absent_positions(clock, ["2019-04-29", "2019-04-30"])
    assert [clock.days[p].isoformat() for p in absent] == ["2019-04-29", "2019-04-30"]
    with pytest.raises(ValueError, match="sesión del calendario"):
        absent_positions(clock, ["2019-05-01"])
    rows = np.arange(len(clock.days))
    with pytest.raises(ValueError, match="ausente en todo el mercado"):
        check_row_positions(rows, absent)
    with pytest.raises(ValueError, match="crecientes"):
        check_row_positions(np.array([3, 2]), absent)


def test_only_market_absences_open_a_window_and_the_slots_end_at_the_decision(clock):
    absent = absent_positions(clock, ["2019-04-29", "2019-04-30"])
    rows = np.setdiff1d(np.arange(len(clock.days)), absent)
    gap = int(absent[0])
    # Ventana completa antes del hueco: todas las sesiones presentes.
    end = int(np.searchsorted(rows, gap - 1))
    assert present_slots(rows, end, 64, absent).all()
    # Ventana que termina el 6 de mayo: los dos huecos son las posiciones 61 y 62.
    end = int(np.searchsorted(rows, gap + 2))
    slots = present_slots(rows, end, 64, absent)
    assert slots.sum() == 62 and not slots[61] and not slots[62] and slots[63]
    built = window_rows(rows, np.array([end]), 64, absent)[0]
    assert (built >= 0).tolist() == slots.tolist()
    assert built[built >= 0].tolist() == list(range(end - 61, end + 1))
    # La ventana que empieza en el hueco también se admite.
    first = int(np.searchsorted(rows, gap + 64))
    slots = present_slots(rows, first, 64, absent)
    assert slots is not None and slots.sum() == 63 and not slots[0]
    # Un hueco propio del activo sigue excluyendo la ventana.
    own = np.delete(rows, 120)
    end = int(np.searchsorted(own, 130))
    assert present_slots(own, end, 64, absent) is None
    with pytest.raises(ValueError, match="solo para este activo"):
        window_rows(own, np.array([end]), 64, absent)
    assert present_slots(rows, 10, 64, absent) is None
    with pytest.raises(ValueError, match="antes del calendario"):
        window_rows(rows, np.array([10]), 64, absent)


def test_future_rows_and_absences_do_not_change_a_window(clock):
    absent = absent_positions(clock, ["2019-04-29", "2019-04-30"])
    rows = np.setdiff1d(np.arange(len(clock.days)), absent)
    end = int(np.searchsorted(rows, int(absent[0]) + 10))
    before = window_rows(rows, np.array([end]), 64, absent)
    later = np.setdiff1d(rows, rows[end + 5 :])
    extra = np.append(absent, rows[end + 3])
    assert np.array_equal(window_rows(later, np.array([end]), 64, absent), before)
    assert np.array_equal(window_rows(np.delete(rows, end + 3), np.array([end]), 64, extra), before)


def windows(present):
    generator = np.random.default_rng(3)
    values = generator.normal(size=(3, 64, 6)).astype(np.float32)
    values[..., 5] = present
    values[..., :5] = np.where(values[..., 5:] > 0, values[..., :5], 0)
    return values


@pytest.mark.parametrize("fill", [1.5, -2.0, 1e6, -0.0])
def test_gate_returns_exact_positive_zero_for_any_fill_and_keeps_observed_values(fill):
    present = np.ones((3, 64), dtype=np.float32)
    present[1, [10, 11]] = 0
    present[2, 0] = 0
    clean = windows(present)
    noisy = clean.copy()
    noisy[present == 0, :5] = fill
    for gated in (gate_price_window(noisy), gate_price_window(torch.from_numpy(noisy)).numpy()):
        assert gated.dtype == np.float32
        assert gated.view(np.uint32).tolist() == gate_price_window(clean).view(np.uint32).tolist()
        assert not np.signbit(gated[present == 0]).any()
    legacy = clean[..., :5].copy()
    assert gate_price_window(legacy) is legacy
    with pytest.raises(ValueError):
        gate_price_window(np.zeros((1, 64, 7), dtype=np.float32))


def test_gate_sends_no_gradient_to_the_fill():
    present = torch.ones(2, 64)
    present[0, 7] = 0
    values = torch.randn(2, 64, 5, dtype=torch.float64, requires_grad=True)
    prices = torch.cat([values, present.unsqueeze(-1).double()], -1)
    gate_price_window(prices).sum().backward()
    assert values.grad[0, 7].abs().sum() == 0
    assert (values.grad[1] == 1).all()


def mark_absent(steps):
    """Marcar pasos como huecos con su relleno correcto, para aislar cada regla."""

    def change(window):
        window[0, steps, :] = 0.0

    return change


@pytest.mark.parametrize(
    "change",
    [
        lambda w: w.__setitem__((0, 3, 5), 0.5),
        mark_absent(63),
        lambda w: w.__setitem__((0, 3, slice(None)), np.array([0, 0, 0, 1e-9, 0, 0])),
        lambda w: w.__setitem__((0, 3, slice(None)), np.array([0, 0, -0.0, 0, 0, 0])),
        mark_absent(slice(0, 63)),
    ],
)
def test_cpu_validator_rejects_a_bad_bit_or_a_nonzero_fill(change):
    present = np.ones((3, 64), dtype=np.float32)
    present[0, 3] = 0
    values = windows(present)
    validate_price_window(values)
    change(values)
    with pytest.raises(ValueError, match="presencia"):
        validate_price_window(values)


def test_chart_leaves_an_absent_session_empty_and_keeps_full_windows_identical():
    generator = np.random.default_rng(5)
    low = generator.uniform(10, 11, 70)
    ohlc = np.column_stack([low + 0.5, low + 1, low, low + 0.7])
    full = chart_png(ohlc, end_index=69, context=64)
    assert chart_png(ohlc, end_index=69, context=64, present=np.ones(64, dtype=bool)) == full
    present = np.ones(64, dtype=bool)
    present[[20, 21]] = False
    gapped = chart_png(ohlc, end_index=69, context=64, present=present)
    assert gapped != full
    from io import BytesIO

    from PIL import Image

    pixels = np.asarray(Image.open(BytesIO(gapped)))
    for slot in (20, 21):
        x = round(12 + slot * 200 / 63)
        assert (pixels[:, x] == pixels[0, 0]).all()
    for bad in (present[:-1], ~np.eye(64, dtype=bool)[63], present.astype(int)):
        with pytest.raises(ValueError):
            chart_png(ohlc, end_index=69, context=64, present=bad)
