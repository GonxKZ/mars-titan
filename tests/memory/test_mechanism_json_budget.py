"""Tamaño canónico y límites previos a la materialización de trazas."""

import json
import math

import pytest

from mars_titan.memory.mechanism_history import (
    HistoryCapacityError,
    _json_bytes,
    _json_only,
)


def encoded_size(value):
    return len(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        -(2**63),
        2**63 - 1,
        -0.0,
        math.nextafter(0.0, 1.0),
        float.fromhex("0x1.fffffffffffffp+1023"),
        "",
        "😀" * 100,
        '\ud800\n\t"\\',
        [],
        (),
        {},
        {"b": [False, None, (1, -0.0)], "a": "ñ😀"},
    ],
)
def test_preflight_bound_covers_canonical_ascii_and_exact_limit(value):
    size = encoded_size(value)
    bound = _json_only(value, [1_000_000, max(size, 1)])
    assert type(bound) is int and bound >= size
    assert _json_bytes(value, size) == size
    with pytest.raises(HistoryCapacityError):
        _json_bytes(value, size - 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**63, -(2**63) - 1, b"a", {1: 2}])
def test_preflight_does_not_serialize_invalid_types_or_numbers(value):
    with pytest.raises(ValueError):
        _json_only(value, [1_000_000, 4096])
    with pytest.raises(ValueError):
        _json_bytes(value, 4096)


def test_preflight_enforces_element_depth_and_string_budgets():
    with pytest.raises(HistoryCapacityError):
        _json_only([0, 1], [2, 4096])
    with pytest.raises(HistoryCapacityError):
        _json_only("abcd", [10, 3])
    value = None
    for _ in range(17):
        value = [value]
    with pytest.raises(ValueError):
        _json_only(value, [100, 4096])


def test_ambiguous_upper_bound_keeps_streaming_without_unbounded_encode(monkeypatch):
    # El límite exacto admite ASCII aunque la cota conservadora sea mayor.
    value = ["a" * 128] * 20
    size = encoded_size(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("No materializar una traza sin cota previa suficiente")

    monkeypatch.setattr(json.JSONEncoder, "encode", forbidden)
    assert _json_bytes(value, size) == size
    with pytest.raises(HistoryCapacityError):
        _json_bytes(value, size - 1)


def test_nested_values_preserve_exact_json_size_across_both_paths():
    values = [
        dict(id=index, weights=[0.125, 0.25, 0.625], view=("precio", "😀", index * -1.25))
        for index in range(128)
    ]
    size = encoded_size(values)
    assert _json_bytes(values, size) == size
    assert _json_bytes(values, size * 32) == size
