"""La medida de coste de los árbitros recorre cada modo con paneles pequeños y no ajusta nada."""

import numpy as np

from benchmarks.online_arbiters_cost import MODES, SHAPES, call, inputs, measure, protocol


def test_the_measure_covers_every_mode_and_its_memory():
    shapes = ((8, 5), (8, 32))
    result = measure(shapes, warmup=0, repetitions=1)
    expected = {(mode, s, o) for s, o in shapes for mode in MODES[o]}
    assert {(r["mode"], r["sessions"], r["options"]) for r in result["summary"]} == expected
    assert all(r["p50_s"] > 0 and r["p95_s"] >= r["p50_s"] for r in result["summary"])
    assert {(r["mode"], r["sessions"], r["options"]) for r in result["memory"]} == expected
    assert all(r["peak_bytes"] > 0 for r in result["memory"])
    # Cada modo se mide en los dos órdenes.
    assert len(result["records"]) == 2 * len(expected)


def test_the_measured_calls_repeat_their_result():
    data = inputs(8, 5)
    for mode in MODES[5]:
        first, second = call(mode, *data), call(mode, *data)
        if isinstance(first, dict):
            np.testing.assert_array_equal(first["prediction"], second["prediction"])
            np.testing.assert_array_equal(first["mixtures"], second["mixtures"])
        else:
            np.testing.assert_array_equal(first, second)


def test_the_protocol_declares_the_measured_shapes():
    declared = protocol()["shapes"]
    assert [(s["sessions_per_market"], s["options"]) for s in declared] == list(SHAPES)
    assert all(tuple(s["modes"]) == MODES[s["options"]] for s in declared)
