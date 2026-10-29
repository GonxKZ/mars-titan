"""Regla y estado del puente hebbiano de tres factores (#784, P1), sin optimizador."""

import math
import random

import numpy as np
import pytest
import torch

from mars_titan.memory.three_factor_bridge import BridgePlasticity
from mars_titan.models.titans.plastic_bridge import (
    MATURE_DIRECTIONAL_HIT,
    MEAN_THIRD_FACTOR,
    THREE_FACTOR_BRIDGE,
    ThreeFactorBridge,
    parse_three_factor_bridge,
)

# Tres áreas: la 0 recibe de la 1 y la 2, y las dos le devuelven su arista.
EDGES = ((0, 1), (0, 2), (1, 0), (2, 0))
FIRST = [0.5, 0.25, 0.0, 1.0]
SECOND = [0.25, 0.5, 0.5, 0.0]


def bridge(kind=THREE_FACTOR_BRIDGE, **changes):
    return BridgePlasticity(ThreeFactorBridge(kind=kind, **changes), EDGES, 3)


def held(plastic, rows):
    """Guardar trazas [(flujo, decisión, traza)] como predicciones emitidas."""
    plastic.hold([(flow, at) for flow, at, _ in rows], np.array([t for _, _, t in rows]))


def opening(plastic):
    """Primer evento de US: fija la expectativa en 2 sin mover los pesos."""
    held(plastic, [("US/A", 1, FIRST), ("US/B", 1, SECOND)])
    plastic.settle([("US/A", 1, 0.5, -0.5), ("US/B", 1, 1.0, 4.0)])


def test_the_first_event_only_sets_the_expectation():
    plastic = bridge()
    opening(plastic)
    state = plastic._markets["US"]
    assert not state["weights"].any()
    assert state["expected"] == 2.0
    assert (state["delta_sum"], state["delta_count"], state["events"]) == (0.0, 2, 1)
    assert plastic.keys() == []


def test_a_later_event_follows_the_rule_written_by_hand():
    plastic = bridge(learning_rate=0.5)
    opening(plastic)
    held(plastic, [("US/A", 2, FIRST), ("US/B", 2, SECOND)])
    # ε = 1 da δ = (2 − 1) / 2 = 0,5 y ε = 3 da δ = −0,5.
    plastic.settle([("US/A", 2, 1.0, 0.0), ("US/B", 2, -1.0, 2.0)])
    state = plastic._markets["US"]
    # η · (0,5 · FIRST − 0,5 · SECOND) / 2, sin escalar porque ningún área pasa de 1.
    assert state["weights"].tolist() == [0.03125, -0.03125, -0.0625, 0.125]
    decay = math.exp(-1 / 21)
    assert state["expected"] == decay * 2.0 + (1.0 - decay) * 2.0
    assert (state["delta_sum"], state["delta_count"], state["events"]) == (0.0, 4, 2)
    assert plastic.biases(["US/Z", "CA/A"]).tolist() == [state["weights"].tolist(), [0.0] * 4]


def test_the_homeostasis_scales_the_edges_of_each_area_in_proportion():
    plastic = bridge(learning_rate=1.0, budget=0.1)
    opening(plastic)
    held(plastic, [("US/A", 2, [0.6, 0.3, 0.05, 0.02])])
    plastic.settle([("US/A", 2, 0.0, 0.0)])
    weights = plastic._markets["US"]["weights"]
    # δ = 1: el área 0 recibe 0,6 + 0,3 y se escala a 0,1 conservando la proporción 2:1.
    assert math.isclose(weights[0] + weights[1], 0.1, rel_tol=1e-15)
    assert math.isclose(weights[0] / weights[1], 2.0, rel_tol=1e-14)
    # Las áreas 1 y 2 quedan por debajo de la cota y no cambian.
    assert weights[2:].tolist() == [0.05, 0.02]


def test_the_third_factor_is_clipped_and_zero_without_an_expectation():
    plastic = bridge(learning_rate=1.0, budget=4.0)
    opening(plastic)
    held(plastic, [("US/A", 2, [1.0, 0.0, 0.0, 0.0]), ("US/B", 2, [0.0, 1.0, 0.0, 0.0])])
    # ε = 10 da (2 − 10) / 2 = −4, que se recorta a −1. ε = 0 da δ = 1.
    plastic.settle([("US/A", 2, 10.0, 0.0), ("US/B", 2, 0.0, 0.0)])
    assert plastic._markets["US"]["weights"].tolist() == [-0.5, 0.5, 0.0, 0.0]
    exact = bridge(learning_rate=1.0)
    held(exact, [("US/A", 1, FIRST)])
    exact.settle([("US/A", 1, 1.0, 1.0)])
    # Con E = 0 el cociente no está definido y δ = 0.
    held(exact, [("US/A", 2, FIRST)])
    exact.settle([("US/A", 2, 3.0, 1.0)])
    assert exact._markets["US"]["expected"] > 0 and not exact._markets["US"]["weights"].any()


def test_the_control_replaces_each_third_factor_by_its_running_mean():
    plastic = bridge(MEAN_THIRD_FACTOR, learning_rate=0.5)
    opening(plastic)
    held(plastic, [("US/A", 2, FIRST), ("US/B", 2, SECOND)])
    # δ = 0,5 y 0,25. Con los dos ceros del primer evento, la media es 0,75 / 4.
    plastic.settle([("US/A", 2, 1.0, 0.0), ("US/B", 2, 1.5, 0.0)])
    mean = 0.75 / 4
    expected = [0.5 * (mean * (a + b) / 2) for a, b in zip(FIRST, SECOND, strict=True)]
    assert plastic._markets["US"]["weights"].tolist() == expected
    assert plastic._markets["US"]["delta_sum"] == 0.75


def common(kind=THREE_FACTOR_BRIDGE, **changes):
    config = ThreeFactorBridge(
        kind=kind, source=MATURE_DIRECTIONAL_HIT, time_constant=None, **changes
    )
    return BridgePlasticity(config, EDGES, 3)


def test_the_common_signal_gives_half_the_gap_between_the_hit_and_its_expectation():
    plastic = common(learning_rate=1.0, budget=4.0)
    held(
        plastic,
        [
            ("US/A", 1, [1.0, 0.0, 0.0, 0.0]),
            ("US/B", 1, [0.0, 1.0, 0.0, 0.0]),
            ("US/C", 1, [0.0, 0.0, 1.0, 0.0]),
            ("CA/A", 1, [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    # Acierto, fallo y predicción nula frente a un acierto esperado de 0,25 en US.
    plastic.settle(
        [
            ("US/A", 1, 0.2, 0.1),
            ("US/B", 1, 0.2, -0.1),
            ("US/C", 1, 0.0, 0.3),
            ("CA/A", 1, -0.4, -0.2),
        ],
        dict(US=0.25, CA=-0.5),
    )
    us, ca = plastic._markets["US"], plastic._markets["CA"]
    assert us["weights"].tolist() == [0.375 / 3, -0.625 / 3, -0.125 / 3, 0.0]
    assert ca["weights"].tolist() == [0.0, 0.0, 0.0, 0.75]
    # La expectativa de la señal no se guarda aquí, porque es la de #791.
    assert us["expected"] is None and us["events"] == 1
    restored = BridgePlasticity.restore(plastic.config, EDGES, 3, plastic.export())
    assert restored._markets["US"]["weights"].tolist() == us["weights"].tolist()


@pytest.mark.parametrize(
    "expectations",
    [None, dict(CA=0.0), dict(US=1.5), dict(US=1), dict(US=math.nan), [("US", 0.0)]],
)
def test_the_common_signal_needs_the_expected_hit_of_each_market(expectations):
    plastic = common()
    held(plastic, [("US/A", 1, FIRST)])
    with pytest.raises(ValueError, match="acierto esperado"):
        plastic.settle([("US/A", 1, 0.2, 0.1)], expectations)
    assert plastic.keys() == [("US/A", 1)] and plastic._markets == {}


def test_the_error_gap_does_not_take_the_common_signal():
    plastic = bridge()
    held(plastic, [("US/A", 1, FIRST)])
    with pytest.raises(ValueError, match="no recibe la señal común"):
        plastic.settle([("US/A", 1, 0.2, 0.1)], dict(US=0.0))


def test_a_restored_common_state_cannot_carry_an_error_expectation():
    plastic = common()
    held(plastic, [("US/A", 1, FIRST)])
    plastic.settle([("US/A", 1, 0.2, 0.1)], dict(US=0.0))
    payload = plastic.export()
    payload["markets"]["US"]["expected"] = 0.5
    with pytest.raises(ValueError, match="no corresponde al puente"):
        BridgePlasticity.restore(plastic.config, EDGES, 3, payload)


def reference_run(kind, events, rate=0.3, budget=0.2, tau=5):
    """La regla del módulo con bucles de Python, en el orden de las claves."""
    decay = math.exp(-1 / tau)
    markets = {}
    for labels in events:
        groups = {}
        for flow, _at, issued, outcome, trace in sorted(labels):
            groups.setdefault(flow.split("/")[0], []).append((abs(issued - outcome), trace))
        for market, rows in sorted(groups.items()):
            state = markets.setdefault(
                market, dict(w=[0.0] * len(EDGES), e=None, total=0.0, count=0)
            )
            expected = state["e"]
            deltas = [
                0.0
                if expected is None or expected == 0
                else min(1.0, max(-1.0, (expected - error) / expected))
                for error, _ in rows
            ]
            state["total"] = math.fsum([state["total"], *deltas])
            state["count"] += len(rows)
            mean = state["total"] / state["count"]
            weights = []
            for column in range(len(EDGES)):
                if kind == MEAN_THIRD_FACTOR:
                    step = mean * sum(trace[column] for _, trace in rows) / len(rows)
                else:
                    products = (
                        d * trace[column] for d, (_, trace) in zip(deltas, rows, strict=True)
                    )
                    step = sum(products) / len(rows)
                weights.append(state["w"][column] + rate * step)
            sums = [0.0] * 3
            for (target, _), value in zip(EDGES, weights, strict=True):
                sums[target] += abs(value)
            state["w"] = [
                value * (budget / sums[target] if sums[target] > budget else 1.0)
                for (target, _), value in zip(EDGES, weights, strict=True)
            ]
            error = math.fsum(error for error, _ in rows) / len(rows)
            state["e"] = error if expected is None else decay * expected + (1 - decay) * error
    return markets


def random_events(seed, count=6):
    generator = random.Random(seed)
    events = []
    for at in range(1, count + 1):
        labels = []
        for flow in ("US/A", "US/B", "US/C", "CA/A", "CA/B"):
            if generator.random() < 0.8:
                trace = [round(generator.random() / 2, 6) for _ in EDGES]
                issued, outcome = generator.gauss(0, 0.02), generator.gauss(0, 0.02)
                labels.append((flow, at, issued, outcome, trace))
        events.append(labels)
    return events


def run_events(plastic, events, *, shuffle=None):
    for labels in events:
        held(plastic, [(flow, at, trace) for flow, at, _, _, trace in labels])
        matured = [(flow, at, issued, outcome) for flow, at, issued, outcome, _ in labels]
        if shuffle is not None:
            shuffle.shuffle(matured)
        plastic.settle(matured)
    return plastic


@pytest.mark.parametrize("kind", [THREE_FACTOR_BRIDGE, MEAN_THIRD_FACTOR])
def test_random_events_match_the_reference_and_ignore_the_label_order(kind):
    events = random_events(5)
    plastic = run_events(bridge(kind, learning_rate=0.3, budget=0.2, time_constant=5), events)
    shuffled = run_events(
        bridge(kind, learning_rate=0.3, budget=0.2, time_constant=5),
        events,
        shuffle=random.Random(9),
    )
    reference = reference_run(kind, events)
    assert set(plastic._markets) == set(reference) == {"US", "CA"}
    for market, state in plastic._markets.items():
        np.testing.assert_allclose(state["weights"], reference[market]["w"], rtol=1e-13, atol=0)
        assert state["expected"] == reference[market]["e"]
        assert state["delta_sum"] == reference[market]["total"]
        assert state["weights"].tolist() == shuffled._markets[market]["weights"].tolist()
    assert any(state["weights"].any() for state in plastic._markets.values())


def test_a_market_does_not_move_the_weights_of_another():
    plastic = bridge(learning_rate=1.0)
    opening(plastic)
    held(plastic, [("US/A", 2, FIRST)])
    plastic.settle([("US/A", 2, 0.0, 0.0)])
    assert plastic.biases(["CA/A"]).tolist() == [[0.0] * 4]
    assert "CA" not in plastic._markets


def test_an_invalid_event_is_refused_before_changing_anything():
    plastic = bridge()
    held(plastic, [("US/A", 1, FIRST), ("US/B", 1, SECOND)])
    before = plastic.export()
    for matured, message in (
        ([("US/A", 1, 0.5, 0.0), ("US/C", 1, 0.5, 0.0)], "traza"),
        ([("US/A", 1, 0.5, 0.0), ("US/A", 1, 0.5, 0.0)], "una vez"),
        ([("US/A", 1, 0.5, 0.0), ("US/B", 1, math.nan, 0.0)], "emitida"),
        ([("US/A", 1, 0.5, 0.0), ("US/B", 1, 0.5, math.inf)], "resultado"),
        ([("US/A", 1, 0.5, 0.0), ("US/B", 1, 1e308, -1e308)], "error maduro"),
        ([("US/A", 1, 0.5, 0.0), ("US/B", 1, 1, 0.0)], "emitida"),
    ):
        with pytest.raises(ValueError, match=message):
            plastic.settle(matured)
        after = plastic.export()
        assert after["markets"] == before["markets"] == {}
        assert after["pending"]["flows"] == before["pending"]["flows"] == ["US/A", "US/B"]


@pytest.mark.parametrize(
    ("keys", "traces", "message"),
    [
        ([("US/A", 1)], [[0.5, 0.5, 0.5]], "todas las aristas"),
        ([("US/A", 1)], [[0.5, 0.5, 0.5, 1.5]], r"\[0, 1\]"),
        ([("US/A", 1)], [[0.5, 0.5, 0.5, -0.1]], r"\[0, 1\]"),
        ([("US/A", 1)], [[0.5, 0.5, 0.5, math.nan]], r"\[0, 1\]"),
        ([("US/A", 1), ("US/A", 1)], [FIRST, SECOND], "pendiente"),
        ([("US/C", 1), ("US/C", 1)], [FIRST, SECOND], "pendiente"),
        ([("US/B", 1), ("US/A", 1)], [FIRST, SECOND], "pendiente"),
        ([("US/C", 1.0)], [FIRST], "instante"),
        ([("US/C", 1, 2)], [FIRST], "instante"),
        ([("USC", 1)], [FIRST], "mercado"),
    ],
)
def test_a_trace_is_held_once_and_atomically(keys, traces, message):
    plastic = bridge()
    held(plastic, [("US/A", 1, FIRST)])
    with pytest.raises(ValueError, match=message):
        plastic.hold(keys, np.array(traces))
    assert plastic.keys() == [("US/A", 1)]


def test_the_state_keeps_its_own_copies_of_traces_and_weights():
    plastic = bridge(learning_rate=1.0)
    opening(plastic)
    held(plastic, [("US/A", 2, FIRST)])
    plastic.settle([("US/A", 2, 1.0, 0.0)])
    weights = plastic._markets["US"]["weights"].tolist()
    assert any(weights)
    traces = np.array([FIRST, SECOND])
    plastic.hold([("US/A", 3), ("US/B", 3)], traces)
    # Cambiar las trazas entregadas no cambia las guardadas.
    traces[:] = 0.0
    saved = plastic.export()
    assert saved["pending"]["traces"].tolist() == [FIRST, SECOND]
    restored = BridgePlasticity.restore(plastic.config, EDGES, 3, saved)
    # Cambiar lo exportado no cambia el puente ni el restaurado desde ello.
    saved["markets"]["US"]["weights"].add_(0.5)
    saved["pending"]["traces"].zero_()
    for state in (plastic, restored):
        assert state._markets["US"]["weights"].tolist() == weights
        assert state.export()["pending"]["traces"].tolist() == [FIRST, SECOND]


def test_resume_from_an_export_continues_bit_for_bit():
    events = random_events(11, count=8)
    continuous = run_events(bridge(learning_rate=0.4, budget=0.3), events)
    first = run_events(bridge(learning_rate=0.4, budget=0.3), events[:4])
    # Trazas emitidas que todavía no han madurado viajan con el estado.
    pending = [(flow, at, trace) for flow, at, _, _, trace in events[4]]
    held(first, pending)
    saved = first.export()
    assert saved["pending"]["traces"].shape == (len(pending), len(EDGES))
    second = BridgePlasticity.restore(
        ThreeFactorBridge(learning_rate=0.4, budget=0.3), EDGES, 3, saved
    )
    second.settle([(flow, at, issued, outcome) for flow, at, issued, outcome, _ in events[4]])
    run_events(second, events[5:])
    left, right = second.export(), continuous.export()
    assert left.keys() == right.keys()
    for market, state in right["markets"].items():
        assert torch.equal(left["markets"][market]["weights"], state["weights"])
        assert {k: v for k, v in left["markets"][market].items() if k != "weights"} == {
            k: v for k, v in state.items() if k != "weights"
        }


def corrupted(change):
    plastic = run_events(bridge(), random_events(3, count=3))
    held(plastic, [("US/A", 9, FIRST), ("US/B", 9, SECOND)])
    payload = plastic.export()
    change(payload)
    return payload


def _set(path, value):
    def change(payload):
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


FOREIGN = "estado guardado no corresponde"
MARKET = "mercado guardado no corresponde"
PENDING = "pendientes guardadas no corresponden"


# Cada corrupción la detecta su propia comprobación, no un error posterior de otra.
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set(("configuration", "budget"), 2.0), FOREIGN),
        (_set(("areas",), 4), FOREIGN),
        (_set(("edges",), [[0, 1], [0, 2], [1, 0]]), FOREIGN),
        (_set(("schema_version",), 2), FOREIGN),
        (_set(("extra",), None), FOREIGN),
        (_set(("markets",), []), FOREIGN),
        (_set(("markets", "US", "weights"), torch.zeros(4, dtype=torch.float32)), MARKET),
        (_set(("markets", "US", "weights"), torch.zeros(3, dtype=torch.float64)), MARKET),
        (_set(("markets", "US", "weights"), [0.0] * 4), MARKET),
        (
            _set(("markets", "US", "weights"), torch.tensor([math.nan, 0, 0, 0]).double()),
            MARKET,
        ),
        (_set(("markets", "US", "weights"), torch.tensor([0.9, 0.9, 0, 0]).double()), "cota"),
        (_set(("markets", "US", "events"), 0), MARKET),
        (_set(("markets", "US", "events"), 1.0), MARKET),
        (_set(("markets", "US", "expected"), None), MARKET),
        (_set(("markets", "US", "expected"), -1.0), "expectativa guardada"),
        (_set(("markets", "US", "expected"), math.inf), "expectativa guardada"),
        (_set(("markets", "US", "delta_sum"), 99.0), "suma de δ"),
        (_set(("markets", "US", "delta_sum"), math.nan), "suma de δ"),
        (_set(("markets", "US", "delta_count"), 0), MARKET),
        (_set(("markets", "US", "delta_count"), 99.0), MARKET),
        (_set(("markets", "US", "extra"), 1), "mercado guardado no conserva"),
        (_set(("markets", "US/A"), {}), "nombre válido"),
        (_set(("pending", "flows"), ["US/B", "US/A"]), "ordenadas"),
        (_set(("pending", "flows"), ("US/A", "US/B")), PENDING),
        (_set(("pending", "traces"), torch.zeros(2, 4, dtype=torch.float32)), PENDING),
        (_set(("pending", "traces"), torch.zeros(2, 3, dtype=torch.float64)), PENDING),
        (_set(("pending", "traces"), [[0.0] * 4] * 2), PENDING),
        (_set(("pending", "decision_at"), torch.tensor([9], dtype=torch.int64)), PENDING),
        (_set(("pending", "decision_at"), torch.tensor([9.0, 9.0])), PENDING),
        (_set(("pending", "decision_at"), [9, 9]), PENDING),
        (_set(("pending", "extra"), []), "pendientes guardadas no conservan"),
    ],
)
def test_restore_refuses_corrupted_or_foreign_states(change, message):
    with pytest.raises(ValueError, match=message):
        BridgePlasticity.restore(ThreeFactorBridge(), EDGES, 3, corrupted(change))


def test_a_restored_state_without_markets_or_pending_traces_is_empty():
    plastic = BridgePlasticity.restore(ThreeFactorBridge(), EDGES, 3, bridge().export())
    assert plastic.keys() == [] and plastic._markets == {}


@pytest.mark.parametrize(
    ("edges", "areas"),
    [
        ((), 3),
        (((0, 0),), 3),
        (((0, 3),), 3),
        (((1, 0), (0, 1)), 3),
        (((0, 1), (0, 1)), 3),
        ([(0, 1)], 3),
        (((0, 1),), 1),
        (((0, True),), 3),
        (((0, 1),), 3.0),
    ],
)
def test_the_edges_must_be_ordered_pairs_of_distinct_areas(edges, areas):
    with pytest.raises(ValueError, match="aristas"):
        BridgePlasticity(ThreeFactorBridge(), edges, areas)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(kind="hebbian"), "tipo"),
        (dict(learning_rate=0.0), "tasa"),
        (dict(learning_rate=1.5), "tasa"),
        (dict(learning_rate=True), "tasa"),
        (dict(learning_rate=math.nan), "tasa"),
        (dict(budget=0.0), "cota"),
        (dict(budget=4.5), "cota"),
        (dict(time_constant=0), "constante"),
        (dict(time_constant=253), "constante"),
        (dict(time_constant=5.0), "constante"),
        (dict(time_constant=None), "constante"),
        (dict(source="hit"), "fuente"),
        (dict(source=MATURE_DIRECTIONAL_HIT), "None"),
    ],
)
def test_invalid_declarations_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        ThreeFactorBridge(**changes)


def test_the_declaration_parses_whole_and_keeps_its_identity():
    config = ThreeFactorBridge(learning_rate=1, budget=2, time_constant=10)
    assert (config.learning_rate, config.budget) == (1.0, 2.0)
    assert type(config.learning_rate) is float and type(config.budget) is float
    assert parse_three_factor_bridge(None) is None
    assert parse_three_factor_bridge(config) is config
    values = dict(
        kind=MEAN_THIRD_FACTOR,
        source="relative_error_gap",
        learning_rate=0.1,
        budget=0.5,
        time_constant=3,
    )
    assert parse_three_factor_bridge(values) == ThreeFactorBridge(**values)
    for raw in (dict(kind=MEAN_THIRD_FACTOR), [], "three_factor_bridge_v1"):
        with pytest.raises(ValueError, match="todos sus campos"):
            parse_three_factor_bridge(raw)
    identity = config.identity()
    assert identity["name"] == THREE_FACTOR_BRIDGE and identity["role"] == "variant"
    control = ThreeFactorBridge(kind=MEAN_THIRD_FACTOR).identity()
    assert control["role"] == "mean_third_factor_control"
    assert control["credit"] == "running_mean_of_the_market_third_factor"
    shared = common().config.identity()
    assert shared["source"] == MATURE_DIRECTIONAL_HIT and shared["time_constant"] is None
    assert (
        shared["expectation"] == "slowest_speed_of_the_mature_reliability_signal_before_the_event"
    )
    assert identity != ThreeFactorBridge().identity()
    assert config.decay == math.exp(-1 / 10)
