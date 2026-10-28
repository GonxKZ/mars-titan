"""Acumulador de la fiabilidad madura de #791: fórmula, causalidad, controles y recuperación."""

import copy
import math

import pytest

from mars_titan.memory.mature_reliability import ReliabilitySignal
from mars_titan.models.titans.reliability_gates import MatureReliability

FLOWS = ("US/A", "US/B", "US/C", "US/D")


def expected_average(scores, config):
    """Promedios de referencia escritos aparte: v ← λ v + (1 − λ) s, con v = 0 al empezar."""
    values = [0.0] * len(config.time_constants)
    for score in scores:
        for index, constant in enumerate(config.time_constants):
            decay = math.exp(-1.0 / constant)
            values[index] = decay * values[index] + (1.0 - decay) * score
    return values


def settled(config, events):
    signal = ReliabilitySignal(config)
    for event in events:
        signal.settle(event)
    return signal


EVENTS = [
    [("US/A", 0.2, 0.1), ("US/B", -0.3, 0.4), ("US/C", 0.5, -0.2)],
    [("US/A", 0.1, 0.3), ("US/B", -0.2, -0.1)],
    [("US/D", 0.0, 0.7), ("US/A", -0.4, -0.6), ("US/C", 0.3, 0.0)],
]
# Aciertos direccionales de cada evento: la media de sign(ŷ) · sign(y).
SCORES = [-1 / 3, 1.0, 1 / 3]


def test_each_speed_averages_the_directional_hits_of_the_matured_labels():
    config = MatureReliability()
    signal = settled(config, EVENTS)
    expected = expected_average(SCORES, config)
    rows = signal.values(("US/X", "US/A"))
    assert rows == [[expected, expected], [expected, expected]]
    assert config.decays == tuple(math.exp(-1.0 / value) for value in (1, 5, 21))
    # Antes de la primera etiqueta la señal es neutral.
    assert ReliabilitySignal(config).values(FLOWS[:1]) == [[[0.0] * 3, [0.0] * 3]]


def test_the_signal_does_not_depend_on_the_order_or_split_of_the_labels():
    config = MatureReliability()
    many = [(f"US/F{i}", 0.1 * (i - 5) + 0.05, (-1) ** i * 0.3) for i in range(11)]
    forward = settled(config, [many])
    backward = settled(config, [list(reversed(many))])
    assert forward.values(FLOWS[:1]) == backward.values(FLOWS[:1])
    assert forward.export() == backward.export()


def test_markets_keep_separate_signals_and_events_without_labels_do_not_move_them():
    config = MatureReliability(time_constants=(1, 5))
    signal = settled(config, [[("US/A", 0.1, 0.2), ("CN/A", 0.1, -0.2)]])
    us, cn = signal.values(("US/Z", "CN/Z"))
    assert us[0] == expected_average([1.0], config)
    assert cn[0] == expected_average([-1.0], config)
    before = signal.export()
    signal.settle([])
    signal.settle([("JP/A", 0.1, 0.1)])
    after = signal.export()
    assert after["markets"]["US"] == before["markets"]["US"]
    assert after["markets"]["CN"] == before["markets"]["CN"]
    assert after["markets"]["JP"][0]["updates"] == 1


def test_the_shuffled_control_draws_each_input_from_the_scores_seen_so_far():
    config = MatureReliability(source="shuffled_in_time", seed=3, time_constants=(1,))
    scores = [-1.0, -0.5, 0.0, 0.5, 1.0, 1 / 3, -1 / 3]
    signal = ReliabilitySignal(config)
    inputs = []
    decay = math.exp(-1.0)
    for count, score in enumerate(scores):
        before = signal.values(FLOWS[:1])[0][0][0]
        # Una etiqueta por evento con la puntuación pedida: ŷ = 1 y el signo de y.
        signal._update("US", 0, score)
        after = signal.values(FLOWS[:1])[0][0][0]
        drawn = (after - decay * before) / (1.0 - decay)
        inputs.append(min(scores[: count + 1], key=lambda value, x=drawn: abs(value - x)))
        assert inputs[-1] == pytest.approx(drawn, abs=1e-12)
    assert inputs[0] == scores[0]
    again = ReliabilitySignal(config)
    other = ReliabilitySignal(
        MatureReliability(source="shuffled_in_time", seed=4, time_constants=(1,))
    )
    for score in scores:
        again._update("US", 0, score)
        other._update("US", 0, score)
    assert again.export()["markets"] == signal.export()["markets"]
    assert (
        other.export()["markets"]["US"][0]["values"]
        != signal.export()["markets"]["US"][0]["values"]
    )
    assert signal.export()["markets"]["US"][0]["history"] == scores


def test_the_surprise_control_compares_each_event_with_its_slowest_average():
    config = MatureReliability(source="titans_surprise", time_constants=(1, 5, 21))
    signal = ReliabilitySignal(config)
    # La sorpresa no espera a ninguna etiqueta: las maduras no la cambian.
    signal.settle(EVENTS[0])
    assert signal.export()["markets"] == {}
    slowest = math.exp(-1.0 / 21)
    reference, scores = None, []
    for losses in ([0.4, 0.2], [0.1, 0.1], [0.9, 0.3], [0.0, 0.0]):
        signal.observe(("US/A", "US/B"), losses)
        mean = math.fsum(losses) / 2
        reference = mean if reference is None else reference
        scores.append(0.0 if reference + mean == 0 else (reference - mean) / (reference + mean))
        reference = slowest * reference + (1 - slowest) * mean
    channel = signal.export()["markets"]["US"][0]
    assert scores[0] == 0.0 and scores[1] > 0 > scores[2]
    assert channel["values"] == expected_average(scores, config)
    assert channel["reference"] == reference
    zero = ReliabilitySignal(config)
    zero.observe(("US/A",), [0.0])
    zero.observe(("US/A",), [0.0])
    assert zero.values(("US/A",))[0][0] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("losses", [[-0.1], [math.nan], [math.inf], [1]])
def test_the_surprise_control_rejects_invalid_losses(losses):
    signal = ReliabilitySignal(MatureReliability(source="titans_surprise"))
    with pytest.raises(ValueError, match="pérdida asociativa"):
        signal.observe(("US/A",), losses)


def test_only_the_surprise_control_observes_losses_and_each_row_needs_one():
    with pytest.raises(ValueError, match="Solo el control"):
        ReliabilitySignal(MatureReliability()).observe(("US/A",), [0.1])
    with pytest.raises(ValueError, match="Cada fila"):
        ReliabilitySignal(MatureReliability(source="titans_surprise")).observe(("US/A",), [])


def test_independent_signals_split_the_flows_between_the_consumers():
    config = MatureReliability(independent=True, seed=11)
    signal = ReliabilitySignal(config)
    flows = [f"US/F{i}" for i in range(40)]
    channels = {flow: signal.channel(flow) for flow in flows}
    assert set(channels.values()) == {0, 1}
    # El reparto depende de la semilla y no del orden de llegada.
    other = ReliabilitySignal(MatureReliability(independent=True, seed=12))
    assert channels != {flow: other.channel(flow) for flow in flows}
    event = [(flow, 1.0, 1.0 if channels[flow] == 0 else -1.0) for flow in flows]
    signal.settle(event)
    write, forget = signal.values(("US/Z",))[0]
    assert write == expected_average([1.0], config)
    assert forget == expected_average([-1.0], config)
    shared = settled(MatureReliability(seed=11), [event])
    first, second = shared.values(("US/Z",))[0]
    assert first == second and shared.channel(flows[1]) == 0


@pytest.mark.parametrize(
    "row",
    [
        ("US/A", 1, 0.5),
        ("US/A", 0.5, True),
        ("US/A", math.nan, 0.5),
        ("US/A", 0.5, -math.inf),
        ("A", 0.5, 0.5),
        (3, 0.5, 0.5),
    ],
)
def test_the_signal_rejects_labels_that_are_not_finite_reals_of_a_flow(row):
    with pytest.raises(ValueError, match="real finito|flujo"):
        ReliabilitySignal(MatureReliability()).settle([row])


@pytest.mark.parametrize(
    "config",
    [
        MatureReliability(),
        MatureReliability(source="shuffled_in_time", seed=5),
        MatureReliability(source="titans_surprise"),
        MatureReliability(independent=True, seed=9, consumers=("write", "forget")),
    ],
    ids=["mature", "shuffled", "surprise", "independent"],
)
def test_a_restored_signal_continues_exactly_like_the_original(config):
    signal = ReliabilitySignal(config)
    feed = (
        (lambda s, e: s.observe([f for f, _, _ in e], [abs(v) for _, v, _ in e]))
        if config.source == "titans_surprise"
        else (lambda s, e: s.settle(e))
    )
    for event in EVENTS[:2]:
        feed(signal, event)
    restored = ReliabilitySignal.restore(config, copy.deepcopy(signal.export()))
    assert restored.export() == signal.export()
    feed(signal, EVENTS[2])
    feed(restored, EVENTS[2])
    assert restored.export() == signal.export()
    assert restored.values(FLOWS) == signal.values(FLOWS)


def _payload(config=None, **change):
    config = config or MatureReliability(source="shuffled_in_time", seed=2)
    signal = settled(config, EVENTS[:2])
    payload = signal.export()
    channel = payload["markets"]["US"][0]
    for key, value in change.items():
        if key in channel:
            channel[key] = value
        else:
            payload[key] = value
    return config, payload


@pytest.mark.parametrize(
    "change",
    [
        dict(schema_version=2),
        dict(configuration={}),
        dict(markets=[]),
        dict(values=[0.0, 0.0]),
        dict(values=[0.0, 2.0, 0.0]),
        dict(values=[0, 0.0, 0.0]),
        dict(updates=-1),
        dict(updates=True),
        dict(history=[1.0]),
        dict(reference=0.5),
        dict(extra=1),
    ],
    ids=lambda change: next(iter(change)),
)
def test_restore_rejects_a_state_that_does_not_belong_to_the_signal(change):
    config, payload = _payload(**change)
    with pytest.raises(ValueError, match="corresponde|conserva"):
        ReliabilitySignal.restore(config, payload)


def test_restore_rejects_markets_channels_and_references_out_of_contract():
    config, payload = _payload()
    payload["markets"][""] = payload["markets"]["US"]
    with pytest.raises(ValueError, match="mercado guardado"):
        ReliabilitySignal.restore(config, payload)
    config, payload = _payload()
    payload["markets"]["US"].append(payload["markets"]["US"][0])
    with pytest.raises(ValueError, match="mercado guardado"):
        ReliabilitySignal.restore(config, payload)
    surprise = MatureReliability(source="titans_surprise")
    signal = ReliabilitySignal(surprise)
    signal.observe(("US/A",), [0.2])
    for reference, message in ((None, "corresponde"), (-0.1, "finita"), (math.nan, "finita")):
        payload = signal.export()
        payload["markets"]["US"][0]["reference"] = reference
        with pytest.raises(ValueError, match=message):
            ReliabilitySignal.restore(surprise, payload)
    with pytest.raises(ValueError, match="configuración"):
        ReliabilitySignal("shared")
