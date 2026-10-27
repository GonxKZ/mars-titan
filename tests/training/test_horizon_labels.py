"""Registro de etiquetas auxiliares de F5 (#749) con flujos sintéticos y sin modelos."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mars_titan.data.input_policy import MODALITIES
from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.training.horizon_labels import HorizonLabels

CONFIG = AuxiliaryHorizons(horizons=(2, 3), weight=1.0, maturity_slack=1)


def block(flows, at):
    """Bloque CPU mínimo. Los precios valen el instante, más 100 en los flujos B."""
    rows = len(flows)
    first = np.asarray([at + 100 * flow.endswith("B") for flow in flows], dtype=np.float32)
    inputs = {
        name: np.repeat(first[:, None] + index, 2, axis=1) for index, name in enumerate(MODALITIES)
    }
    presence = np.ones((rows, 5), dtype=np.bool_)
    return SimpleNamespace(
        flow_ids=tuple(flows), prediction_at=(at,) * rows, inputs=inputs, presence=presence
    )


def emit(ledger, flows, at):
    ledger.observe(flow.split("/", 1)[0] for flow in flows)
    ledger.remember(block(flows, at))


def ready(ledger):
    """(decisión identificada por sus precios, cabeza, objetivo) de cada etiqueta entregada."""
    return sorted(
        (int(values["prices"][0]), head, target)
        for values, heads, targets in ledger.take()
        for head, target in zip(heads, targets, strict=True)
    )


def test_labels_wait_for_the_longest_window_and_share_one_delivery():
    ledger = HorizonLabels(CONFIG)
    # Como en el recorrido, cada evento entrega primero la etiqueta anterior y después emite.
    emit(ledger, ["US/A"], 0)
    assert ledger.mature("US/A", 0, 0.5) == 0
    emit(ledger, ["US/A"], 1)
    # La ventana de dos sesiones desde 0 madura, pero espera a la de tres de la misma decisión.
    assert ledger.mature("US/A", 1, 0.25) == 1
    assert ready(ledger) == []
    emit(ledger, ["US/A"], 2)
    assert ledger.mature("US/A", 2, 1.0) == 2
    # Una sola entrega para la decisión 0 con sus dos horizontes. La de 1 sigue esperando.
    taken = ledger.take()
    assert len(taken) == 1 and taken[0][1] == [0, 1] and taken[0][2] == [0.75, 1.75]
    assert taken[0][0]["prices"][0] == 0
    assert ledger.mature("US/A", 2, 1.0) == 0
    ledger.flush()
    assert ready(ledger) == [(1, 0, 1.25)]
    ledger.flush()
    assert ready(ledger) == []


def test_labels_arriving_out_of_order_complete_each_window_once():
    ledger = HorizonLabels(CONFIG)
    for at in range(3):
        emit(ledger, ["US/A"], at)
    assert ledger.mature("US/A", 1, 2.0) == 0
    assert ledger.mature("US/A", 2, 4.0) == 1
    assert ledger.mature("US/A", 0, 1.0) == 2
    ledger.flush()
    assert ready(ledger) == [(0, 0, 3.0), (0, 1, 7.0), (1, 0, 6.0)]


def test_a_missing_decision_breaks_every_window_that_crosses_it():
    ledger = HorizonLabels(CONFIG)
    emit(ledger, ["US/A", "US/B"], 0)
    emit(ledger, ["US/B"], 1)
    emit(ledger, ["US/A", "US/B"], 2)
    for at in (0, 2):
        ledger.mature("US/A", at, 1.0)
    for at in range(3):
        ledger.mature("US/B", at, 1.0)
    ledger.flush()
    # A no decidió en el ordinal 1 y no completa ninguna ventana. B completa sus tres.
    assert ready(ledger) == [(100, 0, 2.0), (100, 1, 3.0), (101, 0, 2.0)]


def test_a_late_component_beyond_the_slack_is_not_used():
    ledger = HorizonLabels(CONFIG)
    for at in range(2):
        emit(ledger, ["US/A"], at)
    ledger.mature("US/A", 0, 1.0)
    # Dos decisiones más del mercado: ordinal 3 > 1 + 1, fuera de la holgura.
    emit(ledger, ["US/B"], 2)
    emit(ledger, ["US/B"], 3)
    assert ledger.mature("US/A", 1, 1.0) == 0
    other = HorizonLabels(CONFIG)
    for at in range(2):
        emit(other, ["US/A"], at)
    other.mature("US/A", 0, 1.0)
    emit(other, ["US/B"], 2)
    assert other.mature("US/A", 1, 1.0) == 1


def test_a_decision_past_its_deadline_delivers_what_it_completed():
    ledger = HorizonLabels(CONFIG)
    emit(ledger, ["US/A"], 0)
    emit(ledger, ["US/A"], 1)
    ledger.mature("US/A", 0, 1.0)
    ledger.mature("US/A", 1, 2.0)
    # La etiqueta del ordinal 2 no madura y la ventana de tres sesiones no se completa. La de
    # dos espera al plazo de su decisión, ordinal 0 + 3 − 1 + 1, y se entrega al superarlo.
    for at in range(3, 5):
        emit(ledger, ["US/A"], at)
        assert ready(ledger) == []
    emit(ledger, ["US/A"], 5)
    assert ledger.markets["US"] == 4
    assert ready(ledger) == [(0, 0, 3.0)]


def test_markets_keep_separate_ordinals():
    ledger = HorizonLabels(CONFIG)
    emit(ledger, ["US/A"], 0)
    emit(ledger, ["CN/X"], 1)
    emit(ledger, ["US/A", "CN/X"], 2)
    assert ledger.markets == {"US": 1, "CN": 1}
    ledger.mature("US/A", 0, 1.0)
    assert ledger.mature("US/A", 2, 1.0) == 1


def test_stored_decisions_stay_bounded_per_flow():
    ledger = HorizonLabels(CONFIG)
    for at in range(50):
        emit(ledger, ["US/A"], at)
    span = CONFIG.span
    assert len(ledger.flows["US/A"]) == span + 1
    assert len(ledger.keys) == span + 1
    # Las etiquetas de decisiones descartadas ya no pueden completar una ventana.
    assert ledger.mature("US/A", 0, 1.0) == 0
    assert ledger.ready == []


def test_repeated_decision_in_the_same_ordinal_is_refused():
    ledger = HorizonLabels(CONFIG)
    emit(ledger, ["US/A"], 0)
    with pytest.raises(ValueError, match="ordinal"):
        ledger.remember(block(["US/A"], 1))


def test_export_and_restore_continue_with_the_same_windows():
    def history(ledger, stop=None):
        produced = []
        for at in range(10):
            if stop is not None and at == stop:
                ledger = HorizonLabels.restore(CONFIG, ledger.export())
            emit(ledger, ["US/A", "US/B"], at)
            if at:
                for flow in ("US/A", "US/B"):
                    if (at, flow) != (4, "US/B"):
                        ledger.mature(flow, at - 1, at / 8 + (flow == "US/B"))
            produced.append(ready(ledger))
        ledger.flush()
        produced.append(ready(ledger))
        return produced, ledger

    expected, full = history(HorizonLabels(CONFIG))
    assert sum(map(len, expected)) > 10
    for stop in range(1, 10):
        produced, resumed = history(HorizonLabels(CONFIG), stop)
        assert produced == expected
        assert resumed.markets == full.markets and resumed.keys == full.keys
    ledger = HorizonLabels(CONFIG)
    for at in range(3):
        emit(ledger, ["US/A"], at)
    ledger.mature("US/A", 0, 1.0)
    ledger.mature("US/A", 1, 1.0)
    payload = ledger.export()
    assert payload["inputs"]["prices"].dtype == torch.float32
    assert payload["completed"][0] == [(0, 2.0)]
    assert math.isnan(payload["labels"][-1]) and not payload["matured"][-1]
    ledger.ready.append(("pending", [0], [0.0]))
    with pytest.raises(ValueError, match="consumen"):
        ledger.export()


def test_restore_refuses_a_pending_decision_without_its_ordinal():
    ledger = HorizonLabels(CONFIG)
    emit(ledger, ["US/A"], 0)
    payload = ledger.export()
    payload["pending"] = [["US/A", 99]]
    with pytest.raises(ValueError, match="ordinal"):
        HorizonLabels.restore(CONFIG, payload)
    with pytest.raises(ValueError, match="esquema"):
        HorizonLabels.restore(CONFIG, dict(payload, schema_version=2))


def brute_force(schedule, horizons, slack):
    """Definición directa: cada ventana completa con su último sumando dentro del plazo."""
    labels, arrived, result = {}, {}, []
    ordinal = -1
    for kind, flow, at, value in schedule:
        if kind == "emit":
            ordinal += 1
            arrived[flow, at] = ordinal
            continue
        labels[flow, arrived[flow, at]] = value
        for head, horizon in enumerate(horizons):
            for start in range(arrived[flow, at] - horizon + 1, arrived[flow, at] + 1):
                keys = [(flow, start + j) for j in range(horizon)]
                if all(key in labels for key in keys) and start + horizon - 1 + slack >= ordinal:
                    result.append((start, head, math.fsum(labels[key] for key in keys)))
    return sorted(result)


@pytest.mark.parametrize("seed", range(20))
def test_windows_match_the_direct_definition_with_random_gaps_and_delays(seed):
    generator = np.random.default_rng(seed)
    config = AuxiliaryHorizons(horizons=(2, 5), weight=1.0, maturity_slack=int(seed % 3))
    schedule, pending = [], []
    for at in range(40):
        # Las etiquetas maduran con retrasos aleatorios y desordenadas, antes de emitir.
        ready_now = [item for item in pending if item[0] <= at]
        generator.shuffle(ready_now)
        for item in ready_now:
            pending.remove(item)
            schedule.append(("label", "US/A", item[1], float(generator.normal())))
        if generator.random() < 0.9:
            schedule.append(("emit", "US/A", at, None))
            if generator.random() < 0.95:
                pending.append((at + 1 + int(generator.integers(0, 3)), at))
    ledger, ordinal_at = HorizonLabels(config), {}
    for kind, flow, at, value in schedule:
        if kind == "emit":
            emit(ledger, [flow], at)
            ordinal_at[at] = ledger.markets["US"]
        else:
            ledger.mature(flow, at, value)
    ledger.flush()
    delivered = sorted(
        (ordinal_at[int(values["prices"][0])], head, target)
        for values, heads, targets in ledger.take()
        for head, target in zip(heads, targets, strict=True)
    )
    # El registro descarta entradas fuera del plazo y cuenta solo las etiquetas contiguas a la
    # nueva. La definición directa recorre todas las ventanas y debe dar las mismas.
    assert delivered
    assert delivered == brute_force(schedule, config.horizons, config.maturity_slack)
