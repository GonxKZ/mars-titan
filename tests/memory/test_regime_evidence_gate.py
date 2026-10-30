"""Puerta de régimen E4 con valores literales, sin modelos ni optimizadores."""

import json
import math
import random
from dataclasses import replace

import pytest
import torch

from mars_titan.memory import regime_evidence_gate as e4
from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureFeedback,
)
from mars_titan.memory.regime_evidence_gate import RegimeEvidenceGate, RegimeGateConfig

CONFIG = RegimeGateConfig(threshold=2.0, drift=0.5, floor_gate=0.25, half_life=1.0, min_history=2)


def cohort(errors, *, at=1, available=2, market="US"):
    """Resultados con punto emitido cero, de modo que el error es la etiqueta."""
    return tuple(
        (f"{market}/A{index:03d}", at, available, 0.0, float(error))
        for index, error in enumerate(errors)
    )


def run(errors_by_publication, config=CONFIG):
    gate, decisions = RegimeEvidenceGate(config), []
    for index, errors in enumerate(errors_by_publication):
        decision, gate = gate.step(
            cohort(errors, at=2 * index, available=2 * index + 1), cutoff=2 * index + 1
        )
        decisions.append(decision)
    return decisions, gate


def test_two_publications_follow_the_declared_equations():
    decisions, gate = run([(1.0, -1.0), (2.0, -2.0)])
    # Primera cohorte: fija r² = 1 y no actualiza el CUSUM porque aún no hay historial.
    assert decisions[0].by_market() == {"US": 0.25}
    # Segunda: z² = 4, y = 3, S = min(h, 0 + 3 − 0,5) = h, la puerta se abre del todo.
    assert dict(decisions[1].statistics) == {"US": 2.0}
    assert decisions[1].by_market() == {"US": 1.0}
    # Con semivida 1 la media pondera por igual: (1 + 4)/2.
    count, square, statistic, rows, gated = gate.export()["markets"]["US"]
    assert (count, square, statistic, rows) == (4, 2.5, 2.0, 4)
    assert gated == pytest.approx(2 * 0.25 + 2 * 1.0)


def test_stable_errors_keep_the_floor_and_a_shift_opens_then_closes_the_gate():
    stable = [(1.0, -1.0)] * 6
    shifted = [(3.0, -3.0)] * 3
    calm = [(1.0, -1.0)] * 12
    decisions, _ = run(stable + shifted + calm, replace(CONFIG, half_life=63.0))
    gates = [d.by_market()["US"] for d in decisions]
    assert gates[:6] == [0.25] * 6
    assert gates[6] > 0.25 and gates[7] == 1.0
    # Con errores de nuevo estables S baja desde h al menos k − y por publicación.
    assert gates[-1] == 0.25 and gates.index(0.25, 9) <= 9 + 4
    assert all(b <= a for a, b in zip(gates[9:], gates[10:], strict=False))


def test_markets_are_independent_and_reordering_rows_does_not_change_the_result():
    rows = cohort((1.0, 2.0, 3.0), at=0, available=1) + cohort(
        (5.0, 7.0), at=0, available=1, market="CN"
    )
    later = cohort((4.0, 1.0, 9.0), at=2, available=3) + cohort(
        (1.0, 1.0), at=2, available=3, market="CN"
    )
    reference = RegimeEvidenceGate(replace(CONFIG, min_history=1))
    a, after = reference.step(rows, cutoff=1)
    b, final = after.step(later, cutoff=3)
    shuffled = list(later)
    random.Random(3).shuffle(shuffled)
    c, other = after.step(tuple(shuffled), cutoff=3)
    assert b == c and final.export() == other.export()
    alone, _ = RegimeEvidenceGate(replace(CONFIG, min_history=1)).step(rows[:3], cutoff=1)
    assert dict(a.statistics)["US"] == dict(alone.statistics)["US"]
    assert a.by_market()["US"] == alone.by_market()["US"]


def test_the_constant_control_keeps_its_gate_and_tracks_the_same_statistic():
    errors = [(1.0, -1.0), (2.0, -2.0), (4.0, -4.0)]
    variant, _ = run(errors)
    control, state = run(errors, replace(CONFIG, mode="constant", constant=0.6))
    assert [d.by_market()["US"] for d in control] == [0.6] * 3
    assert [d.statistics for d in control] == [d.statistics for d in variant]
    assert state.mean_gate() == pytest.approx(0.6)


def test_mean_gate_weights_each_written_row():
    gate = RegimeEvidenceGate(replace(CONFIG, min_history=1))
    first, gate = gate.step(cohort((1.0, 1.0, 1.0), at=0, available=1), cutoff=1)
    second, gate = gate.step(cohort((3.0,), at=1, available=2), cutoff=2)
    expected = (3 * first.by_market()["US"] + second.by_market()["US"]) / 4
    assert gate.mean_gate() == pytest.approx(expected)
    assert RegimeEvidenceGate(CONFIG).mean_gate() is None


def test_observed_counts_the_mature_errors_of_every_market():
    rows = cohort((1.0, 2.0, 3.0), at=0, available=1) + cohort(
        (5.0,), at=0, available=1, market="CN"
    )
    _, gate = RegimeEvidenceGate(CONFIG).step(rows, cutoff=1)
    _, gate = gate.step((), cutoff=2)
    assert gate.observed == 4 and RegimeEvidenceGate(CONFIG).observed == 0


MEMORY = AssociativeMemoryConfig("proximal", key_size=2, rate=0.5)


def feedback():
    """Dos resultados maduros de un evento con pesos proximales uniformes."""
    return MatureFeedback(
        ids=torch.tensor([1, 2]),
        decision_at=torch.tensor([1, 1]),
        available_at=torch.tensor([2, 2]),
        keys=torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float64),
        values=torch.tensor([[1.0], [-2.0]], dtype=torch.float64),
        weights=torch.full((2,), 0.5, dtype=torch.float64),
    )


def test_a_single_gate_writes_with_the_rate_times_the_gate_and_keeps_the_memory_config():
    written = e4.gated_write(AssociativeMemory(MEMORY), feedback(), [0.25, 0.25], cutoff=2)
    slower = AssociativeMemory(replace(MEMORY, rate=0.125)).write(feedback(), cutoff=2)
    assert torch.equal(written.matrix, slower.matrix) and written.cursor == slower.cursor
    assert written.config == MEMORY and written.writes == slower.writes == 2


def test_a_gated_write_needs_a_simple_memory_and_one_gate_per_row():
    with pytest.raises(ValueError, match="una puerta por fila"):
        e4.gated_write(AssociativeMemory(MEMORY), feedback(), [1.0], cutoff=2)

    class Wrapped(AssociativeMemory):
        pass

    with pytest.raises(ValueError, match="memoria proximal simple"):
        e4.gated_write(Wrapped(MEMORY), feedback(), [1.0, 1.0], cutoff=2)


def test_empty_publications_advance_the_clock_without_changing_statistics():
    _, gate = run([(1.0, -1.0)])
    decision, following = gate.step((), cutoff=7)
    assert decision.gates == () and following.cursor == 7
    assert following.publications == gate.publications + 1
    assert following.export()["markets"] == gate.export()["markets"]


def test_availability_is_checked_before_any_financial_value(monkeypatch):
    finite = e4._finite

    def guarded(value, name):
        if name in ("La etiqueta de E4", "El punto de E4"):
            pytest.fail("Se leyó un valor antes de validar toda la disponibilidad")
        return finite(value, name)

    monkeypatch.setattr(e4, "_finite", guarded)
    rows = cohort((1.0,), at=0, available=1) + cohort((2.0,), at=0, available=5, market="CN")
    with pytest.raises(ValueError, match="disponibilidad"):
        RegimeEvidenceGate(CONFIG).step(rows, cutoff=3)


@pytest.mark.parametrize(
    "rows,cutoff",
    [
        (cohort((1.0,), at=0, available=4), 3),
        (cohort((1.0,), at=2, available=2), 3),
        (cohort((1.0, 1.0))[:1] * 2, 3),
        ((("UK/A", 0, 1, 0.0, 1.0),), 3),
        ((("US/A", 0, 1, 0.0, math.nan),), 3),
        ((("US/A", 0, 1, -1e308, 1e308),), 3),
        ((("US/A", True, 1, 0.0, 1.0),), 3),
        (cohort((1.0,), at=0, available=1), 0),
        ((("US/A", 0, 1, 0.0),), 3),
        ({"US/A"}, 3),
    ],
)
def test_invalid_publications_do_not_change_the_confirmed_state(rows, cutoff):
    _, gate = run([(1.0, -1.0)])
    before = gate.export()
    with pytest.raises(ValueError):
        gate.step(rows, cutoff=cutoff)
    assert gate.export() == before


def test_overflowing_errors_are_rejected_before_the_state_changes():
    gate = RegimeEvidenceGate(replace(CONFIG, min_history=1))
    with pytest.raises(ValueError, match="media cuadrática"):
        gate.step(cohort((1e200,)), cutoff=2)
    _, tiny = gate.step(cohort((1e-300,)), cutoff=2)
    with pytest.raises(ValueError, match="estandarizado"):
        tiny.step(cohort((1e300,), at=3, available=4), cutoff=4)
    assert tiny.publications == 1 and tiny.cursor == 2


@pytest.mark.parametrize(
    "arguments",
    [
        dict(config=None),
        dict(markets={"UK": [1, 1.0, 0.0, 1, 0.25]}),
        dict(markets={"US": [1, 1.0, 0.0]}, cursor=1, publications=1),
        dict(markets={"US": [1, 1.0, 0.0, 1, 0.25]}),
        dict(cursor=3),
        dict(markets={"US": [1, 1.0, 5.0, 1, 0.25]}, cursor=1, publications=1),
    ],
)
def test_incoherent_states_are_rejected(arguments):
    arguments = dict(config=CONFIG) | arguments
    with pytest.raises(ValueError):
        RegimeEvidenceGate(arguments.pop("config"), **arguments)


def test_export_round_trips_through_json_and_rejects_changes():
    _, gate = run([(1.0, -1.0), (2.0, -2.0), (0.5,)])
    payload = json.loads(json.dumps(gate.export()))
    restored = RegimeEvidenceGate.restore(CONFIG, payload)
    assert restored.export() == gate.export()
    nxt = cohort((3.0,), at=10, available=11)
    assert gate.step(nxt, cutoff=11)[0] == restored.step(nxt, cutoff=11)[0]
    for resealed in (False, True):
        changed = json.loads(json.dumps(payload))
        changed["markets"]["US"][3] = changed["markets"]["US"][0] + 1
        if resealed:
            changed["sha256"] = e4._digest({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError):
            RegimeEvidenceGate.restore(CONFIG, changed)
    with pytest.raises(ValueError):
        RegimeEvidenceGate.restore(replace(CONFIG, drift=0.25), payload)
    short = json.loads(json.dumps(payload))
    short["markets"]["US"] = short["markets"]["US"][:4]
    short["sha256"] = e4._digest({k: v for k, v in short.items() if k != "sha256"})
    with pytest.raises(ValueError, match="cuenta, media"):
        RegimeEvidenceGate.restore(CONFIG, short)


@pytest.mark.parametrize(
    "options",
    [
        dict(mode="other"),
        dict(mode="constant"),
        dict(constant=0.5),
        dict(mode="constant", constant=0.0),
        dict(mode="constant", constant=1.5),
        dict(threshold=0.0),
        dict(drift=-0.1),
        dict(floor_gate=0.0),
        dict(floor_gate=1.5),
        dict(half_life=0.5),
        dict(min_history=0),
        dict(min_history=True),
        dict(scale_floor=0.0),
        dict(threshold=math.inf),
    ],
)
def test_invalid_configurations_are_rejected(options):
    with pytest.raises(ValueError):
        RegimeGateConfig(**options)


def test_identity_separates_the_variant_from_its_control():
    variant = RegimeGateConfig().identity()
    control = RegimeGateConfig(mode="constant", constant=0.4).identity()
    assert variant["kind"] == control["kind"] == "regime_evidence_gate_v1"
    assert variant != control and variant["constant"] is None
