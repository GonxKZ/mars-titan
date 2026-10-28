"""E4 pondera las escrituras de B6 con la puerta de su mercado y conserva las emisiones."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_area_correction_window import AREAS
from test_financial_session_associative import DELTA
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_mars_titan_correction_parity import (
    PROXIMAL,
    RECIPE,
    SOURCE,
    core,
    corrected_pass,
    emitted,
)
from test_mars_titan_session_parity import events
from test_multiscale_session import BASE

from mars_titan.memory.associative_memory import AssociativeMemory, MatureFeedback
from mars_titan.memory.regime_evidence_gate import RegimeEvidenceGate, RegimeGateConfig
from mars_titan.training.mars_titan_correction import CorrectionInference
from mars_titan.training.regime_gate_inference import (
    RegimeGateCorrectionInference,
    restore_configuration,
    source_hashes,
)

GATE = RegimeGateConfig(threshold=1.0, drift=0.0, floor_gate=0.25, min_history=1)


def inference(source, config=GATE, correction=PROXIMAL):
    return RegimeGateCorrectionInference(
        core(source), RECIPE, correction, source["codec"], regime_gate=config, audit=True
    )


def test_a_constant_gate_of_one_reproduces_b6_bit_for_bit(four_flow_source):
    original, expected = corrected_pass(four_flow_source, PROXIMAL)
    gated = inference(four_flow_source, RegimeGateConfig(mode="constant", constant=1.0))
    result = gated._pass(SOURCE, events(four_flow_source))
    assert emitted(gated) == emitted(original)
    assert torch.equal(gated.memory.matrix, original.memory.matrix)
    assert gated.memory.cursor == original.memory.cursor
    assert {k: v for k, v in result.items() if k != "regime_gate"} == expected
    assert result["regime_gate"]["mean_gate"] == 1.0


def test_each_write_uses_the_rate_times_the_gate_of_its_market(four_flow_source):
    gated = inference(four_flow_source)
    seen = []
    original = gated._write_memory

    def spy(memory, feedback, staged, at):
        result = original(memory, feedback, staged, at)
        rate = PROXIMAL.memory.rate * gated._gates["US"]
        expected = AssociativeMemory(
            replace(PROXIMAL.memory, rate=rate),
            memory.matrix,
            writes=memory.writes,
            cursor=memory.cursor,
        ).write(feedback, cutoff=at)
        assert torch.equal(result.matrix, expected.matrix)
        assert result.config == PROXIMAL.memory and result.writes == expected.writes
        seen.append(gated._gates["US"])
        return result

    gated._write_memory = spy
    result = gated._pass(SOURCE, events(four_flow_source))
    assert len(seen) == result["regime_gate"]["publications"] > 1
    assert min(seen) == 0.25 and len(set(seen)) > 1


def test_mixed_markets_weight_rows_by_their_gate():
    import math

    from mars_titan.memory.associative_memory import AssociativeMemoryConfig

    config = AssociativeMemoryConfig("proximal", key_size=2, rate=0.5)
    memory = AssociativeMemory(config)
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=torch.float64)
    feedback = MatureFeedback(
        ids=torch.tensor([1, 2, 3]),
        decision_at=torch.tensor([1, 1, 1]),
        available_at=torch.tensor([2, 2, 2]),
        keys=keys,
        values=torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64),
        weights=torch.full((3,), 1.0 / 3, dtype=torch.float64),
    )
    staged = [
        (1, "CN/B", None, None, None),
        (1, "US/A", None, None, None),
        (1, "US/C", None, None, None),
    ]
    holder = SimpleNamespace(_gates={"CN": 1.0, "US": 0.25})
    written = RegimeGateCorrectionInference._write_memory(holder, memory, feedback, staged, 2)
    # Por fila η·w·g: el sistema proximal (I + Σ η w g k kᵀ) A = Σ η w g k y.
    rates = 0.5 * torch.tensor([1.0, 0.25, 0.25], dtype=torch.float64) / 3
    system = torch.eye(2, dtype=torch.float64) + (keys.T * rates) @ keys
    right = (keys.T * rates) @ feedback.values
    expected = torch.linalg.solve(system, right)
    torch.testing.assert_close(written.matrix, expected, rtol=1e-12, atol=1e-14)
    assert written.config == config and written.writes == 3
    assert math.isclose(float(written.matrix.abs().sum()), float(expected.abs().sum()))


def test_the_gate_scores_the_final_emission_with_the_prior_scale(four_flow_source):
    gated = inference(four_flow_source)
    reference = RegimeEvidenceGate(GATE)
    checked = []

    def sequence():
        nonlocal reference
        for event in events(four_flow_source):
            yield event
            if not event.labels:
                continue
            points = {(flow, at): value for flow, at, value in emitted(gated)}
            rows = tuple(
                (flow, at, event.at, points[flow, at], target) for flow, at, target in event.labels
            )
            decision, reference = reference.step(rows, cutoff=event.at)
            assert gated.last_gate_decision == decision
            checked.append(decision.by_market()["US"])

    gated._pass(SOURCE, sequence())
    assert checked and gated.regime_gate.export() == reference.export()


def test_gated_writes_change_the_memory_but_not_the_rows_or_the_parent(four_flow_source):
    original, expected = corrected_pass(four_flow_source, PROXIMAL)
    gated = inference(four_flow_source, RegimeGateConfig(mode="constant", constant=0.5))
    result = gated._pass(SOURCE, events(four_flow_source))
    assert result["labels"] == expected["labels"]
    assert gated.memory.writes == original.memory.writes
    assert not torch.equal(gated.memory.matrix, original.memory.matrix)
    assert [r[:2] for r in emitted(gated)] == [r[:2] for r in emitted(original)]
    assert all(p.grad is None for p in gated.predictor.parameters())


def test_repeating_a_phase_resets_the_gate(four_flow_source):
    gated = inference(four_flow_source)
    first = gated._pass(SOURCE, events(four_flow_source))
    assert gated._pass(SOURCE, events(four_flow_source)) == first
    assert not gated._gate_rows and gated._gates is None


@pytest.mark.parametrize("correction", [DELTA, BASE, AREAS])
def test_unsupported_correctors_are_rejected(four_flow_source, correction):
    with pytest.raises(ValueError, match="E4"):
        inference(four_flow_source, correction=correction)


def test_the_plain_correction_writes_with_its_own_rate(four_flow_source):
    plain = CorrectionInference(
        core(four_flow_source), RECIPE, PROXIMAL, four_flow_source["codec"], audit=True
    )
    keys = torch.ones(1, 64, dtype=torch.float64) / 8
    feedback = PROXIMAL.feedback(
        ids=[1], decision_at=[1], available_at=[2], keys=keys, values=[0.5]
    )
    memory = PROXIMAL.empty()
    written = plain._write_memory(memory, feedback, [(1, "US/A", 0.5, None, None)], 2)
    assert torch.equal(written.matrix, memory.write(feedback, cutoff=2).matrix)


def test_configuration_restoration_requires_the_same_identity_and_sources():
    config = replace(GATE, drift=0.75)
    request = dict(regime_gate=config.identity(), code=source_hashes())
    assert restore_configuration(request) == config
    request["code"]["mars_titan.memory.regime_evidence_gate"] = "a" * 64
    with pytest.raises(ValueError, match="fuentes"):
        restore_configuration(request)


@pytest.mark.parametrize("key,value", [("min_history", True), ("unknown", 1), ("kind", "m2")])
def test_configuration_restoration_rejects_extra_or_changed_fields(key, value):
    identity = dict(GATE.identity(), **{key: value})
    with pytest.raises(ValueError):
        restore_configuration(dict(regime_gate=identity, code=source_hashes()))
