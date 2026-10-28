"""E3 usa las emisiones del recorrido real y conserva las etiquetas de escritura de B6."""

import math

import numpy as np
import pytest
import torch
from test_area_correction_window import AREAS, BRIDGED
from test_financial_session import moment
from test_financial_session_associative import DELTA
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_mars_titan_correction_parity import (
    PROXIMAL,
    RECIPE,
    SOURCE,
    core,
    core_pass,
    corrected_pass,
    emitted,
    shifted,
)
from test_mars_titan_session_parity import events
from test_multiscale_session import AREA, BASE, BRIDGE

from mars_titan.memory.salience_budget import SalienceBudgetConfig
from mars_titan.training.financial_run import ChronologicalInference
from mars_titan.training.salience_inference import (
    SalienceCorrectionInference,
    restore_configuration,
    source_hashes,
)

BACKENDS = (DELTA, PROXIMAL, AREAS, BRIDGED, BASE, AREA, BRIDGE)


def inference(source, correction=PROXIMAL, **options):
    return SalienceCorrectionInference(
        core(source),
        RECIPE,
        correction,
        source["codec"],
        salience=SalienceBudgetConfig(**options),
        audit=True,
    )


@pytest.mark.parametrize("correction", BACKENDS)
def test_large_quota_keeps_original_emissions_and_every_memory_coefficient(
    four_flow_source, correction
):
    original, expected = corrected_pass(four_flow_source, correction)
    selected = inference(four_flow_source, correction, writes_per_market=8192)
    result = selected._pass(SOURCE, events(four_flow_source))
    assert emitted(selected) == emitted(original)
    assert torch.equal(selected.memory.matrix, original.memory.matrix)
    assert selected.memory.cursor == original.memory.cursor
    assert selected.memory.writes == original.memory.writes == result["labels"]
    assert result["salience"]["observed"] == result["salience"]["selected"]
    assert {k: v for k, v in result.items() if k != "salience"} == expected


@pytest.mark.parametrize("correction", BACKENDS)
def test_zero_quota_keeps_the_frozen_core_and_observes_every_mature_error(
    four_flow_source, correction
):
    plain = ChronologicalInference(core(four_flow_source), RECIPE, audit=True)
    core_pass(plain, events(four_flow_source))
    selected = inference(four_flow_source, correction, writes_per_market=0)
    result = selected._pass(SOURCE, events(four_flow_source))
    assert emitted(selected) == [r[2:] for r in plain.audit if r[0] == "prediction"]
    assert selected.memory.writes == 0 and result["salience"]["selected"] == 0
    assert result["salience"]["observed"] == result["labels"] == 16
    assert not selected._mature_experts and not selected._salience_rows


@pytest.mark.parametrize("rule", ("standardized_error", "absolute_error", "uniform"))
@pytest.mark.parametrize("correction", BACKENDS)
def test_small_quota_writes_only_selected_rows_and_cleans_expert_components(
    four_flow_source, rule, correction
):
    selected = inference(four_flow_source, correction, writes_per_market=1, rule=rule)
    result = selected._pass(SOURCE, events(four_flow_source))
    assert result["labels"] == result["salience"]["observed"] == 16
    assert result["associative_writes"] == result["salience"]["selected"] == 4
    assert not selected._mature_experts and not selected._salience_rows
    assert all(p.grad is None for p in selected.predictor.parameters())


def test_scores_use_final_issued_points_and_the_core_residual_remains_the_write_value(
    four_flow_source,
):
    selected = inference(four_flow_source, writes_per_market=1, rule="absolute_error")
    observed = []

    def sequence():
        for event in events(four_flow_source):
            before = selected.memory
            yield event
            if event.labels:
                points = {(flow, at): value for flow, at, value in emitted(selected)}
                scores = dict(selected.last_salience_selection.scores)
                assert scores == {
                    (flow, at): abs(value - points[flow, at]) for flow, at, value in event.labels
                }
                observed.extend(scores)
                chosen = set(selected.last_salience_selection.selected)
                cores = {tuple(r[2:4]): r[4] for r in selected.audit if r[0] == "prediction"}
                staged = sorted(
                    (at, flow, value) for flow, at, value in event.labels if (flow, at) in chosen
                )
                keys = selected.correction.keys(
                    np.concatenate(
                        [four_flow_source["encoded"][flow, at].key_inputs for at, flow, _ in staged]
                    )
                )
                feedback = selected.correction.feedback(
                    ids=[before.writes + i + 1 for i in range(len(staged))],
                    decision_at=[at for at, _, _ in staged],
                    available_at=[event.at] * len(staged),
                    keys=keys,
                    values=[value - cores[flow, at] for at, flow, value in staged],
                )
                expected = before.write(feedback, cutoff=event.at)
                assert torch.equal(selected.memory.matrix, expected.matrix)

    selected._pass(SOURCE, sequence())
    assert len(observed) == 16
    core_points = {tuple(r[2:4]): r[4] for r in selected.audit if r[0] == "prediction"}
    final = {(flow, at): value for flow, at, value in emitted(selected)}
    assert any(final[key] != core_points[key] for key in observed)


def test_scales_are_built_from_all_previous_issued_errors_and_not_the_current_batch(
    four_flow_source,
):
    selected = inference(four_flow_source, writes_per_market=1, min_history=1)
    history, checked = {}, []

    def sequence():
        for event in events(four_flow_source):
            yield event
            if not event.labels:
                continue
            issued = {(flow, at): point for flow, at, point in emitted(selected)}
            scales = dict(selected.last_salience_selection.scales)
            scores = dict(selected.last_salience_selection.scores)
            for flow, at, target in event.labels:
                errors = history.get(flow, [])
                scale = (
                    max(math.sqrt(math.fsum(e * e for e in errors) / len(errors)), 1e-8)
                    if errors
                    else 0.01
                )
                assert scales[flow, at] == pytest.approx(scale, rel=1e-12, abs=1e-15)
                assert scores[flow, at] == pytest.approx(abs(target - issued[flow, at]) / scale)
                checked.append(bool(errors))
            for flow, at, target in event.labels:
                history.setdefault(flow, []).append(target - issued[flow, at])

    selected._pass(SOURCE, sequence())
    assert any(checked) and len(checked) == 16


def test_future_labels_do_not_change_earlier_emissions_and_repeating_a_phase_resets_scales(
    four_flow_source,
):
    first = inference(four_flow_source, writes_per_market=1)
    expected = first._pass(SOURCE, events(four_flow_source))
    following = first._pass(SOURCE, events(four_flow_source))
    assert following == expected
    changed = inference(four_flow_source, writes_per_market=1)
    sequence = shifted(four_flow_source, 128, label=100.0)
    changed._pass(SOURCE, sequence)
    boundary = moment(128)
    assert [r for r in emitted(changed) if r[1] <= boundary] == [
        r for r in emitted(first)[:20] if r[1] <= boundary
    ]


def test_configuration_restoration_requires_the_same_identity_and_sources():
    config = SalienceBudgetConfig(writes_per_market=3)
    request = dict(salience=config.identity(), code=source_hashes())
    assert restore_configuration(request) == config
    assert restore_configuration({}) is None
    request["code"]["mars_titan.memory.salience_budget"] = "a" * 64
    with pytest.raises(ValueError, match="fuentes"):
        restore_configuration(request)


@pytest.mark.parametrize("key,value", [("seed", True), ("unknown", 1), ("kind", "m2")])
def test_configuration_restoration_rejects_extra_or_changed_fields(key, value):
    config = SalienceBudgetConfig(seed=1)
    identity = dict(config.identity(), **{key: value})
    with pytest.raises(ValueError):
        restore_configuration(dict(salience=identity, code=source_hashes()))
