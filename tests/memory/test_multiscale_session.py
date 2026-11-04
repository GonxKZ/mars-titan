"""E1 conserva las predicciones de cada escala hasta la maduración y la recuperación."""

from dataclasses import replace

import pytest
import torch
from test_area_correction_window import BRIDGED, decisions
from test_area_memory import proximal_step
from test_financial_session import moment
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import options, trajectory
from test_financial_session_associative import shared_native as shared_native
from test_mars_titan_correction_parity import (
    RECIPE,
    core,
    core_pass,
    corrected_pass,
    emitted,
    shifted,
)
from test_mars_titan_session_parity import events
from test_regime_routing import daily_routes

from mars_titan.memory import multiscale_mixture
from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.multiscale_mixture import DAY_US, MultiscaleMixtureConfig
from mars_titan.memory.regimes import RegimeRule
from mars_titan.memory.role_binding import RoleBindingConfig
from mars_titan.training.financial_run import ChronologicalInference

MIXTURE = MultiscaleMixtureConfig()
BASE = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25), multiscale=MIXTURE)
LEDGER = replace(BASE, multiscale=replace(MIXTURE, loss_scale=1.0))
AREA = replace(BRIDGED, memory=BASE.memory, multiscale=MIXTURE, query_bridge=None)
BRIDGE = replace(AREA, query_bridge=BRIDGED.query_bridge)
ROLE = replace(BRIDGE, role_binding=RoleBindingConfig())


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    "correction",
    [BASE, LEDGER, AREA, BRIDGE, ROLE],
    ids=["linear", "unclipped_fixture", "areas", "bridge", "roles"],
)
def test_session_window_and_recovery_use_the_same_issued_experts(
    shared_native, four_flow_source, frozen_consumer, tmp_path, correction, monkeypatch
):
    if correction.routing is not None:
        monkeypatch.setattr(RegimeRule, "routes", daily_routes)
    direct = trajectory(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "direct", correction=correction
    )
    recovered = trajectory(
        shared_native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "recovered",
        correction=correction,
        recover=True,
    )
    actual = [(a, t, v) for e in direct["events"] for _, a, t, v in e["predictions"]]
    assert actual == [(a, t, v) for e in recovered["events"] for _, a, t, v in e["predictions"]]
    assert direct["snapshot"] == recovered["snapshot"]
    window, _ = corrected_pass(four_flow_source, correction)
    assert emitted(window) == actual
    assert direct["events"][-1]["memory"]["state_sha256"] == window.memory.export()["state_sha256"]
    if correction is LEDGER:
        assert not torch.equal(window.memory.weights, torch.full((3,), 1 / 3, dtype=torch.float64))


@pytest.mark.parametrize("configuration", [BASE, LEDGER], ids=["default", "unclipped_fixture"])
def test_linear_window_matches_independent_calendar_and_issued_loss_recurrence(
    four_flow_source, configuration
):
    import math

    plain = ChronologicalInference(core(four_flow_source), RECIPE, audit=True)
    core_pass(plain, events(four_flow_source))
    core_values = {(row[2], row[3]): row[4] for row in plain.audit if row[0] == "prediction"}
    matrices = [torch.zeros(64, 1, dtype=torch.float64) for _ in range(3)]
    scores = [0.0, 0.0, 0.0]
    clock = None
    pending = {}
    expected = []
    mixture = configuration.multiscale
    for event in events(four_flow_source):
        aged = [
            matrix * (1.0 if clock is None else 2 ** (-(event.at - clock) / (DAY_US * h)))
            for matrix, h in zip(matrices, mixture.half_lives_days, strict=True)
        ]
        staged = sorted(
            (at, flow, label - core_values[flow, at], pending.pop((flow, at)))
            for flow, at, label in event.labels
        )
        if event.inputs:
            rows, keys = decisions(four_flow_source, event)
            parts = torch.cat([keys @ matrix for matrix in aged], dim=1)
            weights = [math.exp(s - max(scores)) for s in scores]
            total = sum(weights)
            weights = [w / total for w in weights]
            for (flow, at), key, predictions in zip(rows, keys, parts, strict=True):
                correction = sum(w * p for w, p in zip(weights, predictions.tolist(), strict=True))
                expected.append((flow, at, core_values[flow, at] + correction))
                pending[flow, at] = (key, predictions.tolist())
        if staged:
            losses = [
                sum(
                    min(abs(pred[e] - value) / mixture.loss_scale, 1.0)
                    for _, _, value, (_, pred) in staged
                )
                / len(staged)
                for e in range(3)
            ]
            scores = [
                score - mixture.weight_rate * loss
                for score, loss in zip(scores, losses, strict=True)
            ]
            values = torch.tensor([value for _, _, value, _ in staged], dtype=torch.float64)
            keys = torch.stack([key for *_, (key, _) in staged])
            matrices = [proximal_step(matrix, keys, values, BASE.memory) for matrix in aged]
            clock = event.at
    inference, _ = corrected_pass(four_flow_source, configuration)
    actual = emitted(inference)
    assert [row[:2] for row in actual] == [row[:2] for row in expected]
    torch.testing.assert_close(
        torch.tensor([r[2] for r in actual], dtype=torch.float64),
        torch.tensor([r[2] for r in expected], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_a_later_label_does_not_change_predictions_already_issued(four_flow_source):
    plain, _ = corrected_pass(four_flow_source, BASE)
    changed, _ = corrected_pass(four_flow_source, BASE, shifted(four_flow_source, 128, label=20.0))
    before = emitted(plain)
    after = emitted(changed)
    boundary = moment(128)
    assert [r for r in before if r[1] <= boundary] == [r for r in after if r[1] <= boundary]


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_session_rejects_missing_or_misaligned_pending_experts(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    settings = options(shared_native, four_flow_source, frozen_consumer, associative=BASE)
    with FinancialSession(tmp_path / "session", **settings) as session:
        parts = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
        reference = session._stage_associative(
            BASE.empty(), torch.zeros(1, dtype=torch.float64), None, parts
        )
        assert torch.equal(session._associative_state(reference, with_parts=True)[3], parts)
        for extra in (
            {},
            {"issued_experts": torch.zeros(1, 3, dtype=torch.float64)},
            {"issued_experts": torch.empty(0, 2, dtype=torch.float64)},
        ):
            payload = dict(
                memory=BASE.empty().export(), core=torch.empty(0, dtype=torch.float64), **extra
            )
            reference = session._stage(payload, "associative")
            with pytest.raises(ValueError):
                session._associative_state(reference)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_session_identity_pins_the_mixture_source(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch
):
    settings = options(shared_native, four_flow_source, frozen_consumer, associative=BASE)
    with FinancialSession(tmp_path / "first", **settings) as session:
        before = session.model_id
    other = tmp_path / "other.py"
    other.write_text("# Regla diferente.\n")
    monkeypatch.setattr(multiscale_mixture, "__file__", str(other))
    with FinancialSession(tmp_path / "second", **settings) as session:
        assert session.model_id != before


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_session_rejects_a_mixture_that_cannot_fit_its_artifact_budget(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    correction = replace(
        BRIDGE,
        areas=AreaLayout(areas=8, shifts=4),
        multiscale=MultiscaleMixtureConfig(half_lives_days=(7.0, 30.0, 91.0, 365.0)),
    )
    settings = options(shared_native, four_flow_source, frozen_consumer, associative=correction)
    with pytest.raises(ValueError, match="presupuesto"):
        FinancialSession(tmp_path / "oversized", **settings)
