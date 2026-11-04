"""Roles de decisión conservados hasta la etiqueta y tras un corte de sesión."""

from dataclasses import replace

import pytest
import torch
from test_area_correction_window import BRIDGED
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import options, trajectory
from test_financial_session_associative import shared_native as shared_native
from test_mars_titan_correction_parity import corrected_pass, emitted
from test_regime_routing import REGIME, ROUTES, daily_routes, events, oracle, run
from test_regime_routing import parent as parent

from mars_titan.memory import regimes, role_binding
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.regimes import RegimeRule
from mars_titan.memory.role_binding import RoleBindingConfig


def test_changing_regimes_write_with_the_decision_role_against_independent_recurrence(parent):
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01),
        role_binding=RoleBindingConfig(),
        routing=REGIME,
    )
    inference, _ = run(parent, correction)
    expected, memory = oracle(parent, correction, inference, ROUTES)
    assert {(f, at): v for f, at, v in emitted(inference)} == expected
    assert torch.equal(inference.memory.matrix, memory.matrix)
    split, _ = run(parent, correction, events(split=True))
    assert emitted(split) == emitted(inference)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    "mode,bridge",
    [("bipolar", False), ("constant_role", False), ("projected_concat", False), ("bipolar", True)],
)
def test_bound_areas_recover_the_same_decisions_as_the_window(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch, mode, bridge
):
    # La ruta cambia entre decisión y maduración para detectar el uso de información posterior.
    monkeypatch.setattr(RegimeRule, "routes", daily_routes)
    correction = replace(
        BRIDGED,
        query_bridge=BRIDGED.query_bridge if bridge else None,
        role_binding=RoleBindingConfig(mode=mode),
    )
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
    state = correction.restore(direct["events"][-1]["memory"])
    assert torch.equal(state.matrix, window.memory.matrix)
    assert state.writes == 16


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("module", [role_binding, regimes], ids=["binding", "routing"])
def test_session_pins_enabled_binding_source(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch, module
):
    correction = replace(BRIDGED, role_binding=RoleBindingConfig())
    settings = options(shared_native, four_flow_source, frozen_consumer, associative=correction)
    with FinancialSession(tmp_path / "first", **settings) as session:
        before = session.model_id
    changed = tmp_path / "other.py"
    changed.write_text("# Operador diferente.\n")
    monkeypatch.setattr(module, "__file__", str(changed))
    with FinancialSession(tmp_path / "second", **settings) as session:
        assert session.model_id != before
