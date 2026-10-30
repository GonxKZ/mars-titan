"""Roles de decisión conservados hasta la etiqueta y tras un corte de sesión."""

from dataclasses import replace

import numpy as np
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
from test_regime_routing import ATS, FLOWS, REGIME, ROUTES, daily_routes, events, oracle, run
from test_regime_routing import parent as parent
from test_role_sources import factor_windows, numbered, permuted

from mars_titan.data import correlation_regions
from mars_titan.memory import regimes, role_binding, role_sources
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.regimes import SLOTS, RegimeRule
from mars_titan.memory.role_binding import RoleBindingConfig
from mars_titan.memory.role_sources import RowRoleRule

# Noticias y fundamentales de cada flujo en los seis eventos de la ventana.
NEWS = (
    (1, 0, 1, 0, 1),
    (1, 1, 0, 0, 1),
    (0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1),
    (0, 1, 0, 1, 0),
    (1, 0, 0, 1, 1),
)
FUNDAMENTALS = (
    (0, 0, 1, 1, 0),
    (0, 0, 1, 1, 0),
    (1, 1, 1, 1, 1),
    (0, 0, 0, 0, 0),
    (1, 0, 1, 0, 1),
    (0, 1, 1, 0, 0),
)
# Factor de cada flujo en cada evento: cuatro grupos, uno de ellos con dos flujos. El quinto
# evento tiene 41 rendimientos y no se clasifica.
SECTORS = (
    (0, 1, 2, 3, 0),
    (0, 1, 1, 2, 3),
    (3, 2, 1, 0, 1),
    (0, 0, 1, 2, 3),
    None,
    (1, 2, 3, 1, 0),
)


def with_block(event, change):
    """El mismo evento con una copia de su bloque modificada por `change`."""
    block = event.inputs[0]
    inputs = {name: value.copy() for name, value in block["inputs"].items()}
    changed = dict(block, inputs=inputs, presence=block["presence"].copy())
    change(changed)
    return ObservationEvent(event.at, (changed,), event.labels, event.close_phase)


def modality_events():
    """Eventos con las ausencias declaradas y la ruta 1 + noticias + 2·fundamentales."""
    sequence, routes = [], []
    for event, news, fundamentals in zip(events(), NEWS, FUNDAMENTALS, strict=True):
        news, fundamentals = np.array(news, dtype=bool), np.array(fundamentals, dtype=bool)

        def absent(block, news=news, fundamentals=fundamentals):
            block["inputs"]["news"][~news] = 0.0
            block["inputs"]["fundamentals"][~fundamentals] = 0.0
            block["presence"][:, 1], block["presence"][:, 3] = news, fundamentals

        sequence.append(with_block(event, absent))
        routes.append((1 + news.astype(int) + 2 * fundamentals.astype(int)).tolist())
    return sequence, routes


def sector_events():
    """Eventos cuyas ventanas siguen los factores declarados y su ruta numerada."""
    sequence, routes = [], []
    for index, (event, groups) in enumerate(zip(events(), SECTORS, strict=True)):
        absent = tuple(range(22)) if groups is None else ()
        prices = factor_windows(groups or (0, 1, 2, 3, 0), seed=index, absent=absent)

        def windows(block, prices=prices):
            block["inputs"]["prices"] = prices

        sequence.append(with_block(event, windows))
        routes.append([0] * len(FLOWS) if groups is None else numbered(groups))
    return sequence, routes


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


@pytest.mark.parametrize("source", ["modality", "sector", "permuted_sector"])
def test_row_sources_write_each_label_with_the_route_of_its_own_decision(parent, source):
    sequence, routes = modality_events() if source == "modality" else sector_events()
    if source == "permuted_sector":
        routes = [permuted(route, list(FLOWS), 0, "US") for route in routes]
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01),
        role_binding=RoleBindingConfig(source=source),
    )
    inference, metrics = run(parent, correction, sequence)
    expected, memory = oracle(parent, correction, inference, routes, sequence)
    assert {(f, at): v for f, at, v in emitted(inference)} == expected
    assert torch.equal(inference.memory.matrix, memory.matrix) and memory.writes == 25
    rule, table, summary = correction.routing, np.array(routes), metrics["routing"]
    assert len({tuple(route) for route in routes}) > 2 and (table[:-1] != table[1:]).any()
    assert summary["rule"] == rule.name and summary["labels"] == list(rule.labels)
    assert summary["reads_by_route"] == np.bincount(table.ravel(), minlength=SLOTS).tolist()
    assert summary["writes_by_route"] == np.bincount(table[:-1].ravel(), minlength=SLOTS).tolist()
    occupancy = [np.bincount(route, minlength=SLOTS).tolist() for route in routes]
    assert summary["assignments"] == [
        dict(at=at, market="US", assets=5, occupancy=dict(zip(rule.labels, counts, strict=True)))
        for at, counts in zip(ATS, occupancy, strict=True)
    ]
    # Con una fuente por fila se cuentan los cambios de ruta de cada activo.
    assert summary["transitions_by_market"] == {"US": int((table[:-1] != table[1:]).sum())}


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("source", ["market", "modality", "sector", "permuted_sector"])
def test_row_sources_read_the_session_rows_and_recover_as_the_window(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch, source
):
    calls, original = [], RowRoleRule.routes

    def spy(self, prices, flow_ids, at, presence=None):
        calls.append((np.asarray(prices).shape, np.array(presence)))
        return original(self, prices, flow_ids, at, presence)

    monkeypatch.setattr(RowRoleRule, "routes", spy)
    correction = replace(BRIDGED, query_bridge=None, role_binding=RoleBindingConfig(source=source))
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
    session_calls = len(calls)
    window, _ = corrected_pass(four_flow_source, correction)
    assert emitted(window) == actual and len(calls) > session_calls > 0
    # Las filas de la sesión y de la ventana llevan sus precios y su presencia real.
    flags = np.array([True, False, True, False, False])
    for shape, presence in calls:
        assert shape[1:] == (64, 5) and presence.dtype == np.bool_
        assert presence.shape == (shape[0], 5) and (presence == flags).all()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    "module",
    [role_binding, regimes, role_sources, correlation_regions],
    ids=["binding", "routing", "sources", "regions"],
)
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
