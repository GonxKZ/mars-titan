"""Catálogo finito, presupuestos declarados y retornos sin ocultar errores de contrato."""

from dataclasses import replace

import pytest

from mars_titan.models.titans.mechanism_catalog import (
    MechanismAction,
    MechanismAvailability,
    MechanismBudget,
    MechanismCatalog,
)


def available(**options):
    return MechanismAvailability(
        **dict(context=True, reader=True, quantiles=True, static=True, online=True) | options
    )


def test_catalogue_contains_only_36_registered_pipeline_combinations_and_timing():
    catalog = MechanismCatalog()
    actions = catalog.actions()
    assert len(actions) == len({action.key for action in actions}) == 36
    assert actions[0] == MechanismAction()
    identity = catalog.identity()
    assert identity["event_order"] == "predict_all_then_feedback_v1"
    assert identity["stages"] == dict(
        context="before_mac", route="after_k1", calibration="cohort_emission"
    )
    assert identity["arbitrary_code_allowed"] is False
    assert (
        replace(catalog, budget=MechanismBudget(max_reader_steps=2)).fingerprint()
        != catalog.fingerprint()
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(context="eval(x)"),
        dict(route="k3"),
        dict(calibration="exec"),
        dict(route=True),
        dict(context=None),
    ],
)
def test_unknown_actions_are_contract_errors(options):
    with pytest.raises(ValueError):
        MechanismAction(**options)


@pytest.mark.parametrize(
    "budget,availability,action,reason",
    [
        (
            MechanismBudget(),
            available(context=False),
            MechanismAction(context="parent"),
            "unavailable_context",
        ),
        (
            MechanismBudget(),
            available(reader=False),
            MechanismAction(route="k2c"),
            "unavailable_readout",
        ),
        (
            MechanismBudget(max_reader_steps=2),
            available(),
            MechanismAction(route="k4cr"),
            "reader_budget",
        ),
        (
            MechanismBudget(),
            available(reader_limit=1),
            MechanismAction(route="k2c"),
            "reader_budget",
        ),
        (
            MechanismBudget(max_reader_searches=1),
            available(nonempty_bank=True),
            MechanismAction(route="k2r"),
            "search_budget",
        ),
        (
            MechanismBudget(),
            available(quantiles=False),
            MechanismAction(calibration="static"),
            "unavailable_quantiles",
        ),
        (
            MechanismBudget(),
            available(static=False),
            MechanismAction(calibration="static"),
            "unavailable_calibration",
        ),
        (
            MechanismBudget(),
            available(online_capacity=False),
            MechanismAction(calibration="online"),
            "pending_calibration_budget",
        ),
        (
            MechanismBudget(max_context_markets=0),
            available(),
            MechanismAction(context="parent"),
            "context_budget",
        ),
        (
            MechanismBudget(),
            available(context_capacity=False),
            MechanismAction(context="parent"),
            "context_budget",
        ),
    ],
)
def test_unavailable_or_over_budget_actions_return_the_declared_raw_k1(
    budget, availability, action, reason
):
    result = MechanismCatalog(budget).resolve(action, availability, rows=2)
    assert result.applied == MechanismAction()
    assert result.requested == action and result.status == "fallback"
    assert reason in result.reasons


def test_live_bank_search_counts_and_online_row_budget_are_explicit():
    catalog = MechanismCatalog(MechanismBudget(max_online_rows=2))
    action = MechanismAction("parent", "k4rc", "online")
    accepted = catalog.resolve(action, available(nonempty_bank=True), rows=2)
    assert accepted.applied == action and accepted.status == "applied"
    assert accepted.reader_steps == 4 and accepted.reader_searches == 2
    rejected = catalog.resolve(action, available(), rows=3)
    assert rejected.applied == MechanismAction("parent", "k1", "none")
    assert "pending_calibration_budget" in rejected.reasons


def test_post_stage_return_keeps_the_context_selected_before_mac():
    action = MechanismAction("parent", "k2c", "online")
    result = MechanismCatalog().resolve(action, available(online_capacity=False), rows=2)
    assert result.applied == MechanismAction("parent", "k1", "none")
    assert result.status == "fallback"


def test_an_invalid_baseline_is_rejected_instead_of_claiming_a_successful_return():
    for catalog, availability in (
        (MechanismCatalog(), available(reader_limit=0)),
        (MechanismCatalog(MechanismBudget(max_reader_searches=0)), available(nonempty_bank=True)),
    ):
        with pytest.raises(ValueError, match="K1"):
            catalog.resolve(MechanismAction(), availability, rows=1)


@pytest.mark.parametrize("rows", [0, True, 8193])
def test_invalid_population_or_types_are_not_fallbacks(rows):
    with pytest.raises(ValueError):
        MechanismCatalog().resolve(MechanismAction(), available(), rows=rows)


def test_scalar_raw_parent_needs_no_reader_or_interval_and_has_no_reader_work():
    result = MechanismCatalog().resolve(MechanismAction(), MechanismAvailability(), rows=1)
    assert result.status == "applied" and result.reader_steps == result.reader_searches == 0
    with pytest.raises(ValueError):
        MechanismCatalog().resolve(dict(route="k1"), available(), rows=1)
