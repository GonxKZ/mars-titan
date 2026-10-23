"""Recibos adaptativos derivados de rutas reales, con trabajo y copias independientes."""

import importlib
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps

FULL = "adaptive_reference_full"
COMPACT = "adaptive_compact"


def trace_api():
    return importlib.import_module("mars_titan.models.titans.regional_mechanism_trace")


def exported(*, backend=FULL, device="cpu", kind="populated", inactive=False, maximum=4):
    helpers = importlib.import_module("tests.models.titans.test_adaptive_bridge_contracts")
    readout = importlib.import_module("mars_titan.models.titans.episodic_readout")
    routes = importlib.import_module("mars_titan.models.titans.regional_routes")
    config = helpers.adaptive_configuration(backend=backend, k=2, steps=maximum)
    if inactive:
        intervention = importlib.import_module("mars_titan.models.titans.bridge_intervention")
        config = replace(
            config,
            evidence_bridge=replace(
                config.evidence_bridge,
                intervention=intervention.BridgeInterventionConfig((False, False, False)),
            ),
        )
    if kind == "no_bank":
        config = replace(config, mode="no_bank")
    reader = readout.EpisodicReadout(config, device=device, dtype=torch.float64)
    snapshot = (
        None
        if kind == "no_bank"
        else helpers.snapshot(device, torch.float64, count=0 if kind == "empty" else 8)
    )
    _, result = routes.run_route_pair(
        reader,
        helpers.query(device, torch.float64),
        snapshot,
        route="k2c",
        context_id=helpers.CONTEXT,
        cutoff=10,
    )
    return reader, snapshot, result, trace_api().export(result, snapshot, reader=reader)


def verify(row, reader, snapshot):
    trace_api().verify(
        row,
        reader,
        snapshot_sha256=None if snapshot is None else snapshot._digest,
        steps=2,
        searches=1 if row["episodes"] else 0,
        route="k2c",
    )


@pytest.fixture(scope="module")
def receipts():
    return {backend: exported(backend=backend) for backend in (FULL, COMPACT)}


def test_adaptive_receipt_contract_api_is_available_in_call():
    for name in ("export", "verify", "verify_stopping", "observed_work", "verify_work"):
        assert callable(getattr(trace_api(), name, None)), f"Falta la API {name}"


@pytest.mark.parametrize(
    "backend,device",
    [(FULL, "cpu"), (COMPACT, "cpu"), pytest.param(FULL, "cuda:0", marks=pytest.mark.cuda)],
)
def test_real_routes_export_scalar_decisions_and_reconcile_selected_and_executed_work(
    backend, device
):
    reader, snapshot, _, rows = exported(backend=backend, device=device)
    expected_evaluated = 4 if backend == FULL else 1
    assert len(rows) == 3
    for row in rows:
        verify(row, reader, snapshot)
        assert row["kind"] == "adaptive_regional_mechanism_trace_v1"
        assert row["episodes"] == 8 and row["value_projections"] == 2
        assert len(row["reads"]) == 2
        for read in row["reads"]:
            assert read["stopping"] == dict(
                selected_steps=1, evaluated_steps=expected_evaluated, criterion_met=True
            )
            assert type(read["stopping"]["selected_steps"]) is int
            assert type(read["stopping"]["evaluated_steps"]) is int
            assert type(read["stopping"]["criterion_met"]) is bool
    assert trace_api().observed_work(reader, rows) == dict(
        selected_steps=6, evaluated_steps=6 * expected_evaluated
    )
    work = dict(
        bridge_value_projections=6,
        bridge_initial_views=18,
        bridge_selected_node_updates=18,
        bridge_node_updates=18 * expected_evaluated,
    )
    trace_api().verify_work(work, reader, value_projections=6, rows=rows)


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_steps": True},
        {"evaluated_steps": 4.0},
        {"criterion_met": 1},
        {"selected_steps": 3},
        {"selected_steps": 0},
        {"criterion_met": False},
        {"extra": 0},
    ],
    ids=(
        "bool-step",
        "float-work",
        "int-criterion",
        "prefix-three",
        "zero",
        "false-early",
        "field",
    ),
)
def test_receipt_rejects_noncanonical_decision_fields_types_and_prefixes(receipts, changes):
    reader, snapshot, _, rows = receipts[FULL]
    row = deepcopy(rows[0])
    row["reads"][0]["stopping"].update(changes)
    with pytest.raises(ValueError):
        verify(row, reader, snapshot)


def test_selected_prefix_cannot_exceed_the_declared_jmax():
    reader, snapshot, _, rows = exported(maximum=2)
    row = deepcopy(rows[0])
    row["reads"][0]["stopping"]["selected_steps"] = 4
    with pytest.raises(ValueError):
        verify(row, reader, snapshot)


def test_full_backend_rejects_charging_only_the_selected_prefix(receipts):
    reader, snapshot, _, rows = receipts[FULL]
    row = deepcopy(rows[0])
    decision = row["reads"][0]["stopping"]
    assert decision == dict(selected_steps=1, evaluated_steps=4, criterion_met=True)
    decision["evaluated_steps"] = 1
    with pytest.raises(ValueError, match="trabajo.*backend|backend"):
        verify(row, reader, snapshot)


def test_compact_backend_rejects_a_cost_different_from_its_selected_prefix(receipts):
    reader, snapshot, _, rows = receipts[COMPACT]
    row = deepcopy(rows[0])
    row["reads"][0]["stopping"]["evaluated_steps"] = 2
    with pytest.raises(ValueError, match="trabajo.*backend|backend"):
        verify(row, reader, snapshot)


@pytest.mark.parametrize("kind", ("empty", "no_bank"))
def test_empty_and_absent_bank_receipts_keep_zero_work_and_a_false_criterion(kind):
    reader, snapshot, _, rows = exported(kind=kind)
    for row in rows:
        verify(row, reader, snapshot)
        assert row["episodes"] == row["value_projections"] == 0
        for read in row["reads"]:
            assert read["stopping"] == dict(
                selected_steps=0, evaluated_steps=0, criterion_met=False
            )
    assert trace_api().observed_work(reader, rows) == dict(selected_steps=0, evaluated_steps=0)
    damaged = deepcopy(rows[0])
    damaged["reads"][0]["stopping"]["criterion_met"] = True
    with pytest.raises(ValueError):
        verify(damaged, reader, snapshot)


@pytest.mark.parametrize("backend", (FULL, COMPACT))
def test_populated_inactive_regions_keep_the_pool_and_choose_j1(backend):
    reader, snapshot, _, rows = exported(backend=backend, inactive=True)
    for row in rows:
        verify(row, reader, snapshot)
        assert row["episodes"] == 8 and row["value_projections"] == 2
        for read in row["reads"]:
            assert len(read["evidence_ids"]) == 8
            assert read["stopping"] == dict(
                selected_steps=1,
                evaluated_steps=4 if backend == FULL else 1,
                criterion_met=True,
            )
            assert not torch.tensor(read["regional_values"]).count_nonzero()
            assert not torch.tensor(read["read_values"]).count_nonzero()
    damaged = deepcopy(rows[0])
    damaged["reads"][0]["stopping"].update(selected_steps=4, evaluated_steps=4, criterion_met=False)
    with pytest.raises(ValueError):
        verify(damaged, reader, snapshot)


@pytest.mark.parametrize("damage", ("row-total", "projections"))
def test_work_rejects_disagreement_with_row_totals_and_value_projections(receipts, damage):
    reader, _, _, original = receipts[FULL]
    rows = deepcopy(original)
    work = dict(
        bridge_value_projections=6,
        bridge_initial_views=18,
        bridge_selected_node_updates=18,
        bridge_node_updates=72,
    )
    projections = 6
    if damage == "row-total":
        rows[0]["reads"][0]["stopping"]["selected_steps"] = 2
    else:
        projections = work["bridge_value_projections"] = 7
    with pytest.raises(ValueError):
        trace_api().verify_work(work, reader, value_projections=projections, rows=rows)


def test_exported_rows_reads_and_repeated_exports_have_no_mutable_aliases():
    reader, snapshot, result, first = exported()
    expected = deepcopy(first)
    second = trace_api().export(result, snapshot, reader=reader)
    snapshot_before = snapshot.export_cpu()["sha256"]
    first[0]["reads"][0]["stopping"]["selected_steps"] = 2
    first[0]["reads"][0]["evidence_ids"][0] = 999
    first[0]["reads"][0]["regional_values"][0][0] += 1.0
    assert first[1:] == expected[1:]
    assert first[0]["reads"][1] == expected[0]["reads"][1]
    assert second == expected
    assert trace_api().export(result, snapshot, reader=reader) == expected
    assert snapshot.export_cpu()["sha256"] == snapshot_before


def test_receipt_and_live_export_reject_another_backend_configuration(receipts):
    reader, snapshot, result, rows = receipts[FULL]
    other = receipts[COMPACT][0]
    verify(rows[0], reader, snapshot)
    with pytest.raises(ValueError):
        verify(rows[0], other, snapshot)
    with pytest.raises(ValueError):
        trace_api().export(result, snapshot, reader=other)
