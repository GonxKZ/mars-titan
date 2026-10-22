"""Rechazo de trazas regionales incompatibles con la ruta ejecutada."""

from copy import deepcopy

import pytest
import torch
from test_bridge_episodic_readout import bank, configuration

from mars_titan.models.titans import regional_mechanism_trace as trace_api
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.regional_routes import run_route_pair


def packet(mode="bank", count=8):
    reader = EpisodicReadout(configuration(mode=mode), dtype=torch.float64)
    snapshot = bank("cpu", count=count) if mode == "bank" else None
    _, result = run_route_pair(
        reader,
        torch.ones((2, 32), dtype=torch.float64),
        snapshot,
        route="k4cc",
        context_id="b" * 64,
        cutoff=10,
    )
    return reader, snapshot, result, trace_api.export(result, snapshot)


@pytest.mark.parametrize("mode,count", [("bank", 8), ("bank", 0), ("no_bank", 0)])
def test_trace_preserves_evidence_and_handles_a_genuine_empty_snapshot(mode, count):
    reader, snapshot, _, rows = packet(mode, count)
    for trace in rows:
        trace_api.verify(
            trace,
            reader,
            snapshot_sha256=snapshot._digest if snapshot else None,
            steps=4,
            searches=1 if count else 0,
            route="k4cc",
        )


@pytest.mark.parametrize(
    "fault", ["fixed_ids", "searches", "steps", "weights", "shape", "source", "count"]
)
def test_forged_fixed_selection_or_work_cannot_pass_the_controller_archive(fault):
    reader, snapshot, _, rows = packet()
    trace = deepcopy(rows[0])
    steps, searches = 4, 1
    if fault == "fixed_ids":
        trace["reads"][1]["evidence_ids"].reverse()
    if fault == "searches":
        searches = 2
    if fault == "steps":
        steps = True
    if fault == "weights":
        trace["reads"][0]["regional_weights"][0][0] = -1.0
    if fault == "shape":
        trace["reads"][0]["regional_values"] = []
    if fault == "source":
        trace["snapshot_sha256"] = "c" * 64
    if fault == "count":
        trace["episodes"] = 0
    with pytest.raises(ValueError):
        trace_api.verify(
            trace,
            reader,
            snapshot_sha256=snapshot._digest,
            steps=steps,
            searches=searches,
            route="k4cc",
        )


def test_export_cannot_attribute_another_snapshot_to_the_same_read():
    _, _, result, _ = packet()
    with pytest.raises(ValueError, match="snapshot"):
        trace_api.export(result, bank("cpu", count=7))


@pytest.mark.parametrize(
    "fault",
    [
        "fields",
        "duplicate",
        "bool_id",
        "integer_weights",
        "nan",
        "missing_read",
        "projections",
        "huge_bank",
    ],
)
def test_invalid_serialized_views_fail_before_the_archive_accepts_them(fault):
    reader, snapshot, _, rows = packet()
    trace = deepcopy(rows[0])
    read = trace["reads"][0]
    if fault == "fields":
        read["labels"] = [0.125]
    if fault == "duplicate":
        read["evidence_ids"][0] = read["evidence_ids"][1]
    if fault == "bool_id":
        read["evidence_ids"][0] = True
    if fault == "integer_weights":
        read["regional_weights"] = [[0] * 8 for _ in range(3)]
    if fault == "nan":
        read["read_values"][0] = float("nan")
    if fault == "missing_read":
        trace["reads"].pop()
    if fault == "projections":
        trace["value_projections"] = 1
    if fault == "huge_bank":
        trace["episodes"] = 1025
    with pytest.raises(ValueError):
        trace_api.verify(
            trace, reader, snapshot_sha256=snapshot._digest, steps=4, searches=1, route="k4cc"
        )


@pytest.mark.parametrize("mode,count", [("bank", 0), ("no_bank", 0)])
def test_absence_cannot_be_replaced_by_a_fabricated_regional_value(mode, count):
    reader, snapshot, _, rows = packet(mode, count)
    trace = deepcopy(rows[0])
    if snapshot is None:
        trace["reads"][0]["read_values"] = [0.0] * 32
    else:
        trace["reads"][0]["regional_values"][0][0] = 1.0
    with pytest.raises(ValueError):
        trace_api.verify(
            trace,
            reader,
            snapshot_sha256=snapshot._digest if snapshot else None,
            steps=4,
            searches=0,
            route="k4cc",
        )


def test_export_rejects_an_untyped_result_before_reading_its_fields():
    with pytest.raises(ValueError):
        trace_api.export(None, None)
