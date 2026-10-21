"""Contrato de trazas regionales con datos técnicos, sin ajustar parámetros."""

from copy import deepcopy

import pytest
import torch
from test_bridge_episodic_readout import bank, configuration

from mars_titan.memory import regional_episode_trace as trace_api
from mars_titan.models.titans.episodic_readout import EpisodicReadout


def packet(*, mode="bank", selection="first_read"):
    reader = EpisodicReadout(
        configuration(mode=mode, refinements=2, episode_selection=selection), dtype=torch.float64
    )
    snapshot = None if mode == "no_bank" else bank("cpu")
    flows = ("US/A", "US/B")
    result = reader(
        torch.ones((2, 32), dtype=torch.float64), snapshot, context_id="b" * 64, cutoff=10
    )
    trace = trace_api.begin(reader, flows, snapshot, context_id="b" * 64, cutoff=10, block_rows=2)
    trace_api.append(trace, flows, result)
    return reader, trace, flows


def verify(reader, trace, flows):
    return trace_api.verify(
        trace, reader, flows=flows, context_id="b" * 64, cutoff=10, refinements=2, block_rows=2
    )


@pytest.mark.parametrize("mode", ["bank", "no_bank"])
def test_trace_is_detached_and_identifies_only_common_evidence(mode):
    reader, trace, flows = packet(mode=mode)
    verify(reader, trace, flows)
    for record in trace["blocks"][0]["reads"]:
        assert "labels" not in record and "values" not in record
        assert all(
            value.device.type == "cpu" and not value.requires_grad and value.grad_fn is None
            for value in record.values()
        )
    assert (trace["snapshot_sha256"] is None) == (mode == "no_bank")


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "parent",
        "context",
        "cutoff",
        "regions",
        "rows",
        "episodes",
        "blocks",
        "graph",
        "fields",
        "shape",
        "dtype",
        "nonfinite",
        "ids",
        "weights",
        "presence",
        "fixed_ids",
        "flows",
    ],
)
def test_incompatible_or_fabricated_trace_is_rejected(change):
    reader, trace, flows = packet()
    item = trace["blocks"][0]["reads"][0]
    if change == "schema":
        trace["schema_version"] = True
    if change == "parent":
        trace["configuration_sha256"] = "c" * 64
    if change == "context":
        trace["context_id"] = "c" * 64
    if change == "cutoff":
        trace["cutoff"] = 11
    if change == "regions":
        trace["regions"] = None
    if change == "rows":
        trace["rows"] = 3
    if change == "episodes":
        trace["episodes"] = 0
    if change == "blocks":
        trace["blocks"] = []
    if change == "graph":
        trace["bridge_weights"] = trace["bridge_weights"].bool()
    if change == "fields":
        item["labels"] = torch.zeros(2)
    if change == "shape":
        item["regional_values"] = item["regional_values"][0]
    if change == "dtype":
        item["regional_values"] = item["regional_values"].float()
    if change == "nonfinite":
        item["regional_values"][0, 0, 0] = float("inf")
    if change == "ids":
        item["evidence_ids"][0, 1] = item["evidence_ids"][0, 0]
    if change == "weights":
        item["regional_weights"][0, 0, 0] = -1
    if change == "presence":
        item["presence"][0] = False
    if change == "fixed_ids":
        trace["blocks"][0]["reads"][1]["evidence_ids"] = item["evidence_ids"].flip(-1)
    if change == "flows":
        trace["blocks"][0]["flow_ids"] = flows[::-1]
    with pytest.raises(ValueError):
        verify(reader, trace, flows)


def test_trace_cannot_claim_a_read_for_warmup_or_exceed_the_event_budget():
    reader, trace, flows = packet()
    with pytest.raises(ValueError, match="sin lectura"):
        trace_api.verify(
            trace, reader, flows=flows, context_id="b" * 64, cutoff=10, refinements=0, block_rows=2
        )
    from dataclasses import replace

    config = replace(
        reader.config,
        max_working_bytes=1024,
        evidence_bridge=replace(reader.config.evidence_bridge, max_working_bytes=1024),
    )
    limited = EpisodicReadout(config, dtype=torch.float64)
    with pytest.raises(ValueError, match="antes de emitir"):
        trace_api.begin(limited, flows, bank("cpu"), context_id="b" * 64, cutoff=10, block_rows=2)


def test_first_read_verification_does_not_mutate_the_trace():
    reader, trace, flows = packet()
    before = deepcopy(trace)
    verify(reader, trace, flows)
    for left, right in zip(before["blocks"][0]["reads"], trace["blocks"][0]["reads"], strict=True):
        assert all(torch.equal(left[name], right[name]) for name in left)


def test_export_detaches_and_copies_views_without_retaining_the_training_graph():
    reader = EpisodicReadout(configuration(), dtype=torch.float64)
    snapshot = bank("cpu")
    state = torch.ones((2, 32), dtype=torch.float64, requires_grad=True)
    result = reader(state, snapshot, context_id="b" * 64, cutoff=10, differentiable=True)
    trace = trace_api.begin(
        reader, ("US/A", "US/B"), snapshot, context_id="b" * 64, cutoff=10, block_rows=2
    )
    trace_api.append(trace, ("US/A", "US/B"), result)
    exported = trace["blocks"][0]["reads"][0]
    expected = exported["read_values"].clone()
    assert all(not value.requires_grad and value.grad_fn is None for value in exported.values())
    with torch.no_grad():
        result.reads[0].values.add_(1)
    assert torch.equal(exported["read_values"], expected)


def test_export_cannot_combine_a_read_from_another_snapshot():
    from dataclasses import replace

    reader = EpisodicReadout(configuration(), dtype=torch.float64)
    snapshot = bank("cpu")
    result = reader(
        torch.ones((2, 32), dtype=torch.float64), snapshot, context_id="b" * 64, cutoff=10
    )
    wrong = replace(
        result.reads[0], bridge=replace(result.reads[0].bridge, snapshot_sha256="c" * 64)
    )
    result = replace(result, reads=(wrong,))
    trace = trace_api.begin(
        reader, ("US/A", "US/B"), snapshot, context_id="b" * 64, cutoff=10, block_rows=2
    )
    with pytest.raises(ValueError, match="mezcla snapshots"):
        trace_api.append(trace, ("US/A", "US/B"), result)
