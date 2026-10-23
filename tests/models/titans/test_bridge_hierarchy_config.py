"""Contrato estricto e identidad de la jerarquía episódica opcional."""

import importlib

import pytest


def hierarchy_api():
    bridge = importlib.import_module("mars_titan.models.titans.evidence_bridge")
    assert hasattr(bridge, "BridgeHierarchyConfig"), "Falta la configuración jerárquica opcional"
    return bridge


def test_hierarchy_is_explicit_and_disabled_identity_is_unchanged():
    module = hierarchy_api()
    config = module.BridgeHierarchyConfig()
    assert config.local_steps == 2
    assert config.node_updates(4, 38) == 988
    base = module.EvidenceBridgeConfig(codec_id="a" * 64)
    assert base.hierarchy is None
    assert "hierarchy" not in base.identity()


@pytest.mark.parametrize("steps,count", [(1, 18), (2, 26), (4, 42)])
def test_hierarchy_counts_auxiliary_updates_and_keeps_rounds_fixed(steps, count):
    config = hierarchy_api().BridgeHierarchyConfig(local_steps=steps)
    assert config.node_updates(4, 1) == count
    assert config.identity()["rounds"] == 2


@pytest.mark.parametrize(
    "options",
    [
        {"local_steps": True},
        {"local_steps": 3},
        {"feedback": 1},
        {"max_node_updates_per_row": 1597},
        {"topology": "complete"},
    ],
)
def test_hierarchy_rejects_unsupported_or_ambiguous_options(options):
    with pytest.raises(ValueError):
        hierarchy_api().BridgeHierarchyConfig(**options)


def test_hierarchy_rejects_composition_with_the_flat_recurrence():
    module = hierarchy_api()
    from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig

    with pytest.raises(ValueError):
        module.EvidenceBridgeConfig(
            codec_id="a" * 64,
            regions=("a", "b", "c", "d"),
            topology="self",
            cycles=0,
            hierarchy=module.BridgeHierarchyConfig(),
            recurrence=BridgeRecurrenceConfig(2, 988),
        )


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)))
@pytest.mark.parametrize(
    "fault",
    (
        "half",
        "regions",
        "empty",
        "graph_shape",
        "graph_dtype",
        "weights",
        "mask_type",
        "mask_shape",
        "steps",
        "gain",
        "autocast",
    ),
)
def test_hierarchy_rejects_incompatible_tensor_contracts(device, fault):
    from contextlib import nullcontext

    import torch

    from mars_titan.models.titans.bridge_hierarchy import hierarchy_mix

    local = torch.ones((1, 4, 2), device=device, dtype=torch.float64)
    graph = torch.eye(7, device=device, dtype=torch.float64)[None]
    weights = torch.eye(2, device=device, dtype=torch.float64).repeat(4, 1, 1)
    mask = torch.ones((1, 4), device=device, dtype=torch.bool)
    steps, gain = 2, 0.25
    if fault == "half":
        local = local.half()
    elif fault == "regions":
        local = local[:, :3]
    elif fault == "empty":
        local = local[:0]
    elif fault == "graph_shape":
        graph = graph[:, :, :6]
    elif fault == "graph_dtype":
        graph = graph.float()
    elif fault == "weights":
        weights = weights[:3]
    elif fault == "mask_type":
        mask = mask.float()
    elif fault == "mask_shape":
        mask = mask[:, :3]
    elif fault == "steps":
        steps = True
    elif fault == "gain":
        gain = True
    context = (
        torch.autocast(device.split(":")[0], dtype=torch.bfloat16)
        if fault == "autocast"
        else nullcontext()
    )
    with context, pytest.raises(ValueError):
        hierarchy_mix(local, graph, weights, mask, local_steps=steps, gain=gain)


@pytest.mark.parametrize(
    "fault", ("mask_shape", "mask_type", "empty", "edges_shape", "edges_type", "feedback")
)
def test_graph_rejects_ambiguous_masks_and_supports(fault):
    import torch

    from mars_titan.models.titans.bridge_hierarchy import hierarchy_graph

    available = torch.ones((1, 4), dtype=torch.bool)
    edges = torch.eye(7, dtype=torch.bool)
    feedback = True
    if fault == "mask_shape":
        available = available[:, :3]
    elif fault == "mask_type":
        available = available.float()
    elif fault == "empty":
        available = available[:0]
    elif fault == "edges_shape":
        edges = edges[:4]
    elif fault == "edges_type":
        edges = edges.float()
    else:
        feedback = 1
    with pytest.raises(ValueError):
        hierarchy_graph(available, edges, feedback=feedback)


@pytest.mark.parametrize("regions,reads", [(3, 1), (True, 1), (4, True), (4, -1), (4, 39)])
def test_node_quota_rejects_other_region_counts_and_invalid_event_lengths(regions, reads):
    with pytest.raises(ValueError):
        hierarchy_api().BridgeHierarchyConfig().node_updates(regions, reads)


@pytest.mark.parametrize("payload", [None, {"rounds": 3}, {"local_steps": 2}])
def test_json_hierarchy_rejects_partial_or_unknown_contracts(payload):
    from mars_titan.models.titans.bridge_hierarchy import hierarchy_options

    with pytest.raises(ValueError):
        hierarchy_options(payload)
