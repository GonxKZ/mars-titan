"""Referencia escalar y derivadas de la jerarquía, sin ajuste o resultados financieros."""

import importlib
import importlib.util
import math
from dataclasses import replace

import pytest
import torch

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
TOLERANCE = dict(atol=1e-12, rtol=1e-12)
MASKS = (
    (True, True, True, True),
    (True, False, False, True),
    (False, False, True, True),
    (False, False, False, False),
)
LOCAL = (
    ((0.21, -0.31), (0.62, 0.17), (-0.43, 0.58), (0.09, -0.72)),
    ((-0.19, 0.38), (0.41, -0.23), (0.67, 0.12), (-0.56, 0.34)),
    ((0.35, 0.61), (-0.28, 0.49), (0.73, -0.16), (-0.42, -0.59)),
    ((-0.63, 0.18), (0.32, -0.47), (-0.11, 0.54), (0.46, 0.29)),
)
WEIGHTS = (
    ((1.1, 0.3), (-0.2, 0.7)),
    ((0.8, -0.4), (0.15, 1.2)),
    ((0.6, 0.25), (-0.45, 0.9)),
    ((1.25, -0.1), (0.2, 0.5)),
)


def hierarchy_api():
    name = "mars_titan.models.titans.bridge_hierarchy"
    assert importlib.util.find_spec(name) is not None, "Falta el componente jerárquico"
    module = importlib.import_module(name)
    for symbol in ("BridgeHierarchyConfig", "hierarchy_graph", "hierarchy_mix"):
        assert hasattr(module, symbol), f"Falta la API jerárquica {symbol}"
    return module


def numbers(values, device):
    return torch.tensor(values, dtype=torch.float64, device=device)


def node_presence(mask):
    return list(mask) + [any(mask[:2]), any(mask[2:]), any(mask)]


def adjacency(topology):
    pairs = (
        ((0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6))
        if topology == "tree"
        else ((0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6))
    )
    edges = [[receiver == sender for sender in range(7)] for receiver in range(7)]
    for first, second in pairs:
        edges[first][second] = edges[second][first] = True
    return edges


def reference_graph(masks, topology, feedback=True, cut=None):
    edges = adjacency(topology)
    if cut is not None:
        receiver, sender = cut
        edges[receiver][sender] = False
    result = []
    for mask in masks:
        present = node_presence(mask)
        graph = []
        for receiver in range(7):
            allowed = [
                present[receiver] and present[sender] and edges[receiver][sender]
                for sender in range(7)
            ]
            denominator = max(1, sum(allowed))
            row = [int(value) / denominator for value in allowed]
            if receiver < 4 and not feedback:
                row = [float(present[receiver] and sender == receiver) for sender in range(7)]
            graph.append(row)
        result.append(graph)
    return result


def scalar_reference(local, graph, weights, masks, local_steps, gain):
    """Desenrollar con listas y math.tanh, sin llamar operadores de producción."""
    width = len(local[0][0])
    states = [
        [list(value) if mask[i] else [0.0] * width for i, value in enumerate(row)]
        + [[0.0] * width for _ in range(3)]
        for row, mask in zip(local, masks, strict=True)
    ]

    def phase(previous, targets):
        updated = [[value[:] for value in row] for row in previous]
        for batch, row in enumerate(previous):
            present = node_presence(masks[batch])
            for receiver in targets:
                if not present[receiver]:
                    updated[batch][receiver] = [0.0] * width
                    continue
                message = [
                    sum(graph[batch][receiver][sender] * row[sender][h] for sender in range(7))
                    for h in range(width)
                ]
                if receiver < 4:
                    # W actúa a la derecha sobre el vector fila recibido.
                    value = [
                        math.tanh(sum(message[h] * weights[receiver][h][k] for h in range(width)))
                        for k in range(width)
                    ]
                else:
                    value = message
                updated[batch][receiver] = [
                    (1 - gain) * row[receiver][h] + gain * value[h] for h in range(width)
                ]
        return updated

    for _ in range(2):
        states = phase(states, (4, 5))
        states = phase(states, (6,))
        states = phase(states, (4, 5))
        for _ in range(local_steps):
            states = phase(states, (0, 1, 2, 3))
    return states


def operands(device, topology="tree", feedback=True):
    return (
        numbers(LOCAL, device),
        numbers(reference_graph(MASKS, topology, feedback), device),
        numbers(WEIGHTS, device),
        torch.tensor(MASKS, dtype=torch.bool, device=device),
    )


def scalar_case(device, values, *, feedback=True, cut=None, weight=1.0):
    local = numbers([[list((value,)) for value in values]], device)
    graph = numbers(reference_graph([(True,) * 4], "tree", feedback, cut), device)
    weights = torch.full((4, 1, 1), weight, dtype=torch.float64, device=device)
    available = torch.ones((1, 4), dtype=torch.bool, device=device)
    return local, graph, weights, available


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topology", ("tree", "star"))
@pytest.mark.parametrize("feedback", (False, True))
def test_graph_matches_independent_edges_presence_and_masked_means(device, topology, feedback):
    module = hierarchy_api()
    config = module.BridgeHierarchyConfig(topology=topology, feedback=feedback)
    edges = torch.as_tensor(config.edges(), dtype=torch.bool, device=device)
    expected_edges = torch.tensor(adjacency(topology), dtype=torch.bool, device=device)
    assert torch.equal(edges, expected_edges)
    assert int(edges.sum()) == 19 and torch.equal(edges, edges.T)
    available = torch.tensor(MASKS, dtype=torch.bool, device=device)
    graph, present = module.hierarchy_graph(available, edges, feedback=feedback)
    assert graph.shape == (4, 7, 7) and graph.dtype == torch.float64
    assert graph.device == available.device
    assert present.dtype == torch.bool and present.device == available.device
    assert torch.equal(present, torch.tensor([node_presence(row) for row in MASKS], device=device))
    torch.testing.assert_close(
        graph, numbers(reference_graph(MASKS, topology, feedback), device), **TOLERANCE
    )
    torch.testing.assert_close(graph.sum(-1), present.to(torch.float64), **TOLERANCE)
    assert torch.count_nonzero(graph[~present]) == 0
    assert torch.count_nonzero(graph.masked_select(~present[:, None, :])) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topology", ("tree", "star"))
@pytest.mark.parametrize("feedback", (False, True))
@pytest.mark.parametrize("local_steps", (1, 2, 4))
def test_two_rounds_match_scalar_reference_with_distinct_receiver_matrices(
    device, topology, feedback, local_steps
):
    module = hierarchy_api()
    local, graph, weights, available = operands(device, topology, feedback)
    before = [value.clone() for value in (local, graph, weights, available)]
    actual = module.hierarchy_mix(
        local, graph, weights, available, local_steps=local_steps, gain=0.35
    )
    expected = scalar_reference(
        LOCAL, reference_graph(MASKS, topology, feedback), WEIGHTS, MASKS, local_steps, 0.35
    )
    assert actual.shape == (4, 7, 2)
    assert actual.dtype == local.dtype and actual.device == local.device
    torch.testing.assert_close(actual, numbers(expected, device), **TOLERANCE)
    for value, saved in zip((local, graph, weights, available), before, strict=True):
        assert torch.equal(value, saved)
    present = torch.tensor([node_presence(row) for row in MASKS], device=device)
    assert torch.count_nonzero(actual[~present]) == 0
    changed = local.clone()
    changed[~available] = 1e100
    torch.testing.assert_close(
        module.hierarchy_mix(
            changed, graph, weights, available, local_steps=local_steps, gain=0.35
        ),
        actual,
        **TOLERANCE,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topology", ("tree", "star"))
def test_feedback_off_keeps_auxiliaries_but_leaves_equal_four_self_steps(device, topology):
    module = hierarchy_api()
    local, graph, weights, available = operands(device, topology, feedback=False)
    actual = module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=0.35)
    expected = local * available[..., None]
    for _ in range(4):
        expected = (
            0.65 * expected + 0.35 * torch.tanh(torch.einsum("brh,rhk->brk", expected, weights))
        ) * available[..., None]
    torch.testing.assert_close(actual[:, :4], expected, **TOLERANCE)
    assert torch.count_nonzero(actual[:, 4:]) > 0
    isolated = graph.clone()
    isolated[:, 4:] = 0
    for node in (4, 5, 6):
        isolated[:, node, node] = torch.tensor(
            [node_presence(row)[node] for row in MASKS], dtype=torch.float64, device=device
        )
    alternative = module.hierarchy_mix(
        local, isolated, weights, available, local_steps=2, gain=0.35
    )
    torch.testing.assert_close(alternative[:, :4], actual[:, :4], **TOLERANCE)
    assert torch.count_nonzero(alternative[:, 4:]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cut", ((5, 6), (6, 4)))
def test_signal_crosses_groups_only_through_the_declared_root_path(device, cut):
    module = hierarchy_api()
    a = math.tanh(1)
    local, graph, weights, available = scalar_case(device, (a, 0, 0, 0))
    actual = module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=1.0)
    expected = scalar_reference(
        local.cpu().tolist(), graph.cpu().tolist(), weights.cpu().tolist(), [[True] * 4], 2, 1.0
    )
    torch.testing.assert_close(actual, numbers(expected, device), **TOLERANCE)
    assert actual[0, 2, 0] > 0 and local[0, 2, 0] == 0
    opposite = module.hierarchy_mix(-local, graph, weights, available, local_steps=2, gain=1.0)
    torch.testing.assert_close(opposite, -actual, **TOLERANCE)
    cut_graph = numbers(reference_graph([[True] * 4], "tree", cut=cut), device)
    blocked = module.hierarchy_mix(local, cut_graph, weights, available, local_steps=2, gain=1.0)
    assert torch.count_nonzero(blocked[:, 2:4]) == 0
    assert blocked[0, 0, 0] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_cancellation_is_a_negative_reconstruction_case_without_deleting_inputs(device):
    module = hierarchy_api()
    a = math.tanh(1)
    local, graph, weights, available = scalar_case(device, (a, -a, a, -a))
    before = local.clone()
    actual = module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=1.0)
    off_graph = numbers(reference_graph([[True] * 4], "tree", feedback=False), device)
    off = module.hierarchy_mix(local, off_graph, weights, available, local_steps=2, gain=1.0)
    expected, isolated = a, a
    for _ in range(4):
        expected, isolated = math.tanh(expected / 2), math.tanh(isolated)
    torch.testing.assert_close(actual[0, 0, 0], numbers(expected, device), **TOLERANCE)
    torch.testing.assert_close(off[0, 0, 0], numbers(isolated, device), **TOLERANCE)
    torch.testing.assert_close(actual[:, 4:], torch.zeros_like(actual[:, 4:]), **TOLERANCE)
    assert 0 < actual[0, 0, 0] < off[0, 0, 0]
    assert float((actual[0, 0, 0] / isolated - 1).square()) > 0
    torch.testing.assert_close(off[0, 0, 0] / isolated, numbers(1, device), **TOLERANCE)
    assert torch.equal(local, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topology", ("tree", "star"))
def test_calls_rows_blocks_and_rng_are_independent(device, topology):
    module = hierarchy_api()
    local, graph, weights, available = operands(device, topology)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None

    def invoke(values, matrix, mask):
        return module.hierarchy_mix(values, matrix, weights, mask, local_steps=2, gain=0.35)

    full = invoke(local, graph, available)
    invoke(-local, graph, available)
    torch.testing.assert_close(invoke(local, graph, available), full, **TOLERANCE)
    order = torch.tensor((2, 0, 3, 1), device=device)
    inverse = torch.argsort(order)
    permuted = invoke(local[order], graph[order], available[order])
    torch.testing.assert_close(permuted[inverse], full, **TOLERANCE)
    blocked = torch.cat(
        [invoke(local[:1], graph[:1], available[:1]), invoke(local[1:], graph[1:], available[1:])]
    )
    torch.testing.assert_close(blocked, full, **TOLERANCE)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)


@pytest.mark.parametrize("device", DEVICES)
def test_reader_uses_the_same_parameters_and_restarts_real_auxiliary_results(device):
    module = hierarchy_api()
    from test_evidence_bridge import snapshot

    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig, SharedEvidenceBridge

    assert "hierarchy" in EvidenceBridgeConfig.__dataclass_fields__, "Falta la ruta jerárquica"
    base = EvidenceBridgeConfig(
        codec_id="a" * 64,
        hidden_size=2,
        neighbors=2,
        regions=("r0", "r1", "r2", "r3"),
        topology="self",
        cycles=0,
        bridge_gain=0.35,
    )
    plain = SharedEvidenceBridge(base, device=device, dtype=torch.float64).eval()
    reader = SharedEvidenceBridge(
        replace(base, hierarchy=module.BridgeHierarchyConfig()), device=device, dtype=torch.float64
    ).eval()
    assert dict(plain.named_parameters()).keys() == dict(reader.named_parameters()).keys()
    for name, value in plain.named_parameters():
        assert torch.equal(value, dict(reader.named_parameters())[name])
    source = snapshot(device, count=2)
    fingerprint = source.export_cpu()["sha256"]
    state = numbers(((0.2, -0.4), (0.3, 0.8)), device)
    masks = [[True] * 4, [False, True, True, False]]
    available = torch.tensor(masks, dtype=torch.bool, device=device)
    calls = []
    handle = reader.value.register_forward_pre_hook(
        lambda _module, args: calls.append(tuple(args[0].shape))
    )
    try:
        with torch.no_grad():
            initial = plain(state, source, context_id="b" * 64, cutoff=4, available=available)
            first = reader(state, source, context_id="b" * 64, cutoff=4, available=available)
            reader(-state, source, context_id="b" * 64, cutoff=4, available=available)
            again = reader(state, source, context_id="b" * 64, cutoff=4, available=available)
    finally:
        handle.remove()
    assert len(calls) == 3 and all(shape == (4, 65) for shape in calls)
    actual = torch.cat((first.region_values, first.group_values, first.root_values), dim=1)
    expected = scalar_reference(
        initial.region_values.cpu().tolist(),
        reference_graph(masks, "tree"),
        reader.region_value.detach().cpu().tolist(),
        masks,
        2,
        0.35,
    )
    torch.testing.assert_close(actual, numbers(expected, device), **TOLERANCE)
    torch.testing.assert_close(
        first.hierarchy_weights, numbers(reference_graph(masks, "tree"), device), **TOLERANCE
    )
    torch.testing.assert_close(
        first.bridge_weights, first.hierarchy_weights[:, :4, :4], **TOLERANCE
    )
    for name in ("values", "region_values", "group_values", "root_values", "hierarchy_weights"):
        torch.testing.assert_close(getattr(first, name), getattr(again, name), **TOLERANCE)
    assert first.snapshot_sha256 == fingerprint == source.export_cpu()["sha256"]
    assert torch.equal(first.evidence_ids, initial.evidence_ids)


@pytest.mark.parametrize("device", DEVICES)
def test_gain_zero_preserves_masked_initial_views_and_zero_auxiliaries(device):
    local, graph, weights, available = operands(device)
    actual = hierarchy_api().hierarchy_mix(
        local, graph, weights, available, local_steps=4, gain=0.0
    )
    torch.testing.assert_close(actual[:, :4], local * available[..., None], **TOLERANCE)
    assert torch.count_nonzero(actual[:, 4:]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("topology", ("tree", "star"))
def test_full_gradcheck_covers_initial_views_receiver_weights_and_graph(device, topology):
    module = hierarchy_api()
    local = numbers([[[0.2], [-0.35], [0.4], [0.1]]], device).requires_grad_()
    weights = numbers([[[0.9]], [[0.6]], [[1.1]], [[1.25]]], device).requires_grad_()
    logits = torch.zeros((1, 7, 7), dtype=torch.float64, device=device, requires_grad=True)
    allowed = torch.tensor(adjacency(topology), dtype=torch.bool, device=device)[None]
    available = torch.ones((1, 4), dtype=torch.bool, device=device)

    def evaluate(values, matrices, scores):
        graph = torch.where(allowed, scores, -torch.inf).softmax(-1)
        return module.hierarchy_mix(values, graph, matrices, available, local_steps=2, gain=0.37)

    assert torch.autograd.gradcheck(
        evaluate, (local, weights, logits), eps=1e-6, atol=1e-6, rtol=1e-4
    )
    gradients = torch.autograd.grad(
        evaluate(local, weights, logits).square().sum(), (local, weights, logits)
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_gradients_do_not_cross_rows_or_absent_receivers(device):
    module = hierarchy_api()
    masks = [[True, False, True, True], [True, False, False, True]]
    local = numbers(LOCAL[:2], device).requires_grad_()
    weights = numbers(WEIGHTS, device).requires_grad_()
    graph = numbers(reference_graph(masks, "tree"), device)
    available = torch.tensor(masks, dtype=torch.bool, device=device)
    result = module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=0.35)
    local_grad, weights_grad = torch.autograd.grad(result[0].square().sum(), (local, weights))
    assert torch.count_nonzero(local_grad[1]) == 0
    assert torch.count_nonzero(local_grad[0, 1]) == 0
    assert torch.count_nonzero(weights_grad[1]) == 0
    assert torch.count_nonzero(local_grad[0, [0, 2, 3]]) > 0
    assert torch.isfinite(weights_grad).all() and torch.count_nonzero(weights_grad) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_zero_states_can_have_the_declared_noncontractive_full_jacobian(device):
    module = hierarchy_api()
    local, graph, weights, available = scalar_case(device, (0, 0, 0, 0), weight=2.0)
    local.requires_grad_()
    actual = module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=1.0)
    assert torch.count_nonzero(actual) == 0
    gradient = torch.autograd.grad(actual[0, 0, 0], local)[0]
    torch.testing.assert_close(gradient.sum(), numbers(3521 / 576, device), **TOLERANCE)
    assert gradient.sum() > 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "corruption", ("local_nan", "weight_inf", "graph_nan", "absent_local_nan", "absent_weight_nan")
)
def test_nonfinite_inputs_are_rejected_even_in_masked_slots(device, corruption):
    module = hierarchy_api()
    local, graph, weights, available = operands(device)
    if corruption == "local_nan":
        local[0, 0, 0] = torch.nan
    elif corruption == "weight_inf":
        weights[0, 0, 0] = torch.inf
    elif corruption == "graph_nan":
        graph[3, 0, 6] = torch.nan
    elif corruption == "absent_local_nan":
        local[1, 1, 0] = torch.nan
    else:
        available[:, 1] = False
        graph = numbers(reference_graph(available.cpu().tolist(), "tree"), device)
        weights[1, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=0.35)


@pytest.mark.parametrize("device", DEVICES)
def test_finite_parameters_overflowing_before_tanh_are_rejected(device):
    module = hierarchy_api()
    available = torch.ones((1, 4), dtype=torch.bool, device=device)
    graph = numbers(reference_graph([[True] * 4], "tree"), device)
    local = torch.ones((1, 4, 2), dtype=torch.float64, device=device)
    weights = torch.full(
        (4, 2, 2), torch.finfo(torch.float64).max, dtype=torch.float64, device=device
    )
    with pytest.raises(ValueError):
        module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=1.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "fault", ("local_shape", "graph_shape", "weight_shape", "mask_shape", "steps", "gain")
)
def test_invalid_numerical_geometry_or_controls_fail_explicitly(device, fault):
    module = hierarchy_api()
    local, graph, weights, available = operands(device)
    local_steps, gain = 2, 0.35
    if fault == "local_shape":
        local = local[:, :3]
    elif fault == "graph_shape":
        graph = graph[:, :6, :6]
    elif fault == "weight_shape":
        weights = weights[:3]
    elif fault == "mask_shape":
        available = available[:, :3]
    elif fault == "steps":
        local_steps = True
    else:
        gain = float("inf")
    with pytest.raises(ValueError):
        module.hierarchy_mix(local, graph, weights, available, local_steps=local_steps, gain=gain)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "topology,feedback", (("tree", True), ("star", True), ("tree", False), ("star", False))
)
def test_last_group_change_reaches_leaves_only_when_the_final_graph_has_a_path(
    device, topology, feedback, monkeypatch
):
    module = hierarchy_api()
    assert hasattr(module, "_phase"), "Falta la fase jerárquica contrastable"
    local, _, weights, available = scalar_case(device, (0.25, -0.1, 0.3, -0.2))
    graph = numbers(reference_graph([[True] * 4], topology, feedback), device)
    baseline = module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=0.35)
    original = module._phase
    group_calls = 0

    def change_last_group(current, matrix, receivers, present, gain, start, stop):
        nonlocal group_calls
        result = original(current, matrix, receivers, present, gain, start, stop)
        if start == 4:
            group_calls += 1
            if group_calls == 4:
                delta = torch.zeros_like(result)
                delta[:, 4, 0] = 0.125
                result = result + delta
        return result

    monkeypatch.setattr(module, "_phase", change_last_group)
    changed = module.hierarchy_mix(local, graph, weights, available, local_steps=2, gain=0.35)
    assert group_calls == 4
    torch.testing.assert_close(changed[:, 4], baseline[:, 4] + 0.125, **TOLERANCE)
    torch.testing.assert_close(changed[:, 5:], baseline[:, 5:], **TOLERANCE)
    if topology == "tree" and feedback:
        assert not torch.allclose(changed[:, :2], baseline[:, :2], **TOLERANCE)
        torch.testing.assert_close(changed[:, 2:4], baseline[:, 2:4], **TOLERANCE)
    else:
        torch.testing.assert_close(changed[:, :4], baseline[:, :4], **TOLERANCE)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("start,stop", ((4, 6), (6, 7), (0, 4)))
def test_phase_jacobian_matches_the_equation_for_all_seven_independent_states(device, start, stop):
    module = hierarchy_api()
    assert hasattr(module, "_phase"), "Falta la fase jerárquica contrastable"
    values = (0.1, -0.2, 0.3, -0.1, 0.15, -0.05, 0.2)
    receiver_values = (0.7, 0.9, 1.1, 1.3)
    state = numbers([[list((value,)) for value in values]], device).requires_grad_()
    graph_values = reference_graph([[True] * 4], "tree")[0]
    graph = numbers([graph_values], device)
    weights = numbers([[[value]] for value in receiver_values], device)
    present = torch.ones((1, 7), dtype=torch.bool, device=device)
    gain = 0.37
    actual = torch.autograd.functional.jacobian(
        lambda current: module._phase(current, graph, weights, present, gain, start, stop), state
    ).reshape(7, 7)
    expected = []
    for receiver in range(7):
        row = []
        for sender in range(7):
            same = float(receiver == sender)
            if not start <= receiver < stop:
                row.append(same)
                continue
            derivative = 1.0
            if receiver < 4:
                message = sum(graph_values[receiver][j] * values[j] for j in range(7))
                weight = receiver_values[receiver]
                derivative = weight * (1 - math.tanh(message * weight) ** 2)
            row.append((1 - gain) * same + gain * graph_values[receiver][sender] * derivative)
        expected.append(row)
    torch.testing.assert_close(actual, numbers(expected, device), **TOLERANCE)
