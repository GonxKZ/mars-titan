"""Lectura del puente de tres factores (#784, P1) en H1 y en el predictor, sin optimizador."""

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_financial_adapter import api
from test_holographic_network import (
    H1,
    H2,
    H3,
    REPRESENTATION,
    module_inputs,
    name,
    network,
    prepared,
    reference,
    setup,
    single_encoders,
)

from mars_titan.models.baselines.multimodal import MODALITIES
from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.frozen_financial import _SOURCE_MODULES, FrozenFinancialConsumer
from mars_titan.models.titans.holographic_network import (
    HolographicNetwork,
    area_edges,
    plastic_edges,
)
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.plastic_bridge import (
    MATURE_DIRECTIONAL_HIT,
    MEAN_THIRD_FACTOR,
    ThreeFactorBridge,
)
from mars_titan.models.titans.reliability_gates import MatureReliability
from mars_titan.training.dust.financial import FinancialDustAdapter
from mars_titan.training.search_cases import checked_predictor_changes

P1 = ThreeFactorBridge()
COMMON = ThreeFactorBridge(source=MATURE_DIRECTIONAL_HIT, time_constant=None)
PLASTIC = [replace(H1, areas=5), replace(H1, areas=19), replace(H2, areas=38)]


def edge_bias(net, rows=5, seed=4, scale=0.4):
    generator = torch.Generator().manual_seed(seed)
    count = len(net.plastic_edges)
    return (torch.rand(rows, count, generator=generator, dtype=torch.float64) - 0.5) * scale


@pytest.mark.parametrize("areas", [5, 19, 38])
@pytest.mark.parametrize("topology", ["ring", "small_world", "complete"])
def test_the_plastic_edges_are_the_distinct_area_pairs_of_the_topology(areas, topology):
    config = replace(H1, areas=areas, topology=topology)
    edges = area_edges(areas, topology)
    expected = [(i, j) for i in range(areas) for j in range(areas) if i != j and edges[i, j]]
    assert list(plastic_edges(config)) == expected
    assert HolographicNetwork(16, config, REPRESENTATION, seed=1).plastic_edges == tuple(expected)
    if topology == "ring":
        assert len(expected) == 2 * areas
    assert plastic_edges(H3) == ()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("config", PLASTIC, ids=name)
def test_zero_weights_keep_the_bridge_output_bit_for_bit(config, dtype):
    net = network(config)
    blocks, presence, inputs, encoders = module_inputs()
    if dtype == torch.float32:
        net = copy.deepcopy(net).float()
        inputs = {key: value.float() for key, value in inputs.items()}
        blocks, encoders = blocks.float(), single_encoders(encoders)
    plain = net(blocks, presence, inputs, encoders)
    traces = []
    zero = torch.zeros(len(blocks), len(net.plastic_edges), dtype=dtype)
    assert torch.equal(
        net(blocks, presence, inputs, encoders, edge_bias=zero, traces=traces), plain
    )
    assert len(traces) == 1 and traces[0].shape == zero.shape and traces[0].dtype == dtype
    assert not traces[0].requires_grad


@pytest.mark.parametrize(
    "config",
    [*PLASTIC, replace(H1, areas=19, topology="complete", local_steps=2, cycles=3)],
    ids=name,
)
def test_the_bias_and_the_trace_match_a_reference_written_row_by_row(config):
    net, arguments = network(config), module_inputs()
    bias, traces = edge_bias(net), []
    output = net(*arguments, edge_bias=bias, traces=traces)
    expected, expected_traces = reference(net, *arguments, bias=bias)
    torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(traces[0], expected_traces, rtol=1e-10, atol=1e-13)
    # Con pesos distintos de cero la salida se aparta de la referencia de H1.
    assert not torch.equal(output, net(*arguments))


def test_the_trace_is_a_probability_between_present_areas():
    config = replace(H1, areas=19, topology="complete")
    net = network(config)
    blocks, presence, inputs, encoders = module_inputs()
    traces = []
    net(blocks, presence, inputs, encoders, edge_bias=edge_bias(net), traces=traces)
    trace = traces[0]
    assert ((trace >= 0) & (trace <= 1)).all()
    owner = [b for b, members in enumerate(net.layout.members) for _ in members]
    modality = [net.layout.blocks[b][1] for b in owner]
    present = [[bool(presence[r, MODALITIES.index(m)]) for m in modality] for r in range(5)]
    for r in range(len(blocks)):
        sums = [0.0] * 19
        for index, (i, j) in enumerate(net.plastic_edges):
            if not (present[r][i] and present[r][j]):
                assert trace[r, index] == 0, (r, i, j)
            sums[i] += float(trace[r, index])
        # Cada consulta reparte su atención y una parte va a sí misma y a las ranuras.
        assert all(total < 1 for total in sums)
    assert (trace > 0).any()


def test_gradients_cross_the_biased_bridge_and_never_reach_the_weights():
    net = network(replace(H1, areas=19))
    blocks, presence, inputs, encoders = module_inputs()
    blocks.requires_grad_(True)
    output = net(blocks, presence, inputs, encoders, edge_bias=edge_bias(net), traces=[])
    output.square().sum().backward()
    assert blocks.grad is not None and blocks.grad.abs().sum() > 0
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)
    learned = edge_bias(net).requires_grad_()
    with pytest.raises(ValueError, match="sin grafo"):
        net(blocks, presence, inputs, encoders, edge_bias=learned, traces=[])


@pytest.mark.parametrize(
    ("bias", "traces"),
    [
        (lambda net: edge_bias(net), None),
        (lambda net: None, []),
        (lambda net: edge_bias(net)[:, :-1], []),
        (lambda net: edge_bias(net, rows=4), []),
        (lambda net: edge_bias(net).float(), []),
        (lambda net: edge_bias(net).t().contiguous().t(), [torch.zeros(1)]),
        (lambda net: edge_bias(net).tolist(), []),
        (lambda net: edge_bias(net), ()),
        (lambda net: edge_bias(net).to_sparse(), []),
        (lambda net: edge_bias(net).to("meta"), []),
    ],
)
def test_the_weights_need_their_contract(bias, traces):
    net = network(replace(H1, areas=19))
    with pytest.raises(ValueError, match="pesos rápidos"):
        net(*module_inputs(), edge_bias=bias(net), traces=traces)


def test_a_network_without_catalog_areas_has_no_weights_to_read():
    net = network(H3)
    with pytest.raises(ValueError, match="pesos rápidos"):
        net(*module_inputs(), edge_bias=torch.zeros(5, 0, dtype=torch.float64), traces=[])


def test_a_non_finite_weight_is_refused():
    net = network(replace(H1, areas=19))
    bias = edge_bias(net)
    bias[2, 3] = math.nan
    with pytest.raises(ValueError, match="finit"):
        net(*module_inputs(), edge_bias=bias, traces=[])


def bridge_setup(dtype=torch.float64, config=P1, network_config=H1, **changes):
    return setup(dtype, holographic_network=network_config, three_factor_bridge=config, **changes)


def moved(predictor):
    """Con la lectura a cero, H1 devuelve sus bloques y el puente no cambia la salida."""
    with torch.no_grad():
        predictor.holographic_network.readout.weight.fill_(0.05)
    predictor._seal_parameters()
    return predictor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_predictor_with_zero_weights_emits_h1_bit_for_bit(dtype):
    base, batch = setup(dtype, holographic_network=H1)
    predictor = moved(bridge_setup(dtype)[0])
    base = moved(base)
    # P1 no añade parámetros y comparte los de H1.
    assert dict(base.named_parameters()).keys() == dict(predictor.named_parameters()).keys()
    for key, value in base.named_parameters():
        assert torch.equal(dict(predictor.named_parameters())[key], value), key
    edges = len(predictor.holographic_network.plastic_edges)
    zero = torch.zeros(len(batch.flow_ids), edges, dtype=dtype)
    expected = prepared(base, batch)
    actual = predictor.prepare(batch, predictor.initial_state(batch.flow_ids), bridge_bias=zero)
    for field in ("point_predictions", "detached_tokens"):
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field
    assert torch.equal(
        actual.next_state.mac.memory.weights[0], expected.next_state.mac.memory.weights[0]
    )
    assert expected.bridge_traces is None
    assert actual.bridge_traces.shape == zero.shape and actual.bridge_traces.dtype == dtype
    biased = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), bridge_bias=zero + 0.05
    )
    assert not torch.equal(biased.detached_tokens, expected.detached_tokens)


def test_prepare_needs_the_weights_only_when_the_bridge_is_declared():
    base, batch = setup(holographic_network=H1)
    predictor, _ = bridge_setup()
    edges = len(predictor.holographic_network.plastic_edges)
    rows = len(batch.flow_ids)
    zero = torch.zeros(rows, edges, dtype=torch.float64)
    with pytest.raises(ValueError, match="no declara el puente"):
        base.prepare(batch, base.initial_state(batch.flow_ids), bridge_bias=zero)
    state = predictor.initial_state(batch.flow_ids)
    for bias, message in (
        (None, "pesos rápidos de sus filas"),
        (zero[:, 1:], "pesos rápidos de sus filas"),
        (zero[1:], "pesos rápidos de sus filas"),
        (zero.float(), "pesos rápidos de sus filas"),
        (zero.to_sparse(), "pesos rápidos de sus filas"),
        (zero.to("meta"), "pesos rápidos de sus filas"),
        (zero.clone().requires_grad_(), "pesos rápidos de sus filas"),
        (zero + P1.budget * 1.5, "dentro de su cota"),
        (zero + math.inf, "dentro de su cota"),
    ):
        with pytest.raises(ValueError, match=message):
            predictor.prepare(batch, state, bridge_bias=bias)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(holographic_network=None), "áreas de H1 o H2"),
        (dict(holographic_network=H3), "áreas de H1 o H2"),
        (dict(auxiliary_horizons=AuxiliaryHorizons()), "cabezas auxiliares"),
        (dict(three_factor_bridge=dict(kind=MEAN_THIRD_FACTOR)), "todos sus campos"),
        (dict(three_factor_bridge=dict(vars(P1), learning_rate=2.0)), "tasa"),
    ],
)
def test_the_declaration_needs_catalog_areas_and_refuses_row_by_row_tokens(changes, message):
    options = dict(holographic_network=H1, three_factor_bridge=P1) | changes
    with pytest.raises(ValueError, match=message):
        api().FinancialConfig(
            setup()[0].config.inputs, variant="mac_online", hidden_size=32, **options
        )


@pytest.mark.parametrize(
    "reliability",
    [
        None,
        MatureReliability(independent=True),
        MatureReliability(source="titans_surprise"),
        MatureReliability(source="shuffled_in_time"),
    ],
)
def test_the_common_third_factor_needs_the_shared_signal_of_hits(reliability):
    with pytest.raises(ValueError, match="señal compartida de aciertos"):
        api().FinancialConfig(
            setup()[0].config.inputs,
            variant="mac_online",
            hidden_size=32,
            holographic_network=H1,
            three_factor_bridge=COMMON,
            mature_reliability=reliability,
        )
    # La fuente de la issue no necesita la señal de #791.
    assert bridge_setup(mature_reliability=reliability)[0].config.three_factor_bridge == P1


def test_the_bridge_needs_its_own_contrast_with_the_local_control():
    predictor, _ = bridge_setup()
    with pytest.raises(ValueError, match="contraste con C"):
        api().FinancialPredictor(
            predictor.config,
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )


def test_the_disabled_field_keeps_the_identity_and_each_kind_is_another_identity():
    base, _ = setup(holographic_network=H1)
    explicit, _ = bridge_setup(config=None)
    assert explicit.config.identity() == base.config.identity()
    assert "three_factor_bridge" not in base.config.identity()
    declared = [
        P1,
        ThreeFactorBridge(kind=MEAN_THIRD_FACTOR),
        replace(P1, learning_rate=0.1),
        replace(P1, budget=0.5),
        replace(P1, time_constant=5),
        COMMON,
    ]
    identities = {
        json.dumps(
            bridge_setup(config=c, mature_reliability=MatureReliability())[0].config.identity(),
            sort_keys=True,
        )
        for c in declared
    }
    assert len(identities) == len(declared)
    parsed, _ = bridge_setup(config=dict(vars(P1)))
    assert parsed.config.three_factor_bridge == P1
    assert parsed.config.identity()["three_factor_bridge"] == P1.identity()
    # H2 tiene las mismas áreas y también admite el puente.
    assert bridge_setup(network_config=H2)[0].config.three_factor_bridge == P1


def test_the_registry_admits_it_and_the_consumers_refuse_it():
    changes = dict(holographic_network=vars(H1), three_factor_bridge=vars(P1))
    assert checked_predictor_changes(changes) == changes
    predictor, _ = bridge_setup()
    with pytest.raises(ValueError, match="red holográfica"):
        FrozenFinancialConsumer(predictor)
    with pytest.raises(ValueError):
        FinancialDustAdapter(predictor)
    # La configuración entra en la implementación de todo predictor.
    assert "mars_titan.models.titans.plastic_bridge" in _SOURCE_MODULES


def test_the_declared_arms_build_with_their_control():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads((root / "configs/experiments/three-factor-bridge-v1.json").read_text())
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    identities = set()
    for arm_name, arm in declaration["arms"].items():
        changes = arm["predictor_changes"]
        assert checked_predictor_changes(changes) == changes
        predictor, _ = setup(**changes)
        identities.add(json.dumps(predictor.config.identity(), sort_keys=True))
        # Cada configuración declarada comparte los parámetros de su referencia sin P1.
        base = {key: value for key, value in changes.items() if key != "three_factor_bridge"}
        reference_parameters = dict(setup(**base)[0].named_parameters())
        names = dict(predictor.named_parameters())
        assert names.keys() == reference_parameters.keys(), arm_name
        for key, value in reference_parameters.items():
            assert torch.equal(names[key], value), (arm_name, key)
    assert len(identities) == len(declaration["arms"])
    for contrast in declaration["contrasts"]:
        assert {contrast["variant"], contrast["control"]} <= set(declaration["arms"])
