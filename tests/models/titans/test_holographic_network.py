"""Red de redes holográfica de #783 en el nivel de la fila (H1, H2 y H3).

Las pruebas calculan salidas y gradientes sobre lotes ficticios, sin optimizador ni pasos.
"""

import copy
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_adapter import api
from torch import nn
from torch.nn import functional as F

from mars_titan.data.accounting_catalog import JOINT_CONCEPTS, USD_CONCEPTS
from mars_titan.data.company_factors import FACTOR_CONCEPTS
from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.models.baselines.multimodal import MODALITIES
from mars_titan.models.titans.holographic_network import (
    CATALOG,
    CATALOG_MEMBERSHIP_ONLY,
    HIDDEN_GROUPS_ONLY,
    HOLOGRAPHIC_NETWORK,
    MACRO_FAMILIES,
    HolographicNetwork,
    HolographicNetworkConfig,
    area_edges,
    area_layout,
    macro_categories,
)
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.regional_variants import RegionAxisMixingConfig
from mars_titan.training import mars_titan_walk_forward
from mars_titan.training.search_cases import checked_predictor_changes

# Las mismas listas que la edición v3.1: 26 conceptos contables y los 140 indicadores.
FUNDAMENTALS = list(JOINT_CONCEPTS)
INDICATORS = sorted(macro_categories())
REPRESENTATION = dict(fundamental_concepts=FUNDAMENTALS, macro_indicators=INDICATORS)
DIMENSIONS = dict(prices=5, news=2, charts=2, fundamentals=78, macro=420)
AT = 1_609_459_200_000_000
H1 = HolographicNetworkConfig()
H2 = HolographicNetworkConfig(kind=CATALOG_MEMBERSHIP_ONLY)
H3 = HolographicNetworkConfig(
    kind=HIDDEN_GROUPS_ONLY, hidden_groups=6, rank=None, leakage=None, topology="complete"
)
KINDS = (H1, H2, H3)
KIND_DEFAULTS = {config.kind: config for config in KINDS}


def configs():
    """Las tres identidades con 5, 19 y 38 áreas."""
    return [replace(config, areas=areas) for config in KINDS for areas in (5, 19, 38)]


def name(config):
    changed = [
        f"{key}={value}"
        for key, value in asdict(config).items()
        if key not in ("kind", "areas") and value != getattr(KIND_DEFAULTS[config.kind], key)
    ]
    return "-".join([config.kind, str(config.areas), *changed])


def specification(*, historical=True):
    policy = HISTORICAL_MASKED if historical else STRICT_INPUTS
    representation = dict(
        **policy_identity(policy),
        **REPRESENTATION,
        encoders={"fixture": "frozen-v1"},
        representation_code={"fixture.py": "c" * 64},
        text_aggregation="mean",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    return api().FinancialInputSpec(
        source_sha256="a" * 64,
        view_sha256="b" * 64,
        representation=representation,
        dimensions=DIMENSIONS,
        input_policy=policy,
    )


def masked_vector(generator, rows, concepts, present):
    """Valores, máscaras y edades de la disposición histórica, a cero si falta el bloque."""
    observed = generator.random((rows, concepts)) < 0.6
    observed[present & ~observed.any(1), 0] = True
    observed[~present] = False
    data = np.where(observed, generator.normal(size=(rows, concepts)), 0)
    ages = np.where(observed, generator.uniform(0, 5, size=(rows, concepts)), 0)
    return np.concatenate([data, observed, ages], axis=1).astype(np.float32).round(3)


def raw_batch(rows=4, seed=11):
    generator = np.random.default_rng(seed)
    presence = np.ones((rows, 5), dtype=np.bool_)
    presence[1, [1, 3]] = False
    presence[2, 4] = False
    values = {
        key: generator.normal(size=(rows, 64, width) if key == "prices" else (rows, width))
        .astype(np.float32)
        .round(3)
        for key, width in DIMENSIONS.items()
        if key in ("prices", "news", "charts")
    }
    values["news"][~presence[:, 1]] = 0
    values["fundamentals"] = masked_vector(generator, rows, len(FUNDAMENTALS), presence[:, 3])
    values["macro"] = masked_vector(generator, rows, len(INDICATORS), presence[:, 4])
    return dict(
        inputs=values,
        presence=presence,
        sample_ids=[f"US/A{index:03}/{AT}" for index in range(rows)],
        market=["US"] * rows,
        prediction_at=np.full(rows, AT, dtype="datetime64[us]"),
        input_available_at=np.full(rows, AT - 1, dtype="datetime64[us]"),
    )


def setup(dtype=torch.float64, **changes):
    spec = specification()
    config = api().FinancialConfig(
        spec, variant="mac_online", hidden_size=32, layers=1, seed=42, **changes
    )
    model = api().FinancialPredictor(config, dtype=dtype)
    batch = api().DecisionBatch.from_corpus(raw_batch(), spec, dtype=dtype)
    return model, batch


def prepared(model, batch):
    return model.prepare(batch, model.initial_state(batch.flow_ids))


def network(config=H1, *, hidden=16, seed=3, moved=True):
    """Módulo en FP64. Con `moved`, la lectura, los ajustes U y Δ dejan de valer cero."""
    result = HolographicNetwork(hidden, config, REPRESENTATION, seed=seed).double()
    if moved:
        generator = torch.Generator().manual_seed(seed + 1)
        values = dict(result.named_parameters())
        with torch.no_grad():
            for key, scale in (("readout.weight", 4), ("area_up", 4), ("membership_delta", 1)):
                if key in values:
                    shape = values[key].shape
                    values[key].copy_(
                        torch.randn(shape, generator=generator, dtype=torch.float64) / scale
                    )
    return result


def module_inputs(rows=5, hidden=16, seed=9):
    """Bloques, presencia, entradas de familia y codificadores lineales ficticios."""
    generator = torch.Generator().manual_seed(seed)
    presence = torch.ones(rows, 5, dtype=torch.bool)
    presence[1, 1] = presence[2, 4] = presence[3, 3] = False
    blocks = torch.randn(rows, 5, hidden, generator=generator, dtype=torch.float64)
    blocks = blocks * presence.unsqueeze(-1)
    raw = np.random.default_rng(seed)
    inputs = {
        "fundamentals": masked_vector(raw, rows, len(FUNDAMENTALS), presence[:, 3].numpy()),
        "macro": masked_vector(raw, rows, len(INDICATORS), presence[:, 4].numpy()),
    }
    inputs = {key: torch.tensor(value, dtype=torch.float64) for key, value in inputs.items()}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoders = nn.ModuleDict(
            {
                key: nn.Sequential(nn.Linear(DIMENSIONS[key], hidden), nn.SiLU()).double()
                for key in ("fundamentals", "macro")
            }
        )
    return blocks, presence, inputs, encoders


def single_encoders(encoders):
    return copy.deepcopy(encoders).float()


def reference(net, blocks, presence, inputs, encoders):
    """Las ecuaciones del módulo fila a fila, nodo a nodo y bloque a bloque."""
    config, layout, width = net.config, net.layout, net.hidden_size
    areas, slots = config.catalog_areas, config.hidden_groups
    owners = [b for b, members in enumerate(layout.members) for _ in members]
    if areas:
        # Los pesos del catálogo salen de su definición y no del buffer del módulo.
        weights = torch.empty(areas, len(layout.blocks), dtype=torch.float64)
        for b, members in enumerate(layout.members):
            for a in range(areas):
                inside = a in members
                share = 1 - config.leakage if inside else config.leakage
                weights[a, b] = share / (len(members) if inside else areas - len(members))
        logits = weights.log()
        if config.kind == HOLOGRAPHIC_NETWORK:
            logits = logits + net.membership_delta
        p = torch.softmax(logits, dim=0)
    result = blocks.clone()
    for r in range(len(blocks)):
        sources, present = [], []
        for _, modality, columns in layout.blocks:
            index = MODALITIES.index(modality)
            if columns is None:
                sources.append(blocks[r, index])
            else:
                linear, chosen = encoders[modality][0], list(columns)
                value = linear.weight[:, chosen] @ inputs[modality][r, chosen] + linear.bias
                sources.append(F.silu(value) * presence[r, index])
            present.append(bool(presence[r, index]))
        count = len(sources)
        nodes, alive, combine = [], [], []
        for a in range(areas):
            mixed = sum(p[a, b] * sources[b] for b in range(count))
            adapted = mixed + (mixed @ net.area_down[a]) @ net.area_up[a]
            nodes.append(net.area_norm(adapted + net.area_embedding[a]))
            alive.append(present[owners[a]])
        if slots:
            keys = [net.block_norm(source) for source in sources]
            scores = [
                [keys[b] @ net.slot_keys[s] / math.sqrt(width) for b in range(count)]
                for s in range(slots)
            ]
            chosen = [b for b in range(count) if present[b]]
            for s in range(slots):
                weights = torch.softmax(torch.stack([scores[s][b] for b in chosen]), 0)
                gathered = sum(w * sources[b] for w, b in zip(weights, chosen, strict=True))
                nodes.append(net.slot_norm(gathered + net.slot_embedding[s]))
                alive.append(True)
            combine = [
                torch.softmax(torch.stack([scores[s][b] for s in range(slots)]), 0)
                for b in range(count)
            ]
        initial = [n if ok else torch.zeros_like(n) for n, ok in zip(nodes, alive, strict=True)]
        local = list(initial)
        summary, context = blocks.new_zeros(width), blocks.new_zeros(width)
        for _ in range(config.cycles):
            for _ in range(config.local_steps):
                parts = (net.qkv(net.local_norm(n + context)).chunk(3) for n in local)
                q, k, v = zip(*parts, strict=True)
                updated = []
                for i, state in enumerate(local):
                    if not alive[i]:
                        updated.append(state)
                        continue
                    seen = [j for j in range(len(local)) if net.edges[i, j] and alive[j]]
                    weights = torch.softmax(
                        torch.stack([q[i] @ k[j] / math.sqrt(width) for j in seen]), 0
                    )
                    message = sum(w * v[j] for w, j in zip(weights, seen, strict=True))
                    updated.append(state + config.local_gain * net.local_out(message))
                local = updated
            if config.summary:
                pooled = sum(n for n, ok in zip(local, alive, strict=True) if ok) / sum(alive)
                update = net.summary_update(net.summary_norm(summary) + pooled)
                summary = summary + config.summary_gain * update
                context = net.summary_to_local(summary)
        change = [after - before for after, before in zip(local, initial, strict=True)]
        for m, modality in enumerate(MODALITIES):
            if not presence[r, m]:
                continue
            owned = [b for b, (_, owner, _) in enumerate(layout.blocks) if owner == modality]
            total = blocks.new_zeros(width)
            for b in owned:
                delta = context + sum(
                    (p[a, b] * change[a] for a in range(areas)), blocks.new_zeros(width)
                )
                for s in range(slots):
                    delta = delta + combine[b][s] * change[areas + s]
                total = total + delta
            result[r, m] = blocks[r, m] + net.readout(total / len(owned))
    return result


# Variantes de la configuración que recorren las demás ramas del puente.
OPTIONS = [
    replace(H1, hidden_groups=0),
    replace(H2, areas=38, summary=False, local_steps=2, topology="small_world"),
    replace(H3, areas=5, cycles=1, hidden_groups=1),
]


@pytest.mark.parametrize("config", configs() + OPTIONS, ids=name)
def test_the_module_matches_a_reference_written_area_by_area(config):
    net, arguments = network(config), module_inputs()
    torch.testing.assert_close(net(*arguments), reference(net, *arguments), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("config", [replace(H1, areas=38), replace(H3, areas=19)], ids=name)
def test_fp32_follows_the_fp64_module(config):
    net = network(config)
    blocks, presence, inputs, encoders = module_inputs()
    expected = net(blocks, presence, inputs, encoders)
    single = {key: value.float() for key, value in inputs.items()}
    arguments = (blocks.float(), presence, single, single_encoders(encoders))
    output = copy.deepcopy(net).float()(*arguments)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output.double(), expected, rtol=1e-4, atol=1e-5)
    # Recién construido, el módulo trabaja en FP32 y redondea el catálogo FP64 al usarlo.
    fresh = HolographicNetwork(16, config, REPRESENTATION, seed=3)
    if config.catalog_areas:
        assert fresh.log_catalog.dtype == torch.float64
    assert fresh(*arguments).dtype == torch.float32


@pytest.mark.parametrize("areas", [19, 38])
def test_the_layouts_split_every_family_column_once(areas):
    layout = area_layout(areas, REPRESENTATION)
    assert len(layout.areas) == len(set(layout.areas)) == areas
    assert sorted(i for members in layout.members for i in members) == list(range(areas))
    for modality, width in (("fundamentals", 78), ("macro", 420)):
        columns = [c for _, m, chosen in layout.blocks if m == modality for c in chosen]
        assert sorted(columns) == list(range(width)), modality
    families = dict(layout.families)
    assert set(families) == {"fundamentals_balance", "fundamentals_ratios"} | {
        f"macro_{family}" for family in MACRO_FAMILIES
    }
    assert families["fundamentals_balance"] == tuple(
        c for c in JOINT_CONCEPTS if c not in FACTOR_CONCEPTS
    )
    assert families["fundamentals_ratios"] == FACTOR_CONCEPTS
    categories = macro_categories()
    for family in MACRO_FAMILIES:
        assert {categories[i] for i in families[f"macro_{family}"]} == {family}
    blocks = {block: columns for block, _, columns in layout.blocks}
    first = INDICATORS.index(families["macro_inflation"][0])
    if areas == 38:
        # Los valores van a un área y la máscara y la edad, a la otra.
        assert first in blocks["macro_inflation_values"]
        assert {140 + first, 280 + first} <= set(blocks["macro_inflation_observation"])
        assert len(layout.blocks) == 31
    else:
        assert {first, 140 + first, 280 + first} <= set(blocks["macro_inflation"])
        assert len(layout.blocks) == 17


def test_a_us_only_representation_keeps_both_fundamental_families():
    layout = area_layout(19, dict(REPRESENTATION, fundamental_concepts=list(USD_CONCEPTS)))
    families = dict(layout.families)
    assert families["fundamentals_balance"] == USD_CONCEPTS[:8]
    assert families["fundamentals_ratios"] == FACTOR_CONCEPTS
    assert dict(layout.widths)["fundamentals"] == 45


def test_without_presence_every_modality_counts_as_present():
    net = network(replace(H1, areas=5))
    blocks, presence, inputs, encoders = module_inputs()
    full = torch.ones_like(presence)
    assert torch.equal(net(blocks, None), net(blocks, full, inputs, encoders))
    with pytest.raises(ValueError, match="admite"):
        area_layout(7)


def test_the_five_areas_are_the_modalities():
    layout = area_layout(5)
    assert layout.areas == MODALITIES and layout.members == ((0,), (1,), (2,), (3,), (4,))
    assert all(columns is None for _, _, columns in layout.blocks)
    assert not network(replace(H1, areas=5), moved=False).family_blocks


@pytest.mark.parametrize("areas", [5, 19, 38])
def test_the_catalog_membership_sums_to_one_with_its_leakage(areas):
    net = network(replace(H2, areas=areas), moved=False)
    p = torch.softmax(net.log_catalog, dim=0)
    torch.testing.assert_close(p.sum(0), torch.ones(p.shape[1], dtype=p.dtype))
    for block, members in enumerate(net.layout.members):
        torch.testing.assert_close(p[list(members), block].sum(), torch.tensor(0.9, dtype=p.dtype))
        outside = [a for a in range(areas) if a not in members]
        torch.testing.assert_close(p[outside, block], p[outside[0], block].expand(len(outside)))


def test_family_blocks_are_the_encoder_on_the_family_columns():
    net = network(replace(H1, areas=19), moved=False)
    blocks, presence, inputs, encoders = module_inputs()
    sources = net._source_blocks(blocks, presence, inputs, encoders)
    assert sources.shape == (5, 17, 16)
    for index, (_, modality, columns) in enumerate(net.layout.blocks):
        if columns is None:
            assert torch.equal(sources[:, index], blocks[:, MODALITIES.index(modality)])
            continue
        only = torch.zeros_like(inputs[modality])
        only[:, list(columns)] = inputs[modality][:, list(columns)]
        expected = encoders[modality](only) * presence[:, MODALITIES.index(modality), None]
        torch.testing.assert_close(sources[:, index], expected, rtol=1e-12, atol=1e-14)


def test_the_areas_of_one_block_start_apart():
    net = network(replace(H1, areas=19), moved=False)
    blocks, presence, inputs, encoders = module_inputs()
    sources = net._source_blocks(blocks, presence, inputs, encoders)
    p = torch.softmax(net.log_catalog + net.membership_delta, dim=0)
    states = net.area_norm(torch.einsum("ab,rbh->rah", p, sources) + net.area_embedding)
    price = list(net.layout.members[0])
    assert len(price) == 3
    for i in price:
        for j in price:
            if i < j:
                assert not torch.allclose(states[:, i], states[:, j])


@pytest.mark.parametrize("config", configs(), ids=name)
def test_gradients_reach_every_parameter_once_the_readout_moves(config):
    net, arguments = network(config), module_inputs()
    net(*arguments).square().sum().backward()
    for parameter, value in net.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), parameter
        assert torch.count_nonzero(value.grad) > 0, parameter


def test_with_the_readout_at_zero_only_the_readout_learns_first():
    net, arguments = network(H1, moved=False), module_inputs()
    output = net(*arguments)
    assert torch.equal(output, arguments[0])
    output.square().sum().backward()
    for parameter, value in net.named_parameters():
        moved = value.grad is not None and bool(torch.count_nonzero(value.grad))
        assert moved is (parameter == "readout.weight"), parameter


@pytest.mark.parametrize("config", [replace(H1, areas=19), replace(H3, areas=38)], ids=name)
def test_each_row_is_computed_alone(config):
    net = network(config)
    blocks, presence, inputs, encoders = module_inputs()
    base = net(blocks, presence, inputs, encoders)
    changed, other_inputs = blocks.clone(), dict(inputs)
    changed[0] += 3
    other_inputs["macro"] = inputs["macro"].clone()
    other_inputs["macro"][0, :140] += 2
    other = net(changed, presence, other_inputs, encoders)
    assert not torch.equal(other[0], base[0])
    assert torch.equal(other[1:], base[1:])


def test_an_absent_modality_keeps_its_block_and_sends_nothing():
    net = network(replace(H1, areas=19))
    blocks, presence, inputs, encoders = module_inputs()
    output = net(blocks, presence, inputs, encoders)
    # La fila 2 no tiene macro y la 3 no tiene fundamentales.
    assert torch.equal(output[2, 4], blocks[2, 4]) and torch.equal(output[3, 3], blocks[3, 3])
    assert not torch.equal(output[2, 0], blocks[2, 0])
    sources = net._source_blocks(blocks, presence, inputs, encoders)
    macro = [i for i, (_, m, _) in enumerate(net.layout.blocks) if m == "macro"]
    assert torch.count_nonzero(sources[2, macro]) == 0
    # Lo que contenga la entrada de un bloque ausente no llega a la fila.
    altered = dict(inputs, macro=inputs["macro"].clone())
    altered["macro"][2] = 7.0
    assert torch.equal(net(blocks, presence, altered, encoders), output)


def test_an_absent_area_starts_at_zero_and_sends_no_messages():
    net = network(replace(H1, areas=19))
    blocks, presence, inputs, encoders = module_inputs()
    base = net(blocks, presence, inputs, encoders)
    with torch.no_grad():
        net.area_embedding[list(net.layout.members[1])] += 5
    moved = net(blocks, presence, inputs, encoders)
    # La fila 1 no tiene noticias, así que el área de noticias no cambia nada en ella.
    assert torch.equal(moved[1], base[1])
    assert not torch.equal(moved[0], base[0])


def test_the_three_kinds_share_the_slots_and_the_bridge():
    nets = {c.kind: network(replace(c, hidden_groups=6), moved=False) for c in KINDS}
    shared = dict(nets[HIDDEN_GROUPS_ONLY].named_parameters())
    assert not any(key.startswith("area_") for key in shared)
    for kind, net in nets.items():
        own = dict(net.named_parameters())
        for key, value in shared.items():
            assert torch.equal(own[key], value), (kind, key)
    assert "membership_delta" not in dict(nets[CATALOG_MEMBERSHIP_ONLY].named_parameters())
    area_keys = set(dict(nets[HOLOGRAPHIC_NETWORK].named_parameters())) - set(shared)
    assert area_keys == {
        "area_down",
        "area_up",
        "area_embedding",
        "area_norm.weight",
        "area_norm.bias",
        "membership_delta",
    }


def test_the_topologies_of_the_areas():
    ring, small, complete = (area_edges(19, t) for t in ("ring", "small_world", "complete"))
    for edges in (ring, small, complete):
        assert torch.equal(edges, edges.t()) and edges.diagonal().all()
    assert ring.sum(1).tolist() == [3] * 19
    # Las cuerdas a un cuarto del anillo añaden dos vecinos por área.
    assert small.sum(1).tolist() == [5] * 19 and bool(small[0, 4]) and bool(small[0, 15])
    assert complete.all()
    assert torch.equal(area_edges(5, "small_world"), area_edges(5, "ring"))
    net = network(replace(H1, areas=19, hidden_groups=2), moved=False)
    assert torch.equal(net.edges[:19, :19], ring)
    assert net.edges[19:].all() and net.edges[:, 19:].all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("config", KINDS, ids=name)
def test_each_kind_starts_as_the_reference_predictor(config, dtype):
    base, batch = setup(dtype)
    predictor, _ = setup(dtype, holographic_network=config)
    reference_parameters = dict(base.named_parameters())
    names = dict(predictor.named_parameters())
    added = set(names) - set(reference_parameters)
    assert added and all(key.startswith("holographic_network.") for key in added)
    for key, value in reference_parameters.items():
        assert torch.equal(names[key], value), key
    expected, actual = prepared(base, batch), prepared(predictor, batch)
    for field in ("point_predictions", "detached_tokens"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    torch.testing.assert_close(
        actual.next_state.mac.memory.weights[0],
        expected.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    identity = predictor.config.identity()["holographic_network"]
    assert identity == config.identity() and identity["levels"] == 1


def test_the_global_generator_is_not_consumed():
    torch.manual_seed(5)
    setup()
    plain = torch.rand(3)
    torch.manual_seed(5)
    setup(holographic_network=H1)
    assert torch.equal(torch.rand(3), plain)


def test_the_disabled_field_keeps_identity_and_predictions():
    base, batch = setup()
    explicit, _ = setup(holographic_network=None)
    assert explicit.config.identity() == base.config.identity()
    assert "holographic_network" not in base.config.identity()
    torch.testing.assert_close(
        prepared(explicit, batch).point_predictions,
        prepared(base, batch).point_predictions,
        rtol=0,
        atol=0,
    )


def test_each_declaration_is_its_own_identity():
    declared = [*configs(), replace(H1, topology="small_world"), replace(H1, rank=2)]
    identities = {
        json.dumps(setup(holographic_network=c)[0].config.identity(), sort_keys=True)
        for c in declared
    }
    assert len(identities) == len(declared)


def test_the_predictor_passes_its_batch_and_encoders_to_the_network():
    predictor, batch = setup(holographic_network=H1)
    seen = []
    handle = predictor.holographic_network.register_forward_hook(
        lambda _module, arguments, output: seen.append((arguments, output))
    )
    try:
        prepared(predictor, batch)
    finally:
        handle.remove()
    assert len(seen) == 1
    (blocks, presence, inputs, encoders), output = seen[0]
    assert encoders is predictor.encoders and torch.equal(presence, batch.presence)
    for key in ("fundamentals", "macro"):
        assert torch.equal(inputs[key], batch.inputs[key])
    assert blocks.shape == (4, 5, 32) and torch.equal(output, blocks)
    # Con la lectura distinta de cero, la red cambia la predicción y la memoria.
    with torch.no_grad():
        predictor.holographic_network.readout.weight.fill_(0.05)
    predictor._seal_parameters()
    base, _ = setup()
    moved, plain = prepared(predictor, batch), prepared(base, batch)
    assert not torch.equal(moved.point_predictions, plain.point_predictions)
    assert not torch.equal(
        moved.next_state.mac.memory.weights[0], plain.next_state.mac.memory.weights[0]
    )


def test_the_state_dict_round_trips_and_another_declaration_is_refused():
    source, batch = setup(holographic_network=H1)
    with torch.no_grad():
        source.holographic_network.readout.weight.fill_(0.1)
        source.holographic_network.membership_delta.fill_(0.3)
    source._seal_parameters()
    target, _ = setup(holographic_network=H1)
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(
        prepared(target, batch).point_predictions,
        prepared(source, batch).point_predictions,
        rtol=0,
        atol=0,
    )
    for other in (H2, replace(H1, topology="complete")):
        with pytest.raises((ValueError, RuntimeError)):
            setup(holographic_network=other)[0].load_state_dict(source.state_dict())


def test_the_operation_counts_follow_the_nodes():
    config = replace(H1, areas=38, hidden_groups=2, local_steps=2)
    predictor, _ = setup(holographic_network=config)
    assert predictor.operation_counts(10) == dict(
        holographic_node_evaluations=10 * 40 * 2 * 2, holographic_summary_evaluations=10 * 2
    )
    predictor, _ = setup(holographic_network=replace(H3, summary=False))
    assert predictor.operation_counts(3) == dict(
        holographic_node_evaluations=3 * 6 * 2, holographic_summary_evaluations=0
    )


@pytest.mark.parametrize(
    "value, message",
    [
        (dict(asdict(H1), kind="holographic_network_v2"), "admite"),
        (dict(asdict(H1), areas=7), "áreas"),
        (dict(asdict(H1), topology="star"), "topología"),
        (dict(asdict(H1), rank=0), "rango"),
        (dict(asdict(H1), rank=None), "rango"),
        (dict(asdict(H1), leakage=0.0), "fuga"),
        (dict(asdict(H1), leakage=1.0), "fuga"),
        (dict(asdict(H1), leakage=None), "fuga"),
        (dict(asdict(H1), hidden_groups=9), "ranuras"),
        (dict(asdict(H1), cycles=5), "ciclos"),
        (dict(asdict(H1), local_steps=3), "pasos"),
        (dict(asdict(H1), summary=1), "booleano"),
        (dict(asdict(H1), local_gain=1.5), "ganancias"),
        (dict(asdict(H1), summary_gain=float("nan")), "ganancias"),
        (dict(asdict(H1), levels=2), "niveles"),
        (dict(asdict(H1), levels=True), "niveles"),
        (dict(asdict(H3), rank=4), "H3"),
        (dict(asdict(H3), topology="ring"), "H3"),
        (dict(asdict(H3), hidden_groups=0), "ranuras"),
        ({k: v for k, v in asdict(H1).items() if k != "levels"}, "toda su configuración"),
        ("holographic_network_v1", "toda su configuración"),
    ],
)
def test_invalid_declarations_are_rejected(value, message):
    base, _ = setup()
    with pytest.raises(ValueError, match=message):
        replace(base.config, holographic_network=value)


def test_family_areas_need_the_masked_representation():
    spec = specification(historical=False)
    with pytest.raises(ValueError, match="máscaras"):
        api().FinancialConfig(spec, variant="mac_online", holographic_network=H1)
    config = api().FinancialConfig(
        spec, variant="mac_online", holographic_network=replace(H1, areas=5)
    )
    assert config.holographic_network.areas == 5
    with pytest.raises(ValueError, match="categorías"):
        area_layout(19, dict(REPRESENTATION, macro_indicators=["unknown_indicator"]))
    with pytest.raises(ValueError, match="balance o un ratio"):
        area_layout(19, dict(REPRESENTATION, fundamental_concepts=["unknown:Concept:USD"]))
    with pytest.raises(ValueError, match="al menos un concepto"):
        area_layout(19, dict(REPRESENTATION, macro_indicators=INDICATORS[:3]))


@pytest.mark.parametrize(
    "changes",
    [
        dict(variant="mac_frozen"),
        dict(variant="transformer_direct"),
        dict(presence_modulation="film_v1"),
        dict(regional_fusion=RegionalFusionConfig(topology="ring")),
        dict(regional_variant=RegionAxisMixingConfig(RegionalFusionConfig(topology="ring"))),
    ],
)
def test_it_takes_the_modulation_alone_on_mac_online(changes):
    options = dict(variant="mac_online", hidden_size=32, holographic_network=H1) | changes
    with pytest.raises(ValueError, match="mac_online"):
        api().FinancialConfig(specification(), **options)


def test_the_module_refuses_other_contracts():
    net = network(replace(H1, areas=19), moved=False)
    blocks, presence, inputs, encoders = module_inputs()
    with pytest.raises(ValueError, match="contrato"):
        net(blocks.float(), presence, inputs, encoders)
    small = HolographicNetwork(16, H1, REPRESENTATION, seed=1, max_batch=4).double()
    with pytest.raises(ValueError, match="contrato"):
        small(blocks, presence, inputs, encoders)
    with pytest.raises(ValueError, match="presencia"):
        net(blocks, presence.double(), inputs, encoders)
    with pytest.raises(ValueError, match="codificador lineal"):
        net(blocks, presence, dict(inputs, macro=inputs["macro"][:, :30]), encoders)
    with pytest.raises(ValueError, match="codificador lineal"):
        net(blocks, presence, inputs, None)
    with pytest.raises(ValueError, match="codificador lineal"):
        net(blocks, presence, dict(inputs, macro=inputs["macro"].float()), encoders)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        net(blocks, presence, inputs, encoders)
    infinite = blocks.clone()
    infinite[0, 0, 0] = math.inf
    with pytest.raises(ValueError, match="finit"):
        net(infinite, presence, inputs, encoders)
    with pytest.raises(ValueError, match="HolographicNetworkConfig"):
        HolographicNetwork(16, asdict(H1), REPRESENTATION, seed=1)
    with pytest.raises(ValueError, match="rango"):
        HolographicNetwork(8, replace(H1, rank=8), REPRESENTATION, seed=1)


def test_a_non_finite_value_inside_the_bridge_reaches_the_output_check():
    net = network(replace(H1, areas=19))
    with torch.no_grad():
        net.qkv.weight[0, 0] = math.nan
    with pytest.raises(ValueError, match="salida de la red holográfica"):
        net(*module_inputs())


def test_the_predictor_change_registry_admits_it():
    changes = dict(holographic_network=asdict(H3))
    assert checked_predictor_changes(changes) == changes
    base, _ = setup()
    assert replace(base.config, **changes).holographic_network == H3


def test_the_frozen_consumer_rejects_it_until_it_registers_it():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    predictor, _ = setup(holographic_network=H1)
    with pytest.raises(ValueError, match="red holográfica"):
        FrozenFinancialConsumer(predictor)


def test_the_frozen_parent_loader_refuses_it_before_building(tmp_path, monkeypatch):
    document = dict(predictor=dict(hidden_size=32, dtype="float64"))
    report = dict(request=dict(predictor_changes=dict(holographic_network=asdict(H1))))

    def forbidden(*_args, **_options):
        pytest.fail("Se construyó o leyó un padre holográfico sin registro congelado")

    monkeypatch.setattr(mars_titan_walk_forward, "FinancialPredictor", forbidden)
    monkeypatch.setattr(mars_titan_walk_forward, "read_manifest", forbidden)
    with pytest.raises(ValueError, match="red holográfica"):
        mars_titan_walk_forward._frozen_parent(
            document, specification(), 42, "cpu", tmp_path, report
        )


@pytest.mark.parametrize("in_recipe", (False, True))
def test_the_parent_declaration_is_refused_before_checkpoint_verification(
    tmp_path, monkeypatch, in_recipe
):
    view = tmp_path / "view.json"
    view.write_text("{}\n")
    predictor = dict(hidden_size=32, dtype="float64")
    request = dict(
        variant="mac_online", seed=42, view_sha256=hashlib.sha256(view.read_bytes()).hexdigest()
    )
    if in_recipe:
        predictor["holographic_network"] = asdict(H1)
    else:
        request["predictor_changes"] = dict(holographic_network=asdict(H1))
    report = dict(
        kind=mars_titan_walk_forward.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=request,
        identity=dict(recipe=dict(predictor=predictor)),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        mars_titan_walk_forward,
        "_verify",
        lambda *_a: pytest.fail("Se verificó el checkpoint de un padre holográfico sin registro"),
    )
    with pytest.raises(ValueError, match="red holográfica"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)


def test_the_source_and_its_catalog_enter_the_trainer_identity_only_when_active():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    module = "mars_titan.models.titans.holographic_network"
    trainer = object.__new__(ChronologicalTrainer)
    # La receta decide si F2 y F3 añaden su código a la identidad.
    trainer.recipe = ChronologicalRecipe()
    for config, present in ((None, False), (H1, True)):
        trainer.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=None,
                asset_groups=None,
                holographic_network=config,
            )
        )
        code = trainer._code()
        assert (module in code) is present
        assert ("data/catalogs/macro-indicators.csv" in code) is present
    assert (
        code["data/catalogs/macro-indicators.csv"]
        == hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    )


def test_declared_arms_build_with_their_controls():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads((root / "configs/experiments/holographic-network-v1.json").read_text())
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    assert declaration["levels"]["declared"] == [1]
    base, _ = setup()
    reference_parameters = dict(base.named_parameters())
    identities = set()
    for arm_name, arm in declaration["arms"].items():
        changes = arm["predictor_changes"]
        assert checked_predictor_changes(changes or None) == (changes or None)
        predictor, _ = setup(**changes)
        identities.add(json.dumps(predictor.config.identity(), sort_keys=True))
        names = dict(predictor.named_parameters())
        for key, value in reference_parameters.items():
            assert torch.equal(names[key], value), (arm_name, key)
    assert len(identities) == len(declaration["arms"])
    for contrast in declaration["contrasts"]:
        assert {contrast["variant"], contrast["control"]} <= set(declaration["arms"])
    # H3 tiene tantos nodos como H1 con 19 áreas.
    arms = declaration["arms"]
    h1 = HolographicNetworkConfig(**arms["h1_19_areas"]["predictor_changes"]["holographic_network"])
    h3 = HolographicNetworkConfig(**arms["h3_23_slots"]["predictor_changes"]["holographic_network"])
    assert h1.nodes == h3.nodes == 23
