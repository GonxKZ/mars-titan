"""Operador compartido entre los niveles de la jerarquía de una sesión, C3 de #751.

Las pruebas calculan salidas, gradientes y sectores sobre lotes ficticios con factores
plantados, sin optimizador ni pasos de ajuste.
"""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_adapter import setup, specification
from test_session_rank_objective import model, session_batch

from mars_titan.data.correlation_regions import correlation_regions
from mars_titan.models.baselines.multimodal import fuse_modalities
from mars_titan.models.titans.fractal_shared_operator import (
    ALL_SHARED,
    DENSE_BLOCK,
    KINDS,
    LEAVES,
    PER_LEVEL,
    RANDOM_PARTITION,
    SHARED_LORA,
    FractalSharedOperator,
    FractalSharedOperatorConfig,
    control_rank,
    parse_fractal_shared_operator,
)
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.session_ranks import SESSION_RANK_INPUTS, rank_session_inputs
from mars_titan.training.search_cases import checked_predictor_changes

HIDDEN = 16
# Tres sectores para que cada uno reúna varios activos con nueve filas.
SMALL = dict(sectors=3, min_returns=32)
CONFIGS = {kind: FractalSharedOperatorConfig(kind=kind, **SMALL) for kind in KINDS}
VARIANT = FractalSharedOperatorConfig()
FLOWS = tuple(f"US/A{i:02d}" for i in range(6)) + tuple(f"CN/B{i:02d}" for i in range(3))
AT = (7,) * 9


def windows(rows=9, *, seed=4, factors=3, channels=6, absent=(10,)):
    """Ventanas [filas, 64, canales] con cierres cargados en `factors` factores plantados.

    Las sesiones de `absent` faltan en todo el mercado: sus canales y su presencia valen cero.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(factors, 63))
    returns = common[np.arange(rows) % factors] + 0.3 * rng.normal(size=(rows, 63))
    close = np.concatenate([np.zeros((rows, 1)), np.cumsum(0.01 * returns, axis=1)], axis=1)
    window = np.zeros((rows, 64, channels))
    window[..., 3] = close
    window[..., :3] = close[..., None] + 0.002 * rng.normal(size=(rows, 64, 3))
    window[..., 4] = rng.normal(size=(rows, 64))
    if channels == 6:
        window[..., 5] = 1.0
        window[:, list(absent)] = 0.0
    return torch.tensor(window)


def inputs(rows=9, *, hidden=HIDDEN, seed=9):
    """Bloques con hojas ausentes, su presencia y las ventanas de precios."""
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randn(rows, LEAVES, hidden, generator=generator, dtype=torch.float64)
    presence = torch.ones(rows, LEAVES, dtype=torch.bool)
    presence[0, [1, 3]] = False
    presence[4, 4] = False
    presence[7, [1, 2, 3, 4]] = False
    return blocks * presence.unsqueeze(-1), presence, windows(rows)


def layer(config=CONFIGS[SHARED_LORA], *, hidden=HIDDEN, seed=3):
    """Módulo con las proyecciones de salida y las LoRA distintas de cero, para que se vean."""
    result = FractalSharedOperator(hidden, config, seed=seed, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        if config.kind == PER_LEVEL:
            result.left[:, -1].copy_(torch.randn(result.left[:, -1].shape, generator=generator))
        else:
            result.weight[-1].copy_(torch.randn(hidden, hidden, generator=generator) / 4)
        if config.kind in (SHARED_LORA, RANDOM_PARTITION, DENSE_BLOCK):
            result.up.copy_(torch.randn(result.up.shape, generator=generator) / 4)
    return result


def projections(module, level):
    """Normalización y proyecciones de una aplicación, escritas desde sus factores."""
    if module.config.kind == PER_LEVEL:
        return module.norms[level], [
            module.left[level, p] @ module.right[level, p] for p in range(4)
        ]
    if module.config.kind == ALL_SHARED:
        return module.norm, list(module.weight)
    weights = [module.weight[p] + module.up[level, p] @ module.down[level, p] for p in range(4)]
    return module.norm, weights


def attend(module, level, x, context):
    """O_ℓ(x; C) para una consulta, cabeza a cabeza."""
    norm, (query, key, value, out) = projections(module, level)
    heads, width = module.config.heads, module.head_dim
    q = query @ norm(x)
    keys = [key @ norm(item) for item in context]
    values = [value @ norm(item) for item in context]
    parts = []
    for h in range(heads):
        span = slice(h * width, (h + 1) * width)
        scores = torch.stack([q[span] @ k[span] for k in keys]) / math.sqrt(width)
        weights = torch.softmax(scores, dim=0)
        parts.append(sum(w * v[span] for w, v in zip(weights, values, strict=True)))
    return x + out @ torch.cat(parts)


def reference(module, blocks, presence, flows, instants, prices):
    """Las ecuaciones de C3 nodo a nodo, con bucles de Python y los sectores del módulo."""
    output = blocks.clone()
    for rows, sectors in module.sessions(flows, instants, prices):
        present = {i: [m for m in range(LEAVES) if presence[i, m]] for i in rows}
        if module.config.kind == DENSE_BLOCK:
            leaves = {(i, m): blocks[i, m] for i in rows for m in range(LEAVES)}
            keys = [(i, m) for i in rows for m in present[i]]
            for level in range(3):
                leaves = {
                    leaf: attend(module, level, value, [leaves[k] for k in sorted({*keys, leaf})])
                    for leaf, value in leaves.items()
                }
            for i in rows:
                for m in present[i]:
                    output[i, m] = leaves[(i, m)]
            continue
        sector = dict(zip(rows, sectors, strict=True))
        assets = {i: sum(blocks[i, m] for m in present[i]) / len(present[i]) for i in rows}
        members = {g: [i for i in rows if sector[i] == g] for g in sorted(set(sectors))}
        nodes = {g: sum(assets[i] for i in group) / len(group) for g, group in members.items()}
        market = sum(nodes.values()) / len(nodes)
        nodes = {g: attend(module, 0, node, [market, *nodes.values()]) for g, node in nodes.items()}
        updated = {
            i: attend(
                module, 1, assets[i], [nodes[sector[i]], *(assets[j] for j in members[sector[i]])]
            )
            for i in rows
        }
        for i in rows:
            for m in present[i]:
                context = [updated[i], *(blocks[i, k] for k in present[i])]
                output[i, m] = attend(module, 2, blocks[i, m], context)
    return output


@pytest.mark.parametrize(
    "config",
    [
        *CONFIGS.values(),
        FractalSharedOperatorConfig(heads=1, rank=2, **SMALL),
        FractalSharedOperatorConfig(kind=PER_LEVEL, heads=2, rank=3, **SMALL),
    ],
    ids=[*KINDS, "one_head", "per_level_two_heads"],
)
def test_the_operator_matches_a_reference_written_node_by_node(config):
    module = layer(config)
    blocks, presence, prices = inputs()
    actual = module(blocks, presence, FLOWS, AT, prices)
    expected = reference(module, blocks, presence, FLOWS, AT, prices)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not torch.equal(actual, blocks)


def test_without_presence_every_leaf_counts_as_present():
    module = layer()
    blocks, _, prices = inputs()
    everything = torch.ones(blocks.shape[:2], dtype=torch.bool)
    torch.testing.assert_close(
        module(blocks, None, FLOWS, AT, prices),
        module(blocks, everything, FLOWS, AT, prices),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_an_absent_leaf_keeps_its_block_and_sends_nothing(kind):
    module = layer(CONFIGS[kind])
    blocks, presence, prices = inputs()
    output = module(blocks, presence, FLOWS, AT, prices)
    changed = blocks.clone()
    # Una constante se normaliza igual en todas las filas y ocultaría que una hoja ausente
    # entrara como clave, así que el relleno es aleatorio.
    generator = torch.Generator().manual_seed(5)
    changed[~presence] = torch.randn(
        changed[~presence].shape, generator=generator, dtype=changed.dtype
    )
    moved = module(changed, presence, FLOWS, AT, prices)
    assert torch.equal(moved[~presence], changed[~presence])
    torch.testing.assert_close(moved[presence], output[presence], rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_each_session_only_reads_its_own_rows(kind):
    module = layer(CONFIGS[kind])
    blocks, presence, prices = inputs()
    instants = (7,) * 3 + (8,) * 3 + (7,) * 3
    output = module(blocks, presence, FLOWS, instants, prices)
    changed = blocks.clone()
    changed[3:6] = changed[3:6] * -2.0
    moved = module(changed, presence, FLOWS, instants, prices.clone())
    # Las filas 3 a 5 forman la sesión US del instante 8. Las demás no la leen.
    torch.testing.assert_close(moved[:3], output[:3], rtol=0, atol=0)
    torch.testing.assert_close(moved[6:], output[6:], rtol=0, atol=0)
    assert not torch.equal(moved[3:6], output[3:6])
    assert [rows for rows, _ in module.sessions(FLOWS, instants, prices)] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_permuting_the_rows_permutes_the_outputs(kind):
    module = layer(CONFIGS[kind])
    blocks, presence, prices = inputs()
    order = [8, 2, 5, 0, 7, 1, 4, 6, 3]
    expected = module(blocks, presence, FLOWS, AT, prices)[order]
    actual = module(
        blocks[order], presence[order], tuple(FLOWS[i] for i in order), AT, prices[order]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", [SHARED_LORA, PER_LEVEL, ALL_SHARED])
def test_the_sectors_are_the_correlation_regions_of_the_daily_returns(kind):
    module = layer(CONFIGS[kind])
    prices = windows()
    sessions = module.sessions(FLOWS, AT, prices)
    for rows, sectors in sessions:
        close, observed = prices[rows, :, 3].numpy(), prices[rows, :, 5].numpy() > 0
        regions = correlation_regions(
            [FLOWS[i] for i in rows],
            np.diff(close, axis=1),
            observed[:, 1:] & observed[:, :-1],
            clusters=3,
            min_observations=32,
        )
        number = {name: i for i, name in enumerate(sorted(set(regions.values()), key=int))}
        assert sectors == [number[regions[FLOWS[i]]] for i in rows]
    # Las filas cargan en el factor i mod 3: los seis activos de US forman tres pares.
    us = sessions[0][1]
    assert us[0] == us[3] and us[1] == us[4] and us[2] == us[5] and len(set(us)) == 3


def test_only_returns_between_two_present_sessions_enter_the_sectors(monkeypatch):
    from mars_titan.models.titans import fractal_shared_operator as fractal

    seen = []

    def capture(assets, series, mask, **options):
        seen.append((assets, series, mask, options))
        return correlation_regions(assets, series, mask, **options)

    monkeypatch.setattr(fractal, "correlation_regions", capture)
    module = layer()
    prices = windows(absent=(10, 30))
    module.sessions(FLOWS, AT, prices)
    assets, series, mask, options = seen[0]
    assert assets == list(FLOWS[:6]) and options == dict(clusters=3, min_observations=32)
    # Cada sesión ausente invalida el rendimiento que llega a ella y el que sale de ella.
    assert mask.shape == (6, 63) and not mask[:, [9, 10, 29, 30]].any()
    assert mask.sum() == 6 * 59
    returns = np.diff(prices[:6, :, 3].numpy(), axis=1)
    np.testing.assert_array_equal(series[mask], returns[mask])
    # Una ventana de cinco canales no tiene bit de presencia y todas sus sesiones cuentan.
    seen.clear()
    module.sessions(FLOWS, AT, windows(channels=5))
    assert seen[0][2].all()


def test_series_without_min_returns_valid_returns_form_sector_zero():
    module = layer()
    # Con las 33 últimas sesiones ausentes quedan 30 rendimientos válidos, menos de 32.
    short = windows(absent=range(31, 64))
    assert all(set(sectors) == {0} for _, sectors in module.sessions(FLOWS, AT, short))
    relaxed = layer(replace(CONFIGS[SHARED_LORA], min_returns=30))
    assert any(set(sectors) != {0} for _, sectors in relaxed.sessions(FLOWS, AT, short))


def test_the_random_partition_keeps_the_sizes_and_depends_on_the_seed_and_the_session():
    flows = tuple(f"US/C{i:02d}" for i in range(24))
    prices = windows(24, seed=8)

    def sectors(kind=RANDOM_PARTITION, *, seed=3, at=7):
        module = FractalSharedOperator(HIDDEN, CONFIGS[kind], seed=seed, dtype=torch.float64)
        return module.sessions(flows, (at,) * 24, prices)[0][1]

    statistical, shuffled = sectors(SHARED_LORA), sectors()
    assert sorted(statistical) == sorted(shuffled) and statistical != shuffled
    assert shuffled == sectors()
    assert shuffled != sectors(seed=4)
    assert shuffled != sectors(at=8)
    # El reparto estadístico no depende de la semilla ni del instante.
    assert statistical == sectors(SHARED_LORA, seed=4, at=8)


@pytest.mark.parametrize("kind", KINDS)
def test_zero_output_projections_return_the_blocks_exactly(kind):
    module = FractalSharedOperator(HIDDEN, CONFIGS[kind], seed=3, dtype=torch.float64)
    blocks, presence, prices = inputs()
    assert torch.equal(module(blocks, presence, FLOWS, AT, prices), blocks)


def test_the_kinds_with_a_shared_operator_start_with_the_same_weights():
    built = {kind: FractalSharedOperator(64, replace(VARIANT, kind=kind), seed=5) for kind in KINDS}
    counts = {kind: sum(p.numel() for p in module.parameters()) for kind, module in built.items()}
    assert counts == {
        SHARED_LORA: 22_656,
        PER_LEVEL: 23_424,
        ALL_SHARED: 16_512,
        RANDOM_PARTITION: 22_656,
        DENSE_BLOCK: 22_656,
    }
    variant = built[SHARED_LORA]
    for kind in (ALL_SHARED, RANDOM_PARTITION, DENSE_BLOCK):
        assert torch.equal(built[kind].weight, variant.weight)
    for kind in (RANDOM_PARTITION, DENSE_BLOCK):
        assert torch.equal(built[kind].down, variant.down)
    assert torch.count_nonzero(variant.weight[-1]) == 0 and torch.count_nonzero(variant.up) == 0
    assert torch.count_nonzero(built[PER_LEVEL].left[:, -1]) == 0
    # La semilla del predictor cambia los pesos y el generador global no se consume.
    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    other = FractalSharedOperator(64, VARIANT, seed=6)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert not torch.equal(other.weight, variant.weight)


def test_the_lora_and_control_factors_have_the_scale_of_a_linear_layer():
    hidden = 64
    variant = FractalSharedOperator(hidden, VARIANT, seed=5)
    control = FractalSharedOperator(hidden, replace(VARIANT, kind=PER_LEVEL), seed=5)
    bound = 1 / math.sqrt(hidden)
    assert variant.weight[:3].abs().max() <= bound and variant.down.abs().max() <= bound
    assert variant.weight[:3].abs().max() > 0.9 * bound
    rank = control.control_rank
    assert control.right.abs().max() <= bound
    assert control.left[:, :3].abs().max() <= math.sqrt(3 / rank)
    # U V tiene por entrada la varianza 1/(3H) de la proyección compartida.
    products = control.left[:, :3] @ control.right[:, :3]
    ratio = products.var().item() * 3 * hidden
    assert 0.85 < ratio < 1.15


@pytest.mark.parametrize(("hidden", "rank"), [(16, 4), (32, 4), (64, 4), (64, 1), (128, 8)])
def test_the_control_rank_is_the_smallest_that_keeps_the_parameters(hidden, rank):
    def count(kind):
        module = FractalSharedOperator(hidden, replace(VARIANT, kind=kind, rank=rank), seed=1)
        return sum(p.numel() for p in module.parameters())

    variant, control = count(SHARED_LORA), count(PER_LEVEL)
    chosen = control_rank(hidden, rank)
    assert control >= variant
    assert control - 24 * chosen * hidden < variant
    assert control == 3 * (2 * hidden + 8 * chosen * hidden)


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_reach_every_parameter_and_the_present_blocks(kind):
    module = layer(CONFIGS[kind])
    blocks, presence, prices = inputs()
    blocks.requires_grad_(True)
    module(blocks, presence, FLOWS, AT, prices).square().sum().backward()
    for name, value in module.named_parameters():
        assert value.grad is not None and torch.count_nonzero(value.grad) > 0, name
    assert torch.count_nonzero(blocks.grad[presence].abs().sum(-1)) == presence.sum()


@pytest.mark.parametrize("kind", KINDS)
def test_the_gradient_keeps_the_zero_output_projections_learning(kind):
    module = FractalSharedOperator(HIDDEN, CONFIGS[kind], seed=3, dtype=torch.float64)
    blocks, presence, prices = inputs()
    module(blocks, presence, FLOWS, AT, prices).square().sum().backward()
    if kind == PER_LEVEL:
        assert torch.count_nonzero(module.left.grad[:, -1]) > 0
    else:
        assert torch.count_nonzero(module.weight.grad[-1]) > 0
    if kind in (SHARED_LORA, RANDOM_PARTITION, DENSE_BLOCK):
        assert torch.count_nonzero(module.up.grad[:, -1]) > 0


@pytest.mark.parametrize(
    "change",
    [
        lambda b, p, f, t, x: (b.float(), p, f, t, x),
        lambda b, p, f, t, x: (b[:, :4], p, f, t, x),
        lambda b, p, f, t, x: (b[:0], p[:0], (), (), x[:0]),
        lambda b, p, f, t, x: (b, p.int(), f, t, x),
        lambda b, p, f, t, x: (b, p[:, :4], f, t, x),
        lambda b, p, f, t, x: (b, p, list(f), t, x),
        lambda b, p, f, t, x: (b, p, f[:-1], t, x),
        lambda b, p, f, t, x: (b, p, (*f[:-1], f[0]), t, x),
        lambda b, p, f, t, x: (b, p, (*f[:-1], "B08"), t, x),
        lambda b, p, f, t, x: (b, p, f, t[:-1], x),
        lambda b, p, f, t, x: (b, p, f, t, x[:, :32]),
        lambda b, p, f, t, x: (b, p, f, t, x[..., :4]),
        lambda b, p, f, t, x: (b, p, f, t, x[:-1]),
        lambda b, p, f, t, x: (b, p, f, t, x.numpy()),
    ],
)
def test_blocks_presence_flows_and_prices_need_their_contract(change):
    module = layer()
    blocks, presence, prices = inputs()
    with pytest.raises(ValueError, match="C3"):
        module(*change(blocks, presence, FLOWS, AT, prices))


def test_autocast_is_refused():
    module = layer()
    blocks, presence, prices = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        module(blocks, presence, FLOWS, AT, prices)


@pytest.mark.parametrize(
    "value",
    [
        dict(kind="hierarchy", rank=4, heads=4, sectors=8, min_returns=32),
        dict(kind=SHARED_LORA, rank=0, heads=4, sectors=8, min_returns=32),
        dict(kind=SHARED_LORA, rank=4, heads=3, sectors=8, min_returns=32),
        dict(kind=SHARED_LORA, rank=4, heads=4, sectors=1, min_returns=32),
        dict(kind=SHARED_LORA, rank=4, heads=4, sectors=8, min_returns=1),
        dict(kind=SHARED_LORA, rank=True, heads=4, sectors=8, min_returns=32),
        dict(kind=SHARED_LORA, rank=4, heads=4, sectors=8),
        "shared_lora",
    ],
)
def test_invalid_declarations_are_rejected(value):
    with pytest.raises(ValueError):
        parse_fractal_shared_operator(value)


def test_declarations_parse_from_their_values():
    assert parse_fractal_shared_operator(None) is None
    assert parse_fractal_shared_operator(VARIANT) is VARIANT
    assert parse_fractal_shared_operator(asdict(CONFIGS[PER_LEVEL])) == CONFIGS[PER_LEVEL]


def test_construction_refuses_other_configurations_and_precisions():
    with pytest.raises(ValueError, match="FractalSharedOperatorConfig"):
        FractalSharedOperator(HIDDEN, asdict(VARIANT), seed=1)
    with pytest.raises(ValueError, match="múltiplo"):
        FractalSharedOperator(18, VARIANT, seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        FractalSharedOperator(HIDDEN, VARIANT, seed=1, dtype=torch.float16)


def test_loading_weights_checks_the_declared_configuration():
    source = layer()
    target = FractalSharedOperator(HIDDEN, CONFIGS[SHARED_LORA], seed=3, dtype=torch.float64)
    target.load_state_dict(source.state_dict())
    blocks, presence, prices = inputs()
    assert torch.equal(
        target(blocks, presence, FLOWS, AT, prices), source(blocks, presence, FLOWS, AT, prices)
    )
    other = FractalSharedOperator(HIDDEN, CONFIGS[RANDOM_PARTITION], seed=3, dtype=torch.float64)
    with pytest.raises(ValueError):
        other.load_state_dict(source.state_dict())


def test_random_partition_weights_only_load_with_their_partition_seed():
    source = layer(CONFIGS[RANDOM_PARTITION])
    same = FractalSharedOperator(HIDDEN, CONFIGS[RANDOM_PARTITION], seed=3, dtype=torch.float64)
    same.load_state_dict(source.state_dict())
    other = FractalSharedOperator(HIDDEN, CONFIGS[RANDOM_PARTITION], seed=4, dtype=torch.float64)
    with pytest.raises(ValueError):
        other.load_state_dict(source.state_dict())
    # En la variante la semilla solo fija los pesos iniciales y la carga los sustituye.
    variant = FractalSharedOperator(HIDDEN, CONFIGS[SHARED_LORA], seed=4, dtype=torch.float64)
    variant.load_state_dict(layer().state_dict())


def test_the_identity_follows_the_kind():
    identities = {kind: CONFIGS[kind].identity() for kind in KINDS}
    assert len({json.dumps(value, sort_keys=True) for value in identities.values()}) == len(KINDS)
    assert identities[DENSE_BLOCK]["partition"] == "none"
    assert identities[RANDOM_PARTITION]["partition"].endswith("at_random_per_session")
    assert identities[PER_LEVEL]["projections"].startswith("independent")
    assert identities[ALL_SHARED]["projections"] == "shared_without_lora"
    assert all(value["state"] == "none" for value in identities.values())


def moved(config=VARIANT, value=0.05):
    """Predictor con C3 y sus proyecciones de salida distintas de cero."""
    predictor = model(fractal_shared_operator=config)
    module = predictor.fractal_shared_operator
    with torch.no_grad():
        if config.kind == PER_LEVEL:
            module.left[:, -1].fill_(value)
        else:
            module.weight[-1].fill_(value)
    predictor._seal_parameters()
    return predictor


def test_disabled_field_keeps_identity_parameters_and_predictions():
    base, batch = setup()
    explicit = model(fractal_shared_operator=None)
    assert explicit.config.identity() == base.config.identity()
    assert "fractal_shared_operator" not in base.config.identity()
    assert explicit.fractal_shared_operator is None and not explicit.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_starts_as_the_reference(kind):
    base, _ = setup()
    batch = session_batch(6)
    predictor = model(fractal_shared_operator=replace(VARIANT, kind=kind))
    reference_parameters = dict(base.named_parameters())
    names = dict(predictor.named_parameters())
    assert {name.split(".")[0] for name in set(names) - set(reference_parameters)} == {
        "fractal_shared_operator"
    }
    for name, value in reference_parameters.items():
        assert torch.equal(names[name], value), name
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    identity = predictor.config.identity()["fractal_shared_operator"]
    assert identity["operator"] == kind


def test_c3_needs_the_complete_session_in_one_block():
    batch = session_batch()
    predictor = model(fractal_shared_operator=VARIANT)
    assert predictor.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)


def test_the_predictor_passes_the_fused_blocks_through_the_hierarchy():
    batch = session_batch(6)
    predictor = moved()
    prepared = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    price, _ = predictor.price_encoder.encode_last(batch.inputs["prices"])
    module = predictor.fractal_shared_operator
    expected = fuse_modalities(
        price,
        batch.inputs,
        predictor.encoders,
        predictor.fusion,
        batch.presence,
        modulation=lambda blocks, presence: module(
            blocks, presence, batch.flow_ids, batch.prediction_at, batch.inputs["prices"]
        ),
    )
    torch.testing.assert_close(prepared.detached_tokens, expected, rtol=0, atol=0)
    base, _ = setup()
    plain = base.prepare(batch, base.initial_state(batch.flow_ids))
    assert not torch.equal(prepared.detached_tokens, plain.detached_tokens)


def test_the_sectors_use_the_received_window_before_the_session_ranks(monkeypatch):
    batch = session_batch(6)
    ranked = model(fractal_shared_operator=VARIANT, session_rank_inputs=SESSION_RANK_INPUTS)
    markets = ["US"] * 6
    assert not torch.equal(
        rank_session_inputs(batch.inputs, markets)["prices"], batch.inputs["prices"]
    )
    seen = []
    original = type(ranked.fractal_shared_operator).sessions

    def record(module, flows, instants, prices):
        seen.append(prices)
        return original(module, flows, instants, prices)

    monkeypatch.setattr(type(ranked.fractal_shared_operator), "sessions", record)
    ranked.prepare(batch, ranked.initial_state(batch.flow_ids), session_complete=True)
    assert len(seen) == 1 and torch.equal(seen[0], batch.inputs["prices"])


def test_the_state_dict_round_trips_and_the_controls_are_other_identities():
    batch = session_batch()
    source = moved()
    target = model(fractal_shared_operator=VARIANT)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    for kind in (RANDOM_PARTITION, DENSE_BLOCK):
        with pytest.raises(ValueError):
            model(fractal_shared_operator=replace(VARIANT, kind=kind)).load_state_dict(
                source.state_dict()
            )


def test_a_module_that_does_not_match_the_declaration_is_refused():
    predictor = model(fractal_shared_operator=VARIANT)
    predictor.fractal_shared_operator = FractalSharedOperator(
        32, VARIANT, seed=7, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="C3 no corresponde"):
        predictor._seal_parameters()


def test_row_by_row_tokens_refuse_c3():
    predictor = moved()
    batch = session_batch()
    with pytest.raises(ValueError, match="fila a fila"):
        predictor.fused_tokens(batch.inputs, batch.presence)


def test_the_predictor_change_registry_admits_it():
    changes = dict(fractal_shared_operator=asdict(CONFIGS[PER_LEVEL]))
    assert checked_predictor_changes(changes) == changes
    base, _ = setup()
    assert replace(base.config, **changes).fractal_shared_operator == CONFIGS[PER_LEVEL]


@pytest.mark.parametrize(
    "changes",
    [
        dict(variant="mac_frozen"),
        dict(presence_modulation="film_v1"),
        dict(axial_time_region=dict(kind="axial_time_region_v1", window=8, heads=4)),
        dict(cross_asset_attention=dict(heads=4, mode="same_session")),
        dict(
            regional_variant=dict(
                operator=dict(
                    topology="ring",
                    routing="attention",
                    cycles=1,
                    local_steps=1,
                    summary=True,
                    local_gain=0.25,
                    summary_gain=0.25,
                ),
                mixing="none",
            )
        ),
        dict(regional_fusion=RegionalFusionConfig(topology="ring")),
        dict(auxiliary_horizons=dict(horizons=[5], weight=0.5, maturity_slack=0)),
    ],
    ids=["mac_frozen", "pt6", "c2", "c5", "c1", "regional_fusion", "f5"],
)
def test_it_is_exclusive_with_other_modulations_session_mixing_and_row_wise_tokens(changes):
    base, _ = setup()
    with pytest.raises(ValueError):
        replace(base.config, fractal_shared_operator=VARIANT, **changes)


def test_it_does_not_combine_with_the_holographic_network():
    # H1 necesita la edición con el catálogo de sus familias, que su prueba ya construye.
    from test_holographic_network import H1, api, specification

    options = dict(variant="mac_online", hidden_size=32, holographic_network=H1)
    api().FinancialConfig(specification(), **options)
    with pytest.raises(ValueError, match="C3 se declara .* ni la red holográfica"):
        api().FinancialConfig(specification(), fractal_shared_operator=VARIANT, **options)


def test_the_source_and_its_regions_enter_the_trainer_identity_only_when_active():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    names = {
        "mars_titan.models.titans.fractal_shared_operator",
        "mars_titan.data.correlation_regions",
    }
    trainer = object.__new__(ChronologicalTrainer)
    trainer.recipe = ChronologicalRecipe()
    for config, present in ((None, False), (VARIANT, True)):
        trainer.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=None,
                holographic_network=None,
                axial_time_region=None,
                fractal_shared_operator=config,
            )
        )
        code = set(trainer._code())
        assert (names <= code) is present and bool(names & code) is present


def test_the_frozen_consumer_rejects_c3_until_it_registers_it():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    with pytest.raises(ValueError, match="operador fractal de C3"):
        FrozenFinancialConsumer(model(fractal_shared_operator=VARIANT))


def test_the_frozen_parent_loader_refuses_it_before_building(tmp_path, monkeypatch):
    from mars_titan.training import mars_titan_walk_forward as walk

    document = dict(predictor=dict(hidden_size=32, dtype="float64"))
    report = dict(request=dict(predictor_changes=dict(fractal_shared_operator=asdict(VARIANT))))

    def forbidden(*_args, **_options):
        pytest.fail("Se construyó o leyó un padre con C3 sin registro congelado")

    monkeypatch.setattr(walk, "FinancialPredictor", forbidden)
    monkeypatch.setattr(walk, "read_manifest", forbidden)
    with pytest.raises(ValueError, match="operador fractal de C3"):
        walk._frozen_parent(document, specification(), 42, "cpu", tmp_path, report)


@pytest.mark.parametrize("in_recipe", (False, True))
def test_the_parent_declaration_is_refused_before_checkpoint_verification(
    tmp_path, monkeypatch, in_recipe
):
    import hashlib

    from mars_titan.training import mars_titan_walk_forward as walk

    view = tmp_path / "view.json"
    view.write_text("{}\n")
    predictor = dict(hidden_size=32, dtype="float64")
    request = dict(
        variant="mac_online", seed=42, view_sha256=hashlib.sha256(view.read_bytes()).hexdigest()
    )
    if in_recipe:
        predictor["fractal_shared_operator"] = asdict(VARIANT)
    else:
        request["predictor_changes"] = dict(fractal_shared_operator=asdict(VARIANT))
    report = dict(
        kind=walk.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=request,
        identity=dict(recipe=dict(predictor=predictor)),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        walk, "_verify", lambda *_a: pytest.fail("Se verificó el checkpoint de un padre con C3")
    )
    with pytest.raises(ValueError, match="operador fractal de C3"):
        walk._parent(tmp_path, view, 42)


def test_declared_configurations_build_with_their_controls():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads(
        (root / "configs/experiments/fractal-shared-operator-v1.json").read_text(encoding="utf-8")
    )
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    base, _ = setup()
    reference_parameters = dict(base.named_parameters())
    identities, counts = set(), {}
    for name, arm in declaration["arms"].items():
        changes = arm["predictor_changes"]
        assert checked_predictor_changes(changes or None) == (changes or None)
        predictor = model(**changes)
        identities.add(json.dumps(predictor.config.identity(), sort_keys=True))
        counts[name] = sum(p.numel() for p in predictor.parameters())
        for key, value in reference_parameters.items():
            assert torch.equal(dict(predictor.named_parameters())[key], value), (name, key)
    assert len(identities) == len(declaration["arms"])
    plain = counts.pop("reference")
    assert counts["fractal_shared_operator"] == counts["random_partition"] == counts["dense_block"]
    assert counts["per_level"] >= counts["fractal_shared_operator"] > counts["all_shared"] > plain
    for contrast in declaration["contrasts"]:
        assert {contrast["variant"], contrast["control"]} <= set(declaration["arms"])


def test_the_market_node_weighs_each_sector_equally():
    flows = tuple(f"US/D{i:02d}" for i in range(8))
    instants = (7,) * 8
    module = layer()
    blocks, presence, _ = inputs(8)
    prices = windows(8)
    ((rows, sectors),) = module.sessions(flows, instants, prices)
    assert rows == list(range(8)) and sorted(np.bincount(sectors).tolist()) == [2, 3, 3]
    actual = module(blocks, presence, flows, instants, prices)
    expected = reference(module, blocks, presence, flows, instants, prices)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    # Con sectores de tamaños distintos, la media de los sectores no es la de los activos, así
    # que un mercado que promediara los activos daría otra salida.
    assets = torch.stack([blocks[i][presence[i]].mean(0) for i in range(8)])
    by_sector = torch.stack([assets[np.asarray(sectors) == g].mean(0) for g in range(3)])
    assert not torch.allclose(by_sector.mean(0), assets.mean(0))
