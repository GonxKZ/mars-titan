"""Cerebro de atractores de Hopfield escasos con inhibición lateral de orden (RSHRB).

Pruebas de contrato, exactitud de entmax-1.5 (incluyendo gradcheck en FP64), equivalencia
algebraica exacta frente a la referencia cuadrática O(N²), ley de conservación de suma cero e
invariancia a traslaciones uniformes, descenso en energía, acotación frente a ruido, aislamiento
causal entre mercados e instantes, determinismo en FP32 y FP64, tamaño de grafo independiente de
las filas, gradientes vivos en el paso 0 y paso 1 con predicción inicial constante, convivencia
con holograma y CRRM, compatibilidad congelada y declaración de una sola variable en la criba v12.
"""

import importlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_episodic_readout import CONTEXT, reader
from test_financial_adapter import setup
from test_financial_learning_scale import build, emit
from test_frozen_financial import bank, consumer, frozen
from test_session_rank_objective import model, session_batch
from test_technical_dense import DENSE, opened

from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans.asset_groups import AssetGroupsConfig
from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.episodic_readout import apply_episodic_readout
from mars_titan.models.titans.fractal_shared_operator import FractalSharedOperatorConfig
from mars_titan.models.titans.hopfield_rank_brain import (
    ATTRACTIVE_ONLY,
    BETA_MAX,
    BETA_MIN,
    CROWDING_PRIOR,
    PROTOTYPE_NORM_EPS,
    REPULSIVE,
    REPULSIVE_ONLY,
    SHUFFLED_PEERS,
    STEP_BOUND,
    HopfieldRankBrainConfig,
    RepulsiveSparseHopfieldRankBrain,
    entmax15,
    parse_hopfield_rank_brain,
)
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm
from mars_titan.training import comparison_campaign
from mars_titan.training.dust.financial import FinancialDustAdapter
from mars_titan.training.search_cases import checked_predictor_changes
from tests.models.test_price_presence_families import titans_specification

HOPFIELD_MODULE = "mars_titan.models.titans.hopfield_rank_brain"
EARLY = HopfieldRankBrainConfig(prototypes=8, steps=2, mode=REPULSIVE, position="after_mac")
LATE = HopfieldRankBrainConfig(prototypes=32, steps=2, mode=REPULSIVE, position="after_technical")
ATTRACTIVE = HopfieldRankBrainConfig(
    prototypes=8, steps=2, mode=ATTRACTIVE_ONLY, position="after_mac"
)
REPULSIVE_PURE = HopfieldRankBrainConfig(
    prototypes=8, steps=2, mode=REPULSIVE_ONLY, position="after_mac"
)
SHUFFLED = HopfieldRankBrainConfig(prototypes=8, steps=2, mode=SHUFFLED_PEERS, position="after_mac")
MARKETS = ("US", "CN", "US", "US", "CN", "US")
AT = (7,) * len(MARKETS)


def opened_layer(config=EARLY, *, hidden=16, seed=3, dtype=torch.float64):
    """Módulo con la puerta abierta para observar el efecto del mecanismo."""
    layer = RepulsiveSparseHopfieldRankBrain(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed + 101)
        layer.gate.copy_(torch.randn(hidden, generator=gen, dtype=dtype) / 4)
    return layer


def sample_encoded(rows=6, hidden=16, seed=9, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=gen, dtype=dtype)


def naive_reference(layer, values, markets, instants):
    """Referencia explícita O(J · N² · P · H) par a par con la definición energética exacta."""
    cfg = layer.config
    hidden = layer.hidden_size
    prototypes = layer._normalized_prototypes()
    beta = layer._beta()
    eta_a = layer._eta_a()
    eta_r = layer._eta_r()
    use_a = cfg.mode != REPULSIVE_ONLY
    use_r = cfg.mode != ATTRACTIVE_ONLY

    z0 = torch.nn.functional.layer_norm(values, (hidden,), eps=layer.norm.eps)
    z = z0
    scale = beta / math.sqrt(hidden)
    for _ in range(cfg.steps):
        z_hat = layer.norm(z)
        alpha = entmax15((z_hat @ prototypes.transpose(0, 1)) * scale)
        delta_a = alpha @ prototypes if use_a else torch.zeros_like(z)
        if not use_r:
            delta_r = torch.zeros_like(z)
        else:
            rows_r = []
            for i in range(len(values)):
                group = [
                    j
                    for j in range(len(values))
                    if (markets[j], instants[j]) == (markets[i], instants[i])
                ]
                if len(group) == 1:
                    rows_r.append(torch.zeros_like(z_hat[i]))
                    continue
                n_p = torch.stack([alpha[j] for j in group], dim=0).sum(dim=0)
                acc = torch.zeros_like(z_hat[i])
                for j in group:
                    if j == i:
                        continue
                    w_ij = (alpha[i] * alpha[j] / (n_p + CROWDING_PRIOR)).sum()
                    acc = acc + w_ij * (z_hat[i] - z_hat[j])
                rows_r.append(acc)
            delta_r = torch.stack(rows_r, dim=0)
            if cfg.mode == SHUFFLED_PEERS:
                permuted = torch.zeros_like(delta_r)
                for key, members in layer.groups(markets, instants):
                    idx = torch.tensor(members, dtype=torch.int64, device=values.device)
                    part = delta_r.index_select(0, idx)
                    if len(members) > 1:
                        order = layer._permutation(key, len(members)).to(values.device)
                        part = part.index_select(0, order)
                    permuted = permuted.index_copy(0, idx, part)
                delta_r = permuted
        step = torch.zeros_like(z)
        if use_a:
            step = step + eta_a * delta_a
        if use_r:
            step = step + eta_r * delta_r
        z = z + step
    displacement = z - z0
    return values + layer.gate * layer.out(layer.readout_norm(displacement))


def _graph_nodes(tensor):
    visited = set()
    stack = [tensor.grad_fn] if tensor.grad_fn is not None else []
    while stack:
        node = stack.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for nxt, _ in node.next_functions:
            if nxt is not None:
                stack.append(nxt)
    return len(visited)


def test_entmax15_satisfies_simplex_exact_sparsity_edge_cases_and_gradcheck():
    gen = torch.Generator().manual_seed(17)
    logits = torch.randn(12, 32, generator=gen, dtype=torch.float64) * 1.5
    probs = entmax15(logits)
    assert (probs >= 0).all()
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones(12, dtype=torch.float64), rtol=1e-13, atol=1e-13
    )
    # Escasez exacta: la mayoría de prototipos reciben peso exactamente cero.
    assert (probs == 0).sum() > probs.numel() // 2

    # Entrada constante -> distribución uniforme exacta 1/P.
    uniform = entmax15(torch.zeros((3, 8), dtype=torch.float64))
    torch.testing.assert_close(
        uniform, torch.full((3, 8), 1.0 / 8.0, dtype=torch.float64), rtol=1e-13, atol=1e-13
    )

    # Valores extremos en FP32 no desbordan ni producen NaN en forward ni backward.
    extreme = torch.tensor([[0.0, -1e20, -1e30, 5.0]], dtype=torch.float32, requires_grad=True)
    out_ext = entmax15(extreme)
    assert torch.isfinite(out_ext).all()
    torch.testing.assert_close(out_ext, torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
    out_ext.sum().backward()
    assert torch.isfinite(extreme.grad).all()

    # Verificación exacta del jacobiano analítico en FP64.
    sample = (torch.randn(5, 8, generator=gen, dtype=torch.float64) * 1.2).requires_grad_(True)
    assert torch.autograd.gradcheck(entmax15, (sample,), eps=1e-6, atol=1e-7, rtol=1e-5)

    # Gradcheck también sobre empates exactos y entrada nula.
    tied = torch.zeros((2, 6), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(entmax15, (tied,), eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize(
    "config",
    [
        EARLY,
        ATTRACTIVE,
        REPULSIVE_PURE,
        SHUFFLED,
        HopfieldRankBrainConfig(prototypes=4, steps=1, mode=REPULSIVE, position="after_mac"),
        HopfieldRankBrainConfig(prototypes=16, steps=3, mode=REPULSIVE, position="after_mac"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linear_repulsive_step_matches_explicit_quadratic_pairwise_reference(config, dtype):
    layer = opened_layer(config, hidden=16, seed=5, dtype=dtype)
    values = sample_encoded(rows=6, hidden=16, seed=11, dtype=dtype)
    # Sesiones de tamaño 3 ((US, 7)), 2 ((CN, 7)) y 1 ((US, 8)).
    instants = (7, 7, 7, 7, 7, 8)
    actual = layer(values, MARKETS, instants)
    expected = naive_reference(layer, values, MARKETS, instants)
    tol = 1e-5 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def test_linear_repulsive_gradients_match_explicit_quadratic_reference():
    layer_a = opened_layer(EARLY, hidden=16, seed=5, dtype=torch.float64)
    layer_b = opened_layer(EARLY, hidden=16, seed=5, dtype=torch.float64)
    instants = (7, 7, 7, 7, 7, 8)
    x_a = sample_encoded(rows=6, hidden=16, seed=11, dtype=torch.float64).requires_grad_()
    x_b = sample_encoded(rows=6, hidden=16, seed=11, dtype=torch.float64).requires_grad_()
    layer_a(x_a, MARKETS, instants).square().sum().backward()
    naive_reference(layer_b, x_b, MARKETS, instants).square().sum().backward()
    torch.testing.assert_close(x_a.grad, x_b.grad, rtol=1e-12, atol=1e-12)
    for (name, pa), (_, pb) in zip(
        layer_a.named_parameters(), layer_b.named_parameters(), strict=True
    ):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-12, atol=1e-12, msg=name)


def test_repulsive_force_obeys_exact_zero_sum_and_uniform_shift_invariance():
    """Conservación de suma cero (3.ª ley de Newton) e invariancia a traslación uniforme."""
    layer = opened_layer(EARLY, hidden=16, seed=19, dtype=torch.float64)
    values = sample_encoded(rows=7, hidden=16, seed=21, dtype=torch.float64)
    z_hat = layer.norm(values)
    alpha = entmax15(
        (z_hat @ layer._normalized_prototypes().transpose(0, 1))
        * (layer._beta() / math.sqrt(layer.hidden_size))
    )
    repul = layer._repulsive_field(z_hat, alpha)
    # La fuerza repulsiva no desplaza jamás la media transversal de la sesión.
    torch.testing.assert_close(
        repul.sum(dim=0), torch.zeros(16, dtype=torch.float64), rtol=0, atol=1e-13
    )
    # Un desplazamiento uniforme de todos los activos de la sesión se cancela idénticamente.
    shift = torch.linspace(-2.0, 3.0, 16, dtype=torch.float64).unsqueeze(0)
    shifted_repul = layer._repulsive_field(z_hat + shift, alpha)
    torch.testing.assert_close(shifted_repul, repul, rtol=1e-13, atol=1e-13)


def test_energy_descent_decreases_attractive_and_repulsive_energies():
    """Un paso infinitesimal a lo largo de cada campo reduce estrictamente su término energético."""
    layer = opened_layer(EARLY, hidden=16, seed=23, dtype=torch.float64)
    values = sample_encoded(rows=8, hidden=16, seed=29, dtype=torch.float64)
    z_hat = layer.norm(values).detach()
    prototypes = layer._normalized_prototypes().detach()
    beta = layer._beta().detach()
    scale = beta / math.sqrt(layer.hidden_size)

    def attractive_energy(z):
        logits = (z @ prototypes.transpose(0, 1)) * scale
        alpha = entmax15(logits)
        tsallis_entropy = (4.0 / 3.0) * (1.0 - alpha.pow(1.5).sum(dim=-1))
        psi_star = (alpha * logits).sum(dim=-1) + tsallis_entropy
        return -(1.0 / scale) * psi_star.sum()

    alpha0 = entmax15((z_hat @ prototypes.transpose(0, 1)) * scale)
    delta_a = alpha0 @ prototypes
    eps = 1e-4
    assert attractive_energy(z_hat + eps * delta_a) < attractive_energy(z_hat)

    # Energía repulsiva (para afinidades fijas α): −(1/4) Σ_{i,j} W_{ij} ‖z_i − z_j‖².
    n_p = alpha0.sum(dim=0)
    weights = (alpha0 / (n_p + CROWDING_PRIOR).unsqueeze(0)) @ alpha0.transpose(0, 1)

    def repulsive_energy(z):
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        return -0.25 * (weights * diff.square().sum(dim=-1)).sum()

    delta_r = layer._repulsive_field(z_hat, alpha0)
    assert repulsive_energy(z_hat + eps * delta_r) < repulsive_energy(z_hat)


def test_single_asset_and_identical_assets_produce_exact_zero_repulsion_and_finite_gradients():
    # Tres sesiones de un solo activo en modo repulsivo puro -> out == values bit a bit.
    layer = opened_layer(REPULSIVE_PURE, hidden=16, seed=7, dtype=torch.float64)
    values = sample_encoded(rows=3, hidden=16, seed=13, dtype=torch.float64).requires_grad_()
    out = layer(values, ("US", "CN", "US"), (10, 10, 11))
    torch.testing.assert_close(out, values, rtol=0, atol=0)
    out.square().sum().backward()
    assert torch.isfinite(values.grad).all()
    for name, param in layer.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name

    # Activos idénticos en la misma sesión -> repulsión idénticamente cero.
    same = values.detach()[:1].expand(5, -1).clone().requires_grad_()
    layer_r = opened_layer(REPULSIVE_PURE, hidden=16, seed=7, dtype=torch.float64)
    out_same = layer_r(same, ("US",) * 5, (10,) * 5)
    torch.testing.assert_close(out_same, same, rtol=0, atol=0)
    out_same.square().sum().backward()
    assert torch.isfinite(same.grad).all()

    # Coordenadas constantes dentro de una fila -> LayerNorm produce 0 sin NaN.
    const = torch.full((4, 16), 3.5, dtype=torch.float64, requires_grad=True)
    layer_full = opened_layer(EARLY, hidden=16, seed=7, dtype=torch.float64)
    out_const = layer_full(const, ("US",) * 4, (10,) * 4)
    assert torch.isfinite(out_const).all()
    out_const.square().sum().backward()
    assert torch.isfinite(const.grad).all()


def test_step_sizes_and_inverse_temperature_remain_strictly_bounded_under_extreme_logits():
    """Mitigación del riesgo principal: η_a, η_r y β permanecen estrictamente acotados."""
    layer = opened_layer(EARLY, hidden=16, seed=11, dtype=torch.float64)
    with torch.no_grad():
        layer.beta_logit.fill_(50.0)
        layer.eta_a_logit.fill_(50.0)
        layer.eta_r_logit.fill_(50.0)
    assert BETA_MIN < float(layer._beta().detach()) <= BETA_MAX
    assert 0.0 < float(layer._eta_a().detach()) <= STEP_BOUND
    assert 0.0 < float(layer._eta_r().detach()) <= STEP_BOUND

    with torch.no_grad():
        layer.beta_logit.fill_(-50.0)
        layer.eta_a_logit.fill_(-50.0)
        layer.eta_r_logit.fill_(-50.0)
    assert BETA_MIN <= float(layer._beta().detach()) < BETA_MAX
    assert 0.0 <= float(layer._eta_a().detach()) < STEP_BOUND
    assert 0.0 <= float(layer._eta_r().detach()) < STEP_BOUND

    # Prototipos de norma extrema o nula siguen acotados a RMS ≤ 1 tras normalización.
    with torch.no_grad():
        layer.prototypes[0].fill_(1e6)
        layer.prototypes[1].zero_()
    normed = layer._normalized_prototypes().detach()
    rms = torch.sqrt(normed.square().mean(dim=-1) + PROTOTYPE_NORM_EPS)
    assert torch.isfinite(normed).all()
    assert float(rms[0]) <= 1.0 + 1e-6
    assert float(normed[1].abs().max()) == 0.0


def test_each_row_only_reads_rows_of_its_own_market_and_instant_strict_causality():
    """Sin fugas entre mercados ni hacia instantes futuros o pasados del mismo bloque."""
    layer = opened_layer(EARLY, hidden=16, seed=41, dtype=torch.float64)
    values = sample_encoded(rows=6, hidden=16, seed=43, dtype=torch.float64)
    base = layer(values, MARKETS, AT)

    # Modificar filas de CN no cambia ni un bit las filas de US.
    other_market = values.clone()
    other_market[[1, 4]] += 7.5
    changed = layer(other_market, MARKETS, AT)
    us = [0, 2, 3, 5]
    assert torch.equal(changed[us], base[us])

    # Modificar una fila del mismo mercado y mismo instante sí cambia a sus pares.
    same_market = values.clone()
    same_market[0] += 7.5
    assert not torch.equal(layer(same_market, MARKETS, AT)[2], base[2])

    # Mover la fila 5 a un instante futuro (8 > 7): perturbar el futuro no altera el presente (7).
    instants = (7, 7, 7, 7, 7, 8)
    split = layer(values, MARKETS, instants)
    future_perturbed = values.clone()
    future_perturbed[5] += 100.0
    assert torch.equal(layer(future_perturbed, MARKETS, instants)[[0, 2, 3]], split[[0, 2, 3]])


@pytest.mark.parametrize("config", [EARLY, ATTRACTIVE, REPULSIVE_PURE])
def test_permuting_rows_permutes_outputs_equivariantly(config):
    layer = opened_layer(config, hidden=16, seed=47, dtype=torch.float64)
    values = sample_encoded(rows=6, hidden=16, seed=53, dtype=torch.float64)
    order = [3, 0, 5, 1, 4, 2]
    expected = layer(values, MARKETS, AT)[order]
    actual = layer(values[order], tuple(MARKETS[i] for i in order), AT)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_controls_share_exact_parameters_initialization_and_distinct_semantics():
    main = RepulsiveSparseHopfieldRankBrain(16, EARLY, seed=11, dtype=torch.float64)
    ctrl_att = RepulsiveSparseHopfieldRankBrain(16, ATTRACTIVE, seed=11, dtype=torch.float64)
    ctrl_rep = RepulsiveSparseHopfieldRankBrain(16, REPULSIVE_PURE, seed=11, dtype=torch.float64)
    ctrl_shuf = RepulsiveSparseHopfieldRankBrain(16, SHUFFLED, seed=11, dtype=torch.float64)
    for other in (ctrl_att, ctrl_rep, ctrl_shuf):
        assert set(main.state_dict()) == set(other.state_dict())
        for name, tensor in main.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, other.state_dict()[name]), name

    values = sample_encoded(rows=6, hidden=16, seed=59, dtype=torch.float64)
    # attractive_only es estrictamente por fila: perturbar pares de US no altera la fila 0.
    open_att = opened_layer(ATTRACTIVE, hidden=16, seed=11, dtype=torch.float64)
    base_att = open_att(values, MARKETS, AT)
    others = values.clone()
    others[[2, 3, 5]] += 9.0
    assert torch.equal(open_att(others, MARKETS, AT)[0], base_att[0])

    # Todos los modos producen salidas distintas cuando la puerta está abierta.
    open_main = opened_layer(EARLY, hidden=16, seed=11, dtype=torch.float64)
    open_rep = opened_layer(REPULSIVE_PURE, hidden=16, seed=11, dtype=torch.float64)
    open_shuf = opened_layer(SHUFFLED, hidden=16, seed=11, dtype=torch.float64)
    out_main = open_main(values, MARKETS, AT)
    out_rep = open_rep(values, MARKETS, AT)
    out_shuf_1 = open_shuf(values, MARKETS, AT)
    out_shuf_2 = open_shuf(values, MARKETS, AT)
    assert torch.equal(out_shuf_1, out_shuf_2)
    assert not torch.allclose(out_main, base_att)
    assert not torch.allclose(out_main, out_rep)
    assert not torch.allclose(out_main, out_shuf_1)

    # Perturbar CN no altera US en shuffled_peers.
    cn_moved = values.clone()
    cn_moved[[1, 4]] += 5.0
    assert torch.equal(open_shuf(cn_moved, MARKETS, AT)[[0, 2, 3, 5]], out_shuf_1[[0, 2, 3, 5]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deterministic_outputs_and_gradients_in_fp32_and_fp64(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        layer_a = opened_layer(EARLY, hidden=16, seed=61, dtype=dtype)
        layer_b = opened_layer(EARLY, hidden=16, seed=61, dtype=dtype)
        x_a = sample_encoded(rows=6, hidden=16, seed=67, dtype=dtype).requires_grad_()
        x_b = sample_encoded(rows=6, hidden=16, seed=67, dtype=dtype).requires_grad_()
        out_a = layer_a(x_a, MARKETS, AT)
        out_b = layer_b(x_b, MARKETS, AT)
        assert torch.equal(out_a, out_b)
        out_a.square().sum().backward()
        out_b.square().sum().backward()
        assert torch.equal(x_a.grad, x_b.grad)
        for (na, pa), (_, pb) in zip(
            layer_a.named_parameters(), layer_b.named_parameters(), strict=True
        ):
            assert torch.isfinite(pa.grad).all() and torch.equal(pa.grad, pb.grad), na
    finally:
        torch.use_deterministic_algorithms(previous)


def test_autograd_graph_size_does_not_grow_with_number_of_rows():
    """Regla 5 de convivencia: sin bucles por fila; el grafo tiene tamaño independiente de N."""
    layer = opened_layer(EARLY, hidden=16, seed=71, dtype=torch.float32)
    small = sample_encoded(rows=8, hidden=16, seed=73, dtype=torch.float32).requires_grad_()
    large = sample_encoded(rows=128, hidden=16, seed=79, dtype=torch.float32).requires_grad_()
    out_small = layer(small, ("US",) * 8, (1,) * 8)
    out_large = layer(large, ("US",) * 128, (1,) * 128)
    assert _graph_nodes(out_small) == _graph_nodes(out_large)


def test_disabled_default_preserves_identity_fingerprint_parameters_and_predictions_bitwise():
    base, batch = setup()
    explicit = model(hopfield_rank_brain=None)
    assert explicit.config.identity() == base.config.identity()
    assert explicit._config_id() == base._config_id()
    assert explicit._parameter_id == base._parameter_id
    assert "hopfield_rank_brain" not in base.config.identity()
    assert base.hopfield_rank_brain is None and not base.session_level
    expected = base.prepare(
        batch, base.initial_state(batch.flow_ids, differentiable=True), differentiable=True
    )
    actual = explicit.prepare(
        batch, explicit.initial_state(batch.flow_ids, differentiable=True), differentiable=True
    )
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    targets = torch.linspace(
        -0.02, 0.02, len(batch.flow_ids), dtype=expected.point_predictions.dtype
    )
    loss_base = (expected.point_predictions - targets).abs().mean()
    loss_explicit = (actual.point_predictions - targets).abs().mean()
    torch.testing.assert_close(loss_explicit, loss_base, rtol=0, atol=0)
    loss_base.backward()
    loss_explicit.backward()
    for (name, pa), (_, pb) in zip(
        base.named_parameters(), explicit.named_parameters(), strict=True
    ):
        assert (pa.grad is None) == (pb.grad is None), name
        if pa.grad is not None:
            torch.testing.assert_close(pb.grad, pa.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("config", [EARLY, ATTRACTIVE, REPULSIVE_PURE, SHUFFLED])
def test_predictor_with_hopfield_rank_brain_starts_bitwise_identical_to_control(config):
    base, _ = setup()
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(hopfield_rank_brain=config)
    ref_params = dict(base.named_parameters())
    all_params = dict(predictor.named_parameters())
    assert set(all_params) - set(ref_params) == {
        "hopfield_rank_brain.norm.weight",
        "hopfield_rank_brain.norm.bias",
        "hopfield_rank_brain.prototypes",
        "hopfield_rank_brain.beta_logit",
        "hopfield_rank_brain.eta_a_logit",
        "hopfield_rank_brain.eta_r_logit",
        "hopfield_rank_brain.readout_norm.weight",
        "hopfield_rank_brain.out.weight",
        "hopfield_rank_brain.gate",
    }
    for name, value in ref_params.items():
        assert torch.equal(all_params[name], value), name
    assert torch.count_nonzero(predictor.hopfield_rank_brain.gate) == 0
    assert torch.count_nonzero(predictor.hopfield_rank_brain.out.weight) > 0
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [REPULSIVE, ATTRACTIVE_ONLY, REPULSIVE_ONLY, SHUFFLED_PEERS])
def test_step_zero_and_step_one_gradients_are_finite_and_alive_with_zero_median_head(mode):
    """Con mediana inicial en cero (predicción constante) y pinball+IC, los gradientes son
    finitos, la puerta recibe gradiente en el paso 0 y tras un paso todos los parámetros activos
    reciben gradientes no nulos y finitos."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        cfg = HopfieldRankBrainConfig(prototypes=8, steps=2, mode=mode, position="after_technical")
        predictor, batch = build(
            technical_branch=asdict(DENSE),
            hopfield_rank_brain=asdict(cfg),
        )
        opened(predictor.technical_branch)
        predictor._seal_parameters()

        # Paso 0: predicción puntual idénticamente cero (constante), colas activas.
        state0 = predictor.initial_state(batch.flow_ids, differentiable=True)
        prep0 = predictor.prepare(batch, state0, differentiable=True, session_complete=True)
        assert torch.count_nonzero(prep0.point_predictions) == 0

        targets = torch.linspace(
            -0.02, 0.03, len(batch.flow_ids), dtype=prep0.point_predictions.dtype
        )
        sessions = tuple(
            (flow.split("/", 1)[0], at)
            for flow, at in zip(batch.flow_ids, batch.prediction_at, strict=True)
        )
        ic_term = SessionICTerm(weight=3.0, minimum_rows=3)
        loss0 = pinball_loss(prep0.quantiles, targets) + ic_term.term(
            prep0.point_predictions, targets, sessions
        )
        assert torch.isfinite(loss0)
        loss0.backward()
        for name, param in predictor.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
        # La ganancia aprendida recibe gradiente en el paso 0 (sin doble cero en serie).
        assert predictor.hopfield_rank_brain.gate.grad is not None
        assert predictor.hopfield_rank_brain.gate.grad.abs().sum() > 0

        # Un paso de gradiente abre la puerta; en el paso 1 todos los parámetros activos aprenden.
        with torch.no_grad():
            for param in predictor.parameters():
                if param.grad is not None:
                    param.sub_(0.1 * param.grad)
                    param.grad.zero_()
        predictor._seal_parameters()

        state1 = predictor.initial_state(batch.flow_ids, differentiable=True)
        prep1 = predictor.prepare(batch, state1, differentiable=True, session_complete=True)
        loss1 = pinball_loss(prep1.quantiles, targets) + ic_term.term(
            prep1.point_predictions, targets, sessions
        )
        assert torch.isfinite(loss1)
        loss1.backward()
        inactive = {
            REPULSIVE: set(),
            SHUFFLED_PEERS: set(),
            ATTRACTIVE_ONLY: {"eta_r_logit"},
            REPULSIVE_ONLY: {"eta_a_logit"},
        }[mode]
        for name, param in predictor.hopfield_rank_brain.named_parameters():
            if name in inactive:
                assert param.grad is None or float(param.grad.abs().sum()) == 0.0
                continue
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, f"Gradiente nulo tras abrir la puerta: {name}"
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_gradients_are_finite_when_scalar_head_starts_identically_at_zero():
    """Incluso con toda la cabeza escalar en cero (predicción constante 0), no hay NaN."""
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(hopfield_rank_brain=EARLY)
    with torch.no_grad():
        predictor.head.weight.zero_()
        predictor.head.bias.zero_()
        predictor.hopfield_rank_brain.gate.fill_(0.25)
    predictor._seal_parameters()
    state = predictor.initial_state(batch.flow_ids, differentiable=True)
    prep = predictor.prepare(batch, state, differentiable=True, session_complete=True)
    assert torch.count_nonzero(prep.point_predictions) == 0
    targets = torch.linspace(-0.02, 0.02, len(batch.flow_ids), dtype=prep.point_predictions.dtype)
    sessions = tuple(
        (flow.split("/", 1)[0], at)
        for flow, at in zip(batch.flow_ids, batch.prediction_at, strict=True)
    )
    loss = (prep.point_predictions - targets).abs().mean() + SessionICTerm(
        weight=3.0, minimum_rows=3
    ).term(prep.point_predictions, targets, sessions)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in predictor.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_late_position_reads_post_technical_representation_and_coexists_with_hologram_and_crrm():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with pytest.raises(ValueError, match="rama técnica"):
            build(hopfield_rank_brain=asdict(LATE))
        plain, batch = build(technical_branch=asdict(DENSE))
        opened(plain.technical_branch)
        plain._seal_parameters()

        late, _ = build(technical_branch=asdict(DENSE), hopfield_rank_brain=asdict(LATE))
        opened(late.technical_branch)
        with torch.no_grad():
            late.hopfield_rank_brain.gate.fill_(0.2)
        late._seal_parameters()

        early, _ = build(technical_branch=asdict(DENSE), hopfield_rank_brain=asdict(EARLY))
        opened(early.technical_branch)
        with torch.no_grad():
            early.hopfield_rank_brain.gate.fill_(0.2)
        early._seal_parameters()

        markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
        ref = emit(plain, batch)
        after = emit(late, batch, session_complete=True)
        expected_after = late.hopfield_rank_brain(ref.working_state, markets, batch.prediction_at)
        torch.testing.assert_close(after.working_state, expected_after, rtol=0, atol=0)

        before = emit(early, batch, session_complete=True)
        assert not torch.allclose(after.working_state, before.working_state)
        # Ninguna posición altera el estado rápido de MAC.
        torch.testing.assert_close(
            after.next_state.mac.memory.weights[0],
            ref.next_state.mac.memory.weights[0],
            rtol=0,
            atol=0,
        )

        # Convivencia explícita: session_hologram -> hopfield_rank_brain -> CRRM -> head.
        hsi_cfg = SessionHologramConfig(bands=4, phase_dim=8, position="after_technical")
        crrm_cfg = CrossSectionalRankMemoryConfig(dim=8, time_constants=(5,), minimum_rows=3)
        coexist, _ = build(
            technical_branch=asdict(DENSE),
            session_hologram=asdict(hsi_cfg),
            hopfield_rank_brain=asdict(LATE),
            cross_sectional_rank_memory=asdict(crrm_cfg),
        )
        opened(coexist.technical_branch)
        with torch.no_grad():
            coexist.session_hologram.gate.fill_(0.15)
            coexist.hopfield_rank_brain.gate.fill_(0.2)
            coexist.cross_sectional_rank_memory.out.weight.fill_(0.05)
        coexist._seal_parameters()
        state_c = coexist.initial_state(batch.flow_ids)
        rank_mem = coexist.initial_rank_memory()
        prep_c = coexist.prepare(batch, state_c, session_complete=True, rank_memory=rank_mem)
        after_hsi = coexist.session_hologram(ref.working_state, markets, batch.prediction_at)
        after_rshrb = coexist.hopfield_rank_brain(after_hsi, markets, batch.prediction_at)
        keys_c = coexist.cross_sectional_rank_memory.keys(after_rshrb, markets, batch.prediction_at)
        expected_working = after_rshrb + coexist.cross_sectional_rank_memory.read(
            keys_c, markets, rank_mem
        )
        torch.testing.assert_close(prep_c.working_state, expected_working, rtol=0, atol=0)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_session_complete_required_and_mutual_exclusions_enforced():
    batch = session_batch()
    predictor = model(hopfield_rank_brain=EARLY)
    assert predictor.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)

    base, _ = setup()
    with pytest.raises(ValueError, match="no se combina"):
        replace(
            base.config,
            cross_asset_attention=CrossAssetAttentionConfig(),
            hopfield_rank_brain=EARLY,
        )
    with pytest.raises(ValueError, match="cabezas auxiliares"):
        replace(
            base.config,
            auxiliary_horizons=AuxiliaryHorizons(),
            hopfield_rank_brain=EARLY,
        )
    masked_cfg = replace(base.config, inputs=titans_specification())
    with pytest.raises(ValueError, match="no se combina"):
        replace(
            masked_cfg,
            asset_groups=AssetGroupsConfig(),
            hopfield_rank_brain=EARLY,
        )
    with pytest.raises(ValueError, match="no se combina"):
        replace(
            masked_cfg,
            fractal_shared_operator=FractalSharedOperatorConfig(),
            hopfield_rank_brain=EARLY,
        )
    # Dust rechaza hopfield_rank_brain.
    direct = model(variant="transformer_direct", hopfield_rank_brain=EARLY)
    with pytest.raises(ValueError, match="Dust"):
        FinancialDustAdapter(direct)


def test_state_dict_round_trips_and_rejects_different_mode_or_hyperparameters():
    batch = session_batch()
    source = model(hopfield_rank_brain=EARLY)
    with torch.no_grad():
        source.hopfield_rank_brain.gate.fill_(0.15)
    source._seal_parameters()
    target = model(hopfield_rank_brain=EARLY)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model(hopfield_rank_brain=ATTRACTIVE).load_state_dict(source.state_dict())
    with pytest.raises(ValueError):
        model(hopfield_rank_brain=REPULSIVE_PURE).load_state_dict(source.state_dict())
    with pytest.raises(ValueError):
        model(hopfield_rank_brain=SHUFFLED).load_state_dict(source.state_dict())
    with pytest.raises(ValueError):
        model(
            hopfield_rank_brain=HopfieldRankBrainConfig(
                prototypes=8, steps=3, mode=REPULSIVE, position="after_mac"
            )
        ).load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    "bad",
    [
        dict(prototypes=3, steps=2, mode="repulsive"),
        dict(prototypes=129, steps=2, mode="repulsive"),
        dict(prototypes=True, steps=2, mode="repulsive"),
        dict(prototypes=32, steps=0, mode="repulsive"),
        dict(prototypes=32, steps=5, mode="repulsive"),
        dict(prototypes=32, steps=2, mode="unknown"),
        dict(prototypes=32, steps=2, mode="repulsive", position="before_mac"),
        dict(prototypes=32, steps=2, mode="repulsive", extra=1),
        dict(prototypes=32, mode="repulsive"),
        "repulsive_sparse_hopfield_rank_brain_v1",
    ],
)
def test_invalid_declarations_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_hopfield_rank_brain(bad)


def test_construction_and_forward_validation_and_autocast_rejection():
    with pytest.raises(ValueError, match="HopfieldRankBrainConfig"):
        RepulsiveSparseHopfieldRankBrain(16, asdict(EARLY), seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        RepulsiveSparseHopfieldRankBrain(16, EARLY, seed=1, dtype=torch.bfloat16)

    layer = RepulsiveSparseHopfieldRankBrain(16, EARLY, seed=1, dtype=torch.float32)
    x = sample_encoded(rows=6, hidden=16, seed=1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer(x, MARKETS, AT)
    with pytest.raises(ValueError, match="un mercado y un instante"):
        layer(x[:, :8], MARKETS, AT)
    with pytest.raises(ValueError, match="un mercado y un instante"):
        layer(x, MARKETS[:-1], AT)
    with pytest.raises(ValueError, match="finit"):
        bad = x.clone()
        bad[0, 0] = math.nan
        layer(bad, MARKETS, AT)


def test_registry_and_campaign_v12_declaration_accept_single_variable_candidates():
    changes = dict(hopfield_rank_brain=asdict(LATE))
    assert checked_predictor_changes(changes) == changes

    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    for suffix, mode in (
        ("titans_mac_rshrb", REPULSIVE),
        ("titans_mac_rshrb_attractive", ATTRACTIVE_ONLY),
        ("titans_mac_rshrb_shuffled", SHUFFLED_PEERS),
        ("titans_mac_rshrb_repulsive_only", REPULSIVE_ONLY),
    ):
        doc["models"][suffix] = dict(
            doc["models"]["titans_mac"],
            tier="s3_inventions",
            predictor_changes=dict(
                hopfield_rank_brain=dict(
                    prototypes=32, steps=2, mode=mode, position="after_technical"
                )
            ),
        )
    validated = comparison_campaign.checked_declaration(doc, root=root)
    for name in (
        "titans_mac_rshrb",
        "titans_mac_rshrb_attractive",
        "titans_mac_rshrb_shuffled",
        "titans_mac_rshrb_repulsive_only",
    ):
        assert name in validated["models"]
        # Exactamente un campo de diferencia frente a titans_mac.
        candidate = validated["models"][name]
        control = validated["models"]["titans_mac"]
        diff_keys = {
            key
            for key in candidate.keys() | control.keys()
            if candidate.get(key) != control.get(key)
        }
        assert diff_keys == {"tier", "predictor_changes"}
        assert set(candidate["predictor_changes"]) == {"hopfield_rank_brain"}


def test_frozen_consumer_and_trainer_code_register_source_only_when_active():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        plain_engine, _ = frozen("mac_online")
        assert HOPFIELD_MODULE not in plain_engine.identity()["implementation"]
        assert api.declared_sources(plain_engine.predictor.config) == ()

        base, _ = setup("mac_online")
        pred = type(base)(replace(base.config, hopfield_rank_brain=EARLY), dtype=torch.float64)
        with torch.no_grad():
            pred.hopfield_rank_brain.gate.fill_(0.2)
        pred._seal_parameters()
        pred.eval().requires_grad_(False)
        engine = consumer(pred)

        assert api.declared_sources(pred.config) == (HOPFIELD_MODULE,)
        assert HOPFIELD_MODULE in engine.identity()["implementation"]
        assert engine.model_id != plain_engine.model_id
        assert (
            engine.identity()["modules"]["predictor/hopfield_rank_brain"]["settings"][
                "hopfield_rank_brain"
            ]
            == EARLY.identity()
        )
        engine.verify()

        batch = session_batch(6, markets=list(MARKETS))
        state = pred.initial_state(batch.flow_ids)
        expected = pred.prepare(batch, state, session_complete=True)
        actual = engine.prepare(batch, state, context_id=CONTEXT, session_complete=True)
        torch.testing.assert_close(
            actual.point_predictions, expected.point_predictions, rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="mezclador entre activos"):
            engine.prepare(batch, state, context_id=CONTEXT, session_complete=False)

        # Compatibilidad con lector episódico congelado.
        ext = reader(refinements=1).eval().requires_grad_(False)
        engine_r = consumer(pred, ext)
        snap = bank(batch)
        exp_r = apply_episodic_readout(
            expected,
            pred.head,
            ext,
            snapshot=snap,
            context_id=CONTEXT,
            cutoff=batch.prediction_at[0],
            differentiable=False,
        )
        act_r = engine_r.prepare(
            batch, state, snapshot=snap, context_id=CONTEXT, session_complete=True
        )
        torch.testing.assert_close(act_r.point_predictions, exp_r.point_predictions, rtol=0, atol=0)

        # Manipulación tras sellar se detecta.
        pred.hopfield_rank_brain.steps = 1
        with pytest.raises(ValueError):
            engine.verify(strong=False)

        # Trainer._code() solo incluye el módulo cuando hopfield_rank_brain está activo.
        trainer_obj = object.__new__(ChronologicalTrainer)
        trainer_obj.recipe = ChronologicalRecipe()
        for cfg, present in ((None, False), (EARLY, True)):
            trainer_obj.predictor = SimpleNamespace(
                config=SimpleNamespace(
                    temporal_encoder=None,
                    regional_variant=None,
                    cross_asset_attention=None,
                    asset_groups=None,
                    holographic_network=None,
                    axial_time_region=None,
                    mature_reliability=None,
                    fractal_shared_operator=None,
                    hopfield_rank_brain=cfg,
                )
            )
            assert (HOPFIELD_MODULE in trainer_obj._code()) is present
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
