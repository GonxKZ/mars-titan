"""Memoria transversal de rango y residuo maduro (CRRM): contratos, ecuaciones y numérica."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_adapter import DIMENSIONS, setup, specification
from test_financial_learning_scale import statistics
from test_session_rank_objective import model, session_batch

from mars_titan.models.learning_scale import LearningScale
from mars_titan.models.quantile_head import MEDIAN_GAIN, QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.cross_sectional_rank_memory import (
    CROSS_SECTIONAL_RANK_MEMORY,
    KEY_NORM_EPS,
    RIDGE_FLOOR,
    CrossSectionalRankMemory,
    CrossSectionalRankMemoryConfig,
)
from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, declared_sources
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm, standardized
from mars_titan.models.titans.session_ranks import gaussian_ranks
from mars_titan.models.titans.technical_branch import TechnicalDenseConfig
from mars_titan.training.search_cases import checked_predictor_changes

START, DAY = 1_609_459_200_000_000, 86_400_000_000
VARIANT = CrossSectionalRankMemoryConfig(
    dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
)
SHUFFLED = replace(VARIANT, mode="shuffled_targets")
EVAL_FROZEN = replace(VARIANT, mode="eval_frozen")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def active_layer(config=VARIANT, *, hidden=16, seed=7, dtype=torch.float64):
    """Módulo con salida distinta de cero para observar la lectura asociativa."""
    layer = CrossSectionalRankMemory(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 13)
        layer.out.weight.copy_(
            torch.randn(layer.out.weight.shape, generator=generator, dtype=dtype) / 3
        )
    return layer


def encoded(rows=6, hidden=16, seed=11, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=dtype)


def reference_step(layer, sessions, query_encoded, query_markets, query_at):
    """Referencia explícita de las ecuaciones de CRRM sobre varias sesiones maduras."""
    config = layer.config
    dim = config.dim
    decays = [math.exp(-1.0 / tau) for tau in config.time_constants]
    ridge = (RIDGE_FLOOR + torch.nn.functional.softplus(layer.ridge_logits)) / dim
    dtype, device = query_encoded.dtype, query_encoded.device
    stats = {}

    def compute_keys(values, markets, instants):
        raw = layer.key_proj(layer.norm(values))
        groups = {}
        for row, key in enumerate(zip(markets, instants, strict=True)):
            groups.setdefault(key, []).append(row)
        out = torch.empty_like(raw)
        for rows in groups.values():
            idx = torch.tensor(rows, dtype=torch.int64, device=device)
            part = raw.index_select(0, idx)
            if len(rows) > 1:
                part = part - part.mean(dim=0, keepdim=True)
            denom = torch.sqrt(part.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
            out[idx] = part / denom
        return out

    for enc, flows, at, targets, issued in sessions:
        markets = tuple(flow.split("/", 1)[0] for flow in flows)
        keys = compute_keys(enc, markets, (at,) * len(flows))
        by_market = {}
        for i, flow in enumerate(flows):
            by_market.setdefault(flow.split("/", 1)[0], []).append((flow, i))
        for market in sorted(by_market):
            ordered = sorted(by_market[market], key=lambda item: item[0])
            if len(ordered) < config.minimum_rows:
                continue
            idx = [i for _, i in ordered]
            k_mat = keys[idx]
            y_vec = torch.tensor([targets[i] for i in idx], dtype=dtype, device=device)
            p_vec = torch.tensor([issued[i] for i in idx], dtype=dtype, device=device)
            g_vec = gaussian_ranks(y_vec.unsqueeze(-1), torch.ones((len(idx), 1), dtype=torch.bool))
            g_vec = g_vec.squeeze(-1)
            r_vec = g_vec - standardized(p_vec)
            v_mat = torch.stack((g_vec, r_vec), dim=-1)
            cov = (k_mat.T @ k_mat) / len(idx)
            cov = 0.5 * (cov + cov.T)
            rhs = (k_mat.T @ v_mat) / len(idx)
            if market not in stats:
                stats[market] = [
                    [
                        torch.zeros((dim, dim), dtype=dtype, device=device),
                        torch.zeros((dim, 2), dtype=dtype, device=device),
                    ]
                    for _ in decays
                ]
            for r, lam in enumerate(decays):
                stats[market][r][0] = lam * stats[market][r][0] + (1.0 - lam) * cov
                stats[market][r][1] = lam * stats[market][r][1] + (1.0 - lam) * rhs

    q_keys = compute_keys(query_encoded, query_markets, query_at)
    eye = torch.eye(dim, dtype=dtype, device=device)
    features = torch.zeros((len(query_encoded), 2 * len(decays)), dtype=dtype, device=device)
    for row, market in enumerate(query_markets):
        if market not in stats:
            continue
        parts = []
        k_row = q_keys[row]
        k_norm_sq = k_row.square().sum()
        for r in range(len(decays)):
            p_mat, q_mat = stats[market][r]
            a_mat = p_mat + ridge[r] * eye
            chol = torch.linalg.cholesky(a_mat)
            m_mat = torch.cholesky_solve(q_mat, chol)
            z_vec = torch.linalg.solve_triangular(chol, k_row.unsqueeze(-1), upper=False).squeeze(
                -1
            )
            conf = (k_norm_sq - ridge[r] * z_vec.square().sum()).clamp(0.0, 1.0)
            parts.append(conf * torch.tanh(k_row @ m_mat))
        features[row] = torch.cat(parts)
    return layer.out(features)


def test_read_and_settle_match_the_closed_form_ridge_and_leverage_equations():
    layer = active_layer(VARIANT)
    state = layer.initial_state("p")
    flows_1 = ("US/AAA", "US/BBB", "US/CCC", "US/DDD", "CN/600000.SS", "CN/600001.SS")
    enc_1 = encoded(len(flows_1), seed=21)
    issued_1 = [0.2, -0.1, 0.5, -0.4, 0.1, -0.2]
    targets_1 = [0.03, -0.02, 0.01, -0.05, 0.02, -0.01]
    markets_1 = tuple(f.split("/", 1)[0] for f in flows_1)
    keys_1 = layer.keys(enc_1, markets_1, (START,) * len(flows_1))
    # Antes de madurar ninguna etiqueta, la lectura es exactamente cero.
    assert torch.count_nonzero(layer.read(keys_1, markets_1, state)) == 0
    state = layer.stage(
        state,
        flows_1,
        (START,) * len(flows_1),
        keys_1,
        torch.tensor(issued_1, dtype=torch.float64),
        differentiable=True,
    )
    matured_1 = [(flow, START, val) for flow, val in zip(flows_1, targets_1, strict=True)]
    state = layer.settle(state, matured_1, differentiable=True, train=True)
    # CN solo tuvo 2 filas (< minimum_rows=3): no actualiza CN, pero vacía su cola pendiente.
    assert "US" in state.markets and "CN" not in state.markets
    assert len(state.pending) == 0

    flows_2 = ("US/DDD", "US/AAA", "US/CCC", "US/BBB")
    enc_2 = encoded(len(flows_2), seed=31)
    issued_2 = [-0.3, 0.4, 0.0, 0.1]
    targets_2 = [-0.01, 0.04, 0.02, -0.03]
    markets_2 = ("US",) * len(flows_2)
    keys_2 = layer.keys(enc_2, markets_2, (START + DAY,) * len(flows_2))
    state = layer.stage(
        state,
        flows_2,
        (START + DAY,) * len(flows_2),
        keys_2,
        torch.tensor(issued_2, dtype=torch.float64),
        differentiable=True,
    )
    matured_2 = [(flow, START + DAY, val) for flow, val in zip(flows_2, targets_2, strict=True)]
    state = layer.settle(state, matured_2, differentiable=True, train=True)

    q_enc = encoded(5, seed=41)
    q_markets = ("US", "CN", "US", "US", "CN")
    q_at = (START + 2 * DAY,) * 5
    q_keys = layer.keys(q_enc, q_markets, q_at)
    actual = layer.read(q_keys, q_markets, state)
    expected = reference_step(
        layer,
        [
            (enc_1, flows_1, START, targets_1, issued_1),
            (enc_2, flows_2, START + DAY, targets_2, issued_2),
        ],
        q_enc,
        q_markets,
        q_at,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    # CN no recibió actualizaciones válidas, así que sus filas leen cero exacto.
    assert torch.count_nonzero(actual[[1, 4]]) == 0
    assert torch.count_nonzero(actual[[0, 2, 3]]) > 0


def test_leverage_confidence_vanishes_on_orthogonal_unseen_directions():
    config = CrossSectionalRankMemoryConfig(
        dim=4, time_constants=(5,), ridge=0.1, minimum_rows=3, mode="matured_rank", seed=0
    )
    layer = active_layer(config, hidden=8)
    state = layer.initial_state("p")
    # Todas las claves maduras viven estrictamente en las coordenadas (0, 1); (2, 3) son nulas.
    k_train = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-0.8, 0.6, 0.0, 0.0], [0.6, -0.8, 0.0, 0.0]],
        dtype=torch.float64,
    )
    flows = ("US/A", "US/B", "US/C", "US/D")
    state = layer.stage(
        state,
        flows,
        (START,) * 4,
        k_train,
        torch.zeros(4, dtype=torch.float64),
        differentiable=False,
    )
    state = layer.settle(
        state,
        [
            ("US/A", START, 0.04),
            ("US/B", START, 0.01),
            ("US/C", START, -0.02),
            ("US/D", START, 0.0),
        ],
        differentiable=False,
        train=True,
    )
    # Consulta ortogonal pura en coordenadas (2, 3) y consulta vista en (0, 1).
    q_keys = torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    features = layer.read_features(q_keys, ("US", "US"), state)
    # En la dirección nunca vista el apalancamiento y la proyección son exactamente cero.
    torch.testing.assert_close(features[0], torch.zeros_like(features[0]), rtol=0, atol=1e-14)
    assert features[1].abs().sum() > 0.05


def test_markets_stay_isolated_and_row_order_does_not_change_memory_state():
    layer = active_layer(VARIANT)
    flows = ("US/A", "CN/1", "US/B", "US/C", "CN/2", "CN/3", "US/D")
    markets = tuple(f.split("/", 1)[0] for f in flows)
    enc = encoded(len(flows), seed=5)
    targets = [0.05, -0.04, 0.01, -0.02, 0.03, 0.02, -0.06]
    issued = [0.1, 0.2, -0.1, 0.3, -0.2, 0.0, 0.4]

    def run_order(order):
        st = layer.initial_state("p")
        o_flows = tuple(flows[i] for i in order)
        o_markets = tuple(markets[i] for i in order)
        o_enc = enc[list(order)]
        o_keys = layer.keys(o_enc, o_markets, (START,) * len(order))
        o_issued = torch.tensor([issued[i] for i in order], dtype=torch.float64)
        st = layer.stage(st, o_flows, (START,) * len(order), o_keys, o_issued, differentiable=False)
        matured = [(flows[i], START, targets[i]) for i in reversed(order)]
        return layer.settle(st, matured, differentiable=False, train=True)

    base = run_order(list(range(len(flows))))
    perm = run_order([4, 2, 6, 0, 5, 1, 3])
    for market in ("US", "CN"):
        torch.testing.assert_close(
            perm.markets[market].cov, base.markets[market].cov, rtol=1e-13, atol=1e-14
        )
        torch.testing.assert_close(
            perm.markets[market].rhs, base.markets[market].rhs, rtol=1e-13, atol=1e-14
        )
        torch.testing.assert_close(
            perm.markets[market].weight, base.markets[market].weight, rtol=0, atol=0
        )
    # Alterar objetivos de CN no cambia en un solo bit el estado de US.
    st_alt = layer.initial_state("p")
    keys = layer.keys(enc, markets, (START,) * len(flows))
    st_alt = layer.stage(
        st_alt,
        flows,
        (START,) * len(flows),
        keys,
        torch.tensor(issued, dtype=torch.float64),
        differentiable=False,
    )
    alt_targets = [
        v * (-10.0 if flows[i].startswith("CN/") else 1.0) for i, v in enumerate(targets)
    ]
    st_alt = layer.settle(
        st_alt,
        [(f, START, v) for f, v in zip(flows, alt_targets, strict=True)],
        differentiable=False,
        train=True,
    )
    assert torch.equal(st_alt.markets["US"].cov, base.markets["US"].cov)
    assert torch.equal(st_alt.markets["US"].rhs, base.markets["US"].rhs)
    assert not torch.equal(st_alt.markets["CN"].rhs, base.markets["CN"].rhs)


def test_placebo_controls_share_parameters_and_isolate_association_and_eval_updates():
    real = active_layer(VARIANT)
    shuffled = active_layer(SHUFFLED)
    frozen = active_layer(EVAL_FROZEN)
    for (name, a), (_, b), (_, c) in zip(
        real.state_dict().items(),
        shuffled.state_dict().items(),
        frozen.state_dict().items(),
        strict=True,
    ):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b) and torch.equal(a, c), name

    flows = tuple(f"US/S{i}" for i in range(8))
    markets = ("US",) * 8
    enc = encoded(8, seed=19)
    targets = [0.01 * (i - 3.5) for i in range(8)]
    issued = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64)
    matured = [(f, START, y) for f, y in zip(flows, targets, strict=True)]

    st_real = real.stage(
        real.initial_state("p"), flows, (START,) * 8, real.keys(enc, markets, (START,) * 8), issued
    )
    st_real = real.settle(st_real, matured, train=True)
    st_shuf = shuffled.stage(
        shuffled.initial_state("p"),
        flows,
        (START,) * 8,
        shuffled.keys(enc, markets, (START,) * 8),
        issued,
    )
    st_shuf = shuffled.settle(st_shuf, matured, train=True)
    # El placebo barajado conserva la misma covarianza P pero rompe la asociación Q.
    assert torch.equal(st_shuf.markets["US"].cov, st_real.markets["US"].cov)
    assert not torch.equal(st_shuf.markets["US"].rhs, st_real.markets["US"].rhs)

    # eval_frozen actualiza en train=True, pero congela (sin escribir) en train=False.
    st_eval = frozen.stage(
        frozen.initial_state("p"),
        flows,
        (START,) * 8,
        frozen.keys(enc, markets, (START,) * 8),
        issued,
    )
    st_eval = frozen.settle(st_eval, matured, train=False)
    assert "US" not in st_eval.markets and len(st_eval.pending) == 0


def test_disabled_field_keeps_identity_parameters_and_predictions_bit_for_bit():
    base, batch = setup()
    explicit = model(cross_sectional_rank_memory=None)
    assert explicit.config.identity() == base.config.identity()
    assert "cross_sectional_rank_memory" not in base.config.identity()
    assert base.cross_sectional_rank_memory is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert actual.rank_memory_keys is None


@pytest.mark.parametrize("config", [VARIANT, SHUFFLED, EVAL_FROZEN])
def test_all_modes_start_as_the_reference_without_touching_backbone_weights(config):
    base, _ = setup()
    batch = session_batch(6, markets=["US"] * 6)
    predictor = model(cross_sectional_rank_memory=config)
    ref_params = dict(base.named_parameters())
    new_params = dict(predictor.named_parameters())
    assert set(new_params) - set(ref_params) == {
        "cross_sectional_rank_memory.norm.weight",
        "cross_sectional_rank_memory.norm.bias",
        "cross_sectional_rank_memory.key_proj.weight",
        "cross_sectional_rank_memory.ridge_logits",
        "cross_sectional_rank_memory.out.weight",
    }
    for name, value in ref_params.items():
        assert torch.equal(new_params[name], value), name
    assert torch.count_nonzero(predictor.cross_sectional_rank_memory.out.weight) == 0
    rm = predictor.initial_rank_memory()
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True, rank_memory=rm
    )
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    assert actual.rank_memory_keys.shape == (6, config.dim)
    identity = predictor.config.identity()["cross_sectional_rank_memory"]
    assert identity["kind"] == CROSS_SECTIONAL_RANK_MEMORY and identity["mode"] == config.mode


def make_session_batch(rows: int, *, at: int, seed: int, dtype: torch.dtype):
    """Lote multiactivo de un instante con precisión explícita FP32 o FP64."""
    generator = np.random.default_rng(seed)
    markets = ["US"] * rows
    values = {
        name: generator.normal(size=(rows, 64, width) if name == "prices" else (rows, width))
        .astype(np.float32)
        .round(3)
        for name, width in DIMENSIONS.items()
    }
    presence = np.ones((rows, 5), dtype=np.bool_)
    presence[::3, 1] = False
    values["news"][::3] = 0
    values["fundamentals"][:, 1:] = [1, 0.5]
    values["macro"][:] = [0.2, 1, 0]
    names = [f"US/A{index:03}" for index in range(rows)]
    raw = dict(
        inputs=values,
        presence=presence,
        sample_ids=[f"{name}/{at}" for name in names],
        market=markets,
        prediction_at=np.full(rows, at, dtype="datetime64[us]"),
        input_available_at=np.full(rows, at - 1, dtype="datetime64[us]"),
    )
    return DecisionBatch.from_corpus(raw, specification(), dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_meta_gradients_and_constant_zero_head_stay_finite_and_deterministic(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        spec = specification()

        def run_once(zero_out: bool):
            cfg = FinancialConfig(
                spec,
                variant="mac_online",
                hidden_size=32,
                head=QUANTILE_HEAD,
                learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=True),
                median_gain=MEDIAN_GAIN,
                technical_branch=TechnicalDenseConfig(
                    columns=2, hidden=16, depth=2, session_ranks=True
                ),
                gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
                memory_residual_layer_norm=True,
                cross_sectional_rank_memory=VARIANT,
            )
            predictor = FinancialPredictor(cfg, dtype=dtype)
            predictor.set_learning_statistics(statistics(cfg.learning_scale))
            if not zero_out:
                with torch.no_grad():
                    predictor.cross_sectional_rank_memory.out.weight.fill_(0.05)
                predictor._seal_parameters()
            batches = [
                make_session_batch(6, at=START + i * DAY, seed=101 + i, dtype=dtype)
                for i in range(2)
            ]
            state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
            rm = predictor.initial_rank_memory()
            p0 = predictor.prepare(
                batches[0], state, differentiable=True, session_complete=True, rank_memory=rm
            )
            # La mediana arranca exactamente en cero (predicción constante).
            assert torch.count_nonzero(p0.point_predictions) == 0
            rm = predictor.stage_rank_memory(
                rm,
                batches[0].flow_ids,
                batches[0].prediction_at,
                p0.rank_memory_keys,
                p0.point_predictions,
                differentiable=True,
            )
            y0 = [0.02, -0.01, 0.03, -0.04, 0.01, -0.02]
            matured = [
                (flow, START, val) for flow, val in zip(batches[0].flow_ids, y0, strict=True)
            ]
            rm = predictor.settle_rank_memory(rm, matured, differentiable=True, train=True)
            p1 = predictor.prepare(
                batches[1],
                p0.next_state,
                differentiable=True,
                session_complete=True,
                rank_memory=rm,
            )
            y1 = torch.tensor([-0.02, 0.03, -0.01, 0.04, -0.03, 0.02], dtype=dtype)
            sessions = [("US", START + DAY)] * 6
            term = SessionICTerm(weight=3.0, minimum_rows=3)
            loss = pinball_loss(p1.quantiles, y1) + term.term(p1.point_predictions, y1, sessions)
            loss.backward()
            grads = {
                n: p.grad.clone() for n, p in predictor.named_parameters() if p.grad is not None
            }
            return p1.quantiles.detach(), grads

        # Paso 1 con out.weight == 0 y mediana constante en cero: out.weight recibe gradiente != 0.
        q_a, g_a = run_once(zero_out=True)
        q_b, g_b = run_once(zero_out=True)
        assert torch.equal(q_a, q_b)
        for name, grad in g_a.items():
            assert torch.isfinite(grad).all(), name
            assert torch.equal(grad, g_b[name]), name
        assert torch.count_nonzero(g_a["cross_sectional_rank_memory.out.weight"]) > 0

        # Paso 2 (out.weight != 0): el metagradiente atraviesa Cholesky a key_proj y ridge_logits.
        _, g_meta = run_once(zero_out=False)
        for name in (
            "cross_sectional_rank_memory.out.weight",
            "cross_sectional_rank_memory.key_proj.weight",
            "cross_sectional_rank_memory.ridge_logits",
            "cross_sectional_rank_memory.norm.weight",
        ):
            assert torch.isfinite(g_meta[name]).all() and torch.count_nonzero(g_meta[name]) > 0, (
                name
            )
    finally:
        torch.use_deterministic_algorithms(previous)


def test_constant_targets_and_constant_all_zero_scalar_head_produce_finite_gradients():
    spec = specification()
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=True),
        cross_sectional_rank_memory=VARIANT,
    )
    predictor = FinancialPredictor(cfg, dtype=torch.float64)
    predictor.set_learning_statistics(statistics(cfg.learning_scale))
    batches = [
        make_session_batch(5, at=START + i * DAY, seed=201 + i, dtype=torch.float64)
        for i in range(2)
    ]
    state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
    rm = predictor.initial_rank_memory()
    p0 = predictor.prepare(
        batches[0], state, differentiable=True, session_complete=True, rank_memory=rm
    )
    rm = predictor.stage_rank_memory(
        rm,
        batches[0].flow_ids,
        batches[0].prediction_at,
        p0.rank_memory_keys,
        p0.point_predictions,
        differentiable=True,
    )
    # Etiquetas constantes (empate total) + cabeza escalar toda en cero.
    matured = [(flow, START, 0.01) for flow in batches[0].flow_ids]
    rm = predictor.settle_rank_memory(rm, matured, differentiable=True, train=True)
    p1 = predictor.prepare(
        batches[1], p0.next_state, differentiable=True, session_complete=True, rank_memory=rm
    )
    loss = (p1.point_predictions - 0.01).abs().mean()
    loss.backward()
    for name, param in predictor.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def _graph_nodes(tensor):
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(parent for parent, _ in node.next_functions)
    return len(seen)


def test_the_graph_of_a_settled_session_does_not_grow_with_its_rows():
    """Las claves pendientes se reúnen con una operación por emisión, no con una por fila."""
    sizes = {}
    for rows in (8, 64):
        layer = active_layer()
        flows = [f"US/A{index:03d}" for index in range(rows)]
        when = [START] * rows
        keys = layer.keys(encoded(rows=rows), ["US"] * rows, when)
        issued = torch.linspace(-1.0, 1.0, rows, dtype=keys.dtype)
        state = layer.stage(
            layer.initial_state("p"), flows, when, keys, issued, differentiable=True
        )
        assert len(state.pending) == rows
        matured = [
            (flow, START, float(value))
            for flow, value in zip(flows, torch.randn(rows), strict=True)
        ]
        settled = layer.settle(state, matured, differentiable=True)
        assert len(settled.pending) == 0
        sizes[rows] = _graph_nodes(settled.markets["US"].cov)
        # El metagradiente llega a las claves emitidas y es finito.
        (gradient,) = torch.autograd.grad(settled.markets["US"].rhs.sum(), keys)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert sizes[8] == sizes[64], sizes


def test_detaching_keeps_only_the_rows_still_pending_and_exports_them_in_order():
    layer = active_layer()
    rows = 12
    flows = [f"US/A{index:03d}" for index in range(rows)]
    when = [START] * rows
    keys = layer.keys(encoded(rows=rows), ["US"] * rows, when)
    issued = torch.linspace(-1.0, 1.0, rows, dtype=keys.dtype)
    state = layer.stage(layer.initial_state("p"), flows, when, keys, issued, differentiable=True)
    matured = [(flow, START, 0.01 * index) for index, flow in enumerate(flows[:9])]
    partial = layer.detach_state(layer.settle(state, matured, differentiable=True))
    assert sorted(flow for flow, _ in partial.pending) == flows[9:]
    for (flow, _), (source, emitted, row) in partial.pending.items():
        position = flows.index(flow)
        assert source.shape == (3, keys.shape[1]) and not source.requires_grad
        assert emitted.shape == (3,) and not emitted.requires_grad
        assert torch.equal(source[row], keys.detach()[position])
        assert torch.equal(emitted[row], issued[position])
    exported = layer.export_state(partial)["pending"]
    assert exported["flows"] == flows[9:]
    assert torch.equal(exported["keys"], keys.detach()[9:])
    restored = layer.restore_state(layer.export_state(partial), "p", device="cpu")
    again = layer.export_state(restored)["pending"]
    assert again["flows"] == flows[9:] and torch.equal(again["keys"], keys.detach()[9:])


def test_a_session_staged_in_two_blocks_settles_like_one_emission():
    """Una sesión que llega en dos bloques da los mismos estadísticos que emitida de una vez."""
    layer = active_layer()
    rows = 10
    flows = [f"US/A{index:03d}" for index in range(rows)]
    when = [START] * rows
    keys = layer.keys(encoded(rows=rows), ["US"] * rows, when).detach()
    issued = torch.linspace(-1.0, 1.0, rows, dtype=keys.dtype)
    matured = [(flow, START, 0.01 * (index % 4) - 0.02) for index, flow in enumerate(flows)]
    whole = layer.settle(layer.stage(layer.initial_state("p"), flows, when, keys, issued), matured)
    # Los dos bloques llegan en orden distinto al canónico de los flujos.
    first, second = [1, 4, 7, 8, 9], [0, 2, 3, 5, 6]
    state = layer.initial_state("p")
    for block in (first, second):
        state = layer.stage(
            state, [flows[i] for i in block], [START] * len(block), keys[block], issued[block]
        )
    split = layer.settle(state, list(reversed(matured)))
    assert torch.equal(split.markets["US"].cov, whole.markets["US"].cov)
    assert torch.equal(split.markets["US"].rhs, whole.markets["US"].rhs)


def test_export_restore_detach_and_state_dict_contracts():
    predictor = model(cross_sectional_rank_memory=VARIANT)
    with torch.no_grad():
        predictor.cross_sectional_rank_memory.out.weight.fill_(0.08)
    predictor._seal_parameters()
    batch = session_batch(5, markets=["US"] * 5)
    state = predictor.initial_state(batch.flow_ids, differentiable=True)
    rm = predictor.initial_rank_memory()
    p0 = predictor.prepare(batch, state, differentiable=True, session_complete=True, rank_memory=rm)
    rm = predictor.stage_rank_memory(
        rm,
        batch.flow_ids,
        batch.prediction_at,
        p0.rank_memory_keys,
        p0.point_predictions,
        differentiable=True,
    )
    rm = predictor.settle_rank_memory(
        rm,
        [(f, batch.prediction_at[0], 0.01 * i) for i, f in enumerate(batch.flow_ids)],
        differentiable=True,
        train=True,
    )
    # Exportar exige estado desacoplado; antes de detach el grafo atraviesa cov y rhs.
    assert rm.markets["US"].cov.requires_grad and rm.markets["US"].rhs.requires_grad
    detached = predictor.detach_rank_memory(rm)
    assert not detached.markets["US"].cov.requires_grad
    payload = predictor.export_rank_memory(detached)
    restored = predictor.restore_rank_memory(payload)
    assert torch.equal(restored.markets["US"].cov, detached.markets["US"].cov)
    assert torch.equal(restored.markets["US"].rhs, detached.markets["US"].rhs)

    # state_dict round-trip y rechazo entre modos con distinta identidad.
    twin = model(cross_sectional_rank_memory=VARIANT)
    twin.load_state_dict(predictor.state_dict())
    with pytest.raises(ValueError):
        model(cross_sectional_rank_memory=SHUFFLED).load_state_dict(predictor.state_dict())


def test_meta_gradient_flows_through_cholesky_into_written_session_keys():
    layer = active_layer(VARIANT)
    state = layer.initial_state("p")
    flows = ("US/A", "US/B", "US/C", "US/D")
    markets = ("US",) * 4
    enc_1 = encoded(4, seed=71).requires_grad_(True)
    keys_1 = layer.keys(enc_1, markets, (START,) * 4)
    state = layer.stage(
        state,
        flows,
        (START,) * 4,
        keys_1,
        torch.tensor([0.1, -0.2, 0.3, -0.1], dtype=torch.float64),
        differentiable=True,
    )
    matured = [
        ("US/A", START, 0.03),
        ("US/B", START, -0.02),
        ("US/C", START, 0.01),
        ("US/D", START, -0.04),
    ]
    state = layer.settle(state, matured, differentiable=True, train=True)
    assert state.markets["US"].cov.requires_grad and state.markets["US"].rhs.requires_grad
    # Consulta desacoplada: el único camino de gradiente hacia enc_1 y key_proj pasa por Cholesky.
    q_keys = layer.keys(encoded(4, seed=83), markets, (START + DAY,) * 4).detach()
    delta = layer.read(q_keys, markets, state)
    delta.square().sum().backward()
    assert enc_1.grad is not None and torch.isfinite(enc_1.grad).all()
    assert torch.count_nonzero(enc_1.grad) > 0
    assert layer.key_proj.weight.grad is not None
    assert torch.count_nonzero(layer.key_proj.weight.grad) > 0
    assert layer.ridge_logits.grad is not None
    assert torch.count_nonzero(layer.ridge_logits.grad) > 0


def test_registry_declared_sources_and_invalid_declarations():
    raw = asdict(VARIANT)
    assert checked_predictor_changes(dict(cross_sectional_rank_memory=raw)) == dict(
        cross_sectional_rank_memory=raw
    )
    base, batch = setup()
    cfg = replace(base.config, cross_sectional_rank_memory=json.loads(json.dumps(raw)))
    assert cfg.cross_sectional_rank_memory == VARIANT
    assert declared_sources(base.config) == ()
    assert declared_sources(cfg) == ("mars_titan.models.titans.cross_sectional_rank_memory",)

    # El consumidor congelado rechaza CRRM explícitamente.
    predictor = model(cross_sectional_rank_memory=VARIANT).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="memoria transversal de rango"):
        FrozenFinancialConsumer(predictor)

    # Sin declarar CRRM no admite rank_memory; declarándola la exige y exige sesión completa.
    with pytest.raises(ValueError, match="no declara"):
        base.prepare(batch, base.initial_state(batch.flow_ids), rank_memory=object())
    active = model(cross_sectional_rank_memory=VARIANT)
    with pytest.raises(ValueError, match="sesión completa"):
        active.prepare(
            batch,
            active.initial_state(batch.flow_ids),
            rank_memory=active.initial_rank_memory(),
        )
    with pytest.raises(ValueError, match="estado de memoria transversal"):
        active.prepare(batch, active.initial_state(batch.flow_ids), session_complete=True)

    # Control local C no se combina con CRRM.
    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(
            cfg,
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )

    # Variantes distintas de mac_online y declaraciones malformadas se rechazan.
    for variant in ("mac_frozen", "mac_disabled", "transformer_direct"):
        with pytest.raises(ValueError, match="mac_online"):
            replace(base.config, variant=variant, cross_sectional_rank_memory=VARIANT)
    for bad in (
        dict(dim=3, time_constants=[5], ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0),
        dict(dim=8, time_constants=[5, 5], ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0),
        dict(dim=8, time_constants=[], ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0),
        dict(dim=8, time_constants=[5], ridge=0.0, minimum_rows=3, mode="matured_rank", seed=0),
        dict(dim=8, time_constants=[5], ridge=1.0, minimum_rows=2, mode="matured_rank", seed=0),
        dict(dim=8, time_constants=[5], ridge=1.0, minimum_rows=3, mode="unknown", seed=0),
        dict(dim=8, time_constants=[5], ridge=1.0, minimum_rows=3, mode="matured_rank", seed=-1),
        dict(dim=8),
        "cross_sectional_rank_memory_v1",
    ):
        with pytest.raises(ValueError):
            replace(base.config, cross_sectional_rank_memory=bad)

    # Autocast se rechaza.
    layer = CrossSectionalRankMemory(16, VARIANT, seed=1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer.keys(encoded(4, dtype=torch.float32), ("US",) * 4, (START,) * 4)

    # El código entra en ChronologicalTrainer._code solo cuando está activo.
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    tr = object.__new__(ChronologicalTrainer)
    tr.recipe = ChronologicalRecipe()
    mod = "mars_titan.models.titans.cross_sectional_rank_memory"
    for val, present in ((None, False), (VARIANT, True)):
        tr.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=None,
                asset_groups=None,
                holographic_network=None,
                axial_time_region=None,
                mature_reliability=None,
                fractal_shared_operator=None,
                cross_sectional_rank_memory=val,
            )
        )
        assert (mod in tr._code()) is present


def test_campaign_core_recipe_accepts_single_variable_crrm_declaration():
    recipe_path = (
        Path(__file__).resolve().parents[3] / "configs/titans/dgx-us-core-u2-ic3-dense-gr-v9.json"
    )
    doc = json.loads(recipe_path.read_text())
    predictor_opts = dict(doc["predictor"])
    predictor_opts.pop("dtype")
    # En esta especificación sintética solo ajustamos charts=46 para coincidir con la receta real.
    base_spec = specification()
    spec = FinancialInputSpec(
        source_sha256=base_spec.source_sha256,
        view_sha256=base_spec.view_sha256,
        representation=json.loads(base_spec._representation_json),
        dimensions=dict(base_spec.dimensions, charts=46),
        input_policy=base_spec.input_policy,
    )
    candidate = dict(
        dim=32,
        time_constants=[5, 21, 63],
        ridge=1.0,
        minimum_rows=8,
        mode="matured_rank",
        seed=0,
    )
    changes = checked_predictor_changes(dict(cross_sectional_rank_memory=candidate))
    control_cfg = FinancialConfig(spec, variant="mac_online", **predictor_opts)
    candidate_cfg = FinancialConfig(spec, variant="mac_online", **(predictor_opts | changes))
    control_id = control_cfg.identity()
    candidate_id = candidate_cfg.identity()
    assert set(candidate_id) - set(control_id) == {"cross_sectional_rank_memory"}
    assert {
        k: v for k, v in candidate_id.items() if k != "cross_sectional_rank_memory"
    } == control_id
