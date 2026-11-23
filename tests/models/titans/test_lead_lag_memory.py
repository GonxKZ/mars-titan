"""Memoria rápida de desfase dinámico entre grupos de activos (LL-FM): contratos y numérica."""

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
from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, declared_sources
from mars_titan.models.titans.lead_lag_memory import (
    DECAY_CEILING,
    DECAY_FLOOR,
    GROUP_MASS_FLOOR,
    LEAD_LAG_MEMORY,
    LOGIT_BOUND,
    LR_CEILING,
    LR_FLOOR,
    PROJ_EPS,
    LeadLagMemory,
    LeadLagMemoryConfig,
    parse_lead_lag_memory,
)
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm, standardized
from mars_titan.models.titans.session_ranks import gaussian_ranks
from mars_titan.models.titans.technical_branch import TechnicalDenseConfig
from mars_titan.training import comparison_campaign
from mars_titan.training.search_cases import checked_predictor_changes

START, DAY = 1_609_459_200_000_000, 86_400_000_000
VARIANT = LeadLagMemoryConfig(
    groups=6,
    lag_scales=(1, 5),
    decay=0.08,
    learning_rate=0.25,
    damping=1.0,
    minimum_rows=3,
    mode="lead_lag",
    seed=0,
)
SHUFFLED = replace(VARIANT, mode="shuffled_targets")
STATIC_PRIOR = replace(VARIANT, mode="static_prior")
EVAL_FROZEN = replace(VARIANT, mode="eval_frozen")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def active_layer(config=VARIANT, *, hidden=16, seed=7, dtype=torch.float64):
    """Módulo con la puerta vectorial abierta para observar la señal de contagio."""
    layer = LeadLagMemory(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 17)
        layer.gate.copy_(torch.randn(hidden, generator=generator, dtype=dtype) / 3)
    return layer


def encoded(rows=6, hidden=16, seed=11, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=dtype)


def reference_step(layer, sessions, query_encoded, query_markets, query_at):
    """Referencia explícita fila a fila de las ecuaciones exactas de LL-FM."""
    config = layer.config
    groups = config.groups
    scales = len(config.lag_scales)
    dtype, device = query_encoded.dtype, query_encoded.device
    off_diag = 1.0 - torch.eye(groups, dtype=dtype, device=device)
    prior = layer.prior_transition * off_diag.unsqueeze(0)
    alpha = DECAY_FLOOR + (DECAY_CEILING - DECAY_FLOOR) * torch.sigmoid(layer.decay_logits)
    eta = LR_FLOOR + (LR_CEILING - LR_FLOOR) * torch.sigmoid(layer.lr_logits)
    trace_decays = [0.0 if tau == 1 else math.exp(-1.0 / tau) for tau in config.lag_scales]

    def compute_probs(values, markets, instants):
        raw = layer.group_proj(layer.norm(values.detach()))
        by_session = {}
        for row, key in enumerate(zip(markets, instants, strict=True)):
            by_session.setdefault(key, []).append(row)
        out = torch.empty_like(raw)
        for rows in by_session.values():
            idx = torch.tensor(rows, dtype=torch.int64, device=device)
            part = raw.index_select(0, idx)
            if len(rows) > 1:
                part = part - part.mean(dim=0, keepdim=True)
            bounded = LOGIT_BOUND * torch.tanh(part / LOGIT_BOUND)
            out[idx] = torch.softmax(bounded, dim=-1)
        return out

    markets_state = {}
    for enc, flows, at, targets, issued in sessions:
        mkts = tuple(flow.split("/", 1)[0] for flow in flows)
        probs = compute_probs(enc, mkts, (at,) * len(flows))
        by_market = {}
        for i, flow in enumerate(flows):
            by_market.setdefault(flow.split("/", 1)[0], []).append((flow, i))
        for market in sorted(by_market):
            ordered = sorted(by_market[market], key=lambda item: item[0])
            if len(ordered) < config.minimum_rows:
                continue
            idx = [i for _, i in ordered]
            p_mat = probs[idx]
            y_vec = torch.tensor([targets[i] for i in idx], dtype=dtype, device=device)
            pred_vec = torch.tensor([issued[i] for i in idx], dtype=dtype, device=device)
            g_vec = gaussian_ranks(
                y_vec.unsqueeze(-1), torch.ones((len(idx), 1), dtype=torch.bool, device=device)
            ).squeeze(-1)
            g_hat = standardized(pred_vec)
            beta = (g_vec * g_hat).sum() / (g_hat.square().sum() + PROJ_EPS)
            r_perp = g_vec - beta * g_hat
            mass = p_mat.sum(dim=0) + GROUP_MASS_FLOOR
            group_r = (p_mat.T @ r_perp) / mass

            if market not in markets_state:
                delta_t = torch.zeros((scales, groups, groups), dtype=dtype, device=device)
                trace = group_r.unsqueeze(0).expand(scales, groups).clone()
                markets_state[market] = [delta_t, trace]
            else:
                prev_delta, prev_trace = markets_state[market]
                new_delta = prev_delta.clone()
                new_trace = torch.empty_like(prev_trace)
                for h in range(scales):
                    z_prev = prev_trace[h]
                    if config.mode != "static_prior":
                        t_eff = (prior[h] + prev_delta[h]) * off_diag
                        pred_r = t_eff @ z_prev
                        err = group_r - pred_r
                        step = torch.zeros((groups, groups), dtype=dtype, device=device)
                        for b in range(groups):
                            z_no_b = z_prev.clone()
                            z_no_b[b] = 0.0
                            denom = z_no_b.square().sum() + config.damping
                            step[b] = (eta[h] / denom) * err[b] * z_no_b
                        new_delta[h] = (1.0 - alpha[h]) * (prev_delta[h] + step) * off_diag
                    lam = trace_decays[h]
                    new_trace[h] = lam * z_prev + (1.0 - lam) * group_r
                markets_state[market] = [new_delta, new_trace]

    q_probs = compute_probs(query_encoded, query_markets, query_at)
    kappa = torch.sigmoid(layer.receptor_proj(layer.norm(query_encoded.detach())))
    features = torch.zeros((len(query_encoded), 2 * scales * groups), dtype=dtype, device=device)
    by_session = {}
    for row, key in enumerate(zip(query_markets, query_at, strict=True)):
        by_session.setdefault(key, []).append(row)
    for (market, _instant), rows in by_session.items():
        if market not in markets_state:
            continue
        delta_t, trace = markets_state[market]
        t_eff = (prior + delta_t) * off_diag.unsqueeze(0)
        t_asym = 0.5 * (t_eff - t_eff.transpose(-1, -2))
        s_off = torch.tanh(torch.einsum("hgk,hk->hg", t_eff, trace))
        s_asym = torch.tanh(torch.einsum("hgk,hk->hg", t_asym, trace))
        idx = torch.tensor(rows, dtype=torch.int64, device=device)
        scaled_p = float(groups) * q_probs.index_select(0, idx)
        k_m = kappa.index_select(0, idx)
        ch_off = (scaled_p[:, None, :] * s_off[None, :, :]).reshape(len(rows), scales * groups)
        ch_asym = ((scaled_p * k_m)[:, None, :] * s_asym[None, :, :]).reshape(
            len(rows), scales * groups
        )
        part = torch.cat((ch_off, ch_asym), dim=-1)
        if len(rows) > 1:
            part = part - part.mean(dim=0, keepdim=True)
        features[idx] = part
    return layer.gate * layer.out(features)


def test_read_and_settle_match_explicit_off_diagonal_nlms_reference_equations():
    layer = active_layer(VARIANT)
    state = layer.initial_state("p")
    flows_1 = ("US/AAA", "US/BBB", "US/CCC", "US/DDD", "CN/600000.SS", "CN/600001.SS")
    enc_1 = encoded(len(flows_1), seed=21)
    issued_1 = [0.2, -0.1, 0.5, -0.4, 0.1, -0.2]
    targets_1 = [0.03, -0.02, 0.01, -0.05, 0.02, -0.01]
    markets_1 = tuple(f.split("/", 1)[0] for f in flows_1)
    probs_1 = layer.memberships(enc_1, markets_1, (START,) * len(flows_1))
    # Antes de madurar ninguna etiqueta, la lectura es exactamente cero.
    assert torch.count_nonzero(layer.read(enc_1, probs_1, markets_1, (START,) * 6, state)) == 0
    state = layer.stage(
        state,
        flows_1,
        (START,) * len(flows_1),
        probs_1,
        torch.tensor(issued_1, dtype=torch.float64),
        differentiable=True,
    )
    matured_1 = [(flow, START, val) for flow, val in zip(flows_1, targets_1, strict=True)]
    state = layer.settle(state, matured_1, differentiable=True, train=True)
    # CN tuvo solo 2 filas (< minimum_rows=3): no crea estado CN, pero vacía su cola pendiente.
    assert "US" in state.markets and "CN" not in state.markets
    assert state.markets["US"].sessions == 1 and state.markets["US"].updates == 0
    assert len(state.pending) == 0

    flows_2 = ("US/DDD", "US/AAA", "US/CCC", "US/BBB")
    enc_2 = encoded(len(flows_2), seed=31)
    issued_2 = [-0.3, 0.4, 0.0, 0.1]
    targets_2 = [-0.01, 0.04, 0.02, -0.03]
    markets_2 = ("US",) * len(flows_2)
    probs_2 = layer.memberships(enc_2, markets_2, (START + DAY,) * len(flows_2))
    state = layer.stage(
        state,
        flows_2,
        (START + DAY,) * len(flows_2),
        probs_2,
        torch.tensor(issued_2, dtype=torch.float64),
        differentiable=True,
    )
    matured_2 = [(flow, START + DAY, val) for flow, val in zip(flows_2, targets_2, strict=True)]
    state = layer.settle(state, matured_2, differentiable=True, train=True)
    assert state.markets["US"].sessions == 2 and state.markets["US"].updates == 1

    flows_3 = ("US/AAA", "US/BBB", "US/CCC", "US/DDD")
    enc_3 = encoded(len(flows_3), seed=37)
    issued_3 = [0.1, 0.3, -0.2, -0.1]
    targets_3 = [0.05, -0.04, 0.01, -0.02]
    probs_3 = layer.memberships(enc_3, markets_2, (START + 2 * DAY,) * len(flows_3))
    state = layer.stage(
        state,
        flows_3,
        (START + 2 * DAY,) * len(flows_3),
        probs_3,
        torch.tensor(issued_3, dtype=torch.float64),
        differentiable=True,
    )
    matured_3 = [(flow, START + 2 * DAY, val) for flow, val in zip(flows_3, targets_3, strict=True)]
    state = layer.settle(state, matured_3, differentiable=True, train=True)
    assert state.markets["US"].sessions == 3 and state.markets["US"].updates == 2

    q_enc = encoded(5, seed=41)
    q_markets = ("US", "CN", "US", "US", "CN")
    q_at = (START + 3 * DAY,) * 5
    q_probs = layer.memberships(q_enc, q_markets, q_at)
    actual = layer.read(q_enc, q_probs, q_markets, q_at, state)
    expected = reference_step(
        layer,
        [
            (enc_1, flows_1, START, targets_1, issued_1),
            (enc_2, flows_2, START + DAY, targets_2, issued_2),
            (enc_3, flows_3, START + 2 * DAY, targets_3, issued_3),
        ],
        q_enc,
        q_markets,
        q_at,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    assert torch.count_nonzero(actual[[1, 4]]) == 0
    assert torch.count_nonzero(actual[[0, 2, 3]]) > 0


def test_diagonal_of_transition_is_strictly_zero_and_prevents_own_group_rank_regression():
    config = LeadLagMemoryConfig(
        groups=4,
        lag_scales=(1,),
        decay=0.05,
        learning_rate=0.3,
        damping=1.0,
        minimum_rows=3,
        mode="lead_lag",
        seed=0,
    )
    layer = active_layer(config, hidden=8)
    # Poner valores enormes solo en la diagonal del prior y cero fuera de la diagonal.
    with torch.no_grad():
        layer.prior_transition.zero_()
        for g in range(4):
            layer.prior_transition[:, g, g] = 10.0
    state = layer.initial_state("p")
    flows = ("US/A", "US/B", "US/C", "US/D")
    markets = ("US",) * 4
    # Sesión 1: cada activo tiene pertenencia pura a un grupo distinto.
    probs_1 = torch.eye(4, dtype=torch.float64)
    state = layer.stage(
        state,
        flows,
        (START,) * 4,
        probs_1,
        torch.zeros(4, dtype=torch.float64),
        differentiable=False,
    )
    state = layer.settle(
        state,
        [
            ("US/A", START, 0.04),
            ("US/B", START, -0.03),
            ("US/C", START, 0.02),
            ("US/D", START, -0.01),
        ],
        differentiable=False,
        train=True,
    )
    # Tras 1 sesión (solo prior, delta=0): como la diagonal se anula estrictamente, lectura = 0.
    feats = layer.read_features(
        encoded(4, hidden=8, seed=3), probs_1, markets, (START + DAY,) * 4, state
    )
    assert torch.count_nonzero(feats) == 0

    # Tras varias sesiones maduras con delta-rule activa, la diagonal de delta_transition es 0.
    for step in range(1, 6):
        at = START + step * DAY
        enc = encoded(4, hidden=8, seed=10 + step)
        p = layer.memberships(enc, markets, (at,) * 4)
        state = layer.stage(
            state, flows, (at,) * 4, p, torch.zeros(4, dtype=torch.float64), differentiable=False
        )
        matured = [(f, at, 0.01 * ((i + step) % 4 - 1.5)) for i, f in enumerate(flows)]
        state = layer.settle(state, matured, differentiable=False, train=True)
        diag = torch.diagonal(state.markets["US"].delta_transition, dim1=-2, dim2=-1)
        assert torch.count_nonzero(diag) == 0


def test_orthogonalized_innovation_eliminates_backbone_rank_signal_and_keeps_orthogonal_part():
    layer = active_layer(VARIANT)
    flows = tuple(f"US/S{i}" for i in range(8))
    markets = ("US",) * 8
    enc = encoded(8, seed=91)
    probs = layer.memberships(enc, markets, (START,) * 8)
    targets = [0.01 * (i - 3.5) for i in range(8)]
    g_true = gaussian_ranks(
        torch.tensor(targets, dtype=torch.float64).unsqueeze(-1),
        torch.ones((8, 1), dtype=torch.bool),
    ).squeeze(-1)

    # Caso 1: predicción emitida colineal con el rango normal maduro -> innovación ~ 0.
    st_collinear = layer.stage(
        layer.initial_state("p"), flows, (START,) * 8, probs, g_true * 0.05, differentiable=False
    )
    st_collinear = layer.settle(
        st_collinear, [(f, START, y) for f, y in zip(flows, targets, strict=True)], train=True
    )
    torch.testing.assert_close(
        st_collinear.markets["US"].trace,
        torch.zeros_like(st_collinear.markets["US"].trace),
        rtol=0,
        atol=1e-7,
    )

    # Caso 2: predicción emitida nula (cabeza constante en 0) -> innovación = g_true completa.
    st_zero = layer.stage(
        layer.initial_state("p"),
        flows,
        (START,) * 8,
        probs,
        torch.zeros(8, dtype=torch.float64),
        differentiable=False,
    )
    st_zero = layer.settle(
        st_zero, [(f, START, y) for f, y in zip(flows, targets, strict=True)], train=True
    )
    assert st_zero.markets["US"].trace.abs().max() > 0.05

    # Caso 3: predicción con baja correlación (IC ~ 0.11): el signo NO se invierte.
    low_ic_pred = 0.12 * g_true + torch.tensor(
        [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0], dtype=torch.float64
    )
    st_low = layer.stage(
        layer.initial_state("p"), flows, (START,) * 8, probs, low_ic_pred, differentiable=False
    )
    st_low = layer.settle(
        st_low, [(f, START, y) for f, y in zip(flows, targets, strict=True)], train=True
    )
    # La traza ortogonalizada mantiene alta correlación positiva con el caso sin predictor.
    cos = torch.nn.functional.cosine_similarity(
        st_low.markets["US"].trace[0].unsqueeze(0),
        st_zero.markets["US"].trace[0].unsqueeze(0),
    )
    assert float(cos) > 0.90


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nlms_delta_rule_stays_strictly_bounded_over_many_sessions(dtype):
    config = replace(VARIANT, learning_rate=0.85, decay=0.05, damping=0.1)
    layer = active_layer(config, dtype=dtype)
    state = layer.initial_state("p")
    flows = tuple(f"US/A{i:02d}" for i in range(12))
    markets = ("US",) * 12
    for step in range(80):
        at = START + step * DAY
        enc = encoded(12, seed=200 + step, dtype=dtype) * 5.0
        probs = layer.memberships(enc, markets, (at,) * 12)
        issued = torch.zeros(12, dtype=dtype)
        state = layer.stage(state, flows, (at,) * 12, probs, issued, differentiable=False)
        matured = [(f, at, float((i - 5.5) * ((-1) ** (step + i)))) for i, f in enumerate(flows)]
        state = layer.settle(state, matured, differentiable=False, train=True)
    stats = state.markets["US"]
    assert torch.isfinite(stats.delta_transition).all()
    assert torch.isfinite(stats.trace).all()
    assert float(stats.delta_transition.abs().max()) < 50.0


def test_markets_stay_isolated_and_row_order_does_not_change_memory_state():
    layer = active_layer(VARIANT)
    flows = ("US/A", "CN/1", "US/B", "US/C", "CN/2", "CN/3", "US/D")
    markets = tuple(f.split("/", 1)[0] for f in flows)
    enc = encoded(len(flows), seed=5)
    targets = [0.05, -0.04, 0.01, -0.02, 0.03, 0.02, -0.06]
    issued = [0.1, 0.0, -0.1, 0.3, -0.2, 0.2, 0.4]

    def run_order(order):
        st = layer.initial_state("p")
        o_flows = tuple(flows[i] for i in order)
        o_markets = tuple(markets[i] for i in order)
        o_enc = enc[list(order)]
        for step, shift in enumerate((0.0, 0.5)):
            at = START + step * DAY
            o_probs = layer.memberships(o_enc + shift, o_markets, (at,) * len(order))
            o_issued = torch.tensor([issued[i] for i in order], dtype=torch.float64)
            st = layer.stage(
                st, o_flows, (at,) * len(order), o_probs, o_issued, differentiable=False
            )
            matured = [
                (flows[i], at, targets[i] * (1.0 if step == 0 else -1.5)) for i in reversed(order)
            ]
            st = layer.settle(st, matured, differentiable=False, train=True)
        return st

    base = run_order(list(range(len(flows))))
    perm = run_order([4, 2, 6, 0, 5, 1, 3])
    for market in ("US", "CN"):
        torch.testing.assert_close(
            perm.markets[market].delta_transition,
            base.markets[market].delta_transition,
            rtol=1e-13,
            atol=1e-14,
        )
        torch.testing.assert_close(
            perm.markets[market].trace, base.markets[market].trace, rtol=1e-13, atol=1e-14
        )
        assert perm.markets[market].sessions == base.markets[market].sessions == 2
        assert perm.markets[market].updates == base.markets[market].updates == 1

    # Alterar objetivos de CN no cambia ni un bit el estado de US.
    st_alt = layer.initial_state("p")
    for step, shift in enumerate((0.0, 0.5)):
        at = START + step * DAY
        probs = layer.memberships(enc + shift, markets, (at,) * len(flows))
        st_alt = layer.stage(
            st_alt,
            flows,
            (at,) * len(flows),
            probs,
            torch.tensor(issued, dtype=torch.float64),
            differentiable=False,
        )
        alt_targets = [
            (
                0.08
                if step == 1 and flows[i] == "CN/1"
                else -0.05
                if step == 1 and flows[i] == "CN/2"
                else v
            )
            * (1.0 if step == 0 else -1.5)
            for i, v in enumerate(targets)
        ]
        st_alt = layer.settle(
            st_alt,
            [(f, at, v) for f, v in zip(flows, alt_targets, strict=True)],
            differentiable=False,
            train=True,
        )
    assert torch.equal(st_alt.markets["US"].delta_transition, base.markets["US"].delta_transition)
    assert torch.equal(st_alt.markets["US"].trace, base.markets["US"].trace)
    assert not torch.equal(
        st_alt.markets["CN"].delta_transition, base.markets["CN"].delta_transition
    )

    # En lectura, perturbar filas de CN o de un instante futuro no altera filas de US del presente.
    q_enc = encoded(len(flows), seed=55)
    q_at = (START + 2 * DAY,) * 6 + (START + 3 * DAY,)
    q_probs = layer.memberships(q_enc, markets, q_at)
    base_read = layer.read(q_enc, q_probs, markets, q_at, base)
    perturbed = q_enc.clone()
    perturbed[[1, 4, 5, 6]] += 12.0  # CN y la fila US del instante futuro
    p_probs = layer.memberships(perturbed, markets, q_at)
    perturbed_read = layer.read(perturbed, p_probs, markets, q_at, base)
    assert torch.equal(perturbed_read[[0, 2, 3]], base_read[[0, 2, 3]])


def test_placebo_and_ablation_modes_share_exact_parameters_and_isolate_mechanisms():
    real = active_layer(VARIANT)
    shuffled = active_layer(SHUFFLED)
    static = active_layer(STATIC_PRIOR)
    frozen = active_layer(EVAL_FROZEN)
    for other in (shuffled, static, frozen):
        assert set(real.state_dict()) == set(other.state_dict())
        for name, tensor in real.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, other.state_dict()[name]), name

    flows = tuple(f"US/S{i}" for i in range(8))
    markets = ("US",) * 8
    issued = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64)

    def run_two(layer, *, train=True):
        st = layer.initial_state("p")
        for step in range(2):
            at = START + step * DAY
            enc = encoded(8, seed=19 + step)
            probs = layer.memberships(enc, markets, (at,) * 8)
            st = layer.stage(st, flows, (at,) * 8, probs, issued, differentiable=False)
            targets = [0.01 * ((i + 3 * step) % 8 - 3.5) for i in range(8)]
            st = layer.settle(
                st, [(f, at, y) for f, y in zip(flows, targets, strict=True)], train=train
            )
        return st

    st_real = run_two(real)
    st_shuf = run_two(shuffled)
    st_stat = run_two(static)
    st_eval = run_two(frozen, train=False)

    # shuffled_targets destruye la correspondencia real y es determinista.
    assert not torch.equal(st_shuf.markets["US"].trace, st_real.markets["US"].trace)
    assert not torch.equal(
        st_shuf.markets["US"].delta_transition, st_real.markets["US"].delta_transition
    )
    assert torch.equal(
        st_shuf.markets["US"].delta_transition, run_two(shuffled).markets["US"].delta_transition
    )

    # static_prior conserva la misma traza causal que lead_lag pero mantiene delta_transition == 0.
    assert torch.equal(st_stat.markets["US"].trace, st_real.markets["US"].trace)
    assert torch.count_nonzero(st_stat.markets["US"].delta_transition) == 0
    assert st_stat.markets["US"].updates == 0 and st_stat.markets["US"].sessions == 2
    assert torch.count_nonzero(st_real.markets["US"].delta_transition) > 0

    # eval_frozen no escribe en train=False y vacía pending.
    assert "US" not in st_eval.markets and len(st_eval.pending) == 0


def test_disabled_field_keeps_identity_fingerprint_parameters_and_predictions_bit_for_bit():
    base, batch = setup()
    explicit = model(lead_lag_memory=None)
    assert explicit.config.identity() == base.config.identity()
    assert explicit._config_id() == base._config_id()
    assert explicit._parameter_id == base._parameter_id
    assert "lead_lag_memory" not in base.config.identity()
    assert base.lead_lag_memory is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    assert actual.lead_lag_memberships is None


@pytest.mark.parametrize("config", [VARIANT, SHUFFLED, STATIC_PRIOR, EVAL_FROZEN])
def test_all_modes_start_as_the_reference_without_touching_backbone_weights(config):
    base, _ = setup()
    batch = session_batch(6, markets=["US"] * 6)
    predictor = model(lead_lag_memory=config)
    ref_params = dict(base.named_parameters())
    new_params = dict(predictor.named_parameters())
    assert set(new_params) - set(ref_params) == {
        "lead_lag_memory.norm.weight",
        "lead_lag_memory.norm.bias",
        "lead_lag_memory.group_proj.weight",
        "lead_lag_memory.receptor_proj.weight",
        "lead_lag_memory.receptor_proj.bias",
        "lead_lag_memory.prior_transition",
        "lead_lag_memory.decay_logits",
        "lead_lag_memory.lr_logits",
        "lead_lag_memory.out.weight",
        "lead_lag_memory.gate",
    }
    for name, value in ref_params.items():
        assert torch.equal(new_params[name], value), name
    assert torch.count_nonzero(predictor.lead_lag_memory.gate) == 0
    assert torch.count_nonzero(predictor.lead_lag_memory.out.weight) > 0
    ll = predictor.initial_lead_lag_memory()
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True, lead_lag_memory=ll
    )
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    assert actual.lead_lag_memberships.shape == (6, config.groups)
    identity = predictor.config.identity()["lead_lag_memory"]
    assert identity["kind"] == LEAD_LAG_MEMORY and identity["mode"] == config.mode


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
def test_meta_gradients_stop_gradient_and_constant_zero_head_stay_finite_and_deterministic(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        spec = specification()

        def run_once(*, zero_gate: bool, sessions_count: int):
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
                lead_lag_memory=VARIANT,
            )
            predictor = FinancialPredictor(cfg, dtype=dtype)
            predictor.set_learning_statistics(statistics(cfg.learning_scale))
            if not zero_gate:
                with torch.no_grad():
                    predictor.lead_lag_memory.gate.fill_(0.08)
                predictor._seal_parameters()
            batches = [
                make_session_batch(6, at=START + i * DAY, seed=101 + i, dtype=dtype)
                for i in range(sessions_count)
            ]
            state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
            ll = predictor.initial_lead_lag_memory()
            targets_all = [
                [0.02, -0.01, 0.03, -0.04, 0.01, -0.02],
                [-0.02, 0.03, -0.01, 0.04, -0.03, 0.02],
                [0.01, -0.03, 0.02, -0.01, 0.04, -0.02],
            ]
            prepared = None
            for i in range(sessions_count):
                prepared = predictor.prepare(
                    batches[i],
                    state,
                    differentiable=True,
                    session_complete=True,
                    lead_lag_memory=ll,
                )
                if i == 0:
                    assert torch.count_nonzero(prepared.point_predictions) == 0
                state = prepared.next_state
                if i < sessions_count - 1:
                    ll = predictor.stage_lead_lag_memory(
                        ll,
                        batches[i].flow_ids,
                        batches[i].prediction_at,
                        prepared.lead_lag_memberships,
                        prepared.point_predictions,
                        differentiable=True,
                    )
                    matured = [
                        (flow, START + i * DAY, val)
                        for flow, val in zip(batches[i].flow_ids, targets_all[i], strict=True)
                    ]
                    ll = predictor.settle_lead_lag_memory(
                        ll, matured, differentiable=True, train=True
                    )
            y_last = torch.tensor(targets_all[sessions_count - 1], dtype=dtype)
            sessions = [("US", START + (sessions_count - 1) * DAY)] * 6
            term = SessionICTerm(weight=3.0, minimum_rows=3)
            loss = pinball_loss(prepared.quantiles, y_last) + term.term(
                prepared.point_predictions, y_last, sessions
            )
            loss.backward()
            grads = {
                n: p.grad.clone() for n, p in predictor.named_parameters() if p.grad is not None
            }
            return prepared.quantiles.detach(), grads

        # Paso 1 (truncation=2, gate==0, mediana en 0): gate recibe gradiente != 0 ya en el paso 1.
        q_a, g_a = run_once(zero_gate=True, sessions_count=2)
        q_b, g_b = run_once(zero_gate=True, sessions_count=2)
        assert torch.equal(q_a, q_b)
        for name, grad in g_a.items():
            assert torch.isfinite(grad).all(), name
            assert torch.equal(grad, g_b[name]), name
        assert torch.count_nonzero(g_a["lead_lag_memory.gate"]) > 0

        # Con gate != 0 y 2 maduraciones (delta-rule activa): todos los parámetros reciben grad.
        _, g_meta = run_once(zero_gate=False, sessions_count=3)
        for name in (
            "lead_lag_memory.norm.weight",
            "lead_lag_memory.norm.bias",
            "lead_lag_memory.group_proj.weight",
            "lead_lag_memory.receptor_proj.weight",
            "lead_lag_memory.receptor_proj.bias",
            "lead_lag_memory.prior_transition",
            "lead_lag_memory.decay_logits",
            "lead_lag_memory.lr_logits",
            "lead_lag_memory.out.weight",
            "lead_lag_memory.gate",
        ):
            assert torch.isfinite(g_meta[name]).all() and torch.count_nonzero(g_meta[name]) > 0, (
                name
            )
    finally:
        torch.use_deterministic_algorithms(previous)


def test_stop_gradient_protects_backbone_encoded_while_meta_gradient_reaches_lead_lag_params():
    """El gradiente de LL-FM no altera `encoded` del tronco, pero sí atraviesa la regla delta."""
    layer = active_layer(VARIANT)
    state = layer.initial_state("p")
    flows = ("US/A", "US/B", "US/C", "US/D")
    markets = ("US",) * 4
    enc_0 = encoded(4, seed=71).requires_grad_(True)
    enc_1 = encoded(4, seed=72).requires_grad_(True)
    enc_q = encoded(4, seed=73).requires_grad_(True)

    p0 = layer.memberships(enc_0, markets, (START,) * 4)
    state = layer.stage(
        state, flows, (START,) * 4, p0, torch.zeros(4, dtype=torch.float64), differentiable=True
    )
    state = layer.settle(
        state,
        [
            ("US/A", START, 0.03),
            ("US/B", START, -0.02),
            ("US/C", START, 0.01),
            ("US/D", START, -0.04),
        ],
        differentiable=True,
        train=True,
    )
    p1 = layer.memberships(enc_1, markets, (START + DAY,) * 4)
    state = layer.stage(
        state,
        flows,
        (START + DAY,) * 4,
        p1,
        torch.zeros(4, dtype=torch.float64),
        differentiable=True,
    )
    state = layer.settle(
        state,
        [
            ("US/A", START + DAY, -0.02),
            ("US/B", START + DAY, 0.04),
            ("US/C", START + DAY, -0.01),
            ("US/D", START + DAY, 0.02),
        ],
        differentiable=True,
        train=True,
    )
    # Consulta con pertenencias y receptor desacoplados: el único camino de gradiente a los
    # parámetros dinámicos pasa por `delta_transition` y `trace` (metagradiente de la regla delta).
    p_q = layer.memberships(enc_q, markets, (START + 2 * DAY,) * 4)
    delta = layer.read(enc_q, p_q, markets, (START + 2 * DAY,) * 4, state)
    delta.square().sum().backward()
    # Protección anti-CRRM: el tronco `encoded` recibe gradiente nulo desde LL-FM.
    assert enc_0.grad is None and enc_1.grad is None and enc_q.grad is None
    # Todos los parámetros de LL-FM sí reciben metagradientes finitos y no nulos.
    for name, param in layer.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert torch.count_nonzero(param.grad) > 0, name


def test_constant_targets_and_constant_all_zero_scalar_head_produce_finite_gradients():
    spec = specification()
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=True),
        lead_lag_memory=VARIANT,
    )
    predictor = FinancialPredictor(cfg, dtype=torch.float64)
    predictor.set_learning_statistics(statistics(cfg.learning_scale))
    batches = [
        make_session_batch(5, at=START + i * DAY, seed=201 + i, dtype=torch.float64)
        for i in range(3)
    ]
    state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
    ll = predictor.initial_lead_lag_memory()
    for i in range(2):
        prep = predictor.prepare(
            batches[i], state, differentiable=True, session_complete=True, lead_lag_memory=ll
        )
        state = prep.next_state
        ll = predictor.stage_lead_lag_memory(
            ll,
            batches[i].flow_ids,
            batches[i].prediction_at,
            prep.lead_lag_memberships,
            prep.point_predictions,
            differentiable=True,
        )
        # Empate total en las etiquetas maduras.
        matured = [(flow, START + i * DAY, 0.01) for flow in batches[i].flow_ids]
        ll = predictor.settle_lead_lag_memory(ll, matured, differentiable=True, train=True)
    p2 = predictor.prepare(
        batches[2], state, differentiable=True, session_complete=True, lead_lag_memory=ll
    )
    loss = (p2.point_predictions - 0.01).abs().mean()
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
    """Las pertenencias pendientes se reúnen en una selección por emisión, sin crecer con N."""
    sizes = {}
    for rows in (8, 64):
        layer = active_layer()
        flows = [f"US/A{index:03d}" for index in range(rows)]
        state = layer.initial_state("p")
        for step in range(2):
            when = [START + step * DAY] * rows
            probs = layer.memberships(encoded(rows=rows, seed=30 + step), ["US"] * rows, when)
            issued = torch.linspace(-1.0, 1.0, rows, dtype=probs.dtype)
            state = layer.stage(state, flows, when, probs, issued, differentiable=True)
            matured = [
                (flow, START + step * DAY, float(value))
                for flow, value in zip(
                    flows,
                    torch.randn(rows, generator=torch.Generator().manual_seed(90 + step)),
                    strict=True,
                )
            ]
            state = layer.settle(state, matured, differentiable=True)
        assert len(state.pending) == 0
        sizes[rows] = (
            _graph_nodes(state.markets["US"].delta_transition),
            _graph_nodes(state.markets["US"].trace),
        )
        (grad,) = torch.autograd.grad(
            state.markets["US"].delta_transition.square().sum(), layer.group_proj.weight
        )
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert sizes[8] == sizes[64], sizes


def test_detaching_compacts_pending_and_split_blocks_settle_identically():
    layer = active_layer()
    rows = 12
    flows = [f"US/A{index:03d}" for index in range(rows)]
    when = [START] * rows
    probs = layer.memberships(encoded(rows=rows), ["US"] * rows, when)
    issued = torch.linspace(-1.0, 1.0, rows, dtype=probs.dtype)
    state = layer.stage(layer.initial_state("p"), flows, when, probs, issued, differentiable=True)
    matured = [(flow, START, 0.01 * index) for index, flow in enumerate(flows[:9])]
    partial = layer.detach_state(layer.settle(state, matured, differentiable=True))
    assert sorted(flow for flow, _ in partial.pending) == flows[9:]
    for (flow, _), (source, emitted, row) in partial.pending.items():
        position = flows.index(flow)
        assert source.shape == (3, probs.shape[1]) and not source.requires_grad
        assert emitted.shape == (3,) and not emitted.requires_grad
        assert torch.equal(source[row], probs.detach()[position])
        assert torch.equal(emitted[row], issued[position])
    exported = layer.export_state(partial)["pending"]
    assert exported["flows"] == flows[9:]
    assert torch.equal(exported["memberships"], probs.detach()[9:])
    restored = layer.restore_state(layer.export_state(partial), "p", device="cpu")
    again = layer.export_state(restored)["pending"]
    assert again["flows"] == flows[9:] and torch.equal(again["memberships"], probs.detach()[9:])

    # Sesión registrada en dos bloques desordenados coincide bit a bit con una emisión entera.
    matured_all = [(flow, START, 0.01 * (i % 4) - 0.02) for i, flow in enumerate(flows)]
    whole = layer.settle(
        layer.stage(layer.initial_state("p"), flows, when, probs.detach(), issued), matured_all
    )
    first, second = [1, 4, 7, 8, 9, 11], [0, 2, 3, 5, 6, 10]
    split = layer.initial_state("p")
    for block in (first, second):
        split = layer.stage(
            split,
            [flows[i] for i in block],
            [START] * len(block),
            probs.detach()[block],
            issued[block],
        )
    split = layer.settle(split, list(reversed(matured_all)))
    assert torch.equal(split.markets["US"].trace, whole.markets["US"].trace)
    assert torch.equal(split.markets["US"].delta_transition, whole.markets["US"].delta_transition)


def test_export_restore_detach_and_state_dict_contracts():
    predictor = model(lead_lag_memory=VARIANT)
    with torch.no_grad():
        predictor.lead_lag_memory.gate.fill_(0.08)
    predictor._seal_parameters()
    batches = [session_batch(5, markets=["US"] * 5, at=START + i * DAY) for i in range(2)]
    state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
    ll = predictor.initial_lead_lag_memory()
    for i, batch in enumerate(batches):
        prep = predictor.prepare(
            batch, state, differentiable=True, session_complete=True, lead_lag_memory=ll
        )
        state = prep.next_state
        ll = predictor.stage_lead_lag_memory(
            ll,
            batch.flow_ids,
            batch.prediction_at,
            prep.lead_lag_memberships,
            prep.point_predictions,
            differentiable=True,
        )
        ll = predictor.settle_lead_lag_memory(
            ll,
            [(f, batch.prediction_at[0], 0.01 * (j + i)) for j, f in enumerate(batch.flow_ids)],
            differentiable=True,
            train=True,
        )
    assert ll.markets["US"].delta_transition.requires_grad and ll.markets["US"].trace.requires_grad
    detached = predictor.detach_lead_lag_memory(ll)
    assert not detached.markets["US"].delta_transition.requires_grad
    payload = predictor.export_lead_lag_memory(detached)
    restored = predictor.restore_lead_lag_memory(payload)
    assert torch.equal(
        restored.markets["US"].delta_transition, detached.markets["US"].delta_transition
    )
    assert torch.equal(restored.markets["US"].trace, detached.markets["US"].trace)
    assert restored.markets["US"].sessions == detached.markets["US"].sessions == 2
    assert restored.markets["US"].updates == detached.markets["US"].updates == 1

    twin = model(lead_lag_memory=VARIANT)
    twin.load_state_dict(predictor.state_dict())
    with pytest.raises(ValueError):
        model(lead_lag_memory=SHUFFLED).load_state_dict(predictor.state_dict())
    with pytest.raises(ValueError):
        model(lead_lag_memory=STATIC_PRIOR).load_state_dict(predictor.state_dict())


def test_coexistence_with_session_hologram_and_crrm_preserves_order_before_head():
    crrm_cfg = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5,), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )
    holo_cfg = SessionHologramConfig(
        bands=2, phase_dim=4, mode="interferometric", position="after_mac"
    )
    predictor = model(
        session_hologram=holo_cfg,
        cross_sectional_rank_memory=crrm_cfg,
        lead_lag_memory=VARIANT,
    )
    with torch.no_grad():
        predictor.session_hologram.gate.fill_(0.05)
        predictor.cross_sectional_rank_memory.out.weight.fill_(0.05)
        predictor.lead_lag_memory.gate.fill_(0.05)
    predictor._seal_parameters()

    batches = [session_batch(5, markets=["US"] * 5, at=START + i * DAY) for i in range(2)]
    state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
    rm = predictor.initial_rank_memory()
    ll = predictor.initial_lead_lag_memory()
    for _i, batch in enumerate(batches):
        prep = predictor.prepare(
            batch,
            state,
            differentiable=True,
            session_complete=True,
            rank_memory=rm,
            lead_lag_memory=ll,
        )
        assert torch.isfinite(prep.point_predictions).all()
        assert prep.rank_memory_keys is not None and prep.lead_lag_memberships is not None
        state = prep.next_state
        rm = predictor.stage_rank_memory(
            rm,
            batch.flow_ids,
            batch.prediction_at,
            prep.rank_memory_keys,
            prep.point_predictions,
            differentiable=True,
        )
        ll = predictor.stage_lead_lag_memory(
            ll,
            batch.flow_ids,
            batch.prediction_at,
            prep.lead_lag_memberships,
            prep.point_predictions,
            differentiable=True,
        )
        matured = [
            (f, batch.prediction_at[0], 0.01 * (j - 2)) for j, f in enumerate(batch.flow_ids)
        ]
        rm = predictor.settle_rank_memory(rm, matured, differentiable=True, train=True)
        ll = predictor.settle_lead_lag_memory(ll, matured, differentiable=True, train=True)
    # Comprobar que LL-FM se aplica después de CRRM (su residuo es el último antes de head).
    b_next = session_batch(5, markets=["US"] * 5, at=START + 2 * DAY)
    prep_both = predictor.prepare(
        b_next,
        state,
        differentiable=False,
        session_complete=True,
        rank_memory=rm,
        lead_lag_memory=ll,
    )
    with torch.no_grad():
        saved_gate = predictor.lead_lag_memory.gate.clone()
        predictor.lead_lag_memory.gate.zero_()
    predictor._seal_parameters()
    state_zero = replace(state, parameter_id=predictor._parameter_id)
    rm_zero = predictor.detach_rank_memory(rm)
    ll_zero = predictor.detach_lead_lag_memory(ll)
    prep_without_ll = predictor.prepare(
        b_next,
        state_zero,
        differentiable=False,
        session_complete=True,
        rank_memory=rm_zero,
        lead_lag_memory=ll_zero,
    )
    with torch.no_grad():
        predictor.lead_lag_memory.gate.copy_(saved_gate)
    predictor._seal_parameters()
    markets = tuple(f.split("/", 1)[0] for f in b_next.flow_ids)
    ll_probs = predictor.lead_lag_memory.memberships(
        prep_without_ll.working_state, markets, b_next.prediction_at
    )
    expected_working = prep_without_ll.working_state + predictor.lead_lag_memory.read(
        prep_without_ll.working_state,
        ll_probs,
        markets,
        b_next.prediction_at,
        predictor.detach_lead_lag_memory(ll),
    )
    torch.testing.assert_close(prep_both.working_state, expected_working, rtol=1e-12, atol=1e-13)


def test_registry_declared_sources_frozen_consumer_and_invalid_declarations():
    raw = asdict(VARIANT)
    assert checked_predictor_changes(dict(lead_lag_memory=raw)) == dict(lead_lag_memory=raw)
    base, batch = setup()
    cfg = replace(base.config, lead_lag_memory=json.loads(json.dumps(raw)))
    assert cfg.lead_lag_memory == VARIANT
    assert declared_sources(base.config) == ()
    assert declared_sources(cfg) == ("mars_titan.models.titans.lead_lag_memory",)

    # El consumidor congelado rechaza LL-FM explícitamente.
    predictor = model(lead_lag_memory=VARIANT).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="desfase"):
        FrozenFinancialConsumer(predictor)

    # Sin declarar LL-FM no admite lead_lag_memory; declarándola la exige junto a sesión completa.
    with pytest.raises(ValueError, match="no declara"):
        base.prepare(batch, base.initial_state(batch.flow_ids), lead_lag_memory=object())
    active = model(lead_lag_memory=VARIANT)
    with pytest.raises(ValueError, match="sesión completa"):
        active.prepare(
            batch,
            active.initial_state(batch.flow_ids),
            lead_lag_memory=active.initial_lead_lag_memory(),
        )
    with pytest.raises(ValueError, match="desfase"):
        active.prepare(batch, active.initial_state(batch.flow_ids), session_complete=True)

    # Control local C no se combina con LL-FM.
    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(
            cfg,
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )

    # Variantes distintas de mac_online y declaraciones malformadas se rechazan.
    for variant in ("mac_frozen", "mac_disabled", "transformer_direct"):
        with pytest.raises(ValueError, match="mac_online"):
            replace(base.config, variant=variant, lead_lag_memory=VARIANT)

    for bad in (
        dict(raw, groups=1),
        dict(raw, groups=65),
        dict(raw, groups=True),
        dict(raw, lag_scales=[]),
        dict(raw, lag_scales=[5, 5]),
        dict(raw, lag_scales=[5, 1]),
        dict(raw, lag_scales=[1, 2, 3, 4, 5]),
        dict(raw, lag_scales=[0]),
        dict(raw, decay=0.0),
        dict(raw, decay=0.9),
        dict(raw, learning_rate=0.0),
        dict(raw, learning_rate=1.0),
        dict(raw, damping=0.0),
        dict(raw, minimum_rows=2),
        dict(raw, mode="unknown"),
        dict(raw, seed=-1),
        dict(groups=16),
        "lead_lag_fast_memory_v1",
    ):
        with pytest.raises(ValueError):
            parse_lead_lag_memory(bad)

    # Autocast se rechaza.
    layer = LeadLagMemory(16, VARIANT, seed=1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer.memberships(encoded(4, dtype=torch.float32), ("US",) * 4, (START,) * 4)

    # ChronologicalTrainer._code() solo incluye el módulo cuando lead_lag_memory está activo.
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    tr = object.__new__(ChronologicalTrainer)
    tr.recipe = ChronologicalRecipe()
    mod = "mars_titan.models.titans.lead_lag_memory"
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
                lead_lag_memory=val,
            )
        )
        assert (mod in tr._code()) is present


def test_campaign_v12_screen3_accepts_single_variable_lead_lag_declarations():
    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    base_cfg = dict(
        groups=16,
        lag_scales=[1, 5],
        decay=0.05,
        learning_rate=0.2,
        damping=1.0,
        minimum_rows=8,
        seed=0,
    )
    additions = {
        "titans_mac_llfm": dict(base_cfg, mode="lead_lag"),
        "titans_mac_llfm_shuffled": dict(base_cfg, mode="shuffled_targets"),
        "titans_mac_llfm_static": dict(base_cfg, mode="static_prior"),
        "titans_mac_llfm_eval_frozen": dict(base_cfg, mode="eval_frozen"),
    }
    for name, cfg in additions.items():
        doc["models"][name] = dict(
            doc["models"]["titans_mac"],
            tier="s3_inventions",
            predictor_changes=dict(lead_lag_memory=cfg),
        )
    validated = comparison_campaign.checked_declaration(doc, root=root)
    for name in additions:
        assert name in validated["models"]

    # Verificar que en la identidad del predictor solo difiere el campo `lead_lag_memory`.
    recipe = json.loads((root / doc["recipes"]["core"]).read_text())
    opts = dict(recipe["predictor"])
    opts.pop("dtype", None)
    base_spec = specification()
    spec = FinancialInputSpec(
        source_sha256=base_spec.source_sha256,
        view_sha256=base_spec.view_sha256,
        representation=json.loads(base_spec._representation_json),
        dimensions=dict(base_spec.dimensions, charts=46),
        input_policy=base_spec.input_policy,
    )
    control_id = FinancialConfig(spec, variant="mac_online", **opts).identity()
    cand_id = FinancialConfig(
        spec,
        variant="mac_online",
        **opts,
        lead_lag_memory=additions["titans_mac_llfm"],
    ).identity()
    assert set(cand_id) - set(control_id) == {"lead_lag_memory"}
    assert {k: v for k, v in cand_id.items() if k != "lead_lag_memory"} == control_id
