"""Consolidación sináptica de tres factores de rango bajo (D3F-LR): contratos y numérica."""

import json
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
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm, standardized
from mars_titan.models.titans.session_ranks import gaussian_ranks
from mars_titan.models.titans.technical_branch import TechnicalDenseConfig
from mars_titan.models.titans.three_factor_consolidation import (
    CREDIT_SCALE_FLOOR,
    CROSS_ONLY,
    DECAY_MAX,
    DECAY_MIN,
    EVAL_FROZEN,
    KEY_NORM_EPS,
    LOCAL_ONLY,
    MASS_EPS,
    READOUT_NORM_EPS,
    SHUFFLED_CREDIT,
    THREE_FACTOR,
    THREE_FACTOR_CONSOLIDATION,
    ThreeFactorConsolidation,
    ThreeFactorConsolidationConfig,
    parse_three_factor_consolidation,
)
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.search_cases import checked_predictor_changes

START, DAY = 1_609_459_200_000_000, 86_400_000_000
VARIANT = ThreeFactorConsolidationConfig(
    dim=8,
    trace_tau=5.0,
    consolidation_tau=21.0,
    credit_scale=1.0,
    max_gain=0.25,
    minimum_rows=3,
    mode=THREE_FACTOR,
    seed=0,
)
CROSS_CFG = replace(VARIANT, mode=CROSS_ONLY)
LOCAL_CFG = replace(VARIANT, mode=LOCAL_ONLY)
SHUFFLED_CFG = replace(VARIANT, mode=SHUFFLED_CREDIT)
EVAL_FROZEN_CFG = replace(VARIANT, mode=EVAL_FROZEN)
ALL_CONFIGS = (VARIANT, CROSS_CFG, LOCAL_CFG, SHUFFLED_CFG, EVAL_FROZEN_CFG)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def active_layer(config=VARIANT, *, hidden=16, seed=7, dtype=torch.float64):
    """Módulo con puerta activa (no nula) para observar la consolidación y lectura."""
    layer = ThreeFactorConsolidation(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 19)
        layer.gate.copy_(torch.randn(hidden, generator=generator, dtype=dtype) * 0.4)
    return layer


def encoded(rows=6, hidden=16, seed=11, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=dtype)


def _rms(values, weight):
    scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + READOUT_NORM_EPS)
    return values * scale * weight


def reference_step(layer, sessions, query_encoded, query_flows, query_at):
    """Referencia explícita de las ecuaciones D3F-LR sobre sesiones maduras y consulta final."""
    cfg = layer.config
    dim = cfg.dim
    dtype, device = query_encoded.dtype, query_encoded.device
    gamma_e = DECAY_MIN + (DECAY_MAX - DECAY_MIN) * torch.sigmoid(layer.trace_decay_logit)
    alpha_w = DECAY_MIN + (DECAY_MAX - DECAY_MIN) * torch.sigmoid(layer.cross_decay_logit)
    alpha_c = DECAY_MIN + (DECAY_MAX - DECAY_MIN) * torch.sigmoid(layer.local_decay_logit)
    beta = CREDIT_SCALE_FLOOR + torch.nn.functional.softplus(layer.credit_scale_logit)

    def kv(values, markets, instants):
        normed = layer.norm(values.detach())
        raw_k = layer.key_proj(normed)
        raw_v = layer.val_proj(normed)
        groups = {}
        for row, key in enumerate(zip(markets, instants, strict=True)):
            groups.setdefault(key, []).append(row)
        out_k = torch.empty_like(raw_k)
        out_v = torch.empty_like(raw_v)
        for rows in groups.values():
            idx = torch.tensor(rows, dtype=torch.int64, device=device)
            pk, pv = raw_k.index_select(0, idx), raw_v.index_select(0, idx)
            if len(rows) > 1:
                pk = pk - pk.mean(dim=0, keepdim=True)
                pv = pv - pv.mean(dim=0, keepdim=True)
            out_k[idx] = pk / torch.sqrt(pk.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
            out_v[idx] = pv / torch.sqrt(pv.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
        return out_k, out_v

    flow_a, flow_b, flow_c, flow_u = {}, {}, {}, {}
    market_w, market_mass = {}, {}

    for enc, flows, at, targets, issued in sessions:
        markets = tuple(f.split("/", 1)[0] for f in flows)
        k_hat, v_hat = kv(enc, markets, (at,) * len(flows))
        emitted_k, emitted_a, emitted_b, emitted_p = {}, {}, {}, {}
        for i, flow in enumerate(flows):
            prev_a = flow_a.get(flow, torch.zeros(dim, dtype=dtype, device=device))
            prev_b = flow_b.get(flow, torch.zeros(dim, dtype=dtype, device=device))
            next_a = (1.0 - gamma_e) * prev_a + gamma_e * k_hat[i]
            next_b = (1.0 - gamma_e) * prev_b + gamma_e * v_hat[i]
            flow_a[flow] = next_a
            flow_b[flow] = next_b
            emitted_k[flow] = k_hat[i]
            emitted_a[flow] = next_a
            emitted_b[flow] = next_b
            emitted_p[flow] = issued[i]

        by_market = {}
        for i, flow in enumerate(flows):
            by_market.setdefault(flow.split("/", 1)[0], []).append((flow, targets[i]))
        for market in sorted(by_market):
            ordered = sorted(by_market[market], key=lambda item: item[0])
            if len(ordered) < cfg.minimum_rows:
                continue
            y_vec = torch.tensor([y for _, y in ordered], dtype=dtype, device=device)
            p_vec = torch.tensor([emitted_p[f] for f, _ in ordered], dtype=dtype, device=device)
            g_vec = gaussian_ranks(
                y_vec.unsqueeze(-1), torch.ones((len(ordered), 1), dtype=torch.bool)
            ).squeeze(-1)
            prod = standardized(g_vec) * standardized(p_vec)
            delta_ic = prod - prod.mean()
            third = torch.tanh(beta * delta_ic)
            a_mat = torch.stack([emitted_a[f] for f, _ in ordered])
            b_mat = torch.stack([emitted_b[f] for f, _ in ordered])
            k_mat = torch.stack([emitted_k[f] for f, _ in ordered])
            dw = (a_mat.T @ (third.unsqueeze(-1) * b_mat)) / float(len(ordered))
            prev_w = market_w.get(market, torch.zeros((dim, dim), dtype=dtype, device=device))
            prev_m = market_mass.get(market, torch.zeros((), dtype=dtype, device=device))
            market_w[market] = (1.0 - alpha_w) * prev_w + alpha_w * dw
            market_mass[market] = (1.0 - alpha_w) * prev_m + alpha_w
            for idx, (flow, _) in enumerate(ordered):
                psi = (a_mat[idx] * k_mat[idx]).sum() * b_mat[idx]
                prev_c = flow_c.get(flow, torch.zeros(dim, dtype=dtype, device=device))
                prev_u = flow_u.get(flow, torch.zeros((), dtype=dtype, device=device))
                flow_c[flow] = (1.0 - alpha_c) * prev_c + alpha_c * third[idx] * psi
                flow_u[flow] = (1.0 - alpha_c) * prev_u + alpha_c

    q_markets = tuple(f.split("/", 1)[0] for f in query_flows)
    qk, qv = kv(query_encoded, q_markets, query_at)
    groups = {}
    for row, key in enumerate(zip(q_markets, query_at, strict=True)):
        groups.setdefault(key, []).append(row)

    z_cross = torch.zeros((len(query_flows), dim + 1), dtype=dtype, device=device)
    for (market, _), rows in groups.items():
        if market not in market_w:
            continue
        idx = torch.tensor(rows, dtype=torch.int64, device=device)
        w_hat = market_w[market] / (market_mass[market] + MASS_EPS)
        r_cross = qk[idx] @ w_hat
        s_cross = (qv[idx] * r_cross).sum(dim=-1, keepdim=True)
        part = torch.cat((r_cross, s_cross), dim=-1)
        if len(rows) > 1:
            part = part - part.mean(dim=0, keepdim=True)
        z_cross[idx] = market_mass[market] * _rms(part, layer.cross_norm_weight)

    z_local = torch.zeros((len(query_flows), dim + 1), dtype=dtype, device=device)
    for row, flow in enumerate(query_flows):
        c_prev = flow_c.get(flow, torch.zeros(dim, dtype=dtype, device=device))
        u_prev = flow_u.get(flow, torch.zeros((), dtype=dtype, device=device))
        c_hat = c_prev / (u_prev + MASS_EPS)
        s_local = (qv[row] * c_hat).sum().unsqueeze(0)
        z_local[row] = u_prev * _rms(torch.cat((c_hat, s_local)), layer.local_norm_weight)
    for rows in groups.values():
        if len(rows) > 1:
            idx = torch.tensor(rows, dtype=torch.int64, device=device)
            z_local[idx] = z_local[idx] - z_local[idx].mean(dim=0, keepdim=True)

    if cfg.mode == LOCAL_ONLY:
        z_cross = torch.zeros_like(z_cross)
    elif cfg.mode == CROSS_ONLY:
        z_local = torch.zeros_like(z_local)
    features = torch.cat((z_cross, z_local), dim=-1)
    gain = cfg.max_gain * torch.tanh(layer.gate)
    return gain * layer.out(features)


@pytest.mark.parametrize("config", [VARIANT, CROSS_CFG, LOCAL_CFG])
def test_read_and_settle_match_the_three_factor_low_rank_equations(config):
    layer = active_layer(config)
    state = layer.initial_state("p")
    flows_1 = ("US/AAA", "US/BBB", "US/CCC", "US/DDD", "CN/600000.SS", "CN/600001.SS")
    enc_1 = encoded(len(flows_1), seed=21)
    issued_1 = [0.2, -0.1, 0.5, -0.4, 0.1, -0.2]
    targets_1 = [0.03, -0.02, 0.01, -0.05, 0.02, -0.01]
    markets_1 = tuple(f.split("/", 1)[0] for f in flows_1)

    delta_0, em_1 = layer.read_and_trace(enc_1, flows_1, markets_1, (START,) * len(flows_1), state)
    # Antes de madurar ninguna etiqueta, la salida es exactamente cero.
    assert torch.count_nonzero(delta_0) == 0
    state = layer.stage(
        state,
        flows_1,
        (START,) * len(flows_1),
        em_1,
        torch.tensor(issued_1, dtype=torch.float64),
        record_pending=True,
        differentiable=True,
    )
    matured_1 = [(f, START, y) for f, y in zip(flows_1, targets_1, strict=True)]
    state = layer.settle(state, matured_1, differentiable=True, train=True)
    # CN tuvo 2 filas (< minimum_rows=3): no consolida CN pero vacía su cola pendiente.
    assert "US" in state.markets and "CN" not in state.markets
    assert len(state.pending) == 0

    flows_2 = ("US/DDD", "US/AAA", "US/CCC", "US/BBB")
    enc_2 = encoded(len(flows_2), seed=31)
    issued_2 = [-0.3, 0.4, 0.0, 0.1]
    targets_2 = [-0.01, 0.04, 0.02, -0.03]
    markets_2 = ("US",) * len(flows_2)
    _, em_2 = layer.read_and_trace(enc_2, flows_2, markets_2, (START + DAY,) * len(flows_2), state)
    state = layer.stage(
        state,
        flows_2,
        (START + DAY,) * len(flows_2),
        em_2,
        torch.tensor(issued_2, dtype=torch.float64),
        record_pending=True,
        differentiable=True,
    )
    matured_2 = [(f, START + DAY, y) for f, y in zip(flows_2, targets_2, strict=True)]
    state = layer.settle(state, matured_2, differentiable=True, train=True)

    # Incluye un activo nuevo en US (US/NEW) para verificar lectura cruzada sin historia local
    # y activos de CN sin maduraciones previas (deben leer cero exacto).
    q_flows = ("US/AAA", "CN/600000.SS", "US/NEW", "US/BBB", "CN/600001.SS")
    q_markets = tuple(f.split("/", 1)[0] for f in q_flows)
    q_enc = encoded(len(q_flows), seed=41)
    q_at = (START + 2 * DAY,) * len(q_flows)
    actual, _ = layer.read_and_trace(q_enc, q_flows, q_markets, q_at, state)
    expected = reference_step(
        layer,
        [
            (enc_1, flows_1, START, targets_1, issued_1),
            (enc_2, flows_2, START + DAY, targets_2, issued_2),
        ],
        q_enc,
        q_flows,
        q_at,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    assert torch.count_nonzero(actual[[1, 4]]) == 0
    assert torch.count_nonzero(actual[[0, 2, 3]]) > 0


def test_cross_asset_synapse_distinguishes_positive_vs_negative_ic_credit_regimes():
    """El tercer factor consolida la congruencia de IC (g̃·p̃ − ρ), no el rango bruto g."""
    layer = active_layer(CROSS_CFG)
    flows = ("US/A", "US/B", "US/C", "US/D", "US/E", "US/F")
    markets = ("US",) * len(flows)
    enc = encoded(len(flows), seed=55)
    targets = [0.05, 0.03, 0.01, -0.01, -0.03, -0.05]
    issued = [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]

    def run_with(t_vals, p_vals):
        st = layer.initial_state("p")
        _, em = layer.read_and_trace(enc, flows, markets, (START,) * len(flows), st)
        st = layer.stage(
            st,
            flows,
            (START,) * len(flows),
            em,
            torch.tensor(p_vals, dtype=torch.float64),
            record_pending=True,
            differentiable=False,
        )
        matured = [(f, START, y) for f, y in zip(flows, t_vals, strict=True)]
        return layer.settle(st, matured, differentiable=False, train=True)

    aligned = run_with(targets, issued)
    # Invertir simultáneamente objetivo y predicción conserva intacto el crédito marginal de IC!
    both_flipped = run_with([-y for y in targets], [-p for p in issued])
    # Invertir solo la predicción invierte exactamente el signo del crédito marginal de IC!
    anti_aligned = run_with(targets, [-p for p in issued])

    torch.testing.assert_close(
        both_flipped.markets["US"].synapse,
        aligned.markets["US"].synapse,
        rtol=1e-12,
        atol=1e-13,
    )
    torch.testing.assert_close(
        anti_aligned.markets["US"].synapse,
        -aligned.markets["US"].synapse,
        rtol=1e-12,
        atol=1e-13,
    )
    assert aligned.markets["US"].synapse.abs().sum() > 0


def test_markets_stay_isolated_and_row_order_does_not_change_consolidated_state():
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
        _, em = layer.read_and_trace(o_enc, o_flows, o_markets, (START,) * len(order), st)
        o_issued = torch.tensor([issued[i] for i in order], dtype=torch.float64)
        st = layer.stage(
            st,
            o_flows,
            (START,) * len(order),
            em,
            o_issued,
            record_pending=True,
            differentiable=False,
        )
        matured = [(flows[i], START, targets[i]) for i in reversed(order)]
        return layer.settle(st, matured, differentiable=False, train=True)

    base = run_order(list(range(len(flows))))
    perm = run_order([4, 2, 6, 0, 5, 1, 3])
    for market in ("US", "CN"):
        torch.testing.assert_close(
            perm.markets[market].synapse, base.markets[market].synapse, rtol=1e-13, atol=1e-14
        )
        torch.testing.assert_close(
            perm.markets[market].mass, base.markets[market].mass, rtol=0, atol=0
        )
    q_enc = encoded(len(flows), seed=9)
    base_out, _ = layer.read_and_trace(q_enc, flows, markets, (START + DAY,) * len(flows), base)
    perm_out, _ = layer.read_and_trace(q_enc, flows, markets, (START + DAY,) * len(flows), perm)
    torch.testing.assert_close(perm_out, base_out, rtol=1e-12, atol=1e-13)

    # Alterar objetivos de CN no cambia en un solo bit el estado ni la lectura de US.
    st_alt = layer.initial_state("p")
    _, em = layer.read_and_trace(enc, flows, markets, (START,) * len(flows), st_alt)
    st_alt = layer.stage(
        st_alt,
        flows,
        (START,) * len(flows),
        em,
        torch.tensor(issued, dtype=torch.float64),
        record_pending=True,
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
    assert torch.equal(st_alt.markets["US"].synapse, base.markets["US"].synapse)
    assert not torch.equal(st_alt.markets["CN"].synapse, base.markets["CN"].synapse)
    alt_out, _ = layer.read_and_trace(q_enc, flows, markets, (START + DAY,) * len(flows), st_alt)
    us_rows = [i for i, m in enumerate(markets) if m == "US"]
    cn_rows = [i for i, m in enumerate(markets) if m == "CN"]
    assert torch.equal(alt_out[us_rows], base_out[us_rows])
    assert not torch.equal(alt_out[cn_rows], base_out[cn_rows])


def test_placebo_and_ablation_modes_share_exact_parameters_and_isolate_mechanisms():
    layers = [active_layer(cfg) for cfg in ALL_CONFIGS]
    ref_sd = layers[0].state_dict()
    for other in layers[1:]:
        sd = other.state_dict()
        assert sd.keys() == ref_sd.keys()
        for name, tensor in ref_sd.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, sd[name]), (other.config.mode, name)

    flows = tuple(f"US/S{i}" for i in range(8))
    markets = ("US",) * 8
    enc_1 = encoded(8, seed=19)
    enc_2 = encoded(8, seed=29)
    targets = [0.01 * (i - 3.5) for i in range(8)]
    issued = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64)
    matured = [(f, START, y) for f, y in zip(flows, targets, strict=True)]

    def settle_and_read(layer, *, train=True):
        st = layer.initial_state("p")
        _, em = layer.read_and_trace(enc_1, flows, markets, (START,) * 8, st)
        st = layer.stage(st, flows, (START,) * 8, em, issued, record_pending=True)
        st = layer.settle(st, matured, train=train)
        out, _ = layer.read_and_trace(enc_2, flows, markets, (START + DAY,) * 8, st)
        return st, out

    st_real, out_real = settle_and_read(layers[0])
    _, out_cross = settle_and_read(layers[1])
    _, out_local = settle_and_read(layers[2])
    st_shuf, out_shuf = settle_and_read(layers[3])
    st_frozen_eval, out_frozen_eval = settle_and_read(layers[4], train=False)

    # El placebo barajado conserva las mismas masas y trazas, pero rompe la sinapsis consolidada.
    assert torch.equal(st_shuf.markets["US"].mass, st_real.markets["US"].mass)
    assert not torch.equal(st_shuf.markets["US"].synapse, st_real.markets["US"].synapse)
    assert not torch.equal(out_shuf, out_real)
    # La descomposición aditiva cross + local iguala exactamente a three_factor (W_out lineal).
    torch.testing.assert_close(out_cross + out_local, out_real, rtol=1e-12, atol=1e-13)
    assert out_cross.abs().sum() > 0 and out_local.abs().sum() > 0
    # eval_frozen no consolida en train=False y emite exactamente cero.
    assert "US" not in st_frozen_eval.markets and len(st_frozen_eval.pending) == 0
    assert torch.count_nonzero(out_frozen_eval) == 0


def test_disabled_field_keeps_identity_parameters_and_predictions_bit_for_bit():
    base, batch = setup()
    explicit = model(three_factor_consolidation=None)
    assert explicit.config.identity() == base.config.identity()
    assert "three_factor_consolidation" not in base.config.identity()
    assert base.three_factor_consolidation is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert actual.three_factor_emission is None


@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_all_modes_start_as_the_reference_without_touching_backbone_weights(config):
    base, _ = setup()
    batch = session_batch(6, markets=["US"] * 6)
    predictor = model(three_factor_consolidation=config)
    ref_params = dict(base.named_parameters())
    new_params = dict(predictor.named_parameters())
    assert set(new_params) - set(ref_params) == {
        "three_factor_consolidation.norm.weight",
        "three_factor_consolidation.norm.bias",
        "three_factor_consolidation.key_proj.weight",
        "three_factor_consolidation.val_proj.weight",
        "three_factor_consolidation.cross_norm_weight",
        "three_factor_consolidation.local_norm_weight",
        "three_factor_consolidation.out.weight",
        "three_factor_consolidation.gate",
        "three_factor_consolidation.trace_decay_logit",
        "three_factor_consolidation.cross_decay_logit",
        "three_factor_consolidation.local_decay_logit",
        "three_factor_consolidation.credit_scale_logit",
    }
    for name, value in ref_params.items():
        assert torch.equal(new_params[name], value), name
    # Un solo cero en gate sobre W_out Kaiming no nula: sin doble cero en serie.
    assert torch.count_nonzero(predictor.three_factor_consolidation.gate) == 0
    assert torch.count_nonzero(predictor.three_factor_consolidation.out.weight) > 0
    tf_state = predictor.initial_three_factor_state()
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch,
        predictor.initial_state(batch.flow_ids),
        session_complete=True,
        three_factor_state=tf_state,
    )
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    assert actual.three_factor_emission is not None
    assert actual.three_factor_emission[0].shape == (6, config.dim)
    identity = predictor.config.identity()["three_factor_consolidation"]
    assert identity["kind"] == THREE_FACTOR_CONSOLIDATION and identity["mode"] == config.mode


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
def test_first_step_gate_gradient_all_param_meta_gradients_and_determinism(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        spec = specification()

        def run_once(zero_gate: bool, zero_median: bool):
            cfg = FinancialConfig(
                spec,
                variant="mac_online",
                hidden_size=32,
                head=QUANTILE_HEAD,
                learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=zero_median),
                median_gain=MEDIAN_GAIN,
                technical_branch=TechnicalDenseConfig(
                    columns=2, hidden=16, depth=2, session_ranks=True
                ),
                gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
                memory_residual_layer_norm=True,
                three_factor_consolidation=VARIANT,
            )
            predictor = FinancialPredictor(cfg, dtype=dtype)
            predictor.set_learning_statistics(statistics(cfg.learning_scale))
            if not zero_gate:
                with torch.no_grad():
                    predictor.three_factor_consolidation.gate.fill_(0.2)
                predictor._seal_parameters()
            batches = [
                make_session_batch(6, at=START + i * DAY, seed=101 + i, dtype=dtype)
                for i in range(2)
            ]
            state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
            tf = predictor.initial_three_factor_state()
            p0 = predictor.prepare(
                batches[0], state, differentiable=True, session_complete=True, three_factor_state=tf
            )
            if zero_median:
                assert torch.count_nonzero(p0.point_predictions) == 0
            tf = predictor.stage_three_factor_state(
                tf,
                batches[0].flow_ids,
                batches[0].prediction_at,
                p0.three_factor_emission,
                p0.point_predictions,
                record_pending=True,
                differentiable=True,
            )
            y0 = [0.02, -0.01, 0.03, -0.04, 0.01, -0.02]
            matured = [
                (flow, START, val) for flow, val in zip(batches[0].flow_ids, y0, strict=True)
            ]
            tf = predictor.settle_three_factor_state(tf, matured, differentiable=True, train=True)
            p1 = predictor.prepare(
                batches[1],
                p0.next_state,
                differentiable=True,
                session_complete=True,
                three_factor_state=tf,
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

        # 1) Predicción constante (zero_median=True): gradientes finitos y deterministas.
        q_z1, g_z1 = run_once(zero_gate=True, zero_median=True)
        q_z2, g_z2 = run_once(zero_gate=True, zero_median=True)
        assert torch.equal(q_z1, q_z2)
        for name, grad in g_z1.items():
            assert torch.isfinite(grad).all(), name
            assert torch.equal(grad, g_z2[name]), name

        # 2) Predicción no constante con gate=0: gate recibe gradiente != 0 ya en el primer paso!
        q_a, g_a = run_once(zero_gate=True, zero_median=False)
        q_b, g_b = run_once(zero_gate=True, zero_median=False)
        assert torch.equal(q_a, q_b)
        for name, grad in g_a.items():
            assert torch.isfinite(grad).all(), name
            assert torch.equal(grad, g_b[name]), name
        assert torch.count_nonzero(g_a["three_factor_consolidation.gate"]) > 0

        # 3) Con gate != 0: el metagradiente atraviesa stage->settle->read a todos los pesos.
        _, g_meta = run_once(zero_gate=False, zero_median=False)
        for name in (
            "three_factor_consolidation.gate",
            "three_factor_consolidation.out.weight",
            "three_factor_consolidation.cross_norm_weight",
            "three_factor_consolidation.local_norm_weight",
            "three_factor_consolidation.key_proj.weight",
            "three_factor_consolidation.val_proj.weight",
            "three_factor_consolidation.norm.weight",
            "three_factor_consolidation.norm.bias",
            "three_factor_consolidation.trace_decay_logit",
            "three_factor_consolidation.cross_decay_logit",
            "three_factor_consolidation.local_decay_logit",
            "three_factor_consolidation.credit_scale_logit",
        ):
            assert name in g_meta, name
            assert torch.isfinite(g_meta[name]).all() and torch.count_nonzero(g_meta[name]) > 0, (
                name
            )
    finally:
        torch.use_deterministic_algorithms(previous)


def test_backbone_input_is_shielded_from_delayed_credit_meta_gradients():
    """El tronco no recibe metagradientes ruidosos de etiquetas diferidas a través de D3F-LR."""
    layer = active_layer(VARIANT)
    flows = ("US/A", "US/B", "US/C", "US/D")
    markets = ("US",) * 4
    enc_1 = encoded(4, seed=71).requires_grad_(True)
    st = layer.initial_state("p")
    _, em_1 = layer.read_and_trace(enc_1, flows, markets, (START,) * 4, st)
    st = layer.stage(
        st,
        flows,
        (START,) * 4,
        em_1,
        torch.tensor([0.3, -0.2, 0.1, -0.4], dtype=torch.float64),
        record_pending=True,
        differentiable=True,
    )
    matured = [
        ("US/A", START, 0.04),
        ("US/B", START, -0.02),
        ("US/C", START, 0.01),
        ("US/D", START, -0.03),
    ]
    st = layer.settle(st, matured, differentiable=True, train=True)
    enc_2 = encoded(4, seed=83).requires_grad_(True)
    delta, _ = layer.read_and_trace(enc_2, flows, markets, (START + DAY,) * 4, st)
    delta.square().sum().backward()
    # Ningún gradiente entra hacia enc_1 ni enc_2 por la rama interna de D3F-LR.
    assert enc_1.grad is None
    assert enc_2.grad is None
    # Pero TODOS los parámetros propios de D3F-LR reciben gradiente finito y no nulo.
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all() and torch.count_nonzero(param.grad) > 0, name


def test_constant_targets_and_constant_zero_scalar_head_produce_finite_gradients():
    spec = specification()
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=True),
        three_factor_consolidation=VARIANT,
    )
    predictor = FinancialPredictor(cfg, dtype=torch.float64)
    predictor.set_learning_statistics(statistics(cfg.learning_scale))
    batches = [
        make_session_batch(5, at=START + i * DAY, seed=201 + i, dtype=torch.float64)
        for i in range(2)
    ]
    state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
    tf = predictor.initial_three_factor_state()
    p0 = predictor.prepare(
        batches[0], state, differentiable=True, session_complete=True, three_factor_state=tf
    )
    tf = predictor.stage_three_factor_state(
        tf,
        batches[0].flow_ids,
        batches[0].prediction_at,
        p0.three_factor_emission,
        p0.point_predictions,
        record_pending=True,
        differentiable=True,
    )
    matured = [(flow, START, 0.01) for flow in batches[0].flow_ids]
    tf = predictor.settle_three_factor_state(tf, matured, differentiable=True, train=True)
    p1 = predictor.prepare(
        batches[1], p0.next_state, differentiable=True, session_complete=True, three_factor_state=tf
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


def test_the_autograd_graph_does_not_grow_with_the_number_of_rows():
    """Ni el estado por flujo ni las emisiones pendientes crean nodos por fila."""
    sizes_synapse, sizes_read = {}, {}
    for rows in (8, 64):
        layer = active_layer()
        flows = tuple(f"US/A{index:03d}" for index in range(rows))
        markets = ("US",) * rows
        when = (START,) * rows
        st = layer.initial_state("p")
        _, em_1 = layer.read_and_trace(encoded(rows, seed=1), flows, markets, when, st)
        issued = torch.linspace(-1.0, 1.0, rows, dtype=torch.float64)
        st = layer.stage(st, flows, when, em_1, issued, record_pending=True, differentiable=True)
        matured = [
            (flow, START, float(val))
            for flow, val in zip(flows, torch.linspace(-0.04, 0.04, rows), strict=True)
        ]
        st = layer.settle(st, matured, differentiable=True, train=True)
        delta, _ = layer.read_and_trace(
            encoded(rows, seed=2), flows, markets, (START + DAY,) * rows, st
        )
        sizes_synapse[rows] = _graph_nodes(st.markets["US"].synapse)
        sizes_read[rows] = _graph_nodes(delta)
    assert sizes_synapse[8] == sizes_synapse[64], sizes_synapse
    assert sizes_read[8] == sizes_read[64], sizes_read


def test_detaching_compacts_flows_and_pending_and_roundtrips_bit_for_bit():
    layer = active_layer()
    rows = 12
    flows = tuple(f"US/A{index:03d}" for index in range(rows))
    markets = ("US",) * rows
    when = (START,) * rows
    st = layer.initial_state("p")
    _, em = layer.read_and_trace(encoded(rows, seed=3), flows, markets, when, st)
    issued = torch.linspace(-1.0, 1.0, rows, dtype=torch.float64)
    st = layer.stage(st, flows, when, em, issued, record_pending=True, differentiable=True)
    matured = [(flow, START, 0.01 * index - 0.04) for index, flow in enumerate(flows[:9])]
    settled = layer.settle(st, matured, differentiable=True, train=True)
    assert settled.markets["US"].synapse.requires_grad
    partial = layer.detach_state(settled)
    assert not partial.markets["US"].synapse.requires_grad
    assert sorted(flow for flow, _ in partial.pending) == list(flows[9:])
    for _, (k_src, a_src, b_src, p_src, _) in partial.pending.items():
        assert k_src.shape == (3, layer.config.dim) and not k_src.requires_grad
        assert a_src.shape == (3, layer.config.dim) and not a_src.requires_grad
        assert b_src.shape == (3, layer.config.dim) and not b_src.requires_grad
        assert p_src.shape == (3,) and not p_src.requires_grad
    payload = layer.export_state(partial)
    restored = layer.restore_state(payload, "p", device="cpu")
    q_enc = encoded(rows, seed=4)
    expected, _ = layer.read_and_trace(q_enc, flows, markets, (START + DAY,) * rows, partial)
    actual, _ = layer.read_and_trace(q_enc, flows, markets, (START + DAY,) * rows, restored)
    assert torch.equal(actual, expected)
    assert torch.equal(restored.markets["US"].synapse, partial.markets["US"].synapse)


def test_split_emission_blocks_settle_identical_to_single_block():
    layer = active_layer()
    rows = 10
    flows = tuple(f"US/A{index:03d}" for index in range(rows))
    markets = ("US",) * rows
    when = (START,) * rows
    st = layer.initial_state("p")
    _, em = layer.read_and_trace(encoded(rows, seed=6), flows, markets, when, st)
    k_all, a_all, b_all = (t.detach() for t in em)
    issued = torch.linspace(-1.0, 1.0, rows, dtype=torch.float64)
    matured = [(flow, START, 0.01 * (index % 5) - 0.02) for index, flow in enumerate(flows)]

    whole = layer.settle(
        layer.stage(st, flows, when, (k_all, a_all, b_all), issued, record_pending=True), matured
    )
    first, second = [1, 4, 7, 8, 9], [0, 2, 3, 5, 6]
    split = layer.initial_state("p")
    for block in (first, second):
        idx = torch.tensor(block, dtype=torch.int64)
        split = layer.stage(
            split,
            [flows[i] for i in block],
            [START] * len(block),
            (k_all[idx], a_all[idx], b_all[idx]),
            issued[idx],
            record_pending=True,
        )
    split = layer.settle(split, list(reversed(matured)))
    assert torch.equal(split.markets["US"].synapse, whole.markets["US"].synapse)
    q_enc = encoded(rows, seed=8)
    out_whole, _ = layer.read_and_trace(q_enc, flows, markets, (START + DAY,) * rows, whole)
    out_split, _ = layer.read_and_trace(q_enc, flows, markets, (START + DAY,) * rows, split)
    assert torch.equal(out_split, out_whole)


def test_eval_mode_compacts_stale_flow_blocks_without_changing_values():
    """En inferencia sin detach, muchos pasos con bajas de activos no acumulan bloques muertos."""
    layer = active_layer()
    st = layer.initial_state("p")
    for step in range(20):
        flows = tuple(f"US/F{step}_{i}" for i in range(4))
        markets = ("US",) * 4
        at = START + step * DAY
        _, em = layer.read_and_trace(encoded(4, seed=100 + step), flows, markets, (at,) * 4, st)
        st = layer.stage(
            st,
            flows,
            (at,) * 4,
            em,
            torch.zeros(4, dtype=torch.float64),
            record_pending=False,
            differentiable=False,
        )
    distinct_blocks = {id(entry[0]) for entry in st.flows.values()}
    assert len(distinct_blocks) <= 8


def test_warmup_advances_eligibility_traces_without_staging_pending_or_changing_output():
    """El calentamiento alimenta (a, b) sin encolar etiquetas ni emitir señal antes de madurar."""
    layer = active_layer()
    flows = ("US/A", "US/B", "US/C", "US/D")
    markets = ("US",) * 4
    enc_warm = encoded(4, seed=301)
    enc_dec = encoded(4, seed=302)
    issued = torch.tensor([0.4, -0.3, 0.2, -0.1], dtype=torch.float64)
    matured = [
        ("US/A", START + DAY, 0.03),
        ("US/B", START + DAY, -0.02),
        ("US/C", START + DAY, 0.01),
        ("US/D", START + DAY, -0.04),
    ]

    # Sin calentamiento
    cold = layer.initial_state("p")
    d_cold, em_cold = layer.read_and_trace(enc_dec, flows, markets, (START + DAY,) * 4, cold)
    assert torch.count_nonzero(d_cold) == 0
    cold = layer.stage(cold, flows, (START + DAY,) * 4, em_cold, issued, record_pending=True)
    cold = layer.settle(cold, matured, train=True)

    # Con calentamiento previo (`record_pending=False`)
    warm = layer.initial_state("p")
    d_warm0, em_warm0 = layer.read_and_trace(enc_warm, flows, markets, (START,) * 4, warm)
    assert torch.count_nonzero(d_warm0) == 0
    warm = layer.stage(
        warm, flows, (START,) * 4, em_warm0, torch.zeros_like(issued), record_pending=False
    )
    assert len(warm.pending) == 0 and len(warm.markets) == 0 and len(warm.flows) == 4
    d_warm1, em_warm1 = layer.read_and_trace(enc_dec, flows, markets, (START + DAY,) * 4, warm)
    assert torch.count_nonzero(d_warm1) == 0
    warm = layer.stage(warm, flows, (START + DAY,) * 4, em_warm1, issued, record_pending=True)
    warm = layer.settle(warm, matured, train=True)

    # La sinapsis tras la primera maduración refleja el historial de trazas calentadas.
    assert not torch.equal(warm.markets["US"].synapse, cold.markets["US"].synapse)


def test_coexistence_with_hologram_and_crrm_preserves_exact_order_before_head():
    """Orden antes de la cabeza: holograma -> CRRM -> D3F-LR -> cabeza."""
    spec = specification()
    crrm_cfg = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5,), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )
    hsi_cfg = SessionHologramConfig(
        bands=4, phase_dim=8, mode="interferometric", position="after_mac"
    )
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        session_hologram=hsi_cfg,
        cross_sectional_rank_memory=crrm_cfg,
        three_factor_consolidation=VARIANT,
    )
    predictor = FinancialPredictor(cfg, dtype=torch.float64)
    with torch.no_grad():
        predictor.session_hologram.gate.fill_(0.15)
        predictor.cross_sectional_rank_memory.out.weight.fill_(0.08)
        predictor.three_factor_consolidation.gate.fill_(0.2)
    predictor._seal_parameters()

    b0 = make_session_batch(5, at=START, seed=1, dtype=torch.float64)
    b1 = make_session_batch(5, at=START + DAY, seed=2, dtype=torch.float64)
    st = predictor.initial_state(b0.flow_ids)
    rm = predictor.initial_rank_memory()
    tf = predictor.initial_three_factor_state()
    p0 = predictor.prepare(b0, st, session_complete=True, rank_memory=rm, three_factor_state=tf)
    rm = predictor.stage_rank_memory(
        rm, b0.flow_ids, b0.prediction_at, p0.rank_memory_keys, p0.point_predictions
    )
    tf = predictor.stage_three_factor_state(
        tf,
        b0.flow_ids,
        b0.prediction_at,
        p0.three_factor_emission,
        p0.point_predictions,
        record_pending=True,
    )
    matured = [(f, START, 0.01 * i - 0.02) for i, f in enumerate(b0.flow_ids)]
    rm = predictor.settle_rank_memory(rm, matured, train=True)
    tf = predictor.settle_three_factor_state(tf, matured, train=True)

    # Verificar orden de llamada exacto antes de la cabeza.
    calls = []
    orig_hsi = predictor.session_hologram.forward
    orig_crrm = predictor.cross_sectional_rank_memory.read
    orig_d3f = predictor.three_factor_consolidation.read_and_trace
    orig_head = predictor.head.forward

    def wrap(tag, fn):
        def inner(*args, **kwargs):
            calls.append(tag)
            return fn(*args, **kwargs)

        return inner

    predictor.session_hologram.forward = wrap("hologram", orig_hsi)
    predictor.cross_sectional_rank_memory.read = wrap("crrm", orig_crrm)
    predictor.three_factor_consolidation.read_and_trace = wrap("d3f", orig_d3f)
    predictor.head.forward = wrap("head", orig_head)
    p1 = predictor.prepare(
        b1, p0.next_state, session_complete=True, rank_memory=rm, three_factor_state=tf
    )
    assert calls == ["hologram", "crrm", "d3f", "head"]
    assert torch.isfinite(p1.point_predictions).all()


def test_state_dict_contracts_registry_declared_sources_and_guards():
    predictor = model(three_factor_consolidation=VARIANT)
    with torch.no_grad():
        predictor.three_factor_consolidation.gate.fill_(0.1)
    predictor._seal_parameters()
    twin = model(three_factor_consolidation=VARIANT)
    twin.load_state_dict(predictor.state_dict())
    with pytest.raises(ValueError):
        model(three_factor_consolidation=SHUFFLED_CFG).load_state_dict(predictor.state_dict())

    raw = asdict(VARIANT)
    assert checked_predictor_changes(dict(three_factor_consolidation=raw)) == dict(
        three_factor_consolidation=raw
    )
    base, batch = setup()
    cfg = replace(base.config, three_factor_consolidation=json.loads(json.dumps(raw)))
    assert cfg.three_factor_consolidation == VARIANT
    assert declared_sources(base.config) == ()
    assert declared_sources(cfg) == ("mars_titan.models.titans.three_factor_consolidation",)

    frozen_pred = model(three_factor_consolidation=VARIANT).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="tres factores"):
        FrozenFinancialConsumer(frozen_pred)

    with pytest.raises(ValueError, match="no declara"):
        base.prepare(batch, base.initial_state(batch.flow_ids), three_factor_state=object())
    active = model(three_factor_consolidation=VARIANT)
    with pytest.raises(ValueError, match="sesión completa"):
        active.prepare(
            batch,
            active.initial_state(batch.flow_ids),
            three_factor_state=active.initial_three_factor_state(),
        )
    with pytest.raises(ValueError, match="consolidación"):
        active.prepare(batch, active.initial_state(batch.flow_ids), session_complete=True)

    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(
            cfg,
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )

    for variant in ("mac_frozen", "mac_disabled", "transformer_direct"):
        with pytest.raises(ValueError, match="mac_online"):
            replace(base.config, variant=variant, three_factor_consolidation=VARIANT)

    for bad in (
        dict(raw, dim=3),
        dict(raw, dim=65),
        dict(raw, trace_tau=0.5),
        dict(raw, consolidation_tau=300.0),
        dict(raw, credit_scale=0.0),
        dict(raw, max_gain=0.0),
        dict(raw, max_gain=5.0),
        dict(raw, minimum_rows=2),
        dict(raw, mode="unknown"),
        dict(raw, seed=-1),
        dict(dim=16),
        "three_factor_consolidation_v1",
    ):
        with pytest.raises(ValueError):
            parse_three_factor_consolidation(bad)

    layer = ThreeFactorConsolidation(16, VARIANT, seed=1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer.read_and_trace(
            encoded(4, dtype=torch.float32),
            ("US/A", "US/B", "US/C", "US/D"),
            ("US",) * 4,
            (START,) * 4,
            layer.initial_state("p"),
        )

    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    tr = object.__new__(ChronologicalTrainer)
    tr.recipe = ChronologicalRecipe()
    mod = "mars_titan.models.titans.three_factor_consolidation"
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
                three_factor_consolidation=val,
            )
        )
        assert (mod in tr._code()) is present


def test_campaign_v12_screen3_accepts_single_variable_d3f_declaration():
    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    candidate_cfg = dict(
        dim=16,
        trace_tau=5.0,
        consolidation_tau=21.0,
        credit_scale=1.0,
        max_gain=0.25,
        minimum_rows=8,
        mode=THREE_FACTOR,
        seed=0,
    )
    doc["models"]["titans_mac_d3f"] = dict(
        doc["models"]["titans_mac"],
        tier="s3_inventions",
        predictor_changes=dict(three_factor_consolidation=candidate_cfg),
    )
    doc["models"]["titans_mac_d3f_shuffled"] = dict(
        doc["models"]["titans_mac"],
        tier="s3_inventions",
        predictor_changes=dict(
            three_factor_consolidation=dict(candidate_cfg, mode=SHUFFLED_CREDIT)
        ),
    )
    validated = campaign.checked_declaration(doc, root=root)
    assert "titans_mac_d3f" in validated["models"]
    assert "titans_mac_d3f_shuffled" in validated["models"]

    recipe = json.loads((root / doc["recipes"]["core"]).read_text())
    predictor_opts = dict(recipe["predictor"])
    predictor_opts.pop("dtype")
    base_spec = specification()
    spec = FinancialInputSpec(
        source_sha256=base_spec.source_sha256,
        view_sha256=base_spec.view_sha256,
        representation=json.loads(base_spec._representation_json),
        dimensions=dict(base_spec.dimensions, charts=46),
        input_policy=base_spec.input_policy,
    )
    changes = checked_predictor_changes(dict(three_factor_consolidation=candidate_cfg))
    control_id = FinancialConfig(spec, variant="mac_online", **predictor_opts).identity()
    candidate_id = FinancialConfig(
        spec, variant="mac_online", **(predictor_opts | changes)
    ).identity()
    assert set(candidate_id) - set(control_id) == {"three_factor_consolidation"}
    assert {
        k: v for k, v in candidate_id.items() if k != "three_factor_consolidation"
    } == control_id
