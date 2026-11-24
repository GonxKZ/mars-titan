"""Memoria neuronal de Titans supervisada por IC con retención anclada (`ic_titans_mlp`)."""

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
from mars_titan.models.titans.config import LAYER_NORM_EPS
from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, declared_sources
from mars_titan.models.titans.ic_titans_mlp import (
    ALPHA_FLOOR,
    ANCHOR_FLOOR,
    ETA_MAX,
    IC_TITANS_MLP,
    KEY_NORM_EPS,
    THETA_MAX,
    ICTitansMLP,
    ICTitansMLPConfig,
)
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm, standardized
from mars_titan.models.titans.session_ranks import gaussian_ranks
from mars_titan.models.titans.technical_branch import TechnicalDenseConfig
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.search_cases import checked_predictor_changes

START, DAY = 1_609_459_200_000_000, 86_400_000_000
ROOT = Path(__file__).resolve().parents[3]
VARIANT = ICTitansMLPConfig(
    dim=8,
    alpha_half_life=21.0,
    eta=0.5,
    theta=0.05,
    anchor=0.1,
    grad_clip=1.0,
    inner_steps=1,
    minimum_rows=3,
    mode="matured_rank",
    seed=0,
)
SHUFFLED = replace(VARIANT, mode="shuffled_targets")
EVAL_FROZEN = replace(VARIANT, mode="eval_frozen")
UNANCHORED = replace(VARIANT, mode="unanchored")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def active_layer(config=VARIANT, *, hidden=16, seed=7, dtype=torch.float64):
    """Módulo con ganancia residual distinta de cero para observar la lectura asociativa."""
    layer = ICTitansMLP(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        layer.gate.fill_(0.25)
    return layer


def encoded(rows=6, hidden=16, seed=11, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=dtype)


def _apply_ref_mlp(keys, weights):
    hidden = keys @ weights[0].T
    activated = torch.nn.functional.gelu(hidden)
    out = activated @ weights[1].T
    return keys + torch.nn.functional.layer_norm(out, (keys.shape[-1],), eps=LAYER_NORM_EPS)


def _clip_ref(grad, limit):
    norm = grad.norm()
    return grad * (limit / norm) if norm > limit else grad


def reference_autograd_settle_and_read(
    layer, sessions, query_encoded, query_markets, query_at, *, differentiable=True
):
    """Referencia literal mediante `torch.autograd.grad(..., create_graph=differentiable)`."""
    config = layer.config
    dtype, device = query_encoded.dtype, query_encoded.device
    alpha = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * torch.sigmoid(layer.alpha_logits)
    eta = ETA_MAX * torch.sigmoid(layer.eta_logits)
    theta = THETA_MAX * torch.sigmoid(layer.theta_logits)
    mu = ANCHOR_FLOOR + torch.nn.functional.softplus(layer.anchor_logits)
    unit_w = layer.w_read / torch.sqrt(layer.w_read.square().sum() + KEY_NORM_EPS)
    w0 = (layer.initial_weights[0], layer.initial_weights[1])
    anchored = config.mode != "unanchored"

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

    states = {}
    for enc, flows, at, targets in sessions:
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
            g_vec = gaussian_ranks(
                y_vec.unsqueeze(-1), torch.ones((len(idx), 1), dtype=torch.bool, device=device)
            ).squeeze(-1)
            g_std = standardized(g_vec)
            if market not in states:
                cur_w = (w0[0], w0[1]) if differentiable else (w0[0].detach(), w0[1].detach())
                cur_s = (torch.zeros_like(w0[0]), torch.zeros_like(w0[1]))
            else:
                cur_w, cur_s = states[market]
            step_scale = 1.0 / float(config.inner_steps)
            for inner in range(config.inner_steps):
                local = tuple(
                    (w.clone() if differentiable else w.detach().clone()).requires_grad_(True)
                    for w in cur_w
                )
                pred = _apply_ref_mlp(k_mat, local)
                scores = pred @ unit_w
                corr = (standardized(scores) * g_std).mean()
                reg = 0.0
                for layer_idx in range(2):
                    anchor_target = w0[layer_idx] if anchored else torch.zeros_like(w0[layer_idx])
                    reg = (
                        reg
                        + 0.5 * mu[layer_idx] * (local[layer_idx] - anchor_target).square().sum()
                    )
                loss = -corr + reg
                grads = torch.autograd.grad(loss, local, create_graph=differentiable)
                next_w, next_s = [], []
                for layer_idx in range(2):
                    clipped = _clip_ref(grads[layer_idx], config.grad_clip)
                    s_new = (
                        eta[layer_idx] * cur_s[layer_idx]
                        - (theta[layer_idx] * step_scale) * clipped
                    )
                    if inner == 0:
                        if anchored:
                            w_new = (
                                (1.0 - alpha[layer_idx]) * cur_w[layer_idx]
                                + alpha[layer_idx] * w0[layer_idx]
                                + s_new
                            )
                        else:
                            w_new = (1.0 - alpha[layer_idx]) * cur_w[layer_idx] + s_new
                    else:
                        w_new = cur_w[layer_idx] + s_new
                    next_s.append(s_new)
                    next_w.append(w_new)
                cur_w, cur_s = tuple(next_w), tuple(next_s)
            states[market] = (cur_w, cur_s)

    q_keys = compute_keys(query_encoded, query_markets, query_at)
    readouts = torch.empty_like(q_keys)
    for row, market in enumerate(query_markets):
        weights = states[market][0] if market in states else w0
        readouts[row] = _apply_ref_mlp(q_keys[row : row + 1], weights).squeeze(0)
    delta = layer.gate * layer.out(layer.readout_norm(readouts))
    return delta, states


@pytest.mark.parametrize("mode", ["matured_rank", "unanchored"])
@pytest.mark.parametrize("inner_steps", [1, 2])
def test_analytic_inner_gradient_and_metagradients_match_autograd_reference(mode, inner_steps):
    """El gradiente cerrado iguala al autograd de segundo orden en estado y metagradientes."""
    cfg = replace(VARIANT, mode=mode, inner_steps=inner_steps, grad_clip=0.5)
    layer = active_layer(cfg)
    flows_1 = ("US/AAA", "US/BBB", "US/CCC", "US/DDD", "CN/600000.SS", "CN/600001.SS")
    targets_1 = [0.03, -0.02, 0.01, -0.05, 0.02, -0.01]
    flows_2 = ("US/DDD", "US/AAA", "US/CCC", "US/BBB")
    targets_2 = [-0.01, 0.04, 0.02, -0.03]
    enc_1 = encoded(len(flows_1), seed=21)
    enc_2 = encoded(len(flows_2), seed=31)
    q_enc = encoded(5, seed=41)
    q_markets = ("US", "CN", "US", "US", "CN")
    q_at = (START + 2 * DAY,) * 5

    # Ruta analítica bajo prueba.
    e1_a = enc_1.clone().requires_grad_(True)
    e2_a = enc_2.clone().requires_grad_(True)
    state = layer.initial_state("p")
    m1 = tuple(f.split("/", 1)[0] for f in flows_1)
    k1 = layer.keys(e1_a, m1, (START,) * len(flows_1))
    state = layer.stage(state, flows_1, (START,) * len(flows_1), k1, differentiable=True)
    state = layer.settle(
        state,
        [(f, START, y) for f, y in zip(flows_1, targets_1, strict=True)],
        differentiable=True,
        train=True,
    )
    # CN solo tuvo 2 filas (< minimum_rows=3): no actualiza CN pero vacía su cola pendiente.
    assert "US" in state.markets and "CN" not in state.markets
    assert len(state.pending) == 0

    m2 = ("US",) * len(flows_2)
    k2 = layer.keys(e2_a, m2, (START + DAY,) * len(flows_2))
    state = layer.stage(state, flows_2, (START + DAY,) * len(flows_2), k2, differentiable=True)
    state = layer.settle(
        state,
        [(f, START + DAY, y) for f, y in zip(flows_2, targets_2, strict=True)],
        differentiable=True,
        train=True,
    )
    q_keys = layer.keys(q_enc, q_markets, q_at)
    actual = layer.read(q_keys, q_markets, state)
    actual_loss = (
        actual * torch.linspace(-1.0, 1.0, actual.numel(), dtype=torch.float64).reshape_as(actual)
    ).sum()
    actual_grads = torch.autograd.grad(
        actual_loss, [e1_a, e2_a, *layer.parameters()], allow_unused=True
    )

    # Ruta de referencia con torch.autograd.grad(create_graph=True).
    e1_r = enc_1.clone().requires_grad_(True)
    e2_r = enc_2.clone().requires_grad_(True)
    expected, ref_states = reference_autograd_settle_and_read(
        layer,
        [(e1_r, flows_1, START, targets_1), (e2_r, flows_2, START + DAY, targets_2)],
        q_enc,
        q_markets,
        q_at,
        differentiable=True,
    )
    expected_loss = (
        expected
        * torch.linspace(-1.0, 1.0, expected.numel(), dtype=torch.float64).reshape_as(expected)
    ).sum()
    expected_grads = torch.autograd.grad(
        expected_loss, [e1_r, e2_r, *layer.parameters()], allow_unused=True
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    for layer_idx in range(2):
        torch.testing.assert_close(
            state.markets["US"].weights[layer_idx],
            ref_states["US"][0][layer_idx],
            rtol=1e-12,
            atol=1e-13,
        )
        torch.testing.assert_close(
            state.markets["US"].momentum[layer_idx],
            ref_states["US"][1][layer_idx],
            rtol=1e-12,
            atol=1e-13,
        )
    names = ["enc_1", "enc_2", *(n for n, _ in layer.named_parameters())]
    for name, g_act, g_exp in zip(names, actual_grads, expected_grads, strict=True):
        assert g_act is not None and g_exp is not None, name
        assert torch.isfinite(g_act).all(), name
        assert torch.count_nonzero(g_act) > 0, name
        torch.testing.assert_close(g_act, g_exp, rtol=1e-10, atol=1e-12)


def test_inner_update_improves_session_rank_correlation_and_anchors_toward_m0():
    """Una escritura madura sube ρ(w_readᵀ M(K̂), g) y el anclaje devuelve M_m hacia M_0, no a 0."""
    cfg = replace(VARIANT, theta=0.2, anchor=0.05, alpha_half_life=5.0)
    layer = active_layer(cfg)
    flows = tuple(f"US/S{i:02d}" for i in range(16))
    markets = ("US",) * 16
    enc = encoded(16, seed=73)
    targets = [float(i - 7.5) * 0.01 for i in range(16)]
    keys = layer.keys(enc, markets, (START,) * 16).detach()
    g = gaussian_ranks(
        torch.tensor(targets, dtype=torch.float64).unsqueeze(-1),
        torch.ones((16, 1), dtype=torch.bool),
    ).squeeze(-1)
    unit_w = layer.w_read / torch.sqrt(layer.w_read.square().sum() + KEY_NORM_EPS)

    before_scores = _apply_ref_mlp(keys, layer.initial_weights) @ unit_w
    corr_before = float((standardized(before_scores) * standardized(g)).mean().detach())

    state = layer.stage(layer.initial_state("p"), flows, (START,) * 16, keys)
    state = layer.settle(
        state, [(f, START, y) for f, y in zip(flows, targets, strict=True)], train=True
    )
    after_scores = _apply_ref_mlp(keys, state.markets["US"].weights) @ unit_w
    corr_after = float((standardized(after_scores) * standardized(g)).mean().detach())
    assert corr_after > corr_before + 0.05

    # Tras sesiones posteriores con etiquetas constantes (gradiente de IC nulo), `matured_rank`
    # regresa exponencialmente hacia M_0, mientras que `unanchored` decae hacia 0.
    unanch = active_layer(replace(cfg, mode="unanchored"))
    st_anch = state
    st_unanch = unanch.stage(unanch.initial_state("p"), flows, (START,) * 16, keys)
    st_unanch = unanch.settle(
        st_unanch, [(f, START, y) for f, y in zip(flows, targets, strict=True)], train=True
    )
    for step in range(1, 35):
        at = START + step * DAY
        const_matured = [(f, at, 0.0) for f in flows]
        st_anch = layer.settle(layer.stage(st_anch, flows, (at,) * 16, keys), const_matured)
        st_unanch = unanch.settle(unanch.stage(st_unanch, flows, (at,) * 16, keys), const_matured)

    dist_anch_to_m0 = sum(
        (w - w0).norm().item()
        for w, w0 in zip(st_anch.markets["US"].weights, layer.initial_weights, strict=True)
    )
    dist_unanch_to_zero = sum(w.norm().item() for w in st_unanch.markets["US"].weights)
    m0_norm = sum(w0.norm().item() for w0 in layer.initial_weights)
    assert dist_anch_to_m0 < 0.05 * m0_norm
    assert dist_unanch_to_zero < 0.05 * m0_norm


def test_frobenius_clipping_bounds_extreme_updates_and_preserves_small_gradients():
    cfg_tight = replace(VARIANT, grad_clip=0.05, theta=0.25)
    cfg_loose = replace(VARIANT, grad_clip=50.0, theta=0.25)
    tight = active_layer(cfg_tight)
    loose = active_layer(cfg_loose)
    flows = tuple(f"US/A{i}" for i in range(6))
    keys = tight.keys(encoded(6, seed=91), ("US",) * 6, (START,) * 6).detach()
    matured = [(f, START, float(i)) for i, f in enumerate(flows)]
    st_tight = tight.settle(
        tight.stage(tight.initial_state("p"), flows, (START,) * 6, keys), matured
    )
    st_loose = loose.settle(
        loose.stage(loose.initial_state("p"), flows, (START,) * 6, keys), matured
    )
    # Con recorte estricto la norma del paso de momentum queda estrictamente acotada por theta * C.
    theta_val = THETA_MAX * torch.sigmoid(tight.theta_logits).detach()
    for idx in range(2):
        assert st_tight.markets["US"].momentum[idx].norm() <= theta_val[idx] * 0.05 + 1e-12
        assert (
            st_loose.markets["US"].momentum[idx].norm()
            > st_tight.markets["US"].momentum[idx].norm()
        )


def test_markets_stay_isolated_and_row_order_does_not_change_memory_state():
    layer = active_layer(VARIANT)
    flows = ("US/A", "CN/1", "US/B", "US/C", "CN/2", "CN/3", "US/D")
    markets = tuple(f.split("/", 1)[0] for f in flows)
    enc = encoded(len(flows), seed=5)
    targets = [0.05, -0.04, 0.01, -0.02, 0.03, 0.02, -0.06]

    def run_order(order):
        st = layer.initial_state("p")
        o_flows = tuple(flows[i] for i in order)
        o_markets = tuple(markets[i] for i in order)
        o_enc = enc[list(order)]
        o_keys = layer.keys(o_enc, o_markets, (START,) * len(order))
        st = layer.stage(st, o_flows, (START,) * len(order), o_keys, differentiable=False)
        matured = [(flows[i], START, targets[i]) for i in reversed(order)]
        return layer.settle(st, matured, differentiable=False, train=True)

    base = run_order(list(range(len(flows))))
    perm = run_order([4, 2, 6, 0, 5, 1, 3])
    for market in ("US", "CN"):
        for idx in range(2):
            torch.testing.assert_close(
                perm.markets[market].weights[idx],
                base.markets[market].weights[idx],
                rtol=1e-12,
                atol=1e-13,
            )
            torch.testing.assert_close(
                perm.markets[market].momentum[idx],
                base.markets[market].momentum[idx],
                rtol=1e-12,
                atol=1e-13,
            )

    # Alterar objetivos de CN no cambia en un solo bit el estado de US.
    st_alt = layer.initial_state("p")
    keys = layer.keys(enc, markets, (START,) * len(flows))
    st_alt = layer.stage(st_alt, flows, (START,) * len(flows), keys, differentiable=False)
    alt_targets = [
        v * (-10.0 if flows[i].startswith("CN/") else 1.0) for i, v in enumerate(targets)
    ]
    st_alt = layer.settle(
        st_alt,
        [(f, START, v) for f, v in zip(flows, alt_targets, strict=True)],
        differentiable=False,
        train=True,
    )
    for idx in range(2):
        assert torch.equal(st_alt.markets["US"].weights[idx], base.markets["US"].weights[idx])
        assert torch.equal(st_alt.markets["US"].momentum[idx], base.markets["US"].momentum[idx])
        assert not torch.equal(st_alt.markets["CN"].weights[idx], base.markets["CN"].weights[idx])


def test_placebo_and_ablation_modes_share_exact_parameters_and_isolate_mechanisms():
    real = active_layer(VARIANT)
    shuffled = active_layer(SHUFFLED)
    frozen = active_layer(EVAL_FROZEN)
    unanchored = active_layer(UNANCHORED)
    for items in zip(
        real.state_dict().items(),
        shuffled.state_dict().items(),
        frozen.state_dict().items(),
        unanchored.state_dict().items(),
        strict=True,
    ):
        name = items[0][0]
        tensors = [v for _, v in items]
        if isinstance(tensors[0], torch.Tensor):
            for other in tensors[1:]:
                assert torch.equal(tensors[0], other), name

    flows = tuple(f"US/S{i}" for i in range(8))
    markets = ("US",) * 8
    enc = encoded(8, seed=19)
    targets = [0.01 * (i - 3.5) for i in range(8)]
    matured = [(f, START, y) for f, y in zip(flows, targets, strict=True)]

    st_real = real.settle(
        real.stage(
            real.initial_state("p"), flows, (START,) * 8, real.keys(enc, markets, (START,) * 8)
        ),
        matured,
        train=True,
    )
    st_shuf = shuffled.settle(
        shuffled.stage(
            shuffled.initial_state("p"),
            flows,
            (START,) * 8,
            shuffled.keys(enc, markets, (START,) * 8),
        ),
        matured,
        train=True,
    )
    assert st_shuf.markets["US"].updates == st_real.markets["US"].updates == 1
    assert not torch.equal(st_shuf.markets["US"].weights[0], st_real.markets["US"].weights[0])

    # eval_frozen actualiza en train=True igual que matured_rank y congela en train=False.
    st_f_train = frozen.settle(
        frozen.stage(
            frozen.initial_state("p"), flows, (START,) * 8, frozen.keys(enc, markets, (START,) * 8)
        ),
        matured,
        train=True,
    )
    for idx in range(2):
        assert torch.equal(
            st_f_train.markets["US"].weights[idx], st_real.markets["US"].weights[idx]
        )
    st_f_eval = frozen.settle(
        frozen.stage(
            frozen.initial_state("p"), flows, (START,) * 8, frozen.keys(enc, markets, (START,) * 8)
        ),
        matured,
        train=False,
    )
    assert "US" not in st_f_eval.markets and len(st_f_eval.pending) == 0


def test_disabled_field_keeps_identity_parameters_and_predictions_bit_for_bit():
    base, batch = setup()
    explicit = model(ic_titans_mlp=None)
    assert explicit.config.identity() == base.config.identity()
    assert "ic_titans_mlp" not in base.config.identity()
    assert base.ic_titans_mlp is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert actual.ic_mlp_keys is None


@pytest.mark.parametrize("config", [VARIANT, SHUFFLED, EVAL_FROZEN, UNANCHORED])
def test_all_modes_start_as_reference_without_touching_backbone_or_chaining_two_zeros(config):
    base, _ = setup()
    batch = session_batch(6, markets=["US"] * 6)
    predictor = model(ic_titans_mlp=config)
    ref_params = dict(base.named_parameters())
    new_params = dict(predictor.named_parameters())
    assert set(new_params) - set(ref_params) == {
        "ic_titans_mlp.norm.weight",
        "ic_titans_mlp.norm.bias",
        "ic_titans_mlp.key_proj.weight",
        "ic_titans_mlp.initial_weights.0",
        "ic_titans_mlp.initial_weights.1",
        "ic_titans_mlp.w_read",
        "ic_titans_mlp.readout_norm.weight",
        "ic_titans_mlp.readout_norm.bias",
        "ic_titans_mlp.out.weight",
        "ic_titans_mlp.gate",
        "ic_titans_mlp.alpha_logits",
        "ic_titans_mlp.eta_logits",
        "ic_titans_mlp.theta_logits",
        "ic_titans_mlp.anchor_logits",
    }
    for name, value in ref_params.items():
        assert torch.equal(new_params[name], value), name
    # Solo `gate` arranca en cero; `out.weight`, `w_read` e `initial_weights` no son nulos.
    assert torch.count_nonzero(predictor.ic_titans_mlp.gate) == 0
    assert torch.count_nonzero(predictor.ic_titans_mlp.out.weight) > 0
    assert torch.count_nonzero(predictor.ic_titans_mlp.w_read) > 0
    assert torch.count_nonzero(predictor.ic_titans_mlp.initial_weights[0]) > 0
    im = predictor.initial_ic_mlp_memory()
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True, ic_mlp_memory=im
    )
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    assert actual.ic_mlp_keys.shape == (6, config.dim)
    identity = predictor.config.identity()["ic_titans_mlp"]
    assert identity["kind"] == IC_TITANS_MLP and identity["mode"] == config.mode


def make_session_batch(rows: int, *, at: int, seed: int, dtype: torch.dtype):
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

        def run_once(zero_gate: bool):
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
                ic_titans_mlp=VARIANT,
            )
            predictor = FinancialPredictor(cfg, dtype=dtype)
            predictor.set_learning_statistics(statistics(cfg.learning_scale))
            if not zero_gate:
                with torch.no_grad():
                    predictor.ic_titans_mlp.gate.fill_(0.08)
                predictor._seal_parameters()
            batches = [
                make_session_batch(6, at=START + i * DAY, seed=101 + i, dtype=dtype)
                for i in range(3)
            ]
            state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
            im = predictor.initial_ic_mlp_memory()
            p0 = predictor.prepare(
                batches[0], state, differentiable=True, session_complete=True, ic_mlp_memory=im
            )
            assert torch.count_nonzero(p0.point_predictions) == 0
            im = predictor.stage_ic_mlp_memory(
                im,
                batches[0].flow_ids,
                batches[0].prediction_at,
                p0.ic_mlp_keys,
                differentiable=True,
            )
            y0 = [0.02, -0.01, 0.03, -0.04, 0.01, -0.02]
            im = predictor.settle_ic_mlp_memory(
                im,
                [(f, START, v) for f, v in zip(batches[0].flow_ids, y0, strict=True)],
                differentiable=True,
                train=True,
            )
            p1 = predictor.prepare(
                batches[1],
                p0.next_state,
                differentiable=True,
                session_complete=True,
                ic_mlp_memory=im,
            )
            im = predictor.stage_ic_mlp_memory(
                im,
                batches[1].flow_ids,
                batches[1].prediction_at,
                p1.ic_mlp_keys,
                differentiable=True,
            )
            y1 = [-0.02, 0.03, -0.01, 0.04, -0.03, 0.02]
            im = predictor.settle_ic_mlp_memory(
                im,
                [(f, START + DAY, v) for f, v in zip(batches[1].flow_ids, y1, strict=True)],
                differentiable=True,
                train=True,
            )
            p2 = predictor.prepare(
                batches[2],
                p1.next_state,
                differentiable=True,
                session_complete=True,
                ic_mlp_memory=im,
            )
            y2 = torch.tensor([0.01, -0.03, 0.04, -0.02, 0.02, -0.01], dtype=dtype)
            sessions = [("US", START + 2 * DAY)] * 6
            term = SessionICTerm(weight=3.0, minimum_rows=3)
            loss = pinball_loss(p2.quantiles, y2) + term.term(p2.point_predictions, y2, sessions)
            loss.backward()
            grads = {
                n: p.grad.clone() for n, p in predictor.named_parameters() if p.grad is not None
            }
            return p2.quantiles.detach(), grads

        # Paso 1 con gate == 0 y mediana inicial constante en cero: `gate` recibe gradiente != 0.
        q_a, g_a = run_once(zero_gate=True)
        q_b, g_b = run_once(zero_gate=True)
        assert torch.equal(q_a, q_b)
        for name, grad in g_a.items():
            assert torch.isfinite(grad).all(), name
            assert torch.equal(grad, g_b[name]), name
        assert torch.count_nonzero(g_a["ic_titans_mlp.gate"]) > 0

        # Paso 2 (gate != 0): TODOS los parámetros reciben metagradiente finito y no nulo.
        _, g_meta = run_once(zero_gate=False)
        for name in (
            "ic_titans_mlp.gate",
            "ic_titans_mlp.out.weight",
            "ic_titans_mlp.readout_norm.weight",
            "ic_titans_mlp.initial_weights.0",
            "ic_titans_mlp.initial_weights.1",
            "ic_titans_mlp.w_read",
            "ic_titans_mlp.key_proj.weight",
            "ic_titans_mlp.norm.weight",
            "ic_titans_mlp.alpha_logits",
            "ic_titans_mlp.eta_logits",
            "ic_titans_mlp.theta_logits",
            "ic_titans_mlp.anchor_logits",
        ):
            assert torch.isfinite(g_meta[name]).all() and torch.count_nonzero(g_meta[name]) > 0, (
                name
            )
    finally:
        torch.use_deterministic_algorithms(previous)


def test_constant_targets_constant_keys_and_all_zero_scalar_head_produce_finite_gradients():
    spec = specification()
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=True),
        ic_titans_mlp=VARIANT,
    )
    predictor = FinancialPredictor(cfg, dtype=torch.float64)
    predictor.set_learning_statistics(statistics(cfg.learning_scale))
    batches = [
        make_session_batch(5, at=START + i * DAY, seed=201 + i, dtype=torch.float64)
        for i in range(2)
    ]
    state = predictor.initial_state(batches[0].flow_ids, differentiable=True)
    im = predictor.initial_ic_mlp_memory()
    p0 = predictor.prepare(
        batches[0], state, differentiable=True, session_complete=True, ic_mlp_memory=im
    )
    im = predictor.stage_ic_mlp_memory(
        im, batches[0].flow_ids, batches[0].prediction_at, p0.ic_mlp_keys, differentiable=True
    )
    # Etiquetas constantes (empate total) + cabeza escalar en cero.
    matured = [(flow, START, 0.01) for flow in batches[0].flow_ids]
    im = predictor.settle_ic_mlp_memory(im, matured, differentiable=True, train=True)
    p1 = predictor.prepare(
        batches[1], p0.next_state, differentiable=True, session_complete=True, ic_mlp_memory=im
    )
    loss = (p1.point_predictions - 0.01).abs().mean()
    loss.backward()
    for name, param in predictor.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name

    # Claves constantes idénticas en la sesión (dispersión nula de s = w_readᵀ M(K̂)).
    layer = active_layer(VARIANT)
    const_enc = torch.ones((6, 16), dtype=torch.float64, requires_grad=True)
    flows = tuple(f"US/C{i}" for i in range(6))
    k_const = layer.keys(const_enc, ("US",) * 6, (START,) * 6)
    st = layer.stage(layer.initial_state("p"), flows, (START,) * 6, k_const, differentiable=True)
    st = layer.settle(
        st,
        [(f, START, float(i)) for i, f in enumerate(flows)],
        differentiable=True,
        train=True,
    )
    out = layer.read(k_const, ("US",) * 6, st)
    out.square().sum().backward()
    assert const_enc.grad is not None and torch.isfinite(const_enc.grad).all()
    for name, param in layer.named_parameters():
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
    """Las claves pendientes se reúnen con una selección por emisión, no una por fila."""
    sizes = {}
    for rows in (8, 64):
        layer = active_layer()
        flows = [f"US/A{index:03d}" for index in range(rows)]
        when = [START] * rows
        keys = layer.keys(encoded(rows=rows), ["US"] * rows, when)
        state = layer.stage(layer.initial_state("p"), flows, when, keys, differentiable=True)
        assert len(state.pending) == rows
        matured = [
            (flow, START, float(value))
            for flow, value in zip(flows, torch.randn(rows), strict=True)
        ]
        settled = layer.settle(state, matured, differentiable=True)
        assert len(settled.pending) == 0
        sizes[rows] = _graph_nodes(settled.markets["US"].weights[0])
        (gradient,) = torch.autograd.grad(settled.markets["US"].weights[0].sum(), keys)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert sizes[8] == sizes[64], sizes


def test_detaching_compacts_pending_rows_and_split_block_staging_matches_whole_session():
    layer = active_layer()
    rows = 12
    flows = [f"US/A{index:03d}" for index in range(rows)]
    when = [START] * rows
    keys = layer.keys(encoded(rows=rows), ["US"] * rows, when)
    state = layer.stage(layer.initial_state("p"), flows, when, keys, differentiable=True)
    matured = [(flow, START, 0.01 * index) for index, flow in enumerate(flows[:9])]
    partial = layer.detach_state(layer.settle(state, matured, differentiable=True))
    assert sorted(flow for flow, _ in partial.pending) == flows[9:]
    for (flow, _), (source, row) in partial.pending.items():
        position = flows.index(flow)
        assert source.shape == (3, keys.shape[1]) and not source.requires_grad
        assert torch.equal(source[row], keys.detach()[position])
    exported = layer.export_state(partial)
    assert exported["pending"]["flows"] == flows[9:]
    assert torch.equal(exported["pending"]["keys"], keys.detach()[9:])
    restored = layer.restore_state(exported, "p", device="cpu")
    again = layer.export_state(restored)
    assert again["pending"]["flows"] == flows[9:]
    assert torch.equal(again["pending"]["keys"], keys.detach()[9:])
    for idx in range(2):
        assert torch.equal(restored.markets["US"].weights[idx], partial.markets["US"].weights[idx])
        assert torch.equal(
            restored.markets["US"].momentum[idx], partial.markets["US"].momentum[idx]
        )

    # Una sesión preparada en dos bloques produce el mismo estado que emitida de una vez.
    k_det = keys.detach()
    all_matured = [(flow, START, 0.01 * (index % 4) - 0.02) for index, flow in enumerate(flows)]
    whole = layer.settle(layer.stage(layer.initial_state("p"), flows, when, k_det), all_matured)
    first, second = [1, 4, 7, 8, 9, 11], [0, 2, 3, 5, 6, 10]
    st = layer.initial_state("p")
    for block in (first, second):
        st = layer.stage(st, [flows[i] for i in block], [START] * len(block), k_det[block])
    split = layer.settle(st, list(reversed(all_matured)))
    for idx in range(2):
        assert torch.equal(split.markets["US"].weights[idx], whole.markets["US"].weights[idx])
        assert torch.equal(split.markets["US"].momentum[idx], whole.markets["US"].momentum[idx])


def test_export_restore_detach_and_state_dict_contracts():
    predictor = model(ic_titans_mlp=VARIANT)
    with torch.no_grad():
        predictor.ic_titans_mlp.gate.fill_(0.08)
    predictor._seal_parameters()
    batch = session_batch(5, markets=["US"] * 5)
    state = predictor.initial_state(batch.flow_ids, differentiable=True)
    im = predictor.initial_ic_mlp_memory()
    p0 = predictor.prepare(
        batch, state, differentiable=True, session_complete=True, ic_mlp_memory=im
    )
    im = predictor.stage_ic_mlp_memory(
        im, batch.flow_ids, batch.prediction_at, p0.ic_mlp_keys, differentiable=True
    )
    im = predictor.settle_ic_mlp_memory(
        im,
        [(f, batch.prediction_at[0], 0.01 * i) for i, f in enumerate(batch.flow_ids)],
        differentiable=True,
        train=True,
    )
    assert im.markets["US"].weights[0].requires_grad and im.markets["US"].momentum[0].requires_grad
    detached = predictor.detach_ic_mlp_memory(im)
    assert not detached.markets["US"].weights[0].requires_grad
    payload = predictor.export_ic_mlp_memory(detached)
    restored = predictor.restore_ic_mlp_memory(payload)
    for idx in range(2):
        assert torch.equal(restored.markets["US"].weights[idx], detached.markets["US"].weights[idx])
        assert torch.equal(
            restored.markets["US"].momentum[idx], detached.markets["US"].momentum[idx]
        )

    twin = model(ic_titans_mlp=VARIANT)
    twin.load_state_dict(predictor.state_dict())
    with pytest.raises(ValueError):
        model(ic_titans_mlp=SHUFFLED).load_state_dict(predictor.state_dict())
    with pytest.raises(ValueError):
        model(ic_titans_mlp=UNANCHORED).load_state_dict(predictor.state_dict())


def test_coexistence_with_session_hologram_and_crrm_respects_pre_head_order():
    """Holograma -> CRRM -> IC-Titans-MLP -> cabeza conviven sin interferir cuando se apagan."""
    hologram = SessionHologramConfig(
        bands=2, phase_dim=4, mode="interferometric", position="after_mac"
    )
    crrm = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )
    batch = session_batch(6, markets=["US"] * 6)

    # Con ic_titans_mlp=None, holograma + CRRM producen los mismos bits que sin el campo.
    without = model(session_hologram=hologram, cross_sectional_rank_memory=crrm)
    explicit_none = model(
        session_hologram=hologram, cross_sectional_rank_memory=crrm, ic_titans_mlp=None
    )
    assert without.config.identity() == explicit_none.config.identity()
    rm_w = without.initial_rank_memory()
    p_w = without.prepare(
        batch, without.initial_state(batch.flow_ids), session_complete=True, rank_memory=rm_w
    )
    p_e = explicit_none.prepare(
        batch,
        explicit_none.initial_state(batch.flow_ids),
        session_complete=True,
        rank_memory=explicit_none.initial_rank_memory(),
    )
    assert torch.equal(p_w.point_predictions, p_e.point_predictions)

    # Con los tres activos, IC-Titans-MLP lee la representación que sale de holograma + CRRM.
    combined = model(
        session_hologram=hologram, cross_sectional_rank_memory=crrm, ic_titans_mlp=VARIANT
    )
    with torch.no_grad():
        combined.session_hologram.gate.fill_(0.1)
        combined.cross_sectional_rank_memory.out.weight.fill_(0.05)
        combined.ic_titans_mlp.gate.fill_(0.08)
    combined._seal_parameters()

    rm = combined.initial_rank_memory()
    im = combined.initial_ic_mlp_memory()
    p0 = combined.prepare(
        batch,
        combined.initial_state(batch.flow_ids),
        session_complete=True,
        rank_memory=rm,
        ic_mlp_memory=im,
    )
    matured = [(f, batch.prediction_at[0], 0.01 * (i - 2.5)) for i, f in enumerate(batch.flow_ids)]
    rm = combined.settle_rank_memory(
        combined.stage_rank_memory(
            rm, batch.flow_ids, batch.prediction_at, p0.rank_memory_keys, p0.point_predictions
        ),
        matured,
    )
    im = combined.settle_ic_mlp_memory(
        combined.stage_ic_mlp_memory(im, batch.flow_ids, batch.prediction_at, p0.ic_mlp_keys),
        matured,
    )
    batch_2 = make_session_batch(6, at=batch.prediction_at[0] + DAY, seed=77, dtype=torch.float64)
    state_2 = combined.initial_state(batch_2.flow_ids)
    # Comprobar el orden exacto: si solo activamos CRRM y luego aplicamos ic_titans_mlp a su
    # working_state, coincide bit a bit con `combined.working_state`.
    only_holo_crrm = model(session_hologram=hologram, cross_sectional_rank_memory=crrm)
    with torch.no_grad():
        only_holo_crrm.session_hologram.gate.fill_(0.1)
        only_holo_crrm.cross_sectional_rank_memory.out.weight.fill_(0.05)
    only_holo_crrm._seal_parameters()
    rm_ref = only_holo_crrm.restore_rank_memory(combined.export_rank_memory(rm))
    p_before_ic = only_holo_crrm.prepare(
        batch_2,
        only_holo_crrm.initial_state(batch_2.flow_ids),
        session_complete=True,
        rank_memory=rm_ref,
    )
    p_all = combined.prepare(
        batch_2, state_2, session_complete=True, rank_memory=rm, ic_mlp_memory=im
    )
    markets_2 = ("US",) * 6
    expected_keys = combined.ic_titans_mlp.keys(
        p_before_ic.working_state, markets_2, batch_2.prediction_at
    )
    expected_working = p_before_ic.working_state + combined.ic_titans_mlp.read(
        expected_keys, markets_2, im
    )
    torch.testing.assert_close(p_all.working_state, expected_working, rtol=0, atol=0)


def test_registry_declared_sources_and_invalid_declarations():
    raw = asdict(VARIANT)
    assert checked_predictor_changes(dict(ic_titans_mlp=raw)) == dict(ic_titans_mlp=raw)
    base, batch = setup()
    cfg = replace(base.config, ic_titans_mlp=json.loads(json.dumps(raw)))
    assert cfg.ic_titans_mlp == VARIANT
    assert declared_sources(base.config) == ()
    assert declared_sources(cfg) == ("mars_titan.models.titans.ic_titans_mlp",)

    # El consumidor congelado rechaza ic_titans_mlp explícitamente.
    predictor = model(ic_titans_mlp=VARIANT).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="IC-Titans-MLP"):
        FrozenFinancialConsumer(predictor)

    # Sin declarar ic_titans_mlp rechaza ic_mlp_memory; activa la exige con sesión completa.
    with pytest.raises(ValueError, match="no declara"):
        base.prepare(batch, base.initial_state(batch.flow_ids), ic_mlp_memory=object())
    active = model(ic_titans_mlp=VARIANT)
    with pytest.raises(ValueError, match="sesión completa"):
        active.prepare(
            batch,
            active.initial_state(batch.flow_ids),
            ic_mlp_memory=active.initial_ic_mlp_memory(),
        )
    with pytest.raises(ValueError, match="IC-Titans-MLP"):
        active.prepare(batch, active.initial_state(batch.flow_ids), session_complete=True)

    # Control local C no se combina con ic_titans_mlp.
    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(
            cfg,
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )

    # Variantes distintas de mac_online y declaraciones malformadas se rechazan.
    for variant in ("mac_frozen", "mac_disabled", "transformer_direct"):
        with pytest.raises(ValueError, match="mac_online"):
            replace(base.config, variant=variant, ic_titans_mlp=VARIANT)
    for bad in (
        dict(raw, dim=3),
        dict(raw, dim=256),
        dict(raw, alpha_half_life=0.5),
        dict(raw, alpha_half_life=500.0),
        dict(raw, eta=0.0),
        dict(raw, eta=0.95),
        dict(raw, theta=0.0),
        dict(raw, theta=0.6),
        dict(raw, anchor=0.0),
        dict(raw, grad_clip=0.0),
        dict(raw, inner_steps=0),
        dict(raw, inner_steps=5),
        dict(raw, minimum_rows=2),
        dict(raw, mode="unknown"),
        dict(raw, seed=-1),
        dict(dim=8),
        "ic_titans_mlp_v1",
    ):
        with pytest.raises(ValueError):
            replace(base.config, ic_titans_mlp=bad)

    # Autocast se rechaza.
    layer = ICTitansMLP(16, VARIANT, seed=1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer.keys(encoded(4, dtype=torch.float32), ("US",) * 4, (START,) * 4)

    # El código entra en ChronologicalTrainer._code solo cuando está activo.
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    tr = object.__new__(ChronologicalTrainer)
    tr.recipe = ChronologicalRecipe()
    mod = "mars_titan.models.titans.ic_titans_mlp"
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
                ic_titans_mlp=val,
            )
        )
        assert (mod in tr._code()) is present


def test_screen3_campaign_accepts_single_variable_ic_titans_mlp_and_placebo_declarations():
    screen3 = json.loads((ROOT / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    candidate_cfg = dict(
        dim=32,
        alpha_half_life=21.0,
        eta=0.5,
        theta=0.05,
        anchor=0.1,
        grad_clip=1.0,
        inner_steps=1,
        minimum_rows=8,
        mode="matured_rank",
        seed=0,
    )
    doc = json.loads(json.dumps(screen3))
    for suffix, mode in (
        ("", "matured_rank"),
        ("_shuffled", "shuffled_targets"),
        ("_eval_frozen", "eval_frozen"),
        ("_unanchored", "unanchored"),
    ):
        doc["models"][f"titans_mac_ic_mlp{suffix}"] = dict(
            machine="mars_titan",
            recipe="core",
            runner="fit",
            tier="s3_inventions",
            variant="mac_online",
            predictor_changes=dict(ic_titans_mlp=dict(candidate_cfg, mode=mode)),
        )
    validated = campaign.checked_declaration(doc, root=ROOT)
    assert "titans_mac_ic_mlp" in validated["models"]
    assert "titans_mac_ic_mlp_shuffled" in validated["models"]
    assert "titans_mac_ic_mlp_eval_frozen" in validated["models"]
    assert "titans_mac_ic_mlp_unanchored" in validated["models"]

    # Y sobre la receta real `core` solo cambia el campo `ic_titans_mlp` en la identidad.
    recipe_doc = json.loads((ROOT / screen3["recipes"]["core"]).read_text())
    predictor_opts = dict(recipe_doc["predictor"])
    predictor_opts.pop("dtype")
    base_spec = specification()
    spec = FinancialInputSpec(
        source_sha256=base_spec.source_sha256,
        view_sha256=base_spec.view_sha256,
        representation=json.loads(base_spec._representation_json),
        dimensions=dict(base_spec.dimensions, charts=46),
        input_policy=base_spec.input_policy,
    )
    changes = checked_predictor_changes(dict(ic_titans_mlp=candidate_cfg))
    control_id = FinancialConfig(spec, variant="mac_online", **predictor_opts).identity()
    candidate_id = FinancialConfig(
        spec, variant="mac_online", **(predictor_opts | changes)
    ).identity()
    assert set(candidate_id) - set(control_id) == {"ic_titans_mlp"}
    assert {k: v for k, v in candidate_id.items() if k != "ic_titans_mlp"} == control_id
