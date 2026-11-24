"""Memoria asociativa con sesgo atencional contrastivo transversal (InfoNCE-Titans, Idea 5).

Valida el contrato de configuración, las derivadas analíticas cerradas frente a autograd de
segundo orden en FP64, los prototipos deterministas, la ausencia de fugas causales, los controles
placebo con exactamente los mismos parámetros, la estabilidad numérica con predicción constante,
el reinicio entre fases, los checkpoints bit a bit y la paridad con acumulación por bloques.
"""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from test_financial_adapter import specification
from test_financial_learning_scale import build as build_scaled
from test_memory_stability import decision
from test_technical_dense import DENSE, opened
from torch.nn import functional as F

from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans.config import LAYER_NORM_EPS
from mars_titan.models.titans.contrastive_memory import (
    CONTRASTIVE_MEMORY,
    MODES,
    ContrastiveMemory,
    ContrastiveMemoryConfig,
    contrastive_inner_gradients,
    parse_contrastive_memory,
    safe_l2_normalize,
    session_prototypes,
)
from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, declared_sources
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm
from mars_titan.training import comparison_campaign, financial_run, mars_titan_walk_forward
from mars_titan.training.dust.financial import FinancialDustAdapter
from mars_titan.training.financial_run import clip_parameter_groups
from mars_titan.training.search_cases import checked_predictor_changes

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "configs/experiments/us-campaign-v12-screen3.json"
CORE_RECIPE = ROOT / "configs/titans/dgx-us-core-u2-ic3-dense-gr-v9.json"
DAY = 86_400_000_000
START = 1_609_459_200_000_000
SHARED_PREFIXES = ("price_encoder", "encoders", "fusion", "head", "mac")

VARIANT = ContrastiveMemoryConfig(prototypes=32, temperature=0.2, mode="infonce")
SHUFFLED = ContrastiveMemoryConfig(prototypes=32, temperature=0.2, mode="shuffled_values")
EVAL_FROZEN = ContrastiveMemoryConfig(prototypes=32, temperature=0.2, mode="eval_frozen")
L2_SHARED = ContrastiveMemoryConfig(prototypes=32, temperature=0.2, mode="l2_shared")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def make_module(
    mode="infonce",
    *,
    hidden=8,
    prototypes=4,
    temperature=0.2,
    seed=7,
    gain=None,
    dtype=torch.float64,
):
    config = ContrastiveMemoryConfig(prototypes=prototypes, temperature=temperature, mode=mode)
    module = ContrastiveMemory(hidden, config, seed=seed, dtype=dtype)
    if gain is not None:
        generator = torch.Generator().manual_seed(gain)
        with torch.no_grad():
            module.out_projection.weight.copy_(
                0.25 * torch.randn(hidden, hidden, generator=generator, dtype=dtype)
            )
    return module


def make_tokens(count, *, hidden=8, seed=11, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, hidden, generator=generator, dtype=dtype)


def step_session(module, state, rows, flows, at, *, differentiable=False):
    read, pending = module.read(rows, flows, (at,) * len(flows), state)
    return read, module.commit(pending, differentiable=differentiable)


def make_predictor(config=None, *, variant="mac_online", seed=42, dtype=torch.float64, **options):
    financial_config = FinancialConfig(
        specification(),
        variant=variant,
        hidden_size=32,
        seed=seed,
        contrastive_memory=config,
        **options,
    )
    return FinancialPredictor(financial_config, dtype=dtype)


def open_output_gain(predictor, *, seed=19, scale=0.2):
    weights = predictor.state_dict()
    shape = weights["contrastive_memory.out_projection.weight"].shape
    dtype = weights["contrastive_memory.out_projection.weight"].dtype
    generator = torch.Generator().manual_seed(seed)
    weights["contrastive_memory.out_projection.weight"] = scale * torch.randn(
        shape, generator=generator, dtype=dtype
    )
    predictor.load_state_dict(weights)
    return predictor


def _autograd_reference(weights, keys, v_pos, prototypes, *, temperature, loss_mode="infonce"):
    local = tuple(w.clone().requires_grad_(True) for w in weights)
    hidden = keys @ local[0].T
    output = F.gelu(hidden) @ local[1].T
    y = keys + F.layer_norm(output, (keys.shape[-1],), eps=LAYER_NORM_EPS)
    y_hat, _ = safe_l2_normalize(y)
    if loss_mode == "infonce":
        pos = (y_hat * v_pos).sum(dim=-1, keepdim=True) / temperature
        neg = (y_hat @ prototypes.T) / temperature
        logits = torch.cat((pos, neg), dim=-1)
        loss = (torch.logsumexp(logits, dim=-1) - logits[:, 0]).mean()
    else:
        loss = (y_hat - v_pos).square().sum(dim=-1).mean()
    gradients = torch.autograd.grad(loss, local, create_graph=True)
    return gradients, loss


# ---------------------------------------------------------------------------
# 1. Contrato, validación, identidad y comportamiento por defecto bit a bit
# ---------------------------------------------------------------------------


def test_default_none_preserves_identity_parameters_outputs_and_gradients_bit_for_bit():
    base = FinancialPredictor(
        FinancialConfig(specification(), variant="mac_online", hidden_size=32, seed=42),
        dtype=torch.float64,
    )
    explicit = make_predictor(None)
    assert "contrastive_memory" not in base.config.identity()
    assert base.config.identity() == explicit.config.identity()
    assert base._config_id() == explicit._config_id()
    assert base._parameter_id == explicit._parameter_id
    assert declared_sources(base.config) == ()
    flows = ["US/AAA", "US/BBB", "US/CCC"]
    batch = decision(base.config, flows, 0)
    out_base = base.prepare(
        batch, base.initial_state(flows, differentiable=True), differentiable=True
    )
    out_exp = explicit.prepare(
        batch, explicit.initial_state(flows, differentiable=True), differentiable=True
    )
    assert torch.equal(out_base.point_predictions, out_exp.point_predictions)
    assert torch.equal(out_base.working_state, out_exp.working_state)
    assert out_base.next_contrastive is None and out_exp.next_contrastive is None
    out_base.point_predictions.square().sum().backward()
    out_exp.point_predictions.square().sum().backward()
    for (name, p_a), (_, p_b) in zip(
        base.named_parameters(), explicit.named_parameters(), strict=True
    ):
        assert (p_a.grad is None) == (p_b.grad is None), name
        if p_a.grad is not None:
            assert torch.equal(p_a.grad, p_b.grad), name


def test_config_validation_parsing_identity_and_predictor_changes():
    parsed = parse_contrastive_memory(dict(prototypes=32, temperature=0.2, mode="infonce"))
    assert parsed == VARIANT
    assert parse_contrastive_memory(None) is None
    assert parse_contrastive_memory(VARIANT) == VARIANT
    assert VARIANT.identity()["kind"] == CONTRASTIVE_MEMORY
    assert VARIANT.identity()["mode"] == "infonce"
    assert SHUFFLED.identity()["role"] == "shuffled_session_values_placebo"
    assert EVAL_FROZEN.identity()["role"] == "eval_frozen_placebo"
    assert L2_SHARED.identity()["role"] == "normalized_l2_shared_ablation"
    assert checked_predictor_changes(dict(contrastive_memory=asdict(VARIANT))) == dict(
        contrastive_memory=asdict(VARIANT)
    )
    for bad in (
        dict(prototypes=1, temperature=0.2, mode="infonce"),
        dict(prototypes=257, temperature=0.2, mode="infonce"),
        dict(prototypes=True, temperature=0.2, mode="infonce"),
        dict(prototypes=32, temperature=0.04, mode="infonce"),
        dict(prototypes=32, temperature=10.5, mode="infonce"),
        dict(prototypes=32, temperature=float("nan"), mode="infonce"),
        dict(prototypes=32, temperature=True, mode="infonce"),
        dict(prototypes=32, temperature=0.2, mode="unknown"),
        dict(prototypes=32, temperature=0.2),
        dict(prototypes=32, temperature=0.2, mode="infonce", extra=1),
        "contrastive_memory_v1",
    ):
        with pytest.raises(ValueError):
            parse_contrastive_memory(bad)
    with pytest.raises(ValueError, match="mac_online"):
        make_predictor(VARIANT, variant="mac_frozen")
    with pytest.raises(ValueError, match="mac_online"):
        make_predictor(VARIANT, variant="mac_disabled")
    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(
            make_predictor(VARIANT).config, local_control=MACProjectionConfig(), dtype=torch.float64
        )
    frozen_model = make_predictor(VARIANT).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="memoria contrastiva"):
        FrozenFinancialConsumer(frozen_model)
    assert "mars_titan.models.titans.contrastive_memory" in declared_sources(frozen_model.config)


def test_all_modes_share_exact_same_parameters_initialization_and_preserve_global_rng():
    rng = torch.get_rng_state().clone()
    control = make_predictor(None)
    models = {mode: make_predictor(ContrastiveMemoryConfig(mode=mode)) for mode in MODES}
    assert torch.equal(torch.get_rng_state(), rng)
    control_state = control.state_dict()
    for mode, model in models.items():
        state = model.state_dict()
        for name, value in control_state.items():
            if name.split(".", 1)[0] in SHARED_PREFIXES and torch.is_tensor(value):
                assert torch.equal(state[name], value), (mode, name)
    ref_params = dict(models["infonce"].named_parameters())
    for mode in MODES[1:]:
        other_params = dict(models[mode].named_parameters())
        assert list(other_params.keys()) == list(ref_params.keys())
        for name, value in ref_params.items():
            assert torch.equal(other_params[name], value), (mode, name)
        assert models[mode].config.identity() != models["infonce"].config.identity()
        with pytest.raises(ValueError):
            models[mode].load_state_dict(models["infonce"].state_dict())


# ---------------------------------------------------------------------------
# 2. Ecuaciones exactas: derivadas analíticas, metagradientes y prototipos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_mode", ["infonce", "l2_shared"])
@pytest.mark.parametrize("rows,prototypes", [(6, 4), (4, 8), (12, 5)])
def test_analytic_inner_gradients_and_outer_metagradients_match_autograd_in_fp64(
    loss_mode, rows, prototypes
):
    dim = 8
    gen = torch.Generator().manual_seed(101 + rows + prototypes)
    w1 = (torch.randn(dim, dim, generator=gen, dtype=torch.float64) * 0.3).requires_grad_(True)
    w2 = (torch.randn(dim, dim, generator=gen, dtype=torch.float64) * 0.3).requires_grad_(True)
    keys_raw = torch.randn(rows, dim, generator=gen, dtype=torch.float64).requires_grad_(True)
    vals_raw = torch.randn(rows, dim, generator=gen, dtype=torch.float64).requires_grad_(True)
    query_probe = torch.randn(rows, dim, generator=gen, dtype=torch.float64)

    def run(use_analytic):
        k_hat, _ = safe_l2_normalize(keys_raw)
        v_hat, _ = safe_l2_normalize(vals_raw)
        protos = session_prototypes(v_hat, prototypes)
        if use_analytic:
            grads, loss = contrastive_inner_gradients(
                (w1, w2), k_hat, v_hat, protos, temperature=0.25, loss_mode=loss_mode
            )
        else:
            grads, loss = _autograd_reference(
                (w1, w2), k_hat, v_hat, protos, temperature=0.25, loss_mode=loss_mode
            )
        next_w1 = 0.9 * w1 - 0.05 * grads[0]
        next_w2 = 0.9 * w2 - 0.05 * grads[1]
        readout = query_probe @ next_w1.T + F.gelu(query_probe @ next_w1.T) @ next_w2.T
        outer = readout.square().sum()
        meta = torch.autograd.grad(outer, (w1, w2, keys_raw, vals_raw))
        return loss.detach(), tuple(g.detach() for g in grads), meta

    loss_a, grads_a, meta_a = run(True)
    loss_r, grads_r, meta_r = run(False)
    torch.testing.assert_close(loss_a, loss_r, rtol=1e-13, atol=1e-14)
    for left, right in zip(grads_a, grads_r, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-13)
    for left, right in zip(meta_a, meta_r, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)


def test_session_prototypes_keep_all_rows_when_small_and_preserve_distinct_clusters_when_large():
    small = make_tokens(5, hidden=8, seed=1)
    small_hat, _ = safe_l2_normalize(small)
    assert torch.equal(session_prototypes(small_hat, 8), small_hat)
    assert torch.equal(session_prototypes(small_hat, 5), small_hat)

    # Cuatro clústeres ortogonales mezclados entre 64 activos: los prototipos deben cubrir los 4.
    basis = torch.eye(8, dtype=torch.float64)[:4]
    gen = torch.Generator().manual_seed(23)
    rows = []
    for idx in range(64):
        cluster = basis[idx % 4]
        rows.append(cluster + 0.05 * torch.randn(8, generator=gen, dtype=torch.float64))
    v_hat, _ = safe_l2_normalize(torch.stack(rows))
    protos = session_prototypes(v_hat, 8)
    assert protos.shape == (8, 8)
    # Cada clúster verdadero tiene al menos un prototipo con similitud coseno > 0,95.
    max_cos = (basis @ protos.T).max(dim=1).values
    assert (max_cos > 0.95).all(), max_cos
    # Los prototipos no colapsan a la media común (similitud mutua media muy inferior a 1).
    off_diag = protos @ protos.T - torch.eye(8, dtype=torch.float64)
    assert off_diag.mean().item() < 0.5


def test_infonce_suppresses_common_market_mode_compared_to_l2_shared():
    # Construcción sintética con modo común fuerte c y señal transversal discriminativa d_i.
    dim, count = 16, 40
    gen = torch.Generator().manual_seed(31)
    common = torch.zeros(dim, dtype=torch.float64)
    common[0] = 1.0
    keys_raw = torch.randn(count, dim, generator=gen, dtype=torch.float64)
    keys, _ = safe_l2_normalize(keys_raw)
    # Valores = 3.0 * modo común + señal discriminativa alineada con claves en dims 1..15.
    discriminative = keys.clone()
    discriminative[:, 0] = 0.0
    v_hat, _ = safe_l2_normalize(3.0 * common.unsqueeze(0) + discriminative)
    protos = session_prototypes(v_hat, 16)
    # Memoria que arranca prediciendo exactamente el modo común c para todos los activos:
    # con L2 el gradiente en dirección común sigue dominando, mientras InfoNCE empuja
    # estrictamente hacia la componente discriminativa ortogonal a c.
    w1 = torch.zeros(dim, dim, dtype=torch.float64)
    w2 = torch.zeros(dim, dim, dtype=torch.float64)
    # Inicializar pequeña perturbación para que LN tenga varianza no nula.
    w1 = 0.05 * torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    w2 = 0.05 * torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    grads_nce, _ = contrastive_inner_gradients(
        (w1, w2), keys, v_hat, protos, temperature=0.2, loss_mode="infonce"
    )
    grads_l2, _ = contrastive_inner_gradients(
        (w1, w2), keys, v_hat, protos, temperature=0.2, loss_mode="l2_shared"
    )
    # Energía del gradiente en la fila 0 (modo común) frente a las filas 1..15 (discriminativas).
    common_ratio_nce = grads_nce[1][0].square().sum() / grads_nce[1].square().sum()
    common_ratio_l2 = grads_l2[1][0].square().sum() / grads_l2[1].square().sum()
    assert common_ratio_l2 > 0.5
    assert common_ratio_nce < 0.1 * common_ratio_l2


def test_low_temperature_fp32_analytic_gradient_avoids_cancellation_and_matches_fp64():
    dim, rows, prototypes = 16, 20, 8
    gen = torch.Generator().manual_seed(2026)
    w1_64 = (torch.randn(dim, dim, generator=gen, dtype=torch.float64) * 0.25).requires_grad_(True)
    w2_64 = (torch.randn(dim, dim, generator=gen, dtype=torch.float64) * 0.25).requires_grad_(True)
    k_raw = torch.randn(rows, dim, generator=gen, dtype=torch.float64)
    k_64, _ = safe_l2_normalize(k_raw)
    # Hacer v_pos casi colineal con y_hat para forzar p_{i,0} ≈ 1 a temperatura mínima τ = 0.05.
    with torch.no_grad():
        h = F.gelu(k_64 @ w1_64.T) @ w2_64.T
        y_init, _ = safe_l2_normalize(k_64 + F.layer_norm(h, (dim,), eps=LAYER_NORM_EPS))
        v_64, _ = safe_l2_normalize(
            y_init + 0.02 * torch.randn(rows, dim, generator=gen, dtype=torch.float64)
        )
    protos_64 = session_prototypes(v_64, prototypes)
    grads_ref, loss_ref = _autograd_reference(
        (w1_64, w2_64), k_64, v_64, protos_64, temperature=0.05, loss_mode="infonce"
    )
    grads_64, loss_64 = contrastive_inner_gradients(
        (w1_64, w2_64), k_64, v_64, protos_64, temperature=0.05, loss_mode="infonce"
    )
    torch.testing.assert_close(loss_64, loss_ref, rtol=1e-13, atol=1e-14)
    for left, right in zip(grads_64, grads_ref, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-13)

    # En FP32 con τ = 0.05 el gradiente analítico sigue finito, no nulo y cercano al de FP64.
    grads_32, loss_32 = contrastive_inner_gradients(
        (w1_64.detach().float(), w2_64.detach().float()),
        k_64.float(),
        v_64.float(),
        protos_64.float(),
        temperature=0.05,
        loss_mode="infonce",
    )
    assert torch.isfinite(loss_32)
    for g32, g64 in zip(grads_32, grads_ref, strict=True):
        assert torch.isfinite(g32).all() and g32.abs().sum() > 0
        torch.testing.assert_close(g32.double(), g64.detach(), rtol=2e-4, atol=2e-5)


# ---------------------------------------------------------------------------
# 3. Inicio en cero, primer gradiente y flujo completo en BPTT
# ---------------------------------------------------------------------------


def test_predictor_starts_bit_for_bit_at_control_and_receives_first_step_gradient():
    control, batch = build_scaled(technical_branch=asdict(DENSE), contrastive_memory=None)
    model, _ = build_scaled(technical_branch=asdict(DENSE), contrastive_memory=asdict(VARIANT))
    opened(control.technical_branch)
    opened(model.technical_branch)
    control._seal_parameters()
    model._seal_parameters()
    assert torch.count_nonzero(model.contrastive_memory.out_projection.weight) == 0

    flows = list(batch.flow_ids)
    c_state, m_state = control.initial_state(flows), model.initial_state(flows)
    cm_state = model.initial_contrastive_memory()
    for step in range(2):
        step_batch = decision(model.config, flows, step)
        c_out = control.prepare(step_batch, c_state, session_complete=True)
        m_out = model.prepare(step_batch, m_state, contrastive=cm_state, session_complete=True)
        assert torch.equal(m_out.point_predictions, c_out.point_predictions)
        assert torch.equal(m_out.working_state, c_out.working_state)
        c_state, m_state = c_out.next_state, m_out.next_state
        cm_state = model.commit_contrastive_memory(m_out.next_contrastive)

    # Primer paso con cabeza de cuantiles en mediana cero: el gradiente llega a out_projection.
    fresh, _ = build_scaled(technical_branch=asdict(DENSE), contrastive_memory=asdict(VARIANT))
    flows = list(batch.flow_ids)
    step0 = decision(fresh.config, flows, 0)
    out0 = fresh.prepare(
        step0,
        fresh.initial_state(flows, differentiable=True),
        contrastive=fresh.initial_contrastive_memory(differentiable=True),
        differentiable=True,
        session_complete=True,
    )
    assert torch.equal(out0.point_predictions, torch.zeros_like(out0.point_predictions))
    targets = torch.tensor([0.02, -0.01], dtype=torch.float64)
    ic_term = SessionICTerm(weight=3.0, minimum_rows=3)
    sessions = [
        (f.split("/", 1)[0], at) for f, at in zip(step0.flow_ids, step0.prediction_at, strict=True)
    ]
    loss = pinball_loss(out0.quantiles, targets) + ic_term.term(
        out0.point_predictions, targets, sessions
    )
    loss.backward()
    out_grad = fresh.contrastive_memory.out_projection.weight.grad
    assert out_grad is not None and torch.isfinite(out_grad).all() and out_grad.abs().sum() > 0
    for name, param in fresh.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_two_session_bptt_differentiates_every_contrastive_parameter_once_output_opens():
    model = open_output_gain(make_predictor(VARIANT))
    flows = ["US/AAA", "US/BBB", "US/CCC", "US/DDD"]
    # Calentamiento previo de una sesión para que el momentum S_0 sea no nulo (como tras warmup),
    # seguido de un tramo de truncamiento 2: así `eta_projection` también recibe gradiente.
    state0 = model.initial_state(flows, differentiable=False)
    cm0 = model.initial_contrastive_memory(differentiable=False)
    warm = model.prepare(decision(model.config, flows, 0), state0, contrastive=cm0)
    cm_warm = model.commit_contrastive_memory(warm.next_contrastive, differentiable=False)
    state1 = warm.next_state

    # También probamos desde estado inicial puro (3 sesiones) para cubrir initial_weights.
    fresh_state = model.initial_state(flows, differentiable=True)
    fresh_cm = model.initial_contrastive_memory(differentiable=True)
    out0 = model.prepare(
        decision(model.config, flows, 0), fresh_state, contrastive=fresh_cm, differentiable=True
    )
    cm_a = model.commit_contrastive_memory(out0.next_contrastive, differentiable=True)
    out1 = model.prepare(
        decision(model.config, flows, 1), out0.next_state, contrastive=cm_a, differentiable=True
    )
    cm_b = model.commit_contrastive_memory(out1.next_contrastive, differentiable=True)
    out2 = model.prepare(
        decision(model.config, flows, 2), out1.next_state, contrastive=cm_b, differentiable=True
    )
    loss = out2.point_predictions.square().sum()
    loss.backward()
    contrastive_params = [
        (name, param)
        for name, param in model.named_parameters()
        if name.startswith("contrastive_memory.")
    ]
    assert len(contrastive_params) == 12
    for name, param in contrastive_params:
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name

    # Verificar también con truncamiento 2 tras calentamiento (el caso real de la campaña).
    model.zero_grad(set_to_none=True)
    seg1 = model.prepare(
        decision(model.config, flows, 1), state1, contrastive=cm_warm, differentiable=True
    )
    cm_seg1 = model.commit_contrastive_memory(seg1.next_contrastive, differentiable=True)
    seg2 = model.prepare(
        decision(model.config, flows, 2), seg1.next_state, contrastive=cm_seg1, differentiable=True
    )
    seg2.point_predictions.square().sum().backward()
    for name, param in contrastive_params:
        if not name.startswith("contrastive_memory.initial_weights."):
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, name


# ---------------------------------------------------------------------------
# 4. Causalidad estricta, aislamiento entre mercados e invarianza a bloques
# ---------------------------------------------------------------------------


def test_no_future_or_same_session_leakage_and_strict_market_isolation():
    module = make_module("infonce", gain=5)
    flows = ("CN/600000.SS", "CN/600001.SS", "US/AAA", "US/BBB", "US/CCC")
    rows0 = make_tokens(5, seed=10)
    rows0_perturbed = rows0.clone()
    rows0_perturbed[4] += 5.0  # Cambiar US/CCC en la sesión 0.

    read_base, state_base = step_session(module, module.initial_state("p"), rows0, flows, START)
    read_pert, state_pert = step_session(
        module, module.initial_state("p"), rows0_perturbed, flows, START
    )
    # En la sesión 0, cambiar US/CCC NO puede cambiar ni un bit la lectura de ninguna otra fila.
    assert torch.equal(read_base[:4], read_pert[:4])

    # En la sesión 1, US refleja la sesión 0 confirmada y CN queda bit a bit igual.
    rows1 = make_tokens(5, seed=20)
    read1_base, _ = step_session(module, state_base, rows1, flows, START + DAY)
    read1_pert, _ = step_session(module, state_pert, rows1, flows, START + DAY)
    assert torch.equal(read1_base[:2], read1_pert[:2])  # CN intacto.
    assert not torch.equal(read1_base[2:], read1_pert[2:])  # US actualizado causalmente.

    # Modificar una sesión futura (sesión 2) jamás puede afectar a la sesión 1.
    rows1_copy = rows1.clone()
    read1_again, state1 = step_session(module, state_base, rows1_copy, flows, START + DAY)
    assert torch.equal(read1_again, read1_base)
    rows2 = make_tokens(5, seed=30) * 100.0
    step_session(module, state1, rows2, flows, START + 2 * DAY)
    assert torch.equal(read1_again, read1_base)


def test_block_partition_and_row_permutation_leave_committed_state_bit_for_bit_identical():
    module = make_module("infonce", gain=7)
    flows = ("US/AAA", "US/BBB", "US/CCC", "US/DDD", "US/EEE")
    rows = make_tokens(5, seed=17)
    state = module.initial_state("p")
    whole_read, committed = step_session(module, state, rows, flows, START)

    order = (3, 0, 4, 1, 2)
    pending = state
    for block in (order[:2], order[2:]):
        idx = torch.tensor(block)
        part_read, pending = module.read(
            rows[idx], tuple(flows[i] for i in block), (START,) * len(block), pending
        )
        torch.testing.assert_close(part_read, whole_read[idx], rtol=1e-13, atol=1e-15)
    split_committed = module.commit(pending)
    for left, right in zip(
        (*split_committed.markets[0].weights, *split_committed.markets[0].momentum),
        (*committed.markets[0].weights, *committed.markets[0].momentum),
        strict=True,
    ):
        assert torch.equal(left, right)


def test_session_protocol_rejects_reopened_mixed_duplicate_or_uncommitted_states():
    module = make_module("infonce", gain=7)
    state = module.initial_state("p")
    rows = make_tokens(2, seed=3)
    with pytest.raises(ValueError, match="una sesión"):
        module.read(rows, ("US/AAA", "US/BBB"), (START, START + DAY), state)
    _, pending = module.read(rows, ("US/AAA", "CN/600000.SS"), (START, START + 1), state)
    with pytest.raises(ValueError, match="dos veces"):
        module.read(rows[:1], ("US/AAA",), (START,), pending)
    with pytest.raises(ValueError, match="pendiente"):
        module.read(rows[:1], ("US/BBB",), (START + DAY,), pending)
    with pytest.raises(ValueError, match="confirmadas"):
        module.detach(pending, "p")
    committed = module.commit(pending)
    with pytest.raises(ValueError, match="ya se confirmó"):
        module.read(rows[:1], ("US/BBB",), (START,), committed)
    with pytest.raises(ValueError, match="no corresponde"):
        module.validate(committed, "otra")


# ---------------------------------------------------------------------------
# 5. Controles placebo (shuffled_values, eval_frozen, l2_shared)
# ---------------------------------------------------------------------------


def test_shuffled_values_placebo_uses_same_prototypes_and_deranges_values_deterministically():
    infonce = make_module("infonce", gain=9)
    shuffled = make_module("shuffled_values", gain=9)
    rows = make_tokens(6, seed=41)
    flows = tuple(f"US/A{i}" for i in range(6))

    rng_before = torch.get_rng_state().clone()
    _, s_nce = step_session(infonce, infonce.initial_state("p"), rows, flows, START)
    _, s_shuf_a = step_session(shuffled, shuffled.initial_state("p"), rows, flows, START)
    _, s_shuf_b = step_session(shuffled, shuffled.initial_state("p"), rows, flows, START)
    assert torch.equal(torch.get_rng_state(), rng_before)

    # Determinista consigo mismo y distinto del candidato con asociación verdadera.
    for a, b in zip(s_shuf_a.markets[0].weights, s_shuf_b.markets[0].weights, strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(s_nce.markets[0].weights[0], s_shuf_a.markets[0].weights[0])


def test_eval_frozen_placebo_updates_in_train_and_freezes_in_eval():
    infonce = make_module("infonce", gain=9)
    frozen = make_module("eval_frozen", gain=9)
    rows = make_tokens(5, seed=55)
    flows = tuple(f"US/A{i}" for i in range(5))

    # En train() coincide bit a bit con infonce.
    infonce.train()
    frozen.train()
    _, s_nce = step_session(infonce, infonce.initial_state("p"), rows, flows, START)
    _, s_frz_train = step_session(frozen, frozen.initial_state("p"), rows, flows, START)
    for a, b in zip(s_nce.markets[0].weights, s_frz_train.markets[0].weights, strict=True):
        assert torch.equal(a, b)
    assert s_frz_train.markets[0].steps == 1

    # En eval() congela los pesos rápidos en M0 y avanza committed_at sin escribir.
    frozen.eval()
    init = frozen.initial_state("p")
    _, s_frz_eval = step_session(frozen, init, rows, flows, START)
    for a, b in zip(init.markets[0].weights, s_frz_eval.markets[0].weights, strict=True):
        assert torch.equal(a, b)
    assert s_frz_eval.markets[0].steps == 0
    assert s_frz_eval.committed_at == (START, -1)


# ---------------------------------------------------------------------------
# 6. Numérica estricta (FP32, FP64, determinismo, autocast, entradas degeneradas)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strict_numerics_determinism_and_degenerate_inputs(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        module = make_module("infonce", gain=3, dtype=dtype)
        flows = ("US/AAA", "US/BBB", "US/CCC", "US/DDD")
        # Casos límite: todos ceros, todas filas constantes idénticas, fila única, magnitudes altas.
        cases = [
            (torch.zeros(4, 8, dtype=dtype), flows),
            (torch.ones(4, 8, dtype=dtype) * 2.5, flows),
            (make_tokens(1, seed=1, dtype=dtype), ("US/AAA",)),
            (
                make_tokens(4, seed=2, dtype=dtype) * (1e12 if dtype == torch.float32 else 1e100),
                flows,
            ),
        ]
        for rows, case_flows in cases:
            x = rows.clone().requires_grad_(True)
            init = module.initial_state("p", differentiable=True)
            read1, st1 = step_session(module, init, x, case_flows, START, differentiable=True)
            read2, _ = step_session(
                module, st1, x.detach().clone(), case_flows, START + DAY, differentiable=True
            )
            assert torch.isfinite(read1).all() and torch.isfinite(read2).all()
            grads = torch.autograd.grad(
                (read1.square().sum() + read2.square().sum()),
                (x, *module.parameters()),
                allow_unused=True,
            )
            for grad in grads:
                assert grad is None or torch.isfinite(grad).all()

        # Determinismo exacto en dos ejecuciones independientes.
        x = make_tokens(6, seed=9, dtype=dtype)
        f6 = tuple(f"US/S{i}" for i in range(6))
        r_a, s_a = step_session(module, module.initial_state("p"), x, f6, START)
        r_b, s_b = step_session(module, module.initial_state("p"), x, f6, START)
        assert torch.equal(r_a, r_b)
        assert torch.equal(s_a.markets[0].weights[0], s_b.markets[0].weights[0])
    finally:
        torch.use_deterministic_algorithms(previous)

    # Rechazo explícito de autocast.
    module32 = make_module("infonce", gain=3, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            module32.read(
                make_tokens(2, dtype=torch.float32),
                ("US/AAA", "US/BBB"),
                (START, START),
                module32.initial_state("p"),
            )


def test_scalar_zero_head_constant_prediction_produces_finite_gradients():
    model, batch = build_scaled(head="scalar", contrastive_memory=asdict(VARIANT))
    flows = list(batch.flow_ids)
    out = model.prepare(
        decision(model.config, flows, 0),
        model.initial_state(flows, differentiable=True),
        contrastive=model.initial_contrastive_memory(differentiable=True),
        differentiable=True,
    )
    assert torch.equal(out.point_predictions, torch.zeros_like(out.point_predictions))
    loss = (out.point_predictions - torch.tensor([0.05, -0.03], dtype=torch.float64)).abs().mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), name


# ---------------------------------------------------------------------------
# 7. Checkpointing, reinicio de fase y paridad en ChronologicalTrainer (con y sin acumulación)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def streams(tmp_path_factory):
    from tests.training.test_financial_run import corpus

    return corpus(tmp_path_factory.mktemp("infonce"))[1]


def corpus_predictor(streams, config=VARIANT, *, gain=True):
    spec = streams["train"].specification()
    model = FinancialPredictor(
        FinancialConfig(
            spec,
            variant="mac_online",
            hidden_size=32,
            seed=42,
            contrastive_memory=config,
        ),
        dtype=torch.float64,
    )
    return open_output_gain(model) if (config is not None and gain) else model


def test_export_restore_roundtrip_bit_for_bit_and_rejects_tampered_payload():
    module = make_module("infonce", gain=5)
    flows = ("CN/600000.SS", "US/AAA", "US/BBB", "US/CCC")
    _, state = step_session(module, module.initial_state("p"), make_tokens(4, seed=1), flows, START)
    payload = module.export_state(state)
    restored = module.restore_state(payload, "p", device=torch.device("cpu"))
    assert restored.committed_at == state.committed_at

    rows2 = make_tokens(4, seed=2)
    exp_r, exp_s = step_session(module, state, rows2, flows, START + DAY)
    act_r, act_s = step_session(module, restored, rows2, flows, START + DAY)
    assert torch.equal(act_r, exp_r)
    for m_a, m_b in zip(act_s.markets, exp_s.markets, strict=True):
        assert m_a.steps == m_b.steps
        for t_a, t_b in zip(
            (*m_a.weights, *m_a.momentum), (*m_b.weights, *m_b.momentum), strict=True
        ):
            assert torch.equal(t_a, t_b)

    # Rechazar otro modo o payload corrupto.
    with pytest.raises(ValueError):
        make_module("shuffled_values", gain=5).restore_state(
            payload, "p", device=torch.device("cpu")
        )
    bad = copy.deepcopy(payload)
    bad["markets"][0]["steps"] = -1
    with pytest.raises(ValueError):
        module.restore_state(bad, "p", device=torch.device("cpu"))


def test_trainer_phase_reset_checkpoint_resume_and_accumulation_parity(streams, tmp_path):
    from tests.training.test_financial_run import RecordingOptimizer, StopAtStep, named_records

    def make_trainer(
        output, *, config=VARIANT, accumulation_rows=None, observe_without_graph=False, block_rows=2
    ):
        recipe = financial_run.ChronologicalRecipe(
            truncation=2,
            epochs=1,
            block_rows=block_rows,
            accumulation_rows=accumulation_rows,
            observe_without_graph=observe_without_graph,
            checkpoint_updates=1,
            selection=dict(metric="session_mae", patience=5, min_delta=0.0),
        )
        return financial_run.ChronologicalTrainer(
            corpus_predictor(streams, config),
            recipe,
            train=streams["train"],
            validation=streams["validation"],
            output=output,
            optimizer_factory=RecordingOptimizer,
            audit=True,
        )

    # 1. Paridad exacta entre sin acumulación, con acumulación y con observe_without_graph=True.
    plain = make_trainer(tmp_path / "plain", accumulation_rows=None, block_rows=8)
    accum = make_trainer(tmp_path / "accum", accumulation_rows=8, block_rows=8)
    nograph = make_trainer(
        tmp_path / "nograph", accumulation_rows=8, observe_without_graph=True, block_rows=8
    )
    rep_plain = plain.run()
    rep_accum = accum.run()
    rep_nograph = nograph.run()
    assert rep_plain["status"] == rep_accum["status"] == rep_nograph["status"] == "completed"
    assert plain.audit == accum.audit == nograph.audit

    rec_plain = named_records(plain)
    rec_accum = named_records(accum)
    rec_nograph = named_records(nograph)
    assert len(rec_plain) == len(rec_accum) == len(rec_nograph) > 2
    for step_p, step_a, step_n in zip(rec_plain, rec_accum, rec_nograph, strict=True):
        assert step_p.keys() == step_a.keys() == step_n.keys()
        for name in step_p:
            g_p, g_a, g_n = step_p[name], step_a[name], step_n[name]
            assert (g_p is None) == (g_a is None) == (g_n is None), name
            if g_p is not None:
                torch.testing.assert_close(g_a, g_p, rtol=1e-12, atol=1e-13)
                torch.testing.assert_close(g_n, g_p, rtol=1e-12, atol=1e-13)

    # 2. Pausa y reanudación desde checkpoint en mitad de época (con acumulación activa!).
    paused = make_trainer(tmp_path / "resumed", accumulation_rows=8, block_rows=8)
    stop = StopAtStep(2)
    stop.trainer = paused
    assert paused.run(stop=stop)["status"] == "paused"
    resumed = make_trainer(tmp_path / "resumed", accumulation_rows=8, block_rows=8)
    rep_resumed = resumed.run(resume=True)
    assert rep_resumed["status"] == "completed"
    assert rep_resumed["history"] == rep_accum["history"]

    # 3. Reinicio limpio entre fases: evaluar tras entrenar coincide con evaluar fresco.
    fresh_eval = make_trainer(tmp_path / "fresh_eval", block_rows=8)
    metrics_after_train = plain.evaluate(streams["validation"])
    metrics_fresh = fresh_eval.evaluate(streams["validation"])
    assert metrics_after_train == metrics_fresh

    # 4. Sin contrastive_memory, identidad y payload del entrenador no añaden claves nuevas.
    base_a = make_trainer(tmp_path / "base_a", config=None, block_rows=8)
    assert "contrastive_memory" not in base_a.identity["predictor"]
    assert "mars_titan.models.titans.contrastive_memory" not in base_a.identity["implementation"]
    empty_run = financial_run._Pass()
    assert "contrastive" not in base_a._export(empty_run)


def test_accumulation_replay_reattaches_initial_weights_without_warmup(tmp_path):
    """Sin calentamiento previo, el replay debe reenganchar initial_weights al grafo."""
    from tests.training.chronological_fixture import phases
    from tests.training.test_financial_run import (
        RecordingOptimizer,
        StopAtStep,
        corpus,
        named_records,
    )

    train_phase, val_phase = phases(warmup="2022-11-15", train_decisions="2022-11-15")
    _, no_warmup = corpus(tmp_path / "no_warmup", train=train_phase, validation=val_phase)

    def make(name, *, accumulation_rows):
        recipe = financial_run.ChronologicalRecipe(
            truncation=2,
            epochs=1,
            block_rows=8,
            accumulation_rows=accumulation_rows,
            checkpoint_updates=1,
            selection=dict(metric="session_mae", patience=5, min_delta=0.0),
        )
        return financial_run.ChronologicalTrainer(
            corpus_predictor(no_warmup, VARIANT),
            recipe,
            train=no_warmup["train"],
            validation=no_warmup["validation"],
            output=tmp_path / name,
            optimizer_factory=RecordingOptimizer,
            audit=True,
        )

    direct = make("direct", accumulation_rows=None)
    replayed = make("replayed", accumulation_rows=8)
    for engine in (direct, replayed):
        stop = StopAtStep(1)
        stop.trainer = engine
        assert engine.run(stop=stop)["status"] == "paused"
    first_direct, first_replayed = named_records(direct)[0], named_records(replayed)[0]
    for name in (
        "contrastive_memory.initial_weights.0",
        "contrastive_memory.initial_weights.1",
    ):
        assert first_direct[name] is not None and first_direct[name].abs().sum() > 0, name
        assert first_replayed[name] is not None, name
        torch.testing.assert_close(first_replayed[name], first_direct[name], rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# 8. Grafo constante con N, convivencia con CRRM y holograma, valores fijados
#    y declaración de una sola variable sobre us-campaign-v12-screen3.json
# ---------------------------------------------------------------------------


def _graph_nodes(tensor):
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(parent for parent, _ in node.next_functions)
    return len(seen)


def test_the_graph_of_a_committed_session_does_not_grow_with_its_rows():
    """El grafo de lectura y escritura por sesión es O(1) en el número de filas N."""
    sizes_w1, sizes_w2 = {}, {}
    for rows in (8, 64, 256):
        module = make_module("infonce", hidden=8, prototypes=4, gain=13)
        flows = tuple(f"US/A{idx:04d}" for idx in range(rows))
        tokens = make_tokens(rows, hidden=8, seed=100 + rows).requires_grad_(True)
        _, pending = module.read(
            tokens, flows, (START,) * rows, module.initial_state("p", differentiable=True)
        )
        committed = module.commit(pending, differentiable=True)
        w1, w2 = committed.markets[0].weights
        sizes_w1[rows] = _graph_nodes(w1)
        sizes_w2[rows] = _graph_nodes(w2)
        (grad_tokens,) = torch.autograd.grad(w1.square().sum() + w2.square().sum(), tokens)
        assert torch.isfinite(grad_tokens).all() and grad_tokens.abs().sum() > 0
    assert sizes_w1[8] == sizes_w1[64] == sizes_w1[256], sizes_w1
    assert sizes_w2[8] == sizes_w2[64] == sizes_w2[256], sizes_w2


def test_pinned_values_match_standalone_contrastive_memory_on_integration_base():
    """Con solo contrastive_memory activada, los valores y huellas coinciden exactamente."""
    expected = {
        "infonce": dict(
            config=VARIANT,
            preds0=[
                0.010705658743559374,
                -0.15616812363710925,
                -0.10876876024625466,
                -0.18638065458747424,
            ],
            preds1=[
                -0.18253711679461598,
                -0.18681744794340496,
                0.1229620536156161,
                0.05121675708546277,
            ],
            w1_norm=5.7008006906711755,
            w2_norm=5.590790604829608,
            out_grad_norm=0.39905061683203563,
            key_grad_norm=0.002051267400312077,
        ),
        "l2_shared": dict(
            config=L2_SHARED,
            preds0=[
                0.010705658743559374,
                -0.15616812363710925,
                -0.10876876024625466,
                -0.18638065458747424,
            ],
            preds1=[
                -0.18258688458105055,
                -0.18688771268959464,
                0.12237048068030797,
                0.05213106401501291,
            ],
            w1_norm=5.70079094383389,
            w2_norm=5.590793453450131,
            out_grad_norm=0.39894592834864134,
            key_grad_norm=0.0013308793456064739,
        ),
    }
    flows = ["US/AAA", "US/BBB", "US/CCC", "US/DDD"]
    for ref in expected.values():
        model = open_output_gain(
            make_predictor(ref["config"], seed=42, dtype=torch.float64), seed=19, scale=0.2
        )
        explicit_off = open_output_gain(
            make_predictor(
                ref["config"],
                seed=42,
                dtype=torch.float64,
                cross_sectional_rank_memory=None,
                session_hologram=None,
            ),
            seed=19,
            scale=0.2,
        )
        assert model._parameter_id == explicit_off._parameter_id
        assert (
            model._parameter_id
            == "c0f2d84660ea46eabf49af2d2ffecaf2b466d49a99287060ea02c387faed04b8"
        )
        assert "cross_sectional_rank_memory" not in model.config.identity()
        assert "session_hologram" not in model.config.identity()
        assert declared_sources(model.config) == ("mars_titan.models.titans.contrastive_memory",)

        st = model.initial_state(flows, differentiable=True)
        cm = model.initial_contrastive_memory(differentiable=True)
        out0 = model.prepare(
            decision(model.config, flows, 0), st, contrastive=cm, differentiable=True
        )
        cm1 = model.commit_contrastive_memory(out0.next_contrastive, differentiable=True)
        out1 = model.prepare(
            decision(model.config, flows, 1), out0.next_state, contrastive=cm1, differentiable=True
        )
        out1.point_predictions.square().sum().backward()
        torch.testing.assert_close(
            out0.point_predictions.detach(),
            torch.tensor(ref["preds0"], dtype=torch.float64),
            rtol=1e-13,
            atol=1e-14,
        )
        torch.testing.assert_close(
            out1.point_predictions.detach(),
            torch.tensor(ref["preds1"], dtype=torch.float64),
            rtol=1e-13,
            atol=1e-14,
        )
        assert cm1.markets[0].weights[0].detach().norm().item() == pytest.approx(
            ref["w1_norm"], rel=1e-13, abs=1e-14
        )
        assert cm1.markets[0].weights[1].detach().norm().item() == pytest.approx(
            ref["w2_norm"], rel=1e-13, abs=1e-14
        )
        assert model.contrastive_memory.out_projection.weight.grad.norm().item() == pytest.approx(
            ref["out_grad_norm"], rel=1e-13, abs=1e-14
        )
        assert model.contrastive_memory.key_projection.weight.grad.norm().item() == pytest.approx(
            ref["key_grad_norm"], rel=1e-13, abs=1e-14
        )


def test_coexistence_and_before_head_order_with_hologram_and_crrm(streams, tmp_path):
    """Convivencia real y orden exacto: holograma -> memoria contrastiva -> CRRM -> cabeza."""
    from tests.training.test_financial_run import RecordingOptimizer, StopAtStep, named_records

    hologram_after_tech = SessionHologramConfig(bands=4, phase_dim=8, position="after_technical")
    hologram_after_mac = SessionHologramConfig(bands=4, phase_dim=8, position="after_mac")
    crrm_cfg = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )

    # 1. Comprobar el orden exacto tras la rama técnica mediante ganchos de forward.
    probe, probe_batch = build_scaled(
        technical_branch=asdict(DENSE),
        session_hologram=asdict(hologram_after_tech),
        contrastive_memory=asdict(VARIANT),
        cross_sectional_rank_memory=asdict(crrm_cfg),
    )
    order = []
    handles = [
        probe.technical_branch.register_forward_hook(lambda *_: order.append("technical_branch")),
        probe.session_hologram.register_forward_hook(lambda *_: order.append("session_hologram")),
        probe.contrastive_memory.out_projection.register_forward_hook(
            lambda *_: order.append("contrastive_memory")
        ),
        probe.cross_sectional_rank_memory.out.register_forward_hook(
            lambda *_: order.append("cross_sectional_rank_memory")
        ),
        probe.head.register_forward_hook(lambda *_: order.append("head")),
    ]
    try:
        probe.prepare(
            probe_batch,
            probe.initial_state(probe_batch.flow_ids),
            session_complete=True,
            contrastive=probe.initial_contrastive_memory(),
            rank_memory=probe.initial_rank_memory(),
        )
    finally:
        for handle in handles:
            handle.remove()
    assert order == [
        "technical_branch",
        "session_hologram",
        "contrastive_memory",
        "cross_sectional_rank_memory",
        "head",
    ]

    # 2. Convivencia en ChronologicalTrainer: paridad directa vs replay y reanudación.
    def make_joint():
        model = FinancialPredictor(
            FinancialConfig(
                streams["train"].specification(),
                variant="mac_online",
                hidden_size=32,
                seed=42,
                session_hologram=hologram_after_mac,
                contrastive_memory=VARIANT,
                cross_sectional_rank_memory=crrm_cfg,
            ),
            dtype=torch.float64,
        )
        with torch.no_grad():
            model.session_hologram.gate.fill_(0.1)
            model.cross_sectional_rank_memory.out.weight.fill_(0.05)
        open_output_gain(model, seed=19, scale=0.15)
        return model

    def make_trainer(output, *, accumulation_rows=None, observe_without_graph=False):
        recipe = financial_run.ChronologicalRecipe(
            truncation=2,
            epochs=1,
            block_rows=8,
            accumulation_rows=accumulation_rows,
            observe_without_graph=observe_without_graph,
            checkpoint_updates=1,
            selection=dict(metric="session_mae", patience=5, min_delta=0.0),
        )
        return financial_run.ChronologicalTrainer(
            make_joint(),
            recipe,
            train=streams["train"],
            validation=streams["validation"],
            output=output,
            optimizer_factory=RecordingOptimizer,
            audit=True,
        )

    direct = make_trainer(tmp_path / "joint_direct")
    replay = make_trainer(
        tmp_path / "joint_replay", accumulation_rows=8, observe_without_graph=True
    )
    rep_direct = direct.run()
    rep_replay = replay.run()
    assert rep_direct["status"] == "completed"
    assert rep_replay["status"] == "completed"
    assert direct.audit == replay.audit
    for left, right in zip(named_records(direct), named_records(replay), strict=True):
        assert left.keys() == right.keys()
        for name in left:
            if left[name] is not None:
                torch.testing.assert_close(right[name], left[name], rtol=1e-10, atol=1e-12)

    paused = make_trainer(tmp_path / "joint_resume", accumulation_rows=8)
    stop = StopAtStep(2)
    stop.trainer = paused
    assert paused.run(stop=stop)["status"] == "paused"
    resumed = make_trainer(tmp_path / "joint_resume", accumulation_rows=8)
    rep_resumed = resumed.run(resume=True)
    assert rep_resumed["status"] == "completed"
    assert rep_resumed["history"] == rep_replay["history"]
    assert paused.audit + resumed.audit == direct.audit


def test_clip_parameter_groups_dust_and_walk_forward_guards(tmp_path):
    model = make_predictor(VARIANT)
    groups = clip_parameter_groups(model)
    contrastive_names = [
        name for name, _ in model.named_parameters() if name.startswith("contrastive_memory.")
    ]
    assert len(contrastive_names) == 12
    for name in contrastive_names:
        assert name in groups["mac"], name

    with pytest.raises(ValueError, match="Dust"):
        FinancialDustAdapter(model)

    view = tmp_path / "view.json"
    view.write_text("{}\n", encoding="utf-8")
    report = dict(
        kind=mars_titan_walk_forward.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=dict(
            variant="mac_online",
            seed=42,
            view_sha256=hashlib.sha256(view.read_bytes()).hexdigest(),
            predictor_changes=dict(contrastive_memory=asdict(VARIANT)),
        ),
        identity=dict(recipe=dict(predictor=dict(hidden_size=32, dtype="float64"))),
    )
    (tmp_path / "run.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="memoria contrastiva"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
    with pytest.raises(ValueError, match="memoria contrastiva"):
        mars_titan_walk_forward._frozen_parent(
            dict(predictor=dict(hidden_size=32, dtype="float64")),
            specification(),
            42,
            "cpu",
            tmp_path,
            report,
        )


def test_eval_frozen_and_controls_behavior_in_chronological_trainer(streams, tmp_path):
    from tests.training.test_financial_run import RecordingOptimizer, entries

    def run_mode(name, config):
        recipe = financial_run.ChronologicalRecipe(
            truncation=2,
            epochs=1,
            block_rows=8,
            checkpoint_updates=8,
            selection=dict(metric="session_mae", patience=5, min_delta=0.0),
        )
        engine = financial_run.ChronologicalTrainer(
            corpus_predictor(streams, config),
            recipe,
            train=streams["train"],
            validation=streams["validation"],
            output=tmp_path / name,
            optimizer_factory=RecordingOptimizer,
            audit=True,
        )
        assert engine.run()["status"] == "completed"
        return engine

    online = run_mode("infonce", VARIANT)
    frozen = run_mode("eval_frozen", EVAL_FROZEN)
    shuffled = run_mode("shuffled", SHUFFLED)
    l2 = run_mode("l2_shared", L2_SHARED)

    # En ajuste, eval_frozen coincide bit a bit con infonce; en validación difiere al congelar M0.
    assert entries(frozen.audit, "prediction", "train") == entries(
        online.audit, "prediction", "train"
    )
    assert entries(frozen.audit, "prediction", "validation") != entries(
        online.audit, "prediction", "validation"
    )
    # Los placebos shuffled_values y l2_shared producen trayectorias distintas del candidato.
    assert entries(shuffled.audit, "prediction", "train") != entries(
        online.audit, "prediction", "train"
    )
    assert entries(l2.audit, "prediction", "train") != entries(online.audit, "prediction", "train")


def test_strict_causality_future_perturbation_in_chronological_trainer(tmp_path):
    from tests.training.chronological_fixture import decision as ts_decision
    from tests.training.test_financial_run import RecordingOptimizer, entries, named_records
    from tests.training.test_financial_run_session_objective import streams as make_streams

    cut = ts_decision("2022-12-09")
    runs = {}
    for label, perturb in (("base", None), ("future", cut)):
        src = make_streams(tmp_path / label, perturb_after=perturb)
        recipe = financial_run.ChronologicalRecipe(
            truncation=2,
            epochs=1,
            block_rows=8,
            checkpoint_updates=8,
            selection=dict(metric="session_mae", patience=5, min_delta=0.0),
        )
        engine = financial_run.ChronologicalTrainer(
            corpus_predictor(src, VARIANT),
            recipe,
            train=src["train"],
            validation=src["validation"],
            output=tmp_path / label / "run",
            optimizer_factory=RecordingOptimizer,
            audit=True,
        )
        assert engine.run()["status"] == "completed"
        runs[label] = engine

    base_preds = entries(runs["base"].audit, "prediction", "train")
    future_preds = entries(runs["future"].audit, "prediction", "train")
    assert [e for e in base_preds if e[3] <= cut] == [e for e in future_preds if e[3] <= cut]
    assert base_preds != future_preds
    steps = [u[2] for u in entries(runs["base"].audit, "update") if u[1] <= cut]
    assert steps
    for left, right in zip(
        named_records(runs["base"])[: len(steps)], named_records(runs["future"]), strict=False
    ):
        assert left.keys() == right.keys()
        for key in left:
            if left[key] is not None:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_single_variable_declaration_on_us_campaign_v12_screen3():
    document = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    recipe = json.loads(CORE_RECIPE.read_text(encoding="utf-8"))
    added = {
        "titans_mac_infonce": VARIANT,
        "titans_mac_infonce_shuffled": SHUFFLED,
        "titans_mac_infonce_eval_frozen": EVAL_FROZEN,
        "titans_mac_infonce_l2_shared": L2_SHARED,
    }
    extended = copy.deepcopy(document)
    for name, cfg in added.items():
        extended["models"][name] = dict(
            machine="mars_titan",
            recipe="core",
            runner="fit",
            tier="s3_inventions",
            variant="mac_online",
            predictor_changes=dict(contrastive_memory=asdict(cfg)),
        )
    validated = comparison_campaign.checked_declaration(extended, root=ROOT)
    for name in added:
        assert name in validated["models"]

    # Verificar que la única diferencia de identidad frente a `titans_mac` es `contrastive_memory`.
    from mars_titan.models.titans.financial_inputs import FinancialInputSpec

    base_opts = {k: v for k, v in recipe["predictor"].items() if k != "dtype"}
    raw_spec = specification()
    spec = FinancialInputSpec(
        source_sha256=raw_spec.source_sha256,
        view_sha256=raw_spec.view_sha256,
        representation=raw_spec.representation,
        dimensions=raw_spec.dimensions | {"charts": 512 + 46},
        input_policy=raw_spec.input_policy,
    )
    base_cfg = FinancialConfig(spec, variant="mac_online", seed=42, **base_opts)
    for cfg in added.values():
        cand_cfg = FinancialConfig(
            spec,
            variant="mac_online",
            seed=42,
            **(base_opts | dict(contrastive_memory=asdict(cfg))),
        )
        changed = {
            k
            for k in (cand_cfg.identity().keys() | base_cfg.identity().keys())
            if cand_cfg.identity().get(k) != base_cfg.identity().get(k)
        }
        assert changed == {"contrastive_memory"}
