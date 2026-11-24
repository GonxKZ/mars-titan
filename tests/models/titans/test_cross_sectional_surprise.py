"""Sorpresa relativa transversal ortogonalizada (CSOS) en NeuralMemory: contratos, ecuaciones,
paridad analítico/autograd, no fuga, placebo, puntos de control y reinicio de fase.
"""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_adapter import DIMENSIONS, setup, specification
from test_session_rank_objective import model as session_model

from mars_titan.models.learning_scale import LearningScale
from mars_titan.models.quantile_head import MEDIAN_GAIN, QUANTILE_HEAD, median, pinball_loss
from mars_titan.models.titans.config import (
    ANALYTIC_INNER_GRADIENT,
    AUTOGRAD_INNER_GRADIENT,
    FUSED_INNER_GRADIENT,
    GateBias,
    MACConfig,
    MemoryConfig,
)
from mars_titan.models.titans.cross_sectional_surprise import (
    CROSS_SECTIONAL_SURPRISE,
    CrossSectionalSurprise,
    CrossSectionalSurpriseModule,
    session_groups,
)
from mars_titan.models.titans.financial import (
    DecisionBatch,
    FinancialConfig,
    FinancialPredictor,
)
from mars_titan.models.titans.frozen_financial import (
    FrozenFinancialConsumer,
    declared_sources,
)
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.session_ic_objective import SessionICTerm
from mars_titan.training.search_cases import checked_predictor_changes

ROOT = Path(__file__).resolve().parents[3]
START, DAY = 1_609_459_200_000_000, 86_400_000_000
VARIANT = CrossSectionalSurprise()
PLACEBO = CrossSectionalSurprise(mode="shuffled_session")
MARKETS = ("US", "CN", "US", "US", "CN", "US")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def memory_config(
    csos=None,
    *,
    dim=8,
    depth=2,
    inner_gradient=ANALYTIC_INNER_GRADIENT,
    stability=None,
):
    return MemoryConfig(
        dim=dim,
        depth=depth,
        max_batch=16,
        max_tokens=4,
        gate_bias=GateBias(),
        residual_layer_norm=True,
        inner_gradient=inner_gradient,
        stability=stability,
        cross_sectional_surprise=csos,
    )


def financial_config(
    csos=None,
    *,
    variant="mac_online",
    inner_gradient=ANALYTIC_INNER_GRADIENT,
    hidden_size=32,
    **extra,
):
    options = (
        dict(
            gate_bias=GateBias(),
            memory_residual_layer_norm=True,
            memory_inner_gradient=inner_gradient,
            memory_cross_sectional_surprise=csos,
        )
        | extra
    )
    return FinancialConfig(
        specification(),
        variant=variant,
        hidden_size=hidden_size,
        **options,
    )


def reference_same_session(gradients, elements, groups, *, lam, weight, bound, eps):
    """Referencia explícita en Python puro grupo a grupo para same_session."""
    batch = elements.shape[0]
    if groups is None:
        buckets = [list(range(batch))]
    else:
        seen = {}
        for idx, key in enumerate(zip(groups.markets, groups.prediction_at, strict=True)):
            seen.setdefault(key, []).append(idx)
        buckets = list(seen.values())
    out_grads = [g.clone() for g in gradients]
    shifts = torch.zeros(batch, 1, dtype=elements.dtype, device=elements.device)
    losses = elements.sum(dim=-1)
    log_losses = torch.log(losses + eps)
    for members in buckets:
        if len(members) < 2:
            continue
        for layer, grad in enumerate(gradients):
            mean_g = sum(grad[j] for j in members) / len(members)
            for i in members:
                out_grads[layer][i] = grad[i] - lam * mean_g
        mean_s = sum(log_losses[j] for j in members) / len(members)
        var_s = sum((log_losses[j] - mean_s).square() for j in members) / len(members)
        inv_std = torch.rsqrt(var_s + eps)
        for i in members:
            u_i = (log_losses[i] - mean_s) * inv_std
            shifts[i, 0] = bound * torch.tanh(weight * u_i)
    return tuple(out_grads), shifts


def raw_session(
    rows=6,
    *,
    at=START,
    seed=5,
    markets=MARKETS,
    instants=None,
):
    """Diccionario bruto de un lote multimercado listo para DecisionBatch.from_corpus."""
    generator = np.random.default_rng(seed)
    market_list = list(markets[:rows])
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
    names = [f"{market}/A{index:03}" for index, market in enumerate(market_list)]
    at_values = [at] * rows if instants is None else list(instants)
    return dict(
        inputs=values,
        presence=presence,
        sample_ids=[f"{name}/{moment}" for name, moment in zip(names, at_values, strict=True)],
        market=market_list,
        prediction_at=np.array(at_values, dtype="datetime64[us]"),
        input_available_at=np.array([moment - 1 for moment in at_values], dtype="datetime64[us]"),
    )


def multi_step_batches(spec, *, count=3, markets=MARKETS, dtype=torch.float64):
    """Generar varios instantes cronológicos con los mismos flujos multimercado."""
    return [
        DecisionBatch.from_corpus(
            raw_session(len(markets), at=START + step * DAY, seed=5 + step, markets=markets),
            spec,
            dtype=dtype,
        )
        for step in range(count)
    ]


def test_default_disabled_preserves_identity_parameters_outputs_and_gradients():
    plain_mem = MemoryConfig(dim=8)
    explicit_mem = MemoryConfig(dim=8, cross_sectional_surprise=None)
    assert "cross_sectional_surprise" not in plain_mem.identity()
    assert plain_mem.identity() == explicit_mem.identity()
    assert plain_mem.fingerprint() == explicit_mem.fingerprint()

    base, batch = setup()
    explicit = session_model(memory_cross_sectional_surprise=None)
    assert "memory_cross_sectional_surprise" not in base.config.identity()
    assert base.config.identity() == explicit.config.identity()
    assert not base.session_level and base.mac.memory.cross_sectional_surprise is None

    state_a = base.initial_state(batch.flow_ids, differentiable=True)
    state_b = explicit.initial_state(batch.flow_ids, differentiable=True)
    out_a = base.prepare(batch, state_a, differentiable=True)
    out_b = explicit.prepare(batch, state_b, differentiable=True)
    assert torch.equal(out_a.point_predictions, out_b.point_predictions)
    out_a.point_predictions.square().sum().backward()
    out_b.point_predictions.square().sum().backward()
    for (name, pa), (_, pb) in zip(
        base.named_parameters(), explicit.named_parameters(), strict=True
    ):
        assert torch.equal(pa, pb), name
        assert (pa.grad is None) == (pb.grad is None), name
        assert pa.grad is None or torch.equal(pa.grad, pb.grad), name


@pytest.mark.parametrize(
    "bad",
    [
        dict(mode="unknown"),
        dict(mode=1),
        dict(bound=0.0),
        dict(bound=-1.0),
        dict(bound=4.5),
        dict(bound=math.nan),
        dict(bound=math.inf),
        dict(bound=True),
        dict(eps=0.0),
        dict(eps=1e-12),
        dict(eps=0.5),
        dict(eps=math.nan),
        dict(eps=False),
        dict(seed=-1),
        dict(seed=2**32),
        dict(seed=True),
        dict(seed=1.5),
    ],
)
def test_invalid_csos_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        CrossSectionalSurprise(**bad)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"bound": 1.0},
        {"mode": "same_session", "extra": 1},
        "cross_sectional_surprise_v1",
        123,
    ],
)
def test_invalid_csos_declarations_in_financial_config_are_rejected(raw):
    with pytest.raises(ValueError):
        financial_config(csos=raw)


def test_fused_v1_is_rejected_clearly_in_memory_and_financial_configs():
    with pytest.raises(ValueError, match="fused_v1"):
        MemoryConfig(
            dim=64,
            depth=2,
            inner_gradient=FUSED_INNER_GRADIENT,
            cross_sectional_surprise=VARIANT,
        )
    with pytest.raises(ValueError, match="fused_v1"):
        financial_config(csos=VARIANT, hidden_size=64, inner_gradient=FUSED_INNER_GRADIENT)


@pytest.mark.parametrize("variant", ["mac_frozen", "mac_disabled", "transformer_direct"])
def test_csos_requires_mac_online(variant):
    with pytest.raises(ValueError, match="mac_online"):
        financial_config(csos=VARIANT, variant=variant)


def test_mac_config_rejects_csos_outside_online_mode():
    mem = memory_config(VARIANT)
    for mode in ("frozen", "disabled"):
        with pytest.raises(ValueError, match="online"):
            MACConfig(memory=mem, heads=2, persistent_tokens=2, max_segment=1, memory_mode=mode)


def test_csos_rejects_local_control_shared_memory_and_unexpected_session_groups():
    with pytest.raises(ValueError, match="control local|contraste con C"):
        FinancialPredictor(
            financial_config(csos=VARIANT),
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )
    mem_csos = NeuralMemory(memory_config(VARIANT), dtype=torch.float64)
    with pytest.raises(ValueError, match="compartida|PT4"):
        mem_csos.update_shared(
            torch.zeros(2, 8, dtype=torch.float64),
            mem_csos.initial_state(1),
        )
    mem_plain = NeuralMemory(memory_config(None), dtype=torch.float64)
    groups = session_groups(("US/A", "US/B"), (START, START))
    with pytest.raises(ValueError, match="session_groups"):
        mem_plain.update(
            torch.zeros(2, 1, 8, dtype=torch.float64),
            mem_plain.initial_state(2),
            session_groups=groups,
        )


def test_identity_records_csos_and_normalizes_json_and_typed_declarations():
    assert VARIANT.identity()["name"] == CROSS_SECTIONAL_SURPRISE
    assert VARIANT.identity() != PLACEBO.identity()
    assert memory_config(VARIANT).fingerprint() != memory_config(PLACEBO).fingerprint()
    assert memory_config(VARIANT).fingerprint() != memory_config(None).fingerprint()
    assert (
        memory_config(CrossSectionalSurprise(bound=1)).fingerprint()
        == memory_config(CrossSectionalSurprise(bound=1.0)).fingerprint()
    )

    typed = financial_config(csos=VARIANT)
    parsed_full = financial_config(csos=json.loads(json.dumps(asdict(VARIANT))))
    parsed_short = financial_config(csos={"mode": "same_session"})
    assert typed == parsed_full == parsed_short
    assert typed.identity() == parsed_full.identity() == parsed_short.identity()


def test_exact_same_session_equations_match_python_reference_for_single_and_multi_groups():
    generator = torch.Generator().manual_seed(17)
    gradients = (
        torch.randn(6, 4, 4, generator=generator, dtype=torch.float64),
        torch.randn(6, 4, 4, generator=generator, dtype=torch.float64),
    )
    elements = torch.rand(6, 4, generator=generator, dtype=torch.float64) + 0.05
    module = CrossSectionalSurpriseModule(VARIANT, dtype=torch.float64)
    with torch.no_grad():
        module.lambda_logit.fill_(0.7)
        module.write_weight.fill_(-0.4)
    lam = torch.tanh(module.lambda_logit)
    weight = module.write_weight

    # Grupo único (groups=None)
    actual_g, actual_s = module(gradients, elements, None)
    exp_g, exp_s = reference_same_session(
        gradients, elements, None, lam=lam, weight=weight, bound=VARIANT.bound, eps=VARIANT.eps
    )
    for a, e in zip(actual_g, exp_g, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(actual_s, exp_s, rtol=1e-14, atol=1e-14)

    # Múltiples mercados, múltiples instantes y un grupo unitario (N=1)
    flow_ids = ("US/A", "CN/X", "US/B", "US/C", "CN/Y", "JP/Z")
    instants = (START, START, START, START + DAY, START, START)
    groups = session_groups(flow_ids, instants)
    actual_g, actual_s = module(gradients, elements, groups)
    exp_g, exp_s = reference_same_session(
        gradients, elements, groups, lam=lam, weight=weight, bound=VARIANT.bound, eps=VARIANT.eps
    )
    for a, e in zip(actual_g, exp_g, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(actual_s, exp_s, rtol=1e-14, atol=1e-14)
    # Los grupos unitarios (filas 3 y 5) conservan el gradiente intacto y desplazamiento nulo.
    for row in (3, 5):
        for layer in range(2):
            assert torch.equal(actual_g[layer][row], gradients[layer][row])
        assert actual_s[row, 0].item() == 0.0


def test_orthogonalization_removes_common_market_shock_when_lambda_approaches_one():
    generator = torch.Generator().manual_seed(23)
    common_us = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64) * 5.0
    common_cn = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64) * 3.0
    idio = torch.randn(6, 4, 4, generator=generator, dtype=torch.float64) * 0.2
    g0 = idio.clone()
    us_rows = [0, 2, 3, 5]
    cn_rows = [1, 4]
    g0[us_rows] = g0[us_rows] + common_us
    g0[cn_rows] = g0[cn_rows] + common_cn
    elements = torch.rand(6, 4, generator=generator, dtype=torch.float64) + 0.1
    groups = session_groups(
        tuple(f"{m}/A{i}" for i, m in enumerate(MARKETS)),
        (START,) * len(MARKETS),
    )

    module = CrossSectionalSurpriseModule(VARIANT, dtype=torch.float64)
    with torch.no_grad():
        module.lambda_logit.fill_(25.0)  # tanh(25) = 1.0 en FP64
    (tilde_g,), _ = module((g0,), elements, groups)

    # La media transversal de cada mercado se anula completamente.
    torch.testing.assert_close(
        tilde_g[us_rows].mean(dim=0), torch.zeros(4, 4, dtype=torch.float64), rtol=0, atol=1e-13
    )
    torch.testing.assert_close(
        tilde_g[cn_rows].mean(dim=0), torch.zeros(4, 4, dtype=torch.float64), rtol=0, atol=1e-13
    )
    # El componente idiosincrásico centrado se conserva intacto.
    torch.testing.assert_close(
        tilde_g[us_rows],
        idio[us_rows] - idio[us_rows].mean(dim=0, keepdim=True),
        rtol=1e-12,
        atol=1e-13,
    )


def test_neural_memory_applies_exact_orthogonalized_gradient_and_relative_theta(monkeypatch):
    seen = []
    monkeypatch.setattr(NeuralMemory, "observer", staticmethod(lambda **v: seen.append(v)))
    memory = NeuralMemory(memory_config(VARIANT, dim=6), dtype=torch.float64)
    with torch.no_grad():
        memory.cross_sectional_surprise.lambda_logit.fill_(0.8)
        memory.cross_sectional_surprise.write_weight.fill_(0.6)
    generator = torch.Generator().manual_seed(31)
    token = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    state = memory.initial_state(4)
    following = memory.update(token.unsqueeze(1), state)

    # Calcular a mano los gradientes base y el desplazamiento de theta.
    plain = NeuralMemory(memory_config(None, dim=6), dtype=torch.float64)
    seen_plain = []
    monkeypatch.setattr(NeuralMemory, "observer", staticmethod(lambda **v: seen_plain.append(v)))
    plain_next = plain.update(token.unsqueeze(1), plain.initial_state(4))
    plain_theta = seen_plain[0]["theta"]
    raw_grads = tuple(-m / plain_theta for m in plain_next.momentum)
    # Con los mismos pesos iniciales, residuales e = ||M(k)-v||^2 se recuperan desde plain.
    keys = torch.nn.functional.normalize(plain.key_projection(token), dim=-1, eps=1e-12)
    values = plain.value_projection(token)
    pred = plain._apply_memory(keys.unsqueeze(1), state.weights, allow_fused=False).squeeze(1)
    elements = (pred - values).square()
    expected_grads, expected_shift = reference_same_session(
        raw_grads,
        elements,
        None,
        lam=torch.tanh(memory.cross_sectional_surprise.lambda_logit),
        weight=memory.cross_sectional_surprise.write_weight,
        bound=VARIANT.bound,
        eps=VARIANT.eps,
    )
    expected_theta = memory.config.theta_max * (
        memory.theta_projection(token) + expected_shift
    ).sigmoid().unsqueeze(-1)
    torch.testing.assert_close(seen[0]["theta"], expected_theta, rtol=1e-13, atol=1e-14)
    for actual_m, expected_g in zip(following.momentum, expected_grads, strict=True):
        torch.testing.assert_close(actual_m, -expected_theta * expected_g, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("mode", ["same_session", "shuffled_session"])
def test_analytic_and_autograd_paths_match_in_states_and_metagradients(mode):
    csos = CrossSectionalSurprise(mode=mode, bound=1.2, eps=1e-6, seed=7)
    cfg_auto = memory_config(csos, dim=6, inner_gradient=AUTOGRAD_INNER_GRADIENT)
    cfg_ana = memory_config(csos, dim=6, inner_gradient=ANALYTIC_INNER_GRADIENT)
    mem_auto = NeuralMemory(cfg_auto, dtype=torch.float64)
    mem_ana = NeuralMemory(cfg_ana, dtype=torch.float64)
    for mem in (mem_auto, mem_ana):
        with torch.no_grad():
            mem.cross_sectional_surprise.lambda_logit.fill_(0.55)
            mem.cross_sectional_surprise.write_weight.fill_(-0.35)

    generator = torch.Generator().manual_seed(41)
    observed = torch.randn(5, 3, 6, generator=generator, dtype=torch.float64)
    probe = torch.randn(5, 1, 6, generator=generator, dtype=torch.float64)
    groups = session_groups(
        ("US/A", "CN/X", "US/B", "CN/Y", "US/C"),
        (START,) * 5,
    )

    results = []
    for mem in (mem_auto, mem_ana):
        tokens = observed.clone().requires_grad_(True)
        state = mem.initial_state(5, differentiable=True)
        surprises = []
        for step in tokens.split(1, dim=1):
            state = mem.update(
                step, state, differentiable=True, surprise=surprises, session_groups=groups
            )
        read = mem.read(probe, state)
        obj = (read * probe).sum() + sum(m.square().sum() for m in state.momentum)
        grads = torch.autograd.grad(obj, [*mem.parameters(), tokens])
        results.append((state, read, torch.stack(surprises), grads))

    (s0, r0, e0, g0), (s1, r1, e1, g1) = results
    for a, b in zip((*s0.weights, *s0.momentum), (*s1.weights, *s1.momentum), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(r1, r0, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(e1, e0, rtol=1e-12, atol=1e-13)
    for a, b in zip(g0, g1, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["same_session", "shuffled_session"])
def test_gradcheck_passes_on_csos_module_for_single_and_multi_groups(mode):
    csos = CrossSectionalSurprise(mode=mode, bound=1.0, eps=1e-5, seed=3)
    module = CrossSectionalSurpriseModule(csos, dtype=torch.float64)
    with torch.no_grad():
        module.lambda_logit.fill_(0.35)
        module.write_weight.fill_(-0.25)
    generator = torch.Generator().manual_seed(53)
    g1 = torch.randn(5, 3, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    g2 = torch.randn(5, 3, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    el = (torch.rand(5, 3, generator=generator, dtype=torch.float64) + 0.1).requires_grad_(True)
    groups = session_groups(("US/A", "CN/X", "US/B", "CN/Y", "US/C"), (START,) * 5)

    def fn(a, b, e, lam_p, w_p):
        with torch.no_grad():
            module.lambda_logit.copy_(lam_p)
            module.write_weight.copy_(w_p)
        # Usar copia funcional de los parámetros para que gradcheck diferencie respecto a ellos.
        out_g, shift = module._compute((a, b), e, groups, lam_p, w_p, token_index=0, steps=None)
        return out_g[0], out_g[1], shift

    lam_in = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    w_in = torch.tensor(-0.25, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (g1, g2, el, lam_in, w_in), eps=1e-6, atol=1e-6, rtol=1e-4)
    # También en grupo único (groups=None)
    groups = None
    assert torch.autograd.gradcheck(fn, (g1, g2, el, lam_in, w_in), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("csos", [VARIANT, PLACEBO])
def test_zero_init_matches_control_bit_for_bit_and_receives_first_step_gradients(csos):
    rng = torch.get_rng_state().clone()
    control = FinancialPredictor(financial_config(None), dtype=torch.float64)
    candidate = FinancialPredictor(financial_config(csos), dtype=torch.float64)
    assert torch.equal(torch.get_rng_state(), rng)

    ctrl_params = dict(control.named_parameters())
    cand_params = dict(candidate.named_parameters())
    added = set(cand_params) - set(ctrl_params)
    assert added == {
        "mac.memory.cross_sectional_surprise.lambda_logit",
        "mac.memory.cross_sectional_surprise.write_weight",
    }
    for name, value in ctrl_params.items():
        assert torch.equal(cand_params[name], value), name
    for name in added:
        assert cand_params[name].item() == 0.0

    batches = multi_step_batches(control.config.inputs, count=2, dtype=torch.float64)
    st_c = control.initial_state(batches[0].flow_ids, differentiable=True)
    st_m = candidate.initial_state(batches[0].flow_ids, differentiable=True)
    pts_c, pts_m = [], []
    for batch in batches:
        pc = control.prepare(batch, st_c, differentiable=True)
        pm = candidate.prepare(batch, st_m, differentiable=True, session_complete=True)
        assert torch.equal(pm.point_predictions, pc.point_predictions)
        assert torch.equal(pm.working_state, pc.working_state)
        for wa, wb in zip(
            pc.next_state.mac.memory.weights, pm.next_state.mac.memory.weights, strict=True
        ):
            assert torch.equal(wb, wa)
        pts_c.append(pc.point_predictions)
        pts_m.append(pm.point_predictions)
        st_c, st_m = pc.next_state, pm.next_state

    # El gradiente del primer paso llega sin encadenar ceros a ambos parámetros de CSOS.
    torch.stack(pts_m).square().sum().backward()
    for name in added:
        grad = cand_params[name].grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().item() > 0.0, name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_are_finite_with_constant_zero_prediction_and_with_zero_median_quantile_head(
    dtype,
):
    # Caso 1: cabeza escalar exactamente en cero (predicción constante 0) + término de IC.
    scalar = FinancialPredictor(financial_config(VARIANT), dtype=dtype)
    with torch.no_grad():
        scalar.head.weight.zero_()
        scalar.head.bias.zero_()
    scalar._seal_parameters()
    batches = multi_step_batches(scalar.config.inputs, count=2, dtype=dtype)
    st = scalar.initial_state(batches[0].flow_ids, differentiable=True)
    preds = []
    for batch in batches:
        prep = scalar.prepare(batch, st, differentiable=True, session_complete=True)
        preds.append(prep.point_predictions)
        st = prep.next_state
    pred_vec = torch.cat(preds)
    assert torch.count_nonzero(pred_vec) == 0
    target = torch.linspace(-0.02, 0.02, len(pred_vec), dtype=dtype)
    sessions = [
        (flow.split("/", 1)[0], at)
        for batch in batches
        for flow, at in zip(batch.flow_ids, batch.prediction_at, strict=True)
    ]
    ic_term = SessionICTerm(weight=3.0, minimum_rows=3)
    loss = torch.nn.functional.l1_loss(pred_vec, target) + ic_term.term(pred_vec, target, sessions)
    loss.backward()
    for name, param in scalar.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name

    # Caso 2: receta real de campaña (quantile_head_v1 + zero_median + median_gain + pinball + IC).
    from test_financial_learning_scale import statistics

    q_cfg = financial_config(
        VARIANT,
        head=QUANTILE_HEAD,
        learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=True),
        median_gain=MEDIAN_GAIN,
    )
    q_model = FinancialPredictor(q_cfg, dtype=dtype)
    q_model.set_learning_statistics(statistics(q_cfg.learning_scale))
    st = q_model.initial_state(batches[0].flow_ids, differentiable=True)
    q_list = []
    for batch in batches:
        prep = q_model.prepare(batch, st, differentiable=True, session_complete=True)
        q_list.append(prep.quantiles)
        st = prep.next_state
    quantiles = torch.cat(q_list, dim=0)
    assert torch.count_nonzero(median(quantiles)) == 0
    q_loss = pinball_loss(quantiles, target) + ic_term.term(median(quantiles), target, sessions)
    q_loss.backward()
    for name, param in q_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    csos_mod = q_model.mac.memory.cross_sectional_surprise
    assert csos_mod.lambda_logit.grad is not None and csos_mod.lambda_logit.grad.abs().item() > 0
    assert csos_mod.write_weight.grad is not None and csos_mod.write_weight.grad.abs().item() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_degenerate_zero_and_constant_surprise_produce_finite_outputs_and_gradients(dtype):
    module = CrossSectionalSurpriseModule(VARIANT, dtype=dtype)
    with torch.no_grad():
        module.lambda_logit.fill_(0.5)
        module.write_weight.fill_(0.5)
    g = torch.randn(4, 3, 3, dtype=dtype, requires_grad=True)
    for elements in (
        torch.zeros(4, 3, dtype=dtype, requires_grad=True),
        torch.full((4, 3), 2.5, dtype=dtype, requires_grad=True),
    ):
        (og,), shift = module((g,), elements, None)
        assert torch.isfinite(og).all() and torch.isfinite(shift).all()
        assert torch.count_nonzero(shift) == 0
        (og.square().sum() + shift.square().sum()).backward()
        assert torch.isfinite(g.grad).all() and torch.isfinite(elements.grad).all()
        assert torch.isfinite(module.lambda_logit.grad).all()
        assert torch.isfinite(module.write_weight.grad).all()
        g.grad = None
        module.zero_grad()


def test_autocast_is_rejected_and_execution_is_deterministic_in_fp32_and_fp64():
    module = CrossSectionalSurpriseModule(VARIANT, dtype=torch.float32)
    g = torch.randn(4, 4, 4, dtype=torch.float32)
    el = torch.rand(4, 4, dtype=torch.float32) + 0.1
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        module((g,), el, None)

    for dtype in (torch.float32, torch.float64):
        for csos in (VARIANT, PLACEBO):
            pred = FinancialPredictor(financial_config(csos), dtype=dtype)
            with torch.no_grad():
                pred.mac.memory.cross_sectional_surprise.lambda_logit.fill_(0.4)
                pred.mac.memory.cross_sectional_surprise.write_weight.fill_(-0.3)
            pred._seal_parameters()
            batches = multi_step_batches(pred.config.inputs, count=2, dtype=dtype)
            runs = []
            for _ in range(2):
                st = pred.initial_state(batches[0].flow_ids)
                out = []
                for batch in batches:
                    res = pred.prepare(batch, st, session_complete=True)
                    out.append(res.point_predictions)
                    st = res.next_state
                runs.append((torch.stack(out), st))
            assert torch.equal(runs[0][0], runs[1][0])
            for a, b in zip(
                runs[0][1].mac.memory.weights, runs[1][1].mac.memory.weights, strict=True
            ):
                assert torch.equal(a, b)


def test_session_complete_is_enforced_and_there_is_no_future_or_cross_market_leakage():
    predictor = FinancialPredictor(financial_config(VARIANT), dtype=torch.float64)
    assert predictor.session_level
    with torch.no_grad():
        predictor.mac.memory.cross_sectional_surprise.lambda_logit.fill_(0.8)
        predictor.mac.memory.cross_sectional_surprise.write_weight.fill_(0.5)
    predictor._seal_parameters()

    spec = predictor.config.inputs
    raw_base = raw_session(len(MARKETS), at=START, seed=5, markets=MARKETS)
    batch = DecisionBatch.from_corpus(raw_base, spec, dtype=torch.float64)
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))

    # 1) Aislamiento entre mercados: alterar CN no cambia ni un bit las filas de US.
    base_res = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    raw_cn = dict(
        raw_base, inputs={name: value.copy() for name, value in raw_base["inputs"].items()}
    )
    raw_cn["inputs"]["prices"][[1, 4]] += 3.0
    raw_cn["inputs"]["charts"][[1, 4]] -= 2.0
    batch_cn = DecisionBatch.from_corpus(raw_cn, spec, dtype=torch.float64)
    cn_res = predictor.prepare(
        batch_cn, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    us_rows = [0, 2, 3, 5]
    assert torch.equal(cn_res.point_predictions[us_rows], base_res.point_predictions[us_rows])
    for wa, wb in zip(
        base_res.next_state.mac.memory.weights, cn_res.next_state.mac.memory.weights, strict=True
    ):
        assert torch.equal(wb[us_rows], wa[us_rows])

    # Alterar otra fila de US en el mismo instante SÍ modifica la escritura de US.
    raw_us = dict(
        raw_base, inputs={name: value.copy() for name, value in raw_base["inputs"].items()}
    )
    raw_us["inputs"]["prices"][0] += 3.0
    us_res = predictor.prepare(
        DecisionBatch.from_corpus(raw_us, spec, dtype=torch.float64),
        predictor.initial_state(batch.flow_ids),
        session_complete=True,
    )
    assert not torch.equal(us_res.point_predictions[2], base_res.point_predictions[2])

    # 2) Cero fugas temporales en MAC/NeuralMemory: filas en t+1 (futuro) no afectan a filas en t.
    mac = TitansMAC(
        MACConfig(
            memory=memory_config(VARIANT, dim=8), heads=2, persistent_tokens=2, max_segment=1
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        mac.memory.cross_sectional_surprise.lambda_logit.fill_(0.8)
        mac.memory.cross_sectional_surprise.write_weight.fill_(0.5)
    instants_mixed = (START, START, START, START + DAY, START, START + DAY)
    groups_mixed = session_groups(batch.flow_ids, instants_mixed)
    segment = torch.randn(6, 1, 8, generator=torch.Generator().manual_seed(77), dtype=torch.float64)
    st0 = mac.initial_state(6)
    out_base, next_base = mac(segment, st0, session_groups=groups_mixed)
    seg_future = segment.clone()
    seg_future[[3, 5]] += 10.0
    out_fut, next_fut = mac(seg_future, st0, session_groups=groups_mixed)
    past_rows = [0, 1, 2, 4]
    assert torch.equal(out_fut[past_rows], out_base[past_rows])
    for wa, wb in zip(next_base.memory.weights, next_fut.memory.weights, strict=True):
        assert torch.equal(wb[past_rows], wa[past_rows])


@pytest.mark.parametrize("csos", [VARIANT, PLACEBO])
def test_permuting_rows_permutes_outputs_and_states_equivariantly(csos):
    predictor = FinancialPredictor(financial_config(csos), dtype=torch.float64)
    with torch.no_grad():
        predictor.mac.memory.cross_sectional_surprise.lambda_logit.fill_(0.6)
        predictor.mac.memory.cross_sectional_surprise.write_weight.fill_(0.4)
    predictor._seal_parameters()

    batch = multi_step_batches(predictor.config.inputs, count=1, dtype=torch.float64)[0]
    order = [3, 0, 5, 1, 4, 2]
    permuted = batch.select(order)

    res_orig = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    res_perm = predictor.prepare(
        permuted, predictor.initial_state(permuted.flow_ids), session_complete=True
    )
    torch.testing.assert_close(
        res_perm.point_predictions, res_orig.point_predictions[order], rtol=1e-13, atol=1e-13
    )
    for wa, wb in zip(
        res_orig.next_state.mac.memory.weights,
        res_perm.next_state.mac.memory.weights,
        strict=True,
    ):
        torch.testing.assert_close(wb, wa[order], rtol=1e-13, atol=1e-13)


def test_placebo_has_same_parameters_deranges_surprise_and_does_not_cancel_common_shock():
    mod_real = CrossSectionalSurpriseModule(VARIANT, dtype=torch.float64)
    mod_pl = CrossSectionalSurpriseModule(PLACEBO, dtype=torch.float64)
    assert set(dict(mod_real.named_parameters())) == set(dict(mod_pl.named_parameters()))
    for mod in (mod_real, mod_pl):
        with torch.no_grad():
            mod.lambda_logit.fill_(25.0)  # tanh = 1.0
            mod.write_weight.fill_(0.5)

    generator = torch.Generator().manual_seed(61)
    # 6 activos en el mismo mercado (número par) con un choque común fuerte.
    common = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64) * 4.0
    g0 = common + torch.randn(6, 4, 4, generator=generator, dtype=torch.float64) * 0.1
    elements = torch.tensor(
        [[0.1, 0.1], [0.5, 0.5], [1.2, 1.2], [2.5, 2.5], [5.0, 5.0], [10.0, 10.0]],
        dtype=torch.float64,
    )
    groups = session_groups(tuple(f"US/S{i}" for i in range(6)), (START,) * 6)

    rng = torch.get_rng_state().clone()
    (g_real,), s_real = mod_real((g0,), elements, groups)
    (g_pl,), s_pl = mod_pl((g0,), elements, groups)
    assert torch.equal(torch.get_rng_state(), rng)

    # En same_session la media transversal del gradiente queda anulada; en el placebo permanece.
    torch.testing.assert_close(
        g_real.mean(dim=0), torch.zeros(4, 4, dtype=torch.float64), rtol=0, atol=1e-13
    )
    torch.testing.assert_close(g_pl.mean(dim=0), g0.mean(dim=0), rtol=1e-13, atol=1e-13)
    # En el placebo la perturbación por fila tiene exactamente la misma norma que en same_session.
    diff_real = (g0 - g_real).norm(dim=(-2, -1))
    diff_pl = (g0 - g_pl).norm(dim=(-2, -1))
    torch.testing.assert_close(diff_pl, diff_real, rtol=1e-13, atol=1e-13)
    # El desplazamiento de escritura es una permutación sin puntos fijos (derangement) de s_real.
    assert not torch.equal(s_pl, s_real)
    assert bool((s_pl != s_real).all())
    torch.testing.assert_close(s_pl.squeeze(-1).sort().values, s_real.squeeze(-1).sort().values)

    # Otra semilla del placebo cambia la asignación sin tocar el RNG global.
    mod_pl2 = CrossSectionalSurpriseModule(
        CrossSectionalSurprise(mode="shuffled_session", seed=99), dtype=torch.float64
    )
    with torch.no_grad():
        mod_pl2.lambda_logit.fill_(25.0)
        mod_pl2.write_weight.fill_(0.5)
    (g_pl2,), s_pl2 = mod_pl2((g0,), elements, groups)
    assert not torch.equal(s_pl2, s_pl) or not torch.equal(g_pl2, g_pl)


def test_state_dict_fast_state_checkpoint_round_trip_and_mode_mismatch_rejection():
    source = FinancialPredictor(financial_config(VARIANT), dtype=torch.float64)
    with torch.no_grad():
        source.mac.memory.cross_sectional_surprise.lambda_logit.fill_(0.45)
        source.mac.memory.cross_sectional_surprise.write_weight.fill_(-0.2)
    source._seal_parameters()

    target = FinancialPredictor(financial_config(VARIANT), dtype=torch.float64)
    target.load_state_dict(source.state_dict())

    with pytest.raises(ValueError):
        FinancialPredictor(financial_config(PLACEBO), dtype=torch.float64).load_state_dict(
            source.state_dict()
        )
    with pytest.raises(ValueError):
        FinancialPredictor(financial_config(None), dtype=torch.float64).load_state_dict(
            source.state_dict()
        )

    batches = multi_step_batches(source.config.inputs, count=2, dtype=torch.float64)
    st_src = source.prepare(
        batches[0], source.initial_state(batches[0].flow_ids), session_complete=True
    ).next_state
    exported = source.export_state(st_src)
    restored = target.restore_state(exported)
    exported_cpu = source.export_state_cpu(st_src)
    gathered = target.gather_state(
        {flow: (exported_cpu, row) for row, flow in enumerate(batches[0].flow_ids)},
        batches[0].flow_ids,
    )

    next_orig = source.prepare(batches[1], st_src, session_complete=True)
    next_rest = target.prepare(batches[1], restored, session_complete=True)
    next_gath = target.prepare(batches[1], gathered, session_complete=True)
    assert torch.equal(next_rest.point_predictions, next_orig.point_predictions)
    assert torch.equal(next_gath.point_predictions, next_orig.point_predictions)
    for wa, wb, wc in zip(
        next_orig.next_state.mac.memory.weights,
        next_rest.next_state.mac.memory.weights,
        next_gath.next_state.mac.memory.weights,
        strict=True,
    ):
        assert torch.equal(wb, wa) and torch.equal(wc, wa)


def test_frozen_consumer_and_trainer_code_pin_csos_only_when_declared():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    module_name = "mars_titan.models.titans.cross_sectional_surprise"
    plain = (
        FinancialPredictor(financial_config(None), dtype=torch.float64).eval().requires_grad_(False)
    )
    active = (
        FinancialPredictor(financial_config(VARIANT), dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    assert module_name not in declared_sources(plain.config)
    assert module_name in declared_sources(active.config)

    eng_plain = FrozenFinancialConsumer(plain)
    eng_active = FrozenFinancialConsumer(active)
    assert module_name not in eng_plain.identity()["implementation"]
    assert module_name in eng_active.identity()["implementation"]
    assert eng_active.model_id != eng_plain.model_id

    batch = multi_step_batches(active.config.inputs, count=1, dtype=torch.float64)[0]
    with pytest.raises(ValueError, match="sesión completa"):
        eng_active.prepare(
            batch, active.initial_state(batch.flow_ids), context_id="ctx-1", session_complete=False
        )
    out = eng_active.prepare(
        batch, active.initial_state(batch.flow_ids), context_id="ctx-1", session_complete=True
    )
    assert torch.isfinite(out.point_predictions).all()

    trainer = object.__new__(ChronologicalTrainer)
    trainer.recipe = ChronologicalRecipe()
    for cfg, expected in ((None, False), (VARIANT, True)):
        trainer.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=None,
                asset_groups=None,
                holographic_network=None,
                axial_time_region=None,
                mature_reliability=None,
                fractal_shared_operator=None,
                memory_cross_sectional_surprise=cfg,
            )
        )
        assert (module_name in trainer._code()) is expected


def test_chronological_trainer_updates_resets_phases_and_resumes_bit_for_bit(tmp_path, monkeypatch):
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
    from mars_titan.training.learning_hold import HOLD_ENV
    from mars_titan.training.selection import FIXED_BUDGET
    from tests.learning_hold_lowering import allowed_hold
    from tests.training.test_financial_run import SELECTION, StopAtStep, corpus

    monkeypatch.setenv(HOLD_ENV, str(allowed_hold(tmp_path / "hold.json")))
    _, streams = corpus(tmp_path / "data")
    spec = streams["train"].specification()
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        gate_bias=GateBias(),
        memory_residual_layer_norm=True,
        memory_inner_gradient=ANALYTIC_INNER_GRADIENT,
        memory_cross_sectional_surprise=VARIANT,
    )
    recipe = ChronologicalRecipe(
        truncation=2,
        loss="mae",
        epochs=2,
        block_rows=16,
        accumulation_rows=16,
        checkpoint_updates=1,
        checkpoint_seconds=1e6,
        selection=dict(SELECTION, stopping=FIXED_BUDGET),
    )

    # Ejecución continua de referencia con AdamW real: verifica actualización de ambos parámetros.
    full_pred = FinancialPredictor(cfg, dtype=torch.float64)
    full_trainer = ChronologicalTrainer(
        full_pred,
        recipe,
        train=streams["train"],
        validation=streams["validation"],
        output=tmp_path / "full",
    )
    full_report = full_trainer.run()
    assert full_report["status"] == "completed"
    csos_mod = full_pred.mac.memory.cross_sectional_surprise
    assert csos_mod.lambda_logit.item() != 0.0 and csos_mod.write_weight.item() != 0.0

    # Ejecución pausada en el primer paso y reanudada desde checkpoint: idéntica bit a bit.
    stop = StopAtStep(1)
    resumed_pred = FinancialPredictor(cfg, dtype=torch.float64)
    resumed_trainer = ChronologicalTrainer(
        resumed_pred,
        recipe,
        train=streams["train"],
        validation=streams["validation"],
        output=tmp_path / "resumed",
    )
    stop.trainer = resumed_trainer
    paused = resumed_trainer.run(stop=stop)
    assert paused["status"] == "paused"

    resumed_pred2 = FinancialPredictor(cfg, dtype=torch.float64)
    resumed_trainer2 = ChronologicalTrainer(
        resumed_pred2,
        recipe,
        train=streams["train"],
        validation=streams["validation"],
        output=tmp_path / "resumed",
    )
    resumed_report = resumed_trainer2.run(resume=True)
    assert resumed_report["status"] == "completed"
    assert resumed_report["best_score"] == full_report["best_score"]
    for (name, pa), (_, pb) in zip(
        full_pred.named_parameters(), resumed_pred2.named_parameters(), strict=True
    ):
        assert torch.equal(pa, pb), name

    # Reinicio de fase: evaluar dos veces produce exactamente las mismas filas bit a bit.
    rows_a, rows_b = [], []
    full_trainer.predict(streams["validation"], rows_a)
    full_trainer.predict(streams["validation"], rows_b)
    assert rows_a == rows_b and len(rows_a) > 0

    # Un instante partido en bloques se rechaza en el entrenador cronológico.
    split_recipe = replace(recipe, block_rows=1, accumulation_rows=None)
    split_trainer = ChronologicalTrainer(
        FinancialPredictor(cfg, dtype=torch.float64),
        split_recipe,
        train=streams["train"],
        validation=streams["validation"],
        output=tmp_path / "split",
    )
    with pytest.raises(ValueError, match="sesión completa"):
        split_trainer.run()


def test_end_to_end_chronological_training_has_zero_future_leakage(tmp_path):
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
    from mars_titan.training.selection import FIXED_BUDGET
    from tests.training.chronological_fixture import decision
    from tests.training.test_financial_run import (
        SELECTION,
        RecordingOptimizer,
        corpus,
        entries,
        named_records,
    )

    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        _, streams = corpus(tmp_path / name, perturb_after=perturb)
        spec = streams["train"].specification()
        pred = FinancialPredictor(
            FinancialConfig(
                spec,
                variant="mac_online",
                hidden_size=32,
                gate_bias=GateBias(),
                memory_residual_layer_norm=True,
                memory_inner_gradient=ANALYTIC_INNER_GRADIENT,
                memory_cross_sectional_surprise=VARIANT,
            ),
            dtype=torch.float64,
        )
        with torch.no_grad():
            pred.mac.memory.cross_sectional_surprise.lambda_logit.fill_(0.8)
            pred.mac.memory.cross_sectional_surprise.write_weight.fill_(0.5)
        pred._seal_parameters()
        recipe = ChronologicalRecipe(
            truncation=2,
            loss="mae",
            epochs=1,
            block_rows=16,
            accumulation_rows=16,
            selection=dict(SELECTION, stopping=FIXED_BUDGET),
        )
        engine = ChronologicalTrainer(
            pred,
            recipe,
            train=streams["train"],
            validation=streams["validation"],
            output=tmp_path / name / "run",
            optimizer_factory=RecordingOptimizer,
            audit=True,
        )
        engine.run()
        engines.append(engine)
    base, future = engines
    past_base = [e for e in entries(base.audit, "prediction", "train") if e[3] <= cut]
    past_future = [e for e in entries(future.audit, "prediction", "train") if e[3] <= cut]
    assert past_base and past_base == past_future
    assert entries(base.audit, "prediction", "train") != entries(
        future.audit, "prediction", "train"
    )
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps and steps == [u[2] for u in entries(future.audit, "update") if u[1] <= cut]
    for left, right in zip(named_records(base)[: len(steps)], named_records(future), strict=False):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_campaign_v11_declaration_models_pass_predictor_changes_and_build():
    from mars_titan.models.titans.financial_inputs import FinancialInputSpec

    campaign = json.loads((ROOT / "configs/experiments/us-campaign-v11-screen1.json").read_text())
    core_recipe = json.loads((ROOT / campaign["recipes"]["core"]).read_text())
    base_options = {k: v for k, v in core_recipe["predictor"].items() if k != "dtype"}
    base_spec = specification()
    # Ajustar charts a 46 columnas para que technical_branch de la receta real construya.
    spec = FinancialInputSpec(
        source_sha256=base_spec.source_sha256,
        view_sha256=base_spec.view_sha256,
        representation=base_spec.representation,
        dimensions=dict(base_spec.dimensions, charts=46),
        input_policy=base_spec.input_policy,
    )

    control_changes = {"memory_inner_gradient": ANALYTIC_INNER_GRADIENT}
    candidate_changes = {
        "memory_inner_gradient": ANALYTIC_INNER_GRADIENT,
        "memory_cross_sectional_surprise": {"mode": "same_session", "bound": 1.0, "eps": 1e-6},
    }
    placebo_changes = {
        "memory_inner_gradient": ANALYTIC_INNER_GRADIENT,
        "memory_cross_sectional_surprise": {
            "mode": "shuffled_session",
            "bound": 1.0,
            "eps": 1e-6,
            "seed": 0,
        },
    }
    for changes in (control_changes, candidate_changes, placebo_changes):
        assert checked_predictor_changes(changes) == changes

    models = [
        FinancialPredictor(
            FinancialConfig(spec, variant="mac_online", seed=42, **(base_options | changes)),
            dtype=torch.float32,
        )
        for changes in (control_changes, candidate_changes, placebo_changes)
    ]
    identities = {json.dumps(m.config.identity(), sort_keys=True) for m in models}
    assert len(identities) == 3
    # El control base con fused_v1 + CSOS debe ser rechazado.
    with pytest.raises(ValueError, match="fused_v1"):
        FinancialConfig(
            spec,
            variant="mac_online",
            seed=42,
            **(base_options | {"memory_cross_sectional_surprise": {"mode": "same_session"}}),
        )


def _graph_nodes(tensor):
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(parent for parent, _ in node.next_functions)
    return len(seen)


def test_pinned_values_match_previous_patch_in_fp64_and_fp32():
    """Con solo CSOS activada sobre la nueva base, salidas y pesos coinciden con el parche base."""
    import hashlib

    expected = {
        (torch.float64, "same"): (
            [
                0.004629476575222771,
                0.036401730690088965,
                0.049178401184452644,
                0.02936828939746957,
                0.03962231855447587,
                0.030327322263472095,
                0.02236976631148018,
                0.04223125447923543,
                0.028943887226001802,
                0.024022007425147547,
                0.026579555171155927,
                0.04206172506920705,
            ],
            "e6343fbc9e9f2bbcb4ba8de9d8e7825445b91bc944749893a560ad5d2534c129",
        ),
        (torch.float64, "placebo"): (
            [
                0.006690010907154197,
                0.03657151902177483,
                0.0480742812924375,
                0.029128703713281346,
                0.03695868816645778,
                0.03274974786415292,
                0.023557643758285826,
                0.04338873812837783,
                0.02960616302791361,
                0.02490055123160686,
                0.03348159810780633,
                0.049975284527974055,
            ],
            "1ef283ae3e5017b02473f0054538e5a08e9bb525a7313f6913b8f6816ca89528",
        ),
        (torch.float32, "same"): (
            [
                -0.012001491151750088,
                0.021154556423425674,
                -0.01693759113550186,
                -0.028522726148366928,
                -0.0411267951130867,
                -0.052243124693632126,
                -0.040103599429130554,
                -0.005064942874014378,
                -0.029626645147800446,
                -0.043165531009435654,
                0.01110996026545763,
                -0.014298972673714161,
            ],
            "0073491384aaa9389bb8ca31197799608566146ae8ce12ce052a30515e1c9a0a",
        ),
        (torch.float32, "placebo"): (
            [
                -0.014527306891977787,
                0.022178083658218384,
                -0.01668485999107361,
                -0.028714776039123535,
                -0.04346921294927597,
                -0.05173306539654732,
                -0.05338140204548836,
                0.00020287372171878815,
                -0.0320686511695385,
                -0.041456740349531174,
                -0.016677595674991608,
                -0.011073104105889797,
            ],
            "ec349aa90c701ce2757c16e57623e3d017f3986eca047b8fb2094f7252c286d0",
        ),
    }
    for dtype in (torch.float64, torch.float32):
        for label, csos in (("same", VARIANT), ("placebo", PLACEBO)):
            pred = FinancialPredictor(financial_config(csos), dtype=dtype)
            with torch.no_grad():
                pred.mac.memory.cross_sectional_surprise.lambda_logit.fill_(0.65)
                pred.mac.memory.cross_sectional_surprise.write_weight.fill_(-0.35)
            pred._seal_parameters()
            batches = multi_step_batches(pred.config.inputs, count=2, dtype=dtype)
            st = pred.initial_state(batches[0].flow_ids)
            outs = []
            for batch in batches:
                res = pred.prepare(batch, st, session_complete=True)
                outs.append(res.point_predictions)
                st = res.next_state
            exp_vals, exp_hash = expected[(dtype, label)]
            assert torch.equal(torch.stack(outs).reshape(-1), torch.tensor(exp_vals, dtype=dtype))
            w_bytes = b"".join(
                w.detach().cpu().contiguous().numpy().tobytes() for w in st.mac.memory.weights
            )
            assert hashlib.sha256(w_bytes).hexdigest() == exp_hash


@pytest.mark.parametrize("mode", ["same_session", "shuffled_session"])
def test_autograd_graph_size_does_not_grow_with_session_rows(mode):
    csos = CrossSectionalSurprise(mode=mode, seed=11)
    module = CrossSectionalSurpriseModule(csos, dtype=torch.float64)
    with torch.no_grad():
        module.lambda_logit.fill_(0.5)
        module.write_weight.fill_(-0.3)
    single_sizes, multi_sizes, mem_sizes = {}, {}, {}
    mem = NeuralMemory(
        MemoryConfig(
            dim=6,
            depth=2,
            max_batch=128,
            max_tokens=2,
            gate_bias=GateBias(),
            residual_layer_norm=True,
            inner_gradient=ANALYTIC_INNER_GRADIENT,
            cross_sectional_surprise=csos,
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        mem.cross_sectional_surprise.lambda_logit.fill_(0.5)
        mem.cross_sectional_surprise.write_weight.fill_(-0.3)
    for rows in (8, 32, 128):
        gen = torch.Generator().manual_seed(rows)
        g1 = torch.randn(rows, 4, 4, generator=gen, dtype=torch.float64, requires_grad=True)
        g2 = torch.randn(rows, 4, 4, generator=gen, dtype=torch.float64, requires_grad=True)
        el = (torch.rand(rows, 4, generator=gen, dtype=torch.float64) + 0.1).requires_grad_(True)
        grp_one = session_groups(tuple(f"US/A{i:04}" for i in range(rows)), (START,) * rows)
        out_g, shift = module((g1, g2), el, grp_one)
        single_sizes[rows] = (_graph_nodes(out_g[0]), _graph_nodes(out_g[1]), _graph_nodes(shift))

        grp_two = session_groups(
            tuple(f"{'US' if i % 2 == 0 else 'CN'}/A{i:04}" for i in range(rows)),
            (START,) * rows,
        )
        out_g2, shift2 = module((g1, g2), el, grp_two)
        multi_sizes[rows] = (
            _graph_nodes(out_g2[0]),
            _graph_nodes(out_g2[1]),
            _graph_nodes(shift2),
        )

        token = torch.randn(rows, 1, 6, generator=gen, dtype=torch.float64, requires_grad=True)
        nxt = mem.update(
            token,
            mem.initial_state(rows, differentiable=True),
            differentiable=True,
            session_groups=grp_one,
        )
        mem_sizes[rows] = tuple(_graph_nodes(w) for w in nxt.weights)

    assert len(set(single_sizes.values())) == 1
    assert len(set(multi_sizes.values())) == 1
    assert len(set(mem_sizes.values())) == 1


def test_hot_path_never_calls_tensor_tolist_or_item_and_steps_plan_is_valid(monkeypatch):
    """El camino caliente de CSOS no ejecuta .tolist() ni .item(), incluso con groups=None."""
    calls = []
    orig_tolist = torch.Tensor.tolist
    orig_item = torch.Tensor.item

    def forbidden_tolist(self, *args, **kwargs):
        calls.append("tolist")
        return orig_tolist(self, *args, **kwargs)

    def forbidden_item(self, *args, **kwargs):
        calls.append("item")
        return orig_item(self, *args, **kwargs)

    for csos in (VARIANT, PLACEBO):
        module = CrossSectionalSurpriseModule(csos, dtype=torch.float64)
        with torch.no_grad():
            module.lambda_logit.fill_(0.7)
            module.write_weight.fill_(0.5)
        gen = torch.Generator().manual_seed(91)
        g0 = torch.randn(6, 4, 4, generator=gen, dtype=torch.float64)
        el = torch.tensor(
            [[0.1, 0.1], [0.4, 0.4], [1.0, 1.0], [2.2, 2.2], [4.5, 4.5], [9.0, 9.0]],
            dtype=torch.float64,
        )
        groups = session_groups(tuple(f"US/S{i}" for i in range(6)), (START,) * 6)
        steps_a = torch.zeros(6, dtype=torch.int64)
        steps_b = torch.ones(6, dtype=torch.int64)

        monkeypatch.setattr(torch.Tensor, "tolist", forbidden_tolist)
        try:
            out_g_grp, sh_grp = module((g0,), el, groups, token_index=0, steps=None)
            out_g_a, sh_a = module((g0,), el, None, token_index=0, steps=steps_a)
            out_g_b, sh_b = module((g0,), el, None, token_index=0, steps=steps_b)
            out_g_none, sh_none = module((g0,), el, None, token_index=0, steps=None)
            monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
            module._compute(
                (g0,),
                el,
                groups,
                module.lambda_logit,
                module.write_weight,
                token_index=0,
                steps=None,
            )
            module._compute(
                (g0,),
                el,
                None,
                module.lambda_logit,
                module.write_weight,
                token_index=0,
                steps=steps_a,
            )
        finally:
            monkeypatch.setattr(torch.Tensor, "tolist", orig_tolist)
            monkeypatch.setattr(torch.Tensor, "item", orig_item)

        assert calls == [], f"Llamadas síncronas prohibidas en CSOS: {calls}"
        assert torch.isfinite(out_g_grp[0]).all() and torch.isfinite(sh_grp).all()
        assert torch.isfinite(out_g_a[0]).all() and torch.isfinite(sh_a).all()
        assert torch.isfinite(out_g_none[0]).all() and torch.isfinite(sh_none).all()
        if csos.mode == "shuffled_session":
            # El desplazamiento es un desarreglo real sin puntos fijos y cambia al avanzar steps.
            ref_mod = CrossSectionalSurpriseModule(VARIANT, dtype=torch.float64)
            with torch.no_grad():
                ref_mod.lambda_logit.fill_(0.7)
                ref_mod.write_weight.fill_(0.5)
            _, sh_real = ref_mod((g0,), el, None)
            assert bool((sh_a != sh_real).all())
            torch.testing.assert_close(
                sh_a.squeeze(-1).sort().values, sh_real.squeeze(-1).sort().values
            )
            assert not torch.equal(sh_a, sh_b) or not torch.equal(out_g_a[0], out_g_b[0])
            # Con grupos=None y steps=0 debe coincidir con steps=None en token_index=0.
            assert torch.equal(sh_a, sh_none) and torch.equal(out_g_a[0], out_g_none[0])

    # Comprobar también determinismo estricto con groups=None y ausencia estática de .item/.tolist.
    import ast

    source = (ROOT / "src/mars_titan/models/titans/cross_sectional_surprise.py").read_text()
    tree = ast.parse(source)
    forbidden_attrs = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("item", "tolist")
    ]
    assert forbidden_attrs == []

    previous_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        mod = CrossSectionalSurpriseModule(PLACEBO, dtype=torch.float64)
        with torch.no_grad():
            mod.lambda_logit.fill_(0.5)
            mod.write_weight.fill_(0.4)
        g_det = torch.randn(
            6, 4, 4, generator=torch.Generator().manual_seed(97), dtype=torch.float64
        )
        el_det = (
            torch.rand(6, 4, generator=torch.Generator().manual_seed(98), dtype=torch.float64) + 0.1
        )
        st_det = torch.arange(6, dtype=torch.int64)
        g1, s1 = mod((g_det,), el_det, None, token_index=1, steps=st_det)
        g2, s2 = mod((g_det,), el_det, None, token_index=1, steps=st_det)
        assert torch.equal(g1[0], g2[0]) and torch.equal(s1, s2)
    finally:
        torch.use_deterministic_algorithms(previous_det)


def test_csos_coexists_with_crmm_and_session_hologram_without_interference():
    from mars_titan.models.titans.cross_sectional_rank_memory import (
        CrossSectionalRankMemoryConfig,
    )
    from mars_titan.models.titans.session_hologram import SessionHologramConfig

    crmm_cfg = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )
    holo_cfg = SessionHologramConfig(
        bands=4, phase_dim=4, mode="interferometric", position="after_mac"
    )

    # 1) Con solo CSOS activada, CRRM y holograma permanecen ausentes.
    solo = FinancialPredictor(financial_config(VARIANT), dtype=torch.float64)
    assert solo.cross_sectional_rank_memory is None and solo.session_hologram is None
    assert "cross_sectional_rank_memory" not in solo.config.identity()
    assert "session_hologram" not in solo.config.identity()

    # 2) Con CSOS + holograma + CRRM juntos: las tres piezas conviven y reciben gradiente finito.
    combo_cfg = financial_config(
        VARIANT,
        cross_sectional_rank_memory=crmm_cfg,
        session_hologram=holo_cfg,
    )
    combo = FinancialPredictor(combo_cfg, dtype=torch.float64)
    assert combo.mac.memory.cross_sectional_surprise is not None
    assert combo.session_hologram is not None
    assert combo.cross_sectional_rank_memory is not None

    batches = multi_step_batches(combo.config.inputs, count=2, dtype=torch.float64)
    st = combo.initial_state(batches[0].flow_ids, differentiable=True)
    rm_st = combo.initial_rank_memory()
    prep0 = combo.prepare(
        batches[0], st, differentiable=True, session_complete=True, rank_memory=rm_st
    )
    rm_st = combo.stage_rank_memory(
        rm_st,
        batches[0].flow_ids,
        batches[0].prediction_at,
        prep0.rank_memory_keys,
        prep0.point_predictions,
        differentiable=True,
    )
    matured = [
        (flow, int(at), float((i - 2.5) * 0.01))
        for i, (flow, at) in enumerate(
            zip(batches[0].flow_ids, batches[0].prediction_at, strict=True)
        )
    ]
    rm_st = combo.settle_rank_memory(rm_st, matured, differentiable=True, train=True)
    prep1 = combo.prepare(
        batches[1],
        prep0.next_state,
        differentiable=True,
        session_complete=True,
        rank_memory=rm_st,
    )
    loss = prep0.point_predictions.square().sum() + prep1.point_predictions.square().sum()
    loss.backward()
    csos_mod = combo.mac.memory.cross_sectional_surprise
    assert csos_mod.lambda_logit.grad is not None and csos_mod.lambda_logit.grad.abs().item() > 0
    assert csos_mod.write_weight.grad is not None and csos_mod.write_weight.grad.abs().item() > 0
    assert combo.session_hologram.gate.grad is not None
    assert torch.isfinite(combo.session_hologram.gate.grad).all()
    assert combo.cross_sectional_rank_memory.out.weight.grad is not None
    assert combo.cross_sectional_rank_memory.out.weight.grad.abs().max().item() > 0

    # 3) Con CSOS apagada, un modelo con CRRM y holograma es bit a bit igual al sin campo.
    base_other = FinancialPredictor(
        financial_config(None, cross_sectional_rank_memory=crmm_cfg, session_hologram=holo_cfg),
        dtype=torch.float64,
    )
    other_no_field = FinancialPredictor(
        FinancialConfig(
            specification(),
            variant="mac_online",
            hidden_size=32,
            gate_bias=GateBias(),
            memory_residual_layer_norm=True,
            memory_inner_gradient=ANALYTIC_INNER_GRADIENT,
            cross_sectional_rank_memory=crmm_cfg,
            session_hologram=holo_cfg,
        ),
        dtype=torch.float64,
    )
    assert base_other.config.identity() == other_no_field.config.identity()
    res_a = base_other.prepare(
        batches[0],
        base_other.initial_state(batches[0].flow_ids),
        session_complete=True,
        rank_memory=base_other.initial_rank_memory(),
    )
    res_b = other_no_field.prepare(
        batches[0],
        other_no_field.initial_state(batches[0].flow_ids),
        session_complete=True,
        rank_memory=other_no_field.initial_rank_memory(),
    )
    assert torch.equal(res_a.point_predictions, res_b.point_predictions)


def test_csos_rejects_auxiliary_horizons_and_mac_rejects_unexpected_session_groups():
    from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons

    with pytest.raises(ValueError, match="cabezas auxiliares"):
        financial_config(csos=VARIANT, auxiliary_horizons=AuxiliaryHorizons(horizons=(5,)))

    mac_plain = TitansMAC(
        MACConfig(memory=memory_config(None, dim=8), heads=2, persistent_tokens=2, max_segment=1),
        dtype=torch.float64,
    )
    groups = session_groups(("US/A", "US/B"), (START, START))
    with pytest.raises(ValueError, match="session_groups"):
        mac_plain(
            torch.zeros(2, 1, 8, dtype=torch.float64),
            mac_plain.initial_state(2),
            session_groups=groups,
        )
    mac_frozen = TitansMAC(
        MACConfig(
            memory=memory_config(None, dim=8),
            heads=2,
            persistent_tokens=2,
            max_segment=1,
            memory_mode="frozen",
        ),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="online"):
        mac_frozen(
            torch.zeros(2, 1, 8, dtype=torch.float64),
            mac_frozen.initial_state(2),
            session_groups=groups,
        )


def test_frozen_consumer_validates_csos_module_and_rejects_tampering():
    active = (
        FinancialPredictor(financial_config(VARIANT), dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    active.mac.memory.cross_sectional_surprise = None
    with pytest.raises(ValueError, match="sorpresa relativa transversal"):
        FrozenFinancialConsumer(active)

    mismatch = (
        FinancialPredictor(financial_config(VARIANT), dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    mismatch.mac.memory.cross_sectional_surprise = (
        CrossSectionalSurpriseModule(PLACEBO, dtype=torch.float64).eval().requires_grad_(False)
    )
    with pytest.raises(ValueError, match="sorpresa relativa transversal"):
        FrozenFinancialConsumer(mismatch)


def test_campaign_v12_screen3_declaration_models_validate_and_isolate_single_variable():
    from mars_titan.models.titans.financial_inputs import FinancialInputSpec
    from mars_titan.training import comparison_campaign as campaign

    doc = json.loads((ROOT / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    control = {
        "machine": "mars_titan",
        "recipe": "core",
        "runner": "fit",
        "tier": "s3_inventions",
        "variant": "mac_online",
        "predictor_changes": {
            "memory_inner_gradient": ANALYTIC_INNER_GRADIENT,
        },
    }
    candidate = {
        "machine": "mars_titan",
        "recipe": "core",
        "runner": "fit",
        "tier": "s3_inventions",
        "variant": "mac_online",
        "predictor_changes": {
            "memory_inner_gradient": ANALYTIC_INNER_GRADIENT,
            "memory_cross_sectional_surprise": {
                "mode": "same_session",
                "bound": 1.0,
                "eps": 1e-6,
                "seed": 0,
            },
        },
    }
    placebo = {
        "machine": "mars_titan",
        "recipe": "core",
        "runner": "fit",
        "tier": "s3_inventions",
        "variant": "mac_online",
        "predictor_changes": {
            "memory_inner_gradient": ANALYTIC_INNER_GRADIENT,
            "memory_cross_sectional_surprise": {
                "mode": "shuffled_session",
                "bound": 1.0,
                "eps": 1e-6,
                "seed": 0,
            },
        },
    }
    augmented = dict(
        doc,
        models=dict(
            doc["models"],
            titans_mac_analytic=control,
            titans_mac_csos=candidate,
            titans_mac_csos_shuffled=placebo,
        ),
    )
    campaign.checked_declaration(augmented, root=ROOT)

    core_recipe = json.loads((ROOT / doc["recipes"]["core"]).read_text())
    base_opts = {k: v for k, v in core_recipe["predictor"].items() if k != "dtype"}
    ctrl_opts = base_opts | control["predictor_changes"]
    cand_opts = base_opts | candidate["predictor_changes"]
    plac_opts = base_opts | placebo["predictor_changes"]

    assert {k for k in base_opts | ctrl_opts if base_opts.get(k) != ctrl_opts.get(k)} == {
        "memory_inner_gradient"
    }
    assert {k for k in ctrl_opts | cand_opts if ctrl_opts.get(k) != cand_opts.get(k)} == {
        "memory_cross_sectional_surprise"
    }
    assert {k for k in cand_opts | plac_opts if cand_opts.get(k) != plac_opts.get(k)} == {
        "memory_cross_sectional_surprise"
    }

    base_spec = specification()
    spec = FinancialInputSpec(
        source_sha256=base_spec.source_sha256,
        view_sha256=base_spec.view_sha256,
        representation=base_spec.representation,
        dimensions=dict(base_spec.dimensions, charts=46),
        input_policy=base_spec.input_policy,
    )
    m_ctrl = FinancialPredictor(
        FinancialConfig(spec, variant="mac_online", seed=42, **ctrl_opts), dtype=torch.float32
    )
    m_cand = FinancialPredictor(
        FinancialConfig(spec, variant="mac_online", seed=42, **cand_opts), dtype=torch.float32
    )
    m_plac = FinancialPredictor(
        FinancialConfig(spec, variant="mac_online", seed=42, **plac_opts), dtype=torch.float32
    )
    assert sum(p.numel() for p in m_cand.parameters()) == sum(
        p.numel() for p in m_plac.parameters()
    )
    assert (
        sum(p.numel() for p in m_cand.parameters())
        == sum(p.numel() for p in m_ctrl.parameters()) + 2
    )
    assert (
        len({json.dumps(m.config.identity(), sort_keys=True) for m in (m_ctrl, m_cand, m_plac)})
        == 3
    )
