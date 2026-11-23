"""Memoria rápida factorizada con blanqueo espectral de sesión (LFM-SW, Idea 4).

Pruebas unitarias y de integración con `FinancialPredictor`: ecuaciones exactas frente a
referencia paso a paso, blanqueo Newton-Schulz, seguimiento de subespacio, aislamiento entre
mercados, invariancia a permutación, precisión FP32/FP64 determinista, gradientes finitos con
predicción constante, modos placebo con idénticos parámetros y registro de campaña.
"""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_adapter import DIMENSIONS, setup
from test_financial_learning_scale import build as build_scaled
from test_session_rank_objective import AT, model, session_batch
from test_technical_dense import DENSE, opened

from mars_titan.models.titans.factor_memory import (
    ALPHA_MIN,
    EVAL_FROZEN,
    FACTOR_MEMORY,
    GAMMA_MIN,
    ONLINE,
    RANDOM_BASIS,
    THETA_MAX,
    FactorMemoryConfig,
    LowRankFactorMemory,
    newton_schulz,
    parse_factor_memory,
)
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.search_cases import checked_predictor_changes

VARIANT = FactorMemoryConfig(factors=4, ns_steps=3, mode=ONLINE)
FROZEN = FactorMemoryConfig(factors=4, ns_steps=3, mode=EVAL_FROZEN)
RANDOM = FactorMemoryConfig(factors=4, ns_steps=3, mode=RANDOM_BASIS)
MARKETS = ("US", "CN", "US", "US", "CN", "US")
FLOWS = ("US/A0003", "CN/B0002", "US/A0001", "US/A0004", "CN/B0001", "US/A0002")
T1 = (100,) * len(FLOWS)
T2 = (200,) * len(FLOWS)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def opened_layer(config=VARIANT, *, hidden=16, seed=7, dtype=torch.float64):
    """Módulo LFM-SW con proyección de salida distinta de cero para observar la residualización."""
    layer = LowRankFactorMemory(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 11)
        layer.out.weight.copy_(torch.randn(hidden, hidden, generator=generator, dtype=dtype) / 4)
    return layer


def encoded(rows=6, hidden=16, seed=13, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=dtype)


def manual_newton_schulz(matrix, steps=3, eps=1e-12):
    """Referencia explícita de la iteración Newton-Schulz quíntica (estilo Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    y = matrix / torch.sqrt((matrix * matrix).sum() + eps)
    for _ in range(steps):
        gram = y @ y.T
        y = a * y + (b * gram + c * (gram @ gram)) @ y
    return y


def manual_step(layer, values, flows, markets, instants, basis_by_market):
    """Referencia explícita ecuación a ecuación de lectura y actualización causal LFM-SW."""
    alpha = layer.alpha()
    theta = layer.theta()
    gamma = layer.gamma()
    eye = torch.eye(layer.config.factors, dtype=values.dtype, device=values.device)
    residuals = torch.zeros_like(values)
    next_basis = {m: f.clone() for m, f in basis_by_market.items()}
    for market in sorted(set(markets)):
        rows = [i for i, m in enumerate(markets) if m == market]
        x = layer.norm(values[rows])
        f = basis_by_market[market]
        inv = torch.linalg.inv(f @ f.T + gamma * eye)
        loadings = x @ f.T @ inv
        r = x - loadings @ f
        residuals[rows] = r
        order = sorted(range(len(rows)), key=lambda idx: flows[rows[idx]])
        l_ord = loadings[order]
        x_ord = x[order]
        g_f = (l_ord.T @ (l_ord @ f - x_ord)) / len(rows)
        next_basis[market] = (1.0 - alpha) * f - theta * manual_newton_schulz(
            g_f, steps=layer.config.ns_steps
        )
    return values + layer.out(residuals), next_basis


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_newton_schulz_matches_reference_equalizes_spectrum_and_handles_zero(dtype):
    generator = torch.Generator().manual_seed(21)
    raw = torch.randn(8, 32, generator=generator, dtype=dtype)
    # Espectro muy desequilibrado: factor 1 domina 50x sobre el factor 8.
    scales = torch.tensor([50.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0], dtype=dtype)
    skewed = scales[:, None] * raw
    actual = newton_schulz(skewed, steps=3)
    expected = manual_newton_schulz(skewed, steps=3)
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    s_before = torch.linalg.svdvals(skewed)
    s_after = torch.linalg.svdvals(actual)
    # El blanqueo espectral comprime el número de condición drásticamente (>50x -> <3x).
    assert (s_before.max() / s_before.min()) > 20.0
    assert (s_after.max() / s_after.min()) < 3.0
    # En cero exacto devuelve cero con gradiente finito (sin 0/0 de la norma).
    zero = torch.zeros(4, 16, dtype=dtype, requires_grad=True)
    out = newton_schulz(zero, steps=3)
    assert torch.count_nonzero(out) == 0
    (grad,) = torch.autograd.grad((out * torch.ones_like(out)).sum(), zero)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_step_matches_row_by_row_reference_across_two_sessions():
    layer = opened_layer(VARIANT)
    state = layer.initial_state("param-1", differentiable=True)
    x1, x2 = encoded(seed=101), encoded(seed=202)
    basis = {"US": layer.initial_factors.clone(), "CN": layer.initial_factors.clone()}
    out1, state1 = layer.step(x1, FLOWS, MARKETS, T1, state, differentiable=True)
    ref1, basis1 = manual_step(layer, x1, FLOWS, MARKETS, T1, basis)
    torch.testing.assert_close(out1, ref1, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state1.factors[0], basis1["US"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state1.factors[1], basis1["CN"], rtol=1e-12, atol=1e-12)
    assert state1.last_updated_at == (100, 100) and state1.updates == (1, 1)

    out2, state2 = layer.step(x2, FLOWS, MARKETS, T2, state1, differentiable=True)
    ref2, basis2 = manual_step(layer, x2, FLOWS, MARKETS, T2, basis1)
    torch.testing.assert_close(out2, ref2, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state2.factors[0], basis2["US"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state2.factors[1], basis2["CN"], rtol=1e-12, atol=1e-12)
    assert state2.last_updated_at == (200, 200) and state2.updates == (2, 2)


def test_online_basis_tracks_latent_factor_subspace_better_than_random_basis():
    hidden, k, n_assets = 32, 4, 64
    cfg_online = FactorMemoryConfig(
        factors=k, ns_steps=3, mode=ONLINE, alpha_init=0.05, theta_init=0.25, gamma_init=0.05
    )
    cfg_random = FactorMemoryConfig(
        factors=k, ns_steps=3, mode=RANDOM_BASIS, alpha_init=0.05, theta_init=0.25, gamma_init=0.05
    )
    online = LowRankFactorMemory(hidden, cfg_online, seed=5, dtype=torch.float64)
    rand = LowRankFactorMemory(hidden, cfg_random, seed=5, dtype=torch.float64)
    with torch.no_grad():
        online.out.weight.copy_(torch.eye(hidden, dtype=torch.float64))
        rand.out.weight.copy_(torch.eye(hidden, dtype=torch.float64))
    gen = torch.Generator().manual_seed(99)
    # Subespacio verdadero de rango 4 con eigenvalores escalonados (factores de estilo).
    q, _ = torch.linalg.qr(torch.randn(hidden, k, generator=gen, dtype=torch.float64))
    true_basis = q.T  # [4, 32]
    flows = tuple(f"US/A{i:04d}" for i in range(n_assets))
    markets = ("US",) * n_assets
    s_on = online.initial_state("p")
    s_rd = rand.initial_state("p")
    err_online, err_random = [], []
    with torch.no_grad():
        for step in range(25):
            loadings = torch.randn(n_assets, k, generator=gen, dtype=torch.float64) * torch.tensor(
                [4.0, 2.5, 1.5, 1.0], dtype=torch.float64
            )
            noise = 0.05 * torch.randn(n_assets, hidden, generator=gen, dtype=torch.float64)
            x = loadings @ true_basis + noise
            at = (1000 + step,) * n_assets
            y_on, s_on = online.step(x, flows, markets, at, s_on)
            y_rd, s_rd = rand.step(x, flows, markets, at, s_rd)
            # out = I => y - x = r_t (residuo tras proyectar sobre la base vigente).
            x_norm = online.norm(x)
            err_online.append(float((y_on - x).norm() / x_norm.norm()))
            err_random.append(float((y_rd - x).norm() / x_norm.norm()))
    # Tras 25 sesiones, la base en línea captura el subespacio latente y reduce el residuo.
    assert err_online[-1] < 0.35 * err_online[0]
    assert err_online[-1] < 0.40 * err_random[-1]


def test_markets_are_strictly_isolated_in_read_update_and_gradients():
    layer = opened_layer(VARIANT)
    state0 = layer.initial_state("p", differentiable=True)
    x1 = encoded(seed=31).requires_grad_()
    x2 = encoded(seed=32).requires_grad_()
    out1, state1 = layer.step(x1, FLOWS, MARKETS, T1, state0, differentiable=True)
    out2, state2 = layer.step(x2, FLOWS, MARKETS, T2, state1, differentiable=True)
    us = [0, 2, 3, 5]
    cn = [1, 4]
    # Perturbar CN en sesión 1 y sesión 2 no altera ni un bit de US ni de F_US.
    x1_alt = x1.detach().clone()
    x2_alt = x2.detach().clone()
    x1_alt[cn] += 10.0
    x2_alt[cn] -= 7.0
    out1_alt, state1_alt = layer.step(x1_alt, FLOWS, MARKETS, T1, state0, differentiable=False)
    out2_alt, state2_alt = layer.step(x2_alt, FLOWS, MARKETS, T2, state1_alt, differentiable=False)
    assert torch.equal(out1_alt[us], out1.detach()[us])
    assert torch.equal(out2_alt[us], out2.detach()[us])
    assert torch.equal(state1_alt.factors[0], state1.factors[0].detach())
    assert torch.equal(state2_alt.factors[0], state2.factors[0].detach())
    # Los gradientes de las salidas de US en t2 no tocan las filas de CN en t1 ni en t2.
    g1, g2 = torch.autograd.grad(out2[us].sum(), (x1, x2), retain_graph=True)
    assert torch.count_nonzero(g1[cn]) == 0
    assert torch.count_nonzero(g2[cn]) == 0
    assert torch.count_nonzero(g1[us]) > 0 and torch.count_nonzero(g2[us]) > 0
    # Y viceversa: perturbar US no altera ni un bit de CN ni de F_CN, ni sus gradientes.
    x1_us_alt = x1.detach().clone()
    x2_us_alt = x2.detach().clone()
    x1_us_alt[us] += 10.0
    x2_us_alt[us] -= 7.0
    out1_u, state1_u = layer.step(x1_us_alt, FLOWS, MARKETS, T1, state0, differentiable=False)
    out2_u, state2_u = layer.step(x2_us_alt, FLOWS, MARKETS, T2, state1_u, differentiable=False)
    assert torch.equal(out1_u[cn], out1.detach()[cn])
    assert torch.equal(out2_u[cn], out2.detach()[cn])
    assert torch.equal(state1_u.factors[1], state1.factors[1].detach())
    assert torch.equal(state2_u.factors[1], state2.factors[1].detach())
    g1_cn, g2_cn = torch.autograd.grad(out2[cn].sum(), (x1, x2))
    assert torch.count_nonzero(g1_cn[us]) == 0 and torch.count_nonzero(g2_cn[us]) == 0
    assert torch.count_nonzero(g1_cn[cn]) > 0 and torch.count_nonzero(g2_cn[cn]) > 0


def test_permuting_rows_within_session_gives_identical_outputs_and_next_factors():
    layer = opened_layer(VARIANT)
    state = layer.initial_state("p")
    x1 = encoded(seed=41)
    out1, state1 = layer.step(x1, FLOWS, MARKETS, T1, state)
    order = [4, 2, 5, 0, 3, 1]
    out1_p, state1_p = layer.step(
        x1[order],
        tuple(FLOWS[i] for i in order),
        tuple(MARKETS[i] for i in order),
        T1,
        state,
    )
    torch.testing.assert_close(out1_p, out1[order], rtol=0, atol=0)
    # La actualización ordena canónicamente por flow_id: F_{m,t+1} es idéntico bit a bit.
    assert torch.equal(state1_p.factors[0], state1.factors[0])
    assert torch.equal(state1_p.factors[1], state1.factors[1])


def test_non_monotonic_or_mixed_session_instants_are_rejected():
    layer = opened_layer(VARIANT)
    state = layer.initial_state("p")
    x = encoded(seed=51)
    # Dos instantes distintos en el mismo mercado dentro del mismo bloque.
    mixed = (100, 100, 100, 101, 100, 100)
    with pytest.raises(ValueError, match="una sola sesión"):
        layer.step(x, FLOWS, MARKETS, mixed, state)
    _, state1 = layer.step(x, FLOWS, MARKETS, T1, state)
    # Repetir o retroceder en el tiempo se rechaza.
    with pytest.raises(ValueError, match="posterior"):
        layer.step(x, FLOWS, MARKETS, T1, state1)
    with pytest.raises(ValueError, match="posterior"):
        layer.step(x, FLOWS, MARKETS, (50,) * len(FLOWS), state1)


def test_all_modes_share_identical_parameters_shapes_and_initialization():
    layers = [
        LowRankFactorMemory(16, cfg, seed=19, dtype=torch.float64)
        for cfg in (VARIANT, FROZEN, RANDOM)
    ]
    ref_params = dict(layers[0].named_parameters())
    for other in layers[1:]:
        params = dict(other.named_parameters())
        assert params.keys() == ref_params.keys()
        for name, tensor in ref_params.items():
            assert torch.equal(params[name], tensor), name
    # Cada modo tiene su propia identidad y rechaza cargas cruzadas.
    identities = {json.dumps(m.config.identity(), sort_keys=True) for m in layers}
    assert len(identities) == 3
    with pytest.raises(ValueError):
        layers[1].load_state_dict(layers[0].state_dict())
    with pytest.raises(ValueError):
        layers[2].load_state_dict(layers[0].state_dict())


def test_eval_frozen_matches_online_in_train_and_freezes_basis_in_eval():
    online = opened_layer(VARIANT)
    frozen = opened_layer(FROZEN)
    x1, x2 = encoded(seed=61), encoded(seed=62)
    # En modo train, eval_frozen es idéntico bit a bit a online.
    online.train()
    frozen.train()
    s_on = online.initial_state("p")
    s_fr = frozen.initial_state("p")
    y1_on, s_on = online.step(x1, FLOWS, MARKETS, T1, s_on)
    y1_fr, s_fr = frozen.step(x1, FLOWS, MARKETS, T1, s_fr)
    y2_on, s_on = online.step(x2, FLOWS, MARKETS, T2, s_on)
    y2_fr, s_fr = frozen.step(x2, FLOWS, MARKETS, T2, s_fr)
    assert torch.equal(y1_fr, y1_on) and torch.equal(y2_fr, y2_on)
    assert torch.equal(s_fr.factors[0], s_on.factors[0])
    # En modo eval, eval_frozen no adapta F_{m,t}: conserva F_0 tras la sesión 1.
    online.eval()
    frozen.eval()
    s_on = online.initial_state("p")
    s_fr = frozen.initial_state("p")
    _, s_on = online.step(x1, FLOWS, MARKETS, T1, s_on)
    _, s_fr = frozen.step(x1, FLOWS, MARKETS, T1, s_fr)
    assert not torch.equal(s_on.factors[0], online.initial_factors)
    assert torch.equal(s_fr.factors[0], frozen.initial_factors)
    y2_on, _ = online.step(x2, FLOWS, MARKETS, T2, s_on)
    y2_fr, _ = frozen.step(x2, FLOWS, MARKETS, T2, s_fr)
    assert not torch.allclose(y2_fr, y2_on)


def test_random_basis_ignores_past_sessions_and_uses_orthonormal_basis():
    rand = opened_layer(RANDOM)
    x1, x1_alt, x2 = encoded(seed=71), encoded(seed=72), encoded(seed=73)
    s0 = rand.initial_state("p")
    _, s1 = rand.step(x1, FLOWS, MARKETS, T1, s0)
    _, s1_alt = rand.step(x1_alt, FLOWS, MARKETS, T1, s0)
    # La sesión 1 no cambia la salida de la sesión 2 en el placebo random_basis.
    y2_a, _ = rand.step(x2, FLOWS, MARKETS, T2, s1)
    y2_b, _ = rand.step(x2, FLOWS, MARKETS, T2, s1_alt)
    assert torch.equal(y2_a, y2_b)
    # La base aleatoria generada es ortonormal por filas y determinista.
    b1 = rand.random_basis("US", 200, dtype=torch.float64, device=torch.device("cpu"))
    b2 = rand.random_basis("US", 200, dtype=torch.float64, device=torch.device("cpu"))
    assert torch.equal(b1, b2)
    torch.testing.assert_close(
        b1 @ b1.T, torch.eye(RANDOM.factors, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    b_other = rand.random_basis("US", 201, dtype=torch.float64, device=torch.device("cpu"))
    assert not torch.allclose(b1, b_other)


def test_disabled_field_keeps_identity_parameters_outputs_and_gradients_bit_exact():
    base, batch = setup()
    explicit = model(factor_memory=None)
    assert explicit.config.identity() == base.config.identity()
    assert explicit._config_id() == base._config_id()
    assert explicit._parameter_id == base._parameter_id
    assert "factor_memory" not in base.config.identity()
    assert base.factor_memory is None and not base.session_level
    for (n1, p1), (n2, p2) in zip(
        base.named_parameters(), explicit.named_parameters(), strict=True
    ):
        assert n1 == n2 and torch.equal(p1, p2)
    out_base = base.prepare(
        batch, base.initial_state(batch.flow_ids, differentiable=True), differentiable=True
    )
    out_exp = explicit.prepare(
        batch, explicit.initial_state(batch.flow_ids, differentiable=True), differentiable=True
    )
    assert out_exp.next_factor_memory is None
    assert torch.equal(out_exp.point_predictions, out_base.point_predictions)
    assert torch.equal(out_exp.working_state, out_base.working_state)
    g_base = torch.autograd.grad(out_base.point_predictions.sum(), list(base.parameters()))
    g_exp = torch.autograd.grad(out_exp.point_predictions.sum(), list(explicit.parameters()))
    for a, b in zip(g_base, g_exp, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("cfg", [VARIANT, FROZEN, RANDOM])
def test_predictor_starts_identical_to_control_and_sends_first_step_gradient(cfg):
    base, _ = setup()
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(factor_memory=cfg)
    ref_params = dict(base.named_parameters())
    new_params = dict(predictor.named_parameters())
    assert set(new_params) - set(ref_params) == {
        "factor_memory.norm.weight",
        "factor_memory.norm.bias",
        "factor_memory.initial_factors",
        "factor_memory.alpha_logit",
        "factor_memory.theta_logit",
        "factor_memory.gamma_logit",
        "factor_memory.out.weight",
    }
    for name, tensor in ref_params.items():
        assert torch.equal(new_params[name], tensor), name
    assert torch.count_nonzero(predictor.factor_memory.out.weight) == 0
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    f_state = predictor.initial_factor_memory()
    actual = predictor.prepare(
        batch,
        predictor.initial_state(batch.flow_ids),
        session_complete=True,
        factor_memory=f_state,
    )
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    # En el primer paso el gradiente llega a out.weight sin ceros encadenados.
    state = predictor.initial_state(batch.flow_ids, differentiable=True)
    f0 = predictor.initial_factor_memory(differentiable=True)
    emitted = predictor.prepare(
        batch, state, differentiable=True, session_complete=True, factor_memory=f0
    )
    (g_out,) = torch.autograd.grad(
        emitted.point_predictions.sum(), (predictor.factor_memory.out.weight,)
    )
    assert torch.isfinite(g_out).all() and g_out.abs().sum() > 0


def test_second_session_reaches_all_lfmsw_parameters_when_output_is_open():
    batch1 = session_batch(6, markets=list(MARKETS), at=AT, seed=5)
    batch2 = session_batch(6, markets=list(MARKETS), at=AT + 86_400_000_000, seed=6)
    predictor = model(factor_memory=VARIANT)
    with torch.no_grad():
        predictor.factor_memory.out.weight.fill_(0.05)
    predictor._seal_parameters()
    s0 = predictor.initial_state(batch1.flow_ids, differentiable=True)
    f0 = predictor.initial_factor_memory(differentiable=True)
    p1 = predictor.prepare(batch1, s0, differentiable=True, session_complete=True, factor_memory=f0)
    p2 = predictor.prepare(
        batch2,
        p1.next_state,
        differentiable=True,
        session_complete=True,
        factor_memory=p1.next_factor_memory,
    )
    lfmsw_params = [
        (n, p) for n, p in predictor.named_parameters() if n.startswith("factor_memory.")
    ]
    grads = torch.autograd.grad(p2.point_predictions.sum(), [p for _, p in lfmsw_params])
    for (name, _), grad in zip(lfmsw_params, grads, strict=True):
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, f"Gradiente nulo en {name}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constant_prediction_and_zero_inputs_produce_finite_gradients_deterministically(dtype):
    from mars_titan.models.quantile_head import pinball_loss
    from mars_titan.models.titans.session_ic_objective import SessionICTerm

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        # Modelo con zero_median=True (predicción mediana constante = 0) y rama técnica.
        predictor, _ = build_scaled(
            technical_branch=asdict(DENSE),
            median_gain="learned_median_gain_v1",
            factor_memory=asdict(VARIANT),
        )
        opened(predictor.technical_branch)
        predictor.to(dtype=dtype)
        # Lote de 6 activos en la precisión `dtype` para que SessionICTerm (minimum_rows=3) actúe.
        rng = np.random.default_rng(19)
        rows = 6
        values = {
            name: rng.normal(size=(rows, 64, width) if name == "prices" else (rows, width))
            .astype(np.float32)
            .round(3)
            for name, width in DIMENSIONS.items()
        }
        presence = np.ones((rows, 5), dtype=np.bool_)
        values["fundamentals"][:, 1:] = [1, 0.5]
        values["macro"][:] = [0.2, 1, 0]
        raw = dict(
            inputs=values,
            presence=presence,
            sample_ids=[f"US/A{i:03}/{AT}" for i in range(rows)],
            market=["US"] * rows,
            prediction_at=np.full(rows, AT, dtype="datetime64[us]"),
            input_available_at=np.full(rows, AT - 1, dtype="datetime64[us]"),
        )
        batch = DecisionBatch.from_corpus(raw, predictor.config.inputs, dtype=dtype)
        predictor._seal_parameters()
        s0 = predictor.initial_state(batch.flow_ids, differentiable=True)
        f0 = predictor.initial_factor_memory(differentiable=True)
        emitted = predictor.prepare(
            batch, s0, differentiable=True, session_complete=True, factor_memory=f0
        )
        # La mediana es exactamente constante cero en el arranque.
        assert torch.count_nonzero(emitted.point_predictions) == 0
        target = torch.linspace(-0.02, 0.02, len(batch.flow_ids), dtype=dtype)
        sessions = [
            (f.split("/", 1)[0], t)
            for f, t in zip(batch.flow_ids, batch.prediction_at, strict=True)
        ]
        ic_term = SessionICTerm(weight=3.0, minimum_rows=3)
        loss = pinball_loss(emitted.quantiles, target) + ic_term.term(
            emitted.point_predictions, target, sessions
        )
        loss.backward()
        assert predictor.factor_memory.out.weight.grad is not None
        assert predictor.factor_memory.out.weight.grad.abs().sum() > 0
        for name, param in predictor.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradiente no finito en {name}"

        # Determinismo estricto en FP32/FP64 con entradas nulas también da gradientes finitos.
        layer = opened_layer(VARIANT, dtype=dtype)
        zeros = torch.zeros(6, 16, dtype=dtype, requires_grad=True)
        fs0 = layer.initial_state("p", differentiable=True)
        y1, fs1 = layer.step(zeros, FLOWS, MARKETS, T1, fs0, differentiable=True)
        y2, fs2 = layer.step(zeros, FLOWS, MARKETS, T2, fs1, differentiable=True)
        y2.sum().backward()
        assert torch.isfinite(zeros.grad).all()
        for name, param in layer.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
        # Repetir produce exactamente los mismos bits.
        y1_b, fs1_b = layer.step(zeros.detach(), FLOWS, MARKETS, T1, layer.initial_state("p"))
        assert torch.equal(y1_b, y1.detach())
        assert torch.equal(fs1_b.factors[0], fs1.factors[0].detach())
    finally:
        torch.use_deterministic_algorithms(previous)


def test_state_export_restore_round_trip_is_bit_exact_and_rejects_tampering():
    batch = session_batch(6, markets=list(MARKETS), at=AT, seed=5)
    predictor = model(factor_memory=VARIANT)
    s0 = predictor.initial_state(batch.flow_ids)
    f0 = predictor.initial_factor_memory()
    p1 = predictor.prepare(batch, s0, session_complete=True, factor_memory=f0)
    exported = predictor.export_factor_memory(p1.next_factor_memory)
    restored = predictor.restore_factor_memory(exported)
    assert restored.last_updated_at == p1.next_factor_memory.last_updated_at
    assert restored.updates == p1.next_factor_memory.updates
    for a, b in zip(restored.factors, p1.next_factor_memory.factors, strict=True):
        assert torch.equal(a, b)
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    # Continuar desde el estado restaurado coincide bit a bit.
    batch2 = session_batch(6, markets=list(MARKETS), at=AT + 86_400_000_000, seed=6)
    p2_orig = predictor.prepare(
        batch2, p1.next_state, session_complete=True, factor_memory=p1.next_factor_memory
    )
    p2_rest = predictor.prepare(
        batch2, p1.next_state, session_complete=True, factor_memory=restored
    )
    assert torch.equal(p2_rest.point_predictions, p2_orig.point_predictions)
    assert torch.equal(p2_rest.next_factor_memory.factors[0], p2_orig.next_factor_memory.factors[0])
    # Alteraciones de configuración, NaN o forma se rechazan.
    bad_cfg = copy.deepcopy(exported)
    bad_cfg["configuration"]["factors"] = 99
    with pytest.raises(ValueError):
        predictor.restore_factor_memory(bad_cfg)
    bad_nan = copy.deepcopy(exported)
    bad_nan["factors"][0][0, 0] = float("nan")
    with pytest.raises(ValueError):
        predictor.restore_factor_memory(bad_nan)
    bad_shape = copy.deepcopy(exported)
    bad_shape["factors"][0] = bad_shape["factors"][0][:, :-1]
    with pytest.raises(ValueError):
        predictor.restore_factor_memory(bad_shape)
    with pytest.raises(ValueError):
        model(factor_memory=FROZEN).restore_factor_memory(exported)


def test_factor_memory_requires_complete_session_and_explicit_state():
    batch = session_batch()
    predictor = model(factor_memory=VARIANT)
    assert predictor.session_level
    state = predictor.initial_state(batch.flow_ids)
    f0 = predictor.initial_factor_memory()
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, state, factor_memory=f0)
    with pytest.raises(ValueError, match="memoria factorial"):
        predictor.prepare(batch, state, session_complete=True)
    plain = model()
    with pytest.raises(ValueError, match="memoria factorial"):
        plain.prepare(batch, plain.initial_state(batch.flow_ids), factor_memory=f0)


def test_autocast_and_invalid_configurations_are_rejected():
    layer = LowRankFactorMemory(16, VARIANT, seed=1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer.step(encoded(dtype=torch.float32), FLOWS, MARKETS, T1, layer.initial_state("p"))
    with pytest.raises(ValueError, match="FP32 o FP64"):
        LowRankFactorMemory(16, VARIANT, seed=1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="menor que hidden_size"):
        LowRankFactorMemory(4, FactorMemoryConfig(factors=4), seed=1)
    base, _ = setup()
    for bad in (
        dict(factors=0, mode=ONLINE),
        dict(factors=True, mode=ONLINE),
        dict(factors=32, mode=ONLINE),  # == hidden_size (32)
        dict(factors=4, mode="unknown"),
        dict(factors=4, mode=ONLINE, ns_steps=0),
        dict(factors=4, mode=ONLINE, ns_steps=9),
        dict(factors=4, mode=ONLINE, alpha_init=ALPHA_MIN),
        dict(factors=4, mode=ONLINE, alpha_init=1.0),
        dict(factors=4, mode=ONLINE, theta_init=0.0),
        dict(factors=4, mode=ONLINE, theta_init=THETA_MAX),
        dict(factors=4, mode=ONLINE, gamma_init=GAMMA_MIN),
        dict(factors=4, mode=ONLINE, extra_field=1),
        dict(factors=4),
        FACTOR_MEMORY,
    ):
        with pytest.raises(ValueError):
            replace(base.config, factor_memory=bad)
    # Declaración mínima con 2 campos equivale a defaults completos.
    assert parse_factor_memory(dict(factors=16, mode=ONLINE)) == FactorMemoryConfig()


def test_registry_code_identity_campaign_and_consumer_guards():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, declared_sources
    from mars_titan.training.dust.financial import FinancialDustAdapter
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    changes = dict(factor_memory=dict(factors=16, mode=ONLINE))
    assert checked_predictor_changes(changes) == changes
    # `declared_sources` solo incluye el módulo cuando la opción está activa.
    mod = "mars_titan.models.titans.factor_memory"
    plain_pred = model().eval().requires_grad_(False)
    lfmsw_pred = model(factor_memory=VARIANT).eval().requires_grad_(False)
    assert mod not in declared_sources(plain_pred.config)
    assert declared_sources(lfmsw_pred.config) == (mod,)

    # `ChronologicalTrainer._code()` solo añade el módulo cuando está activo.
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
                factor_memory=cfg,
            )
        )
        assert (mod in trainer._code()) is expected

    # El consumidor congelado y Dust rechazan explícitamente factor_memory.
    with pytest.raises(ValueError, match="memoria factorial"):
        FrozenFinancialConsumer(lfmsw_pred)
    direct_lfmsw = model(variant="transformer_direct", factor_memory=VARIANT)
    with pytest.raises(ValueError, match="Dust"):
        FinancialDustAdapter(direct_lfmsw)

    # Los modelos candidato y placebos pasan la validación real de us-campaign-v11-screen1.json.
    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v11-screen1.json").read_text())
    doc["tiers"].append("y1_lfmsw")
    doc["models"]["titans_mac_lfmsw"] = dict(
        machine="mars_titan",
        recipe="core",
        runner="fit",
        tier="y1_lfmsw",
        variant="mac_online",
        predictor_changes=dict(factor_memory=dict(factors=16, mode=ONLINE)),
    )
    doc["models"]["titans_mac_lfmsw_random"] = dict(
        machine="mars_titan",
        recipe="core",
        runner="fit",
        tier="y1_lfmsw",
        variant="mac_online",
        predictor_changes=dict(factor_memory=dict(factors=16, mode=RANDOM_BASIS)),
    )
    doc["models"]["titans_mac_lfmsw_eval_frozen"] = dict(
        machine="mars_titan",
        recipe="core",
        runner="fit",
        tier="y1_lfmsw",
        variant="mac_online",
        predictor_changes=dict(factor_memory=dict(factors=16, mode=EVAL_FROZEN)),
    )
    checked = campaign.checked_declaration(doc, root=root)
    assert "titans_mac_lfmsw" in checked["models"]
    assert "titans_mac_lfmsw_random" in checked["models"]
    assert "titans_mac_lfmsw_eval_frozen" in checked["models"]


def _graph_nodes(tensor):
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(parent for parent, _ in node.next_functions)
    return len(seen)


def test_pinned_values_match_previous_base_patch_with_only_factor_memory_enabled():
    """Con solo factor_memory activada, el módulo y el predictor dan exactamente los valores
    fijados del parche sobre la base anterior (sin contaminación de opciones de v12)."""
    pinned_layer = {
        torch.float64: (
            [-0.009632849888197154, 1.2935389450951955, 1.2099414014597714, 0.5936206527301245],
            [0.4596591613272628, 0.8312564806398676, -0.5315283283560989, -0.7643159888231221],
            [-0.34731147848908106, 0.08759679343308774, -0.19686610875612387, -0.3567047361541333],
            [-0.34157834237287166, 0.10786056902312186, -0.24878935967179588, -0.3667097588967224],
        ),
        torch.float32: (
            [0.28335660696029663, -0.28824830055236816, -1.132095217704773, 0.3879220485687256],
            [-1.5713212490081787, -0.8393475413322449, 0.5197340250015259, -1.8844977617263794],
            [-0.3618566393852234, 0.11482211202383041, -0.18961022794246674, -0.36762285232543945],
            [-0.3676528334617615, 0.11516273766756058, -0.22484424710273743, -0.3438529372215271],
        ),
    }
    for dtype, (exp_y1, exp_y2, exp_us, exp_cn) in pinned_layer.items():
        layer = opened_layer(VARIANT, hidden=16, seed=7, dtype=dtype)
        with torch.no_grad():
            x1 = encoded(seed=101, dtype=dtype)
            x2 = encoded(seed=202, dtype=dtype)
            y1, s1 = layer.step(x1, FLOWS, MARKETS, T1, layer.initial_state("p"))
            y2, s2 = layer.step(x2, FLOWS, MARKETS, T2, s1)
        tol = 1e-13 if dtype == torch.float64 else 1e-6
        torch.testing.assert_close(y1[0, :4], torch.tensor(exp_y1, dtype=dtype), rtol=tol, atol=tol)
        torch.testing.assert_close(y2[0, :4], torch.tensor(exp_y2, dtype=dtype), rtol=tol, atol=tol)
        torch.testing.assert_close(
            s2.factors[0][0, :4], torch.tensor(exp_us, dtype=dtype), rtol=tol, atol=tol
        )
        torch.testing.assert_close(
            s2.factors[1][0, :4], torch.tensor(exp_cn, dtype=dtype), rtol=tol, atol=tol
        )

    b1 = session_batch(6, markets=list(MARKETS), at=AT, seed=5)
    b2 = session_batch(6, markets=list(MARKETS), at=AT + 86_400_000_000, seed=6)
    pred = model(factor_memory=VARIANT)
    assert pred.session_hologram is None and pred.cross_sectional_rank_memory is None
    assert "session_hologram" not in pred.config.identity()
    assert "cross_sectional_rank_memory" not in pred.config.identity()
    with torch.no_grad():
        pred.factor_memory.out.weight.copy_(
            torch.randn(32, 32, generator=torch.Generator().manual_seed(77), dtype=torch.float64)
            / 8
        )
    pred._seal_parameters()
    p1 = pred.prepare(
        b1,
        pred.initial_state(b1.flow_ids),
        session_complete=True,
        factor_memory=pred.initial_factor_memory(),
    )
    p2 = pred.prepare(
        b2,
        p1.next_state,
        session_complete=True,
        factor_memory=p1.next_factor_memory,
    )
    exp_p1 = torch.tensor(
        [
            -0.0011509477259754076,
            -6.957691010428882e-05,
            0.0020548449188961555,
            0.00013412575562920213,
            -0.0047196039602282296,
            0.0023562985944337433,
        ],
        dtype=torch.float64,
    )
    exp_p2 = torch.tensor(
        [
            0.0091313727246377,
            0.009102447470767413,
            0.009713765965515136,
            0.009552409182721863,
            0.009228646805710564,
            0.009633084696407505,
        ],
        dtype=torch.float64,
    )
    exp_f2 = torch.tensor(
        [0.028531273621075826, 0.42230984029315627, 0.16609803358335956, -0.03378293970013034],
        dtype=torch.float64,
    )
    torch.testing.assert_close(p1.point_predictions, exp_p1, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(p2.point_predictions, exp_p2, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        p2.next_factor_memory.factors[0][0, :4], exp_f2, rtol=1e-13, atol=1e-13
    )


def test_cholesky_uses_deferred_cholesky_ex_and_rejects_non_spd_via_require_true(monkeypatch):
    """La factorización usa cholesky_ex(check_errors=False) sin sincronizar y valida info con
    require_true."""
    layer = opened_layer(VARIANT)
    state = layer.initial_state("p", differentiable=True)
    x = encoded(seed=88)

    calls = []
    real_cholesky_ex = torch.linalg.cholesky_ex

    def spy_cholesky_ex(matrix, *, upper=False, check_errors=False, out=None):
        calls.append(check_errors)
        return real_cholesky_ex(matrix, upper=upper, check_errors=check_errors, out=out)

    def forbid_cholesky(*args, **kwargs):
        raise AssertionError("No debe llamarse torch.linalg.cholesky síncrono")

    monkeypatch.setattr(torch.linalg, "cholesky", forbid_cholesky)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", spy_cholesky_ex)
    out, next_state = layer.step(x, FLOWS, MARKETS, T1, state, differentiable=True)
    assert calls == [False, False]
    assert torch.isfinite(out).all() and torch.isfinite(next_state.factors[0]).all()

    # Si info != 0, require_true rechaza la factorización con ValueError explícito.
    def failing_cholesky_ex(matrix, *, upper=False, check_errors=False, out=None):
        chol, _ = real_cholesky_ex(matrix, upper=upper, check_errors=False)
        return chol, torch.ones((), dtype=torch.int32, device=matrix.device)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", failing_cholesky_ex)
    with pytest.raises(ValueError, match="definido positivo"):
        layer.step(x, FLOWS, MARKETS, T1, state, differentiable=True)


@pytest.mark.parametrize("sorted_flows", [True, False])
def test_autograd_graph_size_and_state_size_do_not_grow_with_session_rows(sorted_flows):
    """El grafo de autograd y el tamaño de estado son O(1) en el número de filas de la sesión."""
    out_nodes, state_nodes, state_bytes = {}, {}, {}
    for rows in (8, 64, 256):
        layer = opened_layer(VARIANT, hidden=16, seed=7, dtype=torch.float64)
        flows = tuple(f"US/A{i:04d}" for i in range(rows))
        if not sorted_flows:
            flows = tuple(reversed(flows))
        markets = ("US",) * rows
        instants = (100,) * rows
        x = encoded(rows=rows, hidden=16, seed=55, dtype=torch.float64).requires_grad_()
        s0 = layer.initial_state("p", differentiable=True)
        out, s1 = layer.step(x, flows, markets, instants, s0, differentiable=True)
        out_nodes[rows] = _graph_nodes(out)
        state_nodes[rows] = _graph_nodes(s1.factors[0])
        state_bytes[rows] = sum(t.numel() * t.element_size() for t in s1.factors)
        (gx,) = torch.autograd.grad(s1.factors[0].sum(), x)
        assert torch.isfinite(gx).all() and gx.abs().sum() > 0
    assert out_nodes[8] == out_nodes[64] == out_nodes[256], out_nodes
    assert state_nodes[8] == state_nodes[64] == state_nodes[256], state_nodes
    assert state_bytes[8] == state_bytes[64] == state_bytes[256], state_bytes


def test_coexistence_and_canonical_order_before_head_with_hologram_and_crrm(monkeypatch):
    """Orden antes de la cabeza: lecturas de sesión (holograma) -> memorias no supervisadas
    (factor_memory) -> memorias de etiquetas maduras (CRRM) -> cabeza."""
    from mars_titan.models.titans.cross_sectional_rank_memory import (
        CrossSectionalRankMemoryConfig,
    )
    from mars_titan.models.titans.frozen_financial import declared_sources
    from mars_titan.models.titans.session_hologram import SessionHologramConfig

    holo_cfg = SessionHologramConfig(
        bands=4, phase_dim=8, mode="interferometric", position="after_technical"
    )
    crrm_cfg = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )
    combo, _ = build_scaled(
        technical_branch=asdict(DENSE),
        session_hologram=asdict(holo_cfg),
        factor_memory=asdict(VARIANT),
        cross_sectional_rank_memory=asdict(crrm_cfg),
    )
    combo.to(dtype=torch.float64)
    assert declared_sources(combo.config) == (
        "mars_titan.models.titans.technical_branch",
        "mars_titan.models.titans.cross_sectional_rank_memory",
        "mars_titan.models.titans.session_hologram",
        "mars_titan.models.titans.factor_memory",
        "mars_titan.models.learning_scale",
    )
    with torch.no_grad():
        opened(combo.technical_branch)
        combo.session_hologram.gate.fill_(0.2)
        combo.factor_memory.out.weight.fill_(0.05)
        combo.cross_sectional_rank_memory.out.weight.fill_(0.1)
    combo._seal_parameters()

    b1 = session_batch(6, markets=list(MARKETS), at=AT, seed=5)
    b2 = session_batch(6, markets=list(MARKETS), at=AT + 86_400_000_000, seed=6)
    s0 = combo.initial_state(b1.flow_ids, differentiable=True)
    f0 = combo.initial_factor_memory(differentiable=True)
    r0 = combo.initial_rank_memory()
    p1 = combo.prepare(
        b1, s0, differentiable=True, session_complete=True, factor_memory=f0, rank_memory=r0
    )
    # Madurar etiquetas de la sesión 1 para que CRRM también produzca residuo no nulo en sesión 2.
    r1 = combo.stage_rank_memory(
        r0,
        b1.flow_ids,
        b1.prediction_at,
        p1.rank_memory_keys,
        p1.point_predictions,
        differentiable=True,
    )
    matured = [(flow, AT, 0.01 * (i - 2.5)) for i, flow in enumerate(b1.flow_ids)]
    r1 = combo.settle_rank_memory(r1, matured, differentiable=True, train=True)

    trace = []
    captured = {}
    orig_holo = combo.session_hologram.forward
    orig_fm = combo.factor_memory.step
    orig_crrm_keys = combo.cross_sectional_rank_memory.keys
    orig_crrm_read = combo.cross_sectional_rank_memory.read
    orig_head = combo.head.forward

    def wrap_holo(enc, mkts, at):
        trace.append("session_hologram")
        captured["before_holo"] = enc.detach().clone()
        out = orig_holo(enc, mkts, at)
        captured["after_holo"] = out.detach().clone()
        return out

    def wrap_fm(enc, flows, mkts, at, st, *, differentiable=False):
        trace.append("factor_memory")
        captured["before_fm"] = enc.detach().clone()
        out, nxt = orig_fm(enc, flows, mkts, at, st, differentiable=differentiable)
        captured["after_fm"] = out.detach().clone()
        return out, nxt

    def wrap_crrm_keys(enc, mkts, at):
        trace.append("crrm_keys")
        captured["before_crrm"] = enc.detach().clone()
        return orig_crrm_keys(enc, mkts, at)

    def wrap_crrm_read(keys, mkts, st):
        trace.append("crrm_read")
        delta = orig_crrm_read(keys, mkts, st)
        captured["crrm_delta"] = delta.detach().clone()
        return delta

    def wrap_head(enc):
        trace.append("head")
        captured["before_head"] = enc.detach().clone()
        return orig_head(enc)

    monkeypatch.setattr(combo.session_hologram, "forward", wrap_holo)
    monkeypatch.setattr(combo.factor_memory, "step", wrap_fm)
    monkeypatch.setattr(combo.cross_sectional_rank_memory, "keys", wrap_crrm_keys)
    monkeypatch.setattr(combo.cross_sectional_rank_memory, "read", wrap_crrm_read)
    monkeypatch.setattr(combo.head, "forward", wrap_head)

    p2 = combo.prepare(
        b2,
        p1.next_state,
        differentiable=True,
        session_complete=True,
        factor_memory=p1.next_factor_memory,
        rank_memory=r1,
    )
    assert trace == ["session_hologram", "factor_memory", "crrm_keys", "crrm_read", "head"]
    assert torch.equal(captured["before_fm"], captured["after_holo"])
    assert torch.equal(captured["before_crrm"], captured["after_fm"])
    torch.testing.assert_close(
        captured["before_head"],
        captured["after_fm"] + captured["crrm_delta"],
        rtol=0.0,
        atol=0.0,
    )
    # Los tres módulos modifican activamente la representación y reciben gradientes finitos.
    assert not torch.equal(captured["after_holo"], captured["before_holo"])
    assert not torch.equal(captured["after_fm"], captured["before_fm"])
    assert torch.count_nonzero(captured["crrm_delta"]) > 0
    grads = torch.autograd.grad(
        p2.quantiles.sum(),
        (
            combo.session_hologram.gate,
            combo.factor_memory.out.weight,
            combo.cross_sectional_rank_memory.out.weight,
        ),
    )
    for g in grads:
        assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_v12_screen3_campaign_accepts_lfmsw_candidate_and_placebos_as_single_variable():
    """Candidato y placebos LFM-SW se validan sobre us-campaign-v12-screen3.json y difieren del
    control `titans_mac` en un único campo (`factor_memory`)."""
    from test_financial_adapter import specification

    from mars_titan.models.titans.financial import FinancialConfig
    from mars_titan.models.titans.financial_inputs import FinancialInputSpec

    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    entries = {
        "titans_mac_lfmsw": dict(factors=16, mode=ONLINE),
        "titans_mac_lfmsw_random": dict(factors=16, mode=RANDOM_BASIS),
        "titans_mac_lfmsw_eval_frozen": dict(factors=16, mode=EVAL_FROZEN),
    }
    for name, fm_cfg in entries.items():
        doc["models"][name] = dict(
            machine="mars_titan",
            recipe="core",
            runner="fit",
            tier="s3_inventions",
            variant="mac_online",
            predictor_changes=dict(factor_memory=fm_cfg),
        )
    checked = campaign.checked_declaration(doc, root=root)
    for name in entries:
        assert name in checked["models"]

    # Verificar diferencia de una sola variable en la identidad real de la receta `core`.
    recipe_doc = json.loads((root / doc["recipes"]["core"]).read_text())
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
    control_id = FinancialConfig(spec, variant="mac_online", **predictor_opts).identity()
    for fm_cfg in entries.values():
        changes = checked_predictor_changes(dict(factor_memory=fm_cfg))
        cand_id = FinancialConfig(
            spec, variant="mac_online", **(predictor_opts | changes)
        ).identity()
        assert set(cand_id) - set(control_id) == {"factor_memory"}
        assert {k: v for k, v in cand_id.items() if k != "factor_memory"} == control_id
