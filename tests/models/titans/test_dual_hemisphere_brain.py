"""Cerebro de doble hemisferio con ortogonalización transversal (`dual_hemisphere_brain_v1`).

Pruebas de contrato, equivalencia exacta del operador de pares leave-one-out frente a la
referencia cuadrática O(N²), teorema de ortogonalidad de subespacio completo sobre la proyección
escalar final, bloqueo de gradientes hacia el hemisferio primario (stopgrad), ceguera estructural
frente al bloque gráfico y los rasgos técnicos, causalidad estricta entre mercados e instantes,
determinismo FP32/FP64, gradientes finitos con predicción constante y compatibilidad congelada.
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
from test_financial_adapter import DIMENSIONS, setup
from test_financial_learning_scale import build
from test_frozen_financial import bank, consumer, frozen
from test_session_rank_objective import model, session_batch
from test_technical_dense import DENSE, opened

from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.dual_hemisphere_brain import (
    DUAL_HEMISPHERE_BRAIN,
    ORTHOGONAL,
    PEER_EPS,
    SELF_ONLY,
    SHUFFLED_PEERS,
    UNORTHOGONALIZED,
    DualHemisphereBrain,
    DualHemisphereBrainConfig,
    loo_peer_readout,
    parse_dual_hemisphere_brain,
)
from mars_titan.models.titans.episodic_readout import apply_episodic_readout
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm
from mars_titan.training import comparison_campaign
from mars_titan.training.dust.financial import FinancialDustAdapter
from mars_titan.training.search_cases import checked_predictor_changes

CODHB_MODULE = "mars_titan.models.titans.dual_hemisphere_brain"
ORTHO = DualHemisphereBrainConfig(dim=16, ridge=1e-4, mode=ORTHOGONAL)
UNORTHO = DualHemisphereBrainConfig(dim=16, ridge=1e-4, mode=UNORTHOGONALIZED)
SELF_CFG = DualHemisphereBrainConfig(dim=16, ridge=1e-4, mode=SELF_ONLY)
SHUF_CFG = DualHemisphereBrainConfig(dim=16, ridge=1e-4, mode=SHUFFLED_PEERS)
MARKETS = ("US", "CN", "US", "US", "CN", "US", "US", "CN", "US", "US")
AT = (7,) * len(MARKETS)


def sample_panel(
    rows=10,
    hidden=16,
    *,
    context=64,
    price_channels=6,
    news_dim=8,
    fund_dim=6,
    macro_dim=6,
    seed=9,
    dtype=torch.float64,
):
    """Panel sintético de sesión con precios logarítmicos, modalidades no técnicas y presencia."""
    gen = torch.Generator().manual_seed(seed)
    encoded = torch.randn(rows, hidden, generator=gen, dtype=dtype)
    step_ret = 0.02 * torch.randn(rows, context, generator=gen, dtype=dtype)
    close = step_ret.cumsum(dim=1)
    close = close - close[:, :1]
    opening = close - 0.005 * torch.randn(rows, context, generator=gen, dtype=dtype)
    high = torch.maximum(opening, close) + 0.01 * torch.rand(
        rows, context, generator=gen, dtype=dtype
    )
    low = torch.minimum(opening, close) - 0.01 * torch.rand(
        rows, context, generator=gen, dtype=dtype
    )
    vol = torch.log1p(0.2 + 2.0 * torch.rand(rows, context, generator=gen, dtype=dtype))
    channels = [opening, high, low, close, vol]
    if price_channels == 6:
        channels.append(torch.ones(rows, context, dtype=dtype))
    prices = torch.stack(channels, dim=-1)
    news = torch.randn(rows, news_dim, generator=gen, dtype=dtype)
    fundamentals = torch.randn(rows, fund_dim, generator=gen, dtype=dtype)
    # Macro constante dentro de la sesión, como en los datos financieros reales.
    macro_row = torch.randn(1, macro_dim, generator=gen, dtype=dtype)
    macro = macro_row.expand(rows, macro_dim).clone()
    presence = torch.ones((rows, 5), dtype=torch.bool)
    dims = dict(
        prices=price_channels,
        news=news_dim,
        charts=4,
        fundamentals=fund_dim,
        macro=macro_dim,
    )
    return encoded, prices, news, fundamentals, macro, presence, dims


def opened_brain(config=ORTHO, *, hidden=16, dims=None, seed=3, dtype=torch.float64):
    """Módulo CODHB con puerta y proyección macro abiertas para observar su efecto completo."""
    dims = dims or dict(prices=6, news=8, charts=4, fundamentals=6, macro=6)
    brain = DualHemisphereBrain(hidden, dims, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed + 101)
        brain.gate.copy_(torch.randn(hidden, generator=gen, dtype=dtype) / 3)
        brain.macro_proj.weight.copy_(
            torch.randn(brain.macro_proj.weight.shape, generator=gen, dtype=dtype) / 4
        )
        brain.macro_proj.bias.copy_(
            torch.randn(brain.macro_proj.bias.shape, generator=gen, dtype=dtype) / 4
        )
    return brain


def naive_loo_peer_reference(basis, values):
    """Referencia cuadrática O(N²) par a par para la lectura leave-one-out de pares."""
    rows = len(basis)
    out = torch.zeros_like(values)
    stds = torch.zeros(rows, dtype=values.dtype, device=values.device)
    for i in range(rows):
        peers = [j for j in range(rows) if j != i]
        rhos = torch.stack([(basis[i] * basis[j]).sum() for j in peers])
        rho_bar = rhos.mean()
        centered = rhos - rho_bar
        var = centered.square().mean()
        std = torch.sqrt(var + PEER_EPS)
        peer_vals = torch.stack([values[j] for j in peers], dim=0)
        cov = (centered.unsqueeze(-1) * peer_vals).mean(dim=0)
        out[i] = cov / std
        stds[i] = std
    return out, stds


@pytest.mark.parametrize("rows", [2, 3, 7, 15])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loo_peer_operator_matches_explicit_quadratic_reference(rows, dtype):
    gen = torch.Generator().manual_seed(17 + rows)
    raw = torch.randn(rows, 12, generator=gen, dtype=dtype)
    basis = raw / torch.sqrt(raw.square().sum(dim=-1, keepdim=True) + 1e-8)
    values = torch.randn(rows, 5, generator=gen, dtype=dtype)
    actual, actual_std = loo_peer_readout(basis, values)
    expected, expected_std = naive_loo_peer_reference(basis, values)
    tol = 1e-5 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(actual_std, expected_std, rtol=tol, atol=tol)


def test_loo_peer_operator_has_zero_self_echo_and_annihilates_uniform_shift():
    gen = torch.Generator().manual_seed(29)
    raw = torch.randn(8, 16, generator=gen, dtype=torch.float64)
    basis = raw / torch.sqrt(raw.square().sum(dim=-1, keepdim=True) + 1e-8)
    values = torch.randn(8, 6, generator=gen, dtype=torch.float64).requires_grad_(True)

    readout, _ = loo_peer_readout(basis, values)
    target = 3
    (grad_v,) = torch.autograd.grad(readout[target].sum(), (values,))
    # Cero dependencia respecto al propio activo i; dependencia activa respecto a pares j != i.
    torch.testing.assert_close(grad_v[target], torch.zeros_like(grad_v[target]), rtol=0, atol=1e-14)
    peers = [j for j in range(8) if j != target]
    assert grad_v[peers].abs().sum() > 0

    # Pesos de suma cero: un desplazamiento uniforme v_j -> v_j + c se cancela exactamente.
    shift = torch.linspace(-4.0, 6.0, 6, dtype=torch.float64).unsqueeze(0)
    shifted, _ = loo_peer_readout(basis, values.detach() + shift)
    torch.testing.assert_close(shifted, readout.detach(), rtol=1e-13, atol=1e-13)


def test_subspace_orthogonalization_eliminates_scalar_prediction_correlation_even_adversarially():
    """Demuestra el teorema principal que resuelve la debilidad de la ortogonalización por
    coordenadas: aunque h_rel copie columnas permutadas de e_core (donde Gram-Schmidt coordenada
    a coordenada fallaría con correlación ~1), la proyección de subespacio completo anula la
    correlación de la contribución escalar para CUALQUIER vector de pesos de cabeza."""
    rows, hidden = 128, 16
    gen = torch.Generator().manual_seed(41)
    core = torch.randn(rows, hidden, generator=gen, dtype=torch.float64)
    # Caso 1: hemisferio mezclado con copias desplazadas/rotadas de e_core más señal ortogonal.
    perm = torch.roll(torch.arange(hidden), shifts=1)
    signal = torch.randn(rows, hidden, generator=gen, dtype=torch.float64)
    mixed = 2.0 * core[:, perm] + 0.5 * signal

    cfg = DualHemisphereBrainConfig(dim=hidden, ridge=1e-5, mode=ORTHOGONAL)
    dims = dict(prices=6, news=4, charts=4, fundamentals=4, macro=4)
    brain = opened_brain(cfg, hidden=hidden, dims=dims, seed=5, dtype=torch.float64)
    orth = brain._orthogonalize_session(mixed, core)

    # 1. Media transversal exactamente nula.
    torch.testing.assert_close(
        orth.mean(dim=0), torch.zeros(hidden, dtype=torch.float64), rtol=0, atol=1e-13
    )
    # 2. Toda la matriz de covarianza cruzada E_tilde^T H_perp / N es casi nula.
    core_c = core - core.mean(dim=0, keepdim=True)
    cross_cov = (core_c.T @ orth) / rows
    assert cross_cov.abs().max() < 1e-3

    # 3. Para 5 cabezas aleatorias (como QuantileHead), la contribución escalar w^T(g * W_out h^⊥)
    # es ortogonal a la predicción del núcleo w^T e_core.
    delta = brain.gate * brain.out(orth)
    head_weights = torch.randn(5, hidden, generator=gen, dtype=torch.float64)
    for row in range(5):
        w = head_weights[row]
        y_core = core_c @ w
        y_delta = (delta - delta.mean(dim=0, keepdim=True)) @ w
        corr = (y_core * y_delta).sum() / (y_core.norm() * y_delta.norm())
        assert corr.abs() < 1e-3

    # 4. Contraste con la proyección puramente coordenada a coordenada en el caso adversario:
    # si h_rel = core[:, perm] (copia permutada de e_core), la proyección por coordenadas deja
    # intacta casi toda la energía y produce correlación cercana a 1 tras combinar dimensiones,
    # mientras que la proyección de subespacio completo suprime la norma a O(ridge).
    pure_copy = core[:, perm]
    orth_copy = brain._orthogonalize_session(pure_copy, core)
    assert orth_copy.square().mean().sqrt() < 1e-3

    # 5. Invariancia a escalas anisotrópicas en e_core: si una columna de e_core mide 100 y otra
    # mide 0,01, estandarizar las columnas activas de e_core antes de Gram sigue eliminando la
    # componente sobre la columna pequeña sin que tr(G) la ahogue.
    aniso_core = core.clone()
    aniso_core[:, 0] *= 100.0
    aniso_core[:, 1] *= 0.01
    copy_small = aniso_core[:, 1:2].expand(rows, hidden)
    orth_small = brain._orthogonalize_session(copy_small, aniso_core)
    assert orth_small.square().mean().sqrt() < 1e-3


def test_peer_news_diffusion_reaches_asset_without_own_news_in_orthogonal_but_not_self_only():
    """Un activo sin noticias propias recibe señal de las noticias de sus pares co-móviles en
    `_hemisphere_session` (antes del centrado/ortogonalización), pero no en `self_only`."""
    encoded, prices, news, fund, macro, presence, dims = sample_panel(rows=16, hidden=8, seed=83)
    presence = presence.clone()
    presence[0, 1] = False
    news = news.clone()
    news[0] = 0.0
    markets = ("US",) * 16
    instants = (7,) * 16

    orth_brain = opened_brain(
        DualHemisphereBrainConfig(dim=8, mode=ORTHOGONAL), hidden=8, dims=dims, seed=89
    )
    self_brain = opened_brain(
        DualHemisphereBrainConfig(dim=8, mode=SELF_ONLY), hidden=8, dims=dims, seed=89
    )
    news_shock = news.clone()
    news_shock[1:] += 4.0

    # Directamente en h_rel (sin centrado de sesión): solo la difusión u_peer lleva noticias
    # de pares j != 0 hasta el activo 0 sin noticias propias.
    h_base = orth_brain._hemisphere_session(prices, news, fund, macro, presence)
    h_shock = orth_brain._hemisphere_session(prices, news_shock, fund, macro, presence)
    assert not torch.allclose(h_shock[0], h_base[0])

    # En self_only ni h_rel ni la salida final del activo 0 cambian ni un bit.
    h_self_base = self_brain._hemisphere_session(prices, news, fund, macro, presence)
    h_self_shock = self_brain._hemisphere_session(prices, news_shock, fund, macro, presence)
    assert torch.equal(h_self_shock[0], h_self_base[0])
    base_self = self_brain(encoded, prices, news, fund, macro, presence, markets, instants)
    shock_self = self_brain(encoded, prices, news_shock, fund, macro, presence, markets, instants)
    assert torch.equal(shock_self[0], base_self[0])


def test_relational_price_volume_captures_residual_momentum_against_comoving_peers():
    """Dos grupos co-móviles: si los pares del grupo A suben los últimos 5 días mientras el
    activo 0 del grupo A queda plano, su momento de pares es positivo y su residuo es negativo."""
    from mars_titan.models.titans.dual_hemisphere_brain import _relational_price_volume

    gen = torch.Generator().manual_seed(97)
    rows, steps = 12, 64
    factor_a = 0.02 * torch.randn(steps, generator=gen, dtype=torch.float64)
    factor_b = 0.02 * torch.randn(steps, generator=gen, dtype=torch.float64)
    rets = torch.zeros(rows, steps, dtype=torch.float64)
    for i in range(rows):
        base_factor = factor_a if i < 6 else factor_b
        rets[i] = base_factor + 0.002 * torch.randn(steps, generator=gen, dtype=torch.float64)
    # Últimos 5 días: los pares del grupo A (1..5) suben fuerte, el grupo B (6..11) baja,
    # y el activo 0 (históricamente del grupo A) queda rezagado en 0.
    rets[1:6, -5:] = 0.03
    rets[6:, -5:] = -0.02
    rets[0, -5:] = 0.0
    close = rets.cumsum(dim=1)
    close = close - close[:, :1]
    prices = torch.stack(
        [
            close,
            close + 0.005,
            close - 0.005,
            close,
            torch.ones_like(close),
            torch.ones_like(close),
        ],
        dim=-1,
    )
    rel, _ = _relational_price_volume(prices)
    # Columna 1 = peer_ret_cc_5, columna 13 (12 + 1) = resid_ret_cc_5.
    assert rel[0, 1] > 0.5
    assert rel[0, 13] < -0.5


def test_stopgrad_blocks_all_gradients_from_second_hemisphere_into_core_representation():
    """∂(e' - e)/∂e = 0 exactamente: el ortogonalizador nunca distorsiona el núcleo primario."""
    encoded, prices, news, fund, macro, presence, dims = sample_panel(rows=24, hidden=8)
    brain = opened_brain(
        DualHemisphereBrainConfig(dim=8, ridge=1e-4, mode=ORTHOGONAL),
        hidden=8,
        dims=dims,
        seed=7,
        dtype=torch.float64,
    )
    markets = ("US",) * 24
    instants = (10,) * 24
    x = encoded.clone().requires_grad_(True)
    out = brain(x, prices, news, fund, macro, presence, markets, instants)
    delta = out - x
    assert delta.abs().sum() > 0
    (grad_x,) = torch.autograd.grad(delta.square().sum(), (x,), allow_unused=True)
    assert grad_x is None or torch.equal(grad_x, torch.zeros_like(x))


def test_second_hemisphere_is_strictly_blind_to_charts_and_technical_features():
    """El segundo hemisferio no lee ni el bloque `charts` ni la rama técnica."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        cfg = DualHemisphereBrainConfig(dim=8, ridge=1e-4, mode=ORTHOGONAL)
        predictor, batch = build(
            technical_branch=asdict(DENSE),
            dual_hemisphere_brain=asdict(cfg),
        )
        opened(predictor.technical_branch)
        with torch.no_grad():
            predictor.dual_hemisphere_brain.gate.fill_(0.25)
        predictor._seal_parameters()

        # Construimos un lote idéntico perturbando arbitrariamente todo el bloque `charts` y su bit.
        brain = predictor.dual_hemisphere_brain
        markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
        fixed_core = torch.randn(
            len(batch.flow_ids), predictor.config.hidden_size, dtype=torch.float64
        )
        alt_presence = batch.presence.clone()
        alt_presence[:, 2] = ~alt_presence[:, 2]

        out_a = brain(
            fixed_core,
            batch.inputs["prices"],
            batch.inputs["news"],
            batch.inputs["fundamentals"],
            batch.inputs["macro"],
            batch.presence,
            markets,
            batch.prediction_at,
        )
        out_b = brain(
            fixed_core,
            batch.inputs["prices"],
            batch.inputs["news"],
            batch.inputs["fundamentals"],
            batch.inputs["macro"],
            alt_presence,
            markets,
            batch.prediction_at,
        )
        assert torch.equal(out_a, out_b)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_relational_features_handle_absent_days_constant_assets_legacy_5ch_and_short_windows():
    """Ausencias de calendario, acciones suspendidas, ventanas de 5 canales y ventanas cortas."""
    for context, channels in ((64, 6), (64, 5), (16, 6), (4, 6)):
        encoded, prices, news, fund, macro, presence, dims = sample_panel(
            rows=10, hidden=8, context=context, price_channels=channels, seed=13 + context
        )
        if channels == 6 and context >= 16:
            # Simula días ausentes en todo el mercado (relleno +0.0 y bit 0).
            prices[:, 2, :] = 0.0
            prices[:, context - 3, :] = 0.0
        # Simula una acción suspendida (precios constantes y volumen cero).
        prices[0, :, :4] = 0.0
        prices[0, :, 4] = 0.0
        brain = opened_brain(
            DualHemisphereBrainConfig(dim=8, ridge=1e-4, mode=ORTHOGONAL),
            hidden=8,
            dims=dims,
            seed=19,
            dtype=torch.float64,
        )
        out = brain(encoded, prices, news, fund, macro, presence, MARKETS, AT)
        assert out.shape == encoded.shape
        assert torch.isfinite(out).all()


def test_single_two_and_small_n_sessions_produce_finite_outputs_and_gradients():
    """N=1 devuelve la identidad exacta; N=2 y N<=H mantienen Cholesky SPD y gradientes finitos."""
    # N=1 en tres sesiones distintas.
    encoded, prices, news, fund, macro, presence, dims = sample_panel(rows=3, hidden=16, seed=21)
    brain = opened_brain(ORTHO, hidden=16, dims=dims, seed=23, dtype=torch.float64)
    x = encoded.clone().requires_grad_(True)
    out = brain(x, prices, news, fund, macro, presence, ("US", "CN", "US"), (10, 10, 11))
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, param in brain.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name

    # N=2 y N=5 con H=32 (N <= H): Cholesky con ridge > 0 sigue siendo estrictamente SPD.
    for rows in (2, 5):
        enc, pr, nw, fn, mc, pres, dm = sample_panel(rows=rows, hidden=32, seed=31 + rows)
        b = opened_brain(
            DualHemisphereBrainConfig(dim=16, ridge=1e-4, mode=ORTHOGONAL),
            hidden=32,
            dims=dm,
            seed=37,
            dtype=torch.float64,
        )
        out_small = b(enc, pr, nw, fn, mc, pres, ("US",) * rows, (7,) * rows)
        assert torch.isfinite(out_small).all()
        assert not torch.equal(out_small, enc)
        out_small.square().sum().backward()
        for name, param in b.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_strict_causality_isolation_across_markets_and_instants():
    """Sin fugas entre mercados ni desde instantes futuros del mismo bloque."""
    encoded, prices, news, fund, macro, presence, dims = sample_panel(rows=10, hidden=8, seed=43)
    brain = opened_brain(
        DualHemisphereBrainConfig(dim=8, ridge=1e-4, mode=ORTHOGONAL),
        hidden=8,
        dims=dims,
        seed=47,
        dtype=torch.float64,
    )
    base = brain(encoded, prices, news, fund, macro, presence, MARKETS, AT)
    us = [i for i, m in enumerate(MARKETS) if m == "US"]
    cn = [i for i, m in enumerate(MARKETS) if m == "CN"]

    # Perturbar filas de CN (todas sus modalidades y precios) no cambia ni un bit las de US.
    p_cn, n_cn, f_cn, e_cn = prices.clone(), news.clone(), fund.clone(), encoded.clone()
    p_cn[cn, :, :5] += 2.5
    n_cn[cn] += 5.0
    f_cn[cn] -= 4.0
    e_cn[cn] += 3.0
    changed_cn = brain(e_cn, p_cn, n_cn, f_cn, macro, presence, MARKETS, AT)
    assert torch.equal(changed_cn[us], base[us])

    # Perturbar un par del mismo mercado e instante SÍ cambia a sus pares en US.
    p_us = prices.clone()
    p_us[us[0], -1, 3] += 0.5
    changed_us = brain(encoded, p_us, news, fund, macro, presence, MARKETS, AT)
    assert not torch.equal(changed_us[us[1]], base[us[1]])

    # Mover la última fila US a un instante futuro (8 > 7): perturbarla no cambia el presente (7).
    instants = (7, 7, 7, 7, 7, 7, 7, 7, 7, 8)
    split = brain(encoded, prices, news, fund, macro, presence, MARKETS, instants)
    p_fut, n_fut, e_fut = prices.clone(), news.clone(), encoded.clone()
    p_fut[9, :, :5] += 10.0
    n_fut[9] += 50.0
    e_fut[9] += 20.0
    after_fut = brain(e_fut, p_fut, n_fut, fund, macro, presence, MARKETS, instants)
    assert torch.equal(after_fut[us[:-1]], split[us[:-1]])


@pytest.mark.parametrize("mode", [ORTHOGONAL, UNORTHOGONALIZED, SELF_ONLY])
def test_permuting_rows_permutes_outputs_equivariantly(mode):
    encoded, prices, news, fund, macro, presence, dims = sample_panel(rows=10, hidden=8, seed=53)
    brain = opened_brain(
        DualHemisphereBrainConfig(dim=8, ridge=1e-4, mode=mode),
        hidden=8,
        dims=dims,
        seed=59,
        dtype=torch.float64,
    )
    order = [7, 0, 9, 2, 4, 1, 8, 5, 3, 6]
    expected = brain(encoded, prices, news, fund, macro, presence, MARKETS, AT)[order]
    actual = brain(
        encoded[order],
        prices[order],
        news[order],
        fund[order],
        macro[order],
        presence[order],
        tuple(MARKETS[i] for i in order),
        AT,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_controls_share_exact_parameters_and_have_distinct_scientific_semantics():
    dims = dict(prices=6, news=8, charts=4, fundamentals=6, macro=6)
    main = DualHemisphereBrain(8, dims, DualHemisphereBrainConfig(dim=8, mode=ORTHOGONAL), seed=11)
    others = [
        DualHemisphereBrain(8, dims, DualHemisphereBrainConfig(dim=8, mode=mode), seed=11)
        for mode in (UNORTHOGONALIZED, SELF_ONLY, SHUFFLED_PEERS)
    ]
    for other in others:
        assert set(main.state_dict()) == set(other.state_dict())
        for name, tensor in main.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, other.state_dict()[name]), name

    # Semántica: self_only es estrictamente local por fila.
    encoded, prices, news, fund, macro, presence, dims = sample_panel(rows=32, hidden=8, seed=61)
    us_all = ("US",) * 32
    at_all = (7,) * 32
    open_self = opened_brain(
        DualHemisphereBrainConfig(dim=8, mode=SELF_ONLY), hidden=8, dims=dims, seed=11
    )
    base_self = open_self(encoded, prices, news, fund, macro, presence, us_all, at_all)
    p_other, n_other, e_other = prices.clone(), news.clone(), encoded.clone()
    p_other[1:, :, :5] += 3.0
    n_other[1:] += 8.0
    e_other[1:] += 5.0
    assert torch.equal(
        open_self(e_other, p_other, n_other, fund, macro, presence, us_all, at_all)[0],
        base_self[0],
    )

    # Semántica: unorthogonalized difiere de orthogonal; shuffled_peers es determinista,
    # difiere de orthogonal y sigue ortogonalizado contra e_core.
    open_orth = opened_brain(
        DualHemisphereBrainConfig(dim=8, ridge=1e-5, mode=ORTHOGONAL),
        hidden=8,
        dims=dims,
        seed=11,
    )
    open_unorth = opened_brain(
        DualHemisphereBrainConfig(dim=8, ridge=1e-5, mode=UNORTHOGONALIZED),
        hidden=8,
        dims=dims,
        seed=11,
    )
    open_shuf = opened_brain(
        DualHemisphereBrainConfig(dim=8, ridge=1e-5, mode=SHUFFLED_PEERS),
        hidden=8,
        dims=dims,
        seed=11,
    )
    out_orth = open_orth(encoded, prices, news, fund, macro, presence, us_all, at_all)
    out_unorth = open_unorth(encoded, prices, news, fund, macro, presence, us_all, at_all)
    out_shuf_1 = open_shuf(encoded, prices, news, fund, macro, presence, us_all, at_all)
    out_shuf_2 = open_shuf(encoded, prices, news, fund, macro, presence, us_all, at_all)
    assert not torch.allclose(out_orth, out_unorth)
    assert torch.equal(out_shuf_1, out_shuf_2)
    assert not torch.allclose(out_shuf_1, out_orth)

    # Tanto orthogonal como shuffled_peers mantienen correlación residual casi nula con e_core.
    core_c = encoded - encoded.mean(dim=0, keepdim=True)
    for out in (out_orth, out_shuf_1):
        delta = out - encoded
        cross = (core_c.T @ delta) / len(encoded)
        assert cross.abs().max() < 1e-2


def test_macro_film_modulation_survives_session_centering_and_presence_masks_absent_modalities():
    """Macro es constante en la sesión pero sobrevive al centrado mediante modulación FiLM, y las
    modalidades ausentes (news/fundamentals/macro) se anulan respetando `presence`."""
    encoded, prices, news, fund, macro, presence, dims = sample_panel(rows=16, hidden=8, seed=67)
    markets = ("US",) * 16
    instants = (7,) * 16
    brain = opened_brain(
        DualHemisphereBrainConfig(dim=8, mode=ORTHOGONAL), hidden=8, dims=dims, seed=71
    )
    base = brain(encoded, prices, news, fund, macro, presence, markets, instants)
    # Cambiar el vector macro constante de toda la sesión SÍ cambia la salida ortogonalizada.
    alt_macro = macro + 2.5
    with_macro = brain(encoded, prices, news, fund, alt_macro, presence, markets, instants)
    assert not torch.allclose(with_macro, base)

    # Si macro está ausente (bit 4 falso), cambiar los valores de macro no cambia ni un bit.
    no_macro_pres = presence.clone()
    no_macro_pres[:, 4] = False
    ref_no_macro = brain(encoded, prices, news, fund, macro, no_macro_pres, markets, instants)
    alt_no_macro = brain(encoded, prices, news, fund, alt_macro, no_macro_pres, markets, instants)
    assert torch.equal(ref_no_macro, alt_no_macro)

    # Si news y fundamentals faltan en la fila 0, perturbar sus valores en la fila 0 no afecta nada.
    masked_pres = presence.clone()
    masked_pres[0, 1] = False
    masked_pres[0, 3] = False
    ref_m = brain(encoded, prices, news, fund, macro, masked_pres, markets, instants)
    n2, f2 = news.clone(), fund.clone()
    n2[0] += 100.0
    f2[0] -= 100.0
    alt_m = brain(encoded, prices, n2, f2, macro, masked_pres, markets, instants)
    assert torch.equal(ref_m, alt_m)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deterministic_outputs_and_gradients_in_fp32_and_fp64(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        enc_a, pr, nw, fn, mc, pres, dims = sample_panel(rows=10, hidden=8, seed=73, dtype=dtype)
        enc_b = enc_a.clone()
        enc_a.requires_grad_(True)
        enc_b.requires_grad_(True)
        cfg = DualHemisphereBrainConfig(dim=8, ridge=1e-4, mode=ORTHOGONAL)
        layer_a = opened_brain(cfg, hidden=8, dims=dims, seed=79, dtype=dtype)
        layer_b = opened_brain(cfg, hidden=8, dims=dims, seed=79, dtype=dtype)
        out_a = layer_a(enc_a, pr, nw, fn, mc, pres, MARKETS, AT)
        out_b = layer_b(enc_b, pr, nw, fn, mc, pres, MARKETS, AT)
        assert torch.equal(out_a, out_b)
        out_a.square().sum().backward()
        out_b.square().sum().backward()
        assert torch.equal(enc_a.grad, enc_b.grad)
        for (na, pa), (_, pb) in zip(
            layer_a.named_parameters(), layer_b.named_parameters(), strict=True
        ):
            assert torch.isfinite(pa.grad).all() and torch.equal(pa.grad, pb.grad), na
    finally:
        torch.use_deterministic_algorithms(previous)


def test_disabled_default_preserves_identity_fingerprint_parameters_and_predictions_bitwise():
    base, batch = setup()
    explicit = model(dual_hemisphere_brain=None)
    assert explicit.config.identity() == base.config.identity()
    assert explicit._config_id() == base._config_id()
    assert explicit._parameter_id == base._parameter_id
    assert "dual_hemisphere_brain" not in base.config.identity()
    assert base.dual_hemisphere_brain is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)


@pytest.mark.parametrize("config", [ORTHO, UNORTHO, SELF_CFG, SHUF_CFG])
def test_predictor_with_dual_hemisphere_brain_starts_bitwise_identical_to_control(config):
    base, _ = setup()
    batch = session_batch(6, markets=["US", "CN", "US", "US", "CN", "US"])
    predictor = model(dual_hemisphere_brain=config)
    ref_params = dict(base.named_parameters())
    all_params = dict(predictor.named_parameters())
    expected_new = {
        "dual_hemisphere_brain.rel_proj.weight",
        "dual_hemisphere_brain.rel_proj.bias",
        "dual_hemisphere_brain.news_proj.weight",
        "dual_hemisphere_brain.news_proj.bias",
        "dual_hemisphere_brain.fundamentals_proj.weight",
        "dual_hemisphere_brain.fundamentals_proj.bias",
        "dual_hemisphere_brain.macro_proj.weight",
        "dual_hemisphere_brain.macro_proj.bias",
        "dual_hemisphere_brain.hemisphere_norm.weight",
        "dual_hemisphere_brain.hemisphere_norm.bias",
        "dual_hemisphere_brain.fc1.weight",
        "dual_hemisphere_brain.fc1.bias",
        "dual_hemisphere_brain.fc2.weight",
        "dual_hemisphere_brain.fc2.bias",
        "dual_hemisphere_brain.out.weight",
        "dual_hemisphere_brain.gate",
    }
    assert set(all_params) - set(ref_params) == expected_new
    for name, value in ref_params.items():
        assert torch.equal(all_params[name], value), name
    assert torch.count_nonzero(predictor.dual_hemisphere_brain.gate) == 0
    assert torch.count_nonzero(predictor.dual_hemisphere_brain.out.weight) > 0
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)


def test_step_zero_and_step_one_gradients_alive_and_finite_with_zero_median_head():
    """Con mediana inicial en cero (predicción constante) y pinball + IC, el gradiente llega a
    `gate` en el paso 0 (sin doble cero en serie) y a TODOS los parámetros en el paso 1."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        predictor, _ = build(
            technical_branch=asdict(DENSE),
            dual_hemisphere_brain=asdict(ORTHO),
        )
        opened(predictor.technical_branch)
        predictor._seal_parameters()

        # Lote completo de 6 filas (todas las modalidades presentes) para activar todas las ramas.
        batch = session_batch(6, markets=["US", "US", "US", "US", "US", "US"])
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
        assert predictor.dual_hemisphere_brain.gate.grad is not None
        assert predictor.dual_hemisphere_brain.gate.grad.abs().sum() > 0

        # Paso 1 tras abrir la puerta: todos los parámetros del segundo hemisferio aprenden.
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
        for name, param in predictor.dual_hemisphere_brain.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, f"Gradiente nulo tras abrir la puerta: {name}"
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_gradients_are_finite_when_scalar_head_starts_identically_at_zero():
    """Con toda la cabeza escalar en cero (predicción constante 0), no aparecen NaN ni Inf."""
    batch = session_batch(6, markets=["US", "CN", "US", "US", "CN", "US"])
    predictor = model(dual_hemisphere_brain=ORTHO)
    with torch.no_grad():
        predictor.head.weight.zero_()
        predictor.head.bias.zero_()
        predictor.dual_hemisphere_brain.gate.fill_(0.25)
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


def test_coexistence_with_session_hologram_technical_branch_and_crrm():
    """Convivencia paralela con session_hologram, technical_branch y CRRM en el orden definido."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        hsi_cfg = SessionHologramConfig(bands=4, phase_dim=8, position="after_technical")
        crrm_cfg = CrossSectionalRankMemoryConfig(dim=8, time_constants=(5,), minimum_rows=3)
        base, _ = build(
            technical_branch=asdict(DENSE),
            session_hologram=asdict(hsi_cfg),
            cross_sectional_rank_memory=asdict(crrm_cfg),
        )
        combo, _ = build(
            technical_branch=asdict(DENSE),
            session_hologram=asdict(hsi_cfg),
            cross_sectional_rank_memory=asdict(crrm_cfg),
            dual_hemisphere_brain=asdict(ORTHO),
        )
        opened(base.technical_branch)
        opened(combo.technical_branch)
        with torch.no_grad():
            base.session_hologram.gate.fill_(0.2)
            combo.session_hologram.gate.fill_(0.2)
        base._seal_parameters()
        combo._seal_parameters()

        batch = session_batch(6, markets=["US", "US", "US", "US", "US", "US"])
        # Con puerta CODHB a cero, el combo emite exactamente igual que base con HSI + CRRM.
        ref = base.prepare(
            batch,
            base.initial_state(batch.flow_ids),
            session_complete=True,
            rank_memory=base.initial_rank_memory(),
        )
        act0 = combo.prepare(
            batch,
            combo.initial_state(batch.flow_ids),
            session_complete=True,
            rank_memory=combo.initial_rank_memory(),
        )
        torch.testing.assert_close(act0.point_predictions, ref.point_predictions, rtol=0, atol=0)

        # Al abrir la puerta de CODHB, se aplica DESPUÉS de session_hologram y ANTES de CRRM/head,
        # sin tocar el estado rápido de MAC.
        with torch.no_grad():
            combo.dual_hemisphere_brain.gate.fill_(0.2)
        combo._seal_parameters()
        act1 = combo.prepare(
            batch,
            combo.initial_state(batch.flow_ids),
            session_complete=True,
            rank_memory=combo.initial_rank_memory(),
        )
        assert not torch.allclose(act1.working_state, ref.working_state)
        torch.testing.assert_close(
            act1.next_state.mac.memory.weights[0],
            ref.next_state.mac.memory.weights[0],
            rtol=0,
            atol=0,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_session_complete_required_and_exclusions_enforced():
    batch = session_batch()
    predictor = model(dual_hemisphere_brain=ORTHO)
    assert predictor.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)

    base, _ = setup()
    with pytest.raises(ValueError, match="cabezas auxiliares"):
        replace(
            base.config,
            auxiliary_horizons=AuxiliaryHorizons(horizons=(5,), weight=0.1, maturity_slack=1),
            dual_hemisphere_brain=ORTHO,
        )

    direct, _ = setup("transformer_direct")
    direct_codhb = type(direct)(
        replace(direct.config, dual_hemisphere_brain=ORTHO), dtype=torch.float64
    )
    with pytest.raises(ValueError, match="Dust"):
        FinancialDustAdapter(direct_codhb)


def test_state_dict_round_trips_and_rejects_different_mode_or_ridge():
    batch = session_batch()
    source = model(dual_hemisphere_brain=ORTHO)
    with torch.no_grad():
        source.dual_hemisphere_brain.gate.fill_(0.15)
    source._seal_parameters()
    target = model(dual_hemisphere_brain=ORTHO)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    for other_cfg in (
        UNORTHO,
        SELF_CFG,
        SHUF_CFG,
        DualHemisphereBrainConfig(dim=16, ridge=1e-3, mode=ORTHOGONAL),
    ):
        with pytest.raises(ValueError):
            model(dual_hemisphere_brain=other_cfg).load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    "bad",
    [
        dict(dim=3, mode=ORTHOGONAL),
        dict(dim=257, mode=ORTHOGONAL),
        dict(dim=True, mode=ORTHOGONAL),
        dict(dim=16, ridge=0.0, mode=ORTHOGONAL),
        dict(dim=16, ridge=2.0, mode=ORTHOGONAL),
        dict(dim=16, ridge=float("nan"), mode=ORTHOGONAL),
        dict(dim=16, ridge=True, mode=ORTHOGONAL),
        dict(dim=16, mode="desconocido"),
        dict(dim=16, mode=ORTHOGONAL, extra=1),
        dict(dim=16),
        DUAL_HEMISPHERE_BRAIN,
    ],
)
def test_invalid_declarations_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_dual_hemisphere_brain(bad)


def test_construction_forward_validation_and_autocast_rejection():
    with pytest.raises(ValueError, match="DualHemisphereBrainConfig"):
        DualHemisphereBrain(16, DIMENSIONS, asdict(ORTHO), seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        DualHemisphereBrain(16, DIMENSIONS, ORTHO, seed=1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="modalidades"):
        DualHemisphereBrain(16, dict(prices=5, news=2), ORTHO, seed=1)

    encoded, prices, news, fund, macro, presence, dims = sample_panel(
        rows=6, hidden=16, dtype=torch.float32
    )
    brain = DualHemisphereBrain(16, dims, ORTHO, seed=1, dtype=torch.float32)
    mkts = ("US", "CN", "US", "US", "CN", "US")
    at = (7,) * 6
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        brain(encoded, prices, news, fund, macro, presence, mkts, at)
    with pytest.raises(ValueError):
        brain(encoded[:, :8], prices, news, fund, macro, presence, mkts, at)
    with pytest.raises(ValueError):
        brain(encoded, prices, news, fund, macro, presence, mkts[:-1], at)
    with pytest.raises(ValueError, match="finit"):
        bad = encoded.clone()
        bad[0, 0] = math.nan
        brain(bad, prices, news, fund, macro, presence, mkts, at)


def test_registry_and_campaign_v12_screen3_accept_single_variable_candidate_and_placebos():
    cand = dict(dim=32, ridge=1e-4, mode=ORTHOGONAL)
    changes = dict(dual_hemisphere_brain=cand)
    assert checked_predictor_changes(changes) == changes

    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    for suffix, mode in (
        ("codhb", ORTHOGONAL),
        ("codhb_unorth", UNORTHOGONALIZED),
        ("codhb_self", SELF_ONLY),
        ("codhb_shuffled", SHUFFLED_PEERS),
    ):
        doc["models"][f"titans_mac_{suffix}"] = dict(
            doc["models"]["titans_mac"],
            tier="s3_inventions",
            predictor_changes=dict(dual_hemisphere_brain=dict(dim=32, ridge=1e-4, mode=mode)),
        )
    validated = comparison_campaign.checked_declaration(doc, root=root)
    for name in (
        "titans_mac_codhb",
        "titans_mac_codhb_unorth",
        "titans_mac_codhb_self",
        "titans_mac_codhb_shuffled",
    ):
        assert name in validated["models"]
        assert set(validated["models"][name]["predictor_changes"]) == {"dual_hemisphere_brain"}


def test_frozen_consumer_and_trainer_code_register_source_only_when_active():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        plain_engine, _ = frozen("mac_online")
        assert CODHB_MODULE not in plain_engine.identity()["implementation"]
        assert api.declared_sources(plain_engine.predictor.config) == ()

        base, _ = setup("mac_online")
        pred = type(base)(replace(base.config, dual_hemisphere_brain=ORTHO), dtype=torch.float64)
        with torch.no_grad():
            pred.dual_hemisphere_brain.gate.fill_(0.2)
        pred._seal_parameters()
        pred.eval().requires_grad_(False)
        engine = consumer(pred)

        assert api.declared_sources(pred.config) == (CODHB_MODULE,)
        assert CODHB_MODULE in engine.identity()["implementation"]
        assert engine.model_id != plain_engine.model_id
        assert (
            engine.identity()["modules"]["predictor/dual_hemisphere_brain"]["settings"][
                "dual_hemisphere_brain"
            ]
            == ORTHO.identity()
        )
        engine.verify()

        batch = session_batch(6, markets=["US", "CN", "US", "US", "CN", "US"])
        state = pred.initial_state(batch.flow_ids)
        expected = pred.prepare(batch, state, session_complete=True)
        actual = engine.prepare(batch, state, context_id=CONTEXT, session_complete=True)
        torch.testing.assert_close(
            actual.point_predictions, expected.point_predictions, rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="sesión completa"):
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
        pred.dual_hemisphere_brain.news_dim = 999
        with pytest.raises(ValueError):
            engine.verify(strong=False)

        # Trainer._code() solo incluye el módulo cuando dual_hemisphere_brain está activo.
        trainer = object.__new__(ChronologicalTrainer)
        trainer.recipe = ChronologicalRecipe()
        for cfg, present in ((None, False), (ORTHO, True)):
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
                    session_hologram=None,
                    dual_hemisphere_brain=cfg,
                )
            )
            assert (CODHB_MODULE in trainer._code()) is present
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
