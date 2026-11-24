"""Campo holográfico persistente transversal y temporal multiescala (MPSH).

Pruebas de contrato, equivalencia exacta frente a la referencia O(T·N²), invariancia frente a
desplazamientos uniformes distintos por sesión, cero eco propio, causalidad estricta en el ciclo
preparación-confirmación entre sesiones, aislamiento entre mercados, controles negativos con los
mismos parámetros, tamaño constante del grafo de autograd, determinismo en FP32/FP64, gradientes
vivos desde el paso 0 con predicción constante, convivencia con HSI y CRRM, y guardado/reanudación.
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
from test_financial_learning_scale import build, emit, statistics
from test_session_rank_objective import model, session_batch
from test_technical_dense import DENSE, opened

from mars_titan.models.learning_scale import LearningScale
from mars_titan.models.quantile_head import MEDIAN_GAIN, QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer, declared_sources
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.persistent_session_hologram import (
    DENSITY_BOUND,
    INSTANT_ONLY,
    PERSISTENT,
    PERSISTENT_SESSION_HOLOGRAM,
    SELF_ONLY,
    SHUFFLED_HISTORY,
    PersistentSessionHologram,
    PersistentSessionHologramConfig,
    parse_persistent_session_hologram,
)
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_ic_objective import SessionICTerm
from mars_titan.models.titans.technical_branch import TechnicalDenseConfig
from mars_titan.training import comparison_campaign
from mars_titan.training.dust.financial import FinancialDustAdapter
from mars_titan.training.search_cases import checked_predictor_changes

MPSH_MODULE = "mars_titan.models.titans.persistent_session_hologram"
START, DAY = 1_609_459_200_000_000, 86_400_000_000

EARLY = PersistentSessionHologramConfig(
    bands=4, phase_dim=8, half_lives=(1, 5, 10), mode=PERSISTENT, position="after_mac", seed=0
)
LATE = PersistentSessionHologramConfig(
    bands=4,
    phase_dim=16,
    half_lives=(1, 5, 10),
    mode=PERSISTENT,
    position="after_technical",
    seed=0,
)
INSTANT = replace(EARLY, mode=INSTANT_ONLY)
SHUFFLED = replace(EARLY, mode=SHUFFLED_HISTORY)
SELF = replace(EARLY, mode=SELF_ONLY)
MARKETS = ("US", "CN", "US", "US", "CN", "US")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def opened_layer(config=EARLY, *, hidden=16, seed=3, dtype=torch.float64, nonzero_density=True):
    """Módulo MPSH con puerta y pesos de escala no triviales para observar lectura y rotación."""
    layer = PersistentSessionHologram(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed + 101)
        layer.gate.copy_(torch.randn(hidden, generator=gen, dtype=dtype) / 4)
        layer.scale_mem.copy_(torch.randn(layer.scale_mem.shape, generator=gen, dtype=dtype) / 3)
        layer.scale_rot.copy_(torch.randn(layer.scale_rot.shape, generator=gen, dtype=dtype) / 3)
        if nonzero_density:
            layer.density.weight.copy_(
                torch.randn(layer.density.weight.shape, generator=gen, dtype=dtype) / 2
            )
    return layer


def sample_encoded(rows=6, hidden=16, seed=9, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=gen, dtype=dtype)


def naive_reference_sequence(layer, sessions):
    """Referencia explícita O(T·N²) par a par y sesión a sesión con las ecuaciones de MPSH."""
    cfg = layer.config
    bands, dc, dv = cfg.bands, cfg.phase_dim, layer.band_dim
    decays = [2.0 ** (-1.0 / tau) for tau in cfg.half_lives]
    scales = len(decays)
    # Historial confirmado por mercado: lista de (pi, theta, v) de sesiones estrictamente previas.
    history = {}
    outputs = []

    for values, markets, instants in sessions:
        normed = layer.norm(values)
        projected = layer.proj(normed)
        pw = bands * dc
        theta = projected[:, :pw].reshape(len(values), bands, dc)
        phi = projected[:, pw : 2 * pw].reshape(len(values), bands, dc)
        v = projected[:, 2 * pw :].reshape(len(values), bands, dv)
        g = DENSITY_BOUND * torch.tanh(layer.density(normed))

        # Agrupar la sesión actual por (mercado, instante) para calcular densidades.
        groups = {}
        for idx, key in enumerate(zip(markets, instants, strict=True)):
            groups.setdefault(key, []).append(idx)
        pi_all = {}
        for _key, members in groups.items():
            g_group = torch.stack([g[j] for j in members], dim=0)
            pi_group = torch.softmax(g_group, dim=0)
            for pos, j in enumerate(members):
                pi_all[j] = pi_group[pos]

        row_residuals = []
        for i in range(len(values)):
            if cfg.mode == SELF_ONLY:
                diff = theta[i] - phi[i]
                k_re = torch.cos(diff).mean(dim=-1)
                k_im = torch.sin(diff).mean(dim=-1)
                z_inst = torch.cat(
                    (
                        (k_re.unsqueeze(-1) * v[i]).reshape(-1),
                        (k_im.unsqueeze(-1) * v[i]).reshape(-1),
                        k_re,
                        k_im,
                    )
                )
                u_inst = layer.out_inst(layer.inst_norm(z_inst))
                row_residuals.append(u_inst)
                continue

            market = markets[i]
            key = (market, instants[i])
            members = groups[key]
            if len(members) == 1:
                z_inst = values.new_zeros(layer.readout_dim)
            else:
                peers = [j for j in members if j != i]
                pi_i = pi_all[i]
                d_re_b, d_im_b, rho_re_b, rho_im_b = [], [], [], []
                for m in range(bands):
                    w_peers = torch.stack([pi_all[j][m] / (1.0 - pi_i[m]) for j in peers])
                    k_re = torch.stack([torch.cos(theta[j, m] - phi[i, m]).mean() for j in peers])
                    k_im = torch.stack([torch.sin(theta[j, m] - phi[i, m]).mean() for j in peers])
                    rho_re = (w_peers * k_re).sum()
                    rho_im = (w_peers * k_im).sum()
                    v_peers = torch.stack([v[j, m] for j in peers], dim=0)
                    d_re_b.append(((w_peers * (k_re - rho_re)).unsqueeze(-1) * v_peers).sum(dim=0))
                    d_im_b.append(((w_peers * (k_im - rho_im)).unsqueeze(-1) * v_peers).sum(dim=0))
                    rho_re_b.append(rho_re)
                    rho_im_b.append(rho_im)
                z_inst = torch.cat(
                    (
                        torch.cat(d_re_b),
                        torch.cat(d_im_b),
                        torch.stack(rho_re_b),
                        torch.stack(rho_im_b),
                    )
                )

            u_inst = layer.out_inst(layer.inst_norm(z_inst))
            past = history.get(market, [])
            if cfg.mode == INSTANT_ONLY or not past:
                row_residuals.append(u_inst)
                continue

            # Lectura histórica multiescala exacta: combinación convexa EMA insesgada de las
            # covarianzas de suma cero de cada sesión pasada s < t.
            z_mem_scales = []
            for ell in range(scales):
                lam = decays[ell]
                raw_weights = [
                    (1.0 - lam) * (lam ** (len(past) - 1 - pos)) for pos in range(len(past))
                ]
                total_w = sum(raw_weights)
                alpha = [w / total_w for w in raw_weights]
                d_re_b, d_im_b, rho_re_b, rho_im_b = [], [], [], []
                for m in range(bands):
                    acc_re = values.new_zeros(dv)
                    acc_im = values.new_zeros(dv)
                    acc_rho_re = values.new_zeros(())
                    acc_rho_im = values.new_zeros(())
                    for a_s, (pi_s, theta_s, v_s) in zip(alpha, past, strict=True):
                        k_re = torch.stack(
                            [torch.cos(theta_s[j, m] - phi[i, m]).mean() for j in range(len(v_s))]
                        )
                        k_im = torch.stack(
                            [torch.sin(theta_s[j, m] - phi[i, m]).mean() for j in range(len(v_s))]
                        )
                        w_s = pi_s[:, m]
                        rho_re_s = (w_s * k_re).sum()
                        rho_im_s = (w_s * k_im).sum()
                        cov_re_s = ((w_s * (k_re - rho_re_s)).unsqueeze(-1) * v_s[:, m]).sum(dim=0)
                        cov_im_s = ((w_s * (k_im - rho_im_s)).unsqueeze(-1) * v_s[:, m]).sum(dim=0)
                        acc_re = acc_re + a_s * cov_re_s
                        acc_im = acc_im + a_s * cov_im_s
                        acc_rho_re = acc_rho_re + a_s * rho_re_s
                        acc_rho_im = acc_rho_im + a_s * rho_im_s
                    d_re_b.append(acc_re)
                    d_im_b.append(acc_im)
                    rho_re_b.append(acc_rho_re)
                    rho_im_b.append(acc_rho_im)
                z_mem_scales.append(
                    torch.cat(
                        (
                            torch.cat(d_re_b),
                            torch.cat(d_im_b),
                            torch.stack(rho_re_b),
                            torch.stack(rho_im_b),
                        )
                    )
                )
            z_mem = torch.stack(z_mem_scales, dim=0)
            z_rot = z_inst.unsqueeze(0) - z_mem
            u_mem = layer.out_mem(layer.mem_norm(z_mem))
            u_rot = layer.out_rot(layer.rot_norm(z_rot))
            y_mem = (u_mem * layer.scale_mem).sum(dim=0)
            y_rot = (u_rot * layer.scale_rot).sum(dim=0)
            row_residuals.append(u_inst + y_mem + y_rot)

        outputs.append(values + layer.gate * torch.stack(row_residuals, dim=0))

        # Al cerrar la sesión, cada mercado confirma su estado para sesiones futuras.
        for (market, _), members in groups.items():
            pi_s = torch.stack([pi_all[j] for j in members], dim=0)
            theta_s = torch.stack([theta[j] for j in members], dim=0)
            v_s = torch.stack([v[j] for j in members], dim=0)
            history.setdefault(market, []).append((pi_s, theta_s, v_s))

    return outputs


@pytest.mark.parametrize(
    "config",
    [
        EARLY,
        INSTANT,
        SELF,
        PersistentSessionHologramConfig(
            bands=2, phase_dim=6, half_lives=(1, 4), mode=PERSISTENT, position="after_mac"
        ),
        PersistentSessionHologramConfig(
            bands=1, phase_dim=4, half_lives=(2,), mode=PERSISTENT, position="after_mac"
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multi_session_persistent_hologram_matches_explicit_quadratic_reference(config, dtype):
    layer = opened_layer(config, hidden=16, seed=5, dtype=dtype)
    sessions = [
        (sample_encoded(6, 16, seed=11, dtype=dtype), MARKETS, (START,) * 6),
        (sample_encoded(6, 16, seed=17, dtype=dtype), MARKETS, (START + DAY,) * 6),
        # Sesión 3: US tiene 3 activos y CN solo 1 activo (N=1 con historial previo de CN).
        (
            sample_encoded(4, 16, seed=23, dtype=dtype),
            ("US", "US", "CN", "US"),
            (START + 2 * DAY,) * 4,
        ),
        (sample_encoded(6, 16, seed=29, dtype=dtype), MARKETS, (START + 3 * DAY,) * 6),
    ]
    expected = naive_reference_sequence(layer, sessions)
    state = layer.initial_state("p")
    tol = 2e-5 if dtype == torch.float32 else 1e-12
    for step, (values, markets, instants) in enumerate(sessions):
        actual, staged = layer(values, markets, instants, state)
        torch.testing.assert_close(actual, expected[step], rtol=tol, atol=tol)
        state = layer.commit(staged, differentiable=False)


def test_zero_sum_covariance_cancels_different_uniform_shifts_on_every_session():
    """Desplazar v_{j,s} -> v_{j,s} + c_s con distinto c_s cada día no cambia ninguna salida."""
    layer = opened_layer(EARLY, hidden=16, seed=19, dtype=torch.float64)
    bands, dc, dv = layer.config.bands, layer.config.phase_dim, layer.band_dim
    pw = bands * dc

    def run_with_shifts(shifts):
        state = layer.initial_state("p")
        outs = []
        for idx, shift in enumerate(shifts):
            x = sample_encoded(6, 16, seed=31 + idx, dtype=torch.float64)
            normed = layer.norm(x)
            proj = layer.proj(normed)
            theta = proj[:, :pw].reshape(6, bands, dc)
            phi = proj[:, pw : 2 * pw].reshape(6, bands, dc)
            v = proj[:, 2 * pw :].reshape(6, bands, dv) + shift
            g = DENSITY_BOUND * torch.tanh(layer.density(normed))
            c_t, s_t = torch.cos(theta), torch.sin(theta)
            c_p, s_p = torch.cos(phi), torch.sin(phi)
            inv_dc = 1.0 / dc
            self_re = (c_p * c_t + s_p * s_t).sum(dim=-1) * inv_dc
            self_im = (c_p * s_t - s_p * c_t).sum(dim=-1) * inv_dc
            res, pending = layer._session_step(
                "US",
                START + idx * DAY,
                c_t,
                s_t,
                c_p,
                s_p,
                v,
                g,
                self_re,
                self_im,
                state.markets.get("US"),
            )
            outs.append(res)
            staged = replace(state, pending={"US": pending})
            state = layer.commit(staged, differentiable=False)
        return outs

    zero_shifts = [torch.zeros(1, 1, dv, dtype=torch.float64) for _ in range(3)]
    varied_shifts = [
        torch.linspace(-4.0, 6.0, dv, dtype=torch.float64).view(1, 1, dv),
        torch.linspace(9.0, -2.0, dv, dtype=torch.float64).view(1, 1, dv),
        torch.linspace(-1.5, 11.0, dv, dtype=torch.float64).view(1, 1, dv),
    ]
    base = run_with_shifts(zero_shifts)
    shifted = run_with_shifts(varied_shifts)
    for left, right in zip(base, shifted, strict=True):
        torch.testing.assert_close(right, left, rtol=1e-12, atol=1e-12)


def test_exact_zero_self_echo_on_instantaneous_historical_and_rotation_streams():
    """La salida del activo i en la sesión t no depende de su propio v_{i,t} ni de su θ_{i,t}."""
    layer = opened_layer(EARLY, hidden=16, seed=23, dtype=torch.float64)
    bands, dc, dv = layer.config.bands, layer.config.phase_dim, layer.band_dim
    pw = bands * dc

    # Confirmar primero una sesión pasada con grafo para comprobar también gradiente temporal.
    x0 = sample_encoded(5, 16, seed=27, dtype=torch.float64).requires_grad_(True)
    _, staged0 = layer(x0, ("US",) * 5, (START,) * 5, layer.initial_state("p"))
    state1 = layer.commit(staged0, differentiable=True)

    x1 = sample_encoded(5, 16, seed=29, dtype=torch.float64)
    normed = layer.norm(x1).detach()
    proj = layer.proj(normed).detach()
    theta = proj[:, :pw].reshape(5, bands, dc).clone().requires_grad_(True)
    phi = proj[:, pw : 2 * pw].reshape(5, bands, dc).clone().requires_grad_(True)
    v = proj[:, 2 * pw :].reshape(5, bands, dv).clone().requires_grad_(True)
    g = (DENSITY_BOUND * torch.tanh(layer.density(normed))).detach()
    c_t, s_t = torch.cos(theta), torch.sin(theta)
    c_p, s_p = torch.cos(phi), torch.sin(phi)
    inv_dc = 1.0 / dc
    self_re = (c_p * c_t + s_p * s_t).sum(dim=-1) * inv_dc
    self_im = (c_p * s_t - s_p * c_t).sum(dim=-1) * inv_dc

    residual, _ = layer._session_step(
        "US", START + DAY, c_t, s_t, c_p, s_p, v, g, self_re, self_im, state1.markets["US"]
    )
    target_row = 2
    g_v, g_theta, g_phi, g_x0 = torch.autograd.grad(residual[target_row].sum(), (v, theta, phi, x0))
    torch.testing.assert_close(
        g_v[target_row], torch.zeros_like(g_v[target_row]), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        g_theta[target_row], torch.zeros_like(g_theta[target_row]), rtol=0, atol=1e-14
    )
    # Sí depende de su fase de consulta φ_i, de sus pares actuales y de la sesión previa confirmada.
    assert g_phi[target_row].abs().sum() > 0
    peers = [j for j in range(5) if j != target_row]
    assert g_v[peers].abs().sum() > 0 and g_theta[peers].abs().sum() > 0
    assert g_x0.abs().sum() > 0


def test_strict_cross_session_causality_write_on_commit_read_only_in_later_sessions():
    """El estado solo cambia al confirmar al cierre y solo se lee en sesiones posteriores."""
    layer = opened_layer(EARLY, hidden=16, seed=41, dtype=torch.float64)
    instant = opened_layer(INSTANT, hidden=16, seed=41, dtype=torch.float64)
    x0 = sample_encoded(6, 16, seed=43, dtype=torch.float64)
    x1 = sample_encoded(6, 16, seed=47, dtype=torch.float64)

    st0 = layer.initial_state("p")
    out0, staged0 = layer(x0, MARKETS, (START,) * 6, st0)
    # Antes de confirmar la primera sesión no hay historial confirmado: coincide con instant_only.
    out0_inst, _ = instant(x0, MARKETS, (START,) * 6, instant.initial_state("p"))
    torch.testing.assert_close(out0, out0_inst, rtol=0, atol=0)
    assert st0.markets == {} and staged0.markets == {}
    assert set(staged0.pending) == {"US", "CN"}

    # No se puede leer otra sesión, ni cortar grafo ni exportar mientras haya sesiones pendientes.
    with pytest.raises(ValueError, match="pendiente de confirmar"):
        layer(x1, MARKETS, (START + DAY,) * 6, staged0)
    with pytest.raises(ValueError, match="confirmadas"):
        layer.detach_state(staged0)
    with pytest.raises(ValueError, match="confirmadas"):
        layer.export_state(staged0)

    # Confirmar escribe la sesión 0 para sesiones posteriores.
    st1 = layer.commit(staged0, differentiable=False)
    assert set(st1.markets) == {"US", "CN"} and st1.pending == {}
    assert st1.markets["US"].updates == 1 and st1.markets["US"].committed_at == START

    # No se puede volver a leer ni escribir el mismo instante o uno anterior ya confirmado.
    with pytest.raises(ValueError, match="confirmó"):
        layer(x0, MARKETS, (START,) * 6, st1)
    with pytest.raises(ValueError, match="confirmó"):
        layer(x0, MARKETS, (START - DAY,) * 6, st1)

    # Mezclar dos instantes del mismo mercado en el mismo bloque se rechaza.
    with pytest.raises(ValueError, match="misma sesión"):
        layer(
            x1,
            MARKETS,
            (START + DAY, START + DAY, START + 2 * DAY, START + DAY, START + DAY, START + DAY),
            st1,
        )

    # En la sesión 1 la lectura histórica SÍ cambia respecto a instant_only.
    out1, staged1 = layer(x1, MARKETS, (START + DAY,) * 6, st1)
    out1_inst, _ = instant(x1, MARKETS, (START + DAY,) * 6, instant.initial_state("p"))
    assert not torch.allclose(out1, out1_inst)

    # Aislamiento estricto entre mercados a través de las sesiones: perturbar CN en s=0 y s=1
    # no altera ni un bit las salidas ni el estado confirmado de US.
    x0_cn = x0.clone()
    x0_cn[[1, 4]] += 12.0
    x1_cn = x1.clone()
    x1_cn[[1, 4]] -= 9.0
    out0_alt, staged0_alt = layer(x0_cn, MARKETS, (START,) * 6, st0)
    st1_alt = layer.commit(staged0_alt, differentiable=False)
    out1_alt, staged1_alt = layer(x1_cn, MARKETS, (START + DAY,) * 6, st1_alt)
    st2 = layer.commit(staged1, differentiable=False)
    st2_alt = layer.commit(staged1_alt, differentiable=False)
    us = [0, 2, 3, 5]
    assert torch.equal(out0_alt[us], out0[us])
    assert torch.equal(out1_alt[us], out1[us])
    assert torch.equal(st2_alt.markets["US"].holo_re, st2.markets["US"].holo_re)
    assert torch.equal(st2_alt.markets["US"].holo_im, st2.markets["US"].holo_im)
    assert not torch.equal(st2_alt.markets["CN"].holo_re, st2.markets["CN"].holo_re)


def test_multiscale_half_lives_distinguish_recent_rotation_from_slow_history():
    """Cambiar una sesión antigua afecta más a las semividas largas que a las cortas."""
    layer = opened_layer(EARLY, hidden=16, seed=53, dtype=torch.float64)
    xs = [sample_encoded(6, 16, seed=60 + i, dtype=torch.float64) for i in range(4)]

    def run_seq(seq):
        st = layer.initial_state("p")
        out = None
        for idx, x in enumerate(seq):
            out, staged = layer(x, ("US",) * 6, (START + idx * DAY,) * 6, st)
            st = layer.commit(staged, differentiable=False)
        return out, st

    base_out, base_st = run_seq(xs)
    # Perturbar t=0 con una dirección no uniforme cambia más τ=10 que τ=5 y τ=5 más que τ=1.
    shock = sample_encoded(6, 16, seed=999, dtype=torch.float64)
    xs_old = [xs[0] + shock, xs[1], xs[2], xs[3]]
    old_out, old_st = run_seq(xs_old)
    assert not torch.allclose(old_out, base_out)
    base_hat = base_st.markets["US"].holo_re / base_st.markets["US"].weight[:, None, None, None]
    old_hat = old_st.markets["US"].holo_re / old_st.markets["US"].weight[:, None, None, None]
    diff_per_scale = (old_hat - base_hat).abs().mean(dim=(1, 2, 3))
    assert diff_per_scale[0] < diff_per_scale[1] < diff_per_scale[2]


@pytest.mark.parametrize("config", [EARLY, INSTANT, SELF])
def test_permuting_rows_is_equivariant_and_committed_state_is_invariant(config):
    layer = opened_layer(config, hidden=16, seed=47, dtype=torch.float64)
    x0 = sample_encoded(6, 16, seed=53, dtype=torch.float64)
    x1 = sample_encoded(6, 16, seed=59, dtype=torch.float64)
    order = [3, 0, 5, 1, 4, 2]

    out0, st0 = layer(x0, MARKETS, (START,) * 6, layer.initial_state("p"))
    st1 = layer.commit(st0, differentiable=False)
    out1, _ = layer(x1, MARKETS, (START + DAY,) * 6, st1)

    p_markets = tuple(MARKETS[i] for i in order)
    p_out0, p_st0 = layer(x0[order], p_markets, (START,) * 6, layer.initial_state("p"))
    p_st1 = layer.commit(p_st0, differentiable=False)
    p_out1, _ = layer(x1[order], p_markets, (START + DAY,) * 6, p_st1)

    torch.testing.assert_close(p_out0, out0[order], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(p_out1, out1[order], rtol=1e-12, atol=1e-12)
    if config.mode == PERSISTENT:
        for m in ("US", "CN"):
            torch.testing.assert_close(
                p_st1.markets[m].holo_re, st1.markets[m].holo_re, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                p_st1.markets[m].holo_im, st1.markets[m].holo_im, rtol=1e-12, atol=1e-12
            )


def test_controls_share_exact_parameters_and_semantics_for_instant_shuffled_and_self():
    main = PersistentSessionHologram(16, EARLY, seed=11, dtype=torch.float64)
    ctrl_inst = PersistentSessionHologram(16, INSTANT, seed=11, dtype=torch.float64)
    ctrl_shuf = PersistentSessionHologram(16, SHUFFLED, seed=11, dtype=torch.float64)
    ctrl_self = PersistentSessionHologram(16, SELF, seed=11, dtype=torch.float64)
    for other in (ctrl_inst, ctrl_shuf, ctrl_self):
        assert set(main.state_dict()) == set(other.state_dict())
        for name, tensor in main.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, other.state_dict()[name]), name

    x0 = sample_encoded(6, 16, seed=61, dtype=torch.float64)
    x1 = sample_encoded(6, 16, seed=67, dtype=torch.float64)

    # self_only es puramente fila a fila: cambiar pares ni sesiones previas lo altera.
    open_self = opened_layer(SELF, hidden=16, seed=11, dtype=torch.float64)
    out_s0, st_s0 = open_self(x0, MARKETS, (START,) * 6, open_self.initial_state("p"))
    st_s1 = open_self.commit(st_s0, differentiable=False)
    x0_other = x0.clone()
    x0_other[[2, 3, 5]] += 9.0
    out_s0_other, _ = open_self(x0_other, MARKETS, (START,) * 6, open_self.initial_state("p"))
    assert torch.equal(out_s0_other[0], out_s0[0])
    assert st_s1.markets == {}

    # shuffled_history conserva el mismo estado confirmado que persistent, es determinista,
    # difiere en la lectura histórica de la sesión 1 y conserva el aislamiento entre mercados.
    open_main = opened_layer(EARLY, hidden=16, seed=11, dtype=torch.float64)
    open_shuf = opened_layer(SHUFFLED, hidden=16, seed=11, dtype=torch.float64)
    _, m_st0 = open_main(x0, MARKETS, (START,) * 6, open_main.initial_state("p"))
    m_st1 = open_main.commit(m_st0, differentiable=False)
    m_out1, _ = open_main(x1, MARKETS, (START + DAY,) * 6, m_st1)

    _, s_st0 = open_shuf(x0, MARKETS, (START,) * 6, open_shuf.initial_state("p"))
    s_st1 = open_shuf.commit(s_st0, differentiable=False)
    assert torch.equal(s_st1.markets["US"].holo_re, m_st1.markets["US"].holo_re)
    assert torch.equal(s_st1.markets["US"].holo_im, m_st1.markets["US"].holo_im)
    s_out1_a, _ = open_shuf(x1, MARKETS, (START + DAY,) * 6, s_st1)
    s_out1_b, _ = open_shuf(x1, MARKETS, (START + DAY,) * 6, s_st1)
    assert torch.equal(s_out1_a, s_out1_b)
    assert not torch.allclose(s_out1_a, m_out1)
    x1_cn = x1.clone()
    x1_cn[[1, 4]] += 5.0
    s_out1_cn, _ = open_shuf(x1_cn, MARKETS, (START + DAY,) * 6, s_st1)
    assert torch.equal(s_out1_cn[[0, 2, 3, 5]], s_out1_a[[0, 2, 3, 5]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deterministic_outputs_and_gradients_in_fp32_and_fp64(dtype):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:

        def run_once():
            layer = opened_layer(EARLY, hidden=16, seed=71, dtype=dtype)
            x0 = sample_encoded(6, 16, seed=73, dtype=dtype).requires_grad_()
            x1 = sample_encoded(6, 16, seed=79, dtype=dtype).requires_grad_()
            o0, st0 = layer(x0, MARKETS, (START,) * 6, layer.initial_state("p"))
            st1 = layer.commit(st0, differentiable=True)
            o1, _ = layer(x1, MARKETS, (START + DAY,) * 6, st1)
            (o0.square().sum() + o1.square().sum()).backward()
            grads = {n: p.grad.clone() for n, p in layer.named_parameters()}
            return o1.detach(), x0.grad.clone(), x1.grad.clone(), grads

        out_a, gx0_a, gx1_a, g_a = run_once()
        out_b, gx0_b, gx1_b, g_b = run_once()
        assert torch.equal(out_a, out_b)
        assert torch.equal(gx0_a, gx0_b) and torch.equal(gx1_a, gx1_b)
        for name, grad in g_a.items():
            assert torch.isfinite(grad).all() and torch.equal(grad, g_b[name]), name
    finally:
        torch.use_deterministic_algorithms(previous)


def _graph_nodes(tensor):
    seen, stack = set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(parent for parent, _ in node.next_functions)
    return len(seen)


def test_commit_autograd_graph_size_is_constant_independent_of_session_rows():
    """El holograma confirmado tiene tamaño y grafo independientes del número de activos N."""
    sizes = {}
    for rows in (8, 64):
        layer = opened_layer(EARLY, hidden=16, seed=83, dtype=torch.float64)
        x0 = sample_encoded(rows, 16, seed=89, dtype=torch.float64).requires_grad_(True)
        _, staged = layer(x0, ("US",) * rows, (START,) * rows, layer.initial_state("p"))
        committed = layer.commit(staged, differentiable=True)
        sizes[rows] = _graph_nodes(committed.markets["US"].holo_re)
        (grad_x0,) = torch.autograd.grad(committed.markets["US"].holo_re.sum(), x0)
        assert torch.isfinite(grad_x0).all() and grad_x0.abs().sum() > 0
    assert sizes[8] == sizes[64], sizes


def test_disabled_default_preserves_identity_fingerprint_parameters_and_predictions_bitwise():
    base, batch = setup()
    explicit = model(persistent_session_hologram=None)
    assert explicit.config.identity() == base.config.identity()
    assert explicit._config_id() == base._config_id()
    assert explicit._parameter_id == base._parameter_id
    assert "persistent_session_hologram" not in base.config.identity()
    assert base.persistent_session_hologram is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    assert actual.next_persistent_hologram is None


@pytest.mark.parametrize("config", [EARLY, INSTANT, SHUFFLED, SELF])
def test_predictor_starts_bitwise_identical_to_control_across_all_modes(config):
    base, _ = setup()
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(persistent_session_hologram=config)
    ref_params = dict(base.named_parameters())
    all_params = dict(predictor.named_parameters())
    assert set(all_params) - set(ref_params) == {
        "persistent_session_hologram.norm.weight",
        "persistent_session_hologram.norm.bias",
        "persistent_session_hologram.proj.weight",
        "persistent_session_hologram.density.weight",
        "persistent_session_hologram.inst_norm.weight",
        "persistent_session_hologram.mem_norm.weight",
        "persistent_session_hologram.rot_norm.weight",
        "persistent_session_hologram.out_inst.weight",
        "persistent_session_hologram.out_mem.weight",
        "persistent_session_hologram.out_rot.weight",
        "persistent_session_hologram.scale_mem",
        "persistent_session_hologram.scale_rot",
        "persistent_session_hologram.gate",
    }
    for name, value in ref_params.items():
        assert torch.equal(all_params[name], value), name
    assert torch.count_nonzero(predictor.persistent_session_hologram.gate) == 0
    assert torch.count_nonzero(predictor.persistent_session_hologram.out_inst.weight) > 0
    assert torch.count_nonzero(predictor.persistent_session_hologram.out_mem.weight) > 0
    assert torch.count_nonzero(predictor.persistent_session_hologram.out_rot.weight) > 0

    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    holo = predictor.initial_persistent_hologram()
    actual = predictor.prepare(
        batch,
        predictor.initial_state(batch.flow_ids),
        session_complete=True,
        persistent_hologram=holo,
    )
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    assert torch.equal(actual.working_state, expected.working_state)
    assert actual.next_persistent_hologram is not None
    identity = predictor.config.identity()["persistent_session_hologram"]
    assert identity["kind"] == PERSISTENT_SESSION_HOLOGRAM and identity["mode"] == config.mode


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
def test_step_zero_and_step_one_gradients_and_cross_session_metagradients_with_zero_median_head(
    dtype,
):
    """Con mediana en cero (predicción constante) y pinball+IC, gate recibe gradiente en paso 0
    y tras abrir la puerta TODOS los parámetros de MPSH (incluidos los de memoria y escalas)
    reciben gradientes finitos no nulos a través de las sesiones."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        spec = specification()
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
            persistent_session_hologram=LATE,
        )
        predictor = FinancialPredictor(cfg, dtype=dtype)
        predictor.set_learning_statistics(statistics(cfg.learning_scale))
        batches = [
            make_session_batch(6, at=START + i * DAY, seed=101 + i, dtype=dtype) for i in range(2)
        ]

        # Paso 0: predicción puntual constante en cero; la puerta recibe gradiente != 0.
        st = predictor.initial_state(batches[0].flow_ids, differentiable=True)
        holo = predictor.initial_persistent_hologram()
        p0 = predictor.prepare(
            batches[0], st, differentiable=True, session_complete=True, persistent_hologram=holo
        )
        assert torch.count_nonzero(p0.point_predictions) == 0
        holo = predictor.commit_persistent_hologram(
            p0.next_persistent_hologram, differentiable=True
        )
        p1 = predictor.prepare(
            batches[1],
            p0.next_state,
            differentiable=True,
            session_complete=True,
            persistent_hologram=holo,
        )
        y1 = torch.tensor([-0.02, 0.03, -0.01, 0.04, -0.03, 0.02], dtype=dtype)
        sessions = [("US", START + DAY)] * 6
        term = SessionICTerm(weight=3.0, minimum_rows=3)
        loss0 = pinball_loss(p1.quantiles, y1) + term.term(p1.point_predictions, y1, sessions)
        assert torch.isfinite(loss0)
        loss0.backward()
        for name, param in predictor.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
        assert predictor.persistent_session_hologram.gate.grad is not None
        assert predictor.persistent_session_hologram.gate.grad.abs().sum() > 0

        # Un paso abre la puerta; en el paso 1 todos los parámetros de MPSH reciben gradiente != 0.
        with torch.no_grad():
            for param in predictor.parameters():
                if param.grad is not None:
                    param.sub_(0.1 * param.grad)
                    param.grad.zero_()
        predictor._seal_parameters()

        st = predictor.initial_state(batches[0].flow_ids, differentiable=True)
        holo = predictor.initial_persistent_hologram()
        p0 = predictor.prepare(
            batches[0], st, differentiable=True, session_complete=True, persistent_hologram=holo
        )
        holo = predictor.commit_persistent_hologram(
            p0.next_persistent_hologram, differentiable=True
        )
        p1 = predictor.prepare(
            batches[1],
            p0.next_state,
            differentiable=True,
            session_complete=True,
            persistent_hologram=holo,
        )
        loss1 = pinball_loss(p1.quantiles, y1) + term.term(p1.point_predictions, y1, sessions)
        assert torch.isfinite(loss1)
        loss1.backward()
        for name, param in predictor.persistent_session_hologram.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, f"Gradiente nulo tras abrir la puerta: {name}"
    finally:
        torch.use_deterministic_algorithms(previous)


def test_constant_all_zero_scalar_head_produces_finite_gradients():
    batch0 = make_session_batch(5, at=START, seed=201, dtype=torch.float64)
    batch1 = make_session_batch(5, at=START + DAY, seed=202, dtype=torch.float64)
    cfg = FinancialConfig(
        specification(),
        variant="mac_online",
        hidden_size=32,
        learning_scale=LearningScale(clip=5.0, price_returns=True, zero_median=True),
        persistent_session_hologram=EARLY,
    )
    predictor = FinancialPredictor(cfg, dtype=torch.float64)
    predictor.set_learning_statistics(statistics(cfg.learning_scale))
    with torch.no_grad():
        predictor.persistent_session_hologram.gate.fill_(0.2)
    predictor._seal_parameters()

    st = predictor.initial_state(batch0.flow_ids, differentiable=True)
    holo = predictor.initial_persistent_hologram()
    p0 = predictor.prepare(
        batch0, st, differentiable=True, session_complete=True, persistent_hologram=holo
    )
    assert torch.count_nonzero(p0.point_predictions) == 0
    holo = predictor.commit_persistent_hologram(p0.next_persistent_hologram, differentiable=True)
    p1 = predictor.prepare(
        batch1,
        p0.next_state,
        differentiable=True,
        session_complete=True,
        persistent_hologram=holo,
    )
    y = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.0], dtype=torch.float64)
    sessions = [("US", START + DAY)] * 5
    loss = (p1.point_predictions - y).abs().mean() + SessionICTerm(weight=3.0, minimum_rows=3).term(
        p1.point_predictions, y, sessions
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in predictor.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_late_position_and_coexistence_order_with_session_hologram_and_crrm():
    """Regla 3 de convivencia: HSI -> MPSH -> CRRM -> cabeza, y after_technical exige rama."""
    with pytest.raises(ValueError, match="rama técnica"):
        build(persistent_session_hologram=asdict(LATE))

    # Activar solo MPSH no toca session_hologram ni cross_sectional_rank_memory.
    only_mpsh, batch = build(
        technical_branch=asdict(DENSE), persistent_session_hologram=asdict(LATE)
    )
    assert only_mpsh.session_hologram is None and only_mpsh.cross_sectional_rank_memory is None
    assert "session_hologram" not in only_mpsh.config.identity()
    assert "cross_sectional_rank_memory" not in only_mpsh.config.identity()

    # Convivencia simultánea: HSI + MPSH + CRRM en el orden exacto de la Regla 3.
    crrm_cfg = CrossSectionalRankMemoryConfig(
        dim=8, time_constants=(5, 21), ridge=1.0, minimum_rows=3, mode="matured_rank", seed=0
    )
    hsi_cfg = SessionHologramConfig(
        bands=4, phase_dim=16, mode="interferometric", position="after_technical"
    )
    plain, _ = build(technical_branch=asdict(DENSE))
    combo, _ = build(
        technical_branch=asdict(DENSE),
        session_hologram=asdict(hsi_cfg),
        persistent_session_hologram=asdict(LATE),
        cross_sectional_rank_memory=asdict(crrm_cfg),
    )
    opened(plain.technical_branch)
    plain._seal_parameters()
    opened(combo.technical_branch)
    with torch.no_grad():
        combo.session_hologram.gate.fill_(0.15)
        combo.persistent_session_hologram.gate.fill_(0.20)
        combo.cross_sectional_rank_memory.out.weight.fill_(0.10)
    combo._seal_parameters()

    markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
    ref = emit(plain, batch)
    holo = combo.initial_persistent_hologram()
    rm = combo.initial_rank_memory()
    actual = combo.prepare(
        batch,
        combo.initial_state(batch.flow_ids),
        session_complete=True,
        persistent_hologram=holo,
        rank_memory=rm,
    )
    # Comprobar orden exacto: ref.working_state -> HSI -> MPSH -> CRRM.
    after_hsi = combo.session_hologram(ref.working_state, markets, batch.prediction_at)
    after_mpsh, _ = combo.persistent_session_hologram(after_hsi, markets, batch.prediction_at, holo)
    rm_keys = combo.cross_sectional_rank_memory.keys(after_mpsh, markets, batch.prediction_at)
    expected_ws = after_mpsh + combo.cross_sectional_rank_memory.read(rm_keys, markets, rm)
    torch.testing.assert_close(actual.working_state, expected_ws, rtol=0, atol=0)
    torch.testing.assert_close(actual.rank_memory_keys, rm_keys, rtol=0, atol=0)
    # No modifica el estado rápido de MAC.
    torch.testing.assert_close(
        actual.next_state.mac.memory.weights[0],
        ref.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )


def test_export_restore_detach_and_state_dict_round_trip_bitwise():
    predictor = model(persistent_session_hologram=EARLY)
    with torch.no_grad():
        predictor.persistent_session_hologram.gate.fill_(0.18)
    predictor._seal_parameters()
    batch = session_batch(6, markets=list(MARKETS))
    st = predictor.initial_state(batch.flow_ids, differentiable=True)
    holo = predictor.initial_persistent_hologram()
    p0 = predictor.prepare(
        batch, st, differentiable=True, session_complete=True, persistent_hologram=holo
    )
    # Exportar antes de confirmar se rechaza.
    with pytest.raises(ValueError, match="confirmadas"):
        predictor.export_persistent_hologram(p0.next_persistent_hologram)
    committed = predictor.commit_persistent_hologram(
        p0.next_persistent_hologram, differentiable=True
    )
    assert committed.markets["US"].holo_re.requires_grad
    detached = predictor.detach_persistent_hologram(committed)
    assert not detached.markets["US"].holo_re.requires_grad
    payload = predictor.export_persistent_hologram(detached)
    restored = predictor.restore_persistent_hologram(payload)
    for market in ("US", "CN"):
        assert torch.equal(restored.markets[market].holo_re, detached.markets[market].holo_re)
        assert torch.equal(restored.markets[market].holo_im, detached.markets[market].holo_im)
        assert torch.equal(restored.markets[market].mean_c, detached.markets[market].mean_c)
        assert torch.equal(restored.markets[market].mean_s, detached.markets[market].mean_s)
        assert torch.equal(restored.markets[market].weight, detached.markets[market].weight)
        assert restored.markets[market].committed_at == detached.markets[market].committed_at
        assert restored.markets[market].updates == detached.markets[market].updates

    # Payload manipulado se rechaza.
    bad_payload = dict(payload, schema_version=2)
    with pytest.raises(ValueError):
        predictor.restore_persistent_hologram(bad_payload)

    # state_dict round-trip y rechazo entre modos con distinta identidad.
    twin = model(persistent_session_hologram=EARLY)
    twin.load_state_dict(predictor.state_dict())
    batch2 = session_batch(6, markets=list(MARKETS), at=batch.prediction_at[0] + DAY)
    expected = predictor.prepare(
        batch2, p0.next_state, session_complete=True, persistent_hologram=detached
    )
    actual = twin.prepare(
        batch2, p0.next_state, session_complete=True, persistent_hologram=restored
    )
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    with pytest.raises(ValueError):
        model(persistent_session_hologram=INSTANT).load_state_dict(predictor.state_dict())
    with pytest.raises(ValueError):
        model(persistent_session_hologram=SHUFFLED).load_state_dict(predictor.state_dict())


def test_invalid_declarations_contracts_autocast_and_guards():
    raw = asdict(EARLY)
    assert checked_predictor_changes(dict(persistent_session_hologram=raw)) == dict(
        persistent_session_hologram=raw
    )
    base, batch = setup()
    cfg = replace(base.config, persistent_session_hologram=json.loads(json.dumps(raw)))
    assert cfg.persistent_session_hologram == EARLY
    assert declared_sources(base.config) == ()
    assert declared_sources(cfg) == (MPSH_MODULE,)

    # Consumidor congelado rechaza explícitamente MPSH.
    frozen_pred = model(persistent_session_hologram=EARLY).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="holograma persistente"):
        FrozenFinancialConsumer(frozen_pred)

    # Sin declarar MPSH no admite estado; declarándolo exige estado y sesión completa.
    with pytest.raises(ValueError, match="no declara"):
        base.prepare(batch, base.initial_state(batch.flow_ids), persistent_hologram=object())
    active = model(persistent_session_hologram=EARLY)
    with pytest.raises(ValueError, match="sesión completa"):
        active.prepare(
            batch,
            active.initial_state(batch.flow_ids),
            persistent_hologram=active.initial_persistent_hologram(),
        )
    with pytest.raises(ValueError, match="holograma persistente"):
        active.prepare(batch, active.initial_state(batch.flow_ids), session_complete=True)

    # Control local C no se combina con MPSH.
    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(
            cfg,
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )

    # Variantes distintas de mac_online y declaraciones inválidas se rechazan.
    for variant in ("mac_frozen", "mac_disabled", "transformer_direct"):
        with pytest.raises(ValueError, match="mac_online"):
            replace(base.config, variant=variant, persistent_session_hologram=EARLY)

    for bad in (
        dict(bands=3, phase_dim=8, half_lives=[1, 5], mode="persistent"),
        dict(bands=4, phase_dim=1, half_lives=[1, 5], mode="persistent"),
        dict(bands=4, phase_dim=65, half_lives=[1, 5], mode="persistent"),
        dict(bands=4, phase_dim=8, half_lives=[], mode="persistent"),
        dict(bands=4, phase_dim=8, half_lives=[5, 5], mode="persistent"),
        dict(bands=4, phase_dim=8, half_lives=[10, 5], mode="persistent"),
        dict(bands=4, phase_dim=8, half_lives=[0, 5], mode="persistent"),
        dict(bands=4, phase_dim=8, half_lives=[1, 2, 3, 4, 5], mode="persistent"),
        dict(bands=4, phase_dim=8, half_lives=[1, 5], mode="unknown"),
        dict(bands=4, phase_dim=8, half_lives=[1, 5], mode="persistent", position="before_mac"),
        dict(bands=4, phase_dim=8, half_lives=[1, 5], mode="persistent", seed=-1),
        dict(bands=4, phase_dim=8, half_lives=[1, 5], mode="persistent", extra=1),
        dict(bands=4, phase_dim=8, mode="persistent"),
        "persistent_session_hologram_v1",
    ):
        with pytest.raises(ValueError):
            parse_persistent_session_hologram(bad)

    # Dust rechaza un predictor con MPSH (por no ser transformer_direct sin ampliaciones).
    with pytest.raises(ValueError):
        FinancialDustAdapter(active)

    # Autocast y dimensiones inválidas se rechazan.
    with pytest.raises(ValueError, match="múltiplo"):
        PersistentSessionHologram(30, EARLY, seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        PersistentSessionHologram(16, EARLY, seed=1, dtype=torch.bfloat16)
    layer = PersistentSessionHologram(16, EARLY, seed=1, dtype=torch.float32)
    x = sample_encoded(6, 16, seed=1, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer(x, MARKETS, (START,) * 6, layer.initial_state("p"))
    with pytest.raises(ValueError, match="finit"):
        bad_x = x.clone()
        bad_x[0, 0] = math.nan
        layer(bad_x, MARKETS, (START,) * 6, layer.initial_state("p"))

    # ChronologicalTrainer._code() incluye el módulo solo cuando MPSH está declarado.
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    tr = object.__new__(ChronologicalTrainer)
    tr.recipe = ChronologicalRecipe()
    for val, present in ((None, False), (EARLY, True)):
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
                persistent_session_hologram=val,
            )
        )
        assert (MPSH_MODULE in tr._code()) is present


def test_campaign_v12_screen3_accepts_single_variable_mpsh_and_controls():
    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    for suffix, mode in (
        ("mpsh", PERSISTENT),
        ("mpsh_instant", INSTANT_ONLY),
        ("mpsh_shuffled", SHUFFLED_HISTORY),
        ("mpsh_self", SELF_ONLY),
    ):
        cfg = PersistentSessionHologramConfig(
            bands=4,
            phase_dim=16,
            half_lives=(1, 5, 10),
            mode=mode,
            position="after_technical",
            seed=0,
        )
        doc["models"][f"titans_mac_{suffix}"] = dict(
            doc["models"]["titans_mac"],
            tier="s3_inventions",
            predictor_changes=dict(persistent_session_hologram=asdict(cfg)),
        )
    validated = comparison_campaign.checked_declaration(doc, root=root)
    assert "titans_mac_mpsh" in validated["models"]
    assert "titans_mac_mpsh_instant" in validated["models"]
    assert "titans_mac_mpsh_shuffled" in validated["models"]
    assert "titans_mac_mpsh_self" in validated["models"]
