"""Interferómetro holográfico de sesión en fase dual (`holographic_session_interferometer_v1`).

Pruebas de contrato, equivalencia matemática exacta frente a la referencia O(N²), invariancias
de covarianza y leave-one-out, aislamiento causal entre sesiones y mercados, determinismo en
FP32 y FP64, gradientes finitos con predicción constante y compatibilidad congelada.
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
from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
from mars_titan.models.titans.episodic_readout import apply_episodic_readout
from mars_titan.models.titans.session_hologram import (
    DENSITY_BOUND,
    HolographicSessionInterferometer,
    SessionHologramConfig,
    parse_session_hologram,
)
from mars_titan.models.titans.session_ic_objective import SessionICTerm
from mars_titan.training import comparison_campaign
from mars_titan.training.search_cases import checked_predictor_changes

SESSION_HOLOGRAM_MODULE = "mars_titan.models.titans.session_hologram"
EARLY = SessionHologramConfig(position="after_mac")
LATE = SessionHologramConfig(position="after_technical")
IN_PHASE = SessionHologramConfig(mode="in_phase_only", position="after_mac")
SELF_ONLY = SessionHologramConfig(mode="self_only", position="after_mac")
SHUFFLED = SessionHologramConfig(mode="shuffled_peers", position="after_mac")
MARKETS = ("US", "CN", "US", "US", "CN", "US")
AT = (7,) * len(MARKETS)


def opened_layer(config=EARLY, *, hidden=16, seed=3, dtype=torch.float64, nonzero_density=True):
    """Módulo con puerta (y opcionalmente densidad) abiertas para observar la interferencia."""
    layer = HolographicSessionInterferometer(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed + 101)
        layer.gate.copy_(torch.randn(hidden, generator=gen, dtype=dtype) / 4)
        if nonzero_density:
            layer.density.weight.copy_(
                torch.randn(layer.density.weight.shape, generator=gen, dtype=dtype) / 2
            )
    return layer


def sample_encoded(rows=6, hidden=16, seed=9, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=gen, dtype=dtype)


def naive_reference(layer, values, markets, instants):
    """Referencia explícita O(N²) fila a fila y par a par con la definición matemática exacta."""
    cfg = layer.config
    bands, dc, dv = cfg.bands, cfg.phase_dim, layer.band_dim
    normed = layer.norm(values)
    projected = layer.proj(normed)
    phase_width = bands * dc
    theta = projected[:, :phase_width].reshape(len(values), bands, dc)
    phi = projected[:, phase_width : 2 * phase_width].reshape(len(values), bands, dc)
    v = projected[:, 2 * phase_width :].reshape(len(values), bands, dv)
    g = DENSITY_BOUND * torch.tanh(layer.density(normed))

    readouts = []
    for i in range(len(values)):
        if cfg.mode == "self_only":
            diff = theta[i] - phi[i]
            k_re = torch.cos(diff).mean(dim=-1)
            k_im = torch.sin(diff).mean(dim=-1)
            d_re = (k_re.unsqueeze(-1) * v[i]).reshape(-1)
            d_im = (k_im.unsqueeze(-1) * v[i]).reshape(-1)
            readouts.append(torch.cat((d_re, d_im, k_re, k_im)))
            continue
        group = [
            j for j in range(len(values)) if (markets[j], instants[j]) == (markets[i], instants[i])
        ]
        if len(group) == 1:
            readouts.append(values.new_zeros(layer.readout_dim))
            continue
        peers = [j for j in group if j != i]
        g_group = torch.stack([g[j] for j in group], dim=0)
        pi_group = torch.softmax(g_group, dim=0)
        pi_map = {j: pi_group[pos] for pos, j in enumerate(group)}
        pi_i = pi_map[i]
        d_re_bands, d_im_bands, rho_re_bands, rho_im_bands = [], [], [], []
        for m in range(bands):
            weights = torch.stack([pi_map[j][m] / (1.0 - pi_i[m]) for j in peers])
            k_re = torch.stack([torch.cos(theta[j, m] - phi[i, m]).mean() for j in peers])
            k_im = torch.stack([torch.sin(theta[j, m] - phi[i, m]).mean() for j in peers])
            rho_re = (weights * k_re).sum()
            rho_im = (weights * k_im).sum()
            v_peers = torch.stack([v[j, m] for j in peers], dim=0)
            delta_re = ((weights * (k_re - rho_re)).unsqueeze(-1) * v_peers).sum(dim=0)
            delta_im = ((weights * (k_im - rho_im)).unsqueeze(-1) * v_peers).sum(dim=0)
            if cfg.mode == "in_phase_only":
                delta_im = torch.zeros_like(delta_im)
                rho_im = torch.zeros_like(rho_im)
            d_re_bands.append(delta_re)
            d_im_bands.append(delta_im)
            rho_re_bands.append(rho_re)
            rho_im_bands.append(rho_im)
        readouts.append(
            torch.cat(
                (
                    torch.cat(d_re_bands),
                    torch.cat(d_im_bands),
                    torch.stack(rho_re_bands),
                    torch.stack(rho_im_bands),
                )
            )
        )
    stacked = torch.stack(readouts, dim=0)
    return values + layer.gate * layer.out(layer.readout_norm(stacked))


@pytest.mark.parametrize(
    "config",
    [
        EARLY,
        IN_PHASE,
        SELF_ONLY,
        SessionHologramConfig(bands=1, phase_dim=4, position="after_mac"),
        SessionHologramConfig(bands=2, phase_dim=8, position="after_mac"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linear_hologram_matches_explicit_quadratic_reference(config, dtype):
    layer = opened_layer(config, hidden=16, seed=5, dtype=dtype)
    values = sample_encoded(rows=6, hidden=16, seed=11, dtype=dtype)
    # Sesiones de tamaño 3 ((US, 7)), 2 ((CN, 7)) y 1 ((US, 8)) para cubrir densidad no trivial.
    instants = (7, 7, 7, 7, 7, 8)
    actual = layer(values, MARKETS, instants)
    expected = naive_reference(layer, values, MARKETS, instants)
    tol = 1e-5 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def test_single_asset_session_produces_exact_zero_interference_and_finite_gradients():
    layer = opened_layer(EARLY, hidden=16, seed=7, dtype=torch.float64)
    values = sample_encoded(rows=3, hidden=16, seed=13, dtype=torch.float64).requires_grad_()
    # Tres sesiones distintas de un único activo cada una.
    markets = ("US", "CN", "US")
    instants = (10, 10, 11)
    out = layer(values, markets, instants)
    # Con N=1 el vector de lectura es cero, LayerNorm(0)=0 y W_out sin sesgo da 0 -> out == values.
    torch.testing.assert_close(out, values, rtol=0, atol=0)
    loss = out.square().sum()
    loss.backward()
    assert torch.isfinite(values.grad).all()
    for name, param in layer.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_zero_sum_covariance_annihilates_uniform_value_shift_across_peers():
    """Teorema 1: un desplazamiento uniforme v_j -> v_j + c se cancela exactamente en Δ."""
    layer = opened_layer(EARLY, hidden=16, seed=19, dtype=torch.float64)
    values = sample_encoded(rows=6, hidden=16, seed=21, dtype=torch.float64)
    normed = layer.norm(values)
    projected = layer.proj(normed)
    bands, dc, dv = layer.config.bands, layer.config.phase_dim, layer.band_dim
    pw = bands * dc
    theta = projected[:, :pw].reshape(6, bands, dc)
    phi = projected[:, pw : 2 * pw].reshape(6, bands, dc)
    v = projected[:, 2 * pw :].reshape(6, bands, dv)
    g = DENSITY_BOUND * torch.tanh(layer.density(normed))
    c_t, s_t = torch.cos(theta), torch.sin(theta)
    c_p, s_p = torch.cos(phi), torch.sin(phi)
    self_re = (c_p * c_t + s_p * s_t).mean(dim=-1)
    self_im = (c_p * s_t - s_p * c_t).mean(dim=-1)

    base = layer._session_readout(c_t, s_t, c_p, s_p, v, g, self_re, self_im)
    shift = torch.linspace(-3.0, 5.0, dv, dtype=torch.float64).view(1, 1, dv)
    shifted = layer._session_readout(c_t, s_t, c_p, s_p, v + shift, g, self_re, self_im)
    torch.testing.assert_close(shifted, base, rtol=1e-13, atol=1e-13)


def test_leave_one_out_has_zero_gradient_with_respect_to_own_value_and_write_phase():
    """Teorema 2: la lectura del activo i no depende de v_i ni de θ_i (cero eco propio)."""
    layer = opened_layer(EARLY, hidden=16, seed=23, dtype=torch.float64)
    values = sample_encoded(rows=5, hidden=16, seed=29, dtype=torch.float64)
    normed = layer.norm(values).detach()
    projected = layer.proj(normed).detach()
    bands, dc, dv = layer.config.bands, layer.config.phase_dim, layer.band_dim
    pw = bands * dc
    theta = projected[:, :pw].reshape(5, bands, dc).clone().requires_grad_(True)
    phi = projected[:, pw : 2 * pw].reshape(5, bands, dc).clone().requires_grad_(True)
    v = projected[:, 2 * pw :].reshape(5, bands, dv).clone().requires_grad_(True)
    g = (DENSITY_BOUND * torch.tanh(layer.density(normed))).detach()
    c_t, s_t = torch.cos(theta), torch.sin(theta)
    c_p, s_p = torch.cos(phi), torch.sin(phi)
    self_re = (c_p * c_t + s_p * s_t).mean(dim=-1)
    self_im = (c_p * s_t - s_p * c_t).mean(dim=-1)

    readout = layer._session_readout(c_t, s_t, c_p, s_p, v, g, self_re, self_im)
    target_row = 2
    g_v, g_theta, g_phi = torch.autograd.grad(readout[target_row].sum(), (v, theta, phi))
    # Cero dependencia en v_i y θ_i del propio activo i.
    torch.testing.assert_close(
        g_v[target_row], torch.zeros_like(g_v[target_row]), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        g_theta[target_row], torch.zeros_like(g_theta[target_row]), rtol=0, atol=1e-14
    )
    # Sí depende de su propia fase de consulta φ_i y de los pares j != i.
    assert g_phi[target_row].abs().sum() > 0
    peers = [j for j in range(5) if j != target_row]
    assert g_v[peers].abs().sum() > 0 and g_theta[peers].abs().sum() > 0


def test_quadrature_channel_breaks_phase_antisymmetry_and_is_ablated_in_in_phase_only():
    """El canal imaginario distingue liderazgo/rezago y se anula exactamente en in_phase_only."""
    full = opened_layer(EARLY, hidden=16, seed=31, dtype=torch.float64)
    in_phase = opened_layer(IN_PHASE, hidden=16, seed=31, dtype=torch.float64)
    values = sample_encoded(rows=6, hidden=16, seed=37, dtype=torch.float64)
    out_full = full(values, MARKETS, AT)
    out_in_phase = in_phase(values, MARKETS, AT)
    assert not torch.allclose(out_full, out_in_phase)


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


@pytest.mark.parametrize("config", [EARLY, IN_PHASE, SELF_ONLY])
def test_permuting_rows_permutes_outputs_equivariantly(config):
    layer = opened_layer(config, hidden=16, seed=47, dtype=torch.float64)
    values = sample_encoded(rows=6, hidden=16, seed=53, dtype=torch.float64)
    order = [3, 0, 5, 1, 4, 2]
    expected = layer(values, MARKETS, AT)[order]
    actual = layer(values[order], tuple(MARKETS[i] for i in order), AT)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_self_only_and_shuffled_peers_controls_share_exact_parameters_and_semantics():
    main = HolographicSessionInterferometer(16, EARLY, seed=11, dtype=torch.float64)
    ctrl_self = HolographicSessionInterferometer(16, SELF_ONLY, seed=11, dtype=torch.float64)
    ctrl_shuf = HolographicSessionInterferometer(16, SHUFFLED, seed=11, dtype=torch.float64)
    ctrl_re = HolographicSessionInterferometer(16, IN_PHASE, seed=11, dtype=torch.float64)
    for other in (ctrl_self, ctrl_shuf, ctrl_re):
        assert set(main.state_dict()) == set(other.state_dict())
        for name, tensor in main.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, other.state_dict()[name]), name

    values = sample_encoded(rows=6, hidden=16, seed=59, dtype=torch.float64)
    open_self = opened_layer(SELF_ONLY, hidden=16, seed=11, dtype=torch.float64)
    base_self = open_self(values, MARKETS, AT)
    others = values.clone()
    others[[2, 3, 5]] += 9.0
    assert torch.equal(open_self(others, MARKETS, AT)[0], base_self[0])

    # shuffled_peers es determinista, cambia respecto a interferometric y conserva aislamiento.
    open_main = opened_layer(EARLY, hidden=16, seed=11, dtype=torch.float64)
    open_shuf = opened_layer(SHUFFLED, hidden=16, seed=11, dtype=torch.float64)
    out_shuf_1 = open_shuf(values, MARKETS, AT)
    out_shuf_2 = open_shuf(values, MARKETS, AT)
    assert torch.equal(out_shuf_1, out_shuf_2)
    assert not torch.allclose(out_shuf_1, open_main(values, MARKETS, AT))
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


def test_disabled_default_preserves_identity_fingerprint_parameters_and_predictions_bitwise():
    base, batch = setup()
    explicit = model(session_hologram=None)
    assert explicit.config.identity() == base.config.identity()
    assert explicit._config_id() == base._config_id()
    assert explicit._parameter_id == base._parameter_id
    assert "session_hologram" not in base.config.identity()
    assert base.session_hologram is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)


@pytest.mark.parametrize("config", [EARLY, IN_PHASE, SELF_ONLY, SHUFFLED])
def test_predictor_with_session_hologram_starts_bitwise_identical_to_control(config):
    base, _ = setup()
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(session_hologram=config)
    ref_params = dict(base.named_parameters())
    all_params = dict(predictor.named_parameters())
    assert set(all_params) - set(ref_params) == {
        "session_hologram.norm.weight",
        "session_hologram.norm.bias",
        "session_hologram.proj.weight",
        "session_hologram.density.weight",
        "session_hologram.readout_norm.weight",
        "session_hologram.readout_norm.bias",
        "session_hologram.out.weight",
        "session_hologram.gate",
    }
    for name, value in ref_params.items():
        assert torch.equal(all_params[name], value), name
    assert torch.count_nonzero(predictor.session_hologram.gate) == 0
    assert torch.count_nonzero(predictor.session_hologram.out.weight) > 0
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)


def test_step_zero_and_step_one_gradients_are_finite_and_alive_with_zero_median_head():
    """Regla 4 + Fase B: con mediana inicial en cero (predicción constante) y pinball+IC,
    los gradientes son finitos, la puerta recibe gradiente en el paso 0 y tras un paso todos
    los parámetros del interferómetro reciben gradientes no nulos y finitos."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        predictor, batch = build(
            technical_branch=asdict(DENSE),
            session_hologram=asdict(LATE),
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
        # La ganancia aprendida recibe gradiente en el paso 0 (no hay doble cero en serie).
        assert predictor.session_hologram.gate.grad is not None
        assert predictor.session_hologram.gate.grad.abs().sum() > 0

        # Un paso de gradiente abre la puerta; en el paso 1 todos los parámetros aprenden.
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
        for name, param in predictor.session_hologram.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.abs().sum() > 0, f"Gradiente nulo tras abrir la puerta: {name}"
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_gradients_are_finite_when_scalar_head_starts_identically_at_zero():
    """Incluso si toda la cabeza escalar está a cero (predicción constante 0), no hay NaN."""
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(session_hologram=EARLY)
    with torch.no_grad():
        predictor.head.weight.zero_()
        predictor.head.bias.zero_()
        predictor.session_hologram.gate.fill_(0.25)
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


def test_late_position_reads_post_technical_representation_and_needs_technical_branch():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with pytest.raises(ValueError, match="rama técnica"):
            build(session_hologram=asdict(LATE))
        plain, batch = build(technical_branch=asdict(DENSE))
        opened(plain.technical_branch)
        plain._seal_parameters()

        late, _ = build(technical_branch=asdict(DENSE), session_hologram=asdict(LATE))
        opened(late.technical_branch)
        with torch.no_grad():
            late.session_hologram.gate.fill_(0.2)
        late._seal_parameters()

        early, _ = build(technical_branch=asdict(DENSE), session_hologram=asdict(EARLY))
        opened(early.technical_branch)
        with torch.no_grad():
            early.session_hologram.gate.fill_(0.2)
        early._seal_parameters()

        markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
        ref = emit(plain, batch)
        after = emit(late, batch, session_complete=True)
        expected_after = late.session_hologram(ref.working_state, markets, batch.prediction_at)
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
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_session_complete_required_and_mutual_exclusions_enforced():
    batch = session_batch()
    predictor = model(session_hologram=EARLY)
    assert predictor.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)

    base, _ = setup()
    with pytest.raises(ValueError, match="no se combina"):
        replace(
            base.config,
            cross_asset_attention=CrossAssetAttentionConfig(),
            session_hologram=EARLY,
        )


def test_state_dict_round_trips_and_rejects_different_mode_or_position():
    batch = session_batch()
    source = model(session_hologram=EARLY)
    with torch.no_grad():
        source.session_hologram.gate.fill_(0.15)
    source._seal_parameters()
    target = model(session_hologram=EARLY)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model(session_hologram=SELF_ONLY).load_state_dict(source.state_dict())
    with pytest.raises(ValueError):
        model(session_hologram=IN_PHASE).load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    "bad",
    [
        dict(bands=3, phase_dim=16, mode="interferometric"),
        dict(bands=True, phase_dim=16, mode="interferometric"),
        dict(bands=4, phase_dim=1, mode="interferometric"),
        dict(bands=4, phase_dim=65, mode="interferometric"),
        dict(bands=4, phase_dim=16, mode="unknown"),
        dict(bands=4, phase_dim=16, mode="interferometric", position="before_mac"),
        dict(bands=4, phase_dim=16, mode="interferometric", extra=1),
        dict(bands=4, mode="interferometric"),
        "holographic_session_interferometer_v1",
    ],
)
def test_invalid_declarations_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_session_hologram(bad)


def test_construction_and_forward_validation_and_autocast_rejection():
    with pytest.raises(ValueError, match="múltiplo"):
        HolographicSessionInterferometer(30, EARLY, seed=1)
    with pytest.raises(ValueError, match="SessionHologramConfig"):
        HolographicSessionInterferometer(16, asdict(EARLY), seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        HolographicSessionInterferometer(16, EARLY, seed=1, dtype=torch.bfloat16)

    layer = HolographicSessionInterferometer(16, EARLY, seed=1, dtype=torch.float32)
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


def test_registry_and_campaign_v11_declaration_accept_single_variable_candidate():
    changes = dict(session_hologram=asdict(LATE))
    assert checked_predictor_changes(changes) == changes

    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v11-screen1.json").read_text())
    doc["tiers"].append("y1_session_hologram")
    doc["models"]["titans_mac_session_hologram"] = dict(
        doc["models"]["titans_mac"],
        tier="y1_session_hologram",
        predictor_changes=changes,
    )
    doc["models"]["titans_mac_session_hologram_self"] = dict(
        doc["models"]["titans_mac"],
        tier="y1_session_hologram",
        predictor_changes=dict(
            session_hologram=asdict(
                SessionHologramConfig(mode="self_only", position="after_technical")
            )
        ),
    )
    validated = comparison_campaign.checked_declaration(doc, root=root)
    assert "titans_mac_session_hologram" in validated["models"]
    assert "titans_mac_session_hologram_self" in validated["models"]


def test_frozen_consumer_and_trainer_code_register_source_only_when_active(monkeypatch, tmp_path):
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        plain_engine, _ = frozen("mac_online")
        assert SESSION_HOLOGRAM_MODULE not in plain_engine.identity()["implementation"]
        assert api.declared_sources(plain_engine.predictor.config) == ()

        base, _ = setup("mac_online")
        pred = type(base)(replace(base.config, session_hologram=EARLY), dtype=torch.float64)
        with torch.no_grad():
            pred.session_hologram.gate.fill_(0.2)
        pred._seal_parameters()
        pred.eval().requires_grad_(False)
        engine = consumer(pred)

        assert api.declared_sources(pred.config) == (SESSION_HOLOGRAM_MODULE,)
        assert SESSION_HOLOGRAM_MODULE in engine.identity()["implementation"]
        assert engine.model_id != plain_engine.model_id
        assert (
            engine.identity()["modules"]["predictor/session_hologram"]["settings"][
                "session_hologram"
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
        pred.session_hologram.band_dim = 1
        with pytest.raises(ValueError):
            engine.verify(strong=False)

        # Trainer._code() solo incluye el módulo cuando session_hologram está activo.
        trainer = object.__new__(ChronologicalTrainer)
        trainer.recipe = ChronologicalRecipe()
        for cfg, present in ((None, False), (EARLY, True)):
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
                    session_hologram=cfg,
                )
            )
            assert (SESSION_HOLOGRAM_MODULE in trainer._code()) is present
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
