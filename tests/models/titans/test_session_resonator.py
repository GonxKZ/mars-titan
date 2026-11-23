"""Resonador holográfico de descomposición latente transversal (`session_resonator_v1`, CSRF).

Las pruebas cubren el operador Walsh-Hadamard, la equivalencia exacta con una referencia fila a
fila y celda a celda (con contexto leave-one-out), la recuperación de factores ligados frente al
control sin iteración (T=0), la mitigación del colapso de diccionarios y sus diagnósticos, el
aislamiento causal estricto por mercado e instante, los gradientes finitos y vivos desde el paso 0
(también con mediana inicial nula), los controles negativos emparejados y la integración con el
predictor, el consumidor congelado y el entrenador cronológico.
"""

import hashlib
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
from mars_titan.models.titans.asset_groups import AssetGroupsConfig
from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
from mars_titan.models.titans.cross_sectional_rank_memory import CrossSectionalRankMemoryConfig
from mars_titan.models.titans.episodic_readout import apply_episodic_readout
from mars_titan.models.titans.fractal_shared_operator import FractalSharedOperatorConfig
from mars_titan.models.titans.session_hologram import SessionHologramConfig
from mars_titan.models.titans.session_resonator import (
    CELL_EPS,
    NO_ITERATION,
    NORM_EPS,
    RESONATOR,
    SESSION_RESONATOR,
    SHUFFLED_PEERS,
    SessionResonator,
    SessionResonatorConfig,
    parse_session_resonator,
    walsh_hadamard_matrix,
)
from mars_titan.models.titans.state import deferred_checks
from mars_titan.training import comparison_campaign
from mars_titan.training.search_cases import checked_predictor_changes

VARIANT = SessionResonatorConfig()
CONTROL_T0 = SessionResonatorConfig(iterations=0, mode=NO_ITERATION)
PLACEBO_SHUFFLED = SessionResonatorConfig(mode=SHUFFLED_PEERS)
SMALL = SessionResonatorConfig(codes_a=4, codes_b=4, iterations=2, mode=RESONATOR)
SMALL_T0 = SessionResonatorConfig(codes_a=4, codes_b=4, iterations=0, mode=NO_ITERATION)
SMALL_SHUFFLED = SessionResonatorConfig(codes_a=4, codes_b=4, iterations=2, mode=SHUFFLED_PEERS)

MARKETS = ("US", "CN", "US", "US", "CN", "US")
AT = (7,) * len(MARKETS)
RESONATOR_MODULE = "mars_titan.models.titans.session_resonator"


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def opened_resonator(config=SMALL, *, hidden=16, seed=3, dtype=torch.float64, gain=0.25):
    """Módulo con la ganancia vectorial abierta para observar la contribución transversal."""
    layer = SessionResonator(hidden, config, seed=seed, dtype=dtype)
    with torch.no_grad():
        layer.gain.fill_(gain)
    return layer


def encoded(rows=6, hidden=16, seed=9, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=dtype)


def _reference_unit_rms(dictionary, hidden):
    scale = math.sqrt(hidden)
    return dictionary * (
        scale / torch.sqrt(dictionary.square().sum(dim=-1, keepdim=True) + NORM_EPS)
    )


def _reference_bind(a, b, q):
    return ((a @ q) * (b @ q)) @ q


def _reference_competitive_softmax(logits):
    lme = torch.logsumexp(logits, dim=0, keepdim=True) - math.log(len(logits))
    return torch.softmax(logits - lme, dim=-1)


def reference_forward(layer, values, markets, instants):
    """Referencia explícita fila a fila, código a código y celda a celda con LOO directo."""
    rows, hidden = values.shape
    q = walsh_hadamard_matrix(hidden, dtype=values.dtype).to(values.device)
    a_norm = _reference_unit_rms(layer.dict_a, hidden)
    b_norm = _reference_unit_rms(layer.dict_b, hidden)
    normalized = layer.norm(values)
    mixed = torch.zeros_like(values)
    groups = {}
    for row, key in enumerate(zip(markets, instants, strict=True)):
        groups.setdefault(key, []).append(row)
    inv_sqrt_h = 1.0 / math.sqrt(hidden)
    for key, members in groups.items():
        n = len(members)
        if n < 2:
            continue
        group_vals = torch.stack([normalized[i] for i in members])
        mean = sum(group_vals[k] for k in range(n)) / n
        centered = torch.stack([group_vals[k] - mean for k in range(n)])
        if layer.config.mode == NO_ITERATION:
            logits_a = torch.stack([centered[k] @ a_norm.T * inv_sqrt_h for k in range(n)])
            logits_b = torch.stack([centered[k] @ b_norm.T * inv_sqrt_h for k in range(n)])
            p_a = _reference_competitive_softmax(logits_a)
            p_b = _reference_competitive_softmax(logits_b)
            a_hat = p_a @ a_norm
            b_hat = p_b @ b_norm
            recon = 0.5 * (a_hat + b_hat)
        else:
            b_hat = torch.stack([b_norm.mean(dim=0) for _ in range(n)])
            for _ in range(layer.config.iterations):
                unbound_a = torch.stack(
                    [_reference_bind(centered[k], b_hat[k], q) for k in range(n)]
                )
                p_a = _reference_competitive_softmax((unbound_a @ a_norm.T) * inv_sqrt_h)
                a_hat = p_a @ a_norm
                unbound_b = torch.stack(
                    [_reference_bind(centered[k], a_hat[k], q) for k in range(n)]
                )
                p_b = _reference_competitive_softmax((unbound_b @ b_norm.T) * inv_sqrt_h)
                b_hat = p_b @ b_norm
            recon = torch.stack([_reference_bind(a_hat[k], b_hat[k], q) for k in range(n)])
        m_cells = torch.stack(
            [(p_a[k].unsqueeze(1) * p_b[k].unsqueeze(0)).reshape(-1) for k in range(n)]
        )
        if layer.config.mode == SHUFFLED_PEERS:
            perm = layer._session_permutation(key, n).to(values.device)
            recon = recon[perm]
            m_cells = m_cells[perm]
        residual = centered - recon
        peer_list = []
        cells = m_cells.shape[1]
        for i in range(n):
            acc = torch.zeros(hidden, dtype=values.dtype, device=values.device)
            for c in range(cells):
                num = sum(m_cells[j, c] * centered[j] for j in range(n) if j != i)
                den = sum(m_cells[j, c] for j in range(n) if j != i) + CELL_EPS
                acc = acc + m_cells[i, c] * (num / den)
            peer_list.append(acc)
        peer_context = torch.stack(peer_list)
        joined = torch.cat((residual, peer_context), dim=-1)
        update = layer.out_proj(layer.out_norm(joined))
        for pos, row in enumerate(members):
            mixed[row] = update[pos]
    return values + layer.gain * mixed


def test_walsh_hadamard_matrix_is_orthogonal_symmetric_and_self_inverse_on_bipolar_spectrum():
    for size in (2, 8, 16, 64):
        q = walsh_hadamard_matrix(size, dtype=torch.float64)
        assert q.shape == (size, size)
        torch.testing.assert_close(q, q.T, rtol=0, atol=0)
        torch.testing.assert_close(
            q @ q, torch.eye(size, dtype=torch.float64), rtol=1e-13, atol=1e-13
        )
        torch.testing.assert_close(
            q.abs(), torch.full((size, size), 1.0 / math.sqrt(size), dtype=torch.float64)
        )
    layer = SessionResonator(16, SMALL, seed=5, dtype=torch.float64)
    q = walsh_hadamard_matrix(16, dtype=torch.float64)
    gen = torch.Generator().manual_seed(11)
    a = torch.randn(4, 16, generator=gen, dtype=torch.float64)
    b_spec = torch.where(torch.randn(4, 16, generator=gen, dtype=torch.float64) >= 0, 1.0, -1.0).to(
        torch.float64
    )
    b = b_spec @ q
    bound = layer.bind(a, b)
    recovered = layer.bind(bound, b)
    torch.testing.assert_close(recovered, a, rtol=1e-13, atol=1e-13)
    # Adjunto euclídeo exacto para vectores continuos arbitrarios: <B(a, b), x> == <a, B(x, b)>.
    b_cont = torch.randn(4, 16, generator=gen, dtype=torch.float64)
    x = torch.randn(4, 16, generator=gen, dtype=torch.float64)
    lhs = (layer.bind(a, b_cont) * x).sum(dim=-1)
    rhs = (a * layer.bind(x, b_cont)).sum(dim=-1)
    torch.testing.assert_close(lhs, rhs, rtol=1e-13, atol=1e-13)
    for invalid in (0, 1, 3, 6, 12, 48, 8192, True):
        with pytest.raises(ValueError):
            walsh_hadamard_matrix(invalid)


@pytest.mark.parametrize(
    "config",
    [
        SMALL,
        SMALL_T0,
        SMALL_SHUFFLED,
        SessionResonatorConfig(codes_a=3, codes_b=5, iterations=1, mode=RESONATOR),
    ],
    ids=["resonator", "no_iteration", "shuffled_peers", "asymmetric_t1"],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
def test_forward_matches_explicit_row_and_cell_leave_one_out_reference(config, dtype):
    layer = opened_resonator(config, hidden=16, seed=3, dtype=dtype)
    values = encoded(rows=7, hidden=16, seed=9, dtype=dtype)
    markets = ("US", "CN", "US", "US", "CN", "US", "JP")
    instants = (7, 7, 7, 8, 7, 8, 7)
    actual = layer(values, markets, instants)
    expected = reference_forward(layer, values, markets, instants)
    tol = dict(rtol=1e-12, atol=1e-12) if dtype == torch.float64 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual, expected, **tol)
    # La fila solitaria (JP, 7) queda exactamente intacta incluso con gain y out_norm.bias != 0.
    with torch.no_grad():
        layer.out_norm.bias.fill_(0.5)
    after_bias = layer(values, markets, instants)
    assert torch.equal(after_bias[6], values[6])
    # Comprobar también la ruta de una sola sesión (len(sessions) == 1) y grupos contiguos.
    single = encoded(rows=5, hidden=16, seed=19, dtype=dtype)
    torch.testing.assert_close(
        layer(single, ("US",) * 5, (3,) * 5),
        reference_forward(layer, single, ("US",) * 5, (3,) * 5),
        **tol,
    )
    contig_m = ("US", "US", "US", "CN", "CN")
    contig_t = (3, 3, 3, 3, 3)
    torch.testing.assert_close(
        layer(single, contig_m, contig_t),
        reference_forward(layer, single, contig_m, contig_t),
        **tol,
    )
    # Sesión solitaria única (N=1): identidad exacta incluso con gain != 0 y out_norm.bias != 0.
    lone = encoded(rows=1, hidden=16, seed=23, dtype=dtype)
    assert torch.equal(layer(lone, ("US",), (3,)), lone)


def test_dictionaries_start_orthogonal_with_unit_rms_and_handle_overcomplete_and_singleton():
    layer = SessionResonator(16, SMALL, seed=7, dtype=torch.float64)
    for atoms in (layer.dict_a, layer.dict_b):
        gram = atoms @ atoms.T
        torch.testing.assert_close(
            gram, torch.eye(4, dtype=torch.float64) * 16.0, rtol=1e-12, atol=1e-12
        )
    # Diccionario sobrecompleto (codes > hidden_size) también se inicializa con norma √H.
    over = SessionResonator(
        4, SessionResonatorConfig(codes_a=6, codes_b=6, iterations=2), seed=7, dtype=torch.float64
    )
    norms = over.dict_a.norm(dim=-1)
    torch.testing.assert_close(
        norms, torch.full((6,), 2.0, dtype=torch.float64), rtol=1e-5, atol=1e-5
    )
    # Diagnóstico en sesión solitaria devuelve métricas finitas uniformes.
    lone = encoded(rows=1, hidden=16, seed=5, dtype=torch.float64)
    diag = layer.diagnostics(lone, ("US",), (1,))
    assert diag["perplexity_a"] == pytest.approx(4.0)
    assert diag["residual_energy_ratio"] == pytest.approx(1.0)


def test_resonator_recovers_bound_composites_much_better_than_no_iteration_control():
    hidden, k1, k2 = 64, 4, 4
    q = walsh_hadamard_matrix(hidden, dtype=torch.float64)
    gen = torch.Generator().manual_seed(123)
    # Diccionarios bipolares ortogonales en el dominio Walsh-Hadamard: |Q a|_k = 1.
    a_spec = torch.where(
        torch.randn(k1, hidden, generator=gen, dtype=torch.float64) >= 0, 1.0, -1.0
    ).to(torch.float64)
    b_spec = torch.where(
        torch.randn(k2, hidden, generator=gen, dtype=torch.float64) >= 0, 1.0, -1.0
    ).to(torch.float64)
    a_dict = a_spec @ q
    b_dict = b_spec @ q
    # Sesión balanceada con todos los pares (u, v) repetidos.
    pairs = [(u, v) for u in range(k1) for v in range(k2)] * 3
    rows = len(pairs)
    composites = torch.stack([_reference_bind(a_dict[u], b_dict[v], q) for u, v in pairs])
    composites = composites - composites.mean(dim=0, keepdim=True)
    markets = ("US",) * rows
    instants = (10,) * rows

    def make_layer(mode, iterations):
        cfg = SessionResonatorConfig(codes_a=k1, codes_b=k2, iterations=iterations, mode=mode)
        layer = SessionResonator(hidden, cfg, seed=1, dtype=torch.float64)
        with torch.no_grad():
            layer.dict_a.copy_(a_dict)
            layer.dict_b.copy_(b_dict)
            layer.norm.weight.fill_(1.0)
            layer.norm.bias.zero_()
            layer.gain.fill_(1.0)
        return layer

    res_t2 = make_layer(RESONATOR, 2)
    res_t1 = make_layer(RESONATOR, 1)
    ctrl_t0 = make_layer(NO_ITERATION, 0)

    diag_t2 = res_t2.diagnostics(composites, markets, instants)
    diag_t1 = res_t1.diagnostics(composites, markets, instants)
    diag_t0 = ctrl_t0.diagnostics(composites, markets, instants)

    # El resonador iterativo reduce fuertemente el residuo de factorización frente a T=0 y T=1.
    assert diag_t2["residual_energy_ratio"] < 0.25
    assert diag_t2["residual_energy_ratio"] < diag_t1["residual_energy_ratio"]
    assert diag_t2["residual_energy_ratio"] < diag_t0["residual_energy_ratio"] - 0.35


def test_competitive_softmax_prevents_dictionary_collapse_and_diagnostics_detect_it():
    hidden, k1, k2 = 32, 8, 8
    layer = opened_resonator(
        SessionResonatorConfig(codes_a=k1, codes_b=k2, iterations=2, mode=RESONATOR),
        hidden=hidden,
        seed=7,
        dtype=torch.float64,
    )
    # Invarianza exacta de _competitive_softmax ante sesgos de popularidad por código.
    gen = torch.Generator().manual_seed(17)
    logits = torch.randn(32, k1, generator=gen, dtype=torch.float64)
    popularity_bias = torch.tensor(
        [12.0, -5.0, 0.0, 4.0, -8.0, 1.0, 3.0, -2.0], dtype=torch.float64
    )
    p_base = layer._competitive_softmax(logits)
    p_biased = layer._competitive_softmax(logits + popularity_bias.unsqueeze(0))
    torch.testing.assert_close(p_biased, p_base, rtol=1e-12, atol=1e-12)

    # Sesión fuertemente sesgada (90% alineada con el modo ligado (0, 0) + perturbación pequeña):
    # una softmax ingenua colapsa casi toda la masa en un código; CSRF mantiene uso distribuido.
    q = walsh_hadamard_matrix(hidden, dtype=torch.float64)
    a_norm = _reference_unit_rms(layer.dict_a.detach(), hidden)
    b_norm = _reference_unit_rms(layer.dict_b.detach(), hidden)
    dominant = _reference_bind(a_norm[0], b_norm[0], q)
    n = 80
    skewed = torch.randn(n, hidden, generator=gen, dtype=torch.float64) * 0.25
    skewed[:72] += 4.0 * dominant
    skewed[72:] -= 4.0 * dominant
    markets, instants = ("US",) * n, (5,) * n

    # Verificación del colapso sin competición en logits sesgados:
    centered = layer.norm(skewed)
    centered = centered - centered.mean(dim=0, keepdim=True)
    raw_logits = (centered @ a_norm.T) / math.sqrt(hidden)
    raw_logits[:, 0] += 8.0
    naive_usage = torch.softmax(raw_logits, dim=-1).mean(dim=0)
    naive_perp = math.exp(-(naive_usage * naive_usage.clamp_min(1e-12).log()).sum().item())
    comp_usage = layer._competitive_softmax(raw_logits).mean(dim=0)
    comp_perp = math.exp(-(comp_usage * comp_usage.clamp_min(1e-12).log()).sum().item())
    assert naive_usage.max().item() > 0.85 and naive_perp < 2.0
    assert comp_usage.max().item() < 0.35 and comp_perp > 5.0

    # Diagnóstico integrado del módulo en la sesión sesgada.
    diag = layer.diagnostics(skewed, markets, instants)
    expected_keys = {
        "perplexity_a",
        "perplexity_b",
        "perplexity_cells",
        "normalized_entropy_a",
        "normalized_entropy_b",
        "max_code_share_a",
        "max_code_share_b",
        "residual_energy_ratio",
        "gain_l2",
    }
    assert set(diag) == expected_keys
    assert diag["perplexity_a"] > 4.0 and diag["perplexity_b"] > 4.0
    assert diag["max_code_share_a"] < 0.45 and diag["max_code_share_b"] < 0.45
    assert 0.0 < diag["normalized_entropy_a"] <= 1.0 + 1e-12
    assert 0.0 < diag["normalized_entropy_b"] <= 1.0 + 1e-12
    assert diag["gain_l2"] > 0.0

    # Si se fuerza manualmente el colapso (todos los átomos idénticos), el diagnóstico refleja
    # falta de diferenciación residual (perplejidad máxima uniforme pero residuo alto) y si
    # además un único átomo captura masa en modo degenerado, las métricas siguen siendo finitas.
    with torch.no_grad():
        layer.dict_a.copy_(layer.dict_a[:1].expand_as(layer.dict_a))
    collapsed_diag = layer.diagnostics(skewed, markets, instants)
    assert all(math.isfinite(v) for v in collapsed_diag.values())
    assert collapsed_diag["perplexity_a"] == pytest.approx(float(k1), rel=1e-6)


@pytest.mark.parametrize("config", [SMALL, SMALL_T0, SMALL_SHUFFLED])
def test_strict_causal_isolation_across_markets_and_instants(config):
    layer = opened_resonator(config)
    values = encoded(rows=6, hidden=16, seed=9)
    base = layer(values, MARKETS, AT)
    us = [0, 2, 3, 5]
    cn = [1, 4]
    # Perturbar filas de otro mercado (CN) no cambia ni un bit de las filas de US.
    other_market = values.clone()
    other_market[cn] += 7.5
    assert torch.equal(layer(other_market, MARKETS, AT)[us], base[us])
    # Perturbar filas del futuro (instante 8) no cambia ni un bit del instante presente (7).
    instants = (7, 7, 7, 8, 7, 8)
    split = layer(values, MARKETS, instants)
    future_changed = values.clone()
    future_changed[[3, 5]] += 7.5
    assert torch.equal(layer(future_changed, MARKETS, instants)[[0, 1, 2, 4]], split[[0, 1, 2, 4]])
    # Perturbar un par de la misma sesión (mercado US, instante 7) SÍ cambia a su par US.
    same_session = values.clone()
    same_session[0] += 7.5
    assert not torch.equal(layer(same_session, MARKETS, AT)[2], base[2])


@pytest.mark.parametrize("config", [SMALL, SMALL_T0])
def test_permuting_rows_permutes_outputs_equivariantly(config):
    layer = opened_resonator(config)
    values = encoded(rows=6, hidden=16, seed=9)
    order = [3, 0, 5, 1, 4, 2]
    expected = layer(values, MARKETS, AT)[order]
    actual = layer(values[order], tuple(MARKETS[i] for i in order), AT)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_controls_share_exact_initial_parameters_and_do_not_touch_global_rng():
    torch.random.default_generator.manual_seed(11)
    before = torch.random.get_rng_state()
    res = SessionResonator(16, SMALL, seed=3, dtype=torch.float64)
    assert torch.equal(torch.random.get_rng_state(), before)
    t0 = SessionResonator(16, SMALL_T0, seed=3, dtype=torch.float64)
    shuf = SessionResonator(16, SMALL_SHUFFLED, seed=3, dtype=torch.float64)
    for other in (t0, shuf):
        assert list(res.state_dict()) == list(other.state_dict())
        for name, tensor in res.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, other.state_dict()[name]), name
    # El placebo barajado difiere del resonador en filas distintas y cambia con el instante.
    values = encoded(rows=6, hidden=16, seed=9)
    with torch.no_grad():
        res.gain.fill_(0.25)
        t0.gain.fill_(0.25)
        shuf.gain.fill_(0.25)
    out_res = res(values, MARKETS, AT)
    out_t0 = t0(values, MARKETS, AT)
    out_shuf = shuf(values, MARKETS, AT)
    assert not torch.allclose(out_res, out_t0)
    assert not torch.allclose(out_res, out_shuf)
    assert not torch.equal(
        shuf._session_permutation(("US", 7), 6), shuf._session_permutation(("US", 8), 6)
    )
    # Para N=2 garantiza no ser la permutación identidad.
    for instant in range(20):
        perm = shuf._session_permutation(("US", instant), 2)
        assert perm.tolist() == [1, 0]


def test_gradients_isolate_markets_and_instants_and_reach_all_parameters():
    values = encoded(rows=6, hidden=16, seed=9).requires_grad_()
    instants = (7, 7, 7, 8, 7, 8)
    for config in (SMALL, SMALL_T0, SMALL_SHUFFLED):
        layer = opened_resonator(config)
        out = layer(values, MARKETS, instants)
        (grad,) = torch.autograd.grad(out[0].sum(), values)
        # Fila 0 es (US, 7): solo fila 2 comparte sesión; CN (1, 4) y futuro (3, 5) reciben 0.
        assert torch.count_nonzero(grad[[1, 3, 4, 5]]) == 0
        assert torch.count_nonzero(grad[2]) > 0
        # Al retropropagar con gain abierta, todos los parámetros reciben gradiente finito no nulo.
        layer.zero_grad()
        out = layer(values.detach(), MARKETS, instants)
        out.square().sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert torch.count_nonzero(param.grad) > 0, (config.mode, name)


def test_determinism_and_finite_gradients_under_constant_inputs_in_fp32_and_fp64():
    for dtype in (torch.float32, torch.float64):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            layer = opened_resonator(SMALL, hidden=16, seed=5, dtype=dtype)
            values = encoded(rows=8, hidden=16, seed=13, dtype=dtype)
            markets = ("US", "CN", "US", "US", "CN", "US", "US", "CN")
            instants = (1, 1, 1, 1, 1, 2, 2, 1)
            first = layer(values, markets, instants)
            second = layer(values, markets, instants)
            assert torch.equal(first, second)
            # Entradas constantes en una sesión (centered == 0): salida y gradientes finitos.
            const = torch.full((6, 16), 3.0, dtype=dtype, requires_grad=True)
            out_const = layer(const, MARKETS, AT)
            assert torch.isfinite(out_const).all()
            out_const.sum().backward()
            assert torch.isfinite(const.grad).all()
            for name, param in layer.named_parameters():
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
        finally:
            torch.use_deterministic_algorithms(previous)


def test_disabled_field_keeps_identity_fingerprint_parameters_predictions_and_gradients():
    base, batch = setup()
    explicit = model(session_resonator=None)
    assert explicit.config.identity() == base.config.identity()
    assert explicit._config_id() == base._config_id()
    assert explicit._parameter_id == base._parameter_id
    assert explicit.config.inputs.fingerprint() == base.config.inputs.fingerprint()
    assert "session_resonator" not in base.config.identity()
    assert base.session_resonator is None and not base.session_level
    b_state = base.initial_state(batch.flow_ids, differentiable=True)
    e_state = explicit.initial_state(batch.flow_ids, differentiable=True)
    expected = base.prepare(batch, b_state, differentiable=True)
    actual = explicit.prepare(batch, e_state, differentiable=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    expected.point_predictions.square().sum().backward()
    actual.point_predictions.square().sum().backward()
    for (name, p_base), (_, p_exp) in zip(
        base.named_parameters(), explicit.named_parameters(), strict=True
    ):
        assert torch.equal(p_base, p_exp), name
        if p_base.grad is not None:
            torch.testing.assert_close(p_exp.grad, p_base.grad, rtol=0, atol=0)


@pytest.mark.parametrize("config", [VARIANT, CONTROL_T0, PLACEBO_SHUFFLED])
def test_variant_and_controls_start_bitwise_equal_to_reference_and_receive_step0_gradient(config):
    plain, batch = build(technical_branch=asdict(DENSE))
    opened(plain.technical_branch)
    plain._seal_parameters()
    candidate, _ = build(technical_branch=asdict(DENSE), session_resonator=asdict(config))
    opened(candidate.technical_branch)
    candidate._seal_parameters()
    ref_params = dict(plain.named_parameters())
    cand_params = dict(candidate.named_parameters())
    assert set(cand_params) - set(ref_params) == {
        "session_resonator.norm.weight",
        "session_resonator.norm.bias",
        "session_resonator.dict_a",
        "session_resonator.dict_b",
        "session_resonator.out_norm.weight",
        "session_resonator.out_norm.bias",
        "session_resonator.out_proj.weight",
        "session_resonator.gain",
    }
    for name, value in ref_params.items():
        assert torch.equal(cand_params[name], value), name
    assert torch.count_nonzero(candidate.session_resonator.gain) == 0
    assert torch.count_nonzero(candidate.session_resonator.out_proj.weight) > 0
    expected = emit(plain, batch)
    actual = emit(candidate, batch, session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.quantiles, expected.quantiles, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)

    # Paso 0 con QuantileHead + zero_median=True (mediana constante 0) + pinball:
    # todos los gradientes son finitos y session_resonator.gain recibe gradiente no nulo en t=0.
    state = candidate.initial_state(batch.flow_ids, differentiable=True)
    out = candidate.prepare(batch, state, differentiable=True, session_complete=True)
    assert torch.count_nonzero(out.point_predictions) == 0
    target = torch.linspace(-0.02, 0.02, len(batch.flow_ids), dtype=out.quantiles.dtype)
    loss = pinball_loss(out.quantiles, target)
    loss.backward()
    for name, param in candidate.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    assert torch.count_nonzero(candidate.session_resonator.gain.grad) > 0
    assert torch.count_nonzero(candidate.session_resonator.dict_a.grad) == 0


def test_constant_zero_scalar_head_produces_finite_gradients():
    from mars_titan.models.quantile_head import MEDIAN_GAIN
    from mars_titan.models.titans.session_ic_objective import SessionICTerm

    scalar_model, batch = build(head="scalar", session_resonator=asdict(SMALL))
    state = scalar_model.initial_state(batch.flow_ids, differentiable=True)
    out = scalar_model.prepare(batch, state, differentiable=True, session_complete=True)
    assert torch.count_nonzero(out.point_predictions) == 0
    target = torch.linspace(-0.02, 0.02, len(batch.flow_ids), dtype=out.point_predictions.dtype)
    (out.point_predictions - target).abs().mean().backward()
    for name, param in scalar_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name

    # Receta exacta v11 (QuantileHead + zero_median + median_gain + pinball + session_ic_term):
    # con predicción puntual constante cero, todos los gradientes son finitos y gain aprende en t=0.
    v11_model, _ = build(
        technical_branch=asdict(DENSE),
        median_gain=MEDIAN_GAIN,
        session_resonator=asdict(SMALL),
    )
    opened(v11_model.technical_branch)
    v11_model._seal_parameters()
    six = session_batch(6)
    state = v11_model.initial_state(six.flow_ids, differentiable=True)
    out = v11_model.prepare(six, state, differentiable=True, session_complete=True)
    assert torch.count_nonzero(out.point_predictions) == 0
    target = torch.linspace(-0.02, 0.02, len(six.flow_ids), dtype=out.quantiles.dtype)
    sessions = tuple(
        (flow.split("/", 1)[0], at)
        for flow, at in zip(six.flow_ids, six.prediction_at, strict=True)
    )
    ic_term = SessionICTerm(weight=3.0, minimum_rows=3)
    loss = pinball_loss(out.quantiles, target) + ic_term.term(
        out.point_predictions, target, sessions
    )
    loss.backward()
    for name, param in v11_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    assert torch.count_nonzero(v11_model.session_resonator.gain.grad) > 0


def test_predictor_places_csrf_after_technical_branch_and_does_not_touch_mac_memory():
    plain, batch = build(technical_branch=asdict(DENSE))
    opened(plain.technical_branch)
    plain._seal_parameters()
    cand, _ = build(technical_branch=asdict(DENSE), session_resonator=asdict(SMALL))
    opened(cand.technical_branch)
    with torch.no_grad():
        cand.session_resonator.gain.fill_(0.2)
    cand._seal_parameters()
    ref_out = emit(plain, batch)
    cand_out = emit(cand, batch, session_complete=True)
    markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
    expected_ws = cand.session_resonator(ref_out.working_state, markets, batch.prediction_at)
    torch.testing.assert_close(cand_out.working_state, expected_ws, rtol=0, atol=0)
    torch.testing.assert_close(
        cand_out.next_state.mac.memory.weights[0],
        ref_out.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    # Con gain abierta, el gradiente cruza CSRF hacia dict_a, dict_b y technical_branch.
    state = cand.initial_state(batch.flow_ids, differentiable=True)
    emitted = cand.prepare(batch, state, differentiable=True, session_complete=True)
    emitted.working_state.square().sum().backward()
    assert torch.count_nonzero(cand.session_resonator.dict_a.grad) > 0
    assert torch.count_nonzero(cand.session_resonator.dict_b.grad) > 0
    assert torch.count_nonzero(cand.technical_branch.outer.weight.grad) > 0


def test_csrf_requires_complete_session_and_rejects_incompatible_mixers():
    batch = session_batch()
    predictor = model(session_resonator=SMALL)
    assert predictor.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)
    with pytest.raises(ValueError):
        model(session_resonator=SMALL, cross_asset_attention=CrossAssetAttentionConfig())
    with pytest.raises(ValueError):
        model(session_resonator=SMALL, asset_groups=AssetGroupsConfig())
    with pytest.raises(ValueError):
        model(session_resonator=SMALL, fractal_shared_operator=FractalSharedOperatorConfig())
    with pytest.raises(ValueError, match="operaciones por sesión"):
        model(session_resonator=SMALL, auxiliary_horizons=AuxiliaryHorizons())


def test_state_dict_round_trip_and_distinct_mode_identities():
    batch = session_batch()
    source = model(session_resonator=SMALL)
    with torch.no_grad():
        source.session_resonator.gain.fill_(0.15)
    source._seal_parameters()
    target = model(session_resonator=SMALL)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    for other in (SMALL_T0, SMALL_SHUFFLED, replace(SMALL, codes_a=6)):
        with pytest.raises(ValueError):
            model(session_resonator=other).load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    "bad",
    [
        dict(codes_a=1, codes_b=8, iterations=2, mode=RESONATOR),
        dict(codes_a=33, codes_b=8, iterations=2, mode=RESONATOR),
        dict(codes_a=True, codes_b=8, iterations=2, mode=RESONATOR),
        dict(codes_a=8, codes_b=0, iterations=2, mode=RESONATOR),
        dict(codes_a=8, codes_b=8, iterations=0, mode=RESONATOR),
        dict(codes_a=8, codes_b=8, iterations=5, mode=RESONATOR),
        dict(codes_a=8, codes_b=8, iterations=2, mode=NO_ITERATION),
        dict(codes_a=8, codes_b=8, iterations=0, mode=SHUFFLED_PEERS),
        dict(codes_a=8, codes_b=8, iterations=2, mode="unknown"),
        dict(codes_a=8, codes_b=8, mode=RESONATOR),
        "session_resonator_v1",
    ],
)
def test_invalid_declarations_and_runtime_inputs_are_rejected(bad):
    base, _ = setup()
    with pytest.raises(ValueError):
        replace(base.config, session_resonator=bad)
    with pytest.raises(ValueError):
        parse_session_resonator(bad)


def test_construction_and_forward_validation_guards():
    with pytest.raises(ValueError):
        SessionResonator(12, SMALL, seed=1)
    with pytest.raises(ValueError):
        SessionResonator(16, asdict(SMALL), seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        SessionResonator(16, SMALL, seed=1, dtype=torch.bfloat16)
    layer = opened_resonator(SMALL, hidden=16, dtype=torch.float32)
    valid = encoded(rows=6, hidden=16, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer(valid, MARKETS, AT)
    for args in (
        (valid[:, :8], MARKETS, AT),
        (valid.double(), MARKETS, AT),
        (valid[:0], (), ()),
        (valid, MARKETS[:-1], AT),
        (valid, MARKETS, AT[:-1]),
    ):
        with pytest.raises(ValueError):
            layer(*args)


def test_frozen_consumer_and_trainer_register_csrf_only_when_active(monkeypatch, tmp_path):
    from mars_titan.models.titans import frozen_financial as ff
    from mars_titan.training.dust.financial import FinancialDustAdapter
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    base_engine, _ = frozen("mac_online")
    assert RESONATOR_MODULE not in ff._SOURCE_MODULES
    assert RESONATOR_MODULE not in base_engine.identity()["implementation"]
    assert ff.declared_sources(base_engine.predictor.config) == ()

    predictor = model(session_resonator=SMALL)
    with torch.no_grad():
        predictor.session_resonator.gain.fill_(0.2)
    predictor._seal_parameters()
    predictor.eval().requires_grad_(False)
    engine = consumer(predictor)
    assert ff.declared_sources(predictor.config) == (RESONATOR_MODULE,)
    assert RESONATOR_MODULE in engine.identity()["implementation"]
    assert (
        engine.identity()["modules"]["predictor/session_resonator"]["settings"]["session_resonator"]
        == SMALL.identity()
    )
    engine.verify()

    batch = session_batch(6, markets=list(MARKETS))
    state = predictor.initial_state(batch.flow_ids)
    with pytest.raises(ValueError, match="sesión completa"):
        engine.prepare(batch, state, context_id="ctx", session_complete=False)
    expected = predictor.prepare(batch, state, session_complete=True)
    actual = engine.prepare(batch, state, context_id="ctx", session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)

    # Cambio de fichero fuente invalida verificación fuerte del consumidor con CSRF, no del base.
    original = ff.importlib.import_module
    changed = tmp_path / "different_resonator.py"
    changed.write_text("different\n")
    monkeypatch.setattr(
        ff.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(__file__=str(changed)) if name == RESONATOR_MODULE else original(name)
        ),
    )
    base_engine.verify()
    with pytest.raises(ValueError, match="código"):
        engine.verify()

    # Entrenador cronológico: fuente solo cuando session_resonator está activo.
    trainer = object.__new__(ChronologicalTrainer)
    trainer.recipe = ChronologicalRecipe()
    for cfg, present in ((None, False), (SMALL, True)):
        trainer.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=None,
                asset_groups=None,
                session_resonator=cfg,
                holographic_network=None,
                axial_time_region=None,
                mature_reliability=None,
                fractal_shared_operator=None,
            )
        )
        assert (RESONATOR_MODULE in trainer._code()) is present

    # Dust rechaza el predictor con CSRF.
    with pytest.raises(ValueError, match="sin ampliaciones"):
        FinancialDustAdapter(model(variant="transformer_direct", session_resonator=SMALL))


def test_chronological_trainer_replays_whole_sessions_and_reaches_csrf_gain(tmp_path):
    from tests.training.test_financial_run_session_objective import (
        predictor as run_predictor,
    )
    from tests.training.test_financial_run_session_objective import (
        records,
        streams,
    )
    from tests.training.test_financial_run_session_objective import (
        trainer as run_trainer,
    )

    shared = streams(tmp_path / "corpus")
    csrf_model = run_predictor(
        shared,
        session_rank_inputs=None,
        session_ic_objective=None,
        session_resonator=SMALL,
    )
    direct = run_trainer(shared, tmp_path / "direct", model=csrf_model)
    report = direct.run()
    assert report["status"] == "completed"
    grads = records(direct)
    assert len(grads) > 2
    assert grads[0]["session_resonator.gain"].abs().sum() > 0
    assert grads[0]["session_resonator.dict_a"].abs().sum() == 0
    replayed = run_trainer(
        shared,
        tmp_path / "replayed",
        model=run_predictor(
            shared,
            session_rank_inputs=None,
            session_ic_objective=None,
            session_resonator=SMALL,
        ),
        accumulation_rows=8,
    )
    replayed.run()
    for first, second in zip(grads, records(replayed), strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("tamper", ["config_mismatch", "missing_module", "extra"])
def test_frozen_consumer_rejects_inconsistent_session_resonator(tamper):
    if tamper == "extra":
        predictor, _ = setup("mac_online")
        predictor.session_resonator = SessionResonator(
            predictor.config.hidden_size, SMALL, seed=predictor.config.seed, dtype=torch.float64
        )
        predictor._seal_parameters()
        predictor.eval().requires_grad_(False)
    else:
        predictor = model(session_resonator=SMALL)
        predictor.eval().requires_grad_(False)
        if tamper == "config_mismatch":
            predictor.session_resonator.config = SMALL_T0
        else:
            predictor.session_resonator = None
    with pytest.raises(ValueError, match="resonador de sesión"):
        consumer(predictor)


def test_us_campaign_v11_screen1_single_variable_declarations_build_cleanly():
    root = Path(__file__).resolve().parents[3]
    campaign = json.loads((root / "configs/experiments/us-campaign-v11-screen1.json").read_text())
    core_recipe = json.loads((root / campaign["recipes"]["core"]).read_text())
    assert campaign["models"]["titans_mac"]["recipe"] == "core"

    arms = {
        "session_resonator": dict(
            session_resonator=dict(codes_a=8, codes_b=8, iterations=2, mode="resonator")
        ),
        "session_resonator_t0": dict(
            session_resonator=dict(codes_a=8, codes_b=8, iterations=0, mode="no_iteration")
        ),
        "session_resonator_shuffled": dict(
            session_resonator=dict(codes_a=8, codes_b=8, iterations=2, mode="shuffled_peers")
        ),
    }
    base, _ = build(technical_branch=asdict(DENSE))
    ref_params = dict(base.named_parameters())
    identities = {canonical(base.config.identity())}
    for name, changes in arms.items():
        assert checked_predictor_changes(changes) == changes
        # Validar que el diccionario de cambios se compone sobre la receta core de v11.
        merged = dict(core_recipe["predictor"]) | changes
        assert merged["technical_branch"]["columns"] == 46
        cand, _ = build(technical_branch=asdict(DENSE), **changes)
        identities.add(canonical(cand.config.identity()))
        for key, value in ref_params.items():
            assert torch.equal(dict(cand.named_parameters())[key], value), (name, key)
        assert cand.config.identity()["session_resonator"]["kind"] == SESSION_RESONATOR
    assert len(identities) == 4


def _tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _autograd_node_count(tensor):
    visited = set()
    stack = [tensor.grad_fn] if tensor.grad_fn is not None else []
    while stack:
        node = stack.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for nxt, _ in node.next_functions:
            if nxt is not None:
                stack.append(nxt)
    return len(visited)


def test_single_option_numerical_parity_matches_pinned_previous_patch_values():
    """Con solo session_resonator activo, esta base iguala bit a bit al parche anterior."""
    expected = {
        RESONATOR: dict(
            iters=2,
            param_id="26c7d9390265194467517e448a61b366e83bd00b2cb56feb71ba119194933faf",
            ws_sha="f3e38c1b2e0922f9b8c483c346115bd75b80e7b2c17a4c0386ec838decc00a30",
            q_sha="498b59dc0aa15f28b5381f01931eec09fbc30608f06a9410d666439f221373e0",
            ws_head=[
                -0.11177843376075092,
                0.13861633940776752,
                -0.30106439848317146,
                -0.016618169342512923,
            ],
            gain_grad_sha="f881642a484212722c5c3c51bd92c1bc2fe077143e70a7928f0949f6e4b5d9a5",
            dict_a_grad_sha="48993fda76a6c5ab773e548a5537eb42def015f41e423ad41e8c1af6f4a591d2",
        ),
        NO_ITERATION: dict(
            iters=0,
            param_id="26c7d9390265194467517e448a61b366e83bd00b2cb56feb71ba119194933faf",
            ws_sha="dd88192cdb6e2983eccf24ab67ebfa5fac2439636a385be1855ed22faace9300",
            q_sha="a2e4f0a91ca1cfd7d3c4758cbe233013ef893b4af00f828d2c44d3c6ef00731d",
            ws_head=[
                -0.11864634540952818,
                0.024913596069977995,
                -0.36319190229292025,
                0.14036983880900128,
            ],
            gain_grad_sha="b6aad296591a82ce3add4d0e14bb1bd4291895f168f188cffa2d3f375c845899",
            dict_a_grad_sha="7783ab13d6a6f4751c340d6eaa2d6a756bd2283b798346d7ece1bcc0a0a11297",
        ),
        SHUFFLED_PEERS: dict(
            iters=2,
            param_id="26c7d9390265194467517e448a61b366e83bd00b2cb56feb71ba119194933faf",
            ws_sha="03ccf7cdc0d2e904cd6e67de20da0409ecd19cbfe68a061a054fb65795227838",
            q_sha="2304938f8a3979f2851d57f6668df1fc0d1ec1497e2b69a6660bf5762726ee03",
            ws_head=[
                -0.056106620654085095,
                -0.020202305767845707,
                -0.2819464825088891,
                0.0028651002123463998,
            ],
            gain_grad_sha="8daef82e9ca2c13f88154857f547a27b380fc5d01d698524bf2c55b3d13d2cfa",
            dict_a_grad_sha="1d10d345f8339bd6b2bd7ee95b5c29eb92a500c1f9df4d185d863dd6238f035c",
        ),
    }
    for mode, pinned in expected.items():
        cfg = SessionResonatorConfig(codes_a=4, codes_b=4, iterations=pinned["iters"], mode=mode)
        pred, batch = build(technical_branch=asdict(DENSE), session_resonator=asdict(cfg))
        opened(pred.technical_branch)
        with torch.no_grad():
            pred.session_resonator.gain.fill_(0.25)
        pred._seal_parameters()
        assert pred._parameter_id == pinned["param_id"]
        out = emit(pred, batch, session_complete=True)
        assert _tensor_sha256(out.working_state) == pinned["ws_sha"]
        assert _tensor_sha256(out.quantiles) == pinned["q_sha"]
        torch.testing.assert_close(
            out.working_state[0, :4],
            torch.tensor(pinned["ws_head"], dtype=out.working_state.dtype),
            rtol=0,
            atol=0,
        )
        state = pred.initial_state(batch.flow_ids, differentiable=True)
        emitted = pred.prepare(batch, state, differentiable=True, session_complete=True)
        target = torch.linspace(-0.02, 0.02, len(batch.flow_ids), dtype=emitted.quantiles.dtype)
        pinball_loss(emitted.quantiles, target).backward()
        assert _tensor_sha256(pred.session_resonator.gain.grad) == pinned["gain_grad_sha"]
        assert _tensor_sha256(pred.session_resonator.dict_a.grad) == pinned["dict_a_grad_sha"]


def test_coexistence_with_session_hologram_and_crrm_respects_order_before_head():
    """Orden antes de la cabeza: rama técnica -> holograma -> resonador -> CRRM -> cabeza."""
    from mars_titan.models.titans.frozen_financial import declared_sources

    plain, _ = build(technical_branch=asdict(DENSE))
    opened(plain.technical_branch)
    plain._seal_parameters()
    assert "session_hologram" not in plain.config.identity()
    assert "session_resonator" not in plain.config.identity()
    assert "cross_sectional_rank_memory" not in plain.config.identity()

    only_csrf, _ = build(technical_branch=asdict(DENSE), session_resonator=asdict(SMALL))
    assert only_csrf.session_hologram is None
    assert only_csrf.cross_sectional_rank_memory is None
    assert "session_hologram" not in only_csrf.config.identity()
    assert "cross_sectional_rank_memory" not in only_csrf.config.identity()
    assert declared_sources(only_csrf.config) == (
        "mars_titan.models.titans.technical_branch",
        RESONATOR_MODULE,
        "mars_titan.models.learning_scale",
    )

    hsi_cfg = SessionHologramConfig(bands=2, phase_dim=4, position="after_technical")
    crrm_cfg = CrossSectionalRankMemoryConfig(dim=8, time_constants=(5,), minimum_rows=3)
    combined, _ = build(
        technical_branch=asdict(DENSE),
        session_hologram=asdict(hsi_cfg),
        session_resonator=asdict(SMALL),
        cross_sectional_rank_memory=asdict(crrm_cfg),
    )
    opened(combined.technical_branch)
    with torch.no_grad():
        combined.session_hologram.gate.fill_(0.2)
        combined.session_resonator.gain.fill_(0.25)
        combined.cross_sectional_rank_memory.out.weight.fill_(0.15)
    combined._seal_parameters()

    batch = session_batch(6, markets=list(MARKETS))
    markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
    # Madurar una sesión previa en CRRM para que su lectura produzca un residuo no nulo.
    rm_state = combined.initial_rank_memory()
    past_out = combined.prepare(
        batch,
        combined.initial_state(batch.flow_ids),
        session_complete=True,
        rank_memory=rm_state,
    )
    staged = combined.stage_rank_memory(
        rm_state,
        batch.flow_ids,
        batch.prediction_at,
        past_out.rank_memory_keys,
        past_out.point_predictions,
    )
    outcomes = torch.linspace(-0.03, 0.03, len(batch.flow_ids)).tolist()
    matured = combined.settle_rank_memory(
        staged,
        [
            (flow, at, val)
            for flow, at, val in zip(batch.flow_ids, batch.prediction_at, outcomes, strict=True)
        ],
        differentiable=False,
        train=True,
    )

    ref_out = plain.prepare(batch, plain.initial_state(batch.flow_ids), session_complete=True)
    got_out = combined.prepare(
        batch,
        combined.initial_state(batch.flow_ids),
        session_complete=True,
        rank_memory=matured,
    )

    ws0 = ref_out.working_state
    ws1 = combined.session_hologram(ws0, markets, batch.prediction_at)
    ws2 = combined.session_resonator(ws1, markets, batch.prediction_at)
    crrm_keys = combined.cross_sectional_rank_memory.keys(ws2, markets, batch.prediction_at)
    ws3 = ws2 + combined.cross_sectional_rank_memory.read(crrm_keys, markets, matured)
    torch.testing.assert_close(got_out.working_state, ws3, rtol=0, atol=0)
    torch.testing.assert_close(got_out.rank_memory_keys, crrm_keys, rtol=0, atol=0)

    # Confirmar que el orden importa (no es vacuamente conmutativo).
    swapped = combined.session_hologram(
        combined.session_resonator(ws0, markets, batch.prediction_at), markets, batch.prediction_at
    )
    assert not torch.allclose(swapped, ws2)

    # Gradientes finitos y no nulos atraviesan los tres módulos en convivencia.
    diff_out = combined.prepare(
        batch,
        combined.initial_state(batch.flow_ids, differentiable=True),
        differentiable=True,
        session_complete=True,
        rank_memory=matured,
    )
    diff_out.working_state.square().sum().backward()
    for name in (
        "session_hologram.gate",
        "session_resonator.gain",
        "session_resonator.dict_a",
        "cross_sectional_rank_memory.out.weight",
    ):
        grad = dict(combined.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


@pytest.mark.parametrize("config", [SMALL, SMALL_T0, SMALL_SHUFFLED])
def test_autograd_graph_size_is_independent_of_session_row_count_and_deferred_checks_work(config):
    layer = opened_resonator(config, hidden=16, seed=3, dtype=torch.float64)
    small_x = encoded(rows=8, hidden=16, seed=1, dtype=torch.float64).requires_grad_()
    large_x = encoded(rows=256, hidden=16, seed=2, dtype=torch.float64).requires_grad_()
    with deferred_checks():
        out_small = layer(small_x, ("US",) * 8, (1,) * 8)
        out_large = layer(large_x, ("US",) * 256, (1,) * 256)
    assert _autograd_node_count(out_small) == _autograd_node_count(out_large) > 0


def test_non_strided_input_input_nan_and_shuffled_seed_state_dict_are_rejected():
    layer = SessionResonator(16, SMALL, seed=3, dtype=torch.float32)
    valid = encoded(rows=6, hidden=16, seed=9, dtype=torch.float32)
    with pytest.raises(ValueError, match="un mercado y un instante"):
        layer(valid.to_sparse(), MARKETS, AT)
    bad = valid.clone()
    bad[0, 0] = math.nan
    # Incluso con gain=0 en el paso 0, una entrada no finita se rechaza explícitamente.
    with pytest.raises(ValueError, match="entrada al resonador"):
        layer(bad, MARKETS, AT)
    with deferred_checks(), pytest.raises(ValueError, match="entrada al resonador"):
        layer(bad, MARKETS, AT)

    # En shuffled_peers la semilla de permutación forma parte del estado extra.
    shuf_a = SessionResonator(16, SMALL_SHUFFLED, seed=3, dtype=torch.float64)
    shuf_b = SessionResonator(16, SMALL_SHUFFLED, seed=4, dtype=torch.float64)
    with pytest.raises(ValueError):
        shuf_b.load_state_dict(shuf_a.state_dict())
    res_a = SessionResonator(16, SMALL, seed=3, dtype=torch.float64)
    res_b = SessionResonator(16, SMALL, seed=4, dtype=torch.float64)
    res_b.load_state_dict(res_a.state_dict())


def test_frozen_consumer_supports_episodic_readout_and_rejects_buffer_tamper():
    predictor = model(session_resonator=SMALL)
    with torch.no_grad():
        predictor.session_resonator.gain.fill_(0.2)
    predictor._seal_parameters()
    predictor.eval().requires_grad_(False)
    ext = reader(refinements=1).eval().requires_grad_(False)
    engine = consumer(predictor, ext)
    batch = session_batch(6, markets=list(MARKETS))
    state = predictor.initial_state(batch.flow_ids)
    snap = bank(batch)
    expected = apply_episodic_readout(
        predictor.prepare(batch, state, session_complete=True),
        predictor.head,
        ext,
        snapshot=snap,
        context_id=CONTEXT,
        cutoff=batch.prediction_at[0],
        differentiable=False,
    )
    actual = engine.prepare(batch, state, snapshot=snap, context_id=CONTEXT, session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)

    # Alterar el dtype del buffer hadamard se detecta en la validación del consumidor congelado.
    tampered = model(session_resonator=SMALL)
    tampered.session_resonator.hadamard = tampered.session_resonator.hadamard.float()
    tampered._seal_parameters()
    tampered.eval().requires_grad_(False)
    with pytest.raises(ValueError, match="resonador de sesión"):
        consumer(tampered)


def test_chronological_trainer_split_session_rejection_modes_and_clip_groups(tmp_path):
    from mars_titan.training.financial_run import clip_parameter_groups
    from tests.training.test_financial_run_session_objective import (
        predictor as run_predictor,
    )
    from tests.training.test_financial_run_session_objective import (
        records,
        streams,
    )
    from tests.training.test_financial_run_session_objective import (
        trainer as run_trainer,
    )

    shared = streams(tmp_path / "corpus")
    for cfg in (SMALL, SMALL_T0, SMALL_SHUFFLED):
        csrf_model = run_predictor(
            shared,
            session_rank_inputs=None,
            session_ic_objective=None,
            session_resonator=cfg,
        )
        groups = clip_parameter_groups(csrf_model)
        assert "session_resonator.gain" in groups["backbone_fusion"]
        assert "session_resonator.dict_a" in groups["backbone_fusion"]
        engine = run_trainer(shared, tmp_path / cfg.mode, model=csrf_model)
        report = engine.run()
        assert report["status"] == "completed"
        grads = records(engine)
        assert grads[0]["session_resonator.gain"].abs().sum() > 0
        assert torch.isfinite(grads[0]["session_resonator.gain"]).all()

    split = run_trainer(
        shared,
        tmp_path / "split",
        model=run_predictor(
            shared,
            session_rank_inputs=None,
            session_ic_objective=None,
            session_resonator=SMALL,
        ),
        block_rows=2,
    )
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()


def test_us_campaign_v12_screen3_single_variable_declarations_validate_cleanly():
    root = Path(__file__).resolve().parents[3]
    doc = json.loads((root / "configs/experiments/us-campaign-v12-screen3.json").read_text())
    control = doc["models"]["titans_mac"]
    arms = {
        "titans_mac_session_resonator": dict(
            session_resonator=dict(codes_a=8, codes_b=8, iterations=2, mode="resonator")
        ),
        "titans_mac_session_resonator_t0": dict(
            session_resonator=dict(codes_a=8, codes_b=8, iterations=0, mode="no_iteration")
        ),
        "titans_mac_session_resonator_shuffled": dict(
            session_resonator=dict(codes_a=8, codes_b=8, iterations=2, mode="shuffled_peers")
        ),
    }
    for name, changes in arms.items():
        assert checked_predictor_changes(changes) == changes
        doc["models"][name] = dict(control, tier="s3_inventions", predictor_changes=changes)
    validated = comparison_campaign.checked_declaration(doc, root=root)
    for name in arms:
        assert name in validated["models"]
        assert {
            k: v for k, v in validated["models"][name].items() if k != "predictor_changes"
        } == dict(control, tier="s3_inventions")
