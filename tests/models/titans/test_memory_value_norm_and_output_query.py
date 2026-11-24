"""Opciones WP4: normalización de valores objetivo (`layer_norm_v1`) y consulta de salida
proyectada (`projected_v1`) en `NeuralMemory`, `TitansMAC` y `FinancialPredictor`.
"""

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_financial_adapter import DIMENSIONS, raw_batch, specification
from torch.nn import functional as F

from mars_titan.models.learning_scale import LearningScale, fit_statistics
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.associative_gradient import associative_gradients
from mars_titan.models.titans.config import (
    ANALYTIC_INNER_GRADIENT,
    AUTOGRAD_INNER_GRADIENT,
    FUSED_INNER_GRADIENT,
    LAYER_NORM_EPS,
    GateBias,
    MACConfig,
    MemoryConfig,
    canonical,
)
from mars_titan.models.titans.financial import (
    VARIANTS,
    DecisionBatch,
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.session_context import MARKET_MEMORY
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.dust.financial import FinancialDustAdapter
from mars_titan.training.search_cases import checked_predictor_changes
from tests.suite_support import cpu_threads

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_V11 = ROOT / "configs/experiments/us-campaign-v11-screen1.json"

# Huellas capturadas sobre el commit base `b56cef8` antes de añadir estas dos opciones.
BASE_DIGESTS = {
    ("fp32", AUTOGRAD_INNER_GRADIENT): dict(
        m_fp="c33deeebca270b3b",
        mac_fp="8580f64d1408a66a",
        out="37b2cf387162ac6c",
        st="a24392159e7ef2e6",
        grad="7c8f1953e53dba5e",
    ),
    ("fp32", ANALYTIC_INNER_GRADIENT): dict(
        m_fp="ba2818baebb1ea09",
        mac_fp="29ba3be3381d38c2",
        out="1a1ff39129d7b4e3",
        st="c72b5e73b78a163f",
        grad="4dd07f7e19a5c25a",
    ),
    ("fp64", AUTOGRAD_INNER_GRADIENT): dict(
        m_fp="c33deeebca270b3b",
        mac_fp="8580f64d1408a66a",
        out="44c354e02e08753c",
        st="f1d9c21ca3f4c9f1",
        grad="f18a8be47ecd2a93",
    ),
    ("fp64", ANALYTIC_INNER_GRADIENT): dict(
        m_fp="ba2818baebb1ea09",
        mac_fp="29ba3be3381d38c2",
        out="b20a1da311ceedd1",
        st="1f1bc08d9265912d",
        grad="66ebf21015ff1fa6",
    ),
}
BASE_FINANCIAL_FP64 = dict(
    id="4cf875a3e9f765dc",
    pred="35f41f14d7060baf",
    grad="d324b4b3de283500",
)


def _digest(tensors) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous()
        hasher.update(str((tuple(array.shape), str(array.dtype))).encode())
        hasher.update(array.numpy().tobytes())
    return hasher.hexdigest()


def _grad_digest(parameters, gradients) -> str:
    return _digest(
        [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def _reference_fused_write(
    weights, momentum, keys, values, gates, *, residual_layer_norm: bool, eps: float
):
    """Referencia CPU algebraicamente idéntica al kernel Triton de `fused_associative_write`."""
    alpha, eta, theta = gates
    gradients, elements = associative_gradients(
        weights, keys, values, residual_layer_norm=residual_layer_norm, eps=eps
    )
    following = tuple(eta * m - theta * g for m, g in zip(momentum, gradients, strict=True))
    written = tuple((1 - alpha) * w + s for w, s in zip(weights, following, strict=True))
    return written, following, elements


def _reference_fused_write_and_read(
    weights,
    momentum,
    keys,
    values,
    gates,
    post_query,
    *,
    residual_layer_norm: bool,
    eps: float,
    return_input_absmax: bool = False,
):
    """Referencia CPU algebraicamente idéntica a `fused_associative_write_and_read`."""
    written, following, elements = _reference_fused_write(
        weights,
        momentum,
        keys,
        values,
        gates,
        residual_layer_norm=residual_layer_norm,
        eps=eps,
    )
    hidden = torch.bmm(post_query.unsqueeze(1), written[0].transpose(1, 2))
    output = torch.bmm(F.gelu(hidden), written[1].transpose(1, 2)).squeeze(1)
    if residual_layer_norm:
        read_post = post_query + F.layer_norm(output, (output.shape[-1],), eps=eps)
    else:
        read_post = output
    tracked = (*written, *following)
    if return_input_absmax:
        tracked = (*tracked, *weights, *momentum)
    absmax = torch.stack(
        [
            torch.where(
                torch.isfinite(tensor), tensor.detach().abs(), torch.full_like(tensor, float("inf"))
            ).amax(dim=(1, 2))
            for tensor in tracked
        ]
    )
    return written, following, elements, read_post, absmax


@cpu_threads(2)
@pytest.mark.parametrize(
    ("dtype_name", "dtype"), [("fp32", torch.float32), ("fp64", torch.float64)]
)
@pytest.mark.parametrize("inner_gradient", [AUTOGRAD_INNER_GRADIENT, ANALYTIC_INNER_GRADIENT])
def test_defaults_match_base_commit_identities_outputs_states_and_gradients_bit_for_bit(
    dtype_name, dtype, inner_gradient
):
    memory_cfg = MemoryConfig(
        dim=8,
        depth=2,
        max_batch=2,
        max_tokens=2,
        gate_bias=GateBias(),
        residual_layer_norm=True,
        inner_gradient=inner_gradient,
    )
    assert "value_normalization" not in memory_cfg.identity()
    mac_cfg = MACConfig(
        memory=memory_cfg, heads=2, persistent_tokens=2, max_segment=2, memory_mode="online"
    )
    assert "output_query" not in mac_cfg.identity()
    mac = TitansMAC(mac_cfg, dtype=dtype)
    generator = torch.Generator().manual_seed(101)
    segments = [0.5 * torch.randn(2, 2, 8, generator=generator, dtype=dtype) for _ in range(3)]
    state = mac.initial_state(2, differentiable=True)
    outputs = []
    for segment in segments:
        out, state = mac(segment, state, differentiable=True)
        outputs.append(out)
    loss = sum(out.square().sum() for out in outputs)
    grads = torch.autograd.grad(loss, list(mac.parameters()), allow_unused=True)
    observed = dict(
        m_fp=memory_cfg.fingerprint()[:16],
        mac_fp=mac_cfg.fingerprint()[:16],
        out=_digest(outputs)[:16],
        st=_digest([*state.memory.weights, *state.memory.momentum, state.memory.steps])[:16],
        grad=_grad_digest(mac.parameters(), grads)[:16],
    )
    assert observed == BASE_DIGESTS[(dtype_name, inner_gradient)]

    # Declarar explícitamente los valores por defecto produce exactamente los mismos bits.
    explicit_mem = replace(memory_cfg, value_normalization=None)
    explicit_mac_cfg = replace(mac_cfg, memory=explicit_mem, output_query="raw")
    assert explicit_mem.identity() == memory_cfg.identity()
    assert explicit_mac_cfg.identity() == mac_cfg.identity()
    assert explicit_mem.fingerprint() == memory_cfg.fingerprint()
    assert explicit_mac_cfg.fingerprint() == mac_cfg.fingerprint()


@cpu_threads(2)
def test_default_financial_predictor_matches_base_commit_bit_for_bit():
    spec = specification()
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        gate_bias=GateBias(),
        memory_residual_layer_norm=True,
    )
    identity = cfg.identity()
    assert "memory_value_normalization" not in identity
    assert "output_query" not in identity
    model = FinancialPredictor(cfg, dtype=torch.float64)
    batch = DecisionBatch.from_corpus(raw_batch(), spec, dtype=torch.float64)
    state = model.initial_state(batch.flow_ids, differentiable=True)
    prepared = model.prepare(batch, state, differentiable=True)
    grads = torch.autograd.grad(
        prepared.point_predictions.square().sum(), list(model.parameters()), allow_unused=True
    )
    assert (
        dict(
            id=hashlib.sha256(canonical(identity).encode()).hexdigest()[:16],
            pred=_digest([prepared.point_predictions])[:16],
            grad=_grad_digest(model.parameters(), grads)[:16],
        )
        == BASE_FINANCIAL_FP64
    )


def test_active_options_create_distinct_identities_preserve_parameters_and_isolate_contracts():
    rng = torch.get_rng_state().clone()
    mem_default = MemoryConfig(dim=8, depth=2, residual_layer_norm=True)
    mem_vnorm = MemoryConfig(
        dim=8, depth=2, residual_layer_norm=True, value_normalization="layer_norm_v1"
    )
    assert "value_normalization" not in mem_default.identity()
    assert mem_vnorm.identity()["value_normalization"] == {
        "form": "layer_norm_v1",
        "layer_norm_eps": LAYER_NORM_EPS,
        "layer_norm_affine": False,
    }
    assert mem_vnorm.fingerprint() != mem_default.fingerprint()

    mac_default = MACConfig(memory=mem_default, heads=2, persistent_tokens=2, max_segment=2)
    mac_vnorm = MACConfig(memory=mem_vnorm, heads=2, persistent_tokens=2, max_segment=2)
    mac_outq = MACConfig(
        memory=mem_default,
        heads=2,
        persistent_tokens=2,
        max_segment=2,
        output_query="projected_v1",
    )
    mac_both = MACConfig(
        memory=mem_vnorm,
        heads=2,
        persistent_tokens=2,
        max_segment=2,
        output_query="projected_v1",
    )
    assert "output_query" not in mac_default.identity()
    assert mac_outq.identity()["output_query"] == {
        "form": "projected_v1",
        "projection": "shared_query_projection_then_l2",
        "normalization_eps": 1e-12,
    }
    fingerprints = {cfg.fingerprint() for cfg in (mac_default, mac_vnorm, mac_outq, mac_both)}
    assert len(fingerprints) == 4

    # Ninguna de las dos opciones añade parámetros ni consume RNG.
    models = [
        TitansMAC(cfg, dtype=torch.float64) for cfg in (mac_default, mac_vnorm, mac_outq, mac_both)
    ]
    assert torch.equal(torch.get_rng_state(), rng)
    base_state = models[0].state_dict()
    for other in models[1:]:
        other_state = other.state_dict()
        assert set(other_state) == set(base_state)
        for name, value in base_state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(other_state[name], value), name

    # Contratos incompatibles rechazan estados y pesos de otras combinaciones.
    state_default = models[0].initial_state(1)
    for other in models[1:]:
        with pytest.raises(ValueError, match="contrato|configur"):
            other.load_state_dict(base_state)
        with pytest.raises(ValueError, match="contrato|configur"):
            other.restore_state(models[0].export_state(state_default))
        with pytest.raises(ValueError, match="contrato|configur"):
            other(torch.ones(1, 1, 8, dtype=torch.float64), state_default)


@pytest.mark.parametrize(
    "bad_memory",
    [
        dict(dim=4, value_normalization="layer_norm"),
        dict(dim=4, value_normalization="l2"),
        dict(dim=4, value_normalization=""),
        dict(dim=4, value_normalization=True),
        dict(dim=4, value_normalization=1),
        dict(dim=1, value_normalization="layer_norm_v1"),
    ],
)
def test_invalid_memory_value_normalization_is_rejected(bad_memory):
    with pytest.raises(ValueError):
        MemoryConfig(**bad_memory)


@pytest.mark.parametrize(
    "bad_output_query",
    [None, "", "projected", "RAW", "projected_v2", True, 1],
)
def test_invalid_mac_output_query_is_rejected(bad_output_query):
    with pytest.raises(ValueError):
        MACConfig(memory=MemoryConfig(dim=4), output_query=bad_output_query)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("residual_layer_norm", [False, True])
def test_value_normalization_matches_manual_parameter_free_layer_norm_in_update_and_shared(
    depth, residual_layer_norm
):
    cfg = MemoryConfig(
        dim=6,
        depth=depth,
        max_batch=2,
        max_tokens=3,
        gate_bias=GateBias(),
        residual_layer_norm=residual_layer_norm,
        value_normalization="layer_norm_v1",
    )
    mem = NeuralMemory(cfg, dtype=torch.float64)
    mem_raw = NeuralMemory(replace(cfg, value_normalization=None), dtype=torch.float64)
    generator = torch.Generator().manual_seed(17)
    observed = 5.0 * torch.randn(2, 2, 6, generator=generator, dtype=torch.float64)
    state = mem.initial_state(2)

    actual = mem.update(observed, state)
    raw_actual = mem_raw.update(observed, mem_raw.initial_state(2))
    assert not torch.allclose(actual.weights[0], raw_actual.weights[0])

    # Ecuación manual independiente (sin usar F.layer_norm ni helpers de NeuralMemory).
    weights, momentum = state.weights, state.momentum
    for step in observed.unbind(1):
        with torch.no_grad():
            keys = F.normalize(mem.key_projection(step), dim=-1, eps=1e-12)
            raw_values = mem.value_projection(step)
            centred = raw_values - raw_values.mean(dim=-1, keepdim=True)
            variance = centred.square().mean(dim=-1, keepdim=True)
            target_values = centred / torch.sqrt(variance + LAYER_NORM_EPS)
            alpha = mem.alpha_projection(step).sigmoid().unsqueeze(-1)
            eta = mem.eta_projection(step).sigmoid().unsqueeze(-1)
            theta = cfg.theta_max * mem.theta_projection(step).sigmoid().unsqueeze(-1)
        local = tuple(w.detach().clone().requires_grad_(True) for w in weights)
        prediction = mem._apply_memory(keys.unsqueeze(1), local, allow_fused=False).squeeze(1)
        gradients = torch.autograd.grad((prediction - target_values).square().sum(), local)
        momentum = tuple(eta * m - theta * g for m, g in zip(momentum, gradients, strict=True))
        weights = tuple((1 - alpha) * w + s for w, s in zip(weights, momentum, strict=True))

    for left, right in zip((*actual.weights, *actual.momentum), (*weights, *momentum), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-13)

    # También en `update_shared` (1 fila coincide bit a bit con `update`, y varias filas promedian).
    shared_state = mem.initial_state(1)
    one_row = observed[0, :1]
    from_update = mem.update(one_row.unsqueeze(1), shared_state)
    from_shared = mem.update_shared(one_row, shared_state)
    for left, right in zip(
        (*from_shared.weights, *from_shared.momentum),
        (*from_update.weights, *from_update.momentum),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    multi_rows = observed.reshape(-1, 6)
    shared_multi = mem.update_shared(multi_rows, shared_state)
    raw_shared_multi = mem_raw.update_shared(multi_rows, mem_raw.initial_state(1))
    assert not torch.allclose(shared_multi.weights[0], raw_shared_multi.weights[0])


@pytest.mark.parametrize("mode", ["online", "frozen", "disabled"])
def test_output_query_projected_v1_matches_manual_hadamard_read_without_changing_fast_state(mode):
    mem_cfg = MemoryConfig(
        dim=8, depth=2, max_batch=2, max_tokens=2, gate_bias=GateBias(), residual_layer_norm=True
    )
    raw_cfg = MACConfig(
        memory=mem_cfg, heads=2, persistent_tokens=2, max_segment=2, memory_mode=mode
    )
    proj_cfg = replace(raw_cfg, output_query="projected_v1")
    mac_raw = TitansMAC(raw_cfg, dtype=torch.float64)
    mac_proj = TitansMAC(proj_cfg, dtype=torch.float64)

    generator = torch.Generator().manual_seed(29)
    segment = torch.randn(2, 2, 8, generator=generator, dtype=torch.float64)
    out_raw, st_raw = mac_raw(segment, mac_raw.initial_state(2))
    out_proj, st_proj = mac_proj(segment, mac_proj.initial_state(2))

    # El estado rápido escrito en un paso no cambia por `output_query`: solo la puerta de salida.
    for left, right in zip(
        (*st_proj.memory.weights, *st_proj.memory.momentum, st_proj.memory.steps),
        (*st_raw.memory.weights, *st_raw.memory.momentum, st_raw.memory.steps),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    if mode == "disabled":
        torch.testing.assert_close(out_proj, out_raw, rtol=0, atol=0)
        return

    assert not torch.allclose(out_proj, out_raw)

    # Cálculo manual paso a paso del cierre del segmento con consulta proyectada y normalizada L2.
    with torch.no_grad():
        state0 = mac_proj.initial_state(2)
        queries = F.normalize(mac_proj.query_projection(segment), dim=-1, eps=1e-12)
        retrieved = mac_proj.memory.read(queries, state0.memory)
        prefix = mac_proj.persistent.unsqueeze(0).expand(2, -1, -1)
        packed = torch.cat((prefix, retrieved, segment), dim=1)
        mask = torch.ones(packed.shape[1], packed.shape[1], dtype=torch.bool).triu(1)
        attended, _ = mac_proj.attention(packed, packed, packed, attn_mask=mask, need_weights=False)
        observed = attended[:, -2:]
        next_mem = (
            mac_proj.memory.update(observed, state0.memory) if mode == "online" else state0.memory
        )
        post_query = F.normalize(mac_proj.query_projection(observed[:, -1]), dim=-1, eps=1e-12)
        expected = observed[:, -1] * mac_proj.memory.read(
            post_query.unsqueeze(1), next_mem
        ).squeeze(1)

    torch.testing.assert_close(out_proj, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("dim", [16, 64])
def test_value_normalization_breaks_the_residual_layer_norm_associative_loss_floor(dim):
    """Caso sintético del encargo: con valores de norma grande, sin normalización la pérdida
    asociativa queda atrapada por encima del suelo geométrico (‖v‖ − (√D + 1))², mientras que
    con `value_normalization='layer_norm_v1'` la escritura repetida del mismo par reduce la
    pérdida muy por debajo de ese suelo.
    """
    base_cfg = MemoryConfig(
        dim=dim,
        depth=2,
        max_batch=1,
        max_tokens=1,
        gate_bias=GateBias(alpha_half_life=256.0, eta=0.5, theta=0.05),
        residual_layer_norm=True,
    )
    norm_cfg = replace(base_cfg, value_normalization="layer_norm_v1")
    mem_raw = NeuralMemory(base_cfg, dtype=torch.float64)
    mem_norm = NeuralMemory(norm_cfg, dtype=torch.float64)

    generator = torch.Generator().manual_seed(123)
    token = torch.randn(1, 1, dim, generator=generator, dtype=torch.float64) / math.sqrt(dim)
    with torch.no_grad():
        raw_value = mem_raw.value_projection(token[:, 0])
        # Escalar `value_projection` para que el valor objetivo tenga norma 50 (fuera de
        # [√D−1, √D+1]) manteniendo el token y las puertas α, η, θ en su escala habitual.
        scale = 50.0 / raw_value.norm(dim=-1).item()
        mem_raw.value_projection.weight.mul_(scale)
        mem_norm.value_projection.weight.mul_(scale)
        raw_value_norm = mem_raw.value_projection(token[:, 0]).norm(dim=-1).item()
        norm_value_norm = (
            F.layer_norm(mem_norm.value_projection(token[:, 0]), (dim,), eps=LAYER_NORM_EPS)
            .norm(dim=-1)
            .item()
        )

    assert raw_value_norm == pytest.approx(50.0, rel=1e-12)
    assert norm_value_norm <= math.sqrt(dim) + 1e-9
    geometric_floor = (raw_value_norm - (math.sqrt(dim) + 1.0)) ** 2

    st_raw, st_norm = mem_raw.initial_state(1), mem_norm.initial_state(1)
    losses_raw, losses_norm = [], []
    with torch.no_grad():
        for _ in range(40):
            st_raw = mem_raw.update(token, st_raw, surprise=losses_raw)
            st_norm = mem_norm.update(token, st_norm, surprise=losses_norm)

    raw_series = [entry.item() for entry in losses_raw]
    norm_series = [entry.item() for entry in losses_norm]

    # Sin normalización de valores, ninguna escritura puede bajar del suelo geométrico.
    assert min(raw_series) >= geometric_floor - 1e-6
    # Con `layer_norm_v1`, el objetivo entra en la corona y la escritura repetida baja la pérdida
    # drásticamente y muy por debajo del suelo sin normalización.
    assert norm_series[-1] < norm_series[0]
    assert norm_series[-1] < 0.2
    assert norm_series[-1] < 1e-4 * geometric_floor


@pytest.mark.parametrize(
    ("value_normalization", "output_query"),
    [
        ("layer_norm_v1", "raw"),
        (None, "projected_v1"),
        ("layer_norm_v1", "projected_v1"),
    ],
)
@pytest.mark.parametrize("residual_layer_norm", [True, False])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"), [(torch.float64, 1e-10, 1e-12), (torch.float32, 2e-5, 2e-5)]
)
def test_autograd_and_analytic_inner_gradients_match_with_each_option(
    value_normalization, output_query, residual_layer_norm, dtype, rtol, atol
):
    mem_auto = MemoryConfig(
        dim=8,
        depth=2,
        max_batch=3,
        max_tokens=2,
        gate_bias=GateBias(),
        residual_layer_norm=residual_layer_norm,
        value_normalization=value_normalization,
        inner_gradient=AUTOGRAD_INNER_GRADIENT,
    )
    mem_anal = replace(mem_auto, inner_gradient=ANALYTIC_INNER_GRADIENT)
    macs = [
        TitansMAC(
            MACConfig(
                memory=cfg,
                heads=2,
                persistent_tokens=2,
                max_segment=2,
                output_query=output_query,
            ),
            dtype=dtype,
        )
        for cfg in (mem_auto, mem_anal)
    ]
    generator = torch.Generator().manual_seed(41)
    segments = [0.5 * torch.randn(3, 2, 8, generator=generator, dtype=dtype) for _ in range(3)]

    results = []
    for mac in macs:
        inputs = [seg.clone().requires_grad_(True) for seg in segments]
        state = mac.initial_state(3, differentiable=True)
        surprises, outputs = [], []
        for seg in inputs:
            out, state = mac(seg, state, differentiable=True, surprise=surprises)
            outputs.append(out)
        total = torch.stack(outputs)
        objective = total.square().sum() + sum(m.square().sum() for m in state.memory.momentum)
        grads = torch.autograd.grad(objective, [*mac.parameters(), *inputs])
        results.append((total, state, torch.stack(surprises), grads))

    (o0, s0, e0, g0), (o1, s1, e1, g1) = results
    torch.testing.assert_close(o1, o0, rtol=rtol, atol=atol)
    torch.testing.assert_close(e1, e0, rtol=rtol, atol=atol)
    for a, b in zip(
        (*s0.memory.weights, *s0.memory.momentum),
        (*s1.memory.weights, *s1.memory.momentum),
        strict=True,
    ):
        torch.testing.assert_close(b, a, rtol=rtol, atol=atol)
    for a, b in zip(g0, g1, strict=True):
        torch.testing.assert_close(b, a, rtol=rtol, atol=max(atol, rtol * a.abs().max().item()))

    # Ruta sin grafo (`differentiable=False`): coincide y no conserva grafo.
    det_states, det_outs = [], []
    for mac in macs:
        st = mac.initial_state(3, differentiable=False)
        out, st = mac(segments[0], st, differentiable=False)
        assert not out.requires_grad and out.grad_fn is None
        assert all(
            not w.requires_grad and w.grad_fn is None
            for w in (*st.memory.weights, *st.memory.momentum)
        )
        det_outs.append(out)
        det_states.append(st)
    torch.testing.assert_close(det_outs[1], det_outs[0], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("value_normalization", "output_query"),
    [
        ("layer_norm_v1", "raw"),
        (None, "projected_v1"),
        ("layer_norm_v1", "projected_v1"),
    ],
)
@pytest.mark.parametrize("residual_layer_norm", [True, False])
def test_fused_v1_matches_autograd_and_analytic_on_cpu_via_reference_for_update_and_update_and_read(
    monkeypatch, value_normalization, output_query, residual_layer_norm
):
    from mars_titan.models.titans import fused_memory_write as fused_mod

    monkeypatch.setattr(fused_mod, "fused_associative_write", _reference_fused_write)
    monkeypatch.setattr(
        fused_mod, "fused_associative_write_and_read", _reference_fused_write_and_read
    )

    dim = 64
    base_mem = MemoryConfig(
        dim=dim,
        depth=2,
        max_batch=2,
        max_tokens=1,
        gate_bias=GateBias(),
        residual_layer_norm=residual_layer_norm,
        value_normalization=value_normalization,
    )
    configs = {
        route: MACConfig(
            memory=replace(base_mem, inner_gradient=route),
            heads=4,
            persistent_tokens=2,
            max_segment=1,
            output_query=output_query,
        )
        for route in (AUTOGRAD_INNER_GRADIENT, ANALYTIC_INNER_GRADIENT, FUSED_INNER_GRADIENT)
    }
    macs = {route: TitansMAC(cfg, dtype=torch.float64) for route, cfg in configs.items()}

    generator = torch.Generator().manual_seed(73)
    segments = [
        0.3 * torch.randn(2, 1, dim, generator=generator, dtype=torch.float64) for _ in range(3)
    ]

    # 1) Comparar las tres rutas a través de TitansMAC.forward (en CPU fused_v1 usa update).
    mac_results = {}
    for route, mac in macs.items():
        inputs = [seg.clone().requires_grad_(True) for seg in segments]
        state = mac.initial_state(2, differentiable=True)
        surprises, outputs = [], []
        for seg in inputs:
            out, state = mac(seg, state, differentiable=True, surprise=surprises)
            outputs.append(out)
        total = torch.stack(outputs)
        objective = total.square().sum() + sum(m.square().sum() for m in state.memory.momentum)
        grads = torch.autograd.grad(objective, [*mac.parameters(), *inputs])
        mac_results[route] = (total, state, torch.stack(surprises), grads)

    ref_out, ref_st, ref_surp, ref_grads = mac_results[AUTOGRAD_INNER_GRADIENT]
    for route in (ANALYTIC_INNER_GRADIENT, FUSED_INNER_GRADIENT):
        out, st, surp, grads = mac_results[route]
        torch.testing.assert_close(out, ref_out, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(surp, ref_surp, rtol=1e-11, atol=1e-12)
        for a, b in zip(
            (*ref_st.memory.weights, *ref_st.memory.momentum),
            (*st.memory.weights, *st.memory.momentum),
            strict=True,
        ):
            torch.testing.assert_close(b, a, rtol=1e-11, atol=1e-12)
        for a, b in zip(ref_grads, grads, strict=True):
            torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-10)

    # 2) Verificar también la ruta fusionada `update_and_read` con `_output_query` frente a
    # `update` + `read` de `autograd_v1` (mismos estados, lectura posterior, sorpresa y gradientes).
    fused_mac = macs[FUSED_INNER_GRADIENT]
    auto_mac = macs[AUTOGRAD_INNER_GRADIENT]
    obs_a = segments[0].clone().requires_grad_(True)
    obs_f = segments[0].clone().requires_grad_(True)

    st_a = auto_mac.memory.initial_state(2, differentiable=True)
    surp_a = []
    next_a = auto_mac.memory.update(obs_a, st_a, differentiable=True, surprise=surp_a)
    pq_a = auto_mac._output_query(obs_a[:, -1])
    read_a = auto_mac.memory.read(pq_a.unsqueeze(1), next_a).squeeze(1)
    out_a = obs_a[:, -1] * read_a
    obj_a = out_a.square().sum() + sum(
        w.square().sum() for w in (*next_a.weights, *next_a.momentum)
    )
    grads_a = torch.autograd.grad(
        obj_a,
        [*auto_mac.memory.parameters(), auto_mac.query_projection.weight, obs_a],
        allow_unused=True,
    )

    st_f = fused_mac.memory.initial_state(2, differentiable=True)
    surp_f = []
    pq_f = fused_mac._output_query(obs_f[:, -1])
    next_f, read_f = fused_mac.memory.update_and_read(
        obs_f, st_f, pq_f, differentiable=True, surprise=surp_f
    )
    out_f = obs_f[:, -1] * read_f
    obj_f = out_f.square().sum() + sum(
        w.square().sum() for w in (*next_f.weights, *next_f.momentum)
    )
    grads_f = torch.autograd.grad(
        obj_f,
        [*fused_mac.memory.parameters(), fused_mac.query_projection.weight, obs_f],
        allow_unused=True,
    )

    torch.testing.assert_close(read_f, read_a, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(out_f, out_a, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(surp_f[0], surp_a[0], rtol=1e-11, atol=1e-12)
    for a, b in zip(
        (*next_a.weights, *next_a.momentum), (*next_f.weights, *next_f.momentum), strict=True
    ):
        torch.testing.assert_close(b, a, rtol=1e-11, atol=1e-12)
    for a, b in zip(grads_a, grads_f, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-10)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("value_normalization", "output_query"),
    [
        ("layer_norm_v1", "raw"),
        (None, "projected_v1"),
        ("layer_norm_v1", "projected_v1"),
    ],
)
def test_cuda_fused_v1_kernels_match_analytic_with_each_option(value_normalization, output_query):
    if not torch.cuda.is_available():
        pytest.skip("Sin CUDA visible")
    device = torch.device("cuda:0")
    base_mem = MemoryConfig(
        dim=64,
        depth=2,
        max_batch=4,
        max_tokens=1,
        gate_bias=GateBias(),
        residual_layer_norm=True,
        value_normalization=value_normalization,
    )
    macs = [
        TitansMAC(
            MACConfig(
                memory=replace(base_mem, inner_gradient=route),
                heads=4,
                persistent_tokens=2,
                max_segment=1,
                output_query=output_query,
            ),
            dtype=torch.float64,
        ).to(device)
        for route in (ANALYTIC_INNER_GRADIENT, FUSED_INNER_GRADIENT)
    ]
    segments = torch.randn(3, 4, 1, 64, device=device, dtype=torch.float64)
    outputs, grads = [], []
    for mac in macs:
        state = mac.initial_state(4, differentiable=True)
        emitted = []
        for seg in segments:
            out, state = mac(seg, state, differentiable=True)
            emitted.append(out)
        total = torch.stack(emitted)
        outputs.append(total)
        grads.append(torch.autograd.grad(total.square().sum(), list(mac.parameters())))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-11, atol=1e-12)
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("inner_gradient", [AUTOGRAD_INNER_GRADIENT, ANALYTIC_INNER_GRADIENT])
def test_gradcheck_for_value_normalization_and_output_query_in_float64(inner_gradient):
    mem_cfg = MemoryConfig(
        dim=3,
        depth=2,
        max_batch=1,
        max_tokens=2,
        residual_layer_norm=True,
        value_normalization="layer_norm_v1",
        inner_gradient=inner_gradient,
    )
    mac = TitansMAC(
        MACConfig(
            memory=mem_cfg,
            heads=1,
            persistent_tokens=1,
            max_segment=2,
            output_query="projected_v1",
        ),
        dtype=torch.float64,
    )
    state = mac.initial_state(1, differentiable=True)
    generator = torch.Generator().manual_seed(55)
    segment = (
        0.4 * torch.randn(1, 2, 3, generator=generator, dtype=torch.float64)
    ).requires_grad_()
    w1, w2 = state.memory.weights
    m1 = state.memory.momentum[0].detach().requires_grad_()
    m2 = state.memory.momentum[1].detach().requires_grad_()

    def fn(x, weight1, weight2, mom1, mom2):
        initial = replace(
            state,
            memory=replace(state.memory, weights=(weight1, weight2), momentum=(mom1, mom2)),
        )
        out, updated = mac(x, initial, differentiable=True)
        return (out, *updated.memory.weights, *updated.memory.momentum)

    assert torch.autograd.gradcheck(fn, (segment, w1, w2, m1, m2), eps=1e-6, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize(
    ("value_normalization", "output_query"),
    [
        ("layer_norm_v1", "raw"),
        (None, "projected_v1"),
        ("layer_norm_v1", "projected_v1"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_finiteness_and_determinism_under_strict_algorithms(
    value_normalization, output_query, dtype
):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        cfg = MACConfig(
            memory=MemoryConfig(
                dim=8,
                depth=2,
                max_batch=3,
                max_tokens=2,
                gate_bias=GateBias(),
                residual_layer_norm=True,
                value_normalization=value_normalization,
                inner_gradient=ANALYTIC_INNER_GRADIENT,
            ),
            heads=2,
            persistent_tokens=2,
            max_segment=2,
            output_query=output_query,
        )
        mac = TitansMAC(cfg, dtype=dtype)
        generator = torch.Generator().manual_seed(89)
        seg = torch.randn(3, 2, 8, generator=generator, dtype=dtype)

        runs = []
        for _ in range(2):
            state = mac.initial_state(3, differentiable=True)
            out, next_state = mac(seg, state, differentiable=True)
            assert out.shape == (3, 8) and out.dtype == dtype
            assert torch.isfinite(out).all()
            grads = torch.autograd.grad(out.square().sum(), list(mac.parameters()))
            assert all(torch.isfinite(g).all() for g in grads)
            runs.append((out, next_state, grads))

        (o1, s1, g1), (o2, s2, g2) = runs
        assert torch.equal(o1, o2)
        for a, b in zip(
            (*s1.memory.weights, *s1.memory.momentum, *g1),
            (*s2.memory.weights, *s2.memory.momentum, *g2),
            strict=True,
        ):
            assert torch.equal(a, b)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("inner_gradient", [AUTOGRAD_INNER_GRADIENT, ANALYTIC_INNER_GRADIENT])
def test_gradients_are_finite_for_constant_values_zero_inputs_and_zero_last_layer(
    dtype, inner_gradient
):
    # 1) Valores proyectados constantes (varianza nula antes de LayerNorm) y entradas nulas.
    mem = NeuralMemory(
        MemoryConfig(
            dim=4,
            depth=2,
            max_batch=2,
            max_tokens=2,
            residual_layer_norm=True,
            value_normalization="layer_norm_v1",
            inner_gradient=inner_gradient,
        ),
        dtype=dtype,
    )
    with torch.no_grad():
        # Todas las filas de value_projection iguales => cualquier token da un valor constante.
        mem.value_projection.weight.fill_(1.5)
    for token in (torch.zeros(2, 2, 4, dtype=dtype), torch.ones(2, 2, 4, dtype=dtype)):
        x = token.clone().requires_grad_(True)
        st = mem.update(x, mem.initial_state(2, differentiable=True), differentiable=True)
        loss = sum(w.square().sum() for w in (*st.weights, *st.momentum))
        grads = torch.autograd.grad(loss, [*mem.parameters(), x], allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)

    # 2) TitansMAC con entrada nula y consulta de salida proyectada.
    mac = TitansMAC(
        MACConfig(
            memory=MemoryConfig(
                dim=4,
                depth=2,
                max_batch=2,
                max_tokens=1,
                residual_layer_norm=True,
                value_normalization="layer_norm_v1",
                inner_gradient=inner_gradient,
            ),
            heads=2,
            persistent_tokens=2,
            max_segment=1,
            output_query="projected_v1",
        ),
        dtype=dtype,
    )
    zeros = torch.zeros(2, 1, 4, dtype=dtype, requires_grad=True)
    out, st = mac(zeros, mac.initial_state(2, differentiable=True), differentiable=True)
    assert torch.isfinite(out).all()
    grads = torch.autograd.grad(out.square().sum(), [*mac.parameters(), zeros], allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)

    # 3) Predictor financiero cuya última capa arranca en cero (predicción constante nula).
    import numpy as np

    spec = specification()
    scale_cfg = LearningScale(clip=5.0, price_returns=True, zero_median=True)
    cfg = FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        head=QUANTILE_HEAD,
        gate_bias=GateBias(),
        memory_residual_layer_norm=True,
        memory_inner_gradient=inner_gradient,
        memory_value_normalization="layer_norm_v1",
        output_query="projected_v1",
        learning_scale=scale_cfg,
    )
    predictor = FinancialPredictor(cfg, dtype=dtype)
    rng = np.random.default_rng(7)
    fit_inputs = {
        name: rng.normal(
            0.0, 1.0, size=(64, 64, width) if name == "prices" else (64, width)
        ).astype(np.float32)
        for name, width in DIMENSIONS.items()
    }
    for name in ("fundamentals", "macro"):
        fit_inputs[name][:, 1] = 1
        fit_inputs[name][:, 2] = np.abs(fit_inputs[name][:, 2])
    stats = fit_statistics(
        scale_cfg, fit_inputs, np.ones((64, 5), dtype=np.bool_), rng.normal(scale=0.02, size=64)
    )
    predictor.set_learning_statistics(stats)
    batch = DecisionBatch.from_corpus(raw_batch(), spec, dtype=dtype)
    pstate = predictor.initial_state(batch.flow_ids, differentiable=True)
    prepared = predictor.prepare(batch, pstate, differentiable=True)
    assert torch.equal(prepared.point_predictions, torch.zeros_like(prepared.point_predictions))
    targets = torch.tensor([0.02, -0.01], dtype=dtype)
    loss = (prepared.point_predictions - targets).square().sum() + prepared.quantiles.square().sum()
    pgrads = torch.autograd.grad(loss, list(predictor.parameters()), allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in pgrads)
    assert any(g is not None and g.abs().sum() > 0 for g in pgrads)


def test_paper_projections_keep_query_window_unchanged_and_match_analytic_with_both_options():
    base_mem = MemoryConfig(
        dim=8,
        depth=2,
        max_batch=2,
        max_tokens=2,
        residual_layer_norm=True,
        qkv_silu=True,
        qkv_convolution=4,
        value_normalization="layer_norm_v1",
    )
    mac_raw = TitansMAC(
        MACConfig(memory=base_mem, heads=2, persistent_tokens=2, max_segment=2, output_query="raw"),
        dtype=torch.float64,
    )
    mac_auto = TitansMAC(
        MACConfig(
            memory=base_mem,
            heads=2,
            persistent_tokens=2,
            max_segment=2,
            output_query="projected_v1",
        ),
        dtype=torch.float64,
    )
    mac_anal = TitansMAC(
        MACConfig(
            memory=replace(base_mem, inner_gradient=ANALYTIC_INNER_GRADIENT),
            heads=2,
            persistent_tokens=2,
            max_segment=2,
            output_query="projected_v1",
        ),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(91)
    seg = 0.4 * torch.randn(2, 2, 8, generator=generator, dtype=torch.float64)
    out_raw, st_raw = mac_raw(seg, mac_raw.initial_state(2))
    out_auto, st_auto = mac_auto(
        seg, mac_auto.initial_state(2, differentiable=True), differentiable=True
    )
    out_anal, st_anal = mac_anal(
        seg, mac_anal.initial_state(2, differentiable=True), differentiable=True
    )

    # `output_query='projected_v1'` no avanza ni altera la ventana causal de `query_convolution`.
    assert torch.equal(st_auto.convolution[0], st_raw.convolution[0])
    assert not torch.allclose(out_auto, out_raw)
    torch.testing.assert_close(out_anal, out_auto, rtol=1e-12, atol=1e-13)
    g_auto = torch.autograd.grad(out_auto.square().sum(), list(mac_auto.parameters()))
    g_anal = torch.autograd.grad(out_anal.square().sum(), list(mac_anal.parameters()))
    for left, right in zip(g_auto, g_anal, strict=True):
        torch.testing.assert_close(right, left, rtol=1e-10, atol=1e-12)


def test_output_query_rejects_nonfinite_projection():
    mac = TitansMAC(
        MACConfig(
            memory=MemoryConfig(dim=4, depth=1, max_batch=1, max_tokens=1),
            heads=1,
            persistent_tokens=1,
            max_segment=1,
            output_query="projected_v1",
        ),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="NaN|infinito"):
        mac._output_query(torch.full((1, 4), float("inf"), dtype=torch.float64))


def test_causality_and_flow_isolation_hold_with_both_options():
    cfg = MACConfig(
        memory=MemoryConfig(
            dim=8,
            depth=2,
            max_batch=2,
            max_tokens=2,
            gate_bias=GateBias(),
            residual_layer_norm=True,
            value_normalization="layer_norm_v1",
            inner_gradient=ANALYTIC_INNER_GRADIENT,
        ),
        heads=2,
        persistent_tokens=2,
        max_segment=2,
        output_query="projected_v1",
    )
    mac = TitansMAC(cfg, dtype=torch.float64)
    generator = torch.Generator().manual_seed(63)
    segs = [torch.randn(2, 2, 8, generator=generator, dtype=torch.float64) for _ in range(3)]

    # Causalidad temporal: alterar el segmento futuro no cambia las salidas ni estados previos.
    st_a = mac.initial_state(2)
    outs_a, states_a = [], []
    for seg in segs:
        out, st_a = mac(seg, st_a)
        outs_a.append(out)
        states_a.append(st_a)

    segs_future_changed = [segs[0], segs[1], segs[2] * -5.0 + 3.0]
    st_b = mac.initial_state(2)
    for idx, seg in enumerate(segs_future_changed):
        out, st_b = mac(seg, st_b)
        if idx < 2:
            assert torch.equal(out, outs_a[idx])
            for left, right in zip(
                (*st_b.memory.weights, *st_b.memory.momentum),
                (*states_a[idx].memory.weights, *states_a[idx].memory.momentum),
                strict=True,
            ):
                assert torch.equal(left, right)
        else:
            assert not torch.equal(out, outs_a[idx])

    # Aislamiento entre flujos: perturbar el flujo 1 no altera ni un bit del flujo 0.
    perturbed = segs[0].clone()
    perturbed[1] = perturbed[1] * -4.0 + 2.0
    out0, st0 = mac(segs[0], mac.initial_state(2))
    out1, st1 = mac(perturbed, mac.initial_state(2))
    assert torch.equal(out1[0], out0[0])
    assert not torch.equal(out1[1], out0[1])
    for left, right in zip(
        (*st1.memory.weights, *st1.memory.momentum),
        (*st0.memory.weights, *st0.memory.momentum),
        strict=True,
    ):
        assert torch.equal(left[0], right[0])
        assert not torch.equal(left[1], right[1])


def test_financial_wiring_pairing_recovery_dust_and_campaign_declaration():
    spec = specification()
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        # 1) Emparejamiento y cableado en FinancialConfig / FinancialPredictor / SessionContext.
        models = {}
        for variant in VARIANTS:
            cfg = FinancialConfig(
                spec,
                variant=variant,
                hidden_size=32,
                gate_bias=GateBias(),
                memory_residual_layer_norm=True,
                memory_value_normalization="layer_norm_v1",
                output_query="projected_v1",
            )
            assert cfg.identity()["memory_value_normalization"] == "layer_norm_v1"
            assert cfg.identity()["output_query"] == "projected_v1"
            model = FinancialPredictor(cfg, dtype=torch.float64)
            models[variant] = model
            if model.mac is not None:
                assert model.mac.memory.config.value_normalization == "layer_norm_v1"
                assert model.mac.config.output_query == "projected_v1"

        for variant in ("transformer_direct", "mac_disabled", "mac_frozen"):
            receipt = copy_paired_parameters(models["mac_online"], models[variant])
            assert receipt["runtime_state_transferred"] is False

        # Control sin las opciones no se puede emparejar con el candidato.
        control = FinancialPredictor(
            FinancialConfig(
                spec,
                variant="mac_frozen",
                hidden_size=32,
                gate_bias=GateBias(),
                memory_residual_layer_norm=True,
            ),
            dtype=torch.float64,
        )
        with pytest.raises(ValueError):
            copy_paired_parameters(models["mac_online"], control)

        # 2) Roundtrip export/restore y consumidor congelado.
        online = models["mac_online"]
        batch = DecisionBatch.from_corpus(raw_batch(), spec, dtype=torch.float64)
        first = online.prepare(batch, online.initial_state(batch.flow_ids))
        restored = online.restore_state(copy.deepcopy(online.export_state(first.next_state)))
        later = DecisionBatch.from_corpus(
            raw_batch(at=1_609_459_200_000_000 + 86_400_000_000), spec, dtype=torch.float64
        )
        torch.testing.assert_close(
            online.prepare(later, restored).point_predictions,
            online.prepare(later, first.next_state).point_predictions,
            rtol=0,
            atol=0,
        )
        consumer = FrozenFinancialConsumer(
            FinancialPredictor(online.config, dtype=torch.float64).eval().requires_grad_(False)
        )
        assert consumer.identity()["predictor"]["configuration"]["output_query"] == "projected_v1"

        # 3) SessionContext (PT4) también propaga memory_value_normalization a su NeuralMemory.
        pt4 = FinancialPredictor(
            FinancialConfig(
                spec,
                variant="mac_online",
                hidden_size=32,
                session_context=MARKET_MEMORY,
                memory_value_normalization="layer_norm_v1",
            ),
            dtype=torch.float64,
        )
        assert pt4.session_context.memory.config.value_normalization == "layer_norm_v1"

        # 4) FinancialDustAdapter rechaza configuraciones con las opciones nuevas.
        for extra in (
            dict(memory_value_normalization="layer_norm_v1"),
            dict(output_query="projected_v1"),
        ):
            td = FinancialPredictor(
                FinancialConfig(spec, variant="transformer_direct", hidden_size=32, **extra),
                dtype=torch.float64,
            )
            with pytest.raises(ValueError, match="Dust"):
                FinancialDustAdapter(td)

        # 5) Validación de `predictor_changes` y declaración de campaña v11 de una sola variable.
        for changes in (
            {"memory_value_normalization": "layer_norm_v1"},
            {"output_query": "projected_v1"},
            {"memory_value_normalization": "layer_norm_v1", "output_query": "projected_v1"},
        ):
            assert checked_predictor_changes(changes) == changes
        for bad in (
            {"memory_value_normalization": ""},
            {"memory_value_normalization": {}},
            {"output_query": 1},
            {"output_query": {}},
        ):
            with pytest.raises(ValueError):
                checked_predictor_changes(bad)

        # Rechazos en FinancialConfig.
        with pytest.raises(ValueError, match="memory_value_normalization"):
            FinancialConfig(spec, variant="mac_online", memory_value_normalization="bad")
        with pytest.raises(ValueError, match="output_query"):
            FinancialConfig(spec, variant="mac_online", output_query="bad")

        # Verificar que un candidato sobre `us-campaign-v11-screen1.json` pasa toda la validación.
        doc = json.loads(CAMPAIGN_V11.read_text(encoding="utf-8"))
        doc["tiers"].append("wp4_memory")
        doc["models"]["titans_mac_vnorm"] = dict(
            machine="mars_titan",
            recipe="core",
            runner="fit",
            tier="wp4_memory",
            variant="mac_online",
            predictor_changes={"memory_value_normalization": "layer_norm_v1"},
        )
        doc["models"]["titans_mac_outq"] = dict(
            machine="mars_titan",
            recipe="core",
            runner="fit",
            tier="wp4_memory",
            variant="mac_online",
            predictor_changes={"output_query": "projected_v1"},
        )
        campaign.checked_declaration(doc, root=ROOT)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
