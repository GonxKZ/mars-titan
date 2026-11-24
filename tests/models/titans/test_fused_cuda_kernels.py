"""Pruebas exhaustivas de paridad FP64/FP32, determinismo, casos límite y perfiles sm_121/sm_120
para todos los kernels Triton fusionados de MARS-TITAN.
"""

import pytest
import torch

from mars_titan.data.price_windows import gate_price_window
from mars_titan.models.baselines.transformer import CompactPriceTransformer
from mars_titan.models.learning_scale import price_returns
from mars_titan.models.titans import gpu_profile
from mars_titan.models.titans.cross_asset_attention import (
    CrossAssetAttention,
    CrossAssetAttentionConfig,
)
from mars_titan.models.titans.fused_cuda_kernels import (
    fused_flash_cross_asset,
    fused_last_token_sdpa,
    fused_price_factorized_last_attn,
    fused_price_proj_pos_ln,
    fused_price_returns_and_scale,
    fused_scale_numeric_triple,
    fused_scale_presence,
    fused_stable_topk_desc,
    fused_tie_rank_probabilities,
)
from mars_titan.models.titans.fused_memory_write import (
    fused_associative_write_and_read,
    fused_mac_last_token_attention,
    fused_memory_read,
)
from mars_titan.models.titans.gpu_profile import (
    PROFILE_ENV_VAR,
    SM12X_MAX_SHARED_BYTES,
    active_gpu_profile,
)
from mars_titan.models.titans.session_ranks import gaussian_ranks
from mars_titan.models.titans.state import (
    deferred_checks,
    fulfill_finite,
    reserve_finite,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Los kernels Triton requieren CUDA"
)
CUDA = torch.device("cuda:0")


def _reset_profile_cache():
    gpu_profile._resolve_profile.cache_clear()


@pytest.mark.parametrize("profile_name", ["sm_121", "sm_120", "generic"])
def test_gpu_profiles_respect_sm12x_smem_and_wave_alignment(monkeypatch, profile_name):
    monkeypatch.setenv(PROFILE_ENV_VAR, profile_name)
    _reset_profile_cache()
    prof = active_gpu_profile(CUDA)
    assert prof.max_smem_per_cta <= SM12X_MAX_SHARED_BYTES
    for dtype in (torch.float32, torch.float64):
        for hd in (16, 32, 64):
            bm, bn, nw, ns = prof.c5_tiles(dtype, hd)
            assert bm >= 16 and bn >= 16 and nw in (4, 8) and 2 <= ns <= 3
            # Cota conservadora de SMEM de FlashC5 (K y V por etapa) < techo sm_12x
            smem_est = 2 * bn * hd * (4 if dtype == torch.float32 else 8) * ns
            assert smem_est <= SM12X_MAX_SHARED_BYTES
    rpc, n_ctas = prof.wave_aligned_ctas(4096, prof.fac_bwd_ctas)
    assert rpc >= 1 and n_ctas >= 1 and rpc * n_ctas >= 4096
    _reset_profile_cache()


@pytest.mark.parametrize("profile_name", ["sm_121", "sm_120"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("rows", [1, 37])
@pytest.mark.parametrize("gate_last", [False, True])
def test_factorized_compact_transformer_parity_and_determinism(
    monkeypatch, profile_name, dtype, rows, gate_last
):
    monkeypatch.setenv(PROFILE_ENV_VAR, profile_name)
    _reset_profile_cache()
    torch.manual_seed(17)
    model = CompactPriceTransformer(
        6, context=64, hidden_size=64, heads=4, layers=1, max_batch=256
    ).to(CUDA, dtype=dtype)
    prices = torch.randn(rows, 64, 6, device=CUDA, dtype=dtype)
    prices[..., 5] = 1.0
    if gate_last:
        prices[:, 5:12, 5] = 0.0

    p_ref = prices.clone().requires_grad_(True)
    ref = model.encode_sequence(gate_price_window(p_ref) if gate_last else p_ref)[:, -1]
    upstream = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(ref, (p_ref, *model.parameters()), upstream)

    p_act = prices.clone().requires_grad_(True)
    act, _ = model.encode_last(p_act, gate_last=gate_last)
    act_grads = torch.autograd.grad(act, (p_act, *model.parameters()), upstream)

    rtol, atol = (1e-11, 1e-12) if dtype == torch.float64 else (2e-5, 2e-5)
    torch.testing.assert_close(act, ref, rtol=rtol, atol=atol)
    for g_act, g_ref in zip(act_grads, ref_grads, strict=True):
        torch.testing.assert_close(
            g_act, g_ref, rtol=rtol, atol=atol * max(1.0, g_ref.abs().max().item())
        )

    # Determinismo bit a bit
    with torch.no_grad():
        r1, _ = model.encode_last(prices, gate_last=gate_last)
        r2, _ = model.encode_last(prices, gate_last=gate_last)
        assert torch.equal(r1, r2)
    _reset_profile_cache()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("rows", [1, 7, 129])
@pytest.mark.parametrize("channels", [5, 6])
def test_price_returns_and_scale_bit_exact_with_gaps(dtype, rows, channels):
    torch.manual_seed(23)
    w = torch.randn(rows, 64, channels, device=CUDA, dtype=dtype)
    if channels == 6:
        w[..., 5] = (torch.rand(rows, 64, device=CUDA) > 0.25).to(dtype)
        w[:, 0, 5] = 0.0
        w[:, 10:15, 5] = 0.0
        w[:, -1, 5] = 1.0
    cen = torch.randn(5, device=CUDA, dtype=dtype) * 0.2
    inv = torch.rand(5, device=CUDA, dtype=dtype) + 0.5

    ref_ret = price_returns(w)
    ref_scaled = ((ref_ret[..., :5] - cen) * inv).clamp(-5.0, 5.0)
    if channels == 6:
        zero = torch.zeros((), dtype=dtype, device=CUDA)
        ref_scaled = torch.where(w[..., -1:] > 0, ref_scaled, zero)
        ref = torch.cat([ref_scaled, w[..., 5:]], dim=-1)
    else:
        ref = ref_scaled

    act = fused_price_returns_and_scale(w, cen, inv, do_returns=True, do_scale=True, clip=5.0)
    assert torch.equal(act, ref)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("rows", [1, 33, 257])
def test_learning_scale_dense_and_numeric_bit_exact(dtype, rows):
    torch.manual_seed(31)
    pres = torch.rand(rows, 5, device=CUDA) > 0.3
    zero = torch.zeros((), dtype=dtype, device=CUDA)
    for col, width in ((1, 384), (2, 558)):
        x = torch.randn(rows, width, device=CUDA, dtype=dtype)
        cen = torch.randn(width, device=CUDA, dtype=dtype)
        inv = torch.rand(width, device=CUDA, dtype=dtype) + 0.5
        ref = torch.where(pres[:, col : col + 1], ((x - cen) * inv).clamp(-5.0, 5.0), zero)
        act = fused_scale_presence(x, cen, inv, pres, col, clip=5.0)
        assert torch.equal(act, ref)

    for third in (20, 140):
        width = 3 * third
        x = torch.randn(rows, width, device=CUDA, dtype=dtype)
        x[:, third : 2 * third] = (torch.rand(rows, third, device=CUDA) > 0.25).to(dtype)
        cen = torch.randn(width, device=CUDA, dtype=dtype)
        inv = torch.rand(width, device=CUDA, dtype=dtype) + 0.5
        obs = x[:, third : 2 * third]
        sc = ((x - cen) * inv).clamp(-5.0, 5.0)
        ref = torch.cat([sc[:, :third] * obs, obs, sc[:, 2 * third :] * obs], dim=-1)
        act = fused_scale_numeric_triple(x, cen, inv, third=third, clip=5.0)
        assert torch.equal(act, ref)


@pytest.mark.parametrize("profile_name", ["sm_121", "sm_120"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("rows", [1, 37, 257])
def test_flash_cross_asset_both_blackwell_profiles(monkeypatch, profile_name, dtype, rows):
    monkeypatch.setenv(PROFILE_ENV_VAR, profile_name)
    _reset_profile_cache()
    torch.manual_seed(41)
    q = torch.randn(4, rows, 16, device=CUDA, dtype=dtype, requires_grad=True)
    k = torch.randn(4, rows, 16, device=CUDA, dtype=dtype, requires_grad=True)
    v = torch.randn(4, rows, 16, device=CUDA, dtype=dtype, requires_grad=True)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    g = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(ref, (q, k, v), g)

    act = fused_flash_cross_asset(q, k, v)
    act_grads = torch.autograd.grad(act, (q, k, v), g)

    rtol, atol = (1e-12, 1e-12) if dtype == torch.float64 else (2e-5, 2e-5)
    torch.testing.assert_close(act, ref, rtol=rtol, atol=atol)
    for a, b in zip(act_grads, ref_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol * max(1.0, b.abs().max().item()))
    _reset_profile_cache()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_tie_ranks_and_stable_topk_bit_exact_with_adversarial_ties(dtype):
    torch.manual_seed(53)
    # Empates masivos, columnas constantes y posiciones ausentes
    vals = torch.randint(-3, 4, (137, 19), device=CUDA).to(dtype)
    vals[:, 0] = 7.0
    pres = torch.rand(137, 19, device=CUDA) > 0.2
    pres[:, 1] = False
    pres[0, 2] = True
    pres[1:, 2] = False

    # Referencia CUDA (searchsorted + ndtri CUDA) frente a CUDA Triton (bit a bit idéntica)
    inf = torch.full((), torch.inf, dtype=dtype, device=CUDA)
    masked = torch.where(pres, vals, inf).T.contiguous()
    ordered = masked.sort(dim=-1).values
    lower = torch.searchsorted(ordered, masked, right=False)
    upper = torch.searchsorted(ordered, masked, right=True)
    rank = (lower + upper + 1).to(dtype) / 2
    count = pres.sum(dim=0, keepdim=True).T.to(dtype).clamp_min(1)
    prob_ref = torch.where(pres, ((rank - 0.5) / count).T, 0.5)
    ref_cuda = torch.special.ndtri(prob_ref)
    act = gaussian_ranks(vals, pres)
    assert torch.equal(act, ref_cuda)
    # También coincide con la ruta CPU dentro del redondeo de ndtri (libm vs libdevice)
    ref_cpu = gaussian_ranks(vals.cpu(), pres.cpu()).to(CUDA)
    torch.testing.assert_close(act, ref_cpu, rtol=1e-6, atol=1e-6)

    # Top-k descendente estable en FP64 con empates exactos
    scores = torch.randint(-2, 3, (67, 513), device=CUDA).double()
    ref_idx = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :8]
    act_idx = fused_stable_topk_desc(scores, 8)
    assert torch.equal(act_idx, ref_idx)


@pytest.mark.parametrize("profile_name", ["sm_121", "sm_120"])
@pytest.mark.parametrize("layer_norm", [True, False])
def test_fused_write_read_absmax_in_and_out_nan_propagation(monkeypatch, profile_name, layer_norm):
    monkeypatch.setenv(PROFILE_ENV_VAR, profile_name)
    _reset_profile_cache()
    torch.manual_seed(61)
    B, D = 19, 64
    w1 = torch.randn(B, D, D, device=CUDA) * 0.1
    w2 = torch.randn(B, D, D, device=CUDA) * 0.1
    s1 = torch.randn(B, D, D, device=CUDA) * 0.01
    s2 = torch.randn(B, D, D, device=CUDA) * 0.01
    k = torch.nn.functional.normalize(torch.randn(B, D, device=CUDA), dim=-1)
    v = torch.randn(B, D, device=CUDA) * 0.2
    alpha = torch.rand(B, D, 1, device=CUDA) * 0.1
    eta = torch.rand(B, 1, 1, device=CUDA) * 0.5
    theta = torch.rand(B, 1, 1, device=CUDA) * 0.05
    pq = torch.randn(B, D, device=CUDA) * 0.2

    # Verificar paridad de lectura posterior y absmax [8, B] en condiciones finitas
    (w1o, w2o), (s1o, s2o), _, post_r, absmax = fused_associative_write_and_read(
        (w1, w2),
        (s1, s2),
        k,
        v,
        (alpha, eta, theta),
        pq,
        residual_layer_norm=layer_norm,
        eps=1e-5,
        return_input_absmax=True,
    )
    assert absmax.shape == (8, B)
    assert torch.isfinite(absmax).all()
    ref_post = fused_memory_read((w1o, w2o), pq.unsqueeze(1), residual_layer_norm=layer_norm, eps=1e-5).squeeze(1)
    torch.testing.assert_close(post_r, ref_post, rtol=1e-6, atol=1e-6)

    # Inyectar NaN en s2 de entrada -> absmax[7] debe ser +inf (no finito) y conservar orden de error
    s2_bad = s2.clone()
    s2_bad[7, 13, 29] = float("nan")
    with pytest.raises(ValueError, match="El estado contiene NaN o infinito"):
        with deferred_checks():
            slots = [reserve_finite(t, "El estado") for t in (w1, w2, s1, s2_bad)]
            _, _, _, _, absmax_bad = fused_associative_write_and_read(
                (w1, w2),
                (s1, s2_bad),
                k,
                v,
                (alpha, eta, theta),
                pq,
                residual_layer_norm=layer_norm,
                eps=1e-5,
                return_input_absmax=True,
            )
            ind = torch.linalg.vector_norm(absmax_bad, float("inf"), dim=1)
            for i, slot in enumerate(slots):
                fulfill_finite(slot, ind[4 + i])
    _reset_profile_cache()


def test_sm120_and_sm121_zero_spills_audit(monkeypatch):
    """Audita que ningún kernel Triton compilado en sm_121 ni sm_120 tenga spills de registros."""
    from mars_titan.models.titans.fused_cuda_kernels import (
        _fac_price_attn_bwd_kernel,
        _fac_price_attn_fwd_kernel,
        _flash_c5_bwd_dkv_kernel,
        _flash_c5_bwd_dq_kernel,
        _flash_c5_fwd_kernel,
        _price_returns_scale_vec_kernel,
        _scale_numeric_triple_kernel,
        _scale_presence_kernel,
    )
    from mars_titan.models.titans.fused_memory_write import (
        _backward_kernel,
        _forward_kernel,
        _read_bwd_kernel,
        _read_fwd_kernel,
    )

    kernels = (
        _forward_kernel,
        _backward_kernel,
        _read_fwd_kernel,
        _read_bwd_kernel,
        _fac_price_attn_fwd_kernel,
        _fac_price_attn_bwd_kernel,
        _flash_c5_fwd_kernel,
        _flash_c5_bwd_dq_kernel,
        _flash_c5_bwd_dkv_kernel,
        _price_returns_scale_vec_kernel,
        _scale_presence_kernel,
        _scale_numeric_triple_kernel,
    )
    for profile_name in ("sm_121", "sm_120"):
        monkeypatch.setenv(PROFILE_ENV_VAR, profile_name)
        _reset_profile_cache()
        for kern in kernels:
            kern.device_caches[0][0].clear()
        B, D = 128, 64
        w1 = torch.randn(B, D, D, device=CUDA, requires_grad=True)
        w2 = torch.randn(B, D, D, device=CUDA, requires_grad=True)
        s1 = torch.randn(B, D, D, device=CUDA, requires_grad=True)
        s2 = torch.randn(B, D, D, device=CUDA, requires_grad=True)
        k = torch.randn(B, D, device=CUDA, requires_grad=True)
        v = torch.randn(B, D, device=CUDA, requires_grad=True)
        a = torch.rand(B, D, 1, device=CUDA, requires_grad=True) * 0.1
        e = torch.rand(B, 1, 1, device=CUDA, requires_grad=True) * 0.5
        th = torch.rand(B, 1, 1, device=CUDA, requires_grad=True) * 0.05
        pq = torch.randn(B, D, device=CUDA, requires_grad=True)
        (w1o, w2o), (s1o, s2o), _, r, _ = fused_associative_write_and_read(
            (w1, w2), (s1, s2), k, v, (a, e, th), pq,
            residual_layer_norm=True, eps=1e-5, return_input_absmax=True,
        )
        r.sum().backward()
        q = torch.randn(B, 1, D, device=CUDA, requires_grad=True)
        fused_memory_read((w1.detach().requires_grad_(), w2.detach().requires_grad_()), q, residual_layer_norm=True, eps=1e-5).sum().backward()

        m = CompactPriceTransformer(6, context=64, hidden_size=64, heads=4, layers=1, max_batch=256).to(CUDA)
        p = torch.randn(B, 64, 6, device=CUDA)
        p[..., 5] = 1.0
        out, _ = m.encode_last(p)
        out.sum().backward()

        qc = torch.randn(4, 257, 16, device=CUDA, requires_grad=True)
        kc = torch.randn(4, 257, 16, device=CUDA, requires_grad=True)
        vc = torch.randn(4, 257, 16, device=CUDA, requires_grad=True)
        fused_flash_cross_asset(qc, kc, vc).sum().backward()

        cen = torch.zeros(5, device=CUDA)
        inv = torch.ones(5, device=CUDA)
        fused_price_returns_and_scale(p, cen, inv, do_returns=True, do_scale=True, clip=5.0)
        pres = torch.ones((B, 5), dtype=torch.bool, device=CUDA)
        fused_scale_presence(torch.randn(B, 64, device=CUDA), torch.zeros(64, device=CUDA), torch.ones(64, device=CUDA), pres, 1, clip=5.0)
        fused_scale_numeric_triple(torch.randn(B, 60, device=CUDA), torch.zeros(60, device=CUDA), torch.ones(60, device=CUDA), third=20, clip=5.0)

        for kern in kernels:
            entries = list(kern.device_caches[0][0].values())
            assert entries, f"{kern.__name__} no se compiló en {profile_name}"
            for compiled in entries:
                assert compiled.metadata.shared <= SM12X_MAX_SHARED_BYTES
                assert compiled.n_spills == 0, (
                    f"{kern.__name__} FP32 ({profile_name}) tuvo {compiled.n_spills} spills"
                )
    _reset_profile_cache()
