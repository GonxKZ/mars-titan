"""Pruebas TDD del motor de kernels y rendimiento extremo RTX PRO 6000 Blackwell (`sm_120`).

Verifica de forma estricta (en CPU, sin tocar GPU):
  1. Especificaciones y configuraciones `sm_120` (188 SMs, <=100 KB SMEM/CTA, `num_stages` en 2-3,
     128 MB L2, 96 GB GDDR7) sin duplicar `BLACKWELL_SPECS`.
  2. Paridad numérica exacta (`atol=1e-6` FP32 y `atol=1e-12` FP64, cero truncamiento y resistencia
     a cancelación catastrófica) de `fused_cross_sectional_zscore_clip`, `fused_soft_moe_combine`
     y `fused_pinball_crps_loss` + gradientes autograd + cláusulas `sm_120` en los kernels Triton.
  3. `CUDAGraphSessionRunner`: selección de buckets múltiplos de 64/128/188, rellenado real de
     tensores/lotes dict, des-rellenado exacto y paridad en modo eager CPU.
  4. `blackwell_roofline_audit` y `audit_zero_host_syncs`: intensidad aritmética (FLOP/byte),
     clasificación frente a puntos de cresta `sm_120` (75.1 FP32 / 626.0 BF16), alineación de olas
     con 188 SMs, VRAM por worker en MiB, recomendación MPS <= 96 GB y auditoría AST anti-sync.
  5. `generate_rtx_pro_6000_mps_script`: bash sintácticamente válido (`bash -n`) con `sm_120`,
     `TORCH_CUDA_ARCH_LIST=12.0`, demonio CUDA MPS, límites VRAM por worker y olas completas.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from mars_titan.models.mmarrs import (
    BLACKWELL_SPECS,
    DEFAULT_QUANTILES,
    MMARRS_VARIATIONS,
    HighSNRMoELayer,
    build_variation,
    fused_pt1_huber_gate_update,
)
from mars_titan.models.mmarrs import blackwell_kernels as bk
from mars_titan.models.mmarrs.blackwell_kernels import (
    CANONICAL_TRAIN_CLI_FLAGS,
    SM120_KERNEL_CONFIGS,
    SM120_L2_CACHE_BYTES,
    SM120_MAX_SMEM_BYTES,
    SM120_SMS,
    SM120_VRAM_BYTES,
    CUDAGraphSessionRunner,
    FusedCrossSectionalZScoreClipFunction,
    FusedNeuromodUpdateFunction,
    FusedPinballCRPSLossFunction,
    FusedSoftMoECombineFunction,
    audit_zero_host_syncs,
    blackwell_roofline_audit,
    fused_cross_sectional_zscore_clip,
    fused_crps_pinball_loss,
    fused_neuromod_update,
    fused_pinball_crps_loss,
    fused_soft_moe_combine,
    generate_rtx_pro_6000_mps_script,
)
from mars_titan.models.mmarrs.cognitive_core import TitansCognitiveCore


def _ref_zscore_clip(
    x: torch.Tensor, clip_val: float = 8.0, eps: float = 1e-6
) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=0, keepdim=True)
    return torch.clamp(centered * torch.rsqrt(var + eps), min=-clip_val, max=clip_val)


def _ref_soft_moe_combine(
    expert_outputs: torch.Tensor,
    router_probs: torch.Tensor,
    shared_output: torch.Tensor,
    mix_gate: torch.Tensor | float,
    residual: torch.Tensor | None = None,
    *,
    apply_sigmoid: bool = True,
) -> torch.Tensor:
    routed = torch.einsum("ne,ned->nd", router_probs, expert_outputs)
    gate_t = (
        mix_gate.to(device=expert_outputs.device, dtype=expert_outputs.dtype)
        if isinstance(mix_gate, torch.Tensor)
        else torch.tensor(mix_gate, device=expert_outputs.device, dtype=expert_outputs.dtype)
    )
    if gate_t.ndim == 1 and gate_t.shape[0] == expert_outputs.shape[0]:
        gate_t = gate_t.unsqueeze(-1)
    g = torch.sigmoid(gate_t) if apply_sigmoid else gate_t
    out = g * routed + (1.0 - g) * shared_output
    if residual is not None:
        out = residual + out
    return out


def _ref_pinball_crps(
    quantiles: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    *,
    crps_scale: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    t = taus.to(device=quantiles.device, dtype=quantiles.dtype).reshape(
        *([1] * (quantiles.ndim - 1)), -1
    )
    y = targets if targets.ndim == quantiles.ndim else targets.unsqueeze(-1)
    diff = y - quantiles
    pb = torch.maximum(t * diff, (t - 1.0) * diff)
    per_item = crps_scale * pb.mean(dim=-1)
    if reduction == "mean":
        return per_item.mean()
    if reduction == "sum":
        return per_item.sum()
    return per_item


def test_sm120_constants_reuse_blackwell_specs_and_valid_kernel_configs():
    """Constantes `sm_120` reutilizan `BLACKWELL_SPECS` y respetan SMEM <= 100 KB y 188 SMs."""
    assert bk.BLACKWELL_SPECS is BLACKWELL_SPECS
    assert SM120_SMS == 188
    assert SM120_MAX_SMEM_BYTES == 100 * 1024
    assert SM120_L2_CACHE_BYTES == 128 * 1024 * 1024
    assert SM120_VRAM_BYTES == 96 * 1024 * 1024 * 1024

    expected_kernels = {
        "cross_sectional_zscore_clip",
        "soft_moe_combine",
        "pinball_crps_loss",
    }
    assert expected_kernels.issubset(SM120_KERNEL_CONFIGS.keys())
    for name in expected_kernels:
        cfg = SM120_KERNEL_CONFIGS[name]
        assert cfg["num_stages"] in (2, 3)
        assert cfg["num_warps"] in (2, 4, 8)
        assert cfg["estimated_smem_bytes"] <= SM120_MAX_SMEM_BYTES
    assert SM120_KERNEL_CONFIGS["cross_sectional_zscore_clip"]["block_n"] in (128, 256)
    assert SM120_KERNEL_CONFIGS["pinball_crps_loss"]["block_m"] == 64
    assert SM120_KERNEL_CONFIGS["pinball_crps_loss"]["num_warps"] == 2


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize("shape", [(1, 8), (17, 13), (63, 46), (257, 19), (128,)])
def test_fused_cross_sectional_zscore_clip_parity_fp32_fp64(
    dtype: torch.dtype, atol: float, shape: tuple[int, ...]
):
    """Paridad exacta en FP32 y FP64 (sin truncamiento) y formas irregulares."""
    torch.manual_seed(17)
    x = torch.randn(shape, dtype=dtype) * 5.0
    if x.numel() > 2:
        x.view(-1)[0] = 500.0
        x.view(-1)[-1] = -500.0

    out = fused_cross_sectional_zscore_clip(x, clip_val=8.0, eps=1e-6)
    ref = _ref_zscore_clip(x, clip_val=8.0, eps=1e-6)
    assert out.dtype == dtype
    assert out.shape == shape
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)
    assert float(out.abs().max()) <= 8.0 + 1e-12


def test_fused_cross_sectional_zscore_clip_adversarial_cancellation_constant_and_grad():
    """Resiste cancelación catastrófica en FP64, columnas constantes, vacíos y da gradientes."""
    torch.manual_seed(29)
    # Media gigante + ruido pequeño: E[x^2] - E[x]^2 colapsaría, pero centrado/Welford es exacto
    x_adv = torch.full((128, 12), 1.0e7, dtype=torch.float64) + (
        torch.randn(128, 12, dtype=torch.float64) * 1.0e-2
    )
    out_adv = fused_cross_sectional_zscore_clip(x_adv, clip_val=8.0, eps=1e-12)
    ref_adv = _ref_zscore_clip(x_adv, clip_val=8.0, eps=1e-12)
    torch.testing.assert_close(out_adv, ref_adv, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(out_adv).all()

    # Columna constante -> ceros finitos sin NaN/Inf
    x_const = torch.full((32, 4), 42.0, dtype=torch.float64)
    out_const = fused_cross_sectional_zscore_clip(x_const, clip_val=8.0, eps=1e-6)
    torch.testing.assert_close(out_const, torch.zeros_like(x_const), atol=0.0, rtol=0.0)

    # Tensor vacío
    x_empty = torch.empty((0, 6), dtype=torch.float32)
    out_empty = fused_cross_sectional_zscore_clip(x_empty)
    assert out_empty.shape == (0, 6)

    # Gradientes autograd exactos en FP64
    x_g = torch.randn(32, 8, dtype=torch.float64, requires_grad=True)
    x_ref = x_g.detach().clone().requires_grad_(True)
    fused_cross_sectional_zscore_clip(x_g, clip_val=6.0, eps=1e-6).pow(2).sum().backward()
    _ref_zscore_clip(x_ref, clip_val=6.0, eps=1e-6).pow(2).sum().backward()
    assert x_g.grad is not None
    torch.testing.assert_close(x_g.grad, x_ref.grad, atol=1e-10, rtol=1e-10)

    with pytest.raises(ValueError):
        fused_cross_sectional_zscore_clip(x_const, clip_val=0.0)
    with pytest.raises(ValueError):
        fused_cross_sectional_zscore_clip(x_const, eps=0.0)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("apply_sigmoid", [True, False])
def test_fused_soft_moe_combine_parity_fp32_fp64_and_autograd(
    dtype: torch.dtype, atol: float, with_residual: bool, apply_sigmoid: bool
):
    """Paridad exacta de `fused_soft_moe_combine` en FP32/FP64 con/sin residual y gradientes."""
    torch.manual_seed(41)
    n_assets, n_experts, d_model = 53, 6, 37
    exp_out = torch.randn(n_assets, n_experts, d_model, dtype=dtype, requires_grad=True)
    probs = torch.softmax(torch.randn(n_assets, n_experts, dtype=dtype), dim=-1).detach()
    probs.requires_grad_(True)
    shared = torch.randn(n_assets, d_model, dtype=dtype, requires_grad=True)
    gate_val = 0.35 if apply_sigmoid else 0.62
    mix_gate = torch.tensor(gate_val, dtype=dtype, requires_grad=True)
    residual = (
        torch.randn(n_assets, d_model, dtype=dtype, requires_grad=True)
        if with_residual
        else None
    )

    out = fused_soft_moe_combine(
        exp_out,
        probs,
        shared,
        mix_gate,
        residual=residual,
        apply_sigmoid=apply_sigmoid,
    )
    ref = _ref_soft_moe_combine(
        exp_out.detach().requires_grad_(True),
        probs.detach().requires_grad_(True),
        shared.detach().requires_grad_(True),
        mix_gate.detach().requires_grad_(True),
        residual.detach().requires_grad_(True) if residual is not None else None,
        apply_sigmoid=apply_sigmoid,
    )
    assert out.dtype == dtype
    assert out.shape == (n_assets, d_model)
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)

    # Gradientes autograd finitos y exactos
    out.pow(2).mean().backward()
    assert exp_out.grad is not None and torch.isfinite(exp_out.grad).all()
    assert probs.grad is not None and torch.isfinite(probs.grad).all()
    assert shared.grad is not None and torch.isfinite(shared.grad).all()
    assert mix_gate.grad is not None and torch.isfinite(mix_gate.grad).all()
    if residual is not None:
        assert residual.grad is not None and torch.isfinite(residual.grad).all()


def test_fused_soft_moe_combine_matches_high_snr_moe_layer_and_gate_shapes():
    """Coincide exactamente con `HighSNRMoELayer` y soporta `mix_gate` float / `[N,1]` / `[N,D]`."""
    torch.manual_seed(55)
    moe = HighSNRMoELayer(hidden_dim=24, context_dim=8, n_shared_experts=2, n_routed_experts=6)
    x = torch.randn(32, 24)
    ctx = torch.randn(32, 8)
    with torch.no_grad():
        h = moe.in_norm(x)
        s_hid = torch.nn.functional.silu(
            torch.einsum("nd,edf->nef", h, moe.shared_w1) + moe.shared_b1.unsqueeze(0)
        )
        s_out = (
            torch.einsum("nef,efd->ned", s_hid, moe.shared_w2) + moe.shared_b2.unsqueeze(0)
        ).mean(dim=1)
        out_moe, diag = moe(x, ctx)
        r_hid = torch.nn.functional.silu(
            torch.einsum("nd,edf->nef", h, moe.routed_w1) + moe.routed_b1.unsqueeze(0)
        )
        r_all = torch.einsum("nef,efd->ned", r_hid, moe.routed_w2) + moe.routed_b2.unsqueeze(0)
        out_fused = fused_soft_moe_combine(
            r_all, diag["router_probs"], s_out, moe.mix_gate, residual=x
        )
    torch.testing.assert_close(out_fused, out_moe, atol=1e-6, rtol=1e-6)

    # Gate escalar float, 1D [D], 1D [N], [N, 1], [1, D] y [N, D]
    for g in (
        0.4,
        torch.randn(24),
        torch.randn(32),
        torch.randn(32, 1),
        torch.randn(1, 24),
        torch.randn(32, 24),
    ):
        o = fused_soft_moe_combine(r_all, diag["router_probs"], s_out, g, residual=x)
        r = _ref_soft_moe_combine(r_all, diag["router_probs"], s_out, g, residual=x)
        torch.testing.assert_close(o, r, atol=1e-6, rtol=1e-6)

    # Verificar strides exactos del normalizador 2D usado por el kernel Triton (incl. 1D [D])
    g_d, s_gn, s_gd = bk._normalize_moe_gate_2d(torch.randn(24), n_assets=32, d_model=24)
    assert g_d.shape == (1, 24) and s_gn == 0 and s_gd == 1
    g_n, s_gn_n, s_gd_n = bk._normalize_moe_gate_2d(torch.randn(32), n_assets=32, d_model=24)
    assert g_n.shape == (32, 1) and s_gn_n == 1 and s_gd_n == 0
    g_sc, s_gn_s, s_gd_s = bk._normalize_moe_gate_2d(torch.tensor(0.5), n_assets=32, d_model=24)
    assert g_sc.shape == (1, 1) and s_gn_s == 0 and s_gd_s == 0

    with pytest.raises(ValueError):
        fused_soft_moe_combine(r_all, diag["router_probs"][:, :-1], s_out, 0.5)
    with pytest.raises(ValueError):
        fused_soft_moe_combine(r_all, diag["router_probs"], s_out, torch.randn(7))


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_fused_pinball_crps_loss_parity_fp32_fp64_and_autograd(
    dtype: torch.dtype, atol: float, reduction: str
):
    """Paridad exacta de `fused_pinball_crps_loss` 9-cuantiles en FP32/FP64 y gradientes."""
    torch.manual_seed(73)
    n_assets, n_h = 71, 6
    taus = torch.tensor(DEFAULT_QUANTILES, dtype=dtype)
    q = torch.cumsum(torch.rand(n_assets, n_h, 9, dtype=dtype) * 0.01, dim=-1) - 0.045
    q = q.detach().requires_grad_(True)
    y = torch.randn(n_assets, n_h, dtype=dtype) * 0.02

    loss = fused_pinball_crps_loss(q, y, taus, reduction=reduction)
    q_ref = q.detach().clone().requires_grad_(True)
    ref = _ref_pinball_crps(q_ref, y, taus, reduction=reduction)

    assert loss.dtype == dtype
    torch.testing.assert_close(loss, ref, atol=atol, rtol=atol)
    assert torch.all(loss >= 0.0)

    loss.sum().backward()
    ref.sum().backward()
    assert q.grad is not None
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=atol)


def test_fused_pinball_crps_loss_2d_exact_zero_and_validation():
    """Soporta 2D `[N, Q]`, secuencia Python de `taus`, pérdida cero exacta y valida errores."""
    y = torch.linspace(-0.05, 0.05, 29, dtype=torch.float64)
    q_exact = y.unsqueeze(-1).expand(29, 9).clone()
    loss_zero = fused_pinball_crps_loss(q_exact, y, DEFAULT_QUANTILES)
    torch.testing.assert_close(loss_zero, torch.zeros((), dtype=torch.float64), atol=1e-15, rtol=0)

    # Escala crps_scale=1.0 (pinball puro) == mitad de crps_scale=2.0
    q_noisy = q_exact + 0.01
    l_crps = fused_pinball_crps_loss(q_noisy, y, DEFAULT_QUANTILES, crps_scale=2.0)
    l_pb = fused_pinball_crps_loss(q_noisy, y, DEFAULT_QUANTILES, crps_scale=1.0)
    torch.testing.assert_close(l_crps, 2.0 * l_pb, atol=1e-12, rtol=1e-12)

    with pytest.raises(ValueError):
        fused_pinball_crps_loss(q_exact, y, DEFAULT_QUANTILES[:-1])
    with pytest.raises(ValueError):
        fused_pinball_crps_loss(q_exact, y, DEFAULT_QUANTILES, reduction="median")


def test_triton_kernel_sources_enforce_sm120_rules():
    """Los kernels Triton preservan FP64 constexpr, `tl.full` y `eviction_policy="evict_first"`."""
    src = inspect.getsource(bk)
    assert "tl.full([], eps, x.dtype)" in src
    assert 'eviction_policy="evict_first"' in src
    assert "num_stages=2" in src
    assert "BLOCK_N" in src
    # Verificar que constantes FP64 son tl.constexpr (nunca truncadas a fp32 por ABI Triton)
    assert "clip_val: tl.constexpr" in src
    assert "eps: tl.constexpr" in src
    assert "scale_over_q: tl.constexpr" in src
    if bk._HAS_TRITON:
        z_fn = bk._cross_sectional_zscore_clip_kernel
        p_fn = bk._pinball_crps_loss_kernel
        assert "clip_val" in z_fn.constexprs or z_fn.arg_names.index("clip_val") in z_fn.constexprs
        assert "eps" in z_fn.constexprs or z_fn.arg_names.index("eps") in z_fn.constexprs
        assert (
            "scale_over_q" in p_fn.constexprs
            or p_fn.arg_names.index("scale_over_q") in p_fn.constexprs
        )


def test_cuda_graph_session_runner_bucket_padding_and_cpu_eager_parity():
    """`CUDAGraphSessionRunner` alinea buckets (64/128/188), rellena y conserva paridad exacta."""
    torch.manual_seed(91)
    # 1) Selección de buckets múltiplos de 64, 128 y 188 (SMs Blackwell)
    runner_188 = CUDAGraphSessionRunner(lambda x: x * 2.0, bucket_multiple=188)
    assert runner_188.select_bucket(1) == 188
    assert runner_188.select_bucket(188) == 188
    assert runner_188.select_bucket(189) == 376
    assert runner_188.select_bucket(2600) == 2632

    runner_custom = CUDAGraphSessionRunner(
        lambda x: x, buckets=(64, 128, 188, 376), bucket_multiple=188
    )
    assert runner_custom.select_bucket(50) == 64
    assert runner_custom.select_bucket(100) == 128
    assert runner_custom.select_bucket(150) == 188
    assert runner_custom.select_bucket(200) == 376
    assert runner_custom.select_bucket(400) == 564

    # 2) Módulo tensorial: verifica cubeta rellenada con ceros y salida unpadded exacta
    seen_shapes: list[tuple[int, ...]] = []
    mlp = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 8)).eval()

    def _step(x_pad: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(tuple(x_pad.shape))
        return mlp(x_pad)

    runner = CUDAGraphSessionRunner(_step, bucket_multiple=64, allow_unmasked_padding=True)
    for n_assets, expected_bucket in ((17, 64), (64, 64), (101, 128), (189, 192)):
        x = torch.randn(n_assets, 16)
        with torch.no_grad():
            out_runner = runner(x)
            out_direct = mlp(x)
        assert out_runner.shape == (n_assets, 8)
        assert seen_shapes[-1] == (expected_bucket, 16)
        torch.testing.assert_close(out_runner, out_direct, atol=1e-6, rtol=1e-6)

    st = runner.stats()
    assert st["device_mode"] == "cpu_eager"
    assert st["eager_fallbacks"] == 4
    assert st["graph_captures"] == 0
    assert st["last_bucket_n"] == 192
    assert st["bucket_hits"][64] == 2
    assert st["bucket_hits"][128] == 1

    # 3) Lote diccionario con máscara de validez explícita (preserva media transversal exacta!)
    def _dict_step(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_b = batch["prices"]
        mask = batch["valid_mask"].unsqueeze(-1)
        denom = mask.sum(dim=0, keepdim=True).clamp_min(1.0)
        mean_valid = (x_b * mask).sum(dim=0, keepdim=True) / denom
        centered = (x_b - mean_valid) * mask
        reg = batch["regime_vec"]
        return dict(alpha=centered + reg.mean(), global_norm=reg.norm())

    dict_runner = CUDAGraphSessionRunner(
        _dict_step, bucket_multiple=64, pad_mask_key="valid_mask"
    )
    batch = dict(
        prices=torch.randn(37, 10),
        regime_vec=torch.randn(12),
    )
    out_d = dict_runner(batch)
    expected_alpha = (batch["prices"] - batch["prices"].mean(dim=0, keepdim=True)) + batch[
        "regime_vec"
    ].mean()
    assert out_d["alpha"].shape == (37, 10)
    torch.testing.assert_close(out_d["alpha"], expected_alpha, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        out_d["global_norm"], batch["regime_vec"].norm(), atol=1e-6, rtol=1e-6
    )

    # Caso borde: n_assets == len(regime_vec) no debe rellenar `regime_vec` (non_asset_keys)
    batch_same_n = dict(prices=torch.randn(12, 10), regime_vec=torch.randn(12))
    out_same = dict_runner(batch_same_n)
    assert out_same["alpha"].shape == (12, 10)
    torch.testing.assert_close(
        out_same["global_norm"], batch_same_n["regime_vec"].norm(), atol=1e-6, rtol=1e-6
    )

    # Captura estática (`force_clone=True`) + `_copy_into_static` jamás muta tensores del llamante
    x_first = torch.randn(64, 16)
    x_first_snap = x_first.clone()
    static_t = runner._pad_input(x_first, 64, 64, x_first.device, x_first.dtype, force_clone=True)
    assert static_t.data_ptr() != x_first.data_ptr()
    runner._copy_into_static(static_t, torch.ones(17, 16) * 9.0, 17)
    torch.testing.assert_close(x_first, x_first_snap, atol=0.0, rtol=0.0)

    b_first = dict(prices=torch.randn(64, 10), regime_vec=torch.randn(12))
    b_first_p_snap = b_first["prices"].clone()
    b_first_r_snap = b_first["regime_vec"].clone()
    static_b = dict_runner._pad_input(
        b_first, 64, 64, b_first["prices"].device, b_first["prices"].dtype, force_clone=True
    )
    assert static_b["prices"].data_ptr() != b_first["prices"].data_ptr()
    assert static_b["regime_vec"].data_ptr() != b_first["regime_vec"].data_ptr()
    dict_runner._copy_into_static(
        static_b, dict(prices=torch.full((19, 10), 7.0), regime_vec=torch.full((12,), -3.0)), 19
    )
    torch.testing.assert_close(b_first["prices"], b_first_p_snap, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b_first["regime_vec"], b_first_r_snap, atol=0.0, rtol=0.0)

    dict_runner.reset()
    assert dict_runner.stats()["eager_fallbacks"] == 0


def test_blackwell_roofline_audit_and_zero_host_sync_ast_detector(tmp_path: Path):
    """Roofline `sm_120` calcula FLOP/byte, 188 SMs, VRAM MiB, MPS <= 96 GB y detecta syncs AST."""
    model = build_variation(
        "v13_mmarrs_titan_full",
        prices_dim=46,
        fundamentals_dim=32,
        macro_dim=24,
        news_dim=32,
        charts_dim=32,
        physics_dim=24,
        context_dim=12,
        exposures_dim=9,
        rmt_dim=5,
        hidden_dim=64,
    )

    rep_fp32 = blackwell_roofline_audit(model, n_assets=2600, precision_mode="fp32_strict")
    assert rep_fp32["hardware"]["arch"] == "sm_120"
    assert rep_fp32["hardware"]["sms"] == 188
    assert rep_fp32["ridge_point_flop_per_byte"] == pytest.approx(75.1, rel=1e-4)
    assert rep_fp32["forward_flops"] > 0
    assert rep_fp32["forward_bytes"] > 0
    assert rep_fp32["arithmetic_intensity_flop_per_byte"] > 0.0
    assert rep_fp32["bottleneck"] in ("memory_bound", "compute_bound")
    expected_regime = (
        "memory_bound"
        if rep_fp32["arithmetic_intensity_flop_per_byte"] < rep_fp32["ridge_point_flop_per_byte"]
        else "compute_bound"
    )
    assert rep_fp32["bottleneck"] == expected_regime

    wave = rep_fp32["cta_wave_alignment"]
    assert wave["sms"] == 188
    assert 0.0 < wave["row_wave_efficiency"] <= 1.0
    assert wave["recommended_padded_n_188"] % 188 == 0
    assert wave["recommended_padded_n_188"] >= 2600

    vram = rep_fp32["vram_estimate_mib"]
    assert vram["weights_mib"] > 0.0
    assert vram["optimizer_mib"] > 0.0
    assert vram["activations_mib"] > 0.0
    assert vram["total_per_worker_mib"] > vram["weights_mib"]

    mps = rep_fp32["mps_recommendation"]
    assert mps["recommended_workers"] >= 1
    assert mps["estimated_total_vram_gb"] <= 96.0
    assert 0.0 < mps["vram_fraction_per_worker"] <= 1.0
    assert mps["recommended_sessions_per_pack"] >= 1

    # Modo BF16 tiene punto de cresta 626.0 FLOP/byte (1 PFLOPS / 1,597 GB/s)
    rep_bf16 = blackwell_roofline_audit(model, n_assets=2600, precision_mode="bf16_amp")
    assert rep_bf16["ridge_point_flop_per_byte"] == pytest.approx(626.0, rel=1e-4)
    assert rep_bf16["peak_tflops"] == pytest.approx(1000.0, rel=1e-4)

    with pytest.raises(ValueError):
        blackwell_roofline_audit(model, n_assets=2600, precision_mode="int4_invalid")

    # Auditoría AST de trampas host-sync (.item(), .cpu(), .numpy(), synchronize, clip_grad_norm_)
    clean_file = tmp_path / "clean_mod.py"
    clean_file.write_text(
        "import torch\n"
        "class CleanMod:\n"
        "    def forward(self, x):\n"
        "        return torch.tanh(x * 2.0 + 1.0)\n",
        encoding="utf-8",
    )
    clean_audit = audit_zero_host_syncs([clean_file])
    assert clean_audit["files_scanned"] == 1
    assert clean_audit["forward_findings"] == 0
    assert clean_audit["clean"] is True

    dirty_file = tmp_path / "dirty_mod.py"
    dirty_file.write_text(
        "import torch\n"
        "class DirtyMod:\n"
        "    def forward(self, x):\n"
        "        a = x.mean().item()\n"
        "        b = x.cpu().numpy()\n"
        '        c = x.to("cpu")\n'
        "        torch.cuda.synchronize()\n"
        "        torch.nn.utils.clip_grad_norm_([x], 1.0, error_if_nonfinite=True)\n"
        "        return x + a + float(b.sum()) + c.sum()\n",
        encoding="utf-8",
    )
    dirty_audit = audit_zero_host_syncs([dirty_file])
    assert dirty_audit["clean"] is False
    assert dirty_audit["forward_findings"] >= 6
    detected_calls = {f["call"] for f in dirty_audit["findings"]}
    assert ".item()" in detected_calls
    assert ".cpu()" in detected_calls
    assert ".numpy()" in detected_calls
    assert '.to("cpu")' in detected_calls
    assert ".synchronize()" in detected_calls

    # El propio módulo `blackwell_kernels.py` debe tener 0 trampas host-sync en hot paths
    self_audit = audit_zero_host_syncs([Path(bk.__file__)])
    assert self_audit["forward_findings"] == 0
    assert self_audit["is_zero_host_sync_forward"] is True


def test_generate_rtx_pro_6000_mps_script_valid_bash_and_wave_distribution():
    """Genera script bash válido (`bash -n`) con `sm_120`, MPS, topes VRAM y olas completas."""
    script = generate_rtx_pro_6000_mps_script(
        variations=MMARRS_VARIATIONS,
        script_entry="mars_titan.models.mmarrs.full_history_train",
        out_dir="/tmp/mmarrs run out",
        workers=6,
        precision_mode="fp32_strict",
    )
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in script
    assert "sm_120" in script
    assert "TORCH_CUDA_ARCH_LIST=" in script and "12.0" in script
    assert "CMAKE_CUDA_ARCHITECTURES=" in script and "120" in script
    assert "expandable_segments:True,max_split_size_mb:256" in script
    assert "nvidia-cuda-mps-control -d" in script
    assert "MMARRS_VRAM_FRACTION=" in script
    assert "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=" in script
    assert "Wave 1/3" in script
    assert "Wave 2/3" in script
    assert "Wave 3/3" in script
    for var in MMARRS_VARIATIONS:
        assert var in script

    # Validación sintáctica real con `bash -n`
    proc = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, f"bash -n falló: {proc.stderr}"

    # También funciona con ruta a script .py y modo bf16_amp
    script_py = generate_rtx_pro_6000_mps_script(
        variations=("v00_dense_baseline", "v13_mmarrs_titan_full"),
        script_entry="scripts/train_mmarrs.py",
        out_dir="/tmp/out_bf16",
        workers=4,
        precision_mode="bf16_amp",
    )
    assert "scripts/train_mmarrs.py" in script_py
    assert "bf16_amp" in script_py
    proc2 = subprocess.run(["bash", "-n"], input=script_py, text=True, capture_output=True)
    assert proc2.returncode == 0, f"bash -n falló: {proc2.stderr}"

    with pytest.raises(ValueError):
        generate_rtx_pro_6000_mps_script([], "entry.py", "/tmp/out")
    with pytest.raises(ValueError):
        generate_rtx_pro_6000_mps_script(["v00_dense_baseline"], "entry.py", "/tmp/out", workers=0)
    with pytest.raises(ValueError):
        generate_rtx_pro_6000_mps_script(
            ["v00_dense_baseline"], "entry.py", "/tmp/out", precision_mode="bad"
        )


def test_custom_autograd_functions_gradcheck_fp64_and_fp32_parity():
    """Todos los kernels fusionados usan `torch.autograd.Function` y pasan `gradcheck` FP64."""
    for fn_cls in (
        FusedCrossSectionalZScoreClipFunction,
        FusedSoftMoECombineFunction,
        FusedPinballCRPSLossFunction,
        FusedNeuromodUpdateFunction,
    ):
        assert issubclass(fn_cls, torch.autograd.Function)
    assert fused_crps_pinball_loss is fused_pinball_crps_loss

    torch.manual_seed(2026)

    # 1) fused_cross_sectional_zscore_clip: gradcheck FP64 (interior + recortados)
    x_z = (torch.randn(10, 4, dtype=torch.float64) * 1.5).detach().requires_grad_(True)
    y_z = fused_cross_sectional_zscore_clip(x_z, clip_val=2.0, eps=1e-6)
    assert y_z.grad_fn is not None
    assert torch.autograd.gradcheck(
        lambda t: fused_cross_sectional_zscore_clip(t, clip_val=2.0, eps=1e-6),
        (x_z,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    # 2) fused_soft_moe_combine: gradcheck FP64 con todos los tensores y gate 1D [D] / escalar
    exp_o = torch.randn(6, 4, 5, dtype=torch.float64, requires_grad=True)
    probs = torch.softmax(torch.randn(6, 4, dtype=torch.float64), dim=-1).detach()
    probs.requires_grad_(True)
    shared = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    gate_d = torch.randn(5, dtype=torch.float64, requires_grad=True)
    res = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    y_m = fused_soft_moe_combine(exp_o, probs, shared, gate_d, residual=res, apply_sigmoid=True)
    assert y_m.grad_fn is not None
    assert torch.autograd.gradcheck(
        lambda e, p, s, g, r: fused_soft_moe_combine(e, p, s, g, residual=r, apply_sigmoid=True),
        (exp_o, probs, shared, gate_d, res),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    gate_sc = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda e, p, s, g: fused_soft_moe_combine(e, p, s, g, apply_sigmoid=False),
        (exp_o, probs, shared, gate_sc),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    # 3) fused_pinball_crps_loss: gradcheck FP64 en quantiles Y targets (mean, sum, none)
    taus = torch.tensor(DEFAULT_QUANTILES, dtype=torch.float64)
    q_in = torch.randn(5, 3, 9, dtype=torch.float64) * 0.2
    y_in = torch.randn(5, 3, dtype=torch.float64) * 0.2 + 0.5  # lejos de q==y (suave)
    q_in = q_in.detach().requires_grad_(True)
    y_in = y_in.detach().requires_grad_(True)
    for red in ("mean", "sum", "none"):
        loss_p = fused_pinball_crps_loss(q_in, y_in, taus, reduction=red)
        assert loss_p.grad_fn is not None
        assert torch.autograd.gradcheck(
            lambda q_t, y_t, r=red: fused_pinball_crps_loss(q_t, y_t, taus, reduction=r),
            (q_in, y_in),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )

    # 4) fused_neuromod_update: paridad con PT1+Huber + gradcheck FP64 (full + broadcast)
    s = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    m = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    # Mantener |e| lejos del kink exacto +-1.0 para derivadas finitas limpias
    e = torch.tensor(
        [[0.2, -1.8, 0.4, 2.3, -0.5]] * 6, dtype=torch.float64, requires_grad=True
    )
    a = torch.full((6, 1), 0.15, dtype=torch.float64, requires_grad=True)
    et = torch.full((1, 5), 0.80, dtype=torch.float64, requires_grad=True)
    th = torch.tensor(0.12, dtype=torch.float64, requires_grad=True)

    s_new, m_new = fused_neuromod_update(s, m, e, a, et, th, huber_delta=1.0)
    assert s_new.grad_fn is not None
    s_ref, m_ref = fused_pt1_huber_gate_update(
        s.detach(), m.detach(), e.detach(), a.detach(), et.detach(), th.detach(),
        huber_delta=1.0, use_triton=False,
    )
    torch.testing.assert_close(s_new.detach(), s_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(m_new.detach(), m_ref, atol=1e-12, rtol=1e-12)

    assert torch.autograd.gradcheck(
        lambda st, mo, er, al, eta, tht: fused_neuromod_update(
            st, mo, er, al, eta, tht, huber_delta=1.0
        ),
        (s, m, e, a, et, th),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    # Paridad analítica FP32 contra autograd PyTorch puro para fused_neuromod_update
    s32 = torch.randn(16, 12, dtype=torch.float32, requires_grad=True)
    m32 = torch.randn(16, 12, dtype=torch.float32, requires_grad=True)
    e32 = (torch.randn(16, 12, dtype=torch.float32) * 2.5).detach().requires_grad_(True)
    a32 = torch.full((16, 1), 0.12, dtype=torch.float32, requires_grad=True)
    et32 = torch.full((16, 12), 0.85, dtype=torch.float32, requires_grad=True)
    th32 = torch.full((1, 12), 0.10, dtype=torch.float32, requires_grad=True)
    sn, mn = fused_neuromod_update(s32, m32, e32, a32, et32, th32, huber_delta=1.0)
    (sn.pow(2).sum() + 0.5 * mn.pow(2).sum()).backward()

    refs = [t.detach().clone().requires_grad_(True) for t in (s32, m32, e32, a32, et32, th32)]
    m_r = refs[4] * refs[1] + refs[5] * torch.clamp(refs[2], -1.0, 1.0)
    s_r = (1.0 - refs[3]) * refs[0] + m_r
    (s_r.pow(2).sum() + 0.5 * m_r.pow(2).sum()).backward()
    for got, exp in zip((s32, m32, e32, a32, et32, th32), refs, strict=True):
        assert got.grad is not None and exp.grad is not None
        torch.testing.assert_close(got.grad, exp.grad, atol=1e-5, rtol=1e-5)


def test_mps_script_canonical_cli_flags_and_fm8_market_validation():
    """`generate_rtx_pro_6000_mps_script` emite los 8 flags CLI canónicos y bloquea US+CN."""
    assert CANONICAL_TRAIN_CLI_FLAGS == (
        "--variation",
        "--market",
        "--family",
        "--out-dir",
        "--precision",
        "--vram-fraction",
        "--epochs",
        "--seed",
    )
    script = generate_rtx_pro_6000_mps_script(
        variations=("v00_dense_baseline", "v13_mmarrs_titan_full"),
        script_entry="mars_titan.models.mmarrs.full_history_train",
        out_dir="/tmp/mmarrs_cli_test",
        workers=2,
        precision_mode="fp32_strict",
        market="CN",
        family="titans_cognitive_core",
        epochs=15,
        seed=123,
    )
    for flag in CANONICAL_TRAIN_CLI_FLAGS:
        assert flag in script
    assert "--market CN" in script
    assert "--family titans_cognitive_core" in script
    assert "--epochs 15" in script
    assert "--seed 123" in script

    proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert proc.returncode == 0, f"bash -n falló: {proc.stderr}"

    # FM-8: jamás permitir entrenamiento conjunto US+CN
    with pytest.raises(ValueError):
        generate_rtx_pro_6000_mps_script(
            ["v00_dense_baseline"], "entry.py", "/tmp/out", market="US+CN"
        )
    with pytest.raises(ValueError):
        generate_rtx_pro_6000_mps_script(
            ["v00_dense_baseline"], "entry.py", "/tmp/out", epochs=0
        )
    with pytest.raises(ValueError):
        generate_rtx_pro_6000_mps_script(
            ["v00_dense_baseline"], "entry.py", "/tmp/out", seed=-1
        )


def test_blackwell_roofline_audit_recurrence_multiplier_and_sdpa_flops():
    """Roofline contabiliza multiplicador `max_loops`/`thinking_steps` y FLOPs SDPA O(S^2 D)."""
    # 1) Weight-tied TitansCognitiveCore: thinking_steps=3 debe escalar FLOPs frente a K=1
    core_k1 = TitansCognitiveCore(hidden_dim=32, context_dim=8, thinking_steps=1)
    core_k3 = TitansCognitiveCore(hidden_dim=32, context_dim=8, thinking_steps=3)
    rep_k1 = blackwell_roofline_audit(core_k1, n_assets=256, precision_mode="fp32_strict")
    rep_k3 = blackwell_roofline_audit(core_k3, n_assets=256, precision_mode="fp32_strict")
    assert rep_k3["recurrence_audit"]["detected_thinking_steps"] == 3
    assert rep_k3["recurrence_audit"]["recurrence_extra_flops"] > 0
    assert rep_k3["forward_flops"] > rep_k1["forward_flops"] * 1.8

    # 2) MMARRS v13 (HolographicBrain3L SDPA + PCTRMRecursiveLoop max_loops=3)
    m_v13 = build_variation("v13_mmarrs_titan_full", hidden_dim=32)
    rep_v13 = blackwell_roofline_audit(m_v13, n_assets=256, precision_mode="fp32_strict")
    assert rep_v13["recurrence_audit"]["detected_max_loops"] == m_v13.config.max_loops
    assert rep_v13["recurrence_audit"]["detected_max_loops"] == 3
    assert rep_v13["recurrence_audit"]["recurrence_extra_flops"] > 0
    assert rep_v13["sdpa_audit"]["sdpa_modules_detected"] >= 1
    assert rep_v13["sdpa_audit"]["sdpa_flops"] > 0

    # 3) Atención cross-asset O(N^2 D) vía nn.MultiheadAttention: doblar N cuadruplica sdpa_flops
    mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
    rep_n64 = blackwell_roofline_audit(mha, n_assets=64, precision_mode="fp32_strict")
    rep_n128 = blackwell_roofline_audit(mha, n_assets=128, precision_mode="fp32_strict")
    assert rep_n64["sdpa_audit"]["sdpa_flops"] > 0
    ratio = rep_n128["sdpa_audit"]["sdpa_flops"] / rep_n64["sdpa_audit"]["sdpa_flops"]
    assert ratio == pytest.approx(4.0, rel=1e-3)


def test_p0_2_custom_op_zero_graph_breaks_and_neuromod_boundary_parity():
    """P0-2 + P2: las 4 ops usan `torch.library.custom_op` (0 graph breaks) y paridad `<= delta`."""
    # 1) Paridad exacta `<= d_t` en frontera |e| == huber_delta (CPU fallback y CUDA Triton)
    for dev_b in (
        [torch.device("cpu"), torch.device("cuda")]
        if torch.cuda.is_available()
        else [torch.device("cpu")]
    ):
        s = torch.randn(8, 6, device=dev_b, dtype=torch.float32, requires_grad=True)
        m = torch.randn(8, 6, device=dev_b, dtype=torch.float32, requires_grad=True)
        e = torch.tensor(
            [[1.0, -1.0, 0.5, -0.5, 2.0, -2.0]] * 8, device=dev_b, dtype=torch.float32
        )
        e.requires_grad_(True)
        a = torch.full((8, 1), 0.15, device=dev_b, dtype=torch.float32, requires_grad=True)
        et = torch.full((1, 6), 0.80, device=dev_b, dtype=torch.float32, requires_grad=True)
        th = torch.tensor(0.25, device=dev_b, dtype=torch.float32, requires_grad=True)

        sn, mn = fused_neuromod_update(s, m, e, a, et, th, huber_delta=1.0)
        (sn.sum() + mn.sum()).backward()
        # Con gs=1, gm=1 -> g_tot=2, th=0.25 -> |e|<=1.0 recibe 0.5 y |e|>1.0 recibe 0.0
        expected_de = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5, 0.0, 0.0]] * 8, device=dev_b, dtype=torch.float32
        )
        torch.testing.assert_close(e.grad, expected_de, atol=1e-6, rtol=1e-6)

    # 2) torch.compile(mode="default", dynamic=True, fullgraph=True) -> CERO graph breaks!
    taus_t = torch.tensor(DEFAULT_QUANTILES, dtype=torch.float32)

    def _all_four(
        x: torch.Tensor,
        exp_o: torch.Tensor,
        probs: torch.Tensor,
        shared: torch.Tensor,
        gate: torch.Tensor,
        q_pred: torch.Tensor,
        y_tgt: torch.Tensor,
        taus: torch.Tensor,
        st: torch.Tensor,
        mo: torch.Tensor,
        er: torch.Tensor,
        al: torch.Tensor,
        et_t: torch.Tensor,
        th_t: torch.Tensor,
    ) -> torch.Tensor:
        z = fused_cross_sectional_zscore_clip(x, clip_val=6.0, eps=1e-6)
        moe = fused_soft_moe_combine(exp_o, probs, shared, gate, residual=z, apply_sigmoid=True)
        sn2, mn2 = fused_neuromod_update(st, mo, er, al, et_t, th_t, huber_delta=1.0)
        crps = fused_pinball_crps_loss(q_pred, y_tgt, taus, reduction="mean")
        return moe.pow(2).mean() + sn2.pow(2).mean() + mn2.pow(2).mean() + crps

    compiled = torch.compile(_all_four, mode="default", dynamic=True, fullgraph=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n_cur in (32, 48):
        x = torch.randn(n_cur, 16, device=dev, requires_grad=True)
        exp_o = torch.randn(n_cur, 4, 16, device=dev, requires_grad=True)
        probs = (
            torch.softmax(torch.randn(n_cur, 4, device=dev), dim=-1)
            .detach()
            .requires_grad_(True)
        )
        shared = torch.randn(n_cur, 16, device=dev, requires_grad=True)
        gate = torch.randn(16, device=dev, requires_grad=True)
        q_p = torch.randn(n_cur, 9, device=dev, requires_grad=True)
        y_t = torch.randn(n_cur, device=dev)
        st = torch.randn(n_cur, 16, device=dev, requires_grad=True)
        mo = torch.randn(n_cur, 16, device=dev, requires_grad=True)
        er = torch.randn(n_cur, 16, device=dev, requires_grad=True)
        al = torch.full((n_cur, 1), 0.1, device=dev, requires_grad=True)
        et_t = torch.full((1, 16), 0.8, device=dev, requires_grad=True)
        th_t = torch.tensor(0.1, device=dev, requires_grad=True)
        loss = compiled(
            x, exp_o, probs, shared, gate, q_p, y_t, taus_t.to(dev), st, mo, er, al, et_t, th_t
        )
        assert torch.isfinite(loss).all()
        loss.backward()
        for t in (x, exp_o, probs, shared, gate, q_p, st, mo, er, al, et_t, th_t):
            assert t.grad is not None and torch.isfinite(t.grad).all()


def test_p0_2_cuda_graph_runner_prevents_ghost_row_corruption_and_snapshots_buffers():
    """P0-2: sin `pad_mask_key`, CUDAGraphSessionRunner jamás corrompe ops transversales."""
    torch.manual_seed(808)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Sin pad_mask_key ni allow_unmasked_padding: N=2600 en bucket_multiple=188 -> error 0.0!
    x = torch.randn(2600, 32, device=dev) + 3.0
    runner_xs = CUDAGraphSessionRunner(
        lambda t: fused_cross_sectional_zscore_clip(t, clip_val=8.0),
        bucket_multiple=188,
    )
    with torch.no_grad():
        out_r = runner_xs(x)
        out_ref = fused_cross_sectional_zscore_clip(x, clip_val=8.0)
    torch.testing.assert_close(out_r, out_ref, atol=1e-6, rtol=1e-6)

    # 2) Módulo con estado (buffer contador): warmup+capture NO duplica mutación de buffers
    class _CountingMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(16, 16)
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            self.calls.add_(1)
            return self.lin(inp)

    mod = _CountingMod().to(dev).eval()
    runner_mod = CUDAGraphSessionRunner(mod, bucket_multiple=64, warmup_steps=3)
    with torch.no_grad():
        _ = runner_mod(torch.randn(64, 16, device=dev))
        assert int(mod.calls.item()) == 1
        _ = runner_mod(torch.randn(64, 16, device=dev))
        assert int(mod.calls.item()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere CUDA real")
def test_p0_1_p0_2_real_cuda_triton_bf16_fp32_fp64_accuracy_and_zscore_speedup():
    """P0-1 + P0-2 + #87 #2: precisión BF16/FP32/FP64 en CUDA real (`N=2600`) y gradcheck FP64."""
    torch.manual_seed(2026)
    dev = torch.device("cuda")

    # 1) Z-score 2D-coalesced Split-N: FP32, FP64, BF16 y formas irregulares en CUDA
    for dt, atol in ((torch.float32, 1e-5), (torch.float64, 1e-11), (torch.bfloat16, 2e-2)):
        for shp in ((17, 13), (2600, 24), (2600, 256), (128,)):
            x = (torch.randn(shp, device=dev, dtype=dt) * 3.0).detach().requires_grad_(True)
            x_ref_base = x.detach().float() if dt == torch.bfloat16 else x.detach()
            x_ref = x_ref_base.clone().requires_grad_(True)
            y = fused_cross_sectional_zscore_clip(x, clip_val=6.0, eps=1e-6, use_triton=True)
            y_ref = _ref_zscore_clip(x_ref, clip_val=6.0, eps=1e-6).to(dtype=dt)
            torch.testing.assert_close(y, y_ref, atol=atol, rtol=atol)
            go = torch.randn_like(y)
            y.backward(go)
            y_ref.backward(go)
            torch.testing.assert_close(
                x.grad, x_ref.grad.to(dtype=dt), atol=atol * 2.0, rtol=atol * 2.0
            )

    # Gradcheck FP64 en CUDA para los 4 kernels Triton
    xz64 = (torch.randn(16, 8, device=dev, dtype=torch.float64) * 1.5).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda t: fused_cross_sectional_zscore_clip(t, clip_val=2.0, eps=1e-6, use_triton=True),
        (xz64,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    # 2) fused_neuromod_update BF16 con gates broadcast (1,), (N,1), (1,D) a N=2600, D=256:
    #    antes error=8.00; ahora < 1e-2 en BF16 y < 1e-5 en FP32!
    n_a, d_m = 2600, 256
    for dt, g_tol in ((torch.bfloat16, 1e-2), (torch.float32, 1e-5)):
        s = torch.randn(n_a, d_m, device=dev, dtype=dt, requires_grad=True)
        m = torch.randn(n_a, d_m, device=dev, dtype=dt, requires_grad=True)
        e = (torch.randn(n_a, d_m, device=dev, dtype=dt) * 2.0).detach().requires_grad_(True)
        a = torch.full((n_a, 1), 0.15, device=dev, dtype=dt, requires_grad=True)
        et = torch.full((1, d_m), 0.80, device=dev, dtype=dt, requires_grad=True)
        th = torch.tensor(0.12, device=dev, dtype=dt, requires_grad=True)
        gs = torch.randn(n_a, d_m, device=dev, dtype=dt) * 1e-3
        gm = torch.randn(n_a, d_m, device=dev, dtype=dt) * 1e-3

        sn, mn = fused_neuromod_update(s, m, e, a, et, th, huber_delta=1.0, use_triton=True)
        torch.autograd.backward([sn, mn], [gs, gm])

        # Referencia exacta en FP32
        refs = [t.detach().float().clone().requires_grad_(True) for t in (s, m, e, a, et, th)]
        m_r = refs[4] * refs[1] + refs[5] * torch.clamp(refs[2], -1.0, 1.0)
        s_r = (1.0 - refs[3]) * refs[0] + m_r
        torch.autograd.backward([s_r, m_r], [gs.float(), gm.float()])
        for name, got, exp in zip(
            ("s", "m", "e", "a", "et", "th"), (s, m, e, a, et, th), refs, strict=True
        ):
            err = float((got.grad.float() - exp.grad).abs().max().item())
            rel_bound = g_tol * max(1.0, float(exp.grad.abs().max().item()))
            assert err <= rel_bound, f"neuromod {dt} grad_{name} max_err={err} > {rel_bound}"

    # 3) fused_soft_moe_combine BF16 con apply_sigmoid=True a N=2600, E=8, D=256:
    #    antes drift mismo-dtype=0.03125; con APPLY_SIGMOID FP32 en kernel -> 0.0 ULP vs ref BF16!
    exp_o = torch.randn(n_a, 8, d_m, device=dev, dtype=torch.bfloat16, requires_grad=True)
    probs = (
        torch.softmax(torch.randn(n_a, 8, device=dev), dim=-1)
        .to(torch.bfloat16)
        .detach()
        .requires_grad_(True)
    )
    shared = torch.randn(n_a, d_m, device=dev, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn(d_m, device=dev, dtype=torch.bfloat16, requires_grad=True)
    res = torch.randn(n_a, d_m, device=dev, dtype=torch.bfloat16, requires_grad=True)
    out_m = fused_soft_moe_combine(
        exp_o, probs, shared, gate, residual=res, apply_sigmoid=True, use_triton=True
    )

    refs_m = [
        t.detach().float().clone().requires_grad_(True)
        for t in (exp_o, probs, shared, gate, res)
    ]
    ref_m = _ref_soft_moe_combine(*refs_m, apply_sigmoid=True)
    same_dtype_drift = float((out_m.float() - ref_m.to(torch.bfloat16).float()).abs().max().item())
    assert same_dtype_drift <= 1e-2, f"soft_moe BF16 same-dtype drift alto: {same_dtype_drift}"
    go_m = torch.randn_like(out_m) * 1e-2
    out_m.backward(go_m)
    ref_m.backward(go_m.float())
    for name, got, exp in zip(
        ("exp", "probs", "shared", "gate", "res"),
        (exp_o, probs, shared, gate, res),
        refs_m,
        strict=True,
    ):
        err = float((got.grad.float() - exp.grad).abs().max().item())
        bound = 2e-2 * max(1.0, float(exp.grad.abs().max().item()))
        assert err <= bound, f"soft_moe BF16 grad_{name} err={err} > {bound}"

    # Gradcheck FP64 en CUDA para soft_moe, pinball y neuromod
    e64 = torch.randn(6, 4, 8, device=dev, dtype=torch.float64, requires_grad=True)
    p64 = (
        torch.softmax(torch.randn(6, 4, device=dev, dtype=torch.float64), dim=-1)
        .detach()
        .requires_grad_(True)
    )
    s64 = torch.randn(6, 8, device=dev, dtype=torch.float64, requires_grad=True)
    g64 = torch.randn(8, device=dev, dtype=torch.float64, requires_grad=True)
    r64 = torch.randn(6, 8, device=dev, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda e_, p_, s_, g_, r_: fused_soft_moe_combine(
            e_, p_, s_, g_, residual=r_, apply_sigmoid=True, use_triton=True
        ),
        (e64, p64, s64, g64, r64),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    sn64 = torch.randn(6, 8, device=dev, dtype=torch.float64, requires_grad=True)
    mn64 = torch.randn(6, 8, device=dev, dtype=torch.float64, requires_grad=True)
    en64 = torch.tensor(
        [[0.2, -1.8, 0.4, 2.3, -0.5, 0.1, -2.1, 0.3]] * 6,
        device=dev,
        dtype=torch.float64,
        requires_grad=True,
    )
    an64 = torch.full((6, 1), 0.15, device=dev, dtype=torch.float64, requires_grad=True)
    etn64 = torch.full((1, 8), 0.80, device=dev, dtype=torch.float64, requires_grad=True)
    thn64 = torch.tensor(0.12, device=dev, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda st, mo, er, al, et_, th_: fused_neuromod_update(
            st, mo, er, al, et_, th_, huber_delta=1.0, use_triton=True
        ),
        (sn64, mn64, en64, an64, etn64, thn64),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
    q64 = (
        (torch.randn(5, 3, 9, device=dev, dtype=torch.float64) * 0.2).detach().requires_grad_(True)
    )
    y64 = (
        (torch.randn(5, 3, device=dev, dtype=torch.float64) * 0.2 + 0.5)
        .detach()
        .requires_grad_(True)
    )
    t64 = torch.tensor(DEFAULT_QUANTILES, device=dev, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda q_, y_: fused_pinball_crps_loss(q_, y_, t64, reduction="mean", use_triton=True),
        (q64, y64),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )

    # 4) fused_pinball_crps_loss CUDA N=2600, H=6, Q=9
    q_c = torch.randn(n_a, 6, 9, device=dev, dtype=torch.float32, requires_grad=True)
    y_c = torch.randn(n_a, 6, device=dev, dtype=torch.float32, requires_grad=True)
    taus_c = torch.tensor(DEFAULT_QUANTILES, device=dev, dtype=torch.float32)
    l_c = fused_pinball_crps_loss(q_c, y_c, taus_c, reduction="mean", use_triton=True)
    q_cr = q_c.detach().clone().requires_grad_(True)
    y_cr = y_c.detach().clone().requires_grad_(True)
    l_cr = _ref_pinball_crps(q_cr, y_cr, taus_c, reduction="mean")
    torch.testing.assert_close(l_c, l_cr, atol=1e-5, rtol=1e-5)
    l_c.backward()
    l_cr.backward()
    torch.testing.assert_close(q_c.grad, q_cr.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y_c.grad, y_cr.grad, atol=1e-5, rtol=1e-5)


def test_118_122_custom_ops_noncontiguous_opcheck_compile_and_neuromod_alias_free():
    """Regresión `#118`/`#122`: `register_fake` contiguo, `opcheck` no contiguo y 0 aliasing."""
    torch.manual_seed(118122)
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))

    for dev in devs:
        use_tr = dev.type == "cuda"
        n_a, n_e, d_m = 32, 6, 16
        # Tensores NO contiguos idénticos al patrón real de HighSNRMoELayer (.transpose(0, 1))
        x_nc = torch.randn(d_m, n_a, device=dev, dtype=torch.float32).transpose(0, 1)
        x_nc.requires_grad_(True)
        assert not x_nc.is_contiguous()

        exp_nc = torch.randn(n_e, n_a, d_m, device=dev, dtype=torch.float32).transpose(0, 1)
        exp_nc.requires_grad_(True)
        assert not exp_nc.is_contiguous()
        probs_nc = torch.softmax(torch.randn(n_e, n_a, device=dev), dim=0).transpose(0, 1)
        probs_nc = probs_nc.detach().requires_grad_(True)
        shared_nc = torch.randn(d_m, n_a, device=dev).transpose(0, 1).detach().requires_grad_(True)
        gate = torch.randn(d_m, device=dev, requires_grad=True)
        res_nc = torch.randn(d_m, n_a, device=dev).transpose(0, 1).detach().requires_grad_(True)

        q_nc = (
            torch.randn(9, n_a, 6, device=dev)
            .permute(1, 2, 0)
            .detach()
            .requires_grad_(True)
        )
        assert not q_nc.is_contiguous()
        y_nc = torch.randn(6, n_a, device=dev).transpose(0, 1).detach().requires_grad_(True)
        taus = torch.tensor(DEFAULT_QUANTILES, device=dev, dtype=torch.float32)

        st_nc = torch.randn(d_m, n_a, device=dev).transpose(0, 1).detach().requires_grad_(True)
        mo_nc = torch.randn(d_m, n_a, device=dev).transpose(0, 1).detach().requires_grad_(True)
        er_nc = torch.randn(d_m, n_a, device=dev).transpose(0, 1).detach().requires_grad_(True)
        al = torch.full((n_a, 1), 0.15, device=dev, requires_grad=True)
        et = torch.full((1, d_m), 0.80, device=dev, requires_grad=True)
        th = torch.tensor(0.12, device=dev, requires_grad=True)

        # 1) `torch.library.opcheck` sobre las 4 custom_ops con entradas NO contiguas
        torch.library.opcheck(
            torch.ops.mars_titan_bw.zscore_clip_fwd.default,
            (x_nc, 6.0, 1e-6, use_tr),
        )
        torch.library.opcheck(
            torch.ops.mars_titan_bw.soft_moe_combine_fwd.default,
            (exp_nc, probs_nc, shared_nc, gate, res_nc, True, use_tr),
        )
        torch.library.opcheck(
            torch.ops.mars_titan_bw.pinball_crps_fwd.default,
            (q_nc, y_nc, taus, 2.0, "mean", use_tr),
        )
        torch.library.opcheck(
            torch.ops.mars_titan_bw.neuromod_update_fwd.default,
            (st_nc, mo_nc, er_nc, al, et, th, 1.0, use_tr),
        )

        # 2) Regresión `#122`: state/momentum sin gradiente (sesión 0 BPTT), error/gates con grad
        st_ng = torch.zeros(n_a, d_m, device=dev, dtype=torch.float32, requires_grad=False)
        mo_ng = torch.zeros(n_a, d_m, device=dev, dtype=torch.float32, requires_grad=False)
        er_rg = torch.randn(n_a, d_m, device=dev, dtype=torch.float32, requires_grad=True)
        al_rg = torch.full((n_a, 1), 0.15, device=dev, requires_grad=True)
        et_rg = torch.full((1, d_m), 0.80, device=dev, requires_grad=True)
        th_rg = torch.tensor(0.12, device=dev, requires_grad=True)
        sn_ng, mn_ng = fused_neuromod_update(
            st_ng, mo_ng, er_rg, al_rg, et_rg, th_rg, huber_delta=1.0, use_triton=use_tr
        )
        (sn_ng.pow(2).sum() + mn_ng.pow(2).sum()).backward()
        assert st_ng.grad is None and mo_ng.grad is None
        assert er_rg.grad is not None and torch.isfinite(er_rg.grad).all()
        assert al_rg.grad is not None and torch.isfinite(al_rg.grad).all()
        assert et_rg.grad is not None and torch.isfinite(et_rg.grad).all()
        assert th_rg.grad is not None and torch.isfinite(th_rg.grad).all()

        # 3) `torch.compile(mode="default", dynamic=True, fullgraph=True)` con entradas no contiguas
        def _run_nc(
            x_, e_, p_, s_, g_, r_, q_, y_, st_, mo_, er_, al_, et_, th_,
            *, _use_tr: bool = use_tr, _taus: torch.Tensor = taus,
        ):
            z = fused_cross_sectional_zscore_clip(x_, clip_val=6.0, eps=1e-6, use_triton=_use_tr)
            moe = fused_soft_moe_combine(
                e_, p_, s_, g_, residual=r_ + z, apply_sigmoid=True, use_triton=_use_tr
            )
            sn, mn = fused_neuromod_update(
                st_, mo_, er_, al_, et_, th_, huber_delta=1.0, use_triton=_use_tr
            )
            crps = fused_pinball_crps_loss(q_, y_, _taus, reduction="mean", use_triton=_use_tr)
            return moe.pow(2).mean() + sn.pow(2).mean() + mn.pow(2).mean() + crps

        compiled_nc = torch.compile(_run_nc, mode="default", dynamic=True, fullgraph=True)
        loss_nc = compiled_nc(
            x_nc, exp_nc, probs_nc, shared_nc, gate, res_nc,
            q_nc, y_nc, st_nc, mo_nc, er_nc, al, et, th,
        )
        assert torch.isfinite(loss_nc).all()
        loss_nc.backward()
        for t_chk in (x_nc, exp_nc, probs_nc, shared_nc, gate, res_nc, q_nc, y_nc, st_nc, er_nc):
            assert t_chk.grad is not None and torch.isfinite(t_chk.grad).all()

        # 4) Regresión `#132`: `opcheck` mixto BF16/FP32 en `soft_moe_combine_bwd` y `neuromod_fwd`
        exp_bf = torch.randn(n_a, n_e, d_m, device=dev, dtype=torch.bfloat16, requires_grad=True)
        probs_f32 = (
            torch.softmax(torch.randn(n_a, n_e, device=dev), dim=-1)
            .detach()
            .requires_grad_(True)
        )
        shared_f32 = torch.randn(n_a, d_m, device=dev, dtype=torch.float32, requires_grad=True)
        gate_f32 = torch.randn(d_m, device=dev, dtype=torch.float32, requires_grad=True)
        routed_f32 = torch.randn(n_a, d_m, device=dev, dtype=torch.float32)
        go_bf = torch.randn(n_a, d_m, device=dev, dtype=torch.bfloat16)
        torch.library.opcheck(
            torch.ops.mars_titan_bw.soft_moe_combine_fwd.default,
            (exp_bf, probs_f32, shared_f32, gate_f32, None, True, use_tr),
        )
        for need_s_flag in (True, False):
            torch.library.opcheck(
                torch.ops.mars_titan_bw.soft_moe_combine_bwd.default,
                (
                    exp_bf.detach(),
                    probs_f32.detach(),
                    shared_f32.detach(),
                    gate_f32.detach(),
                    routed_f32,
                    go_bf,
                    True,
                    use_tr,
                    True,
                    True,
                    need_s_flag,
                    True,
                ),
            )
        st_bf = torch.randn(n_a, d_m, device=dev, dtype=torch.bfloat16, requires_grad=True)
        mo_f32 = torch.randn(n_a, d_m, device=dev, dtype=torch.float32, requires_grad=True)
        er_f32 = torch.randn(n_a, d_m, device=dev, dtype=torch.float32, requires_grad=True)
        torch.library.opcheck(
            torch.ops.mars_titan_bw.neuromod_update_fwd.default,
            (st_bf, mo_f32, er_f32, al, et, th, 1.0, use_tr),
        )
        torch.library.opcheck(
            torch.ops.mars_titan_bw.neuromod_update_bwd.default,
            (
                st_bf.detach(),
                mo_f32.detach(),
                er_f32.detach(),
                al.detach(),
                et.detach(),
                th.detach(),
                torch.randn_like(st_bf),
                torch.randn_like(mo_f32),
                1.0,
                use_tr,
                False,
                False,
                True,
                True,
                True,
                True,
            ),
        )



