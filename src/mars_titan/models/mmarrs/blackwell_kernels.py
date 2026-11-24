"""Kernels Triton `sm_120`, CUDA Graphs, auditoría Roofline y generador MPS para RTX PRO 6000.

Motor de rendimiento extremo ajustado exactamente al silicio **NVIDIA RTX PRO 6000 Blackwell**
(`sm_120`, die GB202, 188 SMs, 128 KB SMEM/SM [~100 KB/CTA dinámicos máx.], 128 MB L2 monolítica,
96 GB GDDR7 ECC a 1.597–1.792 GB/s, 120 TFLOPS FP32 estrictos / 1 PFLOPS BF16 `mma.sync`):
  1. Kernels Triton fusionados `sm_120` con respaldo exacto en PyTorch (CPU y autograd):
     - `fused_cross_sectional_zscore_clip`: Welford paralelo sin cancelación catastrófica,
       `BLOCK_N=256`, `num_warps=4`, `num_stages=2` (<=100 KB SMEM/CTA) y constantes tipadas
       `tl.full([], eps, x.dtype)` sin truncamiento FP64/FP32.
     - `fused_soft_moe_combine`: combinación `[N, E, D]` + experto compartido + residual en 1 sola
       pasada con `eviction_policy="evict_first"` para preservar la caché L2 de 128 MB.
     - `fused_pinball_crps_loss`: pérdida Pinball/CRPS de 9 cuantiles en registros evitando
       asignaciones intermedias `[N, H, 9]` en HBM.
  2. `CUDAGraphSessionRunner`: captura y reproducción en CUDA Graphs con cubetas rellenadas
     (`bucket_n` múltiplos de 64/128/188) y respaldo transparente eager en CPU.
  3. `blackwell_roofline_audit` + `audit_zero_host_syncs`: intensidad aritmética (FLOP/byte) frente
     a crestas `sm_120` (75.1 FP32 / 626.0 BF16), alineación de olas con 188 SMs, VRAM en MiB,
     recomendación de workers CUDA MPS (<=96 GB) y auditoría AST de trampas host-sync.
  4. `generate_rtx_pro_6000_mps_script`: generador de script bash completo y verificable con
     `TORCH_CUDA_ARCH_LIST=12.0`, `CMAKE_CUDA_ARCHITECTURES=120`, `nvidia-cuda-mps-control -d`,
     topes de VRAM por worker y planificación por olas.
"""

from __future__ import annotations

import ast
import inspect
import math
import shlex
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .hardware_blackwell import (
    BLACKWELL_SPECS,
    PRECISION_MODES,
    blackwell_mps_launch_plan,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - entornos sin Triton instalado
    triton = None
    tl = None
    _HAS_TRITON = False

# --- Constantes físicas exactas de RTX PRO 6000 Blackwell (sm_120 / GB202) ---
SM120_SMS: int = int(BLACKWELL_SPECS["sms"])  # 188 SMs
SM120_MAX_SMEM_BYTES: int = int(BLACKWELL_SPECS["max_dynamic_smem_kb_per_cta"]) * 1024  # 100 KB
SM120_L2_CACHE_BYTES: int = int(BLACKWELL_SPECS["l2_cache_mb"]) * 1024 * 1024  # 128 MB
SM120_VRAM_BYTES: int = int(float(BLACKWELL_SPECS["vram_gb"]) * (1024**3))  # 96 GB GDDR7
SM120_BANDWIDTH_GBS: float = float(BLACKWELL_SPECS["bandwidth_gbs_server"])  # 1,597 GB/s
SM120_TF32_TFLOPS: float = 234.0

CANONICAL_TRAIN_CLI_FLAGS: tuple[str, ...] = (
    "--variation",
    "--market",
    "--family",
    "--out-dir",
    "--precision",
    "--vram-fraction",
    "--epochs",
    "--seed",
)
VALID_MARKETS: tuple[str, ...] = ("US", "CN")

SM120_KERNEL_CONFIGS: dict[str, dict[str, int | str]] = {
    "cross_sectional_zscore_clip": {
        "block_n": 128,
        "block_f": 32,
        "num_warps": 4,
        "num_stages": 2,
        "estimated_smem_bytes": 128 * 32 * 4 * 2,  # 32 KB en FP32 << 100 KB/CTA
        "eviction_policy": "evict_first",
    },
    "soft_moe_combine": {
        "block_d_max": 256,
        "num_warps": 4,
        "num_stages": 2,
        "estimated_smem_bytes": 256 * 8 * 2,  # 4 KB en FP64 << 100 KB/CTA
        "eviction_policy": "evict_first",
    },
    "pinball_crps_loss": {
        "block_m": 64,
        "block_q": 16,
        "num_warps": 2,
        "num_stages": 2,
        "estimated_smem_bytes": 64 * 16 * 8 * 2,  # 16 KB en FP64 << 100 KB/CTA
        "eviction_policy": "evict_first",
    },
    "neuromod_update": {
        "block_n": 32,
        "block_d": 64,
        "num_warps": 4,
        "num_stages": 2,
        "estimated_smem_bytes": 32 * 64 * 4 * 2,  # 16 KB en FP32 << 100 KB/CTA
        "eviction_policy": "evict_first",
    },
}

_TAUS_TENSOR_CACHE: dict[tuple[tuple[float, ...], torch.device, torch.dtype], torch.Tensor] = {}


def _get_cached_taus_tensor(
    taus: Sequence[float], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Devuelve tensor en dispositivo cacheado para `taus` Python sin copias H2D repetidas."""
    key_vals = tuple(float(t) for t in taus)
    if torch.compiler.is_compiling():
        return torch.tensor(key_vals, device=device, dtype=dtype)
    key = (key_vals, device, dtype)
    cached = _TAUS_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    t = torch.tensor(key_vals, device=device, dtype=dtype)
    if len(_TAUS_TENSOR_CACHE) < 64:
        _TAUS_TENSOR_CACHE[key] = t
    return t


# =============================================================================
# 1. Kernels Triton `sm_120` fusionados + custom_ops + respaldos exactos PyTorch
# =============================================================================
if _HAS_TRITON:  # pragma: no cover - ejecutados únicamente en GPU CUDA con Triton

    @triton.jit
    def _cross_sectional_zscore_clip_kernel(
        x_ptr,
        out_ptr,
        mean_ptr,
        inv_std_ptr,
        n_rows,
        n_cols,
        stride_n,
        stride_f,
        clip_val: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
        SAVE_STATS: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Normalización transversal 2D-coalescida `(BLOCK_N, BLOCK_F)` con Welford de Chan."""
        pid_f = tl.program_id(0)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)

        cnt = zero
        mean = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        m2 = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        for n_start in range(0, n_rows, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < n_rows
            mask = n_mask[:, None] & f_mask[None, :]
            x_raw = tl.load(
                x_ptr + n_offs[:, None] * stride_n + f_offs[None, :] * stride_f,
                mask=mask,
                other=0.0,
            )
            x = x_raw.to(ACC_DTYPE)
            cnt_b = tl.sum(n_mask.to(ACC_DTYPE), axis=0)
            safe_cb = tl.where(cnt_b > zero, cnt_b, one)
            mean_b = tl.sum(tl.where(mask, x, zero), axis=0) / safe_cb
            diff_b = tl.where(mask, x - mean_b[None, :], zero)
            m2_b = tl.sum(diff_b * diff_b, axis=0)

            new_cnt = cnt + cnt_b
            safe_nc = tl.where(new_cnt > zero, new_cnt, one)
            delta = mean_b - mean
            mean = tl.where(cnt_b > zero, mean + delta * (cnt_b / safe_nc), mean)
            m2 = tl.where(
                cnt_b > zero,
                m2 + m2_b + delta * delta * (cnt * cnt_b / safe_nc),
                m2,
            )
            cnt = new_cnt

        safe_cnt = tl.where(cnt > zero, cnt, one)
        var = m2 / safe_cnt
        x = mean
        eps_s = tl.full([], eps, x.dtype)
        clip_t = tl.full([], clip_val, x.dtype)
        inv_std_s = one / tl.sqrt(var + eps_s)
        if SAVE_STATS:
            tl.store(mean_ptr + f_offs, mean, mask=f_mask)
            tl.store(inv_std_ptr + f_offs, inv_std_s, mask=f_mask)

        # Pasada 2: reutiliza `inv_std_s` izado fuera del bucle (cero sqrt redundante)
        for n_start in range(0, n_rows, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < n_rows
            mask = n_mask[:, None] & f_mask[None, :]
            x_raw = tl.load(
                x_ptr + n_offs[:, None] * stride_n + f_offs[None, :] * stride_f,
                mask=mask,
                other=0.0,
            )
            x_tile = x_raw.to(ACC_DTYPE)
            z = (x_tile - mean[None, :]) * inv_std_s[None, :]
            z_clip = tl.minimum(tl.maximum(z, -clip_t), clip_t)
            tl.store(
                out_ptr + n_offs[:, None] * stride_n + f_offs[None, :] * stride_f,
                z_clip.to(x_raw.dtype),
                mask=mask,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _cross_sectional_zscore_split_stats_kernel(
        x_ptr,
        part_mean_ptr,
        part_m2_ptr,
        n_rows,
        n_cols,
        stride_n,
        stride_f,
        rows_per_split,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Split-N Pasada 1: estadísticas parciales Chan-Welford 2D-coalescidas (`>=188` CTAs)."""
        pid_f = tl.program_id(0)
        pid_s = tl.program_id(1)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        s_start = pid_s * rows_per_split
        s_end = tl.minimum(s_start + rows_per_split, n_rows)
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)

        cnt = zero
        mean = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        m2 = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        for n_start in range(s_start, s_end, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < s_end
            mask = n_mask[:, None] & f_mask[None, :]
            x = tl.load(
                x_ptr + n_offs[:, None] * stride_n + f_offs[None, :] * stride_f,
                mask=mask,
                other=0.0,
            ).to(ACC_DTYPE)
            cnt_b = tl.sum(n_mask.to(ACC_DTYPE), axis=0)
            safe_cb = tl.where(cnt_b > zero, cnt_b, one)
            mean_b = tl.sum(tl.where(mask, x, zero), axis=0) / safe_cb
            diff_b = tl.where(mask, x - mean_b[None, :], zero)
            m2_b = tl.sum(diff_b * diff_b, axis=0)

            new_cnt = cnt + cnt_b
            safe_nc = tl.where(new_cnt > zero, new_cnt, one)
            delta = mean_b - mean
            mean = tl.where(cnt_b > zero, mean + delta * (cnt_b / safe_nc), mean)
            m2 = tl.where(
                cnt_b > zero,
                m2 + m2_b + delta * delta * (cnt * cnt_b / safe_nc),
                m2,
            )
            cnt = new_cnt

        tl.store(part_mean_ptr + pid_s * n_cols + f_offs, mean, mask=f_mask)
        tl.store(part_m2_ptr + pid_s * n_cols + f_offs, m2, mask=f_mask)

    @triton.jit
    def _cross_sectional_zscore_merge_stats_kernel(
        part_mean_ptr,
        part_m2_ptr,
        mean_ptr,
        inv_std_ptr,
        n_rows,
        n_cols,
        rows_per_split,
        n_splits,
        eps: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Split-N Fusión: combina `n_splits` parciales en `mean` e `inv_std` exactos (1x/col)."""
        pid_f = tl.program_id(0)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)

        cnt = zero
        mean = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        m2 = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        for s in range(0, n_splits):
            st = s * rows_per_split
            en = tl.minimum(st + rows_per_split, n_rows)
            cnt_s = tl.maximum(en - st, 0).to(ACC_DTYPE)
            pm = tl.load(part_mean_ptr + s * n_cols + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
            pm2 = tl.load(part_m2_ptr + s * n_cols + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
            new_cnt = cnt + cnt_s
            safe_nc = tl.where(new_cnt > zero, new_cnt, one)
            delta = pm - mean
            mean = tl.where(cnt_s > zero, mean + delta * (cnt_s / safe_nc), mean)
            m2 = tl.where(
                cnt_s > zero,
                m2 + pm2 + delta * delta * (cnt * cnt_s / safe_nc),
                m2,
            )
            cnt = new_cnt

        safe_cnt = tl.where(cnt > zero, cnt, one)
        var = m2 / safe_cnt
        eps_s = tl.full([], eps, ACC_DTYPE)
        inv_std_s = one / tl.sqrt(var + eps_s)
        tl.store(mean_ptr + f_offs, mean, mask=f_mask)
        tl.store(inv_std_ptr + f_offs, inv_std_s, mask=f_mask)

    @triton.jit
    def _cross_sectional_zscore_apply_kernel(
        x_ptr,
        out_ptr,
        mean_ptr,
        inv_std_ptr,
        n_rows,
        n_cols,
        stride_n,
        stride_f,
        rows_per_split,
        clip_val: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Split-N Pasada 2: normalización + recorte 2D-coalescidos con `inv_std` izado."""
        pid_f = tl.program_id(0)
        pid_s = tl.program_id(1)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        mean = tl.load(mean_ptr + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
        inv_std = tl.load(inv_std_ptr + f_offs, mask=f_mask, other=1.0).to(ACC_DTYPE)
        clip_t = tl.full([], clip_val, ACC_DTYPE)

        s_start = pid_s * rows_per_split
        s_end = tl.minimum(s_start + rows_per_split, n_rows)
        for n_start in range(s_start, s_end, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < s_end
            mask = n_mask[:, None] & f_mask[None, :]
            x_raw = tl.load(
                x_ptr + n_offs[:, None] * stride_n + f_offs[None, :] * stride_f,
                mask=mask,
                other=0.0,
            )
            x = x_raw.to(ACC_DTYPE)
            z = (x - mean[None, :]) * inv_std[None, :]
            z_clip = tl.minimum(tl.maximum(z, -clip_t), clip_t)
            tl.store(
                out_ptr + n_offs[:, None] * stride_n + f_offs[None, :] * stride_f,
                z_clip.to(x_raw.dtype),
                mask=mask,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _cross_sectional_zscore_clip_bwd_kernel(
        x_ptr,
        grad_out_ptr,
        mean_ptr,
        inv_std_ptr,
        grad_x_ptr,
        n_rows,
        n_cols,
        stride_n,
        stride_f,
        clip_val: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Backward analítico 2D-coalescido de `zscore + clip` para `n_splits == 1`."""
        pid_f = tl.program_id(0)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)
        mean = tl.load(mean_ptr + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
        inv_std = tl.load(inv_std_ptr + f_offs, mask=f_mask, other=1.0).to(ACC_DTYPE)
        clip_t = tl.full([], clip_val, ACC_DTYPE)

        sum_u = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        sum_uz = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        cnt = zero
        for n_start in range(0, n_rows, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < n_rows
            mask = n_mask[:, None] & f_mask[None, :]
            idx = n_offs[:, None] * stride_n + f_offs[None, :] * stride_f
            x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
            dy = tl.load(grad_out_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
            z = (x - mean[None, :]) * inv_std[None, :]
            in_bounds = (z > -clip_t) & (z < clip_t)
            u = tl.where(mask & in_bounds, dy, zero)
            sum_u = sum_u + tl.sum(u, axis=0)
            sum_uz = sum_uz + tl.sum(u * z, axis=0)
            cnt = cnt + tl.sum(n_mask.to(ACC_DTYPE), axis=0)

        safe_cnt = tl.where(cnt > zero, cnt, one)
        mean_u = sum_u / safe_cnt
        mean_uz = sum_uz / safe_cnt

        for n_start in range(0, n_rows, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < n_rows
            mask = n_mask[:, None] & f_mask[None, :]
            idx = n_offs[:, None] * stride_n + f_offs[None, :] * stride_f
            x_raw = tl.load(x_ptr + idx, mask=mask, other=0.0)
            x = x_raw.to(ACC_DTYPE)
            dy = tl.load(grad_out_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
            z = (x - mean[None, :]) * inv_std[None, :]
            in_bounds = (z > -clip_t) & (z < clip_t)
            u = tl.where(in_bounds, dy, zero)
            dx = inv_std[None, :] * (u - mean_u[None, :] - z * mean_uz[None, :])
            tl.store(
                grad_x_ptr + idx,
                dx.to(x_raw.dtype),
                mask=mask,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _cross_sectional_zscore_bwd_split_kernel(
        x_ptr,
        grad_out_ptr,
        mean_ptr,
        inv_std_ptr,
        part_u_ptr,
        part_uz_ptr,
        n_rows,
        n_cols,
        stride_n,
        stride_f,
        rows_per_split,
        clip_val: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Split-N Backward Pasada 1: sumas parciales `u` y `u*z` 2D-coalescidas."""
        pid_f = tl.program_id(0)
        pid_s = tl.program_id(1)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        zero = tl.full([], 0.0, ACC_DTYPE)
        mean = tl.load(mean_ptr + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
        inv_std = tl.load(inv_std_ptr + f_offs, mask=f_mask, other=1.0).to(ACC_DTYPE)
        clip_t = tl.full([], clip_val, ACC_DTYPE)

        sum_u = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        sum_uz = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        s_start = pid_s * rows_per_split
        s_end = tl.minimum(s_start + rows_per_split, n_rows)
        for n_start in range(s_start, s_end, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < s_end
            mask = n_mask[:, None] & f_mask[None, :]
            idx = n_offs[:, None] * stride_n + f_offs[None, :] * stride_f
            x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
            dy = tl.load(grad_out_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
            z = (x - mean[None, :]) * inv_std[None, :]
            in_bounds = (z > -clip_t) & (z < clip_t)
            u = tl.where(mask & in_bounds, dy, zero)
            sum_u = sum_u + tl.sum(u, axis=0)
            sum_uz = sum_uz + tl.sum(u * z, axis=0)

        tl.store(part_u_ptr + pid_s * n_cols + f_offs, sum_u, mask=f_mask)
        tl.store(part_uz_ptr + pid_s * n_cols + f_offs, sum_uz, mask=f_mask)

    @triton.jit
    def _cross_sectional_zscore_bwd_merge_kernel(
        part_u_ptr,
        part_uz_ptr,
        mean_u_ptr,
        mean_uz_ptr,
        n_rows,
        n_cols,
        n_splits,
        BLOCK_F: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Split-N Backward Fusión: reduce `part_u`/`part_uz` en `mean_u`/`mean_uz`."""
        pid_f = tl.program_id(0)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        sum_u = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        sum_uz = tl.zeros([BLOCK_F], dtype=ACC_DTYPE)
        for s in range(0, n_splits):
            sum_u = sum_u + tl.load(part_u_ptr + s * n_cols + f_offs, mask=f_mask, other=0.0).to(
                ACC_DTYPE
            )
            sum_uz = sum_uz + tl.load(
                part_uz_ptr + s * n_cols + f_offs, mask=f_mask, other=0.0
            ).to(ACC_DTYPE)
        inv_n = tl.full([], 1.0, ACC_DTYPE) / tl.maximum(
            n_rows.to(ACC_DTYPE), tl.full([], 1.0, ACC_DTYPE)
        )
        tl.store(mean_u_ptr + f_offs, sum_u * inv_n, mask=f_mask)
        tl.store(mean_uz_ptr + f_offs, sum_uz * inv_n, mask=f_mask)

    @triton.jit
    def _cross_sectional_zscore_bwd_apply_kernel(
        x_ptr,
        grad_out_ptr,
        mean_ptr,
        inv_std_ptr,
        mean_u_ptr,
        mean_uz_ptr,
        grad_x_ptr,
        n_rows,
        n_cols,
        stride_n,
        stride_f,
        rows_per_split,
        clip_val: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_F: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Split-N Backward Pasada 2: escribe `grad_x` 2D-coalescido (`>=188` CTAs)."""
        pid_f = tl.program_id(0)
        pid_s = tl.program_id(1)
        f_offs = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = f_offs < n_cols
        zero = tl.full([], 0.0, ACC_DTYPE)
        mean = tl.load(mean_ptr + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
        inv_std = tl.load(inv_std_ptr + f_offs, mask=f_mask, other=1.0).to(ACC_DTYPE)
        mean_u = tl.load(mean_u_ptr + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
        mean_uz = tl.load(mean_uz_ptr + f_offs, mask=f_mask, other=0.0).to(ACC_DTYPE)
        clip_t = tl.full([], clip_val, ACC_DTYPE)

        s_start = pid_s * rows_per_split
        s_end = tl.minimum(s_start + rows_per_split, n_rows)
        for n_start in range(s_start, s_end, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < s_end
            mask = n_mask[:, None] & f_mask[None, :]
            idx = n_offs[:, None] * stride_n + f_offs[None, :] * stride_f
            x_raw = tl.load(x_ptr + idx, mask=mask, other=0.0)
            x = x_raw.to(ACC_DTYPE)
            dy = tl.load(grad_out_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
            z = (x - mean[None, :]) * inv_std[None, :]
            in_bounds = (z > -clip_t) & (z < clip_t)
            u = tl.where(in_bounds, dy, zero)
            dx = inv_std[None, :] * (u - mean_u[None, :] - z * mean_uz[None, :])
            tl.store(
                grad_x_ptr + idx,
                dx.to(x_raw.dtype),
                mask=mask,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _soft_moe_combine_kernel(
        exp_ptr,
        probs_ptr,
        shared_ptr,
        gate_ptr,
        res_ptr,
        out_ptr,
        routed_out_ptr,
        d_model,
        stride_en,
        stride_ee,
        stride_ed,
        stride_pn,
        stride_pe,
        stride_sn,
        stride_sd,
        stride_gn,
        stride_gd,
        stride_rn,
        stride_rd,
        stride_on,
        stride_od,
        stride_ron,
        stride_rod,
        N_EXPERTS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        APPLY_SIGMOID: tl.constexpr,
        SAVE_ROUTED: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Combina `[N,E,D]` + `shared` + `residual` con sigmoid FP32 en kernel y `E` constexpr."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_model
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)

        acc = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)
        for e in tl.static_range(0, N_EXPERTS):
            p_e = tl.load(probs_ptr + pid_n * stride_pn + e * stride_pe).to(ACC_DTYPE)
            e_val = tl.load(
                exp_ptr + pid_n * stride_en + e * stride_ee + d_offs * stride_ed,
                mask=d_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(ACC_DTYPE)
            acc = acc + p_e * e_val

        shared_raw = tl.load(
            shared_ptr + pid_n * stride_sn + d_offs * stride_sd,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        shared_val = shared_raw.to(ACC_DTYPE)
        if SAVE_ROUTED:
            tl.store(
                routed_out_ptr + pid_n * stride_ron + d_offs * stride_rod,
                acc,
                mask=d_mask,
                eviction_policy="evict_first",
            )

        g_ptrs = gate_ptr + pid_n * stride_gn + d_offs * stride_gd
        g_val = tl.load(g_ptrs, mask=d_mask, other=0.0).to(ACC_DTYPE)
        if APPLY_SIGMOID:
            g_pos = one / (one + tl.exp(-g_val))
            exp_g = tl.exp(g_val)
            g_neg = exp_g / (one + exp_g)
            g_val = tl.where(g_val >= zero, g_pos, g_neg)

        out = g_val * acc + (one - g_val) * shared_val
        if HAS_RESIDUAL:
            res_val = tl.load(
                res_ptr + pid_n * stride_rn + d_offs * stride_rd,
                mask=d_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(ACC_DTYPE)
            out = res_val + out

        tl.store(
            out_ptr + pid_n * stride_on + d_offs * stride_od,
            out.to(shared_raw.dtype),
            mask=d_mask,
        )

    @triton.jit
    def _soft_moe_combine_bwd_kernel(
        exp_ptr,
        probs_ptr,
        shared_ptr,
        gate_raw_ptr,
        routed_ptr,
        grad_out_ptr,
        grad_exp_ptr,
        grad_probs_part_ptr,
        grad_shared_ptr,
        grad_gate_part_ptr,
        d_model,
        n_d_tiles,
        stride_en,
        stride_ee,
        stride_ed,
        stride_pn,
        stride_pe,
        stride_sn,
        stride_sd,
        stride_gn,
        stride_gd,
        stride_ron,
        stride_rod,
        stride_on,
        stride_od,
        N_EXPERTS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        APPLY_SIGMOID: tl.constexpr,
        NEED_EXP_GRAD: tl.constexpr,
        NEED_PROBS_GRAD: tl.constexpr,
        NEED_SHARED_GRAD: tl.constexpr,
        NEED_GATE_GRAD: tl.constexpr,
        GATE_REDUCE_D: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Backward fusionado 1-pasada para grad_exp, grad_shared, grad_probs y grad_gate."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_model
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)

        go_raw = tl.load(
            grad_out_ptr + pid_n * stride_on + d_offs * stride_od,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        go = go_raw.to(ACC_DTYPE)
        g_raw = tl.load(
            gate_raw_ptr + pid_n * stride_gn + d_offs * stride_gd, mask=d_mask, other=0.0
        ).to(ACC_DTYPE)
        if APPLY_SIGMOID:
            g_pos = one / (one + tl.exp(-g_raw))
            exp_g = tl.exp(g_raw)
            g_neg = exp_g / (one + exp_g)
            g_val = tl.where(g_raw >= zero, g_pos, g_neg)
        else:
            g_val = g_raw

        if NEED_SHARED_GRAD:
            g_sh = (one - g_val) * go
            tl.store(
                grad_shared_ptr + pid_n * stride_on + d_offs * stride_od,
                g_sh.to(go_raw.dtype),
                mask=d_mask,
                eviction_policy="evict_first",
            )

        if NEED_EXP_GRAD or NEED_PROBS_GRAD:
            g_routed = g_val * go
            for e in tl.static_range(0, N_EXPERTS):
                if NEED_EXP_GRAD:
                    p_e = tl.load(probs_ptr + pid_n * stride_pn + e * stride_pe).to(ACC_DTYPE)
                    ge_val = p_e * g_routed
                    tl.store(
                        grad_exp_ptr + pid_n * stride_en + e * stride_ee + d_offs * stride_ed,
                        ge_val.to(go_raw.dtype),
                        mask=d_mask,
                        eviction_policy="evict_first",
                    )
                if NEED_PROBS_GRAD:
                    e_val = tl.load(
                        exp_ptr + pid_n * stride_en + e * stride_ee + d_offs * stride_ed,
                        mask=d_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(ACC_DTYPE)
                    gp_e = tl.sum(tl.where(d_mask, e_val * g_routed, zero), axis=0)
                    tl.store(
                        grad_probs_part_ptr + (pid_n * n_d_tiles + pid_d) * N_EXPERTS + e,
                        gp_e,
                    )

        if NEED_GATE_GRAD:
            routed_val = tl.load(
                routed_ptr + pid_n * stride_ron + d_offs * stride_rod,
                mask=d_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(ACC_DTYPE)
            shared_val = tl.load(
                shared_ptr + pid_n * stride_sn + d_offs * stride_sd,
                mask=d_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(ACC_DTYPE)
            dg = go * (routed_val - shared_val)
            if APPLY_SIGMOID:
                dg = dg * g_val * (one - g_val)
            if GATE_REDUCE_D:
                dg_sum = tl.sum(tl.where(d_mask, dg, zero), axis=0)
                tl.store(grad_gate_part_ptr + pid_n * n_d_tiles + pid_d, dg_sum)
            else:
                tl.store(
                    grad_gate_part_ptr + pid_n * d_model + d_offs,
                    dg,
                    mask=d_mask,
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _pinball_crps_loss_kernel(
        q_ptr,
        y_ptr,
        taus_ptr,
        out_ptr,
        n_rows,
        stride_qm,
        stride_qq,
        scale_over_q: tl.constexpr,
        N_QUANTILES: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_M: tl.constexpr,
        REDUCE_BLOCK: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Pinball/CRPS 2D-coalescido `(BLOCK_M=64, BLOCK_Q)` con `N_QUANTILES` `tl.constexpr`."""
        pid = tl.program_id(0)
        m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < n_rows
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)

        y_raw = tl.load(
            y_ptr + m_offs, mask=m_mask, other=0.0, eviction_policy="evict_first"
        )
        y_val = y_raw.to(ACC_DTYPE)
        row_acc = tl.zeros([BLOCK_M], dtype=ACC_DTYPE)

        for q_start in tl.static_range(0, N_QUANTILES, BLOCK_Q):
            q_offs = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offs < N_QUANTILES
            tile_mask = m_mask[:, None] & q_mask[None, :]
            q_tile = tl.load(
                q_ptr + m_offs[:, None] * stride_qm + q_offs[None, :] * stride_qq,
                mask=tile_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(ACC_DTYPE)
            taus_v = tl.load(taus_ptr + q_offs, mask=q_mask, other=0.0).to(ACC_DTYPE)
            diff = y_val[:, None] - q_tile
            pb = tl.maximum(taus_v[None, :] * diff, (taus_v[None, :] - one) * diff)
            row_acc = row_acc + tl.sum(tl.where(q_mask[None, :], pb, zero), axis=1)

        scale_t = tl.full([], scale_over_q, row_acc.dtype)
        row_crps = row_acc * scale_t

        if REDUCE_BLOCK:
            blk_sum = tl.sum(tl.where(m_mask, row_crps, zero), axis=0)
            tl.store(out_ptr + pid, blk_sum)
        else:
            tl.store(
                out_ptr + m_offs,
                row_crps.to(y_raw.dtype),
                mask=m_mask,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _pinball_crps_loss_bwd_kernel(
        q_ptr,
        y_ptr,
        taus_ptr,
        grad_out_ptr,
        grad_q_ptr,
        grad_y_ptr,
        n_rows,
        stride_qm,
        stride_qq,
        scale_over_q: tl.constexpr,
        N_QUANTILES: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_M: tl.constexpr,
        IS_SCALAR_GRAD: tl.constexpr,
        NEED_Q_GRAD: tl.constexpr,
        NEED_Y_GRAD: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Backward fusionado 2D-coalescido `(BLOCK_M=64, BLOCK_Q)` de Pinball/CRPS."""
        pid = tl.program_id(0)
        m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < n_rows
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)
        half = tl.full([], 0.5, ACC_DTYPE)
        scale_t = tl.full([], scale_over_q, ACC_DTYPE)

        y_raw = tl.load(
            y_ptr + m_offs, mask=m_mask, other=0.0, eviction_policy="evict_first"
        )
        y_val = y_raw.to(ACC_DTYPE)
        if IS_SCALAR_GRAD:
            go_val = tl.load(grad_out_ptr).to(ACC_DTYPE)
            c_m = go_val * scale_t
        else:
            go_vec = tl.load(
                grad_out_ptr + m_offs, mask=m_mask, other=0.0, eviction_policy="evict_first"
            ).to(ACC_DTYPE)
            c_m = go_vec * scale_t

        y_grad_acc = tl.zeros([BLOCK_M], dtype=ACC_DTYPE)
        for q_start in tl.static_range(0, N_QUANTILES, BLOCK_Q):
            q_offs = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offs < N_QUANTILES
            tile_mask = m_mask[:, None] & q_mask[None, :]
            q_tile = tl.load(
                q_ptr + m_offs[:, None] * stride_qm + q_offs[None, :] * stride_qq,
                mask=tile_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(ACC_DTYPE)
            taus_v = tl.load(taus_ptr + q_offs, mask=q_mask, other=0.0).to(ACC_DTYPE)
            diff = y_val[:, None] - q_tile
            t_b = taus_v[None, :]
            s = tl.where(diff > zero, t_b, tl.where(diff < zero, t_b - one, t_b - half))
            if NEED_Q_GRAD:
                dq = (-c_m[:, None] * s) if (not IS_SCALAR_GRAD) else (-c_m * s)
                tl.store(
                    grad_q_ptr + m_offs[:, None] * stride_qm + q_offs[None, :] * stride_qq,
                    dq.to(y_raw.dtype),
                    mask=tile_mask,
                    eviction_policy="evict_first",
                )
            if NEED_Y_GRAD:
                y_grad_acc = y_grad_acc + tl.sum(tl.where(q_mask[None, :], s, zero), axis=1)

        if NEED_Y_GRAD:
            dy = c_m * y_grad_acc
            tl.store(
                grad_y_ptr + m_offs,
                dy.to(y_raw.dtype),
                mask=m_mask,
                eviction_policy="evict_first",
            )

    @triton.jit
    def _neuromod_update_fwd_kernel(
        state_ptr,
        mom_ptr,
        err_ptr,
        alpha_ptr,
        eta_ptr,
        theta_ptr,
        out_s_ptr,
        out_m_ptr,
        n_rows,
        n_cols,
        stride_sn,
        stride_sd,
        stride_an,
        stride_ad,
        stride_etn,
        stride_etd,
        stride_thn,
        stride_thd,
        huber_delta: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Actualización PT1 + Huber 2D con strides broadcast sin copias `[N, D]`."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (n_offs[:, None] < n_rows) & (d_offs[None, :] < n_cols)
        idx = n_offs[:, None] * stride_sn + d_offs[None, :] * stride_sd

        s_raw = tl.load(state_ptr + idx, mask=mask, other=0.0)
        s = s_raw.to(ACC_DTYPE)
        m = tl.load(mom_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
        e = tl.load(err_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(
            ACC_DTYPE
        )
        a = tl.load(
            alpha_ptr + n_offs[:, None] * stride_an + d_offs[None, :] * stride_ad,
            mask=mask,
            other=0.0,
        ).to(ACC_DTYPE)
        et = tl.load(
            eta_ptr + n_offs[:, None] * stride_etn + d_offs[None, :] * stride_etd,
            mask=mask,
            other=0.0,
        ).to(ACC_DTYPE)
        th = tl.load(
            theta_ptr + n_offs[:, None] * stride_thn + d_offs[None, :] * stride_thd,
            mask=mask,
            other=0.0,
        ).to(ACC_DTYPE)

        d_t = tl.full([], huber_delta, e.dtype)
        one = tl.full([], 1.0, s.dtype)
        e_clip = tl.minimum(tl.maximum(e, -d_t), d_t)
        m_new = et * m + th * e_clip
        s_new = (one - a) * s + m_new
        tl.store(out_s_ptr + idx, s_new.to(s_raw.dtype), mask=mask)
        tl.store(out_m_ptr + idx, m_new.to(s_raw.dtype), mask=mask)

    @triton.jit
    def _neuromod_update_bwd_kernel(
        state_ptr,
        mom_ptr,
        err_ptr,
        alpha_ptr,
        eta_ptr,
        theta_ptr,
        grad_s_ptr,
        grad_m_ptr,
        d_s_ptr,
        d_m_ptr,
        d_e_ptr,
        d_a_ptr,
        d_et_ptr,
        d_th_ptr,
        n_rows,
        n_cols,
        n_d_tiles,
        stride_sn,
        stride_sd,
        stride_an,
        stride_ad,
        stride_etn,
        stride_etd,
        stride_thn,
        stride_thd,
        huber_delta: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NEED_S_GRAD: tl.constexpr,
        NEED_M_GRAD: tl.constexpr,
        NEED_E_GRAD: tl.constexpr,
        NEED_A_GRAD: tl.constexpr,
        NEED_ET_GRAD: tl.constexpr,
        NEED_TH_GRAD: tl.constexpr,
        A_MODE: tl.constexpr,
        ET_MODE: tl.constexpr,
        TH_MODE: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """Backward fusionado 2D con reducción FP32 de gates broadcast y paridad `<= d_t`."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        n_mask = n_offs < n_rows
        d_mask = d_offs < n_cols
        mask = n_mask[:, None] & d_mask[None, :]
        idx = n_offs[:, None] * stride_sn + d_offs[None, :] * stride_sd

        s_raw = tl.load(state_ptr + idx, mask=mask, other=0.0)
        gs = tl.load(grad_s_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(
            ACC_DTYPE
        )
        gm = tl.load(grad_m_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(
            ACC_DTYPE
        )
        zero = tl.full([], 0.0, ACC_DTYPE)
        one = tl.full([], 1.0, ACC_DTYPE)
        g_tot = gs + gm

        if NEED_S_GRAD:
            a = tl.load(
                alpha_ptr + n_offs[:, None] * stride_an + d_offs[None, :] * stride_ad,
                mask=mask,
                other=0.0,
            ).to(ACC_DTYPE)
            d_s = (one - a) * gs
            tl.store(d_s_ptr + idx, d_s.to(s_raw.dtype), mask=mask, eviction_policy="evict_first")

        if NEED_M_GRAD:
            et = tl.load(
                eta_ptr + n_offs[:, None] * stride_etn + d_offs[None, :] * stride_etd,
                mask=mask,
                other=0.0,
            ).to(ACC_DTYPE)
            d_m = et * g_tot
            tl.store(d_m_ptr + idx, d_m.to(s_raw.dtype), mask=mask, eviction_policy="evict_first")

        if NEED_E_GRAD or NEED_TH_GRAD:
            e = tl.load(err_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(
                ACC_DTYPE
            )
            d_t = tl.full([], huber_delta, e.dtype)
            if NEED_E_GRAD:
                th = tl.load(
                    theta_ptr + n_offs[:, None] * stride_thn + d_offs[None, :] * stride_thd,
                    mask=mask,
                    other=0.0,
                ).to(ACC_DTYPE)
                in_huber = (e >= -d_t) & (e <= d_t)
                d_e = tl.where(in_huber, th * g_tot, zero)
                tl.store(
                    d_e_ptr + idx, d_e.to(s_raw.dtype), mask=mask, eviction_policy="evict_first"
                )
            if NEED_TH_GRAD:
                e_clip = tl.minimum(tl.maximum(e, -d_t), d_t)
                d_th = tl.where(mask, e_clip * g_tot, zero)
                if TH_MODE == 0:
                    tl.store(d_th_ptr + idx, d_th, mask=mask, eviction_policy="evict_first")
                elif TH_MODE == 1:
                    tl.store(
                        d_th_ptr + n_offs * n_d_tiles + pid_d,
                        tl.sum(d_th, axis=1),
                        mask=n_mask,
                    )
                elif TH_MODE == 2:
                    tl.store(
                        d_th_ptr + pid_n * n_cols + d_offs,
                        tl.sum(d_th, axis=0),
                        mask=d_mask,
                    )
                else:
                    tl.store(d_th_ptr + pid_n * n_d_tiles + pid_d, tl.sum(d_th))

        if NEED_A_GRAD:
            s = s_raw.to(ACC_DTYPE)
            d_a = tl.where(mask, -s * gs, zero)
            if A_MODE == 0:
                tl.store(d_a_ptr + idx, d_a, mask=mask, eviction_policy="evict_first")
            elif A_MODE == 1:
                tl.store(
                    d_a_ptr + n_offs * n_d_tiles + pid_d,
                    tl.sum(d_a, axis=1),
                    mask=n_mask,
                )
            elif A_MODE == 2:
                tl.store(
                    d_a_ptr + pid_n * n_cols + d_offs,
                    tl.sum(d_a, axis=0),
                    mask=d_mask,
                )
            else:
                tl.store(d_a_ptr + pid_n * n_d_tiles + pid_d, tl.sum(d_a))

        if NEED_ET_GRAD:
            m = tl.load(mom_ptr + idx, mask=mask, other=0.0).to(ACC_DTYPE)
            d_et = tl.where(mask, m * g_tot, zero)
            if ET_MODE == 0:
                tl.store(d_et_ptr + idx, d_et, mask=mask, eviction_policy="evict_first")
            elif ET_MODE == 1:
                tl.store(
                    d_et_ptr + n_offs * n_d_tiles + pid_d,
                    tl.sum(d_et, axis=1),
                    mask=n_mask,
                )
            elif ET_MODE == 2:
                tl.store(
                    d_et_ptr + pid_n * n_cols + d_offs,
                    tl.sum(d_et, axis=0),
                    mask=d_mask,
                )
            else:
                tl.store(d_et_ptr + pid_n * n_d_tiles + pid_d, tl.sum(d_et))


def _next_pow2(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def _normalize_moe_gate_2d(
    gate_t: torch.Tensor,
    n_assets: int,
    d_model: int,
) -> tuple[torch.Tensor, int, int]:
    """Normaliza `mix_gate` a forma 2D contigua compatible con `(N, D)` y sus strides exactos."""
    if gate_t.ndim == 0 or gate_t.numel() == 1:
        g2 = gate_t.reshape(1, 1).contiguous()
    elif gate_t.ndim == 1:
        if gate_t.shape[0] == n_assets:
            g2 = gate_t.reshape(n_assets, 1).contiguous()
        elif gate_t.shape[0] == d_model:
            g2 = gate_t.reshape(1, d_model).contiguous()
        else:
            raise ValueError(
                f"mix_gate 1D debe tener longitud 1, N={n_assets} o D={d_model}; "
                f"recibido {tuple(gate_t.shape)}"
            )
    elif gate_t.ndim == 2:
        if gate_t.shape[0] not in (1, n_assets) or gate_t.shape[1] not in (1, d_model):
            raise ValueError(
                f"mix_gate 2D debe poder difundirse a {(n_assets, d_model)}; "
                f"recibido {tuple(gate_t.shape)}"
            )
        g2 = gate_t.contiguous()
    else:
        raise ValueError(
            f"mix_gate debe tener 0, 1 o 2 dimensiones; recibido {tuple(gate_t.shape)}"
        )

    stride_gn = g2.stride(0) if g2.shape[0] > 1 else 0
    stride_gd = g2.stride(1) if g2.shape[1] > 1 else 0
    return g2, stride_gn, stride_gd


def _prep_neuromod_gate_2d(
    gate: torch.Tensor,
    n_rows: int,
    n_cols: int,
    state_shape: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    """Prepara gate broadcastable en 2D compacto sin expandir a `[N, D]` y devuelve modo reduce."""
    g_c = gate.to(device=device).contiguous()
    if g_c.numel() == 1:
        g2 = g_c.reshape(1, 1)
        return g2, 0, 0, 3
    if len(state_shape) == 1:
        if g_c.shape == (n_rows,):
            return g_c.reshape(n_rows, 1), 1, 0, 0
        return g_c.expand(state_shape).contiguous().reshape(n_rows, 1), 1, 0, 0
    if len(state_shape) == 2:
        g2 = g_c.reshape(1, g_c.shape[0]) if g_c.ndim == 1 else g_c
        r, c = int(g2.shape[0]), int(g2.shape[1])
        sn = int(g2.stride(0)) if r > 1 else 0
        sd = int(g2.stride(1)) if c > 1 else 0
        if r == n_rows and c == n_cols:
            mode = 0
        elif r == n_rows and c == 1:
            mode = 1
        elif r == 1 and c == n_cols:
            mode = 2
        else:
            mode = 3
        return g2, sn, sd, mode
    # Casos >=3D: detectar broadcast limpio en última dim o leading dims antes de expandir
    if g_c.ndim <= len(state_shape):
        padded = (1,) * (len(state_shape) - g_c.ndim) + tuple(int(s) for s in g_c.shape)
        lead_match = padded[:-1] == tuple(int(s) for s in state_shape[:-1])
        lead_one = all(s == 1 for s in padded[:-1])
        last_match = padded[-1] == n_cols
        last_one = padded[-1] == 1
        if lead_match and last_match:
            g2 = g_c.reshape(n_rows, n_cols)
            return g2, int(g2.stride(0)), int(g2.stride(1)), 0
        if lead_match and last_one:
            g2 = g_c.reshape(n_rows, 1)
            return g2, int(g2.stride(0)), 0, 1
        if lead_one and last_match:
            g2 = g_c.reshape(1, n_cols)
            return g2, 0, int(g2.stride(1)), 2
    g2 = g_c.expand(state_shape).contiguous().reshape(n_rows, n_cols)
    return g2, int(g2.stride(0)), int(g2.stride(1)), 0


def _unbroadcast_grad(grad: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Reduce un gradiente expandido hasta coincidir exactamente con `target_shape`."""
    if grad.shape == target_shape:
        return grad
    if len(target_shape) == 0:
        return grad.sum()
    g = grad
    extra = g.ndim - len(target_shape)
    if extra > 0:
        g = g.sum(dim=tuple(range(extra)))
    reduce_dims = tuple(
        i
        for i, (gs, ts) in enumerate(zip(g.shape, target_shape, strict=True))
        if ts == 1 and gs > 1
    )
    if reduce_dims:
        g = g.sum(dim=reduce_dims, keepdim=True)
    return g.reshape(target_shape)


def _zscore_split_params(
    n_rows: int, n_cols: int, dtype: torch.dtype
) -> tuple[int, int, int, int]:
    """Calcula `(block_n, block_f, n_splits, rows_per_split)` para saturar `>=188` CTAs `sm_120`."""
    block_f = 64 if (n_cols >= 64 and dtype in (torch.float16, torch.bfloat16)) else 32
    block_n = 64 if dtype == torch.float64 else 128
    n_f_tiles = (n_cols + block_f - 1) // block_f
    if n_rows <= block_n:
        return block_n, block_f, 1, max(n_rows, 1)
    target_splits = (SM120_SMS + n_f_tiles - 1) // n_f_tiles
    max_splits = min(32, max(1, (n_rows + 63) // 64))
    n_splits = max(1, min(target_splits, max_splits))
    rows_per_split = (n_rows + n_splits - 1) // n_splits
    return block_n, block_f, n_splits, rows_per_split


# -----------------------------------------------------------------------------
# 1.a. Registro `torch.library.custom_op` (CERO graph breaks en `torch.compile`)
# -----------------------------------------------------------------------------
@torch.library.custom_op("mars_titan_bw::zscore_clip_fwd", mutates_args=())
def _zscore_clip_fwd_op(
    x: torch.Tensor, clip_val: float, eps: float, use_triton: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    n_rows = int(orig_shape[0])
    acc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_2d = x.reshape(n_rows, -1)
    n_cols = int(x_2d.shape[1])
    if n_rows == 0 or n_cols == 0:
        return (
            x.clone(memory_format=torch.contiguous_format),
            torch.zeros((n_cols,), device=x.device, dtype=acc_dtype),
            torch.ones((n_cols,), device=x.device, dtype=acc_dtype),
        )

    if use_triton and _HAS_TRITON and x.is_cuda:  # pragma: no cover - ruta GPU sm_120
        x_c = x_2d.contiguous()
        out_2d = torch.empty_like(x_c)
        mean_buf = torch.empty((n_cols,), device=x.device, dtype=acc_dtype)
        inv_std_buf = torch.empty((n_cols,), device=x.device, dtype=acc_dtype)
        tl_acc = tl.float64 if x.dtype == torch.float64 else tl.float32
        block_n, block_f, n_splits, rps = _zscore_split_params(n_rows, n_cols, x.dtype)
        n_f_tiles = (n_cols + block_f - 1) // block_f
        if n_splits == 1:
            _cross_sectional_zscore_clip_kernel[(n_f_tiles,)](
                x_c,
                out_2d,
                mean_buf,
                inv_std_buf,
                n_rows,
                n_cols,
                x_c.stride(0),
                x_c.stride(1),
                clip_val=float(clip_val),
                eps=float(eps),
                BLOCK_N=block_n,
                BLOCK_F=block_f,
                SAVE_STATS=True,
                ACC_DTYPE=tl_acc,
                num_warps=4,
                num_stages=2,
            )
        else:
            part_mean = torch.empty((n_splits, n_cols), device=x.device, dtype=acc_dtype)
            part_m2 = torch.empty((n_splits, n_cols), device=x.device, dtype=acc_dtype)
            grid_2d = (n_f_tiles, n_splits)
            _cross_sectional_zscore_split_stats_kernel[grid_2d](
                x_c,
                part_mean,
                part_m2,
                n_rows,
                n_cols,
                x_c.stride(0),
                x_c.stride(1),
                rps,
                BLOCK_N=block_n,
                BLOCK_F=block_f,
                ACC_DTYPE=tl_acc,
                num_warps=4,
                num_stages=2,
            )
            _cross_sectional_zscore_merge_stats_kernel[(n_f_tiles,)](
                part_mean,
                part_m2,
                mean_buf,
                inv_std_buf,
                n_rows,
                n_cols,
                rps,
                n_splits,
                eps=float(eps),
                BLOCK_F=block_f,
                ACC_DTYPE=tl_acc,
                num_warps=2,
                num_stages=2,
            )
            _cross_sectional_zscore_apply_kernel[grid_2d](
                x_c,
                out_2d,
                mean_buf,
                inv_std_buf,
                n_rows,
                n_cols,
                x_c.stride(0),
                x_c.stride(1),
                rps,
                clip_val=float(clip_val),
                BLOCK_N=block_n,
                BLOCK_F=block_f,
                ACC_DTYPE=tl_acc,
                num_warps=4,
                num_stages=2,
            )
        return out_2d.reshape(orig_shape), mean_buf, inv_std_buf

    x_acc = x_2d.to(dtype=acc_dtype)
    mean = x_acc.mean(dim=0, keepdim=True)
    centered = x_acc - mean
    var = (centered * centered).mean(dim=0, keepdim=True)
    inv_std = torch.rsqrt(var + float(eps))
    out = torch.clamp(centered * inv_std, min=-float(clip_val), max=float(clip_val))
    return (
        out.to(dtype=x.dtype).reshape(orig_shape).contiguous(),
        mean.squeeze(0).contiguous(),
        inv_std.squeeze(0).contiguous(),
    )


@_zscore_clip_fwd_op.register_fake
def _zscore_clip_fwd_fake(
    x: torch.Tensor, clip_val: float, eps: float, use_triton: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    n_cols = 1
    for s in x.shape[1:]:
        n_cols = n_cols * s
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty((n_cols,), device=x.device, dtype=acc_dtype),
        torch.empty((n_cols,), device=x.device, dtype=acc_dtype),
    )


@torch.library.custom_op("mars_titan_bw::zscore_clip_bwd", mutates_args=())
def _zscore_clip_bwd_op(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    mean_1d: torch.Tensor,
    inv_std_1d: torch.Tensor,
    clip_val: float,
    use_triton: bool,
) -> torch.Tensor:
    orig_shape = x.shape
    n_rows = int(orig_shape[0])
    x_2d = x.reshape(n_rows, -1)
    n_cols = int(x_2d.shape[1])
    if n_rows == 0 or n_cols == 0:
        return torch.zeros_like(x, memory_format=torch.contiguous_format)

    if use_triton and _HAS_TRITON and x.is_cuda:  # pragma: no cover
        x_c = x_2d.contiguous()
        go_c = grad_out.reshape(n_rows, n_cols).contiguous()
        gx_2d = torch.empty_like(x_c)
        acc_dtype = mean_1d.dtype
        tl_acc = tl.float64 if x_c.dtype == torch.float64 else tl.float32
        block_n, block_f, n_splits, rps = _zscore_split_params(n_rows, n_cols, x_c.dtype)
        n_f_tiles = (n_cols + block_f - 1) // block_f
        if n_splits == 1:
            _cross_sectional_zscore_clip_bwd_kernel[(n_f_tiles,)](
                x_c,
                go_c,
                mean_1d,
                inv_std_1d,
                gx_2d,
                n_rows,
                n_cols,
                x_c.stride(0),
                x_c.stride(1),
                clip_val=float(clip_val),
                BLOCK_N=block_n,
                BLOCK_F=block_f,
                ACC_DTYPE=tl_acc,
                num_warps=4,
                num_stages=2,
            )
        else:
            part_u = torch.empty((n_splits, n_cols), device=x.device, dtype=acc_dtype)
            part_uz = torch.empty((n_splits, n_cols), device=x.device, dtype=acc_dtype)
            mean_u = torch.empty((n_cols,), device=x.device, dtype=acc_dtype)
            mean_uz = torch.empty((n_cols,), device=x.device, dtype=acc_dtype)
            grid_2d = (n_f_tiles, n_splits)
            _cross_sectional_zscore_bwd_split_kernel[grid_2d](
                x_c,
                go_c,
                mean_1d,
                inv_std_1d,
                part_u,
                part_uz,
                n_rows,
                n_cols,
                x_c.stride(0),
                x_c.stride(1),
                rps,
                clip_val=float(clip_val),
                BLOCK_N=block_n,
                BLOCK_F=block_f,
                ACC_DTYPE=tl_acc,
                num_warps=4,
                num_stages=2,
            )
            _cross_sectional_zscore_bwd_merge_kernel[(n_f_tiles,)](
                part_u,
                part_uz,
                mean_u,
                mean_uz,
                n_rows,
                n_cols,
                n_splits,
                BLOCK_F=block_f,
                ACC_DTYPE=tl_acc,
                num_warps=2,
                num_stages=2,
            )
            _cross_sectional_zscore_bwd_apply_kernel[grid_2d](
                x_c,
                go_c,
                mean_1d,
                inv_std_1d,
                mean_u,
                mean_uz,
                gx_2d,
                n_rows,
                n_cols,
                x_c.stride(0),
                x_c.stride(1),
                rps,
                clip_val=float(clip_val),
                BLOCK_N=block_n,
                BLOCK_F=block_f,
                ACC_DTYPE=tl_acc,
                num_warps=4,
                num_stages=2,
            )
        return gx_2d.reshape(orig_shape)

    acc_dtype = mean_1d.dtype
    x_acc = x_2d.to(dtype=acc_dtype)
    go_acc = grad_out.reshape(n_rows, n_cols).to(dtype=acc_dtype)
    mean = mean_1d.unsqueeze(0)
    inv_std = inv_std_1d.unsqueeze(0)
    z = (x_acc - mean) * inv_std
    in_bounds = (z > -float(clip_val)) & (z < float(clip_val))
    u = torch.where(in_bounds, go_acc, torch.zeros_like(go_acc))
    mean_u = u.mean(dim=0, keepdim=True)
    mean_uz = (u * z).mean(dim=0, keepdim=True)
    gx = inv_std * (u - mean_u - z * mean_uz)
    return gx.to(dtype=x.dtype).reshape(orig_shape).contiguous()


@_zscore_clip_bwd_op.register_fake
def _zscore_clip_bwd_fake(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    mean_1d: torch.Tensor,
    inv_std_1d: torch.Tensor,
    clip_val: float,
    use_triton: bool,
) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _zscore_clip_setup_ctx(ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
    x, clip_val, _eps, use_triton = inputs
    _out, mean_1d, inv_std_1d = output
    ctx.save_for_backward(x, mean_1d, inv_std_1d)
    ctx.clip_val = float(clip_val)
    ctx.use_triton = bool(use_triton)


def _zscore_clip_backward(
    ctx: Any,
    grad_out: torch.Tensor | None,
    _grad_mean: torch.Tensor | None,
    _grad_inv_std: torch.Tensor | None,
) -> tuple[torch.Tensor | None, None, None, None]:
    if grad_out is None or not ctx.needs_input_grad[0]:
        return None, None, None, None
    x, mean_1d, inv_std_1d = ctx.saved_tensors
    gx = _zscore_clip_bwd_op(x, grad_out, mean_1d, inv_std_1d, ctx.clip_val, ctx.use_triton)
    return gx, None, None, None


_zscore_clip_fwd_op.register_autograd(_zscore_clip_backward, setup_context=_zscore_clip_setup_ctx)


# --- Soft MoE Combine custom_op ---
@torch.library.custom_op("mars_titan_bw::soft_moe_combine_fwd", mutates_args=())
def _soft_moe_combine_fwd_op(
    expert_outputs: torch.Tensor,
    router_probs: torch.Tensor,
    shared_output: torch.Tensor,
    mix_gate: torch.Tensor,
    residual: torch.Tensor | None,
    apply_sigmoid: bool,
    use_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_assets, n_experts, d_model = (
        int(expert_outputs.shape[0]),
        int(expert_outputs.shape[1]),
        int(expert_outputs.shape[2]),
    )
    device, dtype = expert_outputs.device, expert_outputs.dtype
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    gate_2d_raw, stride_gn, stride_gd = _normalize_moe_gate_2d(mix_gate, n_assets, d_model)

    if use_triton and _HAS_TRITON and expert_outputs.is_cuda:  # pragma: no cover
        exp_c = expert_outputs.contiguous()
        probs_c = router_probs.to(device=device, dtype=dtype).contiguous()
        shared_c = shared_output.to(device=device, dtype=dtype).contiguous()
        g_raw_c = gate_2d_raw.to(device=device).contiguous()
        stride_gn = int(g_raw_c.stride(0)) if g_raw_c.shape[0] > 1 else 0
        stride_gd = int(g_raw_c.stride(1)) if g_raw_c.shape[1] > 1 else 0
        has_res = residual is not None
        res_c = residual.to(device=device, dtype=dtype).contiguous() if has_res else shared_c
        out = torch.empty((n_assets, d_model), device=device, dtype=dtype)
        routed_buf = torch.empty((n_assets, d_model), device=device, dtype=acc_dtype)
        block_d = min(max(32, _next_pow2(d_model)), 256)
        grid = (n_assets, (d_model + block_d - 1) // block_d)
        tl_acc = tl.float64 if dtype == torch.float64 else tl.float32
        _soft_moe_combine_kernel[grid](
            exp_c,
            probs_c,
            shared_c,
            g_raw_c,
            res_c,
            out,
            routed_buf,
            d_model,
            exp_c.stride(0),
            exp_c.stride(1),
            exp_c.stride(2),
            probs_c.stride(0),
            probs_c.stride(1),
            shared_c.stride(0),
            shared_c.stride(1),
            stride_gn,
            stride_gd,
            res_c.stride(0),
            res_c.stride(1),
            out.stride(0),
            out.stride(1),
            routed_buf.stride(0),
            routed_buf.stride(1),
            N_EXPERTS=n_experts,
            BLOCK_D=block_d,
            HAS_RESIDUAL=has_res,
            APPLY_SIGMOID=bool(apply_sigmoid),
            SAVE_ROUTED=True,
            ACC_DTYPE=tl_acc,
            num_warps=4,
            num_stages=2,
        )
        return out, routed_buf

    g_a = gate_2d_raw.to(device=device, dtype=acc_dtype)
    if apply_sigmoid:
        g_a = torch.sigmoid(g_a)
    exp_a = expert_outputs.to(dtype=acc_dtype)
    probs_a = router_probs.to(device=device, dtype=acc_dtype)
    shared_a = shared_output.to(device=device, dtype=acc_dtype)
    routed_a = torch.einsum("ne,ned->nd", probs_a, exp_a)
    out_a = g_a * routed_a + (1.0 - g_a) * shared_a
    if residual is not None:
        out_a = residual.to(device=device, dtype=acc_dtype) + out_a
    return out_a.to(dtype=dtype).contiguous(), routed_a.contiguous()


@_soft_moe_combine_fwd_op.register_fake
def _soft_moe_combine_fwd_fake(
    expert_outputs: torch.Tensor,
    router_probs: torch.Tensor,
    shared_output: torch.Tensor,
    mix_gate: torch.Tensor,
    residual: torch.Tensor | None,
    apply_sigmoid: bool,
    use_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_assets, _, d_model = expert_outputs.shape
    dtype = expert_outputs.dtype
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    return (
        torch.empty((n_assets, d_model), device=expert_outputs.device, dtype=dtype),
        torch.empty((n_assets, d_model), device=expert_outputs.device, dtype=acc_dtype),
    )


@torch.library.custom_op("mars_titan_bw::soft_moe_combine_bwd", mutates_args=())
def _soft_moe_combine_bwd_op(
    expert_outputs: torch.Tensor,
    router_probs: torch.Tensor,
    shared_output: torch.Tensor,
    mix_gate: torch.Tensor,
    routed_acc: torch.Tensor,
    grad_out: torch.Tensor,
    apply_sigmoid: bool,
    use_triton: bool,
    need_exp: bool,
    need_probs: bool,
    need_shared: bool,
    need_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_assets, n_experts, d_model = (
        int(expert_outputs.shape[0]),
        int(expert_outputs.shape[1]),
        int(expert_outputs.shape[2]),
    )
    device, dtype = expert_outputs.device, expert_outputs.dtype
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    gate_orig_shape = tuple(mix_gate.shape)
    gate_2d_raw, stride_gn, stride_gd = _normalize_moe_gate_2d(mix_gate, n_assets, d_model)

    if (
        use_triton
        and _HAS_TRITON
        and expert_outputs.is_cuda
        and (need_exp or need_probs or need_shared or need_gate)
    ):  # pragma: no cover
        exp_c = expert_outputs.contiguous()
        probs_c = router_probs.to(device=device, dtype=dtype).contiguous()
        shared_c = shared_output.to(device=device, dtype=dtype).contiguous()
        g_raw_c = gate_2d_raw.to(device=device).contiguous()
        routed_c = routed_acc.to(device=device, dtype=acc_dtype).contiguous()
        go_c = grad_out.to(device=device, dtype=dtype).contiguous()
        stride_gn = int(g_raw_c.stride(0)) if g_raw_c.shape[0] > 1 else 0
        stride_gd = int(g_raw_c.stride(1)) if g_raw_c.shape[1] > 1 else 0
        block_d = min(max(32, _next_pow2(d_model)), 256)
        n_d_tiles = (d_model + block_d - 1) // block_d
        gate_reduce_d = stride_gd == 0
        grad_exp = (
            torch.empty_like(exp_c) if need_exp else torch.empty((0,), device=device, dtype=dtype)
        )
        grad_shared = (
            torch.empty_like(shared_c)
            if need_shared
            else torch.empty((0,), device=device, dtype=shared_output.dtype)
        )
        grad_probs_part = (
            torch.empty((n_assets, n_d_tiles, n_experts), device=device, dtype=acc_dtype)
            if need_probs
            else torch.empty((0,), device=device, dtype=acc_dtype)
        )
        if need_gate:
            grad_gate_part = (
                torch.empty((n_assets, n_d_tiles), device=device, dtype=acc_dtype)
                if gate_reduce_d
                else torch.empty((n_assets, d_model), device=device, dtype=acc_dtype)
            )
        else:
            grad_gate_part = torch.empty((0,), device=device, dtype=acc_dtype)

        tl_acc = tl.float64 if dtype == torch.float64 else tl.float32
        _soft_moe_combine_bwd_kernel[(n_assets, n_d_tiles)](
            exp_c,
            probs_c,
            shared_c,
            g_raw_c,
            routed_c,
            go_c,
            grad_exp,
            grad_probs_part,
            grad_shared,
            grad_gate_part,
            d_model,
            n_d_tiles,
            exp_c.stride(0),
            exp_c.stride(1),
            exp_c.stride(2),
            probs_c.stride(0),
            probs_c.stride(1),
            shared_c.stride(0),
            shared_c.stride(1),
            stride_gn,
            stride_gd,
            routed_c.stride(0),
            routed_c.stride(1),
            go_c.stride(0),
            go_c.stride(1),
            N_EXPERTS=n_experts,
            BLOCK_D=block_d,
            APPLY_SIGMOID=bool(apply_sigmoid),
            NEED_EXP_GRAD=bool(need_exp),
            NEED_PROBS_GRAD=bool(need_probs),
            NEED_SHARED_GRAD=bool(need_shared),
            NEED_GATE_GRAD=bool(need_gate),
            GATE_REDUCE_D=bool(gate_reduce_d),
            ACC_DTYPE=tl_acc,
            num_warps=4,
            num_stages=2,
        )
        if need_probs:
            gp_acc = (
                grad_probs_part.squeeze(1) if n_d_tiles == 1 else grad_probs_part.sum(dim=1)
            )
            grad_probs = gp_acc.to(dtype=router_probs.dtype).contiguous()
        else:
            grad_probs = torch.empty((0,), device=device, dtype=router_probs.dtype)

        if need_gate:
            if gate_reduce_d:
                gg_n = (
                    grad_gate_part.squeeze(1) if n_d_tiles == 1 else grad_gate_part.sum(dim=1)
                )
                gg_2d = gg_n.sum().reshape(1, 1) if stride_gn == 0 else gg_n.reshape(n_assets, 1)
            else:
                gg_2d = (
                    grad_gate_part.sum(dim=0, keepdim=True)
                    if stride_gn == 0
                    else grad_gate_part
                )
            grad_gate = gg_2d.reshape(gate_orig_shape).to(dtype=mix_gate.dtype).contiguous()
        else:
            grad_gate = torch.empty((0,), device=device, dtype=mix_gate.dtype)
        if need_shared:
            grad_shared = grad_shared.to(dtype=shared_output.dtype).contiguous()
        return grad_exp, grad_probs, grad_shared, grad_gate

    go_a = grad_out.to(device=device, dtype=acc_dtype)
    g_a = gate_2d_raw.to(device=device, dtype=acc_dtype)
    if apply_sigmoid:
        g_a = torch.sigmoid(g_a)
    gr_a = g_a * go_a

    grad_exp = (
        (router_probs.to(device=device, dtype=acc_dtype).unsqueeze(-1) * gr_a.unsqueeze(1))
        .to(dtype=dtype)
        .contiguous()
        if need_exp
        else torch.empty((0,), device=device, dtype=dtype)
    )
    grad_shared = (
        ((1.0 - g_a) * go_a).to(dtype=shared_output.dtype).contiguous()
        if need_shared
        else torch.empty((0,), device=device, dtype=shared_output.dtype)
    )
    grad_probs = (
        torch.einsum("ned,nd->ne", expert_outputs.to(dtype=acc_dtype), gr_a)
        .to(dtype=router_probs.dtype)
        .contiguous()
        if need_probs
        else torch.empty((0,), device=device, dtype=router_probs.dtype)
    )
    if need_gate:
        dg_nd = go_a * (
            routed_acc.to(device=device, dtype=acc_dtype)
            - shared_output.to(device=device, dtype=acc_dtype)
        )
        if apply_sigmoid:
            dg_nd = dg_nd * g_a * (1.0 - g_a)
        grad_gate = (
            _unbroadcast_grad(dg_nd, tuple(gate_2d_raw.shape))
            .reshape(gate_orig_shape)
            .to(dtype=mix_gate.dtype)
            .contiguous()
        )
    else:
        grad_gate = torch.empty((0,), device=device, dtype=mix_gate.dtype)
    return grad_exp, grad_probs, grad_shared, grad_gate


@_soft_moe_combine_bwd_op.register_fake
def _soft_moe_combine_bwd_fake(
    expert_outputs: torch.Tensor,
    router_probs: torch.Tensor,
    shared_output: torch.Tensor,
    mix_gate: torch.Tensor,
    routed_acc: torch.Tensor,
    grad_out: torch.Tensor,
    apply_sigmoid: bool,
    use_triton: bool,
    need_exp: bool,
    need_probs: bool,
    need_shared: bool,
    need_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = expert_outputs.device
    return (
        torch.empty_like(expert_outputs, memory_format=torch.contiguous_format)
        if need_exp
        else torch.empty((0,), device=dev, dtype=expert_outputs.dtype),
        torch.empty_like(router_probs, memory_format=torch.contiguous_format)
        if need_probs
        else torch.empty((0,), device=dev, dtype=router_probs.dtype),
        torch.empty_like(shared_output, memory_format=torch.contiguous_format)
        if need_shared
        else torch.empty((0,), device=dev, dtype=shared_output.dtype),
        torch.empty_like(mix_gate, memory_format=torch.contiguous_format)
        if need_gate
        else torch.empty((0,), device=dev, dtype=mix_gate.dtype),
    )


def _soft_moe_setup_ctx(ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
    exp_o, probs, shared, mix_gate, residual, apply_sigmoid, use_triton = inputs
    _out, routed_acc = output
    ctx.save_for_backward(exp_o, probs, shared, mix_gate, routed_acc)
    ctx.apply_sigmoid = bool(apply_sigmoid)
    ctx.use_triton = bool(use_triton)
    ctx.has_residual = residual is not None
    ctx.res_dtype = residual.dtype if residual is not None else exp_o.dtype


def _soft_moe_backward(
    ctx: Any,
    grad_out: torch.Tensor | None,
    _grad_routed: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    None,
]:
    if grad_out is None:
        return None, None, None, None, None, None, None
    exp_o, probs, shared, mix_gate, routed_acc = ctx.saved_tensors
    need_exp, need_probs, need_shared, need_gate, need_res = ctx.needs_input_grad[:5]
    ge, gp, gs, gg = _soft_moe_combine_bwd_op(
        exp_o,
        probs,
        shared,
        mix_gate,
        routed_acc,
        grad_out,
        ctx.apply_sigmoid,
        ctx.use_triton,
        bool(need_exp),
        bool(need_probs),
        bool(need_shared),
        bool(need_gate),
    )
    grad_res = grad_out.to(dtype=ctx.res_dtype) if (ctx.has_residual and need_res) else None
    return (
        ge if need_exp else None,
        gp if need_probs else None,
        gs if need_shared else None,
        gg if need_gate else None,
        grad_res,
        None,
        None,
    )


_soft_moe_combine_fwd_op.register_autograd(_soft_moe_backward, setup_context=_soft_moe_setup_ctx)


# --- Pinball / CRPS Loss custom_op ---
@torch.library.custom_op("mars_titan_bw::pinball_crps_fwd", mutates_args=())
def _pinball_crps_fwd_op(
    quantiles: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    crps_scale: float,
    reduction: str,
    use_triton: bool,
) -> torch.Tensor:
    device, dtype = quantiles.device, quantiles.dtype
    expected_item_shape = quantiles.shape[:-1]
    n_q = int(quantiles.shape[-1])
    n_rows = int(targets.numel())
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    if n_rows == 0:
        if reduction == "none":
            return torch.empty(expected_item_shape, device=device, dtype=dtype)
        return torch.zeros((), device=device, dtype=dtype)

    if use_triton and _HAS_TRITON and quantiles.is_cuda:  # pragma: no cover
        q_2d = quantiles.reshape(n_rows, n_q).contiguous()
        y_1d = targets.reshape(n_rows).contiguous()
        taus_1d = taus.reshape(n_q).contiguous()
        block_m = 64
        block_q = min(64, _next_pow2(n_q))
        n_blocks = (n_rows + block_m - 1) // block_m
        reduce_blk = reduction != "none"
        out_buf = (
            torch.empty((n_blocks,), device=device, dtype=acc_dtype)
            if reduce_blk
            else torch.empty((n_rows,), device=device, dtype=dtype)
        )
        tl_acc = tl.float64 if dtype == torch.float64 else tl.float32
        _pinball_crps_loss_kernel[(n_blocks,)](
            q_2d,
            y_1d,
            taus_1d,
            out_buf,
            n_rows,
            q_2d.stride(0),
            q_2d.stride(1),
            scale_over_q=float(crps_scale) / float(n_q),
            N_QUANTILES=n_q,
            BLOCK_Q=block_q,
            BLOCK_M=block_m,
            REDUCE_BLOCK=reduce_blk,
            ACC_DTYPE=tl_acc,
            num_warps=2,
            num_stages=2,
        )
        if reduction == "none":
            return out_buf.reshape(expected_item_shape)
        total = out_buf.sum()
        if reduction == "mean":
            total = total / float(n_rows)
        return total.to(dtype=dtype)

    q_a = quantiles.to(dtype=acc_dtype)
    y_a = targets.to(dtype=acc_dtype).unsqueeze(-1)
    taus_view = taus.to(dtype=acc_dtype).reshape(*([1] * (quantiles.ndim - 1)), n_q)
    diff = y_a - q_a
    pinball = torch.maximum(taus_view * diff, (taus_view - 1.0) * diff)
    per_item = float(crps_scale) * pinball.mean(dim=-1)
    if reduction == "mean":
        return per_item.mean().to(dtype=dtype)
    if reduction == "sum":
        return per_item.sum().to(dtype=dtype)
    return per_item.to(dtype=dtype).contiguous()


@_pinball_crps_fwd_op.register_fake
def _pinball_crps_fwd_fake(
    quantiles: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    crps_scale: float,
    reduction: str,
    use_triton: bool,
) -> torch.Tensor:
    if reduction == "none":
        return torch.empty(quantiles.shape[:-1], device=quantiles.device, dtype=quantiles.dtype)
    return torch.empty((), device=quantiles.device, dtype=quantiles.dtype)


@torch.library.custom_op("mars_titan_bw::pinball_crps_bwd", mutates_args=())
def _pinball_crps_bwd_op(
    quantiles: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    grad_out: torch.Tensor,
    crps_scale: float,
    reduction: str,
    use_triton: bool,
    need_q: bool,
    need_y: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = quantiles.device, quantiles.dtype
    n_q = int(quantiles.shape[-1])
    n_rows = int(targets.numel())
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    scale_over_q = float(crps_scale) / float(n_q)

    if n_rows == 0 or not (need_q or need_y):
        return (
            torch.zeros_like(quantiles, memory_format=torch.contiguous_format)
            if need_q
            else torch.empty((0,), device=device, dtype=dtype),
            torch.zeros_like(targets, memory_format=torch.contiguous_format)
            if need_y
            else torch.empty((0,), device=device, dtype=targets.dtype),
        )

    if use_triton and _HAS_TRITON and quantiles.is_cuda:  # pragma: no cover
        q_2d = quantiles.reshape(n_rows, n_q).contiguous()
        y_1d = targets.reshape(n_rows).contiguous()
        taus_1d = taus.reshape(n_q).contiguous()
        is_scalar = reduction != "none"
        if reduction == "mean":
            go_eff = (grad_out.to(dtype=acc_dtype) / float(n_rows)).contiguous()
        elif reduction == "sum":
            go_eff = grad_out.to(dtype=acc_dtype).contiguous()
        else:
            go_eff = grad_out.reshape(n_rows).to(dtype=acc_dtype).contiguous()
        gq_2d = torch.empty_like(q_2d) if need_q else torch.empty((0,), device=device, dtype=dtype)
        gy_1d = (
            torch.empty_like(y_1d)
            if need_y
            else torch.empty((0,), device=device, dtype=targets.dtype)
        )
        block_m = 64
        block_q = min(64, _next_pow2(n_q))
        n_blocks = (n_rows + block_m - 1) // block_m
        tl_acc = tl.float64 if dtype == torch.float64 else tl.float32
        _pinball_crps_loss_bwd_kernel[(n_blocks,)](
            q_2d,
            y_1d,
            taus_1d,
            go_eff,
            gq_2d,
            gy_1d,
            n_rows,
            q_2d.stride(0),
            q_2d.stride(1),
            scale_over_q=scale_over_q,
            N_QUANTILES=n_q,
            BLOCK_Q=block_q,
            BLOCK_M=block_m,
            IS_SCALAR_GRAD=is_scalar,
            NEED_Q_GRAD=bool(need_q),
            NEED_Y_GRAD=bool(need_y),
            ACC_DTYPE=tl_acc,
            num_warps=2,
            num_stages=2,
        )
        return (
            gq_2d.reshape(quantiles.shape) if need_q else gq_2d,
            gy_1d.reshape(targets.shape).to(dtype=targets.dtype) if need_y else gy_1d,
        )

    q_a = quantiles.reshape(n_rows, n_q).to(dtype=acc_dtype)
    y_a = targets.reshape(n_rows).to(dtype=acc_dtype).unsqueeze(-1)
    t_a = taus.reshape(n_q).to(dtype=acc_dtype).unsqueeze(0)
    diff = y_a - q_a
    s = torch.where(diff > 0.0, t_a, torch.where(diff < 0.0, t_a - 1.0, t_a - 0.5))
    if reduction == "mean":
        c_m = grad_out.to(dtype=acc_dtype) * (scale_over_q / float(n_rows))
    elif reduction == "sum":
        c_m = grad_out.to(dtype=acc_dtype) * scale_over_q
    else:
        c_m = grad_out.reshape(n_rows, 1).to(dtype=acc_dtype) * scale_over_q

    grad_q = (
        (-c_m * s).to(dtype=dtype).reshape(quantiles.shape).contiguous()
        if need_q
        else torch.empty((0,), device=device, dtype=dtype)
    )
    grad_y = (
        (c_m * s.sum(dim=-1, keepdim=True))
        .to(dtype=targets.dtype)
        .reshape(targets.shape)
        .contiguous()
        if need_y
        else torch.empty((0,), device=device, dtype=targets.dtype)
    )
    return grad_q, grad_y


@_pinball_crps_bwd_op.register_fake
def _pinball_crps_bwd_fake(
    quantiles: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    grad_out: torch.Tensor,
    crps_scale: float,
    reduction: str,
    use_triton: bool,
    need_q: bool,
    need_y: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = quantiles.device
    return (
        torch.empty_like(quantiles, memory_format=torch.contiguous_format)
        if need_q
        else torch.empty((0,), device=dev, dtype=quantiles.dtype),
        torch.empty_like(targets, memory_format=torch.contiguous_format)
        if need_y
        else torch.empty((0,), device=dev, dtype=targets.dtype),
    )


def _pinball_setup_ctx(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    _ = output
    quantiles, targets, taus, crps_scale, reduction, use_triton = inputs
    ctx.save_for_backward(quantiles, targets, taus)
    ctx.crps_scale = float(crps_scale)
    ctx.reduction = str(reduction)
    ctx.use_triton = bool(use_triton)


def _pinball_backward(
    ctx: Any, grad_out: torch.Tensor | None
) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
    if grad_out is None:
        return None, None, None, None, None, None
    need_q, need_y = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
    if not (need_q or need_y):
        return None, None, None, None, None, None
    quantiles, targets, taus = ctx.saved_tensors
    gq, gy = _pinball_crps_bwd_op(
        quantiles,
        targets,
        taus,
        grad_out,
        ctx.crps_scale,
        ctx.reduction,
        ctx.use_triton,
        bool(need_q),
        bool(need_y),
    )
    return gq if need_q else None, gy if need_y else None, None, None, None, None


_pinball_crps_fwd_op.register_autograd(_pinball_backward, setup_context=_pinball_setup_ctx)


# --- Neuromod Update custom_op ---
@torch.library.custom_op("mars_titan_bw::neuromod_update_fwd", mutates_args=())
def _neuromod_update_fwd_op(
    state: torch.Tensor,
    momentum: torch.Tensor,
    error: torch.Tensor,
    alpha: torch.Tensor,
    eta: torch.Tensor,
    theta: torch.Tensor,
    huber_delta: float,
    use_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = state.dtype
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    if state.numel() == 0:
        return (
            state.clone(memory_format=torch.contiguous_format),
            momentum.clone(memory_format=torch.contiguous_format),
        )

    if use_triton and _HAS_TRITON and state.is_cuda:  # pragma: no cover
        orig_shape = tuple(int(s) for s in state.shape)
        if state.ndim == 1:
            n_rows, n_cols = int(state.shape[0]), 1
        else:
            n_cols = int(state.shape[-1])
            n_rows = int(state.numel()) // max(n_cols, 1)
        s_2d = state.contiguous().reshape(n_rows, n_cols)
        m_2d = momentum.to(device=state.device, dtype=dtype).contiguous().reshape(n_rows, n_cols)
        e_2d = error.to(device=state.device, dtype=dtype).contiguous().reshape(n_rows, n_cols)
        a_2d, s_an, s_ad, _ = _prep_neuromod_gate_2d(
            alpha, n_rows, n_cols, orig_shape, state.device
        )
        et_2d, s_etn, s_etd, _ = _prep_neuromod_gate_2d(
            eta, n_rows, n_cols, orig_shape, state.device
        )
        th_2d, s_thn, s_thd, _ = _prep_neuromod_gate_2d(
            theta, n_rows, n_cols, orig_shape, state.device
        )
        out_s = torch.empty_like(s_2d)
        out_m = torch.empty_like(m_2d)
        block_n, block_d = 32, 64
        grid = ((n_rows + block_n - 1) // block_n, (n_cols + block_d - 1) // block_d)
        tl_acc = tl.float64 if dtype == torch.float64 else tl.float32
        _neuromod_update_fwd_kernel[grid](
            s_2d,
            m_2d,
            e_2d,
            a_2d,
            et_2d,
            th_2d,
            out_s,
            out_m,
            n_rows,
            n_cols,
            s_2d.stride(0),
            s_2d.stride(1),
            s_an,
            s_ad,
            s_etn,
            s_etd,
            s_thn,
            s_thd,
            huber_delta=float(huber_delta),
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            ACC_DTYPE=tl_acc,
            num_warps=4,
            num_stages=2,
        )
        return (
            out_s.reshape(orig_shape),
            out_m.reshape(orig_shape).to(dtype=momentum.dtype).contiguous(),
        )

    s_a = state.to(dtype=acc_dtype)
    m_a = momentum.to(device=state.device, dtype=acc_dtype)
    e_a = error.to(device=state.device, dtype=acc_dtype)
    a_a = alpha.to(device=state.device, dtype=acc_dtype)
    et_a = eta.to(device=state.device, dtype=acc_dtype)
    th_a = theta.to(device=state.device, dtype=acc_dtype)
    e_clip = torch.clamp(e_a, min=-float(huber_delta), max=float(huber_delta))
    m_new = et_a * m_a + th_a * e_clip
    s_new = (1.0 - a_a) * s_a + m_new
    return (
        s_new.to(dtype=dtype).contiguous(),
        m_new.to(dtype=momentum.dtype).contiguous(),
    )


@_neuromod_update_fwd_op.register_fake
def _neuromod_update_fwd_fake(
    state: torch.Tensor,
    momentum: torch.Tensor,
    error: torch.Tensor,
    alpha: torch.Tensor,
    eta: torch.Tensor,
    theta: torch.Tensor,
    huber_delta: float,
    use_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(state, memory_format=torch.contiguous_format),
        torch.empty_like(momentum, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op("mars_titan_bw::neuromod_update_bwd", mutates_args=())
def _neuromod_update_bwd_op(
    state: torch.Tensor,
    momentum: torch.Tensor,
    error: torch.Tensor,
    alpha: torch.Tensor,
    eta: torch.Tensor,
    theta: torch.Tensor,
    grad_s: torch.Tensor,
    grad_m: torch.Tensor,
    huber_delta: float,
    use_triton: bool,
    need_s: bool,
    need_m: bool,
    need_e: bool,
    need_a: bool,
    need_et: bool,
    need_th: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = state.device, state.dtype
    acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    delta = float(huber_delta)
    alpha_shape = tuple(int(s) for s in alpha.shape)
    eta_shape = tuple(int(s) for s in eta.shape)
    theta_shape = tuple(int(s) for s in theta.shape)

    def _empty_like_or_zero(t: torch.Tensor, needed: bool) -> torch.Tensor:
        return (
            torch.zeros_like(t, memory_format=torch.contiguous_format)
            if needed
            else torch.empty((0,), device=device, dtype=t.dtype)
        )

    if state.numel() == 0:
        return (
            _empty_like_or_zero(state, need_s),
            _empty_like_or_zero(momentum, need_m),
            _empty_like_or_zero(error, need_e),
            _empty_like_or_zero(alpha, need_a),
            _empty_like_or_zero(eta, need_et),
            _empty_like_or_zero(theta, need_th),
        )

    if use_triton and _HAS_TRITON and state.is_cuda:  # pragma: no cover
        orig_shape = tuple(int(s) for s in state.shape)
        if state.ndim == 1:
            n_rows, n_cols = int(state.shape[0]), 1
        else:
            n_cols = int(state.shape[-1])
            n_rows = int(state.numel()) // max(n_cols, 1)
        s_2d = state.contiguous().reshape(n_rows, n_cols)
        m_2d = momentum.to(device=device, dtype=dtype).contiguous().reshape(n_rows, n_cols)
        e_2d = error.to(device=device, dtype=dtype).contiguous().reshape(n_rows, n_cols)
        gs_2d = grad_s.to(device=device, dtype=dtype).contiguous().reshape(n_rows, n_cols)
        gm_2d = grad_m.to(device=device, dtype=dtype).contiguous().reshape(n_rows, n_cols)

        a_2d, s_an, s_ad, a_mode = _prep_neuromod_gate_2d(alpha, n_rows, n_cols, orig_shape, device)
        et_2d, s_etn, s_etd, et_mode = _prep_neuromod_gate_2d(
            eta, n_rows, n_cols, orig_shape, device
        )
        th_2d, s_thn, s_thd, th_mode = _prep_neuromod_gate_2d(
            theta, n_rows, n_cols, orig_shape, device
        )

        block_n, block_d = 32, 64
        n_n_tiles = (n_rows + block_n - 1) // block_n
        n_d_tiles = (n_cols + block_d - 1) // block_d

        def _alloc_gate_buf(needed: bool, mode: int) -> torch.Tensor:
            if not needed:
                return torch.empty((0,), device=device, dtype=acc_dtype)
            if mode == 0:
                return torch.empty((n_rows, n_cols), device=device, dtype=acc_dtype)
            if mode == 1:
                return torch.empty((n_rows, n_d_tiles), device=device, dtype=acc_dtype)
            if mode == 2:
                return torch.empty((n_n_tiles, n_cols), device=device, dtype=acc_dtype)
            return torch.empty((n_n_tiles, n_d_tiles), device=device, dtype=acc_dtype)

        # Asignar tensores vacíos FRESCOS por salida (jamás aliasar retornos en custom_op, #122!)
        d_s_buf = (
            torch.empty_like(s_2d) if need_s else torch.empty((0,), device=device, dtype=dtype)
        )
        d_m_buf = (
            torch.empty_like(m_2d)
            if need_m
            else torch.empty((0,), device=device, dtype=momentum.dtype)
        )
        d_e_buf = (
            torch.empty_like(e_2d)
            if need_e
            else torch.empty((0,), device=device, dtype=error.dtype)
        )
        d_a_buf = _alloc_gate_buf(need_a, a_mode)
        d_et_buf = _alloc_gate_buf(need_et, et_mode)
        d_th_buf = _alloc_gate_buf(need_th, th_mode)

        tl_acc = tl.float64 if dtype == torch.float64 else tl.float32
        _neuromod_update_bwd_kernel[(n_n_tiles, n_d_tiles)](
            s_2d,
            m_2d,
            e_2d,
            a_2d,
            et_2d,
            th_2d,
            gs_2d,
            gm_2d,
            d_s_buf,
            d_m_buf,
            d_e_buf,
            d_a_buf,
            d_et_buf,
            d_th_buf,
            n_rows,
            n_cols,
            n_d_tiles,
            s_2d.stride(0),
            s_2d.stride(1),
            s_an,
            s_ad,
            s_etn,
            s_etd,
            s_thn,
            s_thd,
            huber_delta=delta,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            NEED_S_GRAD=bool(need_s),
            NEED_M_GRAD=bool(need_m),
            NEED_E_GRAD=bool(need_e),
            NEED_A_GRAD=bool(need_a),
            NEED_ET_GRAD=bool(need_et),
            NEED_TH_GRAD=bool(need_th),
            A_MODE=a_mode,
            ET_MODE=et_mode,
            TH_MODE=th_mode,
            ACC_DTYPE=tl_acc,
            num_warps=4,
            num_stages=2,
        )

        def _finalize_gate(
            buf: torch.Tensor,
            needed: bool,
            mode: int,
            target_shape: tuple[int, ...],
            target_dtype: torch.dtype,
        ) -> torch.Tensor:
            if not needed:
                return torch.empty((0,), device=device, dtype=target_dtype)
            if mode == 0:
                out_acc = _unbroadcast_grad(buf.reshape(orig_shape), target_shape)
            elif mode == 1:
                out_acc = buf.sum(dim=1).reshape(target_shape)
            elif mode == 2:
                out_acc = buf.sum(dim=0).reshape(target_shape)
            else:
                out_acc = buf.sum().reshape(target_shape)
            return out_acc.to(dtype=target_dtype).contiguous()

        return (
            d_s_buf.reshape(orig_shape) if need_s else d_s_buf,
            d_m_buf.reshape(orig_shape).to(dtype=momentum.dtype).contiguous()
            if need_m
            else d_m_buf,
            d_e_buf.reshape(orig_shape).to(dtype=error.dtype).contiguous() if need_e else d_e_buf,
            _finalize_gate(d_a_buf, need_a, a_mode, alpha_shape, alpha.dtype),
            _finalize_gate(d_et_buf, need_et, et_mode, eta_shape, eta.dtype),
            _finalize_gate(d_th_buf, need_th, th_mode, theta_shape, theta.dtype),
        )

    s_a = state.to(dtype=acc_dtype)
    m_a = momentum.to(device=device, dtype=acc_dtype)
    e_a = error.to(device=device, dtype=acc_dtype)
    a_a = alpha.to(device=device, dtype=acc_dtype)
    et_a = eta.to(device=device, dtype=acc_dtype)
    th_a = theta.to(device=device, dtype=acc_dtype)
    gs_a = grad_s.to(device=device, dtype=acc_dtype)
    gm_a = grad_m.to(device=device, dtype=acc_dtype)
    g_tot = gs_a + gm_a
    e_clip = torch.clamp(e_a, min=-delta, max=delta)
    in_huber = (e_a >= -delta) & (e_a <= delta)

    d_s = (
        ((1.0 - a_a) * gs_a).to(dtype=dtype).contiguous()
        if need_s
        else torch.empty((0,), device=device, dtype=dtype)
    )
    d_m = (
        (et_a * g_tot).to(dtype=momentum.dtype).contiguous()
        if need_m
        else torch.empty((0,), device=device, dtype=momentum.dtype)
    )
    d_e = (
        torch.where(in_huber, th_a * g_tot, torch.zeros_like(g_tot))
        .to(dtype=error.dtype)
        .contiguous()
        if need_e
        else torch.empty((0,), device=device, dtype=error.dtype)
    )
    d_a = (
        _unbroadcast_grad(-s_a * gs_a, alpha_shape).to(dtype=alpha.dtype).contiguous()
        if need_a
        else torch.empty((0,), device=device, dtype=alpha.dtype)
    )
    d_et = (
        _unbroadcast_grad(m_a * g_tot, eta_shape).to(dtype=eta.dtype).contiguous()
        if need_et
        else torch.empty((0,), device=device, dtype=eta.dtype)
    )
    d_th = (
        _unbroadcast_grad(e_clip * g_tot, theta_shape).to(dtype=theta.dtype).contiguous()
        if need_th
        else torch.empty((0,), device=device, dtype=theta.dtype)
    )
    return d_s, d_m, d_e, d_a, d_et, d_th


@_neuromod_update_bwd_op.register_fake
def _neuromod_update_bwd_fake(
    state: torch.Tensor,
    momentum: torch.Tensor,
    error: torch.Tensor,
    alpha: torch.Tensor,
    eta: torch.Tensor,
    theta: torch.Tensor,
    grad_s: torch.Tensor,
    grad_m: torch.Tensor,
    huber_delta: float,
    use_triton: bool,
    need_s: bool,
    need_m: bool,
    need_e: bool,
    need_a: bool,
    need_et: bool,
    need_th: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = state.device
    return (
        torch.empty_like(state, memory_format=torch.contiguous_format)
        if need_s
        else torch.empty((0,), device=dev, dtype=state.dtype),
        torch.empty_like(momentum, memory_format=torch.contiguous_format)
        if need_m
        else torch.empty((0,), device=dev, dtype=momentum.dtype),
        torch.empty_like(error, memory_format=torch.contiguous_format)
        if need_e
        else torch.empty((0,), device=dev, dtype=error.dtype),
        torch.empty_like(alpha, memory_format=torch.contiguous_format)
        if need_a
        else torch.empty((0,), device=dev, dtype=alpha.dtype),
        torch.empty_like(eta, memory_format=torch.contiguous_format)
        if need_et
        else torch.empty((0,), device=dev, dtype=eta.dtype),
        torch.empty_like(theta, memory_format=torch.contiguous_format)
        if need_th
        else torch.empty((0,), device=dev, dtype=theta.dtype),
    )


def _neuromod_setup_ctx(ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
    _ = output
    state, momentum, error, alpha, eta, theta, huber_delta, use_triton = inputs
    ctx.save_for_backward(state, momentum, error, alpha, eta, theta)
    ctx.huber_delta = float(huber_delta)
    ctx.use_triton = bool(use_triton)


def _neuromod_backward(
    ctx: Any,
    grad_s: torch.Tensor | None,
    grad_m: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    None,
]:
    state, momentum, error, alpha, eta, theta = ctx.saved_tensors
    need = ctx.needs_input_grad[:6]
    if not any(need):
        return None, None, None, None, None, None, None, None
    gs = torch.zeros_like(state) if grad_s is None else grad_s
    gm = torch.zeros_like(momentum) if grad_m is None else grad_m
    ds, dm, de, da, det, dth = _neuromod_update_bwd_op(
        state,
        momentum,
        error,
        alpha,
        eta,
        theta,
        gs,
        gm,
        ctx.huber_delta,
        ctx.use_triton,
        bool(need[0]),
        bool(need[1]),
        bool(need[2]),
        bool(need[3]),
        bool(need[4]),
        bool(need[5]),
    )
    return (
        ds if need[0] else None,
        dm if need[1] else None,
        de if need[2] else None,
        da if need[3] else None,
        det if need[4] else None,
        dth if need[5] else None,
        None,
        None,
    )


_neuromod_update_fwd_op.register_autograd(_neuromod_backward, setup_context=_neuromod_setup_ctx)


# -----------------------------------------------------------------------------
# 1.b. Envoltorios `torch.autograd.Function` (compatibilidad hacia atrás)
# -----------------------------------------------------------------------------
class FusedCrossSectionalZScoreClipFunction(torch.autograd.Function):
    """Autograd `sm_120` para `fused_cross_sectional_zscore_clip` (compatibilidad directa)."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        clip_val: float,
        eps: float,
        use_triton: bool,
    ) -> torch.Tensor:
        out, mean_1d, inv_std_1d = _zscore_clip_fwd_op(x, float(clip_val), float(eps), use_triton)
        ctx.save_for_backward(x, mean_1d, inv_std_1d)
        ctx.clip_val = float(clip_val)
        ctx.use_triton = bool(use_triton)
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        x, mean_1d, inv_std_1d = ctx.saved_tensors
        return (
            _zscore_clip_bwd_op(x, grad_out, mean_1d, inv_std_1d, ctx.clip_val, ctx.use_triton),
            None,
            None,
            None,
        )


class FusedSoftMoECombineFunction(torch.autograd.Function):
    """Autograd `sm_120` para `fused_soft_moe_combine` (compatibilidad directa)."""

    @staticmethod
    def forward(
        ctx: Any,
        expert_outputs: torch.Tensor,
        router_probs: torch.Tensor,
        shared_output: torch.Tensor,
        mix_gate: torch.Tensor,
        residual: torch.Tensor | None,
        apply_sigmoid: bool,
        use_triton: bool,
    ) -> torch.Tensor:
        out, routed_acc = _soft_moe_combine_fwd_op(
            expert_outputs,
            router_probs,
            shared_output,
            mix_gate,
            residual,
            bool(apply_sigmoid),
            bool(use_triton),
        )
        ctx.save_for_backward(expert_outputs, router_probs, shared_output, mix_gate, routed_acc)
        ctx.apply_sigmoid = bool(apply_sigmoid)
        ctx.use_triton = bool(use_triton)
        ctx.has_residual = residual is not None
        ctx.res_dtype = residual.dtype if residual is not None else expert_outputs.dtype
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        return _soft_moe_backward(ctx, grad_out, None)


class FusedPinballCRPSLossFunction(torch.autograd.Function):
    """Autograd `sm_120` para `fused_pinball_crps_loss` (compatibilidad directa)."""

    @staticmethod
    def forward(
        ctx: Any,
        quantiles: torch.Tensor,
        targets: torch.Tensor,
        taus: torch.Tensor,
        crps_scale: float,
        reduction: str,
        use_triton: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(quantiles, targets, taus)
        ctx.crps_scale = float(crps_scale)
        ctx.reduction = str(reduction)
        ctx.use_triton = bool(use_triton)
        return _pinball_crps_fwd_op(
            quantiles, targets, taus, float(crps_scale), str(reduction), bool(use_triton)
        )

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        return _pinball_backward(ctx, grad_out)


class FusedNeuromodUpdateFunction(torch.autograd.Function):
    """Autograd `sm_120` para `fused_neuromod_update` (compatibilidad directa)."""

    @staticmethod
    def forward(
        ctx: Any,
        state: torch.Tensor,
        momentum: torch.Tensor,
        error: torch.Tensor,
        alpha: torch.Tensor,
        eta: torch.Tensor,
        theta: torch.Tensor,
        huber_delta: float,
        use_triton: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(state, momentum, error, alpha, eta, theta)
        ctx.huber_delta = float(huber_delta)
        ctx.use_triton = bool(use_triton)
        return _neuromod_update_fwd_op(
            state, momentum, error, alpha, eta, theta, float(huber_delta), bool(use_triton)
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_s: torch.Tensor | None,
        grad_m: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        return _neuromod_backward(ctx, grad_s, grad_m)


def fused_cross_sectional_zscore_clip(
    x: torch.Tensor,
    clip_val: float = 8.0,
    eps: float = 1e-6,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Tipificación transversal 2D-coalescida Split-N + recorte vía `torch.library.custom_op`.

    Registra `mars_titan_bw::zscore_clip_fwd`/`bwd` para CERO graph breaks bajo
    `torch.compile(mode="default", dynamic=True)`, tiling 2D `(BLOCK_N=128, BLOCK_F=32/64)`
    con Welford paralelo de Chan Split-N (`>=188` CTAs) y constantes `tl.constexpr` sin truncar.
    """
    if clip_val <= 0.0:
        raise ValueError(f"clip_val debe ser estrictamente positivo; recibido {clip_val}")
    if eps <= 0.0:
        raise ValueError(f"eps debe ser estrictamente positivo; recibido {eps}")
    if x.ndim == 0:
        raise ValueError("x debe tener al menos 1 dimensión (eje transversal dim=0)")
    if x.shape[0] == 0:
        return x.clone()
    out, _, _ = _zscore_clip_fwd_op(x, float(clip_val), float(eps), bool(use_triton))
    return out


def fused_soft_moe_combine(
    expert_outputs: torch.Tensor,
    router_probs: torch.Tensor,
    shared_output: torch.Tensor,
    mix_gate: torch.Tensor | float,
    residual: torch.Tensor | None = None,
    *,
    apply_sigmoid: bool = True,
    use_triton: bool = True,
) -> torch.Tensor:
    """Funde combinación MoE `[N, E, D]` + experto compartido + residual vía `custom_op` `sm_120`.

    Aplica sigmoid FP32 dentro del kernel Triton (`APPLY_SIGMOID: tl.constexpr`) y funde el
    backward de `grad_exp`, `grad_shared`, `grad_probs` y `grad_gate` en una sola pasada.
    """
    if expert_outputs.ndim != 3:
        raise ValueError(
            f"expert_outputs debe tener forma [N, E, D]; recibido {tuple(expert_outputs.shape)}"
        )
    n_assets, n_experts, d_model = expert_outputs.shape
    if router_probs.shape != (n_assets, n_experts):
        raise ValueError(
            f"router_probs debe tener forma {(n_assets, n_experts)}; "
            f"recibido {tuple(router_probs.shape)}"
        )
    if shared_output.shape != (n_assets, d_model):
        raise ValueError(
            f"shared_output debe tener forma {(n_assets, d_model)}; "
            f"recibido {tuple(shared_output.shape)}"
        )
    if residual is not None and residual.shape != (n_assets, d_model):
        raise ValueError(
            f"residual debe tener forma {(n_assets, d_model)}; recibido {tuple(residual.shape)}"
        )

    device, dtype = expert_outputs.device, expert_outputs.dtype
    gate_t = (
        mix_gate.to(device=device)
        if isinstance(mix_gate, torch.Tensor)
        else shared_output.new_full((1, 1), float(mix_gate))
    )
    if n_assets == 0 or d_model == 0:
        return torch.empty((n_assets, d_model), device=device, dtype=dtype)

    out, _ = _soft_moe_combine_fwd_op(
        expert_outputs,
        router_probs,
        shared_output,
        gate_t,
        residual,
        bool(apply_sigmoid),
        bool(use_triton),
    )
    return out


def fused_pinball_crps_loss(
    quantiles: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor | Sequence[float],
    *,
    crps_scale: float = 2.0,
    reduction: str = "mean",
    use_triton: bool = True,
) -> torch.Tensor:
    """Pérdida Pinball/CRPS `Q`-cuantiles 2D-coalescida (`BLOCK_M=64, warps=2`) vía `custom_op`."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction inválida {reduction!r}; válidas: ('mean', 'sum', 'none')")
    if quantiles.ndim not in (2, 3):
        raise ValueError(
            f"quantiles debe tener forma [N, Q] o [N, H, Q]; recibido {tuple(quantiles.shape)}"
        )

    device, dtype = quantiles.device, quantiles.dtype
    n_q = quantiles.shape[-1]
    taus_t = (
        taus.to(device=device, dtype=dtype).reshape(-1)
        if isinstance(taus, torch.Tensor)
        else _get_cached_taus_tensor(taus, device=device, dtype=dtype)
    )
    if taus_t.numel() != n_q:
        raise ValueError(
            f"Longitud de taus ({taus_t.numel()}) != dimensión cuantil Q ({n_q})"
        )

    expected_item_shape = quantiles.shape[:-1]
    y_t = targets.to(device=device, dtype=dtype)
    if y_t.ndim == quantiles.ndim and y_t.shape[-1] == 1:
        y_t = y_t.squeeze(-1)
    if y_t.shape != expected_item_shape:
        raise ValueError(
            f"targets debe tener forma {expected_item_shape}; recibido {tuple(targets.shape)}"
        )

    if y_t.numel() == 0:
        if reduction == "none":
            return torch.empty(expected_item_shape, device=device, dtype=dtype)
        return torch.zeros((), device=device, dtype=dtype)

    return _pinball_crps_fwd_op(
        quantiles,
        y_t,
        taus_t,
        float(crps_scale),
        reduction,
        bool(use_triton),
    )


fused_crps_pinball_loss = fused_pinball_crps_loss


def fused_neuromod_update(
    state: torch.Tensor,
    momentum: torch.Tensor,
    error: torch.Tensor,
    alpha: torch.Tensor,
    eta: torch.Tensor,
    theta: torch.Tensor,
    *,
    huber_delta: float = 1.0,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Actualización PT1 + Huber 2D con strides broadcast, reducción FP32 y `custom_op` `sm_120`."""
    if huber_delta <= 0.0:
        raise ValueError(f"huber_delta debe ser estrictamente positivo; recibido {huber_delta}")
    if state.shape != momentum.shape or state.shape != error.shape:
        raise ValueError(
            f"state {tuple(state.shape)}, momentum {tuple(momentum.shape)} y "
            f"error {tuple(error.shape)} deben tener exactamente la misma forma"
        )
    if state.numel() == 0:
        return (
            state.clone(memory_format=torch.contiguous_format),
            momentum.clone(memory_format=torch.contiguous_format),
        )
    return _neuromod_update_fwd_op(
        state,
        momentum,
        error,
        alpha,
        eta,
        theta,
        float(huber_delta),
        bool(use_triton),
    )


# =============================================================================
# 2. CUDAGraphSessionRunner (cubetas rellenadas 64/128/188 + eager en CPU)
# =============================================================================
class CUDAGraphSessionRunner:
    """Ejecutor de pasos de sesión con CUDA Graphs por cubeta (`bucket_n`) y respaldo CPU eager.

    Elimina la latencia de lanzamiento Python en los 188 SMs de RTX PRO 6000 (`sm_120`) al capturar
    un grafo CUDA estático por tamaño de cubeta rellenado (`bucket_n`, ej. múltiplo de 64/128/188).
    En CPU o cuando autograd está activo (salvo `allow_grad=True`), aplica el mismo rellenado y
    des-rellenado de cubetas en modo eager transparente para garantizar paridad verificable.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        bucket_multiple: int = 64,
        buckets: Sequence[int] | None = None,
        warmup_steps: int = 3,
        use_cuda_graph: bool = True,
        pad_on_cpu: bool = True,
        pad_mask_key: str | None = None,
        non_asset_keys: Iterable[str] = ("regime_vec", "ne_signal"),
        allow_grad: bool = False,
        allow_unmasked_padding: bool = False,
    ) -> None:
        if bucket_multiple < 1:
            raise ValueError(f"bucket_multiple debe ser >= 1; recibido {bucket_multiple}")
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps debe ser >= 1; recibido {warmup_steps}")
        sorted_buckets: tuple[int, ...] | None = None
        if buckets is not None:
            b_list = sorted({int(b) for b in buckets})
            if not b_list or b_list[0] < 1:
                raise ValueError("buckets debe contener enteros positivos")
            sorted_buckets = tuple(b_list)

        self.fn = fn
        self.bucket_multiple = int(bucket_multiple)
        self.buckets = sorted_buckets
        self.warmup_steps = int(warmup_steps)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.pad_on_cpu = bool(pad_on_cpu)
        self.pad_mask_key = pad_mask_key
        self.non_asset_keys: frozenset[str] = frozenset(non_asset_keys)
        self.allow_grad = bool(allow_grad)
        self.allow_unmasked_padding = bool(allow_unmasked_padding)

        self._graphs: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._bucket_hits: dict[int, int] = {}
        self._eager_fallbacks: int = 0
        self._graph_captures: int = 0
        self._graph_replays: int = 0
        self._last_bucket_n: int | None = None
        self._last_device_mode: str = "cpu_eager"

    @staticmethod
    def _resolve_stateful_module(fn: Callable[[Any], Any]) -> nn.Module | None:
        if isinstance(fn, nn.Module):
            return fn
        owner = getattr(fn, "__self__", None)
        if isinstance(owner, nn.Module):
            return owner
        return None

    @staticmethod
    def _snapshot_module_buffers(mod: nn.Module | None) -> dict[str, torch.Tensor]:
        if mod is None:
            return {}
        return {name: buf.detach().clone() for name, buf in mod.named_buffers()}

    @staticmethod
    def _restore_module_buffers(
        mod: nn.Module | None, snapshot: dict[str, torch.Tensor]
    ) -> None:
        if mod is None or not snapshot:
            return
        with torch.no_grad():
            for name, buf in mod.named_buffers():
                saved = snapshot.get(name)
                if saved is not None:
                    buf.copy_(saved)

    def select_bucket(self, n_assets: int) -> int:
        """Devuelve el tamaño de cubeta `bucket_n >= n_assets` alineado a 64/128/188."""
        if n_assets < 1:
            raise ValueError(f"n_assets debe ser >= 1; recibido {n_assets}")
        if self.buckets is not None:
            for b in self.buckets:
                if b >= n_assets:
                    return b
        m = self.bucket_multiple
        return ((n_assets + m - 1) // m) * m

    @staticmethod
    def _infer_n_assets_and_device(inp: Any) -> tuple[int, torch.device, torch.dtype]:
        if isinstance(inp, torch.Tensor):
            if inp.ndim < 1:
                raise ValueError("El tensor de entrada debe tener al menos 1 dimensión (N)")
            return int(inp.shape[0]), inp.device, inp.dtype
        if isinstance(inp, dict):
            if "prices" in inp and isinstance(inp["prices"], torch.Tensor):
                t = inp["prices"]
                return int(t.shape[0]), t.device, t.dtype
            for v in inp.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 2:
                    return int(v.shape[0]), v.device, v.dtype
            for v in inp.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    return int(v.shape[0]), v.device, v.dtype
        raise TypeError(
            "CUDAGraphSessionRunner requiere un torch.Tensor o un dict[str, Tensor] con eje N"
        )

    def _pad_tensor(
        self,
        t: torch.Tensor,
        n_assets: int,
        bucket_n: int,
        *,
        force_clone: bool = False,
    ) -> torch.Tensor:
        if t.ndim < 1 or t.shape[0] != n_assets or bucket_n == n_assets:
            return t.clone() if force_clone else t
        out = torch.zeros((bucket_n, *t.shape[1:]), device=t.device, dtype=t.dtype)
        out[:n_assets].copy_(t)
        return out

    def _pad_input(
        self,
        inp: Any,
        n_assets: int,
        bucket_n: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        force_clone: bool = False,
    ) -> Any:
        if isinstance(inp, torch.Tensor):
            return self._pad_tensor(inp, n_assets, bucket_n, force_clone=force_clone)
        if isinstance(inp, dict):
            padded: dict[str, Any] = {}
            for k, v in inp.items():
                if not isinstance(v, torch.Tensor):
                    padded[k] = v
                elif k in self.non_asset_keys:
                    padded[k] = v.clone() if force_clone else v
                else:
                    padded[k] = self._pad_tensor(
                        v, n_assets, bucket_n, force_clone=force_clone
                    )
            if self.pad_mask_key is not None:
                mask = torch.zeros((bucket_n,), device=device, dtype=dtype)
                mask[:n_assets] = 1.0
                padded[self.pad_mask_key] = mask
            return padded
        return inp

    def _copy_into_static(self, static_in: Any, inp: Any, n_assets: int) -> None:
        if isinstance(static_in, torch.Tensor) and isinstance(inp, torch.Tensor):
            static_in[:n_assets].copy_(inp)
            if static_in.shape[0] > n_assets:
                static_in[n_assets:].zero_()
            return
        if isinstance(static_in, dict) and isinstance(inp, dict):
            for k, buf in static_in.items():
                if k == self.pad_mask_key and isinstance(buf, torch.Tensor):
                    buf.zero_()
                    buf[:n_assets] = 1.0
                elif (
                    k in inp
                    and isinstance(buf, torch.Tensor)
                    and isinstance(inp[k], torch.Tensor)
                ):
                    src = inp[k]
                    is_asset = (
                        k not in self.non_asset_keys
                        and buf.ndim >= 1
                        and src.ndim >= 1
                        and src.shape[0] == n_assets
                    )
                    if is_asset:
                        buf[:n_assets].copy_(src)
                        if buf.shape[0] > n_assets:
                            buf[n_assets:].zero_()
                    else:
                        buf.copy_(src)

    def _unpad_output(self, out: Any, n_assets: int, bucket_n: int, *, clone: bool) -> Any:
        if isinstance(out, torch.Tensor):
            if out.ndim >= 1 and out.shape[0] == bucket_n:
                sliced = out[:n_assets]
                return sliced.clone() if clone else sliced
            return out.clone() if clone else out
        if isinstance(out, dict):
            return {
                k: (
                    (v.clone() if (clone and isinstance(v, torch.Tensor)) else v)
                    if k in self.non_asset_keys
                    else self._unpad_output(v, n_assets, bucket_n, clone=clone)
                )
                for k, v in out.items()
            }
        if isinstance(out, tuple):
            return tuple(self._unpad_output(v, n_assets, bucket_n, clone=clone) for v in out)
        if isinstance(out, list):
            return [self._unpad_output(v, n_assets, bucket_n, clone=clone) for v in out]
        return out

    def _input_signature(self, inp: Any, n_assets: int, bucket_n: int) -> tuple[Any, ...]:
        if isinstance(inp, torch.Tensor):
            return ("tensor", bucket_n, tuple(inp.shape[1:]), inp.dtype, str(inp.device))
        if isinstance(inp, dict):
            sig = []
            for k in sorted(inp.keys()):
                v = inp[k]
                if isinstance(v, torch.Tensor):
                    is_asset = (
                        k not in self.non_asset_keys
                        and v.ndim >= 1
                        and v.shape[0] == n_assets
                    )
                    tail = tuple(v.shape[1:]) if is_asset else tuple(v.shape)
                    sig.append((k, is_asset, tail, v.dtype, str(v.device)))
                else:
                    sig.append((k, type(v).__name__))
            return ("dict", bucket_n, tuple(sig))
        return ("other", bucket_n)

    def __call__(self, inp: Any) -> Any:
        n_assets, device, dtype = self._infer_n_assets_and_device(inp)
        if self.pad_mask_key is not None and isinstance(inp, torch.Tensor):
            raise ValueError(
                "pad_mask_key requiere entrada dict[str, Tensor] para inyectar valid_mask"
            )
        selected_bucket = self.select_bucket(n_assets)
        # P0-2: sin pad_mask_key ni allow_unmasked_padding, usar SOLO tamaño exacto (0 ghost rows)
        can_pad = (self.pad_mask_key is not None) or self.allow_unmasked_padding
        bucket_n = selected_bucket if can_pad else n_assets
        self._last_bucket_n = bucket_n
        self._bucket_hits[bucket_n] = self._bucket_hits.get(bucket_n, 0) + 1

        can_graph = (
            self.use_cuda_graph
            and device.type == "cuda"
            and torch.cuda.is_available()
            and (self.allow_grad or not torch.is_grad_enabled())
        )
        if can_graph:  # pragma: no cover - ruta exclusiva CUDA Graph en GPU
            key = self._input_signature(inp, n_assets, bucket_n)
            entry = self._graphs.get(key)
            if entry is None:
                static_in = self._pad_input(
                    inp, n_assets, bucket_n, device, dtype, force_clone=True
                )
                warmup_in = self._pad_input(
                    inp, n_assets, bucket_n, device, dtype, force_clone=True
                )
                mod = self._resolve_stateful_module(self.fn)
                buf_snap = self._snapshot_module_buffers(mod)

                side_stream = torch.cuda.Stream(device=device)
                side_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side_stream):
                    for _ in range(self.warmup_steps):
                        _ = self.fn(warmup_in)
                    # Restaurar buffers antes de capturar para que warmup no corrompa estado
                    self._restore_module_buffers(mod, buf_snap)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=side_stream):
                        static_out = self.fn(static_in)
                    # Restaurar buffers tras captura para que el replay inicial aplique el estado 1x
                    self._restore_module_buffers(mod, buf_snap)
                torch.cuda.current_stream(device).wait_stream(side_stream)

                entry = dict(graph=graph, static_in=static_in, static_out=static_out)
                self._graphs[key] = entry
                self._graph_captures += 1
            else:
                self._copy_into_static(entry["static_in"], inp, n_assets)

            entry["graph"].replay()
            self._graph_replays += 1
            self._last_device_mode = "cuda_graph"
            return self._unpad_output(entry["static_out"], n_assets, bucket_n, clone=True)

        # Respaldo transparente eager (CPU o modo con gradientes)
        self._eager_fallbacks += 1
        self._last_device_mode = "cpu_eager" if device.type == "cpu" else "cuda_eager"
        if (self.pad_on_cpu or device.type == "cuda") and bucket_n != n_assets:
            padded = self._pad_input(inp, n_assets, bucket_n, device, dtype)
            raw_out = self.fn(padded)
            return self._unpad_output(raw_out, n_assets, bucket_n, clone=False)
        if self.pad_mask_key is not None and isinstance(inp, dict):
            padded = self._pad_input(inp, n_assets, bucket_n, device, dtype)
            return self.fn(padded)
        return self.fn(inp)

    def reset(self) -> None:
        """Libera los grafos CUDA capturados y reinicia contadores de telemetría."""
        self._graphs.clear()
        self._bucket_hits.clear()
        self._eager_fallbacks = 0
        self._graph_captures = 0
        self._graph_replays = 0
        self._last_bucket_n = None

    def stats(self) -> dict[str, object]:
        """Devuelve estadísticas de cubetas, capturas CUDA Graph y respaldos eager."""
        return dict(
            device_mode=self._last_device_mode,
            bucket_multiple=self.bucket_multiple,
            captured_buckets=len(self._graphs),
            graph_captures=self._graph_captures,
            graph_replays=self._graph_replays,
            eager_fallbacks=self._eager_fallbacks,
            last_bucket_n=self._last_bucket_n,
            bucket_hits=dict(self._bucket_hits),
        )


# =============================================================================
# 3. Auditoría AST anti host-sync y Auditoría Roofline RTX PRO 6000 (`sm_120`)
# =============================================================================
_HOST_SYNC_ATTRS: frozenset[str] = frozenset({"item", "cpu", "numpy", "tolist", "synchronize"})


class _HostSyncASTVisitor(ast.NodeVisitor):
    """Detector AST de llamadas host-sync ocultas (`.item()`, `.cpu()`, `.numpy()`, etc.)."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._class_stack: list[str] = []
        self._func_stack: list[tuple[str, bool]] = []
        self._no_grad_depth: int = 0
        self.findings: list[dict[str, object]] = []

    def _current_fn(self) -> tuple[str, bool]:
        if not self._func_stack:
            return "<module>", False
        fn_name, is_fwd = self._func_stack[-1]
        if self._class_stack:
            return f"{self._class_stack[-1]}.{fn_name}", is_fwd
        return fn_name, is_fwd

    @staticmethod
    def _is_no_grad_ctx(item: ast.withitem) -> bool:
        expr = item.context_expr
        if isinstance(expr, ast.Call):
            fn = expr.func
            if isinstance(fn, ast.Attribute) and fn.attr in ("no_grad", "inference_mode"):
                return True
            if isinstance(fn, ast.Name) and fn.id in ("no_grad", "inference_mode"):
                return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_fwd = (
            node.name == "forward"
            or node.name.startswith("forward_")
            or node.name in ("_step_loss",)
        )
        self._func_stack.append((node.name, is_fwd))
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        is_fwd = node.name == "forward" or node.name.startswith("forward_")
        self._func_stack.append((node.name, is_fwd))
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_With(self, node: ast.With) -> None:
        entered = any(self._is_no_grad_ctx(it) for it in node.items)
        if entered:
            self._no_grad_depth += 1
        self.generic_visit(node)
        if entered:
            self._no_grad_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        fn_label, is_fwd = self._current_fn()
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in _HOST_SYNC_ATTRS:
                self.findings.append(
                    dict(
                        file=self.filepath,
                        line=int(node.lineno),
                        function=fn_label,
                        call=f".{attr}()",
                        in_forward=is_fwd,
                        in_no_grad=self._no_grad_depth > 0,
                        severity="high" if is_fwd else "medium",
                    )
                )
            elif attr == "to":
                has_cpu_arg = any(
                    isinstance(a, ast.Constant) and a.value == "cpu" for a in node.args
                ) or any(
                    kw.arg == "device"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "cpu"
                    for kw in node.keywords
                )
                if has_cpu_arg:
                    self.findings.append(
                        dict(
                            file=self.filepath,
                            line=int(node.lineno),
                            function=fn_label,
                            call='.to("cpu")',
                            in_forward=is_fwd,
                            in_no_grad=self._no_grad_depth > 0,
                            severity="high" if is_fwd else "medium",
                        )
                    )
            elif attr == "clip_grad_norm_":
                for kw in node.keywords:
                    if (
                        kw.arg == "error_if_nonfinite"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        self.findings.append(
                            dict(
                                file=self.filepath,
                                line=int(node.lineno),
                                function=fn_label,
                                call="clip_grad_norm_(error_if_nonfinite=True)",
                                in_forward=is_fwd,
                                in_no_grad=self._no_grad_depth > 0,
                                severity="high" if is_fwd else "medium",
                            )
                        )
        self.generic_visit(node)


def audit_zero_host_syncs(
    module_paths: Sequence[str | Path] | str | Path,
) -> dict[str, object]:
    """Audita estáticamente archivos Python buscando trampas host-sync en rutas calientes GPU."""
    paths = [module_paths] if isinstance(module_paths, (str, Path)) else list(module_paths)
    findings: list[dict[str, object]] = []
    files_scanned = 0

    for raw_p in paths:
        p = Path(raw_p)
        if not p.is_file():
            continue
        source = p.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(p))
        visitor = _HostSyncASTVisitor(str(p))
        visitor.visit(tree)
        findings.extend(visitor.findings)
        files_scanned += 1

    forward_findings = sum(1 for f in findings if f["in_forward"])
    return dict(
        files_scanned=files_scanned,
        total_findings=len(findings),
        forward_findings=forward_findings,
        clean=len(findings) == 0,
        is_zero_host_sync_forward=forward_findings == 0,
        findings=findings,
    )


def _discover_model_source_files(model: nn.Module) -> list[Path]:
    """Descubre archivos `.py` de usuario definidos por el modelo y sus submódulos."""
    seen: set[Path] = set()
    files: list[Path] = []
    for mod in model.modules():
        try:
            src = inspect.getsourcefile(type(mod))
        except (TypeError, OSError):
            continue
        if not src:
            continue
        p = Path(src).resolve()
        if "site-packages" in p.parts or "dist-packages" in p.parts or not p.is_file():
            continue
        if p not in seen:
            seen.add(p)
            files.append(p)
    return files


def _leaf_module_cost(
    mod: nn.Module,
    n_assets: int,
    compute_bytes: int,
) -> tuple[int, int, int]:
    """Calcula `(flops, bytes, act_elems)` de un solo paso para los parámetros propios de `mod`."""
    if isinstance(mod, nn.Linear):
        in_f, out_f = int(mod.in_features), int(mod.out_features)
        flops = 2 * n_assets * in_f * out_f + (n_assets * out_f if mod.bias is not None else 0)
        mem_b = compute_bytes * (n_assets * in_f + in_f * out_f + n_assets * out_f)
        if mod.bias is not None:
            mem_b += compute_bytes * out_f
        return flops, mem_b, n_assets * (in_f + out_f)

    if isinstance(mod, nn.LayerNorm):
        dim = math.prod(int(s) for s in mod.normalized_shape)
        return 5 * n_assets * dim, compute_bytes * (2 * n_assets * dim + 2 * dim), n_assets * dim

    if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
        k_prod = math.prod(int(k) for k in mod.kernel_size)
        c_in = int(mod.in_channels) // int(mod.groups)
        c_out = int(mod.out_channels)
        flops = 2 * n_assets * c_in * c_out * k_prod
        mem_b = compute_bytes * (
            n_assets * int(mod.in_channels) + c_in * c_out * k_prod + n_assets * c_out
        )
        return flops, mem_b, n_assets * c_out

    flops = 0
    mem_b = 0
    acts = 0
    for param in mod.parameters(recurse=False):
        if param.ndim == 3:
            e_cnt, k_dim, m_dim = int(param.shape[0]), int(param.shape[1]), int(param.shape[2])
            flops += 2 * n_assets * e_cnt * k_dim * m_dim
            mem_b += compute_bytes * (
                n_assets * k_dim + e_cnt * k_dim * m_dim + n_assets * e_cnt * m_dim
            )
            acts += n_assets * e_cnt * m_dim
        elif param.ndim == 2:
            r_dim, c_dim = int(param.shape[0]), int(param.shape[1])
            flops += 2 * n_assets * r_dim * c_dim
            mem_b += compute_bytes * (n_assets * c_dim + r_dim * c_dim + n_assets * r_dim)
            acts += n_assets * max(r_dim, c_dim)
        elif param.ndim == 1:
            dim = int(param.shape[0])
            flops += n_assets * dim
            mem_b += compute_bytes * (n_assets * dim + dim)
            acts += n_assets * dim
    return flops, mem_b, acts


def _sdpa_module_cost(
    mod: nn.Module,
    n_assets: int,
    compute_bytes: int,
    precision_mode: str,
) -> tuple[int, int, int, dict[str, object] | None]:
    """Estima FLOPs/tráfico/activaciones SDPA `O(S^2 D)` (cross-asset, áreas y cross-modal)."""
    materialize_attn = precision_mode == "fp32_strict"

    # 1) Atención multi-cabeza estándar (cross-asset O(N^2 D) por omisión salvo `attention_seq_len`)
    if isinstance(mod, nn.MultiheadAttention):
        d_h = int(mod.embed_dim)
        n_heads = int(mod.num_heads)
        seq_len = int(getattr(mod, "attention_seq_len", n_assets))
        batch_cnt = 1 if seq_len == n_assets else n_assets
        s2 = seq_len * seq_len
        flops = 4 * batch_cnt * s2 * d_h + 5 * batch_cnt * n_heads * s2
        attn_b = 2 * batch_cnt * n_heads * s2 if materialize_attn else 0
        mem_b = compute_bytes * (4 * batch_cnt * seq_len * d_h + attn_b)
        acts = batch_cnt * seq_len * d_h + (
            batch_cnt * n_heads * seq_len * seq_len
            if materialize_attn
            else batch_cnt * n_heads * seq_len
        )
        return (
            flops,
            mem_b,
            acts,
            dict(
                module_type=type(mod).__name__,
                kind="cross_asset_sdpa" if seq_len == n_assets else "sequence_sdpa",
                batch_count=batch_cnt,
                seq_len=seq_len,
                hidden_dim=d_h,
                num_heads=n_heads,
                base_flops=flops,
            ),
        )

    # 2) Cerebro Holográfico 3L / Atención entre áreas corticales + ranuras ocultas
    if hasattr(mod, "n_cortical_areas") and hasattr(mod, "n_hidden_slots"):
        n_nodes = int(
            getattr(mod, "n_nodes", int(mod.n_cortical_areas) + int(mod.n_hidden_slots))
        )
        d_h = int(getattr(mod, "hidden_dim", 128))
        n_heads = int(getattr(mod, "n_heads", getattr(mod, "num_heads", 4)))
        flops = 4 * n_assets * n_nodes * n_nodes * d_h + 5 * n_assets * n_heads * n_nodes * n_nodes
        attn_b = 2 * n_assets * n_heads * n_nodes * n_nodes if materialize_attn else 0
        mem_b = compute_bytes * (4 * n_assets * n_nodes * d_h + attn_b)
        acts = n_assets * n_nodes * d_h + (
            n_assets * n_heads * n_nodes * n_nodes
            if materialize_attn
            else n_assets * n_heads * n_nodes
        )
        return (
            flops,
            mem_b,
            acts,
            dict(
                module_type=type(mod).__name__,
                kind="cross_area_sdpa",
                batch_count=n_assets,
                seq_len=n_nodes,
                hidden_dim=d_h,
                num_heads=n_heads,
                base_flops=flops,
            ),
        )

    # 3) Fusión / alineamiento cross-modal con SDPA sobre modalidades + token física
    if hasattr(mod, "n_modalities") and (hasattr(mod, "n_heads") or hasattr(mod, "head_dim")):
        n_mod = int(mod.n_modalities)
        cls_name = type(mod).__name__
        q_len = n_mod if "Alignment" in cls_name else (n_mod + 1)
        kv_len = n_mod + 1
        d_h = int(getattr(mod, "hidden_dim", 128))
        n_heads = int(getattr(mod, "n_heads", 4))
        flops = 4 * n_assets * q_len * kv_len * d_h + 5 * n_assets * n_heads * q_len * kv_len
        attn_b = 2 * n_assets * n_heads * q_len * kv_len if materialize_attn else 0
        mem_b = compute_bytes * (2 * n_assets * (q_len + kv_len) * d_h + attn_b)
        acts = n_assets * q_len * d_h + (
            n_assets * n_heads * q_len * kv_len if materialize_attn else n_assets * n_heads * q_len
        )
        return (
            flops,
            mem_b,
            acts,
            dict(
                module_type=cls_name,
                kind="cross_modal_sdpa",
                batch_count=n_assets,
                seq_len=kv_len,
                hidden_dim=d_h,
                num_heads=n_heads,
                base_flops=flops,
            ),
        )

    return 0, 0, 0, None


def _estimate_model_flops_bytes_and_acts(
    model: nn.Module,
    n_assets: int,
    compute_bytes: int,
    *,
    precision_mode: str = "fp32_strict",
    recurrence_steps: int | None = None,
) -> dict[str, Any]:
    """Estima FLOPs, bytes HBM y activaciones incluyendo recurrencia (`max_loops`/`thinking_steps`)
    y atención SDPA `O(S^2 D)` / cross-area / cross-modal sin doble conteo."""
    total_flops = 0
    base_flops = 0
    total_bytes = 0
    total_act_elems = 0
    sdpa_flops_total = 0
    sdpa_bytes_total = 0
    sdpa_details: list[dict[str, object]] = []
    detected_max_loops = 1
    detected_thinking_steps = 1

    # Detectar si el modelo raíz tiene un bucle `recursive_core` que evalúa `head` `max_loops` veces
    parent_head_loops = 1
    rec_core = getattr(model, "recursive_core", None)
    if rec_core is not None and isinstance(getattr(rec_core, "max_loops", None), int):
        parent_head_loops = max(1, int(rec_core.max_loops))

    def _walk(mod: nn.Module, mult: int) -> None:
        nonlocal total_flops, base_flops, total_bytes, total_act_elems
        nonlocal sdpa_flops_total, sdpa_bytes_total
        nonlocal detected_max_loops, detected_thinking_steps

        lf, lb, la = _leaf_module_cost(mod, n_assets, compute_bytes)
        base_flops += lf
        total_flops += lf * mult
        total_bytes += lb * mult
        total_act_elems += la * mult

        sf, sb, sa, s_info = _sdpa_module_cost(mod, n_assets, compute_bytes, precision_mode)
        if s_info is not None:
            base_flops += sf
            sdpa_flops_total += sf * mult
            sdpa_bytes_total += sb * mult
            total_flops += sf * mult
            total_bytes += sb * mult
            total_act_elems += sa * mult
            s_entry = dict(s_info)
            s_entry["effective_flops"] = sf * mult
            s_entry["recurrence_multiplier"] = mult
            sdpa_details.append(s_entry)

        # Determinar multiplicador de recurrencia para hijos
        t_steps = getattr(mod, "thinking_steps", None)
        m_loops = getattr(mod, "max_loops", None)
        n_cycles = getattr(mod, "n_cycles", None)
        if isinstance(t_steps, int) and t_steps > 1:
            detected_thinking_steps = max(detected_thinking_steps, t_steps)
        if isinstance(m_loops, int) and m_loops > 1:
            detected_max_loops = max(detected_max_loops, m_loops)
        if isinstance(n_cycles, int) and n_cycles > 1:
            detected_max_loops = max(detected_max_loops, n_cycles)

        has_unrolled_blocks = isinstance(getattr(mod, "step_blocks", None), nn.ModuleList)

        for child_name, child in mod.named_children():
            child_mult = mult
            if mod is model and child_name == "head" and parent_head_loops > 1:
                child_mult = mult * parent_head_loops
            elif isinstance(t_steps, int) and t_steps > 1:
                # Core cognitivo con pesos compartidos: contexto corre 1x; core y tools Kx
                if child_name not in ("context_engineer",):
                    child_mult = mult * t_steps
            elif has_unrolled_blocks and isinstance(m_loops, int) and m_loops > 1:
                # PCTRMRecursiveLoop: step_blocks ya son K copias distintas; y_feedback corre K-1x
                if child_name == "y_feedback":
                    child_mult = mult * max(1, m_loops - 1)
            elif isinstance(n_cycles, int) and n_cycles > 1:
                enc_names = (
                    "enc_prices",
                    "enc_fundamentals",
                    "enc_macro",
                    "enc_news",
                    "enc_charts",
                )
                if child_name not in enc_names:
                    child_mult = mult * n_cycles
            elif isinstance(m_loops, int) and m_loops > 1 and not has_unrolled_blocks:
                child_mult = mult * m_loops
            _walk(child, child_mult)

    root_mult = max(1, int(recurrence_steps)) if recurrence_steps is not None else 1
    _walk(model, root_mult)

    recurrence_extra_flops = max(0, total_flops - base_flops)
    eff_mult = float(total_flops) / float(max(base_flops, 1))
    return dict(
        forward_flops=total_flops,
        base_single_pass_flops=base_flops,
        forward_bytes=total_bytes,
        act_elems=total_act_elems,
        recurrence_audit=dict(
            detected_max_loops=int(detected_max_loops),
            detected_thinking_steps=int(detected_thinking_steps),
            effective_recurrence_multiplier=round(eff_mult, 4),
            recurrence_extra_flops=int(recurrence_extra_flops),
        ),
        sdpa_audit=dict(
            sdpa_modules_detected=len(sdpa_details),
            sdpa_flops=int(sdpa_flops_total),
            sdpa_bytes=int(sdpa_bytes_total),
            sdpa_details=sdpa_details,
        ),
    )


def blackwell_roofline_audit(
    model: nn.Module,
    n_assets: int = 2600,
    precision_mode: str = "fp32_strict",
    *,
    training: bool = True,
    safety_headroom: float = 0.88,
    block_n: int = 256,
    recurrence_steps: int | None = None,
    module_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    """Auditoría Roofline + olas de 188 SMs + VRAM MiB + CUDA MPS + AST host-sync en `sm_120`.

    Evalúa un modelo PyTorch (incluyendo multiplicadores de recurrencia `max_loops`/
    `thinking_steps` y atención SDPA `O(S^2 D)` / cross-area / cross-modal) frente a las
    especificaciones exactas de RTX PRO 6000 Blackwell:
      - Cresta FP32 estricto: `75.1 FLOP/byte` (`120 TFLOPS / 1,597 GB/s`).
      - Cresta BF16 `mma.sync`: `626.0 FLOP/byte` (`1,000 TFLOPS / 1,597 GB/s`).
      - Cresta TF32: `146.5 FLOP/byte` (`234 TFLOPS / 1,597 GB/s`).
    """
    if precision_mode not in PRECISION_MODES:
        raise ValueError(
            f"precision_mode inválido {precision_mode!r}; válidos: {PRECISION_MODES}"
        )
    if n_assets < 1:
        raise ValueError(f"n_assets debe ser >= 1; recibido {n_assets}")
    if not (0.1 <= safety_headroom <= 0.98):
        raise ValueError("safety_headroom debe pertenecer a [0.1, 0.98]")

    if precision_mode == "fp32_strict":
        peak_tflops = float(BLACKWELL_SPECS["fp32_strict_tflops"])
        ridge_point = float(BLACKWELL_SPECS["fp32_ridge_flop_per_byte"])
        compute_bytes = 4
    elif precision_mode == "bf16_amp":
        peak_tflops = float(BLACKWELL_SPECS["bf16_tflops"])
        ridge_point = float(BLACKWELL_SPECS["bf16_ridge_flop_per_byte"])
        compute_bytes = 2
    else:  # tf32_fast
        peak_tflops = SM120_TF32_TFLOPS
        ridge_point = round((SM120_TF32_TFLOPS * 1000.0) / SM120_BANDWIDTH_GBS, 1)
        compute_bytes = 4

    est = _estimate_model_flops_bytes_and_acts(
        model,
        n_assets=n_assets,
        compute_bytes=compute_bytes,
        precision_mode=precision_mode,
        recurrence_steps=recurrence_steps,
    )
    fwd_flops = int(est["forward_flops"])
    fwd_bytes = int(est["forward_bytes"])
    act_elems = int(est["act_elems"])
    ai = float(fwd_flops) / float(max(fwd_bytes, 1))
    bottleneck = "memory_bound" if ai < ridge_point else "compute_bound"
    achievable_tflops = min(peak_tflops, (ai * SM120_BANDWIDTH_GBS) / 1000.0)

    # Alineación de olas con los 188 SMs de GB202
    row_waves = math.ceil(n_assets / SM120_SMS)
    row_wave_eff = float(n_assets) / float(row_waves * SM120_SMS)
    tile_ctas = math.ceil(n_assets / max(1, block_n))
    padded_188 = math.ceil(n_assets / SM120_SMS) * SM120_SMS
    padded_64 = math.ceil(n_assets / 64) * 64

    # Estimación detallada de VRAM pico por worker (en MiB)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    if training and precision_mode == "bf16_amp":
        # Pesos maestros FP32 + copia BF16 en forward/backward
        weights_bytes = int(weights_bytes * 1.5)
    grads_bytes = n_trainable * 4 if training else 0
    optimizer_bytes = n_trainable * 8 if training else 0  # AdamW (exp_avg + exp_avg_sq FP32)
    activations_bytes = int(act_elems * compute_bytes * (2.0 if training else 1.0))

    mib = 1024.0 * 1024.0
    weights_mib = weights_bytes / mib
    grads_mib = grads_bytes / mib
    optimizer_mib = optimizer_bytes / mib
    activations_mib = activations_bytes / mib
    cuda_context_mib = 650.0  # Contexto CUDA/cuBLAS/asignador por cliente MPS
    total_worker_mib = (
        weights_mib + grads_mib + optimizer_mib + activations_mib + cuda_context_mib
    )
    total_worker_gb = total_worker_mib / 1024.0

    vram_gb_total = float(BLACKWELL_SPECS["vram_gb"])
    usable_vram_mib = vram_gb_total * 1024.0 * safety_headroom
    fits_single_gpu = total_worker_mib <= usable_vram_mib
    max_workers_by_vram = max(1, int(usable_vram_mib // max(total_worker_mib, 1.0)))
    # Punto óptimo práctico CUDA MPS en 188 SMs / 96 GB GDDR7 (6-8 workers sin thrashing CPU/PCIe)
    recommended_workers = max(1, min(max_workers_by_vram, 8))
    est_total_vram_gb = round(recommended_workers * total_worker_gb, 3)
    vram_frac = round(min(safety_headroom / recommended_workers, 0.90), 3)
    vram_cap_gb = round((vram_gb_total * safety_headroom) / recommended_workers, 2)

    # Empaquetado de sesiones sugerido: cuántas sesiones caben cómodamente por worker y en L2
    l2_working_set_mb = (weights_bytes + (activations_bytes / (2.0 if training else 1.0))) / mib
    fits_in_l2 = l2_working_set_mb <= float(BLACKWELL_SPECS["l2_cache_mb"])
    per_worker_budget_mib = usable_vram_mib / recommended_workers
    headroom_for_packs = max(0.0, per_worker_budget_mib - (total_worker_mib - activations_mib))
    recommended_sessions_per_pack = max(
        1, min(8, int(headroom_for_packs // max(activations_mib, 1.0)))
    )

    # Auditoría estática AST de trampas host-sync
    paths_to_scan = (
        list(module_paths) if module_paths is not None else _discover_model_source_files(model)
    )
    sync_audit = audit_zero_host_syncs(paths_to_scan)

    return dict(
        hardware=dict(
            arch=BLACKWELL_SPECS["arch"],
            die=BLACKWELL_SPECS["die"],
            sms=SM120_SMS,
            vram_gb=vram_gb_total,
            l2_cache_mb=int(BLACKWELL_SPECS["l2_cache_mb"]),
            bandwidth_gbs=SM120_BANDWIDTH_GBS,
            max_dynamic_smem_kb_per_cta=int(BLACKWELL_SPECS["max_dynamic_smem_kb_per_cta"]),
        ),
        precision_mode=precision_mode,
        n_assets=int(n_assets),
        total_params=int(n_params),
        trainable_params=int(n_trainable),
        forward_flops=int(fwd_flops),
        forward_bytes=int(fwd_bytes),
        arithmetic_intensity_flop_per_byte=round(ai, 4),
        ridge_point_flop_per_byte=ridge_point,
        bottleneck=bottleneck,
        achievable_tflops=round(achievable_tflops, 3),
        peak_tflops=peak_tflops,
        cta_wave_alignment=dict(
            sms=SM120_SMS,
            block_n=int(block_n),
            row_ctas=int(n_assets),
            tile_ctas=int(tile_ctas),
            row_waves=int(row_waves),
            row_wave_efficiency=round(row_wave_eff, 4),
            recommended_padded_n_188=int(padded_188),
            recommended_padded_n_64=int(padded_64),
            l2_working_set_mb=round(l2_working_set_mb, 3),
            fits_in_l2_128mb=bool(fits_in_l2),
        ),
        vram_estimate_mib=dict(
            weights_mib=round(weights_mib, 3),
            gradients_mib=round(grads_mib, 3),
            optimizer_mib=round(optimizer_mib, 3),
            activations_mib=round(activations_mib, 3),
            cuda_context_mib=cuda_context_mib,
            total_per_worker_mib=round(total_worker_mib, 3),
            total_per_worker_gb=round(total_worker_gb, 4),
            usable_gpu_vram_mib=round(usable_vram_mib, 2),
        ),
        mps_recommendation=dict(
            recommended_workers=int(recommended_workers),
            max_workers_by_vram=int(max_workers_by_vram),
            fits_single_gpu=bool(fits_single_gpu),
            vram_fraction_per_worker=vram_frac,
            vram_cap_gb_per_worker=vram_cap_gb,
            estimated_total_vram_gb=est_total_vram_gb,
            recommended_sessions_per_pack=int(recommended_sessions_per_pack),
            recommended_bucket_multiple=SM120_SMS if n_assets >= SM120_SMS else 64,
        ),
        recurrence_audit=est["recurrence_audit"],
        sdpa_audit=est["sdpa_audit"],
        host_sync_audit=sync_audit,
    )


# =============================================================================
# 4. Generador de script Bash CUDA MPS para RTX PRO 6000 (`sm_120`)
# =============================================================================
def generate_rtx_pro_6000_mps_script(
    variations: Iterable[str],
    script_entry: str,
    out_dir: str | Path,
    workers: int = 6,
    precision_mode: str = "fp32_strict",
    *,
    market: str = "US",
    family: str = "mmarrs_titan",
    epochs: int = 10,
    seed: int = 42,
    vram_gb: float = 96.0,
    python_bin: str = "python3",
    extra_args: Sequence[str] = (),
) -> str:
    """Genera un script Bash ejecutable de saturación CUDA MPS para RTX PRO 6000 (`sm_120`).

    Alineado con los 8 flags CLI canónicos (`CANONICAL_TRAIN_CLI_FLAGS`) de `full_history_train.py`:
      `--variation --market --family --out-dir --precision --vram-fraction --epochs --seed`.
    Configura `TORCH_CUDA_ARCH_LIST=12.0`, `CMAKE_CUDA_ARCHITECTURES=120`, asignador PyTorch sin
    fragmentación (`expandable_segments:True,max_split_size_mb:256`), demonio
    `nvidia-cuda-mps-control -d`, límites de VRAM por worker y planificación por olas.
    """
    var_list = [str(v).strip() for v in variations if str(v).strip()]
    if not var_list:
        raise ValueError("variations debe contener al menos una variación")
    if workers < 1:
        raise ValueError(f"workers debe ser >= 1; recibido {workers}")
    if precision_mode not in PRECISION_MODES:
        raise ValueError(
            f"precision_mode inválido {precision_mode!r}; válidos: {PRECISION_MODES}"
        )
    if market not in VALID_MARKETS:
        raise ValueError(
            f"market inválido {market!r}; válidos (FM-8 estrictamente separados): {VALID_MARKETS}"
        )
    family_str = str(family).strip()
    if not family_str:
        raise ValueError("family no puede estar vacío")
    if epochs < 1:
        raise ValueError(f"epochs debe ser >= 1; recibido {epochs}")
    if seed < 0:
        raise ValueError(f"seed debe ser >= 0; recibido {seed}")
    entry = str(script_entry).strip()
    if not entry:
        raise ValueError("script_entry no puede estar vacío")

    plan = blackwell_mps_launch_plan(
        var_list,
        vram_gb=vram_gb,
        workers=workers,
        precision_mode=precision_mode,
    )
    waves: list[list[str]] = plan["waves"]  # type: ignore[assignment]
    w_per_wave = int(plan["workers_per_wave"])
    vram_cap_gb = float(plan["vram_gb_per_worker"])
    vram_cap_mb = int(vram_cap_gb * 1024)
    vram_frac = float(plan["vram_fraction_per_worker"])
    sm_thread_pct = max(10, min(100, int(100 // w_per_wave)))

    is_module_entry = ("/" not in entry) and (not entry.endswith(".py")) and (" " not in entry)
    quoted_out = shlex.quote(str(out_dir))
    quoted_py = shlex.quote(python_bin)
    quoted_entry = shlex.quote(entry)
    quoted_market = shlex.quote(market)
    quoted_family = shlex.quote(family_str)
    extra_str = (
        " " + " ".join(shlex.quote(str(a)) for a in extra_args) if extra_args else ""
    )

    lines: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# =============================================================================",
        "# Lanzador CUDA MPS — NVIDIA RTX PRO 6000 Blackwell (sm_120 / GB202)",
        f"# Hardware: {SM120_SMS} SMs | 100 KB/CTA SMEM | 128 MB L2 | {vram_gb:.0f} GB GDDR7 ECC",
        f"# Mercado: {market} | Familia: {family_str} | Precisión: {precision_mode}",
        f"# Variaciones: {len(var_list)} | Olas: {len(waves)} x {w_per_wave} workers | "
        f"Tope VRAM/worker: {vram_cap_gb:.2f} GiB",
        "# =============================================================================",
        "",
        f"OUT_DIR={quoted_out}",
        f"PYTHON_BIN=\"${{PYTHON_BIN:-{quoted_py}}}\"",
        'export TORCH_CUDA_ARCH_LIST="12.0"',
        'export CMAKE_CUDA_ARCHITECTURES="120"',
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"',
        'export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"',
        'export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"',
        f'export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="{sm_thread_pct}"',
        f'export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0={vram_cap_mb}M"',
        f"export MMARRS_MARKET={quoted_market}",
        f"export MMARRS_FAMILY={quoted_family}",
        f"export MMARRS_PRECISION_MODE={shlex.quote(precision_mode)}",
        f'export MMARRS_VRAM_FRACTION="{vram_frac}"',
        f'export MMARRS_VRAM_CAP_GB="{vram_cap_gb}"',
        f'export MMARRS_EPOCHS="{int(epochs)}"',
        f'export MMARRS_SEED="{int(seed)}"',
        'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"',
        'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"',
        "",
        'mkdir -p "$OUT_DIR" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"',
        "",
        "cleanup_mps() {",
        '    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true',
        "}",
        "trap cleanup_mps EXIT",
        "",
        'echo "[RTX PRO 6000 sm_120] Iniciando demonio CUDA MPS..."',
        "nvidia-cuda-mps-control -d",
        "",
        "FAILURES=0",
        "",
    ]

    for wave_idx, wave_vars in enumerate(waves, start=1):
        lines.append(
            f'echo "[Wave {wave_idx}/{len(waves)}] Lanzando {len(wave_vars)} '
            f'variaciones en paralelo (VRAM cap {vram_cap_gb:.2f} GiB/worker)..."'
        )
        lines.append("PIDS=()")
        for var in wave_vars:
            q_var = shlex.quote(var)
            var_out = f'"$OUT_DIR"/{q_var}'
            log_file = f'"$OUT_DIR"/{q_var}.log'
            runner_prefix = f'"$PYTHON_BIN" -m {quoted_entry}' if is_module_entry else (
                f'"$PYTHON_BIN" {quoted_entry}'
            )
            cmd = (
                f"{runner_prefix} "
                f"--variation {q_var} "
                f"--market {quoted_market} "
                f"--family {quoted_family} "
                f"--out-dir {var_out} "
                f"--precision {shlex.quote(precision_mode)} "
                f"--vram-fraction {vram_frac} "
                f"--epochs {int(epochs)} "
                f"--seed {int(seed)}{extra_str}"
            )
            lines.append(f"mkdir -p {var_out}")
            lines.append(f"{cmd} > {log_file} 2>&1 &")
            lines.append('PIDS+=("$!")')
        lines.append('for pid in "${PIDS[@]}"; do')
        lines.append('    if ! wait "$pid"; then')
        lines.append("        FAILURES=$((FAILURES + 1))")
        lines.append("    fi")
        lines.append("done")
        lines.append(
            f'echo "[Wave {wave_idx}/{len(waves)}] Completada (fallos acumulados: $FAILURES)"'
        )
        lines.append("")

    lines.extend(
        [
            'echo "[RTX PRO 6000 sm_120] Campaña MPS finalizada con $FAILURES fallos."',
            'exit "$FAILURES"',
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BLACKWELL_SPECS",
    "CANONICAL_TRAIN_CLI_FLAGS",
    "SM120_BANDWIDTH_GBS",
    "SM120_KERNEL_CONFIGS",
    "SM120_L2_CACHE_BYTES",
    "SM120_MAX_SMEM_BYTES",
    "SM120_SMS",
    "SM120_VRAM_BYTES",
    "VALID_MARKETS",
    "CUDAGraphSessionRunner",
    "FusedCrossSectionalZScoreClipFunction",
    "FusedNeuromodUpdateFunction",
    "FusedPinballCRPSLossFunction",
    "FusedSoftMoECombineFunction",
    "audit_zero_host_syncs",
    "blackwell_roofline_audit",
    "fused_cross_sectional_zscore_clip",
    "fused_crps_pinball_loss",
    "fused_neuromod_update",
    "fused_pinball_crps_loss",
    "fused_soft_moe_combine",
    "generate_rtx_pro_6000_mps_script",
]
