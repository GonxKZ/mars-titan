"""Perfiles de lanzamiento Triton conscientes de la arquitectura GPU (Blackwell sm_121 / sm_120).

Centraliza los parámetros de bloque, warps, etapas de pipeline, swizzling L2 y política de
evicción para las dos GPU objetivo de MARS-TITAN en FP32 IEEE estricto (y paridad FP64):

- **DGX Spark (NVIDIA GB10, ``sm_121`` / GB20B)**: 48 SMs, 24 MB L2, 128 KB L1/SMEM por SM
  (~100 KB SMEM dinámica por CTA), memoria unificada LPDDR5X a 273 GB/s.
- **RTX PRO 6000 Blackwell (``sm_120`` / GB202, Google Colab G4 / Server / Workstation)**:
  188 SMs, 128 MB L2 monolítica, 128 KB L1/SMEM por SM (~100 KB SMEM dinámica por CTA),
  96 GB GDDR7 a 1.597–1.792 GB/s y 120 TFLOPS FP32 estricto.

Todos los perfiles producen resultados deterministas en FP32 IEEE estricto (sin TF32 ni
precisiones reducidas) y respetan estrictamente el techo de ~100 KB de memoria compartida
por CTA de ``sm_12x`` (evitando el fallo ``OutOfResources`` de configuraciones de H100/B200).
La variable ``MARS_TITAN_GPU_PROFILE`` permite forzar ``sm_120``, ``sm_121`` o ``generic``
para verificación y contraste controlado.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

PROFILE_ENV_VAR = "MARS_TITAN_GPU_PROFILE"
# Techo de memoria compartida configurable por CTA en Blackwell sm_120 y sm_121 (~100 KB).
SM12X_MAX_SHARED_BYTES = 101376


@dataclass(frozen=True)
class GPUKernelProfile:
    """Parámetros de ejecución Triton ajustados a la microarquitectura de la GPU."""

    name: str
    arch: str
    sm_count: int
    l2_bytes: int
    max_smem_per_cta: int
    # Kernels de memoria asociativa (fused_memory_write.py)
    mem_fwd_warps: int
    mem_bwd_warps: int
    mem_read_warps: int
    mem_evict_stream: bool
    # Atención MAC último token (reducción parcial persistente)
    mac_bwd_ctas: int
    # CompactPriceTransformer factorizado + fusionado (proj+pos+LN1+QKV+SDPA)
    fac_fwd_block_t: int
    fac_fwd_warps: int
    fac_bwd_block_t: int
    fac_bwd_warps: int
    fac_bwd_ctas: int
    # CrossAssetAttention FlashC5 (swizzling L2 y etapas seguras para <=100 KB SMEM)
    c5_group_size_m: int
    c5_fp32_hd32_stages: int
    c5_fp32_hd64_stages: int
    # Escalado de entradas (LearningScaler)
    price_returns_warps: int
    scale_block_b: int
    scale_block_f: int
    scale_warps: int

    def c5_tiles(self, dtype: torch.dtype, head_dim: int) -> tuple[int, int, int, int]:
        """Devuelve ``(BLOCK_M, BLOCK_N, num_warps, num_stages)`` sin spills y <=100 KB SMEM."""
        if dtype == torch.float64:
            if head_dim <= 32:
                return 32, 32, 4, 2
            return 32, 16, 4, 2
        if head_dim <= 32:
            return 128, 32, 4, self.c5_fp32_hd32_stages
        if head_dim <= 64:
            return 64, 32, 4, self.c5_fp32_hd64_stages
        return 32, 32, 4, 2

    def wave_aligned_ctas(self, rows: int, base_ctas: int) -> tuple[int, int]:
        """Calcula ``(rows_per_cta, num_ctas)`` alineado a ondas completas de SMs."""
        target = max(1, min(rows, base_ctas))
        rows_per_cta = max(1, (rows + target - 1) // target)
        num_ctas = (rows + rows_per_cta - 1) // rows_per_cta
        return rows_per_cta, num_ctas


def _profile_dgx_gb10(sm_count: int = 48, l2_bytes: int = 24 * 1024 * 1024) -> GPUKernelProfile:
    """Perfil ajustado para DGX Spark (NVIDIA GB10, sm_121, 48 SMs, 24 MB L2, 273 GB/s)."""
    return GPUKernelProfile(
        name="dgx_gb10_sm121",
        arch="sm_121",
        sm_count=sm_count,
        l2_bytes=l2_bytes,
        max_smem_per_cta=SM12X_MAX_SHARED_BYTES,
        mem_fwd_warps=4,
        mem_bwd_warps=8,
        mem_read_warps=8,
        mem_evict_stream=True,
        mac_bwd_ctas=max(96, sm_count * 2),
        fac_fwd_block_t=64,
        fac_fwd_warps=8,
        fac_bwd_block_t=32,
        fac_bwd_warps=4,
        fac_bwd_ctas=max(192, sm_count * 4),
        c5_group_size_m=4,
        c5_fp32_hd32_stages=2,
        c5_fp32_hd64_stages=2,
        price_returns_warps=2,
        scale_block_b=32,
        scale_block_f=64,
        scale_warps=4,
    )


def _profile_rtx_pro_6000(
    sm_count: int = 188, l2_bytes: int = 128 * 1024 * 1024
) -> GPUKernelProfile:
    """Perfil ajustado para RTX PRO 6000 Blackwell (sm_120 / GB202, Colab G4, 188 SMs, 128 MB L2)."""
    return GPUKernelProfile(
        name="rtx_pro_6000_sm120",
        arch="sm_120",
        sm_count=sm_count,
        l2_bytes=l2_bytes,
        max_smem_per_cta=SM12X_MAX_SHARED_BYTES,
        # 8 warps en _forward_kernel reduce registros (168 -> 122, 0 spills) para 2 CTAs/SM
        # × 188 SMs = 376 flujos en vuelo saturando el bus GDDR7 de 1.597-1.792 GB/s.
        mem_fwd_warps=8,
        mem_bwd_warps=8,
        mem_read_warps=8,
        # `evict_first` en los flujos S1/S2 mantiene W1/W2 calientes en los 128 MB de L2.
        mem_evict_stream=True,
        # Grids alineados a múltiplos exactos de 188 SMs (376 y 752 CTAs, sin cuantización de onda).
        mac_bwd_ctas=max(376, sm_count * 2),
        fac_fwd_block_t=64,
        fac_fwd_warps=8,
        fac_bwd_block_t=32,
        fac_bwd_warps=4,
        fac_bwd_ctas=max(752, sm_count * 4),
        # Swizzling GROUP_SIZE_M=8 aprovecha los 128 MB de L2; stages=3 en HD<=32 cabe en <100 KB SMEM.
        c5_group_size_m=8,
        c5_fp32_hd32_stages=3,
        c5_fp32_hd64_stages=2,
        price_returns_warps=2,
        scale_block_b=32,
        scale_block_f=64,
        scale_warps=4,
    )


def _profile_generic(major: int, minor: int, sm_count: int, l2_bytes: int) -> GPUKernelProfile:
    """Perfil conservador determinista para otras GPU CUDA (Ampere/Ada/Hopper)."""
    return GPUKernelProfile(
        name=f"cuda_sm_{major}{minor}",
        arch=f"sm_{major}{minor}",
        sm_count=sm_count,
        l2_bytes=l2_bytes,
        max_smem_per_cta=98304,
        mem_fwd_warps=4,
        mem_bwd_warps=8,
        mem_read_warps=8,
        mem_evict_stream=False,
        mac_bwd_ctas=max(96, sm_count * 2),
        fac_fwd_block_t=32,
        fac_fwd_warps=4,
        fac_bwd_block_t=16,
        fac_bwd_warps=4,
        fac_bwd_ctas=max(128, sm_count * 2),
        c5_group_size_m=4,
        c5_fp32_hd32_stages=2,
        c5_fp32_hd64_stages=2,
        price_returns_warps=2,
        scale_block_b=32,
        scale_block_f=64,
        scale_warps=4,
    )


@functools.lru_cache(maxsize=16)
def _resolve_profile(device_index: int, override: str) -> GPUKernelProfile:
    if override in ("sm_120", "rtx_pro_6000", "rtx_pro_6000_sm120", "colab_g4"):
        return _profile_rtx_pro_6000()
    if override in ("sm_121", "dgx_gb10", "dgx_gb10_sm121", "gb10"):
        return _profile_dgx_gb10()
    if override == "generic":
        return _profile_generic(8, 9, 64, 48 * 1024 * 1024)
    if override and override != "auto":
        raise ValueError(
            f"{PROFILE_ENV_VAR} desconocido: {override!r} (use auto, sm_120, sm_121 o generic)"
        )
    if not torch.cuda.is_available():
        return _profile_dgx_gb10()
    props = torch.cuda.get_device_properties(device_index)
    major, minor = props.major, props.minor
    sm_count = int(props.multi_processor_count)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or (24 * 1024 * 1024))
    if (major, minor) == (12, 0):
        return _profile_rtx_pro_6000(sm_count=sm_count, l2_bytes=l2_bytes)
    if (major, minor) == (12, 1):
        return _profile_dgx_gb10(sm_count=sm_count, l2_bytes=l2_bytes)
    return _profile_generic(major, minor, sm_count, l2_bytes)


def active_gpu_profile(device: torch.device | int | None = None) -> GPUKernelProfile:
    """Devuelve el perfil de kernels Triton para el dispositivo CUDA indicado."""
    if device is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    elif isinstance(device, int):
        index = device
    else:
        index = 0 if device.index is None else int(device.index)
    override = os.environ.get(PROFILE_ENV_VAR, "auto").strip().lower()
    return _resolve_profile(index, override)
