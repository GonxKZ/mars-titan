#!/usr/bin/env python3
"""Plan de pruebas (Nivel 0 + Nivel 1) y barrido `sm_120` para `fused_memory_write.py`.

Cubre el kernel vigente en `material/repo-v11/src/mars_titan/models/titans/fused_memory_write.py`
sobre NVIDIA RTX PRO 6000 Blackwell (`sm_120`, GB202, 188 SMs, 48 warps/SM, 128 KiB SMEM/SM,
99 KiB SMEM máx./CTA, 128 MB L2, 1.597 GB/s, 120 TFLOPS FP32):
- Nivel 0: huella SHA-256 exacta y `torch.equal` bit a bit (determinismo, referencia FP32,
  invariancia al tamaño de lote y disposiciones no contiguas).
- Nivel 1: paridad frente a oráculo matemático independiente en FP64 (distancias ULP FP32,
  max_abs, max_rel, rel_l2 y prueba TOST de equivalencia estadística sin sesgo).
- Generador de rejilla `sm_120`: valida potencias de 2, techo estricto de 99 KiB/CTA,
  granularidad de 256 registros/warp, descarta configuraciones inválidas/redundantes y prohíbe
  `@triton.autotune` en producción para preservar reproducibilidad bit a bit.
- Auditoría PTX offline real con Triton 3.8.0 sobre `fused_memory_write.py` en `sm_120`.
- Ejecutor GPU listo para correr en Colab (`run_sm120_gpu_sweep`, `[no verificado en GPU]`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from mars_titan.models.titans.associative_gradient import associative_gradients


@dataclass(frozen=True)
class Sm120Spec:
    """Especificación verificada de RTX PRO 6000 Blackwell Server Edition (`sm_120`, GB202)."""

    arch: str = "sm_120"
    sm_count: int = 188
    cuda_cores_per_sm: int = 128
    max_warps_per_sm: int = 48
    max_blocks_per_sm: int = 32
    warp_size: int = 32
    regs_per_sm: int = 65536
    max_regs_per_thread: int = 255
    reg_alloc_granularity_per_warp: int = 256
    max_smem_per_sm_bytes: int = 128 * 1024
    max_smem_per_block_bytes: int = 99 * 1024
    l2_cache_bytes: int = 128 * 1024 * 1024
    vram_bytes: int = 96 * 1024**3
    bandwidth_gbps: float = 1597.0
    fp32_tflops: float = 120.0

    @property
    def fp32_ridge_point(self) -> float:
        return (self.fp32_tflops * 1e12) / (self.bandwidth_gbps * 1e9)


SM120_SPEC = Sm120Spec()
KERNEL_DIM = 64
LEVEL0_BITWISE_ATOL = 0.0
LEVEL1_FP64_RTOL = 1e-12
LEVEL1_FP64_ATOL = 1e-13
LEVEL1_FP32_FWD_RTOL = 1e-5
LEVEL1_FP32_FWD_ATOL = 1e-6
LEVEL1_FP32_FWD_MAX_ULP = 256
LEVEL1_FP32_BWD_RTOL = 5e-5
LEVEL1_FP32_BWD_ATOL = 1e-5
LEVEL1_FP32_BWD_MAX_ULP = 1024
UNVERIFIED_GPU_TAG = "[no verificado en GPU]"


def verify_strict_fp32_deterministic(*, check_triton: bool = True) -> dict[str, Any]:
    """Comprueba sin modificar estado global que FP32 IEEE estricto y determinismo están activos."""
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("autocast está activo: prohibido bajo el contrato FP32 IEEE estricto")
    triton_fp32_default = None
    if check_triton:
        import triton

        triton_fp32_default = triton.knobs.language.fp32_default
        if triton_fp32_default != "ieee":
            raise RuntimeError(
                f"triton.knobs.language.fp32_default debe ser 'ieee' (es {triton_fp32_default!r})"
            )

    state = {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "triton_fp32_default": triton_fp32_default,
    }
    if (
        not state["deterministic_algorithms"]
        or state["matmul_allow_tf32"]
        or state["cudnn_allow_tf32"]
        or state["matmul_precision"] != "highest"
        or not state["cudnn_deterministic"]
        or state["cudnn_benchmark"]
    ):
        raise RuntimeError(f"Contrato FP32 estricto/determinista violado: {state}")
    return state


def enforce_strict_fp32_deterministic(*, check_triton: bool = True) -> dict[str, Any]:
    """Impone y verifica FP32 IEEE estricto + determinismo en PyTorch y Triton."""
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("autocast está activo: prohibido bajo el contrato FP32 IEEE estricto")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if check_triton:
        import triton

        triton.knobs.language.fp32_default = "ieee"
    return verify_strict_fp32_deterministic(check_triton=check_triton)


def tensor_sha256(tensor: Tensor) -> str:
    """Huella SHA-256 exacta de forma, dtype y bytes IEEE del tensor."""
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    header = f"{tuple(t.shape)}|{t.dtype}|".encode("ascii")
    h.update(header)
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def ulp_distance_fp32(a: Tensor, b: Tensor) -> Tensor:
    """Distancia exacta en ULPs entre dos tensores IEEE-754 float32 finitos."""
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("ulp_distance_fp32 exige tensores torch.float32")
    if a.shape != b.shape:
        raise ValueError(f"Formas incompatibles: {tuple(a.shape)} != {tuple(b.shape)}")
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("ulp_distance_fp32 exige valores finitos")

    ia = a.detach().contiguous().view(torch.int32).to(torch.int64)
    ib = b.detach().contiguous().view(torch.int32).to(torch.int64)
    sign_mask = (1 << 31) - 1
    # Proyección monótona de signo-magnitud: +0.0 y -0.0 se mapean exactamente a 0.
    oa = torch.where(ia < 0, -(ia & sign_mask), ia)
    ob = torch.where(ib < 0, -(ib & sign_mask), ib)
    return (oa - ob).abs()


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def tost_zero_bias(diff: Tensor, margin: float, alpha: float = 0.05) -> dict[str, Any]:
    """Prueba TOST de equivalencia estadística para sesgo medio dentro de `±margin`."""
    if margin <= 0.0 or not math.isfinite(margin):
        raise ValueError("margin debe ser finito positivo")
    vals = diff.detach().to(torch.float64).reshape(-1)
    if vals.numel() < 2:
        raise ValueError("TOST requiere al menos 2 elementos")
    if not torch.isfinite(vals).all():
        return {
            "equivalent": False,
            "mean": float("nan"),
            "se": float("nan"),
            "p_tost": 1.0,
            "margin": margin,
        }
    mean = float(vals.mean().item())
    std = float(vals.std(unbiased=True).item())
    se = std / math.sqrt(vals.numel())
    if se == 0.0:
        p_tost = 0.0 if abs(mean) < margin else 1.0
    else:
        p_low = 1.0 - _normal_cdf((mean + margin) / se)
        p_high = _normal_cdf((mean - margin) / se)
        p_tost = max(p_low, p_high)
    half_90 = 1.6448536269514722 * se
    return {
        "equivalent": bool(p_tost <= alpha),
        "mean": mean,
        "se": se,
        "interval_90": [mean - half_90, mean + half_90],
        "p_tost": p_tost,
        "margin": margin,
    }


def make_write_inputs(
    batch: int,
    *,
    dim: int = KERNEL_DIM,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Genera entradas y cotangentes deterministas con semilla fija."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale = dim**-0.5

    def _draw(*shape: int, mult: float) -> Tensor:
        return (torch.randn(shape, generator=gen, dtype=dtype) * mult).to(device)

    w1 = _draw(batch, dim, dim, mult=scale)
    w2 = _draw(batch, dim, dim, mult=scale)
    s1 = _draw(batch, dim, dim, mult=0.2 * scale)
    s2 = _draw(batch, dim, dim, mult=0.2 * scale)
    keys = F.normalize(_draw(batch, dim, mult=1.0), dim=-1, eps=1e-12)
    values = _draw(batch, dim, mult=0.5)
    alpha = torch.sigmoid(_draw(batch, dim, 1, mult=1.0) - 3.0)
    eta = torch.sigmoid(_draw(batch, 1, 1, mult=1.0))
    theta = 0.1 * torch.sigmoid(_draw(batch, 1, 1, mult=1.0))

    upstream = (
        _draw(batch, dim, dim, mult=scale),
        _draw(batch, dim, dim, mult=scale),
        _draw(batch, dim, dim, mult=scale),
        _draw(batch, dim, dim, mult=scale),
        _draw(batch, dim, mult=0.25),
    )
    return (w1, w2, s1, s2, keys, values, alpha, eta, theta), upstream


def reference_write_fp64(
    weights: tuple[Tensor, Tensor],
    momentum: tuple[Tensor, Tensor],
    keys: Tensor,
    values: Tensor,
    gates: tuple[Tensor, Tensor, Tensor],
    *,
    residual_layer_norm: bool,
    eps: float = 1e-5,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
    """Oráculo matemático independiente en FP64 puro (sin compartir código con el kernel)."""
    w1, w2 = weights
    s1, s2 = momentum
    alpha, eta, theta = gates
    outer = torch.is_grad_enabled() and any(
        t.requires_grad for t in (w1, w2, s1, s2, keys, values, alpha, eta, theta)
    )
    with torch.enable_grad():
        lw1 = (w1.clone() if outer else w1.detach().clone()).requires_grad_(True)
        lw2 = (w2.clone() if outer else w2.detach().clone()).requires_grad_(True)
        lk = keys if outer else keys.detach()
        lv = values if outer else values.detach()
        h = torch.bmm(lk.unsqueeze(1), lw1.transpose(1, 2))
        a = F.gelu(h, approximate="none")
        z = torch.bmm(a, lw2.transpose(1, 2))
        if residual_layer_norm:
            z = lk.unsqueeze(1) + F.layer_norm(z, (lk.shape[-1],), eps=eps)
        residual = z.squeeze(1) - lv
        elements = residual.square()
        g1, g2 = torch.autograd.grad(elements.sum(), (lw1, lw2), create_graph=outer)
    s1_next = eta * s1 - theta * g1
    s2_next = eta * s2 - theta * g2
    keep = 1.0 - alpha
    w1_next = keep * w1 + s1_next
    w2_next = keep * w2 + s2_next
    if not outer:
        return (
            (w1_next.detach(), w2_next.detach()),
            (s1_next.detach(), s2_next.detach()),
            elements.detach(),
        )
    return (w1_next, w2_next), (s1_next, s2_next), elements


class _NormalizeBackwardUpstream(torch.autograd.Function):
    """Normaliza cotangentes no contiguas en backward, como `_FusedWrite.backward`."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.contiguous()


def reference_write_fp32_deterministic(
    weights: tuple[Tensor, Tensor],
    momentum: tuple[Tensor, Tensor],
    keys: Tensor,
    values: Tensor,
    gates: tuple[Tensor, Tensor, Tensor],
    *,
    residual_layer_norm: bool,
    eps: float = 1e-5,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
    """Referencia FP32 determinista por flujo (un CTA por flujo, como `fused_memory_write.py`).

    Evalúa cada flujo con `associative_gradients` de `repo-v11` sobre cortes contiguos [1, ...],
    evitando diferencias de vectorización por tamaño de lote de OpenBLAS/OneDNN en CPU y
    reproduciendo la invariancia exacta al tamaño de lote `B` del kernel Triton sin atómicas.
    """
    w1, w2 = weights[0].contiguous(), weights[1].contiguous()
    s1, s2 = momentum[0].contiguous(), momentum[1].contiguous()
    k, v = keys.contiguous(), values.contiguous()
    alpha, eta, theta = gates[0].contiguous(), gates[1].contiguous(), gates[2].contiguous()
    batch = k.shape[0]
    w1_outs, w2_outs, s1_outs, s2_outs, elem_outs = [], [], [], [], []
    for idx in range(batch):
        sl = slice(idx, idx + 1)
        grads, elem = associative_gradients(
            (w1[sl], w2[sl]),
            k[sl],
            v[sl],
            residual_layer_norm=residual_layer_norm,
            eps=eps,
        )
        ns1 = eta[sl] * s1[sl] - theta[sl] * grads[0]
        ns2 = eta[sl] * s2[sl] - theta[sl] * grads[1]
        nw1 = (1.0 - alpha[sl]) * w1[sl] + ns1
        nw2 = (1.0 - alpha[sl]) * w2[sl] + ns2
        w1_outs.append(nw1)
        w2_outs.append(nw2)
        s1_outs.append(ns1)
        s2_outs.append(ns2)
        elem_outs.append(elem)
    return (
        (
            _NormalizeBackwardUpstream.apply(torch.cat(w1_outs, dim=0)),
            _NormalizeBackwardUpstream.apply(torch.cat(w2_outs, dim=0)),
        ),
        (
            _NormalizeBackwardUpstream.apply(torch.cat(s1_outs, dim=0)),
            _NormalizeBackwardUpstream.apply(torch.cat(s2_outs, dim=0)),
        ),
        _NormalizeBackwardUpstream.apply(torch.cat(elem_outs, dim=0)),
    )


def _run_with_grads(
    fn, inputs: tuple[Tensor, ...], upstream: tuple[Tensor, ...], *, layer_norm: bool, eps: float
):
    leaves = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    written, following, elements = fn(
        leaves[:2],
        leaves[2:4],
        leaves[4],
        leaves[5],
        leaves[6:9],
        residual_layer_norm=layer_norm,
        eps=eps,
    )
    outs = (*written, *following, elements)
    grads = torch.autograd.grad(outs, leaves, upstream)
    return tuple(t.detach() for t in outs), tuple(t.detach() for t in grads)


def _make_noncontiguous(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Construye vistas no contiguas diferenciables con centinelas NaN fuera del stride."""
    out: list[Tensor] = []
    for t in tensors:
        if t.ndim == 3 and t.shape[1] > 1 and t.shape[2] > 1:
            out.append(t.transpose(1, 2).contiguous().transpose(1, 2))
        elif t.ndim == 2 and t.shape[1] > 1:
            nan_pad = torch.full_like(t, float("nan"))
            stacked = torch.stack((t, nan_pad), dim=-1).reshape(t.shape[0], t.shape[1] * 2)
            out.append(stacked[:, 0::2])
        elif t.ndim == 3 and t.shape[1] > 1:
            nan_pad = torch.full_like(t, float("nan"))
            stacked = torch.stack((t, nan_pad), dim=2).reshape(
                t.shape[0], t.shape[1] * 2, t.shape[2]
            )
            out.append(stacked[:, 0::2, :])
        else:
            nan_pad = torch.full_like(t, float("nan"))
            stacked = torch.stack((t, nan_pad), dim=1).reshape(t.shape[0] * 2, *t.shape[1:])
            out.append(stacked[0::2])
    return tuple(out)


def _run_noncontiguous_with_grads(
    fn, inputs: tuple[Tensor, ...], upstream: tuple[Tensor, ...], *, layer_norm: bool, eps: float
):
    leaves = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    nc_in = _make_noncontiguous(leaves)
    nc_up = _make_noncontiguous(tuple(u.detach().clone() for u in upstream))
    written, following, elements = fn(
        nc_in[:2],
        nc_in[2:4],
        nc_in[4],
        nc_in[5],
        nc_in[6:9],
        residual_layer_norm=layer_norm,
        eps=eps,
    )
    outs = (*written, *following, elements)
    grads = torch.autograd.grad(outs, leaves, nc_up)
    return tuple(t.detach() for t in outs), tuple(t.detach() for t in grads)


def evaluate_level0_bitwise(
    fn_candidate,
    fn_reference,
    inputs: tuple[Tensor, ...],
    upstream: tuple[Tensor, ...],
    *,
    layer_norm: bool = True,
    eps: float = 1e-5,
    repeats: int = 3,
) -> dict[str, Any]:
    """Nivel 0: identidad bit a bit (SHA-256 + `torch.equal`), invariancia a lote y no contiguos."""
    if repeats < 2:
        raise ValueError("repeats debe ser >= 2 para comprobar determinismo")
    base_out, base_grad = _run_with_grads(
        fn_candidate, inputs, upstream, layer_norm=layer_norm, eps=eps
    )
    base_hashes_out = [tensor_sha256(t) for t in base_out]
    base_hashes_grad = [tensor_sha256(t) for t in base_grad]

    deterministic = True
    for _ in range(repeats - 1):
        cur_out, cur_grad = _run_with_grads(
            fn_candidate, inputs, upstream, layer_norm=layer_norm, eps=eps
        )
        if not (
            all(torch.equal(a, b) for a, b in zip(base_out, cur_out, strict=True))
            and all(torch.equal(a, b) for a, b in zip(base_grad, cur_grad, strict=True))
            and [tensor_sha256(t) for t in cur_out] == base_hashes_out
            and [tensor_sha256(t) for t in cur_grad] == base_hashes_grad
        ):
            deterministic = False

    ref_out, ref_grad = _run_with_grads(
        fn_reference, inputs, upstream, layer_norm=layer_norm, eps=eps
    )
    ref_match = all(torch.equal(a, b) for a, b in zip(base_out, ref_out, strict=True)) and all(
        torch.equal(a, b) for a, b in zip(base_grad, ref_grad, strict=True)
    )

    # Invariancia al tamaño de lote (fila 0 con B=1 frente a fila 0 del lote completo)
    in1 = tuple(t[:1].contiguous() for t in inputs)
    up1 = tuple(t[:1].contiguous() for t in upstream)
    out1, grad1 = _run_with_grads(fn_candidate, in1, up1, layer_norm=layer_norm, eps=eps)
    batch_inv = all(torch.equal(a[0], b[0]) for a, b in zip(out1, base_out, strict=True)) and all(
        torch.equal(a[0], b[0]) for a, b in zip(grad1, base_grad, strict=True)
    )

    # Disposiciones no contiguas pasadas directamente al candidato (sin normalizar en el arnés)
    nc_out, nc_grad = _run_noncontiguous_with_grads(
        fn_candidate, inputs, upstream, layer_norm=layer_norm, eps=eps
    )
    noncontig_ok = all(torch.equal(a, b) for a, b in zip(nc_out, base_out, strict=True)) and all(
        torch.equal(a, b) for a, b in zip(nc_grad, base_grad, strict=True)
    )

    return {
        "passed": bool(deterministic and ref_match and batch_inv and noncontig_ok),
        "deterministic_bitwise": deterministic,
        "reference_match_bitwise": ref_match,
        "batch_invariant_bitwise": batch_inv,
        "noncontiguous_match_bitwise": noncontig_ok,
        "output_hashes": base_hashes_out,
        "grad_hashes": base_hashes_grad,
    }


def _tensor_parity_metrics(
    name: str,
    actual: Tensor,
    ref_fp64: Tensor,
    *,
    rtol: float,
    atol: float,
    max_ulp_limit: int | None,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if actual.shape != ref_fp64.shape:
        violations.append(f"{name}: forma {tuple(actual.shape)} != {tuple(ref_fp64.shape)}")
        return {"max_abs": float("inf"), "max_rel": float("inf"), "max_ulp": None}, violations
    if not torch.isfinite(actual).all():
        violations.append(f"{name}: contiene valores no finitos (NaN/Inf)")
        return {"max_abs": float("nan"), "max_rel": float("nan"), "max_ulp": None}, violations
    if not torch.isfinite(ref_fp64).all():
        violations.append(f"{name}: referencia FP64 no finita")
        return {"max_abs": float("nan"), "max_rel": float("nan"), "max_ulp": None}, violations

    got64 = actual.detach().cpu().to(torch.float64)
    ref64 = ref_fp64.detach().cpu().to(torch.float64)
    diff = (got64 - ref64).abs()
    max_abs = float(diff.max().item())
    ref_inf = float(ref64.abs().max().item())
    max_rel = max_abs / (ref_inf if ref_inf > 0.0 else 1.0)
    ref_l2 = float(torch.linalg.vector_norm(ref64).item())
    rel_l2 = float(torch.linalg.vector_norm(diff).item()) / (ref_l2 if ref_l2 > 0.0 else 1.0)

    limit = atol + rtol * ref64.abs()
    if bool((diff > limit).any().item()):
        violations.append(
            f"{name}: excede tolerancia (max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, "
            f"rtol={rtol}, atol={atol})"
        )

    max_ulp: int | None = None
    max_ulp_all: int | None = None
    if actual.dtype == torch.float32:
        ref32 = ref64.to(torch.float32)
        if torch.isfinite(ref32).all():
            ulps = ulp_distance_fp32(actual.detach().cpu(), ref32)
            max_ulp_all = int(ulps.max().item())
            # Evaluar la cota ULP elemento a elemento solo donde |error| > atol (evita falsos
            # positivos por cancelación cercana a cero en e² ~ 1e-10 con error absoluto ~ 1e-12).
            above_floor = diff > atol
            max_ulp = int(ulps[above_floor].max().item()) if bool(above_floor.any().item()) else 0
            if max_ulp_limit is not None and max_ulp > max_ulp_limit:
                violations.append(f"{name}: excede ULP máximo ({max_ulp} > {max_ulp_limit})")

    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rel_l2": rel_l2,
        "max_ulp": max_ulp,
        "max_ulp_all": max_ulp_all,
    }, violations


def evaluate_level1_fp64_parity(
    actual_outputs: tuple[Tensor, ...],
    actual_grads: tuple[Tensor, ...],
    ref_fp64_outputs: tuple[Tensor, ...],
    ref_fp64_grads: tuple[Tensor, ...],
    *,
    fwd_rtol: float = LEVEL1_FP32_FWD_RTOL,
    fwd_atol: float = LEVEL1_FP32_FWD_ATOL,
    fwd_max_ulp: int = LEVEL1_FP32_FWD_MAX_ULP,
    bwd_rtol: float = LEVEL1_FP32_BWD_RTOL,
    bwd_atol: float = LEVEL1_FP32_BWD_ATOL,
    bwd_max_ulp: int = LEVEL1_FP32_BWD_MAX_ULP,
    stat_margin: float = 1e-6,
) -> dict[str, Any]:
    """Nivel 1: paridad frente a FP64 (ULP/abs/rel) + equivalencia estadística TOST."""
    violations: list[str] = []
    fwd_metrics: dict[str, Any] = {}
    bwd_metrics: dict[str, Any] = {}
    residuals: list[Tensor] = []

    for idx, (got, ref) in enumerate(zip(actual_outputs, ref_fp64_outputs, strict=True)):
        name = f"out_{idx}"
        m, v = _tensor_parity_metrics(
            name, got, ref, rtol=fwd_rtol, atol=fwd_atol, max_ulp_limit=fwd_max_ulp
        )
        fwd_metrics[name] = m
        violations.extend(v)
        if got.shape == ref.shape and torch.isfinite(got).all() and torch.isfinite(ref).all():
            residuals.append(
                (
                    got.detach().cpu().to(torch.float64) - ref.detach().cpu().to(torch.float64)
                ).reshape(-1)
            )

    for idx, (got, ref) in enumerate(zip(actual_grads, ref_fp64_grads, strict=True)):
        name = f"grad_{idx}"
        m, v = _tensor_parity_metrics(
            name, got, ref, rtol=bwd_rtol, atol=bwd_atol, max_ulp_limit=bwd_max_ulp
        )
        bwd_metrics[name] = m
        violations.extend(v)
        if got.shape == ref.shape and torch.isfinite(got).all() and torch.isfinite(ref).all():
            residuals.append(
                (
                    got.detach().cpu().to(torch.float64) - ref.detach().cpu().to(torch.float64)
                ).reshape(-1)
            )

    if residuals:
        tost_res = tost_zero_bias(torch.cat(residuals), margin=stat_margin)
        if not tost_res["equivalent"]:
            violations.append(
                f"TOST sin equivalencia estadística (mean={tost_res['mean']:.3e}, "
                f"p={tost_res['p_tost']:.3e})"
            )
    else:
        tost_res = {
            "equivalent": False,
            "mean": float("nan"),
            "se": float("nan"),
            "p_tost": 1.0,
            "margin": stat_margin,
        }

    max_ulp_fwd = max(
        (m["max_ulp"] for m in fwd_metrics.values() if m["max_ulp"] is not None), default=0
    )
    max_ulp_bwd = max(
        (m["max_ulp"] for m in bwd_metrics.values() if m["max_ulp"] is not None), default=0
    )
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "forward": fwd_metrics,
        "backward": bwd_metrics,
        "max_ulp_fwd": max_ulp_fwd,
        "max_ulp_bwd": max_ulp_bwd,
        "statistical_equivalent": bool(tost_res["equivalent"]),
        "tost": tost_res,
    }


def run_cpu_verification_plan(
    *,
    batches: tuple[int, ...] = (1, 7, 13, 37),
    layer_norms: tuple[bool, ...] = (True, False),
    eps: float = 1e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Ejecuta en CPU el arnés completo Nivel 0 + Nivel 1 verificando referencias y métricas."""
    fp32_state = enforce_strict_fp32_deterministic(check_triton=True)
    cases: list[dict[str, Any]] = []
    for batch in batches:
        for ln in layer_norms:
            in64, up64 = make_write_inputs(batch, dtype=torch.float64, seed=seed + batch)
            ref_out64, ref_grad64 = _run_with_grads(
                reference_write_fp64, in64, up64, layer_norm=ln, eps=eps
            )
            in32 = tuple(t.to(torch.float32) for t in in64)
            up32 = tuple(t.to(torch.float32) for t in up64)

            lvl0 = evaluate_level0_bitwise(
                reference_write_fp32_deterministic,
                reference_write_fp32_deterministic,
                in32,
                up32,
                layer_norm=ln,
                eps=eps,
                repeats=3,
            )
            out32, grad32 = _run_with_grads(
                reference_write_fp32_deterministic, in32, up32, layer_norm=ln, eps=eps
            )
            lvl1 = evaluate_level1_fp64_parity(
                out32, grad32, ref_out64, ref_grad64, stat_margin=1e-6
            )
            cases.append({"batch": batch, "layer_norm": ln, "level0": lvl0, "level1": lvl1})

    lvl0_ok = all(c["level0"]["passed"] for c in cases)
    lvl1_ok = all(c["level1"]["passed"] for c in cases)
    return {
        "schema_version": 1,
        "kind": "sm120_kernel_cpu_verification_v1",
        "strict_fp32_verified": True,
        "fp32_state": fp32_state,
        "level0_all_passed": lvl0_ok,
        "level1_all_passed": lvl1_ok,
        "all_passed": lvl0_ok and lvl1_ok,
        "cases": cases,
    }


def _is_power_of_two(n: int) -> bool:
    return type(n) is int and n > 0 and (n & (n - 1)) == 0


def estimate_sm120_smem_bytes(
    *,
    regime: str,
    block_d: int | None = None,
    block_m: int | None = None,
    block_k: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    dtype_bytes: int = 4,
    buffered_tensors: int = 2,
) -> int:
    """Estima la memoria compartida por CTA en `sm_120` según régimen (registros vs tiled)."""
    if regime == "register_resident":
        # En el kernel actual sin bucle for, Triton usa únicamente scratchpad de reducción por warp.
        return int(num_warps) * 128
    if regime == "tiled_smem":
        if block_m is None or block_k is None:
            raise ValueError("tiled_smem requiere block_m y block_k")
        return (
            int(num_stages) * int(buffered_tensors) * int(block_m) * int(block_k) * int(dtype_bytes)
        )
    raise ValueError(f"Régimen desconocido: {regime!r}")


def sm120_occupancy(
    *,
    num_warps: int,
    regs_per_thread: int,
    smem_bytes: int,
    spec: Sm120Spec = SM120_SPEC,
) -> dict[str, Any]:
    """Modelo exacto de ocupación en `sm_120` (48 warps/SM, 256 regs/warp, 99 KiB/CTA)."""
    if type(num_warps) is not int or not 1 <= num_warps <= spec.max_warps_per_sm:
        raise ValueError(f"num_warps fuera de rango [1, {spec.max_warps_per_sm}]: {num_warps!r}")
    if type(regs_per_thread) is not int or not 1 <= regs_per_thread <= spec.max_regs_per_thread:
        raise ValueError(
            f"regs_per_thread fuera de rango [1, {spec.max_regs_per_thread}]: {regs_per_thread!r}"
        )
    if type(smem_bytes) is not int or smem_bytes < 0:
        raise ValueError(f"smem_bytes inválido: {smem_bytes!r}")
    if smem_bytes > spec.max_smem_per_block_bytes:
        raise ValueError(
            f"smem_bytes={smem_bytes} supera el límite de 99 KiB ({spec.max_smem_per_block_bytes} "
            "B) por CTA en sm_120"
        )

    gran = spec.reg_alloc_granularity_per_warp
    regs_per_warp = ((regs_per_thread * spec.warp_size + gran - 1) // gran) * gran
    regs_per_block = regs_per_warp * num_warps
    blocks_by_warps = spec.max_warps_per_sm // num_warps
    blocks_by_regs = (
        spec.regs_per_sm // regs_per_block if regs_per_block > 0 else spec.max_blocks_per_sm
    )
    blocks_by_smem = (
        spec.max_smem_per_sm_bytes // smem_bytes if smem_bytes > 0 else spec.max_blocks_per_sm
    )
    active_blocks = min(blocks_by_warps, blocks_by_regs, blocks_by_smem, spec.max_blocks_per_sm)
    active_warps = active_blocks * num_warps
    return {
        "num_warps": num_warps,
        "regs_per_thread": regs_per_thread,
        "regs_per_block": regs_per_block,
        "smem_bytes": smem_bytes,
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "occupancy": active_warps / float(spec.max_warps_per_sm),
        "gpu_active_ctas": active_blocks * spec.sm_count,
    }


def generate_sm120_sweep_grid(
    *,
    dim: int = 64,
    regime: str = "register_resident",
    block_ms: tuple[int, ...] = (32, 64, 128),
    block_ks: tuple[int, ...] = (32, 64, 128),
    num_warps_list: tuple[int, ...] = (2, 4, 8, 16),
    num_stages_list: tuple[int, ...] = (1, 2, 3),
    dtype_bytes: int = 4,
    buffered_tensors: int = 2,
) -> dict[str, Any]:
    """Genera y filtra el espacio de barrido estático para `sm_120` (sin `@triton.autotune`)."""
    if regime not in ("register_resident", "tiled_smem"):
        raise ValueError(f"Régimen inválido: {regime!r}")
    valid: list[dict[str, Any]] = []
    eliminated: list[dict[str, Any]] = []

    if regime == "register_resident":
        block_d = 1 << (dim - 1).bit_length()
        for nw in num_warps_list:
            for ns in num_stages_list:
                cfg = {"block_d": block_d, "num_warps": nw, "num_stages": ns}
                if nw not in (1, 2, 4, 8, 16, 32):
                    eliminated.append({**cfg, "reason": "invalid_num_warps"})
                    continue
                if ns != 2:
                    eliminated.append({**cfg, "reason": "redundant_num_stages_without_loop"})
                    continue
                smem = estimate_sm120_smem_bytes(
                    regime="register_resident", block_d=block_d, num_warps=nw, num_stages=ns
                )
                valid.append({**cfg, "smem_bytes": smem, "fits_sm120_smem": True})
    else:
        for bm in block_ms:
            for bk in block_ks:
                for nw in num_warps_list:
                    for ns in num_stages_list:
                        cfg = {"block_m": bm, "block_k": bk, "num_warps": nw, "num_stages": ns}
                        if not (_is_power_of_two(bm) and _is_power_of_two(bk)):
                            eliminated.append({**cfg, "reason": "non_power_of_two_block"})
                            continue
                        if nw not in (1, 2, 4, 8, 16, 32):
                            eliminated.append({**cfg, "reason": "invalid_num_warps"})
                            continue
                        if ns < 1:
                            eliminated.append({**cfg, "reason": "invalid_num_stages"})
                            continue
                        smem = estimate_sm120_smem_bytes(
                            regime="tiled_smem",
                            block_m=bm,
                            block_k=bk,
                            num_warps=nw,
                            num_stages=ns,
                            dtype_bytes=dtype_bytes,
                            buffered_tensors=buffered_tensors,
                        )
                        if smem > SM120_SPEC.max_smem_per_block_bytes:
                            eliminated.append(
                                {
                                    **cfg,
                                    "smem_bytes": smem,
                                    "reason": "smem_exceeds_sm120_99kib_cta_limit",
                                }
                            )
                            continue
                        valid.append(
                            {
                                **cfg,
                                "smem_bytes": smem,
                                "blocks_by_smem": SM120_SPEC.max_smem_per_sm_bytes // max(smem, 1),
                                "fits_sm120_smem": True,
                            }
                        )

    return {
        "schema_version": 1,
        "kind": "sm120_block_sweep_grid_v1",
        "arch": SM120_SPEC.arch,
        "dim": dim,
        "regime": regime,
        "autotune_forbidden": True,
        "autotune_forbidden_reason": (
            "Cambiar num_warps modifica el árbol de reducción tl.sum en 1-2 ULP; bajo concurrencia "
            "MPS el ruido de latencia haría @triton.autotune no determinista. Fijar por tabla "
            "estática."
        ),
        "sm120_limits": {
            "max_warps_per_sm": SM120_SPEC.max_warps_per_sm,
            "max_smem_per_block_bytes": SM120_SPEC.max_smem_per_block_bytes,
            "max_smem_per_sm_bytes": SM120_SPEC.max_smem_per_sm_bytes,
            "fp32_ridge_point": round(SM120_SPEC.fp32_ridge_point, 2),
        },
        "valid_configs": valid,
        "eliminated_configs": eliminated,
    }


def audit_repo_v11_kernel_sm120_ptx(
    *,
    num_warps_fwd: int = 4,
    num_warps_bwd: int = 8,
    num_stages: int = 2,
    layer_norm: bool = True,
) -> dict[str, Any]:
    """Compila offline a PTX `sm_120` el kernel real de `repo-v11` y audita atómicas/TF32/SMEM."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from mars_titan.models.titans.fused_memory_write import _backward_kernel, _forward_kernel

    enforce_strict_fp32_deterministic(check_triton=True)
    target = GPUTarget("cuda", 120, 32)

    fwd_sig = {
        name: "*fp32"
        for name in (
            "w1_ptr",
            "w2_ptr",
            "s1_ptr",
            "s2_ptr",
            "k_ptr",
            "v_ptr",
            "alpha_ptr",
            "eta_ptr",
            "theta_ptr",
            "w1_out",
            "w2_out",
            "s1_out",
            "s2_out",
            "elements_out",
        )
    }
    bwd_sig = {
        name: "*fp32"
        for name in (
            "w1_ptr",
            "w2_ptr",
            "s1_ptr",
            "s2_ptr",
            "k_ptr",
            "v_ptr",
            "alpha_ptr",
            "eta_ptr",
            "theta_ptr",
            "dw1_next_ptr",
            "dw2_next_ptr",
            "ds1_next_ptr",
            "ds2_next_ptr",
            "delements_ptr",
            "dw1_ptr",
            "dw2_ptr",
            "ds1_ptr",
            "ds2_ptr",
            "dk_ptr",
            "dv_ptr",
            "dalpha_ptr",
            "deta_ptr",
            "dtheta_ptr",
        )
    }
    constexprs = {"EPS": 1e-5, "D": KERNEL_DIM, "LAYER_NORM": layer_norm}

    def _compile_and_inspect(fn, sig, nw):
        src = ASTSource(fn=fn, signature=sig, constexprs=constexprs)
        compiled = triton.compile(
            src, target=target, options={"num_warps": nw, "num_stages": num_stages}
        )
        ptx = compiled.asm["ptx"]
        return {
            "num_warps": int(compiled.metadata.num_warps),
            "num_stages": int(num_stages),
            "shared_bytes": int(compiled.metadata.shared),
            "ptx_bytes": len(ptx),
            "atomic_ops": len(re.findall(r"\batom\.", ptx)),
            "tf32_ops": len(re.findall(r"\btf32\b", ptx)),
            "sqrt_approx_ftz_ops": len(re.findall(r"\bsqrt\.approx\.ftz\.f32\b", ptx)),
            "div_full_ops": len(re.findall(r"\bdiv\.full\.f32\b", ptx)),
        }

    return {
        "schema_version": 1,
        "kind": "repo_v11_fused_memory_write_sm120_ptx_audit_v1",
        "arch": SM120_SPEC.arch,
        "layer_norm": layer_norm,
        "forward": _compile_and_inspect(_forward_kernel, fwd_sig, num_warps_fwd),
        "backward": _compile_and_inspect(_backward_kernel, bwd_sig, num_warps_bwd),
    }


def run_sm120_gpu_sweep(
    *,
    batches: tuple[int, ...] = (1, 13, 64, 129, 4198),
    warp_pairs: tuple[tuple[int, int], ...] = ((4, 8), (8, 8), (16, 8), (16, 16)),
    eps: float = 1e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Ejecutor GPU para RTX PRO 6000 Blackwell (`sm_120`) — [no verificado en GPU]."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"run_sm120_gpu_sweep requiere GPU CUDA sm_120 ({UNVERIFIED_GPU_TAG})")

    import triton
    import triton.testing

    from mars_titan.models.titans import fused_memory_write as fmw

    fp32_state = enforce_strict_fp32_deterministic(check_triton=True)
    cap = torch.cuda.get_device_capability(0)
    results = []
    # Paso 1: Nivel 0 y Nivel 1 en la configuración por defecto (4, 8)
    orig_fwd, orig_bwd = fmw._FORWARD_WARPS, fmw._BACKWARD_WARPS
    try:
        for w_fwd, w_bwd in warp_pairs:
            fmw._FORWARD_WARPS, fmw._BACKWARD_WARPS = w_fwd, w_bwd
            # Validar Nivel 0 + Nivel 1 en lote representativo (129) antes de cronometrar
            in64_cpu, up64_cpu = make_write_inputs(129, dtype=torch.float64, seed=seed)
            ref_out64, ref_grad64 = _run_with_grads(
                reference_write_fp64, in64_cpu, up64_cpu, layer_norm=True, eps=eps
            )
            in32_gpu = tuple(t.to(torch.float32).cuda() for t in in64_cpu)
            up32_gpu = tuple(t.to(torch.float32).cuda() for t in up64_cpu)

            lvl0 = evaluate_level0_bitwise(
                fmw.fused_associative_write,
                fmw.fused_associative_write,
                in32_gpu,
                up32_gpu,
                layer_norm=True,
                eps=eps,
                repeats=3,
            )
            out_gpu, grad_gpu = _run_with_grads(
                fmw.fused_associative_write, in32_gpu, up32_gpu, layer_norm=True, eps=eps
            )
            lvl1 = evaluate_level1_fp64_parity(out_gpu, grad_gpu, ref_out64, ref_grad64)

            # Paso 2: Cronometraje sincronizado en tamaño de producción (B = 4198)
            prod_b = max(batches)
            in_p, up_p = make_write_inputs(
                prod_b, dtype=torch.float32, seed=seed + 1, device="cuda:0"
            )
            fwd_ms = float(
                triton.testing.do_bench(
                    lambda in_p=in_p: fmw.fused_associative_write(
                        in_p[:2],
                        in_p[2:4],
                        in_p[4],
                        in_p[5],
                        in_p[6:9],
                        residual_layer_norm=True,
                        eps=eps,
                    ),
                    warmup=25,
                    rep=100,
                )
            )
            # 132.1 KiB/flujo en forward D=64 FP32
            fwd_bytes = prod_b * (8 * KERNEL_DIM * KERNEL_DIM + 4 * KERNEL_DIM + 2) * 4
            fwd_gbps = (fwd_bytes / 1e9) / (fwd_ms / 1e3)
            results.append(
                {
                    "num_warps_fwd": w_fwd,
                    "num_warps_bwd": w_bwd,
                    "level0_passed": lvl0["passed"],
                    "level1_passed": lvl1["passed"],
                    "max_ulp_fwd": lvl1["max_ulp_fwd"],
                    "max_ulp_bwd": lvl1["max_ulp_bwd"],
                    "prod_batch": prod_b,
                    "fwd_ms": fwd_ms,
                    "fwd_gbps": fwd_gbps,
                    "fwd_bw_efficiency_pct": 100.0 * fwd_gbps / SM120_SPEC.bandwidth_gbps,
                }
            )
    finally:
        fmw._FORWARD_WARPS, fmw._BACKWARD_WARPS = orig_fwd, orig_bwd

    return {
        "schema_version": 1,
        "kind": "sm120_gpu_sweep_report_v1",
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(cap),
        "fp32_state": fp32_state,
        "compute_sanitizer_commands": [
            "compute-sanitizer --tool memcheck python3 kernel_sm120_plan.py --mode gpu-sweep",
            "compute-sanitizer --tool racecheck python3 kernel_sm120_plan.py --mode gpu-sweep",
        ],
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan de pruebas y barrido sm_120 de fused_memory_write.py."
    )
    parser.add_argument(
        "--mode",
        choices=("cpu-verify", "grid", "ptx-audit", "gpu-sweep"),
        required=True,
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--regime", choices=("register_resident", "tiled_smem"), default="register_resident"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.mode == "cpu-verify":
        doc = run_cpu_verification_plan()
    elif args.mode == "grid":
        doc = generate_sm120_sweep_grid(dim=args.dim, regime=args.regime)
    elif args.mode == "ptx-audit":
        doc = audit_repo_v11_kernel_sm120_ptx()
    else:
        doc = run_sm120_gpu_sweep()

    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
