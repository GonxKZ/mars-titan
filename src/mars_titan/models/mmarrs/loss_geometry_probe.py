"""Deep Learning Optimization Geometry, Hessian Spectrum, Neural Collapse & IB Probe.

Model-agnostic, GPU/CPU-native diagnostic engine designed to work identically across ALL
4 independent model families in `MARS-TITAN`:
  1. `titans_original` (`MMARRSTitan` Titans presets + DualAxis memory + TabM)
  2. `holographic_brain` (`StandaloneHolographicBrainModel`)
  3. `mmarrs_titan` (`MMARRSTitan` MoE / Looped / Counterfactual / Multimodal SOTA)
  4. `titans_cognitive_core` (`MMARRSTitan` + `TitansCognitiveCore` weight-tied toolbox)
as well as any arbitrary PyTorch `nn.Module`.

Design invariants (RTX PRO 6000 Blackwell `sm_120` & numerical stability):
  - Zero host-syncs (`.item()`, `.cpu()`, data-dependent Python branches) inside hot
    loops (`GradientDynamicsTracker.record_step`, `record_post_step`, Lanczos / power
    iteration inner loops, Hutchinson probe accumulation).
  - Master FP32/FP64 promotion inside `torch.amp.autocast(enabled=False)` for all
    eigenvalue, SVD, covariance, log-det, Welford, and CKA computations; safe under
    BF16/FP16 training and adversarial scales (`1e-4..1e4`).
  - Zero weight or `.grad` mutation: Hessian/SAM probing uses `torch.autograd.grad` and
    restores exact bit-identical parameter clones inside `try ... finally`.
  - Safe fallback when custom ops are not double-differentiable: switches cleanly to
    central finite-difference HVP (`central_fd_fallback`) or non-crashing skip
    (`skipped_non_double_diff`) while recording the exact method used.
  - Deterministic, ultra-compact, lossless RFC-1952 `.jsonl.gz` trace persistence
    (`write_geometry_trace_gz` / `read_geometry_trace_gz`) with strict RFC 8259 JSON
    sanitization (`NaN`/`Inf` -> `None`).
"""

from __future__ import annotations

import contextlib
import gzip
import inspect
import json
import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CANONICAL_DIAGNOSES: tuple[str, ...] = (
    "healthy_generalizing",
    "underfitting",
    "memorizing_overfit",
    "representation_collapse",
    "sharp_minimum",
    "gradient_instability",
)

_INPUT_BATCH_KEYS: tuple[str, ...] = (
    "prices",
    "physics_v4",
    "fundamentals",
    "high_snr_context",
    "news",
    "charts",
)


# ---------------------------------------------------------------------------
# Internal Numerical, Context & Serialization Utilities
# ---------------------------------------------------------------------------


def _calc_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Promote BF16/FP16 to FP32 master precision while preserving FP64 references."""
    if tensors and all(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def _math_sdpa_context() -> contextlib.AbstractContextManager[Any]:
    """Force math SDPA backend during second-order HVP so attention is double-differentiable."""
    attn_mod = getattr(torch.nn, "attention", None)
    if (
        attn_mod is not None
        and hasattr(attn_mod, "sdpa_kernel")
        and hasattr(attn_mod, "SDPBackend")
    ):
        return attn_mod.sdpa_kernel(attn_mod.SDPBackend.MATH)
    return contextlib.nullcontext()


def _autocast_disabled(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    dev_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    return torch.amp.autocast(device_type=dev_type, enabled=False)


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return [int(idx)]
    return []


def _capture_rng_state(
    fork_devices: Sequence[int],
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    cpu_state = torch.get_rng_state()
    cuda_states = [(idx, torch.cuda.get_rng_state(idx)) for idx in fork_devices]
    return cpu_state, cuda_states


def _restore_rng_state(
    state: tuple[torch.Tensor, list[tuple[int, torch.Tensor]]],
) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    for idx, c_state in cuda_states:
        torch.cuda.set_rng_state(c_state, idx)


def _resolve_trainable_params(
    model_or_params: nn.Module | Sequence[nn.Parameter],
) -> list[nn.Parameter]:
    if isinstance(model_or_params, nn.Module):
        return [p for p in model_or_params.parameters() if p.requires_grad]
    return [p for p in model_or_params if p.requires_grad]


def _unflatten_like(
    flat: torch.Tensor,
    params: Sequence[nn.Parameter],
) -> list[torch.Tensor]:
    parts: list[torch.Tensor] = []
    offset = 0
    for p in params:
        n = p.numel()
        chunk = flat[offset : offset + n].view_as(p)
        parts.append(chunk.to(device=p.device, dtype=p.dtype))
        offset += n
    return parts


def _flatten_tensors(
    tensors: Sequence[torch.Tensor | None],
    params: Sequence[nn.Parameter],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for t, p in zip(tensors, params, strict=True):
        if t is None:
            parts.append(torch.zeros(p.numel(), device=device, dtype=dtype))
        else:
            parts.append(t.detach().reshape(-1).to(device=device, dtype=dtype))
    if not parts:
        return torch.zeros(0, device=device, dtype=dtype)
    return torch.cat(parts, dim=0)


def _average_ranks_1d(x: torch.Tensor) -> torch.Tensor:
    """Compute tie-aware average ranks in FP64 on device with zero scipy dependency."""
    n = int(x.numel())
    if n == 0:
        return torch.empty(0, device=x.device, dtype=torch.float64)
    xf = x.reshape(-1).to(dtype=torch.float64)
    order = torch.argsort(xf, stable=True)
    sorted_x = xf[order]
    # Mid-ranks for ties: locate start and end of each equal-value run
    arange = torch.arange(n, device=x.device, dtype=torch.float64)
    if n == 1:
        return torch.zeros(1, device=x.device, dtype=torch.float64)
    diff = sorted_x[1:] != sorted_x[:-1]
    change_idx = torch.nonzero(diff, as_tuple=False).reshape(-1) + 1
    boundaries = torch.cat(
        [
            torch.zeros(1, device=x.device, dtype=torch.long),
            change_idx,
            torch.full((1,), n, device=x.device, dtype=torch.long),
        ]
    )
    starts = boundaries[:-1]
    ends = boundaries[1:]
    counts = ends - starts
    avg_val = 0.5 * (starts.to(torch.float64) + (ends - 1).to(torch.float64))
    sorted_ranks = torch.repeat_interleave(avg_val, counts)
    ranks = torch.empty_like(arange)
    ranks[order] = sorted_ranks
    return ranks


def _spearman_corr_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Finite tie-aware Spearman rank correlation (returns 0.0 on constant/degenerate vectors)."""
    a1 = a.reshape(-1).to(dtype=torch.float64)
    b1 = b.reshape(-1).to(device=a1.device, dtype=torch.float64)
    mask = torch.isfinite(a1) & torch.isfinite(b1)
    if int(mask.sum().item()) < 3:
        return 0.0
    ra = _average_ranks_1d(a1[mask])
    rb = _average_ranks_1d(b1[mask])
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    denom = torch.linalg.vector_norm(ra_c) * torch.linalg.vector_norm(rb_c)
    if float(denom.item()) < 1e-12:
        return 0.0
    val = float((torch.dot(ra_c, rb_c) / denom).item())
    return max(-1.0, min(1.0, val)) if math.isfinite(val) else 0.0


def _json_sanitize(obj: Any) -> Any:
    """Convert arbitrary objects to strict RFC 8259 JSON primitives (NaN/Inf -> None)."""
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, torch.Tensor):
        return _json_sanitize(obj.detach().cpu().tolist())
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _json_sanitize(obj.to_dict())
    if isinstance(obj, Mapping):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Pillar 1: Hessian Spectrum & Loss Landscape Curvature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HessianConfig:
    """Configuration for `HessianSpectrumAnalyzer`."""

    top_k: int = 5
    method: Literal["lanczos", "power_iteration"] = "lanczos"
    lanczos_steps: int = 20
    power_iters: int = 40
    hutchinson_probes: int = 16
    sam_rho: float = 0.05
    sam_adaptive: bool = False
    fallback_mode: Literal["finite_difference", "skip"] = "finite_difference"
    fd_eps: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}")
        if self.method not in ("lanczos", "power_iteration"):
            raise ValueError(f"Unsupported Hessian method {self.method!r}")
        if self.fallback_mode not in ("finite_difference", "skip"):
            raise ValueError(f"Unsupported fallback_mode {self.fallback_mode!r}")
        if self.sam_rho < 0.0:
            raise ValueError(f"sam_rho must be >= 0; got {self.sam_rho}")
        if self.fd_eps <= 0.0:
            raise ValueError(f"fd_eps must be > 0; got {self.fd_eps}")


@dataclass(frozen=True)
class HessianSpectrumResult:
    """Structured Hessian spectrum & SAM neighborhood sharpness diagnostics."""

    top_eigenvalues: list[float]
    lambda_max: float
    min_eigenvalue: float
    hessian_trace: float
    hessian_trace_se: float
    spectral_trace: float
    negative_eigenvalue_ratio: float
    saddle_curvature_ratio: float
    effective_condition_number: float
    sam_sharpness: float
    sam_relative_sharpness: float
    sam_rho: float
    hvp_method: Literal["exact_autograd", "central_fd_fallback", "skipped_non_double_diff"]
    fallback_reason: str | None = None
    n_params: int = 0
    ritz_eigenvalues: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HessianSpectrumAnalyzer:
    """Matrix-free Pearlmutter HVP, Lanczos/Power-Iteration spectrum, Hutchinson trace & SAM.

    All parameter perturbations restore exact bit-identical clones in `try ... finally`,
    and `.grad` attributes on model parameters are never modified.
    """

    def __init__(self, config: HessianConfig | None = None) -> None:
        self.config = config or HessianConfig()

    @staticmethod
    def _build_autograd_hvp_fn(
        loss_closure: Callable[[], torch.Tensor],
        params: list[nn.Parameter],
        device: torch.device,
        calc_dtype: torch.dtype,
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, str | None]:
        """Build a retained-graph exact Pearlmutter HVP operator, or report fallback reason."""
        try:
            with _autocast_disabled(device), torch.enable_grad(), _math_sdpa_context():
                loss = loss_closure()
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    loss = loss.mean()
                grads = torch.autograd.grad(
                    loss, params, create_graph=True, allow_unused=True
                )
        except RuntimeError as exc:
            return None, f"first_order_create_graph_failed: {exc}"

        active_idx = [
            i for i, g in enumerate(grads) if g is not None and bool(g.requires_grad)
        ]
        if not active_idx:
            return None, "no_double_differentiable_grad_graph"

        active_params = [params[i] for i in active_idx]
        active_grads = [grads[i] for i in active_idx]
        offsets: list[tuple[int, int, torch.Size, torch.device, torch.dtype]] = []
        cursor = 0
        for p in params:
            n = p.numel()
            offsets.append((cursor, cursor + n, p.shape, p.device, p.dtype))
            cursor += n
        total_dim = cursor

        def _hvp_flat(v_flat: torch.Tensor) -> torch.Tensor:
            v_active = [
                v_flat[offsets[i][0] : offsets[i][1]]
                .view(offsets[i][2])
                .to(device=active_grads[k].device, dtype=active_grads[k].dtype)
                for k, i in enumerate(active_idx)
            ]
            with _autocast_disabled(device):
                hv_active = torch.autograd.grad(
                    active_grads,
                    active_params,
                    grad_outputs=v_active,
                    retain_graph=True,
                    allow_unused=True,
                )
            out = torch.zeros(total_dim, device=device, dtype=calc_dtype)
            for k, i in enumerate(active_idx):
                hvk = hv_active[k]
                if hvk is not None:
                    start, end = offsets[i][0], offsets[i][1]
                    out[start:end] = hvk.detach().reshape(-1).to(device=device, dtype=calc_dtype)
            return out

        # Probe all active parameters with a unit-norm vector to catch deferred double-bwd errors
        try:
            test_v = torch.zeros(total_dim, device=device, dtype=calc_dtype)
            if total_dim > 0:
                val = 1.0 / math.sqrt(float(total_dim))
                test_v.fill_(val)
            probe_out = _hvp_flat(test_v)
            if not bool(torch.isfinite(probe_out).all().item()):
                return None, "nonfinite_exact_hvp_probe"
        except RuntimeError as exc:
            return None, f"second_order_autograd_failed: {exc}"

        return _hvp_flat, None

    @staticmethod
    def _build_central_fd_hvp_fn(
        loss_closure: Callable[[], torch.Tensor],
        params: list[nn.Parameter],
        device: torch.device,
        calc_dtype: torch.dtype,
        fd_eps: float,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Central finite-difference HVP `(g(w + h v) - g(w - h v)) / (2 h)` with exact restore."""
        orig_clones = [p.detach().clone() for p in params]
        has_low_prec = any(p.dtype in (torch.float16, torch.bfloat16) for p in params)
        eff_step = max(float(fd_eps), 2e-2 if has_low_prec else 0.0)

        def _grad_flat() -> torch.Tensor:
            with _autocast_disabled(device), torch.enable_grad():
                loss = loss_closure()
                if loss.ndim != 0:
                    loss = loss.mean()
                grads = torch.autograd.grad(
                    loss, params, create_graph=False, allow_unused=True
                )
            return _flatten_tensors(grads, params, device, calc_dtype)

        def _fd_hvp_flat(v_flat: torch.Tensor) -> torch.Tensor:
            v_norm = torch.linalg.vector_norm(v_flat)
            safe_norm = v_norm.clamp_min(1e-12)
            v_unit = v_flat / safe_norm
            v_parts = _unflatten_like(v_unit, params)
            step = eff_step
            fork_devs = _fork_rng_devices(device)
            with torch.random.fork_rng(devices=fork_devs):
                rng_snap = _capture_rng_state(fork_devs)
                try:
                    with torch.no_grad():
                        for p, orig, vp in zip(params, orig_clones, v_parts, strict=True):
                            p.copy_(orig)
                            p.add_(vp, alpha=step)
                    g_plus = _grad_flat()

                    # Replay identical RNG (Common Random Numbers) so stochastic ops cancel cleanly
                    _restore_rng_state(rng_snap)
                    with torch.no_grad():
                        for p, orig, vp in zip(params, orig_clones, v_parts, strict=True):
                            p.copy_(orig)
                            p.add_(vp, alpha=-step)
                    g_minus = _grad_flat()
                finally:
                    with torch.no_grad():
                        for p, orig in zip(params, orig_clones, strict=True):
                            p.copy_(orig)
            return ((g_plus - g_minus) / (2.0 * step)) * v_norm

        return _fd_hvp_flat

    def compute_hvp(
        self,
        loss_closure: Callable[[], torch.Tensor],
        params: Sequence[nn.Parameter],
        vectors: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Compute matrix-free Hessian-Vector Product `H @ v` matching parameter shapes."""
        param_list = [p for p in params if p.requires_grad]
        if not param_list:
            return []
        device = param_list[0].device
        cdtype = _calc_dtype(*param_list)
        v_flat = _flatten_tensors(vectors, param_list, device, cdtype)

        hvp_fn, reason = self._build_autograd_hvp_fn(loss_closure, param_list, device, cdtype)
        if hvp_fn is None:
            if self.config.fallback_mode == "finite_difference":
                hvp_fn = self._build_central_fd_hvp_fn(
                    loss_closure, param_list, device, cdtype, self.config.fd_eps
                )
            else:
                raise RuntimeError(
                    f"Exact HVP unavailable ({reason}) and fallback_mode='skip'"
                )
        hv_flat = hvp_fn(v_flat)
        return _unflatten_like(hv_flat, param_list)

    def compute_sam_sharpness(
        self,
        model_or_params: nn.Module | Sequence[nn.Parameter],
        loss_closure: Callable[[], torch.Tensor],
        *,
        rho: float | None = None,
        adaptive: bool | None = None,
    ) -> tuple[float, float]:
        """Compute 1-step SAM neighborhood sharpness `max_{||eps||<=rho} [L(w+eps) - L(w)]`.

        Guarantees exact bit-for-bit parameter restoration and preserves existing `.grad` buffers.
        Returns `(sam_sharpness, sam_relative_sharpness)`.
        """
        eff_rho = float(self.config.sam_rho if rho is None else rho)
        eff_adaptive = bool(self.config.sam_adaptive if adaptive is None else adaptive)
        params = _resolve_trainable_params(model_or_params)
        if not params or eff_rho == 0.0:
            return 0.0, 0.0

        device = params[0].device
        cdtype = _calc_dtype(*params)
        orig_clones = [p.detach().clone() for p in params]
        fork_devs = _fork_rng_devices(device)

        with torch.random.fork_rng(devices=fork_devs), _autocast_disabled(device):
            rng_snap = _capture_rng_state(fork_devs)
            with torch.enable_grad():
                loss_0 = loss_closure()
                if loss_0.ndim != 0:
                    loss_0 = loss_0.mean()
                grads = torch.autograd.grad(
                    loss_0, params, create_graph=False, allow_unused=True
                )

            with torch.no_grad():
                sq_sum = torch.zeros((), device=device, dtype=torch.float64)
                scaled_grads: list[torch.Tensor | None] = []
                for p, g in zip(params, grads, strict=True):
                    if g is None:
                        scaled_grads.append(None)
                        continue
                    gf = g.detach().to(device=device, dtype=cdtype)
                    if eff_adaptive:
                        scale = p.detach().to(device=device, dtype=cdtype).abs()
                        eff_g = gf * (scale * scale)
                        sq_sum = sq_sum + (gf * scale).to(torch.float64).pow(2).sum()
                    else:
                        eff_g = gf
                        sq_sum = sq_sum + gf.to(torch.float64).pow(2).sum()
                    scaled_grads.append(eff_g)

                grad_norm = torch.sqrt(sq_sum)
                denom = grad_norm.clamp_min(1e-12)
                step_scale = torch.where(
                    grad_norm > 1e-12,
                    torch.full((), eff_rho, device=device, dtype=torch.float64) / denom,
                    torch.zeros((), device=device, dtype=torch.float64),
                )

                try:
                    for p, sg in zip(params, scaled_grads, strict=True):
                        if sg is not None:
                            delta = (sg * step_scale.to(dtype=sg.dtype)).to(
                                device=p.device, dtype=p.dtype
                            )
                            p.add_(delta)
                    # Replay identical RNG so stochastic dropout/noise cannot corrupt sharpness
                    _restore_rng_state(rng_snap)
                    loss_pert = loss_closure()
                    if loss_pert.ndim != 0:
                        loss_pert = loss_pert.mean()
                finally:
                    for p, orig in zip(params, orig_clones, strict=True):
                        p.copy_(orig)

        l0_val = float(loss_0.detach().to(dtype=torch.float64).item())
        lp_val = float(loss_pert.detach().to(dtype=torch.float64).item())
        sharpness = lp_val - l0_val
        rel_sharpness = sharpness / max(abs(l0_val), 1e-8)
        return sharpness, rel_sharpness

    @staticmethod
    def _random_unit_vector(
        dim: int,
        gen_cpu: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        v = torch.randn((dim,), generator=gen_cpu, dtype=dtype).to(device=device)
        norm = torch.linalg.vector_norm(v).clamp_min(1e-12)
        return v / norm

    @classmethod
    def _lanczos_spectrum(
        cls,
        hvp_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        n_steps: int,
        gen_cpu: torch.Generator,
        device: torch.device,
        calc_dtype: torch.dtype,
    ) -> list[float]:
        """Full-reorthogonalization symmetric Lanczos with 0-sync invariant-subspace restart."""
        m = min(max(1, n_steps), dim)
        tol = 1e-10
        zero_s = torch.zeros((), device=device, dtype=calc_dtype)
        q_cur = cls._random_unit_vector(dim, gen_cpu, device, calc_dtype)
        q_list: list[torch.Tensor] = [q_cur]
        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        beta_prev = zero_s

        with _autocast_disabled(device):
            for j in range(m):
                w = hvp_fn(q_cur)
                if j > 0:
                    w = w - beta_prev * q_list[j - 1]
                alpha_j = torch.dot(q_cur, w)
                w = w - alpha_j * q_cur

                # Double Modified Gram-Schmidt via BLAS-2 matvec (Kahan-Parlett: twice is enough)
                q_mat = torch.stack(q_list, dim=0)  # [j+1, dim]
                for _ in range(2):
                    coeffs = q_mat @ w
                    w = w - coeffs @ q_mat

                alphas.append(alpha_j)
                if j == m - 1:
                    break

                beta_j = torch.linalg.vector_norm(w)
                # Zero-host-sync backup orthogonal vector in case of invariant-subspace breakdown
                r = torch.randn((dim,), generator=gen_cpu, dtype=calc_dtype).to(device=device)
                for _ in range(2):
                    r = r - (q_mat @ r) @ q_mat
                r_norm = torch.linalg.vector_norm(r).clamp_min(1e-12)
                r_unit = r / r_norm

                valid = beta_j > tol
                beta_eff = torch.where(valid, beta_j, zero_s)
                q_next = torch.where(valid, w / beta_j.clamp_min(tol), r_unit)

                betas.append(beta_eff)
                q_list.append(q_next)
                q_cur = q_next
                beta_prev = beta_eff

        # Single batched device -> host transfer for tiny m x m tridiagonal eigendecomposition
        alphas_cpu = torch.stack(alphas).detach().to(device="cpu", dtype=torch.float64)
        t_mat = torch.diag(alphas_cpu)
        if betas:
            betas_cpu = torch.stack(betas).detach().to(device="cpu", dtype=torch.float64)
            t_mat = t_mat + torch.diag(betas_cpu, diagonal=1) + torch.diag(betas_cpu, diagonal=-1)
        ritz = torch.linalg.eigvalsh(t_mat)
        return [float(x) for x in ritz.tolist()]

    @classmethod
    def _deflated_power_iteration(
        cls,
        hvp_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        top_k: int,
        power_iters: int,
        gen_cpu: torch.Generator,
        device: torch.device,
        calc_dtype: torch.dtype,
    ) -> list[float]:
        """Deterministic deflated power iteration for top-k eigenvalues (0-sync inner loop)."""
        k_eff = min(max(1, top_k), dim)
        found_vals: list[torch.Tensor] = []
        found_vecs: list[torch.Tensor] = []

        with _autocast_disabled(device):
            for _ in range(k_eff):
                v = cls._random_unit_vector(dim, gen_cpu, device, calc_dtype)
                if found_vecs:
                    q_prev = torch.stack(found_vecs, dim=0)
                    for _ in range(2):
                        v = v - (q_prev @ v) @ q_prev
                    v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)

                for _ in range(max(1, power_iters)):
                    w = hvp_fn(v)
                    if found_vecs:
                        q_prev = torch.stack(found_vecs, dim=0)
                        lam_prev = torch.stack(found_vals)
                        w = w - (lam_prev * (q_prev @ v)) @ q_prev
                        w = w - (q_prev @ w) @ q_prev
                    norm_w = torch.linalg.vector_norm(w)
                    v = torch.where(norm_w > 1e-12, w / norm_w.clamp_min(1e-12), v)

                hv = hvp_fn(v)
                if found_vecs:
                    q_prev = torch.stack(found_vecs, dim=0)
                    lam_prev = torch.stack(found_vals)
                    hv = hv - (lam_prev * (q_prev @ v)) @ q_prev
                lam = torch.dot(v, hv)
                found_vals.append(lam)
                found_vecs.append(v)

        vals_cpu = torch.stack(found_vals).detach().to(device="cpu", dtype=torch.float64)
        return sorted((float(x) for x in vals_cpu.tolist()), reverse=True)

    @staticmethod
    def _hutchinson_trace(
        hvp_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        n_probes: int,
        gen_cpu: torch.Generator,
        device: torch.device,
        calc_dtype: torch.dtype,
    ) -> tuple[float, float]:
        """Unbiased Hutchinson Hessian trace via Rademacher +-1 probes (0-sync loop)."""
        m = max(1, int(n_probes))
        samples: list[torch.Tensor] = []
        with _autocast_disabled(device):
            for _ in range(m):
                bits = torch.randint(0, 2, (dim,), generator=gen_cpu, dtype=torch.int8)
                z = (bits.to(device=device, dtype=calc_dtype) * 2.0) - 1.0
                hz = hvp_fn(z)
                samples.append(torch.dot(z, hz).to(dtype=torch.float64))
        stacked = torch.stack(samples)
        mean_t = stacked.mean()
        if m > 1:
            centered = stacked - mean_t
            se_t = torch.sqrt(centered.pow(2).sum() / float(m * (m - 1)))
        else:
            se_t = torch.zeros((), device=device, dtype=torch.float64)
        pair = torch.stack([mean_t, se_t]).detach().cpu()
        return float(pair[0].item()), float(pair[1].item())

    def analyze(
        self,
        model_or_params: nn.Module | Sequence[nn.Parameter],
        loss_closure: Callable[[], torch.Tensor],
    ) -> HessianSpectrumResult:
        """Run full Hessian spectrum + Hutchinson trace + SAM sharpness audit."""
        params = _resolve_trainable_params(model_or_params)
        total_params = sum(p.numel() for p in params)
        cfg = self.config

        # 1) First-order SAM neighborhood sharpness (always safe, never mutates weights!)
        sam_sharpness, sam_rel = self.compute_sam_sharpness(params, loss_closure)

        if total_params == 0:
            return HessianSpectrumResult(
                top_eigenvalues=[],
                lambda_max=0.0,
                min_eigenvalue=0.0,
                hessian_trace=0.0,
                hessian_trace_se=0.0,
                spectral_trace=0.0,
                negative_eigenvalue_ratio=0.0,
                saddle_curvature_ratio=0.0,
                effective_condition_number=1.0,
                sam_sharpness=sam_sharpness,
                sam_relative_sharpness=sam_rel,
                sam_rho=cfg.sam_rho,
                hvp_method="skipped_non_double_diff",
                fallback_reason="no_trainable_parameters",
                n_params=0,
            )

        device = params[0].device
        cdtype = _calc_dtype(*params)
        hvp_fn, reason = self._build_autograd_hvp_fn(loss_closure, params, device, cdtype)
        hvp_method: Literal["exact_autograd", "central_fd_fallback", "skipped_non_double_diff"]

        if hvp_fn is not None:
            hvp_method = "exact_autograd"
        elif cfg.fallback_mode == "finite_difference":
            hvp_fn = self._build_central_fd_hvp_fn(
                loss_closure, params, device, cdtype, cfg.fd_eps
            )
            hvp_method = "central_fd_fallback"
        else:
            return HessianSpectrumResult(
                top_eigenvalues=[],
                lambda_max=float("nan"),
                min_eigenvalue=float("nan"),
                hessian_trace=float("nan"),
                hessian_trace_se=float("nan"),
                spectral_trace=float("nan"),
                negative_eigenvalue_ratio=float("nan"),
                saddle_curvature_ratio=float("nan"),
                effective_condition_number=float("nan"),
                sam_sharpness=sam_sharpness,
                sam_relative_sharpness=sam_rel,
                sam_rho=cfg.sam_rho,
                hvp_method="skipped_non_double_diff",
                fallback_reason=reason,
                n_params=total_params,
            )

        gen_spec = torch.Generator(device="cpu").manual_seed(cfg.seed)
        gen_hutch = torch.Generator(device="cpu").manual_seed(cfg.seed + 10_007)

        if cfg.method == "lanczos":
            ritz_asc = self._lanczos_spectrum(
                hvp_fn,
                total_params,
                max(cfg.lanczos_steps, cfg.top_k),
                gen_spec,
                device,
                cdtype,
            )
            eigs_desc = list(reversed(ritz_asc))
        else:
            eigs_desc = self._deflated_power_iteration(
                hvp_fn,
                total_params,
                cfg.top_k,
                cfg.power_iters,
                gen_spec,
                device,
                cdtype,
            )
            ritz_asc = list(reversed(eigs_desc))

        k_out = min(cfg.top_k, len(eigs_desc))
        top_eigs = eigs_desc[:k_out]
        lambda_max = float(eigs_desc[0])
        min_eig = float(ritz_asc[0])
        spectral_trace = float(sum(ritz_asc))

        hutch_trace, hutch_se = self._hutchinson_trace(
            hvp_fn,
            total_params,
            cfg.hutchinson_probes,
            gen_hutch,
            device,
            cdtype,
        )

        max_abs = max((abs(v) for v in ritz_asc), default=1.0)
        neg_tol = 1e-6 * max(1.0, max_abs)
        n_neg = sum(1 for v in ritz_asc if v < -neg_tol)
        neg_ratio = float(n_neg) / float(max(len(ritz_asc), 1))
        saddle_ratio = max(0.0, -min_eig) / max(abs(lambda_max), abs(min_eig), 1e-12)
        # Unbounded PyHessian-style outlier-to-bulk ratio (unlike lambda_max/mean|ritz| <= m):
        # median(|ritz|) tracks bulk curvature while |Tr(H)|/P guards rank-deficient zeros.
        abs_ritz = sorted(abs(v) for v in ritz_asc)
        n_r = len(abs_ritz)
        if n_r == 0:
            median_abs_ritz = 0.0
        elif n_r % 2 == 1:
            median_abs_ritz = abs_ritz[n_r // 2]
        else:
            median_abs_ritz = 0.5 * (abs_ritz[n_r // 2 - 1] + abs_ritz[n_r // 2])
        bulk_floor = abs(hutch_trace) / float(max(total_params, 1))
        bulk_scale = max(median_abs_ritz, bulk_floor, 1e-12)
        eff_cond = max(lambda_max, 0.0) / bulk_scale

        return HessianSpectrumResult(
            top_eigenvalues=top_eigs,
            lambda_max=lambda_max,
            min_eigenvalue=min_eig,
            hessian_trace=hutch_trace,
            hessian_trace_se=hutch_se,
            spectral_trace=spectral_trace,
            negative_eigenvalue_ratio=neg_ratio,
            saddle_curvature_ratio=saddle_ratio,
            effective_condition_number=eff_cond,
            sam_sharpness=sam_sharpness,
            sam_relative_sharpness=sam_rel,
            sam_rho=cfg.sam_rho,
            hvp_method=hvp_method,
            fallback_reason=reason,
            n_params=total_params,
            ritz_eigenvalues=ritz_asc,
        )


# ---------------------------------------------------------------------------
# Pillar 2: Latent Representation Health, Financial Neural Collapse & Linear CKA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentationHealthResult:
    """Spectral dimensionality & collapse diagnostics on latent representations `Z`."""

    n_samples: int
    dim: int
    effective_rank: float
    effective_rank_ratio: float
    rankme: float
    stable_rank: float
    dead_unit_fraction: float
    spectral_decay_exponent: float
    singular_values_top_k: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NeuralCollapseResult:
    """Continuous-target decile Neural Collapse (`NC1`, `NC2`, ordinal monotonicity, `NC3`)."""

    nc1_within_between_ratio: float
    nc1_trace_ratio: float
    nc2_etf_simplex_deviation: float
    nc2_equinorm_cv: float
    ordinal_monotonicity_rho: float
    nc3_head_duality_cosine: float | None
    n_quantiles_used: int
    decile_mean_norms: list[float]
    nc1_active_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopCKAResult:
    """Linear CKA trajectory across recursive loop / cognitive thinking steps `k=0..K-1`."""

    n_steps: int
    adjacent_cka: list[float]
    first_to_last_cka: float
    mean_adjacent_cka: float
    min_adjacent_cka: float
    stagnant_loop_detected: bool
    stagnation_threshold: float
    cka_matrix: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_linear_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    debiased: bool = False,
) -> float:
    """Exact $O(N D^2 + D_1 D_2)$ Linear CKA (Kornblith et al. 2019).

    Invariant to orthogonal transformations, isotropic scaling, and translation; supports
    representations of different widths `D_1 != D_2` without materializing `N x N` Grams.
    """
    x2 = x.reshape(-1, x.shape[-1])
    y2 = y.reshape(-1, y.shape[-1])
    if x2.shape[0] != y2.shape[0]:
        raise ValueError(
            f"CKA requires identical sample count N; got {x2.shape[0]} vs {y2.shape[0]}"
        )
    device = x2.device
    with _autocast_disabled(device), torch.no_grad():
        xf = x2.to(device=device, dtype=torch.float64)
        yf = y2.to(device=device, dtype=torch.float64)
        finite_mask = torch.isfinite(xf).all(dim=-1) & torch.isfinite(yf).all(dim=-1)
        xf = xf[finite_mask]
        yf = yf[finite_mask]
        n = int(xf.shape[0])
        if n < 2:
            return 0.0

        xc = xf - xf.mean(dim=0, keepdim=True)
        yc = yf - yf.mean(dim=0, keepdim=True)
        # Scale-normalize by Frobenius norm to eliminate overflow/underflow at extreme scales
        nx = torch.linalg.matrix_norm(xc, ord="fro")
        ny = torch.linalg.matrix_norm(yc, ord="fro")
        if float(nx.item()) < 1e-12 or float(ny.item()) < 1e-12:
            return 0.0
        xc = xc / nx
        yc = yc / ny

        if not debiased or n < 4:
            cross = yc.T @ xc
            xx = xc.T @ xc
            yy = yc.T @ yc
            hsic_xy = cross.pow(2).sum()
            hsic_xx = xx.pow(2).sum()
            hsic_yy = yy.pow(2).sum()
            denom = torch.sqrt((hsic_xx * hsic_yy).clamp_min(0.0))
            if float(denom.item()) < 1e-15:
                return 0.0
            val = float((hsic_xy / denom).item())
            return max(0.0, min(1.0, val))

        # Unbiased HSIC estimator (Song et al. 2012 / Nguyen et al. 2021) for small N
        k_mat = xc @ xc.T
        l_mat = yc @ yc.T
        k_mat.fill_diagonal_(0.0)
        l_mat.fill_diagonal_(0.0)
        nf = float(n)

        def _unbiased_hsic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            term1 = (a * b).sum()
            term2 = (a.sum() * b.sum()) / ((nf - 1.0) * (nf - 2.0))
            term3 = (2.0 / (nf - 2.0)) * (a.sum(dim=1) * b.sum(dim=1)).sum()
            return (term1 + term2 - term3) / (nf * (nf - 3.0))

        h_xy = _unbiased_hsic(k_mat, l_mat)
        h_xx = _unbiased_hsic(k_mat, k_mat)
        h_yy = _unbiased_hsic(l_mat, l_mat)
        denom_u = torch.sqrt((h_xx * h_yy).clamp_min(0.0))
        if float(denom_u.item()) < 1e-15:
            return 0.0
        val_u = float((h_xy / denom_u).item())
        return max(-1.0, min(1.0, val_u)) if math.isfinite(val_u) else 0.0


class RepresentationGeometryAnalyzer:
    """Analyzes latent rank health, continuous-target Neural Collapse, and Loop CKA."""

    compute_linear_cka = staticmethod(compute_linear_cka)

    @staticmethod
    def _entropy_effective_rank(singular_values: torch.Tensor) -> float:
        s_pos = singular_values[singular_values > 1e-12]
        if s_pos.numel() == 0:
            return 0.0
        p = s_pos / s_pos.sum().clamp_min(1e-15)
        entropy = -(p * torch.log(p.clamp_min(1e-15))).sum()
        return float(torch.exp(entropy).item())

    @classmethod
    def analyze_representation_health(
        cls,
        z: torch.Tensor,
        *,
        rel_dead_tol: float = 1e-4,
        top_k_sv: int = 8,
    ) -> RepresentationHealthResult:
        """Compute Roy & Vetterli eRank, RankMe, Stable Rank, Dead Units & Spectral Decay."""
        z2 = z.reshape(-1, z.shape[-1])
        device = z2.device
        dim = int(z2.shape[-1])
        with _autocast_disabled(device), torch.no_grad():
            zf = z2.to(device=device, dtype=torch.float64)
            finite_rows = torch.isfinite(zf).all(dim=-1)
            zf = zf[finite_rows]
            n = int(zf.shape[0])
            if n < 2 or dim == 0:
                return RepresentationHealthResult(
                    n_samples=n,
                    dim=dim,
                    effective_rank=0.0,
                    effective_rank_ratio=0.0,
                    rankme=0.0,
                    stable_rank=0.0,
                    dead_unit_fraction=1.0,
                    spectral_decay_exponent=0.0,
                    singular_values_top_k=[],
                )

            # Scale-aware dead-unit fraction (detects zero OR constant-variance collapsed neurons)
            zc = zf - zf.mean(dim=0, keepdim=True)
            std_j = torch.sqrt(zc.pow(2).mean(dim=0))
            max_std = float(std_j.max().item())
            if max_std <= 1e-12:
                dead_frac = 1.0
            else:
                dead_thresh = max(1e-12, float(rel_dead_tol) * max_std)
                dead_frac = float((std_j <= dead_thresh).to(torch.float64).mean().item())

            # Scale-stabilized SVD of uncentered Z (RankMe, Garrido et al. 2023)
            raw_scale = float(zf.abs().max().item())
            if raw_scale <= 1e-12:
                sv_raw = torch.zeros(min(n, dim), device=device, dtype=torch.float64)
                rankme = 0.0
            else:
                sv_raw = torch.linalg.svdvals(zf / raw_scale) * raw_scale
                rankme = cls._entropy_effective_rank(sv_raw)

            # Scale-stabilized SVD of centered Z_c (Roy & Vetterli eRank, Stable Rank, Decay)
            c_scale = float(zc.abs().max().item())
            if c_scale <= 1e-12:
                return RepresentationHealthResult(
                    n_samples=n,
                    dim=dim,
                    effective_rank=0.0,
                    effective_rank_ratio=0.0,
                    rankme=rankme,
                    stable_rank=0.0,
                    dead_unit_fraction=dead_frac,
                    spectral_decay_exponent=0.0,
                    singular_values_top_k=[],
                )

            sv_c = torch.linalg.svdvals(zc / c_scale) * c_scale
            erank = cls._entropy_effective_rank(sv_c)
            max_rank = float(min(n - 1, dim))
            erank_ratio = erank / max(max_rank, 1.0)

            s1_sq = float(sv_c[0].pow(2).item())
            stable_rank = float(sv_c.pow(2).sum().item() / s1_sq) if s1_sq > 1e-24 else 0.0

            # Spectral decay exponent alpha: covariance eigenvalues lambda_k ~ k^{-alpha}
            eig_cov = sv_c.pow(2) / float(n)
            active = eig_cov > (1e-10 * eig_cov[0].clamp_min(1e-24))
            n_act = int(active.sum().item())
            if n_act >= 2:
                log_k = torch.log(torch.arange(1, n_act + 1, device=device, dtype=torch.float64))
                log_lam = torch.log(eig_cov[:n_act])
                lk_c = log_k - log_k.mean()
                ll_c = log_lam - log_lam.mean()
                denom_slope = lk_c.pow(2).sum().clamp_min(1e-15)
                slope = float((torch.dot(lk_c, ll_c) / denom_slope).item())
                decay_exp = max(0.0, -slope)
            else:
                decay_exp = 0.0

            k_sv = min(top_k_sv, int(sv_c.numel()))
            top_sv_list = [float(v) for v in sv_c[:k_sv].cpu().tolist()]

        return RepresentationHealthResult(
            n_samples=n,
            dim=dim,
            effective_rank=erank,
            effective_rank_ratio=erank_ratio,
            rankme=rankme,
            stable_rank=stable_rank,
            dead_unit_fraction=dead_frac,
            spectral_decay_exponent=decay_exp,
            singular_values_top_k=top_sv_list,
        )

    @classmethod
    def analyze_neural_collapse(
        cls,
        z: torch.Tensor,
        y: torch.Tensor,
        *,
        w_head: torch.Tensor | None = None,
        n_quantiles: int = 10,
        rcond: float = 1e-2,
    ) -> NeuralCollapseResult:
        """Continuous-target decile Neural Collapse (`NC1`, `NC2`, ordinal monotonicity, `NC3`).

        Uses truncated Moore-Penrose pseudoinverse on `Sigma_B` (default `rcond=1e-2`) so
        both full-rank Simplex ETF geometries and 1D continuous Neural Regression Collapse
        (where orthogonal finite-sample sampling noise lives orders of magnitude below
        `lambda_max(Sigma_B)`) are measured without spurious noise-eigenvalue amplification.
        """
        z2 = z.reshape(-1, z.shape[-1])
        y1 = y[:, 0] if y.ndim == 2 else y.reshape(-1)
        if z2.shape[0] != y1.shape[0]:
            raise ValueError(f"Sample mismatch: Z has {z2.shape[0]}, Y has {y1.shape[0]}")

        device = z2.device
        with _autocast_disabled(device), torch.no_grad():
            zf = z2.to(device=device, dtype=torch.float64)
            yf = y1.to(device=device, dtype=torch.float64)
            valid = torch.isfinite(zf).all(dim=-1) & torch.isfinite(yf)
            zf = zf[valid]
            yf = yf[valid]
            n, d = int(zf.shape[0]), int(zf.shape[1])
            q_eff = min(max(2, int(n_quantiles)), max(2, n // 2))
            if n < 4 or d == 0:
                return NeuralCollapseResult(
                    nc1_within_between_ratio=float("nan"),
                    nc1_trace_ratio=float("nan"),
                    nc2_etf_simplex_deviation=float("nan"),
                    nc2_equinorm_cv=float("nan"),
                    ordinal_monotonicity_rho=0.0,
                    nc3_head_duality_cosine=None,
                    n_quantiles_used=0,
                    decile_mean_norms=[],
                    nc1_active_rank=0,
                )

            order = torch.argsort(yf, stable=True)
            chunks = torch.tensor_split(order, q_eff)
            means = torch.stack([zf[idx].mean(dim=0) for idx in chunks], dim=0)  # [Q, D]
            mu_g = means.mean(dim=0, keepdim=True)  # [1, D]
            centered_means = means - mu_g  # [Q, D]

            # Between-decile covariance Sigma_B and within-decile covariance Sigma_W
            sigma_b = (centered_means.T @ centered_means) / float(q_eff)
            residuals = torch.cat(
                [zf[idx] - means[q_i].unsqueeze(0) for q_i, idx in enumerate(chunks)],
                dim=0,
            )
            sigma_w = (residuals.T @ residuals) / float(n)

            # NC1: Tr(Sigma_W @ pinv(Sigma_B)) / Q via truncated symmetric eigendecomposition
            evals_b, evecs_b = torch.linalg.eigh(sigma_b)
            lam_b_max = float(evals_b[-1].item())
            tr_w = float(torch.trace(sigma_w).item())
            tr_b = float(torch.trace(sigma_b).item())
            active_rank = 0

            if lam_b_max <= 1e-12:
                nc1 = float("inf") if tr_w > 1e-12 else float("nan")
                nc1_tr = float("inf") if tr_w > 1e-12 else float("nan")
            else:
                thresh = max(1e-12, float(rcond) * lam_b_max)
                active_mask = evals_b > thresh
                # Rank of Sigma_B cannot exceed Q - 1
                active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
                if active_indices.numel() > q_eff - 1:
                    active_indices = active_indices[-(q_eff - 1) :]
                active_rank = int(active_indices.numel())
                u_act = evecs_b[:, active_indices]  # [D, r]
                lam_act = evals_b[active_indices]  # [r]
                # u_j^T Sigma_W u_j for each active direction
                proj_w = (u_act * (sigma_w @ u_act)).sum(dim=0).clamp_min(0.0)
                nc1 = float((proj_w / lam_act).sum().item() / float(q_eff))
                nc1_tr = tr_w / max(tr_b, 1e-12)

            # NC2: Simplex ETF deviation ||G_M / ||G_M||_F - G_ETF / ||G_ETF||_F||_F
            gram_m = centered_means @ centered_means.T
            norm_gm = torch.linalg.matrix_norm(gram_m, ord="fro")
            eye_q = torch.eye(q_eff, device=device, dtype=torch.float64)
            ones_q = torch.ones((q_eff, q_eff), device=device, dtype=torch.float64)
            g_etf = (float(q_eff) / float(q_eff - 1)) * eye_q - (
                1.0 / float(q_eff - 1)
            ) * ones_q
            g_etf_norm = g_etf / torch.linalg.matrix_norm(g_etf, ord="fro").clamp_min(1e-12)
            if float(norm_gm.item()) <= 1e-12:
                nc2_etf = float(math.sqrt(2.0))
            else:
                nc2_etf = float(
                    torch.linalg.matrix_norm(gram_m / norm_gm - g_etf_norm, ord="fro").item()
                )

            mean_norms = torch.linalg.vector_norm(centered_means, dim=-1)
            mean_norm_avg = float(mean_norms.mean().item())
            if mean_norm_avg <= 1e-12:
                equinorm_cv = 0.0
            else:
                equinorm_cv = float(
                    (mean_norms.std(unbiased=False).item()) / mean_norm_avg
                )

            # Ordinal monotonicity along empirical top-minus-bottom axis (mu_top - mu_bot)
            delta_mu = means[-1] - means[0]
            delta_norm = torch.linalg.vector_norm(delta_mu)
            if float(delta_norm.item()) <= 1e-12:
                ord_rho = 0.0
            else:
                u_ord = delta_mu / delta_norm
                proj_scores = centered_means @ u_ord
                q_grid = torch.arange(q_eff, device=device, dtype=torch.float64)
                ord_rho = _spearman_corr_1d(q_grid, proj_scores)

            # NC3: Head self-duality cosine cos(w_head, mu_top - mu_bot)
            nc3_cos: float | None = None
            if w_head is not None:
                wf = w_head.detach().to(device=device, dtype=torch.float64)
                if wf.ndim == 2:
                    wf = wf[0]
                wf = wf.reshape(-1)
                if wf.numel() == d:
                    w_norm = torch.linalg.vector_norm(wf)
                    if float(w_norm.item()) > 1e-12 and float(delta_norm.item()) > 1e-12:
                        val_c = float((torch.dot(wf, delta_mu) / (w_norm * delta_norm)).item())
                        nc3_cos = max(-1.0, min(1.0, val_c))
                    else:
                        nc3_cos = 0.0

            norms_list = [float(x) for x in mean_norms.cpu().tolist()]

        return NeuralCollapseResult(
            nc1_within_between_ratio=nc1,
            nc1_trace_ratio=nc1_tr,
            nc2_etf_simplex_deviation=nc2_etf,
            nc2_equinorm_cv=equinorm_cv,
            ordinal_monotonicity_rho=ord_rho,
            nc3_head_duality_cosine=nc3_cos,
            n_quantiles_used=q_eff,
            decile_mean_norms=norms_list,
            nc1_active_rank=active_rank,
        )

    @classmethod
    def analyze_loop_cka(
        cls,
        loop_latents: Sequence[torch.Tensor],
        *,
        stagnation_threshold: float = 0.995,
        debiased: bool = False,
    ) -> LoopCKAResult:
        """Compare representations across loop steps `k=0..K-1` via Linear CKA."""
        k = len(loop_latents)
        if k == 0:
            return LoopCKAResult(
                n_steps=0,
                adjacent_cka=[],
                first_to_last_cka=1.0,
                mean_adjacent_cka=1.0,
                min_adjacent_cka=1.0,
                stagnant_loop_detected=False,
                stagnation_threshold=stagnation_threshold,
                cka_matrix=[],
            )
        if k == 1:
            return LoopCKAResult(
                n_steps=1,
                adjacent_cka=[],
                first_to_last_cka=1.0,
                mean_adjacent_cka=1.0,
                min_adjacent_cka=1.0,
                stagnant_loop_detected=False,
                stagnation_threshold=stagnation_threshold,
                cka_matrix=[[1.0]],
            )

        mat: list[list[float]] = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                val = compute_linear_cka(loop_latents[i], loop_latents[j], debiased=debiased)
                mat[i][j] = val
                mat[j][i] = val

        adj = [mat[i][i + 1] for i in range(k - 1)]
        mean_adj = sum(adj) / float(len(adj))
        min_adj = min(adj)
        stagnant = any(v > stagnation_threshold for v in adj)
        return LoopCKAResult(
            n_steps=k,
            adjacent_cka=adj,
            first_to_last_cka=mat[0][k - 1],
            mean_adjacent_cka=mean_adj,
            min_adjacent_cka=min_adj,
            stagnant_loop_detected=stagnant,
            stagnation_threshold=stagnation_threshold,
            cka_matrix=mat,
        )


# ---------------------------------------------------------------------------
# Pillar 3: Information Bottleneck Trajectory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformationBottleneckResult:
    """Compression `I_Gauss(X;Z)` vs OOS Linear/Nonlinear Sufficiency `I_probe(Z;Y)`."""

    i_xz_gauss: float
    i_xz_per_dim: float
    input_latent_cka: float
    linear_probe_r2: float
    linear_probe_rank_ic: float
    nonlinear_probe_r2: float
    nonlinear_probe_rank_ic: float
    nonlinear_linear_gap_r2: float
    nonlinear_linear_gap_ic: float
    i_zy_linear: float
    i_zy_nonlinear: float
    compression_to_sufficiency_ratio: float
    ib_phase: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InformationBottleneckProbe:
    """Tracks Compression `I_Gauss(X; Z)` vs OOS Sufficiency `I_probe(Z; Y)` across epochs.

    Uses zero-cancellation canonical correlations for `I_Gauss(X; Z)` and strict OOS
    GCV-tuned Ridge (linear) + RBF Kernel Ridge (nonlinear) probes with zero leakage.
    """

    def __init__(
        self,
        *,
        ridge_mi: float = 1e-2,
        train_frac: float = 0.7,
        max_krr_samples: int = 512,
        ridge_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        seed: int = 42,
    ) -> None:
        if not (0.1 <= train_frac <= 0.95):
            raise ValueError(f"train_frac must be in [0.1, 0.95]; got {train_frac}")
        self.ridge_mi = float(ridge_mi)
        self.train_frac = float(train_frac)
        self.max_krr_samples = int(max_krr_samples)
        self.ridge_grid = tuple(float(a) for a in ridge_grid)
        self.seed = int(seed)

    def compute_gaussian_mi(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        *,
        ridge: float | None = None,
    ) -> float:
        """Compute exact regularized Gaussian MI `-0.5 * sum(log1p(-rho_i^2))` >= 0."""
        eps = float(self.ridge_mi if ridge is None else ridge)
        x2 = x.reshape(-1, x.shape[-1])
        z2 = z.reshape(-1, z.shape[-1])
        if x2.shape[0] != z2.shape[0]:
            raise ValueError(f"Sample mismatch: X={x2.shape[0]}, Z={z2.shape[0]}")

        device = x2.device
        with _autocast_disabled(device), torch.no_grad():
            xf = x2.to(device=device, dtype=torch.float64)
            zf = z2.to(device=device, dtype=torch.float64)
            valid = torch.isfinite(xf).all(dim=-1) & torch.isfinite(zf).all(dim=-1)
            xf = xf[valid]
            zf = zf[valid]
            n = int(xf.shape[0])
            dx, dz = int(xf.shape[1]), int(zf.shape[1])
            if n < 3 or dx == 0 or dz == 0:
                return 0.0

            xc = xf - xf.mean(dim=0, keepdim=True)
            zc = zf - zf.mean(dim=0, keepdim=True)

            # Isotropic RMS normalization keeps dead dims at 0 while staying scale-invariant
            sx = torch.sqrt(xc.pow(2).mean())
            sz = torch.sqrt(zc.pow(2).mean())
            if float(sx.item()) <= 1e-12 or float(sz.item()) <= 1e-12:
                return 0.0
            xn = xc / sx
            zn = zc / sz

            inv_n = 1.0 / float(n)
            cxx = inv_n * (xn.T @ xn) + eps * torch.eye(dx, device=device, dtype=torch.float64)
            czz = inv_n * (zn.T @ zn) + eps * torch.eye(dz, device=device, dtype=torch.float64)
            cxz = inv_n * (xn.T @ zn)

            ex, ux = torch.linalg.eigh(cxx)
            ez, uz = torch.linalg.eigh(czz)
            inv_sqrt_x = (ux * ex.clamp_min(1e-12).rsqrt().unsqueeze(0)) @ ux.T
            inv_sqrt_z = (uz * ez.clamp_min(1e-12).rsqrt().unsqueeze(0)) @ uz.T

            cca_mat = inv_sqrt_x @ cxz @ inv_sqrt_z
            rhos = torch.linalg.svdvals(cca_mat).clamp(0.0, 1.0 - 1e-7)
            mi = -0.5 * torch.log1p(-rhos.pow(2)).sum()
            val = float(mi.item())
            return max(0.0, val) if math.isfinite(val) else 0.0

    @staticmethod
    def _oos_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        yt = y_true.reshape(-1).to(dtype=torch.float64)
        yp = y_pred.reshape(-1).to(device=yt.device, dtype=torch.float64)
        ss_tot = (yt - yt.mean()).pow(2).sum()
        if float(ss_tot.item()) <= 1e-15:
            return 0.0
        ss_res = (yt - yp).pow(2).sum()
        val = float((1.0 - ss_res / ss_tot).item())
        return val if math.isfinite(val) else 0.0

    def _fit_eval_linear_ridge_oos(
        self,
        z_tr: torch.Tensor,
        y_tr: torch.Tensor,
        z_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> tuple[float, float]:
        """SVD-GCV Ridge fit strictly on train split, evaluated strictly OOS on val split."""
        n_tr = int(z_tr.shape[0])
        if n_tr < 3 or int(z_val.shape[0]) < 3:
            return 0.0, 0.0

        mu_z = z_tr.mean(dim=0, keepdim=True)
        zc_tr = z_tr - mu_z
        std_z = torch.sqrt(zc_tr.pow(2).mean(dim=0, keepdim=True))
        active = std_z > 1e-8
        safe_std = torch.where(active, std_z, torch.ones_like(std_z))
        zs_tr = torch.where(active, zc_tr / safe_std, torch.zeros_like(zc_tr))
        zs_val = torch.where(active, (z_val - mu_z) / safe_std, torch.zeros_like(z_val))

        mu_y = y_tr.mean()
        yc_tr = y_tr - mu_y

        u, s, vh = torch.linalg.svd(zs_tr, full_matrices=False)
        s2 = s.pow(2)
        uty = u.T @ yc_tr
        y_norm2 = yc_tr.pow(2).sum()

        best_gcv = float("inf")
        best_alpha = self.ridge_grid[len(self.ridge_grid) // 2]
        for alpha_rel in self.ridge_grid:
            alpha = max(alpha_rel * float(n_tr) / max(float(zs_tr.shape[1]), 1.0), 1e-8)
            filt = s2 / (s2 + alpha)
            # Residual norm: ||y - U diag(f) U^T y||^2 = ||y||^2 - sum((2f - f^2) (u^T y)^2)
            rss = (y_norm2 - ((2.0 * filt - filt.pow(2)) * uty.pow(2)).sum()).clamp_min(0.0)
            df = filt.sum()
            denom = max(1.0 - float(df.item()) / float(n_tr), 1e-3)
            gcv = float(rss.item()) / (float(n_tr) * denom * denom)
            if math.isfinite(gcv) and gcv < best_gcv:
                best_gcv = gcv
                best_alpha = alpha

        w = vh.T @ ((s / (s2 + best_alpha)) * uty)
        y_pred = mu_y + zs_val @ w
        return self._oos_r2(y_val, y_pred), _spearman_corr_1d(y_val, y_pred)

    def _fit_eval_rbf_krr_oos(
        self,
        z_tr: torch.Tensor,
        y_tr: torch.Tensor,
        z_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> tuple[float, float]:
        """Exact closed-form GCV-tuned RBF Kernel Ridge probe fit on train, evaluated OOS."""
        n_tr = int(z_tr.shape[0])
        if n_tr < 4 or int(z_val.shape[0]) < 3:
            return 0.0, 0.0

        # Deterministic subsample when training set is very large (keeps KRR fast on CPU/GPU)
        if n_tr > self.max_krr_samples:
            idx = torch.linspace(0, n_tr - 1, steps=self.max_krr_samples, device=z_tr.device).long()
            z_tr = z_tr[idx]
            y_tr = y_tr[idx]
            n_tr = int(z_tr.shape[0])

        mu_z = z_tr.mean(dim=0, keepdim=True)
        zc_tr = z_tr - mu_z
        std_z = torch.sqrt(zc_tr.pow(2).mean(dim=0, keepdim=True))
        active = std_z > 1e-8
        safe_std = torch.where(active, std_z, torch.ones_like(std_z))
        zs_tr = torch.where(active, zc_tr / safe_std, torch.zeros_like(zc_tr))
        zs_val = torch.where(active, (z_val - mu_z) / safe_std, torch.zeros_like(z_val))

        # Stable squared pairwise distances via torch.cdist (zero catastrophic cancellation)
        d2_tr = torch.cdist(zs_tr, zs_tr).pow(2)
        pos_d2 = d2_tr[d2_tr > 1e-12]
        if pos_d2.numel() == 0:
            return 0.0, 0.0
        bw_sq = float(torch.median(pos_d2).item())
        bw_sq = max(bw_sq, 1e-4)

        k_tr = torch.exp(-d2_tr / (2.0 * bw_sq))
        d2_val = torch.cdist(zs_val, zs_tr).pow(2)
        k_val = torch.exp(-d2_val / (2.0 * bw_sq))

        mu_y = y_tr.mean()
        yc_tr = y_tr - mu_y
        evals, evecs = torch.linalg.eigh(k_tr)
        evals = evals.clamp_min(0.0)
        uty = evecs.T @ yc_tr

        best_gcv = float("inf")
        best_lam = 1e-1
        for lam in self.ridge_grid:
            filt = evals / (evals + lam)
            rss = (((1.0 - filt) * uty).pow(2)).sum()
            df = filt.sum()
            denom = max(1.0 - float(df.item()) / float(n_tr), 1e-3)
            gcv = float(rss.item()) / (float(n_tr) * denom * denom)
            if math.isfinite(gcv) and gcv < best_gcv:
                best_gcv = gcv
                best_lam = lam

        dual_alpha = evecs @ (uty / (evals + best_lam))
        y_pred = mu_y + k_val @ dual_alpha
        return self._oos_r2(y_val, y_pred), _spearman_corr_1d(y_val, y_pred)

    @staticmethod
    def classify_trajectory(
        history: Sequence[InformationBottleneckResult | Mapping[str, Any]],
    ) -> str:
        """Classify epoch trajectory into `fitting | compression | memorization | collapse`."""
        if len(history) < 2:
            return "insufficient_history"

        def _get(
            rec: InformationBottleneckResult | Mapping[str, Any], key: str
        ) -> float | None:
            raw = (
                getattr(rec, key, None)
                if isinstance(rec, InformationBottleneckResult)
                else rec.get(key)
            )
            if raw is None:
                return None
            val = float(raw)
            return val if math.isfinite(val) else None

        prev = history[-2]
        curr = history[-1]
        ixz_prev, ixz_curr = _get(prev, "i_xz_gauss"), _get(curr, "i_xz_gauss")
        izy_prev, izy_curr = _get(prev, "i_zy_linear"), _get(curr, "i_zy_linear")
        vl_prev, vl_curr = _get(prev, "val_loss"), _get(curr, "val_loss")

        if ixz_prev is None or ixz_curr is None or izy_prev is None or izy_curr is None:
            return "insufficient_history"

        d_ixz = ixz_curr - ixz_prev
        d_izy = izy_curr - izy_prev
        val_worsening = (
            vl_prev is not None and vl_curr is not None and (vl_curr - vl_prev) > 1e-4
        )

        if ixz_curr < 1e-3 and izy_curr < 1e-3:
            return "collapse"
        if (d_izy < -0.02 and d_ixz >= -0.05) or (val_worsening and d_ixz > 0.05 and d_izy <= 0.0):
            return "memorization"
        if d_ixz < -0.05 and d_izy >= -0.02 and not val_worsening:
            return "compression"
        if d_ixz >= 0.0 and d_izy >= 0.0:
            return "fitting"
        return "compression" if d_ixz < 0.0 else "memorization"

    def probe(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        *,
        x_val: torch.Tensor | None = None,
        z_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        history: Sequence[InformationBottleneckResult | Mapping[str, Any]] | None = None,
    ) -> InformationBottleneckResult:
        """Compute full IB compression & strict OOS linear/nonlinear sufficiency diagnostics."""
        x2 = x.reshape(-1, x.shape[-1])
        z2 = z.reshape(-1, z.shape[-1])
        y1 = y[:, 0] if y.ndim == 2 else y.reshape(-1)
        device = z2.device

        with _autocast_disabled(device), torch.no_grad():
            xf = x2.to(device=device, dtype=torch.float64)
            zf = z2.to(device=device, dtype=torch.float64)
            yf = y1.to(device=device, dtype=torch.float64)
            valid = (
                torch.isfinite(xf).all(dim=-1)
                & torch.isfinite(zf).all(dim=-1)
                & torch.isfinite(yf)
            )
            xf, zf, yf = xf[valid], zf[valid], yf[valid]

            if x_val is not None and z_val is not None and y_val is not None:
                xv = x_val.reshape(-1, x_val.shape[-1]).to(device=device, dtype=torch.float64)
                zv = z_val.reshape(-1, z_val.shape[-1]).to(device=device, dtype=torch.float64)
                yv_raw = y_val[:, 0] if y_val.ndim == 2 else y_val.reshape(-1)
                yv = yv_raw.to(device=device, dtype=torch.float64)
                v_mask = (
                    torch.isfinite(xv).all(dim=-1)
                    & torch.isfinite(zv).all(dim=-1)
                    & torch.isfinite(yv)
                )
                xv, zv, yv = xv[v_mask], zv[v_mask], yv[v_mask]
                x_mi, z_mi = xv if xv.shape[0] >= 3 else xf, zv if zv.shape[0] >= 3 else zf
                z_tr, y_tr, z_ev, y_ev = zf, yf, zv, yv
            else:
                n_all = int(zf.shape[0])
                n_tr = max(3, min(n_all - 3, int(round(n_all * self.train_frac))))
                x_mi, z_mi = xf, zf
                z_tr, y_tr = zf[:n_tr], yf[:n_tr]
                z_ev, y_ev = zf[n_tr:], yf[n_tr:]

            i_xz = self.compute_gaussian_mi(x_mi, z_mi)
            min_dim = max(1, min(int(x_mi.shape[-1]), int(z_mi.shape[-1])))
            i_xz_per_dim = i_xz / float(min_dim)
            cka_xz = compute_linear_cka(x_mi, z_mi)

            lin_r2, lin_ic = self._fit_eval_linear_ridge_oos(z_tr, y_tr, z_ev, y_ev)
            nl_r2, nl_ic = self._fit_eval_rbf_krr_oos(z_tr, y_tr, z_ev, y_ev)

        gap_r2 = max(0.0, nl_r2 - lin_r2)
        gap_ic = nl_ic - lin_ic
        i_zy_lin = -0.5 * math.log(max(1.0 - max(0.0, min(lin_r2, 1.0 - 1e-7)), 1e-7))
        i_zy_nl = -0.5 * math.log(max(1.0 - max(0.0, min(nl_r2, 1.0 - 1e-7)), 1e-7))
        comp_ratio = i_xz / max(i_zy_lin, 1e-6)

        prelim = InformationBottleneckResult(
            i_xz_gauss=i_xz,
            i_xz_per_dim=i_xz_per_dim,
            input_latent_cka=cka_xz,
            linear_probe_r2=lin_r2,
            linear_probe_rank_ic=lin_ic,
            nonlinear_probe_r2=nl_r2,
            nonlinear_probe_rank_ic=nl_ic,
            nonlinear_linear_gap_r2=gap_r2,
            nonlinear_linear_gap_ic=gap_ic,
            i_zy_linear=i_zy_lin,
            i_zy_nonlinear=i_zy_nl,
            compression_to_sufficiency_ratio=comp_ratio,
            ib_phase="single_snapshot",
        )
        if history:
            phase = self.classify_trajectory([*history, prelim])
            return InformationBottleneckResult(**{**prelim.to_dict(), "ib_phase": phase})
        return prelim


# ---------------------------------------------------------------------------
# Pillar 4: Gradient Dynamics Tracker, Health Diagnosis & Trace Persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochDynamicsSnapshot:
    """Epoch-level training dynamics, GSNR, update-to-weight ratio & convergence slope."""

    epoch: int
    train_loss: float
    val_loss: float | None
    generalization_gap: float | None
    relative_generalization_gap: float | None
    grad_norm: float
    grad_variance: float
    gsnr: float
    grad_cosine_stability: float
    weight_norm: float
    update_to_weight_ratio: float
    convergence_rate: float
    n_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GradientDynamicsTracker:
    """Zero-host-sync step gradient tracker with online Welford GSNR & Karpathy ratios.

    Calling `record_step(...)` and `record_post_step(...)` inside a training loop performs
    pure device-side tensor arithmetic (0 `.item()` / `.cpu()` syncs). Host materialization
    happens strictly once per epoch inside `finalize_epoch(...)`.
    """

    def __init__(
        self,
        *,
        convergence_window: int = 5,
        vec_dtype: torch.dtype | None = None,
    ) -> None:
        self.convergence_window = max(2, int(convergence_window))
        self.vec_dtype = vec_dtype
        self._history: list[EpochDynamicsSnapshot] = []
        self._reset_epoch_accumulators()

    def _resolve_vec_dtype(self, device: torch.device) -> torch.dtype:
        if self.vec_dtype is not None:
            return self.vec_dtype
        # On CUDA Blackwell sm_120 keep parameter-sized running vectors in FP32 (64x faster,
        # 2x less VRAM); scalar Welford/norm/cosine accumulators stay in FP64!
        return torch.float32 if device.type == "cuda" else torch.float64

    def _reset_epoch_accumulators(self) -> None:
        self._step_count: int = 0
        self._mean_g: torch.Tensor | None = None
        self._prev_g: torch.Tensor | None = None
        self._m2_sum: torch.Tensor | None = None
        self._grad_norm_sum: torch.Tensor | None = None
        self._cos_sum: torch.Tensor | None = None
        self._cos_count: int = 0
        self._pre_step_w: torch.Tensor | None = None
        self._upd_ratio_sum: torch.Tensor | None = None
        self._upd_count: int = 0
        self._weight_norm_last: torch.Tensor | None = None
        self._nonfinite_grad_count: torch.Tensor | None = None

    @property
    def history(self) -> list[EpochDynamicsSnapshot]:
        return list(self._history)

    @torch.no_grad()
    def record_step(
        self,
        model_or_params: nn.Module | Sequence[nn.Parameter],
        *,
        grads: Sequence[torch.Tensor | None] | None = None,
        track_weight_snapshot: bool = True,
    ) -> None:
        """Record one pre-step gradient vector on device via Welford online update (0 syncs)."""
        params = _resolve_trainable_params(model_or_params)
        if not params:
            return
        device = params[0].device
        grad_seq: Sequence[torch.Tensor | None] = (
            grads if grads is not None else [p.grad for p in params]
        )

        vec_dtype = self._resolve_vec_dtype(device)
        with _autocast_disabled(device):
            g_flat = _flatten_tensors(grad_seq, params, device, vec_dtype)
            is_fin = torch.isfinite(g_flat).all()
            g_safe = torch.where(torch.isfinite(g_flat), g_flat, torch.zeros_like(g_flat))
            g_norm = torch.linalg.vector_norm(g_safe).to(dtype=torch.float64)

            if self._step_count == 0:
                self._mean_g = g_safe.clone()
                self._prev_g = g_safe.clone()
                self._m2_sum = torch.zeros((), device=device, dtype=torch.float64)
                self._grad_norm_sum = g_norm.clone()
                self._cos_sum = torch.zeros((), device=device, dtype=torch.float64)
                self._upd_ratio_sum = torch.zeros((), device=device, dtype=torch.float64)
                self._nonfinite_grad_count = (~is_fin).to(dtype=torch.float64)
                self._step_count = 1
            else:
                assert self._mean_g is not None
                assert self._prev_g is not None
                assert self._m2_sum is not None
                assert self._grad_norm_sum is not None
                assert self._cos_sum is not None
                assert self._nonfinite_grad_count is not None
                self._step_count += 1
                delta = g_safe - self._mean_g
                self._mean_g.add_(delta, alpha=1.0 / float(self._step_count))
                delta2 = g_safe - self._mean_g
                self._m2_sum.add_(torch.dot(delta, delta2).to(dtype=torch.float64))
                self._grad_norm_sum.add_(g_norm)
                self._nonfinite_grad_count.add_((~is_fin).to(dtype=torch.float64))

                prev_norm = torch.linalg.vector_norm(self._prev_g).to(dtype=torch.float64)
                denom = (g_norm * prev_norm).clamp_min(1e-12)
                valid_pair = (g_norm > 1e-12) & (prev_norm > 1e-12)
                dot_prev = torch.dot(g_safe, self._prev_g).to(dtype=torch.float64)
                cos_val = torch.where(
                    valid_pair,
                    (dot_prev / denom).clamp(-1.0, 1.0),
                    torch.zeros((), device=device, dtype=torch.float64),
                )
                self._cos_sum.add_(cos_val)
                self._cos_count += 1
                self._prev_g.copy_(g_safe)

            if track_weight_snapshot:
                w_flat = _flatten_tensors(
                    [p.detach() for p in params], params, device, vec_dtype
                )
                self._pre_step_w = w_flat
                self._weight_norm_last = torch.linalg.vector_norm(w_flat).to(
                    dtype=torch.float64
                )

    @torch.no_grad()
    def record_post_step(
        self,
        model_or_params: nn.Module | Sequence[nn.Parameter],
    ) -> None:
        """Record post-`optimizer.step()` weights to accumulate exact `||dW|| / ||W||` (0 syncs)."""
        params = _resolve_trainable_params(model_or_params)
        if not params:
            return
        device = params[0].device
        vec_dtype = self._resolve_vec_dtype(device)
        with _autocast_disabled(device):
            w_after = _flatten_tensors(
                [p.detach() for p in params], params, device, vec_dtype
            )
            w_norm = torch.linalg.vector_norm(w_after).to(dtype=torch.float64)
            self._weight_norm_last = w_norm
            if self._pre_step_w is not None and self._pre_step_w.shape == w_after.shape:
                dw_norm = torch.linalg.vector_norm(w_after - self._pre_step_w).to(
                    dtype=torch.float64
                )
                ratio = dw_norm / w_norm.clamp_min(1e-12)
                if self._upd_ratio_sum is None:
                    self._upd_ratio_sum = ratio.clone()
                else:
                    self._upd_ratio_sum.add_(ratio)
                self._upd_count += 1

    def _compute_convergence_rate(self, current_epoch: int, current_train_loss: float) -> float:
        """Log-loss OLS slope when all window losses > 0; normalized linear slope if any <= 0."""
        raw_pts: list[tuple[float, float]] = [
            (float(s.epoch), float(s.train_loss))
            for s in self._history[-(self.convergence_window - 1) :]
            if math.isfinite(s.train_loss)
        ]
        if math.isfinite(current_train_loss):
            raw_pts.append((float(current_epoch), float(current_train_loss)))
        if len(raw_pts) < 2:
            return 0.0
        xs = np.array([p[0] for p in raw_pts], dtype=np.float64)
        losses = np.array([p[1] for p in raw_pts], dtype=np.float64)
        xc = xs - xs.mean()
        denom = float(np.dot(xc, xc))
        if denom <= 1e-12:
            return 0.0
        if bool(np.all(losses > 0.0)):
            ys = np.log(losses)
            return float(np.dot(xc, ys - ys.mean()) / denom)
        scale = max(abs(float(losses[0])), 1.0)
        return float(np.dot(xc, losses - losses.mean()) / denom) / scale

    def finalize_epoch(
        self,
        *,
        epoch: int,
        train_loss: float | torch.Tensor,
        val_loss: float | torch.Tensor | None = None,
        model_or_params: nn.Module | Sequence[nn.Parameter] | None = None,
    ) -> EpochDynamicsSnapshot:
        """Materialize epoch metrics with a single batched host transfer and reset accumulators."""
        tr_l = (
            float(train_loss.detach().to(dtype=torch.float64).item())
            if isinstance(train_loss, torch.Tensor)
            else float(train_loss)
        )
        va_l = (
            None
            if val_loss is None
            else (
                float(val_loss.detach().to(dtype=torch.float64).item())
                if isinstance(val_loss, torch.Tensor)
                else float(val_loss)
            )
        )
        gen_gap = None if va_l is None else (va_l - tr_l)
        rel_gap = None if gen_gap is None else (gen_gap / max(abs(tr_l), 1e-8))

        if self._step_count > 0 and self._mean_g is not None and self._m2_sum is not None:
            assert self._grad_norm_sum is not None
            assert self._cos_sum is not None
            assert self._nonfinite_grad_count is not None
            device = self._mean_g.device
            mean_norm_sq = self._mean_g.to(dtype=torch.float64).pow(2).sum()
            grad_var_t = (self._m2_sum / float(self._step_count)).clamp_min(0.0)
            gsnr_t = mean_norm_sq / grad_var_t.clamp_min(1e-12)
            grad_norm_t = self._grad_norm_sum / float(self._step_count)
            cos_t = (
                self._cos_sum / float(self._cos_count)
                if self._cos_count > 0
                else torch.ones((), device=device, dtype=torch.float64)
            )
            upd_t = (
                self._upd_ratio_sum / float(self._upd_count)
                if (self._upd_count > 0 and self._upd_ratio_sum is not None)
                else torch.zeros((), device=device, dtype=torch.float64)
            )
            if self._weight_norm_last is not None:
                w_norm_t = self._weight_norm_last
            elif model_or_params is not None:
                p_list = _resolve_trainable_params(model_or_params)
                w_norm_t = torch.linalg.vector_norm(
                    _flatten_tensors([p.detach() for p in p_list], p_list, device, torch.float64)
                )
            else:
                w_norm_t = torch.zeros((), device=device, dtype=torch.float64)

            packed = torch.stack(
                [
                    grad_norm_t,
                    grad_var_t,
                    gsnr_t,
                    cos_t,
                    w_norm_t,
                    upd_t,
                    self._nonfinite_grad_count,
                ]
            ).cpu()
            grad_norm = float(packed[0].item())
            grad_var = float(packed[1].item())
            gsnr = float(packed[2].item())
            cos_stab = float(packed[3].item())
            w_norm = float(packed[4].item())
            upd_ratio = float(packed[5].item())
            if float(packed[6].item()) > 0.0:
                grad_norm = float("inf")
        else:
            grad_norm = 0.0
            grad_var = 0.0
            gsnr = 0.0
            cos_stab = 0.0
            upd_ratio = 0.0
            if model_or_params is not None:
                p_list = _resolve_trainable_params(model_or_params)
                if p_list:
                    w_norm = float(
                        torch.linalg.vector_norm(
                            _flatten_tensors(
                                [p.detach() for p in p_list],
                                p_list,
                                p_list[0].device,
                                torch.float64,
                            )
                        ).item()
                    )
                else:
                    w_norm = 0.0
            else:
                w_norm = 0.0

        conv_rate = self._compute_convergence_rate(epoch, tr_l)
        snap = EpochDynamicsSnapshot(
            epoch=int(epoch),
            train_loss=tr_l,
            val_loss=va_l,
            generalization_gap=gen_gap,
            relative_generalization_gap=rel_gap,
            grad_norm=grad_norm,
            grad_variance=grad_var,
            gsnr=gsnr,
            grad_cosine_stability=cos_stab,
            weight_norm=w_norm,
            update_to_weight_ratio=upd_ratio,
            convergence_rate=conv_rate,
            n_steps=int(self._step_count),
        )
        self._history.append(snap)
        self._reset_epoch_accumulators()
        return snap


@dataclass(frozen=True)
class HealthDiagnosisThresholds:
    """Configurable thresholds for `diagnose_training_health`."""

    max_grad_norm: float = 1e3
    max_update_to_weight_ratio: float = 0.10
    min_erank_ratio: float = 0.10
    min_erank_abs: float = 2.0
    max_dead_unit_fraction: float = 0.50
    max_generalization_gap: float = 0.15
    max_rel_generalization_gap: float = 0.35
    max_sam_sharpness: float = 0.50
    max_lambda_max: float = 1e3
    max_effective_condition_number: float = 50.0
    underfit_min_epoch: int = 3
    underfit_max_abs_conv_rate: float = 1e-3
    underfit_max_gsnr: float = 0.10
    underfit_max_probe_r2: float = 0.02
    underfit_min_probe_ic: float = 0.015


def diagnose_training_health(
    *,
    hessian: HessianSpectrumResult | None = None,
    representation: RepresentationHealthResult | None = None,
    neural_collapse: NeuralCollapseResult | None = None,
    loop_cka: LoopCKAResult | None = None,
    information_bottleneck: InformationBottleneckResult | None = None,
    dynamics: EpochDynamicsSnapshot | None = None,
    thresholds: HealthDiagnosisThresholds | None = None,
) -> tuple[str, list[str], dict[str, Any], list[str]]:
    """Emit a canonical diagnosis label, secondary flags, quantitative evidence & remedies."""
    thr = thresholds or HealthDiagnosisThresholds()
    flags: list[str] = []
    evidence: dict[str, Any] = {}
    recs: list[str] = []

    # 1) Gradient instability (highest operational priority)
    if dynamics is not None:
        nonfin = not math.isfinite(dynamics.train_loss) or not math.isfinite(dynamics.grad_norm)
        exploding = (
            dynamics.grad_norm > thr.max_grad_norm
            or dynamics.update_to_weight_ratio > thr.max_update_to_weight_ratio
        )
        diverging = (
            dynamics.convergence_rate > 0.05
            and dynamics.grad_cosine_stability < -0.1
            and dynamics.gsnr < 0.05
        )
        if nonfin or exploding or diverging:
            flags.append("gradient_instability")
            evidence["grad_norm"] = dynamics.grad_norm
            evidence["update_to_weight_ratio"] = dynamics.update_to_weight_ratio
            evidence["convergence_rate"] = dynamics.convergence_rate
            recs.append(
                "Lower learning rate, tighten gradient clipping, and extend warmup."
            )

    # 2) Representation collapse (require n_samples >= 2 so degenerate N<=1 never false-fires)
    if representation is not None and representation.dim > 0 and representation.n_samples >= 2:
        low_erank = (
            representation.effective_rank_ratio < thr.min_erank_ratio
            or (representation.dim >= 8 and representation.effective_rank < thr.min_erank_abs)
        )
        high_dead = representation.dead_unit_fraction > thr.max_dead_unit_fraction
        if low_erank or high_dead:
            flags.append("representation_collapse")
            evidence["effective_rank"] = representation.effective_rank
            evidence["effective_rank_ratio"] = representation.effective_rank_ratio
            evidence["dead_unit_fraction"] = representation.dead_unit_fraction
            recs.append(
                "Add VICReg/RankMe covariance regularization, check LayerNorm/init scale, "
                "or reduce weight decay."
            )

    # 3) Memorizing / overfitting
    overfit_triggered = False
    if dynamics is not None and dynamics.generalization_gap is not None:
        rel_gap = dynamics.relative_generalization_gap or 0.0
        if (
            dynamics.generalization_gap > thr.max_generalization_gap
            and rel_gap > thr.max_rel_generalization_gap
        ):
            overfit_triggered = True
            evidence["generalization_gap"] = dynamics.generalization_gap
            evidence["relative_generalization_gap"] = rel_gap
    if information_bottleneck is not None and information_bottleneck.ib_phase == "memorization":
        overfit_triggered = True
        evidence["ib_phase"] = information_bottleneck.ib_phase
        evidence["i_xz_gauss"] = information_bottleneck.i_xz_gauss
        evidence["i_zy_linear"] = information_bottleneck.i_zy_linear
    if overfit_triggered:
        flags.append("memorizing_overfit")
        recs.append(
            "Increase decoupled weight decay / structured modality dropout, enable EMA/Model Soup, "
            "or early-stop before memorization."
        )

    # 4) Sharp minimum
    if hessian is not None:
        sharp_sam = math.isfinite(hessian.sam_sharpness) and (
            hessian.sam_sharpness > thr.max_sam_sharpness
        )
        sharp_eig = math.isfinite(hessian.lambda_max) and (hessian.lambda_max > thr.max_lambda_max)
        bad_cond = math.isfinite(hessian.effective_condition_number) and (
            hessian.effective_condition_number > thr.max_effective_condition_number
            and hessian.lambda_max > 10.0
        )
        if sharp_sam or sharp_eig or bad_cond:
            flags.append("sharp_minimum")
            evidence["lambda_max"] = hessian.lambda_max
            evidence["sam_sharpness"] = hessian.sam_sharpness
            evidence["effective_condition_number"] = hessian.effective_condition_number
            recs.append(
                "Switch to ScheduleFreeAdamW or SAM (rho=0.02..0.05) and check cosine/SWA tail."
            )

    # 5) Underfitting (calibrated for low-SNR finance: IC >= 0.015 is healthy even if R^2 < 0.02)
    if dynamics is not None and "memorizing_overfit" not in flags:
        stalled = (
            dynamics.epoch >= thr.underfit_min_epoch
            and abs(dynamics.convergence_rate) <= thr.underfit_max_abs_conv_rate
            and (dynamics.generalization_gap is None or dynamics.generalization_gap <= 0.05)
        )
        if information_bottleneck is not None:
            weak_signal = (
                abs(information_bottleneck.linear_probe_rank_ic) < thr.underfit_min_probe_ic
                and information_bottleneck.linear_probe_r2 <= thr.underfit_max_probe_r2
            )
        else:
            weak_signal = dynamics.gsnr <= thr.underfit_max_gsnr
        if stalled and weak_signal:
            flags.append("underfitting")
            evidence["convergence_rate"] = dynamics.convergence_rate
            evidence["gsnr"] = dynamics.gsnr
            if information_bottleneck is not None:
                evidence["linear_probe_r2"] = information_bottleneck.linear_probe_r2
                evidence["linear_probe_rank_ic"] = information_bottleneck.linear_probe_rank_ic
            recs.append(
                "Increase LR or model capacity, relax regularization, or verify target scaling."
            )

    # Supplemental advisory evidence (loop stagnation, nonlinear decoding gap, NC3 misalignment)
    if loop_cka is not None and loop_cka.stagnant_loop_detected:
        evidence["stagnant_loop_mean_cka"] = loop_cka.mean_adjacent_cka
        recs.append("Recursive loop is stagnant (CKA > 0.995): inspect contractive step / FiLM.")
    if (
        information_bottleneck is not None
        and information_bottleneck.nonlinear_linear_gap_r2 > 0.15
    ):
        evidence["nonlinear_linear_gap_r2"] = information_bottleneck.nonlinear_linear_gap_r2
        recs.append(
            "Large nonlinear-vs-linear sufficiency gap: latent carries signal not yet linearized "
            "for the linear readout head."
        )
    if (
        neural_collapse is not None
        and neural_collapse.nc3_head_duality_cosine is not None
        and neural_collapse.nc3_head_duality_cosine < 0.0
    ):
        evidence["nc3_head_duality_cosine"] = neural_collapse.nc3_head_duality_cosine

    primary = flags[0] if flags else "healthy_generalizing"
    if not recs:
        recs.append("Optimization geometry, representation rank, and IB trajectory look healthy.")
    return primary, flags, evidence, recs


@dataclass(frozen=True)
class DeepLearningHealthReport:
    """Unified geometry, representation, IB & training-dynamics report for one checkpoint."""

    family: str
    variant: str
    epoch: int = 0
    step: int = 0
    diagnosis: str = "healthy_generalizing"
    secondary_flags: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    hessian: HessianSpectrumResult | None = None
    representation: RepresentationHealthResult | None = None
    neural_collapse: NeuralCollapseResult | None = None
    loop_cka: LoopCKAResult | None = None
    information_bottleneck: InformationBottleneckResult | None = None
    dynamics: EpochDynamicsSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_sanitize(
            dict(
                family=self.family,
                variant=self.variant,
                epoch=self.epoch,
                step=self.step,
                diagnosis=self.diagnosis,
                secondary_flags=list(self.secondary_flags),
                evidence=dict(self.evidence),
                recommendations=list(self.recommendations),
                hessian=None if self.hessian is None else self.hessian.to_dict(),
                representation=(
                    None if self.representation is None else self.representation.to_dict()
                ),
                neural_collapse=(
                    None if self.neural_collapse is None else self.neural_collapse.to_dict()
                ),
                loop_cka=None if self.loop_cka is None else self.loop_cka.to_dict(),
                information_bottleneck=(
                    None
                    if self.information_bottleneck is None
                    else self.information_bottleneck.to_dict()
                ),
                dynamics=None if self.dynamics is None else self.dynamics.to_dict(),
            )
        )


def write_geometry_trace_gz(
    path: str | Path,
    records: Sequence[DeepLearningHealthReport | Mapping[str, Any]],
    *,
    append: bool = False,
    compresslevel: int = 6,
) -> Path:
    """Write or append compact lossless RFC-1952 `.jsonl.gz` geometry traces deterministically.

    Uses `filename=""` and `mtime=0` so identical inputs produce byte-identical `.jsonl.gz`
    files, atomic `os.replace` on overwrite, and concatenated RFC-1952 members on `append=True`.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            _json_sanitize(rec),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for rec in records
    ]
    payload = "".join(lines).encode("utf-8")

    if append:
        with (
            out_path.open("ab") as raw_fh,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_fh,
                compresslevel=compresslevel,
                mtime=0,
            ) as gz_fh,
        ):
            gz_fh.write(payload)
        return out_path

    tmp = out_path.with_name(
        f".{out_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with (
            tmp.open("wb") as raw_fh,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_fh,
                compresslevel=compresslevel,
                mtime=0,
            ) as gz_fh,
        ):
            gz_fh.write(payload)
        os.replace(tmp, out_path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return out_path


def read_geometry_trace_gz(path: str | Path) -> list[dict[str, Any]]:
    """Read `.jsonl.gz` (including multi-member RFC-1952) or plain `.jsonl` losslessly."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("rb") as raw_fh:
        magic = raw_fh.read(2)
    is_gz = magic == b"\x1f\x8b"
    opener = gzip.open(p, "rt", encoding="utf-8") if is_gz else p.open("r", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    with opener as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSONL in {p} line {line_no}: {exc}") from exc
    return rows


# ---------------------------------------------------------------------------
# Unified Model-Agnostic Orchestrator Across All 4 Families
# ---------------------------------------------------------------------------


def _extract_input_matrix(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """Concatenate available 2D asset feature tensors from a session batch into `[N, D_in]`."""
    n_assets: int | None = None
    cols: list[torch.Tensor] = []
    for key in _INPUT_BATCH_KEYS:
        t = batch.get(key)
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            continue
        if n_assets is None:
            n_assets = int(t.shape[0])
        if int(t.shape[0]) != n_assets:
            continue
        tf = t.detach().to(device=device, dtype=torch.float32)
        cols.append(torch.where(torch.isfinite(tf), tf, torch.zeros_like(tf)))
    if not cols:
        raise ValueError("Batch contains no valid 2D asset feature tensors for IB probing")
    return torch.cat(cols, dim=-1)


def _extract_head_direction(
    model: nn.Module,
    latent_dim: int,
    *,
    horizon_idx: int = 0,
) -> torch.Tensor | None:
    """Extract linear readout weight vector `w_head [D]` across all 4 families & generic models."""
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "alpha_head"):
        alpha_lin = head.alpha_head
        if isinstance(alpha_lin, nn.Linear) and alpha_lin.in_features == latent_dim:
            h_i = min(max(0, int(horizon_idx)), alpha_lin.out_features - 1)
            w = alpha_lin.weight[h_i].detach().clone()
            tabm_k = int(getattr(head, "tabm_k", 1))
            tabm_r = getattr(head, "tabm_r", None)
            tabm_s = getattr(head, "tabm_s", None)
            if tabm_k > 1 and isinstance(tabm_r, torch.Tensor) and isinstance(tabm_s, torch.Tensor):
                # Effective linear weight averaged over TabM BatchEnsemble members
                scale = (
                    tabm_s.detach()[:, h_i : h_i + 1].to(device=w.device, dtype=w.dtype)
                    * tabm_r.detach().to(device=w.device, dtype=w.dtype)
                ).mean(dim=0)
                w = w * scale
            return w

    for mod in reversed(list(model.modules())):
        if isinstance(mod, nn.Linear) and mod.in_features == latent_dim:
            h_i = min(max(0, int(horizon_idx)), mod.out_features - 1)
            return mod.weight[h_i].detach().clone()
    return None


def _invoke_model_forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    state: dict[str, object] | None = None,
    disable_early_exit: bool = False,
) -> Any:
    """Call `model.forward_session` or `model(batch)` cleanly across all 4 families."""
    fn = getattr(model, "forward_session", None)
    if callable(fn):
        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {}
        if "state" in sig.parameters:
            kwargs["state"] = state
        if disable_early_exit and "enable_early_exit" in sig.parameters:
            kwargs["enable_early_exit"] = False
        return fn(batch, **kwargs)
    return model(batch)


def _default_surrogate_loss(
    out: Any,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Clean double-differentiable financial loss (Pearson IC + Huber) for Hessian probing."""
    y_raw = batch.get("target_returns")
    if isinstance(out, Mapping):
        pred = out.get("pred_return")
        if pred is None:
            pred = out.get("alpha")
    else:
        pred = out
    if not isinstance(pred, torch.Tensor):
        raise ValueError("Model output does not expose a tensor prediction for default loss")

    p1 = pred[:, 0] if pred.ndim >= 2 else pred.reshape(-1)
    if not isinstance(y_raw, torch.Tensor):
        return 0.5 * p1.pow(2).mean()

    y1 = (y_raw[:, 0] if y_raw.ndim >= 2 else y_raw.reshape(-1)).to(
        device=p1.device, dtype=p1.dtype
    )
    valid = torch.isfinite(y1)
    y_clean = torch.where(valid, y1, torch.zeros_like(y1))
    p_clean = torch.where(valid, p1, torch.zeros_like(p1))
    cnt = valid.to(dtype=p1.dtype).sum().clamp_min(1.0)
    pc = torch.where(valid, p_clean - p_clean.sum() / cnt, torch.zeros_like(p_clean))
    yc = torch.where(valid, y_clean - y_clean.sum() / cnt, torch.zeros_like(y_clean))
    corr = (pc * yc).sum() / (
        torch.sqrt(pc.pow(2).sum() + 1e-8) * torch.sqrt(yc.pow(2).sum() + 1e-8)
    )
    huber = F.huber_loss(p_clean, y_clean, delta=0.05)
    return (1.0 - corr) + 5.0 * huber


class DeepLearningGeometryProbe:
    """Model-agnostic orchestrator unifying all 4 diagnostic pillars across all 4 families."""

    def __init__(
        self,
        *,
        hessian_cfg: HessianConfig | None = None,
        n_quantiles: int = 10,
        stagnation_threshold: float = 0.995,
        ib_probe: InformationBottleneckProbe | None = None,
        diagnosis_thresholds: HealthDiagnosisThresholds | None = None,
    ) -> None:
        self.hessian_analyzer = HessianSpectrumAnalyzer(hessian_cfg)
        self.n_quantiles = int(n_quantiles)
        self.stagnation_threshold = float(stagnation_threshold)
        self.ib_probe = ib_probe or InformationBottleneckProbe()
        self.diagnosis_thresholds = diagnosis_thresholds or HealthDiagnosisThresholds()
        self._ib_history: list[InformationBottleneckResult] = []

    def extract_latents_and_head_direction(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        state: dict[str, object] | None = None,
        latent_module: nn.Module | None = None,
        horizon_idx: int = 0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None, Any]:
        """Capture final + per-loop latents and linear head vector via temporary pre-hook."""
        target_mod = latent_module or getattr(model, "head", None)
        if target_mod is None:
            linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
            if not linears:
                raise ValueError(
                    "Could not infer latent readout module; pass `latent_module` explicitly."
                )
            target_mod = linears[-1]

        captured: list[torch.Tensor] = []

        def _pre_hook(_mod: nn.Module, args: tuple[Any, ...]) -> None:
            if args and isinstance(args[0], torch.Tensor):
                h_in = args[0].detach()
                # When hooking MultiHorizonGranularHead, apply its pre-linear LayerNorm so
                # NC3 cosine duality matches the exact vector entering `alpha_head`.
                norm_mod = getattr(_mod, "norm", None)
                if isinstance(norm_mod, nn.Module) and hasattr(_mod, "alpha_head"):
                    with torch.no_grad():
                        h_in = norm_mod(h_in).detach()
                captured.append(h_in)

        handle = target_mod.register_forward_pre_hook(_pre_hook)
        try:
            with torch.no_grad():
                out = _invoke_model_forward(model, batch, state=state, disable_early_exit=True)
        finally:
            handle.remove()

        # Prefer post-monotonic/post-halt `out["final_hidden"]` when exposed by the model
        # (e.g., Family 2 `StandaloneHolographicBrainModel`), applying head LayerNorm so NC3
        # aligns with the exact vector multiplied by `alpha_head.weight`.
        raw_final = out.get("final_hidden") if isinstance(out, Mapping) else None
        if isinstance(raw_final, torch.Tensor) and raw_final.ndim == 2:
            final_latent = raw_final.detach()
            norm_mod = getattr(target_mod, "norm", None)
            if isinstance(norm_mod, nn.Module) and hasattr(target_mod, "alpha_head"):
                with torch.no_grad():
                    final_latent = norm_mod(final_latent).detach()
        elif captured:
            final_latent = captured[-1]
        else:
            raise RuntimeError("Latent hook captured no activations during model forward pass")

        loop_latents = captured if captured else [final_latent]
        w_head = _extract_head_direction(
            model, int(final_latent.shape[-1]), horizon_idx=horizon_idx
        )
        return final_latent, loop_latents, w_head, out

    def probe_model(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        loss_closure: Callable[[], torch.Tensor] | None = None,
        val_batch: dict[str, torch.Tensor] | None = None,
        dynamics_snapshot: EpochDynamicsSnapshot | None = None,
        family: str | None = None,
        variant: str | None = None,
        epoch: int = 0,
        step: int = 0,
        latent_module: nn.Module | None = None,
        run_hessian: bool = True,
        eval_mode: bool = True,
    ) -> DeepLearningHealthReport:
        """Run full 4-pillar geometry probe on any model without mutating weights or mode."""
        saved_modes = {mod: mod.training for mod in model.modules()}
        cfg_obj = getattr(model, "config", None)
        resolved_family = str(
            family
            or getattr(cfg_obj, "family", None)
            or (
                "holographic_brain"
                if model.__class__.__name__ == "StandaloneHolographicBrainModel"
                else "custom"
            )
        )
        resolved_variant = str(
            variant
            or getattr(cfg_obj, "variation", None)
            or getattr(cfg_obj, "ablation", None)
            or model.__class__.__name__
        )

        try:
            if eval_mode:
                model.eval()

            # 1) Latent extraction (train + optional validation split)
            z_tr, loop_tr, w_head, _ = self.extract_latents_and_head_direction(
                model, batch, latent_module=latent_module
            )
            device = z_tr.device
            x_tr = _extract_input_matrix(batch, device)
            y_tr = batch.get("target_returns")

            if val_batch is not None:
                z_ev, loop_ev, _, _ = self.extract_latents_and_head_direction(
                    model, val_batch, latent_module=latent_module
                )
                x_ev = _extract_input_matrix(val_batch, device)
                y_ev = val_batch.get("target_returns")
            else:
                z_ev, loop_ev, x_ev, y_ev = z_tr, loop_tr, x_tr, y_tr

            # 2) Representation health, decile Neural Collapse & Loop CKA
            rep_res = RepresentationGeometryAnalyzer.analyze_representation_health(z_ev)
            nc_res = (
                RepresentationGeometryAnalyzer.analyze_neural_collapse(
                    z_ev, y_ev, w_head=w_head, n_quantiles=self.n_quantiles
                )
                if isinstance(y_ev, torch.Tensor)
                else None
            )
            loop_res = (
                RepresentationGeometryAnalyzer.analyze_loop_cka(
                    loop_ev, stagnation_threshold=self.stagnation_threshold
                )
                if len(loop_ev) >= 2
                else None
            )

            # 3) Information Bottleneck compression vs OOS sufficiency
            ib_res: InformationBottleneckResult | None = None
            if isinstance(y_tr, torch.Tensor):
                if val_batch is not None and isinstance(y_ev, torch.Tensor):
                    ib_res = self.ib_probe.probe(
                        x_tr,
                        z_tr,
                        y_tr,
                        x_val=x_ev,
                        z_val=z_ev,
                        y_val=y_ev,
                        history=self._ib_history,
                    )
                else:
                    ib_res = self.ib_probe.probe(
                        x_tr, z_tr, y_tr, history=self._ib_history
                    )
                self._ib_history.append(ib_res)

            # 4) Hessian Spectrum & SAM Neighborhood Sharpness
            hess_res: HessianSpectrumResult | None = None
            if run_hessian:
                eff_closure = loss_closure or (
                    lambda: _default_surrogate_loss(
                        _invoke_model_forward(model, batch, state=None, disable_early_exit=False),
                        batch,
                    )
                )
                hess_res = self.hessian_analyzer.analyze(model, eff_closure)
        finally:
            for mod, mode_flag in saved_modes.items():
                mod.train(mode_flag)

        diagnosis, flags, evidence, recs = diagnose_training_health(
            hessian=hess_res,
            representation=rep_res,
            neural_collapse=nc_res,
            loop_cka=loop_res,
            information_bottleneck=ib_res,
            dynamics=dynamics_snapshot,
            thresholds=self.diagnosis_thresholds,
        )
        return DeepLearningHealthReport(
            family=resolved_family,
            variant=resolved_variant,
            epoch=int(epoch),
            step=int(step),
            diagnosis=diagnosis,
            secondary_flags=flags,
            evidence=evidence,
            recommendations=recs,
            hessian=hess_res,
            representation=rep_res,
            neural_collapse=nc_res,
            loop_cka=loop_res,
            information_bottleneck=ib_res,
            dynamics=dynamics_snapshot,
        )
