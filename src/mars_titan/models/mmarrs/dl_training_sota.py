"""Deep Learning SOTA Training & Architecture Taxonomy Kit (`MARS-TITAN`).

Modular, zero-host-sync, BF16/`torch.compile`-safe, Blackwell (`sm_120`) optimized
building blocks spanning every axis of the Deep Learning taxonomy so ablations run
identically across all 4 independent model families (`titans_original`,
`holographic_brain`, `mmarrs_titan`, `titans_cognitive_core`):

1. **Modern SOTA Layers**:
   - `DynamicTanh` (DyT, Zhu et al. CVPR 2025: `gamma * tanh(alpha * x) + beta`,
     zero reduction overhead, FP32-promoted under BF16).
   - `RMSNormSOTA` (`x * rsqrt(mean(x^2) + eps) * weight`) + `build_norm_layer`.
   - `SwiGLUFFN` (`W_down(silu(W_gate(x)) * W_up(x))` with `2/3` iso-param hidden
     scaling aligned to Tensor Core multiples of 8).
   - `LayerScale` (`gamma_init=1e-2`) and `DeepNorm` residual gain helpers
     (`deepnorm_gains`, `deepnorm_residual`, `DeepNormResidual`).
2. **Maximal Update Parameterization (`muP`) + Coordinate Check (`#164` P0-3)**:
   - `apply_mup_init_and_param_groups`: scales ONLY 2D hidden weight matrices
     (`lr ∝ base_width / width`, fan-in `std ∝ 1 / sqrt(fan_in)`) and linear readout
     heads (`std ∝ 1 / width`, `lr ∝ base_width / width`), preserving input encoders,
     embeddings, norms, biases, routers, and BatchEnsemble adapters at `1.0x`.
   - `run_mup_coordinate_check`: empirically verifies width-invariance of hidden
     update magnitudes `||Delta h||_rms` across `D = (64, 128, 256)`.
3. **Scale-Invariant SOTA Losses & Anti-Collapse Regularizers (`#164` P0-2 & P0-3)**:
   - `tail_listnet_loss`: symmetric long-tail (`+`) & short-tail (`-`) KL divergence
     strictly over tie-averaged 0-sync `gaussian_rank_features(target)` and
     cross-sectionally z-scored `pred` (`tau=0.5`, exact scale invariance, FM-7 safe,
     >70% mass on D10/D1, CN all-tie & NaN safe).
   - `vicreg_covariance_loss`: zero-sync off-diagonal covariance + mild variance floor
     preventing dimensional collapse (`eRank` preservation).
   - `cross_sectional_rank_contrastive_loss`: `O(N * D)` leave-one-out tail-prototype
     InfoNCE pulling top/bottom quintiles into coherent opposite clusters (strictly
     opt-in ablation, CN-tie & NaN safe).
4. **Zero-Sync Gradient Dynamics & Stabilization**:
   - `adaptive_gradient_clip_` (Brock et al. AGC, vectorized via `torch._foreach_*`,
     zero `.item()` host syncs).
   - `cautious_weight_decay_mask` & `apply_cautious_weight_decay_` (`(g * p) >= 0`).
5. **Strictly Causal Training Paradigms (Zero Future Leakage)**:
   - `CausalCurriculumScheduler`: monotonic annealing of observable liquidity gamma,
     input Gaussian noise, and multi-horizon warmup weights (chronological order only).
   - `CausalSSLPretrainer`: masked-modality representation + next-day realized
     log-volatility & cross-sectional dispersion reconstruction working across all 4
     independent families (skipping absent/NaN CN fundamentals via `torch.where`).
   - `StructuredModalityNoiseAugmenter`: 50/50 session-wide + per-asset modality
     dropout (strictly preserving `prices` channel 0) + optional Gaussian noise only
     on finite present features during `train()`, bit-exact identity in `eval()`.

Numerical stability contract:
  - Every reduction, variance, `tanh`, `rsqrt`, `softmax`, `log_softmax`, covariance,
    and per-unit norm promotes BF16/FP16 to FP32 master precision while preserving
    FP64 references for double-precision parity checks.
  - Variance always uses two-pass centered deviations (`mean((x - mean(x))^2)`),
    never `E[x^2] - E[x]^2`.
  - Softmax / log-softmax always subtracts per-column maxima before `exp` and
    computes direct `shifted - log(sum(exp(shifted)))` with empty-mask guards.
  - Zero host synchronization calls anywhere in this module (`audit_zero_host_syncs`
    total findings == 0).
"""

from __future__ import annotations

import contextlib
import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features

NormKind = Literal["layernorm", "rmsnorm", "dyt"]

_MODALITY_KEYS: tuple[str, ...] = (
    "prices",
    "fundamentals",
    "macro",
    "news",
    "charts",
)

_MODALITY_KEY_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(_MODALITY_KEYS)}

_MUP_ROUTER_TOKENS: tuple[str, ...] = (
    "router",
    "balance_bias",
    "halt_gate",
    "mix_gate",
)

_MUP_NORM_TOKENS: tuple[str, ...] = (
    "norm",
    "dyt",
    "layerscale",
    "layer_scale",
)

_MUP_EMBED_OR_INPUT_TOKENS: tuple[str, ...] = (
    "embed",
    "_emb",
    "area_emb",
    "slot_emb",
    "slot_keys",
    "queries",
    "landmarks",
    "init_notes",
    "anchor_token",
    "membership_logits",
    "phase_prior",
    "modality_prior",
    "age_decay",
    "modality_precision",
    "factor_load",
    "target_encoder",
    "regime_proj",
    "input_proj",
    "in_proj",
    "enc_in",
    "fc_in",
)

_EPISODIC_BUFFER_SUFFIXES: tuple[str, ...] = (
    "slot_keys",
    "slot_vals",
    "slot_filled",
    "write_ptr",
    "traces",
    "proto_latent",
    "proto_context",
    "proto_target",
    "proto_target_mask",
    "proto_priority",
    "proto_filled",
)


@torch.no_grad()
def _reset_episodic_buffers(model: nn.Module) -> None:
    """Zero HRR / SWR episodic buffers between SSL epochs and before Stage-1 fine-tuning."""
    for b_name, buf in model.named_buffers():
        if any(b_name.endswith(sfx) for sfx in _EPISODIC_BUFFER_SUFFIXES):
            buf.zero_()

_MUP_READOUT_TOKENS: tuple[str, ...] = (
    "alpha_head",
    "q_median",
    "q_deltas_lo",
    "q_deltas_hi",
    "tb_head",
    "turn_head",
    "vol_head",
    "epistemic_head",
    "skew_t_head",
    "readout",
    "lm_head",
    "output_head",
    "out_head",
    "pred_head",
    "classifier",
)


# ---------------------------------------------------------------------------
# Internal Numerical Precision & Context Utilities
# ---------------------------------------------------------------------------


def _calc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote BF16/FP16 to FP32 master precision while preserving FP64."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _autocast_disabled(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    dev_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    return torch.amp.autocast(device_type=dev_type, enabled=False)


def _safe_l2_normalize(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Zero-safe L2 normalization in master precision without infinite grad at 0."""
    calc_dt = _calc_dtype(x.dtype)
    xc = x.to(dtype=calc_dt)
    sq = xc.pow(2).sum(dim=dim, keepdim=True)
    eps_sq = float(eps) * float(eps)
    safe_sq = torch.where(sq > eps_sq, sq, torch.full_like(sq, eps_sq))
    norm = torch.sqrt(safe_sq)
    out = torch.where(sq > eps_sq, xc / norm, torch.zeros_like(xc))
    return out.to(dtype=x.dtype)


def _safe_layer_norm(x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Zero-safe, exact scale-invariant parameter-free LayerNorm over the last dim."""
    calc_dt = _calc_dtype(x.dtype)
    xc = x.to(dtype=calc_dt)
    mean = xc.mean(dim=-1, keepdim=True)
    centered = xc - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    eps_sq = float(eps) * float(eps)
    safe_var = torch.where(var > eps_sq, var, torch.ones_like(var))
    out = torch.where(
        var > eps_sq, centered / torch.sqrt(safe_var), torch.zeros_like(centered)
    )
    return out.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# 1. Modern SOTA Layers (Norms, Activations, Residuals)
# ---------------------------------------------------------------------------


class DynamicTanh(nn.Module):
    """Dynamic Tanh (`DyT`, Zhu et al. CVPR 2025): `gamma * tanh(alpha * x) + beta`.

    Zero-reduction elementwise normalization drop-in. Evaluates `tanh(alpha * x)` in
    master FP32/FP64 precision so BF16 training retains smooth gradients without
    mantissa collapse or overflow on extreme outliers.
    """

    def __init__(
        self,
        dim: int,
        *,
        alpha_init: float = 0.5,
        alpha_min: float | None = None,
        alpha_max: float | None = None,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        if alpha_min is not None and alpha_max is not None and alpha_min > alpha_max:
            raise ValueError(f"alpha_min ({alpha_min}) must be <= alpha_max ({alpha_max})")
        self.dim = int(dim)
        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min) if alpha_min is not None else None
        self.alpha_max = float(alpha_max) if alpha_max is not None else None
        self.elementwise_affine = bool(elementwise_affine)

        self.alpha = nn.Parameter(torch.full((1,), self.alpha_init, dtype=torch.float32))
        if self.elementwise_affine:
            self.gamma: nn.Parameter | None = nn.Parameter(
                torch.ones(self.dim, dtype=torch.float32)
            )
            self.beta: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.dim, dtype=torch.float32)
            )
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        calc_dt = _calc_dtype(orig_dtype)
        with _autocast_disabled(x.device):
            x_c = x.to(dtype=calc_dt)
            a_c = self.alpha.to(device=x.device, dtype=calc_dt)
            if self.alpha_min is not None or self.alpha_max is not None:
                a_c = torch.clamp(a_c, min=self.alpha_min, max=self.alpha_max)
            y_c = torch.tanh(a_c * x_c)
            if self.gamma is not None:
                y_c = y_c * self.gamma.to(device=x.device, dtype=calc_dt)
            if self.beta is not None:
                y_c = y_c + self.beta.to(device=x.device, dtype=calc_dt)
        return y_c.to(dtype=orig_dtype)


class RMSNormSOTA(nn.Module):
    """Master-precision RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight` (BF16-safe)."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0; got {eps}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight: nn.Parameter | None = nn.Parameter(
                torch.ones(self.dim, dtype=torch.float32)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        calc_dt = _calc_dtype(orig_dtype)
        with _autocast_disabled(x.device):
            x_c = x.to(dtype=calc_dt)
            mean_sq = x_c.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(mean_sq + self.eps)
            y_c = x_c * inv_rms
            if self.weight is not None:
                y_c = y_c * self.weight.to(device=x.device, dtype=calc_dt)
        return y_c.to(dtype=orig_dtype)


def build_norm_layer(
    kind: NormKind,
    dim: int,
    *,
    eps: float = 1e-6,
    alpha_init: float = 0.5,
) -> nn.Module:
    """Factory returning a plug-and-play normalization module (`layernorm`, `rmsnorm`, `dyt`)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    key = str(kind).lower()
    if key == "layernorm":
        return nn.LayerNorm(int(dim), eps=float(eps))
    if key == "rmsnorm":
        return RMSNormSOTA(int(dim), eps=float(eps))
    if key == "dyt":
        return DynamicTanh(int(dim), alpha_init=float(alpha_init))
    raise ValueError(f"Unsupported norm kind {kind!r}; expected 'layernorm', 'rmsnorm', or 'dyt'")


def compute_swiglu_hidden_dim(
    dim: int,
    *,
    expansion: float = 4.0,
    multiple_of: int = 8,
) -> int:
    """Compute `2/3` iso-parameter SwiGLU inner width aligned to Tensor Core multiples."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    if expansion <= 0.0:
        raise ValueError(f"expansion must be > 0; got {expansion}")
    if multiple_of < 1:
        raise ValueError(f"multiple_of must be >= 1; got {multiple_of}")
    raw = max(1, int(round((2.0 / 3.0) * float(expansion) * float(dim))))
    if multiple_of > 1:
        raw = int(multiple_of) * ((raw + int(multiple_of) - 1) // int(multiple_of))
    return max(1, raw)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: `W_down(silu(W_gate(x)) * W_up(x))` with `2/3` iso-param hidden scaling."""

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
        expansion: float = 4.0,
        multiple_of: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        self.dim = int(dim)
        self.hidden_dim = (
            int(hidden_dim)
            if hidden_dim is not None
            else compute_swiglu_hidden_dim(
                self.dim, expansion=expansion, multiple_of=multiple_of
            )
        )
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1; got {self.hidden_dim}")
        self.w_gate = nn.Linear(self.dim, self.hidden_dim, bias=bias)
        self.w_up = nn.Linear(self.dim, self.hidden_dim, bias=bias)
        self.w_down = nn.Linear(self.hidden_dim, self.dim, bias=bias)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))


class LayerScale(nn.Module):
    """Per-channel residual gain `x * gamma` initialized to `gamma_init` (Touvron et al. 2021)."""

    def __init__(self, dim: int, *, gamma_init: float = 1e-2) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        self.dim = int(dim)
        self.gamma_init = float(gamma_init)
        self.gamma = nn.Parameter(torch.full((self.dim,), self.gamma_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        calc_dt = _calc_dtype(orig_dtype)
        with _autocast_disabled(x.device):
            y_c = x.to(dtype=calc_dt) * self.gamma.to(device=x.device, dtype=calc_dt)
        return y_c.to(dtype=orig_dtype)


def deepnorm_gains(
    num_layers: int,
    *,
    mode: Literal["encoder", "decoder"] = "encoder",
) -> tuple[float, float]:
    """Return DeepNorm `(alpha, beta)` residual & init gain pair (Wang et al. 2022)."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}")
    n = float(num_layers)
    if mode == "encoder":
        return (2.0 * n) ** 0.25, (8.0 * n) ** -0.25
    if mode == "decoder":
        return (3.0 * n) ** 0.25, (12.0 * n) ** -0.25
    raise ValueError(f"Unsupported DeepNorm mode {mode!r}; expected 'encoder' or 'decoder'")


def deepnorm_residual(
    x: torch.Tensor,
    sublayer_out: torch.Tensor,
    *,
    alpha: float,
    norm: nn.Module | None = None,
) -> torch.Tensor:
    """Apply DeepNorm residual merge `norm(alpha * x + sublayer_out)` in master precision."""
    orig_dtype = x.dtype
    calc_dt = _calc_dtype(orig_dtype)
    with _autocast_disabled(x.device):
        merged = (
            float(alpha) * x.to(dtype=calc_dt) + sublayer_out.to(device=x.device, dtype=calc_dt)
        ).to(dtype=orig_dtype)
    return norm(merged) if norm is not None else merged


class DeepNormResidual(nn.Module):
    """Plug-and-play DeepNorm residual block `norm(alpha * x + sublayer_out)`."""

    def __init__(
        self,
        dim: int,
        *,
        num_layers: int,
        mode: Literal["encoder", "decoder"] = "encoder",
        norm_kind: NormKind = "rmsnorm",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.mode = mode
        alpha, beta = deepnorm_gains(self.num_layers, mode=mode)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.norm = build_norm_layer(norm_kind, self.dim, eps=eps)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return deepnorm_residual(x, sublayer_out, alpha=self.alpha, norm=self.norm)


# ---------------------------------------------------------------------------
# 2. Maximal Update Parameterization (muP) + Coordinate Check
# ---------------------------------------------------------------------------


def _has_prefix_or_segment(low_name: str, prefix: str) -> bool:
    return low_name.startswith(prefix) or f".{prefix}" in low_name


def _build_module_lookup(model: nn.Module) -> dict[int, tuple[str, nn.Module]]:
    """Map parameter `id(p)` -> `(module_name, parent_module)` for exact classification."""
    lookup: dict[int, tuple[str, nn.Module]] = {}
    for mod_name, mod in model.named_modules():
        for p in mod.parameters(recurse=False):
            lookup[id(p)] = (mod_name, mod)
    return lookup


def _classify_mup_parameter(
    name: str,
    param: nn.Parameter,
    *,
    width: int,
    parent_mod: nn.Module | None,
    first_input_linear_id: int | None,
    last_readout_linear_id: int | None,
    has_named_readout: bool,
) -> str:
    """Classify a trainable parameter into its exact muP role (`#164` P0-3 & `#183` P0-1).

    Priority:
      1. Routers / halt gates / balance biases -> `router_decay` / `router_no_decay` (`1.0x` LR)
      2. 1D tensors, biases (incl. 2D MoE/dendritic `*_b`, `*_b1`, `*_b2`), norm/DyT/LayerScale
         params, and TabM multiplicative adapters (`tabm_r`, `tabm_s`, `r_in`, `s_mid`)
         -> `norm_or_bias_no_decay` (`1.0x` LR, `wd = 0.0`)
      3. Input encoders (`enc_*`), embeddings, queries, `factor_load` -> `input_or_embed_*`
      4. Linear readout heads (2D weight with `fan_in == width`) -> `readout` (`base_width / width`)
      5. Width-scaling 2D hidden matrices (`fan_in >= max(16, width // 2)`) and 3D hidden
         expert/branch matrices (`HighSNRMoELayer`, `DendriticCompartmentLayer`,
         `HolographicBrain3L` `area_u/v`) -> `hidden_2d` (`base_width / width`)
      6. Fixed small-fan-in projections (e.g. F4 scalar tool heads `T3-T8` with `fan_in < 16`)
         and remaining tensors -> `other_decay` (`1.0x` LR)
    """
    low = name.lower()
    leaf = low.rsplit(".", 1)[-1]
    is_norm_mod = isinstance(
        parent_mod,
        (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm, RMSNormSOTA, DynamicTanh, LayerScale),
    )
    is_norm_name = any(tok in low for tok in _MUP_NORM_TOKENS)
    is_tabm = ("tabm_r" in low) or ("tabm_s" in low) or leaf in ("r_in", "s_mid")
    is_bias_like = low.endswith(".bias") or leaf in (
        "bias",
        "shared_b1",
        "shared_b2",
        "routed_b1",
        "routed_b2",
        "basal_b",
    )
    is_no_decay = param.ndim <= 1 or is_bias_like or is_norm_mod or is_norm_name or is_tabm

    # 1) Routers & gates keep 1.0x LR (checked BEFORE readout so `halt_gate.out_head` stays router)
    if any(tok in low for tok in _MUP_ROUTER_TOKENS):
        return "router_no_decay" if is_no_decay else "router_decay"

    # 2) Norms, biases, 1D scalars/vectors, and TabM multiplicative adapters keep 1.0x LR, 0 WD
    if is_no_decay:
        return "norm_or_bias_no_decay"

    # 3) Embeddings & input encoders keep 1.0x LR (never re-initialized by hidden/readout scaling)
    is_embed_mod = isinstance(parent_mod, nn.Embedding)
    is_enc_name = _has_prefix_or_segment(low, "enc_")
    is_embed_or_in_name = any(tok in low for tok in _MUP_EMBED_OR_INPUT_TOKENS)
    is_generic_input = first_input_linear_id is not None and id(parent_mod) == first_input_linear_id
    if is_embed_mod or is_enc_name or is_embed_or_in_name or is_generic_input:
        return "input_or_embed_no_decay" if is_embed_mod else "input_or_embed_decay"

    min_hidden_fan_in = max(16, int(width) // 2)

    # 4) Linear readout heads & width-scaling 2D hidden matrices
    if param.ndim == 2:
        out_features, in_features = int(param.shape[0]), int(param.shape[1])
        is_head_prefix = low.startswith("head.") or ".head." in low
        is_readout_name = is_head_prefix or any(tok in low for tok in _MUP_READOUT_TOKENS)
        if is_readout_name and in_features == int(width):
            return "readout"
        if (
            not has_named_readout
            and last_readout_linear_id is not None
            and id(parent_mod) == last_readout_linear_id
            and in_features == int(width)
        ):
            return "readout"

        # Only width-scaling 2D hidden matrices get `base_width / width` LR
        # (fixed small-fan-in F4 scalar tools `T3-T8` stay at 1.0x LR)
        if min(out_features, in_features) >= 2 and in_features >= min_hidden_fan_in:
            return "hidden_2d"

    # 5) Width-scaling 3D hidden expert/branch matrices (F3 MoE w1/w2, F2 dendritic & area_u/v)
    if param.ndim == 3:
        mat_in, mat_out = int(param.shape[-2]), int(param.shape[-1])
        if min(mat_in, mat_out) >= 2 and mat_in >= min_hidden_fan_in:
            return "hidden_2d"

    return "other_decay"


def apply_mup_init_and_param_groups(
    model: nn.Module,
    *,
    base_width: int = 128,
    width: int = 256,
    base_lr: float = 3e-4,
    weight_decay: float = 1e-2,
    hidden_init_gain: float = 1.0,
    readout_init_gain: float = 1.0,
    apply_init: bool = True,
    reinit_weights: bool = False,
    generator: torch.Generator | None = None,
) -> list[dict[str, Any]]:
    """Apply Maximal Update Parameterization (`muP`) init & build optimizer param groups.

    Strictly honors `#164` P0-3 and `#183` P0-1 guards:
      - **Width-scaling 2D & 3D-MoE/dendritic hidden matrices** (`fan_in >= max(16, width // 2)`):
        `lr = base_lr * (base_width / width)`. By default (`reinit_weights=False`), preserves
        calibrated financial module inits (already `1/sqrt(fan_in)`); set `reinit_weights=True`
        for fresh `Normal(0, hidden_init_gain / sqrt(fan_in))` re-initialization.
      - **Linear readout heads**: `lr = base_lr * (base_width / width)`. By default
        (`reinit_weights=False`), rescales calibrated readout weights by `base_width / width`
        (1.0x no-op at `width == base_width`, `std ∝ 1/width` as width grows); set
        `reinit_weights=True` for fresh `Normal(0, readout_init_gain / width)` draw.
      - **Input encoders, embeddings, norms, biases, routers, TabM adapters (`tabm_r/s`,
        `r_in/s_mid`), and fixed small-fan-in scalar tools**: kept strictly at `1.0x` LR.
    """
    if base_width < 1 or width < 1:
        raise ValueError(f"base_width ({base_width}) and width ({width}) must be >= 1")
    if base_lr <= 0.0:
        raise ValueError(f"base_lr must be > 0; got {base_lr}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0; got {weight_decay}")

    lr_scale = float(base_width) / float(width)
    scaled_lr = float(base_lr) * lr_scale
    mod_lookup = _build_module_lookup(model)

    # Detect generic sequential input/readout linears when model lacks explicit MARS-TITAN naming
    named_linears: list[tuple[str, nn.Linear]] = [
        (m_name, mod) for m_name, mod in model.named_modules() if isinstance(mod, nn.Linear)
    ]
    has_named_encoder = any(
        _has_prefix_or_segment(n.lower(), "enc_")
        or any(tok in n.lower() for tok in ("input_proj", "in_proj", "enc_in", "fc_in"))
        for n, _ in named_linears
    )
    has_named_readout = any(
        n.lower() == "head"
        or n.lower().startswith("head.")
        or ".head." in n.lower()
        or any(tok in n.lower() for tok in _MUP_READOUT_TOKENS)
        for n, _ in named_linears
    )
    first_input_linear_id: int | None = None
    if not has_named_encoder:
        for _n, lin in named_linears:
            if lin.in_features != int(width) and lin.out_features == int(width):
                first_input_linear_id = id(lin)
                break

    last_readout_linear_id: int | None = None
    if not has_named_readout:
        for _n, lin in reversed(named_linears):
            if lin.in_features == int(width) and lin.out_features < int(width):
                last_readout_linear_id = id(lin)
                break

    buckets: dict[str, list[nn.Parameter]] = {
        "hidden_2d": [],
        "readout": [],
        "input_or_embed_decay": [],
        "input_or_embed_no_decay": [],
        "router_decay": [],
        "router_no_decay": [],
        "norm_or_bias_no_decay": [],
        "other_decay": [],
    }

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            parent_entry = mod_lookup.get(id(param))
            parent_mod = parent_entry[1] if parent_entry is not None else None
            role = _classify_mup_parameter(
                name,
                param,
                width=int(width),
                parent_mod=parent_mod,
                first_input_linear_id=first_input_linear_id,
                last_readout_linear_id=last_readout_linear_id,
                has_named_readout=has_named_readout,
            )
            buckets[role].append(param)

            if apply_init:
                if role == "hidden_2d" and reinit_weights:
                    raw_fan = int(param.shape[1]) if param.ndim == 2 else int(param.shape[-2])
                    fan_in = max(1, raw_fan)
                    std = float(hidden_init_gain) / math.sqrt(float(fan_in))
                    nn.init.normal_(param, mean=0.0, std=std, generator=generator)
                elif role == "readout":
                    if reinit_weights:
                        std = float(readout_init_gain) / float(width)
                        nn.init.normal_(param, mean=0.0, std=std, generator=generator)
                    else:
                        # Preserve calibrated base_width financial init scales (base_width / width)
                        param.mul_(lr_scale)

    if apply_init and reinit_weights:
        upd_spec = getattr(model, "update_spectral_buffers", None)
        if callable(upd_spec):
            upd_spec()

    group_specs: tuple[tuple[str, float, float, float], ...] = (
        ("hidden_2d", scaled_lr, float(weight_decay), lr_scale),
        ("readout", scaled_lr, float(weight_decay), lr_scale),
        ("input_or_embed_decay", float(base_lr), float(weight_decay), 1.0),
        ("input_or_embed_no_decay", float(base_lr), 0.0, 1.0),
        ("router_decay", float(base_lr), float(weight_decay), 1.0),
        ("router_no_decay", float(base_lr), 0.0, 1.0),
        ("norm_or_bias_no_decay", float(base_lr), 0.0, 1.0),
        ("other_decay", float(base_lr), float(weight_decay), 1.0),
    )

    param_groups: list[dict[str, Any]] = []
    for role, grp_lr, grp_wd, grp_scale in group_specs:
        plist = buckets[role]
        if plist:
            param_groups.append(
                dict(
                    params=plist,
                    lr=grp_lr,
                    weight_decay=grp_wd,
                    mup_role=role,
                    mup_lr_scale=grp_scale,
                )
            )
    return param_groups


def _extract_model_prediction_1d(out: Any) -> torch.Tensor:
    if isinstance(out, Mapping):
        if "pred_return" in out and isinstance(out["pred_return"], torch.Tensor):
            p = out["pred_return"]
            return p[:, 0] if p.ndim >= 2 else p
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v.reshape(-1)
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"Unsupported model output type for coordinate check: {type(out)!r}")


def _attach_hidden_capture_hook(
    model: nn.Module,
    width: int,
    sink: list[torch.Tensor],
) -> torch.utils.hooks.RemovableHandle | None:
    """Hook the final hidden representation entering the readout head (all 4 families + generic)."""
    head_mod = getattr(model, "head", None)
    if isinstance(head_mod, nn.Module):
        return head_mod.register_forward_pre_hook(
            lambda _m, args: (
                sink.append(args[0]) if args and isinstance(args[0], torch.Tensor) else None
            )
        )
    # Fallback: hook the last nn.Linear whose `in_features == width`
    last_lin: nn.Linear | None = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear) and mod.in_features == int(width):
            last_lin = mod
    if last_lin is not None:
        return last_lin.register_forward_pre_hook(
            lambda _m, args: (
                sink.append(args[0]) if args and isinstance(args[0], torch.Tensor) else None
            )
        )
    return None


def _call_model_clean(
    model: nn.Module,
    batch_or_x: Any,
    *,
    state: dict[str, object] | None = None,
) -> Any:
    """Invoke `model` cleanly for `(x)` or `(batch, state=..., enable_early_exit=...)`."""
    sig = inspect.signature(model.forward)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "state" in params:
        kwargs["state"] = state
    if "enable_early_exit" in params:
        kwargs["enable_early_exit"] = False
    return model(batch_or_x, **kwargs)


def _log_width_slope(width_list: Sequence[int], vals: Sequence[float]) -> float:
    """Compute OLS slope of `log(val)` vs `log(width)` (~0.0 for muP, ~+1.0 for SP)."""
    if len(width_list) < 2 or any(not math.isfinite(v) or v <= 0.0 for v in vals):
        return float("nan")
    log_w = [math.log(float(w)) for w in width_list]
    log_d = [math.log(v) for v in vals]
    mw = sum(log_w) / float(len(log_w))
    md = sum(log_d) / float(len(log_d))
    var_w = sum((a - mw) ** 2 for a in log_w)
    cov_wd = sum((a - mw) * (b - md) for a, b in zip(log_w, log_d, strict=True))
    return cov_wd / max(var_w, 1e-12)


def run_mup_coordinate_check(
    model_factory: Callable[[int], nn.Module],
    widths: Sequence[int] = (64, 128, 256),
    *,
    base_width: int = 128,
    input_dim: int = 16,
    batch_size: int = 32,
    n_steps: int = 5,
    base_lr: float = 1e-2,
    weight_decay: float = 0.0,
    seed: int = 42,
    reinit_weights: bool = True,
    input_fn: Callable[[int, torch.device, torch.dtype], tuple[Any, torch.Tensor]] | None = None,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor] | None = None,
    hidden_extractor: Callable[[nn.Module, Any], torch.Tensor] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    max_ratio_threshold: float = 2.0,
) -> dict[str, Any]:
    """Empirically verify `muP` coordinate invariance across widths (Yang et al. 2022).

    Measures AND gates BOTH hidden update magnitude `||h_t - h_0||_rms` AND un-normalized
    readout update magnitude `||pred_t - pred_0||_rms` (preventing pre-head RMSNorm/LayerNorm
    from masking SP divergence, `#183` P0-2). Passes iff all updates are finite and both
    `max_to_min_ratio <= max_ratio_threshold` and `out_max_to_min_ratio <= max_ratio_threshold`.
    """
    width_list = [int(w) for w in widths]
    if len(width_list) < 2 or any(w < 2 for w in width_list):
        raise ValueError(f"widths must contain at least 2 integers >= 2; got {width_list}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    probe_gen = torch.Generator(device=dev).manual_seed(int(seed) + 17)
    if input_fn is not None:
        probe_x, probe_y = input_fn(int(batch_size), dev, dtype)
    else:
        probe_x = torch.randn(
            (int(batch_size), int(input_dim)), generator=probe_gen, device=dev, dtype=dtype
        )
        probe_y = torch.randn(
            (int(batch_size), 1), generator=probe_gen, device=dev, dtype=dtype
        )

    per_width: dict[int, dict[str, Any]] = {}
    final_deltas: dict[int, float] = {}
    mean_step_deltas: dict[int, float] = {}
    final_out_deltas: dict[int, float] = {}
    mean_step_out_deltas: dict[int, float] = {}
    all_finite = True

    for w in width_list:
        torch.manual_seed(int(seed))
        model = model_factory(w).to(device=dev, dtype=dtype)
        init_gen = torch.Generator(device=dev).manual_seed(int(seed) + w)
        param_groups = apply_mup_init_and_param_groups(
            model,
            base_width=int(base_width),
            width=w,
            base_lr=float(base_lr),
            weight_decay=float(weight_decay),
            apply_init=True,
            reinit_weights=bool(reinit_weights),
            generator=init_gen,
        )
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

        captured: list[torch.Tensor] = []
        hook = (
            None
            if hidden_extractor is not None
            else _attach_hidden_capture_hook(model, w, captured)
        )

        def _measure_hidden_and_out(
            m: nn.Module = model, sink: list[torch.Tensor] = captured
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sink.clear()
            out_m = _call_model_clean(m, probe_x)
            pred_t = _extract_model_prediction_1d(out_m).detach().to(dtype=torch.float32)
            if hidden_extractor is not None:
                h_t = hidden_extractor(m, out_m).detach().to(dtype=torch.float32)
            elif isinstance(out_m, Mapping) and isinstance(out_m.get("final_hidden"), torch.Tensor):
                h_t = out_m["final_hidden"].detach().to(dtype=torch.float32)
            elif sink:
                h_t = sink[-1].detach().to(dtype=torch.float32)
            else:
                h_t = pred_t
            return h_t, pred_t

        try:
            model.train()
            with torch.no_grad():
                h_init_t, out_init_t = _measure_hidden_and_out()
                h_init = h_init_t.clone()
                out_init = out_init_t.clone()
            h_prev = h_init
            out_prev = out_init
            delta_by_step: list[float] = []
            step_delta_by_step: list[float] = []
            out_delta_by_step: list[float] = []
            step_out_delta_by_step: list[float] = []
            last_loss_val = float("nan")
            w_finite = bool(torch.isfinite(h_init).all()) and bool(torch.isfinite(out_init).all())

            for _ in range(int(n_steps)):
                opt.zero_grad(set_to_none=True)
                captured.clear()
                out = _call_model_clean(model, probe_x)
                if loss_fn is not None:
                    loss = loss_fn(out, probe_y)
                else:
                    pred = _extract_model_prediction_1d(out)
                    target = probe_y.to(device=pred.device, dtype=pred.dtype).reshape_as(pred)
                    loss = F.mse_loss(pred, target)
                loss.backward()
                opt.step()

                last_loss_val = float(loss.detach())
                with torch.no_grad():
                    h_cur, out_cur = _measure_hidden_and_out()
                    dh_rms = float((h_cur - h_init).pow(2).mean().sqrt())
                    step_dh_rms = float((h_cur - h_prev).pow(2).mean().sqrt())
                    dout_rms = float((out_cur - out_init).pow(2).mean().sqrt())
                    step_dout_rms = float((out_cur - out_prev).pow(2).mean().sqrt())
                    h_prev = h_cur
                    out_prev = out_cur
                w_finite = (
                    w_finite
                    and math.isfinite(last_loss_val)
                    and math.isfinite(dh_rms)
                    and math.isfinite(step_dh_rms)
                    and math.isfinite(dout_rms)
                    and math.isfinite(step_dout_rms)
                )
                delta_by_step.append(dh_rms)
                step_delta_by_step.append(step_dh_rms)
                out_delta_by_step.append(dout_rms)
                step_out_delta_by_step.append(step_dout_rms)

            final_h_rms = float(h_prev.pow(2).mean().sqrt())
            final_out_rms = float(out_prev.pow(2).mean().sqrt())
        finally:
            if hook is not None:
                hook.remove()

        all_finite = (
            all_finite and w_finite and math.isfinite(final_h_rms) and math.isfinite(final_out_rms)
        )
        final_dh = delta_by_step[-1]
        mean_sdh = sum(step_delta_by_step) / float(len(step_delta_by_step))
        final_dout = out_delta_by_step[-1]
        mean_sdout = sum(step_out_delta_by_step) / float(len(step_out_delta_by_step))
        final_deltas[w] = final_dh
        mean_step_deltas[w] = mean_sdh
        final_out_deltas[w] = final_dout
        mean_step_out_deltas[w] = mean_sdout
        per_width[w] = dict(
            delta_h_rms_by_step=delta_by_step,
            step_delta_h_rms_by_step=step_delta_by_step,
            delta_out_rms_by_step=out_delta_by_step,
            step_delta_out_rms_by_step=step_out_delta_by_step,
            final_delta_h_rms=final_dh,
            mean_step_delta_h_rms=mean_sdh,
            final_delta_out_rms=final_dout,
            mean_step_delta_out_rms=mean_sdout,
            final_h_rms=final_h_rms,
            final_out_rms=final_out_rms,
            final_loss=last_loss_val,
            all_finite=w_finite,
        )

    vals = [final_deltas[w] for w in width_list]
    step_vals = [mean_step_deltas[w] for w in width_list]
    out_vals = [final_out_deltas[w] for w in width_list]
    step_out_vals = [mean_step_out_deltas[w] for w in width_list]

    max_to_min_ratio = max(vals) / max(min(vals), 1e-12) if all_finite else float("inf")
    step_max_to_min_ratio = (
        max(step_vals) / max(min(step_vals), 1e-12) if all_finite else float("inf")
    )
    out_max_to_min_ratio = (
        max(out_vals) / max(min(out_vals), 1e-12) if all_finite else float("inf")
    )
    step_out_max_to_min_ratio = (
        max(step_out_vals) / max(min(step_out_vals), 1e-12) if all_finite else float("inf")
    )

    width_scaling_exponent = _log_width_slope(width_list, vals) if all_finite else float("nan")
    out_width_scaling_exponent = (
        _log_width_slope(width_list, out_vals) if all_finite else float("nan")
    )

    passed = bool(
        all_finite
        and math.isfinite(max_to_min_ratio)
        and math.isfinite(out_max_to_min_ratio)
        and max_to_min_ratio <= float(max_ratio_threshold)
        and out_max_to_min_ratio <= float(max_ratio_threshold)
    )
    return dict(
        widths=width_list,
        base_width=int(base_width),
        per_width=per_width,
        delta_h_rms=final_deltas,
        step_delta_h_rms=mean_step_deltas,
        delta_out_rms=final_out_deltas,
        step_delta_out_rms=mean_step_out_deltas,
        max_to_min_ratio=max_to_min_ratio,
        step_max_to_min_ratio=step_max_to_min_ratio,
        out_max_to_min_ratio=out_max_to_min_ratio,
        step_out_max_to_min_ratio=step_out_max_to_min_ratio,
        width_scaling_exponent=width_scaling_exponent,
        out_width_scaling_exponent=out_width_scaling_exponent,
        all_finite=all_finite,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# 3. Scale-Invariant SOTA Losses & Anti-Collapse Regularizers (0 Host-Syncs)
# ---------------------------------------------------------------------------


def _masked_log_softmax_and_prob(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift-by-max masked softmax & direct log-softmax with zero NaN on empty columns."""
    col_valid = valid.any(dim=dim, keepdim=True)
    neg_inf = torch.full_like(logits, float("-inf"))
    raw_max = torch.where(valid, logits, neg_inf).max(dim=dim, keepdim=True).values
    safe_max = torch.where(col_valid, raw_max, torch.zeros_like(raw_max))
    shifted = torch.where(valid, logits - safe_max, torch.zeros_like(logits))
    exp_s = torch.where(valid, torch.exp(shifted), torch.zeros_like(logits))
    sum_exp = exp_s.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    log_z = torch.log(sum_exp)
    log_prob = torch.where(valid, shifted - log_z, torch.zeros_like(logits))
    prob = torch.where(valid, exp_s / sum_exp, torch.zeros_like(logits))
    return prob, log_prob


def tail_listnet_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    tau: float = 0.5,
    mask: torch.Tensor | None = None,
    liquidity_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
    rank_clip: float = 6.0,
) -> torch.Tensor:
    """Scale-invariant symmetric Tail-ListNet KL loss (`#164` P0-2 guard, 0 host-syncs).

    Computes symmetric long-tail (`+`) and short-tail (`-`) KL divergence strictly on:
      - `g_y = gaussian_rank_features(target)` (tie-averaged, 0-sync, `~N(0, 1)`), and
      - cross-sectionally z-scored `pred` (`z_p = (p - mean) / std`, exact scale & shift
        invariance with zero radial incentive to inflate `|pred|` — 100% FM-7 safe).
    At `tau = 0.5`, `softmax(±g_y / tau)` concentrates >70% probability mass on D10 / D1.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0; got {tau}")
    if pred.ndim not in (1, 2):
        raise ValueError(f"pred must be 1D [N] or 2D [N, H]; got ndim={pred.ndim}")

    calc_dt = _calc_dtype(pred.dtype)
    with _autocast_disabled(pred.device):
        p2 = pred.unsqueeze(-1) if pred.ndim == 1 else pred
        y_raw = target.to(device=pred.device)
        y2 = y_raw.unsqueeze(-1) if y_raw.ndim == 1 else y_raw
        if p2.shape != y2.shape:
            raise ValueError(f"pred shape {tuple(p2.shape)} != target shape {tuple(y2.shape)}")

        p_c = p2.to(dtype=calc_dt)
        y_c = y2.to(dtype=calc_dt)
        n_assets, n_h = p_c.shape
        if n_assets < 2:
            return torch.zeros((), device=pred.device, dtype=calc_dt)

        valid = torch.isfinite(p_c) & torch.isfinite(y_c)
        if mask is not None:
            m_t = mask.to(device=pred.device, dtype=torch.bool)
            if m_t.ndim == 1:
                m_t = m_t.unsqueeze(-1)
            valid = valid & m_t.expand(n_assets, n_h)

        if liquidity_weights is not None:
            lw = liquidity_weights.to(device=pred.device, dtype=calc_dt)
            if lw.ndim == 1:
                lw = lw.unsqueeze(-1)
            lw = lw.expand(n_assets, n_h)
            lw_ok = torch.isfinite(lw) & (lw > 0.0)
            valid = valid & lw_ok
            w = torch.where(valid, lw.clamp_min(0.0), torch.zeros_like(lw))
        else:
            w = valid.to(dtype=calc_dt)

        p_clean = torch.where(valid, p_c, torch.zeros_like(p_c))
        y_clean = torch.where(valid, y_c, torch.zeros_like(y_c))

        # Tie-averaged 0-sync Gaussian rank targets (`N(0, 1)`, all-tie CN -> 0.0)
        g_y_raw = gaussian_rank_features(y_clean, mask=valid, clip_val=float(rank_clip)).to(
            dtype=calc_dt
        )

        # Two-pass weighted z-score of BOTH `g_y` and `pred` for exact finite-N 0-KL alignment
        # and exact scale invariance (`pred * c == pred`) with zero radial gradient (FM-7 safe)
        w_sum = w.sum(dim=0, keepdim=True)
        w_norm = torch.where(w_sum > 1e-12, w / w_sum.clamp_min(1e-12), torch.zeros_like(w))
        eps_sq = float(eps) * float(eps)

        g_mean = (w_norm * g_y_raw).sum(dim=0, keepdim=True)
        g_centered = torch.where(valid, g_y_raw - g_mean, torch.zeros_like(g_y_raw))
        g_var = (w_norm * g_centered.pow(2)).sum(dim=0, keepdim=True)
        g_std = torch.sqrt(torch.where(g_var > eps_sq, g_var, torch.ones_like(g_var)))
        g_y = torch.where(valid, g_centered / g_std, torch.zeros_like(g_y_raw))

        p_mean = (w_norm * p_clean).sum(dim=0, keepdim=True)
        p_centered = torch.where(valid, p_clean - p_mean, torch.zeros_like(p_clean))
        p_var = (w_norm * p_centered.pow(2)).sum(dim=0, keepdim=True)
        # Fallback std=1.0 on degenerate constant pred prevents 1/eps grad spike at step 0
        safe_var = torch.where(p_var > eps_sq, p_var, torch.ones_like(p_var))
        p_std = torch.sqrt(safe_var)
        z_p = torch.where(valid, p_centered / p_std, torch.zeros_like(p_clean))

        inv_tau = 1.0 / float(tau)
        if liquidity_weights is not None:
            log_w = torch.where(valid, torch.log(w.clamp_min(1e-12)), torch.zeros_like(w))
            tgt_pos_logits = g_y * inv_tau + log_w
            tgt_neg_logits = -g_y * inv_tau + log_w
            prd_pos_logits = z_p * inv_tau + log_w
            prd_neg_logits = -z_p * inv_tau + log_w
        else:
            tgt_pos_logits = g_y * inv_tau
            tgt_neg_logits = -g_y * inv_tau
            prd_pos_logits = z_p * inv_tau
            prd_neg_logits = -z_p * inv_tau

        q_pos, log_q_pos = _masked_log_softmax_and_prob(tgt_pos_logits, valid, dim=0)
        q_neg, log_q_neg = _masked_log_softmax_and_prob(tgt_neg_logits, valid, dim=0)
        _, log_p_pos = _masked_log_softmax_and_prob(prd_pos_logits, valid, dim=0)
        _, log_p_neg = _masked_log_softmax_and_prob(prd_neg_logits, valid, dim=0)

        kl_pos = (q_pos * (log_q_pos - log_p_pos)).sum(dim=0).clamp_min(0.0)
        kl_neg = (q_neg * (log_q_neg - log_p_neg)).sum(dim=0).clamp_min(0.0)
        kl_sym = 0.5 * (kl_pos + kl_neg)

        valid_count = valid.sum(dim=0)
        has_target_var = g_var.squeeze(0) > eps_sq
        active_h = ((valid_count >= 2) & has_target_var).to(dtype=calc_dt)
        active_sum = active_h.sum()
        loss = torch.where(
            active_sum > 0.5,
            (kl_sym * active_h).sum() / active_sum.clamp_min(1.0),
            torch.zeros((), device=pred.device, dtype=calc_dt),
        )
    return loss


def vicreg_covariance_loss(
    hidden: torch.Tensor,
    *,
    gamma: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 1.0,
    eps: float = 1e-4,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-sync width-invariant VICReg off-diagonal covariance + variance floor (`#164` P0-3).

    Prevents dimensional collapse and boosts `eRank` on `final_hidden [N, D]` in `<0.05ms`
    using two-pass centered covariance in FP32/FP64 master precision. Off-diagonal squared
    covariance is normalized by `D * (D - 1)` so penalty magnitude stays `O(1)` across widths.
    """
    if hidden.ndim not in (2, 3):
        raise ValueError(f"hidden must be 2D [N, D] or 3D [B, N, D]; got ndim={hidden.ndim}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0; got {gamma}")

    calc_dt = _calc_dtype(hidden.dtype)
    with _autocast_disabled(hidden.device):
        h = hidden.reshape(-1, hidden.shape[-1]).to(dtype=calc_dt)
        n_rows, dim = h.shape
        if n_rows < 2 or dim < 2:
            return torch.zeros((), device=hidden.device, dtype=calc_dt)

        row_valid = torch.isfinite(h).all(dim=-1)
        if mask is not None:
            m_flat = mask.to(device=hidden.device, dtype=torch.bool).reshape(-1)
            row_valid = row_valid & m_flat

        w = row_valid.to(dtype=calc_dt).unsqueeze(-1)  # [N, 1]
        n_valid = w.sum()
        has_enough = n_valid >= 2.0
        h_clean = torch.where(row_valid.unsqueeze(-1), h, torch.zeros_like(h))

        # Two-pass centered variance & covariance (never E[x^2] - E[x]^2)
        mean = (h_clean * w).sum(dim=0, keepdim=True) / n_valid.clamp_min(1.0)
        h_c = torch.where(row_valid.unsqueeze(-1), h_clean - mean, torch.zeros_like(h_clean))
        denom = (n_valid - 1.0).clamp_min(1.0)

        var = (h_c.pow(2) * w).sum(dim=0) / denom
        std = torch.sqrt(var.clamp_min(0.0) + float(eps))
        var_loss = F.relu(float(gamma) - std).mean()

        cov = (h_c.transpose(0, 1) @ h_c) / denom  # [D, D]
        off_diag = 1.0 - torch.eye(dim, device=hidden.device, dtype=calc_dt)
        cov_loss = (cov.pow(2) * off_diag).sum() / float(dim * (dim - 1))

        total = float(cov_weight) * cov_loss + float(var_weight) * var_loss
        out = torch.where(has_enough, total, torch.zeros_like(total))
    return out


def cross_sectional_rank_contrastive_loss(
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    top_frac: float = 0.2,
    temp: float = 0.1,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """`O(N * D)` leave-one-out tail-prototype contrastive loss (opt-in ablation, 0 syncs).

    Pulls top-quintile stocks toward their leave-one-out top-tail centroid and bottom-quintile
    stocks toward their leave-one-out bottom-tail centroid while pushing opposite tails apart
    via overflow-proof `softplus((s_neg - s_pos) / temp)`. Safely returns `0.0` on CN all-tie
    or sparse sessions with zero host syncs.
    """
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be 2D [N, D]; got ndim={hidden.ndim}")
    if not (0.0 < top_frac <= 0.5):
        raise ValueError(f"top_frac must be in (0, 0.5]; got {top_frac}")
    if temp <= 0.0:
        raise ValueError(f"temp must be > 0; got {temp}")

    calc_dt = _calc_dtype(hidden.dtype)
    with _autocast_disabled(hidden.device):
        h = hidden.to(dtype=calc_dt)
        n_assets, dim = h.shape
        if n_assets < 4 or dim < 2:
            return torch.zeros((), device=hidden.device, dtype=calc_dt)

        y_raw = target.to(device=hidden.device, dtype=calc_dt)
        y2 = y_raw.unsqueeze(-1) if y_raw.ndim == 1 else y_raw
        if y2.shape[0] != n_assets:
            raise ValueError(
                f"target first dim {y2.shape[0]} != hidden first dim {n_assets}"
            )
        n_h = y2.shape[1]

        h_ok = torch.isfinite(h).all(dim=-1, keepdim=True)
        valid = h_ok & torch.isfinite(y2)
        if mask is not None:
            m_t = mask.to(device=hidden.device, dtype=torch.bool)
            if m_t.ndim == 1:
                m_t = m_t.unsqueeze(-1)
            valid = valid & m_t.expand(n_assets, n_h)

        h_clean = torch.where(h_ok, h, torch.zeros_like(h))
        z = _safe_l2_normalize(h_clean, dim=-1, eps=eps)  # [N, D]
        y_clean = torch.where(valid, y2, torch.zeros_like(y2))

        # Exact tie-averaged percentile rank `u in (0, 1)` via 0-sync Gaussian rank -> Phi(g)
        g_y = gaussian_rank_features(y_clean, mask=valid, clip_val=6.0).to(dtype=calc_dt)
        u_rank = torch.special.ndtr(g_y)

        # Strict tail masks: all-tied CN sessions have u_rank == 0.5 -> neither tail triggers!
        top_bool = valid & (u_rank >= (1.0 - float(top_frac))) & (g_y > 0.0)
        bot_bool = valid & (u_rank <= float(top_frac)) & (g_y < 0.0)
        m_top = top_bool.to(dtype=calc_dt)  # [N, H]
        m_bot = bot_bool.to(dtype=calc_dt)  # [N, H]
        n_top = m_top.sum(dim=0)  # [H]
        n_bot = m_bot.sum(dim=0)  # [H]
        has_both = (n_top >= 2.0) & (n_bot >= 2.0)

        sum_top = torch.einsum("nd,nh->hd", z, m_top)  # [H, D]
        sum_bot = torch.einsum("nd,nh->hd", z, m_bot)  # [H, D]
        c_top = _safe_l2_normalize(sum_top, dim=-1, eps=eps)  # [H, D]
        c_bot = _safe_l2_normalize(sum_bot, dim=-1, eps=eps)  # [H, D]

        # Leave-one-out same-tail prototypes exclude stock `i` itself (zero self-shortcut!)
        loo_top = sum_top.unsqueeze(0) - z.unsqueeze(1) * m_top.unsqueeze(-1)  # [N, H, D]
        loo_bot = sum_bot.unsqueeze(0) - z.unsqueeze(1) * m_bot.unsqueeze(-1)  # [N, H, D]
        c_top_loo = _safe_l2_normalize(loo_top, dim=-1, eps=eps)
        c_bot_loo = _safe_l2_normalize(loo_bot, dim=-1, eps=eps)

        inv_temp = 1.0 / float(temp)
        s_top_pos = torch.einsum("nd,nhd->nh", z, c_top_loo) * inv_temp
        s_top_neg = torch.einsum("nd,hd->nh", z, c_bot) * inv_temp
        loss_top_el = torch.where(
            top_bool, F.softplus(s_top_neg - s_top_pos), torch.zeros_like(s_top_pos)
        )

        s_bot_pos = torch.einsum("nd,nhd->nh", z, c_bot_loo) * inv_temp
        s_bot_neg = torch.einsum("nd,hd->nh", z, c_top) * inv_temp
        loss_bot_el = torch.where(
            bot_bool, F.softplus(s_bot_neg - s_bot_pos), torch.zeros_like(s_bot_pos)
        )

        loss_h = 0.5 * (
            loss_top_el.sum(dim=0) / n_top.clamp_min(1.0)
            + loss_bot_el.sum(dim=0) / n_bot.clamp_min(1.0)
        )
        active_h = has_both.to(dtype=calc_dt)
        active_sum = active_h.sum()
        out = torch.where(
            active_sum > 0.5,
            (loss_h * active_h).sum() / active_sum.clamp_min(1.0),
            torch.zeros((), device=hidden.device, dtype=calc_dt),
        )
    return out


# ---------------------------------------------------------------------------
# 4. Gradient Dynamics & Stabilization (Vectorized `_foreach_*`, 0 Syncs)
# ---------------------------------------------------------------------------


def _unit_reduce_dims(t: torch.Tensor) -> tuple[int, ...]:
    if t.ndim >= 2:
        return tuple(range(1, t.ndim))
    if t.ndim == 1:
        return (0,)
    return ()


@torch.no_grad()
def adaptive_gradient_clip_(
    params: Iterable[nn.Parameter] | nn.Module,
    *,
    clip_factor: float = 0.01,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Brock et al. (2021) Adaptive Gradient Clipping (AGC) via `torch._foreach_*`.

    Enforces per-unit `||g_i||_2 / max(||w_i||_2, eps) <= clip_factor` in-place with
    zero `.item()` host syncs and master FP32/FP64 norm precision under BF16.
    Returns a device scalar tensor containing the fraction of parameter units clipped.
    """
    if clip_factor <= 0.0:
        raise ValueError(f"clip_factor must be > 0; got {clip_factor}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps}")

    param_iter = params.parameters() if isinstance(params, nn.Module) else params
    buckets: dict[tuple[torch.device, torch.dtype], list[nn.Parameter]] = {}
    first_dev: torch.device | None = None
    for p in param_iter:
        if first_dev is None:
            first_dev = p.device
        if p.grad is None or not p.requires_grad:
            continue
        key = (p.device, _calc_dtype(p.dtype))
        buckets.setdefault(key, []).append(p)

    if not buckets:
        return torch.zeros((), device=first_dev or torch.device("cpu"), dtype=torch.float32)

    clipped_units_total: torch.Tensor | None = None
    units_total = 0

    for (dev, calc_dt), plist in buckets.items():
        raw_grads = [p.grad for p in plist if p.grad is not None]
        g_calc = [g.to(dtype=calc_dt) if g.dtype != calc_dt else g for g in raw_grads]
        w_calc = [
            p.detach().to(dtype=calc_dt) if p.dtype != calc_dt else p.detach() for p in plist
        ]

        g_sq = torch._foreach_mul(g_calc, g_calc)
        w_sq = torch._foreach_mul(w_calc, w_calc)

        g_sq_sums: list[torch.Tensor] = []
        w_sq_sums: list[torch.Tensor] = []
        for gs, ws in zip(g_sq, w_sq, strict=True):
            dims = _unit_reduce_dims(gs)
            if dims:
                g_sq_sums.append(gs.sum(dim=dims, keepdim=True))
                w_sq_sums.append(ws.sum(dim=dims, keepdim=True))
            else:
                g_sq_sums.append(gs)
                w_sq_sums.append(ws)

        g_norms = torch._foreach_sqrt(g_sq_sums)
        w_norms = torch._foreach_sqrt(w_sq_sums)

        w_clamped = [wn.clamp_min(float(eps)) for wn in w_norms]
        max_norms = torch._foreach_mul(w_clamped, float(clip_factor))
        g_denoms = [gn.clamp_min(1e-12) for gn in g_norms]
        ratios = torch._foreach_div(max_norms, g_denoms)

        scales: list[torch.Tensor] = []
        bucket_clipped = torch.zeros((), device=dev, dtype=torch.float32)
        for gn, mn, r in zip(g_norms, max_norms, ratios, strict=True):
            is_clipped = gn > mn
            bucket_clipped = bucket_clipped + is_clipped.to(dtype=torch.float32).sum()
            units_total += int(gn.numel())
            # Exact 1.0 when unclipped (bit-exact preservation of unclipped units!)
            scales.append(torch.where(is_clipped, r.clamp(0.0, 1.0), torch.ones_like(r)))

        same_grads: list[torch.Tensor] = []
        same_scales: list[torch.Tensor] = []
        cast_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for g_orig, s in zip(raw_grads, scales, strict=True):
            s_exp = s.expand_as(g_orig)
            if g_orig.dtype == s_exp.dtype:
                same_grads.append(g_orig)
                same_scales.append(s_exp)
            else:
                cast_pairs.append((g_orig, s_exp))

        if same_grads:
            torch._foreach_mul_(same_grads, same_scales)
        for g_orig, s_exp in cast_pairs:
            g_orig.copy_((g_orig.to(dtype=calc_dt) * s_exp).to(dtype=g_orig.dtype))

        if clipped_units_total is None:
            clipped_units_total = bucket_clipped
        else:
            clipped_units_total = clipped_units_total + bucket_clipped.to(
                device=clipped_units_total.device
            )

    assert clipped_units_total is not None
    return clipped_units_total / float(max(units_total, 1))


def cautious_weight_decay_mask(
    grad: torch.Tensor,
    param: torch.Tensor,
    *,
    rescale: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aligned-sign cautious weight decay mask (`(g * p) >= 0`, underflow-free, 0 syncs)."""
    g = grad.detach()
    p = param.detach()
    calc_dt = _calc_dtype(p.dtype)
    with _autocast_disabled(p.device):
        g_c = g.to(dtype=calc_dt)
        p_c = p.to(dtype=calc_dt)
        # Direct sign comparison prevents BF16/FP32 subnormal underflow (`-1e-20 * 1e-20 -> 0.0`)
        aligned = ((g_c >= 0.0) & (p_c >= 0.0)) | ((g_c <= 0.0) & (p_c <= 0.0))
        mask_c = aligned.to(dtype=calc_dt)
        if rescale:
            denom = mask_c.mean().clamp_min(float(eps))
            mask_c = mask_c / denom
    return mask_c.to(dtype=p.dtype)


@torch.no_grad()
def apply_cautious_weight_decay_(
    params: Iterable[nn.Parameter] | nn.Module,
    *,
    lr: float,
    weight_decay: float,
    rescale: bool = False,
) -> None:
    """Apply aligned-sign decoupled cautious weight decay in-place via `torch._foreach_*`."""
    if weight_decay <= 0.0 or lr <= 0.0:
        return
    param_iter = params.parameters() if isinstance(params, nn.Module) else params
    buckets: dict[
        tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]
    ] = {}
    for p in param_iter:
        if p.grad is None or not p.requires_grad:
            continue
        key = (p.device, p.dtype)
        ps, deltas = buckets.setdefault(key, ([], []))
        mask = cautious_weight_decay_mask(p.grad, p, rescale=rescale)
        ps.append(p)
        deltas.append(p.detach() * mask)

    factor = -float(lr) * float(weight_decay)
    for ps, deltas in buckets.values():
        torch._foreach_add_(ps, deltas, alpha=factor)


# ---------------------------------------------------------------------------
# 5. Strictly Causal Training Paradigms (Zero Future Leakage)
# ---------------------------------------------------------------------------


class CausalCurriculumScheduler:
    """Strictly causal curriculum annealing observable liquidity, noise, and horizon weights.

    Never reorders sessions (preserving causal Titans/HRR/Replay state transitions).
    """

    def __init__(
        self,
        total_epochs: int,
        liq_gamma_start: float = 2.0,
        liq_gamma_end: float = 1.0,
        noise_start: float = 0.05,
        noise_end: float = 0.0,
        horizon_warmup_epochs: int = 2,
        *,
        schedule: Literal["cosine", "linear"] = "cosine",
    ) -> None:
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1; got {total_epochs}")
        if liq_gamma_start < 0.0 or liq_gamma_end < 0.0:
            raise ValueError("liquidity gammas must be >= 0")
        if noise_start < 0.0 or noise_end < 0.0:
            raise ValueError("noise std endpoints must be >= 0")
        if horizon_warmup_epochs < 0:
            raise ValueError(f"horizon_warmup_epochs must be >= 0; got {horizon_warmup_epochs}")
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"Unsupported schedule {schedule!r}; expected 'cosine' or 'linear'")

        self.total_epochs = int(total_epochs)
        self.liq_gamma_start = float(liq_gamma_start)
        self.liq_gamma_end = float(liq_gamma_end)
        self.noise_start = float(noise_start)
        self.noise_end = float(noise_end)
        self.horizon_warmup_epochs = int(horizon_warmup_epochs)
        self.schedule = schedule

    def progress(self, epoch: int | float) -> float:
        if self.total_epochs <= 1:
            return 1.0
        raw = float(epoch) / float(self.total_epochs - 1)
        return min(max(raw, 0.0), 1.0)

    def _interp_factor(self, epoch: int | float) -> float:
        p = self.progress(epoch)
        if self.schedule == "linear":
            return p
        return 0.5 * (1.0 - math.cos(math.pi * p))

    def liq_gamma(self, epoch: int | float) -> float:
        a = self._interp_factor(epoch)
        return self.liq_gamma_start + a * (self.liq_gamma_end - self.liq_gamma_start)

    def noise_std(self, epoch: int | float) -> float:
        a = self._interp_factor(epoch)
        return self.noise_start + a * (self.noise_end - self.noise_start)

    def horizon_weights(
        self,
        epoch: int | float,
        n_horizons: int = 6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        decay_strength: float = 2.0,
    ) -> torch.Tensor:
        """Compute strictly causal multi-horizon emphasis weights `[H]`."""
        if n_horizons < 1:
            raise ValueError(f"n_horizons must be >= 1; got {n_horizons}")
        if self.horizon_warmup_epochs <= 0:
            ramp = 1.0
        else:
            ramp = min(max(float(epoch) / float(self.horizon_warmup_epochs), 0.0), 1.0)
        idx = torch.arange(int(n_horizons), device=device, dtype=torch.float32)
        if n_horizons == 1:
            return torch.ones((1,), device=device, dtype=dtype)
        frac = idx / float(n_horizons - 1)
        weights = torch.exp(-float(decay_strength) * (1.0 - ramp) * frac)
        return weights.to(dtype=dtype)

    def weight_liquidity(
        self,
        liquidity_pct: torch.Tensor,
        epoch: int | float,
        *,
        floor: float = 0.05,
    ) -> torch.Tensor:
        """Compute normalized per-asset liquidity weights from observable causal `liquidity_pct`."""
        calc_dt = _calc_dtype(liquidity_pct.dtype)
        gamma = self.liq_gamma(epoch)
        liq = liquidity_pct.to(dtype=calc_dt)
        finite = torch.isfinite(liq)
        clean = torch.where(
            finite, liq.clamp(float(floor), 1.0), torch.full_like(liq, float(floor))
        )
        raw = clean.pow(gamma)
        mean_w = raw.mean().clamp_min(1e-8)
        return (raw / mean_w).to(dtype=liquidity_pct.dtype)

    def state_at(self, epoch: int | float, n_horizons: int = 6) -> dict[str, Any]:
        return dict(
            epoch=float(epoch),
            progress=self.progress(epoch),
            liq_gamma=self.liq_gamma(epoch),
            noise_std=self.noise_std(epoch),
            horizon_weights=self.horizon_weights(epoch, n_horizons=n_horizons),
        )


class StructuredModalityNoiseAugmenter(nn.Module):
    """50/50 session-wide + per-asset modality dropout (preserving `prices`) + feature noise.

    Active strictly during `train()`, bit-exact identity in `eval()`, zero host-syncs,
    and respects missingness so absent CN fundamentals stay exact `0.0` while preserving
    finite/non-finite flags in `prices` (for `SessionHealthGate`).
    """

    def __init__(
        self,
        *,
        modality_dropout: float = 0.20,
        session_level_prob: float = 0.50,
        noise_std: float = 0.05,
        preserve_modality_indices: tuple[int, ...] = (0,),
        noise_keys: tuple[str, ...] = ("prices", "physics_v4", "fundamentals", "news", "charts"),
    ) -> None:
        super().__init__()
        if not (0.0 <= modality_dropout < 1.0):
            raise ValueError(f"modality_dropout must be in [0, 1); got {modality_dropout}")
        if not (0.0 <= session_level_prob <= 1.0):
            raise ValueError(f"session_level_prob must be in [0, 1]; got {session_level_prob}")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        self.modality_dropout = float(modality_dropout)
        self.session_level_prob = float(session_level_prob)
        self.noise_std = float(noise_std)
        self.preserve_modality_indices = tuple(int(i) for i in preserve_modality_indices)
        self.noise_keys = tuple(noise_keys)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        modality_dropout: float | None = None,
        noise_std: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.training:
            return dict(batch)

        eff_drop = self.modality_dropout if modality_dropout is None else float(modality_dropout)
        eff_noise = self.noise_std if noise_std is None else float(noise_std)
        out: dict[str, torch.Tensor] = dict(batch)

        new_mask: torch.Tensor | None = None
        if "modality_mask" in batch and isinstance(batch["modality_mask"], torch.Tensor):
            orig_mask = batch["modality_mask"]
            n_assets, n_mod = orig_mask.shape[0], orig_mask.shape[-1]
            dev = orig_mask.device
            if eff_drop > 0.0 and n_mod > 1:
                use_session = (
                    torch.rand((1, 1), device=dev, generator=generator) < self.session_level_prob
                )
                keep_session = (
                    torch.rand((1, n_mod), device=dev, generator=generator) >= eff_drop
                )
                keep_asset = (
                    torch.rand((n_assets, n_mod), device=dev, generator=generator) >= eff_drop
                )
                keep = torch.where(use_session, keep_session, keep_asset)

                preserve = torch.zeros((1, n_mod), device=dev, dtype=torch.bool)
                for idx in self.preserve_modality_indices:
                    if 0 <= idx < n_mod:
                        preserve[0, idx] = True
                keep = keep | preserve
                new_mask = orig_mask * keep.to(dtype=orig_mask.dtype)
            else:
                new_mask = orig_mask.clone()
            out["modality_mask"] = new_mask

        if eff_noise > 0.0:
            for key in self.noise_keys:
                val = batch.get(key)
                if not isinstance(val, torch.Tensor) or not val.is_floating_point():
                    continue
                ch_idx = _MODALITY_KEY_TO_IDX.get(key)
                present: torch.Tensor | None = None
                if ch_idx is not None and new_mask is not None and ch_idx < new_mask.shape[-1]:
                    present = (new_mask[:, ch_idx : ch_idx + 1] > 0.5).to(device=val.device)

                # Preserve binary `obs_flag` & `log_age` bit-exact in encoded-v3.1 triplets
                n_triplet_slots = (
                    26
                    if (key == "fundamentals" and val.ndim == 2 and val.shape[-1] == 78)
                    else (140 if (key == "macro" and val.ndim == 2 and val.shape[-1] == 420) else 0)
                )
                if n_triplet_slots > 0:
                    v_raw, obs_raw, age_raw = torch.split(val, n_triplet_slots, dim=-1)
                    active_v = torch.isfinite(v_raw) & torch.isfinite(obs_raw) & (obs_raw > 0.5)
                    if present is not None:
                        active_v = active_v & present
                    v_noise = (
                        torch.randn(
                            v_raw.shape,
                            device=val.device,
                            dtype=val.dtype,
                            generator=generator,
                        )
                        * eff_noise
                    )
                    v_aug = torch.where(active_v, v_raw + v_noise, v_raw)
                    out[key] = torch.cat([v_aug, obs_raw, age_raw], dim=-1)
                else:
                    active = torch.isfinite(val)
                    if present is not None:
                        active = active & present
                    noise = (
                        torch.randn(
                            val.shape, device=val.device, dtype=val.dtype, generator=generator
                        )
                        * eff_noise
                    )
                    out[key] = torch.where(active, val + noise, val)

        return out


class CausalSSLPretrainer(nn.Module):
    """Causal Self-Supervised Pretrainer working across all 4 independent model families.

    Combines two strictly causal high-SNR objectives with zero host syncs:
      1. **Masked-modality representation reconstruction**: randomly masks present auxiliary
         modalities (`fundamentals`, `macro`, `news`, `charts` — never `prices`), skips
         already-absent/NaN CN fundamentals via `torch.where`, and reconstructs stop-gradient
         modality embeddings (or unmasked latent fallback for generic models).
      2. **Next-day realized log-volatility & cross-sectional dispersion reconstruction**:
         predicts next-day per-asset `log(|r_{t+1}| + eps)` and session-level cross-sectional
         return dispersion `log(std_cs(r_{t+1}) + eps)` (~50x return SNR).
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_modalities: int = 5,
        mask_prob: float = 0.40,
        w_masked_mod: float = 1.0,
        w_vol: float = 1.0,
        w_dispersion: float = 0.5,
        vol_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim must be >= 2; got {hidden_dim}")
        if n_modalities < 2:
            raise ValueError(f"n_modalities must be >= 2; got {n_modalities}")
        if not (0.0 < mask_prob < 1.0):
            raise ValueError(f"mask_prob must be in (0, 1); got {mask_prob}")

        self.hidden_dim = int(hidden_dim)
        self.n_modalities = int(n_modalities)
        self.mask_prob = float(mask_prob)
        self.w_masked_mod = float(w_masked_mod)
        self.w_vol = float(w_vol)
        self.w_dispersion = float(w_dispersion)
        self.vol_eps = float(vol_eps)

        self.recon_norm = RMSNormSOTA(self.hidden_dim)
        self.mod_recon_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.n_modalities * self.hidden_dim),
        )
        self.latent_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.vol_head = nn.Sequential(
            RMSNormSOTA(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.dispersion_head = nn.Sequential(
            RMSNormSOTA(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _sanitize_batch_for_ssl(
        batch: Mapping[str, torch.Tensor],
        mask_bool: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Zero-out absent/NaN auxiliary modalities via `torch.where` (CN 99.3% NaN-safe)."""
        clean: dict[str, torch.Tensor] = dict(batch)
        clean["modality_mask"] = mask_bool.to(dtype=batch["prices"].dtype)
        for ch_idx, key in enumerate(_MODALITY_KEYS):
            val = batch.get(key)
            if not isinstance(val, torch.Tensor) or not val.is_floating_point():
                continue
            if ch_idx == 0:
                clean[key] = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
            elif ch_idx < mask_bool.shape[-1]:
                present = mask_bool[:, ch_idx : ch_idx + 1].to(device=val.device)
                clean[key] = torch.where(
                    present & torch.isfinite(val), val, torch.zeros_like(val)
                )
        return clean

    @staticmethod
    def _forward_with_hidden(
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        state: dict[str, object] | None = None,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        captured: list[torch.Tensor] = []
        head_mod = getattr(model, "head", None)
        hook = (
            head_mod.register_forward_pre_hook(
                lambda _m, args: captured.append(args[0])
                if args and isinstance(args[0], torch.Tensor)
                else None
            )
            if isinstance(head_mod, nn.Module)
            else None
        )
        try:
            out_raw = _call_model_clean(model, batch, state=state)
        finally:
            if hook is not None:
                hook.remove()

        if isinstance(out_raw, Mapping):
            out_dict = dict(out_raw)
            if isinstance(out_dict.get("final_hidden"), torch.Tensor):
                return out_dict, out_dict["final_hidden"]
            if captured:
                return out_dict, captured[-1]
            raise ValueError("Model output lacks 'final_hidden' and no `model.head` hook fired")
        if isinstance(out_raw, torch.Tensor):
            return {"final_hidden": out_raw}, out_raw
        raise TypeError(f"Unsupported model output type {type(out_raw)!r}")

    def compute_ssl_loss(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        next_returns: torch.Tensor | None = None,
        next_vol: torch.Tensor | None = None,
        state: dict[str, object] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Compute causal SSL loss dict (total, masked_mod_loss, vol_loss, dispersion_loss)."""
        prices = batch["prices"]
        n_assets = prices.shape[0]
        device = prices.device
        calc_dt = _calc_dtype(prices.dtype)

        orig_mask_raw = batch.get("modality_mask")
        if isinstance(orig_mask_raw, torch.Tensor):
            orig_mask_bool = orig_mask_raw.to(device=device) > 0.5
        else:
            orig_mask_bool = torch.ones(
                (n_assets, self.n_modalities), device=device, dtype=torch.bool
            )
        n_mod = orig_mask_bool.shape[-1]

        # Sanitize CN missing/NaN modalities BEFORE teacher encoding or student forward
        clean_full_batch = self._sanitize_batch_for_ssl(batch, orig_mask_bool)

        # Sample random mask ONLY on present auxiliary modalities (channel 0 `prices` preserved)
        drop_rand = (
            torch.rand((n_assets, n_mod), device=device, generator=generator) < self.mask_prob
        )
        aux_cols = torch.arange(n_mod, device=device).unsqueeze(0) > 0
        actually_masked = orig_mask_bool & drop_rand & aux_cols
        student_mask_bool = orig_mask_bool & (~actually_masked)
        masked_batch = self._sanitize_batch_for_ssl(clean_full_batch, student_mask_bool)

        # Teacher stop-gradient targets from sanitized unmasked view
        target_tokens: torch.Tensor | None = None
        target_latent: torch.Tensor | None = None
        enc_fn = getattr(model, "_encode_modalities", None)
        with torch.no_grad():
            if callable(enc_fn):
                enc_out = enc_fn(clean_full_batch)
                tok = enc_out[0] if isinstance(enc_out, tuple) else enc_out
                if (
                    isinstance(tok, torch.Tensor)
                    and tok.ndim == 3
                    and tok.shape[-1] == self.hidden_dim
                ):
                    target_tokens = tok.detach()
            if target_tokens is None:
                _, h_teach = self._forward_with_hidden(model, clean_full_batch, state=state)
                target_latent = h_teach.detach()

        # Student forward on masked view (full gradient flow into backbone + encoders)
        out_dict, h_student = self._forward_with_hidden(model, masked_batch, state=state)

        with _autocast_disabled(device):
            h_f = h_student.to(dtype=calc_dt)
            h_norm = self.recon_norm(h_f)

            # 1) Masked-modality representation reconstruction (norm-invariant + CN NaN-safe)
            if target_tokens is not None and target_tokens.shape[1] == self.n_modalities:
                pred_tok = self.mod_recon_head(h_norm).view(
                    n_assets, self.n_modalities, self.hidden_dim
                )
                tgt_tok = target_tokens.to(device=device, dtype=calc_dt)
                valid_tok = actually_masked & torch.isfinite(tgt_tok).all(dim=-1)
                tgt_safe = torch.where(valid_tok.unsqueeze(-1), tgt_tok, torch.zeros_like(tgt_tok))
                pred_u = _safe_l2_normalize(pred_tok, dim=-1)
                tgt_u = _safe_l2_normalize(tgt_safe, dim=-1)
                cos_loss = 1.0 - (pred_u * tgt_u).sum(dim=-1)
                # Parameter-free scale-invariant LayerNorm defeats norm-collapse shortcut
                pred_ln = _safe_layer_norm(pred_tok)
                tgt_ln = _safe_layer_norm(tgt_safe)
                tgt_ln = torch.where(valid_tok.unsqueeze(-1), tgt_ln, torch.zeros_like(tgt_ln))
                mse_loss = F.smooth_l1_loss(pred_ln, tgt_ln, reduction="none").mean(dim=-1)
                el_loss = torch.where(valid_tok, cos_loss + mse_loss, torch.zeros_like(cos_loss))
                n_masked = valid_tok.to(dtype=calc_dt).sum()
                masked_mod_loss = torch.where(
                    n_masked > 0.5,
                    el_loss.sum() / n_masked.clamp_min(1.0),
                    torch.zeros((), device=device, dtype=calc_dt),
                )
            else:
                assert target_latent is not None
                pred_lat = self.latent_predictor(h_norm)
                tgt_lat = target_latent.to(device=device, dtype=calc_dt)
                row_has_mask = actually_masked.any(dim=-1) & torch.isfinite(tgt_lat).all(dim=-1)
                tgt_safe = torch.where(
                    row_has_mask.unsqueeze(-1), tgt_lat, torch.zeros_like(tgt_lat)
                )
                cos_l = 1.0 - (
                    _safe_l2_normalize(pred_lat) * _safe_l2_normalize(tgt_safe)
                ).sum(dim=-1)
                pred_ln = _safe_layer_norm(pred_lat)
                tgt_ln = _safe_layer_norm(tgt_safe)
                tgt_ln = torch.where(row_has_mask.unsqueeze(-1), tgt_ln, torch.zeros_like(tgt_ln))
                mse_l = F.smooth_l1_loss(pred_ln, tgt_ln, reduction="none").mean(dim=-1)
                el_l = torch.where(row_has_mask, cos_l + mse_l, torch.zeros_like(cos_l))
                n_masked = row_has_mask.to(dtype=calc_dt).sum()
                masked_mod_loss = torch.where(
                    n_masked > 0.5,
                    el_l.sum() / n_masked.clamp_min(1.0),
                    torch.zeros((), device=device, dtype=calc_dt),
                )

            # 2) Causal next-day realized log-volatility & cross-sectional dispersion
            y_ret_raw = next_returns if next_returns is not None else batch.get("target_returns")
            y1: torch.Tensor | None = None
            if isinstance(y_ret_raw, torch.Tensor):
                yr = y_ret_raw.to(device=device, dtype=calc_dt)
                y1 = yr[:, 0] if yr.ndim >= 2 else yr

            if next_vol is not None:
                v_raw = next_vol.to(device=device, dtype=calc_dt)
                v1 = v_raw[:, 0] if v_raw.ndim >= 2 else v_raw
                vol_fin = torch.isfinite(v1) & (v1 >= 0.0)
                v_safe = torch.where(vol_fin, v1, torch.zeros_like(v1))
                tgt_log_vol = torch.log(v_safe.clamp_min(0.0) + self.vol_eps).clamp(-7.0, 1.0)
            elif y1 is not None:
                vol_fin = torch.isfinite(y1)
                y1_safe = torch.where(vol_fin, y1, torch.zeros_like(y1))
                tgt_log_vol = torch.log(y1_safe.abs() + self.vol_eps).clamp(-7.0, 1.0)
            else:
                vol_fin = torch.zeros((n_assets,), device=device, dtype=torch.bool)
                tgt_log_vol = torch.zeros((n_assets,), device=device, dtype=calc_dt)

            pred_log_vol = self.vol_head(h_f).squeeze(-1)
            vol_el = F.huber_loss(pred_log_vol, tgt_log_vol, delta=0.5, reduction="none")
            vol_el = torch.where(vol_fin, vol_el, torch.zeros_like(vol_el))
            n_vol = vol_fin.to(dtype=calc_dt).sum()
            vol_loss = torch.where(
                n_vol > 0.5,
                vol_el.sum() / n_vol.clamp_min(1.0),
                torch.zeros((), device=device, dtype=calc_dt),
            )

            # Session-level next-day cross-sectional dispersion (two-pass centered std)
            if y1 is not None:
                ret_fin = torch.isfinite(y1)
                r_cnt = ret_fin.to(dtype=calc_dt).sum()
                r_safe = torch.where(ret_fin, y1, torch.zeros_like(y1))
                r_mean = r_safe.sum() / r_cnt.clamp_min(1.0)
                r_c = torch.where(ret_fin, r_safe - r_mean, torch.zeros_like(r_safe))
                r_var = r_c.pow(2).sum() / (r_cnt - 1.0).clamp_min(1.0)
                tgt_disp = torch.log(
                    torch.sqrt(r_var.clamp_min(0.0)) + self.vol_eps
                ).clamp(-7.0, 1.0)
                pred_disp = self.dispersion_head(h_f.mean(dim=0, keepdim=True)).reshape(())
                disp_loss = torch.where(
                    r_cnt >= 2.0,
                    F.huber_loss(pred_disp, tgt_disp, delta=0.5),
                    torch.zeros((), device=device, dtype=calc_dt),
                )
            else:
                disp_loss = torch.zeros((), device=device, dtype=calc_dt)

            total = (
                self.w_masked_mod * masked_mod_loss
                + self.w_vol * vol_loss
                + self.w_dispersion * disp_loss
            )
            denom_aux = (orig_mask_bool & aux_cols).to(dtype=calc_dt).sum().clamp_min(1.0)
            masked_fraction = actually_masked.to(dtype=calc_dt).sum() / denom_aux

        return dict(
            total=total,
            masked_mod_loss=masked_mod_loss,
            vol_loss=vol_loss,
            dispersion_loss=disp_loss,
            masked_fraction=masked_fraction,
            state=out_dict.get("state"),
        )

    def pretrain_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Mapping[str, torch.Tensor],
        *,
        next_returns: torch.Tensor | None = None,
        next_vol: torch.Tensor | None = None,
        state: dict[str, object] | None = None,
        max_grad_norm: float | None = 2.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Run one zero-sync causal SSL optimization step and return loss/state dict."""
        model.train()
        self.train()
        optimizer.zero_grad(set_to_none=True)
        out = self.compute_ssl_loss(
            model,
            batch,
            next_returns=next_returns,
            next_vol=next_vol,
            state=state,
            generator=generator,
        )
        out["total"].backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            all_params = list(model.parameters()) + list(self.parameters())
            torch.nn.utils.clip_grad_norm_(
                all_params, max_norm=float(max_grad_norm), error_if_nonfinite=False
            )
        optimizer.step()
        upd_spec = getattr(model, "update_spectral_buffers", None)
        if callable(upd_spec):
            upd_spec()
        return out

    def pretrain_epochs(
        self,
        model: nn.Module,
        sessions: Sequence[Mapping[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int = 1,
        max_grad_norm: float | None = 2.0,
        seed: int = 42,
    ) -> list[torch.Tensor]:
        """Pretrain strictly chronologically across `sessions` (never shuffling time)."""
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1; got {epochs}")
        if not sessions:
            return []
        dev = sessions[0]["prices"].device
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        epoch_losses: list[torch.Tensor] = []

        for _ in range(int(epochs)):
            _reset_episodic_buffers(model)
            state: dict[str, object] | None = None
            ep_acc = torch.zeros((), device=dev, dtype=torch.float32)
            for batch in sessions:
                res = self.pretrain_step(
                    model,
                    optimizer,
                    batch,
                    state=state,
                    max_grad_norm=max_grad_norm,
                    generator=gen,
                )
                ep_acc = ep_acc + res["total"].detach().float()
                next_state = res.get("state")
                detach_fn = getattr(model, "detach_state", None)
                state = (
                    detach_fn(next_state)
                    if callable(detach_fn) and isinstance(next_state, dict)
                    else None
                )
            epoch_losses.append(ep_acc / float(len(sessions)))
        _reset_episodic_buffers(model)
        return epoch_losses


__all__ = [
    "CausalCurriculumScheduler",
    "CausalSSLPretrainer",
    "DeepNormResidual",
    "DynamicTanh",
    "LayerScale",
    "NormKind",
    "RMSNormSOTA",
    "StructuredModalityNoiseAugmenter",
    "SwiGLUFFN",
    "adaptive_gradient_clip_",
    "apply_cautious_weight_decay_",
    "apply_mup_init_and_param_groups",
    "build_norm_layer",
    "cautious_weight_decay_mask",
    "compute_swiglu_hidden_dim",
    "cross_sectional_rank_contrastive_loss",
    "deepnorm_gains",
    "deepnorm_residual",
    "run_mup_coordinate_check",
    "tail_listnet_loss",
    "vicreg_covariance_loss",
]
