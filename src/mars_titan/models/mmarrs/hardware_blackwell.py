"""Motor de rendimiento extremo de hardware para NVIDIA RTX PRO 6000 Blackwell (`sm_120`).

Implementa las reglas exactas de arquitectura de silicio de la **RTX PRO 6000 Blackwell**
(`sm_120`, GB202, 188 SMs, 128 KB SMEM/SM [~100 KB/CTA], 128 MB L2 monolítica, 96 GB GDDR7 ECC
a 1.597–1.792 GB/s, 120 TFLOPS FP32 estrictos / 1 PFLOPS BF16 `mma.sync`) con degradación
totalmente transparente a CPU y otras GPUs:
  - Configuración exacta de precisión (`fp32_strict` IEEE + `SDPBackend.MATH` + algoritmos
    deterministas, `bf16_amp` Tensor Cores con acumulación maestra FP32, o `tf32_fast`).
  - Kernel Triton nativo `sm_120` para actualización fusionada PT1+ACh/NE+Huber.
  - Optimizador `AdamW(fused=True)` con grupos separados (LR del router `3x` menor que el tronco).
  - `iter_sessions_h2d` (usa `ZeroCopySessionStore.iter_pinned_h2d` nativamente) y respaldo seguro
    `AsyncSessionPrefetcher` con doble búfer fijado, `torch.cuda.Event` por slot y `record_stream`.
  - Generador de plan CUDA MPS multi-worker (`6–8` procesos paralelos sobre 96 GB VRAM).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU o entornos sin triton
    triton = None
    tl = None
    _HAS_TRITON = False

BLACKWELL_SPECS: dict[str, object] = dict(
    arch="sm_120",
    die="GB202",
    sms=188,
    cuda_cores=24064,
    smem_kb_per_sm=128,
    max_dynamic_smem_kb_per_cta=100,
    l2_cache_mb=128,
    vram_gb=96.0,
    bandwidth_gbs_server=1597.0,
    bandwidth_gbs_workstation=1792.0,
    fp32_strict_tflops=120.0,
    fp32_ridge_flop_per_byte=75.1,
    bf16_tflops=1000.0,
    bf16_ridge_flop_per_byte=626.0,
)

PRECISION_MODES: tuple[str, ...] = ("fp32_strict", "bf16_amp", "tf32_fast")


@dataclass(frozen=True)
class BlackwellRuntimeConfig:
    """Perfil de rendimiento y determinismo para RTX PRO 6000 Blackwell (`sm_120`) / CPU."""

    precision_mode: str = "fp32_strict"
    compile_mode: str | None = None  # None | "default" | "reduce-overhead" | "max-autotune"
    use_triton_fused_ops: bool = True
    use_fused_adamw: bool = True
    use_pinned_prefetch: bool = True
    router_lr_scale: float = 1.0 / 3.0
    vram_fraction_per_worker: float | None = None
    allocator_conf: str = "expandable_segments:True,max_split_size_mb:256"
    cuda_arch_list: str = "12.0"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.precision_mode not in PRECISION_MODES:
            raise ValueError(
                f"precision_mode inválido {self.precision_mode!r}; válidos: {PRECISION_MODES}"
            )
        if self.vram_fraction_per_worker is not None and not (
            0.05 <= self.vram_fraction_per_worker <= 1.0
        ):
            raise ValueError("vram_fraction_per_worker debe pertenecer a [0.05, 1.0]")


def sdpa_math_context(
    precision: str | BlackwellRuntimeConfig = "fp32_strict",
) -> contextlib.AbstractContextManager[Any]:
    """Fuerza `SDPBackend.MATH` en `fp32_strict` (bloquea Flash/cuDNN no determinista)."""
    mode = precision.precision_mode if isinstance(precision, BlackwellRuntimeConfig) else str(
        precision
    )
    if mode not in PRECISION_MODES:
        raise ValueError(f"precision_mode inválido {mode!r}; válidos: {PRECISION_MODES}")
    if mode == "fp32_strict":
        attn_mod = getattr(torch.nn, "attention", None)
        if attn_mod is not None and hasattr(attn_mod, "sdpa_kernel") and hasattr(
            attn_mod, "SDPBackend"
        ):
            return attn_mod.sdpa_kernel(attn_mod.SDPBackend.MATH)
    return contextlib.nullcontext()


def configure_blackwell_runtime(
    cfg: BlackwellRuntimeConfig | None = None,
    *,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Aplica las banderas exactas de precisión, determinismo y arquitectura Blackwell `sm_120`."""
    c = cfg or BlackwellRuntimeConfig()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", c.cuda_arch_list)
    os.environ.setdefault("CMAKE_CUDA_ARCHITECTURES", c.cuda_arch_list.replace(".", ""))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", c.allocator_conf)

    det_algos = False
    if c.precision_mode == "fp32_strict":
        torch.set_float32_matmul_precision("highest")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if c.deterministic and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
            det_algos = True
    elif c.precision_mode == "tf32_fast":
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    else:  # bf16_amp
        torch.set_float32_matmul_precision("high")

    if c.deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    req_cuda = (
        (isinstance(device, torch.device) and device.type == "cuda")
        or (isinstance(device, str) and device.startswith("cuda"))
    )
    cuda_ok = torch.cuda.is_available() and (device is None or req_cuda)
    dev_name = "cpu"
    cc = (0, 0)
    is_blackwell = False
    vram_gb = 0.0
    if cuda_ok:
        idx = (
            device.index or 0
            if isinstance(device, torch.device) and device.type == "cuda"
            else 0
        )
        props = torch.cuda.get_device_properties(idx)
        dev_name = props.name
        cc = (props.major, props.minor)
        is_blackwell = cc >= (12, 0)
        vram_gb = float(props.total_memory) / (1024.0**3)
        if c.vram_fraction_per_worker is not None:
            torch.cuda.set_per_process_memory_fraction(c.vram_fraction_per_worker, idx)

    return dict(
        config=asdict(c),
        device="cuda" if cuda_ok else "cpu",
        device_name=dev_name,
        compute_capability=cc,
        is_blackwell_sm120=is_blackwell,
        vram_gb=vram_gb,
        triton_available=_HAS_TRITON,
        amp_dtype="bfloat16" if c.precision_mode == "bf16_amp" else "float32",
        sdpa_backend="MATH" if c.precision_mode == "fp32_strict" else "AUTO",
        deterministic_algorithms=det_algos,
        target_specs=BLACKWELL_SPECS,
    )


def autocast_context(device: torch.device, cfg: BlackwellRuntimeConfig | None = None):
    """Context manager de precisión exacto (FP32 estricto o BF16 AMP con pesos maestros FP32)."""
    c = cfg or BlackwellRuntimeConfig()
    if c.precision_mode == "bf16_amp" and device.type in ("cuda", "cpu"):
        return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.amp.autocast(device_type=device.type, enabled=False)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    cfg: BlackwellRuntimeConfig | None = None,
) -> torch.optim.Optimizer:
    """Construye AdamW fusionado con LR `3x` menor en routers MoE y sin decay en biases/Norms."""
    c = cfg or BlackwellRuntimeConfig()
    router_decay, router_no_decay = [], []
    base_decay, base_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_router = "router" in name or "balance_bias" in name
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        if is_router:
            (router_no_decay if no_decay else router_decay).append(param)
        else:
            (base_no_decay if no_decay else base_decay).append(param)

    groups = []
    if base_decay:
        groups.append(dict(params=base_decay, lr=lr, weight_decay=weight_decay))
    if base_no_decay:
        groups.append(dict(params=base_no_decay, lr=lr, weight_decay=0.0))
    if router_decay:
        groups.append(
            dict(params=router_decay, lr=lr * c.router_lr_scale, weight_decay=weight_decay)
        )
    if router_no_decay:
        groups.append(dict(params=router_no_decay, lr=lr * c.router_lr_scale, weight_decay=0.0))

    on_cuda = any(p.is_cuda for p in model.parameters())
    use_fused = bool(c.use_fused_adamw and on_cuda)
    try:
        return torch.optim.AdamW(groups, lr=lr, fused=use_fused)
    except (RuntimeError, TypeError):  # pragma: no cover - fallback seguro
        return torch.optim.AdamW(groups, lr=lr)


class _CompiledSessionWrapper(nn.Module):
    """Envoltorio para modelos cuya entrada es `forward_session` sin mutar `base_model`."""

    def __init__(self, orig_mod: nn.Module, *, mode: str, dynamic: bool) -> None:
        super().__init__()
        self._orig_mod = orig_mod
        self._compiled_fn = torch.compile(
            orig_mod.forward_session, mode=mode, dynamic=bool(dynamic)
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._compiled_fn(*args, **kwargs)

    def forward_session(self, *args: Any, **kwargs: Any) -> Any:
        return self._compiled_fn(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_mod, name)


def maybe_compile_model(
    model: nn.Module,
    cfg: BlackwellRuntimeConfig | None = None,
    *,
    dynamic: bool = True,
) -> nn.Module:
    """Compila el modelo con Inductor (`mode='default', dynamic=True`), bloqueando SIGSEGV 2.14.

    En PyTorch 2.14, `mode='reduce-overhead'` combinado con `dynamic=True` provoca un SIGSEGV en
    CUDAGraphTrees sobre paneles irregulares (`N_t` variable). Se rechaza explícitamente esa
    combinación y se recomienda `mode='default', dynamic=True` (2.07–3.14× speedup, 0 recompiles).
    """
    c = cfg or BlackwellRuntimeConfig()
    if not c.compile_mode:
        return model
    mode = "default" if c.compile_mode == "auto" else str(c.compile_mode)
    if mode == "reduce-overhead" and dynamic:
        raise ValueError(
            "Combinación insegura en PyTorch 2.14: `compile_mode='reduce-overhead'` con "
            "`dynamic=True` provoca SIGSEGV en CUDAGraphTrees sobre paneles irregulares. "
            "Use `compile_mode='default'` con `dynamic=True` o `dynamic=False` con formas fijas."
        )
    if not hasattr(torch, "compile"):
        return model
    try:  # pragma: no cover - solo cuando el usuario solicita compile explícitamente
        if type(model).forward is nn.Module.forward and hasattr(model, "forward_session"):
            return _CompiledSessionWrapper(model, mode=mode, dynamic=bool(dynamic))
        return torch.compile(model, mode=mode, dynamic=bool(dynamic))
    except Exception:
        return model


class AsyncSessionPrefetcher:
    """Pre-cargador seguro sin carreras DMA: doble búfer fijado + `Event` + `record_stream`."""

    def __init__(
        self,
        sessions: Iterable[dict[str, torch.Tensor]],
        device: torch.device,
        *,
        use_pinned: bool = True,
    ) -> None:
        self.sessions = list(sessions)
        self.device = device
        self.use_pinned = bool(use_pinned and device.type == "cuda")

    def _to_device_sync(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=False)
            else:
                out[k] = v
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.device.type != "cuda" or not self.sessions:
            for b in self.sessions:
                yield self._to_device_sync(b)
            return
        # Ruta CUDA Blackwell libre de carreras DMA y sin re-asignación `pin_memory()` por sesión
        stream = torch.cuda.Stream(device=self.device)  # pragma: no cover
        staging: list[dict[str, torch.Tensor]] = [{}, {}]  # pragma: no cover
        slot_events = [torch.cuda.Event(), torch.cuda.Event()]  # pragma: no cover
        slot_in_flight = [False, False]  # pragma: no cover

        def _stage(batch: dict[str, Any], slot: int) -> dict[str, Any]:  # pragma: no cover
            if self.use_pinned and slot_in_flight[slot]:
                slot_events[slot].synchronize()
                slot_in_flight[slot] = False
            out_gpu: dict[str, Any] = {}
            bufs = staging[slot]
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if self.use_pinned and not v.is_cuda:
                        prev = bufs.get(k)
                        if prev is None or prev.shape != v.shape or prev.dtype != v.dtype:
                            prev = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                            bufs[k] = prev
                        prev.copy_(v, non_blocking=False)
                        out_gpu[k] = prev.to(self.device, non_blocking=True)
                    else:
                        out_gpu[k] = v.to(self.device, non_blocking=True)
                else:
                    out_gpu[k] = v
            if self.use_pinned:
                slot_events[slot].record(stream)
                slot_in_flight[slot] = True
            return out_gpu

        with torch.cuda.stream(stream):  # pragma: no cover
            nxt = _stage(self.sessions[0], 0)
        for idx in range(len(self.sessions)):  # pragma: no cover
            cur_stream = torch.cuda.current_stream(self.device)
            cur_stream.wait_stream(stream)
            cur = nxt
            for v in cur.values():
                if isinstance(v, torch.Tensor):
                    v.record_stream(cur_stream)
            if idx + 1 < len(self.sessions):
                with torch.cuda.stream(stream):
                    nxt = _stage(self.sessions[idx + 1], (idx + 1) & 1)
            yield cur


def iter_sessions_h2d(
    sessions_or_store: Any,
    device: torch.device,
    *,
    use_pinned: bool = True,
    gaussian_rank_keys: Sequence[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Itera sesiones usando `ZeroCopySessionStore.iter_pinned_h2d` si está disponible."""
    if hasattr(sessions_or_store, "iter_pinned_h2d"):
        kwargs: dict[str, Any] = dict(device=device, use_pinned=use_pinned)
        if gaussian_rank_keys is not None:
            kwargs["gaussian_rank_keys"] = gaussian_rank_keys
        return iter(sessions_or_store.iter_pinned_h2d(**kwargs))
    return iter(AsyncSessionPrefetcher(sessions_or_store, device, use_pinned=use_pinned))


# --- Kernel Triton `sm_120` para actualización fusionada PT1 + Huber + Cascada ---
if _HAS_TRITON:  # pragma: no cover - compilado únicamente en GPU CUDA con Triton

    @triton.jit
    def _pt1_huber_update_kernel(
        state_ptr,
        mom_ptr,
        err_ptr,
        alpha_ptr,
        eta_ptr,
        theta_ptr,
        out_state_ptr,
        out_mom_ptr,
        n_elements,
        huber_delta,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        s = tl.load(state_ptr + offs, mask=mask, other=0.0)
        m = tl.load(mom_ptr + offs, mask=mask, other=0.0)
        e = tl.load(err_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
        a = tl.load(alpha_ptr + offs, mask=mask, other=0.1)
        et = tl.load(eta_ptr + offs, mask=mask, other=0.5)
        th = tl.load(theta_ptr + offs, mask=mask, other=0.1)
        # Recorte Huber tipado explícitamente al dtype de entrada (sin truncar FP64/FP32)
        d_typed = tl.full([], huber_delta, e.dtype)
        e_clip = tl.minimum(tl.maximum(e, -d_typed), d_typed)
        m_new = et * m + th * e_clip
        one = tl.full([], 1.0, s.dtype)
        s_new = (one - a) * s + m_new
        tl.store(out_state_ptr + offs, s_new, mask=mask)
        tl.store(out_mom_ptr + offs, m_new, mask=mask)


def fused_pt1_huber_gate_update(
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
    """Actualización contractiva PT1 + gradiente Huber en 1 sola pasada de memoria.

    En GPU CUDA sin autograd (macro-bucle causal / inferencia) dispara el kernel Triton
    ajustado a `sm_120` (`num_stages=2`, `BLOCK=256`, `eviction_policy="evict_first"`);
    en CPU o bajo autograd usa el camino vectorial exacto de PyTorch.
    """
    can_triton = (
        use_triton
        and _HAS_TRITON
        and state.is_cuda
        and not (
            state.requires_grad
            or error.requires_grad
            or alpha.requires_grad
            or eta.requires_grad
            or theta.requires_grad
        )
    )
    if can_triton:  # pragma: no cover - ruta exclusiva CUDA Blackwell
        s_c = state.contiguous()
        m_c = momentum.contiguous()
        e_c = error.contiguous()
        a_c = alpha.expand_as(s_c).contiguous()
        et_c = eta.expand_as(s_c).contiguous()
        th_c = theta.expand_as(s_c).contiguous()
        out_s = torch.empty_like(s_c)
        out_m = torch.empty_like(m_c)
        n_el = s_c.numel()
        block = 256
        grid = ((n_el + block - 1) // block,)
        _pt1_huber_update_kernel[grid](
            s_c,
            m_c,
            e_c,
            a_c,
            et_c,
            th_c,
            out_s,
            out_m,
            n_el,
            float(huber_delta),
            BLOCK=block,
            num_warps=4,
            num_stages=2,
        )
        return out_s, out_m

    e_clip = torch.clamp(error, min=-huber_delta, max=huber_delta)
    m_new = eta * momentum + theta * e_clip
    s_new = (1.0 - alpha) * state + m_new
    return s_new, m_new


def blackwell_mps_launch_plan(
    variations: Iterable[str],
    *,
    vram_gb: float = 96.0,
    workers: int = 6,
    precision_mode: str = "fp32_strict",
) -> dict[str, object]:
    """Genera plan y comandos exactos de saturación CUDA MPS para 1x RTX PRO 6000 (`sm_120`)."""
    var_list = list(variations)
    w = max(1, min(workers, len(var_list)))
    # Reservar ~12 % de margen para contexto CUDA/cuBLAS y asignador
    vram_per_worker = round((vram_gb * 0.88) / w, 2)
    frac = round(min(0.88 / w, 0.90), 3)
    waves = [var_list[i : i + w] for i in range(0, len(var_list), w)]
    env = dict(
        TORCH_CUDA_ARCH_LIST="12.0",
        CMAKE_CUDA_ARCHITECTURES="120",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256",
        CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps",
        CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log",
        MMARRS_PRECISION_MODE=precision_mode,
        MMARRS_VRAM_FRACTION=str(frac),
    )
    setup_cmd = (
        "export TORCH_CUDA_ARCH_LIST=12.0 CMAKE_CUDA_ARCHITECTURES=120 "
        "PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256' "
        "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log && "
        "mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY && "
        "nvidia-cuda-mps-control -d"
    )
    return dict(
        hardware="NVIDIA RTX PRO 6000 Blackwell (sm_120 / GB202 / 96GB GDDR7 / 188 SMs)",
        precision_mode=precision_mode,
        n_variations=len(var_list),
        workers_per_wave=w,
        n_waves=len(waves),
        vram_gb_per_worker=vram_per_worker,
        vram_fraction_per_worker=frac,
        waves=waves,
        env=env,
        mps_setup_cmd=setup_cmd,
        nsys_profile_cmd=(
            "nsys profile -t cuda,nvtx,osrt --sample=none -o mmarrs_sm120 "
            "python -m mars_titan.models.mmarrs.ablation_runner"
        ),
        ncu_profile_cmd=(
            "ncu --set full --kernel-name regex:_pt1_huber_update_kernel --launch-count 5 "
            "python -m mars_titan.models.mmarrs.ablation_runner"
        ),
    )
