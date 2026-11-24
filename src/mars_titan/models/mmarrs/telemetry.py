"""Observabilidad y telemetría de investigación para pre-entrenamiento y ablaciones `MMARRS-TITAN`.

Diseñado explícitamente para **cero bloqueos GPU↔CPU en el bucle interno** de RTX PRO 6000
Blackwell (`sm_120`):
  - En pasos no muestreados (`step % log_every_steps != 0`), todos los monitores y los hooks de
    `ActivationProbe` retornan inmediatamente sin lanzar kernels ni sincronizaciones `.item()`.
  - En pasos muestreados, las estadísticas de pesos, gradientes pre/post-clip, ratio Karpathy
    `||ΔW|| / ||W||`, estado efectivo de `AdamW`, pérdidas y activaciones se reducen en dispositivo
    y se transfieren a host en el mínimo número de copias agrupadas.
  - Emite salida estructurada sin dependencias externas (`telemetry.jsonl` append-only en JSON
    estricto RFC 8259), integración opcional con `torch.utils.tensorboard.SummaryWriter` con
    degradación limpia cuando TensorBoard no está instalado, e informe completo `summarize_run()`.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .cognitive_core import TOOL_NAMES, TOOL_NAMES_V2

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except Exception:  # pragma: no cover - entornos sin paquete tensorboard
    _SummaryWriter = None


ARCHITECTURAL_ORGANS: tuple[str, ...] = (
    "encoders",
    "holographic_3l",
    "hypergraph",
    "neuromod_memory",
    "hrr_hopfield",
    "cognitive_core",
    "moe_routers",
    "moe_experts",
    "contractive_loop",
    "granular_heads",
    "dendritic",
    "lateral_inhibition",
    "three_factor",
    "replay",
    "cross_modal_fusion",
    "ssl_alignment",
    "skew_student_t",
)

_ANOMALY_FLAG_CODES: tuple[str, ...] = (
    "vanishing_gradients",
    "exploding_gradients",
    "nonfinite_gradients",
    "nonfinite_weights",
    "dead_updates",
    "unstable_updates",
    "dead_routers",
    "expert_collapse",
    "activation_saturation",
    "nonfinite_activations",
    "pt1_bound_violation",
)

_DOMAIN_MEAN_KEYS: tuple[str, ...] = (
    "moe_router_entropy",
    "moe_load_max_min_ratio",
    "moe_expert_mean_corr",
    "loop_k_used",
    "loop_delta_energy_neg_pct",
    "loop_halt_at_k1_pct",
    "cf_fragility_final",
    "cf_sign_flip_rate_final",
    "hrr_ood_energy_mean",
    "cognitive_pruned_context_ratio",
    "cognitive_tool_entropy",
    "cognitive_tool_load_max_min_ratio",
    "cognitive_tool_active_count",
)

_LOSS_CANONICAL_KEYS: tuple[str, ...] = (
    "total",
    "ic",
    "crps",
    "tb",
    "energy",
    "sharpe_turnover",
    "decor",
)

_ACTIVATION_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.SiLU,
    nn.ReLU,
    nn.GELU,
    nn.Tanh,
    nn.Sigmoid,
    nn.ELU,
    nn.LeakyReLU,
)

_ORGAN_PROBE_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "_ModalityEncoder",
        "HighSNRMoELayer",
        "_ContractiveStepBlock",
        "LoopedCrossAssetTransformer",
        "InducedFactorCrossAttention",
        "_WeightTiedContractiveIntegrator",
        "HolographicBrain3L",
        "RMTSpilloverHypergraph",
        "DualAxisTitansMemory",
        "MultiHorizonGranularHead",
        "TitansCognitiveCore",
        "ContextEngineerGPU",
        "_WeightTiedContractiveCore",
        "LateralInhibitionCompetition",
        "DendriticCompartmentLayer",
        "ThreeFactorDopaminePlasticity",
        "TurrigianoHomeostaticScaling",
        "ActiveInferencePrecisionGate",
        "HippocampalSharpWaveReplay",
        "_ContractivePredictiveCodingCycle",
        "CrossModalConflictFusion",
        "SkewStudentTDistributionHead",
    }
)

_NAMED_BOUNDARY_MODULES: frozenset[str] = frozenset(
    {
        "fusion_norm",
        "neuromod_memory.out_proj",
        "hrr_hopfield.out_proj",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _strip_wrapper_prefix(name: str) -> str:
    """Elimina prefijos de compilación/DDP (`_orig_mod.`, `module.`) preservando jerarquía."""
    clean = name
    for prefix in ("_orig_mod.", "module."):
        while clean.startswith(prefix):
            clean = clean[len(prefix) :]
    return clean


def classify_param_organ(name: str) -> str:
    """Clasifica el nombre de un parámetro o módulo en su órgano arquitectónico canónico."""
    clean = _strip_wrapper_prefix(name)
    if not clean:
        return "other"

    parts = clean.split(".")
    head = parts[0]

    # 1) Encoders multimodales, física v4, bypass TabM y proyección/fusión inicial
    if head.startswith("enc_") or head in (
        "regime_proj",
        "fusion_norm",
        "fusion_proj",
        "target_encoder",
        "tabm_skip",
    ):
        return "encoders"

    # 2) Núcleo cognitivo (comprobar antes de hypergraph/router por sub-módulos internos)
    if head in (
        "cognitive_core",
        "context_engineer",
        "tool_router",
        "toolbox",
        "core_cell",
        "halt_gate",
        "scratchpad_init_notes",
    ):
        return "cognitive_core"

    # 3) Cerebro holográfico 3L (incluye `holographic_3l.hypergraph_l2.*`)
    if head == "holographic_3l":
        return "holographic_3l"

    # 4) Hipergrafo transversal RMT de nivel superior
    if head == "hypergraph":
        return "hypergraph"

    # 5) Memoria Titans neuromodulada ACh/NE + PT1 (v1 + DualAxisTitansMemory v2)
    if head == "neuromod_memory":
        return "neuromod_memory"

    # 6) Memoria asociativa HRR-Modern-Hopfield
    if head == "hrr_hopfield":
        return "hrr_hopfield"

    # 6b) Familia 2 Neuro-AI (`neuro_brain_sota.py` v1 + v2)
    if head == "dendritic" or head.startswith("dendritic_"):
        return "dendritic"
    if head == "lateral_inhibition" or head.startswith("lateral_inhib"):
        return "lateral_inhibition"
    if (
        head in ("three_factor", "three_factor_plasticity", "homeostatic_scaling")
        or head.startswith("three_factor")
        or head.startswith("homeostatic_")
    ):
        return "three_factor"
    if head in ("replay", "replay_memory", "hippocampal_replay") or head.startswith("replay_"):
        return "replay"
    if head == "active_inference_pi" or head.startswith("active_inference"):
        return "contractive_loop"

    # 6c) Fusión multimodal SOTA, alineación SSL y cabeza Skew Student-t (`multimodal_sota.py`)
    if head in ("cross_modal_fusion", "conflict_fusion", "multimodal_fusion"):
        return "cross_modal_fusion"
    if head in ("ssl_alignment", "ssl_loss", "multimodal_ssl", "alignment_loss"):
        return "ssl_alignment"
    if head in ("skew_student_t", "skew_t_head", "dist_head", "distribution_head"):
        return "skew_student_t"

    # 7) Cabezas granulares multi-horizonte y verificador financiero
    if head in ("head", "verifier"):
        return "granular_heads"

    # 8) Tronco recursivo: separar routers MoE, expertos MoE y pasos contractivos / FFN denso
    if head == "pc_cycle":
        return "contractive_loop"
    if head == "moe_bridges":
        tail = ".".join(parts[2:]) if len(parts) > 2 else ""
        if tail.startswith("router") or tail.startswith("ctx_norm") or "balance_bias" in tail:
            return "moe_routers"
        if (
            tail.startswith("shared_")
            or tail.startswith("routed_")
            or tail == "mix_gate"
            or tail.startswith("in_norm")
        ):
            return "moe_experts"
        return "contractive_loop"

    if head == "recursive_core":
        if len(parts) >= 2 and parts[1] == "step_blocks":
            tail = ".".join(parts[3:]) if len(parts) > 3 else ""
            if (
                tail.startswith("router")
                or tail.startswith("ctx_norm")
                or "balance_bias" in tail
            ):
                return "moe_routers"
            if (
                tail.startswith("shared_")
                or tail.startswith("routed_")
                or tail == "mix_gate"
                or tail.startswith("in_norm")
            ):
                return "moe_experts"
            # Bloques densos (`nn.Sequential`) en variaciones sin MoE o módulo padre
            return "contractive_loop"
        if len(parts) >= 2 and parts[1] == "step_routers":
            return "moe_routers"
        if len(parts) >= 2 and parts[1] == "shared_moe":
            tail = ".".join(parts[2:]) if len(parts) > 2 else ""
            if (
                tail.startswith("router")
                or tail.startswith("ctx_norm")
                or "balance_bias" in tail
            ):
                return "moe_routers"
            return "moe_experts"
        return "contractive_loop"

    # 8b) `LoopedCrossAssetTransformer` independiente (sin prefijo `recursive_core.`)
    if head == "step_routers":
        return "moe_routers"
    if head == "shared_moe":
        tail = ".".join(parts[1:]) if len(parts) > 1 else ""
        if tail.startswith("router") or tail.startswith("ctx_norm") or "balance_bias" in tail:
            return "moe_routers"
        return "moe_experts"
    if head in (
        "step_embed",
        "step_emb_proj",
        "step_film",
        "x0_inj",
        "inj_gate",
        "u_norm",
        "feedback_proj",
        "feedback_gate",
        "cross_asset_attn",
        "contractive_integrator",
        "shared_ffn",
        "cf_sim",
    ):
        return "contractive_loop"

    # 9) Capas `HighSNRMoELayer` usadas de forma independiente
    if head in ("router", "ctx_norm", "balance_bias"):
        return "moe_routers"
    if head.startswith(("shared_", "routed_")) or head in ("mix_gate", "in_norm"):
        return "moe_experts"

    return "other"


classify_organ = classify_param_organ


def _json_sanitize(obj: Any) -> Any:
    """Convierte valores arbitrarios a tipos JSON estrictos (NaN/Inf -> `None`)."""
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            v = float(obj.detach().cpu().item())
            return v if math.isfinite(v) else None
        return _json_sanitize(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_sanitize(v) for v in obj]
    # Compatibilidad con escalares NumPy sin requerir importación explícita de numpy
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_sanitize(obj.item())
        except Exception:
            pass
    return str(obj)


def _to_float_scalar(val: Any) -> float | None:
    """Convierte un escalar Python o tensor 0-D a `float` (solo usado en pasos muestreados)."""
    if val is None:
        return None
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, int | float):
        return float(val)
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return float(val.detach().float().cpu().item())
        return None
    if hasattr(val, "item") and callable(val.item):
        try:
            return float(val.item())
        except Exception:
            return None
    return None


def _spectral_norm_approx(tensor: torch.Tensor, *, n_iters: int = 4) -> torch.Tensor:
    """Aproximación determinista de norma espectral por iteración de potencia en dispositivo."""
    t_f = tensor.detach().float()
    if t_f.numel() == 0:
        return t_f.new_zeros(())
    t_clean = torch.where(torch.isfinite(t_f), t_f, torch.zeros_like(t_f))
    if t_clean.ndim <= 1:
        return t_clean.norm()

    if t_clean.ndim == 3:
        b, _, n = t_clean.shape
        v = torch.linspace(0.5, 1.5, n, device=t_clean.device, dtype=torch.float32)
        v = (v / v.norm().clamp_min(1e-12)).view(1, n, 1).expand(b, n, 1)
        u = t_clean.new_zeros((b, t_clean.shape[1], 1))
        for _ in range(n_iters):
            u = torch.bmm(t_clean, v)
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            v = torch.bmm(t_clean.transpose(1, 2), u)
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        mv = torch.bmm(t_clean, v)
        return (u * mv).sum(dim=(1, 2)).abs().max()

    mat = t_clean if t_clean.ndim == 2 else t_clean.reshape(t_clean.shape[0], -1)
    n_cols = mat.shape[1]
    v = torch.linspace(0.5, 1.5, n_cols, device=mat.device, dtype=torch.float32)
    v = v / v.norm().clamp_min(1e-12)
    u = mat.new_zeros((mat.shape[0],))
    for _ in range(n_iters):
        u = mat @ v
        u = u / u.norm().clamp_min(1e-12)
        v = mat.T @ u
        v = v / v.norm().clamp_min(1e-12)
    return torch.dot(u, mat @ v).abs()


def _batch_scalars_to_floats(tensors: list[torch.Tensor]) -> list[float]:
    """Transfiere una lista de tensores escalares agrupados por dispositivo en un solo sync."""
    if not tensors:
        return []
    out: list[float] = [0.0] * len(tensors)
    by_dev: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for idx, t in enumerate(tensors):
        by_dev.setdefault(t.device, []).append((idx, t.detach().float().reshape(())))
    for items in by_dev.values():
        stacked = torch.stack([it[1] for it in items]).cpu().tolist()
        for (idx, _), val in zip(items, stacked, strict=True):
            out[idx] = float(val)
    return out


def _batch_vectors_to_lists(tensors: list[torch.Tensor]) -> list[list[float]]:
    """Concatena tensores 1D heterogéneos por dispositivo en 1 único `.cpu().tolist()`."""
    if not tensors:
        return []
    out: list[list[float]] = [[] for _ in tensors]
    by_dev: dict[torch.device, list[tuple[int, torch.Tensor, int]]] = {}
    for idx, t in enumerate(tensors):
        flat = t.detach().float().reshape(-1)
        n = int(flat.numel())
        if n > 0:
            by_dev.setdefault(flat.device, []).append((idx, flat, n))
    for items in by_dev.values():
        cat_vals = torch.cat([it[1] for it in items], dim=0).cpu().tolist()
        cursor = 0
        for idx, _, n in items:
            out[idx] = [float(x) for x in cat_vals[cursor : cursor + n]]
            cursor += n
    return out


def _is_gzip_file(path: Path) -> bool:
    """Detecta compresión gzip por bytes mágicos `\x1f\x8b` (o extensión `.gz` si vacío)."""
    if path.is_file() and path.stat().st_size >= 2:
        with path.open("rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    return path.suffix == ".gz"


def read_telemetry_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Lee de forma transparente trazas append-only `.jsonl` o comprimidas `.jsonl.gz`."""
    p = Path(path)
    if not p.is_file():
        return []
    records: list[dict[str, Any]] = []
    opener = gzip.open if _is_gzip_file(p) else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Línea JSONL corrupta en {p} línea {line_no}: {exc}") from exc
    return records


@dataclass(frozen=True)
class TelemetryConfig:
    """Contrato inmutable de telemetría de investigación con muestreo libre de bloqueos GPU."""

    log_dir: str
    log_every_steps: int = 10
    histogram_every_steps: int = 100
    spectral_every_k: int = 50
    max_history_records: int = 2048
    track_weights: bool = True
    track_gradients: bool = True
    track_update_ratios: bool = True
    track_activations: bool = True
    enable_tensorboard: bool = True
    enable_jsonl: bool = True
    grad_explode_threshold: float = 50.0
    grad_vanish_threshold: float = 1e-7
    update_ratio_target: float = 1e-3
    update_ratio_dead_threshold: float = 1e-6
    update_ratio_unstable_threshold: float = 1e-1
    dead_activation_threshold: float = 0.50
    saturation_threshold: float = 0.90
    dead_router_entropy_threshold: float = 0.25
    moe_imbalance_alert_threshold: float = 3.0
    moe_corr_alert_threshold: float = 0.85
    zero_eps: float = 1e-12
    compress_jsonl: bool = True
    jsonl_filename: str = "telemetry.jsonl.gz"

    def __post_init__(self) -> None:
        if isinstance(self.log_dir, Path):
            object.__setattr__(self, "log_dir", str(self.log_dir))
        if not self.compress_jsonl and self.jsonl_filename == "telemetry.jsonl.gz":
            object.__setattr__(self, "jsonl_filename", "telemetry.jsonl")
        elif self.compress_jsonl and self.jsonl_filename == "telemetry.jsonl":
            object.__setattr__(self, "jsonl_filename", "telemetry.jsonl.gz")
        if not self.log_dir or not str(self.log_dir).strip():
            raise ValueError("log_dir no puede estar vacío")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps debe ser >= 1")
        if self.histogram_every_steps < 1:
            raise ValueError("histogram_every_steps debe ser >= 1")
        if self.spectral_every_k < 1:
            raise ValueError("spectral_every_k debe ser >= 1")
        if self.max_history_records < 1:
            raise ValueError("max_history_records debe ser >= 1")
        if self.grad_explode_threshold <= 0.0:
            raise ValueError("grad_explode_threshold debe ser > 0")
        if self.grad_vanish_threshold < 0.0:
            raise ValueError("grad_vanish_threshold debe ser >= 0")
        if self.grad_vanish_threshold >= self.grad_explode_threshold:
            raise ValueError("grad_vanish_threshold debe ser < grad_explode_threshold")
        if self.update_ratio_target <= 0.0:
            raise ValueError("update_ratio_target debe ser > 0")
        if self.update_ratio_dead_threshold <= 0.0:
            raise ValueError("update_ratio_dead_threshold debe ser > 0")
        if self.update_ratio_unstable_threshold <= self.update_ratio_dead_threshold:
            raise ValueError(
                "update_ratio_unstable_threshold debe ser > update_ratio_dead_threshold"
            )
        if not (0.0 < self.dead_activation_threshold <= 1.0):
            raise ValueError("dead_activation_threshold debe pertenecer a (0, 1]")
        if not (0.0 < self.saturation_threshold <= 1.0):
            raise ValueError("saturation_threshold debe pertenecer a (0, 1]")
        if self.moe_imbalance_alert_threshold < 1.0:
            raise ValueError("moe_imbalance_alert_threshold debe ser >= 1.0")
        if not (0.0 < self.moe_corr_alert_threshold <= 1.0):
            raise ValueError("moe_corr_alert_threshold debe pertenecer a (0, 1]")
        if self.zero_eps < 0.0:
            raise ValueError("zero_eps debe ser >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuración completa como diccionario JSON-compatible."""
        return asdict(self)


@dataclass(frozen=True)
class ModuleWeightGradSnapshot:
    """Instantánea estructurada de pesos, gradientes pre/post-clip y ratio Karpathy por órgano."""

    organ: str
    n_params: int
    n_tensors: int
    weight_l2: float = 0.0
    weight_rms: float = 0.0
    weight_abs_max: float = 0.0
    weight_std: float = 0.0
    weight_spectral_approx: float = 0.0
    weight_zero_fraction: float = 0.0
    weight_nan_count: int = 0
    weight_inf_count: int = 0
    grad_pre_clip_l2: float | None = None
    grad_post_clip_l2: float | None = None
    grad_abs_max: float | None = None
    grad_nan_count: int = 0
    grad_inf_count: int = 0
    grad_vanishing: bool = False
    grad_exploding: bool = False
    update_l2: float | None = None
    update_to_weight_ratio: float | None = None
    update_ratio_log10: float | None = None
    update_dead: bool = False
    update_unstable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convierte la instantánea a diccionario plano."""
        return asdict(self)


class ActivationProbe:
    """Sonda ligera de activaciones con hooks muestreados (coste cero en pasos no activos).

    Cuando `set_active(False)` está vigente, el hook ejecuta una única comprobación booleana en
    Python y retorna de inmediato sin tocar tensores ni lanzar kernels CUDA. En pasos activos,
    acumula reducciones escalares en dispositivo y las transfiere en un único `.cpu().tolist()`
    cuando se invoca `collect()`.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        module_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._active: bool = False
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._hooked_modules: list[str] = []
        self._module_organs: dict[str, str] = {}
        self._step_buffers: dict[str, list[tuple[torch.Tensor, ...]]] = {}
        self._active_captures: int = 0

        targets = self._select_modules(model, module_names=module_names)
        for mod_name, mod in targets:
            clean_name = _strip_wrapper_prefix(mod_name)
            self._hooked_modules.append(clean_name)
            self._module_organs[clean_name] = classify_param_organ(clean_name)
            handle = mod.register_forward_hook(self._make_hook(clean_name))
            self._handles.append(handle)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def active_captures(self) -> int:
        return self._active_captures

    @property
    def hooked_modules(self) -> tuple[str, ...]:
        return tuple(self._hooked_modules)

    @staticmethod
    def _select_modules(
        model: nn.Module,
        *,
        module_names: list[str] | tuple[str, ...] | None = None,
    ) -> list[tuple[str, nn.Module]]:
        if module_names is not None:
            wanted = set(module_names)
            return [
                (name, mod)
                for name, mod in model.named_modules()
                if name in wanted or _strip_wrapper_prefix(name) in wanted
            ]

        selected: list[tuple[str, nn.Module]] = []
        for name, mod in model.named_modules():
            if not name:
                continue
            clean = _strip_wrapper_prefix(name)
            cls_name = mod.__class__.__name__
            if (
                isinstance(mod, _ACTIVATION_MODULE_TYPES)
                or cls_name in _ORGAN_PROBE_CLASS_NAMES
                or clean in _NAMED_BOUNDARY_MODULES
            ):
                selected.append((name, mod))

        if not selected:
            for name, mod in model.named_modules():
                if name and isinstance(mod, nn.Linear):
                    selected.append((name, mod))
        return selected

    @staticmethod
    def _extract_tensor(obj: Any) -> torch.Tensor | None:
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, tuple | list):
            for item in obj:
                if isinstance(item, torch.Tensor):
                    return item
            return None
        if isinstance(obj, dict):
            for key in ("pred_return", "retrieved", "alpha", "final_hidden"):
                val = obj.get(key)
                if isinstance(val, torch.Tensor):
                    return val
            for val in obj.values():
                if isinstance(val, torch.Tensor):
                    return val
        return None

    def _make_hook(self, clean_name: str):
        def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self._active:
                return
            out_t = self._extract_tensor(output)
            if out_t is None or out_t.numel() == 0 or not out_t.dtype.is_floating_point:
                return
            with torch.no_grad():
                t = out_t.detach().float()
                nan_c = torch.isnan(t).sum()
                inf_c = torch.isinf(t).sum()
                finite = torch.isfinite(t)
                t_clean = torch.where(finite, t, torch.zeros_like(t))
                abs_t = t_clean.abs()

                mean_t = t_clean.mean()
                std_t = t_clean.std(unbiased=False)
                abs_max_t = abs_t.max()
                zero_frac = (abs_t <= 1e-6).float().mean()

                pre_t = self._extract_tensor(inputs) if inputs else None
                if isinstance(module, nn.SiLU):
                    if pre_t is not None and pre_t.shape == t.shape:
                        pre_f = pre_t.detach().float()
                        dead_frac = ((pre_f < -5.0) | (abs_t <= 1e-6)).float().mean()
                    else:
                        dead_frac = ((t_clean <= 0.0) & (abs_t < 1e-2)).float().mean()
                    sat_frac = ((abs_t > 50.0) | (~finite)).float().mean()
                elif isinstance(module, nn.ReLU):
                    dead_frac = (t_clean <= 1e-6).float().mean()
                    sat_frac = ((abs_t > 50.0) | (~finite)).float().mean()
                elif isinstance(module, nn.GELU):
                    if pre_t is not None and pre_t.shape == t.shape:
                        pre_f = pre_t.detach().float()
                        dead_frac = ((pre_f < -4.0) | (abs_t <= 1e-6)).float().mean()
                    else:
                        dead_frac = zero_frac
                    sat_frac = ((abs_t > 50.0) | (~finite)).float().mean()
                elif isinstance(module, nn.Tanh):
                    dead_frac = zero_frac
                    sat_frac = ((abs_t > 0.95) | (~finite)).float().mean()
                elif isinstance(module, nn.Sigmoid):
                    dead_frac = zero_frac
                    sat_frac = (((t_clean < 0.05) | (t_clean > 0.95)) | (~finite)).float().mean()
                else:
                    dead_frac = zero_frac
                    sat_frac = ((abs_t > 50.0) | (~finite)).float().mean()

                self._step_buffers.setdefault(clean_name, []).append(
                    (mean_t, std_t, abs_max_t, zero_frac, dead_frac, sat_frac, nan_c, inf_c)
                )
                self._active_captures += 1

        return _hook

    def set_active(self, active: bool) -> None:
        """Activa o desactiva la recolección en los forward hooks."""
        self._active = bool(active)
        if self._active:
            self._step_buffers.clear()

    def clear(self) -> None:
        """Descarta las activaciones acumuladas del paso actual."""
        self._step_buffers.clear()

    def collect(self) -> dict[str, dict[str, Any]]:
        """Transfiere y agrega todas las estadísticas de activación capturadas en un solo sync."""
        if not self._step_buffers:
            return {}

        flat_tensors: list[torch.Tensor] = []
        spans: list[tuple[str, int]] = []
        for mod_name, entries in self._step_buffers.items():
            spans.append((mod_name, len(entries)))
            for tup in entries:
                flat_tensors.extend(tup)

        vals = _batch_scalars_to_floats(flat_tensors)
        result: dict[str, dict[str, Any]] = {}
        cursor = 0
        for mod_name, count in spans:
            chunk = vals[cursor : cursor + 8 * count]
            cursor += 8 * count
            means = chunk[0::8]
            stds = chunk[1::8]
            abs_maxs = chunk[2::8]
            zeros = chunk[3::8]
            deads = chunk[4::8]
            sats = chunk[5::8]
            nans = chunk[6::8]
            infs = chunk[7::8]
            result[mod_name] = dict(
                organ=self._module_organs.get(mod_name, "other"),
                n_calls=count,
                mean=sum(means) / count,
                std=sum(stds) / count,
                abs_max=max(abs_maxs),
                zero_fraction=sum(zeros) / count,
                dead_fraction=sum(deads) / count,
                saturation_fraction=sum(sats) / count,
                nan_count=int(round(sum(nans))),
                inf_count=int(round(sum(infs))),
            )

        self._step_buffers.clear()
        return result

    def remove(self) -> None:
        """Desregistra todos los forward hooks de forma segura e idempotente."""
        self._active = False
        self._step_buffers.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class ResearchTelemetryMonitor:
    """Monitor integral de observabilidad e investigación para `MMARRS-TITAN`."""

    def __init__(
        self,
        model: nn.Module,
        config: TelemetryConfig,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        tb_writer: Any | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

        self._log_dir = Path(config.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._log_dir / config.jsonl_filename

        # Agrupar parámetros por órgano arquitectónico en orden determinista
        self._param_id_to_name: dict[int, str] = {}
        self._organ_params: dict[str, list[tuple[str, nn.Parameter]]] = {}
        self._refresh_param_index()

        # TensorBoard opcional con degradación totalmente limpia si no está instalado
        self._tb_writer: Any | None = None
        self._owns_tb_writer: bool = False
        if tb_writer is not None:
            self._tb_writer = tb_writer
            self._owns_tb_writer = True
        elif config.enable_tensorboard and _SummaryWriter is not None:  # pragma: no cover
            try:
                self._tb_writer = _SummaryWriter(log_dir=str(self._log_dir / "tensorboard"))
                self._owns_tb_writer = True
            except Exception:
                self._tb_writer = None
                self._owns_tb_writer = False

        # Sonda de activaciones muestreada
        self._activation_probe: ActivationProbe | None = (
            ActivationProbe(model) if config.track_activations else None
        )

        # Estado de paso e historial ligero para `summarize_run()`
        self._total_steps_seen: int = 0
        self._sessions_since_last_log: int = 0
        self._last_log_wall_time: float | None = None
        self._step_start_time: float | None = None
        self._pre_step_weights: dict[str, list[torch.Tensor]] = {}
        self._pre_step_organ_has_grad: dict[str, bool] = {}
        self._pre_clip_grad_stats: dict[str, dict[str, Any]] = {}
        self._pre_clip_global: dict[str, Any] = {}
        self._post_clip_grad_stats: dict[str, dict[str, Any]] = {}
        self._post_clip_global: dict[str, Any] = {}
        self._captured_pre_step: int | None = None
        self._captured_post_clip_step: int | None = None

        self._last_snapshots: dict[str, ModuleWeightGradSnapshot] = {}
        self._last_spectral_approx: dict[str, float] = {}
        self._spectral_eval_steps: int = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=config.max_history_records)
        self._anomalies: deque[dict[str, Any]] = deque(
            maxlen=max(config.max_history_records * 4, 2048)
        )

        # Acumuladores online O(1) RAM para que `summarize_run()` sea exacto en 120k+ pasos
        self._n_logged_steps: int = 0
        self._last_logged_step: int | None = None
        self._total_anomalies_seen: int = 0
        self._anomaly_flags: dict[str, bool] = {code: False for code in _ANOMALY_FLAG_CODES}
        self._affected_sets: dict[str, set[str]] = {code: set() for code in _ANOMALY_FLAG_CODES}
        self._organ_online: dict[str, dict[str, Any]] = {}
        self._loss_online: dict[str, dict[str, Any]] = {}
        self._last_opt_telemetry: dict[str, Any] = {}
        self._dom_mean_online: dict[str, list[float]] = {k: [0.0, 0.0] for k in _DOMAIN_MEAN_KEYS}
        self._dom_pt1_alpha_min: float | None = None
        self._dom_pt1_eta_max: float | None = None
        self._dom_pt1_theta_max: float | None = None
        self._hw_peak_vram_alloc_mb: float = 0.0
        self._hw_peak_vram_res_mb: float = 0.0
        self._hw_tput_sum: float = 0.0
        self._hw_tput_count: int = 0

    def __enter__(self) -> ResearchTelemetryMonitor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    @property
    def tensorboard_active(self) -> bool:
        return self._tb_writer is not None

    @property
    def activation_probe(self) -> ActivationProbe | None:
        return self._activation_probe

    @property
    def last_snapshots(self) -> dict[str, ModuleWeightGradSnapshot]:
        return dict(self._last_snapshots)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def _refresh_param_index(self) -> None:
        self._param_id_to_name.clear()
        grouped: dict[str, list[tuple[str, nn.Parameter]]] = {}
        for name, param in self.model.named_parameters():
            self._param_id_to_name[id(param)] = _strip_wrapper_prefix(name)
            organ = classify_param_organ(name)
            grouped.setdefault(organ, []).append((name, param))

        ordered: dict[str, list[tuple[str, nn.Parameter]]] = {}
        for organ in ARCHITECTURAL_ORGANS:
            if organ in grouped:
                ordered[organ] = grouped[organ]
        for organ, plist in grouped.items():
            if organ not in ordered:
                ordered[organ] = plist
        self._organ_params = ordered

    def should_log(self, step: int) -> bool:
        """Indica si `step` corresponde a un paso muestreado de telemetría."""
        return int(step) % self.config.log_every_steps == 0

    def should_log_histograms(self, step: int) -> bool:
        """Indica si `step` debe emitir histogramas TensorBoard."""
        return int(step) % self.config.histogram_every_steps == 0

    def should_log_spectral(self, step: int) -> bool:
        """Indica si `step` debe recalcular la aproximación espectral por iteración de potencia."""
        return int(step) % self.config.spectral_every_k == 0

    def begin_step(self, step: int) -> None:
        """Prepara el paso: activa hooks de activación SOLO si `step` es muestreado."""
        is_logged = self.should_log(step)
        if self._activation_probe is not None:
            self._activation_probe.set_active(is_logged)
        if not is_logged:
            if self._last_log_wall_time is None:
                self._last_log_wall_time = time.perf_counter()
            return
        now = time.perf_counter()
        self._step_start_time = now
        if self._last_log_wall_time is None:
            self._last_log_wall_time = now
        self._pre_step_weights.clear()
        self._pre_step_organ_has_grad.clear()
        self._pre_clip_grad_stats.clear()
        self._pre_clip_global.clear()
        self._post_clip_grad_stats.clear()
        self._post_clip_global.clear()
        self._captured_pre_step = None
        self._captured_post_clip_step = None

    @torch.no_grad()
    def _collect_grad_stats(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Computa normas L2, max abs y conteos NaN/Inf de gradientes en una sola sync."""
        scalars: list[torch.Tensor] = []
        organ_meta: list[tuple[str, int]] = []

        for organ, plist in self._organ_params.items():
            grads = [p.grad.detach().float() for _, p in plist if p.grad is not None]
            if not grads:
                continue
            organ_meta.append((organ, len(grads)))
            sq_sum = grads[0].new_zeros(())
            abs_max = grads[0].new_zeros(())
            nan_cnt = grads[0].new_zeros(())
            inf_cnt = grads[0].new_zeros(())
            for g in grads:
                nans = torch.isnan(g).sum()
                infs = torch.isinf(g).sum()
                nan_cnt = nan_cnt + nans
                inf_cnt = inf_cnt + infs
                g_fin = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
                sq_sum = sq_sum + (g_fin * g_fin).sum()
                abs_max = torch.maximum(abs_max, g_fin.abs().max())
            scalars.extend([sq_sum, abs_max, nan_cnt, inf_cnt])

        vals = _batch_scalars_to_floats(scalars)
        per_organ: dict[str, dict[str, Any]] = {}
        global_sq = 0.0
        global_max = 0.0
        global_nan = 0
        global_inf = 0

        for idx, (organ, n_g) in enumerate(organ_meta):
            sq_v, max_v, nan_v, inf_v = vals[4 * idx : 4 * idx + 4]
            nan_i = int(round(nan_v))
            inf_i = int(round(inf_v))
            if nan_i > 0:
                l2 = float("nan")
            elif inf_i > 0:
                l2 = float("inf")
            else:
                l2 = math.sqrt(max(sq_v, 0.0))
                global_sq += max(sq_v, 0.0)
            global_max = max(global_max, max_v)
            global_nan += nan_i
            global_inf += inf_i
            per_organ[organ] = dict(
                l2=l2,
                abs_max=max_v,
                nan_count=nan_i,
                inf_count=inf_i,
                n_grad_tensors=n_g,
            )

        if global_nan > 0:
            global_l2 = float("nan")
        elif global_inf > 0:
            global_l2 = float("inf")
        else:
            global_l2 = math.sqrt(max(global_sq, 0.0))

        global_stats = dict(
            l2=global_l2,
            abs_max=global_max,
            nan_count=global_nan,
            inf_count=global_inf,
            has_nonfinite=(global_nan > 0 or global_inf > 0),
        )
        return per_organ, global_stats

    @torch.no_grad()
    def before_optimizer_step(self, step: int) -> None:
        """Captura gradientes pre-clip y pesos pre-step (NO-OP en pasos no muestreados)."""
        if not self.should_log(step):
            return
        self._captured_pre_step = int(step)
        if self.config.track_gradients:
            self._pre_clip_grad_stats, self._pre_clip_global = self._collect_grad_stats()
        if self.config.track_update_ratios:
            self._pre_step_weights.clear()
            self._pre_step_organ_has_grad.clear()
            for organ, plist in self._organ_params.items():
                clones = [p.detach().clone() for _, p in plist if p.requires_grad]
                has_g = any(p.requires_grad and p.grad is not None for _, p in plist)
                self._pre_step_weights[organ] = clones
                self._pre_step_organ_has_grad[organ] = has_g

    @torch.no_grad()
    def after_grad_clip(self, step: int) -> None:
        """Captura gradientes post-clip (NO-OP en pasos no muestreados)."""
        if not self.should_log(step):
            return
        self._captured_post_clip_step = int(step)
        if self.config.track_gradients:
            self._post_clip_grad_stats, self._post_clip_global = self._collect_grad_stats()

    def clip_grad_norm(
        self,
        step: int,
        max_norm: float,
        *,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """Recorta gradientes registrando pre-clip y post-clip SOLO en pasos muestreados."""
        if self.should_log(step):
            self.before_optimizer_step(step)
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_norm, norm_type=norm_type
            )
            self.after_grad_clip(step)
            return total_norm
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=max_norm, norm_type=norm_type
        )

    @torch.no_grad()
    def _collect_weight_and_update_stats(
        self,
        *,
        compute_spectral: bool = True,
    ) -> tuple[dict[str, ModuleWeightGradSnapshot], dict[str, Any]]:
        """Computa estadísticas de pesos, gradientes y ratio Karpathy `||ΔW||/||W||` por órgano."""
        scalars: list[torch.Tensor] = []
        organ_meta: list[tuple[str, int, int, bool]] = []
        z_eps = self.config.zero_eps

        for organ, plist in self._organ_params.items():
            n_tensors = len(plist)
            n_params = sum(p.numel() for _, p in plist)
            first_p = plist[0][1]
            sq_sum = first_p.new_zeros((), dtype=torch.float32)
            elem_sum = first_p.new_zeros((), dtype=torch.float32)
            fin_cnt = first_p.new_zeros((), dtype=torch.float32)
            abs_max = first_p.new_zeros((), dtype=torch.float32)
            zero_cnt = first_p.new_zeros((), dtype=torch.float32)
            spec_max = first_p.new_zeros((), dtype=torch.float32)
            nan_cnt = first_p.new_zeros((), dtype=torch.float32)
            inf_cnt = first_p.new_zeros((), dtype=torch.float32)

            if self.config.track_weights:
                for _, p in plist:
                    w = p.detach().float()
                    nan_cnt.add_(torch.isnan(w).sum())
                    inf_cnt.add_(torch.isinf(w).sum())
                    fin = torch.isfinite(w)
                    w_fin = torch.where(fin, w, torch.zeros_like(w))
                    fin_cnt.add_(fin.sum())
                    elem_sum.add_(w_fin.sum())
                    sq_sum.add_((w_fin * w_fin).sum())
                    abs_w = w_fin.abs()
                    abs_max = torch.maximum(abs_max, abs_w.max())
                    zero_cnt.add_((abs_w <= z_eps).sum())
                    if compute_spectral:
                        spec_max = torch.maximum(spec_max, _spectral_norm_approx(w_fin))

            # Karpathy update-to-weight ratio ||W_after - W_before||_2 / ||W_before||_2
            has_pre = self.config.track_update_ratios and organ in self._pre_step_weights
            dw_sq = first_p.new_zeros((), dtype=torch.float32)
            w_pre_sq = first_p.new_zeros((), dtype=torch.float32)
            if has_pre:
                cur_trainable = [p.detach() for _, p in plist if p.requires_grad]
                pre_list = self._pre_step_weights[organ]
                for w_after, w_before in zip(cur_trainable, pre_list, strict=True):
                    wa = w_after.float()
                    wb = w_before.float()
                    diff = wa - wb
                    diff_fin = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
                    wb_fin = torch.where(torch.isfinite(wb), wb, torch.zeros_like(wb))
                    dw_sq.add_((diff_fin * diff_fin).sum())
                    w_pre_sq.add_((wb_fin * wb_fin).sum())

            organ_meta.append((organ, n_params, n_tensors, has_pre))
            scalars.extend(
                [
                    sq_sum,
                    elem_sum,
                    fin_cnt,
                    abs_max,
                    zero_cnt,
                    spec_max,
                    nan_cnt,
                    inf_cnt,
                    dw_sq,
                    w_pre_sq,
                ]
            )

        vals = _batch_scalars_to_floats(scalars)
        if compute_spectral and self.config.track_weights:
            self._spectral_eval_steps += 1
        snapshots: dict[str, ModuleWeightGradSnapshot] = {}
        g_w_sq = 0.0
        g_w_n = 0
        g_w_nan = 0
        g_w_inf = 0
        g_dw_sq = 0.0
        g_wpre_sq = 0.0
        any_update = False

        for idx, (organ, n_params, n_tensors, has_pre) in enumerate(organ_meta):
            base = 10 * idx
            (
                sq_sum_v,
                elem_sum_v,
                fin_cnt_v,
                abs_max_v,
                zero_cnt_v,
                spec_max_v,
                nan_v,
                inf_v,
                dw_sq_v,
                w_pre_sq_v,
            ) = vals[base : base + 10]

            w_nan = int(round(nan_v))
            w_inf = int(round(inf_v))
            g_w_nan += w_nan
            g_w_inf += w_inf
            g_w_n += n_params

            if self.config.track_weights:
                if w_nan > 0:
                    w_l2 = float("nan")
                    w_rms = float("nan")
                elif w_inf > 0:
                    w_l2 = float("inf")
                    w_rms = float("inf")
                else:
                    w_l2 = math.sqrt(max(sq_sum_v, 0.0))
                    w_rms = w_l2 / math.sqrt(max(float(n_params), 1.0))
                    g_w_sq += max(sq_sum_v, 0.0)
                if fin_cnt_v > 0.0:
                    mean_v = elem_sum_v / fin_cnt_v
                    var_v = max(sq_sum_v / fin_cnt_v - mean_v * mean_v, 0.0)
                    w_std = math.sqrt(var_v)
                else:
                    w_std = 0.0
                w_zero_frac = zero_cnt_v / max(float(n_params), 1.0)
                if compute_spectral:
                    self._last_spectral_approx[organ] = spec_max_v
                else:
                    spec_max_v = self._last_spectral_approx.get(organ, 0.0)
            else:
                w_l2 = w_rms = abs_max_v = w_std = spec_max_v = w_zero_frac = 0.0

            # Gradientes pre-clip y post-clip
            pre_g = self._pre_clip_grad_stats.get(organ)
            post_g = self._post_clip_grad_stats.get(organ)
            g_pre_l2 = pre_g["l2"] if pre_g is not None else None
            g_post_l2 = post_g["l2"] if post_g is not None else None
            ref_g = pre_g if pre_g is not None else post_g
            g_abs_max = ref_g["abs_max"] if ref_g is not None else None
            g_nan = (
                max(
                    pre_g["nan_count"] if pre_g is not None else 0,
                    post_g["nan_count"] if post_g is not None else 0,
                )
                if (pre_g is not None or post_g is not None)
                else 0
            )
            g_inf = (
                max(
                    pre_g["inf_count"] if pre_g is not None else 0,
                    post_g["inf_count"] if post_g is not None else 0,
                )
                if (pre_g is not None or post_g is not None)
                else 0
            )

            ref_l2 = g_pre_l2 if g_pre_l2 is not None else g_post_l2
            grad_vanishing = False
            grad_exploding = False
            if ref_l2 is not None:
                if g_inf > 0 or (
                    math.isfinite(ref_l2) and ref_l2 > self.config.grad_explode_threshold
                ):
                    grad_exploding = True
                if (
                    g_nan == 0
                    and g_inf == 0
                    and math.isfinite(ref_l2)
                    and ref_l2 < self.config.grad_vanish_threshold
                ):
                    grad_vanishing = True

            # Ratio Karpathy ||ΔW|| / ||W||
            upd_l2: float | None = None
            upd_ratio: float | None = None
            upd_log10: float | None = None
            upd_dead = False
            upd_unstable = False
            if has_pre:
                any_update = True
                upd_l2 = math.sqrt(max(dw_sq_v, 0.0))
                w_pre_l2 = math.sqrt(max(w_pre_sq_v, 0.0))
                upd_ratio = upd_l2 / max(w_pre_l2, 1e-12)
                upd_log10 = math.log10(max(upd_ratio, 1e-15))
                g_dw_sq += max(dw_sq_v, 0.0)
                g_wpre_sq += max(w_pre_sq_v, 0.0)
                organ_had_grad = self._pre_step_organ_has_grad.get(organ, False)
                if organ_had_grad:
                    if upd_ratio < self.config.update_ratio_dead_threshold:
                        upd_dead = True
                    if upd_ratio > self.config.update_ratio_unstable_threshold:
                        upd_unstable = True

            snapshots[organ] = ModuleWeightGradSnapshot(
                organ=organ,
                n_params=n_params,
                n_tensors=n_tensors,
                weight_l2=w_l2,
                weight_rms=w_rms,
                weight_abs_max=abs_max_v,
                weight_std=w_std,
                weight_spectral_approx=spec_max_v,
                weight_zero_fraction=w_zero_frac,
                weight_nan_count=w_nan,
                weight_inf_count=w_inf,
                grad_pre_clip_l2=g_pre_l2,
                grad_post_clip_l2=g_post_l2,
                grad_abs_max=g_abs_max,
                grad_nan_count=g_nan,
                grad_inf_count=g_inf,
                grad_vanishing=grad_vanishing,
                grad_exploding=grad_exploding,
                update_l2=upd_l2,
                update_to_weight_ratio=upd_ratio,
                update_ratio_log10=upd_log10,
                update_dead=upd_dead,
                update_unstable=upd_unstable,
            )

        global_w_l2 = (
            float("nan")
            if g_w_nan > 0
            else (float("inf") if g_w_inf > 0 else math.sqrt(max(g_w_sq, 0.0)))
        )
        global_upd_ratio = (
            math.sqrt(max(g_dw_sq, 0.0)) / max(math.sqrt(max(g_wpre_sq, 0.0)), 1e-12)
            if any_update
            else None
        )
        global_summary = dict(
            total_params=g_w_n,
            global_weight_l2=global_w_l2,
            global_weight_nan_count=g_w_nan,
            global_weight_inf_count=g_w_inf,
            global_update_to_weight_ratio=global_upd_ratio,
            global_update_ratio_log10=(
                math.log10(max(global_upd_ratio, 1e-15)) if global_upd_ratio is not None else None
            ),
            spectral_computed=bool(compute_spectral and self.config.track_weights),
        )
        return snapshots, global_summary

    @torch.no_grad()
    def _collect_optimizer_telemetry(
        self, optimizer: torch.optim.Optimizer | None
    ) -> dict[str, Any]:
        """Registra LR por grupo + estado AdamW, Muon y ScheduleFreeAdamW con cero copias H2D."""
        if optimizer is None:
            return dict(
                lr_by_group={},
                backbone_lr=None,
                router_lr=None,
                muon_lr=None,
                router_to_backbone_lr_ratio=None,
                adamw_effective_step_rms=0.0,
                adamw_adaptive_lr_rms=0.0,
                adamw_exp_avg_norm=0.0,
                adamw_exp_avg_sq_rms=0.0,
                muon_momentum_buffer_l2=0.0,
                muon_momentum_buffer_rms=0.0,
                muon_momentum_buffer_abs_max=0.0,
                muon_param_count=0,
                muon_momentum_by_organ={},
                schedule_free_z_x_diff_l2=0.0,
                schedule_free_z_x_diff_rms=0.0,
                schedule_free_z_x_diff_abs_max=0.0,
                schedule_free_z_x_rel_diff=0.0,
                schedule_free_adaptive_lr_rms=0.0,
                schedule_free_exp_avg_sq_rms=0.0,
                schedule_free_param_count=0,
                schedule_free_z_x_diff_by_organ={},
            )

        lr_by_group: dict[str, float] = {}
        backbone_lrs: list[float] = []
        router_lrs: list[float] = []
        muon_lrs: list[float] = []

        # Acumuladores en dispositivo por device (0 asignaciones `torch.tensor()` / H2D!)
        adam_acc: dict[torch.device, list[torch.Tensor]] = {}
        adam_elem_count = 0

        muon_acc: dict[torch.device, list[torch.Tensor]] = {}
        muon_organ_acc: dict[tuple[str, torch.device], torch.Tensor] = {}
        muon_elem_count = 0

        sf_acc: dict[torch.device, list[torch.Tensor]] = {}
        sf_organ_acc: dict[tuple[str, torch.device], torch.Tensor] = {}
        sf_elem_count = 0
        sf_v_elem_count = 0

        for idx, group in enumerate(optimizer.param_groups):
            lr_val = float(group.get("lr", 0.0))
            wd_val = float(group.get("weight_decay", 0.0))
            params = group.get("params", [])
            names = [self._param_id_to_name.get(id(p), "") for p in params]
            is_router_group = bool(names) and all(
                ("router" in n or "balance_bias" in n) for n in names if n
            )
            has_named = any(bool(n) for n in names)
            is_muon_group = bool(group.get("use_muon", False))

            if "name" in group and group["name"]:
                label = str(group["name"])
            elif "group_role" in group and group["group_role"]:
                label = str(group["group_role"])
            elif has_named:
                prefix = "router" if is_router_group else "backbone"
                suffix = "decay" if wd_val > 0.0 else "no_decay"
                label = f"{prefix}_{suffix}"
            else:
                label = f"group_{idx}"

            if label in lr_by_group:
                label = f"{label}_{idx}"
            lr_by_group[label] = lr_val

            if is_muon_group or "muon" in label.lower():
                muon_lrs.append(lr_val)
            if is_router_group or "router" in label.lower():
                router_lrs.append(lr_val)
            else:
                backbone_lrs.append(lr_val)

            betas = group.get("betas", (0.9, 0.999))
            beta1 = float(betas[0]) if isinstance(betas, tuple | list) else 0.9
            beta2 = float(betas[1]) if isinstance(betas, tuple | list) else 0.999
            log_beta1 = math.log(beta1) if 0.0 < beta1 < 1.0 else None
            log_beta2 = math.log(beta2) if 0.0 < beta2 < 1.0 else None
            eps = float(group.get("eps", 1e-8))
            group_step = group.get("step")
            warmup_steps = int(group.get("warmup_steps", 0) or 0)

            for p in params:
                st = optimizer.state.get(p)
                if not isinstance(st, dict):
                    continue

                # 1) Estado Muon / SGD (`momentum_buffer`)
                mom_buf = st.get("momentum_buffer")
                if isinstance(mom_buf, torch.Tensor) and mom_buf.numel() > 0:
                    mb_f = mom_buf.detach().float()
                    mb_fin = torch.nan_to_num(mb_f, nan=0.0, posinf=0.0, neginf=0.0)
                    sq_t = (mb_fin * mb_fin).sum()
                    max_t = mb_fin.abs().max()
                    dev = mb_fin.device
                    m_slot = muon_acc.get(dev)
                    if m_slot is None:
                        m_slot = [mb_fin.new_zeros(()), mb_fin.new_zeros(())]
                        muon_acc[dev] = m_slot
                    m_slot[0].add_(sq_t)
                    m_slot[1] = torch.maximum(m_slot[1], max_t)
                    p_name = self._param_id_to_name.get(id(p), "")
                    p_organ = classify_param_organ(p_name) if p_name else "other"
                    okey = (p_organ, dev)
                    if okey not in muon_organ_acc:
                        muon_organ_acc[okey] = mb_fin.new_zeros(())
                    muon_organ_acc[okey].add_(sq_t)
                    muon_elem_count += int(mb_f.numel())

                # 2) Estado ScheduleFreeAdamW (`z`, `x`, y opcionalmente `exp_avg_sq`)
                sf_z = st.get("z")
                sf_x = st.get("x")
                if (
                    isinstance(sf_z, torch.Tensor)
                    and isinstance(sf_x, torch.Tensor)
                    and sf_z.numel() > 0
                ):
                    z_f = sf_z.detach().float()
                    x_f = sf_x.detach().float()
                    diff_fin = torch.nan_to_num(z_f - x_f, nan=0.0, posinf=0.0, neginf=0.0)
                    x_fin = torch.nan_to_num(x_f, nan=0.0, posinf=0.0, neginf=0.0)
                    diff_sq = (diff_fin * diff_fin).sum()
                    diff_max = diff_fin.abs().max()
                    x_sq = (x_fin * x_fin).sum()
                    dev = diff_fin.device
                    s_slot = sf_acc.get(dev)
                    if s_slot is None:
                        s_slot = [
                            diff_fin.new_zeros(()),  # 0: diff_sq
                            diff_fin.new_zeros(()),  # 1: diff_max
                            diff_fin.new_zeros(()),  # 2: x_sq
                            diff_fin.new_zeros(()),  # 3: adapt_lr_sq
                            diff_fin.new_zeros(()),  # 4: v_sum
                        ]
                        sf_acc[dev] = s_slot
                    s_slot[0].add_(diff_sq)
                    s_slot[1] = torch.maximum(s_slot[1], diff_max)
                    s_slot[2].add_(x_sq)
                    p_name = self._param_id_to_name.get(id(p), "")
                    p_organ = classify_param_organ(p_name) if p_name else "other"
                    s_okey = (p_organ, dev)
                    if s_okey not in sf_organ_acc:
                        sf_organ_acc[s_okey] = diff_fin.new_zeros(())
                    sf_organ_acc[s_okey].add_(diff_sq)
                    sf_elem_count += int(diff_fin.numel())

                    sf_v = st.get("exp_avg_sq")
                    step_val_sf = st.get("step", group_step)
                    if (
                        isinstance(sf_v, torch.Tensor)
                        and sf_v.numel() > 0
                        and step_val_sf is not None
                    ):
                        v_f = sf_v.detach().float().clamp_min(0.0)
                        if isinstance(step_val_sf, torch.Tensor) and step_val_sf.is_cuda:
                            step_f = step_val_sf.detach().float()
                            bc2_sf: Any = (
                                (1.0 - torch.exp(step_f * log_beta2)).clamp_min(1e-8)
                                if log_beta2 is not None
                                else 1.0
                            )
                            lr_eff_sf: Any = (
                                lr_val * (step_f / float(warmup_steps)).clamp(0.0, 1.0)
                                if warmup_steps > 0
                                else lr_val
                            )
                            valid_sf = True
                        else:
                            s_num_sf = float(step_val_sf)
                            valid_sf = s_num_sf > 0.0
                            if valid_sf:
                                bc2_sf = (
                                    max(1.0 - (beta2**s_num_sf), 1e-8)
                                    if log_beta2 is not None
                                    else 1.0
                                )
                                sched_sf = (
                                    min(1.0, s_num_sf / float(warmup_steps))
                                    if warmup_steps > 0
                                    else 1.0
                                )
                                lr_eff_sf = lr_val * sched_sf
                        if valid_sf:
                            denom_sf = (v_f / bc2_sf).sqrt() + eps
                            adapt_lr_sf = lr_eff_sf / denom_sf
                            s_slot[3].add_((adapt_lr_sf * adapt_lr_sf).sum())
                            s_slot[4].add_(v_f.sum())
                            sf_v_elem_count += int(v_f.numel())

                # 3) Estado AdamW (`exp_avg`, `exp_avg_sq`) — cero asignaciones H2D
                exp_avg = st.get("exp_avg")
                exp_avg_sq = st.get("exp_avg_sq")
                step_val = st.get("step", group_step)
                if not (isinstance(exp_avg, torch.Tensor) and isinstance(exp_avg_sq, torch.Tensor)):
                    continue
                if step_val is None:
                    continue
                if isinstance(step_val, torch.Tensor) and step_val.is_cuda:
                    step_f = step_val.detach().float()
                    bc1: Any = (
                        (1.0 - torch.exp(step_f * log_beta1)).clamp_min(1e-8)
                        if log_beta1 is not None
                        else 1.0
                    )
                    bc2: Any = (
                        (1.0 - torch.exp(step_f * log_beta2)).clamp_min(1e-8)
                        if log_beta2 is not None
                        else 1.0
                    )
                else:
                    s_num = float(step_val)
                    if s_num <= 0.0:
                        continue
                    bc1 = max(1.0 - (beta1**s_num), 1e-8) if log_beta1 is not None else 1.0
                    bc2 = max(1.0 - (beta2**s_num), 1e-8) if log_beta2 is not None else 1.0

                m_f = exp_avg.detach().float()
                v_f = exp_avg_sq.detach().float().clamp_min(0.0)
                denom = (v_f / bc2).sqrt() + eps
                adapt_lr = lr_val / denom
                eff_step = adapt_lr * (m_f / bc1)
                dev = m_f.device
                a_slot = adam_acc.get(dev)
                if a_slot is None:
                    a_slot = [
                        m_f.new_zeros(()),
                        m_f.new_zeros(()),
                        m_f.new_zeros(()),
                        m_f.new_zeros(()),
                    ]
                    adam_acc[dev] = a_slot
                a_slot[0].add_((eff_step * eff_step).sum())
                a_slot[1].add_((adapt_lr * adapt_lr).sum())
                a_slot[2].add_((m_f * m_f).sum())
                a_slot[3].add_(v_f.sum())
                adam_elem_count += int(m_f.numel())

        # Un único sync agrupado para AdamW + Muon + ScheduleFreeAdamW + órganos
        packed_scalars: list[torch.Tensor] = []
        adam_dev_slices: list[int] = []
        for a_slot in adam_acc.values():
            adam_dev_slices.append(len(packed_scalars))
            packed_scalars.extend(a_slot)

        muon_dev_slices: list[int] = []
        for m_slot in muon_acc.values():
            muon_dev_slices.append(len(packed_scalars))
            packed_scalars.extend(m_slot)

        muon_organ_items = sorted(muon_organ_acc.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
        muon_organ_start = len(packed_scalars)
        for _, sq_t in muon_organ_items:
            packed_scalars.append(sq_t)

        sf_dev_slices: list[int] = []
        for s_slot in sf_acc.values():
            sf_dev_slices.append(len(packed_scalars))
            packed_scalars.extend(s_slot)

        sf_organ_items = sorted(sf_organ_acc.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
        sf_organ_start = len(packed_scalars)
        for _, sq_t in sf_organ_items:
            packed_scalars.append(sq_t)

        synced = _batch_scalars_to_floats(packed_scalars)

        adam_eff_rms = 0.0
        adam_adapt_rms = 0.0
        adam_m_norm = 0.0
        adam_v_rms = 0.0
        if adam_elem_count > 0:
            eff_sq = sum(synced[off] for off in adam_dev_slices)
            adp_sq = sum(synced[off + 1] for off in adam_dev_slices)
            m_sq = sum(synced[off + 2] for off in adam_dev_slices)
            v_sum = sum(synced[off + 3] for off in adam_dev_slices)
            adam_eff_rms = math.sqrt(max(eff_sq, 0.0) / float(adam_elem_count))
            adam_adapt_rms = math.sqrt(max(adp_sq, 0.0) / float(adam_elem_count))
            adam_m_norm = math.sqrt(max(m_sq, 0.0))
            adam_v_rms = math.sqrt(max(v_sum, 0.0) / float(adam_elem_count))

        muon_buf_l2 = 0.0
        muon_buf_rms = 0.0
        muon_buf_max = 0.0
        muon_by_organ: dict[str, float] = {}
        if muon_elem_count > 0:
            m_tot_sq = sum(synced[off] for off in muon_dev_slices)
            muon_buf_l2 = math.sqrt(max(m_tot_sq, 0.0))
            muon_buf_rms = muon_buf_l2 / math.sqrt(float(muon_elem_count))
            muon_buf_max = max(synced[off + 1] for off in muon_dev_slices)
            org_sq_sum: dict[str, float] = {}
            for i, ((org, _), _) in enumerate(muon_organ_items):
                org_sq_sum[org] = org_sq_sum.get(org, 0.0) + max(synced[muon_organ_start + i], 0.0)
            for org in sorted(org_sq_sum):
                muon_by_organ[org] = math.sqrt(org_sq_sum[org])

        sf_diff_l2 = 0.0
        sf_diff_rms = 0.0
        sf_diff_max = 0.0
        sf_rel_diff = 0.0
        sf_adapt_rms = 0.0
        sf_v_rms = 0.0
        sf_by_organ: dict[str, float] = {}
        if sf_elem_count > 0:
            d_tot_sq = sum(synced[off] for off in sf_dev_slices)
            sf_diff_l2 = math.sqrt(max(d_tot_sq, 0.0))
            sf_diff_rms = sf_diff_l2 / math.sqrt(float(sf_elem_count))
            sf_diff_max = max(synced[off + 1] for off in sf_dev_slices)
            x_tot_l2 = math.sqrt(max(sum(synced[off + 2] for off in sf_dev_slices), 0.0))
            sf_rel_diff = sf_diff_l2 / max(x_tot_l2, 1e-12)
            if sf_v_elem_count > 0:
                sf_adp_sq = sum(synced[off + 3] for off in sf_dev_slices)
                sf_v_s = sum(synced[off + 4] for off in sf_dev_slices)
                sf_adapt_rms = math.sqrt(max(sf_adp_sq, 0.0) / float(sf_v_elem_count))
                sf_v_rms = math.sqrt(max(sf_v_s, 0.0) / float(sf_v_elem_count))
            sf_org_sq: dict[str, float] = {}
            for i, ((org, _), _) in enumerate(sf_organ_items):
                sf_org_sq[org] = sf_org_sq.get(org, 0.0) + max(synced[sf_organ_start + i], 0.0)
            for org in sorted(sf_org_sq):
                sf_by_organ[org] = math.sqrt(sf_org_sq[org])

        backbone_lr = backbone_lrs[0] if backbone_lrs else None
        router_lr = router_lrs[0] if router_lrs else None
        muon_lr = muon_lrs[0] if muon_lrs else None
        ratio = (
            router_lr / backbone_lr
            if (router_lr is not None and backbone_lr is not None and backbone_lr > 0.0)
            else None
        )
        return dict(
            lr_by_group=lr_by_group,
            backbone_lr=backbone_lr,
            router_lr=router_lr,
            muon_lr=muon_lr,
            router_to_backbone_lr_ratio=ratio,
            adamw_effective_step_rms=adam_eff_rms,
            adamw_adaptive_lr_rms=adam_adapt_rms,
            adamw_exp_avg_norm=adam_m_norm,
            adamw_exp_avg_sq_rms=adam_v_rms,
            muon_momentum_buffer_l2=muon_buf_l2,
            muon_momentum_buffer_rms=muon_buf_rms,
            muon_momentum_buffer_abs_max=muon_buf_max,
            muon_param_count=muon_elem_count,
            muon_momentum_by_organ=muon_by_organ,
            schedule_free_z_x_diff_l2=sf_diff_l2,
            schedule_free_z_x_diff_rms=sf_diff_rms,
            schedule_free_z_x_diff_abs_max=sf_diff_max,
            schedule_free_z_x_rel_diff=sf_rel_diff,
            schedule_free_adaptive_lr_rms=sf_adapt_rms,
            schedule_free_exp_avg_sq_rms=sf_v_rms,
            schedule_free_param_count=sf_elem_count,
            schedule_free_z_x_diff_by_organ=sf_by_organ,
        )

    @staticmethod
    @torch.no_grad()
    def _collect_losses(loss_dict: dict[str, Any] | None) -> dict[str, float]:
        """Extrae descomposición completa de pérdidas en un único sync agrupado."""
        if not loss_dict:
            return {}
        out: dict[str, float] = {}
        tensor_keys: list[str] = []
        tensor_vals: list[torch.Tensor] = []

        for key, val in loss_dict.items():
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    tensor_keys.append(str(key))
                    tensor_vals.append(val)
            elif isinstance(val, int | float) and not isinstance(val, bool):
                out[str(key)] = float(val)

        if tensor_vals:
            floats = _batch_scalars_to_floats(tensor_vals)
            for k, fv in zip(tensor_keys, floats, strict=True):
                out[k] = fv
        return out

    @torch.no_grad()
    def _collect_domain_telemetry(
        self,
        *,
        model_output: dict[str, Any] | None,
        titans_update_diag: dict[str, Any] | None,
        cognitive_core_diag: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Extrae métricas de dominio en UN solo sync agrupado `.cpu().tolist()` (0 `.item()`)."""
        diag = (
            model_output.get("diagnostics", {})
            if isinstance(model_output, dict) and isinstance(model_output.get("diagnostics"), dict)
            else {}
        )
        state = (
            model_output.get("state", {})
            if isinstance(model_output, dict) and isinstance(model_output.get("state"), dict)
            else {}
        )

        # Recolector por lotes: registra escalares/vectores GPU y los transfiere en 1 sola copia
        pending_tensors: list[torch.Tensor] = []
        scalar_slots: dict[str, int] = {}
        scalar_vals: dict[str, float | None] = {}
        vector_slots: dict[str, int] = {}
        vector_vals: dict[str, list[float]] = {}

        def _reg_scalar(key: str, val: Any) -> None:
            if val is None:
                scalar_vals[key] = None
                return
            if isinstance(val, bool):
                scalar_vals[key] = float(val)
                return
            if isinstance(val, int | float):
                scalar_vals[key] = float(val)
                return
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    scalar_slots[key] = len(pending_tensors)
                    pending_tensors.append(val.detach().float().reshape(1))
                else:
                    scalar_vals[key] = None
                return
            if hasattr(val, "item") and callable(val.item):
                try:
                    scalar_vals[key] = float(val.item())
                except Exception:
                    scalar_vals[key] = None
                return
            scalar_vals[key] = None

        # 1) MoE
        _reg_scalar("moe_entropy", diag.get("router_entropy"))
        _reg_scalar("moe_load_ratio", diag.get("load_max_min_ratio"))
        _reg_scalar("moe_expert_corr", diag.get("expert_mean_corr"))
        raw_loads = diag.get("expert_loads")
        if isinstance(raw_loads, list) and raw_loads:
            if all(isinstance(t, torch.Tensor) for t in raw_loads):
                stacked = torch.stack([t.detach().float() for t in raw_loads], dim=0).mean(dim=0)
                vector_slots["moe_expert_loads"] = len(pending_tensors)
                pending_tensors.append(stacked.reshape(-1))
            elif all(isinstance(x, int | float) and not isinstance(x, bool) for x in raw_loads):
                vector_vals["moe_expert_loads"] = [float(x) for x in raw_loads]
        elif isinstance(raw_loads, torch.Tensor):
            vector_slots["moe_expert_loads"] = len(pending_tensors)
            pending_tensors.append(raw_loads.detach().float().reshape(-1))

        # 2) Doble bucle contractivo PC-TRM-Eq + LoopedCrossAssetTransformer + Contrafactual v2
        _reg_scalar("k_used", diag.get("k_used"))
        _reg_scalar("de_neg_rate", diag.get("delta_energy_neg_rate"))
        _reg_scalar("max_spec", diag.get("max_spectral_norm"))
        _reg_scalar("loop_halt_at_k1_pct", diag.get("loop_halt_at_k1_pct"))
        _reg_scalar("cf_fragility_k0", diag.get("cf_fragility_k0"))
        _reg_scalar("cf_fragility_final", diag.get("cf_fragility_final"))
        _reg_scalar("cf_sign_flip_rate_final", diag.get("cf_sign_flip_rate_final"))
        _reg_scalar("cf_worst_dd_final", diag.get("cf_worst_dd_final"))
        _reg_scalar("loop_step_router_js_div", diag.get("loop_step_router_js_div"))
        _reg_scalar("landmark_attn_entropy", diag.get("landmark_attn_entropy"))
        _reg_scalar("ponder_kl_loss", diag.get("ponder_kl_loss"))
        e_traj = diag.get("energy_trajectory")
        if isinstance(e_traj, list) and e_traj:
            _reg_scalar("energy_final", e_traj[-1])
        elif isinstance(model_output, dict) and isinstance(model_output.get("energy"), dict):
            _reg_scalar("energy_final", model_output["energy"].get("total"))
        else:
            scalar_vals["energy_final"] = None

        # 3) Titans PT1 (desde `titans_update_diag` o `diag["neuromod"]`) + DualAxis v2
        nm_diag = diag.get("neuromod") if isinstance(diag.get("neuromod"), dict) else {}
        pt1_src: dict[str, Any] = {}
        if isinstance(nm_diag, dict):
            pt1_src.update(nm_diag)
        if isinstance(titans_update_diag, dict):
            pt1_src.update(titans_update_diag)
        _reg_scalar("pt1_alpha_min", pt1_src.get("alpha_min"))
        _reg_scalar("pt1_alpha_mean", pt1_src.get("alpha_mean"))
        _reg_scalar("pt1_eta_max", pt1_src.get("eta_max"))
        _reg_scalar("pt1_theta_max", pt1_src.get("theta_max"))
        _reg_scalar("pt1_theta_mean", pt1_src.get("theta_mean"))
        _reg_scalar("ne_slow_shift", nm_diag.get("ne_slow_shift"))
        _reg_scalar("mem_norm", nm_diag.get("mem_norm"))
        _reg_scalar("neuromod_mkt_mem_norm", nm_diag.get("mkt_mem_norm"))
        _reg_scalar("neuromod_tukey_zero_frac", pt1_src.get("tukey_zero_frac"))
        _reg_scalar("neuromod_cond_upper_bound", pt1_src.get("cond_upper_bound"))

        # 4) HRR-Modern-Hopfield OOD
        _reg_scalar("hrr_ood_mean", diag.get("ood_energy_mean"))
        raw_hrr_slots = state.get("hrr_slots_used")
        if raw_hrr_slots is None:
            hrr_mod = getattr(self.model, "hrr_hopfield", None)
            if hrr_mod is not None and hasattr(hrr_mod, "slot_filled"):
                _reg_scalar("hrr_slots_used", hrr_mod.slot_filled.detach().sum())
            else:
                scalar_vals["hrr_slots_used"] = 0.0
        else:
            _reg_scalar("hrr_slots_used", raw_hrr_slots)

        # 5) Cognitive Core (v1 6-tools + v2 10-tools specialization en el mismo sync!)
        cog_src = (
            cognitive_core_diag
            if isinstance(cognitive_core_diag, dict)
            else (
                diag.get("cognitive_core")
                if isinstance(diag.get("cognitive_core"), dict)
                else None
            )
        )
        cog_tool_names: tuple[str, ...] = TOOL_NAMES
        cog_toolbox_version: str | None = None
        cog_dict_keys: list[str] = []
        cog_raw_map: dict[str, float] = {}
        cog_step_w_shape: tuple[int, int] | None = None
        if cog_src is not None:
            raw_ver = cog_src.get("toolbox_version")
            cog_toolbox_version = str(raw_ver) if raw_ver is not None else None
            default_names = TOOL_NAMES_V2 if cog_toolbox_version == "v2_10tools" else TOOL_NAMES
            cog_tool_names = tuple(str(x) for x in cog_src.get("tool_names", default_names))
            raw_dist = cog_src.get("tool_usage_distribution")
            if isinstance(raw_dist, torch.Tensor):
                vector_slots["cog_dist"] = len(pending_tensors)
                pending_tensors.append(raw_dist.detach().float().reshape(-1))
            elif isinstance(raw_dist, list | tuple):
                if all(isinstance(t, torch.Tensor) for t in raw_dist) and raw_dist:
                    vector_slots["cog_dist"] = len(pending_tensors)
                    pending_tensors.append(
                        torch.stack([t.detach().float().reshape(()) for t in raw_dist])
                    )
                else:
                    vector_vals["cog_dist"] = [float(x) for x in raw_dist]
            elif isinstance(raw_dist, dict):
                for tk, tv in raw_dist.items():
                    sk = str(tk)
                    if isinstance(tv, torch.Tensor) and tv.numel() == 1:
                        cog_dict_keys.append(sk)
                        _reg_scalar(f"__cog_tool_{sk}", tv)
                    else:
                        cog_raw_map[sk] = float(tv)
            raw_by_step = cog_src.get("tool_weights_by_step")
            if isinstance(raw_by_step, torch.Tensor) and raw_by_step.ndim == 2:
                cog_step_w_shape = (int(raw_by_step.shape[0]), int(raw_by_step.shape[1]))
                vector_slots["cog_weights_by_step"] = len(pending_tensors)
                pending_tensors.append(raw_by_step.detach().float().reshape(-1))
            _reg_scalar("cog_tool_entropy", cog_src.get("tool_entropy"))
            _reg_scalar("cog_pruned_ratio", cog_src.get("pruned_context_ratio"))
            _reg_scalar("cog_lip", cog_src.get("lipschitz_upper_bound"))
            _reg_scalar("cog_load_max_min", cog_src.get("tool_load_max_min_ratio"))
            _reg_scalar("cog_active_count", cog_src.get("tool_active_count"))
            _reg_scalar("cog_corr_max", cog_src.get("tool_corr_max"))
            _reg_scalar(
                "cog_exp_steps_mean",
                cog_src.get("expected_steps_mean", cog_src.get("expected_thinking_steps")),
            )
            _reg_scalar("cog_exp_steps_easy_q25", cog_src.get("expected_steps_easy_q25"))
            _reg_scalar("cog_exp_steps_hard_q75", cog_src.get("expected_steps_hard_q75"))
            _reg_scalar("cog_cf_flip_k0", cog_src.get("cf_sign_flip_k0"))
            _reg_scalar("cog_cf_flip_final", cog_src.get("cf_sign_flip_final"))
            _reg_scalar("cog_conflict_mean", cog_src.get("conflict_mean"))
            _reg_scalar("cog_scratchpad_novelty", cog_src.get("scratchpad_novelty_mean"))
            if scalar_vals.get("ponder_kl_loss") is None and "ponder_kl_loss" not in scalar_slots:
                _reg_scalar("ponder_kl_loss", cog_src.get("ponder_kl_loss"))
        else:
            for k_none in (
                "cog_tool_entropy",
                "cog_pruned_ratio",
                "cog_lip",
                "cog_load_max_min",
                "cog_active_count",
                "cog_corr_max",
                "cog_exp_steps_mean",
                "cog_exp_steps_easy_q25",
                "cog_exp_steps_hard_q75",
                "cog_cf_flip_k0",
                "cog_cf_flip_final",
                "cog_conflict_mean",
                "cog_scratchpad_novelty",
            ):
                scalar_vals[k_none] = None

        # ÚNICA transferencia agrupada dispositivo -> host para TODOS los diagnósticos de dominio
        transferred = _batch_vectors_to_lists(pending_tensors)
        for key, slot in scalar_slots.items():
            vals_s = transferred[slot]
            scalar_vals[key] = vals_s[0] if vals_s else None
        for key, slot in vector_slots.items():
            vector_vals[key] = transferred[slot]

        moe_entropy = scalar_vals["moe_entropy"]
        moe_load_ratio = scalar_vals["moe_load_ratio"]
        moe_expert_corr = scalar_vals["moe_expert_corr"]
        moe_expert_loads = vector_vals.get("moe_expert_loads", [])

        k_used = scalar_vals["k_used"]
        de_neg_rate = scalar_vals["de_neg_rate"]
        de_neg_pct = 100.0 * de_neg_rate if de_neg_rate is not None else None
        max_spec = scalar_vals["max_spec"]
        energy_final = scalar_vals["energy_final"]

        pt1_alpha_min = scalar_vals["pt1_alpha_min"]
        pt1_alpha_mean = scalar_vals["pt1_alpha_mean"]
        pt1_eta_max = scalar_vals["pt1_eta_max"]
        pt1_theta_max = scalar_vals["pt1_theta_max"]
        pt1_theta_mean = scalar_vals["pt1_theta_mean"]
        ne_slow_shift = scalar_vals["ne_slow_shift"]
        mem_norm = scalar_vals["mem_norm"]

        # Verificación de la caja contractiva PT1 contra límites del modelo
        pt1_violation = False
        nm_mod = getattr(self.model, "neuromod_memory", None)
        alpha_lo = float(getattr(nm_mod, "pt1_alpha_lo", 0.05)) if nm_mod is not None else 0.05
        eta_hi = float(getattr(nm_mod, "pt1_eta_hi", 0.95)) if nm_mod is not None else 0.95
        theta_hi = float(getattr(nm_mod, "pt1_theta_max", 0.25)) if nm_mod is not None else 0.25
        if pt1_alpha_min is not None and pt1_alpha_min < alpha_lo - 1e-5:
            pt1_violation = True
        if pt1_eta_max is not None and pt1_eta_max > eta_hi + 1e-5:
            pt1_violation = True
        if pt1_theta_max is not None and pt1_theta_max > theta_hi + 1e-5:
            pt1_violation = True

        hrr_ood_mean = scalar_vals["hrr_ood_mean"]
        hrr_slots_f = scalar_vals.get("hrr_slots_used")
        hrr_slots_used = int(round(hrr_slots_f)) if hrr_slots_f is not None else 0

        tool_usage_map: dict[str, float] = dict(cog_raw_map)
        for sk in cog_dict_keys:
            sv = scalar_vals.get(f"__cog_tool_{sk}")
            if sv is not None:
                tool_usage_map[sk] = sv
        dist_list = vector_vals.get("cog_dist", [])
        if dist_list:
            fallback_names = (
                TOOL_NAMES_V2 if len(dist_list) == len(TOOL_NAMES_V2) else cog_tool_names
            )
            for idx, val in enumerate(dist_list):
                t_name = (
                    cog_tool_names[idx]
                    if idx < len(cog_tool_names)
                    else (fallback_names[idx] if idx < len(fallback_names) else f"tool_{idx}")
                )
                tool_usage_map[str(t_name)] = val
        cog_weights_by_step: list[list[float]] = []
        raw_step_flat = vector_vals.get("cog_weights_by_step", [])
        if raw_step_flat and cog_step_w_shape is not None:
            n_k, n_t = cog_step_w_shape
            if len(raw_step_flat) == n_k * n_t:
                cog_weights_by_step = [
                    raw_step_flat[i * n_t : (i + 1) * n_t] for i in range(n_k)
                ]
        cog_tool_entropy = scalar_vals["cog_tool_entropy"]
        cog_pruned_ratio = scalar_vals["cog_pruned_ratio"]
        cog_lip = scalar_vals["cog_lip"]

        return dict(
            moe_router_entropy=moe_entropy,
            moe_load_max_min_ratio=moe_load_ratio,
            moe_expert_mean_corr=moe_expert_corr,
            moe_expert_loads=moe_expert_loads,
            loop_k_used=k_used,
            loop_delta_energy_neg_rate=de_neg_rate,
            loop_delta_energy_neg_pct=de_neg_pct,
            loop_max_spectral_norm=max_spec,
            loop_energy_final=energy_final,
            loop_halt_at_k1_pct=scalar_vals["loop_halt_at_k1_pct"],
            cf_fragility_k0=scalar_vals["cf_fragility_k0"],
            cf_fragility_final=scalar_vals["cf_fragility_final"],
            cf_sign_flip_rate_final=scalar_vals["cf_sign_flip_rate_final"],
            cf_worst_dd_final=scalar_vals["cf_worst_dd_final"],
            loop_step_router_js_div=scalar_vals["loop_step_router_js_div"],
            landmark_attn_entropy=scalar_vals["landmark_attn_entropy"],
            ponder_kl_loss=scalar_vals["ponder_kl_loss"],
            pt1_alpha_min=pt1_alpha_min,
            pt1_alpha_mean=pt1_alpha_mean,
            pt1_eta_max=pt1_eta_max,
            pt1_theta_max=pt1_theta_max,
            pt1_theta_mean=pt1_theta_mean,
            pt1_bound_violation=pt1_violation,
            neuromod_ne_slow_shift=ne_slow_shift,
            neuromod_mem_norm=mem_norm,
            neuromod_mkt_mem_norm=scalar_vals["neuromod_mkt_mem_norm"],
            neuromod_tukey_zero_frac=scalar_vals["neuromod_tukey_zero_frac"],
            neuromod_cond_upper_bound=scalar_vals["neuromod_cond_upper_bound"],
            hrr_ood_energy_mean=hrr_ood_mean,
            hrr_slots_used=hrr_slots_used,
            cognitive_toolbox_version=cog_toolbox_version,
            cognitive_tool_usage=tool_usage_map,
            cognitive_tool_weights_by_step=cog_weights_by_step,
            cognitive_tool_entropy=cog_tool_entropy,
            cognitive_tool_load_max_min_ratio=scalar_vals["cog_load_max_min"],
            cognitive_tool_active_count=scalar_vals["cog_active_count"],
            cognitive_tool_corr_max=scalar_vals["cog_corr_max"],
            cognitive_expected_steps_mean=scalar_vals["cog_exp_steps_mean"],
            cognitive_expected_steps_easy_q25=scalar_vals["cog_exp_steps_easy_q25"],
            cognitive_expected_steps_hard_q75=scalar_vals["cog_exp_steps_hard_q75"],
            cognitive_cf_sign_flip_k0=scalar_vals["cog_cf_flip_k0"],
            cognitive_cf_sign_flip_final=scalar_vals["cog_cf_flip_final"],
            cognitive_conflict_mean=scalar_vals["cog_conflict_mean"],
            cognitive_scratchpad_novelty_mean=scalar_vals["cog_scratchpad_novelty"],
            cognitive_pruned_context_ratio=cog_pruned_ratio,
            cognitive_lipschitz_upper_bound=cog_lip,
        )

    def _collect_hardware_telemetry(self, now: float) -> dict[str, Any]:
        """Registra VRAM CUDA (si el modelo corre en GPU) y throughput sesiones/seg."""
        alloc_mb = 0.0
        reserved_mb = 0.0
        max_alloc_mb = 0.0
        first_param = next(self.model.parameters(), None)
        on_cuda = first_param is not None and first_param.is_cuda
        if on_cuda and torch.cuda.is_available():  # pragma: no cover - entorno CPU sin GPU
            dev = first_param.device
            scale = 1.0 / (1024.0 * 1024.0)
            alloc_mb = float(torch.cuda.memory_allocated(dev)) * scale
            reserved_mb = float(torch.cuda.memory_reserved(dev)) * scale
            max_alloc_mb = float(torch.cuda.max_memory_allocated(dev)) * scale

        ref_t = self._last_log_wall_time if self._last_log_wall_time is not None else now
        elapsed = max(now - ref_t, 1e-6)
        sessions_per_sec = float(self._sessions_since_last_log) / elapsed
        step_ms = (
            (now - self._step_start_time) * 1000.0
            if self._step_start_time is not None
            else elapsed * 1000.0
        )
        self._sessions_since_last_log = 0
        self._last_log_wall_time = now
        return dict(
            device=str(first_param.device) if first_param is not None else "cpu",
            cuda_vram_allocated_mb=alloc_mb,
            cuda_vram_reserved_mb=reserved_mb,
            cuda_vram_max_allocated_mb=max_alloc_mb,
            sessions_per_sec=sessions_per_sec,
            step_time_ms=step_ms,
        )

    def _summarize_activations(
        self, raw_acts: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Agrega estadísticas de activación por módulo, órgano y totales globales."""
        if not raw_acts:
            return dict(
                n_modules_probed=0,
                max_dead_fraction=0.0,
                max_saturation_fraction=0.0,
                total_activation_nans=0,
                total_activation_infs=0,
                per_organ={},
                per_module={},
            )

        by_organ: dict[str, list[dict[str, Any]]] = {}
        for stats in raw_acts.values():
            by_organ.setdefault(str(stats["organ"]), []).append(stats)

        per_organ: dict[str, dict[str, Any]] = {}
        for organ, entries in by_organ.items():
            cnt = len(entries)
            per_organ[organ] = dict(
                n_modules=cnt,
                mean=sum(e["mean"] for e in entries) / cnt,
                std=sum(e["std"] for e in entries) / cnt,
                abs_max=max(e["abs_max"] for e in entries),
                zero_fraction=sum(e["zero_fraction"] for e in entries) / cnt,
                dead_fraction=max(e["dead_fraction"] for e in entries),
                saturation_fraction=max(e["saturation_fraction"] for e in entries),
                nan_count=sum(e["nan_count"] for e in entries),
                inf_count=sum(e["inf_count"] for e in entries),
            )

        return dict(
            n_modules_probed=len(raw_acts),
            max_dead_fraction=max(e["dead_fraction"] for e in raw_acts.values()),
            max_saturation_fraction=max(e["saturation_fraction"] for e in raw_acts.values()),
            total_activation_nans=sum(e["nan_count"] for e in raw_acts.values()),
            total_activation_infs=sum(e["inf_count"] for e in raw_acts.values()),
            per_organ=per_organ,
            per_module=raw_acts,
        )

    def _detect_step_anomalies(
        self,
        *,
        step: int,
        snapshots: dict[str, ModuleWeightGradSnapshot],
        activations: dict[str, Any],
        domain: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Evalúa todas las reglas de anomalía científica del paso actual."""
        found: list[dict[str, Any]] = []
        c = self.config

        for organ, snap in snapshots.items():
            if snap.weight_nan_count > 0 or snap.weight_inf_count > 0:
                found.append(
                    dict(
                        code="nonfinite_weights",
                        severity="critical",
                        organ=organ,
                        step=step,
                        value=snap.weight_nan_count + snap.weight_inf_count,
                        threshold=0,
                        message=(
                            f"Pesos no finitos en {organ}: "
                            f"nan={snap.weight_nan_count}, inf={snap.weight_inf_count}"
                        ),
                    )
                )
            if snap.grad_nan_count > 0 or snap.grad_inf_count > 0:
                found.append(
                    dict(
                        code="nonfinite_gradients",
                        severity="critical",
                        organ=organ,
                        step=step,
                        value=snap.grad_nan_count + snap.grad_inf_count,
                        threshold=0,
                        message=(
                            f"Gradientes no finitos en {organ}: "
                            f"nan={snap.grad_nan_count}, inf={snap.grad_inf_count}"
                        ),
                    )
                )
            if snap.grad_exploding:
                ref_v = (
                    snap.grad_pre_clip_l2
                    if snap.grad_pre_clip_l2 is not None
                    else snap.grad_post_clip_l2
                )
                found.append(
                    dict(
                        code="exploding_gradients",
                        severity="critical",
                        organ=organ,
                        step=step,
                        value=ref_v,
                        threshold=c.grad_explode_threshold,
                        message=f"Gradiente explosivo en {organ}: l2={ref_v}",
                    )
                )
            if snap.grad_vanishing:
                ref_v = (
                    snap.grad_pre_clip_l2
                    if snap.grad_pre_clip_l2 is not None
                    else snap.grad_post_clip_l2
                )
                found.append(
                    dict(
                        code="vanishing_gradients",
                        severity="warning",
                        organ=organ,
                        step=step,
                        value=ref_v,
                        threshold=c.grad_vanish_threshold,
                        message=f"Gradiente evanescente en {organ}: l2={ref_v}",
                    )
                )
            if snap.update_dead:
                found.append(
                    dict(
                        code="dead_updates",
                        severity="warning",
                        organ=organ,
                        step=step,
                        value=snap.update_to_weight_ratio,
                        threshold=c.update_ratio_dead_threshold,
                        message=(
                            f"Ratio Karpathy ||ΔW||/||W|| muerto en {organ}: "
                            f"{snap.update_to_weight_ratio:.3e}"
                        ),
                    )
                )
            if snap.update_unstable:
                found.append(
                    dict(
                        code="unstable_updates",
                        severity="critical",
                        organ=organ,
                        step=step,
                        value=snap.update_to_weight_ratio,
                        threshold=c.update_ratio_unstable_threshold,
                        message=(
                            f"Ratio Karpathy ||ΔW||/||W|| inestable en {organ}: "
                            f"{snap.update_to_weight_ratio:.3e}"
                        ),
                    )
                )

        # Activaciones: saturación, dead-SiLU y no finitos
        for organ, a_stat in activations.get("per_organ", {}).items():
            if a_stat["nan_count"] > 0 or a_stat["inf_count"] > 0:
                found.append(
                    dict(
                        code="nonfinite_activations",
                        severity="critical",
                        organ=organ,
                        step=step,
                        value=a_stat["nan_count"] + a_stat["inf_count"],
                        threshold=0,
                        message=f"Activaciones no finitas en {organ}",
                    )
                )
            if (
                a_stat["saturation_fraction"] >= c.saturation_threshold
                or a_stat["dead_fraction"] >= c.dead_activation_threshold
            ):
                found.append(
                    dict(
                        code="activation_saturation",
                        severity="warning",
                        organ=organ,
                        step=step,
                        value=max(a_stat["saturation_fraction"], a_stat["dead_fraction"]),
                        threshold=min(c.saturation_threshold, c.dead_activation_threshold),
                        message=(
                            f"Saturación/muerte de activaciones en {organ}: "
                            f"dead={a_stat['dead_fraction']:.3f}, "
                            f"sat={a_stat['saturation_fraction']:.3f}"
                        ),
                    )
                )

        # Dominio MoE (solo cuando el modelo tiene routers MoE activos)
        has_moe = "moe_routers" in self._organ_params
        if has_moe:
            ent = domain.get("moe_router_entropy")
            load_r = domain.get("moe_load_max_min_ratio")
            e_corr = domain.get("moe_expert_mean_corr")
            if (ent is not None and ent < c.dead_router_entropy_threshold) or (
                load_r is not None and load_r > c.moe_imbalance_alert_threshold
            ):
                found.append(
                    dict(
                        code="dead_routers",
                        severity="warning",
                        organ="moe_routers",
                        step=step,
                        value=ent if ent is not None else load_r,
                        threshold=c.dead_router_entropy_threshold,
                        message=(
                            f"Anomalía de router MoE: entropy={ent}, load_max_min={load_r}"
                        ),
                    )
                )
            if e_corr is not None and e_corr > c.moe_corr_alert_threshold:
                found.append(
                    dict(
                        code="expert_collapse",
                        severity="warning",
                        organ="moe_experts",
                        step=step,
                        value=e_corr,
                        threshold=c.moe_corr_alert_threshold,
                        message=f"Colapso de especialización MoE: corr={e_corr:.3f}",
                    )
                )

        if domain.get("pt1_bound_violation"):
            found.append(
                dict(
                    code="pt1_bound_violation",
                    severity="critical",
                    organ="neuromod_memory",
                    step=step,
                    value=dict(
                        alpha_min=domain.get("pt1_alpha_min"),
                        eta_max=domain.get("pt1_eta_max"),
                        theta_max=domain.get("pt1_theta_max"),
                    ),
                    threshold="pt1_box",
                    message="Violación de la caja contractiva de Lyapunov PT1",
                )
            )
        return found

    def _emit_tensorboard(self, step: int, record: dict[str, Any]) -> None:
        """Emite escalares agrupados e histogramas muestreados a TensorBoard si está activo."""
        writer = self._tb_writer
        if writer is None:
            return

        def _add(tag: str, val: Any) -> None:
            if isinstance(val, int | float) and math.isfinite(float(val)):
                writer.add_scalar(tag, float(val), step)

        for k, v in record["losses"].items():
            _add(f"Loss/{k}", v)

        opt_r = record["optimizer"]
        for g_name, lr_v in opt_r["lr_by_group"].items():
            _add(f"LR/{g_name}", lr_v)
        _add("LR/muon_lr", opt_r.get("muon_lr"))
        _add("LR/router_to_backbone_ratio", opt_r["router_to_backbone_lr_ratio"])
        _add("Optimizer/adamw_effective_step_rms", opt_r["adamw_effective_step_rms"])
        _add("Optimizer/adamw_adaptive_lr_rms", opt_r["adamw_adaptive_lr_rms"])
        if opt_r.get("muon_param_count", 0) > 0:
            _add("Optimizer/muon_momentum_buffer_l2", opt_r["muon_momentum_buffer_l2"])
            _add("Optimizer/muon_momentum_buffer_rms", opt_r["muon_momentum_buffer_rms"])
            _add("Optimizer/muon_momentum_buffer_abs_max", opt_r["muon_momentum_buffer_abs_max"])
        if opt_r.get("schedule_free_param_count", 0) > 0:
            _add("Optimizer/schedule_free_z_x_diff_l2", opt_r["schedule_free_z_x_diff_l2"])
            _add("Optimizer/schedule_free_z_x_diff_rms", opt_r["schedule_free_z_x_diff_rms"])
            _add("Optimizer/schedule_free_z_x_rel_diff", opt_r["schedule_free_z_x_rel_diff"])
            _add(
                "Optimizer/schedule_free_adaptive_lr_rms",
                opt_r["schedule_free_adaptive_lr_rms"],
            )

        grads = record["gradients"]
        _add("Gradients/global_pre_clip_l2", grads["global_pre_clip_l2"])
        _add("Gradients/global_post_clip_l2", grads["global_post_clip_l2"])
        _add("Gradients/clip_ratio", grads["clip_ratio"])

        for organ, snap in record["organs"].items():
            _add(f"Weights/{organ}/l2", snap["weight_l2"])
            _add(f"Weights/{organ}/rms", snap["weight_rms"])
            _add(f"Weights/{organ}/abs_max", snap["weight_abs_max"])
            _add(f"Weights/{organ}/spectral_approx", snap["weight_spectral_approx"])
            _add(f"Weights/{organ}/zero_fraction", snap["weight_zero_fraction"])
            _add(f"Gradients/{organ}/pre_clip_l2", snap["grad_pre_clip_l2"])
            _add(f"Gradients/{organ}/post_clip_l2", snap["grad_post_clip_l2"])
            _add(f"Gradients/{organ}/abs_max", snap["grad_abs_max"])
            _add(f"UpdateRatio/{organ}/ratio", snap["update_to_weight_ratio"])
            _add(f"UpdateRatio/{organ}/log10", snap["update_ratio_log10"])

        for organ, a_stat in record["activations"]["per_organ"].items():
            _add(f"Activations/{organ}/mean", a_stat["mean"])
            _add(f"Activations/{organ}/std", a_stat["std"])
            _add(f"Activations/{organ}/dead_fraction", a_stat["dead_fraction"])
            _add(f"Activations/{organ}/saturation_fraction", a_stat["saturation_fraction"])

        dom = record["domain"]
        _add("Domain/MoE/router_entropy", dom["moe_router_entropy"])
        _add("Domain/MoE/load_max_min_ratio", dom["moe_load_max_min_ratio"])
        _add("Domain/MoE/expert_mean_corr", dom["moe_expert_mean_corr"])
        _add("Domain/Loop/k_used", dom["loop_k_used"])
        _add("Domain/Loop/delta_energy_neg_pct", dom["loop_delta_energy_neg_pct"])
        _add("Domain/Loop/max_spectral_norm", dom["loop_max_spectral_norm"])
        _add("Domain/Loop/halt_at_k1_pct", dom.get("loop_halt_at_k1_pct"))
        _add("Domain/Loop/cf_fragility_final", dom.get("cf_fragility_final"))
        _add("Domain/Loop/cf_sign_flip_rate_final", dom.get("cf_sign_flip_rate_final"))
        _add("Domain/Loop/step_router_js_div", dom.get("loop_step_router_js_div"))
        _add("Domain/Loop/landmark_attn_entropy", dom.get("landmark_attn_entropy"))
        _add("Domain/TitansPT1/alpha_min", dom["pt1_alpha_min"])
        _add("Domain/TitansPT1/alpha_mean", dom["pt1_alpha_mean"])
        _add("Domain/TitansPT1/eta_max", dom["pt1_eta_max"])
        _add("Domain/TitansPT1/theta_max", dom["pt1_theta_max"])
        _add("Domain/HRR/ood_energy_mean", dom["hrr_ood_energy_mean"])
        _add("Domain/CognitiveCore/pruned_context_ratio", dom["cognitive_pruned_context_ratio"])
        _add("Domain/CognitiveCore/tool_entropy", dom["cognitive_tool_entropy"])
        _add(
            "Domain/CognitiveCore/tool_load_max_min_ratio",
            dom.get("cognitive_tool_load_max_min_ratio"),
        )
        _add("Domain/CognitiveCore/tool_active_count", dom.get("cognitive_tool_active_count"))
        _add("Domain/CognitiveCore/tool_corr_max", dom.get("cognitive_tool_corr_max"))
        _add("Domain/CognitiveCore/cf_sign_flip_final", dom.get("cognitive_cf_sign_flip_final"))
        for t_name, t_val in dom["cognitive_tool_usage"].items():
            _add(f"Domain/CognitiveCore/tool_usage/{t_name}", t_val)

        hw = record["hardware"]
        _add("Hardware/vram_allocated_mb", hw["cuda_vram_allocated_mb"])
        _add("Hardware/vram_reserved_mb", hw["cuda_vram_reserved_mb"])
        _add("Hardware/sessions_per_sec", hw["sessions_per_sec"])

        # Histogramas SOLO cada `histogram_every_steps`
        if self.should_log_histograms(step) and hasattr(writer, "add_histogram"):
            with torch.no_grad():
                for organ, plist in self._organ_params.items():
                    w_flat = torch.cat([p.detach().float().reshape(-1) for _, p in plist])
                    w_fin = w_flat[torch.isfinite(w_flat)]
                    if w_fin.numel() > 0:
                        writer.add_histogram(f"HistWeights/{organ}", w_fin.cpu(), step)
                    g_list = [
                        p.grad.detach().float().reshape(-1) for _, p in plist if p.grad is not None
                    ]
                    if g_list:
                        g_flat = torch.cat(g_list)
                        g_fin = g_flat[torch.isfinite(g_flat)]
                        if g_fin.numel() > 0:
                            writer.add_histogram(f"HistGrads/{organ}", g_fin.cpu(), step)

        if hasattr(writer, "flush"):
            writer.flush()

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self.config.enable_jsonl:
            return
        sanitized = _json_sanitize(record)
        line = json.dumps(sanitized, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        if self.config.compress_jsonl or self._jsonl_path.suffix == ".gz":
            # Miembro gzip concatenado RFC 1952: append-only, legible en caliente y ultra-compacto
            with gzip.open(self._jsonl_path, "ab", compresslevel=6) as fh:
                fh.write(line.encode("utf-8"))
                fh.flush()
        else:
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    @torch.no_grad()
    def end_step(
        self,
        step: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        loss_dict: dict[str, Any] | None = None,
        model_output: dict[str, Any] | None = None,
        titans_update_diag: dict[str, Any] | None = None,
        cognitive_core_diag: dict[str, Any] | None = None,
        extra_metrics: dict[str, Any] | None = None,
        n_sessions: int = 1,
    ) -> dict[str, Any] | None:
        """Cierra el paso: retorna `None` con coste cero en pasos no muestreados."""
        self._total_steps_seen += 1
        self._sessions_since_last_log += max(int(n_sessions), 0)

        if not self.should_log(step):
            if self._activation_probe is not None and self._activation_probe.active:
                self._activation_probe.set_active(False)
            return None

        # Si el usuario no llamó `after_grad_clip`, capturar gradientes actuales como post-clip
        if self.config.track_gradients and self._captured_post_clip_step != int(step):
            self._post_clip_grad_stats, self._post_clip_global = self._collect_grad_stats()

        snapshots, global_weight_upd = self._collect_weight_and_update_stats(
            compute_spectral=self.should_log_spectral(step),
        )
        self._last_snapshots = snapshots

        # Resumen global de gradientes pre/post-clip
        pre_g_l2 = self._pre_clip_global.get("l2")
        post_g_l2 = self._post_clip_global.get("l2")
        clip_ratio: float | None = None
        if (
            pre_g_l2 is not None
            and post_g_l2 is not None
            and math.isfinite(pre_g_l2)
            and math.isfinite(post_g_l2)
            and pre_g_l2 > 1e-12
        ):
            clip_ratio = post_g_l2 / pre_g_l2

        grad_summary = dict(
            global_pre_clip_l2=pre_g_l2,
            global_post_clip_l2=post_g_l2,
            clip_ratio=clip_ratio,
            global_abs_max=self._pre_clip_global.get(
                "abs_max", self._post_clip_global.get("abs_max")
            ),
            total_nan_count=max(
                int(self._pre_clip_global.get("nan_count", 0)),
                int(self._post_clip_global.get("nan_count", 0)),
            ),
            total_inf_count=max(
                int(self._pre_clip_global.get("inf_count", 0)),
                int(self._post_clip_global.get("inf_count", 0)),
            ),
            has_nonfinite=bool(
                self._pre_clip_global.get("has_nonfinite", False)
                or self._post_clip_global.get("has_nonfinite", False)
            ),
        )

        opt_used = optimizer if optimizer is not None else self.optimizer
        opt_telemetry = self._collect_optimizer_telemetry(opt_used)
        losses = self._collect_losses(loss_dict)

        raw_acts = self._activation_probe.collect() if self._activation_probe is not None else {}
        if self._activation_probe is not None:
            self._activation_probe.set_active(False)
        activations = self._summarize_activations(raw_acts)

        domain = self._collect_domain_telemetry(
            model_output=model_output,
            titans_update_diag=titans_update_diag,
            cognitive_core_diag=cognitive_core_diag,
        )
        hardware = self._collect_hardware_telemetry(time.perf_counter())

        step_anomalies = self._detect_step_anomalies(
            step=int(step),
            snapshots=snapshots,
            activations=activations,
            domain=domain,
        )
        self._anomalies.extend(step_anomalies)

        record: dict[str, Any] = dict(
            step=int(step),
            timestamp_utc=_utc_now_iso(),
            weights=global_weight_upd,
            gradients=grad_summary,
            organs={k: v.to_dict() for k, v in snapshots.items()},
            optimizer=opt_telemetry,
            losses=losses,
            activations=activations,
            domain=domain,
            hardware=hardware,
            anomalies=step_anomalies,
            extra=_json_sanitize(extra_metrics) if extra_metrics else {},
        )

        self._write_jsonl(record)
        self._emit_tensorboard(int(step), record)
        self._update_online_summary(record)
        self._history.append(record)

        # Liberar clones de pesos inmediatamente para no retener VRAM
        self._pre_step_weights.clear()
        self._pre_step_organ_has_grad.clear()
        return record

    def _update_online_summary(self, record: dict[str, Any]) -> None:
        """Actualiza acumuladores online O(1) RAM para `summarize_run()` exacto en 120k+ pasos."""
        self._n_logged_steps += 1
        self._last_logged_step = int(record["step"])
        self._last_opt_telemetry = dict(record["optimizer"])

        for item in record["anomalies"]:
            self._total_anomalies_seen += 1
            code = str(item["code"])
            if code in self._anomaly_flags:
                self._anomaly_flags[code] = True
                if item.get("organ"):
                    self._affected_sets[code].add(str(item["organ"]))

        acts_per_organ = record["activations"]["per_organ"]
        for organ in self._organ_params:
            acc = self._organ_online.get(organ)
            if acc is None:
                acc = dict(
                    weight_l2_last=None,
                    weight_l2_max=None,
                    grad_l2_sum=0.0,
                    grad_l2_count=0,
                    grad_l2_max=None,
                    ur_sum=0.0,
                    ur_count=0,
                    ur_min=None,
                    ur_max=None,
                    act_dead_max=0.0,
                    act_sat_max=0.0,
                )
                self._organ_online[organ] = acc
            snap = record["organs"].get(organ)
            if snap is not None:
                w_l2 = snap["weight_l2"]
                if isinstance(w_l2, float) and math.isfinite(w_l2):
                    acc["weight_l2_last"] = w_l2
                    acc["weight_l2_max"] = (
                        w_l2 if acc["weight_l2_max"] is None else max(acc["weight_l2_max"], w_l2)
                    )
                g_ref = (
                    snap["grad_pre_clip_l2"]
                    if snap["grad_pre_clip_l2"] is not None
                    else snap["grad_post_clip_l2"]
                )
                if isinstance(g_ref, float) and math.isfinite(g_ref):
                    acc["grad_l2_sum"] += g_ref
                    acc["grad_l2_count"] += 1
                    acc["grad_l2_max"] = (
                        g_ref if acc["grad_l2_max"] is None else max(acc["grad_l2_max"], g_ref)
                    )
                ur = snap["update_to_weight_ratio"]
                if isinstance(ur, float) and math.isfinite(ur):
                    acc["ur_sum"] += ur
                    acc["ur_count"] += 1
                    acc["ur_min"] = ur if acc["ur_min"] is None else min(acc["ur_min"], ur)
                    acc["ur_max"] = ur if acc["ur_max"] is None else max(acc["ur_max"], ur)
            a_org = acts_per_organ.get(organ)
            if a_org is not None:
                acc["act_dead_max"] = max(acc["act_dead_max"], float(a_org["dead_fraction"]))
                acc["act_sat_max"] = max(acc["act_sat_max"], float(a_org["saturation_fraction"]))

        for lk, lv_raw in record["losses"].items():
            lv = float(lv_raw)
            if not math.isfinite(lv):
                continue
            l_acc = self._loss_online.get(lk)
            if l_acc is None:
                self._loss_online[lk] = dict(first=lv, last=lv, min=lv, max=lv, sum=lv, count=1)
            else:
                l_acc["last"] = lv
                if lv < l_acc["min"]:
                    l_acc["min"] = lv
                if lv > l_acc["max"]:
                    l_acc["max"] = lv
                l_acc["sum"] += lv
                l_acc["count"] += 1

        dom = record["domain"]
        for d_key in _DOMAIN_MEAN_KEYS:
            dv = dom.get(d_key)
            if dv is not None:
                fv = float(dv)
                if math.isfinite(fv):
                    slot = self._dom_mean_online[d_key]
                    slot[0] += fv
                    slot[1] += 1.0
        a_min = dom.get("pt1_alpha_min")
        if a_min is not None and math.isfinite(float(a_min)):
            fa = float(a_min)
            self._dom_pt1_alpha_min = (
                fa if self._dom_pt1_alpha_min is None else min(self._dom_pt1_alpha_min, fa)
            )
        e_max = dom.get("pt1_eta_max")
        if e_max is not None and math.isfinite(float(e_max)):
            fe = float(e_max)
            self._dom_pt1_eta_max = (
                fe if self._dom_pt1_eta_max is None else max(self._dom_pt1_eta_max, fe)
            )
        t_max = dom.get("pt1_theta_max")
        if t_max is not None and math.isfinite(float(t_max)):
            ft = float(t_max)
            self._dom_pt1_theta_max = (
                ft if self._dom_pt1_theta_max is None else max(self._dom_pt1_theta_max, ft)
            )

        hw = record["hardware"]
        self._hw_peak_vram_alloc_mb = max(
            self._hw_peak_vram_alloc_mb, float(hw["cuda_vram_allocated_mb"])
        )
        self._hw_peak_vram_res_mb = max(
            self._hw_peak_vram_res_mb, float(hw["cuda_vram_reserved_mb"])
        )
        self._hw_tput_sum += float(hw["sessions_per_sec"])
        self._hw_tput_count += 1

    def summarize_run(self) -> dict[str, Any]:
        """Genera un informe completo de salud en O(1) RAM exacto sobre todos los pasos vistos."""
        anomaly_flags = dict(self._anomaly_flags)
        affected_organs = {k: sorted(v) for k, v in self._affected_sets.items() if v}

        # Resumen por órgano desde acumuladores online exactos
        organ_summary: dict[str, dict[str, Any]] = {}
        target_log10 = math.log10(self.config.update_ratio_target)
        for organ in self._organ_params:
            acc = self._organ_online.get(organ)
            if acc is None:
                organ_summary[organ] = dict(
                    weight_l2_last=None,
                    weight_l2_max=None,
                    grad_l2_mean=None,
                    grad_l2_max=None,
                    update_ratio_mean=None,
                    update_ratio_min=None,
                    update_ratio_max=None,
                    update_ratio_target_decade_gap=None,
                    activation_dead_max=0.0,
                    activation_saturation_max=0.0,
                )
                continue
            mean_g = acc["grad_l2_sum"] / acc["grad_l2_count"] if acc["grad_l2_count"] > 0 else None
            mean_ur = acc["ur_sum"] / acc["ur_count"] if acc["ur_count"] > 0 else None
            organ_summary[organ] = dict(
                weight_l2_last=acc["weight_l2_last"],
                weight_l2_max=acc["weight_l2_max"],
                grad_l2_mean=mean_g,
                grad_l2_max=acc["grad_l2_max"],
                update_ratio_mean=mean_ur,
                update_ratio_min=acc["ur_min"],
                update_ratio_max=acc["ur_max"],
                update_ratio_target_decade_gap=(
                    abs(math.log10(max(mean_ur, 1e-15)) - target_log10)
                    if mean_ur is not None
                    else None
                ),
                activation_dead_max=acc["act_dead_max"],
                activation_saturation_max=acc["act_sat_max"],
            )

        # Resumen de pérdidas exacto (incluso `first` de pasos antiguos ya expulsados del deque)
        loss_summary: dict[str, dict[str, float]] = {}
        for lk in sorted(self._loss_online.keys()):
            l_acc = self._loss_online[lk]
            loss_summary[lk] = dict(
                first=l_acc["first"],
                last=l_acc["last"],
                min=l_acc["min"],
                max=l_acc["max"],
                mean=l_acc["sum"] / float(l_acc["count"]),
            )

        # Resumen de LRs y optimizador
        last_opt = self._last_opt_telemetry
        lr_summary = dict(
            last_backbone_lr=last_opt.get("backbone_lr"),
            last_router_lr=last_opt.get("router_lr"),
            last_muon_lr=last_opt.get("muon_lr"),
            last_router_to_backbone_ratio=last_opt.get("router_to_backbone_lr_ratio"),
            last_lr_by_group=last_opt.get("lr_by_group", {}),
            last_adamw_effective_step_rms=last_opt.get("adamw_effective_step_rms", 0.0),
            last_muon_momentum_buffer_l2=last_opt.get("muon_momentum_buffer_l2", 0.0),
            last_schedule_free_z_x_diff_l2=last_opt.get("schedule_free_z_x_diff_l2", 0.0),
        )

        # Resumen de dominio online exacto
        def _mean_dom(key: str) -> float | None:
            s_val, c_val = self._dom_mean_online[key]
            return s_val / c_val if c_val > 0.0 else None

        domain_summary = dict(
            moe_router_entropy_mean=_mean_dom("moe_router_entropy"),
            moe_load_max_min_mean=_mean_dom("moe_load_max_min_ratio"),
            moe_expert_mean_corr=_mean_dom("moe_expert_mean_corr"),
            loop_k_used_mean=_mean_dom("loop_k_used"),
            loop_delta_energy_neg_pct_mean=_mean_dom("loop_delta_energy_neg_pct"),
            loop_halt_at_k1_pct_mean=_mean_dom("loop_halt_at_k1_pct"),
            cf_fragility_final_mean=_mean_dom("cf_fragility_final"),
            cf_sign_flip_rate_final_mean=_mean_dom("cf_sign_flip_rate_final"),
            pt1_alpha_min_overall=self._dom_pt1_alpha_min,
            pt1_eta_max_overall=self._dom_pt1_eta_max,
            pt1_theta_max_overall=self._dom_pt1_theta_max,
            hrr_ood_energy_mean=_mean_dom("hrr_ood_energy_mean"),
            cognitive_pruned_context_ratio_mean=_mean_dom("cognitive_pruned_context_ratio"),
            cognitive_tool_entropy_mean=_mean_dom("cognitive_tool_entropy"),
            cognitive_tool_load_max_min_mean=_mean_dom("cognitive_tool_load_max_min_ratio"),
            cognitive_tool_active_count_mean=_mean_dom("cognitive_tool_active_count"),
        )

        # Resumen de hardware online exacto
        hardware_summary = dict(
            peak_cuda_vram_allocated_mb=self._hw_peak_vram_alloc_mb,
            peak_cuda_vram_reserved_mb=self._hw_peak_vram_res_mb,
            mean_sessions_per_sec=(
                self._hw_tput_sum / float(self._hw_tput_count) if self._hw_tput_count > 0 else 0.0
            ),
            tensorboard_active=self.tensorboard_active,
            jsonl_path=str(self._jsonl_path) if self.config.enable_jsonl else None,
        )

        healthy = not any(anomaly_flags.values())
        return dict(
            healthy=healthy,
            n_logged_steps=self._n_logged_steps,
            history_retained_steps=len(self._history),
            total_steps_seen=self._total_steps_seen,
            last_step=self._last_logged_step,
            anomaly_flags=anomaly_flags,
            affected_organs=affected_organs,
            anomalies=list(self._anomalies),
            total_anomalies_count=self._total_anomalies_seen,
            organ_summary=organ_summary,
            loss_summary=loss_summary,
            lr_summary=lr_summary,
            domain_summary=domain_summary,
            hardware_summary=hardware_summary,
        )

    def close(self) -> None:
        """Limpia hooks de activación y cierra el escritor de TensorBoard si está abierto."""
        if self._activation_probe is not None:
            self._activation_probe.remove()
        self._pre_step_weights.clear()
        self._pre_step_organ_has_grad.clear()
        if self._tb_writer is not None and self._owns_tb_writer:
            if hasattr(self._tb_writer, "flush"):
                self._tb_writer.flush()
            if hasattr(self._tb_writer, "close"):
                self._tb_writer.close()
            self._tb_writer = None


__all__ = [
    "ARCHITECTURAL_ORGANS",
    "ActivationProbe",
    "ModuleWeightGradSnapshot",
    "ResearchTelemetryMonitor",
    "TelemetryConfig",
    "classify_organ",
    "classify_param_organ",
    "read_telemetry_jsonl",
]
