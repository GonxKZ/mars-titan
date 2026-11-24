"""Arnés de larga duración para `MMARRS-TITAN` (historia 2000–2023+ -> test sellado `2024+`).

Aplica el patrón *Effective Harnesses for Long-Running Agents* (Inicializador + Trabajador
Incremental + Ledger Estructurado + Compactación Zero-Drift + Verificación Automática E2E) al
entrenamiento e inferencia causal multi-década de `MMARRS-TITAN` sobre NVIDIA RTX PRO 6000
Blackwell (`sm_120`) / Colab G4 / CPU:

  1. `HarnessConfig`: contrato inmutable para modos `full_history_final` (`2000–2022/2023` todo
     de golpe -> `2024+` sellado) y `expanding_walk_forward`, checkpoints periódicos, compactación
     cada `compaction_interval` sesiones y umbrales de salud.
  2. `SessionHealthGate` + `HealthReport`: puerta automática tras cada sesión/checkpoint que
     verifica (a) finitud de pesos/predicciones/estado, (b) 9 cuantiles no cruzados, (c) cota
     de Lyapunov PT1 de memoria, (d) balanceo MoE `max/min`, (e) picos OOD de Hopfield y
     (f) alerta CUSUM de decaimiento de IC; dispara rollback seguro ante fallos duros.
  3. `compact_working_state`: compactación periódica *Zero-Drift* que atenúa suavemente los
     buffers rápidos (`fast`, `fast_mom`) y descarta cachés efímeros preservando bit-a-bit los
     bancos cristalizados Benna-Fusi (`medium`, `slow`) y la memoria episódica `HRR-Hopfield`.
  4. `LongRunningTitansHarness`: fase inicializadora (`manifest.json` SHA-256 + `progress_ledger`
     append-only sin sobrescritura silenciosa) y trabajador incremental con checkpoints atómicos
     reanudables y cumplimiento estricto del sello `2024+`.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import inspect
import json
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .ablation_runner import _infer_dims, _is_episodic_buffer
from .config import DEFAULT_HORIZONS, VARIATION_REGISTRY
from .full_history_train import (
    SPLIT_MODES,
    FullHistorySplitConfig,
    _apply_structured_modality_dropout,
    _update_ema,
    split_chronological_sessions,
)
from .granular_heads import MMARRSMultiTaskLoss
from .hardware_blackwell import (
    BlackwellRuntimeConfig,
    autocast_context,
    build_optimizer,
    configure_blackwell_runtime,
    maybe_compile_model,
)
from .model import MMARRSTitan, build_variation

_TRANSIENT_STATE_KEYS: tuple[str, ...] = (
    "last_x0",
    "last_specialist_preds",
    "last_expert_loads",
)

PIPELINE_STAGES: tuple[str, ...] = (
    "stage1_pretraining",
    "stage2_posttraining",
)

HARNESS_OPTIMIZER_NAMES: tuple[str, ...] = (
    "adamw_fused",
    "muon_hybrid",
    "schedule_free_adamw",
)

DEFAULT_MATCHED_CONTROLS: dict[str, str] = {
    "dense_iso_flop": "v00_dense_baseline",
    "dense_iso_param": "v01_dense_wide_iso_param",
    "uniform_1_over_e": "v04_moe_uniform_control",
    "shuffled_placebo": "v05_moe_shuffled_ctx_placebo",
    "k1_fixed_depth": "v03_moe_only_k1",
}


class SealedTestViolationError(ValueError):
    """Error lanzado ante cualquier intento de filtrar sesiones `year >= sealed_test_start_year`."""


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _is_gzip_file(path: Path) -> bool:
    if path.suffix == ".gz":
        return True
    try:
        with path.open("rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _append_jsonl_batch(
    path: Path,
    records: Sequence[dict[str, Any]],
    *,
    sync: bool = False,
) -> None:
    """Anexa un lote de registros JSON en `.jsonl` o un único miembro RFC 1952 `.jsonl.gz`."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        for rec in records
    ).encode("utf-8")
    if path.suffix == ".gz":
        with path.open("ab") as raw_fh:
            with gzip.GzipFile(
                filename="", fileobj=raw_fh, mode="wb", compresslevel=6, mtime=0
            ) as gz_fh:
                gz_fh.write(payload)
            raw_fh.flush()
            if sync:
                os.fsync(raw_fh.fileno())
    else:
        with path.open("ab") as fh:
            fh.write(payload)
            fh.flush()
            if sync:
                os.fsync(fh.fileno())


def _append_jsonl(path: Path, record: dict[str, Any], *, sync: bool = False) -> None:
    """Anexa un registro JSON compacto en `.jsonl` o `.jsonl.gz` (multi-miembro RFC 1952)."""
    _append_jsonl_batch(path, (record,), sync=sync)


def _supports_gpu_found_inf(opt: torch.optim.Optimizer, device: torch.device) -> bool:
    """Detecta `AdamW(fused=True)` en CUDA con soporte nativo de `found_inf` sin sync host."""
    return (
        device.type == "cuda"
        and isinstance(opt, torch.optim.AdamW)
        and bool(opt.param_groups)
        and all(bool(g.get("fused", False)) for g in opt.param_groups)
    )


@torch.no_grad()
def _update_ema_masked(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
    step_finite: torch.Tensor,
) -> None:
    """Actualiza EMA de parámetros y buffers no episódicos sin syncs host y a prueba de NaN."""
    sf = step_finite.to(dtype=torch.bool)
    alpha_f32 = sf.to(dtype=torch.float32) * (1.0 - float(decay))
    for p_ema, p_src in zip(ema_model.parameters(), model.parameters(), strict=True):
        p_d = p_src.detach().to(device=p_ema.device, dtype=p_ema.dtype)
        sf_p = sf.to(device=p_ema.device)
        safe_p = torch.where(sf_p & torch.isfinite(p_d), p_d, p_ema)
        p_ema.lerp_(safe_p, alpha_f32.to(device=p_ema.device, dtype=p_ema.dtype))
    for (b_name, b_ema), (_, b_src) in zip(
        ema_model.named_buffers(), model.named_buffers(), strict=True
    ):
        if _is_episodic_buffer(b_name) or not b_ema.dtype.is_floating_point:
            continue
        b_d = b_src.detach().to(device=b_ema.device, dtype=b_ema.dtype)
        sf_b = sf.to(device=b_ema.device)
        b_ema.copy_(torch.where(sf_b & torch.isfinite(b_d), b_d, b_ema))


def _align_rows_by_asset_ids(
    cur_ids: torch.Tensor,
    prev_ids: torch.Tensor,
    prev_tensor: torch.Tensor,
) -> torch.Tensor:
    """Alinea eje 0 de `prev_tensor` por `asset_ids` en dispositivo (O(N log N), 0 syncs)."""
    p_ids = prev_ids.to(device=cur_ids.device, dtype=cur_ids.dtype)
    p_t = prev_tensor.to(device=cur_ids.device)
    if p_ids.numel() == 0 or cur_ids.numel() == 0:
        return torch.zeros(
            (cur_ids.shape[0], *p_t.shape[1:]),
            device=cur_ids.device,
            dtype=p_t.dtype,
        )
    order = torch.argsort(p_ids)
    sorted_prev = p_ids[order]
    sorted_t = p_t[order]
    pos = torch.searchsorted(sorted_prev, cur_ids)
    safe_pos = pos.clamp_max(sorted_prev.numel() - 1)
    matched = (pos < sorted_prev.numel()) & (sorted_prev[safe_pos] == cur_ids)
    gathered = sorted_t[safe_pos]
    mask = matched.view((-1,) + (1,) * (gathered.ndim - 1))
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def _sanitize_matured_input_state(
    state: dict[str, object],
    step_finite: torch.Tensor,
) -> dict[str, object]:
    """Sanea entradas efímeras de `step_matured` en GPU si `step_finite=False` (0 syncs host)."""
    sf = step_finite.to(dtype=torch.bool)
    san = dict(state)
    for key in ("last_x0", "last_regime_vec", "last_specialist_preds"):
        val = san.get(key)
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            cond = sf.to(device=val.device) & torch.isfinite(val)
            san[key] = torch.where(cond, val, torch.zeros_like(val))
    loads = san.get("last_expert_loads")
    if isinstance(loads, list):
        safe_loads: list[object] = []
        for item in loads:
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                cond = sf.to(device=item.device) & torch.isfinite(item)
                unif = torch.full_like(item, 1.0 / float(max(1, item.numel())))
                safe_loads.append(torch.where(cond, item, unif))
            else:
                safe_loads.append(item)
        san["last_expert_loads"] = safe_loads
    return san


def _sanitize_causal_state_masked(
    prev_state: dict[str, object] | None,
    next_state: dict[str, object],
    step_finite: torch.Tensor,
) -> dict[str, object]:
    """Preserva `titans_state` previos (incl. paneles ragged) si `step_finite=False` (0 syncs)."""
    sf = step_finite.to(dtype=torch.bool)
    merged = dict(next_state)
    prev_ts = prev_state.get("titans_state") if isinstance(prev_state, dict) else None
    next_ts = next_state.get("titans_state")
    prev_ids_any = (
        prev_ts.get("asset_ids")
        if isinstance(prev_ts, dict) and isinstance(prev_ts.get("asset_ids"), torch.Tensor)
        else (prev_state.get("last_asset_ids") if isinstance(prev_state, dict) else None)
    )
    cur_ids_any = (
        next_ts.get("asset_ids")
        if isinstance(next_ts, dict) and isinstance(next_ts.get("asset_ids"), torch.Tensor)
        else next_state.get("last_asset_ids")
    )

    if isinstance(next_ts, dict):
        merged_ts: dict[str, object] = {}
        for k, v_new in next_ts.items():
            v_old = prev_ts.get(k) if isinstance(prev_ts, dict) else None
            if not isinstance(v_new, torch.Tensor):
                merged_ts[k] = v_new
                continue
            cond = sf.to(device=v_new.device)
            if v_new.is_floating_point():
                if isinstance(v_old, torch.Tensor) and v_new.shape == v_old.shape:
                    fb = v_old.to(device=v_new.device, dtype=v_new.dtype)
                elif (
                    isinstance(v_old, torch.Tensor)
                    and v_new.ndim >= 1
                    and v_old.ndim == v_new.ndim
                    and v_new.shape[1:] == v_old.shape[1:]
                    and isinstance(cur_ids_any, torch.Tensor)
                    and isinstance(prev_ids_any, torch.Tensor)
                    and cur_ids_any.shape[0] == v_new.shape[0]
                    and prev_ids_any.shape[0] == v_old.shape[0]
                ):
                    fb = _align_rows_by_asset_ids(
                        cur_ids_any.to(device=v_new.device), prev_ids_any, v_old
                    ).to(dtype=v_new.dtype)
                else:
                    fb = torch.zeros_like(v_new)
                fb = torch.where(torch.isfinite(fb), fb, torch.zeros_like(fb))
                merged_ts[k] = torch.where(cond & torch.isfinite(v_new), v_new, fb)
            elif isinstance(v_old, torch.Tensor) and v_new.shape == v_old.shape:
                merged_ts[k] = torch.where(
                    cond, v_new, v_old.to(device=v_new.device, dtype=v_new.dtype)
                )
            else:
                merged_ts[k] = v_new
        merged["titans_state"] = merged_ts

    aw_new = next_state.get("agent_w_prev")
    if isinstance(aw_new, torch.Tensor) and aw_new.is_floating_point():
        aw_old = prev_state.get("agent_w_prev") if isinstance(prev_state, dict) else None
        if isinstance(aw_old, torch.Tensor) and aw_old.shape == aw_new.shape:
            aw_fb = aw_old.to(device=aw_new.device, dtype=aw_new.dtype)
        elif (
            isinstance(aw_old, torch.Tensor)
            and aw_new.ndim >= 1
            and aw_old.ndim == aw_new.ndim
            and aw_new.shape[1:] == aw_old.shape[1:]
            and isinstance(cur_ids_any, torch.Tensor)
            and isinstance(prev_ids_any, torch.Tensor)
            and cur_ids_any.shape[0] == aw_new.shape[0]
            and prev_ids_any.shape[0] == aw_old.shape[0]
        ):
            aw_fb = _align_rows_by_asset_ids(
                cur_ids_any.to(device=aw_new.device), prev_ids_any, aw_old
            ).to(dtype=aw_new.dtype)
        else:
            aw_fb = torch.zeros_like(aw_new)
        aw_fb = torch.where(torch.isfinite(aw_fb), aw_fb, torch.zeros_like(aw_fb))
        merged["agent_w_prev"] = torch.where(
            sf.to(device=aw_new.device) & torch.isfinite(aw_new), aw_new, aw_fb
        )
    return merged


def read_progress_ledger(ledger_path: str | Path) -> list[dict[str, Any]]:
    """Lee entradas válidas de `progress_ledger.jsonl` o `.jsonl.gz` de forma transparente."""
    p = Path(ledger_path)
    if not p.is_file():
        # Resolución transparente entre .jsonl y .jsonl.gz dentro del mismo run_dir
        alt = (
            p.with_name(f"{p.name}.gz")
            if p.suffix == ".jsonl"
            else (p.with_suffix("") if p.name.endswith(".jsonl.gz") else None)
        )
        if alt is not None and alt.is_file():
            p = alt
        else:
            return []
    opener = gzip.open if _is_gzip_file(p) else open
    entries: list[dict[str, Any]] = []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Línea JSONL corrupta en {p} línea {line_no}: {exc}"
                ) from exc
    return entries


def required_purge_embargo(active_horizons: Sequence[int]) -> tuple[int, int]:
    """Devuelve `(min_purge, min_embargo)` seguros para evitar fuga multi-horizonte (P0-4)."""
    if not active_horizons:
        raise ValueError("active_horizons no puede estar vacío")
    max_h = max(int(h) for h in active_horizons)
    if max_h < 1:
        raise ValueError("Todos los horizontes activos deben ser >= 1")
    return max_h, max(5, math.ceil(0.2 * max_h))


def _align_w_prev_by_asset_ids(
    cur_ids: torch.Tensor,
    prev_ids: torch.Tensor | None,
    prev_w: torch.Tensor | None,
) -> torch.Tensor | None:
    """Alinea `w_prev` por `asset_ids` en dispositivo (O(N log N), cero syncs CPU)."""
    if prev_ids is None or prev_w is None or prev_ids.numel() == 0 or cur_ids.numel() == 0:
        return None
    p_ids = prev_ids.to(device=cur_ids.device, dtype=cur_ids.dtype)
    p_w = prev_w.to(device=cur_ids.device)
    order = torch.argsort(p_ids)
    sorted_prev = p_ids[order]
    sorted_w = p_w[order]
    pos = torch.searchsorted(sorted_prev, cur_ids)
    safe_pos = pos.clamp_max(sorted_prev.numel() - 1)
    matched = (pos < sorted_prev.numel()) & (sorted_prev[safe_pos] == cur_ids)
    return torch.where(matched, sorted_w[safe_pos], torch.zeros_like(sorted_w[safe_pos]))


def _iter_sessions_h2d(
    sessions: Any,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    *,
    use_pinned: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    """Itera sesiones vía `ZeroCopySessionStore.iter_pinned_h2d` o doble búfer seguro."""
    if hasattr(sessions, "iter_pinned_h2d"):
        sig = inspect.signature(sessions.iter_pinned_h2d)
        if "indices" in sig.parameters:
            yield from sessions.iter_pinned_h2d(
                device, indices=range(start_idx, end_idx), use_pinned=use_pinned
            )
        else:
            sub = sessions[start_idx:end_idx]
            yield from sub.iter_pinned_h2d(device, use_pinned=use_pinned)
        return

    slice_batches = list(sessions[start_idx:end_idx])
    if not slice_batches:
        return
    if device.type != "cuda":
        for batch in slice_batches:
            yield {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
            }
        return

    # Doble búfer fijado seguro en CUDA con eventos por ranura + record_stream (sin race DMA)
    stream = torch.cuda.Stream(device=device)  # pragma: no cover
    staging: list[dict[str, torch.Tensor]] = [{}, {}]  # pragma: no cover
    slot_events = [torch.cuda.Event(), torch.cuda.Event()]  # pragma: no cover
    slot_in_flight = [False, False]  # pragma: no cover

    def _stage(batch_in: dict[str, Any], slot: int) -> dict[str, Any]:  # pragma: no cover
        if use_pinned and slot_in_flight[slot]:
            slot_events[slot].synchronize()
            slot_in_flight[slot] = False
        out_gpu: dict[str, Any] = {}
        bufs = staging[slot]
        for k, v in batch_in.items():
            if not isinstance(v, torch.Tensor):
                out_gpu[k] = v
            elif use_pinned and not v.is_cuda:
                prev = bufs.get(k)
                if prev is None or prev.shape != v.shape or prev.dtype != v.dtype:
                    prev = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                    bufs[k] = prev
                prev.copy_(v, non_blocking=False)
                out_gpu[k] = prev.to(device, non_blocking=True)
            else:
                out_gpu[k] = v.to(device, non_blocking=True)
        if use_pinned:
            slot_events[slot].record(stream)
            slot_in_flight[slot] = True
        return out_gpu

    with torch.cuda.stream(stream):  # pragma: no cover
        nxt = _stage(slice_batches[0], 0)
    for pos in range(len(slice_batches)):  # pragma: no cover
        cur_stream = torch.cuda.current_stream(device)
        cur_stream.wait_stream(stream)
        cur = nxt
        for val in cur.values():
            if isinstance(val, torch.Tensor):
                val.record_stream(cur_stream)
        if pos + 1 < len(slice_batches):
            with torch.cuda.stream(stream):
                nxt = _stage(slice_batches[pos + 1], (pos + 1) & 1)
        yield cur


def validate_no_sealed_leakage(
    session_years: Sequence[int],
    *,
    sealed_test_start_year: int = 2024,
) -> None:
    """Rechaza cualquier lista de sesiones de entrenamiento/ajuste con `year >= sello`."""
    offending = sorted({int(y) for y in session_years if int(y) >= sealed_test_start_year})
    if offending:
        raise SealedTestViolationError(
            f"Violación del sello {sealed_test_start_year}+: años prohibidos en entrenamiento/"
            f"ajuste detectados: {offending}"
        )


def _clone_state_to_cpu(state: dict[str, object] | None) -> dict[str, object] | None:
    """Serializa el estado causal a CPU sin compartir memoria ni grafo de autograd."""
    if state is None:
        return None
    out: dict[str, object] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, dict):
            out[k] = {
                sk: sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else copy.deepcopy(sv)
                for sk, sv in v.items()
            }
        elif isinstance(v, list):
            out[k] = [
                item.detach().cpu().clone()
                if isinstance(item, torch.Tensor)
                else copy.deepcopy(item)
                for item in v
            ]
        else:
            out[k] = copy.deepcopy(v)
    return out


def _move_state_to_device(
    state: dict[str, object] | None, device: torch.device
) -> dict[str, object] | None:
    """Restaura el estado causal en el dispositivo destino sin arrastrar gradientes."""
    if state is None:
        return None
    out: dict[str, object] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to(device=device).clone()
        elif isinstance(v, dict):
            out[k] = {
                sk: sv.detach().to(device=device).clone()
                if isinstance(sv, torch.Tensor)
                else copy.deepcopy(sv)
                for sk, sv in v.items()
            }
        elif isinstance(v, list):
            out[k] = [
                item.detach().to(device=device).clone()
                if isinstance(item, torch.Tensor)
                else copy.deepcopy(item)
                for item in v
            ]
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass(frozen=True)
class HarnessConfig:
    """Contrato inmutable del arnés de larga duración `LongRunningTitansHarness`."""

    run_dir: str
    variation: str = "v13_mmarrs_titan_full"
    split_mode: str = "full_history_final"
    checkpoint_every_sessions: int = 250
    compaction_interval: int = 63
    max_ood_energy: float = 4.0
    max_load_imbalance: float = 2.5
    cusum_alert_threshold: float = 0.25
    auto_recover_on_nan: bool = True
    sealed_test_start_year: int = 2024
    # Parámetros opcionales de estabilización y entrenamiento incremental
    max_memory_norm: float = 50.0
    fast_compaction_decay: float = 0.70
    momentum_compaction_decay: float = 0.50
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.995
    modality_dropout: float = 0.10
    grad_clip_norm: float = 2.0
    seed: int = 42
    device: str = "cuda"
    pipeline_stage: str = "stage1_pretraining"
    active_horizons: tuple[int, ...] = DEFAULT_HORIZONS
    health_check_every_k: int = 10
    compress_ledger: bool = True
    ledger_flush_every_k: int = 64
    optimizer_name: str = "adamw_fused"

    def __post_init__(self) -> None:
        if isinstance(self.run_dir, Path):
            object.__setattr__(self, "run_dir", str(self.run_dir))
        if not self.run_dir or not str(self.run_dir).strip():
            raise ValueError("run_dir no puede estar vacío")
        if self.variation not in VARIATION_REGISTRY:
            raise ValueError(
                f"Variación desconocida {self.variation!r}; válidas: {tuple(VARIATION_REGISTRY)}"
            )
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode inválido {self.split_mode!r}; válidos: {SPLIT_MODES}")
        if self.checkpoint_every_sessions < 1:
            raise ValueError("checkpoint_every_sessions debe ser >= 1")
        if self.compaction_interval < 1:
            raise ValueError("compaction_interval debe ser >= 1")
        if self.max_ood_energy <= 0.0:
            raise ValueError("max_ood_energy debe ser > 0")
        if self.max_load_imbalance < 1.0:
            raise ValueError("max_load_imbalance debe ser >= 1.0")
        if self.cusum_alert_threshold <= 0.0:
            raise ValueError("cusum_alert_threshold debe ser > 0")
        if self.sealed_test_start_year < 2001:
            raise ValueError("sealed_test_start_year debe ser >= 2001")
        if self.max_memory_norm <= 0.0:
            raise ValueError("max_memory_norm debe ser > 0")
        if not (0.0 < self.fast_compaction_decay < 1.0):
            raise ValueError("fast_compaction_decay debe pertenecer a (0, 1)")
        if not (0.0 <= self.momentum_compaction_decay < 1.0):
            raise ValueError("momentum_compaction_decay debe pertenecer a [0, 1)")
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim debe ser >= 8")
        if self.lr <= 0.0:
            raise ValueError("lr debe ser > 0")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay debe pertenecer a [0, 1)")
        if not (0.0 <= self.modality_dropout < 1.0):
            raise ValueError("modality_dropout debe pertenecer a [0, 1)")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm debe ser > 0")
        if self.pipeline_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"pipeline_stage inválido {self.pipeline_stage!r}; válidos: {PIPELINE_STAGES}"
            )
        if not self.active_horizons or any(int(h) < 1 for h in self.active_horizons):
            raise ValueError("active_horizons debe contener enteros >= 1")
        if self.health_check_every_k < 1:
            raise ValueError("health_check_every_k debe ser >= 1")
        if self.ledger_flush_every_k < 1:
            raise ValueError("ledger_flush_every_k debe ser >= 1")
        if self.optimizer_name not in HARNESS_OPTIMIZER_NAMES:
            raise ValueError(
                f"optimizer_name inválido {self.optimizer_name!r}; "
                f"válidos: {HARNESS_OPTIMIZER_NAMES}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuración completa como diccionario JSON-compatible."""
        return asdict(self)


@dataclass
class HealthReport:
    """Resultado estructurado de verificación de salud por sesión o checkpoint."""

    passed: bool
    warnings: list[str]
    hard_failures: list[str]
    metrics: dict[str, Any]
    rollback_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convierte el reporte de salud en diccionario listo para el ledger JSONL."""
        return dict(
            passed=self.passed,
            warnings=list(self.warnings),
            hard_failures=list(self.hard_failures),
            metrics=dict(self.metrics),
            rollback_triggered=self.rollback_triggered,
        )


def _batch_scalar_tensors_to_floats(tensors: Sequence[torch.Tensor]) -> list[float]:
    """Transfiere escalares agrupados por dispositivo en una sola sync D2H por dispositivo."""
    if not tensors:
        return []
    by_dev: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for idx, t in enumerate(tensors):
        by_dev.setdefault(t.device, []).append((idx, t.detach().float().reshape(())))
    out = [0.0] * len(tensors)
    for items in by_dev.values():
        stacked = torch.stack([t for _, t in items]).cpu().tolist()
        for (idx, _), val in zip(items, stacked, strict=True):
            out[idx] = float(val)
    return out


def compact_working_state(
    state: dict[str, object] | None,
    *,
    fast_decay: float = 0.70,
    momentum_decay: float = 0.50,
    model: MMARRSTitan | None = None,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Compactación *Zero-Drift* de buffers rápidos preservando memoria cristalizada y HRR.

    Atenúa suavemente `titans_state["fast"]` y `titans_state["fast_mom"]` y limpia tensores
    efímeros de la sesión anterior, sin modificar ni un bit de los bancos cristalizados
    Benna-Fusi (`medium`, `slow`) ni las ranuras/trazas episódicas de `HRRModernHopfieldMemory`.
    """
    if not (0.0 < fast_decay < 1.0):
        raise ValueError("fast_decay debe pertenecer a (0, 1)")
    if not (0.0 <= momentum_decay < 1.0):
        raise ValueError("momentum_decay debe pertenecer a [0, 1)")

    hrr_slots = 0
    if model is not None and getattr(model, "hrr_hopfield", None) is not None:
        hrr_slots = int(model.hrr_hopfield.slot_filled.detach().sum().item())
    elif state is not None:
        hrr_slots = int(state.get("hrr_slots_used", 0) or 0)

    if state is None:
        return {}, dict(
            compacted=False,
            fast_norm_before=0.0,
            fast_norm_after=0.0,
            fast_mom_norm_before=0.0,
            fast_mom_norm_after=0.0,
            medium_norm_before=0.0,
            medium_norm_after=0.0,
            slow_norm_before=0.0,
            slow_norm_after=0.0,
            hrr_slots_preserved=hrr_slots,
            fast_decay=fast_decay,
            momentum_decay=momentum_decay,
            dropped_transient_keys=[],
        )

    new_state: dict[str, object] = {}
    dropped_keys: list[str] = []
    for key, val in state.items():
        if key in _TRANSIENT_STATE_KEYS:
            dropped_keys.append(key)
        elif key == "titans_state":
            continue
        elif isinstance(val, torch.Tensor):
            new_state[key] = val.detach().clone()
        else:
            new_state[key] = copy.deepcopy(val)

    norms = [0.0] * 8  # fast_b, fast_a, mom_b, mom_a, med_b, med_a, slow_b, slow_a
    compacted_flag = False

    titans_in = state.get("titans_state")
    if isinstance(titans_in, dict):
        new_titans: dict[str, torch.Tensor] = {}
        scalar_slots: list[tuple[int, torch.Tensor]] = []

        def _bank_norm(t: torch.Tensor) -> torch.Tensor:
            return t.detach().float().norm(dim=-1).mean() if t.numel() > 0 else t.new_zeros(())

        fast_t = titans_in.get("fast")
        if isinstance(fast_t, torch.Tensor):
            fast_damped = fast_t.detach() * float(fast_decay)
            new_titans["fast"] = fast_damped
            scalar_slots.append((0, _bank_norm(fast_t)))
            scalar_slots.append((1, _bank_norm(fast_damped)))
            compacted_flag = True

        mom_t = titans_in.get("fast_mom")
        if isinstance(mom_t, torch.Tensor):
            mom_damped = mom_t.detach() * float(momentum_decay)
            new_titans["fast_mom"] = mom_damped
            scalar_slots.append((2, _bank_norm(mom_t)))
            scalar_slots.append((3, _bank_norm(mom_damped)))
            compacted_flag = True

        # Bancos cristalizados Benna-Fusi: preservación exacta bit-a-bit
        for slot_base, bank_name in ((4, "medium"), (6, "slow")):
            bank_t = titans_in.get(bank_name)
            if isinstance(bank_t, torch.Tensor):
                preserved = bank_t.detach().clone()
                new_titans[bank_name] = preserved
                b_norm = _bank_norm(preserved)
                scalar_slots.append((slot_base, b_norm))
                scalar_slots.append((slot_base + 1, b_norm))

        # Conservar cualquier banco auxiliar adicional sin mutarlo
        for extra_k, extra_v in titans_in.items():
            if extra_k not in new_titans and isinstance(extra_v, torch.Tensor):
                new_titans[extra_k] = extra_v.detach().clone()

        new_state["titans_state"] = new_titans
        # Una sola transferencia D2H agrupada por dispositivo para toda la telemetría
        if scalar_slots:
            vals = _batch_scalar_tensors_to_floats([t for _, t in scalar_slots])
            for (slot_idx, _), val in zip(scalar_slots, vals, strict=True):
                norms[slot_idx] = val

    metrics: dict[str, Any] = dict(
        compacted=compacted_flag or bool(dropped_keys),
        fast_norm_before=norms[0],
        fast_norm_after=norms[1],
        fast_mom_norm_before=norms[2],
        fast_mom_norm_after=norms[3],
        medium_norm_before=norms[4],
        medium_norm_after=norms[5],
        slow_norm_before=norms[6],
        slow_norm_after=norms[7],
        hrr_slots_preserved=hrr_slots,
        fast_decay=float(fast_decay),
        momentum_decay=float(momentum_decay),
        dropped_transient_keys=sorted(dropped_keys),
    )
    return new_state, metrics


class SessionHealthGate:
    """Puerta de verificación automática E2E tras cada sesión / checkpoint de `MMARRS-TITAN`."""

    def __init__(
        self,
        *,
        max_ood_energy: float = 4.0,
        max_load_imbalance: float = 2.5,
        cusum_alert_threshold: float = 0.25,
        max_memory_norm: float = 50.0,
        cusum_drift: float = 0.01,
        ewma_halflife: float = 21.0,
        quantile_eps: float = 0.0,
        auto_recover_on_nan: bool = True,
        ood_hard_multiplier: float = 2.5,
        imbalance_hard_multiplier: float = 4.0,
    ) -> None:
        self.max_ood_energy = float(max_ood_energy)
        self.max_load_imbalance = float(max_load_imbalance)
        self.cusum_alert_threshold = float(cusum_alert_threshold)
        self.max_memory_norm = float(max_memory_norm)
        self.cusum_drift = float(cusum_drift)
        self.ewma_decay = math.pow(0.5, 1.0 / max(float(ewma_halflife), 1.0))
        self.quantile_eps = float(quantile_eps)
        self.auto_recover_on_nan = bool(auto_recover_on_nan)
        self.ood_hard_multiplier = float(ood_hard_multiplier)
        self.imbalance_hard_multiplier = float(imbalance_hard_multiplier)

        self.ewma_ic: float = 0.0
        self.peak_ic: float = 0.0
        self.cusum_stat: float = 0.0
        self.n_ic_updates: int = 0

    @classmethod
    def from_config(cls, cfg: HarnessConfig) -> SessionHealthGate:
        """Construye un `SessionHealthGate` alineado con `HarnessConfig`."""
        return cls(
            max_ood_energy=cfg.max_ood_energy,
            max_load_imbalance=cfg.max_load_imbalance,
            cusum_alert_threshold=cfg.cusum_alert_threshold,
            max_memory_norm=cfg.max_memory_norm,
            auto_recover_on_nan=cfg.auto_recover_on_nan,
        )

    def state_dict(self) -> dict[str, Any]:
        """Exporta el estado del detector CUSUM para checkpoints reanudables."""
        return dict(
            ewma_ic=self.ewma_ic,
            peak_ic=self.peak_ic,
            cusum_stat=self.cusum_stat,
            n_ic_updates=self.n_ic_updates,
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restaura el estado del detector CUSUM desde un checkpoint."""
        self.ewma_ic = float(state.get("ewma_ic", 0.0))
        self.peak_ic = float(state.get("peak_ic", 0.0))
        self.cusum_stat = float(state.get("cusum_stat", 0.0))
        self.n_ic_updates = int(state.get("n_ic_updates", 0))

    @staticmethod
    def _all_tensors_finite(tensors: Sequence[torch.Tensor]) -> bool:
        """Verifica finitud de todos los tensores agrupando en 1 sola sincronización."""
        if not tensors:
            return True
        by_dev: dict[torch.device, list[torch.Tensor]] = {}
        for t in tensors:
            if t.numel() > 0 and (t.dtype.is_floating_point or t.dtype.is_complex):
                by_dev.setdefault(t.device, []).append(torch.isfinite(t.detach()).all())
        if not by_dev:
            return True
        for flags in by_dev.values():
            if not bool(torch.stack(flags).all().item()):
                return False
        return True

    @staticmethod
    def _pearson_ic_tensor(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.detach().float().reshape(-1)
        y = target.detach().float().reshape(-1)
        if p.numel() < 3 or p.numel() != y.numel():
            return torch.tensor(float("nan"), device=p.device, dtype=torch.float32)
        p_c = p - p.mean()
        y_c = y - y.mean()
        denom = p_c.norm() * y_c.norm()
        return torch.where(
            denom > 1e-12,
            (p_c * y_c).sum() / denom.clamp_min(1e-12),
            torch.zeros((), device=p.device, dtype=torch.float32),
        )

    @torch.no_grad()
    def verify_session(
        self,
        *,
        model: nn.Module | None = None,
        outputs: dict[str, Any] | None = None,
        state: dict[str, object] | None = None,
        post_matured_state: dict[str, object] | None = None,
        targets: torch.Tensor | None = None,
        loss: torch.Tensor | float | None = None,
        session_ic: float | None = None,
        prior_nonfinite_steps: torch.Tensor | int | float | None = None,
        rollback_fn: Callable[[], Any] | None = None,
    ) -> HealthReport:
        """Verifica las 6 invariantes críticas (+ post-matured opcional) en 1 sola sync D2H."""
        warnings: list[str] = []
        hard_failures: list[str] = []

        # Agrupar TODAS las finitudes y escalares 0-D en UNA sola sincronización D2H
        scalar_slots: dict[str, int] = {}
        scalar_tensors: list[torch.Tensor] = []

        def _enqueue_scalar(name: str, val: torch.Tensor) -> None:
            scalar_slots[name] = len(scalar_tensors)
            scalar_tensors.append(val.detach().float().reshape(()))

        def _enqueue_finite_group(prefix: str, tensors: Sequence[torch.Tensor]) -> None:
            by_dev: dict[torch.device, list[torch.Tensor]] = {}
            for t in tensors:
                if t.numel() > 0 and (t.dtype.is_floating_point or t.dtype.is_complex):
                    by_dev.setdefault(t.device, []).append(torch.isfinite(t.detach()).all())
            for d_i, flags in enumerate(by_dev.values()):
                _enqueue_scalar(f"{prefix}_{d_i}", torch.stack(flags).all().float())

        # (a) Finitud de pesos del modelo y buffers flotantes
        if model is not None:
            model_tensors: list[torch.Tensor] = [p for p in model.parameters()]
            model_tensors.extend(
                b for b in model.buffers() if b.dtype.is_floating_point or b.dtype.is_complex
            )
            _enqueue_finite_group("w_fin", model_tensors)

        # (a') Finitud de predicciones y pérdida
        pred_tensors: list[torch.Tensor] = []
        if outputs is not None:
            for key in (
                "pred_return",
                "alpha",
                "factor_return",
                "quantiles",
                "tb_probs",
                "turning_prob",
                "vol_pred",
                "epistemic_error",
            ):
                val = outputs.get(key)
                if isinstance(val, torch.Tensor):
                    pred_tensors.append(val)
            energy_d = outputs.get("energy")
            if isinstance(energy_d, dict) and isinstance(energy_d.get("total"), torch.Tensor):
                pred_tensors.append(energy_d["total"])

        loss_val: float | None = None
        if isinstance(loss, torch.Tensor):
            pred_tensors.append(loss)
            _enqueue_scalar("loss", loss)
        elif loss is not None:
            loss_val = float(loss)
            if not math.isfinite(loss_val):
                hard_failures.append("non_finite_loss")
        _enqueue_finite_group("p_fin", pred_tensors)

        # (a'') Finitud de estado causal y (c) Cota PT1 de Lyapunov en bancos de memoria
        state_tensors: list[torch.Tensor] = []
        pt1_norm_tensors: list[torch.Tensor] = []
        if state is not None:
            titans_st = state.get("titans_state")
            if isinstance(titans_st, dict):
                for bank_t in titans_st.values():
                    if isinstance(bank_t, torch.Tensor):
                        state_tensors.append(bank_t)
                        if bank_t.numel() > 0 and bank_t.is_floating_point():
                            pt1_norm_tensors.append(bank_t.detach().float().norm(dim=-1).max())
            for k_st in ("agent_w_prev", "last_x0"):
                v_st = state.get(k_st)
                if isinstance(v_st, torch.Tensor):
                    state_tensors.append(v_st)
        _enqueue_finite_group("s_fin", state_tensors)

        if pt1_norm_tensors:
            by_dev_pt1: dict[torch.device, list[torch.Tensor]] = {}
            for pt_t in pt1_norm_tensors:
                by_dev_pt1.setdefault(pt_t.device, []).append(pt_t.detach().float().reshape(()))
            for d_i, grp in enumerate(by_dev_pt1.values()):
                _enqueue_scalar(f"pt1_max_{d_i}", torch.stack(grp).max())

        # (b) No-cruce estricto de los 9 cuantiles
        n_quantiles = 0
        if outputs is not None and isinstance(outputs.get("quantiles"), torch.Tensor):
            q = outputs["quantiles"].detach().float()
            n_quantiles = int(q.shape[-1])
            if n_quantiles != 9:
                warnings.append(f"unexpected_quantile_count(n={n_quantiles}, expected=9)")
            if n_quantiles >= 2:
                diffs = q[..., 1:] - q[..., :-1]
                _enqueue_scalar("q_cross", (diffs < -self.quantile_eps).sum().float())
                _enqueue_scalar("q_min_gap", diffs.min())
                _enqueue_scalar("q_max_viol", torch.relu(-diffs).max())

        # (d) Ratio de balanceo MoE max/min y (e) Pico OOD de Hopfield (0D tensor o float)
        diag = outputs.get("diagnostics", {}) if isinstance(outputs, dict) else {}
        if not isinstance(diag, dict):
            diag = {}

        raw_load = diag.get("load_max_min_ratio", 1.0)
        load_ratio = 1.0
        if isinstance(raw_load, torch.Tensor):
            _enqueue_scalar("load_ratio", raw_load)
        else:
            load_ratio = float(raw_load)

        raw_ood = diag.get("ood_energy_mean", 0.0)
        ood_energy = 0.0
        if isinstance(raw_ood, torch.Tensor):
            _enqueue_scalar("ood_energy", raw_ood)
        else:
            ood_energy = float(raw_ood)

        # (f) Monitor CUSUM de decaimiento persistente de IC por sesión
        ic_value = session_ic
        if (
            ic_value is None
            and outputs is not None
            and targets is not None
            and isinstance(outputs.get("pred_return"), torch.Tensor)
        ):
            p1 = outputs["pred_return"][:, 0]
            y1 = targets[:, 0] if targets.ndim == 2 else targets
            _enqueue_scalar("ic", self._pearson_ic_tensor(p1, y1.to(device=p1.device)))

        # (g) Estado post-`step_matured` opcional (consolidado en la MISMA única sync D2H!)
        if post_matured_state is not None:
            pm_state_tensors: list[torch.Tensor] = []
            pm_pt1_tensors: list[torch.Tensor] = []
            pm_titans = post_matured_state.get("titans_state")
            if isinstance(pm_titans, dict):
                for bank_t in pm_titans.values():
                    if isinstance(bank_t, torch.Tensor):
                        pm_state_tensors.append(bank_t)
                        if bank_t.numel() > 0 and bank_t.is_floating_point():
                            pm_pt1_tensors.append(bank_t.detach().float().norm(dim=-1).max())
            for k_st in ("agent_w_prev", "last_x0"):
                v_st = post_matured_state.get(k_st)
                if isinstance(v_st, torch.Tensor):
                    pm_state_tensors.append(v_st)
            _enqueue_finite_group("pm_s_fin", pm_state_tensors)
            if pm_pt1_tensors:
                by_dev_pm: dict[torch.device, list[torch.Tensor]] = {}
                for pt_t in pm_pt1_tensors:
                    by_dev_pm.setdefault(pt_t.device, []).append(pt_t.detach().float().reshape(()))
                for d_i, grp in enumerate(by_dev_pm.values()):
                    _enqueue_scalar(f"pm_pt1_max_{d_i}", torch.stack(grp).max())

        # (h) Contador sticky en dispositivo de pasos no finitos saltados desde último check
        prior_nf_val = 0.0
        if isinstance(prior_nonfinite_steps, torch.Tensor):
            _enqueue_scalar("prior_nf", prior_nonfinite_steps)
        elif prior_nonfinite_steps is not None:
            prior_nf_val = float(prior_nonfinite_steps)

        # UNA sola sincronización D2H agrupada para TODAS las comprobaciones del HealthGate
        pulled = _batch_scalar_tensors_to_floats(scalar_tensors) if scalar_tensors else []

        def _group_all_true(prefix: str) -> bool:
            vals = [pulled[idx] for k, idx in scalar_slots.items() if k.startswith(f"{prefix}_")]
            return all(v >= 0.5 for v in vals)

        weights_finite = _group_all_true("w_fin")
        if not weights_finite:
            hard_failures.append("non_finite_weights")

        predictions_finite = _group_all_true("p_fin")
        if not predictions_finite:
            hard_failures.append("non_finite_predictions")

        state_finite = _group_all_true("s_fin")
        if not state_finite:
            hard_failures.append("non_finite_state")

        if "loss" in scalar_slots:
            loss_val = pulled[scalar_slots["loss"]]
        pt1_vals = [pulled[idx] for k, idx in scalar_slots.items() if k.startswith("pt1_max_")]
        pt1_memory_norm_max = max(pt1_vals) if pt1_vals else 0.0
        if not math.isfinite(pt1_memory_norm_max) or pt1_memory_norm_max > self.max_memory_norm:
            hard_failures.append(
                f"pt1_memory_norm_exceeded(norm={pt1_memory_norm_max:.4f} > "
                f"{self.max_memory_norm:.4f})"
            )
        elif pt1_memory_norm_max > 0.85 * self.max_memory_norm:
            warnings.append(
                f"pt1_memory_norm_elevated(norm={pt1_memory_norm_max:.4f} / "
                f"{self.max_memory_norm:.4f})"
            )

        quantile_crossings = 0
        quantile_min_gap = float("inf")
        if "q_cross" in scalar_slots:
            quantile_crossings = int(round(pulled[scalar_slots["q_cross"]]))
            quantile_min_gap = float(pulled[scalar_slots["q_min_gap"]])
            max_viol = float(pulled[scalar_slots["q_max_viol"]])
            if quantile_crossings > 0:
                hard_failures.append(
                    f"crossing_quantiles(count={quantile_crossings}, max_viol={max_viol:.6g})"
                )

        if "load_ratio" in scalar_slots:
            load_ratio = float(pulled[scalar_slots["load_ratio"]])
        load_hard_limit = self.max_load_imbalance * self.imbalance_hard_multiplier
        if not math.isfinite(load_ratio) or load_ratio > load_hard_limit:
            hard_failures.append(
                f"extreme_moe_load_imbalance(ratio={load_ratio:.3f} > {load_hard_limit:.3f})"
            )
        elif load_ratio > self.max_load_imbalance:
            warnings.append(
                f"moe_load_imbalance(ratio={load_ratio:.3f} > {self.max_load_imbalance:.3f})"
            )

        if "ood_energy" in scalar_slots:
            ood_energy = float(pulled[scalar_slots["ood_energy"]])
        ood_hard_limit = self.max_ood_energy * self.ood_hard_multiplier
        if not math.isfinite(ood_energy) or ood_energy > ood_hard_limit:
            hard_failures.append(
                f"extreme_ood_energy_spike(ood={ood_energy:.3f} > {ood_hard_limit:.3f})"
            )
        elif ood_energy > self.max_ood_energy:
            warnings.append(
                f"ood_energy_spike(ood={ood_energy:.3f} > {self.max_ood_energy:.3f})"
            )

        if "ic" in scalar_slots and predictions_finite:
            cand_ic = float(pulled[scalar_slots["ic"]])
            if math.isfinite(cand_ic):
                ic_value = cand_ic

        if ic_value is not None and math.isfinite(ic_value):
            self.n_ic_updates += 1
            if self.n_ic_updates == 1:
                self.ewma_ic = float(ic_value)
                self.peak_ic = max(float(ic_value), 0.0)
            else:
                self.ewma_ic = (
                    self.ewma_decay * self.ewma_ic + (1.0 - self.ewma_decay) * float(ic_value)
                )
                self.peak_ic = max(self.peak_ic * 0.995, self.ewma_ic)
            drop = (self.peak_ic - float(ic_value)) - self.cusum_drift
            self.cusum_stat = max(0.0, self.cusum_stat + drop)

        effective_cusum = self.cusum_stat
        gov = getattr(model, "agent_governor", None) if model is not None else None
        if gov is not None and hasattr(gov, "cusum_stat") and len(gov.cusum_stat) > 0:
            effective_cusum = max(effective_cusum, float(np.max(gov.cusum_stat)))

        cusum_alert = effective_cusum >= self.cusum_alert_threshold
        if cusum_alert:
            warnings.append(
                f"cusum_ic_decay_alert(stat={effective_cusum:.4f} >= "
                f"{self.cusum_alert_threshold:.4f})"
            )

        if "prior_nf" in scalar_slots:
            prior_nf_val = float(pulled[scalar_slots["prior_nf"]])
        prior_nf_count = int(round(prior_nf_val)) if math.isfinite(prior_nf_val) else 1
        if prior_nf_count > 0:
            hard_failures.append(f"non_finite_skipped_steps(count={prior_nf_count})")

        post_matured_state_finite: bool | None = None
        post_matured_pt1_max: float | None = None
        if post_matured_state is not None:
            post_matured_state_finite = _group_all_true("pm_s_fin")
            if not post_matured_state_finite:
                hard_failures.append("non_finite_post_matured_state")
            pm_pt1_vals = [
                pulled[idx] for k, idx in scalar_slots.items() if k.startswith("pm_pt1_max_")
            ]
            post_matured_pt1_max = max(pm_pt1_vals) if pm_pt1_vals else 0.0
            if (
                not math.isfinite(post_matured_pt1_max)
                or post_matured_pt1_max > self.max_memory_norm
            ):
                hard_failures.append(
                    f"pt1_post_matured_memory_norm_exceeded(norm={post_matured_pt1_max:.4f} > "
                    f"{self.max_memory_norm:.4f})"
                )
            elif post_matured_pt1_max > 0.85 * self.max_memory_norm:
                warnings.append(
                    f"pt1_post_matured_memory_norm_elevated(norm={post_matured_pt1_max:.4f} / "
                    f"{self.max_memory_norm:.4f})"
                )

        passed = len(hard_failures) == 0
        rollback_triggered = False
        if not passed and self.auto_recover_on_nan and rollback_fn is not None:
            rollback_fn()
            rollback_triggered = True

        metrics: dict[str, Any] = dict(
            weights_finite=weights_finite,
            predictions_finite=predictions_finite,
            state_finite=state_finite,
            n_quantiles=n_quantiles,
            quantile_crossings=quantile_crossings,
            quantile_min_gap=quantile_min_gap if math.isfinite(quantile_min_gap) else None,
            pt1_memory_norm_max=pt1_memory_norm_max,
            moe_load_max_min_ratio=load_ratio,
            ood_energy_mean=ood_energy,
            session_ic=ic_value,
            ewma_ic=self.ewma_ic,
            cusum_stat=effective_cusum,
            cusum_alert=cusum_alert,
            loss=loss_val,
            non_finite_skipped_steps=prior_nf_count,
        )
        if post_matured_state is not None:
            metrics["post_matured_buffers_finite"] = weights_finite
            metrics["post_matured_state_finite"] = post_matured_state_finite
            metrics["post_matured_pt1_memory_norm_max"] = post_matured_pt1_max
        return HealthReport(
            passed=passed,
            warnings=warnings,
            hard_failures=hard_failures,
            metrics=metrics,
            rollback_triggered=rollback_triggered,
        )

    @torch.no_grad()
    def verify_post_matured_state(
        self,
        *,
        state: dict[str, object] | None,
        model: nn.Module | None = None,
        rollback_fn: Callable[[], Any] | None = None,
    ) -> HealthReport:
        """Verifica finitud y cota PT1 de Lyapunov tras `step_matured` en una sola sync D2H."""
        warnings: list[str] = []
        hard_failures: list[str] = []

        scalar_slots: dict[str, int] = {}
        scalar_tensors: list[torch.Tensor] = []

        def _enqueue_scalar(name: str, val: torch.Tensor) -> None:
            scalar_slots[name] = len(scalar_tensors)
            scalar_tensors.append(val.detach().float().reshape(()))

        def _enqueue_finite_group(prefix: str, tensors: Sequence[torch.Tensor]) -> None:
            by_dev: dict[torch.device, list[torch.Tensor]] = {}
            for t in tensors:
                if t.numel() > 0 and (t.dtype.is_floating_point or t.dtype.is_complex):
                    by_dev.setdefault(t.device, []).append(torch.isfinite(t.detach()).all())
            for d_i, flags in enumerate(by_dev.values()):
                _enqueue_scalar(f"{prefix}_{d_i}", torch.stack(flags).all().float())

        if model is not None:
            buf_tensors = [
                b for b in model.buffers() if b.dtype.is_floating_point or b.dtype.is_complex
            ]
            _enqueue_finite_group("b_fin", buf_tensors)

        state_tensors: list[torch.Tensor] = []
        pt1_norm_tensors: list[torch.Tensor] = []
        if state is not None:
            titans_st = state.get("titans_state")
            if isinstance(titans_st, dict):
                for bank_t in titans_st.values():
                    if isinstance(bank_t, torch.Tensor):
                        state_tensors.append(bank_t)
                        if bank_t.numel() > 0 and bank_t.is_floating_point():
                            pt1_norm_tensors.append(bank_t.detach().float().norm(dim=-1).max())
            for k_st in ("agent_w_prev", "last_x0"):
                v_st = state.get(k_st)
                if isinstance(v_st, torch.Tensor):
                    state_tensors.append(v_st)
        _enqueue_finite_group("s_fin", state_tensors)

        if pt1_norm_tensors:
            by_dev_pt1: dict[torch.device, list[torch.Tensor]] = {}
            for pt_t in pt1_norm_tensors:
                by_dev_pt1.setdefault(pt_t.device, []).append(pt_t.detach().float().reshape(()))
            for d_i, grp in enumerate(by_dev_pt1.values()):
                _enqueue_scalar(f"pt1_max_{d_i}", torch.stack(grp).max())

        pulled = _batch_scalar_tensors_to_floats(scalar_tensors) if scalar_tensors else []
        buffers_finite = all(
            pulled[idx] >= 0.5 for k, idx in scalar_slots.items() if k.startswith("b_fin_")
        )
        if not buffers_finite:
            hard_failures.append("non_finite_post_matured_buffers")

        state_finite = all(
            pulled[idx] >= 0.5 for k, idx in scalar_slots.items() if k.startswith("s_fin_")
        )
        if not state_finite:
            hard_failures.append("non_finite_post_matured_state")

        pt1_vals = [pulled[idx] for k, idx in scalar_slots.items() if k.startswith("pt1_max_")]
        pt1_memory_norm_max = max(pt1_vals) if pt1_vals else 0.0
        if not math.isfinite(pt1_memory_norm_max) or pt1_memory_norm_max > self.max_memory_norm:
            hard_failures.append(
                f"pt1_post_matured_memory_norm_exceeded(norm={pt1_memory_norm_max:.4f} > "
                f"{self.max_memory_norm:.4f})"
            )
        elif pt1_memory_norm_max > 0.85 * self.max_memory_norm:
            warnings.append(
                f"pt1_post_matured_memory_norm_elevated(norm={pt1_memory_norm_max:.4f} / "
                f"{self.max_memory_norm:.4f})"
            )

        passed = len(hard_failures) == 0
        rollback_triggered = False
        if not passed and self.auto_recover_on_nan and rollback_fn is not None:
            rollback_fn()
            rollback_triggered = True

        return HealthReport(
            passed=passed,
            warnings=warnings,
            hard_failures=hard_failures,
            metrics=dict(
                post_matured_buffers_finite=buffers_finite,
                post_matured_state_finite=state_finite,
                post_matured_pt1_memory_norm_max=pt1_memory_norm_max,
            ),
            rollback_triggered=rollback_triggered,
        )


def _extract_six_dim_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Extrae las 8 métricas clave de 6 dimensiones (informe anidado o plano)."""
    src = report.get("granular_report", report)
    if not isinstance(src, Mapping):
        raise ValueError("El informe granular debe ser un mapeo válido")

    dim2 = src.get("dim2_ranking", src)
    dim4 = src.get("dim4_uncertainty", src)
    dim6 = src.get("dim6_economics", src)

    def _pick(mapping: Mapping[str, Any], *keys: str) -> float:
        for k in keys:
            if k in mapping and mapping[k] is not None:
                return float(mapping[k])
        raise KeyError(f"Métrica requerida ausente en informe granular; claves buscadas: {keys}")

    return dict(
        rank_ic=_pick(dim2, "rank_ic_mean", "rank_ic"),
        pure_alpha_ic=_pick(dim2, "pure_alpha_ic_mean", "pure_alpha_ic"),
        rank_ic_t_nw=_pick(dim2, "rank_ic_t_nw", "newey_west_t"),
        crps=_pick(dim4, "crps"),
        cov_80=_pick(dim4, "cov_80", "cov80"),
        net_sharpe_1x=_pick(dim6, "net_sharpe_1x"),
        net_sharpe_2x=_pick(dim6, "net_sharpe_2x"),
        dsr=_pick(dim6, "dsr"),
    )


def _extract_slice_rank_ic_and_sharpe(
    entry: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    """Extrae `(rank_ic_mean, net_sharpe_1x)` de informes planos o anidados de 6 dimensiones."""
    raw_sub = entry.get("granular_report", entry)
    sub = raw_sub if isinstance(raw_sub, Mapping) else entry
    raw_d2 = sub.get("dim2_ranking", sub)
    d2 = raw_d2 if isinstance(raw_d2, Mapping) else sub
    raw_d6 = sub.get("dim6_economics", sub)
    d6 = raw_d6 if isinstance(raw_d6, Mapping) else sub

    ic_raw: Any = None
    for mapping in (d2, sub):
        for key in ("rank_ic_mean", "rank_ic"):
            if key in mapping and mapping[key] is not None:
                ic_raw = mapping[key]
                break
        if ic_raw is not None:
            break

    sr_raw: Any = None
    for mapping in (d6, sub):
        if "net_sharpe_1x" in mapping and mapping["net_sharpe_1x"] is not None:
            sr_raw = mapping["net_sharpe_1x"]
            break

    return (
        None if ic_raw is None else float(ic_raw),
        None if sr_raw is None else float(sr_raw),
    )


@dataclass(frozen=True)
class GainVerdict:
    """Veredicto auditable `PASS`/`REJECT` de ganancia OOS frente a controles emparejados."""

    variation: str
    verdict: str
    promoted: bool
    candidate_metrics: dict[str, float]
    deltas_vs_controls: dict[str, dict[str, float]]
    checks: dict[str, bool]
    failed_criteria: tuple[str, ...]
    controls_evaluated: tuple[str, ...]
    survivorship_disclosure: bool = True
    champion_mode: bool = False
    survivorship_note: str = (
        "Cohort may carry survivorship conditioning; co-gate on liquid top-third and "
        "2008/2020/2022 crisis regimes."
    )

    def to_dict(self) -> dict[str, Any]:
        """Serializa el veredicto de ganancia en formato JSON para manifiesto y ledger."""
        return dict(
            variation=self.variation,
            verdict=self.verdict,
            promoted=self.promoted,
            candidate_metrics=dict(self.candidate_metrics),
            deltas_vs_controls={k: dict(v) for k, v in self.deltas_vs_controls.items()},
            checks=dict(self.checks),
            failed_criteria=list(self.failed_criteria),
            controls_evaluated=list(self.controls_evaluated),
            survivorship_disclosure=self.survivorship_disclosure,
            champion_mode=self.champion_mode,
            survivorship_note=self.survivorship_note,
        )


class GainValidationGate:
    """Puerta de promoción OOS que valida cada variación frente a sus controles emparejados.

    Exige superar controles emparejados (`dense_iso_flop`, `dense_iso_param`, `uniform_1_over_e`,
    `shuffled_placebo`, `k1_fixed_depth`) en las 6 dimensiones granulares (Rank IC, Pure Alpha IC,
    CRPS, Cov80, Net Sharpe 1x/2x, DSR >= 0.90 [0.95 campeón], Newey-West t-stat, PBO <= 0.50,
    margen de seguridad >= 2.5x, tercio líquido y pisos en crisis 2008/2020/2022).
    """

    def __init__(
        self,
        *,
        required_controls: Sequence[str] = tuple(DEFAULT_MATCHED_CONTROLS.keys()),
        min_rank_ic_gain: float = 1e-4,
        min_pure_alpha_ic_gain: float = 1e-4,
        max_crps_regression: float = 0.0,
        cov80_target: float = 0.80,
        max_cov80_error: float = 0.08,
        min_net_sharpe_1x_gain: float = 0.05,
        min_net_sharpe_2x_gain: float = 0.0,
        min_dsr: float = 0.90,
        champion_min_dsr: float = 0.95,
        champion_mode: bool = False,
        min_newey_west_t: float = 1.96,
        max_pbo: float = 0.50,
        min_safety_margin: float = 2.5,
        crisis_years: Sequence[int] = (2008, 2020, 2022),
        min_crisis_rank_ic: float = 0.0,
        min_crisis_net_sharpe_1x: float = -0.50,
        min_liquid_third_rank_ic: float = 0.0,
        require_positive_net_sharpe_2x: bool = True,
        survivorship_disclosure: bool = True,
    ) -> None:
        self.required_controls = tuple(required_controls)
        self.min_rank_ic_gain = float(min_rank_ic_gain)
        self.min_pure_alpha_ic_gain = float(min_pure_alpha_ic_gain)
        self.max_crps_regression = float(max_crps_regression)
        self.cov80_target = float(cov80_target)
        self.max_cov80_error = float(max_cov80_error)
        self.min_net_sharpe_1x_gain = float(min_net_sharpe_1x_gain)
        self.min_net_sharpe_2x_gain = float(min_net_sharpe_2x_gain)
        self.min_dsr = float(min_dsr)
        self.champion_min_dsr = float(champion_min_dsr)
        self.champion_mode = bool(champion_mode)
        self.min_newey_west_t = float(min_newey_west_t)
        self.max_pbo = float(max_pbo)
        self.min_safety_margin = float(min_safety_margin)
        self.crisis_years = tuple(int(y) for y in crisis_years)
        self.min_crisis_rank_ic = float(min_crisis_rank_ic)
        self.min_crisis_net_sharpe_1x = float(min_crisis_net_sharpe_1x)
        self.min_liquid_third_rank_ic = float(min_liquid_third_rank_ic)
        self.require_positive_net_sharpe_2x = bool(require_positive_net_sharpe_2x)
        self.survivorship_disclosure = bool(survivorship_disclosure)

    @classmethod
    def for_champion(cls, **kwargs: Any) -> GainValidationGate:
        """Atajo para puerta de campeón final (`champion_mode=True`, `min_dsr >= 0.95`)."""
        kwargs["champion_mode"] = True
        return cls(**kwargs)

    def evaluate(
        self,
        variation: str,
        candidate_report: Mapping[str, Any],
        control_reports: Mapping[str, Mapping[str, Any]],
    ) -> GainVerdict:
        """Emite `GainVerdict(PASS|REJECT)` auditando al candidato frente a todos los controles."""
        cand = _extract_six_dim_metrics(candidate_report)
        checks: dict[str, bool] = {}
        failed: list[str] = []
        deltas: dict[str, dict[str, float]] = {}

        cand_finite = all(math.isfinite(v) for v in cand.values())
        checks["candidate_metrics_finite"] = cand_finite
        if not cand_finite:
            failed.append("non_finite_candidate_metrics")

        if not control_reports:
            checks["matched_controls_present"] = False
            failed.append("missing_matched_controls")
            return GainVerdict(
                variation=variation,
                verdict="REJECT",
                promoted=False,
                candidate_metrics=cand,
                deltas_vs_controls={},
                checks=checks,
                failed_criteria=tuple(failed),
                controls_evaluated=(),
                survivorship_disclosure=self.survivorship_disclosure,
                champion_mode=self.champion_mode,
            )

        checks["matched_controls_present"] = True
        missing_req = [c for c in self.required_controls if c not in control_reports]
        checks["required_controls_present"] = len(missing_req) == 0
        if missing_req:
            failed.append(f"missing_required_controls({','.join(missing_req)})")

        # Puertas absolutas de calibración, significación estadística y anti-sobreajuste (P0-5)
        t_nw_ok = cand["rank_ic_t_nw"] >= self.min_newey_west_t
        checks["rank_ic_t_nw"] = t_nw_ok
        if not t_nw_ok:
            failed.append(
                f"rank_ic_t_nw({cand['rank_ic_t_nw']:.3f} < {self.min_newey_west_t:.3f})"
            )

        eff_min_dsr = (
            max(self.min_dsr, self.champion_min_dsr) if self.champion_mode else self.min_dsr
        )
        dsr_ok = cand["dsr"] >= eff_min_dsr
        checks["dsr"] = dsr_ok
        if not dsr_ok:
            failed.append(f"dsr({cand['dsr']:.3f} < {eff_min_dsr:.3f})")

        cov_err = abs(cand["cov_80"] - self.cov80_target)
        cov_ok = cov_err <= self.max_cov80_error
        checks["cov_80"] = cov_ok
        if not cov_ok:
            failed.append(
                f"cov_80_calibration(cov_80={cand['cov_80']:.3f}, err={cov_err:.3f} > "
                f"{self.max_cov80_error:.3f})"
            )

        if self.require_positive_net_sharpe_2x:
            sr2_pos = cand["net_sharpe_2x"] > 0.0
            checks["net_sharpe_2x_positive"] = sr2_pos
            if not sr2_pos:
                failed.append(f"net_sharpe_2x_non_positive({cand['net_sharpe_2x']:.3f})")

        # Puertas P0-5 opcionales cuando las métricas están presentes: PBO, safety_margin,
        # tercio líquido y pisos en regímenes de crisis (2008 / 2020 / 2022)
        src = candidate_report.get("granular_report", candidate_report)
        dim6 = src.get("dim6_economics", src) if isinstance(src, Mapping) else {}
        pbo_raw = (
            dim6.get("pbo")
            if isinstance(dim6, Mapping) and "pbo" in dim6
            else candidate_report.get("pbo")
        )
        if pbo_raw is not None:
            pbo_val = float(pbo_raw)
            cand["pbo"] = pbo_val
            pbo_ok = math.isfinite(pbo_val) and pbo_val <= self.max_pbo
            checks["pbo"] = pbo_ok
            if not pbo_ok:
                failed.append(f"pbo_exceeded({pbo_val:.3f} > {self.max_pbo:.3f})")

        sm_raw = (
            dim6.get("safety_margin")
            if isinstance(dim6, Mapping) and "safety_margin" in dim6
            else candidate_report.get("safety_margin")
        )
        if sm_raw is None and isinstance(dim6, Mapping):
            g_bps = dim6.get("gross_mean_bps")
            c_bps = dim6.get("cost_mean_bps")
            if c_bps is None and any(
                k in dim6 and dim6[k] is not None
                for k in ("linear_cost_bps", "impact_cost_bps")
            ):
                # `linear_cost_bps` ya incluye `c_stamp` en `granular_evaluator.py:666`
                c_bps = float(dim6.get("linear_cost_bps") or 0.0) + float(
                    dim6.get("impact_cost_bps") or 0.0
                )
            if g_bps is not None and c_bps is not None:
                g_val = float(g_bps)
                c_val = float(c_bps)
                if g_val <= 0.0:
                    sm_raw = 0.0
                elif c_val > 1e-12:
                    sm_raw = g_val / c_val
        if sm_raw is not None:
            sm_val = float(sm_raw)
            cand["safety_margin"] = sm_val
            sm_ok = math.isfinite(sm_val) and sm_val >= self.min_safety_margin
            checks["safety_margin"] = sm_ok
            if not sm_ok:
                failed.append(
                    f"safety_margin_insufficient({sm_val:.3f} < {self.min_safety_margin:.3f})"
                )

        liq_rep = candidate_report.get("liquid_third") or (
            src.get("liquid_third") if isinstance(src, Mapping) else None
        )
        if isinstance(liq_rep, Mapping):
            liq_ic, liq_sr = _extract_slice_rank_ic_and_sharpe(liq_rep)
            if liq_ic is None and liq_sr is None and len(liq_rep) > 0:
                checks["liquid_third_metrics_present"] = False
                failed.append("liquid_third_missing_metrics")
            if liq_ic is not None:
                cand["liquid_third_rank_ic"] = liq_ic
                liq_ok = math.isfinite(liq_ic) and liq_ic >= self.min_liquid_third_rank_ic
                checks["liquid_third_rank_ic"] = liq_ok
                if not liq_ok:
                    failed.append(
                        f"liquid_third_rank_ic({liq_ic:.4f} < {self.min_liquid_third_rank_ic:.4f})"
                    )
            if liq_sr is not None:
                cand["liquid_third_net_sharpe_1x"] = liq_sr
                liq_sr_ok = math.isfinite(liq_sr) and liq_sr > 0.0
                checks["liquid_third_net_sharpe_1x"] = liq_sr_ok
                if not liq_sr_ok:
                    failed.append(f"liquid_third_net_sharpe_1x({liq_sr:.3f} <= 0.0)")

        crisis_map = candidate_report.get("crisis_metrics") or (
            src.get("crisis_metrics") if isinstance(src, Mapping) else None
        )
        if isinstance(crisis_map, Mapping):
            for yr in self.crisis_years:
                c_entry = crisis_map.get(yr, crisis_map.get(str(yr)))
                if not isinstance(c_entry, Mapping):
                    continue
                c_ic, c_sr = _extract_slice_rank_ic_and_sharpe(c_entry)
                if c_ic is None and c_sr is None and len(c_entry) > 0:
                    checks[f"crisis_{yr}_metrics_present"] = False
                    failed.append(f"crisis_{yr}_missing_metrics")
                if c_ic is not None:
                    c_ic_ok = math.isfinite(c_ic) and c_ic >= self.min_crisis_rank_ic
                    checks[f"crisis_{yr}_rank_ic"] = c_ic_ok
                    if not c_ic_ok:
                        failed.append(
                            f"crisis_{yr}_rank_ic({c_ic:.4f} < {self.min_crisis_rank_ic:.4f})"
                        )
                if c_sr is not None:
                    c_sr_ok = math.isfinite(c_sr) and c_sr >= self.min_crisis_net_sharpe_1x
                    checks[f"crisis_{yr}_net_sharpe_1x"] = c_sr_ok
                    if not c_sr_ok:
                        failed.append(
                            f"crisis_{yr}_net_sharpe_1x({c_sr:.3f} < "
                            f"{self.min_crisis_net_sharpe_1x:.3f})"
                        )

        # Puertas relativas frente a cada control emparejado
        for ctrl_name, ctrl_rep in control_reports.items():
            ctrl = _extract_six_dim_metrics(ctrl_rep)
            ctrl_finite = all(math.isfinite(v) for v in ctrl.values())
            checks[f"finite_{ctrl_name}"] = ctrl_finite
            if not ctrl_finite:
                failed.append(f"non_finite_control({ctrl_name})")
                continue

            d_ric = cand["rank_ic"] - ctrl["rank_ic"]
            d_pure = cand["pure_alpha_ic"] - ctrl["pure_alpha_ic"]
            crps_imp = ctrl["crps"] - cand["crps"]  # positivo = mejora (menor CRPS)
            d_sr1 = cand["net_sharpe_1x"] - ctrl["net_sharpe_1x"]
            d_sr2 = cand["net_sharpe_2x"] - ctrl["net_sharpe_2x"]
            d_dsr = cand["dsr"] - ctrl["dsr"]
            d_tnw = cand["rank_ic_t_nw"] - ctrl["rank_ic_t_nw"]
            deltas[ctrl_name] = dict(
                delta_rank_ic=d_ric,
                delta_pure_alpha_ic=d_pure,
                crps_improvement=crps_imp,
                delta_net_sharpe_1x=d_sr1,
                delta_net_sharpe_2x=d_sr2,
                delta_dsr=d_dsr,
                delta_rank_ic_t_nw=d_tnw,
            )

            ric_ok = d_ric >= self.min_rank_ic_gain
            checks[f"rank_ic_vs_{ctrl_name}"] = ric_ok
            if not ric_ok:
                failed.append(
                    f"rank_ic_vs_{ctrl_name}(delta={d_ric:.5f} < {self.min_rank_ic_gain:.5f})"
                )

            pure_ok = d_pure >= self.min_pure_alpha_ic_gain
            checks[f"pure_alpha_ic_vs_{ctrl_name}"] = pure_ok
            if not pure_ok:
                failed.append(
                    f"pure_alpha_ic_vs_{ctrl_name}(delta={d_pure:.5f} < "
                    f"{self.min_pure_alpha_ic_gain:.5f})"
                )

            crps_ok = crps_imp >= -self.max_crps_regression
            checks[f"crps_vs_{ctrl_name}"] = crps_ok
            if not crps_ok:
                failed.append(
                    f"crps_regression_vs_{ctrl_name}(improvement={crps_imp:.6f} < "
                    f"{-self.max_crps_regression:.6f})"
                )

            sr1_ok = d_sr1 >= self.min_net_sharpe_1x_gain
            checks[f"net_sharpe_1x_vs_{ctrl_name}"] = sr1_ok
            if not sr1_ok:
                failed.append(
                    f"net_sharpe_1x_vs_{ctrl_name}(delta={d_sr1:.4f} < "
                    f"{self.min_net_sharpe_1x_gain:.4f})"
                )

            sr2_ok = d_sr2 >= self.min_net_sharpe_2x_gain
            checks[f"net_sharpe_2x_vs_{ctrl_name}"] = sr2_ok
            if not sr2_ok:
                failed.append(
                    f"net_sharpe_2x_vs_{ctrl_name}(delta={d_sr2:.4f} < "
                    f"{self.min_net_sharpe_2x_gain:.4f})"
                )

        passed = len(failed) == 0
        return GainVerdict(
            variation=variation,
            verdict="PASS" if passed else "REJECT",
            promoted=passed,
            candidate_metrics=cand,
            deltas_vs_controls=deltas,
            checks=checks,
            failed_criteria=tuple(failed),
            controls_evaluated=tuple(control_reports.keys()),
            survivorship_disclosure=self.survivorship_disclosure,
            champion_mode=self.champion_mode,
        )

    def evaluate_batch(
        self,
        variation_reports: Mapping[str, Mapping[str, Any]],
        *,
        control_map: Mapping[str, str] | None = None,
    ) -> dict[str, GainVerdict]:
        """Valida un lote de variaciones pre-entrenadas contra los controles disponibles."""
        cmap = dict(DEFAULT_MATCHED_CONTROLS if control_map is None else control_map)
        verdicts: dict[str, GainVerdict] = {}
        for var_name, rep in variation_reports.items():
            matched = {
                role: variation_reports[ctrl_var]
                for role, ctrl_var in cmap.items()
                if ctrl_var in variation_reports and ctrl_var != var_name
            }
            verdicts[var_name] = self.evaluate(var_name, rep, matched)
        return verdicts


def _export_governor_state(model: MMARRSTitan) -> dict[str, Any] | None:
    gov = getattr(model, "agent_governor", None)
    if gov is None:
        return None
    return dict(
        specialist_weights=gov.specialist_weights.copy(),
        ewma_ic=gov.ewma_ic.copy(),
        peak_ic=gov.peak_ic.copy(),
        cusum_stat=gov.cusum_stat.copy(),
        decay_alerts=gov.decay_alerts.copy(),
        history_updates=int(gov.history_updates),
    )


def _restore_governor_state(model: MMARRSTitan, state: dict[str, Any] | None) -> None:
    gov = getattr(model, "agent_governor", None)
    if gov is None or state is None:
        return
    gov.specialist_weights = np.asarray(state["specialist_weights"], dtype=np.float64).copy()
    gov.ewma_ic = np.asarray(state["ewma_ic"], dtype=np.float64).copy()
    gov.peak_ic = np.asarray(state["peak_ic"], dtype=np.float64).copy()
    gov.cusum_stat = np.asarray(state["cusum_stat"], dtype=np.float64).copy()
    gov.decay_alerts = np.asarray(state["decay_alerts"], dtype=bool).copy()
    gov.history_updates = int(state["history_updates"])


class LongRunningTitansHarness:
    """Arnés oficial de larga duración (Inicializador + Trabajador Incremental + Rollback)."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        runtime_cfg: BlackwellRuntimeConfig | None = None,
    ) -> None:
        self.config = config
        self.runtime_cfg = runtime_cfg or BlackwellRuntimeConfig(precision_mode="fp32_strict")
        self.health_gate = SessionHealthGate.from_config(config)

        self._run_dir = Path(config.run_dir)
        self._manifest_path = self._run_dir / "manifest.json"
        ledger_name = (
            "progress_ledger.jsonl.gz" if config.compress_ledger else "progress_ledger.jsonl"
        )
        self._ledger_path = self._run_dir / ledger_name
        self._checkpoints_dir = self._run_dir / "checkpoints"
        self._latest_ckpt_path = self._checkpoints_dir / "latest_verified.pt"

        self._hw_info: dict[str, Any] = {}
        self._device: torch.device = torch.device("cpu")
        self._model: MMARRSTitan | None = None
        self._compiled_model: nn.Module | None = None
        self._ema_model: MMARRSTitan | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._scheduler_total_steps: int | None = None
        self._loss_fn: MMARRSMultiTaskLoss | None = None
        self._drop_gen: torch.Generator | None = None

        self._causal_state: dict[str, object] | None = None
        self._w_prev: torch.Tensor | None = None
        self._w_prev_asset_ids: torch.Tensor | None = None
        self._epoch: int = 0
        self._session_cursor: int = 0
        self._global_step: int = 0
        self._completed_keys: set[str] = set()
        self._manifest: dict[str, Any] = {}
        self._ledger_buffer: list[dict[str, Any]] = []
        self._nonfinite_steps_since_check: torch.Tensor = torch.zeros((), dtype=torch.float32)

    def __enter__(self) -> LongRunningTitansHarness:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def flush_ledger(self, *, sync: bool = False) -> int:
        """Vuelca el búfer JSONL/JSONL.GZ en un único miembro RFC 1952 compacto."""
        if not self._ledger_buffer:
            return 0
        batch = list(self._ledger_buffer)
        self._ledger_buffer.clear()
        _append_jsonl_batch(self._ledger_path, batch, sync=sync)
        return len(batch)

    def _enqueue_ledger_event(
        self,
        record: dict[str, Any],
        *,
        flush: bool = False,
        sync: bool = False,
    ) -> None:
        self._ledger_buffer.append(record)
        if flush or sync or len(self._ledger_buffer) >= self.config.ledger_flush_every_k:
            self.flush_ledger(sync=sync)

    def close(self) -> None:
        """Garantiza el volcado íntegro del ledger comprimido al cerrar el arnés."""
        self.flush_ledger(sync=True)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path

    @property
    def checkpoints_dir(self) -> Path:
        return self._checkpoints_dir

    @property
    def latest_checkpoint_path(self) -> Path:
        return self._latest_ckpt_path

    @property
    def model(self) -> MMARRSTitan:
        if self._model is None:
            raise RuntimeError("El arnés no ha sido inicializado; invoque initialize() primero")
        return self._model

    @property
    def ema_model(self) -> MMARRSTitan:
        if self._ema_model is None:
            raise RuntimeError("El arnés no ha sido inicializado; invoque initialize() primero")
        return self._ema_model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            raise RuntimeError("El arnés no ha sido inicializado; invoque initialize() primero")
        return self._optimizer

    @property
    def causal_state(self) -> dict[str, object] | None:
        return self._causal_state

    @property
    def session_cursor(self) -> int:
        return self._session_cursor

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def _validate_split_config(
        self,
        split_cfg: FullHistorySplitConfig,
        *,
        active_horizons: Sequence[int] | None = None,
        allow_unsafe_purge: bool = False,
    ) -> None:
        seal_yr = self.config.sealed_test_start_year
        if split_cfg.allow_sealed_test:
            raise SealedTestViolationError(
                f"allow_sealed_test=True está prohibido en el arnés de entrenamiento/ajuste "
                f"(sello {seal_yr}+ activo)"
            )
        if split_cfg.train_end_year >= seal_yr or split_cfg.val_year >= seal_yr:
            raise SealedTestViolationError(
                f"train_end_year={split_cfg.train_end_year} y val_year={split_cfg.val_year} "
                f"deben preceder estrictamente al sello {seal_yr}+"
            )
        if split_cfg.sealed_test_start_year > seal_yr:
            raise SealedTestViolationError(
                f"split_cfg.sealed_test_start_year={split_cfg.sealed_test_start_year} relaja "
                f"el sello configurado ({seal_yr}+)"
            )
        if not allow_unsafe_purge:
            min_purge, min_embargo = required_purge_embargo(
                active_horizons if active_horizons is not None else self.config.active_horizons
            )
            if split_cfg.purge_sessions < min_purge:
                raise ValueError(
                    f"Fuga multi-horizonte (P0-4): purge_sessions={split_cfg.purge_sessions} < "
                    f"max(active_horizons)={min_purge}"
                )
            if split_cfg.embargo_sessions < min_embargo:
                raise ValueError(
                    f"Embargo insuficiente (P0-4): embargo_sessions={split_cfg.embargo_sessions} "
                    f"< {min_embargo} para max_h={min_purge}"
                )

    def build_split_config(
        self,
        *,
        train_start_year: int = 2000,
        train_end_year: int = 2022,
        val_year: int = 2023,
        purge_sessions: int | None = None,
        embargo_sessions: int | None = None,
        allow_unsafe_purge: bool = False,
    ) -> FullHistorySplitConfig:
        """Construye un `FullHistorySplitConfig` sellado y seguro frente a fuga multi-horizonte."""
        seal_yr = self.config.sealed_test_start_year
        if train_end_year >= seal_yr or val_year >= seal_yr:
            raise SealedTestViolationError(
                f"No se permite train_end_year={train_end_year} ni val_year={val_year} >= "
                f"sello {seal_yr}+"
            )
        min_purge, min_embargo = required_purge_embargo(self.config.active_horizons)
        s_cfg = FullHistorySplitConfig(
            mode=self.config.split_mode,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            val_year=val_year,
            sealed_test_start_year=seal_yr,
            purge_sessions=min_purge if purge_sessions is None else int(purge_sessions),
            embargo_sessions=min_embargo if embargo_sessions is None else int(embargo_sessions),
            allow_sealed_test=False,
        )
        self._validate_split_config(s_cfg, allow_unsafe_purge=allow_unsafe_purge)
        return s_cfg

    def build_walk_forward_splits(
        self,
        eval_years: Sequence[int],
        *,
        train_start_year: int = 2000,
        purge_sessions: int | None = None,
        embargo_sessions: int | None = None,
        allow_unsafe_purge: bool = False,
    ) -> list[FullHistorySplitConfig]:
        """Genera particiones `expanding_walk_forward` (`2000..T-1 -> T`) sellando `2024+`."""
        seal_yr = self.config.sealed_test_start_year
        min_purge, min_embargo = required_purge_embargo(self.config.active_horizons)
        p_sess = min_purge if purge_sessions is None else int(purge_sessions)
        e_sess = min_embargo if embargo_sessions is None else int(embargo_sessions)
        splits: list[FullHistorySplitConfig] = []
        for yr in eval_years:
            t_yr = int(yr)
            if t_yr >= seal_yr:
                raise SealedTestViolationError(
                    f"Año walk-forward {t_yr} viola el sello estricto {seal_yr}+"
                )
            if t_yr <= train_start_year:
                raise ValueError(
                    f"eval_year={t_yr} debe ser estrictamente posterior a "
                    f"train_start_year={train_start_year}"
                )
            s_cfg = FullHistorySplitConfig(
                mode="expanding_walk_forward",
                train_start_year=train_start_year,
                train_end_year=t_yr - 1,
                val_year=t_yr,
                sealed_test_start_year=seal_yr,
                purge_sessions=p_sess,
                embargo_sessions=e_sess,
                allow_sealed_test=False,
            )
            self._validate_split_config(s_cfg, allow_unsafe_purge=allow_unsafe_purge)
            splits.append(s_cfg)
        return splits

    def prepare_splits(
        self,
        sessions: Sequence[dict[str, torch.Tensor]] | Any,
        session_years: Sequence[int] | None = None,
        *,
        split_cfg: FullHistorySplitConfig | None = None,
    ) -> dict[str, Any]:
        """Aplica partición cronológica sellada `2024+` (zero-copy en `ZeroCopySessionStore`)."""
        s_cfg = split_cfg or self.build_split_config()
        self._validate_split_config(s_cfg)

        if hasattr(sessions, "chronological_splits"):
            store_parts = sessions.chronological_splits(
                train_start_year=s_cfg.train_start_year,
                train_end_year=s_cfg.train_end_year,
                val_year=s_cfg.val_year,
                sealed_test_start_year=s_cfg.sealed_test_start_year,
                active_horizons=self.config.active_horizons,
                purge_sessions=s_cfg.purge_sessions,
                embargo_sessions=s_cfg.embargo_sessions,
                allow_zero_purge=False,
                allow_sealed_test=False,
            )
            train_years = [int(y) for y in store_parts["train"].session_years.tolist()]
            val_years = [int(y) for y in store_parts["val"].session_years.tolist()]
            validate_no_sealed_leakage(
                train_years, sealed_test_start_year=self.config.sealed_test_start_year
            )
            validate_no_sealed_leakage(
                val_years, sealed_test_start_year=self.config.sealed_test_start_year
            )
            return dict(
                train=store_parts["train"],
                train_years=train_years,
                val=store_parts["val"],
                val_years=val_years,
                sealed_test=store_parts["sealed_test"],
                n_sealed_withheld=int(store_parts["n_sealed_withheld"]),
                split_config=s_cfg,
            )

        if session_years is None:
            raise ValueError("session_years es obligatorio para secuencias en memoria")
        parts = split_chronological_sessions(sessions, session_years, s_cfg)

        years = np.asarray(session_years, dtype=np.int64)
        train_idx = np.flatnonzero(
            (years >= s_cfg.train_start_year) & (years <= s_cfg.train_end_year)
        )
        val_idx = np.flatnonzero(
            (years > s_cfg.train_end_year)
            & (years <= s_cfg.val_year)
            & (years < s_cfg.sealed_test_start_year)
        )
        # Purga/embargo estrictos P0-4: jamás saltarse purga si len <= purge/embargo
        if s_cfg.purge_sessions > 0:
            if len(train_idx) > s_cfg.purge_sessions:
                train_idx = train_idx[: -s_cfg.purge_sessions]
            else:
                train_idx = np.empty(0, dtype=np.int64)
                parts["train"] = []
        if s_cfg.embargo_sessions > 0:
            if len(val_idx) > s_cfg.embargo_sessions:
                val_idx = val_idx[s_cfg.embargo_sessions :]
            else:
                val_idx = np.empty(0, dtype=np.int64)
                parts["val"] = []

        train_years = [int(years[int(i)]) for i in train_idx]
        val_years = [int(years[int(i)]) for i in val_idx]
        # Defensa en profundidad: verificar que ningún año sellado alcanzó train o val
        validate_no_sealed_leakage(
            train_years, sealed_test_start_year=self.config.sealed_test_start_year
        )
        validate_no_sealed_leakage(
            val_years, sealed_test_start_year=self.config.sealed_test_start_year
        )
        return dict(
            train=parts["train"],
            train_years=train_years,
            val=parts["val"],
            val_years=val_years,
            sealed_test=parts["sealed_test"],
            n_sealed_withheld=parts["n_sealed_withheld"],
            split_config=s_cfg,
        )

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        t_steps = max(1, int(total_steps))
        warmup_steps = max(1, t_steps // 10)

        def _lr_mult(step_i: int) -> float:
            if step_i < warmup_steps:
                return float(step_i + 1) / float(warmup_steps)
            prog = float(step_i - warmup_steps) / float(max(t_steps - warmup_steps, 1))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

        self._scheduler_total_steps = t_steps
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_mult)

    def _ensure_scheduler(self, total_planned_steps: int | None, fallback_steps: int) -> None:
        if self._scheduler is None:
            steps = total_planned_steps if total_planned_steps is not None else fallback_steps
            self._scheduler = self._build_scheduler(max(1, int(steps)))

    def initialize(
        self,
        *,
        sample_session: dict[str, torch.Tensor] | None = None,
        feature_dims: dict[str, int] | None = None,
        split_cfg: FullHistorySplitConfig | None = None,
        total_planned_steps: int | None = None,
    ) -> dict[str, Any]:
        """Fase Inicializadora: crea `manifest.json` SHA-256 + `progress_ledger.jsonl` o reanuda."""
        s_cfg = split_cfg or self.build_split_config()
        self._validate_split_config(s_cfg)

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._hw_info = configure_blackwell_runtime(self.runtime_cfg, device=self.config.device)
        self._device = torch.device(str(self._hw_info["device"]))

        dims: dict[str, int] = {}
        if feature_dims is not None:
            dims = {k: int(v) for k, v in feature_dims.items()}
        elif sample_session is not None:
            dims = _infer_dims(sample_session)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._model = build_variation(
            self.config.variation,
            hidden_dim=self.config.hidden_dim,
            seed=self.config.seed,
            **dims,
        ).to(self._device)
        self._ema_model = copy.deepcopy(self._model).eval()
        self._compiled_model = maybe_compile_model(self._model, self.runtime_cfg)
        self._loss_fn = MMARRSMultiTaskLoss().to(self._device)
        if self.config.optimizer_name == "adamw_fused":
            self._optimizer = build_optimizer(
                self._model,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                cfg=self.runtime_cfg,
            )
        else:
            from .autotuner_sota import build_sota_optimizer

            self._optimizer = build_sota_optimizer(
                self._model,
                optimizer_type=self.config.optimizer_name,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                router_lr_scale=self.runtime_cfg.router_lr_scale,
                cfg=self.runtime_cfg,
            )
        self._nonfinite_steps_since_check = torch.zeros(
            (), device=self._device, dtype=torch.float32
        )
        self._scheduler = (
            self._build_scheduler(total_planned_steps)
            if total_planned_steps is not None
            else None
        )
        self._drop_gen = torch.Generator(device=self._device)
        self._drop_gen.manual_seed(self.config.seed + 999)

        audit = self._model.parameter_audit()
        resolved_dims = dict(
            prices_dim=self._model.config.prices_dim,
            fundamentals_dim=self._model.config.fundamentals_dim,
            macro_dim=self._model.config.macro_dim,
            news_dim=self._model.config.news_dim,
            charts_dim=self._model.config.charts_dim,
            physics_dim=self._model.config.physics_dim,
            context_dim=self._model.config.context_dim,
            exposures_dim=self._model.config.exposures_dim,
            rmt_dim=self._model.config.rmt_dim,
        )
        feature_contract = dict(
            modalities=["prices", "fundamentals", "macro", "news", "charts"],
            auxiliary_channels=[
                "modality_mask",
                "physics_v4",
                "high_snr_context",
                "exposures",
                "rmt_loadings",
            ],
            dims=resolved_dims,
            horizons=list(self._model.config.horizons),
            quantile_levels=list(self._model.config.quantile_levels),
        )
        seal_status = dict(
            sealed_test_start_year=self.config.sealed_test_start_year,
            allow_sealed_test=False,
            seal_enforced=True,
            policy=f"year >= {self.config.sealed_test_start_year} refused for train/tune",
        )
        pipeline_stages = dict(
            active_stage=self.config.pipeline_stage,
            stage1_pretraining=dict(
                name="Stage 1: End-to-End Pre-Training (2000-2023 Multimodal)",
                train_window=f"{s_cfg.train_start_year}-{s_cfg.train_end_year}",
                val_window=str(s_cfg.val_year),
                sealed_test_start_year=self.config.sealed_test_start_year,
                updates_model_weights=True,
                post_training_layers_active=False,
                status=(
                    "active"
                    if self.config.pipeline_stage == "stage1_pretraining"
                    else "completed_prerequisite"
                ),
            ),
            stage2_posttraining=dict(
                name="Stage 2: Post-Training Calibration / Convex P+ / Metacognitive A5-A8",
                calibration_window=str(s_cfg.val_year),
                sealed_test_start_year=self.config.sealed_test_start_year,
                updates_model_weights=False,
                post_training_layers=[
                    "calibration",
                    "convex_portfolio_p_plus",
                    "metacognitive_a5_a8",
                ],
                requires_stage1_gain_pass=True,
                status=(
                    "locked_pending_stage1"
                    if self.config.pipeline_stage == "stage1_pretraining"
                    else "active"
                ),
            ),
        )
        deterministic_contract = dict(
            schema_version=1,
            variation=self.config.variation,
            architecture_fingerprint=str(audit["fingerprint"]),
            model_config=self._model.config.identity(),
            split_config=asdict(s_cfg),
            seal_status=seal_status,
            pipeline_stages=pipeline_stages,
            feature_contract=feature_contract,
        )
        manifest_sha256 = _canonical_sha256(deterministic_contract)

        # Reanudación limpia: nunca sobrescribir silenciosamente un `run_dir` existente
        if self._manifest_path.is_file():
            existing = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if existing.get("architecture_fingerprint") != str(audit["fingerprint"]):
                raise ValueError(
                    "Conflicto de architecture_fingerprint al reanudar run_dir: "
                    f"existente={existing.get('architecture_fingerprint')!r} vs "
                    f"actual={audit['fingerprint']!r}"
                )
            if existing.get("variation") != self.config.variation:
                raise ValueError(
                    f"Conflicto de variación en run_dir: existente={existing.get('variation')!r} "
                    f"vs actual={self.config.variation!r}"
                )
            existing_seal = existing.get("seal_status", {}).get("sealed_test_start_year")
            if existing_seal != self.config.sealed_test_start_year:
                raise ValueError(
                    f"Conflicto de sello temporal en run_dir: existente={existing_seal} vs "
                    f"actual={self.config.sealed_test_start_year}"
                )

            self._manifest = existing
            # Conservar el formato de ledger existente en run_dir para no fragmentar progreso
            if not self._ledger_path.is_file():
                alt_ledger = self._run_dir / (
                    "progress_ledger.jsonl"
                    if self._ledger_path.suffix == ".gz"
                    else "progress_ledger.jsonl.gz"
                )
                if alt_ledger.is_file():
                    self._ledger_path = alt_ledger
            if self._latest_ckpt_path.is_file():
                self.load_checkpoint(self._latest_ckpt_path)
            else:
                # Reconstruir cursor desde el ledger si el checkpoint todavía no existía
                ledger_entries = read_progress_ledger(self._ledger_path)
                completed = [e for e in ledger_entries if e.get("event") == "session_completed"]
                self._completed_keys = {str(e["session_key"]) for e in completed}
                if completed:
                    last_e = completed[-1]
                    self._epoch = int(last_e.get("epoch", 0))
                    self._session_cursor = int(last_e.get("session_idx", -1)) + 1
                    self._global_step = int(last_e.get("global_step", len(completed)))

            _append_jsonl(
                self._ledger_path,
                dict(
                    event="resume_initialized",
                    timestamp_utc=_utc_now_iso(),
                    variation=self.config.variation,
                    pipeline_stage=self.config.pipeline_stage,
                    architecture_fingerprint=str(audit["fingerprint"]),
                    epoch=self._epoch,
                    session_cursor=self._session_cursor,
                    global_step=self._global_step,
                    completed_sessions=len(self._completed_keys),
                ),
            )
            return dict(
                manifest=dict(self._manifest),
                resumed=True,
                epoch=self._epoch,
                session_cursor=self._session_cursor,
                global_step=self._global_step,
                completed_sessions=len(self._completed_keys),
            )

        # Proteger cualquier ledger previo huérfano (.jsonl o .jsonl.gz) frente a sobrescritura
        for cand_ledger in (
            self._run_dir / "progress_ledger.jsonl",
            self._run_dir / "progress_ledger.jsonl.gz",
        ):
            if cand_ledger.is_file() and cand_ledger.stat().st_size > 0:
                raise RuntimeError(
                    f"{cand_ledger.name} ya existe en {self._run_dir} sin manifest.json; "
                    "se aborta para evitar sobrescribir progreso previo"
                )

        manifest_payload: dict[str, Any] = dict(
            schema_version=1,
            harness_version="long_running_titans_v1",
            created_at_utc=_utc_now_iso(),
            run_dir=str(self._run_dir),
            variation=self.config.variation,
            architecture_fingerprint=str(audit["fingerprint"]),
            manifest_sha256=manifest_sha256,
            parameter_audit=audit,
            model_config=self._model.config.identity(),
            harness_config=self.config.to_dict(),
            split_config=asdict(s_cfg),
            seal_status=seal_status,
            pipeline_stages=pipeline_stages,
            hardware_telemetry=self._hw_info,
            feature_contract=feature_contract,
        )
        _atomic_write_json(self._manifest_path, manifest_payload)
        self._manifest = manifest_payload

        _append_jsonl(
            self._ledger_path,
            dict(
                event="run_initialized",
                timestamp_utc=_utc_now_iso(),
                variation=self.config.variation,
                pipeline_stage=self.config.pipeline_stage,
                split_mode=self.config.split_mode,
                architecture_fingerprint=str(audit["fingerprint"]),
                manifest_sha256=manifest_sha256,
                sealed_test_start_year=self.config.sealed_test_start_year,
                device=str(self._device),
            ),
            sync=True,
        )

        # Guardar checkpoint inicial verificado (cursor 0) para permitir rollback desde sesión 0
        self.save_checkpoint(milestone=False, log_event=False)
        return dict(
            manifest=dict(self._manifest),
            resumed=False,
            epoch=self._epoch,
            session_cursor=self._session_cursor,
            global_step=self._global_step,
            completed_sessions=0,
        )

    def _build_checkpoint_payload(self) -> dict[str, Any]:
        return dict(
            schema_version=1,
            saved_at_utc=_utc_now_iso(),
            variation=self.config.variation,
            architecture_fingerprint=self.model.config.fingerprint(),
            manifest_sha256=self._manifest.get("manifest_sha256"),
            epoch=self._epoch,
            session_cursor=self._session_cursor,
            global_step=self._global_step,
            completed_keys=sorted(self._completed_keys),
            model_state_dict=self.model.state_dict(),
            ema_state_dict=self.ema_model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=(
                self._scheduler.state_dict() if self._scheduler is not None else None
            ),
            scheduler_total_steps=self._scheduler_total_steps,
            causal_state=_clone_state_to_cpu(self._causal_state),
            w_prev=self._w_prev.detach().cpu().clone() if self._w_prev is not None else None,
            w_prev_asset_ids=(
                self._w_prev_asset_ids.detach().cpu().clone()
                if self._w_prev_asset_ids is not None
                else None
            ),
            health_gate_state=self.health_gate.state_dict(),
            agent_governor_state=_export_governor_state(self.model),
            ema_governor_state=_export_governor_state(self.ema_model),
            torch_rng_state=torch.get_rng_state().cpu(),
            cuda_rng_state=(
                [s.cpu() for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
            drop_gen_state=(
                self._drop_gen.get_state().cpu() if self._drop_gen is not None else None
            ),
        )

    def save_checkpoint(self, *, milestone: bool = False, log_event: bool = True) -> Path:
        """Guarda un checkpoint atómico reanudable (modelo + EMA + optimizador + estado causal)."""
        # Vaciar registros previos del ledger antes del checkpoint para consistencia en disco
        if not log_event:
            self.flush_ledger(sync=True)
        payload = self._build_checkpoint_payload()
        _atomic_save_checkpoint(self._latest_ckpt_path, payload)
        saved_path = self._latest_ckpt_path
        if milestone:
            ms_name = f"ckpt_e{self._epoch:03d}_s{self._session_cursor:06d}.pt"
            saved_path = self._checkpoints_dir / ms_name
            _atomic_save_checkpoint(saved_path, payload)
        if log_event:
            self._enqueue_ledger_event(
                dict(
                    event="checkpoint_saved",
                    timestamp_utc=_utc_now_iso(),
                    epoch=self._epoch,
                    session_cursor=self._session_cursor,
                    global_step=self._global_step,
                    milestone=milestone,
                    checkpoint_path=str(saved_path),
                ),
                flush=True,
                sync=True,
            )
        return saved_path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Carga y restaura íntegramente un checkpoint atómico verificado."""
        ckpt = torch.load(Path(path), map_location=self._device, weights_only=False)
        if ckpt.get("architecture_fingerprint") != self.model.config.fingerprint():
            raise ValueError(
                "Checkpoint incompatible con el architecture_fingerprint actual: "
                f"{ckpt.get('architecture_fingerprint')!r} != {self.model.config.fingerprint()!r}"
            )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        sched_steps = ckpt.get("scheduler_total_steps")
        sched_state = ckpt.get("scheduler_state_dict")
        if sched_steps is not None and sched_state is not None:
            self._scheduler = self._build_scheduler(int(sched_steps))
            self._scheduler.load_state_dict(sched_state)
            for grp, restored_lr in zip(
                self.optimizer.param_groups, self._scheduler.get_last_lr(), strict=True
            ):
                grp["lr"] = restored_lr
        else:
            self._scheduler = None
            self._scheduler_total_steps = None

        self._causal_state = _move_state_to_device(ckpt.get("causal_state"), self._device)
        w_prev = ckpt.get("w_prev")
        self._w_prev = (
            w_prev.detach().to(device=self._device).clone()
            if isinstance(w_prev, torch.Tensor)
            else None
        )
        w_prev_ids = ckpt.get("w_prev_asset_ids")
        self._w_prev_asset_ids = (
            w_prev_ids.detach().to(device=self._device).clone()
            if isinstance(w_prev_ids, torch.Tensor)
            else None
        )
        self._epoch = int(ckpt.get("epoch", 0))
        self._session_cursor = int(ckpt.get("session_cursor", 0))
        self._global_step = int(ckpt.get("global_step", 0))
        self._completed_keys = set(ckpt.get("completed_keys", []))

        if ckpt.get("health_gate_state") is not None:
            self.health_gate.load_state_dict(ckpt["health_gate_state"])
        _restore_governor_state(self.model, ckpt.get("agent_governor_state"))
        _restore_governor_state(self.ema_model, ckpt.get("ema_governor_state"))

        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            saved_cuda_rng = [s.cpu() for s in ckpt["cuda_rng_state"]]
            n_devices = int(torch.cuda.device_count())
            if n_devices <= 0 or len(saved_cuda_rng) == n_devices:
                torch.cuda.set_rng_state_all(saved_cuda_rng)
            else:
                for dev_idx in range(min(len(saved_cuda_rng), n_devices)):
                    torch.cuda.set_rng_state(saved_cuda_rng[dev_idx], device=dev_idx)
        if self._drop_gen is not None and ckpt.get("drop_gen_state") is not None:
            self._drop_gen.set_state(ckpt["drop_gen_state"].cpu())
        self._nonfinite_steps_since_check = torch.zeros(
            (), device=self._device, dtype=torch.float32
        )
        return ckpt

    def rollback_to_last_verified(self) -> bool:
        """Revierte de forma segura modelo, EMA, optimizador y estado al último checkpoint sano."""
        if not self._latest_ckpt_path.is_file():
            return False
        self.load_checkpoint(self._latest_ckpt_path)
        return True

    def compact_working_state(
        self,
        state: dict[str, object] | None,
        *,
        model: MMARRSTitan | None = None,
    ) -> tuple[dict[str, object], dict[str, Any]]:
        """Ejecuta compactación *Zero-Drift* usando las tasas de decaimiento de `HarnessConfig`."""
        return compact_working_state(
            state,
            fast_decay=self.config.fast_compaction_decay,
            momentum_decay=self.config.momentum_compaction_decay,
            model=model if model is not None else self._model,
        )

    def validate_and_record_gain(
        self,
        candidate_report: Mapping[str, Any],
        control_reports: Mapping[str, Mapping[str, Any]],
        *,
        gate: GainValidationGate | None = None,
    ) -> GainVerdict:
        """Audita la ganancia OOS de Stage 1 contra controles y actualiza manifiesto + ledger."""
        validator = gate or GainValidationGate()
        verdict = validator.evaluate(self.config.variation, candidate_report, control_reports)

        if self._manifest:
            stages = self._manifest.setdefault("pipeline_stages", {})
            s1 = stages.setdefault("stage1_pretraining", {})
            s2 = stages.setdefault("stage2_posttraining", {})
            s1["gain_verdict"] = verdict.to_dict()
            s1["status"] = "passed_gain_gate" if verdict.promoted else "rejected_gain_gate"
            s2["status"] = "unlocked_ready" if verdict.promoted else "blocked_gain_rejected"
            _atomic_write_json(self._manifest_path, self._manifest)

        self._enqueue_ledger_event(
            dict(
                event="gain_validation",
                timestamp_utc=_utc_now_iso(),
                variation=self.config.variation,
                pipeline_stage=self.config.pipeline_stage,
                verdict=verdict.verdict,
                promoted=verdict.promoted,
                details=verdict.to_dict(),
            ),
            flush=True,
            sync=True,
        )
        return verdict

    def run_incremental_sessions(
        self,
        sessions: Sequence[dict[str, torch.Tensor]] | Any,
        session_years: Sequence[int] | None = None,
        *,
        max_sessions: int | None = None,
        epoch: int = 0,
        total_planned_steps: int | None = None,
        session_ids: Sequence[str] | None = None,
        split_cfg: FullHistorySplitConfig | None = None,
        update_weights: bool = True,
        continue_after_rollback: bool = False,
        post_step_hook: Callable[[int, MMARRSTitan, dict[str, Any], dict[str, Any]], None]
        | None = None,
    ) -> dict[str, Any]:
        """Procesa sesiones cronológicas con verificación, compactación y checkpoints."""
        if self.config.pipeline_stage == "stage2_posttraining" and update_weights:
            raise ValueError(
                "Stage 2 (post-training calibration / P+ / A5-A8) no permite actualizar "
                "pesos del tronco de Stage 1; pase update_weights=False"
            )
        if session_years is None:
            if hasattr(sessions, "session_years"):
                session_years = [int(y) for y in np.asarray(sessions.session_years).tolist()]
            else:
                raise ValueError("session_years es obligatorio salvo al pasar ZeroCopySessionStore")
        if len(sessions) != len(session_years):
            raise ValueError("sessions y session_years deben tener exactamente la misma longitud")
        if len(sessions) == 0:
            return dict(
                sessions_processed=0,
                epoch=self._epoch,
                session_cursor=self._session_cursor,
                global_step=self._global_step,
                compactions=0,
                checkpoints_saved=0,
                rollbacks=0,
                health_reports=[],
                last_health_report=None,
            )

        years_arr = np.asarray(session_years, dtype=np.int64)
        if np.any(np.diff(years_arr) < 0):
            raise ValueError("Las sesiones deben llegar en orden estrictamente cronológico")

        n_sealed_withheld = 0
        if split_cfg is not None:
            parts = self.prepare_splits(sessions, session_years, split_cfg=split_cfg)
            work_sessions: Any = parts["train"]
            work_years: list[int] = list(parts["train_years"])
            n_sealed_withheld = int(parts["n_sealed_withheld"])
        else:
            validate_no_sealed_leakage(
                session_years, sealed_test_start_year=self.config.sealed_test_start_year
            )
            work_sessions = sessions
            work_years = [int(y) for y in session_years]

        if session_ids is not None and len(session_ids) != len(work_sessions):
            raise ValueError("session_ids debe coincidir en longitud con las sesiones activas")

        if self._model is None:
            first_sample = (
                work_sessions.get_session(0, zero_copy=True)
                if hasattr(work_sessions, "get_session")
                else work_sessions[0]
            )
            self.initialize(
                sample_session=first_sample,
                split_cfg=split_cfg,
                total_planned_steps=total_planned_steps,
            )

        if epoch < self._epoch:
            return dict(
                sessions_processed=0,
                epoch=self._epoch,
                session_cursor=self._session_cursor,
                global_step=self._global_step,
                compactions=0,
                checkpoints_saved=0,
                rollbacks=0,
                n_sealed_withheld=n_sealed_withheld,
                health_reports=[],
                last_health_report=None,
            )

        if epoch > self._epoch:
            self._epoch = epoch
            self._session_cursor = 0
            self._causal_state = None
            self._w_prev = None
            self._w_prev_asset_ids = None

        self._ensure_scheduler(total_planned_steps, fallback_steps=len(work_sessions))

        start_idx = self._session_cursor
        end_idx = (
            len(work_sessions)
            if max_sessions is None
            else min(len(work_sessions), start_idx + max(0, int(max_sessions)))
        )

        if start_idx >= end_idx:
            return dict(
                sessions_processed=0,
                epoch=self._epoch,
                session_cursor=self._session_cursor,
                global_step=self._global_step,
                compactions=0,
                checkpoints_saved=0,
                rollbacks=0,
                n_sealed_withheld=n_sealed_withheld,
                health_reports=[],
                last_health_report=None,
            )

        active_model = self._compiled_model if self._compiled_model is not None else self.model
        assert self._loss_fn is not None
        assert self._drop_gen is not None

        sessions_processed = 0
        compactions = 0
        checkpoints_saved = 0
        rollbacks = 0
        dirty_since_checkpoint = False
        health_reports: list[HealthReport] = []

        prefetch = _iter_sessions_h2d(
            work_sessions,
            start_idx,
            end_idx,
            self._device,
            use_pinned=self.runtime_cfg.use_pinned_prefetch,
        )

        def _invoke_step_matured(
            mdl: MMARRSTitan,
            st_in: dict[str, object],
            b_mat: dict[str, torch.Tensor],
            y_mat: torch.Tensor,
            p_mat: torch.Tensor,
        ) -> dict[str, object]:
            kw: dict[str, Any] = dict(
                matured_batch=b_mat,
                matured_targets=y_mat,
                matured_preds=p_mat,
            )
            aids_t = b_mat.get("asset_ids")
            if aids_t is not None and "asset_ids" in inspect.signature(mdl.step_matured).parameters:
                kw["asset_ids"] = aids_t
            st_next = mdl.step_matured(st_in, **kw)
            if isinstance(aids_t, torch.Tensor):
                st_next["last_asset_ids"] = aids_t.detach()
            return st_next

        def _safe_scalar_diag(d_map: Mapping[str, Any], key: str, default: float) -> float | None:
            v = d_map.get(key, default)
            if isinstance(v, torch.Tensor):
                return None  # Evitar sync host en pasos no muestreados
            return float(v)

        use_gpu_found_inf = _supports_gpu_found_inf(self.optimizer, self._device)

        try:
            for offset, batch in enumerate(prefetch):
                idx = start_idx + offset
                yr = work_years[idx]
                # Defensa en profundidad: jamás procesar una sesión del bloque sellado
                if yr >= self.config.sealed_test_start_year:
                    raise SealedTestViolationError(
                        f"Sesión {idx} pertenece al año sellado {yr} >= "
                        f"{self.config.sealed_test_start_year}"
                    )

                session_key = (
                    str(session_ids[idx]) if session_ids is not None else f"e{epoch}:s{idx}"
                )
                if session_key in self._completed_keys:
                    self._session_cursor = max(self._session_cursor, idx + 1)
                    continue

                next_global = self._global_step + 1
                is_last_in_chunk = idx == end_idx - 1
                will_checkpoint = (
                    next_global % self.config.checkpoint_every_sessions == 0 or is_last_in_chunk
                )
                should_check_health = (
                    next_global % self.config.health_check_every_k == 0
                    or will_checkpoint
                    or post_step_hook is not None
                )

                loss_dict: dict[str, torch.Tensor] = {}
                loss_tensor: torch.Tensor | None = None
                step_finite: torch.Tensor | None = None
                step_ok_host: bool | None = None
                y_ret = batch.get("target_returns")
                cur_asset_ids = batch.get("asset_ids")

                # Alinear `agent_w_prev` del estado causal por `asset_ids` en paneles irregulares
                state_in: dict[str, object] | None = self._causal_state
                if (
                    state_in is not None
                    and isinstance(cur_asset_ids, torch.Tensor)
                    and isinstance(state_in.get("agent_w_prev"), torch.Tensor)
                    and isinstance(state_in.get("last_asset_ids"), torch.Tensor)
                ):
                    aligned_agent_w = _align_w_prev_by_asset_ids(
                        cur_asset_ids,
                        state_in["last_asset_ids"],  # type: ignore[arg-type]
                        state_in["agent_w_prev"],  # type: ignore[arg-type]
                    )
                    state_in = dict(state_in)
                    if aligned_agent_w is not None:
                        state_in["agent_w_prev"] = aligned_agent_w

                if update_weights:
                    if y_ret is None:
                        raise ValueError(
                            f"Sesión {idx} carece de 'target_returns' requerido en entrenamiento"
                        )
                    self.model.train()
                    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                        self.optimizer.train()
                    b_in = _apply_structured_modality_dropout(
                        batch, self.config.modality_dropout, self._drop_gen
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    with autocast_context(self._device, self.runtime_cfg):
                        out = active_model.forward_session(
                            b_in, state=state_in, enable_early_exit=False
                        )
                        if isinstance(cur_asset_ids, torch.Tensor):
                            w_prev_in = _align_w_prev_by_asset_ids(
                                cur_asset_ids, self._w_prev_asset_ids, self._w_prev
                            )
                        else:
                            w_prev_in = (
                                self._w_prev
                                if (
                                    self._w_prev is not None
                                    and len(self._w_prev) == y_ret.shape[0]
                                )
                                else None
                            )
                        loss_dict = self._loss_fn(
                            step_outputs=out["loop_outputs"],
                            targets=dict(returns=y_ret),
                            exposures=b_in.get("exposures"),
                            energy_total=out["energy"]["total"],
                            w_prev=w_prev_in,
                        )
                    loss_tensor = loss_dict["total"]
                    loss_finite = torch.isfinite(loss_tensor.detach())
                    loss_tensor.backward()
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.grad_clip_norm
                    )
                    step_finite = loss_finite & torch.isfinite(total_grad_norm.detach())
                    assert self._scheduler is not None

                    if use_gpu_found_inf:
                        # Cero syncs host en CUDA: `_fused_adamw_` salta el paso si `found_inf==1`
                        self.optimizer.found_inf = (~step_finite).to(dtype=torch.float32)  # type: ignore[attr-defined]
                        try:
                            self.optimizer.step()
                        finally:
                            self.optimizer.found_inf = None  # type: ignore[attr-defined]
                        self._scheduler.step()
                        if not should_check_health:
                            self._nonfinite_steps_since_check.add_((~step_finite).float())
                    else:
                        step_ok_host = bool(step_finite.item())
                        if step_ok_host:
                            self.optimizer.step()
                            self._scheduler.step()
                        else:
                            self.optimizer.zero_grad(set_to_none=True)
                            if not should_check_health:
                                self._nonfinite_steps_since_check.add_(1.0)

                    if post_step_hook is not None:
                        post_step_hook(idx, self.model, out, loss_dict)
                    target_model = self.model
                else:
                    self.ema_model.eval()
                    b_in = batch
                    with torch.no_grad():
                        out = self.ema_model.forward_session(
                            b_in, state=state_in, enable_early_exit=True
                        )
                    target_model = self.ema_model

                # Macro-bucle causal `step_matured` (protegido ante paso NaN en CPU/GPU)
                if y_ret is not None:
                    if step_ok_host is False:
                        next_causal_state = self._causal_state or out["state"]
                    else:
                        matured_p0 = out["pred_return"][:, 0].detach()
                        state_for_matured = out["state"]
                        if use_gpu_found_inf and step_finite is not None:
                            sf_m = step_finite.to(device=matured_p0.device, dtype=torch.bool)
                            matured_p0 = torch.where(
                                sf_m & torch.isfinite(matured_p0),
                                matured_p0,
                                torch.zeros_like(matured_p0),
                            )
                            state_for_matured = _sanitize_matured_input_state(
                                out["state"], step_finite
                            )
                        raw_next_state = _invoke_step_matured(
                            target_model,
                            state_for_matured,
                            b_in,
                            y_ret,
                            matured_p0,
                        )
                        next_causal_state = (
                            _sanitize_causal_state_masked(
                                self._causal_state, raw_next_state, step_finite
                            )
                            if (use_gpu_found_inf and step_finite is not None)
                            else raw_next_state
                        )
                else:
                    next_causal_state = out["state"]
                    if isinstance(cur_asset_ids, torch.Tensor):
                        next_causal_state["last_asset_ids"] = cur_asset_ids.detach()

                report: HealthReport | None = None
                if should_check_health:
                    # UNA sola sync D2H agrupada: pesos + preds + pérdida + estado pre/post-matured
                    report = self.health_gate.verify_session(
                        model=target_model,
                        outputs=out,
                        state=out["state"],
                        post_matured_state=next_causal_state,
                        targets=y_ret,
                        loss=loss_tensor,
                        prior_nonfinite_steps=self._nonfinite_steps_since_check,
                        rollback_fn=self.rollback_to_last_verified,
                    )
                    self._nonfinite_steps_since_check.zero_()
                    health_reports.append(report)

                    if not report.passed:
                        self._enqueue_ledger_event(
                            dict(
                                event="session_health_failure",
                                timestamp_utc=_utc_now_iso(),
                                epoch=epoch,
                                session_idx=idx,
                                session_key=session_key,
                                session_year=yr,
                                global_step=self._global_step,
                                health=report.to_dict(),
                            ),
                            flush=True,
                            sync=True,
                        )
                        if report.rollback_triggered:
                            rollbacks += 1
                            dirty_since_checkpoint = False
                            weights_ok = SessionHealthGate._all_tensors_finite(
                                list(self.model.parameters())
                            )
                            self._enqueue_ledger_event(
                                dict(
                                    event="rollback_recovered",
                                    timestamp_utc=_utc_now_iso(),
                                    failed_session_idx=idx,
                                    failed_session_key=session_key,
                                    restored_epoch=self._epoch,
                                    restored_session_cursor=self._session_cursor,
                                    restored_global_step=self._global_step,
                                    weights_finite_after_rollback=weights_ok,
                                    hard_failures=list(report.hard_failures),
                                ),
                                flush=True,
                                sync=True,
                            )
                        if not continue_after_rollback:
                            break
                        continue

                # Confirmar estado, EMA, `w_prev` y cursor (protegido ante NaN en todos los pasos)
                if step_ok_host is not False:
                    self._causal_state = next_causal_state
                ema_updated = False
                if update_weights:
                    if use_gpu_found_inf and step_finite is not None:
                        _update_ema_masked(
                            self.ema_model, self.model, self.config.ema_decay, step_finite
                        )
                        cand_w = loss_dict["portfolio_weights"]
                        self._w_prev = torch.where(
                            step_finite & torch.isfinite(cand_w),
                            cand_w,
                            torch.zeros_like(cand_w),
                        )
                        self._w_prev_asset_ids = (
                            cur_asset_ids.detach()
                            if isinstance(cur_asset_ids, torch.Tensor)
                            else None
                        )
                        ema_updated = True
                    elif step_ok_host:
                        _update_ema(self.ema_model, self.model, self.config.ema_decay)
                        self._w_prev = loss_dict["portfolio_weights"]
                        self._w_prev_asset_ids = (
                            cur_asset_ids.detach()
                            if isinstance(cur_asset_ids, torch.Tensor)
                            else None
                        )
                        ema_updated = True

                self._session_cursor = idx + 1
                self._global_step += 1
                self._completed_keys.add(session_key)
                sessions_processed += 1
                dirty_since_checkpoint = True

                # Compactación periódica Zero-Drift cada `compaction_interval` sesiones
                comp_metrics: dict[str, Any] | None = None
                if self._global_step % self.config.compaction_interval == 0:
                    self._causal_state, comp_metrics = self.compact_working_state(
                        self._causal_state, model=target_model
                    )
                    compactions += 1
                    self._enqueue_ledger_event(
                        dict(
                            event="context_compaction",
                            timestamp_utc=_utc_now_iso(),
                            epoch=epoch,
                            session_idx=idx,
                            session_key=session_key,
                            global_step=self._global_step,
                            metrics=comp_metrics,
                        )
                    )

                diag = out.get("diagnostics", {})
                k_used_raw = diag.get("k_used", 1)
                k_used_val = None if isinstance(k_used_raw, torch.Tensor) else int(k_used_raw)
                self._enqueue_ledger_event(
                    dict(
                        event="session_completed",
                        timestamp_utc=_utc_now_iso(),
                        pipeline_stage=self.config.pipeline_stage,
                        epoch=epoch,
                        session_idx=idx,
                        session_key=session_key,
                        session_year=yr,
                        global_step=self._global_step,
                        health_checked=should_check_health,
                        loss=report.metrics.get("loss") if report is not None else None,
                        ema_updated=ema_updated,
                        compacted=comp_metrics is not None,
                        router_entropy=_safe_scalar_diag(diag, "router_entropy", 0.0),
                        load_max_min_ratio=_safe_scalar_diag(diag, "load_max_min_ratio", 1.0),
                        k_used=k_used_val,
                        agent_active_rate=_safe_scalar_diag(diag, "agent_active_rate", 1.0),
                        health=report.to_dict() if report is not None else None,
                    )
                )

                # Checkpoint atómico de hito cada `checkpoint_every_sessions` sesiones
                if self._global_step % self.config.checkpoint_every_sessions == 0:
                    self.save_checkpoint(milestone=True, log_event=True)
                    checkpoints_saved += 1
                    dirty_since_checkpoint = False

            # Garantizar checkpoint atómico al cerrar el tramo para reanudación exacta sin repetir
            if dirty_since_checkpoint:
                self.save_checkpoint(milestone=False, log_event=True)
                checkpoints_saved += 1
        finally:
            self.flush_ledger(sync=False)

        return dict(
            sessions_processed=sessions_processed,
            epoch=self._epoch,
            session_cursor=self._session_cursor,
            global_step=self._global_step,
            compactions=compactions,
            checkpoints_saved=checkpoints_saved,
            rollbacks=rollbacks,
            n_sealed_withheld=n_sealed_withheld,
            health_reports=health_reports,
            last_health_report=health_reports[-1] if health_reports else None,
        )
