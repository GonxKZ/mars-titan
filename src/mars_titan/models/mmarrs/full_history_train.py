"""Entrenador oficial de historia completa (`2000–2023` todo de golpe -> `2024+` test sellado).

Integra de extremo a extremo el stack SOTA sobre NVIDIA RTX PRO 6000 Blackwell (`sm_120`):
  - **P0-1:** Alineación canónica por `asset_ids` en `w_prev` y evaluación de paneles irregulares.
  - **P0-2:** BPTT truncado (`bptt_steps=4`) en `NeuromodulatedTitansMemory` + `target_encoder`.
  - **P0-4:** Purga/embargo estrictos multi-horizonte (`purge >= max_h`, `embargo >= 0.2*max_h`)
    y `horizon_mask` automático en targets 1D.
  - **P0-5:** Jamás pasa `vol_pred` al evaluador; pasa `realized_vol`, `adv_usd` real y `n_trials`.
  - **P0-7:** `ZeroCopySessionStore.iter_pinned_h2d`, `sdpa_math_context` en `fp32_strict`,
    pérdida en GPU (1 sync/época) y CLI canónico de 8 flags con separación FM-8 `US|CN`.
  - **P1-1:** `_update_ema` sin corromper HRR + `reset_episodic_buffers`, `OptimalFitController`
    (Top-3 Model Soup + parada adaptativa suavizada), `build_sota_optimizer` (`adamw_fused` /
    `muon_hybrid` / `schedule_free_adamw`, `wd=1e-2`), `ResearchTelemetryMonitor` (`.jsonl.gz`)
    y ensamblado por rangos de 3 semillas (`train_multiseed_rank_ensemble`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mars_titan.data.dataset_enrichment_sota import MissingnessResilientFeaturizer
from mars_titan.evaluation.granular_evaluator import evaluate_granular_six_dimensions

from .ablation_runner import (
    _EPISODIC_BUFFER_SUFFIXES as _EPISODIC_BUFFER_SUFFIXES,
)
from .ablation_runner import (
    _collect_aux_losses as _collect_aux_losses,
)
from .ablation_runner import (
    _extract_liquidity_weights,
    _infer_dims,
    _is_episodic_buffer,
    _reset_episodic_buffers,
    _update_spectral_buffers_post_step,
)
from .ablation_runner import (
    _promote_fp32_outputs as _promote_fp32_outputs,
)
from .config import (
    DEFAULT_HORIZONS,
    MMARRS_FAMILIES,
    MMARRS_VARIATIONS,
    VALID_MARKETS,
    required_purge_embargo,
)
from .dl_training_sota import (
    CausalCurriculumScheduler,
    CausalSSLPretrainer,
    DynamicTanh,
    LayerScale,
    RMSNormSOTA,
    StructuredModalityNoiseAugmenter,
    adaptive_gradient_clip_,
    apply_cautious_weight_decay_,
    apply_mup_init_and_param_groups,
    cautious_weight_decay_mask,
)
from .granular_heads import MMARRSMultiTaskLoss, infer_horizon_mask
from .hardware_blackwell import (
    PRECISION_MODES,
    BlackwellRuntimeConfig,
    autocast_context,
    blackwell_mps_launch_plan,
    configure_blackwell_runtime,
    iter_sessions_h2d,
    maybe_compile_model,
    sdpa_math_context,
)
from .loss_geometry_probe import DeepLearningGeometryProbe, HessianConfig
from .model import MMARRSTitan, build_family_model
from .telemetry import ResearchTelemetryMonitor, TelemetryConfig

SPLIT_MODES = ("full_history_final", "expanding_walk_forward")


@dataclass(frozen=True)
class FullHistorySplitConfig:
    """Partición temporal estricta Point-in-Time con purga, embargo y sello de 2024+."""

    mode: str = "full_history_final"
    train_start_year: int = 2000
    train_end_year: int = 2022
    val_year: int = 2023
    sealed_test_start_year: int = 2024
    purge_sessions: int = 63
    embargo_sessions: int = 13
    active_horizons: tuple[int, ...] | None = None
    allow_sealed_test: bool = False
    allow_short_synthetic: bool = False

    def __post_init__(self) -> None:
        if self.mode not in SPLIT_MODES:
            raise ValueError(f"mode inválido {self.mode!r}; válidos: {SPLIT_MODES}")
        if self.train_end_year < self.train_start_year:
            raise ValueError("train_end_year debe ser >= train_start_year")
        if self.val_year <= self.train_end_year:
            raise ValueError("val_year debe ser estrictamente posterior a train_end_year")
        if self.val_year >= self.sealed_test_start_year:
            raise ValueError("val_year debe preceder estrictamente al test sellado (2024+)")
        if self.purge_sessions < 0 or self.embargo_sessions < 0:
            raise ValueError("purge_sessions y embargo_sessions deben ser >= 0")
        if self.active_horizons is not None and not self.allow_short_synthetic:
            min_p, min_e = required_purge_embargo(self.active_horizons)
            if self.purge_sessions < min_p:
                raise ValueError(
                    f"P0-4 fuga multi-horizonte: purge_sessions={self.purge_sessions} < "
                    f"max(active_horizons)={min_p}"
                )
            if self.embargo_sessions < min_e:
                raise ValueError(
                    f"P0-4 embargo insuficiente: embargo_sessions={self.embargo_sessions} < "
                    f"{min_e} para max_h={min_p}"
                )


def _validate_purge_embargo_for_horizons(
    cfg: FullHistorySplitConfig,
    horizons: Sequence[int],
    *,
    allow_short_synthetic: bool | None = None,
) -> None:
    allow_syn = (
        cfg.allow_short_synthetic if allow_short_synthetic is None else allow_short_synthetic
    )
    if allow_syn:
        return
    min_p, min_e = required_purge_embargo(horizons)
    if cfg.purge_sessions < min_p:
        raise ValueError(
            f"P0-4 fuga multi-horizonte: purge_sessions={cfg.purge_sessions} < "
            f"max(active_horizons)={min_p} (pase allow_short_synthetic=True solo en tests)"
        )
    if cfg.embargo_sessions < min_e:
        raise ValueError(
            f"P0-4 embargo insuficiente: embargo_sessions={cfg.embargo_sessions} < "
            f"{min_e} para max_h={min_p} (pase allow_short_synthetic=True solo en tests)"
        )


def split_chronological_sessions(
    sessions: Any,
    session_years: Sequence[int] | None,
    cfg: FullHistorySplitConfig,
    *,
    active_horizons: Sequence[int] | None = None,
    allow_short_synthetic: bool | None = None,
) -> dict[str, Any]:
    """Divide sesiones o `ZeroCopySessionStore` con purga, embargo y sello `2024+`."""
    eff_horizons = active_horizons if active_horizons is not None else cfg.active_horizons
    eff_allow_syn = (
        cfg.allow_short_synthetic if allow_short_synthetic is None else bool(allow_short_synthetic)
    )
    if eff_horizons is not None:
        _validate_purge_embargo_for_horizons(cfg, eff_horizons, allow_short_synthetic=eff_allow_syn)

    # Ruta zero-copy nativa cuando `sessions` es un `ZeroCopySessionStore`
    if hasattr(sessions, "chronological_splits"):
        hs = tuple(eff_horizons) if eff_horizons is not None else DEFAULT_HORIZONS
        return sessions.chronological_splits(
            train_start_year=cfg.train_start_year,
            train_end_year=cfg.train_end_year,
            val_year=cfg.val_year,
            sealed_test_start_year=cfg.sealed_test_start_year,
            active_horizons=hs,
            purge_sessions=cfg.purge_sessions,
            embargo_sessions=cfg.embargo_sessions,
            allow_zero_purge=eff_allow_syn,
            allow_sealed_test=cfg.allow_sealed_test,
        )

    if session_years is None:
        raise ValueError("session_years es obligatorio para secuencias de sesiones en memoria")
    if len(sessions) != len(session_years):
        raise ValueError("sessions y session_years deben tener exactamente la misma longitud")

    years = np.asarray(session_years, dtype=np.int64)
    if len(years) > 1 and np.any(np.diff(years) < 0):
        raise ValueError("Las sesiones deben llegar en orden estrictamente cronológico")

    train_idx = np.flatnonzero((years >= cfg.train_start_year) & (years <= cfg.train_end_year))
    val_mask = (
        (years > cfg.train_end_year)
        & (years <= cfg.val_year)
        & (years < cfg.sealed_test_start_year)
    )
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(years >= cfg.sealed_test_start_year)

    # Purga/embargo estrictos (jamás saltarse purga en silencioso)
    if cfg.purge_sessions > 0:
        train_idx = (
            train_idx[: -cfg.purge_sessions]
            if len(train_idx) > cfg.purge_sessions
            else np.empty(0, dtype=np.int64)
        )
    if cfg.embargo_sessions > 0:
        val_idx = (
            val_idx[cfg.embargo_sessions :]
            if len(val_idx) > cfg.embargo_sessions
            else np.empty(0, dtype=np.int64)
        )

    if len(test_idx) > 0 and not cfg.allow_sealed_test:
        test_list: list[dict[str, torch.Tensor]] = []
    else:
        test_list = [sessions[int(i)] for i in test_idx]

    return dict(
        train=[sessions[int(i)] for i in train_idx],
        val=[sessions[int(i)] for i in val_idx],
        sealed_test=test_list,
        n_sealed_withheld=0 if cfg.allow_sealed_test else int(len(test_idx)),
    )


def _apply_structured_modality_dropout(
    batch: dict[str, torch.Tensor],
    prob: float,
    rng: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Apaga aleatoriamente modalidades auxiliares (no precios) para robustez ante ausencias."""
    if prob <= 0.0 or "modality_mask" not in batch:
        return batch
    mask = batch["modality_mask"].clone()
    # Nunca apagar precios (canal 0); aplicar dropout estructurado a canales 1..4
    if mask.shape[-1] > 1:
        keep = (
            torch.rand((mask.shape[0], mask.shape[-1] - 1), generator=rng, device=mask.device)
            >= prob
        ).to(dtype=mask.dtype)
        mask[:, 1:] = mask[:, 1:] * keep
    out = dict(batch)
    out["modality_mask"] = mask
    return out


@torch.no_grad()
def reset_episodic_buffers(model: torch.nn.Module) -> None:
    """Reinicia buffers episódicos HRR + `HippocampalSharpWaveReplay` antes de evaluar EMA/val."""
    _reset_episodic_buffers(model)


def _effective_ema_decay(
    ema_decay: float,
    ema_step: int,
    *,
    bias_correct: bool = True,
) -> float:
    """Decaimiento EMA con corrección de sesgo inicial `min(d, (1+s)/(10+s))` (0 host-syncs)."""
    base = float(ema_decay)
    if not bias_correct:
        return base
    step_f = float(max(0, int(ema_step)))
    return min(base, (1.0 + step_f) / (10.0 + step_f))


def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Actualización EMA de parámetros omitiendo buffers episódicos HRR/Replay (P1-1 fix!)."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters(), strict=True):
            p_ema.mul_(decay).add_(p, alpha=1.0 - decay)
        for (b_name, b_ema), (_, b) in zip(
            ema_model.named_buffers(), model.named_buffers(), strict=True
        ):
            if _is_episodic_buffer(b_name):
                continue
            if b_ema.dtype.is_floating_point:
                b_ema.copy_(b)


def _align_w_prev_ids(
    cur_ids: torch.Tensor | None,
    prev_ids: torch.Tensor | None,
    w_prev: torch.Tensor | None,
    n_assets: int,
) -> torch.Tensor | None:
    """Alinea `w_prev` por `asset_ids` canónicos en paneles irregulares (IPOs -> 0, P0-1)."""
    if w_prev is None:
        return None
    if cur_ids is not None and prev_ids is not None:
        return MMARRSTitan._align_vector_by_asset_ids(cur_ids, prev_ids, w_prev)
    return w_prev if w_prev.shape[0] == n_assets else None


def _build_deranged_context_indices(n_sessions: int, seed: int) -> np.ndarray:
    """Permutación determinista estrictamente sin puntos fijos (`perm[t] != t`) para v05 (P1-3)."""
    if n_sessions <= 1:
        return np.zeros(n_sessions, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sessions)
    for i in range(n_sessions):
        if perm[i] == i:
            swap_j = (i + 1) % n_sessions
            perm[i], perm[swap_j] = perm[swap_j], perm[i]
    return perm.astype(np.int64)


def _extract_realized_vol(batch: dict[str, torch.Tensor]) -> np.ndarray | None:
    """Extrae volatilidad histórica realizada de la sesión (JAMÁS `vol_pred` del modelo, P0-5)."""
    for key in ("realized_vol", "volatility", "hist_vol"):
        val = batch.get(key)
        if val is not None:
            arr = val.detach().float().cpu().numpy()
            return arr[:, 0] if arr.ndim == 2 else arr
    return None


def _extract_adv_usd(batch: dict[str, torch.Tensor]) -> np.ndarray | None:
    """Extrae liquidez ADV USD real de la sesión si existe (P0-5)."""
    for key in ("adv_usd", "liquidity", "dollar_volume"):
        val = batch.get(key)
        if val is not None:
            arr = val.detach().float().cpu().numpy()
            return arr[:, 0] if arr.ndim == 2 else arr
    return None


def _safe_session_rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    if int(np.sum(mask)) < 3:
        return float("nan")
    p = pred[mask]
    y = target[mask]
    if float(np.std(p)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    rp = np.argsort(np.argsort(p)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    c = float(np.corrcoef(rp, ry)[0, 1])
    return c if math.isfinite(c) else 0.0


def _register_optional_spectral_hooks(
    base_model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    """Engancha hooks post-step espectrales si el modelo o submódulos los exponen."""
    for mod in (
        base_model,
        getattr(base_model, "recursive_core", None),
        getattr(base_model, "cognitive_core", None),
    ):
        hook_fn = getattr(mod, "register_spectral_post_step_hook", None)
        if callable(hook_fn):
            hook_fn(optimizer)


def _collect_no_decay_param_ids(model: torch.nn.Module) -> set[int]:
    """Identifica todos los parámetros exentos de weight decay (incl. 2D `no_decay`, #233 P0-2)."""
    mod_by_param: dict[int, torch.nn.Module] = {}
    for _, mod in model.named_modules():
        for p in mod.parameters(recurse=False):
            mod_by_param[id(p)] = mod

    no_decay_ids: set[int] = set()
    no_decay_leaves = frozenset(
        {"bias", "shared_b1", "shared_b2", "routed_b1", "routed_b2", "basal_b", "r_in", "s_mid"}
    )
    no_decay_norm_types = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.GroupNorm,
        torch.nn.Embedding,
        RMSNormSOTA,
        DynamicTanh,
        LayerScale,
    )
    for name, p in model.named_parameters():
        low = name.lower()
        leaf = low.rsplit(".", 1)[-1]
        parent = mod_by_param.get(id(p))
        is_excluded = (
            p.ndim <= 1
            or low.endswith(".bias")
            or leaf in no_decay_leaves
            or isinstance(parent, no_decay_norm_types)
            or "norm" in low
            or "dyt" in low
            or "layerscale" in low
            or "layer_scale" in low
            or "tabm_r" in low
            or "tabm_s" in low
            or "embed" in low
        )
        if is_excluded:
            no_decay_ids.add(id(p))
    return no_decay_ids


def _build_train_optimizer(
    base_model: torch.nn.Module,
    *,
    opt_kind: str,
    lr: float,
    weight_decay: float,
    rt_cfg: BlackwellRuntimeConfig,
    dev: torch.device,
    eff_sf_warmup: int,
    mup_groups: list[dict[str, Any]] | None,
    use_cautious_wd: bool,
) -> torch.optim.Optimizer:
    """Construye optimizador preservando `router_lr_scale` en muP (#233 P1-1) y `cautious_wd`."""
    from .autotuner_sota import (
        MuonHybridOptimizer,
        ScheduleFreeAdamW,
        build_sota_optimizer,
    )

    real_wd = float(weight_decay)
    if mup_groups is not None:
        scaled_groups: list[dict[str, Any]] = []
        for grp in mup_groups:
            g_copy = dict(grp)
            if g_copy.get("mup_role") in ("router_decay", "router_no_decay"):
                g_copy["lr"] = float(g_copy["lr"]) * float(rt_cfg.router_lr_scale)
            if opt_kind == "muon_hybrid":
                g_copy["use_muon"] = bool(g_copy.get("mup_role") == "hidden_2d")
            scaled_groups.append(g_copy)
        if opt_kind == "muon_hybrid":
            opt: torch.optim.Optimizer = MuonHybridOptimizer(
                scaled_groups,
                lr=float(lr),
                weight_decay=real_wd,
                router_lr_scale=rt_cfg.router_lr_scale,
                use_bf16_ns=bool(rt_cfg.precision_mode == "bf16_amp"),
            )
        elif opt_kind == "schedule_free_adamw":
            opt = ScheduleFreeAdamW(
                scaled_groups,
                lr=float(lr),
                weight_decay=real_wd,
                warmup_steps=eff_sf_warmup,
                router_lr_scale=rt_cfg.router_lr_scale,
            )
        else:
            use_fused_opt = bool(rt_cfg.use_fused_adamw and dev.type == "cuda")
            opt = torch.optim.AdamW(
                scaled_groups,
                lr=float(lr),
                weight_decay=real_wd,
                fused=use_fused_opt,
            )
    else:
        opt = build_sota_optimizer(
            base_model,
            optimizer_type=opt_kind,
            lr=lr,
            weight_decay=real_wd,
            router_lr_scale=rt_cfg.router_lr_scale,
            cfg=rt_cfg,
            sf_warmup_steps=eff_sf_warmup,
        )

    if use_cautious_wd:
        for grp in opt.param_groups:
            grp_wd = float(grp.get("weight_decay", 0.0))
            grp["cautious_wd_enabled"] = grp_wd > 0.0
            grp["cautious_wd"] = grp_wd
            grp["weight_decay"] = 0.0
    return opt


@torch.no_grad()
def _apply_cautious_wd_before_step(
    opt: torch.optim.Optimizer,
    no_decay_ids: set[int],
    default_lr: float,
) -> None:
    """Aplica Cautious WD sin syncs en AdamW/Muon y directo en estado `z` de ScheduleFree."""
    from .autotuner_sota import ScheduleFreeAdamW

    is_sf = isinstance(opt, ScheduleFreeAdamW)
    for grp in opt.param_groups:
        if not grp.get("cautious_wd_enabled", False):
            continue
        wd_g = float(grp.get("cautious_wd", 0.0))
        if wd_g <= 0.0:
            continue
        base_lr = float(grp.get("lr", default_lr))
        eligible = [
            p
            for p in grp["params"]
            if p.requires_grad and p.grad is not None and id(p) not in no_decay_ids
        ]
        if not eligible:
            continue
        if is_sf:
            wu = int(grp.get("warmup_steps", 0))
            for p in eligible:
                st_p = opt.state[p]
                next_step = int(st_p.get("step", 0)) + 1
                sched_m = min(1.0, float(next_step) / float(wu)) if wu > 0 else 1.0
                lr_t = base_lr * sched_m
                if lr_t <= 0.0:
                    continue
                if "z" in st_p:
                    # Exact SF-CWD: decay internal `z` using aligned-sign `y_t` before step()!
                    z_buf = st_p["z"]
                    y_c = p.detach().to(dtype=z_buf.dtype)
                    g_c = p.grad.detach().to(dtype=z_buf.dtype)
                    mask = cautious_weight_decay_mask(g_c, y_c)
                    z_buf.add_(mask * y_c, alpha=-(lr_t * wd_g))
                else:
                    # Step 1: ScheduleFree clones `z=x=p` before first update (`ckp1 == 1.0`)
                    apply_cautious_weight_decay_([p], lr=lr_t, weight_decay=wd_g)
        else:
            apply_cautious_weight_decay_(eligible, lr=base_lr, weight_decay=wd_g)


def _evaluate_model_causal(
    eval_model: torch.nn.Module,
    val_s: Any,
    dev: torch.device,
    rt_cfg: BlackwellRuntimeConfig,
    *,
    market: str,
    n_trials: int,
    use_placebo_shuffle: bool,
    seed: int,
    n_horizons: int,
    featurizer: MissingnessResilientFeaturizer | None = None,
) -> tuple[dict[str, dict[str, object]], np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Evalúa un modelo causalmente sobre `val_s` (cero `vol_pred`, ADV/vol reales, P0-5)."""
    reset_episodic_buffers(eval_model)
    eval_model.eval()
    state = None
    p_all, y_all, q_all, pr_all, e_all, w_all = [], [], [], [], [], []
    vol_real_all, adv_all, aids_all = [], [], []
    bounds: list[tuple[int, int]] = []
    offset = 0

    placebo_ctxs: list[torch.Tensor | None] | None = None
    if use_placebo_shuffle and hasattr(val_s, "__len__") and len(val_s) > 1:
        perm = _build_deranged_context_indices(len(val_s), seed + 707)
        placebo_ctxs = [val_s[int(p)].get("high_snr_context") for p in perm]

    with torch.no_grad(), sdpa_math_context(rt_cfg):
        for s_idx, batch in enumerate(
            iter_sessions_h2d(val_s, dev, use_pinned=rt_cfg.use_pinned_prefetch)
        ):
            y_ret = batch["target_returns"]
            b_ev = featurizer.featurize_batch(batch) if featurizer is not None else dict(batch)
            b_ev.setdefault("horizon_mask", infer_horizon_mask(y_ret, n_horizons))
            if placebo_ctxs is not None and placebo_ctxs[s_idx] is not None:
                b_ev["placebo_high_snr_context"] = placebo_ctxs[s_idx].to(dev)

            if isinstance(eval_model, MMARRSTitan):
                out = eval_model.forward_session(b_ev, state=state, enable_early_exit=True)
            else:
                out = eval_model.forward_session(b_ev, state=state)
            y1 = (y_ret[:, 0] if y_ret.ndim == 2 else y_ret).detach().float().cpu().numpy()
            p1 = out["pred_return"][:, 0].detach().float().cpu().numpy()
            q1 = out["quantiles"][:, 0, :].detach().float().cpu().numpy()
            tbp = out["tb_probs"][:, 0, :].detach().float().cpu().numpy()
            p_up = tbp[:, 2] / np.maximum(tbp[:, 0] + tbp[:, 2], 1e-6)
            n_s = len(y1)
            bounds.append((offset, offset + n_s))
            offset += n_s
            p_all.append(p1)
            y_all.append(y1)
            q_all.append(q1)
            pr_all.append(p_up)

            rv = _extract_realized_vol(batch)
            if rv is not None:
                vol_real_all.append(rv)
            adv = _extract_adv_usd(batch)
            if adv is not None:
                adv_all.append(adv)
            if batch.get("asset_ids") is not None:
                aids_all.append(batch["asset_ids"].detach().cpu().numpy())
            if batch.get("exposures") is not None:
                e_all.append(batch["exposures"].detach().float().cpu().numpy())
            if out.get("agent_weights") is not None:
                aw = out["agent_weights"]
                w_all.append(
                    aw.detach().float().cpu().numpy() if isinstance(aw, torch.Tensor) else aw
                )

            if isinstance(eval_model, MMARRSTitan):
                state = eval_model.step_matured(
                    out["state"],
                    matured_batch=b_ev,
                    matured_targets=y_ret,
                    matured_preds=out["pred_return"][:, 0].detach(),
                )
            else:
                state = eval_model.step_matured(
                    out["state"],
                    matured_targets=y_ret,
                    matured_preds=out["pred_return"][:, 0].detach(),
                    asset_ids=b_ev.get("asset_ids"),
                )

    p_cat = np.concatenate(p_all)
    y_cat = np.concatenate(y_all)
    val_report = evaluate_granular_six_dimensions(
        y=y_cat,
        pred=p_cat,
        bounds=bounds,
        quantiles=np.vstack(q_all),
        probabilities=np.concatenate(pr_all),
        exposures=np.vstack(e_all) if e_all else None,
        volatility=np.concatenate(vol_real_all) if len(vol_real_all) == len(bounds) else None,
        liquidity=np.concatenate(adv_all) if len(adv_all) == len(bounds) else None,
        asset_ids=np.concatenate(aids_all) if len(aids_all) == len(bounds) else None,
        portfolio_weights=np.concatenate(w_all) if len(w_all) == len(bounds) else None,
        n_trials=max(1, int(n_trials)),
        market=market,  # type: ignore[arg-type]
    )
    return val_report, p_cat, y_cat, bounds


def train_full_history_variation(
    sessions: Any,
    session_years: Sequence[int] | None = None,
    *,
    variation: str = "v13_mmarrs_titan_full",
    market: str = "US",
    family: str | None = None,
    split_cfg: FullHistorySplitConfig | None = None,
    runtime_cfg: BlackwellRuntimeConfig | None = None,
    fit_cfg: Any = None,
    hidden_dim: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    optimizer_type: str = "adamw_fused",
    bptt_steps: int = 4,
    ema_decay: float = 0.995,
    ema_bias_correct: bool = True,
    sf_disable_ema: bool = True,
    sf_warmup_steps: int | None = None,
    modality_dropout: float = 0.10,
    n_trials: int = 1,
    seed: int = 42,
    device: str = "cuda",
    checkpoint_dir: str | Path | None = None,
    telemetry_dir: str | Path | None = None,
    use_optimal_fit: bool = True,
    allow_short_synthetic: bool | None = None,
    # Wave-3C opt-in DL taxonomy, missingness immunity, SSL & geometry probe (default OFF!)
    w_crps: float | None = None,
    w_listnet_tail: float = 0.0,
    w_latent_cov: float = 0.0,
    w_supcon: float = 0.0,
    w_churn: float = 0.0,
    use_mup: bool = False,
    mup_base_width: int = 128,
    use_agc: bool = False,
    agc_clip_factor: float = 0.01,
    use_cautious_wd: bool = False,
    use_missingness_featurizer: bool = False,
    mod_drop_prob: float = 0.0,
    input_noise_std: float = 0.0,
    curriculum_enabled: bool = False,
    ssl_pretrain_epochs: int = 0,
    collect_geometry_probe: bool = False,
    # Wave-3C opt-in model frontier switches (default OFF / "layernorm" -> bit-exact legacy!)
    use_polar_alpha_head: bool = False,
    use_set_modality_fusion: bool = False,
    set_fusion_mnar_placebo: bool = False,
    use_diff_xasset_attn: bool = False,
    diff_xasset_zero_lambda_control: bool = False,
    norm_kind: str = "layernorm",
) -> dict[str, object]:
    """Entrena una variación de cualquiera de las 4 familias en historia completa (`2000–2023`)."""
    if market not in VALID_MARKETS:
        raise ValueError(
            f"FM-8: market inválido {market!r}; solo se permiten mercados separados {VALID_MARKETS}"
        )
    if family is not None and family not in MMARRS_FAMILIES:
        raise ValueError(f"family inválida {family!r}; válidas: {MMARRS_FAMILIES}")
    if sf_warmup_steps is not None and int(sf_warmup_steps) < 0:
        raise ValueError(f"sf_warmup_steps debe ser >= 0; recibido {sf_warmup_steps}")
    if ssl_pretrain_epochs < 0:
        raise ValueError(f"ssl_pretrain_epochs debe ser >= 0; recibido {ssl_pretrain_epochs}")

    s_cfg = split_cfg or FullHistorySplitConfig()
    rt_cfg = runtime_cfg or BlackwellRuntimeConfig(precision_mode="fp32_strict")
    hw_info = configure_blackwell_runtime(rt_cfg, device=device)
    dev = torch.device(str(hw_info["device"]))

    # Inferir horizontes activos desde el primer batch para validar P0-4 estrictamente
    first_sample = sessions[0]
    y_sample = first_sample["target_returns"]
    n_h_data = 1 if y_sample.ndim == 1 else int(y_sample.shape[-1])
    active_h = (
        s_cfg.active_horizons
        if s_cfg.active_horizons is not None
        else DEFAULT_HORIZONS[: min(n_h_data, len(DEFAULT_HORIZONS))]
    )
    eff_allow_syn = (
        s_cfg.allow_short_synthetic
        if allow_short_synthetic is None
        else bool(allow_short_synthetic)
    )
    _validate_purge_embargo_for_horizons(s_cfg, active_h, allow_short_synthetic=eff_allow_syn)

    parts = split_chronological_sessions(
        sessions,
        session_years,
        s_cfg,
        active_horizons=active_h,
        allow_short_synthetic=eff_allow_syn,
    )
    train_s = parts["train"]
    val_s = parts["val"]
    if len(train_s) < 2 or len(val_s) < 1:
        raise ValueError(
            f"Partición insuficiente: train={len(train_s)}, val={len(val_s)} con {s_cfg}"
        )

    dims = _infer_dims(train_s[0])
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_overrides: dict[str, object] = dict(
        hidden_dim=hidden_dim,
        seed=seed,
        bptt_steps=max(1, int(bptt_steps)),
        weight_decay=float(weight_decay),
        allow_short_synthetic=eff_allow_syn,
        purge_sessions=s_cfg.purge_sessions,
        embargo_sessions=s_cfg.embargo_sessions,
        use_polar_alpha_head=bool(use_polar_alpha_head),
        use_set_modality_fusion=bool(use_set_modality_fusion),
        set_fusion_mnar_placebo=bool(set_fusion_mnar_placebo),
        use_diff_xasset_attn=bool(use_diff_xasset_attn),
        diff_xasset_zero_lambda_control=bool(diff_xasset_zero_lambda_control),
        norm_kind=str(norm_kind),
        **dims,
    )
    base_model = build_family_model(
        variation, market=market, family=family, **model_overrides
    ).to(dev)
    audit = base_model.parameter_audit()
    resolved_family = str(
        audit.get("family")
        or getattr(getattr(base_model, "config", None), "family", None)
        or ("holographic_brain" if not isinstance(base_model, MMARRSTitan) else "mmarrs_titan")
    )

    featurizer: MissingnessResilientFeaturizer | None = (
        MissingnessResilientFeaturizer().to(dev) if use_missingness_featurizer else None
    )

    # Aplicar muP ANTES de clonar EMA / compilar / construir optimizador (conservando WD real!)
    no_decay_param_ids: set[int] = (
        _collect_no_decay_param_ids(base_model) if use_cautious_wd else set()
    )
    mup_groups: list[dict[str, Any]] | None = None
    if use_mup:
        mup_groups = apply_mup_init_and_param_groups(
            base_model,
            base_width=int(mup_base_width),
            width=int(hidden_dim),
            base_lr=float(lr),
            weight_decay=float(weight_decay),
            apply_init=True,
            reinit_weights=False,
        )

    # Pre-entrenamiento SSL estrictamente causal sobre `train_s` ANTES de épocas supervisadas
    ssl_pretrain_losses: list[float] = []
    if ssl_pretrain_epochs > 0:
        ssl_pretrainer = CausalSSLPretrainer(hidden_dim=int(hidden_dim)).to(dev)
        ssl_opt = torch.optim.AdamW(
            list(base_model.parameters()) + list(ssl_pretrainer.parameters()),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        ssl_gen = torch.Generator(device=dev).manual_seed(seed + 1701)
        base_model.train()
        for _ssl_ep in range(int(ssl_pretrain_epochs)):
            reset_episodic_buffers(base_model)
            ssl_state: dict[str, object] | None = None
            ssl_acc = torch.zeros((), device=dev, dtype=torch.float32)
            ssl_cnt = 0
            for raw_b in iter_sessions_h2d(train_s, dev, use_pinned=rt_cfg.use_pinned_prefetch):
                b_ssl = featurizer.featurize_batch(raw_b) if featurizer is not None else raw_b
                ssl_res = ssl_pretrainer.pretrain_step(
                    base_model,
                    ssl_opt,
                    b_ssl,
                    state=ssl_state,
                    max_grad_norm=2.0,
                    generator=ssl_gen,
                )
                _update_spectral_buffers_post_step(base_model)
                ssl_acc = ssl_acc + ssl_res["total"].detach().float()
                ssl_cnt += 1
                next_st = ssl_res.get("state")
                detach_fn = getattr(base_model, "detach_state", None)
                ssl_state = (
                    detach_fn(next_st)
                    if callable(detach_fn) and isinstance(next_st, dict)
                    else None
                )
            ssl_pretrain_losses.append(float(ssl_acc.cpu().item()) / max(ssl_cnt, 1))
        reset_episodic_buffers(base_model)

    opt_kind = str(optimizer_type).strip().lower()
    is_schedule_free = opt_kind == "schedule_free_adamw"
    use_ema_eval = not (is_schedule_free and sf_disable_ema)
    ema_model: torch.nn.Module | None = None
    if use_ema_eval:
        ema_model = copy.deepcopy(base_model).eval()
        reset_episodic_buffers(ema_model)
    eval_model: torch.nn.Module = ema_model if ema_model is not None else base_model
    model = maybe_compile_model(base_model, rt_cfg)

    cfg_obj = getattr(base_model, "config", None)
    noic_flag = bool(getattr(cfg_obj, "noic_pinball_twin", False))
    use_fused = bool(getattr(cfg_obj, "use_fused_blackwell_kernels", True))
    use_gauss_targets = bool(getattr(cfg_obj, "use_gaussian_rank_targets", False))
    w_stab_down = float(getattr(cfg_obj, "w_stab_downside", 0.0))
    n_horizons = len(getattr(cfg_obj, "horizons", DEFAULT_HORIZONS))
    loss_kwargs: dict[str, Any] = dict(
        noic_pinball_twin=noic_flag,
        use_fused_blackwell_kernels=use_fused,
        use_gaussian_rank_targets=use_gauss_targets,
        w_stab_downside=w_stab_down,
        w_listnet_tail=float(w_listnet_tail),
        w_latent_cov=float(w_latent_cov),
        w_supcon=float(w_supcon),
        w_churn=float(w_churn),
    )
    if w_crps is not None:
        loss_kwargs["w_crps"] = float(w_crps)
    loss_fn = MMARRSMultiTaskLoss(**loss_kwargs).to(dev)
    from .autotuner_sota import OptimalFitConfig, OptimalFitController

    has_bptt_state = (
        getattr(base_model, "neuromod_memory", None) is not None
        or getattr(base_model, "three_factor_plasticity", None) is not None
    )
    eff_bptt = max(1, int(bptt_steps)) if has_bptt_state else 1
    n_train = len(train_s)
    steps_per_epoch = max(1, math.ceil(n_train / eff_bptt))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, total_steps // 10)
    eff_sf_warmup = (
        max(1, total_steps // 20)
        if (is_schedule_free and sf_warmup_steps is None)
        else (int(sf_warmup_steps) if sf_warmup_steps is not None else 0)
    )
    opt = _build_train_optimizer(
        base_model,
        opt_kind=opt_kind,
        lr=float(lr),
        weight_decay=float(weight_decay),
        rt_cfg=rt_cfg,
        dev=dev,
        eff_sf_warmup=eff_sf_warmup,
        mup_groups=mup_groups,
        use_cautious_wd=bool(use_cautious_wd),
    )
    use_sched = not is_schedule_free

    def _lr_mult(step_i: int) -> float:
        if step_i < warmup_steps:
            return float(step_i + 1) / float(warmup_steps)
        prog = float(step_i - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_mult) if use_sched else None
    drop_gen = torch.Generator(device=dev)
    drop_gen.manual_seed(seed + 999)

    curriculum: CausalCurriculumScheduler | None = (
        CausalCurriculumScheduler(
            total_epochs=max(1, int(epochs)),
            noise_start=float(input_noise_std) if input_noise_std > 0.0 else 0.05,
            noise_end=0.0,
        )
        if curriculum_enabled
        else None
    )
    use_struct_aug = (mod_drop_prob > 0.0) or (input_noise_std > 0.0) or curriculum_enabled
    eff_aug_drop = float(mod_drop_prob) if mod_drop_prob > 0.0 else float(modality_dropout)
    struct_aug: StructuredModalityNoiseAugmenter | None = (
        StructuredModalityNoiseAugmenter(
            modality_dropout=eff_aug_drop,
            session_level_prob=0.5,
            noise_std=float(input_noise_std),
        )
        if use_struct_aug
        else None
    )

    use_placebo = bool(
        getattr(getattr(base_model, "config", None), "moe_mode", "") == "shuffled_context"
    )
    fit_controller = (
        OptimalFitController(fit_cfg or OptimalFitConfig()) if use_optimal_fit else None
    )

    tel_dir = telemetry_dir if telemetry_dir is not None else (
        Path(checkpoint_dir) / "telemetry" if checkpoint_dir is not None else None
    )
    telemetry: ResearchTelemetryMonitor | None = None
    if tel_dir is not None:
        telemetry = ResearchTelemetryMonitor(
            base_model,
            TelemetryConfig(
                log_dir=str(tel_dir),
                log_every_steps=max(1, min(10, steps_per_epoch)),
                histogram_every_steps=max(2, steps_per_epoch * 2),
                enable_tensorboard=False,
                enable_jsonl=True,
                compress_jsonl=True,
                jsonl_filename="telemetry.jsonl.gz",
            ),
            optimizer=opt,
        )

    t0 = time.perf_counter()
    train_losses: list[float] = []
    train_ics: list[float] = []
    fit_decisions: list[dict[str, Any]] = []
    global_step = 0
    ema_step = 0
    geometry_report: dict[str, Any] | None = None

    try:
        for ep in range(epochs):
            model.train()
            if struct_aug is not None:
                struct_aug.train()
            if hasattr(opt, "train"):
                opt.train()
            reset_episodic_buffers(base_model)
            loss_fn.reset_stab_ema()
            state = None
            w_prev: torch.Tensor | None = None
            w_prev_ids: torch.Tensor | None = None
            prev_alpha_u: torch.Tensor | None = None
            prev_alpha_ids: torch.Tensor | None = None
            ep_loss_t = torch.zeros((), device=dev, dtype=torch.float32)
            ep_ic_sum_t = torch.zeros((), device=dev, dtype=torch.float32)
            ep_ic_cnt_t = torch.zeros((), device=dev, dtype=torch.float32)

            ep_noise_std = (
                curriculum.noise_std(ep) if curriculum is not None else float(input_noise_std)
            )
            ep_mod_drop = (
                float(mod_drop_prob) if mod_drop_prob > 0.0 else float(modality_dropout)
            )

            # Placebo v05: permutación determinista cross-session del contexto de entrenamiento
            tr_placebo: list[torch.Tensor | None] | None = None
            if use_placebo and n_train > 1:
                perm_tr = _build_deranged_context_indices(n_train, seed + 505 + ep)
                tr_placebo = [train_s[int(p)].get("high_snr_context") for p in perm_tr]

            chunk_loss: torch.Tensor | None = None
            chunk_count = 0
            last_out: dict[str, object] | None = None
            last_ld: dict[str, torch.Tensor] | None = None

            for s_idx, batch in enumerate(
                iter_sessions_h2d(train_s, dev, use_pinned=rt_cfg.use_pinned_prefetch)
            ):
                if chunk_count == 0 and telemetry is not None:
                    telemetry.begin_step(global_step)

                b_clean = (
                    featurizer.featurize_batch(batch) if featurizer is not None else batch
                )
                if struct_aug is not None:
                    b_in = struct_aug(
                        b_clean,
                        modality_dropout=ep_mod_drop,
                        noise_std=ep_noise_std,
                        generator=drop_gen,
                    )
                else:
                    b_in = _apply_structured_modality_dropout(b_clean, modality_dropout, drop_gen)
                y_ret = b_in["target_returns"]
                b_in["horizon_mask"] = infer_horizon_mask(y_ret, n_horizons)
                if tr_placebo is not None and tr_placebo[s_idx] is not None:
                    b_in["placebo_high_snr_context"] = tr_placebo[s_idx].to(dev)

                cur_ids = b_in.get("asset_ids")
                w_prev_aligned = _align_w_prev_ids(cur_ids, w_prev_ids, w_prev, y_ret.shape[0])
                if w_prev_aligned is not None:
                    b_in["w_prev"] = w_prev_aligned.detach()

                with autocast_context(dev, rt_cfg), sdpa_math_context(rt_cfg):
                    if isinstance(base_model, MMARRSTitan):
                        out = model(b_in, state=state, enable_early_exit=False)
                    else:
                        out = model(b_in, state=state)

                # Pérdidas y reducciones siempre en FP32 maestro (cero degradación en bf16_amp!)
                with torch.amp.autocast(device_type=dev.type, enabled=False):
                    aux = _collect_aux_losses(
                        out, base_model, y_ret, horizon_mask=b_in["horizon_mask"]
                    )
                    raw_steps = out.get("loop_outputs") or out.get("cycle_outputs") or [out]
                    step_outs = (
                        _promote_fp32_outputs(raw_steps)
                        if rt_cfg.precision_mode == "bf16_amp"
                        else raw_steps
                    )
                    en_dict = out.get("energy")
                    en_raw = en_dict.get("total") if isinstance(en_dict, dict) else None
                    en_total = en_raw.float() if isinstance(en_raw, torch.Tensor) else None
                    liq_w = _extract_liquidity_weights(b_in)
                    if curriculum is not None and liq_w is not None:
                        liq_w = curriculum.weight_liquidity(liq_w, ep)
                    fh_raw = out.get("final_hidden")
                    fh_t = fh_raw.float() if isinstance(fh_raw, torch.Tensor) else None
                    ld = loss_fn(
                        step_outputs=step_outs,
                        targets=dict(returns=y_ret.float()),
                        exposures=(
                            b_in["exposures"].float() if b_in.get("exposures") is not None else None
                        ),
                        energy_total=en_total,
                        w_prev=(
                            w_prev_aligned.float() if w_prev_aligned is not None else None
                        ),
                        horizon_mask=b_in["horizon_mask"].float(),
                        liquidity_weights=liq_w,
                        aux_losses=aux if aux else None,
                        final_hidden=fh_t,
                        prev_alpha_u=prev_alpha_u if w_churn > 0.0 else None,
                        cur_asset_ids=cur_ids if w_churn > 0.0 else None,
                        prev_asset_ids=prev_alpha_ids if w_churn > 0.0 else None,
                    )

                step_total = ld["total"]
                chunk_loss = step_total if chunk_loss is None else (chunk_loss + step_total)
                chunk_count += 1
                ep_loss_t = ep_loss_t + step_total.detach().float()
                rho_t = ld.get("rho_ic")
                if isinstance(rho_t, torch.Tensor):
                    rho_f = rho_t.detach().float().reshape(())
                    rho_fin = torch.isfinite(rho_f)
                    ep_ic_sum_t = ep_ic_sum_t + torch.where(
                        rho_fin, rho_f, torch.zeros_like(rho_f)
                    )
                    ep_ic_cnt_t = ep_ic_cnt_t + rho_fin.to(dtype=torch.float32)
                last_out = out
                last_ld = ld

                w_prev = ld["portfolio_weights"]
                w_prev_ids = cur_ids.detach() if cur_ids is not None else None
                if w_churn > 0.0:
                    cur_u_src = out.get("alpha_u")
                    if not isinstance(cur_u_src, torch.Tensor):
                        cur_u_src = out.get("alpha")
                    prev_alpha_u = (
                        cur_u_src.detach().float()
                        if isinstance(cur_u_src, torch.Tensor)
                        else None
                    )
                    prev_alpha_ids = cur_ids.detach() if cur_ids is not None else None

                is_boundary = (chunk_count >= eff_bptt) or (s_idx + 1 == n_train)

                if is_boundary:
                    assert chunk_loss is not None
                    opt.zero_grad(set_to_none=True)
                    (chunk_loss / float(chunk_count)).backward()
                    if telemetry is not None:
                        telemetry.before_optimizer_step(global_step)
                    if use_agc:
                        adaptive_gradient_clip_(base_model, clip_factor=float(agc_clip_factor))
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(), max_norm=2.0, error_if_nonfinite=False
                    )
                    if telemetry is not None:
                        telemetry.after_grad_clip(global_step)
                    if use_cautious_wd and float(weight_decay) > 0.0:
                        _apply_cautious_wd_before_step(
                            opt, no_decay_param_ids, default_lr=float(lr)
                        )
                    opt.step()
                    _update_spectral_buffers_post_step(base_model)
                    if sched is not None:
                        sched.step()
                    if ema_model is not None:
                        eff_ema_decay = _effective_ema_decay(
                            ema_decay, ema_step, bias_correct=ema_bias_correct
                        )
                        _update_ema(ema_model, base_model, eff_ema_decay)
                        ema_step += 1

                    # Avanzar estado causal tras backward() para evitar conflictos in-place
                    out_state_det = (
                        base_model.detach_state(out["state"])
                        if hasattr(base_model, "detach_state")
                        else out["state"]
                    )
                    if isinstance(base_model, MMARRSTitan):
                        state = base_model.step_matured(
                            out_state_det,
                            matured_batch=b_in,
                            matured_targets=y_ret,
                            matured_preds=out["pred_return"][:, 0].detach(),
                            keep_grad=False,
                        )
                    else:
                        state = base_model.step_matured(
                            _promote_fp32_outputs(out_state_det),
                            matured_targets=y_ret.float(),
                            matured_preds=out["pred_return"][:, 0].detach().float(),
                            asset_ids=b_in.get("asset_ids"),
                        )
                    if telemetry is not None:
                        telemetry.end_step(
                            global_step,
                            optimizer=opt,
                            loss_dict=last_ld,
                            model_output=last_out,
                            titans_update_diag=(
                                state.get("last_titans_diag") if isinstance(state, dict) else None
                            ),
                            n_sessions=chunk_count,
                        )
                    state = (
                        base_model.detach_state(state)
                        if hasattr(base_model, "detach_state")
                        else state
                    )
                    chunk_loss = None
                    chunk_count = 0
                    global_step += 1
                else:
                    if isinstance(base_model, MMARRSTitan):
                        state = base_model.step_matured(
                            out["state"],
                            matured_batch=b_in,
                            matured_targets=y_ret,
                            matured_preds=out["pred_return"][:, 0].detach(),
                            keep_grad=True,
                        )
                    else:
                        state = base_model.step_matured(
                            _promote_fp32_outputs(out["state"]),
                            matured_targets=y_ret.float(),
                            matured_preds=out["pred_return"][:, 0].detach().float(),
                            asset_ids=b_in.get("asset_ids"),
                        )

            # 1 sola sync D2H por época para pérdida e IC medio de entrenamiento (0 syncs extra!)
            ep_loss_sum, ep_ic_sum, ep_ic_cnt = (
                torch.stack((ep_loss_t, ep_ic_sum_t, ep_ic_cnt_t)).detach().cpu().tolist()
            )
            ep_loss_val = float(ep_loss_sum) / max(n_train, 1)
            ep_train_ic = float(ep_ic_sum) / float(ep_ic_cnt) if ep_ic_cnt > 0.0 else 0.0
            train_losses.append(ep_loss_val)
            train_ics.append(ep_train_ic)

            # Validación por época para OptimalFitController (Top-3 Model Soup + parada suavizada)
            if fit_controller is not None:
                if hasattr(opt, "eval"):
                    opt.eval()
                val_ep_rep, _, _, _ = _evaluate_model_causal(
                    eval_model,
                    val_s,
                    dev,
                    rt_cfg,
                    market=market,
                    n_trials=n_trials,
                    use_placebo_shuffle=use_placebo,
                    seed=seed,
                    n_horizons=n_horizons,
                    featurizer=featurizer,
                )
                val_ic_ep = float(val_ep_rep["dim2_ranking"]["rank_ic_mean"])
                dim4_ep = val_ep_rep.get("dim4_uncertainty", {})
                crps_raw = dim4_ep.get("crps") if isinstance(dim4_ep, dict) else None
                val_crps_ep = float(crps_raw) if crps_raw is not None else float("nan")
                dec = fit_controller.step(
                    epoch=ep,
                    train_ic=ep_train_ic if math.isfinite(ep_train_ic) else 0.0,
                    val_ic=val_ic_ep if math.isfinite(val_ic_ep) else 0.0,
                    val_crps=val_crps_ep if math.isfinite(val_crps_ep) else None,
                    train_loss=ep_loss_val,
                    model=eval_model,
                    optimizer=opt,
                )
                fit_decisions.append(dec.to_dict())
                if dec.should_stop:
                    break
                if hasattr(opt, "train"):
                    opt.train()
                model.train()

        # Colocar ScheduleFree en `x_t` ANTES de restaurar Top-K Soup y de la evaluación final
        if hasattr(opt, "eval"):
            opt.eval()

        # Restaurar Top-3 Model Soup si OptimalFitController registró checkpoints válidos
        if fit_controller is not None and fit_controller.top_checkpoint_epochs:
            fit_controller.restore_top_k_soup(eval_model)

        # Evaluación final estrictamente causal sobre `val_s` (cero `vol_pred`, ADV real, P0-5)
        val_report, val_preds, val_targets, val_bounds = _evaluate_model_causal(
            eval_model,
            val_s,
            dev,
            rt_cfg,
            market=market,
            n_trials=n_trials,
            use_placebo_shuffle=use_placebo,
            seed=seed,
            n_horizons=n_horizons,
            featurizer=featurizer,
        )

        # Sonda de geometría de pérdida y representaciones (0 mutaciones de pesos ni buffers!)
        if collect_geometry_probe:
            saved_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
            saved_loss_state = {k: v.detach().clone() for k, v in loss_fn.state_dict().items()}
            try:
                probe_iter = iter_sessions_h2d(val_s, dev, use_pinned=False)
                probe_raw = next(probe_iter)
                probe_batch = (
                    featurizer.featurize_batch(probe_raw)
                    if featurizer is not None
                    else dict(probe_raw)
                )
                probe_y = probe_batch["target_returns"]
                probe_batch.setdefault("horizon_mask", infer_horizon_mask(probe_y, n_horizons))

                def _probe_loss_closure() -> torch.Tensor:
                    eval_model.eval()
                    if isinstance(eval_model, MMARRSTitan):
                        p_out = eval_model.forward_session(
                            probe_batch, state=None, enable_early_exit=False
                        )
                    else:
                        p_out = eval_model.forward_session(probe_batch, state=None)
                    raw_s = p_out.get("loop_outputs") or p_out.get("cycle_outputs") or [p_out]
                    ld_p = loss_fn(
                        step_outputs=raw_s,
                        targets=dict(returns=probe_y.float()),
                        exposures=(
                            probe_batch["exposures"].float()
                            if probe_batch.get("exposures") is not None
                            else None
                        ),
                        horizon_mask=probe_batch["horizon_mask"].float(),
                    )
                    return ld_p["total"]

                probe = DeepLearningGeometryProbe(
                    hessian_cfg=HessianConfig(
                        top_k=3, lanczos_steps=8, power_iters=8, hutchinson_probes=4
                    )
                )
                rep_obj = probe.probe_model(
                    eval_model,
                    probe_batch,
                    loss_closure=_probe_loss_closure,
                    family=resolved_family,
                    variant=variation,
                    epoch=len(train_losses),
                    step=global_step,
                    run_hessian=True,
                    eval_mode=True,
                )
                geometry_report = rep_obj.to_dict()
            finally:
                eval_model.zero_grad(set_to_none=True)
                eval_model.load_state_dict(saved_state, strict=True)
                loss_fn.load_state_dict(saved_loss_state, strict=True)
    finally:
        if telemetry is not None:
            telemetry.close()

    ckpt_path = None
    if checkpoint_dir is not None:
        cdir = Path(checkpoint_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        fp = str(audit["fingerprint"])[:16]
        ckpt_path = cdir / f"mmarrs_{variation}_{market}_{fp}_seed{seed}.pt"
        cfg_id = base_model.config.identity() if hasattr(base_model, "config") else {}
        torch.save(
            dict(
                variation=variation,
                family=resolved_family,
                market=market,
                fingerprint=audit["fingerprint"],
                config=cfg_id,
                split_config=asdict(s_cfg),
                ema_enabled=bool(ema_model is not None),
                ema_state_dict=eval_model.state_dict(),
                base_state_dict=base_model.state_dict(),
                val_report=val_report,
                fit_decisions=fit_decisions,
            ),
            ckpt_path,
        )

    result_dict: dict[str, object] = dict(
        variation=variation,
        family=resolved_family,
        market=market,
        parameter_audit=audit,
        hardware=hw_info,
        split_summary=dict(
            mode=s_cfg.mode,
            n_train=len(train_s),
            n_val=len(val_s),
            purge_sessions=s_cfg.purge_sessions,
            embargo_sessions=s_cfg.embargo_sessions,
            n_sealed_withheld=parts["n_sealed_withheld"],
            sealed_test_accessible=bool(s_cfg.allow_sealed_test),
        ),
        train_losses=train_losses,
        train_ics=train_ics,
        training_dynamics=dict(
            optimizer_type=opt_kind,
            sf_disable_ema=bool(sf_disable_ema),
            sf_warmup_steps=int(eff_sf_warmup),
            ema_enabled=bool(ema_model is not None),
            ema_bias_correct=bool(ema_bias_correct),
            ema_updates=int(ema_step),
            use_mup=bool(use_mup),
            mup_base_width=int(mup_base_width),
            use_agc=bool(use_agc),
            agc_clip_factor=float(agc_clip_factor),
            use_cautious_wd=bool(use_cautious_wd),
            use_missingness_featurizer=bool(use_missingness_featurizer),
            curriculum_enabled=bool(curriculum_enabled),
            ssl_pretrain_epochs=int(ssl_pretrain_epochs),
            ssl_pretrain_losses=ssl_pretrain_losses,
        ),
        fit_decisions=fit_decisions,
        top_soup_epochs=list(fit_controller.top_checkpoint_epochs) if fit_controller else [],
        val_report=val_report,
        val_predictions=val_preds,
        val_targets=val_targets,
        val_bounds=val_bounds,
        checkpoint_path=str(ckpt_path) if ckpt_path else None,
        telemetry_path=str(telemetry.jsonl_path) if telemetry is not None else None,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )
    if geometry_report is not None:
        result_dict["geometry_report"] = geometry_report
    return result_dict


def train_multiseed_rank_ensemble(
    sessions: Any,
    session_years: Sequence[int] | None = None,
    *,
    variation: str = "v13_mmarrs_titan_full",
    market: str = "US",
    family: str | None = None,
    seeds: Sequence[int] = (42, 43, 44),
    n_trials: int | None = None,
    telemetry_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    **train_kwargs: Any,
) -> dict[str, object]:
    """Entrena un ensamble determinista por rangos de `3+` semillas (`+0.002..+0.004` IC)."""
    if len(seeds) == 0:
        raise ValueError("seeds debe contener al menos una semilla")
    eff_trials = max(int(n_trials or 1), len(seeds))

    seed_results: list[dict[str, object]] = []
    seed_preds: list[np.ndarray] = []
    y_ref: np.ndarray | None = None
    bounds_ref: list[tuple[int, int]] = []

    for s in seeds:
        s_tel = Path(telemetry_dir) / f"seed_{s}" if telemetry_dir is not None else None
        s_ckpt = Path(checkpoint_dir) / f"seed_{s}" if checkpoint_dir is not None else None
        res = train_full_history_variation(
            sessions,
            session_years,
            variation=variation,
            market=market,
            family=family,
            seed=int(s),
            n_trials=eff_trials,
            telemetry_dir=s_tel,
            checkpoint_dir=s_ckpt,
            **train_kwargs,
        )
        seed_results.append(res)
        seed_preds.append(np.asarray(res["val_predictions"], dtype=np.float64))
        if y_ref is None:
            y_ref = np.asarray(res["val_targets"], dtype=np.float64)
            bounds_ref = list(res["val_bounds"])  # type: ignore[arg-type]

    from .autotuner_sota import evaluate_rank_ensemble, rank_ensemble_predictions

    assert y_ref is not None
    ens_pred = rank_ensemble_predictions(seed_preds, bounds=bounds_ref, method="rank_zscore")
    ens_summary = evaluate_rank_ensemble(
        seed_preds, y_ref, bounds=bounds_ref, method="rank_zscore"
    )
    ens_val_report = evaluate_granular_six_dimensions(
        y=y_ref,
        pred=ens_pred.astype(np.float64),
        bounds=bounds_ref,
        n_trials=eff_trials,
        market=market,  # type: ignore[arg-type]
    )
    return dict(
        variation=variation,
        market=market,
        seeds=list(seeds),
        seed_results=seed_results,
        ensemble_predictions=ens_pred,
        ensemble_summary=ens_summary,
        ensemble_val_report=ens_val_report,
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI canónico alineado 1:1 con `CANONICAL_TRAIN_CLI_FLAGS` (P0-7)."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento completo 2000-2023 MMARRS-TITAN en RTX PRO 6000 (sm_120)"
    )
    parser.add_argument("--variation", default="v13_mmarrs_titan_full")
    parser.add_argument("--market", choices=VALID_MARKETS, default="US")
    parser.add_argument("--family", choices=MMARRS_FAMILIES, default="mmarrs_titan")
    parser.add_argument("--out-dir", dest="out_dir", default=None)
    parser.add_argument("--precision", choices=PRECISION_MODES, default="fp32_strict")
    parser.add_argument("--vram-fraction", dest="vram_fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-mps-plan", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI RTX PRO 6000
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if args.print_mps_plan:
        plan = blackwell_mps_launch_plan(
            MMARRS_VARIATIONS, vram_gb=96.0, workers=args.workers, precision_mode=args.precision
        )
        print(json.dumps(plan, indent=2))
        return
    rt_cfg = BlackwellRuntimeConfig(
        precision_mode=args.precision,
        vram_fraction_per_worker=args.vram_fraction,
    )
    hw = configure_blackwell_runtime(rt_cfg)
    summary = dict(
        variation=args.variation,
        market=args.market,
        family=args.family,
        out_dir=args.out_dir,
        precision=args.precision,
        vram_fraction=args.vram_fraction,
        epochs=args.epochs,
        seed=args.seed,
        hardware=hw,
    )
    if args.out_dir:
        out_p = Path(args.out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "launch_manifest.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
