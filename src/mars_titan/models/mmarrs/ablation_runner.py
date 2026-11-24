"""Arnés sistemático de ablación científica para las variaciones emparejadas de `MMARRS-TITAN`.

Entrena y evalúa todas las variaciones seleccionadas bajo condiciones estrictamente idénticas:
  - Misma semilla determinista y división cronológica estricta con purga + embargo multi-horizonte
    (`purge >= max_h`, `embargo >= max(5, ceil(0.2*max_h))`, P0-4).
  - Mantiene `base_model` separado de `maybe_compile_model` (P2-16) y soporta `asset_ids` (P0-1),
    BPTT truncado (P0-2), `ZeroCopySessionStore.iter_pinned_h2d` (P0-7), `sdpa_math_context`,
    placebo temporal cross-session para `v05` (P1-3) y `build_sota_optimizer` (`wd=1e-2`).
  - Jamás pasa `vol_pred` del modelo al evaluador: usa volatilidad realizada, ADV USD real y
    `n_trials` acumulado para DSR exacto (P0-5).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from mars_titan.evaluation.granular_evaluator import evaluate_granular_six_dimensions

from .config import (
    DEFAULT_HORIZONS,
    MMARRS_VARIATIONS,
    VALID_MARKETS,
    required_purge_embargo,
)
from .granular_heads import MMARRSMultiTaskLoss, infer_horizon_mask
from .hardware_blackwell import (
    BlackwellRuntimeConfig,
    autocast_context,
    configure_blackwell_runtime,
    iter_sessions_h2d,
    maybe_compile_model,
    sdpa_math_context,
)
from .model import MMARRSTitan, build_family_model


def _infer_dims(sample: dict[str, torch.Tensor]) -> dict[str, int]:
    """Infiere automáticamente dimensiones multimodales desde la primera sesión."""
    return dict(
        prices_dim=int(sample["prices"].shape[-1]),
        fundamentals_dim=int(sample["fundamentals"].shape[-1]) if "fundamentals" in sample else 8,
        macro_dim=int(sample["macro"].shape[-1]) if "macro" in sample else 6,
        news_dim=int(sample["news"].shape[-1]) if "news" in sample else 8,
        charts_dim=int(sample["charts"].shape[-1]) if "charts" in sample else 8,
        physics_dim=int(sample["physics_v4"].shape[-1]) if "physics_v4" in sample else 12,
        context_dim=int(sample["high_snr_context"].shape[-1])
        if "high_snr_context" in sample
        else 10,
        exposures_dim=int(sample["exposures"].shape[-1]) if "exposures" in sample else 4,
        rmt_dim=int(sample["rmt_loadings"].shape[-1]) if "rmt_loadings" in sample else 4,
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


def _is_episodic_buffer(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in _EPISODIC_BUFFER_SUFFIXES)


@torch.no_grad()
def _reset_episodic_buffers(model: torch.nn.Module) -> None:
    """Reinicia buffers episódicos HRR + `HippocampalSharpWaveReplay` (evita fuga train->val)."""
    for b_name, buf in model.named_buffers():
        if _is_episodic_buffer(b_name):
            buf.zero_()


def _extract_liquidity_weights(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Extrae pesos de liquidez `[N]` (`liquidity_weights` o `high_snr_context[:, 7]` `liq_pct`)."""
    lw = batch.get("liquidity_weights")
    if isinstance(lw, torch.Tensor):
        w = lw.detach().float().reshape(-1)
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w)).clamp_min(1e-4)
        return w / w.mean().clamp_min(1e-6)
    ctx = batch.get("high_snr_context")
    if isinstance(ctx, torch.Tensor) and ctx.ndim == 2 and ctx.shape[-1] > 7:
        liq_pct = ctx[:, 7].detach().float()
        liq_clean = torch.where(torch.isfinite(liq_pct), liq_pct, torch.full_like(liq_pct, 0.5))
        w = 0.25 + 1.5 * liq_clean.clamp(0.0, 1.0)
        return w / w.mean().clamp_min(1e-6)
    return None


@torch.no_grad()
def _update_spectral_buffers_post_step(model: torch.nn.Module) -> None:
    """Invoca `update_spectral_buffers()` tras `optimizer.step()` en cualquier modelo/submódulo."""
    top_fn = getattr(model, "update_spectral_buffers", None)
    if callable(top_fn):
        top_fn()
        return
    for mod in model.modules():
        if mod is model:
            continue
        fn = getattr(mod, "update_spectral_buffers", None)
        if callable(fn):
            fn()


def _promote_fp32_outputs(obj: Any) -> Any:
    """Promueve recursivamente tensores flotantes a FP32 maestro antes de pérdidas y estados."""
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.is_floating_point() and obj.dtype != torch.float32 else obj
    if isinstance(obj, dict):
        return {k: _promote_fp32_outputs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_promote_fp32_outputs(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_promote_fp32_outputs(v) for v in obj)
    return obj


def _compute_skew_t_aux_loss(
    skew_head: Any,
    skew_t: dict[str, Any],
    y_ret: torch.Tensor,
    horizon_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calcula `0.05 * skew_head.loss` recortando a `n_h` activos y aplicando `horizon_mask`."""
    st_fp32 = _promote_fp32_outputs(skew_t)
    loc = st_fp32.get("loc")
    y_f = y_ret.float()
    y_2d = y_f.unsqueeze(-1) if y_f.ndim == 1 else y_f
    if isinstance(loc, torch.Tensor) and loc.ndim >= 2:
        h_model = int(loc.shape[1])
        n_h = min(int(y_2d.shape[-1]), h_model)
        st_sliced = {
            k: (
                v[:, :n_h]
                if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == h_model
                else v
            )
            for k, v in st_fp32.items()
        }
        y_sub = y_2d[:, :n_h]
        st_mask: torch.Tensor | None = None
        if horizon_mask is not None:
            hm_f = horizon_mask.to(device=loc.device, dtype=torch.float32)
            st_mask = hm_f[:n_h].unsqueeze(0) if hm_f.ndim == 1 else hm_f[:, :n_h]
        st_loss = skew_head.loss(st_sliced, y_sub, mask=st_mask)
    else:
        st_loss = skew_head.loss(st_fp32, y_f)
    return 0.05 * st_loss["total"].float()


def _collect_aux_losses(
    out: dict[str, object],
    model: torch.nn.Module,
    y_ret: torch.Tensor,
    *,
    horizon_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Recopila pérdidas auxiliares en FP32 respetando `horizon_mask` (P0-4)."""
    aux: dict[str, torch.Tensor] = {}
    cfg = getattr(model, "config", None)
    if isinstance(out.get("ssl_loss"), torch.Tensor):
        aux["ssl"] = out["ssl_loss"].float()  # type: ignore[union-attr]
    if isinstance(out.get("weighted_ponder_loss"), torch.Tensor):
        aux["ponder"] = out["weighted_ponder_loss"].float()  # type: ignore[union-attr]
    elif isinstance(out.get("ponder_kl_loss"), torch.Tensor):
        w_p = float(getattr(cfg, "cognitive_ponder_weight", 0.01))
        if w_p > 0.0:
            aux["ponder"] = w_p * out["ponder_kl_loss"].float()  # type: ignore[union-attr]
    if isinstance(out.get("cf_fragility_loss"), torch.Tensor):
        w_cf = float(getattr(cfg, "w_cf_fragility", 0.05))
        if w_cf > 0.0:
            aux["cf_fragility"] = w_cf * out["cf_fragility_loss"].float()  # type: ignore[union-attr]
    if isinstance(out.get("tool_aux_loss"), torch.Tensor):
        w_ta = float(getattr(cfg, "w_tool_aux", 0.01))
        if w_ta > 0.0:
            aux["tool_aux"] = w_ta * out["tool_aux_loss"].float()  # type: ignore[union-attr]
    skew_t = out.get("skew_t")
    skew_head = getattr(model, "skew_t_head", None)
    if isinstance(skew_t, dict) and skew_head is not None:
        aux["skew_t_nll"] = _compute_skew_t_aux_loss(
            skew_head, skew_t, y_ret, horizon_mask=horizon_mask
        )
    replay_fn = getattr(model, "compute_replay_loss", None)
    if callable(replay_fn):
        aux["swr_replay"] = 0.1 * replay_fn().float()
    return aux


def _deranged_perm(n: int, seed: int) -> np.ndarray:
    """Permutación determinista estrictamente sin puntos fijos (`perm[i] != i` para todo `n>=2`)."""
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for idx in range(n):
        if perm[idx] == idx:
            j = (idx + 1) % n
            perm[idx], perm[j] = perm[j], perm[idx]
    return perm.astype(np.int64)


def run_variation_ablation(
    sessions: Any,
    *,
    variations: Sequence[str] = MMARRS_VARIATIONS,
    train_end: int,
    epochs: int = 4,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    optimizer_type: str = "adamw_fused",
    bptt_steps: int = 4,
    hidden_dim: int = 32,
    seed: int = 42,
    device: str = "cpu",
    market: str = "US",
    active_horizons: Sequence[int] | None = None,
    purge_sessions: int | None = None,
    embargo_sessions: int | None = None,
    allow_short_synthetic: bool = False,
    n_trials: int | None = None,
    runtime_cfg: BlackwellRuntimeConfig | None = None,
) -> dict[str, object]:
    """Ejecuta el estudio de ablación controlado sobre sesiones cronológicas o `ZeroCopyStore`."""
    if market not in VALID_MARKETS:
        raise ValueError(
            f"FM-8: market inválido {market!r}; solo se permiten mercados separados {VALID_MARKETS}"
        )
    if train_end < 2 or train_end >= len(sessions):
        raise ValueError("train_end debe dividir sesiones en train (>=2) y eval (>=1)")

    sample0 = sessions[0]
    y0 = sample0["target_returns"]
    n_h_data = 1 if y0.ndim == 1 else int(y0.shape[-1])
    eff_horizons = (
        tuple(int(h) for h in active_horizons)
        if active_horizons is not None
        else DEFAULT_HORIZONS[: min(n_h_data, len(DEFAULT_HORIZONS))]
    )
    min_purge, min_embargo = required_purge_embargo(eff_horizons)
    eff_purge = min_purge if purge_sessions is None else int(purge_sessions)
    eff_embargo = min_embargo if embargo_sessions is None else int(embargo_sessions)
    if eff_purge < 0 or eff_embargo < 0:
        raise ValueError("purge_sessions y embargo_sessions deben ser >= 0")
    if not allow_short_synthetic:
        if eff_purge < min_purge:
            raise ValueError(
                f"P0-4 fuga multi-horizonte: purge_sessions={eff_purge} < "
                f"max(active_horizons)={min_purge} (use allow_short_synthetic=True solo en tests)"
            )
        if eff_embargo < min_embargo:
            raise ValueError(
                f"P0-4 embargo insuficiente: embargo_sessions={eff_embargo} < {min_embargo}"
            )

    rt_cfg = runtime_cfg or BlackwellRuntimeConfig(precision_mode="fp32_strict")
    hw_info = configure_blackwell_runtime(rt_cfg, device=device)
    dev = torch.device(str(hw_info["device"]))
    dims = _infer_dims(sample0)

    train_stop = max(0, train_end - eff_purge)
    eval_start = min(len(sessions), train_end + eff_embargo)
    if hasattr(sessions, "iter_pinned_h2d"):
        train_sessions: Any = sessions[:train_stop]
        eval_sessions: Any = sessions[eval_start:]
    else:
        train_sessions = list(sessions[:train_stop])
        eval_sessions = list(sessions[eval_start:])

    if len(train_sessions) < 2 or len(eval_sessions) < 1:
        raise ValueError(
            f"Partición insuficiente tras purga={eff_purge} y embargo={eff_embargo}: "
            f"train={len(train_sessions)}, eval={len(eval_sessions)}"
        )

    eff_n_trials = max(int(n_trials or 1), len(variations))
    results: dict[str, dict[str, object]] = {}

    for name in variations:
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # P2-16: mantener `base_model` separado de `maybe_compile_model`
        base_model = build_family_model(
            name,
            market=market,
            hidden_dim=hidden_dim,
            seed=seed,
            bptt_steps=max(1, int(bptt_steps)),
            weight_decay=float(weight_decay),
            purge_sessions=eff_purge,
            embargo_sessions=eff_embargo,
            allow_short_synthetic=allow_short_synthetic,
            **dims,
        ).to(dev)
        audit = base_model.parameter_audit()
        model = maybe_compile_model(base_model, rt_cfg)

        cfg_obj = getattr(base_model, "config", None)
        noic_flag = bool(getattr(cfg_obj, "noic_pinball_twin", False))
        use_fused = bool(getattr(cfg_obj, "use_fused_blackwell_kernels", True))
        use_gauss_targets = bool(getattr(cfg_obj, "use_gaussian_rank_targets", False))
        w_stab_down = float(getattr(cfg_obj, "w_stab_downside", 0.0))
        n_horizons = len(getattr(cfg_obj, "horizons", DEFAULT_HORIZONS))
        loss_fn = MMARRSMultiTaskLoss(
            noic_pinball_twin=noic_flag,
            use_fused_blackwell_kernels=use_fused,
            use_gaussian_rank_targets=use_gauss_targets,
            w_stab_downside=w_stab_down,
        ).to(dev)
        from .autotuner_sota import build_sota_optimizer

        opt = build_sota_optimizer(
            base_model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            router_lr_scale=rt_cfg.router_lr_scale,
            cfg=rt_cfg,
        )

        has_bptt_state = (
            getattr(base_model, "neuromod_memory", None) is not None
            or getattr(base_model, "three_factor_plasticity", None) is not None
        )
        eff_bptt = max(1, int(bptt_steps)) if has_bptt_state else 1
        use_placebo = bool(getattr(cfg_obj, "moe_mode", "") == "shuffled_context")

        # --- FASE 1: Entrenamiento cronológico causal con BPTT truncado ---
        model.train()
        if hasattr(opt, "train"):
            opt.train()
        n_tr = len(train_sessions)
        for ep in range(epochs):
            _reset_episodic_buffers(base_model)
            loss_fn.reset_stab_ema()
            state = None
            w_prev: torch.Tensor | None = None
            w_prev_ids: torch.Tensor | None = None
            tr_placebo: list[torch.Tensor | None] | None = None
            if use_placebo and n_tr > 1:
                perm_tr = _deranged_perm(n_tr, seed + 505 + ep)
                tr_placebo = [train_sessions[int(p)].get("high_snr_context") for p in perm_tr]

            chunk_loss: torch.Tensor | None = None
            chunk_cnt = 0
            for s_idx, batch in enumerate(
                iter_sessions_h2d(train_sessions, dev, use_pinned=rt_cfg.use_pinned_prefetch)
            ):
                y_ret = batch["target_returns"]
                b_in = dict(batch)
                b_in["horizon_mask"] = infer_horizon_mask(y_ret, n_horizons)
                if tr_placebo is not None and tr_placebo[s_idx] is not None:
                    b_in["placebo_high_snr_context"] = tr_placebo[s_idx].to(dev)

                cur_ids = b_in.get("asset_ids")
                if w_prev is not None and cur_ids is not None and w_prev_ids is not None:
                    w_prev_in: torch.Tensor | None = MMARRSTitan._align_vector_by_asset_ids(
                        cur_ids, w_prev_ids, w_prev
                    )
                elif w_prev is not None and w_prev.shape[0] == y_ret.shape[0]:
                    w_prev_in = w_prev
                else:
                    w_prev_in = None
                if w_prev_in is not None:
                    b_in["w_prev"] = w_prev_in.detach()

                with autocast_context(dev, rt_cfg), sdpa_math_context(rt_cfg):
                    if isinstance(base_model, MMARRSTitan):
                        out = model(b_in, state=state, enable_early_exit=False)
                    else:
                        out = model(b_in, state=state)
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
                    loss_dict = loss_fn(
                        step_outputs=step_outs,
                        targets=dict(returns=y_ret.float()),
                        exposures=(
                            b_in["exposures"].float() if b_in.get("exposures") is not None else None
                        ),
                        energy_total=en_total,
                        w_prev=(w_prev_in.float() if w_prev_in is not None else None),
                        horizon_mask=b_in["horizon_mask"].float(),
                        liquidity_weights=liq_w,
                        aux_losses=aux if aux else None,
                    )

                step_l = loss_dict["total"]
                chunk_loss = step_l if chunk_loss is None else (chunk_loss + step_l)
                chunk_cnt += 1
                w_prev = loss_dict["portfolio_weights"]
                w_prev_ids = cur_ids.detach() if cur_ids is not None else None

                is_boundary = (chunk_cnt >= eff_bptt) or (s_idx + 1 == n_tr)
                if is_boundary:
                    assert chunk_loss is not None
                    opt.zero_grad(set_to_none=True)
                    (chunk_loss / float(chunk_cnt)).backward()
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(), max_norm=2.0, error_if_nonfinite=False
                    )
                    opt.step()
                    _update_spectral_buffers_post_step(base_model)
                    out_st = (
                        base_model.detach_state(out["state"])
                        if hasattr(base_model, "detach_state")
                        else out["state"]
                    )
                    if isinstance(base_model, MMARRSTitan):
                        state = base_model.step_matured(
                            out_st,
                            matured_batch=b_in,
                            matured_targets=y_ret,
                            matured_preds=out["pred_return"][:, 0].detach(),
                            keep_grad=False,
                        )
                    else:
                        state = base_model.step_matured(
                            _promote_fp32_outputs(out_st),
                            matured_targets=y_ret.float(),
                            matured_preds=out["pred_return"][:, 0].detach().float(),
                            asset_ids=b_in.get("asset_ids"),
                        )
                    if hasattr(base_model, "detach_state"):
                        state = base_model.detach_state(state)
                    chunk_loss = None
                    chunk_cnt = 0
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

        # --- FASE 2: Evaluación OOS estrictamente causal (cero `vol_pred`, ADV real, P0-5) ---
        base_model.eval()
        if hasattr(opt, "eval"):
            opt.eval()
        _reset_episodic_buffers(base_model)

        state = None
        preds_all, y_all, q_all, prob_all, exp_all, w_agent_all = [], [], [], [], [], []
        vol_real_all, adv_all, aids_all = [], [], []
        bounds: list[tuple[int, int]] = []
        offset = 0
        k_used_list, de_neg_list, ent_list, load_ratio_list, ecorr_list, act_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        max_spec = 0.0

        ev_placebo: list[torch.Tensor | None] | None = None
        if use_placebo and len(eval_sessions) > 1:
            perm_ev = _deranged_perm(len(eval_sessions), seed + 707)
            ev_placebo = [eval_sessions[int(p)].get("high_snr_context") for p in perm_ev]

        with torch.no_grad(), sdpa_math_context(rt_cfg):
            for s_idx, batch in enumerate(
                iter_sessions_h2d(eval_sessions, dev, use_pinned=rt_cfg.use_pinned_prefetch)
            ):
                y_ret = batch["target_returns"]
                b_ev = dict(batch)
                b_ev["horizon_mask"] = infer_horizon_mask(y_ret, n_horizons)
                if ev_placebo is not None and ev_placebo[s_idx] is not None:
                    b_ev["placebo_high_snr_context"] = ev_placebo[s_idx].to(dev)

                if isinstance(base_model, MMARRSTitan):
                    out = base_model.forward_session(b_ev, state=state, enable_early_exit=True)
                else:
                    out = base_model.forward_session(b_ev, state=state)
                p1 = out["pred_return"][:, 0].detach().float().cpu().numpy()
                y1 = (y_ret[:, 0] if y_ret.ndim == 2 else y_ret).detach().float().cpu().numpy()
                q1 = out["quantiles"][:, 0, :].detach().float().cpu().numpy()
                tbp = out["tb_probs"][:, 0, :].detach().float().cpu().numpy()
                p_up = tbp[:, 2] / np.maximum(tbp[:, 0] + tbp[:, 2], 1e-6)

                n_s = len(y1)
                bounds.append((offset, offset + n_s))
                offset += n_s
                preds_all.append(p1)
                y_all.append(y1)
                q_all.append(q1)
                prob_all.append(p_up)

                for vkey in ("realized_vol", "volatility", "hist_vol"):
                    if batch.get(vkey) is not None:
                        va = batch[vkey].detach().float().cpu().numpy()
                        vol_real_all.append(va[:, 0] if va.ndim == 2 else va)
                        break
                for lkey in ("adv_usd", "liquidity", "dollar_volume"):
                    if batch.get(lkey) is not None:
                        la = batch[lkey].detach().float().cpu().numpy()
                        adv_all.append(la[:, 0] if la.ndim == 2 else la)
                        break
                if batch.get("asset_ids") is not None:
                    aids_all.append(batch["asset_ids"].detach().cpu().numpy())
                if batch.get("exposures") is not None:
                    exp_all.append(batch["exposures"].detach().float().cpu().numpy())
                if out.get("agent_weights") is not None:
                    aw = out["agent_weights"]
                    w_agent_all.append(
                        aw.detach().float().cpu().numpy() if isinstance(aw, torch.Tensor) else aw
                    )

                d = out["diagnostics"]
                k_used_list.append(float(d.get("k_used", d.get("max_cycles", 1.0))))
                de_neg_list.append(
                    float(d.get("delta_energy_neg_rate", d.get("monotonic_accept_rate", 1.0)))
                )
                ent_list.append(float(d.get("router_entropy", d.get("moe_router_entropy", 0.0))))
                load_ratio_list.append(float(d.get("load_max_min_ratio", 1.0)))
                ecorr_list.append(float(d.get("expert_mean_corr", 0.0)))
                act_list.append(float(d.get("agent_active_rate", 1.0)))
                max_spec = max(
                    max_spec, float(d.get("max_spectral_norm", d.get("lipschitz_upper_bound", 0.0)))
                )

                if isinstance(base_model, MMARRSTitan):
                    state = base_model.step_matured(
                        out["state"],
                        matured_batch=b_ev,
                        matured_targets=y_ret,
                        matured_preds=out["pred_return"][:, 0].detach(),
                    )
                else:
                    state = base_model.step_matured(
                        out["state"],
                        matured_targets=y_ret,
                        matured_preds=out["pred_return"][:, 0].detach(),
                        asset_ids=b_ev.get("asset_ids"),
                    )

        p_cat = np.concatenate(preds_all)
        y_cat = np.concatenate(y_all)
        q_cat = np.vstack(q_all)
        pr_cat = np.concatenate(prob_all)
        e_cat = np.vstack(exp_all) if exp_all else None
        w_cat = np.concatenate(w_agent_all) if len(w_agent_all) == len(bounds) else None

        loop_diag = dict(
            k_used_mean=float(np.mean(k_used_list)),
            delta_energy_neg_rate=float(np.mean(de_neg_list)),
            max_spectral_norm=max_spec,
            router_entropy_mean=float(np.mean(ent_list)),
            load_max_min_mean=float(np.mean(load_ratio_list)),
            expert_mean_corr=float(np.mean(ecorr_list)),
            agent_active_rate=float(np.mean(act_list)),
        )
        rep = evaluate_granular_six_dimensions(
            y=y_cat,
            pred=p_cat,
            bounds=bounds,
            quantiles=q_cat,
            probabilities=pr_cat,
            exposures=e_cat,
            volatility=np.concatenate(vol_real_all) if len(vol_real_all) == len(bounds) else None,
            liquidity=np.concatenate(adv_all) if len(adv_all) == len(bounds) else None,
            asset_ids=np.concatenate(aids_all) if len(aids_all) == len(bounds) else None,
            portfolio_weights=w_cat,
            loop_diagnostics=loop_diag,
            n_trials=eff_n_trials,
            market=market,  # type: ignore[arg-type]
        )
        results[name] = dict(
            parameter_audit=audit,
            granular_report=rep,
            diagnostics=loop_diag,
            elapsed_s=round(time.perf_counter() - t0, 3),
        )

    # Comparaciones automáticas contra los 4 controles emparejados
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for ctrl in (
        "v00_dense_baseline",
        "v01_dense_wide_iso_param",
        "v04_moe_uniform_control",
        "v05_moe_shuffled_ctx_placebo",
    ):
        if ctrl not in results:
            continue
        c_ic = float(results[ctrl]["granular_report"]["dim2_ranking"]["rank_ic_mean"])
        c_sr = float(results[ctrl]["granular_report"]["dim6_economics"]["net_sharpe_1x"])
        comparisons[f"delta_vs_{ctrl}"] = {
            name: dict(
                delta_rank_ic=round(
                    float(results[name]["granular_report"]["dim2_ranking"]["rank_ic_mean"]) - c_ic,
                    5,
                ),
                delta_net_sharpe_1x=round(
                    float(results[name]["granular_report"]["dim6_economics"]["net_sharpe_1x"])
                    - c_sr,
                    4,
                ),
            )
            for name in results
            if math.isfinite(
                float(results[name]["granular_report"]["dim2_ranking"]["rank_ic_mean"])
            )
        }

    return dict(
        hardware=hw_info,
        market=market,
        purge_sessions=eff_purge,
        embargo_sessions=eff_embargo,
        n_trials=eff_n_trials,
        n_train_sessions=len(train_sessions),
        n_eval_sessions=len(eval_sessions),
        results=results,
        control_comparisons=comparisons,
    )
