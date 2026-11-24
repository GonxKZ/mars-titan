"""Pruebas TDD exhaustivas de observabilidad y telemetría de investigación para `MMARRS-TITAN`.

Verifica de forma estricta:
  1. `TelemetryConfig` inmutable, valores por defecto y validación de umbrales.
  2. Clasificación exacta por órgano arquitectónico (`encoders`, `holographic_3l`, `hypergraph`,
     `neuromod_memory`, `hrr_hopfield`, `cognitive_core`, `moe_routers`, `moe_experts`,
     `contractive_loop`, `granular_heads`) sin parámetros huérfanos.
  3. `ActivationProbe`: coste cero en pasos no muestreados (`step % log_every_steps != 0`),
     detección de dead-SiLU y saturación, y limpieza total de hooks al cerrar.
  4. Bucle real multi-paso en `v13_mmarrs_titan_full` y `v07_moe_double_loop`:
     pesos, gradientes pre/post-clip, ratio Karpathy `||ΔW|| / ||W||` verificado contra cálculo
     manual exacto, LRs por grupo (backbone vs router `3x` menor), escala efectiva AdamW,
     descomposición de pérdidas, diagnósticos de dominio (MoE, doble bucle, PT1, HRR,
     Cognitive Core), VRAM/throughput, persistencia JSONL append-only y TensorBoard opcional.
  5. `summarize_run()`: informe de salud y detección exacta de gradientes evanescentes,
     explosivos (incluso tras clipping), NaN/Inf, ratios de actualización muertos/inestables,
     routers MoE colapsados, saturación de activaciones y violación de cotas PT1.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.models.mmarrs.autotuner_sota import MuonHybridOptimizer
from mars_titan.models.mmarrs.cognitive_core import TOOL_NAMES, TitansCognitiveCore
from mars_titan.models.mmarrs.granular_heads import MMARRSMultiTaskLoss
from mars_titan.models.mmarrs.hardware_blackwell import BlackwellRuntimeConfig, build_optimizer
from mars_titan.models.mmarrs.model import build_variation
from mars_titan.models.mmarrs.multimodal_sota import (
    CrossModalConflictFusion,
    SelfSupervisedMultimodalAlignmentLoss,
    SkewStudentTDistributionHead,
)
from mars_titan.models.mmarrs.neuro_brain_sota import (
    NeuroBrainConfig,
    StandaloneHolographicBrainModel,
)
from mars_titan.models.mmarrs.telemetry import (
    ARCHITECTURAL_ORGANS,
    ActivationProbe,
    ModuleWeightGradSnapshot,
    ResearchTelemetryMonitor,
    TelemetryConfig,
    classify_organ,
    classify_param_organ,
    read_telemetry_jsonl,
)


def _synthetic_session(
    rng: np.random.Generator,
    n_assets: int = 16,
    *,
    prices_dim: int = 16,
    fundamentals_dim: int = 8,
    macro_dim: int = 6,
    news_dim: int = 8,
    charts_dim: int = 8,
    physics_dim: int = 12,
    context_dim: int = 10,
    exposures_dim: int = 4,
    rmt_dim: int = 4,
) -> dict[str, torch.Tensor]:
    def _t(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(rng.standard_normal(shape), dtype=torch.float32)

    mask = torch.ones((n_assets, 5), dtype=torch.float32)
    mask[0::4, 3] = 0.0
    prices = _t((n_assets, prices_dim))
    physics = _t((n_assets, physics_dim))
    y1 = 0.01 * prices[:, 0] + 0.008 * physics[:, 0] + 0.01 * _t((n_assets,))
    target_returns = torch.stack(
        [y1 * math.sqrt(h) + 0.004 * _t((n_assets,)) for h in (1, 2, 5, 10, 21, 63)],
        dim=-1,
    )
    return dict(
        prices=prices,
        fundamentals=_t((n_assets, fundamentals_dim)),
        macro=_t((n_assets, macro_dim)),
        news=_t((n_assets, news_dim)),
        charts=_t((n_assets, charts_dim)),
        modality_mask=mask,
        physics_v4=physics,
        high_snr_context=_t((n_assets, context_dim)),
        exposures=_t((n_assets, exposures_dim)),
        rmt_loadings=_t((n_assets, rmt_dim)),
        regime_vec=_t((context_dim,)),
        ach_signal=torch.abs(_t((n_assets, 1))) * 0.5,
        ne_signal=torch.tensor([0.25], dtype=torch.float32),
        target_returns=target_returns,
    )


def _build_small_model(variation: str, *, seed: int = 7) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_variation(
        variation,
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
        n_cortical_areas=8,
        n_hidden_slots=2,
        n_hyperedges=6,
        hrr_max_slots=32,
        seed=seed,
    )


class _RecordingTBWriter:
    """Escritor ligero compatible con `SummaryWriter` para probar escalares e histogramas."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.histograms: list[tuple[str, int, int]] = []
        self.flushed: int = 0
        self.closed: bool = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0) -> None:
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def add_histogram(self, tag: str, values: torch.Tensor, global_step: int = 0) -> None:
        self.histograms.append((tag, int(values.numel()), int(global_step)))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed = True


def test_telemetry_config_defaults_immutability_and_validation(tmp_path: Path) -> None:
    """`TelemetryConfig` es inmutable, normaliza `Path` y rechaza umbrales inválidos."""
    cfg = TelemetryConfig(log_dir=tmp_path)  # type: ignore[arg-type]
    assert cfg.log_dir == str(tmp_path)
    assert cfg.log_every_steps == 10
    assert cfg.histogram_every_steps == 100
    assert cfg.track_weights is True
    assert cfg.track_gradients is True
    assert cfg.track_update_ratios is True
    assert cfg.track_activations is True
    assert cfg.enable_tensorboard is True
    assert cfg.enable_jsonl is True
    assert cfg.grad_explode_threshold == 50.0
    assert cfg.grad_vanish_threshold == 1e-7
    assert cfg.update_ratio_target == 1e-3

    with pytest.raises(AttributeError):
        cfg.log_every_steps = 5  # type: ignore[misc]

    with pytest.raises(ValueError):
        TelemetryConfig(log_dir="")
    with pytest.raises(ValueError):
        TelemetryConfig(log_dir=str(tmp_path), log_every_steps=0)
    with pytest.raises(ValueError):
        TelemetryConfig(log_dir=str(tmp_path), histogram_every_steps=0)
    with pytest.raises(ValueError):
        TelemetryConfig(log_dir=str(tmp_path), grad_explode_threshold=0.0)
    with pytest.raises(ValueError):
        TelemetryConfig(
            log_dir=str(tmp_path), grad_vanish_threshold=10.0, grad_explode_threshold=5.0
        )


def test_organ_classification_covers_all_mmarrs_and_cognitive_core_params() -> None:
    """100 % de parámetros de todas las familias (v13, v00, CogCore, NeuroBrain, SOTA) mapean."""
    assert ARCHITECTURAL_ORGANS == (
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
    assert classify_organ is classify_param_organ

    m_full = _build_small_model("v13_mmarrs_titan_full")
    organs_full: dict[str, int] = {}
    for name, _ in m_full.named_parameters():
        organ = classify_param_organ(name)
        assert organ in ARCHITECTURAL_ORGANS, f"Parámetro sin órgano canónico: {name} -> {organ}"
        organs_full[organ] = organs_full.get(organ, 0) + 1
        # Prefijo `_orig_mod.` (torch.compile) produce exactamente el mismo órgano
        assert classify_param_organ(f"_orig_mod.{name}") == organ

    expected_v13_organs = {
        "encoders",
        "holographic_3l",
        "hypergraph",
        "neuromod_memory",
        "hrr_hopfield",
        "moe_routers",
        "moe_experts",
        "contractive_loop",
        "granular_heads",
    }
    assert set(organs_full.keys()) == expected_v13_organs

    # Verificar separación fina entre hypergraph_l2 (holographic_3l) y hypergraph top-level
    assert classify_param_organ("holographic_3l.hypergraph_l2.edge_queries") == "holographic_3l"
    assert classify_param_organ("hypergraph.edge_queries") == "hypergraph"
    assert classify_param_organ("recursive_core.step_blocks.0.router.weight") == "moe_routers"
    assert classify_param_organ("recursive_core.step_blocks.0.routed_w1") == "moe_experts"
    assert (
        classify_param_organ("recursive_core.contractive_steps.0.lin1.bias") == "contractive_loop"
    )

    # Baseline denso (sin MoE): step_blocks densos van a `contractive_loop`
    m_dense = _build_small_model("v00_dense_baseline")
    for name, _ in m_dense.named_parameters():
        assert classify_param_organ(name) in ARCHITECTURAL_ORGANS

    # Cognitive Core acoplado o independiente -> órgano `cognitive_core`
    cc = TitansCognitiveCore(hidden_dim=16, context_dim=10, use_context_engineer=True)
    for name, _ in cc.named_parameters():
        assert classify_param_organ(f"cognitive_core.{name}") == "cognitive_core"
        assert classify_param_organ(name) == "cognitive_core"


def test_activation_probe_zero_cost_on_non_logged_steps_and_pathology_detection() -> None:
    """`ActivationProbe` no computa nada en pasos no muestreados y detecta dead-SiLU/saturación."""
    rng = np.random.default_rng(19)
    model = _build_small_model("v07_moe_double_loop", seed=19)
    batch = _synthetic_session(rng, n_assets=16)

    probe = ActivationProbe(model)
    assert len(probe.hooked_modules) > 0

    # Paso inactivo: 0 capturas y diccionario vacío
    probe.set_active(False)
    with torch.no_grad():
        model.forward_session(batch, enable_early_exit=False)
    assert probe.active_captures == 0
    assert probe.collect() == {}

    # Paso activo: captura estadísticas finitas por módulo y órgano
    probe.set_active(True)
    with torch.no_grad():
        model.forward_session(batch, enable_early_exit=False)
    captures_after_one = probe.active_captures
    assert captures_after_one > 0
    stats = probe.collect()
    assert len(stats) > 0
    for mod_stat in stats.values():
        assert math.isfinite(mod_stat["mean"])
        assert math.isfinite(mod_stat["std"])
        assert 0.0 <= mod_stat["dead_fraction"] <= 1.0
        assert 0.0 <= mod_stat["saturation_fraction"] <= 1.0

    # 4 pasos inactivos consecutivos: `active_captures` no incrementa ni una vez!
    probe.set_active(False)
    with torch.no_grad():
        for _ in range(4):
            model.forward_session(batch, enable_early_exit=False)
    assert probe.active_captures == captures_after_one

    # Inyectar dead-SiLU severo en `enc_prices` (pre-activación << -5) -> dead_fraction ~ 1.0
    with torch.no_grad():
        model.enc_prices.net[1].weight.zero_()
        model.enc_prices.net[1].bias.fill_(-12.0)
    probe.set_active(True)
    with torch.no_grad():
        model.forward_session(batch, enable_early_exit=False)
    dead_stats = probe.collect()
    assert dead_stats["enc_prices.net.2"]["dead_fraction"] > 0.95

    # Limpieza total de hooks
    probe.remove()
    leftover_hooks = sum(len(m._forward_hooks) for m in model.modules())
    assert leftover_hooks == 0


def test_full_v13_training_loop_telemetry_karpathy_ratio_lrs_domain_and_jsonl(
    tmp_path: Path,
) -> None:
    """Monitoreo completo en `v13_mmarrs_titan_full`: pesos, grads, `||ΔW||/||W||`, LRs y JSONL."""
    rng = np.random.default_rng(42)
    model = _build_small_model("v13_mmarrs_titan_full", seed=42)
    rt_cfg = BlackwellRuntimeConfig(precision_mode="fp32_strict", router_lr_scale=1.0 / 3.0)
    opt = build_optimizer(model, lr=3e-3, weight_decay=1e-4, cfg=rt_cfg)
    loss_fn = MMARRSMultiTaskLoss()

    # Módulo Cognitive Core auxiliar para verificar telemetría de herramientas y poda
    cog_core = TitansCognitiveCore(
        hidden_dim=16, context_dim=10, thinking_steps=2, use_context_engineer=True
    )

    cfg = TelemetryConfig(
        log_dir=str(tmp_path / "run_v13"),
        log_every_steps=2,
        histogram_every_steps=4,
        enable_tensorboard=True,  # debe degradar limpiamente si tensorboard no está instalado
        enable_jsonl=True,
    )

    logged_records: list[dict[str, object]] = []
    manual_ratio_checks: list[tuple[float, float]] = []

    with ResearchTelemetryMonitor(model, cfg, optimizer=opt) as monitor:
        state = None
        w_prev = None
        n_steps = 6  # logged steps: 0, 2, 4; non-logged: 1, 3, 5
        for step in range(n_steps):
            batch = _synthetic_session(rng, n_assets=16)
            captures_before = (
                monitor.activation_probe.active_captures if monitor.activation_probe else 0
            )
            monitor.begin_step(step)

            opt.zero_grad(set_to_none=True)
            out = model.forward_session(batch, state=state, enable_early_exit=False)
            y_ret = batch["target_returns"]
            loss_dict = loss_fn(
                step_outputs=out["loop_outputs"],
                targets=dict(returns=y_ret),
                exposures=batch["exposures"],
                energy_total=out["energy"]["total"],
                w_prev=w_prev,
            )
            loss_dict["total"].backward()

            # Captura manual de referencia en paso 2 para verificar `||ΔW|| / ||W||` de `encoders`
            enc_before = None
            if step == 2:
                enc_before = [
                    p.detach().clone()
                    for n, p in model.named_parameters()
                    if classify_param_organ(n) == "encoders"
                ]

            monitor.before_optimizer_step(step)
            # Clipping estricto a 0.25 para comprobar que pre-clip > post-clip se registra exacto
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            monitor.after_grad_clip(step)

            opt.step()

            if enc_before is not None:
                enc_after = [
                    p.detach()
                    for n, p in model.named_parameters()
                    if classify_param_organ(n) == "encoders"
                ]
                dw_sq = sum(
                    float((a - b).float().pow(2).sum().item())
                    for a, b in zip(enc_after, enc_before, strict=True)
                )
                w_sq = sum(float(b.float().pow(2).sum().item()) for b in enc_before)
                manual_ratio = math.sqrt(dw_sq) / max(math.sqrt(w_sq), 1e-12)
                manual_ratio_checks.append((manual_ratio, 0.0))

            # Diagnósticos reales de PT1 (`neuromod_memory.update`) y Cognitive Core
            with torch.no_grad():
                _, pt1_diag = model.neuromod_memory.update(
                    out["state"]["last_x0"],
                    state.get("titans_state") if state is not None else None,
                    ach_signal=batch["ach_signal"],
                    ne_signal=batch["ne_signal"],
                )
                mod_toks = torch.randn(16, 5, 16)
                phys_emb = torch.randn(16, 16)
                _, _, cog_diag = cog_core(
                    None,
                    batch["high_snr_context"],
                    mod_tokens=mod_toks,
                    modality_mask=batch["modality_mask"],
                    physics_emb=phys_emb,
                )

            rec = monitor.end_step(
                step,
                loss_dict=loss_dict,
                model_output=out,
                titans_update_diag=pt1_diag,
                cognitive_core_diag=cog_diag,
                n_sessions=1,
            )

            captures_after = (
                monitor.activation_probe.active_captures if monitor.activation_probe else 0
            )
            if step % cfg.log_every_steps == 0:
                assert rec is not None
                assert captures_after > captures_before
                logged_records.append(rec)
                if step == 2:
                    logged_ratio = rec["organs"]["encoders"]["update_to_weight_ratio"]
                    manual_ratio_checks[-1] = (manual_ratio_checks[-1][0], float(logged_ratio))
            else:
                assert rec is None
                # Cero capturas de hooks en pasos no muestreados!
                assert captures_after == captures_before

            w_prev = loss_dict["portfolio_weights"]
            state = model.step_matured(
                out["state"],
                matured_batch=batch,
                matured_targets=y_ret,
                matured_preds=out["pred_return"][:, 0].detach(),
            )

        summary = monitor.summarize_run()

    # 1) Exactamente 3 pasos registrados (0, 2, 4)
    assert len(logged_records) == 3
    assert [r["step"] for r in logged_records] == [0, 2, 4]

    # 2) Verificación exacta del ratio Karpathy `||ΔW|| / ||W||` contra cálculo manual
    manual_r, logged_r = manual_ratio_checks[0]
    assert logged_r == pytest.approx(manual_r, rel=1e-4, abs=1e-7)
    assert logged_r > 0.0

    # 3) Inspección completa del paso 4 (cuando memoria Titans y HRR-Hopfield ya están pobladas)
    rec4 = logged_records[-1]
    organs = rec4["organs"]
    for organ_name in (
        "encoders",
        "holographic_3l",
        "hypergraph",
        "neuromod_memory",
        "hrr_hopfield",
        "moe_routers",
        "moe_experts",
        "contractive_loop",
        "granular_heads",
    ):
        snap = organs[organ_name]
        assert snap["weight_l2"] > 0.0
        assert snap["weight_rms"] > 0.0
        assert snap["weight_abs_max"] > 0.0
        assert snap["weight_spectral_approx"] > 0.0
        assert 0.0 <= snap["weight_zero_fraction"] <= 1.0
        assert snap["weight_nan_count"] == 0
        assert snap["weight_inf_count"] == 0
        assert snap["update_to_weight_ratio"] is not None
        assert snap["update_to_weight_ratio"] > 0.0

    # 4) Gradientes pre-clip vs post-clip (clip max_norm=0.25 activo)
    grads = rec4["gradients"]
    assert grads["global_pre_clip_l2"] > grads["global_post_clip_l2"]
    assert grads["global_post_clip_l2"] <= 0.25 + 1e-3
    assert 0.0 < grads["clip_ratio"] < 1.0
    assert grads["has_nonfinite"] is False

    # 5) LRs por grupo (backbone 3e-3 vs router 1e-3) + estado efectivo AdamW
    opt_info = rec4["optimizer"]
    assert opt_info["backbone_lr"] == pytest.approx(3e-3, rel=1e-5)
    assert opt_info["router_lr"] == pytest.approx(1e-3, rel=1e-5)
    assert opt_info["router_to_backbone_lr_ratio"] == pytest.approx(1.0 / 3.0, rel=1e-4)
    assert opt_info["adamw_effective_step_rms"] > 0.0
    assert opt_info["adamw_adaptive_lr_rms"] > 0.0

    # 6) Descomposición completa de pérdida MMARRS
    losses = rec4["losses"]
    for key in ("total", "ic", "crps", "tb", "energy", "sharpe_turnover", "decor"):
        assert key in losses
        assert math.isfinite(losses[key])

    # 7) Activaciones muestreadas
    acts = rec4["activations"]
    assert acts["n_modules_probed"] > 0
    assert len(acts["per_organ"]) > 0

    # 8) Dominio: MoE, bucle, PT1, HRR-Hopfield, Cognitive Core
    dom = rec4["domain"]
    assert dom["moe_router_entropy"] > 0.0
    assert dom["moe_load_max_min_ratio"] >= 1.0
    assert math.isfinite(dom["moe_expert_mean_corr"])
    assert dom["loop_k_used"] == float(model.config.max_loops)
    assert 0.0 <= dom["loop_delta_energy_neg_pct"] <= 100.0
    assert dom["pt1_alpha_min"] >= 0.05 - 1e-6
    assert dom["pt1_eta_max"] <= 0.95 + 1e-6
    assert dom["pt1_theta_max"] <= 0.25 + 1e-6
    assert dom["hrr_ood_energy_mean"] >= 0.0
    assert dom["hrr_slots_used"] > 0
    assert set(dom["cognitive_tool_usage"].keys()) == set(TOOL_NAMES)
    assert sum(dom["cognitive_tool_usage"].values()) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= dom["cognitive_pruned_context_ratio"] <= 1.0

    # 9) Hardware (CPU sin CUDA)
    hw = rec4["hardware"]
    assert hw["cuda_vram_allocated_mb"] == 0.0
    assert hw["cuda_vram_reserved_mb"] == 0.0
    assert hw["sessions_per_sec"] > 0.0

    # 10) Persistencia JSONL append-only y `summarize_run()`
    jsonl_path = Path(cfg.log_dir) / cfg.jsonl_filename
    assert jsonl_path.is_file()
    persisted = read_telemetry_jsonl(jsonl_path)
    assert len(persisted) == 3
    assert [p["step"] for p in persisted] == [0, 2, 4]
    assert summary["n_logged_steps"] == 3
    assert summary["total_steps_seen"] == 6
    assert "organ_summary" in summary
    assert "loss_summary" in summary
    assert "domain_summary" in summary
    assert "hardware_summary" in summary


def test_v07_moe_double_loop_tensorboard_writer_scalars_and_histograms(tmp_path: Path) -> None:
    """TensorBoard emite escalares cada `log_every_steps` e histogramas cada `histogram_every`."""
    rng = np.random.default_rng(101)
    model = _build_small_model("v07_moe_double_loop", seed=101)
    opt = build_optimizer(model, lr=2e-3, weight_decay=1e-4)
    loss_fn = MMARRSMultiTaskLoss()
    fake_tb = _RecordingTBWriter()

    cfg = TelemetryConfig(
        log_dir=str(tmp_path / "run_v07"),
        log_every_steps=2,
        histogram_every_steps=4,
        enable_tensorboard=True,
        enable_jsonl=True,
    )

    with ResearchTelemetryMonitor(model, cfg, optimizer=opt, tb_writer=fake_tb) as monitor:
        assert monitor.tensorboard_active is True
        for step in range(6):  # logged: 0, 2, 4; histograms: 0, 4
            batch = _synthetic_session(rng, n_assets=16)
            monitor.begin_step(step)
            opt.zero_grad(set_to_none=True)
            out = model.forward_session(batch, enable_early_exit=False)
            loss_dict = loss_fn(
                step_outputs=out["loop_outputs"],
                targets=dict(returns=batch["target_returns"]),
                exposures=batch["exposures"],
                energy_total=out["energy"]["total"],
            )
            loss_dict["total"].backward()
            monitor.clip_grad_norm(step, max_norm=2.0)
            opt.step()
            monitor.end_step(step, loss_dict=loss_dict, model_output=out)

        snap = monitor.last_snapshots["moe_routers"]
        assert isinstance(snap, ModuleWeightGradSnapshot)
        assert snap.organ == "moe_routers"

    assert fake_tb.closed is True
    assert fake_tb.flushed >= 3

    scalar_steps = sorted({s for _, _, s in fake_tb.scalars})
    assert scalar_steps == [0, 2, 4]
    scalar_tags = {tag for tag, _, _ in fake_tb.scalars}
    assert any(t.startswith("Loss/") for t in scalar_tags)
    assert any(t.startswith("LR/") for t in scalar_tags)
    assert any(t.startswith("Weights/moe_routers/") for t in scalar_tags)
    assert any(t.startswith("Gradients/moe_experts/") for t in scalar_tags)
    assert any(t.startswith("UpdateRatio/contractive_loop/") for t in scalar_tags)
    assert any(t.startswith("Activations/") for t in scalar_tags)
    assert any(t.startswith("Domain/MoE/") for t in scalar_tags)
    assert any(t.startswith("Hardware/") for t in scalar_tags)

    hist_steps = sorted({s for _, _, s in fake_tb.histograms})
    assert hist_steps == [0, 4]


def test_summarize_run_detects_gradient_update_router_and_pt1_anomalies(tmp_path: Path) -> None:
    """Patologías inyectadas (grads, updates, routers, saturación, PT1) disparan flags exactos."""
    rng = np.random.default_rng(202)
    loss_fn = MMARRSMultiTaskLoss()

    # --- Caso 1: Gradientes evanescentes en `encoders` y explosivos en `granular_heads` ---
    model1 = _build_small_model("v07_moe_double_loop", seed=1)
    opt1 = build_optimizer(model1, lr=1e-3)
    cfg1 = TelemetryConfig(
        log_dir=str(tmp_path / "case_grad_pathology"),
        log_every_steps=1,
        enable_tensorboard=False,
        grad_explode_threshold=20.0,
        grad_vanish_threshold=1e-6,
    )
    with ResearchTelemetryMonitor(model1, cfg1, optimizer=opt1) as mon1:
        batch = _synthetic_session(rng, n_assets=16)
        mon1.begin_step(0)
        opt1.zero_grad(set_to_none=True)
        out = model1.forward_session(batch, enable_early_exit=False)
        ld = loss_fn(
            step_outputs=out["loop_outputs"],
            targets=dict(returns=batch["target_returns"]),
            exposures=batch["exposures"],
            energy_total=out["energy"]["total"],
        )
        ld["total"].backward()

        # Inyectar vanishing grads en `encoders` y exploding grads en `granular_heads`
        with torch.no_grad():
            for name, p in model1.named_parameters():
                if p.grad is None:
                    continue
                organ = classify_param_organ(name)
                if organ == "encoders":
                    p.grad.mul_(1e-12)
                elif organ == "granular_heads":
                    p.grad.fill_(100.0)

        mon1.before_optimizer_step(0)
        # Incluso tras clipping agresivo a 1.0, la explosión pre-clip debe quedar registrada!
        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
        mon1.after_grad_clip(0)
        opt1.step()
        mon1.end_step(0, loss_dict=ld, model_output=out)
        rep1 = mon1.summarize_run()

    assert rep1["healthy"] is False
    assert rep1["anomaly_flags"]["vanishing_gradients"] is True
    assert "encoders" in rep1["affected_organs"]["vanishing_gradients"]
    assert rep1["anomaly_flags"]["exploding_gradients"] is True
    assert "granular_heads" in rep1["affected_organs"]["exploding_gradients"]

    # --- Caso 2: Gradiente NaN en `moe_experts` -> flag exacto + JSONL estrictamente válido ---
    model2 = _build_small_model("v07_moe_double_loop", seed=2)
    opt2 = build_optimizer(model2, lr=1e-3)
    cfg2 = TelemetryConfig(
        log_dir=str(tmp_path / "case_nan_grad"),
        log_every_steps=1,
        enable_tensorboard=False,
    )
    with ResearchTelemetryMonitor(model2, cfg2, optimizer=opt2) as mon2:
        batch = _synthetic_session(rng, n_assets=16)
        mon2.begin_step(0)
        opt2.zero_grad(set_to_none=True)
        out = model2.forward_session(batch, enable_early_exit=False)
        ld = loss_fn(
            step_outputs=out["loop_outputs"],
            targets=dict(returns=batch["target_returns"]),
            exposures=batch["exposures"],
            energy_total=out["energy"]["total"],
        )
        ld["total"].backward()
        with torch.no_grad():
            model2.recursive_core.step_blocks[0].routed_w1.grad.view(-1)[0] = float("nan")
        mon2.before_optimizer_step(0)
        mon2.after_grad_clip(0)
        mon2.end_step(0, loss_dict=ld, model_output=out)
        rep2 = mon2.summarize_run()

    assert rep2["healthy"] is False
    assert rep2["anomaly_flags"]["nonfinite_gradients"] is True
    assert "moe_experts" in rep2["affected_organs"]["nonfinite_gradients"]
    parsed_records = read_telemetry_jsonl(Path(cfg2.log_dir) / cfg2.jsonl_filename)
    assert len(parsed_records) == 1
    assert parsed_records[0]["organs"]["moe_experts"]["grad_nan_count"] >= 1

    # --- Caso 3: Update-to-weight ratio muerto (<1e-6) e inestable (>1e-1) ---
    model3_dead = _build_small_model("v07_moe_double_loop", seed=3)
    opt3_dead = build_optimizer(model3_dead, lr=1e-10)
    cfg3_dead = TelemetryConfig(
        log_dir=str(tmp_path / "case_dead_update"),
        log_every_steps=1,
        enable_tensorboard=False,
    )
    with ResearchTelemetryMonitor(model3_dead, cfg3_dead, optimizer=opt3_dead) as mon_dead:
        batch = _synthetic_session(rng, n_assets=16)
        mon_dead.begin_step(0)
        opt3_dead.zero_grad(set_to_none=True)
        out = model3_dead.forward_session(batch, enable_early_exit=False)
        ld = loss_fn(
            step_outputs=out["loop_outputs"],
            targets=dict(returns=batch["target_returns"]),
            exposures=batch["exposures"],
            energy_total=out["energy"]["total"],
        )
        ld["total"].backward()
        mon_dead.before_optimizer_step(0)
        opt3_dead.step()
        mon_dead.end_step(0, loss_dict=ld, model_output=out)
        rep_dead = mon_dead.summarize_run()
    assert rep_dead["healthy"] is False
    assert rep_dead["anomaly_flags"]["dead_updates"] is True

    model3_hi = _build_small_model("v07_moe_double_loop", seed=4)
    opt3_hi = build_optimizer(model3_hi, lr=2.0)
    cfg3_hi = TelemetryConfig(
        log_dir=str(tmp_path / "case_unstable_update"),
        log_every_steps=1,
        enable_tensorboard=False,
    )
    with ResearchTelemetryMonitor(model3_hi, cfg3_hi, optimizer=opt3_hi) as mon_hi:
        batch = _synthetic_session(rng, n_assets=16)
        mon_hi.begin_step(0)
        opt3_hi.zero_grad(set_to_none=True)
        out = model3_hi.forward_session(batch, enable_early_exit=False)
        ld = loss_fn(
            step_outputs=out["loop_outputs"],
            targets=dict(returns=batch["target_returns"]),
            exposures=batch["exposures"],
            energy_total=out["energy"]["total"],
        )
        ld["total"].backward()
        mon_hi.before_optimizer_step(0)
        opt3_hi.step()
        mon_hi.end_step(0, loss_dict=ld, model_output=out)
        rep_hi = mon_hi.summarize_run()
    assert rep_hi["healthy"] is False
    assert rep_hi["anomaly_flags"]["unstable_updates"] is True

    # --- Caso 4: Router MoE muerto, colapso de expertos y violación de caja PT1 ---
    model4 = _build_small_model("v13_mmarrs_titan_full", seed=5)
    opt4 = build_optimizer(model4, lr=1e-3)
    cfg4 = TelemetryConfig(
        log_dir=str(tmp_path / "case_router_pt1"),
        log_every_steps=1,
        enable_tensorboard=False,
    )
    with ResearchTelemetryMonitor(model4, cfg4, optimizer=opt4) as mon4:
        batch = _synthetic_session(rng, n_assets=16)
        mon4.begin_step(0)
        opt4.zero_grad(set_to_none=True)
        out = model4.forward_session(batch, enable_early_exit=False)
        ld = loss_fn(
            step_outputs=out["loop_outputs"],
            targets=dict(returns=batch["target_returns"]),
            exposures=batch["exposures"],
            energy_total=out["energy"]["total"],
        )
        ld["total"].backward()
        mon4.before_optimizer_step(0)
        opt4.step()
        # Inyectar router muerto, expertos colapsados y caja PT1 violada
        fake_out = dict(out)
        fake_diag = dict(out["diagnostics"])
        fake_diag["router_entropy"] = 0.05
        fake_diag["load_max_min_ratio"] = 25.0
        fake_diag["expert_mean_corr"] = 0.96
        fake_out["diagnostics"] = fake_diag
        bad_pt1 = dict(alpha_min=0.001, alpha_mean=0.02, eta_max=0.99, theta_max=0.60)
        mon4.end_step(0, loss_dict=ld, model_output=fake_out, titans_update_diag=bad_pt1)
        rep4 = mon4.summarize_run()

    assert rep4["healthy"] is False
    assert rep4["anomaly_flags"]["dead_routers"] is True
    assert rep4["anomaly_flags"]["expert_collapse"] is True
    assert rep4["anomaly_flags"]["pt1_bound_violation"] is True

    # --- Caso 5: Saturación / dead-SiLU severo en activaciones -> `activation_saturation` ---
    model5 = _build_small_model("v07_moe_double_loop", seed=6)
    with torch.no_grad():
        model5.enc_prices.net[1].weight.zero_()
        model5.enc_prices.net[1].bias.fill_(-12.0)
    opt5 = build_optimizer(model5, lr=1e-3)
    cfg5 = TelemetryConfig(
        log_dir=str(tmp_path / "case_activation_sat"),
        log_every_steps=1,
        enable_tensorboard=False,
    )
    with ResearchTelemetryMonitor(model5, cfg5, optimizer=opt5) as mon5:
        batch = _synthetic_session(rng, n_assets=16)
        mon5.begin_step(0)
        opt5.zero_grad(set_to_none=True)
        out = model5.forward_session(batch, enable_early_exit=False)
        ld = loss_fn(
            step_outputs=out["loop_outputs"],
            targets=dict(returns=batch["target_returns"]),
            exposures=batch["exposures"],
            energy_total=out["energy"]["total"],
        )
        ld["total"].backward()
        mon5.before_optimizer_step(0)
        opt5.step()
        mon5.end_step(0, loss_dict=ld, model_output=out)
        rep5 = mon5.summarize_run()

    assert rep5["healthy"] is False
    assert rep5["anomaly_flags"]["activation_saturation"] is True
    assert "encoders" in rep5["affected_organs"]["activation_saturation"]


def test_family2_neuro_brain_and_multimodal_sota_organ_classification_and_telemetry(
    tmp_path: Path,
) -> None:
    """Órganos de Familia 2 (`neuro_brain_sota`) y `multimodal_sota` clasifican y monitorean."""
    neuro_cfg = NeuroBrainConfig(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
        n_cortical_areas=8,
        n_hidden_slots=2,
        n_hyperedges=6,
        hrr_max_slots=32,
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_hrr_hopfield=True,
        use_three_factor=True,
        use_replay=True,
        use_moe_bridge=True,
        max_cycles=2,
    )
    neuro_model = StandaloneHolographicBrainModel(neuro_cfg)
    neuro_organs: dict[str, int] = {}
    for name, _ in neuro_model.named_parameters():
        organ = classify_organ(name)
        assert organ in ARCHITECTURAL_ORGANS, f"Parámetro NeuroBrain sin órgano: {name} -> {organ}"
        neuro_organs[organ] = neuro_organs.get(organ, 0) + 1

    for expected in (
        "encoders",
        "holographic_3l",
        "dendritic",
        "lateral_inhibition",
        "hrr_hopfield",
        "three_factor",
        "replay",
        "moe_routers",
        "moe_experts",
        "contractive_loop",
        "granular_heads",
    ):
        assert expected in neuro_organs, f"Falta órgano Familia 2: {expected}"

    # Contenedor multimodal SOTA con los 3 módulos (`cross_modal_fusion`, `ssl_alignment`, `skew`)
    class _MultimodalSotaWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross_modal_fusion = CrossModalConflictFusion(hidden_dim=16, n_heads=2)
            self.ssl_alignment = SelfSupervisedMultimodalAlignmentLoss(hidden_dim=16, n_heads=2)
            self.skew_student_t = SkewStudentTDistributionHead(hidden_dim=16, horizons=(1, 5))

    mm_sota = _MultimodalSotaWrapper()
    mm_organs = {classify_organ(n) for n, _ in mm_sota.named_parameters()}
    assert mm_organs == {"cross_modal_fusion", "ssl_alignment", "skew_student_t"}

    # Verificar monitoreo completo sobre `StandaloneHolographicBrainModel`
    rng = np.random.default_rng(77)
    batch = _synthetic_session(rng, n_assets=16)
    opt = MuonHybridOptimizer.from_model(neuro_model, lr=2e-3, weight_decay=1e-4)
    cfg = TelemetryConfig(
        log_dir=str(tmp_path / "run_neuro_muon"),
        log_every_steps=1,
        enable_tensorboard=False,
    )
    with ResearchTelemetryMonitor(neuro_model, cfg, optimizer=opt) as monitor:
        monitor.begin_step(0)
        opt.zero_grad(set_to_none=True)
        out = neuro_model.forward_session(batch)
        loss = (out["pred_return"][:, 0] - batch["target_returns"][:, 0]).pow(2).mean()
        loss.backward()
        monitor.clip_grad_norm(0, max_norm=2.0)
        opt.step()
        rec = monitor.end_step(0, loss_dict=dict(total=loss), model_output=out)
        summary = monitor.summarize_run()

    assert rec is not None
    for organ in ("dendritic", "lateral_inhibition", "three_factor", "replay", "holographic_3l"):
        assert organ in rec["organs"]
        assert rec["organs"][organ]["weight_l2"] > 0.0
    assert rec["optimizer"]["muon_momentum_buffer_l2"] > 0.0
    assert rec["optimizer"]["muon_momentum_buffer_rms"] > 0.0
    assert rec["optimizer"]["muon_momentum_buffer_abs_max"] > 0.0
    assert rec["optimizer"]["muon_param_count"] > 0
    assert rec["optimizer"]["muon_lr"] == pytest.approx(2e-3, rel=1e-5)
    assert rec["optimizer"]["adamw_effective_step_rms"] > 0.0
    assert len(rec["optimizer"]["muon_momentum_by_organ"]) > 0
    assert summary["lr_summary"]["last_muon_momentum_buffer_l2"] > 0.0


def test_muon_hybrid_optimizer_and_sgd_momentum_buffer_telemetry(tmp_path: Path) -> None:
    """Telemetría registra `momentum_buffer` en `MuonHybridOptimizer` y emite escalares TB."""
    rng = np.random.default_rng(88)
    model = _build_small_model("v07_moe_double_loop", seed=88)
    opt = MuonHybridOptimizer.from_model(
        model, lr=3e-3, weight_decay=1e-4, router_lr_scale=1.0 / 3.0
    )
    loss_fn = MMARRSMultiTaskLoss()
    fake_tb = _RecordingTBWriter()
    cfg = TelemetryConfig(
        log_dir=str(tmp_path / "run_muon_v07"),
        log_every_steps=1,
        histogram_every_steps=10,
        enable_tensorboard=True,
    )

    with ResearchTelemetryMonitor(model, cfg, optimizer=opt, tb_writer=fake_tb) as monitor:
        for step in range(2):
            batch = _synthetic_session(rng, n_assets=16)
            monitor.begin_step(step)
            opt.zero_grad(set_to_none=True)
            out = model.forward_session(batch, enable_early_exit=False)
            ld = loss_fn(
                step_outputs=out["loop_outputs"],
                targets=dict(returns=batch["target_returns"]),
                exposures=batch["exposures"],
                energy_total=out["energy"]["total"],
            )
            ld["total"].backward()
            monitor.clip_grad_norm(step, max_norm=2.0)
            opt.step()
            rec = monitor.end_step(step, loss_dict=ld, model_output=out)

    assert rec is not None
    opt_t = rec["optimizer"]
    assert opt_t["muon_momentum_buffer_l2"] > 0.0
    assert opt_t["muon_momentum_buffer_rms"] > 0.0
    assert opt_t["muon_param_count"] > 0
    assert opt_t["muon_lr"] == pytest.approx(3e-3, rel=1e-5)
    assert opt_t["router_lr"] == pytest.approx(1e-3, rel=1e-5)
    assert opt_t["adamw_effective_step_rms"] > 0.0
    assert "encoders" in opt_t["muon_momentum_by_organ"]

    tb_tags = {tag for tag, _, _ in fake_tb.scalars}
    assert "Optimizer/muon_momentum_buffer_l2" in tb_tags
    assert "Optimizer/muon_momentum_buffer_rms" in tb_tags
    assert "LR/muon_lr" in tb_tags


def test_compressed_jsonl_gz_lossless_roundtrip_and_compact_disk_footprint(tmp_path: Path) -> None:
    """Por defecto escribe `.jsonl.gz` sin pérdida, legible en caliente y mucho más compacto."""
    rng_a = np.random.default_rng(303)
    rng_b = np.random.default_rng(303)
    m_gz = _build_small_model("v07_moe_double_loop", seed=303)
    m_raw = _build_small_model("v07_moe_double_loop", seed=303)
    opt_gz = build_optimizer(m_gz, lr=2e-3)
    opt_raw = build_optimizer(m_raw, lr=2e-3)
    loss_fn = MMARRSMultiTaskLoss()

    cfg_gz = TelemetryConfig(
        log_dir=str(tmp_path / "gz_run"),
        log_every_steps=1,
        enable_tensorboard=False,
    )
    cfg_raw = TelemetryConfig(
        log_dir=str(tmp_path / "raw_run"),
        log_every_steps=1,
        enable_tensorboard=False,
        compress_jsonl=False,
    )
    assert cfg_gz.compress_jsonl is True
    assert cfg_gz.jsonl_filename.endswith(".jsonl.gz")
    assert cfg_raw.compress_jsonl is False
    assert cfg_raw.jsonl_filename.endswith(".jsonl")

    with (
        ResearchTelemetryMonitor(m_gz, cfg_gz, optimizer=opt_gz) as mon_gz,
        ResearchTelemetryMonitor(m_raw, cfg_raw, optimizer=opt_raw) as mon_raw,
    ):
        for step in range(4):
            b_gz = _synthetic_session(rng_a, n_assets=16)
            b_raw = _synthetic_session(rng_b, n_assets=16)

            mon_gz.begin_step(step)
            opt_gz.zero_grad(set_to_none=True)
            o_gz = m_gz.forward_session(b_gz, enable_early_exit=False)
            l_gz = loss_fn(
                step_outputs=o_gz["loop_outputs"],
                targets=dict(returns=b_gz["target_returns"]),
                exposures=b_gz["exposures"],
                energy_total=o_gz["energy"]["total"],
            )
            l_gz["total"].backward()
            mon_gz.clip_grad_norm(step, max_norm=2.0)
            opt_gz.step()
            mon_gz.end_step(step, loss_dict=l_gz, model_output=o_gz)

            mon_raw.begin_step(step)
            opt_raw.zero_grad(set_to_none=True)
            o_raw = m_raw.forward_session(b_raw, enable_early_exit=False)
            l_raw = loss_fn(
                step_outputs=o_raw["loop_outputs"],
                targets=dict(returns=b_raw["target_returns"]),
                exposures=b_raw["exposures"],
                energy_total=o_raw["energy"]["total"],
            )
            l_raw["total"].backward()
            mon_raw.clip_grad_norm(step, max_norm=2.0)
            opt_raw.step()
            mon_raw.end_step(step, loss_dict=l_raw, model_output=o_raw)

        # Lectura en caliente (antes de `close()`) funciona en ambos formatos
        mid_recs = read_telemetry_jsonl(mon_gz.jsonl_path)
        assert len(mid_recs) == 4

    gz_path = Path(cfg_gz.log_dir) / cfg_gz.jsonl_filename
    raw_path = Path(cfg_raw.log_dir) / cfg_raw.jsonl_filename
    # Verificar cabecera mágica gzip `\x1f\x8b` y reducción real en disco
    assert gz_path.read_bytes()[:2] == b"\x1f\x8b"
    assert gz_path.stat().st_size < raw_path.stat().st_size * 0.55

    # Fidelidad 100 % sin pérdida (excepto timestamp y tiempos de reloj de pared)
    recs_gz = read_telemetry_jsonl(gz_path)
    recs_raw = read_telemetry_jsonl(raw_path)
    assert len(recs_gz) == len(recs_raw) == 4
    for rg, rr in zip(recs_gz, recs_raw, strict=True):
        assert rg["step"] == rr["step"]
        assert rg["organs"] == rr["organs"]
        assert rg["losses"] == rr["losses"]
        assert rg["optimizer"] == rr["optimizer"]
        assert rg["domain"] == rr["domain"]


def test_p1_6_domain_telemetry_batches_all_tensors_into_single_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-6: `_collect_domain_telemetry` agrupa 20+ tensores en 1 `.tolist()` y 0 `.item()`."""
    model = _build_small_model("v13_mmarrs_titan_full", seed=17)
    cfg = TelemetryConfig(
        log_dir=str(tmp_path / "domain_batch"),
        log_every_steps=1,
        enable_tensorboard=False,
        enable_jsonl=False,
    )
    with ResearchTelemetryMonitor(model, cfg) as monitor:
        model_output = dict(
            diagnostics=dict(
                router_entropy=torch.tensor(1.23),
                load_max_min_ratio=torch.tensor(1.45),
                expert_mean_corr=torch.tensor(0.31),
                expert_loads=[
                    torch.tensor([0.1, 0.2, 0.3, 0.4]),
                    torch.tensor([0.3, 0.2, 0.1, 0.4]),
                ],
                k_used=torch.tensor(3.0),
                delta_energy_neg_rate=torch.tensor(0.85),
                max_spectral_norm=torch.tensor(0.77),
                energy_trajectory=[torch.tensor(2.0), torch.tensor(1.1)],
                neuromod=dict(
                    alpha_min=torch.tensor(0.08),
                    alpha_mean=torch.tensor(0.22),
                    eta_max=torch.tensor(0.88),
                    theta_max=torch.tensor(0.19),
                    theta_mean=torch.tensor(0.09),
                    ne_slow_shift=torch.tensor(0.04),
                    mem_norm=torch.tensor(3.5),
                ),
                ood_energy_mean=torch.tensor(0.62),
            ),
            state={},
        )
        cog_diag = dict(
            tool_names=TOOL_NAMES,
            tool_usage_distribution=torch.tensor([0.1, 0.2, 0.15, 0.25, 0.15, 0.15]),
            tool_entropy=torch.tensor(1.72),
            pruned_context_ratio=torch.tensor(0.42),
            lipschitz_upper_bound=torch.tensor(0.83),
        )

        orig_tolist = torch.Tensor.tolist
        tolist_calls = 0

        def _counting_tolist(self: torch.Tensor):
            nonlocal tolist_calls
            tolist_calls += 1
            return orig_tolist(self)

        def _forbid_item(self: torch.Tensor):
            raise AssertionError("P1-6: `.item()` prohibido dentro de `_collect_domain_telemetry`")

        monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
        monkeypatch.setattr(torch.Tensor, "tolist", _counting_tolist)

        dom = monitor._collect_domain_telemetry(
            model_output=model_output,
            titans_update_diag=None,
            cognitive_core_diag=cog_diag,
        )

    assert tolist_calls == 1
    assert dom["moe_router_entropy"] == pytest.approx(1.23, abs=1e-5)
    assert dom["moe_load_max_min_ratio"] == pytest.approx(1.45, abs=1e-5)
    assert dom["moe_expert_mean_corr"] == pytest.approx(0.31, abs=1e-5)
    assert dom["moe_expert_loads"] == pytest.approx([0.2, 0.2, 0.2, 0.4], abs=1e-5)
    assert dom["loop_k_used"] == pytest.approx(3.0, abs=1e-5)
    assert dom["loop_delta_energy_neg_pct"] == pytest.approx(85.0, abs=1e-4)
    assert dom["loop_max_spectral_norm"] == pytest.approx(0.77, abs=1e-5)
    assert dom["loop_energy_final"] == pytest.approx(1.1, abs=1e-5)
    assert dom["pt1_alpha_min"] == pytest.approx(0.08, abs=1e-5)
    assert dom["pt1_eta_max"] == pytest.approx(0.88, abs=1e-5)
    assert dom["pt1_theta_max"] == pytest.approx(0.19, abs=1e-5)
    assert dom["neuromod_ne_slow_shift"] == pytest.approx(0.04, abs=1e-5)
    assert dom["neuromod_mem_norm"] == pytest.approx(3.5, abs=1e-5)
    assert dom["hrr_ood_energy_mean"] == pytest.approx(0.62, abs=1e-5)
    assert dom["hrr_slots_used"] == 0
    assert dom["cognitive_tool_entropy"] == pytest.approx(1.72, abs=1e-5)
    assert dom["cognitive_pruned_context_ratio"] == pytest.approx(0.42, abs=1e-5)
    assert dom["cognitive_lipschitz_upper_bound"] == pytest.approx(0.83, abs=1e-5)
    assert sum(dom["cognitive_tool_usage"].values()) == pytest.approx(1.0, abs=1e-5)


def test_p1_6_optimizer_telemetry_zero_h2d_allocs_and_schedule_free_adamw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-6: cero `torch.tensor()`/`new_tensor()` en `_collect_optimizer_telemetry` + SF-AdamW."""
    from mars_titan.models.mmarrs.autotuner_sota import build_sota_optimizer

    rng = np.random.default_rng(91)
    model_adam = _build_small_model("v07_moe_double_loop", seed=91)
    opt_adam = build_optimizer(model_adam, lr=2e-3, weight_decay=1e-4)
    batch = _synthetic_session(rng, n_assets=16)
    out = model_adam.forward_session(batch, enable_early_exit=False)
    (out["pred_return"][:, 0] - batch["target_returns"][:, 0]).pow(2).mean().backward()
    opt_adam.step()

    cfg_a = TelemetryConfig(
        log_dir=str(tmp_path / "opt_adam_no_h2d"),
        log_every_steps=1,
        enable_tensorboard=False,
        enable_jsonl=False,
    )
    with ResearchTelemetryMonitor(model_adam, cfg_a, optimizer=opt_adam) as mon_a:
        def _forbid_tensor(*args, **kwargs):
            raise AssertionError("P1-6: `torch.tensor()` prohibido en optimizer telemetry")

        def _forbid_new_tensor(self, *args, **kwargs):
            raise AssertionError("P1-6: `.new_tensor()` prohibido en optimizer telemetry")

        monkeypatch.setattr(torch, "tensor", _forbid_tensor)
        monkeypatch.setattr(torch.Tensor, "new_tensor", _forbid_new_tensor)
        opt_tel = mon_a._collect_optimizer_telemetry(opt_adam)

    monkeypatch.undo()
    assert opt_tel["adamw_effective_step_rms"] > 0.0
    assert opt_tel["adamw_adaptive_lr_rms"] > 0.0
    assert opt_tel["adamw_exp_avg_norm"] > 0.0
    assert opt_tel["adamw_exp_avg_sq_rms"] > 0.0

    # Verificar telemetría real de `ScheduleFreeAdamW` (norma `z - x` + adaptative LR + TB)
    model_sf = _build_small_model("v07_moe_double_loop", seed=92)
    opt_sf = build_sota_optimizer(
        model_sf,
        optimizer_type="schedule_free_adamw",
        lr=3e-3,
        weight_decay=1e-4,
        sf_warmup_steps=2,
    )
    fake_tb = _RecordingTBWriter()
    cfg_sf = TelemetryConfig(
        log_dir=str(tmp_path / "opt_schedule_free"),
        log_every_steps=1,
        enable_tensorboard=True,
        enable_jsonl=True,
    )
    with ResearchTelemetryMonitor(model_sf, cfg_sf, optimizer=opt_sf, tb_writer=fake_tb) as mon_sf:
        for step in range(3):
            b_s = _synthetic_session(rng, n_assets=16)
            mon_sf.begin_step(step)
            opt_sf.zero_grad(set_to_none=True)
            o_s = model_sf.forward_session(b_s, enable_early_exit=False)
            loss_s = (o_s["pred_return"][:, 0] - b_s["target_returns"][:, 0]).pow(2).mean()
            loss_s.backward()
            mon_sf.clip_grad_norm(step, max_norm=2.0)
            opt_sf.step()
            rec = mon_sf.end_step(step, loss_dict=dict(total=loss_s), model_output=o_s)
        summary_sf = mon_sf.summarize_run()

    assert rec is not None
    manual_diff_sq = sum(
        float((st["z"].float() - st["x"].float()).pow(2).sum().item())
        for st in opt_sf.state.values()
        if "z" in st and "x" in st
    )
    opt_sf_t = rec["optimizer"]
    assert opt_sf_t["schedule_free_param_count"] > 0
    assert opt_sf_t["schedule_free_z_x_diff_l2"] == pytest.approx(
        math.sqrt(manual_diff_sq), rel=1e-4, abs=1e-7
    )
    assert opt_sf_t["schedule_free_z_x_diff_l2"] > 0.0
    assert opt_sf_t["schedule_free_z_x_diff_rms"] > 0.0
    assert opt_sf_t["schedule_free_z_x_diff_abs_max"] > 0.0
    assert opt_sf_t["schedule_free_z_x_rel_diff"] > 0.0
    assert opt_sf_t["schedule_free_adaptive_lr_rms"] > 0.0
    assert opt_sf_t["schedule_free_exp_avg_sq_rms"] > 0.0
    assert len(opt_sf_t["schedule_free_z_x_diff_by_organ"]) > 0
    assert summary_sf["lr_summary"]["last_schedule_free_z_x_diff_l2"] == pytest.approx(
        opt_sf_t["schedule_free_z_x_diff_l2"], rel=1e-5
    )
    tb_tags = {tag for tag, _, _ in fake_tb.scalars}
    assert "Optimizer/schedule_free_z_x_diff_l2" in tb_tags
    assert "Optimizer/schedule_free_z_x_diff_rms" in tb_tags


def test_p1_6_spectral_norm_throttled_to_spectral_every_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-6: `_spectral_norm_approx` solo se ejecuta cada `spectral_every_k` pasos (defecto 50)."""
    import mars_titan.models.mmarrs.telemetry as tel_mod

    default_cfg = TelemetryConfig(log_dir=str(tmp_path / "def_spec"))
    assert default_cfg.spectral_every_k == 50
    with pytest.raises(ValueError):
        TelemetryConfig(log_dir=str(tmp_path / "bad_spec"), spectral_every_k=0)

    rng = np.random.default_rng(55)
    model = _build_small_model("v00_dense_baseline", seed=55)
    opt = build_optimizer(model, lr=1e-3)
    n_param_tensors = sum(1 for _ in model.parameters())

    orig_spec = tel_mod._spectral_norm_approx
    spec_calls = 0

    def _counting_spec(tensor: torch.Tensor, *, n_iters: int = 4) -> torch.Tensor:
        nonlocal spec_calls
        spec_calls += 1
        return orig_spec(tensor, n_iters=n_iters)

    monkeypatch.setattr(tel_mod, "_spectral_norm_approx", _counting_spec)

    cfg = TelemetryConfig(
        log_dir=str(tmp_path / "spec_throttle"),
        log_every_steps=1,
        spectral_every_k=5,
        enable_tensorboard=False,
        enable_jsonl=False,
    )
    records = []
    with ResearchTelemetryMonitor(model, cfg, optimizer=opt) as mon:
        for step in range(11):  # spectral steps: 0, 5, 10 (exactamente 3 de 11!)
            batch = _synthetic_session(rng, n_assets=8)
            mon.begin_step(step)
            opt.zero_grad(set_to_none=True)
            out = model.forward_session(batch, enable_early_exit=False)
            loss = (out["pred_return"][:, 0] - batch["target_returns"][:, 0]).pow(2).mean()
            loss.backward()
            mon.before_optimizer_step(step)
            opt.step()
            rec = mon.end_step(step, loss_dict=dict(total=loss), model_output=out)
            assert rec is not None
            records.append(rec)

    assert spec_calls == 3 * n_param_tensors
    for r in records:
        expected_flag = r["step"] % 5 == 0
        assert r["weights"]["spectral_computed"] is expected_flag
        for snap in r["organs"].values():
            assert snap["weight_spectral_approx"] > 0.0


def test_p1_6_bounded_history_deque_o1_ram_and_exact_online_summarize_run(
    tmp_path: Path,
) -> None:
    """P1-6: `_history` acotado (`deque`) + resumen online exacto + JSONL sin pérdida."""
    from collections import deque

    default_cfg = TelemetryConfig(log_dir=str(tmp_path / "def_hist"))
    assert default_cfg.max_history_records == 2048
    with pytest.raises(ValueError):
        TelemetryConfig(log_dir=str(tmp_path / "bad_hist"), max_history_records=0)

    rng_a = np.random.default_rng(606)
    rng_b = np.random.default_rng(606)
    m_full = _build_small_model("v07_moe_double_loop", seed=606)
    m_bnd = _build_small_model("v07_moe_double_loop", seed=606)
    opt_full = build_optimizer(m_full, lr=2e-3)
    opt_bnd = build_optimizer(m_bnd, lr=2e-3)

    cfg_full = TelemetryConfig(
        log_dir=str(tmp_path / "hist_full"),
        log_every_steps=1,
        max_history_records=100,
        enable_tensorboard=False,
    )
    cfg_bnd = TelemetryConfig(
        log_dir=str(tmp_path / "hist_bounded"),
        log_every_steps=1,
        max_history_records=4,
        enable_tensorboard=False,
    )

    with (
        ResearchTelemetryMonitor(m_full, cfg_full, optimizer=opt_full) as mon_full,
        ResearchTelemetryMonitor(m_bnd, cfg_bnd, optimizer=opt_bnd) as mon_bnd,
    ):
        assert isinstance(mon_bnd._history, deque)
        assert mon_bnd._history.maxlen == 4

        for step in range(15):
            b_f = _synthetic_session(rng_a, n_assets=12)
            b_b = _synthetic_session(rng_b, n_assets=12)

            mon_full.begin_step(step)
            opt_full.zero_grad(set_to_none=True)
            o_f = m_full.forward_session(b_f, enable_early_exit=False)
            l_f = (o_f["pred_return"][:, 0] - b_f["target_returns"][:, 0]).pow(2).mean()
            l_f.backward()
            mon_full.clip_grad_norm(step, max_norm=2.0)
            opt_full.step()

            mon_bnd.begin_step(step)
            opt_bnd.zero_grad(set_to_none=True)
            o_b = m_bnd.forward_session(b_b, enable_early_exit=False)
            l_b = (o_b["pred_return"][:, 0] - b_b["target_returns"][:, 0]).pow(2).mean()
            l_b.backward()
            mon_bnd.clip_grad_norm(step, max_norm=2.0)
            opt_bnd.step()

            # Inyectar anomalía temprana en paso 1 (expulsado de `deque(maxlen=4)` en paso 14!)
            pt1_diag = (
                dict(alpha_min=0.001, alpha_mean=0.02, eta_max=0.99, theta_max=0.60)
                if step == 1
                else dict(alpha_min=0.10, alpha_mean=0.20, eta_max=0.80, theta_max=0.15)
            )
            mon_full.end_step(
                step, loss_dict=dict(total=l_f), model_output=o_f, titans_update_diag=pt1_diag
            )
            mon_bnd.end_step(
                step, loss_dict=dict(total=l_b), model_output=o_b, titans_update_diag=pt1_diag
            )

        assert len(mon_bnd.history) == 4
        assert [r["step"] for r in mon_bnd.history] == [11, 12, 13, 14]
        sum_full = mon_full.summarize_run()
        sum_bnd = mon_bnd.summarize_run()

    # Disco conserva los 15 registros completos sin pérdida
    disk_recs = read_telemetry_jsonl(Path(cfg_bnd.log_dir) / cfg_bnd.jsonl_filename)
    assert len(disk_recs) == 15
    assert [r["step"] for r in disk_recs] == list(range(15))

    # `summarize_run()` online es 100 % exacto frente al historial completo
    assert sum_bnd["n_logged_steps"] == sum_full["n_logged_steps"] == 15
    assert sum_bnd["total_steps_seen"] == sum_full["total_steps_seen"] == 15
    assert sum_bnd["last_step"] == sum_full["last_step"] == 14
    assert sum_bnd["healthy"] is False
    assert sum_bnd["anomaly_flags"] == sum_full["anomaly_flags"]
    assert sum_bnd["anomaly_flags"]["pt1_bound_violation"] is True
    assert sum_bnd["affected_organs"] == sum_full["affected_organs"]

    for organ, s_full in sum_full["organ_summary"].items():
        s_bnd = sum_bnd["organ_summary"][organ]
        for k, v_full in s_full.items():
            if v_full is None:
                assert s_bnd[k] is None
            else:
                assert s_bnd[k] == pytest.approx(v_full, rel=1e-6, abs=1e-9)

    assert sum_bnd["loss_summary"]["total"] == pytest.approx(
        sum_full["loss_summary"]["total"], rel=1e-6, abs=1e-9
    )
    for k, v_full in sum_full["domain_summary"].items():
        if v_full is None:
            assert sum_bnd["domain_summary"][k] is None
        else:
            assert sum_bnd["domain_summary"][k] == pytest.approx(v_full, rel=1e-6, abs=1e-9)


def test_wave2_v2_four_families_zero_unclassified_params_and_batched_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wave-2: 0 parámetros 'other' en las 4 familias v2 + telemetría 1-sync F3 CF y F4 10-tools."""
    from mars_titan.models.mmarrs.cognitive_core import TOOL_NAMES_V2
    from mars_titan.models.mmarrs.model import build_family_model
    from mars_titan.models.mmarrs.mor_loom_pc import LoopedCrossAssetTransformer

    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
        n_cortical_areas=8,
        n_hidden_slots=2,
        n_hyperedges=6,
        hrr_max_slots=32,
        allow_short_synthetic=True,
    )

    v2_flagships = (
        ("t15b_full_v2", "titans_original"),
        ("nb_v2_full", "holographic_brain"),
        ("nb_v2_moe_soft_k3", "holographic_brain"),
        ("v14c_tied_xasset_cf", "mmarrs_titan"),
        ("cog_v2_10tools_soft_k3", "titans_cognitive_core"),
    )
    for vid, fam in v2_flagships:
        model = build_family_model(vid, market="US", family=fam, **dims)
        unclassified = [
            name
            for name, _ in model.named_parameters()
            if classify_param_organ(name) not in ARCHITECTURAL_ORGANS
        ]
        assert unclassified == [], f"{vid} tiene parámetros sin clasificar: {unclassified}"

    # Verificar separación fina de órganos en F3 v2 LoopedCrossAssetTransformer
    m_f3 = build_family_model("v14c_tied_xasset_cf", market="US", **dims)
    f3_organs = {classify_param_organ(n) for n, _ in m_f3.named_parameters()}
    assert {"moe_routers", "moe_experts", "contractive_loop", "cross_modal_fusion"}.issubset(
        f3_organs
    )
    standalone_loop = LoopedCrossAssetTransformer(hidden_dim=16, context_dim=10, max_loops=3)
    assert all(
        classify_param_organ(n) in ARCHITECTURAL_ORGANS
        for n, _ in standalone_loop.named_parameters()
    )
    standalone_cog_v2 = TitansCognitiveCore(
        hidden_dim=16, context_dim=10, toolbox_version="v2_10tools", use_context_engineer=False
    )
    assert all(
        classify_param_organ(n) == "cognitive_core" for n, _ in standalone_cog_v2.named_parameters()
    )

    # Telemetría agrupada 1-sync para F3 v2 (Loop/CF) y F4 v2 (10-tool specialization) -> .jsonl.gz
    rng = np.random.default_rng(2026)
    batch = _synthetic_session(rng, n_assets=16)
    batch["asset_ids"] = torch.arange(16, dtype=torch.long)

    m_f4 = build_family_model("cog_v2_10tools_soft_k3", market="US", **dims)
    m_f3.eval()
    m_f4.eval()
    with torch.no_grad():
        out_f3 = m_f3.forward_session(batch)
        out_f4 = m_f4.forward_session(batch)

    orig_tolist = torch.Tensor.tolist
    tolist_calls = 0

    def _counting_tolist(self: torch.Tensor):
        nonlocal tolist_calls
        tolist_calls += 1
        return orig_tolist(self)

    def _forbid_item(self: torch.Tensor):
        raise AssertionError("`.item()` prohibido en `_collect_domain_telemetry`")

    cfg_f3 = TelemetryConfig(
        log_dir=str(tmp_path / "tel_f3_v2"), log_every_steps=1, enable_tensorboard=False
    )
    with ResearchTelemetryMonitor(m_f3, cfg_f3) as mon_f3:
        monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
        monkeypatch.setattr(torch.Tensor, "tolist", _counting_tolist)
        tolist_calls = 0
        dom_f3 = mon_f3._collect_domain_telemetry(
            model_output=out_f3, titans_update_diag=None, cognitive_core_diag=None
        )
        assert tolist_calls == 1
        monkeypatch.undo()
        rec_f3 = mon_f3.end_step(0, loss_dict=dict(total=torch.tensor(1.0)), model_output=out_f3)

    assert rec_f3 is not None
    for k in (
        "loop_halt_at_k1_pct",
        "cf_fragility_k0",
        "cf_fragility_final",
        "cf_sign_flip_rate_final",
        "cf_worst_dd_final",
        "loop_step_router_js_div",
        "landmark_attn_entropy",
        "ponder_kl_loss",
    ):
        assert dom_f3[k] is not None and math.isfinite(float(dom_f3[k])), f"Falta campo F3 v2: {k}"
    assert len(read_telemetry_jsonl(mon_f3.jsonl_path)) == 1

    cfg_f4 = TelemetryConfig(
        log_dir=str(tmp_path / "tel_f4_v2"), log_every_steps=1, enable_tensorboard=False
    )
    with ResearchTelemetryMonitor(m_f4, cfg_f4) as mon_f4:
        monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
        monkeypatch.setattr(torch.Tensor, "tolist", _counting_tolist)
        tolist_calls = 0
        dom_f4 = mon_f4._collect_domain_telemetry(
            model_output=out_f4, titans_update_diag=None, cognitive_core_diag=None
        )
        assert tolist_calls == 1
        monkeypatch.undo()
        rec_f4 = mon_f4.end_step(0, loss_dict=dict(total=torch.tensor(1.0)), model_output=out_f4)

    assert rec_f4 is not None
    assert dom_f4["cognitive_toolbox_version"] == "v2_10tools"
    assert set(dom_f4["cognitive_tool_usage"].keys()) == set(TOOL_NAMES_V2)
    assert len(dom_f4["cognitive_tool_usage"]) == 10
    assert sum(dom_f4["cognitive_tool_usage"].values()) == pytest.approx(1.0, abs=1e-4)
    for k in (
        "cognitive_tool_load_max_min_ratio",
        "cognitive_tool_active_count",
        "cognitive_tool_corr_max",
        "cognitive_expected_steps_mean",
        "cognitive_expected_steps_easy_q25",
        "cognitive_expected_steps_hard_q75",
        "cognitive_cf_sign_flip_k0",
        "cognitive_cf_sign_flip_final",
        "cognitive_conflict_mean",
        "cognitive_scratchpad_novelty_mean",
    ):
        assert dom_f4[k] is not None and math.isfinite(float(dom_f4[k])), f"Falta campo F4 v2: {k}"
    assert len(dom_f4["cognitive_tool_weights_by_step"]) == 3
    assert len(dom_f4["cognitive_tool_weights_by_step"][0]) == 10
    recs_f4 = read_telemetry_jsonl(mon_f4.jsonl_path)
    assert len(recs_f4) == 1
    assert set(recs_f4[0]["domain"]["cognitive_tool_usage"].keys()) == set(TOOL_NAMES_V2)
