"""Arquitectura unificada `MMARRSTitan`, fábrica de las 4 familias independientes y BPTT causal.

Integra limpiamente todos los componentes SOTA con **cero host-syncs GPU**
(`audit_zero_host_syncs == 0`) y separación estricta entre las **4 familias independientes**:
  - Familia 1 (`titans_original`): Presets Titans neuromodulados / MAC + rangos gaussianos + TabM.
  - Familia 2 (`holographic_brain`): `StandaloneHolographicBrainModel` (`neuro_brain_sota.py`).
  - Familia 3 (`mmarrs_titan`): 16 variaciones emparejadas `v00..v13c` (incluye
    `v13b_mmarrs_multimodal_sota` con `CrossModalConflictFusion` + `VICReg/Barlow SSL` +
    `SkewStudentTDistributionHead`, y `v13c_mmarrs_noic_pinball_twin` FM-7).
  - Familia 4 (`titans_cognitive_core`): `TitansCognitiveCore` weight-tied con `head` compartido,
    `ponder_kl_loss` y notas persistentes `state["cognitive_notes"]`.

Soporta:
  - Alineación canónica por `asset_ids: int64[N]` en paneles irregulares (P0-1).
  - `self.target_encoder = nn.Linear(1, d_model)` + BPTT truncado (`keep_grad=True`, P0-2).
  - Normalización transversal por rangos gaussianos (`gaussian_rank_features`) + bypass `TabM`
    ligero + `fused_cross_sectional_zscore_clip` (P1-1 / P0-7).
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features
from mars_titan.evaluation.metacognitive_agent import (
    ConvexPortfolioOptimizerPPlus,
    MetacognitiveGovernor,
)

from .blackwell_kernels import fused_cross_sectional_zscore_clip
from .cognitive_core import TitansCognitiveCore
from .config import VARIATION_ALIASES, VARIATION_REGISTRY, MMARRSConfig, variation_config
from .dl_training_sota import build_norm_layer
from .frontier_modules_sota import (
    MissingnessAwareSetModalityFusion,
    ObservedSlotEncoder,
    PolarAlphaHead,
)
from .granular_heads import MultiHorizonGranularHead
from .hypergraph_hrr import (
    HolographicBrain3L,
    HRRModernHopfieldMemory,
    RMTSpilloverHypergraph,
)
from .mor_loom_pc import (
    FinancialEnergyVerifier,
    LoopedCrossAssetTransformer,
    PCTRMRecursiveLoop,
)
from .multimodal_sota import (
    CrossModalConflictFusion,
    SelfSupervisedMultimodalAlignmentLoss,
    SkewStudentTDistributionHead,
)
from .neuro_brain_sota import (
    NEURO_BRAIN_ABLATIONS,
    NeuroBrainConfig,
    build_neuro_brain_ablation,
)
from .neuromod_memory import DualAxisTitansMemory, NeuromodulatedTitansMemory
from .tfm_catalog import TFM_VARIANT_ALIASES, TFM_VARIANT_CATALOG

# Mapeo de los IDs canónicos de Familia 3 en `tfm_catalog.py` a `VARIATION_REGISTRY`
_CATALOG_TO_MMARRS_VARIATION: dict[str, str] = {
    "v00_dense_iso_flop": "v00_dense_baseline",
    "v01_dense_wide_iso_param": "v01_dense_wide_iso_param",
    "v02_neuromod_only": "v02_titans_neuromod_only",
    "v03_moe_only_k1": "v03_moe_only_k1",
    "v04_moe_uniform_control": "v04_moe_uniform_control",
    "v05_moe_time_shuffled_placebo": "v05_moe_shuffled_ctx_placebo",
    "v06_loop_only_k3": "v06_double_loop_pc_trm_only",
    "v07_moe_loop_k3": "v07_moe_double_loop",
    "v08_holo_bridge_comparison": "v08_holographic_3l_hrr_only",
    "v09_holo_moe_bridge_comparison": "v09_holographic_3l_moe_loop",
    "v10_rmt_hypergraph_only": "v10_cross_asset_hypergraph_only",
    "v11_moe_agent_p_plus": "v11_moe_plus_agent_core",
    "v12_moe_loop_p_plus": "v12_moe_loop_plus_agent_core",
    "v13_mmarrs_full": "v13_mmarrs_titan_full",
    "v13b_mmarrs_multimodal_sota": "v13b_mmarrs_multimodal_sota",
    "v13c_mmarrs_noic_pinball_twin": "v13c_mmarrs_noic_pinball_twin",
}


class _ModalityEncoder(nn.Module):
    """Codificador multimodal compacto con normalización y activación SiLU."""

    def __init__(
        self, in_dim: int, hidden_dim: int, *, norm_kind: str = "layernorm"
    ) -> None:
        super().__init__()
        norm_layer: nn.Module = (
            nn.LayerNorm(in_dim)
            if norm_kind == "layernorm"
            else build_norm_layer(norm_kind, in_dim)  # type: ignore[arg-type]
        )
        self.net = nn.Sequential(
            norm_layer,
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TabMTechnicalSkip(nn.Module):
    """Rama residual ligera estilo TabM (BatchEnsemble K miembros) sobre precios + física v4."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        n_members: int = 4,
        norm_kind: str = "layernorm",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_members = max(1, int(n_members))
        mid_dim = max(8, hidden_dim // 2)

        self.in_norm: nn.Module = (
            nn.LayerNorm(in_dim)
            if norm_kind == "layernorm"
            else build_norm_layer(norm_kind, in_dim)  # type: ignore[arg-type]
        )
        self.r_in = nn.Parameter(
            torch.ones(self.n_members, in_dim) + 0.05 * torch.randn(self.n_members, in_dim)
        )
        self.s_mid = nn.Parameter(
            torch.ones(self.n_members, mid_dim) + 0.05 * torch.randn(self.n_members, mid_dim)
        )
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, hidden_dim)
        self.skip_scale = nn.Parameter(torch.full((), 0.15, dtype=torch.float32))

    def forward(self, tech_x: torch.Tensor) -> torch.Tensor:
        # [N, F] -> [N, K, F] mediante adaptadores BatchEnsemble de rango 1
        xn = self.in_norm(tech_x)
        x_k = xn.unsqueeze(1) * self.r_in.to(dtype=xn.dtype).unsqueeze(0)
        h_k = F.silu(self.fc1(x_k)) * self.s_mid.to(dtype=xn.dtype).unsqueeze(0)
        out = self.fc2(h_k).mean(dim=1)
        return self.skip_scale.to(dtype=xn.dtype) * out


class MMARRSTitan(nn.Module):
    """Modelo unificado `MMARRS-TITAN` (soporta Familias 1, 3 y 4 limpiamente separadas)."""

    def __init__(self, config: MMARRSConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        nk = config.norm_kind

        # 1) Codificadores 5 modalidades + física v4 (canal dual v2 / ObservedSlot opcional)
        rank_mult = 2 if config.use_dual_level_rank_input else 1
        self.enc_prices = _ModalityEncoder(config.prices_dim * rank_mult, d, norm_kind=nk)
        if config.use_set_modality_fusion and config.fundamentals_dim == 78:
            self.enc_fundamentals: nn.Module = ObservedSlotEncoder(n_slots=26, hidden_dim=d)
        else:
            self.enc_fundamentals = _ModalityEncoder(
                config.fundamentals_dim * rank_mult, d, norm_kind=nk
            )
        if config.use_set_modality_fusion and config.macro_dim == 420:
            self.enc_macro: nn.Module = ObservedSlotEncoder(n_slots=140, hidden_dim=d)
        else:
            self.enc_macro = _ModalityEncoder(config.macro_dim, d, norm_kind=nk)
        self.enc_news = _ModalityEncoder(config.news_dim, d, norm_kind=nk)
        self.enc_charts = _ModalityEncoder(config.charts_dim, d, norm_kind=nk)
        self.enc_physics = _ModalityEncoder(config.physics_dim * rank_mult, d, norm_kind=nk)
        self.regime_proj = nn.Linear(config.context_dim, d)

        # P0-2: codificador aprendido de targets maduros (reemplaza vector rango-1 constante)
        self.target_encoder = nn.Linear(1, d)
        nn.init.normal_(self.target_encoder.weight, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.target_encoder.bias)

        # P1-1: bypass técnico ligero TabM sobre precios+física (`enc_` -> órgano encoders)
        self.enc_tabm_skip = (
            _TabMTechnicalSkip(
                (config.prices_dim + config.physics_dim) * rank_mult,
                d,
                n_members=config.tabm_members,
                norm_kind=nk,
            )
            if config.use_tabm_skip
            else None
        )

        # Fusión multimodal: CrossModalConflictFusion (v13b), SetModalityFusion (opt-in) o uniforme
        self.fusion_norm: nn.Module = (
            nn.LayerNorm(d)
            if nk == "layernorm"
            else build_norm_layer(nk, d)  # type: ignore[arg-type]
        )
        self.fusion_proj = nn.Linear(2 * d, d)
        n_heads_mm = 4 if d % 4 == 0 else (2 if d % 2 == 0 else 1)
        self.conflict_fusion = (
            CrossModalConflictFusion(d, n_modalities=5, n_heads=n_heads_mm)
            if config.use_multimodal_sota
            else None
        )
        # Jamás sobrescribe F3 `CrossModalConflictFusion` ni F4 `ContextEngineerGPU`!
        uses_cog_ce = bool(config.use_cognitive_core and config.cognitive_use_context_engineer)
        self.set_modality_fusion: MissingnessAwareSetModalityFusion | None = (
            MissingnessAwareSetModalityFusion(
                hidden_dim=d,
                n_aux_modalities=4,
                n_heads=n_heads_mm,
                mnar_placebo_shuffle=config.set_fusion_mnar_placebo,
            )
            if (
                config.use_set_modality_fusion
                and self.conflict_fusion is None
                and not uses_cog_ce
            )
            else None
        )
        self.ssl_alignment = (
            SelfSupervisedMultimodalAlignmentLoss(
                d,
                n_modalities=5,
                n_heads=n_heads_mm,
                ssl_weight=config.ssl_weight,
            )
            if config.use_multimodal_sota
            else None
        )
        self.skew_t_head = (
            SkewStudentTDistributionHead(
                d,
                horizons=config.horizons,
                quantile_levels=config.quantile_levels,
            )
            if config.use_multimodal_sota
            else None
        )

        # 2) Componentes opcionales gobernados exactamente por `config`
        self.neuromod_memory: NeuromodulatedTitansMemory | DualAxisTitansMemory | None
        if config.use_neuromod_memory:
            if config.use_dual_axis_titans:
                n_heads_mem = 4 if d % 4 == 0 else (2 if d % 2 == 0 else 1)
                self.neuromod_memory = DualAxisTitansMemory(
                    d,
                    n_heads=n_heads_mem,
                    pt1_alpha_lo=config.pt1_alpha_lo,
                    pt1_eta_hi=config.pt1_eta_hi,
                    pt1_theta_max=config.pt1_theta_max,
                    huber_delta=config.huber_delta,
                    use_fused_blackwell_kernels=config.use_fused_blackwell_kernels,
                )
            else:
                self.neuromod_memory = NeuromodulatedTitansMemory(
                    d,
                    pt1_alpha_lo=config.pt1_alpha_lo,
                    pt1_eta_hi=config.pt1_eta_hi,
                    pt1_theta_max=config.pt1_theta_max,
                    huber_delta=config.huber_delta,
                    use_fused_blackwell_kernels=config.use_fused_blackwell_kernels,
                )
        else:
            self.neuromod_memory = None
        self.hypergraph = (
            RMTSpilloverHypergraph(
                d, n_hyperedges=config.n_hyperedges, rmt_dim=config.rmt_dim
            )
            if config.use_hypergraph
            else None
        )
        self.holographic_3l = (
            HolographicBrain3L(
                d,
                n_modalities=5,
                n_cortical_areas=config.n_cortical_areas,
                n_hidden_slots=config.n_hidden_slots,
                n_hyperedges=config.n_hyperedges,
                regime_dim=config.context_dim,
                rmt_dim=config.rmt_dim,
            )
            if config.use_holographic_3l
            else None
        )
        self.hrr_hopfield = (
            HRRModernHopfieldMemory(
                d,
                max_slots=config.hrr_max_slots,
                k_proto=config.hrr_k_proto,
            )
            if config.use_hrr_hopfield
            else None
        )

        # 3) Tronco: Familia 4 (`TitansCognitiveCore`) O Familia 1/3 (`Looped` v2 / `PCTRM` v1)
        self.recursive_core: PCTRMRecursiveLoop | LoopedCrossAssetTransformer | None
        if config.use_cognitive_core:
            self.cognitive_core: TitansCognitiveCore | None = TitansCognitiveCore(
                hidden_dim=d,
                context_dim=config.context_dim,
                thinking_steps=config.max_loops,
                tool_mode=config.cognitive_tool_mode,
                toolbox_version=config.cognitive_toolbox_version,
                n_modalities=5,
                n_hyperedges=config.n_hyperedges,
                rmt_dim=config.rmt_dim,
                horizons=config.horizons,
                lipschitz_bound=config.lipschitz_bound,
                ponder_weight=config.cognitive_ponder_weight,
                use_context_engineer=config.cognitive_use_context_engineer,
                cf_placebo_mode=config.cognitive_cf_placebo,
                norm_kind=nk,
            )
            # Puente MoE opcional únicamente si se solicita explícitamente (`cog_core_plus_moe_k3`)
            self.recursive_core = (
                PCTRMRecursiveLoop(
                    hidden_dim=d,
                    context_dim=config.context_dim,
                    max_loops=1,
                    use_moe=True,
                    n_shared_experts=config.n_shared_experts,
                    n_routed_experts=config.n_routed_experts,
                    moe_mode=config.moe_mode,
                    ffn_mult=config.ffn_mult,
                    lipschitz_bound=config.lipschitz_bound,
                    early_exit_tol=config.early_exit_tol,
                    use_fused_blackwell_kernels=config.use_fused_blackwell_kernels,
                )
                if config.use_moe
                else None
            )
        else:
            self.cognitive_core = None
            if config.use_looped_transformer:
                self.recursive_core = LoopedCrossAssetTransformer(
                    hidden_dim=d,
                    context_dim=config.context_dim,
                    max_loops=config.max_loops,
                    use_cross_asset_attn=config.loop_use_cross_attn,
                    use_diff_xasset_attn=config.use_diff_xasset_attn,
                    diff_xasset_zero_lambda_control=config.diff_xasset_zero_lambda_control,
                    n_landmarks=config.loop_n_landmarks,
                    use_moe=config.use_moe,
                    n_shared_experts=config.n_shared_experts,
                    n_routed_experts=config.n_routed_experts,
                    moe_mode=config.moe_mode,
                    ffn_mult=config.ffn_mult,
                    lipschitz_bound=config.lipschitz_bound,
                    use_counterfactual=config.use_counterfactual_loop,
                    placebo_mode=config.cf_placebo_mode,
                    use_fused_blackwell_kernels=config.use_fused_blackwell_kernels,
                    norm_kind=nk,
                )
            else:
                n_loops = config.max_loops if config.use_double_loop else 1
                self.recursive_core = PCTRMRecursiveLoop(
                    hidden_dim=d,
                    context_dim=config.context_dim,
                    max_loops=n_loops,
                    use_moe=config.use_moe,
                    n_shared_experts=config.n_shared_experts,
                    n_routed_experts=config.n_routed_experts,
                    moe_mode=config.moe_mode,
                    ffn_mult=config.ffn_mult,
                    lipschitz_bound=config.lipschitz_bound,
                    early_exit_tol=config.early_exit_tol,
                    use_fused_blackwell_kernels=config.use_fused_blackwell_kernels,
                )

        # 4) Verificador estructural + Cabezas Granulares Multi-Horizonte (compartida con CogCore!)
        self.verifier = FinancialEnergyVerifier(horizons=config.horizons)
        self.head = MultiHorizonGranularHead(
            hidden_dim=d,
            horizons=config.horizons,
            quantile_levels=config.quantile_levels,
            exposures_dim=config.exposures_dim,
            tabm_k=config.tabm_k,
            norm_kind=nk,
        )
        self.polar_alpha_head: PolarAlphaHead | None = (
            PolarAlphaHead(n_horizons=len(config.horizons), default_vol=0.015)
            if config.use_polar_alpha_head
            else None
        )

        # 5) Núcleo Agéntico Metacognitivo (A5–A8 + Cartera Convexa P+)
        self.agent_governor = (
            MetacognitiveGovernor(n_specialists=3, epistemic_kappa=0.4, ood_threshold=2.5)
            if config.use_agent_core
            else None
        )
        self.portfolio_opt = (
            ConvexPortfolioOptimizerPPlus(
                risk_aversion=2.0, factor_penalty=15.0, l1_cost=0.001, impact_cost=0.01
            )
            if config.use_agent_core
            else None
        )
        # Buffers en dispositivo para actualizar el gobernador sin host-syncs en GPU
        self.register_buffer("_gov_weights", torch.full((3,), 1.0 / 3.0, dtype=torch.float32))
        self.register_buffer("_gov_ewma_ic", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("_gov_peak_ic", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("_gov_cusum", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("_gov_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_spectral_buffers(self) -> None:
        """Actualiza buffers `_SpectralLinear` en submódulos activos tras `optimizer.step()`."""
        for sub in (self.recursive_core, self.cognitive_core):
            fn = getattr(sub, "update_spectral_buffers", None)
            if callable(fn):
                fn()

    @staticmethod
    def detach_state(state: dict[str, object] | None) -> dict[str, object] | None:
        """Desconecta tensores persistentes del grafo autograd al final de un bloque BPTT."""
        if state is None:
            return None
        out: dict[str, object] = {}
        for k, v in state.items():
            if k == "titans_state" and isinstance(v, dict):
                out[k] = NeuromodulatedTitansMemory.detach_state(v)
            elif isinstance(v, torch.Tensor):
                out[k] = v.detach()
            elif isinstance(v, list):
                out[k] = [t.detach() if isinstance(t, torch.Tensor) else t for t in v]
            else:
                out[k] = v
        return out

    @staticmethod
    def _align_vector_by_asset_ids(
        cur_ids: torch.Tensor,
        prev_ids: torch.Tensor,
        prev_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Alinea un vector `[M]` (`w_prev`) a `cur_ids [N]` vía `searchsorted` (IPOs -> 0)."""
        device, dtype = prev_vec.device, prev_vec.dtype
        c_ids = cur_ids.to(device=device, dtype=torch.long).reshape(-1)
        p_ids = prev_ids.to(device=device, dtype=torch.long).reshape(-1)
        if p_ids.shape[0] == 0:
            return torch.zeros(c_ids.shape[0], device=device, dtype=dtype)
        sort_idx = torch.argsort(p_ids)
        p_sorted = p_ids[sort_idx]
        v_sorted = prev_vec[sort_idx]
        pos = torch.searchsorted(p_sorted, c_ids)
        pos_safe = pos.clamp(max=p_sorted.shape[0] - 1)
        matched = (pos < p_sorted.shape[0]) & (p_sorted[pos_safe] == c_ids)
        return torch.where(matched, v_sorted[pos_safe], torch.zeros_like(v_sorted[pos_safe]))

    def _preprocess_cross_section(
        self,
        x: torch.Tensor,
        *,
        apply_gaussian_rank: bool,
        preserve_nonfinite: bool = False,
        present_row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Higiene transversal sin syncs: canal dual [z, g_rank] v2 o rango gaussiano/zscore v1."""
        is_fin = torch.isfinite(x)
        if present_row_mask is not None:
            row_m = present_row_mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1)
            valid_mask = is_fin & row_m
        else:
            row_m = None
            valid_mask = is_fin
        clean = torch.where(valid_mask, x, torch.zeros_like(x))
        if apply_gaussian_rank and self.config.use_dual_level_rank_input:
            clipped_z = torch.where(
                valid_mask, clean.clamp(-8.0, 8.0), torch.zeros_like(clean)
            )
            g_rank = gaussian_rank_features(clean, mask=valid_mask, clip_val=8.0)
            if preserve_nonfinite:
                keep_nf = ((~is_fin) & row_m) if row_m is not None else (~is_fin)
                clipped_z = torch.where(keep_nf, x, clipped_z)
            return torch.cat([clipped_z, g_rank], dim=-1)
        if apply_gaussian_rank and self.config.use_gaussian_ranks and clean.shape[0] >= 3:
            clean = gaussian_rank_features(clean)
        elif self.config.use_fused_blackwell_kernels and clean.shape[0] >= 2:
            clean = fused_cross_sectional_zscore_clip(clean, clip_val=8.0, use_triton=True)
        if preserve_nonfinite:
            clean = torch.where(is_fin, clean, x)
        return clean

    def _encode_modalities(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Codifica las 5 modalidades + precios con aislamiento seguro ante ausentes."""
        prices_raw = batch["prices"]
        n_assets = prices_raw.shape[0]
        device, dtype = prices_raw.device, prices_raw.dtype
        c = self.config

        mask = batch.get("modality_mask")
        if mask is None:
            mask = torch.ones((n_assets, 5), device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)
        mask_bool = mask > 0.5

        def _get(key: str, dim: int) -> torch.Tensor:
            val = batch.get(key)
            if val is None:
                return torch.zeros((n_assets, dim), device=device, dtype=dtype)
            t = val.to(device=device, dtype=dtype)
            return torch.where(torch.isfinite(t), t, torch.zeros_like(t))

        # Preservar NaN en precios presentes (`SessionHealthGate`), aislando ausentes (`mask==0`)
        prices_t = prices_raw.to(device=device, dtype=dtype)
        prices_masked = torch.where(mask_bool[:, 0:1], prices_t, torch.zeros_like(prices_t))
        prices_in = self._preprocess_cross_section(
            prices_masked,
            apply_gaussian_rank=True,
            preserve_nonfinite=True,
            present_row_mask=mask_bool[:, 0] if c.use_dual_level_rank_input else None,
        )
        if isinstance(self.enc_fundamentals, ObservedSlotEncoder):
            f_raw = batch.get("fundamentals")
            if f_raw is None:
                fund_in = torch.zeros((n_assets, c.fundamentals_dim), device=device, dtype=dtype)
            else:
                fund_in = torch.where(
                    mask_bool[:, 1:2],
                    f_raw.to(device=device, dtype=dtype),
                    torch.zeros((n_assets, c.fundamentals_dim), device=device, dtype=dtype),
                )
        else:
            fund_raw = _get("fundamentals", c.fundamentals_dim)
            fund_in = self._preprocess_cross_section(
                fund_raw,
                apply_gaussian_rank=True,
                present_row_mask=mask_bool[:, 1] if c.use_dual_level_rank_input else None,
            )
        # Macro es constante entre activos en la sesión -> NO aplicar rango gaussiano transversal
        if isinstance(self.enc_macro, ObservedSlotEncoder):
            m_raw = batch.get("macro")
            if m_raw is None:
                macro_in = torch.zeros((n_assets, c.macro_dim), device=device, dtype=dtype)
            else:
                macro_in = torch.where(
                    mask_bool[:, 2:3],
                    m_raw.to(device=device, dtype=dtype),
                    torch.zeros((n_assets, c.macro_dim), device=device, dtype=dtype),
                )
        else:
            macro_in = _get("macro", c.macro_dim)
        news_in = _get("news", c.news_dim)
        charts_in = _get("charts", c.charts_dim)

        m0 = self.enc_prices(prices_in)
        m1 = self.enc_fundamentals(fund_in)
        m2 = self.enc_macro(macro_in)
        m3 = self.enc_news(news_in)
        m4 = self.enc_charts(charts_in)
        tokens = torch.stack([m0, m1, m2, m3, m4], dim=1)  # [N, 5, D]

        tokens = torch.where(mask_bool.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens, mask, prices_in

    def forward_session(
        self,
        batch: dict[str, torch.Tensor],
        state: dict[str, object] | None = None,
        *,
        enable_early_exit: bool | None = None,
    ) -> dict[str, object]:
        """Ejecuta una sesión completa de `N` activos (100% causal, cero host-syncs GPU)."""
        st: dict[str, object] = {} if state is None else dict(state)
        mod_tokens, mod_mask, prices_in = self._encode_modalities(batch)
        n_assets = mod_tokens.shape[0]
        device, dtype = mod_tokens.device, mod_tokens.dtype
        c = self.config

        phys_raw = batch.get("physics_v4")
        if phys_raw is None:
            phys_zeros = torch.zeros((n_assets, c.physics_dim), device=device, dtype=dtype)
            phys_in = (
                torch.cat([phys_zeros, phys_zeros], dim=-1)
                if c.use_dual_level_rank_input
                else phys_zeros
            )
        else:
            phys_in = self._preprocess_cross_section(
                phys_raw.to(device=device, dtype=dtype), apply_gaussian_rank=True
            )
        phys_emb = self.enc_physics(phys_in)

        ctx = batch.get("high_snr_context")
        if ctx is None:
            ctx = torch.zeros((n_assets, c.context_dim), device=device, dtype=dtype)
        else:
            ctx = ctx.to(device=device, dtype=dtype)
            ctx = torch.where(torch.isfinite(ctx), ctx, torch.zeros_like(ctx))

        regime_vec = batch.get("regime_vec")
        if regime_vec is None:
            regime_vec = ctx.mean(dim=0)
        else:
            regime_vec = regime_vec.to(device=device, dtype=dtype)

        rmt_loadings = batch.get("rmt_loadings")
        exposures = batch.get("exposures")
        ne_signal = batch.get("ne_signal")
        if ne_signal is None and isinstance(st.get("last_ne_signal"), torch.Tensor):
            ne_signal = st["last_ne_signal"]  # type: ignore[assignment]
        asset_ids = batch.get("asset_ids")
        horizon_mask = batch.get("horizon_mask")
        placebo_ctx = batch.get("placebo_high_snr_context")
        if placebo_ctx is None:
            placebo_ctx = batch.get("shuffled_context")
        if placebo_ctx is None and c.moe_mode == "shuffled_context":
            # Usar resumen de sesión previa del anillo causal si existe
            prev_ctx_ring = st.get("placebo_prev_ctx")
            if isinstance(prev_ctx_ring, torch.Tensor):
                placebo_ctx = prev_ctx_ring

        # Alinear `w_prev` canónico por `asset_ids` para F3 v2 (CF) y F4 v2 (Tool 4 FISTA)
        w_prev_core: torch.Tensor | None = None
        raw_batch_wp = batch.get("w_prev")
        if isinstance(raw_batch_wp, torch.Tensor) and raw_batch_wp.shape[0] == n_assets:
            w_prev_core = raw_batch_wp.detach().to(device=device, dtype=dtype)
        else:
            for wk_key, id_key in (
                ("last_tool_w_prev", "last_tool_w_prev_ids"),
                ("agent_w_prev", "agent_w_prev_ids"),
            ):
                wp_st = st.get(wk_key)
                wp_ids = st.get(id_key)
                if isinstance(wp_st, torch.Tensor):
                    if asset_ids is not None and isinstance(wp_ids, torch.Tensor):
                        w_prev_core = self._align_vector_by_asset_ids(
                            asset_ids, wp_ids, wp_st.to(device=device, dtype=dtype)
                        )
                    elif wp_st.shape[0] == n_assets:
                        w_prev_core = wp_st.to(device=device, dtype=dtype)
                    break

        # 1) Cerebro Holográfico 3 Niveles (si activo)
        holo_diag: dict[str, object] = {}
        if self.holographic_3l is not None:
            mod_tokens, holo_diag = self.holographic_3l(
                mod_tokens, mod_mask, regime_vec=regime_vec, rmt_loadings=rmt_loadings
            )

        # Fusión multimodal: CrossModalConflictFusion (v13b), SetModalityFusion (opt-in) o uniforme
        conflict_score: torch.Tensor | None = None
        ssl_loss: torch.Tensor | None = None
        set_fusion_diag: dict[str, torch.Tensor] = {}
        if self.conflict_fusion is not None:
            x0, conflict_score, _ = self.conflict_fusion(
                mod_tokens,
                mod_mask,
                phys_emb,
                modality_age=batch.get("modality_age"),
            )
            if self.ssl_alignment is not None and self.training:
                ssl_out = self.ssl_alignment(mod_tokens, mod_mask, physics_emb=phys_emb)
                ssl_loss = ssl_out["total"]
        elif self.set_modality_fusion is not None:
            core_anchor = self.fusion_norm(
                self.fusion_proj(torch.cat([mod_tokens[:, 0], phys_emb], dim=-1))
            )
            mod_age = batch.get("modality_age")
            aux_age = (
                mod_age[:, 1:5]
                if isinstance(mod_age, torch.Tensor) and mod_age.ndim == 2 and mod_age.shape[1] >= 5
                else None
            )
            x0, set_fusion_diag = self.set_modality_fusion(
                mod_tokens[:, 1:5],
                mod_mask[:, 1:5],
                core_anchor,
                modality_age=aux_age,
            )
        else:
            denom = mod_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mod_pooled = (mod_tokens * mod_mask.unsqueeze(-1)).sum(dim=1) / denom
            x0 = self.fusion_norm(self.fusion_proj(torch.cat([mod_pooled, phys_emb], dim=-1)))

        # Bypass técnico TabM (P1-1) + z-score clip Blackwell (P0-7)
        if self.enc_tabm_skip is not None:
            x0 = x0 + self.enc_tabm_skip(torch.cat([prices_in, phys_in], dim=-1))
        if c.use_fused_blackwell_kernels and n_assets >= 2:
            x0 = x0 + 0.05 * fused_cross_sectional_zscore_clip(x0, clip_val=8.0, use_triton=True)

        x0_pre_mem = x0

        # 2) Hipergrafo transversal RMT / Lead-Lag O(N*K) (si activo y no exclusivo en CogCore)
        hg_diag: dict[str, torch.Tensor] = {}
        if self.hypergraph is not None and self.cognitive_core is None:
            x0, hg_diag = self.hypergraph(x0, rmt_loadings=rmt_loadings)

        # 3) Lectura de Memoria Titans Neuromodulada ACh/NE con alineación `asset_ids` (P0-1)
        mem_diag: dict[str, torch.Tensor] = {}
        if self.neuromod_memory is not None:
            x0, mem_diag = self.neuromod_memory.read(
                x0,
                st.get("titans_state"),  # type: ignore[arg-type]
                ne_signal=ne_signal,
                asset_ids=asset_ids,
            )

        # 4) Recuperación Asociativa HRR-Modern-Hopfield + trazas multiescala + energía OOD
        ood_energy = torch.zeros(n_assets, device=device, dtype=dtype)
        if self.hrr_hopfield is not None:
            reg_key = self.regime_proj(regime_vec)
            hrr_out = self.hrr_hopfield.retrieve(x0, regime_key=reg_key)
            x0 = x0 + hrr_out["retrieved"]
            ood_energy = hrr_out["ood_energy"]

        # 5) Núcleo: Familia 4 (`TitansCognitiveCore`) O Familia 1/3 (`Looped` v2 / `PCTRM`)
        ponder_kl_loss: torch.Tensor | None = None
        weighted_ponder_loss: torch.Tensor | None = None
        cf_fragility_loss: torch.Tensor | None = None
        tool_aux_loss: torch.Tensor | None = None
        if self.cognitive_core is not None:
            reg_key_cog = self.regime_proj(regime_vec) if self.hrr_hopfield is not None else None
            cog_early_exit = True if enable_early_exit is None else bool(enable_early_exit)
            if c.cognitive_use_context_engineer:
                h_acc, step_hiddens, cog_diag = self.cognitive_core(
                    None,
                    ctx,
                    mod_tokens=mod_tokens,
                    modality_mask=mod_mask,
                    physics_emb=phys_emb,
                    modality_age=batch.get("modality_age"),
                    state=st,
                    exposures=exposures,
                    rmt_loadings=rmt_loadings,
                    hrr_memory=self.hrr_hopfield,
                    hypergraph=self.hypergraph,
                    head=self.head,
                    w_prev=w_prev_core,
                    regime_key=reg_key_cog,
                    ood_energy=ood_energy,
                    horizon_mask=horizon_mask,
                    cf_placebo_mode=c.cognitive_cf_placebo,
                    enable_early_exit=cog_early_exit,
                )
            else:
                h_acc, step_hiddens, cog_diag = self.cognitive_core(
                    x0,
                    ctx,
                    state=st,
                    exposures=exposures,
                    rmt_loadings=rmt_loadings,
                    hrr_memory=self.hrr_hopfield,
                    hypergraph=self.hypergraph,
                    head=self.head,
                    w_prev=w_prev_core,
                    regime_key=reg_key_cog,
                    ood_energy=ood_energy,
                    horizon_mask=horizon_mask,
                    cf_placebo_mode=c.cognitive_cf_placebo,
                    enable_early_exit=cog_early_exit,
                )
            # Integrar memoria Titans/HRR residual cuando ContextEngineer construyó anchor propio
            h_final = h_acc + 0.1 * (x0 - x0_pre_mem) if c.cognitive_use_context_engineer else h_acc
            if self.recursive_core is not None:
                bridge_out = self.recursive_core(
                    h_final,
                    ctx,
                    head=self.head,
                    verifier=self.verifier,
                    exposures=exposures,
                    ood_energy=ood_energy,
                    horizon_mask=horizon_mask,
                    placebo_context=placebo_ctx,
                    enable_early_exit=enable_early_exit,
                )
                pred = bridge_out["final_pred"]
                final_hidden = bridge_out["final_hidden"]
                step_outputs = bridge_out["step_outputs"]
                energy = bridge_out["energy"]
                diag: dict[str, object] = dict(bridge_out["diagnostics"])
            else:
                step_preds = cog_diag["step_predictions"]
                step_outputs = (
                    list(step_preds)
                    if isinstance(step_preds, list) and len(step_preds) > 0
                    else [self.head(hk, exposures=exposures) for hk in step_hiddens]
                )
                pred = self.head(h_final, exposures=exposures)
                final_hidden = h_final
                energy = self.verifier(
                    pred,
                    exposures=exposures,
                    ood_energy=ood_energy,
                    horizon_mask=horizon_mask,
                )
                z0 = torch.zeros((), device=device, dtype=dtype)
                one0 = torch.ones((), device=device, dtype=dtype)
                diag = dict(
                    k_used=cog_diag["expected_thinking_steps"],
                    energy_trajectory=[energy["total"].detach()],
                    delta_energy_neg_rate=cog_diag["monotonic_accept_rate"],
                    max_spectral_norm=cog_diag["lipschitz_upper_bound"],
                    router_entropy=cog_diag["tool_entropy"],
                    load_max_min_ratio=cog_diag.get("tool_load_max_min_ratio", one0),
                    expert_mean_corr=cog_diag.get("tool_corr_mean", z0),
                    router_entropy_per_asset=cog_diag["tool_entropy_per_asset"],
                    expert_loads=[],
                )
            diag["cognitive_core"] = cog_diag
            ponder_kl_loss = cog_diag["ponder_kl_loss"]  # type: ignore[assignment]
            weighted_ponder_loss = cog_diag["weighted_ponder_loss"]  # type: ignore[assignment]
            raw_tool_aux = cog_diag.get("tool_aux_loss")
            if isinstance(raw_tool_aux, torch.Tensor):
                tool_aux_loss = raw_tool_aux
            raw_tool_loads = cog_diag.get("tool_usage_distribution")
            if isinstance(raw_tool_loads, torch.Tensor):
                st["last_tool_loads"] = raw_tool_loads.detach()
            cog_last_aux = cog_diag.get("last_tool_aux")
            if isinstance(cog_last_aux, dict) and isinstance(
                cog_last_aux.get("portfolio_weights"), torch.Tensor
            ):
                st["last_tool_w_prev"] = cog_last_aux["portfolio_weights"].detach()
                if asset_ids is not None:
                    st["last_tool_w_prev_ids"] = asset_ids.detach()
            cog_state = cog_diag.get("state")
            if isinstance(cog_state, dict) and "cognitive_notes" in cog_state:
                st["cognitive_notes"] = cog_state["cognitive_notes"]
        else:
            assert self.recursive_core is not None
            if isinstance(self.recursive_core, LoopedCrossAssetTransformer):
                cf_conflict = (
                    conflict_score if conflict_score is not None else batch.get("conflict_score")
                )
                loop_out = self.recursive_core(
                    x0,
                    ctx,
                    head=self.head,
                    verifier=self.verifier,
                    exposures=exposures,
                    rmt_loadings=rmt_loadings,
                    ood_energy=ood_energy,
                    horizon_mask=horizon_mask,
                    conflict=cf_conflict,
                    w_opt=w_prev_core,
                    regime_vec=regime_vec,
                    placebo_context=placebo_ctx,
                    placebo_mode=c.cf_placebo_mode,
                    enable_early_exit=enable_early_exit,
                )
                raw_pkl = loop_out.get("ponder_kl_loss")
                if isinstance(raw_pkl, torch.Tensor):
                    ponder_kl_loss = raw_pkl
                    weighted_ponder_loss = raw_pkl * c.cognitive_ponder_weight
                raw_cf_l = loop_out.get("cf_fragility_loss")
                if isinstance(raw_cf_l, torch.Tensor):
                    cf_fragility_loss = raw_cf_l
            else:
                loop_out = self.recursive_core(
                    x0,
                    ctx,
                    head=self.head,
                    verifier=self.verifier,
                    exposures=exposures,
                    ood_energy=ood_energy,
                    horizon_mask=horizon_mask,
                    placebo_context=placebo_ctx,
                    enable_early_exit=enable_early_exit,
                )
            pred = loop_out["final_pred"]
            final_hidden = loop_out["final_hidden"]
            step_outputs = loop_out["step_outputs"]
            energy = loop_out["energy"]
            diag = dict(loop_out["diagnostics"])

        # PolarAlphaHead opt-in (#195 P0-1a/b/c: exact zero-mean alpha_cal + anti-FM-7 cross-detach)
        if self.polar_alpha_head is not None:
            vol_det: torch.Tensor | None = None
            for v_key in ("realized_vol", "volatility", "hist_vol"):
                v_raw = batch.get(v_key)
                if isinstance(v_raw, torch.Tensor):
                    vol_det = v_raw.detach().to(device=device, dtype=dtype)
                    break

            def _apply_polar(p_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                if "alpha_cal_for_cal" in p_dict:
                    return p_dict
                v_step = vol_det
                if v_step is None:
                    v_cand = p_dict.get("rough_vol")
                    if not isinstance(v_cand, torch.Tensor):
                        v_cand = p_dict.get("vol_pred")
                    v_step = v_cand.detach() if isinstance(v_cand, torch.Tensor) else None
                p_out = self.polar_alpha_head(p_dict["alpha"], vol=v_step)  # type: ignore[misc]
                p_new = dict(p_dict)
                p_new["alpha_raw"] = p_dict["alpha"]
                p_new["alpha_u"] = p_out["u"]
                p_new["alpha_cal"] = p_out["alpha_cal"]
                p_new["alpha_cal_for_cal"] = p_out["alpha_cal_for_cal"]
                p_new["alpha_cal_for_pplus"] = p_out["alpha_cal_for_pplus"]
                p_new["alpha"] = p_out["alpha_cal"]
                # #233 P0-1: Pure calibrated zero-mean alpha (no untrained factor_return noise!)
                p_new["factor_return"] = torch.zeros_like(p_out["alpha_cal"])
                p_new["pred_return"] = p_out["alpha_cal"]
                return p_new

            pred_is_last = bool(step_outputs) and (pred is step_outputs[-1])
            step_outputs = [_apply_polar(s_p) for s_p in step_outputs]
            pred = step_outputs[-1] if pred_is_last else _apply_polar(pred)

        # Cabeza paramétrica Skew Student-t (v13b_mmarrs_multimodal_sota)
        skew_t_out: dict[str, torch.Tensor] | None = None
        if self.skew_t_head is not None:
            skew_t_out = self.skew_t_head(final_hidden, conflict_score=conflict_score)

        diag["holographic"] = holo_diag
        diag["hypergraph"] = hg_diag
        diag["neuromod"] = mem_diag
        diag["ood_energy_mean"] = ood_energy.detach().mean()
        if set_fusion_diag:
            diag["set_modality_fusion"] = set_fusion_diag

        # 6) Núcleo Agéntico Metacognitivo nativo en dispositivo (cero syncs GPU!)
        agent_weights: torch.Tensor | None = None
        agent_active_rate: torch.Tensor = torch.ones((), device=device, dtype=dtype)
        if self.agent_governor is not None and self.portfolio_opt is not None and not self.training:
            with torch.no_grad():
                p_tot = pred["pred_return"][:, 0].detach()
                p_alf = pred["alpha"][:, 0].detach()
                p_med = pred["quantiles"][:, 0, 4].detach()
                spec_t = torch.stack([p_tot, p_alf, p_med], dim=-1)
                dec_t = self.agent_governor.decide_torch(
                    specialist_preds=spec_t,
                    hopfield_energy=ood_energy.detach(),
                    router_entropy=diag["router_entropy_per_asset"].detach(),  # type: ignore[union-attr]
                    loop_energy=energy["per_asset"].detach(),
                    estimated_costs=torch.full((n_assets,), 0.0005, device=device, dtype=dtype),
                )
                agent_active_rate = dec_t["active_mask"].to(dtype=dtype).mean()
                w_prev_t = st.get("agent_w_prev")
                prev_w_ids = st.get("agent_w_prev_ids")
                if (
                    isinstance(w_prev_t, torch.Tensor)
                    and asset_ids is not None
                    and isinstance(prev_w_ids, torch.Tensor)
                ):
                    w_prev_aligned: torch.Tensor | None = self._align_vector_by_asset_ids(
                        asset_ids, prev_w_ids, w_prev_t
                    )
                elif isinstance(w_prev_t, torch.Tensor) and w_prev_t.shape[0] == n_assets:
                    w_prev_aligned = w_prev_t
                else:
                    w_prev_aligned = None

                w_opt_t = self.portfolio_opt.solve_torch(
                    dec_t["gated_signal"],
                    vol=pred["vol_pred"][:, 0].detach(),
                    exposures=exposures,
                    w_prev=w_prev_aligned,
                )
                agent_weights = w_opt_t.detach()
                st["agent_w_prev"] = w_opt_t.detach()
                if asset_ids is not None:
                    st["agent_w_prev_ids"] = asset_ids.detach()
                st["last_specialist_preds"] = spec_t.detach()

        diag["agent_active_rate"] = agent_active_rate
        st["last_x0"] = x0_pre_mem.detach()
        if self.training and self.neuromod_memory is not None:
            st["last_x0_grad"] = x0_pre_mem
        st["last_regime_vec"] = regime_vec.detach()
        st["last_expert_loads"] = diag.get("expert_loads", [])
        if asset_ids is not None:
            st["last_asset_ids"] = asset_ids.detach()
        if c.moe_mode == "shuffled_context":
            st["placebo_prev_ctx"] = ctx.detach()
        st.setdefault("matured_steps", 0)
        st.setdefault("hrr_slots_used", 0)

        out: dict[str, object] = dict(
            pred_return=pred["pred_return"],
            alpha=pred["alpha"],
            factor_return=pred["factor_return"],
            quantiles=pred["quantiles"],
            tb_logits=pred["tb_logits"],
            tb_probs=pred["tb_probs"],
            turning_prob=pred["turning_prob"],
            log_vol=pred["log_vol"],
            vol_pred=pred["vol_pred"],
            epistemic_error=pred["epistemic_error"],
            final_hidden=final_hidden,
            loop_outputs=step_outputs,
            energy=energy,
            diagnostics=diag,
            agent_weights=agent_weights,
            state=st,
        )
        if self.polar_alpha_head is not None:
            for p_key in (
                "alpha_raw",
                "alpha_u",
                "alpha_cal",
                "alpha_cal_for_cal",
                "alpha_cal_for_pplus",
            ):
                out[p_key] = pred[p_key]
        if ssl_loss is not None:
            out["ssl_loss"] = ssl_loss
        if conflict_score is not None:
            out["conflict_score"] = conflict_score
        if skew_t_out is not None:
            out["skew_t"] = skew_t_out
        if ponder_kl_loss is not None:
            out["ponder_kl_loss"] = ponder_kl_loss
            out["weighted_ponder_loss"] = weighted_ponder_loss
        if cf_fragility_loss is not None:
            out["cf_fragility_loss"] = cf_fragility_loss
        if tool_aux_loss is not None:
            out["tool_aux_loss"] = tool_aux_loss
        return out

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        state: dict[str, object] | None = None,
        *,
        enable_early_exit: bool | None = None,
    ) -> dict[str, object]:
        """Delegación `nn.Module.forward` para `torch.compile(mode='default', dynamic=True)`."""
        return self.forward_session(batch, state=state, enable_early_exit=enable_early_exit)

    def _update_governor_on_device(self, ics_t: torch.Tensor) -> None:
        """Actualización Page-Hinkley/CUSUM 100% en dispositivo (cero `.cpu()/.numpy()`)."""
        ics = ics_t.detach().to(device=self._gov_weights.device, dtype=torch.float32)
        self._gov_updates.add_(1)
        is_first = self._gov_updates == 1
        decay = float(self.agent_governor.decay) if self.agent_governor is not None else 0.90
        drift = float(self.agent_governor.cusum_drift) if self.agent_governor is not None else 0.02
        thr = (
            float(self.agent_governor.cusum_threshold) if self.agent_governor is not None else 0.15
        )
        new_ewma = torch.where(is_first, ics, decay * self._gov_ewma_ic + (1.0 - decay) * ics)
        new_peak = torch.where(
            is_first,
            torch.clamp(ics, min=0.0),
            torch.maximum(self._gov_peak_ic * 0.995, new_ewma),
        )
        drop = (new_peak - ics) - drift
        new_cusum = torch.clamp(self._gov_cusum + drop, min=0.0)
        alerts = new_cusum >= thr
        scores = torch.clamp(new_ewma, min=0.0) + 1e-4
        scores = torch.where(alerts, scores * 0.15, scores)
        weights = scores / scores.sum().clamp_min(1e-8)

        self._gov_ewma_ic.copy_(new_ewma)
        self._gov_peak_ic.copy_(new_peak)
        self._gov_cusum.copy_(new_cusum)
        self._gov_weights.copy_(weights)
        # En CPU sincronizar gobernador numpy sin llamadas `.cpu()`/`.numpy()` (cero sync CUDA!)
        if self.agent_governor is not None and ics.device.type == "cpu":
            self.agent_governor.update_matured(np.asarray(ics, dtype=np.float64))

    def step_matured(
        self,
        state: dict[str, object] | None,
        *,
        matured_batch: dict[str, torch.Tensor],
        matured_targets: torch.Tensor,
        matured_ach: torch.Tensor | None = None,
        matured_ne: torch.Tensor | None = None,
        matured_preds: torch.Tensor | None = None,
        keep_grad: bool = False,
    ) -> dict[str, object]:
        """Macro-bucle causal tras madurar etiquetas (soporta `keep_grad=True` para BPTT P0-2)."""
        st: dict[str, object] = {} if state is None else dict(state)
        use_grad = bool(keep_grad and self.training and torch.is_grad_enabled())

        x0_grad = st.pop("last_x0_grad", None)
        x0_ref = x0_grad if (use_grad and isinstance(x0_grad, torch.Tensor)) else st.get("last_x0")
        if not isinstance(x0_ref, torch.Tensor):
            with torch.set_grad_enabled(use_grad):
                mod_tokens, mod_mask, _ = self._encode_modalities(matured_batch)
                denom = mod_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                mod_pooled = (mod_tokens * mod_mask.unsqueeze(-1)).sum(dim=1) / denom
                phys_raw = matured_batch.get("physics_v4")
                if phys_raw is None:
                    phys_z = torch.zeros(
                        (mod_tokens.shape[0], self.config.physics_dim),
                        device=mod_tokens.device,
                        dtype=mod_tokens.dtype,
                    )
                    phys_in = (
                        torch.cat([phys_z, phys_z], dim=-1)
                        if self.config.use_dual_level_rank_input
                        else phys_z
                    )
                else:
                    phys_in = self._preprocess_cross_section(
                        phys_raw.to(device=mod_tokens.device, dtype=mod_tokens.dtype),
                        apply_gaussian_rank=True,
                    )
                x0_ref = self.fusion_norm(
                    self.fusion_proj(torch.cat([mod_pooled, self.enc_physics(phys_in)], dim=-1))
                )

        master_dtype = self.target_encoder.weight.dtype
        device = x0_ref.device
        dtype = master_dtype
        if x0_ref.dtype != master_dtype:
            x0_ref = x0_ref.to(dtype=master_dtype)
        y_raw = matured_targets.to(device=device, dtype=dtype)
        y_raw_1d = y_raw[:, 0] if y_raw.ndim == 2 else y_raw
        y_fin_mask = torch.isfinite(y_raw_1d).to(dtype=dtype)
        y_t = torch.where(torch.isfinite(y_raw_1d), y_raw_1d, torch.zeros_like(y_raw_1d))

        asset_ids = matured_batch.get("asset_ids")
        if asset_ids is None and isinstance(st.get("last_asset_ids"), torch.Tensor):
            asset_ids = st["last_asset_ids"]  # type: ignore[assignment]

        # 1) Actualizar Memoria Titans Neuromodulada ACh/NE (CON gradientes si `keep_grad=True`!)
        if self.neuromod_memory is not None:
            with torch.set_grad_enabled(use_grad):
                ach = matured_ach if matured_ach is not None else matured_batch.get("ach_signal")
                if ach is None:
                    y_std = y_t.std().clamp_min(1e-4)
                    ach = ((y_t - y_t.mean()).abs() / y_std).unsqueeze(-1)
                ne_explicit = (
                    matured_ne if matured_ne is not None else matured_batch.get("ne_signal")
                )
                if matured_preds is not None:
                    p_m = matured_preds.detach().to(device=device, dtype=dtype)
                    p1 = p_m[:, 0] if p_m.ndim == 2 else p_m
                    err_vec = (y_t - p1).abs()
                else:
                    err_vec = y_t.abs()
                # P0-2: target_encoder aprendido `nn.Linear(1, D)` en lugar de vector rango-1
                hint = self.target_encoder(torch.clamp(y_t.unsqueeze(-1), -0.5, 0.5) * 10.0)
                if isinstance(self.neuromod_memory, DualAxisTitansMemory):
                    new_titans, titans_diag = self.neuromod_memory.update(
                        x0_ref,
                        st.get("titans_state"),  # type: ignore[arg-type]
                        ach_signal=ach,
                        ne_signal=ne_explicit,
                        target_hint=hint,
                        asset_ids=asset_ids,
                        write_mask=y_fin_mask,
                        quality_weights=matured_batch.get("liquidity_weights"),
                        matured_error=err_vec,
                    )
                    if isinstance(new_titans.get("last_ne_signal"), torch.Tensor):
                        st["last_ne_signal"] = new_titans["last_ne_signal"].detach()
                else:
                    if ne_explicit is None:
                        # Pulso NE causal libre de escala vía EMA en estado (descongela ne_slow!)
                        e_cur = err_vec.mean().detach().to(dtype=torch.float32)
                        prev_mean = (
                            st["ne_ema_mean"].detach().to(device=device, dtype=torch.float32)
                            if isinstance(st.get("ne_ema_mean"), torch.Tensor)
                            else e_cur
                        )
                        prev_var = (
                            st["ne_ema_var"].detach().to(device=device, dtype=torch.float32)
                            if isinstance(st.get("ne_ema_var"), torch.Tensor)
                            else torch.full((), 1e-2, device=device, dtype=torch.float32)
                        )
                        prev_steps = (
                            st["ne_ema_steps"].detach().to(device=device, dtype=torch.float32)
                            if isinstance(st.get("ne_ema_steps"), torch.Tensor)
                            else torch.zeros((), device=device, dtype=torch.float32)
                        )
                        raw_ne = torch.clamp(
                            (e_cur - prev_mean) / (torch.sqrt(prev_var.clamp_min(1e-8)) + 1e-4),
                            min=-4.0,
                            max=4.0,
                        )
                        ne_val = torch.where(prev_steps > 0.5, raw_ne, torch.zeros_like(raw_ne))
                        is_first = prev_steps < 0.5
                        st["ne_ema_mean"] = torch.where(
                            is_first, e_cur, 0.95 * prev_mean + 0.05 * e_cur
                        ).detach()
                        st["ne_ema_var"] = torch.where(
                            is_first,
                            prev_var,
                            0.95 * prev_var + 0.05 * (e_cur - prev_mean).pow(2),
                        ).detach()
                        st["ne_ema_steps"] = (prev_steps + 1.0).detach()
                        ne_for_upd = ne_val.to(dtype=dtype).unsqueeze(0)
                    else:
                        ne_for_upd = ne_explicit
                        ne_val = ne_explicit.detach().to(device=device, dtype=torch.float32).mean()
                    st["last_ne_signal"] = ne_val.detach()
                    new_titans, titans_diag = self.neuromod_memory.update(
                        x0_ref,
                        st.get("titans_state"),  # type: ignore[arg-type]
                        ach_signal=ach,
                        ne_signal=ne_for_upd,
                        target_hint=hint,
                        asset_ids=asset_ids,
                    )
                st["titans_state"] = (
                    new_titans
                    if use_grad
                    else NeuromodulatedTitansMemory.detach_state(new_titans)
                )
                st["last_titans_diag"] = titans_diag

        with torch.no_grad():
            x0_det = x0_ref.detach()
            # 2) Almacenar prototipos maduros en Memoria HRR-Modern-Hopfield (P0-2)
            if self.hrr_hopfield is not None:
                reg_vec = st.get("last_regime_vec")
                reg_key = (
                    self.regime_proj(reg_vec.to(device=device, dtype=dtype))
                    if isinstance(reg_vec, torch.Tensor)
                    else None
                )
                val_mem = x0_det * (1.0 + torch.tanh(y_t.unsqueeze(-1) * 20.0))
                slots_used = self.hrr_hopfield.store(
                    x0_det,
                    val_mem,
                    regime_key=reg_key,
                    return_signal=y_t,
                )
                st["hrr_slots_used"] = slots_used

            # 3) Actualizar precisiones Pi_l acotadas <= 1.0 (cero `.item()`, P1-3)
            if self.recursive_core is not None and matured_preds is not None:
                p_m = matured_preds.detach().to(device=device, dtype=dtype)
                p1 = p_m[:, 0] if p_m.ndim == 2 else p_m
                err_scale = (y_t - p1).abs().mean()
                pi_target = torch.clamp(0.02 / err_scale.clamp_min(0.005), min=0.25, max=1.0)
                lp = self.recursive_core.layer_precision
                lp.mul_(0.9).add_(0.1 * pi_target.to(device=lp.device, dtype=lp.dtype))
                lp.clamp_(min=0.25, max=1.0)

            # 4) Sesgos de equilibrio Aux-Loss-Free estrictamente en `train()` (P1-3 + F3 v2)
            if (
                self.training
                and self.recursive_core is not None
                and self.config.use_moe
                and self.config.moe_mode in ("soft", "sparsemax")
            ):
                loads_list = st.get("last_expert_loads") or []
                if hasattr(self.recursive_core, "update_balance_biases"):
                    self.recursive_core.update_balance_biases(loads_list, lr=0.05)  # type: ignore[arg-type]
                else:
                    for blk, loads in zip(
                        self.recursive_core.step_blocks, loads_list, strict=False
                    ):
                        if hasattr(blk, "update_balance_bias"):
                            blk.update_balance_bias(loads, lr=0.05)

            # 4b) Sesgo DeepSeek-V3 + calendario coseno del ToolRouter (Familia 4, solo en train()!)
            if (
                self.training
                and self.cognitive_core is not None
                and self.config.cognitive_tool_mode in ("soft", "sparsemax")
            ):
                tool_loads = st.get("last_tool_loads")
                if isinstance(tool_loads, torch.Tensor):
                    self.cognitive_core.tool_router.update_balance_bias(tool_loads, lr=0.05)
                self.cognitive_core.step_router_schedule(1)

            # 5) Gobernador Metacognitivo A5–A8 en dispositivo (cero sync en GPU!)
            spec_raw = st.get("last_specialist_preds")
            if (
                self.agent_governor is not None
                and isinstance(spec_raw, torch.Tensor)
                and spec_raw.shape[0] == y_t.shape[0]
                and y_t.shape[0] >= 5
            ):
                spec_f = spec_raw.to(device=device, dtype=torch.float32)
                y_f = y_t.to(dtype=torch.float32)
                sc = spec_f - spec_f.mean(dim=0, keepdim=True)
                yc = y_f - y_f.mean()
                s_norm = sc.norm(dim=0)
                y_norm = yc.norm()
                denom = (s_norm * y_norm).clamp_min(1e-12)
                valid = (s_norm > 1e-8) & (y_norm > 1e-8)
                ics_t = torch.where(
                    valid,
                    (sc * yc.unsqueeze(-1)).sum(dim=0) / denom,
                    torch.zeros_like(s_norm),
                )
                st["last_specialist_ics"] = ics_t
                self._update_governor_on_device(ics_t)

        st["matured_steps"] = int(st.get("matured_steps", 0)) + 1
        return st

    def parameter_audit(self) -> dict[str, object]:
        """Audita parámetros totales, entrenables y huella SHA-256 de la variación."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(
            variation=self.config.variation,
            family=self.config.family,
            market=self.config.market,
            fingerprint=self.config.fingerprint(),
            total_params=int(total),
            trainable_params=int(trainable),
            use_moe=self.config.use_moe,
            moe_mode=self.config.moe_mode,
            max_loops=(
                self.config.max_loops
                if (
                    self.config.use_double_loop
                    or self.config.use_looped_transformer
                    or self.config.use_cognitive_core
                )
                else 1
            ),
            use_agent_core=self.config.use_agent_core,
            use_cognitive_core=self.config.use_cognitive_core,
            use_multimodal_sota=self.config.use_multimodal_sota,
            use_dual_axis_titans=self.config.use_dual_axis_titans,
            use_looped_transformer=self.config.use_looped_transformer,
            use_counterfactual_loop=self.config.use_counterfactual_loop,
            cognitive_toolbox_version=self.config.cognitive_toolbox_version,
            noic_pinball_twin=self.config.noic_pinball_twin,
        )


def build_variation(name: str, **overrides: object) -> MMARRSTitan:
    """Fábrica oficial de todas las variaciones emparejadas v1 y v2 de `MMARRS-TITAN`."""
    cfg = variation_config(name, **overrides)
    return MMARRSTitan(cfg)


def build_family_model(
    variant_id: str,
    *,
    market: str = "US",
    family: str | None = None,
    **overrides: object,
) -> nn.Module:
    """Fábrica unificada de las 4 familias estrictamente independientes (FM-8 y anti-sopa).

    Instancia:
      - Familia 1 (`titans_original`): `MMARRSTitan` sin MoE/Holo3L/CogCore (v1 + v2 `t15*`).
      - Familia 2 (`holographic_brain`): `StandaloneHolographicBrainModel` (v1 + v2 `nb_v2_*`).
      - Familia 3 (`mmarrs_titan`): `MMARRSTitan` (`v00..v13c` + v2 `v14*`).
      - Familia 4 (`titans_cognitive_core`): `MMARRSTitan` con `TitansCognitiveCore` (v1 + v2 10T).
    """
    if market not in ("US", "CN"):
        raise ValueError(
            f"FM-8: market inválido {market!r}; solo se permiten mercados separados ('US', 'CN')"
        )

    ov = dict(overrides)
    if family is None and "family" in ov and ov["family"] is not None:
        family = str(ov["family"])
    ov.pop("family", None)

    def _check_family(expected: str) -> None:
        if family is not None and family != expected:
            raise ValueError(
                f"Incompatibilidad de familia (anti-sopa): la variante {variant_id!r} "
                f"pertenece a {expected!r}, no a {family!r}"
            )

    # Resolver aliases de catálogo o registro (`cog_v1_6tools_k3`, `v13_mmarrs_full`)
    resolved_id = TFM_VARIANT_ALIASES.get(variant_id, VARIATION_ALIASES.get(variant_id, variant_id))

    # Atajo directo si es un nombre de `VARIATION_REGISTRY` (F1 v2, F3 v1/v2, F4 v2)
    if resolved_id in VARIATION_REGISTRY:
        reg_family = str(VARIATION_REGISTRY[resolved_id].get("family", "mmarrs_titan"))
        _check_family(reg_family)
        ov.setdefault("market", market)
        ov["family"] = reg_family
        return build_variation(resolved_id, **ov)

    # Atajo directo si es un preset de `NEURO_BRAIN_ABLATIONS` (Familia 2 v1 + v2 `nb_v2_*`)
    if resolved_id in NEURO_BRAIN_ABLATIONS:
        _check_family("holographic_brain")
        if bool(ov.get("use_diff_xasset_attn", False)):
            raise ValueError(
                "use_diff_xasset_attn es exclusivo de Family 3 ('mmarrs_titan'); "
                "recibido family='holographic_brain'"
            )
        valid_nb = {f.name for f in fields(NeuroBrainConfig)}
        nb_ov = {k: v for k, v in ov.items() if k in valid_nb}
        return build_neuro_brain_ablation(resolved_id, **nb_ov)

    if resolved_id not in TFM_VARIANT_CATALOG:
        raise ValueError(
            f"Variante desconocida {variant_id!r}; no encontrada en VARIATION_REGISTRY ni "
            "TFM_VARIANT_CATALOG"
        )

    spec = TFM_VARIANT_CATALOG[resolved_id]
    _check_family(spec.family)
    noic = spec.order_level_mode == "noic_pinball_twin"

    if spec.family == "holographic_brain":
        if bool(ov.get("use_diff_xasset_attn", False)):
            raise ValueError(
                "use_diff_xasset_attn es exclusivo de Family 3 ('mmarrs_titan'); "
                "recibido family='holographic_brain'"
            )
        valid_nb = {f.name for f in fields(NeuroBrainConfig)}
        nb_ov = {k: v for k, v in ov.items() if k in valid_nb}
        ablation = str(spec.config_overrides.get("ablation", ""))
        if ablation in NEURO_BRAIN_ABLATIONS:
            return build_neuro_brain_ablation(ablation, **nb_ov)
        # Variantes Entrega 2 / DGX v12 de Familia 2 sobre `StandaloneHolographicBrainModel`
        if "n_cortical_areas" in spec.config_overrides:
            nb_ov.setdefault("n_cortical_areas", int(spec.config_overrides["n_cortical_areas"]))
        nb_ov.setdefault("max_cycles", min(int(spec.max_loops), 3))
        preset = "bare_holographic_3l" if spec.origin == "matched_control" else "full"
        return build_neuro_brain_ablation(preset, **nb_ov)

    if spec.family == "mmarrs_titan":
        var_name = _CATALOG_TO_MMARRS_VARIATION.get(
            variant_id, str(spec.config_overrides.get("variation", "v13_mmarrs_titan_full"))
        )
        ov["family"] = "mmarrs_titan"
        ov.setdefault("market", market)
        ov.setdefault("matched_control", spec.matched_control_id)
        ov.setdefault("max_loops", int(spec.max_loops))
        if noic:
            ov.setdefault("noic_pinball_twin", True)
        if spec.config_overrides.get("use_cross_modal_conflict_fusion"):
            ov.setdefault("use_multimodal_sota", True)
        if "ssl_weight" in spec.config_overrides:
            ov.setdefault("ssl_weight", float(spec.config_overrides["ssl_weight"]))
        return build_variation(var_name, **ov)

    if spec.family == "titans_cognitive_core":
        tool_mode = str(spec.config_overrides.get("tool_mode", "soft"))
        use_ce = bool(spec.config_overrides.get("use_context_engineer", True))
        use_moe_bridge = bool(spec.config_overrides.get("use_moe_bridge", False))
        cfg_kwargs: dict[str, object] = dict(
            variation="v02_titans_neuromod_only",
            family="titans_cognitive_core",
            market=market,
            matched_control=spec.matched_control_id,
            use_neuromod_memory=True,
            use_hrr_hopfield=True,
            use_hypergraph=True,
            use_holographic_3l=False,
            use_double_loop=False,
            use_moe=use_moe_bridge,
            use_agent_core=False,
            use_cognitive_core=True,
            cognitive_tool_mode=tool_mode,
            cognitive_use_context_engineer=use_ce,
            cognitive_ponder_weight=float(spec.config_overrides.get("w_ponder_kl", 0.01)),
            max_loops=int(spec.max_loops),
            noic_pinball_twin=noic,
        )
        cfg_kwargs.update(ov)
        return MMARRSTitan(MMARRSConfig(**cfg_kwargs))

    # Familia 1 (`titans_original`): sin MoE, sin Holo3L, sin CogCore
    use_mem = (
        spec.config_overrides.get("memory_mode") != "disabled"
        and "baseline" not in spec.tags
    )
    use_hrr = "episodic" in spec.tags and spec.config_overrides.get("admission_policy") != "M0"
    cfg_f1: dict[str, object] = dict(
        variation="v02_titans_neuromod_only" if use_mem else "v00_dense_baseline",
        family="titans_original",
        market=market,
        matched_control=spec.matched_control_id,
        use_neuromod_memory=use_mem,
        use_hrr_hopfield=use_hrr,
        use_holographic_3l=False,
        use_hypergraph=False,
        use_moe=False,
        use_double_loop=False,
        use_agent_core=False,
        use_cognitive_core=False,
        use_gaussian_ranks=True,
        use_tabm_skip=True,
        max_loops=int(spec.max_loops),
        noic_pinball_twin=noic,
    )
    cfg_f1.update(ov)
    return MMARRSTitan(MMARRSConfig(**cfg_f1))
