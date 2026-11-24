"""Contratos inmutables SHA-256 y registro de las 16 variaciones emparejadas de `MMARRS-TITAN`.

Separa limpiamente las 4 familias independientes TFM (`titans_original`, `holographic_brain`,
`mmarrs_titan`, `titans_cognitive_core`), exige separación estricta de mercados `US` vs `CN`
(FM-8), aplica el tope duro de bucles (`max_loops=3` por omisión, tope duro `<= 4`), exige
purga/embargo anti-fuga multi-horizonte (`purge >= max_h`, `embargo >= max(5, ceil(0.2*max_h))`)
y declara todas las variaciones `v00..v13c` con sus controles emparejados obligatorios.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

MMARRS_FAMILIES: tuple[str, ...] = (
    "titans_original",
    "holographic_brain",
    "mmarrs_titan",
    "titans_cognitive_core",
)

VALID_MARKETS: tuple[str, ...] = ("US", "CN")

MMARRS_VARIATIONS: tuple[str, ...] = (
    "v00_dense_baseline",
    "v01_dense_wide_iso_param",
    "v02_titans_neuromod_only",
    "v03_moe_only_k1",
    "v04_moe_uniform_control",
    "v05_moe_shuffled_ctx_placebo",
    "v06_double_loop_pc_trm_only",
    "v07_moe_double_loop",
    "v08_holographic_3l_hrr_only",
    "v09_holographic_3l_moe_loop",
    "v10_cross_asset_hypergraph_only",
    "v11_moe_plus_agent_core",
    "v12_moe_loop_plus_agent_core",
    "v13_mmarrs_titan_full",
    "v13b_mmarrs_multimodal_sota",
    "v13c_mmarrs_noic_pinball_twin",
)

MMARRS_V2_VARIATIONS: tuple[str, ...] = (
    # Familia 1 v2 (`titans_original`)
    "t15a_dual_axis_mem",
    "t15b_full_v2",
    "t15c_noic_twin",
    # Familia 3 v2 (`mmarrs_titan` Looped Transformer + Counterfactual)
    "v14a_tied_loop",
    "v14b_tied_xasset",
    "v14c_tied_xasset_cf",
    "v14c_placebo",
    "v14c_noic_twin",
    # Familia 4 v2 (`titans_cognitive_core` 10-Tool Karpathy/Anthropic)
    "cog_v2_10tools_uniform_k3",
    "cog_v2_10tools_shuffled_cf_placebo_k3",
    "cog_v2_10tools_soft_k3",
    "cog_v2_10tools_sparsemax_k3",
    "cog_v2_10tools_noic_twin",
)

MOE_MODES: tuple[str, ...] = ("soft", "sparsemax", "uniform", "shuffled_context")
DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 21, 63)
DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def required_purge_embargo(active_horizons: Sequence[int]) -> tuple[int, int]:
    """Devuelve `(min_purge, min_embargo)` anti-fuga para los horizontes activos (P0-4)."""
    hs = [int(h) for h in active_horizons]
    if not hs or any(h < 1 for h in hs):
        raise ValueError("active_horizons debe contener al menos un horizonte >= 1")
    max_h = max(hs)
    return max_h, max(5, math.ceil(0.2 * max_h))


# Definición científica exacta de cada una de las 16 variaciones de ablación
VARIATION_REGISTRY: dict[str, dict[str, object]] = {
    "v00_dense_baseline": dict(
        description="Control 1 (Denso Iso-FLOP): fusión multimodal uniforme + FFN denso, K=1.",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v01_dense_wide_iso_param": dict(
        description="Control 1b (Denso Ancho Iso-Parámetros): iguala parámetros totales de v03.",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=10.0,
        use_neuromod_memory=False,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v02_titans_neuromod_only": dict(
        description="Aislamiento Titans Neuromodulado (ACh/NE + Gated DeltaNet-2 + PT1 + Huber).",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v03_moe_only_k1": dict(
        description="Solo MoE MoR-LOOM-Fin (2 shared + 6 routed, Aux-Free, K=1, sin doble bucle).",
        family="mmarrs_titan",
        matched_control="v01_dense_wide_iso_param",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v04_moe_uniform_control": dict(
        description="Control 3 (MoE Uniforme 1/E): mismos expertos que v03 sin especialización.",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=True,
        moe_mode="uniform",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v05_moe_shuffled_ctx_placebo": dict(
        description="Control 4 (Placebo Contexto Barajado entre sesiones): rompe régimen en MoE.",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=True,
        moe_mode="shuffled_context",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v06_double_loop_pc_trm_only": dict(
        description="Solo Doble Bucle PC-TRM-Eq (K<=3, Lipschitz<1, Early Exit, sin MoE).",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v07_moe_double_loop": dict(
        description="MoE + Doble Bucle (MoR-LOOM-Fin + PC-TRM-Eq con routers R^(k), K<=3).",
        family="mmarrs_titan",
        matched_control="v03_moe_only_k1",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v08_holographic_3l_hrr_only": dict(
        description="Puente controlado: Cerebro Holográfico 3L + HRR-Hopfield (K=1).",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v09_holographic_3l_moe_loop": dict(
        description="Puente controlado: Cerebro Holográfico 3L + HRR + MoE + Doble Bucle (K<=3).",
        family="mmarrs_titan",
        matched_control="v07_moe_double_loop",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v10_cross_asset_hypergraph_only": dict(
        description="Solo Hipergrafo Transversal RMT/Lead-Lag O(N*K) sin MoE ni bucle.",
        family="mmarrs_titan",
        matched_control="v00_dense_baseline",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=True,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v11_moe_plus_agent_core": dict(
        description="MoE (K=1) + Núcleo Agéntico Metacognitivo (A5-A8 + Cartera Convexa P+).",
        family="mmarrs_titan",
        matched_control="v03_moe_only_k1",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=True,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v12_moe_loop_plus_agent_core": dict(
        description="MoE + Doble Bucle (K<=3) + Núcleo Agéntico Metacognitivo (A5-A8 + P+).",
        family="mmarrs_titan",
        matched_control="v07_moe_double_loop",
        ffn_mult=2.0,
        use_neuromod_memory=False,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=True,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v13_mmarrs_titan_full": dict(
        description=(
            "MMARRS-TITAN Completo: Multimodal 5M + Física v4 + Titans ACh/NE + Holográfico 3L + "
            "HRR-Hopfield + Hipergrafo RMT + MoR-LOOM-Fin + Doble Bucle PC-TRM-Eq (K<=3) + P+"
        ),
        family="mmarrs_titan",
        matched_control="v04_moe_uniform_control",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "v13b_mmarrs_multimodal_sota": dict(
        description=(
            "MMARRS-TITAN Multimodal SOTA+: v13 + CrossModalConflictFusion + VICReg/Barlow SSL "
            "(ssl_weight=0.02, vista B solo asset-varying) + SkewStudentTDistributionHead."
        ),
        family="mmarrs_titan",
        matched_control="v13_mmarrs_titan_full",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=True,
        noic_pinball_twin=False,
    ),
    "v13c_mmarrs_noic_pinball_twin": dict(
        description=(
            "Gemelo FM-7/F7-b Pinball-Only de v13 (w_ic=0) para calibración exacta de anchuras."
        ),
        family="mmarrs_titan",
        matched_control="v13_mmarrs_titan_full",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=False,
        noic_pinball_twin=True,
    ),
    # -------------------------------------------------------------------------
    # Wave-2 v2 — Familia 1 (`titans_original`): DualAxisTitansMemory + TabM + stab05
    # -------------------------------------------------------------------------
    "t15a_dual_axis_mem": dict(
        description=(
            "Familia 1 v2 ablación: DualAxisTitansMemory (matriz cohorte proximal FP32 + Tukey + "
            "CSOS + banco Benna-Fusi + NE causal libre de escala), anclada a titans_neuromod_sota."
        ),
        family="titans_original",
        matched_control="titans_neuromod_sota",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_dual_axis_titans=True,
        use_dual_level_rank_input=False,
        tabm_k=1,
        use_gaussian_rank_targets=False,
        w_stab_downside=0.0,
        use_gaussian_ranks=True,
        use_tabm_skip=True,
        tabm_members=8,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "t15b_full_v2": dict(
        description=(
            "Familia 1 v2 flagship: DualAxisTitansMemory + canal dual [clipped_z, g_rank] + "
            "TabM head-only K=8 + IC gaussiano + liqw3 + semivarianza bajista stab05."
        ),
        family="titans_original",
        matched_control="titans_neuromod_sota",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_dual_axis_titans=True,
        use_dual_level_rank_input=True,
        tabm_k=8,
        use_gaussian_rank_targets=True,
        w_stab_downside=0.05,
        use_gaussian_ranks=True,
        use_tabm_skip=True,
        tabm_members=8,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=False,
    ),
    "t15c_noic_twin": dict(
        description=(
            "Gemelo FM-7/F7-b Pinball-Only de t15b_full_v2 (w_ic=0, w_stab_downside=0) para "
            "calibración exacta de cuantiles."
        ),
        family="titans_original",
        matched_control="t15b_full_v2",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_dual_axis_titans=True,
        use_dual_level_rank_input=True,
        tabm_k=8,
        use_gaussian_rank_targets=False,
        w_stab_downside=0.0,
        use_gaussian_ranks=True,
        use_tabm_skip=True,
        tabm_members=8,
        use_moe=False,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_holographic_3l=False,
        use_hrr_hopfield=False,
        use_hypergraph=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        noic_pinball_twin=True,
    ),
    # -------------------------------------------------------------------------
    # Wave-2 v2 — Familia 3 (`mmarrs_titan`): LoopedCrossAssetTransformer + Counterfactual
    # -------------------------------------------------------------------------
    "v14a_tied_loop": dict(
        description=(
            "Familia 3 v2 ablación K=1: LoopedCrossAssetTransformer weight-tied sin atención "
            "transversal ni contrafactual (anclada a v13b_mmarrs_multimodal_sota)."
        ),
        family="mmarrs_titan",
        matched_control="v13b_mmarrs_multimodal_sota",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=False,
        max_loops=1,
        use_looped_transformer=True,
        loop_use_cross_attn=False,
        use_counterfactual_loop=False,
        cf_placebo_mode=False,
        w_cf_fragility=0.0,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=True,
        noic_pinball_twin=False,
    ),
    "v14b_tied_xasset": dict(
        description=(
            "Familia 3 v2 ablación K=3: LoopedCrossAssetTransformer weight-tied + "
            "InducedFactorCrossAttention O(N*R) invariante a F (sin ciclo contrafactual)."
        ),
        family="mmarrs_titan",
        matched_control="v14a_tied_loop",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_looped_transformer=True,
        loop_use_cross_attn=True,
        use_counterfactual_loop=False,
        cf_placebo_mode=False,
        w_cf_fragility=0.0,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=True,
        noic_pinball_twin=False,
    ),
    "v14c_tied_xasset_cf": dict(
        description=(
            "Familia 3 v2 flagship K=3: LoopedCrossAssetTransformer weight-tied + "
            "InducedFactorCrossAttention + Counterfactual Verify-Retry Loop (5 escenarios)."
        ),
        family="mmarrs_titan",
        matched_control="v14b_tied_xasset",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_looped_transformer=True,
        loop_use_cross_attn=True,
        use_counterfactual_loop=True,
        cf_placebo_mode=False,
        w_cf_fragility=0.05,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=True,
        noic_pinball_twin=False,
    ),
    "v14c_placebo": dict(
        description=(
            "Control placebo emparejado de v14c: derangement sin puntos fijos (N//2) sobre los "
            "shocks contrafactuales (mismos pesos y distribución marginal, causalidad rota)."
        ),
        family="mmarrs_titan",
        matched_control="v14b_tied_xasset",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_looped_transformer=True,
        loop_use_cross_attn=True,
        use_counterfactual_loop=True,
        cf_placebo_mode=True,
        w_cf_fragility=0.05,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=True,
        noic_pinball_twin=False,
    ),
    "v14c_noic_twin": dict(
        description=(
            "Gemelo FM-7/F7-b Pinball-Only de v14c_tied_xasset_cf (w_ic=0) para calibración "
            "exacta de cuantiles."
        ),
        family="mmarrs_titan",
        matched_control="v14c_tied_xasset_cf",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_moe=True,
        moe_mode="soft",
        use_double_loop=True,
        max_loops=3,
        use_looped_transformer=True,
        loop_use_cross_attn=True,
        use_counterfactual_loop=True,
        cf_placebo_mode=False,
        w_cf_fragility=0.0,
        use_holographic_3l=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_agent_core=True,
        use_multimodal_sota=True,
        noic_pinball_twin=True,
    ),
    # -------------------------------------------------------------------------
    # Wave-2 v2 — Familia 4 (`titans_cognitive_core`): 10-Tool Karpathy/Anthropic
    # -------------------------------------------------------------------------
    "cog_v2_10tools_uniform_k3": dict(
        description=(
            "Control emparejado Familia 4 v2: pesos uniformes 1/10 sobre las 10 herramientas GPU "
            "ortogonales (K=3, sin especialización del ToolRouter)."
        ),
        family="titans_cognitive_core",
        matched_control="cog_no_tools_k3",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_holographic_3l=False,
        use_double_loop=False,
        use_moe=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        use_cognitive_core=True,
        cognitive_tool_mode="uniform",
        cognitive_toolbox_version="v2_10tools",
        cognitive_use_context_engineer=True,
        cognitive_cf_placebo=False,
        cognitive_ponder_weight=0.01,
        w_tool_aux=0.01,
        max_loops=3,
        noic_pinball_twin=False,
    ),
    "cog_v2_10tools_shuffled_cf_placebo_k3": dict(
        description=(
            "Control placebo emparejado Familia 4 v2: ToolRouter soft sobre 10 herramientas con "
            "T8 contrafactual en modo placebo (derangement sin puntos fijos)."
        ),
        family="titans_cognitive_core",
        matched_control="cog_v2_10tools_uniform_k3",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_holographic_3l=False,
        use_double_loop=False,
        use_moe=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        use_cognitive_core=True,
        cognitive_tool_mode="soft",
        cognitive_toolbox_version="v2_10tools",
        cognitive_use_context_engineer=True,
        cognitive_cf_placebo=True,
        cognitive_ponder_weight=0.01,
        w_tool_aux=0.01,
        max_loops=3,
        noic_pinball_twin=False,
    ),
    "cog_v2_10tools_soft_k3": dict(
        description=(
            "Familia 4 v2 flagship: Núcleo Cognitivo weight-tied Karpathy + ContextEngineerGPU "
            "Anthropic + ToolRouter v2 lazo cerrado (soft) sobre 10 herramientas GPU (K<=3)."
        ),
        family="titans_cognitive_core",
        matched_control="cog_v2_10tools_uniform_k3",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_holographic_3l=False,
        use_double_loop=False,
        use_moe=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        use_cognitive_core=True,
        cognitive_tool_mode="soft",
        cognitive_toolbox_version="v2_10tools",
        cognitive_use_context_engineer=True,
        cognitive_cf_placebo=False,
        cognitive_ponder_weight=0.01,
        w_tool_aux=0.01,
        max_loops=3,
        noic_pinball_twin=False,
    ),
    "cog_v2_10tools_sparsemax_k3": dict(
        description=(
            "Familia 4 v2 SOTA: ToolRouter v2 sparsemax (invocación dispersa exacta sobre las "
            "10 herramientas GPU ortogonales, K<=3)."
        ),
        family="titans_cognitive_core",
        matched_control="cog_v2_10tools_uniform_k3",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_holographic_3l=False,
        use_double_loop=False,
        use_moe=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        use_cognitive_core=True,
        cognitive_tool_mode="sparsemax",
        cognitive_toolbox_version="v2_10tools",
        cognitive_use_context_engineer=True,
        cognitive_cf_placebo=False,
        cognitive_ponder_weight=0.01,
        w_tool_aux=0.01,
        max_loops=3,
        noic_pinball_twin=False,
    ),
    "cog_v2_10tools_noic_twin": dict(
        description=(
            "Gemelo FM-7/F7-b Pinball-Only de cog_v2_10tools_soft_k3 (w_ic=0) para calibración "
            "exacta de cuantiles."
        ),
        family="titans_cognitive_core",
        matched_control="cog_v2_10tools_soft_k3",
        ffn_mult=2.0,
        use_neuromod_memory=True,
        use_hrr_hopfield=True,
        use_hypergraph=True,
        use_holographic_3l=False,
        use_double_loop=False,
        use_moe=False,
        use_agent_core=False,
        use_multimodal_sota=False,
        use_cognitive_core=True,
        cognitive_tool_mode="soft",
        cognitive_toolbox_version="v2_10tools",
        cognitive_use_context_engineer=True,
        cognitive_cf_placebo=False,
        cognitive_ponder_weight=0.01,
        w_tool_aux=0.01,
        max_loops=3,
        noic_pinball_twin=True,
    ),
}


@dataclass(frozen=True)
class MMARRSConfig:
    """Contrato inmutable completo de `MMARRS-TITAN` con huella criptográfica SHA-256."""

    variation: str = "v13_mmarrs_titan_full"
    family: Literal[
        "titans_original", "holographic_brain", "mmarrs_titan", "titans_cognitive_core"
    ] = "mmarrs_titan"
    market: Literal["US", "CN"] = "US"
    matched_control: str | None = None
    # Dimensiones de todas las modalidades de entrada + física v4 + contexto alto SNR
    prices_dim: int = 46
    fundamentals_dim: int = 32
    macro_dim: int = 24
    news_dim: int = 32
    charts_dim: int = 32
    physics_dim: int = 24
    context_dim: int = 12
    exposures_dim: int = 9
    rmt_dim: int = 5
    # Tronco y cabezas granulares
    hidden_dim: int = 128
    ffn_mult: float = 2.0
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    # Higiene de entrada, bypass TabM y multimodal SOTA / FM-7
    use_gaussian_ranks: bool = True
    use_tabm_skip: bool = True
    tabm_members: int = 4
    use_multimodal_sota: bool = False
    ssl_weight: float = 0.02
    noic_pinball_twin: bool = False
    # Interruptores de ablación por variación
    use_neuromod_memory: bool = True
    use_dual_axis_titans: bool = False
    use_dual_level_rank_input: bool = False
    tabm_k: int = 1
    use_gaussian_rank_targets: bool = False
    w_stab_downside: float = 0.0
    bptt_steps: int = 4
    use_moe: bool = True
    moe_mode: str = "soft"
    n_shared_experts: int = 2
    n_routed_experts: int = 6
    use_double_loop: bool = True
    max_loops: int = 3
    use_looped_transformer: bool = False
    use_counterfactual_loop: bool = False
    cf_placebo_mode: bool = False
    loop_use_cross_attn: bool = True
    loop_n_landmarks: int = 16
    w_cf_fragility: float = 0.05
    use_holographic_3l: bool = True
    n_cortical_areas: int = 19
    n_hidden_slots: int = 4
    use_hrr_hopfield: bool = True
    hrr_max_slots: int = 512
    hrr_k_proto: int = 4
    use_hypergraph: bool = True
    n_hyperedges: int = 16
    use_agent_core: bool = True
    # Familia 4 (`titans_cognitive_core`) — desactivado por defecto en Familia 3
    use_cognitive_core: bool = False
    cognitive_tool_mode: str = "soft"
    cognitive_toolbox_version: Literal["v1_6tools", "v2_10tools"] = "v1_6tools"
    cognitive_use_context_engineer: bool = True
    cognitive_cf_placebo: bool = False
    cognitive_ponder_weight: float = 0.01
    w_tool_aux: float = 0.01
    # Interruptores de frontera Wave-3C (opt-in; valores por defecto conservan 100% retrocompat!)
    use_polar_alpha_head: bool = False
    use_set_modality_fusion: bool = False
    set_fusion_mnar_placebo: bool = False
    use_diff_xasset_attn: bool = False
    diff_xasset_zero_lambda_control: bool = False
    norm_kind: str = "layernorm"
    # Kernels Blackwell sm_120 y regularización
    use_fused_blackwell_kernels: bool = True
    weight_decay: float = 1e-2
    # Protocolo anti-fuga multi-horizonte (P0-4)
    purge_sessions: int = 63
    embargo_sessions: int = 13
    allow_short_synthetic: bool = False
    # Parámetros matemáticos certificados (PT1, Huber, Lipschitz, Early Exit)
    pt1_alpha_lo: float = 0.05
    pt1_eta_hi: float = 0.95
    pt1_theta_max: float = 0.25
    huber_delta: float = 1.0
    lipschitz_bound: float = 0.90
    early_exit_tol: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.variation not in VARIATION_REGISTRY:
            raise ValueError(
                f"Variación desconocida {self.variation!r}; "
                f"válidas: {tuple(VARIATION_REGISTRY.keys())}"
            )
        if self.family not in MMARRS_FAMILIES:
            raise ValueError(f"family inválida {self.family!r}; válidas: {MMARRS_FAMILIES}")
        if self.use_diff_xasset_attn and self.family != "mmarrs_titan":
            raise ValueError(
                f"use_diff_xasset_attn es exclusivo de Family 3 ('mmarrs_titan'); "
                f"recibido family={self.family!r}"
            )
        if self.norm_kind not in ("layernorm", "rmsnorm", "dyt"):
            raise ValueError(
                f"norm_kind inválido {self.norm_kind!r}; válidos: ('layernorm', 'rmsnorm', 'dyt')"
            )
        if self.market not in VALID_MARKETS:
            raise ValueError(
                f"FM-8: market inválido {self.market!r}; solo se permiten mercados separados "
                f"{VALID_MARKETS} (prohibido mezclar 'US+CN')"
            )
        if self.moe_mode not in MOE_MODES:
            raise ValueError(f"moe_mode inválido {self.moe_mode!r}; válidos: {MOE_MODES}")
        if self.cognitive_toolbox_version not in ("v1_6tools", "v2_10tools"):
            raise ValueError(
                f"cognitive_toolbox_version inválido {self.cognitive_toolbox_version!r}; "
                "válidos: ('v1_6tools', 'v2_10tools')"
            )
        if self.cognitive_tool_mode not in ("soft", "sparsemax", "uniform", "no_tools"):
            raise ValueError(f"cognitive_tool_mode inválido {self.cognitive_tool_mode!r}")
        if self.hidden_dim < 8 or not (1 <= self.max_loops <= 4):
            raise ValueError(
                f"hidden_dim >= 8 y tope duro 1 <= max_loops <= 4 son obligatorios "
                f"(recibido hidden_dim={self.hidden_dim}, max_loops={self.max_loops})"
            )
        if self.bptt_steps < 1:
            raise ValueError("bptt_steps debe ser >= 1")
        if self.hrr_max_slots < 1 or self.hrr_k_proto < 1:
            raise ValueError("hrr_max_slots y hrr_k_proto deben ser >= 1")
        if self.tabm_members < 1 or self.tabm_k < 1:
            raise ValueError("tabm_members y tabm_k deben ser >= 1")
        if self.loop_n_landmarks < 1:
            raise ValueError("loop_n_landmarks debe ser >= 1")
        if (
            self.weight_decay < 0.0
            or self.ssl_weight < 0.0
            or self.w_stab_downside < 0.0
            or self.w_cf_fragility < 0.0
            or self.w_tool_aux < 0.0
        ):
            raise ValueError(
                "weight_decay, ssl_weight, w_stab_downside, w_cf_fragility y w_tool_aux "
                "deben ser >= 0.0"
            )
        if not (0.0 < self.pt1_alpha_lo < 1.0):
            raise ValueError("pt1_alpha_lo debe pertenecer a (0, 1)")
        if not (0.0 < self.pt1_eta_hi < 1.0):
            raise ValueError("pt1_eta_hi debe pertenecer a (0, 1)")
        if not (0.0 < self.pt1_theta_max <= 1.0):
            raise ValueError("pt1_theta_max debe pertenecer a (0, 1]")
        if not (0.1 <= self.lipschitz_bound < 1.0):
            raise ValueError("lipschitz_bound debe ser estrictamente < 1.0 (Banach)")
        if tuple(sorted(self.quantile_levels)) != self.quantile_levels:
            raise ValueError("quantile_levels debe ser estrictamente creciente")
        if not self.allow_short_synthetic:
            min_purge, min_embargo = required_purge_embargo(self.horizons)
            if self.purge_sessions < min_purge:
                raise ValueError(
                    f"P0-4 fuga multi-horizonte: purge_sessions={self.purge_sessions} < "
                    f"max(horizons)={min_purge} (pase allow_short_synthetic=True solo en tests)"
                )
            if self.embargo_sessions < min_embargo:
                raise ValueError(
                    f"P0-4 embargo insuficiente: embargo_sessions={self.embargo_sessions} < "
                    f"{min_embargo} para max_h={min_purge} "
                    "(pase allow_short_synthetic=True solo en tests)"
                )

    def identity(self) -> dict[str, object]:
        """Representación canónica JSON-serializable de toda la arquitectura."""
        return dict(schema_version=2, architecture="MMARRS-TITAN", **asdict(self))

    def fingerprint(self) -> str:
        """Huella SHA-256 determinista del contrato completo."""
        payload = json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


# Alias 1:1 desde `tfm_catalog.py` (Familia 3) hacia las claves canónicas de `VARIATION_REGISTRY`
VARIATION_ALIASES: dict[str, str] = {
    "v00_dense_iso_flop": "v00_dense_baseline",
    "v02_neuromod_only": "v02_titans_neuromod_only",
    "v05_moe_time_shuffled_placebo": "v05_moe_shuffled_ctx_placebo",
    "v06_loop_only_k3": "v06_double_loop_pc_trm_only",
    "v07_moe_loop_k3": "v07_moe_double_loop",
    "v08_holo_bridge_comparison": "v08_holographic_3l_hrr_only",
    "v09_holo_moe_bridge_comparison": "v09_holographic_3l_moe_loop",
    "v10_rmt_hypergraph_only": "v10_cross_asset_hypergraph_only",
    "v11_moe_agent_p_plus": "v11_moe_plus_agent_core",
    "v12_moe_loop_p_plus": "v12_moe_loop_plus_agent_core",
    "v13_mmarrs_full": "v13_mmarrs_titan_full",
}


def resolve_variation_name(name: str) -> str:
    """Resuelve un nombre canónico o alias de `tfm_catalog.py` a `VARIATION_REGISTRY`."""
    resolved = VARIATION_ALIASES.get(name, name)
    if resolved not in VARIATION_REGISTRY:
        raise ValueError(f"Variación desconocida {name!r}; válidas: {MMARRS_VARIATIONS}")
    return resolved


def variation_config(name: str, **overrides: object) -> MMARRSConfig:
    """Construye el `MMARRSConfig` oficial para cualquier variación o alias de `tfm_catalog`."""
    resolved = resolve_variation_name(name)
    spec = {k: v for k, v in VARIATION_REGISTRY[resolved].items() if k != "description"}
    spec["variation"] = resolved

    # Normalizar `config_overrides` declarados en `tfm_catalog.py` (#84 P1-6)
    norm_overrides = dict(overrides)
    norm_overrides.pop("variation", None)
    norm_overrides.pop("shuffle_axis", None)
    cm_flag = bool(norm_overrides.pop("use_cross_modal_conflict_fusion", False))
    ssl_flag = bool(norm_overrides.pop("use_ssl_alignment", False))
    skew_flag = bool(norm_overrides.pop("use_skew_student_t_head", False))
    if cm_flag or ssl_flag or skew_flag:
        norm_overrides["use_multimodal_sota"] = True
    loss_mode = norm_overrides.pop("loss_mode", None)
    w_ic_ov = norm_overrides.pop("w_ic", None)
    if loss_mode == "pinball_only" or (w_ic_ov is not None and float(w_ic_ov) == 0.0):
        norm_overrides["noic_pinball_twin"] = True

    spec.update(norm_overrides)
    return MMARRSConfig(**spec)
