"""Catálogo Unificado de las 4 Familias Independientes TFM (`MMARRS-TITAN` + Entrega 2 + DGX).

Módulo autocontenido (cero mutaciones en `config.py` ni dependencias de GPU) que declara de forma
inmutable y sin omisiones el registro completo de variaciones, controles emparejados, políticas de
orden/nivel (FM-7), separación estricta de mercados `US` vs `CN` (FM-8) y tope duro de bucles
(`1 <= max_loops <= 4`, por omisión `<= 3`) para las **4 familias estrictamente independientes**
construidas sobre el fundamento Titans:
  1. `titans_original` — Baselines Entrega 2 (reglas, tabulares, secuenciales, RL), `Titans-MAC`,
     `MARS-TITAN` episódico (`M0..M3`), inventos DGX v12 (`TabM`, `CRRM`, `LFM-SW`, `CSOS`) y
     `titans_neuromod_sota` (+ gemelo `noic_pinball`).
  2. `holographic_brain` — Red de áreas Entrega 2 (`H1..H4`), inventos holográficos DGX v12
     (`HSI`, `CSRF`) y los 9 presets Neuro-IA independientes de `neuro_brain_sota.py`.
  3. `mmarrs_titan` — Todas las 16 variaciones y controles emparejados MoE + Doble Bucle `PC-TRM`
     + Hipergrafo RMT + Multimodal SOTA (`v00..v13c`).
  4. `titans_cognitive_core` — Núcleo Cognitivo weight-tied de Karpathy + Ingeniería de Contexto
     de Anthropic + 6 herramientas cuantitativas diferenciables en GPU (`cog_no_tools_k1..noic`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from .config import VARIATION_REGISTRY, MMARRSConfig, variation_config

_MMARRS_CONFIG_FIELDS: frozenset[str] = frozenset(f.name for f in dataclass_fields(MMARRSConfig))
_BUILD_VARIATION_PREFIX: str = "mars_titan.models.mmarrs.model:build_variation"

TFM_FAMILIES: tuple[str, ...] = (
    "titans_original",
    "holographic_brain",
    "mmarrs_titan",
    "titans_cognitive_core",
)

TFM_ORIGINS: tuple[str, ...] = (
    "entrega2_pdf",
    "dgx_v10_v12",
    "sota_innovation",
    "matched_control",
)

TFM_MARKETS: tuple[str, ...] = ("US", "CN")

ORDER_LEVEL_MODES: tuple[str, ...] = (
    "decoupled_f7b",
    "noic_pinball_twin",
    "raw",
)

DEFAULT_MAX_LOOPS: int = 3
MAX_ALLOWED_LOOPS: int = 4

_MARKET_MODELS: dict[str, str] = {
    "US": "spy_ols_252d",
    "CN": "csi300_required",
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ModelVariantSpec:
    """Especificación inmutable de una variación TFM con invariantes FM-7, FM-8 y anti-sopa."""

    variant_id: str
    family: str
    origin: str
    description: str
    matched_control_id: str
    supports_markets: tuple[str, ...]
    max_loops: int
    order_level_mode: str
    builder_target: str
    config_overrides: dict[str, Any]
    tags: tuple[str, ...]
    is_independent_family: bool = True
    cross_family_bridge_of: tuple[str, ...] = ()
    registry_key: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.variant_id or not self.variant_id.strip():
            raise ValueError("variant_id no puede estar vacío")
        if self.family not in TFM_FAMILIES:
            raise ValueError(f"family inválida {self.family!r}; válidas: {TFM_FAMILIES}")
        if self.origin not in TFM_ORIGINS:
            raise ValueError(f"origin inválido {self.origin!r}; válidos: {TFM_ORIGINS}")
        if not self.description or not self.description.strip():
            raise ValueError("description no puede estar vacía")
        if not self.matched_control_id or not self.matched_control_id.strip():
            raise ValueError("matched_control_id no puede estar vacío")

        if not self.supports_markets:
            raise ValueError("supports_markets no puede estar vacío")
        for mkt in self.supports_markets:
            if mkt not in TFM_MARKETS:
                raise ValueError(
                    f"Mercado inválido {mkt!r} en supports_markets; válidos: {TFM_MARKETS} "
                    "(FM-8 prohíbe mezclar mercados como 'US+CN')"
                )

        if not isinstance(self.max_loops, int) or isinstance(self.max_loops, bool):
            raise ValueError("max_loops debe ser entero")
        if not (1 <= self.max_loops <= MAX_ALLOWED_LOOPS):
            raise ValueError(
                f"max_loops={self.max_loops} viola el tope duro 1 <= max_loops <= "
                f"{MAX_ALLOWED_LOOPS}"
            )
        if self.order_level_mode not in ORDER_LEVEL_MODES:
            raise ValueError(
                f"order_level_mode inválido {self.order_level_mode!r}; válidos: {ORDER_LEVEL_MODES}"
            )
        if not self.builder_target or ":" not in self.builder_target:
            raise ValueError(
                f"builder_target debe seguir el formato 'modulo:simbolo', recibido: "
                f"{self.builder_target!r}"
            )
        if not isinstance(self.config_overrides, dict):
            raise ValueError("config_overrides debe ser un dict")
        if not self.is_independent_family:
            raise ValueError(
                "is_independent_family=True es obligatorio: las 4 familias son independientes"
            )

        # Invariante anti-sopa entre familias: nunca mezclar >1 familia externa ni omitir control
        if self.cross_family_bridge_of:
            if len(self.cross_family_bridge_of) > 1:
                raise ValueError(
                    f"Sopa inter-familia prohibida en {self.variant_id!r}: "
                    f"cross_family_bridge_of={self.cross_family_bridge_of} (>1 familia externa)"
                )
            ext = self.cross_family_bridge_of[0]
            if ext not in TFM_FAMILIES or ext == self.family:
                raise ValueError(
                    f"cross_family_bridge_of={self.cross_family_bridge_of!r} inválido para "
                    f"family={self.family!r}"
                )
            if self.matched_control_id == self.variant_id:
                raise ValueError(
                    f"Puente inter-familia {self.variant_id!r} exige un matched_control_id "
                    "distinto declarado dentro de su propia familia"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serializa la especificación a diccionario JSON-compatible."""
        return dict(
            variant_id=self.variant_id,
            family=self.family,
            origin=self.origin,
            description=self.description,
            matched_control_id=self.matched_control_id,
            supports_markets=list(self.supports_markets),
            max_loops=self.max_loops,
            order_level_mode=self.order_level_mode,
            builder_target=self.builder_target,
            config_overrides=dict(self.config_overrides),
            tags=list(self.tags),
            is_independent_family=self.is_independent_family,
            cross_family_bridge_of=list(self.cross_family_bridge_of),
            registry_key=self.registry_key,
            aliases=list(self.aliases),
        )

    def fingerprint_sha256(self) -> str:
        """Huella SHA-256 determinista del contrato completo de la variación."""
        return _canonical_sha256(self.to_dict())


def _spec(
    variant_id: str,
    family: str,
    origin: str,
    description: str,
    matched_control_id: str,
    builder_target: str,
    *,
    max_loops: int = 1,
    order_level_mode: str = "decoupled_f7b",
    config_overrides: dict[str, Any] | None = None,
    tags: tuple[str, ...] = (),
    supports_markets: tuple[str, ...] = TFM_MARKETS,
    cross_family_bridge_of: tuple[str, ...] = (),
    registry_key: str | None = None,
    aliases: tuple[str, ...] = (),
) -> ModelVariantSpec:
    return ModelVariantSpec(
        variant_id=variant_id,
        family=family,
        origin=origin,
        description=description,
        matched_control_id=matched_control_id,
        supports_markets=supports_markets,
        max_loops=max_loops,
        order_level_mode=order_level_mode,
        builder_target=builder_target,
        config_overrides=dict(config_overrides or {}),
        tags=tags,
        is_independent_family=True,
        cross_family_bridge_of=cross_family_bridge_of,
        registry_key=registry_key,
        aliases=aliases,
    )


def _build_default_catalog() -> dict[str, ModelVariantSpec]:
    items: list[ModelVariantSpec] = [
        # =====================================================================
        # FAMILY 1 — titans_original (Entrega 2 + Titans-MAC + MARS + DGX v12 + SOTA)
        # =====================================================================
        _spec(
            "zero",
            "titans_original",
            "matched_control",
            "Entrega 2 mandatory zero-return baseline reference.",
            "zero",
            "mars_titan.models.baselines:zero_baseline",
            order_level_mode="raw",
            tags=("baseline", "rule", "anchor"),
        ),
        _spec(
            "past_median",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 60d rolling past median return baseline (lag 2).",
            "zero",
            "mars_titan.models.baselines:past_median_baseline",
            order_level_mode="raw",
            config_overrides={"window": 60, "lag": 2},
            tags=("baseline", "rule"),
        ),
        _spec(
            "momentum_reversal",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 5d reversal + window momentum rule baseline.",
            "zero",
            "mars_titan.models.baselines.momentum_reversal:MomentumReversalBaseline",
            order_level_mode="raw",
            tags=("baseline", "rule"),
        ),
        _spec(
            "ridge",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 cross-sectional session Ridge regression baseline.",
            "zero",
            "mars_titan.models.baselines.session_ridge:SessionRidgeBaseline",
            order_level_mode="decoupled_f7b",
            tags=("baseline", "linear"),
        ),
        _spec(
            "xgboost",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 gradient-boosted trees XGBoost baseline.",
            "zero",
            "mars_titan.models.baselines.boosting:XGBoostBaseline",
            order_level_mode="decoupled_f7b",
            tags=("baseline", "tabular", "gbdts"),
        ),
        _spec(
            "lightgbm",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 same-inputs / Qlib Alpha158 LightGBM tabular benchmark.",
            "zero",
            "mars_titan.models.baselines.same_inputs_lightgbm:SameInputsLightGBMBaseline",
            order_level_mode="decoupled_f7b",
            tags=("baseline", "tabular", "gbdts"),
        ),
        _spec(
            "rnn",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 multimodal vanilla RNN sequence baseline.",
            "zero",
            "mars_titan.models.baselines.multimodal:RecurrentMultimodalBaseline[rnn]",
            config_overrides={"cell": "rnn"},
            tags=("baseline", "sequence"),
        ),
        _spec(
            "lstm",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 multimodal LSTM sequence baseline.",
            "zero",
            "mars_titan.models.baselines.multimodal:RecurrentMultimodalBaseline[lstm]",
            config_overrides={"cell": "lstm"},
            tags=("baseline", "sequence"),
        ),
        _spec(
            "gru",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 multimodal GRU sequence baseline.",
            "zero",
            "mars_titan.models.baselines.multimodal:RecurrentMultimodalBaseline[gru]",
            config_overrides={"cell": "gru"},
            tags=("baseline", "sequence"),
        ),
        _spec(
            "dlinear",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 DLinear trend/seasonal decomposition sequence baseline.",
            "zero",
            "mars_titan.models.baselines.dlinear:DLinearBaseline",
            tags=("baseline", "sequence"),
        ),
        _spec(
            "transformer_compact",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 compact Transformer sequence baseline.",
            "zero",
            "mars_titan.models.baselines.transformer:CompactTransformerBaseline",
            tags=("baseline", "sequence", "transformer"),
        ),
        _spec(
            "kronlstm",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 Annex B KronLSTM structured price encoder baseline.",
            "lstm",
            "mars_titan.models.kronlstm.price_encoder:KronLSTMPriceEncoder",
            config_overrides={"variant": "kronlstm_v1"},
            tags=("baseline", "sequence", "annex_b"),
        ),
        _spec(
            "rl_ppo",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 §4.2.6 PPO clipped-KL portfolio policy benchmark.",
            "zero",
            "mars_titan.rl.policies:PPOPolicy",
            order_level_mode="raw",
            config_overrides={"variant": "ppo_clip_full_kl"},
            tags=("baseline", "rl_portfolio"),
        ),
        _spec(
            "rl_double_dqn",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 §4.2.6 Double DQN discrete portfolio policy benchmark.",
            "zero",
            "mars_titan.rl.policies:DoubleDQNPolicy",
            order_level_mode="raw",
            tags=("baseline", "rl_portfolio"),
        ),
        _spec(
            "rl_klpo",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 §4.2.6 KLPO quadratic trust-region portfolio policy.",
            "rl_ppo",
            "mars_titan.models.klpo:KLPOPolicy",
            order_level_mode="raw",
            tags=("baseline", "rl_portfolio"),
        ),
        _spec(
            "rl_grpo",
            "titans_original",
            "entrega2_pdf",
            "Entrega 2 §4.2.6 GRPO / Dr.GRPO group-relative portfolio policy.",
            "rl_ppo",
            "mars_titan.rl.policies:GRPOPolicy",
            order_level_mode="raw",
            config_overrides={"variant": "dr_grpo_outcome"},
            tags=("baseline", "rl_portfolio"),
        ),
        # Core Titans-MAC & matched memory-mode controls
        _spec(
            "titans_mac_disabled",
            "titans_original",
            "matched_control",
            "Titans-MAC matched control with neural memory disabled.",
            "zero",
            "mars_titan.models.titans.mac:TitansMAC[disabled]",
            config_overrides={"memory_mode": "disabled"},
            tags=("titans_mac", "control"),
        ),
        _spec(
            "titans_mac_frozen",
            "titans_original",
            "matched_control",
            "Titans-MAC matched control with memory frozen during evaluation.",
            "titans_mac_disabled",
            "mars_titan.models.titans.mac:TitansMAC[frozen]",
            config_overrides={"memory_mode": "frozen"},
            tags=("titans_mac", "control"),
        ),
        _spec(
            "titans_mac_online",
            "titans_original",
            "entrega2_pdf",
            "Titans-MAC online depth-2 MLP fast memory + WP4 FM-1/FM-2 fixes.",
            "titans_mac_frozen",
            "mars_titan.models.titans.mac:TitansMAC[online]",
            config_overrides={
                "memory_mode": "online",
                "memory_value_normalization": "layer_norm_v1",
                "output_query": "projected_v1",
                "session_gaussian_ranks": True,
            },
            tags=("titans_mac", "online_memory"),
        ),
        _spec(
            "titans_mac_noic_pinball",
            "titans_original",
            "dgx_v10_v12",
            "FM-7/F7-b pinball-only width twin for Titans-MAC (w_ic=0).",
            "titans_mac_online",
            "mars_titan.models.titans.mac:TitansMAC[noic_pinball]",
            order_level_mode="noic_pinball_twin",
            config_overrides={"memory_mode": "online", "loss_mode": "pinball_only", "w_ic": 0.0},
            tags=("titans_mac", "f7b_twin", "calibration"),
        ),
        # MARS-TITAN Episodic Bank ladder (M0..M3, K=1,2,4, familiarity_gate_r3)
        _spec(
            "mars_titan_m0_no_bank",
            "titans_original",
            "matched_control",
            "MARS-TITAN M0 control: refiner active without episodic bank.",
            "titans_mac_online",
            "mars_titan.models.titans.episodic_readout:MARSTitanEpisodic[M0]",
            config_overrides={"admission_policy": "M0", "k_reads": 1},
            tags=("mars_titan", "episodic", "control"),
        ),
        _spec(
            "mars_titan_m1_uniform",
            "titans_original",
            "matched_control",
            "MARS-TITAN M1 control: uniform episodic bank admission.",
            "mars_titan_m0_no_bank",
            "mars_titan.models.titans.episodic_readout:MARSTitanEpisodic[M1]",
            config_overrides={"admission_policy": "M1", "k_reads": 1},
            tags=("mars_titan", "episodic", "control"),
        ),
        _spec(
            "mars_titan_m2_mature_error",
            "titans_original",
            "entrega2_pdf",
            "MARS-TITAN M2 selective bank (50% uniform / 25% top error / 25% recent).",
            "mars_titan_m1_uniform",
            "mars_titan.models.titans.episodic_readout:MARSTitanEpisodic[M2]",
            config_overrides={"admission_policy": "M2", "k_reads": 1},
            tags=("mars_titan", "episodic"),
        ),
        _spec(
            "mars_titan_m3_k1",
            "titans_original",
            "entrega2_pdf",
            "MARS-TITAN M3 full surprise-scored episodic bank (K=1 read).",
            "mars_titan_m1_uniform",
            "mars_titan.models.titans.episodic_readout:MARSTitanEpisodic[M3_K1]",
            max_loops=1,
            config_overrides={
                "admission_policy": "M3",
                "k_reads": 1,
                "label_path": "direct_v1",
            },
            tags=("mars_titan", "episodic"),
        ),
        _spec(
            "mars_titan_m3_k2",
            "titans_original",
            "entrega2_pdf",
            "MARS-TITAN M3 episodic bank with K=2 iterative refinement reads.",
            "mars_titan_m3_k1",
            "mars_titan.models.titans.episodic_readout:MARSTitanEpisodic[M3_K2]",
            max_loops=2,
            config_overrides={
                "admission_policy": "M3",
                "k_reads": 2,
                "label_path": "direct_v1",
            },
            tags=("mars_titan", "episodic", "iterative"),
        ),
        _spec(
            "mars_titan_m3_k4",
            "titans_original",
            "entrega2_pdf",
            "MARS-TITAN M3 episodic bank at hard cap K=4 iterative reads (Entrega 2 Table 4).",
            "mars_titan_m3_k1",
            "mars_titan.models.titans.episodic_readout:MARSTitanEpisodic[M3_K4]",
            max_loops=4,
            config_overrides={
                "admission_policy": "M3",
                "k_reads": 4,
                "label_path": "direct_v1",
            },
            tags=("mars_titan", "episodic", "iterative"),
        ),
        _spec(
            "mars_titan_m3_familiarity_gate_r3",
            "titans_original",
            "entrega2_pdf",
            "MARS-TITAN M3 + R3 familiarity gate controlling bank injection strength.",
            "mars_titan_m3_k2",
            "mars_titan.models.titans.episodic_readout:MARSTitanEpisodic[M3_R3]",
            max_loops=2,
            config_overrides={
                "admission_policy": "M3",
                "k_reads": 2,
                "familiarity_gate": True,
                "label_path": "direct_v1",
            },
            tags=("mars_titan", "episodic", "familiarity_gate"),
        ),
        # DGX v12 breakthroughs
        _spec(
            "titans_tabm",
            "titans_original",
            "dgx_v10_v12",
            "DGX v12 WP7 TabM BatchEnsemble (K=8) + masked ranks + long_horizon_v5.",
            "titans_mac_online",
            "mars_titan.models.titans.technical_branch:TitansTabM[K8]",
            config_overrides={
                "ensemble_members": 8,
                "absent_rows": "masked_v1",
                "feature_set": "long_horizon_v5",
            },
            tags=("dgx_v12", "tabm", "wp7"),
        ),
        _spec(
            "titans_crrm",
            "titans_original",
            "dgx_v10_v12",
            "DGX v12 WP10 CRRM supervised cross-sectional Cholesky rank memory (5/21/63d).",
            "titans_mac_frozen",
            "mars_titan.models.titans.cross_sectional_rank_memory:CRRMModel",
            config_overrides={"horizons": [5, 21, 63], "ood_gate": True},
            tags=("dgx_v12", "crrm", "wp10"),
        ),
        _spec(
            "titans_lfm_sw",
            "titans_original",
            "dgx_v10_v12",
            "DGX v12 WP10 LFM-SW low-rank factor memory with Newton-Schulz whitening.",
            "titans_mac_frozen",
            "mars_titan.models.titans.factor_memory:LFMSWModel",
            config_overrides={"spectral_whitening": "newton_schulz"},
            tags=("dgx_v12", "lfm_sw", "wp10"),
        ),
        _spec(
            "titans_csos",
            "titans_original",
            "dgx_v10_v12",
            "DGX v12 WP10 CSOS cross-sectional orthogonalized surprise memory update.",
            "titans_mac_online",
            "mars_titan.models.titans.mac:TitansMAC[csos]",
            config_overrides={"orthogonalize_session_mean_surprise": True},
            tags=("dgx_v12", "csos", "wp10"),
        ),
        # Family 1 SOTA Innovation + NOIC pinball twin
        _spec(
            "titans_neuromod_sota",
            "titans_original",
            "sota_innovation",
            "Family 1 SOTA: ACh/NE + Gated DeltaNet-2 + PT1 Lyapunov + Huber + Benna-Fusi "
            "3-scale + Gaussian ranks + TabM skip (standalone, NO MoE/PC-TRM/CogCore).",
            "titans_mac_frozen",
            "mars_titan.models.mmarrs.model:build_variation[v02_titans_neuromod_only]",
            max_loops=1,
            order_level_mode="decoupled_f7b",
            registry_key="v02_titans_neuromod_only",
            config_overrides={
                "family": "titans_original",
                "use_neuromod_memory": True,
                "huber_delta": 1.0,
                "use_gaussian_ranks": True,
                "use_tabm_skip": True,
                "tabm_members": 8,
                "use_moe": False,
                "use_double_loop": False,
            },
            tags=("sota", "neuromod", "pt1", "benna_fusi", "tabm"),
        ),
        _spec(
            "titans_neuromod_sota_noic_pinball",
            "titans_original",
            "sota_innovation",
            "FM-7/F7-b pinball-only width twin for `titans_neuromod_sota` (w_ic=0).",
            "titans_neuromod_sota",
            "mars_titan.models.mmarrs.model:build_variation[v02_titans_neuromod_only]",
            max_loops=1,
            order_level_mode="noic_pinball_twin",
            registry_key="v02_titans_neuromod_only",
            config_overrides={
                "family": "titans_original",
                "use_neuromod_memory": True,
                "noic_pinball_twin": True,
                "use_moe": False,
                "use_double_loop": False,
            },
            tags=("sota", "neuromod", "f7b_twin", "calibration"),
        ),
        # Family 1 v2 SOTA (DualAxisTitansMemory + Dual [z, rank] + TabM K=8 + stab05)
        _spec(
            "t15a_dual_axis_mem",
            "titans_original",
            "sota_innovation",
            "Family 1 v2 ablation: DualAxisTitansMemory (proximal FP32 market matrix + Tukey + "
            "CSOS + Benna-Fusi asset bank + scale-free NE), anchored to titans_neuromod_sota.",
            "titans_neuromod_sota",
            "mars_titan.models.mmarrs.model:build_variation[t15a_dual_axis_mem]",
            max_loops=1,
            order_level_mode="decoupled_f7b",
            registry_key="t15a_dual_axis_mem",
            config_overrides={
                "family": "titans_original",
                "use_dual_axis_titans": True,
            },
            tags=("sota", "v2", "dual_axis_titans", "neuromod", "pt1"),
        ),
        _spec(
            "t15b_full_v2",
            "titans_original",
            "sota_innovation",
            "Family 1 v2 flagship: DualAxisTitansMemory + dual [clipped_z, g_rank] input + "
            "head-only TabM K=8 + Gaussian IC + liqw3 + downside semivariance stab05.",
            "titans_neuromod_sota",
            "mars_titan.models.mmarrs.model:build_variation[t15b_full_v2]",
            max_loops=1,
            order_level_mode="decoupled_f7b",
            registry_key="t15b_full_v2",
            config_overrides={
                "family": "titans_original",
                "use_dual_axis_titans": True,
                "use_dual_level_rank_input": True,
                "tabm_k": 8,
                "use_gaussian_rank_targets": True,
                "w_stab_downside": 0.05,
            },
            tags=("sota", "v2", "flagship", "dual_axis_titans", "tabm", "stab05"),
        ),
        _spec(
            "t15c_noic_twin",
            "titans_original",
            "sota_innovation",
            "FM-7/F7-b pinball-only width twin for `t15b_full_v2` (w_ic=0, w_stab_downside=0).",
            "t15b_full_v2",
            "mars_titan.models.mmarrs.model:build_variation[t15c_noic_twin]",
            max_loops=1,
            order_level_mode="noic_pinball_twin",
            registry_key="t15c_noic_twin",
            config_overrides={
                "family": "titans_original",
                "use_dual_axis_titans": True,
                "use_dual_level_rank_input": True,
                "tabm_k": 8,
                "noic_pinball_twin": True,
            },
            tags=("sota", "v2", "f7b_twin", "calibration"),
        ),
        # =====================================================================
        # FAMILY 2 — holographic_brain (Entrega 2/DGX L1 + 9 v1 + 4 v2 Neuro-AI Presets)
        # =====================================================================
        _spec(
            "h2_catalog_only",
            "holographic_brain",
            "matched_control",
            "Entrega 2 H2 control: fixed catalog membership only (Delta=0).",
            "h2_catalog_only",
            "mars_titan.models.titans.holographic_network:HolographicNetwork[H2]",
            config_overrides={"catalog_membership_only": True, "delta_zero": True},
            tags=("entrega2", "holographic_l1", "control"),
        ),
        _spec(
            "h3_hidden_groups",
            "holographic_brain",
            "matched_control",
            "Entrega 2 H3 control: Soft-MoE hidden groups only without catalog areas.",
            "h2_catalog_only",
            "mars_titan.models.titans.holographic_network:HolographicNetwork[H3]",
            config_overrides={"hidden_groups_only": True},
            tags=("entrega2", "holographic_l1", "control"),
        ),
        _spec(
            "h1_single_zero_5",
            "holographic_brain",
            "entrega2_pdf",
            "Entrega 2 H1 area network (5 areas) with WP8 `single_zero_v1` init.",
            "h2_catalog_only",
            "mars_titan.models.titans.holographic_network:HolographicNetwork[H1_5]",
            config_overrides={"n_cortical_areas": 5, "init_mode": "single_zero_v1"},
            tags=("entrega2", "holographic_l1", "wp8"),
        ),
        _spec(
            "h1_single_zero_19",
            "holographic_brain",
            "entrega2_pdf",
            "Entrega 2 H1 area network (19 cortical areas) with WP8 `single_zero_v1` init.",
            "h2_catalog_only",
            "mars_titan.models.titans.holographic_network:HolographicNetwork[H1_19]",
            config_overrides={"n_cortical_areas": 19, "init_mode": "single_zero_v1"},
            tags=("entrega2", "holographic_l1", "wp8"),
        ),
        _spec(
            "h1_single_zero_38",
            "holographic_brain",
            "entrega2_pdf",
            "Entrega 2 H1 area network (38 fine areas) with WP8 `single_zero_v1` init.",
            "h2_catalog_only",
            "mars_titan.models.titans.holographic_network:HolographicNetwork[H1_38]",
            config_overrides={"n_cortical_areas": 38, "init_mode": "single_zero_v1"},
            tags=("entrega2", "holographic_l1", "wp8"),
        ),
        _spec(
            "h4_asset_groups_shuffled_ctrl",
            "holographic_brain",
            "matched_control",
            "Entrega 2 H4 shuffled-rows placebo control for AssetGroupMembership.",
            "h2_catalog_only",
            "mars_titan.models.titans.asset_groups:AssetGroupMembership[shuffled]",
            config_overrides={"shuffled_rows": True},
            tags=("entrega2", "asset_groups", "placebo", "control"),
        ),
        _spec(
            "h4_asset_groups",
            "holographic_brain",
            "entrega2_pdf",
            "Entrega 2 H4 causal 64d AssetGroupMembership (5 buckets + 8 fuzzy c-means).",
            "h4_asset_groups_shuffled_ctrl",
            "mars_titan.models.titans.asset_groups:AssetGroupMembership[H4]",
            config_overrides={"n_fuzzy_clusters": 8, "causal_window": 64},
            tags=("entrega2", "asset_groups"),
        ),
        _spec(
            "hsi_session_hologram",
            "holographic_brain",
            "dgx_v10_v12",
            "DGX v12 WP11 HSI O(N) complex interferometric leave-one-out session hologram.",
            "h2_catalog_only",
            "mars_titan.models.titans.session_hologram:HolographicSessionInterference",
            config_overrides={"leave_one_out": True, "complex_phase": True},
            tags=("dgx_v12", "hsi", "wp11"),
        ),
        _spec(
            "csrf_hadamard_resonator",
            "holographic_brain",
            "dgx_v10_v12",
            "DGX v12 WP11 CSRF Walsh-Hadamard cross-sectional session resonator (K<=3).",
            "h2_catalog_only",
            "mars_titan.models.titans.session_resonator:CrossSectionalSessionResonator",
            max_loops=3,
            config_overrides={"walsh_hadamard": True, "leave_one_out": True},
            tags=("dgx_v12", "csrf", "wp11"),
        ),
        # All 9 standalone Neuro-AI SOTA presets from `neuro_brain_sota.py`
        _spec(
            "nb_bare_holographic_3l",
            "holographic_brain",
            "matched_control",
            "Standalone Neuro-Brain control: bare 3-level HolographicBrain3L (K=1).",
            "h2_catalog_only",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation"
            "[bare_holographic_3l]",
            max_loops=1,
            config_overrides={"ablation": "bare_holographic_3l"},
            tags=("neuro_brain", "control"),
        ),
        _spec(
            "nb_standalone_k1",
            "holographic_brain",
            "matched_control",
            "Standalone Neuro-Brain K=1 control (dendritic+inhibition+3-factor+replay, no loops).",
            "nb_bare_holographic_3l",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation[standalone_k1]",
            max_loops=1,
            config_overrides={"ablation": "standalone_k1"},
            tags=("neuro_brain", "control"),
        ),
        _spec(
            "nb_full",
            "holographic_brain",
            "sota_innovation",
            "Family 2 SOTA: StandaloneHolographicBrainModel full (3L + Dendritic + Lateral "
            "Inhibition + 3-Factor Dopamine PT1 + Sharp-Wave Replay + PC K<=3, NO MoE).",
            "nb_standalone_k1",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation[full]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "full", "max_cycles": DEFAULT_MAX_LOOPS},
            tags=("neuro_brain", "sota", "standalone"),
        ),
        _spec(
            "nb_no_dendritic",
            "holographic_brain",
            "matched_control",
            "Neuro-Brain lesion ablation removing DendriticCompartmentLayer.",
            "nb_bare_holographic_3l",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation[no_dendritic]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "no_dendritic"},
            tags=("neuro_brain", "lesion", "control"),
        ),
        _spec(
            "nb_no_lateral_inhibition",
            "holographic_brain",
            "matched_control",
            "Neuro-Brain lesion ablation removing LateralInhibitionCompetition.",
            "nb_bare_holographic_3l",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation"
            "[no_lateral_inhibition]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "no_lateral_inhibition"},
            tags=("neuro_brain", "lesion", "control"),
        ),
        _spec(
            "nb_no_three_factor",
            "holographic_brain",
            "matched_control",
            "Neuro-Brain lesion ablation removing ThreeFactorDopaminePlasticity.",
            "nb_bare_holographic_3l",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation"
            "[no_three_factor]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "no_three_factor"},
            tags=("neuro_brain", "lesion", "control"),
        ),
        _spec(
            "nb_no_replay",
            "holographic_brain",
            "matched_control",
            "Neuro-Brain lesion ablation removing HippocampalSharpWaveReplay.",
            "nb_bare_holographic_3l",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation[no_replay]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "no_replay"},
            tags=("neuro_brain", "lesion", "control"),
        ),
        _spec(
            "nb_holographic_moe_uniform_k3",
            "holographic_brain",
            "matched_control",
            "Controlled bridge placebo: Neuro-Brain + uniform 1/E MoE bridge (K<=3).",
            "nb_full",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation"
            "[holographic_moe_uniform_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "holographic_moe_uniform_k3"},
            tags=("neuro_brain", "bridge", "control"),
            cross_family_bridge_of=("mmarrs_titan",),
        ),
        _spec(
            "nb_holographic_moe_soft_k3",
            "holographic_brain",
            "sota_innovation",
            "Controlled bridge test: Neuro-Brain + soft High-SNR MoE bridge (K<=3).",
            "nb_holographic_moe_uniform_k3",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation"
            "[holographic_moe_soft_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "holographic_moe_soft_k3"},
            tags=("neuro_brain", "bridge", "sota"),
            cross_family_bridge_of=("mmarrs_titan",),
        ),
        # Family 2 v2 Computational Neuroscience SOTA (4 presets, exact IDs without double prefix)
        _spec(
            "nb_v2_fhrr_lgn_pi",
            "holographic_brain",
            "sota_innovation",
            "Family 2 v2 ablation: Unitary real-pair FHRR (0 FFT) + LGN dual [z, xrank] + "
            "Active-Inference per-asset precision Pi_{i,t} (K<=3).",
            "nb_full",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation"
            "[nb_v2_fhrr_lgn_pi]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "nb_v2_fhrr_lgn_pi", "max_cycles": DEFAULT_MAX_LOOPS},
            tags=("neuro_brain", "v2", "sota", "fhrr", "lgn", "active_inference"),
        ),
        _spec(
            "nb_v2_full",
            "holographic_brain",
            "sota_innovation",
            "Family 2 v2 flagship: Unitary FHRR + LGN dual + Active-Inference Pi + Turrigiano "
            "homeostatic scaling + causal BCM/W_tag + 3-strata SWR replay (K<=3, NO MoE).",
            "nb_full",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation[nb_v2_full]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "nb_v2_full", "max_cycles": DEFAULT_MAX_LOOPS},
            tags=("neuro_brain", "v2", "sota", "flagship", "standalone"),
        ),
        _spec(
            "nb_v2_noic_twin",
            "holographic_brain",
            "sota_innovation",
            "FM-7/F7-b pinball-only width twin for `nb_v2_full` (w_ic=0).",
            "nb_v2_full",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation[nb_v2_noic_twin]",
            max_loops=DEFAULT_MAX_LOOPS,
            order_level_mode="noic_pinball_twin",
            config_overrides={
                "ablation": "nb_v2_noic_twin",
                "max_cycles": DEFAULT_MAX_LOOPS,
                "noic_pinball_twin": True,
            },
            tags=("neuro_brain", "v2", "f7b_twin", "calibration"),
        ),
        _spec(
            "nb_v2_moe_soft_k3",
            "holographic_brain",
            "sota_innovation",
            "Controlled bridge test: Family 2 v2 full + soft High-SNR MoE bridge (K<=3).",
            "nb_v2_full",
            "mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation"
            "[nb_v2_moe_soft_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={"ablation": "nb_v2_moe_soft_k3", "max_cycles": DEFAULT_MAX_LOOPS},
            tags=("neuro_brain", "v2", "bridge", "sota"),
            cross_family_bridge_of=("mmarrs_titan",),
        ),
        # =====================================================================
        # FAMILY 3 — mmarrs_titan (16 Frontier Variations aligned 1:1 with VARIATION_REGISTRY)
        # =====================================================================
        _spec(
            "v00_dense_baseline",
            "mmarrs_titan",
            "matched_control",
            "Matched Control 1 (Iso-FLOP dense FFN, K=1, no MoE/loop/memory).",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v00_dense_baseline]",
            max_loops=1,
            registry_key="v00_dense_baseline",
            aliases=("v00_dense_iso_flop",),
            config_overrides={},
            tags=("mmarrs", "control", "iso_flop"),
        ),
        _spec(
            "v01_dense_wide_iso_param",
            "mmarrs_titan",
            "matched_control",
            "Matched Control 2 (Wide dense FFN matching total params of v03 MoE, K=1).",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v01_dense_wide_iso_param]",
            max_loops=1,
            registry_key="v01_dense_wide_iso_param",
            config_overrides={},
            tags=("mmarrs", "control", "iso_param"),
        ),
        _spec(
            "v02_titans_neuromod_only",
            "mmarrs_titan",
            "sota_innovation",
            "Ablation isolating NeuromodulatedTitansMemory (ACh/NE + PT1 + Huber, K=1).",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v02_titans_neuromod_only]",
            max_loops=1,
            registry_key="v02_titans_neuromod_only",
            aliases=("v02_neuromod_only",),
            config_overrides={},
            tags=("mmarrs", "ablation", "neuromod"),
        ),
        _spec(
            "v03_moe_only_k1",
            "mmarrs_titan",
            "matched_control",
            "High-SNR MoE (2 shared + 6 routed, Aux-Loss-Free, K=1 fixed depth control).",
            "v01_dense_wide_iso_param",
            "mars_titan.models.mmarrs.model:build_variation[v03_moe_only_k1]",
            max_loops=1,
            registry_key="v03_moe_only_k1",
            config_overrides={},
            tags=("mmarrs", "moe", "control", "k1_fixed"),
        ),
        _spec(
            "v04_moe_uniform_control",
            "mmarrs_titan",
            "matched_control",
            "Matched Control 3: MoE with uniform 1/E routing (zero specialization).",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v04_moe_uniform_control]",
            max_loops=1,
            registry_key="v04_moe_uniform_control",
            config_overrides={"moe_mode": "uniform"},
            tags=("mmarrs", "moe", "control", "uniform"),
        ),
        _spec(
            "v05_moe_shuffled_ctx_placebo",
            "mmarrs_titan",
            "matched_control",
            "Matched Control 4: MoE placebo with time-session shuffled routing context.",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v05_moe_shuffled_ctx_placebo]",
            max_loops=1,
            registry_key="v05_moe_shuffled_ctx_placebo",
            aliases=("v05_moe_time_shuffled_placebo",),
            config_overrides={"moe_mode": "shuffled_context"},
            tags=("mmarrs", "moe", "control", "placebo"),
        ),
        _spec(
            "v06_double_loop_pc_trm_only",
            "mmarrs_titan",
            "sota_innovation",
            "Ablation isolating contractive PC-TRM double loop (K<=3, Lipschitz<1, no MoE).",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v06_double_loop_pc_trm_only]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v06_double_loop_pc_trm_only",
            aliases=("v06_loop_only_k3",),
            config_overrides={"max_loops": DEFAULT_MAX_LOOPS},
            tags=("mmarrs", "pc_trm", "loop"),
        ),
        _spec(
            "v07_moe_double_loop",
            "mmarrs_titan",
            "sota_innovation",
            "Recursive High-SNR MoE + PC-TRM double loop (per-step routers R^(k), K<=3).",
            "v03_moe_only_k1",
            "mars_titan.models.mmarrs.model:build_variation[v07_moe_double_loop]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v07_moe_double_loop",
            aliases=("v07_moe_loop_k3",),
            config_overrides={"max_loops": DEFAULT_MAX_LOOPS},
            tags=("mmarrs", "moe", "pc_trm", "loop"),
        ),
        _spec(
            "v08_holographic_3l_hrr_only",
            "mmarrs_titan",
            "sota_innovation",
            "Controlled bridge comparison: HolographicBrain3L + HRR inside MMARRS shell (K=1).",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v08_holographic_3l_hrr_only]",
            max_loops=1,
            registry_key="v08_holographic_3l_hrr_only",
            aliases=("v08_holo_bridge_comparison",),
            config_overrides={},
            tags=("mmarrs", "bridge", "comparison"),
            cross_family_bridge_of=("holographic_brain",),
        ),
        _spec(
            "v09_holographic_3l_moe_loop",
            "mmarrs_titan",
            "sota_innovation",
            "Controlled bridge comparison: Holographic3L + MoE + PC-TRM loop (K<=3).",
            "v07_moe_double_loop",
            "mars_titan.models.mmarrs.model:build_variation[v09_holographic_3l_moe_loop]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v09_holographic_3l_moe_loop",
            aliases=("v09_holo_moe_bridge_comparison",),
            config_overrides={"max_loops": DEFAULT_MAX_LOOPS},
            tags=("mmarrs", "bridge", "comparison", "moe", "loop"),
            cross_family_bridge_of=("holographic_brain",),
        ),
        _spec(
            "v10_cross_asset_hypergraph_only",
            "mmarrs_titan",
            "sota_innovation",
            "Ablation isolating O(N*K) permutation-equivariant RMT spillover hypergraph (K=1).",
            "v00_dense_baseline",
            "mars_titan.models.mmarrs.model:build_variation[v10_cross_asset_hypergraph_only]",
            max_loops=1,
            registry_key="v10_cross_asset_hypergraph_only",
            aliases=("v10_rmt_hypergraph_only",),
            config_overrides={},
            tags=("mmarrs", "hypergraph", "rmt"),
        ),
        _spec(
            "v11_moe_plus_agent_core",
            "mmarrs_titan",
            "sota_innovation",
            "MoE (K=1) + metacognitive governor A5-A8 & differentiable convex P+ layer.",
            "v03_moe_only_k1",
            "mars_titan.models.mmarrs.model:build_variation[v11_moe_plus_agent_core]",
            max_loops=1,
            registry_key="v11_moe_plus_agent_core",
            aliases=("v11_moe_agent_p_plus",),
            config_overrides={},
            tags=("mmarrs", "moe", "portfolio_p_plus"),
        ),
        _spec(
            "v12_moe_loop_plus_agent_core",
            "mmarrs_titan",
            "sota_innovation",
            "MoE + PC-TRM loop (K<=3) + metacognitive governor A5-A8 & convex P+.",
            "v07_moe_double_loop",
            "mars_titan.models.mmarrs.model:build_variation[v12_moe_loop_plus_agent_core]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v12_moe_loop_plus_agent_core",
            aliases=("v12_moe_loop_p_plus",),
            config_overrides={"max_loops": DEFAULT_MAX_LOOPS},
            tags=("mmarrs", "moe", "loop", "portfolio_p_plus"),
        ),
        _spec(
            "v13_mmarrs_titan_full",
            "mmarrs_titan",
            "sota_innovation",
            "Family 3 flagship `MMARRS-TITAN` full stack (MoE + PC-TRM K<=3 + Hypergraph + "
            "Neuromod + HRR + Granular Heads).",
            "v04_moe_uniform_control",
            "mars_titan.models.mmarrs.model:build_variation[v13_mmarrs_titan_full]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v13_mmarrs_titan_full",
            aliases=("v13_mmarrs_full",),
            config_overrides={"max_loops": DEFAULT_MAX_LOOPS},
            tags=("mmarrs", "sota", "flagship"),
        ),
        _spec(
            "v13b_mmarrs_multimodal_sota",
            "mmarrs_titan",
            "sota_innovation",
            "Family 3 SOTA+: `v13_mmarrs_titan_full` + CrossModalConflictFusion + asset-varying "
            "VICReg/Barlow SSL (w=0.02) + SkewStudentTDistributionHead.",
            "v13_mmarrs_titan_full",
            "mars_titan.models.mmarrs.model:build_variation[v13b_mmarrs_multimodal_sota]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v13b_mmarrs_multimodal_sota",
            config_overrides={
                "max_loops": DEFAULT_MAX_LOOPS,
                "use_multimodal_sota": True,
                "ssl_weight": 0.02,
            },
            tags=("mmarrs", "sota", "multimodal_sota", "skew_t"),
        ),
        _spec(
            "v13c_mmarrs_noic_pinball_twin",
            "mmarrs_titan",
            "sota_innovation",
            "FM-7/F7-b pinball-only width twin for `v13_mmarrs_titan_full` (w_ic=0).",
            "v13_mmarrs_titan_full",
            "mars_titan.models.mmarrs.model:build_variation[v13c_mmarrs_noic_pinball_twin]",
            max_loops=DEFAULT_MAX_LOOPS,
            order_level_mode="noic_pinball_twin",
            registry_key="v13c_mmarrs_noic_pinball_twin",
            config_overrides={
                "max_loops": DEFAULT_MAX_LOOPS,
                "noic_pinball_twin": True,
            },
            tags=("mmarrs", "f7b_twin", "calibration"),
        ),
        # Family 3 v2: Weight-Tied LoopedCrossAssetTransformer + Counterfactual Verify-Retry Loop
        _spec(
            "v14a_tied_loop",
            "mmarrs_titan",
            "sota_innovation",
            "Family 3 v2 K=1 ablation: weight-tied LoopedCrossAssetTransformer (no cross-attn, "
            "no counterfactual), anchored to v13b_mmarrs_multimodal_sota.",
            "v13b_mmarrs_multimodal_sota",
            "mars_titan.models.mmarrs.model:build_variation[v14a_tied_loop]",
            max_loops=1,
            registry_key="v14a_tied_loop",
            config_overrides={
                "use_looped_transformer": True,
                "loop_use_cross_attn": False,
                "use_counterfactual_loop": False,
                "max_loops": 1,
            },
            tags=("mmarrs", "v2", "looped_transformer", "k1_fixed"),
        ),
        _spec(
            "v14b_tied_xasset",
            "mmarrs_titan",
            "sota_innovation",
            "Family 3 v2 K=3 ablation: weight-tied LoopedCrossAssetTransformer + F-invariant "
            "InducedFactorCrossAttention O(N*R) (without counterfactual loop).",
            "v14a_tied_loop",
            "mars_titan.models.mmarrs.model:build_variation[v14b_tied_xasset]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v14b_tied_xasset",
            config_overrides={
                "use_looped_transformer": True,
                "loop_use_cross_attn": True,
                "use_counterfactual_loop": False,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("mmarrs", "v2", "looped_transformer", "xasset_attn"),
        ),
        _spec(
            "v14c_tied_xasset_cf",
            "mmarrs_titan",
            "sota_innovation",
            "Family 3 v2 flagship K=3: weight-tied LoopedCrossAssetTransformer + "
            "InducedFactorCrossAttention + 5-scenario Counterfactual Verify-Retry Loop.",
            "v14b_tied_xasset",
            "mars_titan.models.mmarrs.model:build_variation[v14c_tied_xasset_cf]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v14c_tied_xasset_cf",
            config_overrides={
                "use_looped_transformer": True,
                "loop_use_cross_attn": True,
                "use_counterfactual_loop": True,
                "cf_placebo_mode": False,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("mmarrs", "v2", "sota", "flagship", "looped_transformer", "counterfactual"),
        ),
        _spec(
            "v14c_placebo",
            "mmarrs_titan",
            "matched_control",
            "Matched counterfactual placebo for v14c: zero-fixed-point derangement (N//2) on "
            "CF shock deltas preserving marginal distribution while breaking causal alignment.",
            "v14b_tied_xasset",
            "mars_titan.models.mmarrs.model:build_variation[v14c_placebo]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="v14c_placebo",
            config_overrides={
                "use_looped_transformer": True,
                "loop_use_cross_attn": True,
                "use_counterfactual_loop": True,
                "cf_placebo_mode": True,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("mmarrs", "v2", "control", "placebo", "counterfactual"),
        ),
        _spec(
            "v14c_noic_twin",
            "mmarrs_titan",
            "sota_innovation",
            "FM-7/F7-b pinball-only width twin for `v14c_tied_xasset_cf` (w_ic=0).",
            "v14c_tied_xasset_cf",
            "mars_titan.models.mmarrs.model:build_variation[v14c_noic_twin]",
            max_loops=DEFAULT_MAX_LOOPS,
            order_level_mode="noic_pinball_twin",
            registry_key="v14c_noic_twin",
            config_overrides={
                "use_looped_transformer": True,
                "loop_use_cross_attn": True,
                "use_counterfactual_loop": True,
                "noic_pinball_twin": True,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("mmarrs", "v2", "f7b_twin", "calibration"),
        ),
        # =====================================================================
        # FAMILY 4 — titans_cognitive_core (Weight-Tied Core + 6 v1 / 10 v2 GPU Tools)
        # =====================================================================
        _spec(
            "cog_no_tools_k1",
            "titans_cognitive_core",
            "matched_control",
            "Family 4 anchor control: weight-tied Cognitive Core K=1 with tools disabled.",
            "cog_no_tools_k1",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[no_tools_k1]",
            max_loops=1,
            config_overrides={
                "tool_mode": "no_tools",
                "thinking_steps": 1,
                "use_context_engineer": True,
            },
            tags=("cognitive_core", "control", "no_tools"),
        ),
        _spec(
            "cog_no_tools_k3",
            "titans_cognitive_core",
            "matched_control",
            "Family 4 matched control: weight-tied Cognitive Core K=3 without quant tools.",
            "cog_no_tools_k1",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[no_tools_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={
                "tool_mode": "no_tools",
                "thinking_steps": DEFAULT_MAX_LOOPS,
                "use_context_engineer": True,
            },
            tags=("cognitive_core", "control", "no_tools"),
        ),
        _spec(
            "cog_uniform_tools_k3",
            "titans_cognitive_core",
            "matched_control",
            "Family 4 matched control: uniform 1/6 tool weights across all 6 GPU tools (K=3).",
            "cog_no_tools_k3",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[uniform_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={
                "tool_mode": "uniform",
                "thinking_steps": DEFAULT_MAX_LOOPS,
                "use_context_engineer": True,
            },
            tags=("cognitive_core", "control", "uniform_tools"),
        ),
        _spec(
            "cog_soft_tools_k3",
            "titans_cognitive_core",
            "sota_innovation",
            "Family 4 SOTA: Karpathy weight-tied Core + Anthropic ContextEngineerGPU + "
            "soft ToolRouter over 6 GPU quant tools + AdaptiveHaltGate (K<=3).",
            "cog_uniform_tools_k3",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[soft_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            aliases=("cog_v1_6tools_k3",),
            config_overrides={
                "tool_mode": "soft",
                "thinking_steps": DEFAULT_MAX_LOOPS,
                "use_context_engineer": True,
                "use_adaptive_halt": True,
                "w_ponder_kl": 0.01,
            },
            tags=("cognitive_core", "sota", "soft_tools"),
        ),
        _spec(
            "cog_sparsemax_tools_k3",
            "titans_cognitive_core",
            "sota_innovation",
            "Family 4 SOTA: sparsemax ToolRouter (exact sparse tool invocation, K<=3).",
            "cog_uniform_tools_k3",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[sparsemax_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={
                "tool_mode": "sparsemax",
                "thinking_steps": DEFAULT_MAX_LOOPS,
                "use_context_engineer": True,
                "use_adaptive_halt": True,
                "w_ponder_kl": 0.01,
            },
            tags=("cognitive_core", "sota", "sparsemax_tools"),
        ),
        _spec(
            "cog_no_context_eng_k3",
            "titans_cognitive_core",
            "matched_control",
            "Family 4 ablation removing Anthropic ContextEngineerGPU pruning & working notes.",
            "cog_soft_tools_k3",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[no_context_eng_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={
                "tool_mode": "soft",
                "thinking_steps": DEFAULT_MAX_LOOPS,
                "use_context_engineer": False,
                "use_adaptive_halt": True,
            },
            tags=("cognitive_core", "ablation", "control"),
        ),
        _spec(
            "cog_core_plus_moe_k3",
            "titans_cognitive_core",
            "sota_innovation",
            "Controlled bridge test: Cognitive Core (K<=3) + HighSNRMoELayer bridge.",
            "cog_soft_tools_k3",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[plus_moe_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            config_overrides={
                "tool_mode": "soft",
                "thinking_steps": DEFAULT_MAX_LOOPS,
                "use_context_engineer": True,
                "use_moe_bridge": True,
            },
            tags=("cognitive_core", "bridge", "sota"),
            cross_family_bridge_of=("mmarrs_titan",),
        ),
        _spec(
            "cog_core_noic_twin",
            "titans_cognitive_core",
            "sota_innovation",
            "FM-7/F7-b pinball-only width twin for `cog_soft_tools_k3` (w_ic=0).",
            "cog_soft_tools_k3",
            "mars_titan.models.mmarrs.cognitive_core:TitansCognitiveCore[noic_twin]",
            max_loops=DEFAULT_MAX_LOOPS,
            order_level_mode="noic_pinball_twin",
            config_overrides={
                "tool_mode": "soft",
                "thinking_steps": DEFAULT_MAX_LOOPS,
                "use_context_engineer": True,
                "loss_mode": "pinball_only",
                "w_ic": 0.0,
            },
            tags=("cognitive_core", "f7b_twin", "calibration"),
        ),
        # Family 4 v2: 10-Tool Karpathy/Anthropic Cognitive Core + Closed-Loop Router
        _spec(
            "cog_v2_10tools_uniform_k3",
            "titans_cognitive_core",
            "matched_control",
            "Family 4 v2 matched control: uniform 1/10 tool weights over all 10 GPU tools (K=3).",
            "cog_no_tools_k3",
            "mars_titan.models.mmarrs.model:build_variation[cog_v2_10tools_uniform_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="cog_v2_10tools_uniform_k3",
            config_overrides={
                "family": "titans_cognitive_core",
                "use_cognitive_core": True,
                "cognitive_toolbox_version": "v2_10tools",
                "cognitive_tool_mode": "uniform",
                "cognitive_cf_placebo": False,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("cognitive_core", "v2", "10tools", "control", "uniform_tools"),
        ),
        _spec(
            "cog_v2_10tools_shuffled_cf_placebo_k3",
            "titans_cognitive_core",
            "matched_control",
            "Family 4 v2 matched placebo control: 10-tool soft router with T8 CF deranged placebo.",
            "cog_v2_10tools_uniform_k3",
            "mars_titan.models.mmarrs.model:build_variation[cog_v2_10tools_shuffled_cf_placebo_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="cog_v2_10tools_shuffled_cf_placebo_k3",
            config_overrides={
                "family": "titans_cognitive_core",
                "use_cognitive_core": True,
                "cognitive_toolbox_version": "v2_10tools",
                "cognitive_tool_mode": "soft",
                "cognitive_cf_placebo": True,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("cognitive_core", "v2", "10tools", "control", "placebo"),
        ),
        _spec(
            "cog_v2_10tools_soft_k3",
            "titans_cognitive_core",
            "sota_innovation",
            "Family 4 v2 flagship: weight-tied Cognitive Core + ContextEngineerGPU + closed-loop "
            "soft ToolRouter over 10 orthogonal GPU quant tools + AdaptiveHaltGate v2 (K<=3).",
            "cog_v2_10tools_uniform_k3",
            "mars_titan.models.mmarrs.model:build_variation[cog_v2_10tools_soft_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="cog_v2_10tools_soft_k3",
            config_overrides={
                "family": "titans_cognitive_core",
                "use_cognitive_core": True,
                "cognitive_toolbox_version": "v2_10tools",
                "cognitive_tool_mode": "soft",
                "cognitive_cf_placebo": False,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("cognitive_core", "v2", "10tools", "sota", "flagship", "soft_tools"),
        ),
        _spec(
            "cog_v2_10tools_sparsemax_k3",
            "titans_cognitive_core",
            "sota_innovation",
            "Family 4 v2 SOTA: closed-loop sparsemax ToolRouter over 10 GPU quant tools (K<=3).",
            "cog_v2_10tools_uniform_k3",
            "mars_titan.models.mmarrs.model:build_variation[cog_v2_10tools_sparsemax_k3]",
            max_loops=DEFAULT_MAX_LOOPS,
            registry_key="cog_v2_10tools_sparsemax_k3",
            config_overrides={
                "family": "titans_cognitive_core",
                "use_cognitive_core": True,
                "cognitive_toolbox_version": "v2_10tools",
                "cognitive_tool_mode": "sparsemax",
                "cognitive_cf_placebo": False,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("cognitive_core", "v2", "10tools", "sota", "sparsemax_tools"),
        ),
        _spec(
            "cog_v2_10tools_noic_twin",
            "titans_cognitive_core",
            "sota_innovation",
            "FM-7/F7-b pinball-only width twin for `cog_v2_10tools_soft_k3` (w_ic=0).",
            "cog_v2_10tools_soft_k3",
            "mars_titan.models.mmarrs.model:build_variation[cog_v2_10tools_noic_twin]",
            max_loops=DEFAULT_MAX_LOOPS,
            order_level_mode="noic_pinball_twin",
            registry_key="cog_v2_10tools_noic_twin",
            config_overrides={
                "family": "titans_cognitive_core",
                "use_cognitive_core": True,
                "cognitive_toolbox_version": "v2_10tools",
                "cognitive_tool_mode": "soft",
                "noic_pinball_twin": True,
                "max_loops": DEFAULT_MAX_LOOPS,
            },
            tags=("cognitive_core", "v2", "10tools", "f7b_twin", "calibration"),
        ),
    ]

    catalog: dict[str, ModelVariantSpec] = {}
    for item in items:
        if item.variant_id in catalog:
            raise ValueError(f"variant_id duplicado en TFM_VARIANT_CATALOG: {item.variant_id!r}")
        catalog[item.variant_id] = item
    return catalog


def _build_aliases(catalog: Mapping[str, ModelVariantSpec]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical_id, spec in catalog.items():
        for alias in spec.aliases:
            if alias in catalog:
                raise ValueError(f"Alias {alias!r} colisiona con variant_id canónico")
            if alias in aliases and aliases[alias] != canonical_id:
                raise ValueError(
                    f"Alias duplicado {alias!r}: {aliases[alias]!r} vs {canonical_id!r}"
                )
            aliases[alias] = canonical_id
    return aliases


TFM_VARIANT_CATALOG: dict[str, ModelVariantSpec] = _build_default_catalog()
TFM_VARIANT_ALIASES: dict[str, str] = _build_aliases(TFM_VARIANT_CATALOG)


def resolve_variant_id(variant_id: str) -> str:
    """Resuelve un `variant_id` canónico o alias legado a su identificador canónico."""
    if variant_id in TFM_VARIANT_CATALOG:
        return variant_id
    if variant_id in TFM_VARIANT_ALIASES:
        return TFM_VARIANT_ALIASES[variant_id]
    raise KeyError(
        f"Variación desconocida {variant_id!r}; total registradas: {len(TFM_VARIANT_CATALOG)}"
    )


def get_variant(variant_id: str) -> ModelVariantSpec:
    """Recupera la especificación inmutable de `variant_id` (acepta aliases) o lanza `KeyError`."""
    canonical = resolve_variant_id(variant_id)
    return TFM_VARIANT_CATALOG[canonical]


def list_variants(
    family: str | None = None,
    origin: str | None = None,
    *,
    tag: str | None = None,
    market: str | None = None,
) -> list[ModelVariantSpec]:
    """Filtra el catálogo por familia, origen, etiqueta y/o mercado."""
    if family is not None and family not in TFM_FAMILIES:
        raise ValueError(f"family inválida {family!r}; válidas: {TFM_FAMILIES}")
    if origin is not None and origin not in TFM_ORIGINS:
        raise ValueError(f"origin inválido {origin!r}; válidos: {TFM_ORIGINS}")
    if market is not None and market not in TFM_MARKETS:
        raise ValueError(
            f"market inválido {market!r}; válidos: {TFM_MARKETS} (FM-8 prohíbe mezclar mercados)"
        )

    out: list[ModelVariantSpec] = []
    for spec in TFM_VARIANT_CATALOG.values():
        if family is not None and spec.family != family:
            continue
        if origin is not None and spec.origin != origin:
            continue
        if tag is not None and tag not in spec.tags:
            continue
        if market is not None and market not in spec.supports_markets:
            continue
        out.append(spec)
    return out


def build_ablation_matrix(
    family: str,
    market: str = "US",
    *,
    include_bridges: bool = True,
    seeds: tuple[int, ...] = (42, 43, 44),
    n_hpo_trials_per_variant: int = 1,
) -> dict[str, Any]:
    """Construye la matriz ejecutable de ablación para una familia y un único mercado (FM-8)."""
    if family not in TFM_FAMILIES:
        raise ValueError(f"family inválida {family!r}; válidas: {TFM_FAMILIES}")
    if market not in TFM_MARKETS:
        raise ValueError(
            f"market inválido {market!r}; válidos: {TFM_MARKETS}. "
            "FM-8 prohíbe entrenar/evaluar mercados combinados (como 'US+CN')."
        )
    if not seeds:
        raise ValueError("seeds no puede estar vacío")
    if int(n_hpo_trials_per_variant) < 1:
        raise ValueError("n_hpo_trials_per_variant debe ser >= 1")

    fam_specs = list_variants(family=family, market=market)
    if not include_bridges:
        fam_specs = [s for s in fam_specs if not s.cross_family_bridge_of]

    controls = [s.to_dict() for s in fam_specs if s.origin == "matched_control"]
    candidates = [s.to_dict() for s in fam_specs if s.origin != "matched_control"]
    ordered_variants = controls + candidates
    n_seeds = len(seeds)
    n_hpo = int(n_hpo_trials_per_variant)
    n_trials = len(ordered_variants) * n_seeds * n_hpo

    body = dict(
        schema_version=1,
        family=family,
        market=market,
        market_model=_MARKET_MODELS[market],
        markets_strictly_separate=True,
        families_strictly_independent=True,
        train_window="2000-2022" if market == "US" else "2006-2022",
        val_window="2023",
        sealed_test_start_year=2024,
        include_bridges=bool(include_bridges),
        seeds=[int(s) for s in seeds],
        n_seeds=n_seeds,
        n_hpo_trials_per_variant=n_hpo,
        n_controls=len(controls),
        n_candidates=len(candidates),
        n_variants=len(ordered_variants),
        n_trials=n_trials,
        controls=controls,
        candidates=candidates,
        variants=ordered_variants,
    )
    return dict(**body, matrix_sha256=_canonical_sha256(body))


def validate_catalog_integrity(
    catalog: Mapping[str, ModelVariantSpec] | None = None,
) -> dict[str, Any]:
    """Audita todas las invariantes del catálogo y lanza `ValueError` ante cualquier fallo."""
    cat = TFM_VARIANT_CATALOG if catalog is None else catalog
    errors: list[str] = []

    counts_by_family = {fam: 0 for fam in TFM_FAMILIES}
    controls_by_family = {fam: 0 for fam in TFM_FAMILIES}
    independent_non_controls = {fam: 0 for fam in TFM_FAMILIES}
    counts_by_origin = {orig: 0 for orig in TFM_ORIGINS}
    alias_seen: dict[str, str] = {}

    for key, spec in cat.items():
        if key != spec.variant_id:
            errors.append(f"Clave desalineada: key={key!r} != variant_id={spec.variant_id!r}")
        if spec.family not in TFM_FAMILIES:
            errors.append(f"Familia desconocida en {spec.variant_id!r}: {spec.family!r}")
            continue

        for alias in spec.aliases:
            if alias in cat:
                errors.append(f"{spec.variant_id}: alias {alias!r} colisiona con clave canónica")
            elif alias in alias_seen and alias_seen[alias] != spec.variant_id:
                errors.append(
                    f"{spec.variant_id}: alias duplicado {alias!r} "
                    f"(ya apunta a {alias_seen[alias]!r})"
                )
            else:
                alias_seen[alias] = spec.variant_id

        counts_by_family[spec.family] += 1
        counts_by_origin[spec.origin] += 1
        if spec.origin == "matched_control":
            controls_by_family[spec.family] += 1
        elif not spec.cross_family_bridge_of:
            independent_non_controls[spec.family] += 1

        if not (1 <= spec.max_loops <= MAX_ALLOWED_LOOPS):
            errors.append(f"{spec.variant_id}: max_loops={spec.max_loops} > {MAX_ALLOWED_LOOPS}")

        resolved_ctrl = spec.matched_control_id
        if resolved_ctrl not in cat and resolved_ctrl in alias_seen:
            resolved_ctrl = alias_seen[resolved_ctrl]
        if resolved_ctrl not in cat:
            errors.append(
                f"{spec.variant_id}: matched_control_id={spec.matched_control_id!r} no existe"
            )
        else:
            ctrl = cat[resolved_ctrl]
            if spec.cross_family_bridge_of and ctrl.family != spec.family:
                errors.append(
                    f"{spec.variant_id}: cross-family bridge control {ctrl.variant_id!r} "
                    f"must belong to the same family ({spec.family!r}, got {ctrl.family!r})"
                )

        if spec.order_level_mode == "noic_pinball_twin" and (
            spec.matched_control_id == spec.variant_id
        ):
            errors.append(f"{spec.variant_id}: noic_pinball_twin debe emparejar un modelo primario")

        if spec.builder_target.startswith(_BUILD_VARIATION_PREFIX):
            rkey = spec.registry_key
            if not rkey or rkey not in VARIATION_REGISTRY:
                errors.append(
                    f"{spec.variant_id}: registry_key={rkey!r} no existe en VARIATION_REGISTRY"
                )
            else:
                if spec.family == "mmarrs_titan" and spec.variant_id != rkey:
                    errors.append(
                        f"{spec.variant_id}: Familia 3 debe coincidir con registry_key={rkey!r}"
                    )
                unknown_keys = sorted(set(spec.config_overrides) - _MMARRS_CONFIG_FIELDS)
                if unknown_keys:
                    errors.append(
                        f"{spec.variant_id}: config_overrides inválidos para MMARRSConfig: "
                        f"{unknown_keys}"
                    )
                else:
                    try:
                        variation_config(rkey, **spec.config_overrides)
                    except Exception as exc:
                        errors.append(
                            f"{spec.variant_id}: variation_config({rkey!r}, ...) falló: {exc}"
                        )

    for fam in TFM_FAMILIES:
        if controls_by_family[fam] < 1:
            errors.append(f"Familia {fam!r} carece de controles emparejados (matched_control)")
        if independent_non_controls[fam] < 1:
            errors.append(f"Familia {fam!r} carece de variaciones independientes propias")

    if errors:
        raise ValueError("Fallos de integridad en TFM_VARIANT_CATALOG:\n- " + "\n- ".join(errors))

    return dict(
        valid=True,
        total_variants=len(cat),
        counts_by_family=counts_by_family,
        controls_by_family=controls_by_family,
        counts_by_origin=counts_by_origin,
        max_loops_observed=max((s.max_loops for s in cat.values()), default=0),
    )


def serialize_catalog_json(
    path: str | Path | None = None,
    *,
    compress: bool | None = None,
) -> str:
    """Serializa el catálogo completo en JSON compacto (y `.json.gz` opcional sin pérdida)."""
    integrity = validate_catalog_integrity()
    body = dict(
        schema_version=1,
        families=list(TFM_FAMILIES),
        markets=list(TFM_MARKETS),
        sealed_test_start_year=2024,
        max_allowed_loops=MAX_ALLOWED_LOOPS,
        total_variants=len(TFM_VARIANT_CATALOG),
        integrity=integrity,
        variants={vid: spec.to_dict() for vid, spec in TFM_VARIANT_CATALOG.items()},
    )
    payload = dict(**body, catalog_sha256=_canonical_sha256(body))
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        use_gz = bool(compress) if compress is not None else p.suffix == ".gz"
        tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
        if use_gz:
            with gzip.open(tmp, "wb", compresslevel=6) as fh:
                fh.write(compact.encode("utf-8"))
        else:
            tmp.write_text(compact, encoding="utf-8")
        os.replace(tmp, p)
    return compact


# Autoverificación al importar: impide regresiones silenciosas en el registro canónico
validate_catalog_integrity(TFM_VARIANT_CATALOG)


__all__ = [
    "DEFAULT_MAX_LOOPS",
    "MAX_ALLOWED_LOOPS",
    "ORDER_LEVEL_MODES",
    "TFM_FAMILIES",
    "TFM_MARKETS",
    "TFM_ORIGINS",
    "TFM_VARIANT_ALIASES",
    "TFM_VARIANT_CATALOG",
    "ModelVariantSpec",
    "build_ablation_matrix",
    "get_variant",
    "list_variants",
    "resolve_variant_id",
    "serialize_catalog_json",
    "validate_catalog_integrity",
]
