"""Núcleo Cognitivo Compacto (`TitansCognitiveCore`), Ingeniería de Contexto y Caja Cuantitativa.

Aplica como variación financiera nativa en GPU de `MMARRS-TITAN` tres principios complementarios:
  1. **Anthropic Effective Context Engineering (`ContextEngineerGPU`)**: decide qué alimentar y
     qué podar en cada sesión mediante compuertas causales de relación señal/ruido (SNR),
     decaimiento exponencial de frescura (`modality_age`), aislamiento exacto de modalidades
     ausentes (`torch.where`) y notas externas de trabajo comprimidas (`[M_notes, D]`):
     en `v1_6tools` 4 ranuras persistentes; en `v2_10tools` 8 ranuras unificadas (`0..3`
     persistentes entre sesiones `t-1 -> t` + `4..7` bloque de borrador intra-sesión reseteado
     en cada sesión `t` para evitar degradación de contexto).
  2. **Caja de Herramientas Cuantitativas Diferenciables (`DifferentiableQuantToolbox`)**:
     - `v1_6tools` (compatibilidad estricta hacia atrás): 6 herramientas deterministas en GPU
       (`rmt_factor_decompose`, `hrr_episodic_lookup`, `hypergraph_spillover`,
       `structural_verifier`, `portfolio_feasibility`, `epistemic_risk_check`).
     - `v2_10tools` (actualización SOTA `#130` + `#133`): 10 herramientas estrictamente
       ortogonales añadiendo `cross_sectional_rank_calibrator` (T6),
       `cross_modal_conflict_arbiter` (T7), `regime_stress_counterfactual` (T8, delegando en
       `CounterfactualStressSimulator` compartido) y `working_memory_scratchpad_ops` (T9,
       fusionado directamente sobre `cognitive_notes [8, D]`).
     Todas con MLPs cuello de botella de 2 capas en herramientas escalares, `LayerNorm` de RMS
     unitario por herramienta, cero bypass crudo de `h`, ruta rápida Triton FISTA
     (`aux_req_grad=False`) y **cero sincronizaciones CPU↔GPU** (`audit_zero_host_syncs == 0`).
  3. **Andrej Karpathy Compact Weight-Tied Cognitive Core (`TitansCognitiveCore`)**: núcleo
     recurrente compacto con compartición estricta de pesos (`thinking_steps=3` por defecto,
     tope duro `1 <= thinking_steps <= 4`), cuello de botella `core_dim <= hidden_dim`,
     estabilidad contractiva normalizada espectralmente (`Lipschitz < 1`), `ToolRouter v2` de
     lazo cerrado con prior débil de fase `±0.35` (`Gather -> Verify -> Repair`) y annealing
     coseno de temperatura, trío limpio anti-colapso (sesgo DeepSeek-V3 + `tool_balance_loss`
     en media de lote + `tool_decor_loss`), y `AdaptiveHaltGate v2` condicionado a dificultad
     compuesta desacoplada con congelación zero-sync en evaluación (`torch.where`).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features
from mars_titan.evaluation import metacognitive_agent as _ma
from mars_titan.evaluation.metacognitive_agent import (
    DifferentiablePortfolioLayer,
    project_dollar_neutral_box_torch,
)

from .config import DEFAULT_HORIZONS
from .counterfactual_sim import (
    CF_TRACE_DIM,
    CTX_LIQ_PCT,
    CTX_VOL_PCT,
    CTX_WASS,
    CounterfactualStressSimulator,
    prepare_cf_factor_cache,
)
from .dl_training_sota import build_norm_layer
from .hypergraph_hrr import HRRModernHopfieldMemory, RMTSpilloverHypergraph
from .mor_loom_pc import _sparsemax

DEFAULT_THINKING_STEPS: int = 3
MAX_THINKING_STEPS: int = 4
DEFAULT_PONDER_WEIGHT: float = 0.01

TOOLBOX_VERSIONS: tuple[str, ...] = ("v1_6tools", "v2_10tools")

TOOL_NAMES: tuple[str, ...] = (
    "rmt_factor_decompose",
    "hrr_episodic_lookup",
    "hypergraph_spillover",
    "structural_verifier",
    "portfolio_feasibility",
    "epistemic_risk_check",
)

TOOL_NAMES_V2: tuple[str, ...] = TOOL_NAMES + (
    "cross_sectional_rank_calibrator",
    "cross_modal_conflict_arbiter",
    "regime_stress_counterfactual",
    "working_memory_scratchpad_ops",
)

TOOL_ROUTER_MODES: tuple[str, ...] = ("soft", "sparsemax", "uniform", "no_tools")


def _inv_softplus(value: float) -> float:
    """Inversa numéricamente estable de `softplus` para inicialización positiva exacta."""
    v = max(float(value), 1e-4)
    if v > 20.0:
        return v
    return math.log(math.expm1(v))


def _stable_calc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promociona FP16/BF16 a FP32 para álgebra lineal/logsumexp y preserva FP32/FP64."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _make_scalar_tool_mlp(
    in_features: int, hidden_features: int, out_features: int
) -> nn.Sequential:
    """MLP cuello de botella de 2 capas (`feat -> core_dim//2 -> D`) para herramientas escalares."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.SiLU(),
        nn.Linear(hidden_features, out_features),
    )


def _init_phase_prior(max_steps: int, n_tools: int) -> torch.Tensor:
    """Prior débil aprendible `±0.35` (`Gather -> Verify -> Repair`) para `ToolRouter v2`."""
    prior = torch.zeros(max_steps, n_tools, dtype=torch.float32)
    if n_tools != 10:
        return prior
    gather_idx = (0, 1, 2, 6, 7, 9)
    verify_idx = (3, 4, 5, 7, 8)
    repair_idx = (0, 4, 5, 6, 8, 9)
    for k in range(max_steps):
        prior[k].fill_(-0.35)
        if k == 0:
            active = gather_idx
        elif k == 1:
            active = verify_idx
        else:
            active = repair_idx
        for idx in active:
            prior[k, idx] = 0.35
    return prior


def _spectral_norm_sq_psd(gram: torch.Tensor, *, n_iters: int = 5) -> torch.Tensor:
    """Autovalor dominante de matriz Gram PSD `[K, K]` en tensores puros (cero cuSOLVER/syncs!)."""
    col_l1 = gram.abs().sum(dim=-1)
    gersh_max = col_l1.max().clamp_min(0.0)
    best_col = torch.argmax(col_l1)
    v = F.normalize(gram.index_select(1, best_col.reshape(1)).squeeze(1), dim=0, eps=1e-12)
    for _ in range(max(1, int(n_iters))):
        gv = gram @ v
        v = F.normalize(gv, dim=0, eps=1e-12)
    rayleigh = torch.dot(v, gram @ v).clamp_min(0.0)
    return torch.minimum(gersh_max, rayleigh * 1.02)


def _project_dollar_neutral_box(
    z: torch.Tensor,
    *,
    pos_cap: float,
    gross_limit: float,
) -> torch.Tensor:
    """Proyección euclídea diferenciable exacta (1-CTA Triton + derivada implícita, 0 syncs)."""
    return project_dollar_neutral_box_torch(z, pos_cap=pos_cap, gross_limit=gross_limit)


def _blend_head_predictions(
    pred_cand: dict[str, torch.Tensor],
    pred_prev: dict[str, torch.Tensor],
    accept_k: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combina hipótesis/predicciones por `accept_k` (exacto cuando `accept_k=1`, ahorra `head`)."""
    a2 = accept_k.unsqueeze(-1)
    b2 = 1.0 - a2
    a3 = a2.unsqueeze(-1)
    b3 = 1.0 - a3

    out: dict[str, torch.Tensor] = dict(
        pred_return=a2 * pred_cand["pred_return"] + b2 * pred_prev["pred_return"],
        alpha=a2 * pred_cand["alpha"] + b2 * pred_prev["alpha"],
        epistemic_error=a2 * pred_cand["epistemic_error"] + b2 * pred_prev["epistemic_error"],
    )
    if "factor_return" in pred_cand and "factor_return" in pred_prev:
        out["factor_return"] = a2 * pred_cand["factor_return"] + b2 * pred_prev["factor_return"]
    if "quantiles" in pred_cand and pred_cand["quantiles"] is not None:
        if "quantiles" in pred_prev and pred_prev["quantiles"] is not None:
            out["quantiles"] = a3 * pred_cand["quantiles"] + b3 * pred_prev["quantiles"]
        else:
            out["quantiles"] = pred_cand["quantiles"]
    if "tb_logits" in pred_cand and "tb_logits" in pred_prev:
        tb_logits = a3 * pred_cand["tb_logits"] + b3 * pred_prev["tb_logits"]
        out["tb_logits"] = tb_logits
        out["tb_probs"] = torch.softmax(tb_logits, dim=-1)
    if "turn_logits" in pred_cand and "turn_logits" in pred_prev:
        turn_logits = a2 * pred_cand["turn_logits"] + b2 * pred_prev["turn_logits"]
        out["turn_logits"] = turn_logits
        out["turning_prob"] = torch.sigmoid(turn_logits)
    if "log_vol" in pred_cand and "log_vol" in pred_prev:
        log_vol = a2 * pred_cand["log_vol"] + b2 * pred_prev["log_vol"]
        out["log_vol"] = log_vol
        out["vol_pred"] = torch.exp(log_vol)
    else:
        out["vol_pred"] = a2 * pred_cand["vol_pred"] + b2 * pred_prev["vol_pred"]
    return out


class ContextEngineerGPU(nn.Module):
    """Ingeniería de Contexto causal en GPU: compuertas SNR, poda por frescura y notas compactas."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        n_modalities: int = 5,
        n_notes: int = 4,
        n_persistent_notes: int | None = None,
        bottleneck_dim: int | None = None,
        prune_threshold: float = 0.08,
        stale_halflife: float = 5.0,
        notes_ema_decay: float = 0.85,
        use_sparsemax: bool = True,
        temperature: float = 1.0,
        norm_kind: str = "layernorm",
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or context_dim < 1:
            raise ValueError("hidden_dim >= 8 y context_dim >= 1 son obligatorios")
        if n_modalities < 1 or n_notes < 1:
            raise ValueError("n_modalities >= 1 y n_notes >= 1 son obligatorios")
        if stale_halflife <= 0.0 or temperature <= 0.0:
            raise ValueError("stale_halflife y temperature deben ser estrictamente positivos")
        if not (0.0 < notes_ema_decay < 1.0):
            raise ValueError("notes_ema_decay debe pertenecer a (0, 1)")

        resolved_persist = n_notes if n_persistent_notes is None else int(n_persistent_notes)
        if not (1 <= resolved_persist <= n_notes):
            raise ValueError(
                "n_persistent_notes debe satisfacer 1 <= n_persistent_notes <= n_notes"
            )

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.n_modalities = n_modalities
        self.n_notes = n_notes
        self.n_persistent_notes = resolved_persist
        self.bottleneck_dim = (
            bottleneck_dim if bottleneck_dim is not None else max(8, hidden_dim // 4)
        )
        self.prune_threshold = float(prune_threshold)
        self.use_sparsemax = bool(use_sparsemax)
        self.temperature = float(temperature)
        self.norm_kind = str(norm_kind)

        def _mk_norm(dim: int) -> nn.Module:
            if self.norm_kind == "layernorm":
                return nn.LayerNorm(dim)
            return build_norm_layer(self.norm_kind, dim)  # type: ignore[arg-type]

        # Evaluador causal de relación señal/ruido (SNR) por modalidad en cuello de botella
        self.tok_norm = _mk_norm(hidden_dim)
        self.tok_scorer = nn.Sequential(
            nn.Linear(hidden_dim, self.bottleneck_dim),
            nn.SiLU(),
            nn.Linear(self.bottleneck_dim, 1),
        )
        self.ctx_norm = _mk_norm(context_dim)
        self.phys_norm = _mk_norm(hidden_dim)
        self.ctx_scorer = nn.Sequential(
            nn.Linear(context_dim + hidden_dim, self.bottleneck_dim),
            nn.SiLU(),
            nn.Linear(self.bottleneck_dim, n_modalities),
        )
        self.modality_prior = nn.Parameter(torch.zeros(1, n_modalities, dtype=torch.float32))

        # Tasas aprendibles de decaimiento exponencial por frescura (> 0 vía softplus)
        init_rate = math.log(2.0) / float(stale_halflife)
        self.age_decay_raw = nn.Parameter(
            torch.full((1, n_modalities), _inv_softplus(init_rate), dtype=torch.float32)
        )

        # Fusión compacta de canales seleccionados + física + contexto de alto SNR
        self.ctx_embed = nn.Linear(context_dim, hidden_dim)
        self.fusion_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fusion_norm = _mk_norm(hidden_dim)

        # Notas externas de trabajo comprimidas [M_notes, D] (persistentes + borrador intra-sesión)
        scale = 1.0 / math.sqrt(float(hidden_dim))
        self.note_queries = nn.Parameter(torch.randn(n_notes, hidden_dim) * scale)
        self.init_notes = nn.Parameter(torch.zeros(n_notes, hidden_dim, dtype=torch.float32))
        self.note_q = nn.Linear(hidden_dim, hidden_dim)
        self.note_k = nn.Linear(hidden_dim, hidden_dim)
        self.note_v = nn.Linear(hidden_dim, hidden_dim)
        self.note_norm = _mk_norm(hidden_dim)
        init_decay_logit = math.log(notes_ema_decay / (1.0 - notes_ema_decay))
        self.notes_decay_logit = nn.Parameter(torch.tensor(init_decay_logit, dtype=torch.float32))

        # Lectura de notas hacia cada activo (estrictamente equivariante por permutación)
        self.read_q = nn.Linear(hidden_dim, hidden_dim)
        self.read_k = nn.Linear(hidden_dim, hidden_dim)
        self.read_v = nn.Linear(hidden_dim, hidden_dim)
        self.read_out = nn.Linear(hidden_dim, hidden_dim)
        self.note_read_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        mod_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        physics_emb: torch.Tensor,
        high_snr_context: torch.Tensor,
        modality_age: torch.Tensor | None = None,
        prev_notes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selecciona/poda modalidades causales y actualiza notas externas `[M_notes, D]`."""
        device, dtype = mod_tokens.device, mod_tokens.dtype
        mask_bool = modality_mask.to(device=device) > 0.5
        mask = mask_bool.to(dtype=dtype)
        # Aislar modalidades ausentes con torch.where (seguro ante NaN/Inf * 0!)
        present_tokens = torch.where(
            mask_bool.unsqueeze(-1), mod_tokens, torch.zeros_like(mod_tokens)
        )

        phys = physics_emb.to(device=device, dtype=dtype)
        ctx = high_snr_context.to(device=device, dtype=dtype)
        ctx_n = self.ctx_norm(ctx)
        phys_n = self.phys_norm(phys)

        if modality_age is None:
            age = torch.zeros_like(mask)
        else:
            raw_age = modality_age.to(device=device, dtype=dtype)
            safe_age = torch.where(
                mask_bool & torch.isfinite(raw_age), raw_age, torch.zeros_like(mask)
            )
            age = torch.clamp(safe_age, min=0.0)

        # 1) Puntuación SNR causal + decaimiento exponencial de frescura
        decay_rate = F.softplus(self.age_decay_raw.to(device=device, dtype=dtype)) + 1e-3
        freshness = torch.exp(-decay_rate * age)  # (0, 1]
        tok_score = self.tok_scorer(self.tok_norm(present_tokens)).squeeze(-1)  # [N, M]
        ctx_score = self.ctx_scorer(torch.cat([ctx_n, phys_n], dim=-1))  # [N, M]
        snr_logits = (
            tok_score
            + ctx_score
            + self.modality_prior.to(device=device, dtype=dtype)
            - 2.0 * decay_rate * age
        )

        # 2) Compuerta competitiva (sparsemax o softmax) + máscara + frescura (en calc_dtype)
        calc_dtype = _stable_calc_dtype(dtype)
        has_any = (mask.sum(dim=-1, keepdim=True) > 0.5).to(dtype=calc_dtype)
        snr_calc = snr_logits.to(dtype=calc_dtype)
        mask_calc = mask.to(dtype=calc_dtype)
        masked_logits = torch.where(
            mask_calc > 0.5, snr_calc, torch.full_like(snr_calc, -1e4)
        )
        scaled_logits = masked_logits / self.temperature
        comp_w = (
            _sparsemax(scaled_logits, dim=-1)
            if self.use_sparsemax
            else torch.softmax(scaled_logits, dim=-1)
        )
        salience_weights = (
            comp_w * mask_calc * freshness.to(dtype=calc_dtype) * has_any
        ).to(dtype=dtype)
        pruned_ratio = (salience_weights < self.prune_threshold).to(dtype=dtype).mean()

        # 3) Fusión compacta por activo
        mod_pooled = torch.einsum("nm,nmd->nd", salience_weights, present_tokens)
        ctx_emb = self.ctx_embed(ctx_n)
        base_x = self.fusion_norm(self.fusion_proj(torch.cat([mod_pooled, phys, ctx_emb], dim=-1)))

        # 4) Compresión equivariante en notas externas [M_notes, D]:
        #    slots 0..n_persistent_notes-1 persisten t-1->t; los restantes se resetean por sesión
        init_n = self.init_notes.to(device=device, dtype=dtype)
        if prev_notes is not None:
            prev_t = prev_notes.to(device=device, dtype=dtype)
            n_p = min(self.n_persistent_notes, self.n_notes, prev_t.shape[0])
            if n_p >= self.n_notes:
                prior_notes = prev_t[: self.n_notes]
            else:
                prior_notes = torch.cat([prev_t[:n_p], init_n[n_p:]], dim=0)
        else:
            prior_notes = init_n

        inv_sqrt_d = 1.0 / math.sqrt(float(self.hidden_dim))
        q_notes = self.note_q(self.note_queries.to(device=device, dtype=dtype) + prior_notes)
        k_assets = self.note_k(base_x)
        v_assets = self.note_v(base_x)
        note_attn = torch.softmax((q_notes @ k_assets.T) * inv_sqrt_d, dim=-1)  # [M_notes, N]
        fresh_notes = self.note_norm(note_attn @ v_assets)
        decay = torch.sigmoid(self.notes_decay_logit).to(device=device, dtype=dtype)
        context_notes = decay * prior_notes + (1.0 - decay) * fresh_notes

        # 5) Lectura residual desde notas hacia cada activo
        q_read = self.read_q(base_x)
        k_read = self.read_k(context_notes)
        v_read = self.read_v(context_notes)
        read_attn = torch.softmax((q_read @ k_read.T) * inv_sqrt_d, dim=-1)  # [N, M_notes]
        note_ctx = self.read_out(read_attn @ v_read)
        gate = torch.sigmoid(self.note_read_gate).to(device=device, dtype=dtype)
        compact_x = base_x + gate * note_ctx
        return compact_x, salience_weights, pruned_ratio, context_notes


class DifferentiableQuantToolbox(nn.Module):
    """Caja de herramientas cuantitativas en GPU (`v1_6tools` o `v2_10tools`, 0 syncs)."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        core_dim: int | None = None,
        toolbox_version: Literal["v1_6tools", "v2_10tools"] = "v1_6tools",
        rmt_dim: int = 5,
        n_hyperedges: int = 16,
        n_modalities: int = 5,
        n_notes: int = 8,
        n_persistent_notes: int = 4,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        portfolio_steps: int = 6,
        portfolio_risk_aversion: float = 2.0,
        portfolio_factor_penalty: float = 8.0,
        portfolio_l1_cost: float = 0.001,
        portfolio_holding_days: float = 10.0,
        portfolio_gross_limit: float = 1.0,
        portfolio_pos_cap: float = 0.1,
        epistemic_kappa_init: float = 1.0,
        cf_placebo_mode: bool = False,
        cf_softmin_beta: float = 6.0,
        cf_flip_tau: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim >= 8 es obligatorio")
        if toolbox_version not in TOOLBOX_VERSIONS:
            raise ValueError(
                f"toolbox_version inválido {toolbox_version!r}; válidos: {TOOLBOX_VERSIONS}"
            )
        if portfolio_holding_days <= 0.0:
            raise ValueError("portfolio_holding_days debe ser estrictamente positivo")

        self.hidden_dim = hidden_dim
        self.core_dim = core_dim if core_dim is not None else max(8, hidden_dim // 2)
        self.toolbox_version = toolbox_version
        self.is_v2 = toolbox_version == "v2_10tools"
        self.tool_names = TOOL_NAMES_V2 if self.is_v2 else TOOL_NAMES
        self.n_tools = len(self.tool_names)
        self.rmt_dim = rmt_dim
        self.n_hyperedges = n_hyperedges
        self.n_modalities = int(n_modalities)
        self.n_notes = int(n_notes)
        self.n_persistent_notes = min(int(n_persistent_notes), int(n_notes))
        self.horizons = horizons
        self.n_h = len(horizons)
        self.portfolio_holding_days = float(portfolio_holding_days)
        self.cf_placebo_mode = bool(cf_placebo_mode)

        scalar_mid = max(8, self.core_dim // 2)

        # Escalas sqrt(h) y bases deterministas registradas en dispositivo (cero syncs)
        self.register_buffer(
            "horizon_scales",
            torch.tensor([math.sqrt(float(h)) for h in horizons], dtype=torch.float32),
        )
        dim_grid = torch.arange(hidden_dim, dtype=torch.float32)
        self.register_buffer(
            "sector_basis_freq",
            0.37 + 0.19 * (dim_grid / max(float(hidden_dim), 1.0)) * math.pi,
        )
        self.register_buffer(
            "sector_basis_phase",
            0.5 * dim_grid * (math.pi / max(float(hidden_dim), 1.0)),
        )
        self.register_buffer(
            "liq_tercile_centers",
            torch.tensor([1.0 / 6.0, 0.5, 5.0 / 6.0], dtype=torch.float32),
        )
        self.register_buffer(
            "t7_av_idx",
            torch.tensor(
                [m for m in range(self.n_modalities) if not (self.n_modalities >= 3 and m == 2)],
                dtype=torch.long,
            ),
        )

        # Sonda ligera de hipótesis interna usada cuando el llamador no aporta `pred_k` externo
        self.probe_norm = nn.LayerNorm(hidden_dim)
        self.probe_head = nn.Linear(hidden_dim, self.n_h * 3)
        nn.init.normal_(self.probe_head.weight, std=0.01)
        nn.init.zeros_(self.probe_head.bias)

        # Tool 2: Hipergrafo transversal RMT O(N*K) (reutiliza clase oficial)
        self.hypergraph = RMTSpilloverHypergraph(
            hidden_dim, n_hyperedges=n_hyperedges, rmt_dim=rmt_dim, n_rounds=1
        )
        with torch.no_grad():
            self.hypergraph.out_scale.fill_(0.25)

        # Tool 4: Cartera convexa P+ (ruta rápida Triton FISTA 1.9ms cuando aux_req_grad=False)
        self.portfolio_layer = DifferentiablePortfolioLayer(
            steps=portfolio_steps,
            risk_aversion=portfolio_risk_aversion,
            factor_penalty=portfolio_factor_penalty,
            l1_cost=portfolio_l1_cost,
            gross_limit=portfolio_gross_limit,
            pos_cap=portfolio_pos_cap,
            holding_days=portfolio_holding_days,
        )

        # Tool 5: aversión epistémica positiva kappa > 0 (+ escalas positivas en v2)
        self.epistemic_kappa_raw = nn.Parameter(
            torch.tensor(_inv_softplus(epistemic_kappa_init), dtype=torch.float32)
        )
        if self.is_v2:
            self.epistemic_scale_raw: nn.Parameter | None = nn.Parameter(
                torch.tensor(
                    [
                        _inv_softplus(0.25),
                        _inv_softplus(0.002),
                        _inv_softplus(0.002),
                        _inv_softplus(0.01),
                        _inv_softplus(0.01),
                    ],
                    dtype=torch.float32,
                )
            )
        else:
            self.epistemic_scale_raw = None

        # Proyecciones exactas de deltas (cero bypass crudo de h + MLPs 2L en escalares)
        self.tool0_proj = nn.Linear(hidden_dim + (2 if self.is_v2 else 1), hidden_dim)
        self.tool1_proj = nn.Linear(hidden_dim + 1, hidden_dim)
        self.tool2_proj = nn.Linear(hidden_dim + (2 if self.is_v2 else 1), hidden_dim)
        self.tool3_proj = _make_scalar_tool_mlp(7, scalar_mid, hidden_dim)
        self.tool4_proj = _make_scalar_tool_mlp(7, scalar_mid, hidden_dim)
        self.tool5_proj = _make_scalar_tool_mlp(8 if self.is_v2 else 6, scalar_mid, hidden_dim)

        # Herramientas ortogonales v2 (T6, T7, T8, T9)
        if self.is_v2:
            # T6: cross_sectional_rank_calibrator (10 escalares -> 2L MLP -> D)
            self.tool6_proj: nn.Module | None = _make_scalar_tool_mlp(10, scalar_mid, hidden_dim)

            # T7: cross_modal_conflict_arbiter (sin macro en conflicto; release age[:,2]<=1)
            t7_dir_dim = min(32, hidden_dim)
            self.t7_age_rate_raw: nn.Parameter | None = nn.Parameter(
                torch.full((1, self.n_modalities), _inv_softplus(0.15), dtype=torch.float32)
            )
            self.t7_mod_prec: nn.Parameter | None = nn.Parameter(
                torch.zeros(1, self.n_modalities, dtype=torch.float32)
            )
            self.t7_dir_proj: nn.Linear | None = nn.Linear(hidden_dim, t7_dir_dim)
            self.t7_vote_head: nn.Linear | None = nn.Linear(t7_dir_dim, 1)
            self.tool7_proj: nn.Module | None = nn.Linear(hidden_dim + 8, hidden_dim)

            # T8: regime_stress_counterfactual (delega en CounterfactualStressSimulator compartido)
            self.cf_sim: CounterfactualStressSimulator | None = CounterfactualStressSimulator(
                pos_cap=portfolio_pos_cap,
                softmin_beta=cf_softmin_beta,
                flip_tau=cf_flip_tau,
                placebo_mode=cf_placebo_mode,
            )
            self.tool8_proj: nn.Module | None = _make_scalar_tool_mlp(
                CF_TRACE_DIM, scalar_mid, hidden_dim
            )

            # T9: working_memory_scratchpad_ops (sobre cognitive_notes [8, D], sin Gram-Schmidt)
            t9_attn_dim = max(16, self.core_dim // 2)
            self.t9_attn_dim = t9_attn_dim
            self.t9_h_norm: nn.LayerNorm | None = nn.LayerNorm(hidden_dim)
            self.t9_note_norm: nn.LayerNorm | None = nn.LayerNorm(hidden_dim)
            self.t9_slot_queries: nn.Parameter | None = nn.Parameter(
                torch.randn(self.n_notes, hidden_dim) / math.sqrt(float(hidden_dim))
            )
            self.t9_write_q: nn.Linear | None = nn.Linear(hidden_dim, t9_attn_dim)
            self.t9_write_k: nn.Linear | None = nn.Linear(hidden_dim, t9_attn_dim)
            self.t9_write_v: nn.Linear | None = nn.Linear(hidden_dim, hidden_dim)
            self.t9_write_norm: nn.LayerNorm | None = nn.LayerNorm(hidden_dim)
            self.t9_nov_gamma: nn.Parameter | None = nn.Parameter(
                torch.tensor(_inv_softplus(2.0), dtype=torch.float32)
            )
            self.t9_nov_bias: nn.Parameter | None = nn.Parameter(
                torch.tensor(-0.5, dtype=torch.float32)
            )
            self.t9_out_norm: nn.LayerNorm | None = nn.LayerNorm(hidden_dim)
            self.t9_read_q: nn.Linear | None = nn.Linear(hidden_dim, t9_attn_dim)
            self.t9_read_k: nn.Linear | None = nn.Linear(hidden_dim, t9_attn_dim)
            self.t9_read_v: nn.Linear | None = nn.Linear(hidden_dim, hidden_dim)
            self.tool9_proj: nn.Module | None = nn.Linear(hidden_dim + 2, hidden_dim)
        else:
            self.tool6_proj = None
            self.t7_age_rate_raw = None
            self.t7_mod_prec = None
            self.t7_dir_proj = None
            self.t7_vote_head = None
            self.tool7_proj = None
            self.cf_sim = None
            self.tool8_proj = None
            self.t9_attn_dim = 0
            self.t9_h_norm = None
            self.t9_note_norm = None
            self.t9_slot_queries = None
            self.t9_write_q = None
            self.t9_write_k = None
            self.t9_write_v = None
            self.t9_write_norm = None
            self.t9_nov_gamma = None
            self.t9_nov_bias = None
            self.t9_out_norm = None
            self.t9_read_q = None
            self.t9_read_k = None
            self.t9_read_v = None
            self.tool9_proj = None

        # LayerNorm unit-RMS individual por herramienta (iguala escala entre herramientas)
        self.tool_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_tools)])

    def estimate_hypothesis(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Construye una hipótesis diferenciable compacta `(ret, alpha, vol, epist)` desde `h`."""
        n_assets = h.shape[0]
        raw = self.probe_head(self.probe_norm(h)).view(n_assets, self.n_h, 3)
        ret = raw[..., 0]
        alpha = ret - ret.mean(dim=0, keepdim=True)
        log_vol = torch.clamp(raw[..., 1] - 3.9, min=-6.0, max=0.5)
        vol = torch.exp(log_vol)
        epist = F.softplus(raw[..., 2]) * 0.02
        return dict(
            pred_return=ret,
            alpha=alpha,
            vol_pred=vol,
            epistemic_error=epist,
        )

    def compute_probe_aux_loss(
        self,
        hypothesis: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Pérdida auxiliar supervisada para `probe_head` (evita hipótesis sin supervisar)."""
        pred_ret = hypothesis["pred_return"]
        epist = hypothesis["epistemic_error"]
        device, dtype = pred_ret.device, pred_ret.dtype
        calc_dtype = _stable_calc_dtype(dtype)

        y = targets.to(device=device, dtype=calc_dtype)
        if y.ndim == 1:
            y = y.unsqueeze(-1).expand(-1, pred_ret.shape[-1])
        n_h = min(y.shape[-1], pred_ret.shape[-1])
        p_h = pred_ret[:, :n_h].to(dtype=calc_dtype)
        y_h = y[:, :n_h]
        fin = torch.isfinite(y_h)
        fin_f = fin.to(dtype=calc_dtype)
        cnt = fin_f.sum().clamp_min(1.0)
        y_safe = torch.where(fin, y_h, torch.zeros_like(y_h))
        p_safe = torch.where(fin, p_h, torch.zeros_like(p_h))

        huber_el = F.huber_loss(p_safe, y_safe, reduction="none", delta=0.03) * fin_f
        loss_ret = huber_el.sum() / cnt

        # Calibración de incertidumbre epistémica frente al error absoluto real
        e_h = epist[:, :n_h].to(dtype=calc_dtype)
        abs_err = (y_safe - p_safe.detach()).abs()
        loss_epist = (
            F.huber_loss(e_h, abs_err, reduction="none", delta=0.02) * fin_f
        ).sum() / cnt

        # Término IC transversal en el horizonte principal
        p0 = p_safe[:, 0]
        y0 = y_safe[:, 0]
        f0 = fin[:, 0]
        f0_cnt = f0.to(dtype=calc_dtype).sum().clamp_min(1.0)
        p0_c = torch.where(f0, p0 - p0.sum() / f0_cnt, torch.zeros_like(p0))
        y0_c = torch.where(f0, y0 - y0.sum() / f0_cnt, torch.zeros_like(y0))
        corr = (p0_c * y0_c).sum() / (
            p0_c.norm().clamp_min(1e-6) * y0_c.norm().clamp_min(1e-6)
        )
        loss_ic = torch.where(f0_cnt > 1.5, 1.0 - corr, torch.zeros_like(corr))
        return (5.0 * loss_ret + 0.5 * loss_ic + 2.0 * loss_epist).to(dtype=dtype)

    def prepare_factor_cache(
        self,
        *,
        n_assets: int,
        device: torch.device,
        dtype: torch.dtype,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        ridge: float = 1e-3,
    ) -> dict[str, torch.Tensor | None]:
        """Precalcula normalizaciones, Gram y operadores ridge 1 sola vez por sesión (hoisted)."""
        calc_dtype = _stable_calc_dtype(dtype)
        cache: dict[str, torch.Tensor | None] = dict(
            decomp_b_n=None,
            decomp_proj=None,
            exp_z=None,
            port_b_mat=None,
            port_s_max2=None,
            cf_b_n=None,
            cf_b_proj=None,
            cf_r_n=None,
            cf_r_proj=None,
            cf_r_row=None,
        )
        if n_assets <= 1:
            return cache

        has_exp = exposures is not None and exposures.shape[-1] > 0
        has_rmt = rmt_loadings is not None and rmt_loadings.shape[-1] > 0

        exp_c_s: torch.Tensor | None = None
        exp_z: torch.Tensor | None = None
        if has_exp:
            assert exposures is not None
            exp_s = torch.nan_to_num(
                exposures.detach().to(device=device, dtype=calc_dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            exp_c_s = exp_s - exp_s.mean(dim=0, keepdim=True)
            exp_std = exp_c_s.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-4)
            exp_z = exp_c_s / exp_std
            cache["exp_z"] = exp_z

            if self.portfolio_layer.factor_penalty > 0.0:
                b_norm = exp_c_s.norm(dim=0, keepdim=True).clamp_min(1e-4)
                b_mat = exp_c_s / b_norm
                gram_p = b_mat.detach().T @ b_mat.detach()
                s_max2 = _spectral_norm_sq_psd(gram_p, n_iters=5).clamp_min(1.0)
                cache["port_b_mat"] = b_mat
                cache["port_s_max2"] = s_max2

        decomp_cols: list[torch.Tensor] = []
        if exp_z is not None:
            decomp_cols.append(exp_z)
        if has_rmt:
            assert rmt_loadings is not None
            rmt_s = torch.nan_to_num(
                rmt_loadings.detach().to(device=device, dtype=calc_dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            rmt_c = rmt_s - rmt_s.mean(dim=0, keepdim=True)
            rmt_z = rmt_c / rmt_c.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-4)
            decomp_cols.append(rmt_z)

        if decomp_cols:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                b_n = torch.cat(decomp_cols, dim=-1)
                n_f = b_n.shape[-1]
                gram_d = b_n.T @ b_n + ridge * torch.eye(n_f, device=device, dtype=calc_dtype)
                proj_op, _ = torch.linalg.solve_ex(gram_d, b_n.T, check_errors=False)  # [F, N]
                cache["decomp_b_n"] = b_n
                cache["decomp_proj"] = proj_op

        # En v2 izamos también los operadores del simulador contrafactual (cero solve_ex en bucle K)
        if self.is_v2 and (has_exp or has_rmt):
            cf_cache = prepare_cf_factor_cache(
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                ridge=ridge,
                device=device,
                dtype=calc_dtype,
            )
            cache.update(cf_cache)

        return cache

    def derive_sector_key(
        self,
        *,
        factor_cache: dict[str, torch.Tensor | None] | None = None,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Proyección determinista equivariante de `[exposures, rmt_loadings]` en `R^D` para T1."""
        calc_dtype = _stable_calc_dtype(dtype)
        b_n: torch.Tensor | None = (
            factor_cache.get("decomp_b_n") if factor_cache is not None else None
        )
        if b_n is None:
            cols: list[torch.Tensor] = []
            if exposures is not None and exposures.ndim == 2 and exposures.shape[-1] > 0:
                cols.append(
                    torch.nan_to_num(
                        exposures.detach().to(device=device, dtype=calc_dtype),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                )
            if rmt_loadings is not None and rmt_loadings.ndim == 2 and rmt_loadings.shape[-1] > 0:
                cols.append(
                    torch.nan_to_num(
                        rmt_loadings.detach().to(device=device, dtype=calc_dtype),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                )
            if not cols:
                return None
            b_n = torch.cat(cols, dim=-1)
        else:
            b_n = b_n.detach().to(device=device, dtype=calc_dtype)

        f_tot = b_n.shape[-1]
        col_idx = torch.arange(1, f_tot + 1, device=device, dtype=calc_dtype).unsqueeze(-1)
        freq = self.sector_basis_freq.to(device=device, dtype=calc_dtype).unsqueeze(0)  # [1, D]
        phase = self.sector_basis_phase.to(device=device, dtype=calc_dtype).unsqueeze(0)  # [1, D]
        basis = torch.cos(col_idx * freq + phase) / math.sqrt(float(max(f_tot, 1)))  # [F, D]
        return (b_n @ basis).to(dtype=dtype)

    @staticmethod
    def _clip_mad_residuals(h_c_s: torch.Tensor) -> torch.Tensor:
        """Recorta residuos extremos `h_c` a `±6*MAD` pre-OLS (seguro ante límites CN/halts)."""
        if h_c_s.shape[0] < 4:
            return h_c_s
        med = h_c_s.detach().median(dim=0, keepdim=True).values
        mad = (h_c_s.detach() - med).abs().median(dim=0, keepdim=True).values * 1.4826
        std_fb = h_c_s.detach().std(dim=0, unbiased=False, keepdim=True)
        scale = torch.where(mad > 1e-5, mad, std_fb).clamp_min(1e-4)
        lim = 6.0 * scale
        return med + torch.clamp(h_c_s - med, min=-lim, max=lim)

    @staticmethod
    def decompose_factors(
        h: torch.Tensor,
        *,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        ridge: float = 1e-3,
        factor_cache: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool 0: proyección ortogonal exacta en GPU con recorte robusto `±6*MAD` pre-OLS."""
        n_assets = h.shape[0]
        device, dtype = h.device, h.dtype
        solve_dtype = _stable_calc_dtype(dtype)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            h_s = h.to(dtype=solve_dtype)
            h_fit = DifferentiableQuantToolbox._clip_mad_residuals(h_s)
            h_center_s = h_fit.mean(dim=0, keepdim=True)
            h_c_fit = h_fit - h_center_s
            h_center = h_center_s.to(dtype=dtype)

            if factor_cache is not None:
                b_n_cached = factor_cache.get("decomp_b_n")
                proj_cached = factor_cache.get("decomp_proj")
                if b_n_cached is None or proj_cached is None or n_assets <= 1:
                    return h_center.expand_as(h), h - h_center
                sys_proj = (b_n_cached @ (proj_cached @ h_c_fit)).to(dtype=dtype)
                h_sys = h_center + sys_proj
                return h_sys, h - h_sys

            factor_cols: list[torch.Tensor] = []
            if exposures is not None and exposures.shape[-1] > 0:
                factor_cols.append(exposures.to(device=device, dtype=solve_dtype))
            if rmt_loadings is not None and rmt_loadings.shape[-1] > 0:
                factor_cols.append(rmt_loadings.to(device=device, dtype=solve_dtype))

            if not factor_cols or n_assets <= 1:
                return h_center.expand_as(h), h - h_center

            b_raw = torch.cat(factor_cols, dim=-1)
            b_s = torch.nan_to_num(b_raw, nan=0.0, posinf=0.0, neginf=0.0)
            b_c = b_s - b_s.mean(dim=0, keepdim=True)
            b_std = b_c.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-4)
            b_n = b_c / b_std
            n_f = b_n.shape[-1]
            gram = b_n.T @ b_n + ridge * torch.eye(n_f, device=device, dtype=solve_dtype)
            beta, _ = torch.linalg.solve_ex(gram, b_n.T @ h_c_fit, check_errors=False)
            sys_proj = (b_n @ beta).to(dtype=dtype)
        h_sys = h_center + sys_proj
        h_idio = h - h_sys
        return h_sys, h_idio

    @staticmethod
    def lookup_hrr_memory(
        h: torch.Tensor,
        hrr_memory: HRRModernHopfieldMemory | None = None,
        *,
        regime_key: torch.Tensor | None = None,
        sector_key: torch.Tensor | None = None,
        fallback_ood: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool 1: delega en `hrr_memory.retrieve` enlazando `regime_key` y `sector_key`."""
        n_assets = h.shape[0]
        device, dtype = h.device, h.dtype
        if hrr_memory is None:
            zero_ood = (
                torch.zeros(n_assets, device=device, dtype=dtype)
                if fallback_ood is None
                else fallback_ood.to(device=device, dtype=dtype)
            )
            return torch.zeros_like(h), zero_ood

        param_dtype = hrr_memory.out_proj.weight.dtype
        h_in = h if h.dtype == param_dtype else h.to(dtype=param_dtype)
        res = hrr_memory.retrieve(h_in, regime_key=regime_key, sector_key=sector_key)
        retrieved = res["retrieved"].to(device=device, dtype=dtype)
        ood_out = res["ood_energy"].to(device=device, dtype=dtype)
        if fallback_ood is not None:
            ood_out = torch.maximum(ood_out, fallback_ood.to(device=device, dtype=dtype))
        return retrieved, ood_out

    def run_hypergraph_spillover(
        self,
        h: torch.Tensor,
        hypergraph: RMTSpilloverHypergraph | None = None,
        *,
        rmt_loadings: torch.Tensor | None = None,
        return_peer_disp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool 2: propagación transversal `O(N*K)` + radar de dispersión de pares `peer_disp`."""
        hg = hypergraph if hypergraph is not None else self.hypergraph
        out = h
        calc_dtype = _stable_calc_dtype(h.dtype)
        scale = 1.0 / math.sqrt(float(hg.hidden_dim))
        logits = (hg.node_norm(out) @ hg.edge_queries.T) * scale
        if rmt_loadings is not None and rmt_loadings.shape[-1] == hg.rmt_dim:
            rmt_in = rmt_loadings.to(device=h.device, dtype=h.dtype)
        else:
            rmt_in = torch.zeros((h.shape[0], hg.rmt_dim), device=h.device, dtype=h.dtype)
        logits = logits + hg.rmt_proj(rmt_in)
        inc = torch.softmax(logits, dim=-1)  # [N, K]
        edge_deg = inc.sum(dim=0, keepdim=True).clamp_min(1e-6)
        inc_norm = inc / edge_deg

        for _ in range(hg.n_rounds):
            v = F.silu(hg.msg_in(hg.node_norm(out)))
            edge_state = inc_norm.T @ v
            node_msg = hg.msg_out(inc @ edge_state)
            out = out + hg.out_scale.to(dtype=out.dtype) * node_msg

        inc_c = inc.to(dtype=calc_dtype)
        ent_per_asset = -(inc_c * torch.log(inc_c.clamp_min(1e-8))).sum(dim=-1).to(dtype=h.dtype)
        if not return_peer_disp:
            return out, ent_per_asset

        inc_norm_c = inc_norm.to(dtype=calc_dtype)
        out_c = out.to(dtype=calc_dtype)
        edge_mean = inc_norm_c.T @ out_c
        edge_m2 = inc_norm_c.T @ out_c.pow(2)
        edge_var = (edge_m2 - edge_mean.pow(2)).clamp_min(0.0).mean(dim=-1)  # [K]
        peer_disp = (inc_c @ edge_var).clamp_min(0.0).to(dtype=h.dtype)  # [N]
        return out, ent_per_asset, peer_disp

    def verify_structure(
        self,
        pred_k: dict[str, torch.Tensor],
        *,
        exposures: torch.Tensor | None = None,
        ood_energy: torch.Tensor | None = None,
        factor_cache: dict[str, torch.Tensor | None] | None = None,
        horizon_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool 3: invariantes financieros estructurales respetando `horizon_mask` (P0-4)."""
        ret = pred_k["pred_return"]
        alpha = pred_k["alpha"]
        vol = pred_k["vol_pred"]
        n_assets, n_h = ret.shape
        device, dtype = ret.device, ret.dtype

        if horizon_mask is not None:
            h_mask = (horizon_mask.to(device=device).reshape(-1)[:n_h] > 0.5).to(dtype=dtype)
        else:
            h_mask = torch.ones(n_h, device=device, dtype=dtype)
        h_sum = h_mask.sum().clamp_min(1.0)

        # 1) Coherencia multi-horizonte en escala sqrt(h) (cero penalización en inactivos!)
        if n_h >= 2:
            scales = self.horizon_scales[:n_h].to(device=device, dtype=dtype)
            norm_ret = ret / scales.unsqueeze(0)
            pair_mask = h_mask[:-1] * h_mask[1:]
            pair_sum = pair_mask.sum()
            has_pairs = (pair_sum > 0.5).to(dtype=dtype)
            diffs_sq = (norm_ret[:, 1:] - norm_ret[:, :-1]) ** 2
            e_hor = (
                (diffs_sq * pair_mask.unsqueeze(0)).sum(dim=-1) / pair_sum.clamp_min(1.0)
            ) * has_pairs
        else:
            e_hor = torch.zeros(n_assets, device=device, dtype=dtype)

        # 2) Consistencia CDF/volatilidad respetando `horizon_mask` + asimetría de colas
        if "quantiles" in pred_k and pred_k["quantiles"] is not None:
            q = pred_k["quantiles"]
            cross_h = F.relu(q[..., :-1] - q[..., 1:]).pow(2).mean(dim=-1)  # [N, H]
            cross_viol = (cross_h * h_mask.unsqueeze(0)).sum(dim=-1) / h_sum
            if q.shape[-1] >= 9:
                iqr_sigma = (q[..., 6] - q[..., 2]) / 2.5631
                q_lo, q_med, q_hi = q[:, 0, 2], q[:, 0, 4], q[:, 0, 6]
            else:
                iqr_sigma = (q[..., -1] - q[..., 0]) / 2.5631
                q_lo, q_med, q_hi = q[:, 0, 0], q[:, 0, q.shape[-1] // 2], q[:, 0, -1]
            vol_match = (((iqr_sigma - vol) ** 2) * h_mask.unsqueeze(0)).sum(dim=-1) / h_sum
            e_cdf_vol = cross_viol + vol_match
            tail_skew = torch.tanh(
                ((q_hi - q_med) - (q_med - q_lo)) / (iqr_sigma[:, 0].abs().clamp_min(1e-4) + 1e-4)
            )
        else:
            ratio_rv = ret.abs() / vol.clamp_min(1e-4)
            e_cdf_vol = (((ratio_rv - 1.0).pow(2)) * h_mask.unsqueeze(0)).sum(dim=-1) / h_sum * 0.01
            tail_skew = torch.tanh((ret[:, 0] - alpha[:, 0]) / vol[:, 0].clamp_min(1e-4))

        # 3) Desviación transversal media-cero + penalización de ortogonalidad factorial
        a0 = alpha[:, 0]
        cs_mean = a0.mean()
        cs_dev = a0 - cs_mean
        mean_pen = cs_mean.pow(2).expand(n_assets)
        exp_z_cached = factor_cache.get("exp_z") if factor_cache is not None else None
        if exp_z_cached is not None and n_assets > 1:
            b_n = exp_z_cached.to(dtype=dtype)
            fac_proj = (b_n.T @ a0) / float(max(n_assets, 1))
            fac_pen = (fac_proj**2).mean().expand(n_assets)
        elif (
            factor_cache is None
            and exposures is not None
            and exposures.shape[-1] > 0
            and n_assets > 1
        ):
            b = exposures.to(device=device, dtype=dtype)
            b_c = b - b.mean(dim=0, keepdim=True)
            b_n = b_c / b_c.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-4)
            fac_proj = (b_n.T @ a0) / float(max(n_assets, 1))
            fac_pen = (fac_proj**2).mean().expand(n_assets)
        else:
            fac_pen = torch.zeros(n_assets, device=device, dtype=dtype)
        e_cs = mean_pen + fac_pen

        # 4) Anclaje epistémico OOD
        if ood_energy is not None:
            ood = ood_energy.to(device=device, dtype=dtype).clamp_min(0.0)
            e_ood = ood * (a0**2)
        else:
            e_ood = torch.zeros(n_assets, device=device, dtype=dtype)

        e_total = e_hor + e_cdf_vol + e_cs + e_ood
        feats = torch.stack(
            [
                torch.log1p(e_hor.clamp_min(0.0)),
                torch.log1p(e_cdf_vol.clamp_min(0.0)),
                torch.tanh(cs_dev * 20.0),
                torch.log1p(e_cs.clamp_min(0.0)),
                torch.log1p(e_ood.clamp_min(0.0)),
                torch.log1p(e_total.clamp_min(0.0)),
                tail_skew,
            ],
            dim=-1,
        )
        return feats, e_total

    def run_portfolio_feasibility(
        self,
        mu: torch.Tensor,
        *,
        vol: torch.Tensor | None = None,
        exposures: torch.Tensor | None = None,
        w_prev: torch.Tensor | None = None,
        holding_days: float | None = None,
        liq_pct: torch.Tensor | None = None,
        factor_cache: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Tool 4: FISTA `P+` 1.9ms Triton fast path (`aux_req_grad==False` garantizado, P0-5)."""
        layer = self.portfolio_layer
        n_assets = mu.shape[0]
        device, dtype = mu.device, mu.dtype
        if n_assets <= 1:
            return torch.zeros_like(mu)

        calc_dtype = _stable_calc_dtype(dtype)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            # P0-5: desconectar vol, liq_pct, w_prev y exposures -> garantiza ruta Triton FISTA
            vol_base = (
                torch.full((n_assets,), 0.02, device=device, dtype=calc_dtype)
                if vol is None
                else torch.nan_to_num(
                    vol.detach().to(device=device, dtype=calc_dtype),
                    nan=0.02,
                    posinf=1.0,
                    neginf=1e-4,
                ).clamp_min(1e-4)
            )
            if liq_pct is not None:
                liq_d = torch.nan_to_num(
                    liq_pct.detach().to(device=device, dtype=calc_dtype),
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                vol_eff = vol_base * (1.0 + 1.5 * (1.0 - liq_d).pow(2))
            else:
                vol_eff = vol_base

            w_prev_d = (
                None
                if w_prev is None
                else torch.nan_to_num(
                    w_prev.detach().to(device=device, dtype=calc_dtype),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            exp_d = (
                None
                if exposures is None
                else exposures.detach().to(device=device, dtype=calc_dtype)
            )

            eff_holding = (
                float(layer.holding_days)
                if holding_days is None
                else max(float(holding_days), 1.0)
            )
            if (
                factor_cache is not None
                and factor_cache.get("port_b_mat") is not None
                and factor_cache.get("port_s_max2") is not None
            ) or abs(eff_holding - float(layer.holding_days)) > 1e-9:
                # Reutilizar `port_b_mat`/`port_s_max2` izados o `holding_days` vía custom_op
                mu_c = mu.to(dtype=calc_dtype)
                var_diag = vol_eff.clamp_min(1e-4).pow(2)
                w0 = (
                    torch.zeros(n_assets, device=device, dtype=calc_dtype)
                    if w_prev_d is None
                    else w_prev_d
                )
                lip = layer.risk_aversion * var_diag.max().detach().clamp_min(1e-8)
                b_mat: torch.Tensor | None = None
                if factor_cache is not None:
                    b_c = factor_cache.get("port_b_mat")
                    s_c = factor_cache.get("port_s_max2")
                    if b_c is not None and s_c is not None and layer.factor_penalty > 0.0:
                        b_mat = b_c.detach().to(device=device, dtype=calc_dtype)
                        lip = lip + layer.factor_penalty * s_c.detach().to(
                            device=device, dtype=calc_dtype
                        )
                elif exp_d is not None and layer.factor_penalty > 0.0 and exp_d.shape[-1] > 0:
                    exp_clean = torch.nan_to_num(exp_d, nan=0.0, posinf=0.0, neginf=0.0)
                    b_cent = exp_clean - exp_clean.mean(dim=0, keepdim=True)
                    col_n = torch.sqrt((b_cent * b_cent).sum(dim=0, keepdim=True) + 1e-12)
                    b_mat = torch.where(col_n > 1e-6, b_cent / col_n, torch.zeros_like(b_cent))
                    gram = b_mat.T @ b_mat
                    s_max2 = _spectral_norm_sq_psd(gram, n_iters=5).clamp_min(1.0)
                    lip = lip + layer.factor_penalty * s_max2

                step_t = 1.0 / lip.clamp_min(1e-4)
                eff_l1 = float(layer.l1_cost) / eff_holding
                b_arg = (
                    b_mat
                    if b_mat is not None
                    else torch.empty((n_assets, 0), device=device, dtype=calc_dtype)
                )
                w_out, _, _, _, _ = _ma._diff_portfolio_fista_fwd_op(
                    mu_c,
                    var_diag,
                    w0,
                    b_arg,
                    step_t,
                    int(layer.steps),
                    float(layer.risk_aversion),
                    float(layer.factor_penalty),
                    float(eff_l1),
                    float(layer.pos_cap),
                    float(layer.gross_limit),
                )
                return w_out.to(dtype=dtype)

            # Delegación directa a `DifferentiablePortfolioLayer` (con todas las aux desconectadas)
            return layer(mu, vol=vol_eff, exposures=exp_d, w_prev=w_prev_d).to(dtype=dtype)

    def check_epistemic_risk(
        self,
        pred_k: dict[str, torch.Tensor],
        *,
        ood_energy: torch.Tensor | None = None,
        structural_energy: torch.Tensor | None = None,
        conflict_score: torch.Tensor | None = None,
        cf_energy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool 5: margen epistémico de Energía Libre `|mu| - kappa * sigma_epist` en GPU."""
        mu = pred_k["pred_return"][:, 0]
        alpha0 = pred_k["alpha"][:, 0]
        vol0 = pred_k["vol_pred"][:, 0]
        epist0 = pred_k["epistemic_error"][:, 0]
        device, dtype = mu.device, mu.dtype

        disagree = (mu - alpha0).abs()
        if "quantiles" in pred_k and pred_k["quantiles"] is not None:
            q = pred_k["quantiles"]
            q_med = q[:, 0, q.shape[-1] // 2]
            disagree = 0.5 * (disagree + (mu - q_med).abs())

        ood = (
            torch.zeros_like(mu)
            if ood_energy is None
            else ood_energy.to(device=device, dtype=dtype).clamp_min(0.0)
        )
        e_struct = (
            torch.zeros_like(mu)
            if structural_energy is None
            else structural_energy.to(device=device, dtype=dtype).clamp_min(0.0)
        )
        kappa = F.softplus(self.epistemic_kappa_raw.to(device=device, dtype=dtype))

        if self.is_v2 and self.epistemic_scale_raw is not None:
            conf = (
                torch.zeros_like(mu)
                if conflict_score is None
                else conflict_score.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            )
            e_cf = (
                torch.zeros_like(mu)
                if cf_energy is None
                else cf_energy.to(device=device, dtype=dtype).clamp_min(0.0)
            )
            scales = F.softplus(self.epistemic_scale_raw.to(device=device, dtype=dtype))
            sigma_epist = (
                epist0
                + scales[0] * disagree
                + scales[1] * ood
                + scales[2] * e_struct
                + scales[3] * conf
                + scales[4] * e_cf
            )
            fe_margin = mu.abs() - kappa * sigma_epist
            confidence = torch.sigmoid(fe_margin / vol0.clamp_min(1e-3))
            gated_mu = confidence * mu
            snr_bounded = 5.0 * torch.tanh((mu / sigma_epist.clamp_min(1e-4)) / 5.0)
            feats = torch.stack(
                [
                    fe_margin,
                    sigma_epist,
                    confidence,
                    gated_mu,
                    snr_bounded,
                    torch.log1p(ood),
                    conf,
                    torch.log1p(e_cf),
                ],
                dim=-1,
            )
            return feats, fe_margin, sigma_epist

        sigma_epist = epist0 + 0.25 * disagree + 0.002 * ood + 0.002 * e_struct
        fe_margin = mu.abs() - kappa * sigma_epist
        confidence = torch.sigmoid(fe_margin / vol0.clamp_min(1e-3))
        gated_mu = confidence * mu
        snr_bounded = 5.0 * torch.tanh((mu / sigma_epist.clamp_min(1e-4)) / 5.0)
        feats = torch.stack(
            [fe_margin, sigma_epist, confidence, gated_mu, snr_bounded, torch.log1p(ood)],
            dim=-1,
        )
        return feats, fe_margin, sigma_epist

    def calibrate_cross_sectional_ranks(
        self,
        alpha0: torch.Tensor,
        vol0: torch.Tensor,
        *,
        liq_pct: torch.Tensor | None = None,
        factor_cache: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool 6 (`v2_10tools`): rangos gaussianos promediados + MAD/Huber + terciles liquidez."""
        n_assets = alpha0.shape[0]
        device, dtype = alpha0.device, alpha0.dtype
        if n_assets <= 1:
            return (
                torch.zeros((n_assets, 10), device=device, dtype=dtype),
                torch.zeros(n_assets, device=device, dtype=dtype),
            )

        calc_dtype = _stable_calc_dtype(dtype)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            a_raw = alpha0.to(dtype=calc_dtype)
            a_fin = torch.isfinite(a_raw)
            a = torch.nan_to_num(a_raw, nan=0.0, posinf=0.0, neginf=0.0)
            v = torch.nan_to_num(
                vol0.detach().to(device=device, dtype=calc_dtype),
                nan=0.02,
                posinf=1.0,
                neginf=1e-4,
            ).clamp_min(1e-4)

            # 1) Rango gaussiano exacto con empates promediados (seguro ante límites CN y NaN)
            g_exact = gaussian_rank_features(
                a.detach().unsqueeze(-1), mask=a_fin.unsqueeze(-1), clip_val=4.0
            ).squeeze(-1)

            # 2) Cuartiles exactos, MAD y CDF suave diferenciable O(N * 16)
            a_sorted = torch.sort(a.detach()).values
            n_m1 = n_assets - 1
            q25 = a_sorted[n_m1 // 4]
            q50 = a_sorted[n_m1 // 2]
            q75 = a_sorted[(n_m1 * 3) // 4]
            iqr_scale = ((q75 - q25) / 1.349).clamp_min(1e-4)
            mad_raw = (a.detach() - q50).abs().median() * 1.4826
            mad_scale = torch.maximum(mad_raw, iqr_scale * 0.25).clamp_min(1e-4)

            lm_idx = (
                torch.linspace(0.0, float(n_m1), steps=16, device=device, dtype=calc_dtype)
                .round()
                .long()
                .clamp(0, n_m1)
            )
            landmarks = a_sorted.index_select(0, lm_idx).detach()  # [16]
            smooth_cdf = torch.sigmoid(
                (a.unsqueeze(-1) - landmarks.unsqueeze(0)) / (0.25 * iqr_scale.detach())
            ).mean(dim=-1)
            g_smooth = math.sqrt(2.0) * torch.erfinv((2.0 * smooth_cdf - 1.0).clamp(-0.999, 0.999))

            # 3) Z-score MAD robusto y winsorización Huber (delta = 2.5)
            z_mad = ((a - q50.detach()) / mad_scale.detach()).clamp(-12.0, 12.0)
            z_huber = 2.5 * torch.tanh(z_mad / 2.5)
            outlier_mag = F.relu(z_mad.abs() - 2.5)

            # 4) Rango relativo suave por terciles de liquidez (`liq_pct`)
            if liq_pct is not None:
                lp = torch.nan_to_num(
                    liq_pct.detach().to(device=device, dtype=calc_dtype),
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
            else:
                lp = torch.full((n_assets,), 0.5, device=device, dtype=calc_dtype)
            centers = self.liq_tercile_centers.to(device=device, dtype=calc_dtype)
            tier_dist = (lp.unsqueeze(-1) - centers.unsqueeze(0)) / 0.22
            w_tier = torch.softmax(-tier_dist.pow(2), dim=-1)
            tier_mass = w_tier.sum(dim=0).clamp_min(1e-4)  # [3]
            tier_mean = (w_tier.T @ a) / tier_mass  # [3]
            tier_diff = a.unsqueeze(-1) - tier_mean.unsqueeze(0)  # [N, 3]
            tier_mad = (
                ((w_tier * tier_diff.abs()).sum(dim=0) / tier_mass) * 1.4826
            ).clamp_min(1e-4)
            tier_z = (tier_diff / tier_mad.unsqueeze(0)).clamp(-10.0, 10.0)
            z_liq_rel = (w_tier * tier_z).sum(dim=-1)

            # 5) Z-score MAD neutralizado por factores (reutiliza factor_cache izado)
            b_n_c = factor_cache.get("decomp_b_n") if factor_cache is not None else None
            proj_c = factor_cache.get("decomp_proj") if factor_cache is not None else None
            if b_n_c is not None and proj_c is not None and b_n_c.shape[0] == n_assets:
                a_c = a - a.mean()
                a_neut = a_c - (b_n_c @ (proj_c @ a_c))
            else:
                a_neut = a - q50.detach()
            neut_med = a_neut.detach().median()
            neut_mad = ((a_neut.detach() - neut_med).abs().median() * 1.4826).clamp_min(1e-4)
            z_fac_neut = ((a_neut - neut_med) / neut_mad).clamp(-10.0, 10.0)

            # 6) Rango Sharpe ajustado por volatilidad + ratio de dispersión anti-FM-7
            sharpe_raw = a / v
            sh_med = sharpe_raw.detach().median()
            sh_mad = ((sharpe_raw.detach() - sh_med).abs().median() * 1.4826).clamp_min(1e-4)
            z_vol_adj = ((sharpe_raw - sh_med) / sh_mad).clamp(-10.0, 10.0)

            vol_med = v.median().clamp_min(1e-4)
            disp_ratio = torch.log1p(iqr_scale.detach() / vol_med).expand(n_assets)
            rank_gap = (g_smooth - g_exact).clamp(-4.0, 4.0)

            feats = torch.stack(
                [
                    g_exact,
                    g_smooth,
                    z_mad / 4.0,
                    z_huber / 2.5,
                    torch.log1p(outlier_mag),
                    z_liq_rel / 4.0,
                    z_fac_neut / 4.0,
                    z_vol_adj / 4.0,
                    disp_ratio,
                    rank_gap,
                ],
                dim=-1,
            )
            feats = torch.where(a_fin.unsqueeze(-1), feats, torch.zeros_like(feats))
            outlier_clean = torch.where(a_fin, outlier_mag, torch.zeros_like(outlier_mag))
            return feats.to(dtype=dtype), outlier_clean.to(dtype=dtype)

    def prepare_modality_cache(
        self,
        mod_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        *,
        modality_age: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tool 7 (`v2_10tools`): conflicto cross-modal izado 1 vez por sesión (0 syncs)."""
        assert self.t7_age_rate_raw is not None
        assert self.t7_mod_prec is not None
        assert self.t7_dir_proj is not None
        assert self.t7_vote_head is not None

        n_assets, n_mod, _ = mod_tokens.shape
        device, dtype = mod_tokens.device, mod_tokens.dtype
        calc_dtype = _stable_calc_dtype(dtype)

        # Aislamiento bit-a-bit de modalidades ausentes/NaN/Inf (CN fundamentales ausentes)
        mask_bool = modality_mask.to(device=device) > 0.5
        finite_tok = torch.isfinite(mod_tokens).all(dim=-1)
        valid_bool = mask_bool & finite_tok
        valid_f = valid_bool.to(dtype=calc_dtype)
        safe_tok = torch.where(valid_bool.unsqueeze(-1), mod_tokens, torch.zeros_like(mod_tokens))

        if modality_age is None:
            safe_age = torch.zeros((n_assets, n_mod), device=device, dtype=calc_dtype)
        else:
            raw_age = modality_age.to(device=device, dtype=calc_dtype)
            safe_age = torch.where(
                valid_bool & torch.isfinite(raw_age),
                raw_age.clamp_min(0.0),
                torch.zeros_like(raw_age),
            )

        age_raw = self.t7_age_rate_raw[:, :n_mod].to(device=device, dtype=calc_dtype)
        rates = F.softplus(age_raw) + 1e-3
        fresh = torch.exp(-rates * safe_age) * valid_f  # [N, M]

        # Proyección direccional compacta y votos en [-1, 1]
        dir_raw = self.t7_dir_proj(safe_tok)  # [N, M, d_dir]
        dir_n = F.normalize(dir_raw.to(dtype=calc_dtype), dim=-1, eps=1e-6)
        dir_unit = dir_n * valid_f.unsqueeze(-1)
        votes_raw = torch.tanh(self.t7_vote_head(dir_unit.to(dtype=dtype)).squeeze(-1)).to(
            dtype=calc_dtype
        )
        votes = torch.where(valid_bool, votes_raw, torch.zeros_like(votes_raw))

        # Excluir `macro` (índice 2, constante por sesión) del desacuerdo transversal (0 HtoD!)
        if n_mod == self.n_modalities:
            av_idx_t = self.t7_av_idx.to(device=device)
        else:
            all_m = torch.arange(n_mod, device=device, dtype=torch.long)
            av_idx_t = torch.cat([all_m[:2], all_m[3:]], dim=0) if n_mod >= 3 else all_m
        av_valid = valid_bool.index_select(1, av_idx_t)  # [N, M_av]
        av_dir = dir_unit.index_select(1, av_idx_t)  # [N, M_av, d_dir]
        av_votes = votes.index_select(1, av_idx_t)  # [N, M_av]
        av_fresh = fresh.index_select(1, av_idx_t)  # [N, M_av]
        av_age = safe_age.index_select(1, av_idx_t)  # [N, M_av]
        n_av = av_idx_t.shape[0]

        # Contradicción por pares en modalidades variables (0 si idénticas, 1 si opuestas)
        cos_ij = torch.einsum("nid,njd->nij", av_dir, av_dir).clamp(-1.0, 1.0)
        vote_diff_ij = 0.25 * (av_votes.unsqueeze(-1) - av_votes.unsqueeze(-2)).pow(2)
        c_ij = (0.75 * 0.5 * (1.0 - cos_ij) + 0.25 * vote_diff_ij).clamp(0.0, 1.0)

        triu_mask = torch.triu(
            torch.ones((n_av, n_av), device=device, dtype=torch.bool), diagonal=1
        )
        pair_valid = av_valid.unsqueeze(-1) & av_valid.unsqueeze(-2) & triu_mask.unsqueeze(0)
        pair_w = (av_fresh.unsqueeze(-1) * av_fresh.unsqueeze(-2)) * pair_valid.to(dtype=calc_dtype)
        has_pairs = (pair_valid.sum(dim=(-1, -2)) > 0).to(dtype=calc_dtype)
        overall_conflict = (
            (c_ij * pair_w).sum(dim=(-1, -2)) / pair_w.sum(dim=(-1, -2)).clamp_min(1e-6)
        ) * has_pairs

        # Conflicto mercado ({prices=0, charts=4}) vs información ({fundamentals=1, news=3})
        if n_mod >= 5:
            has_mkt = valid_bool[:, 0] | valid_bool[:, 4]
            has_info = valid_bool[:, 1] | valid_bool[:, 3]
            w_mkt = (fresh[:, 0] + fresh[:, 4]).clamp_min(1e-6)
            w_info = (fresh[:, 1] + fresh[:, 3]).clamp_min(1e-6)
            v_mkt = (fresh[:, 0] * votes[:, 0] + fresh[:, 4] * votes[:, 4]) / w_mkt
            v_info = (fresh[:, 1] * votes[:, 1] + fresh[:, 3] * votes[:, 3]) / w_info
            d_mkt = F.normalize(
                fresh[:, 0:1] * dir_unit[:, 0, :] + fresh[:, 4:5] * dir_unit[:, 4, :],
                dim=-1,
                eps=1e-6,
            )
            d_info = F.normalize(
                fresh[:, 1:2] * dir_unit[:, 1, :] + fresh[:, 3:4] * dir_unit[:, 3, :],
                dim=-1,
                eps=1e-6,
            )
            cos_mi = (d_mkt * d_info).sum(dim=-1).clamp(-1.0, 1.0)
            mi_raw = 0.75 * 0.5 * (1.0 - cos_mi) + 0.25 * 0.25 * (v_mkt - v_info).pow(2)
            mkt_vs_info = torch.where(
                has_mkt & has_info,
                mi_raw.clamp(0.0, 1.0),
                torch.zeros(n_assets, device=device, dtype=calc_dtype),
            )
        else:
            mkt_vs_info = overall_conflict

        # Conflicto fresco (age <= 2) vs obsoleto (age > 2)
        is_fresh = av_valid & (av_age <= 2.0)
        is_stale = av_valid & (av_age > 2.0)
        f_w = av_fresh * is_fresh.to(dtype=calc_dtype)
        s_w = av_fresh * is_stale.to(dtype=calc_dtype)
        v_fresh = (f_w * av_votes).sum(dim=-1) / f_w.sum(dim=-1).clamp_min(1e-6)
        v_stale = (s_w * av_votes).sum(dim=-1) / s_w.sum(dim=-1).clamp_min(1e-6)
        d_fresh = F.normalize((f_w.unsqueeze(-1) * av_dir).sum(dim=1), dim=-1, eps=1e-6)
        d_stale = F.normalize((s_w.unsqueeze(-1) * av_dir).sum(dim=1), dim=-1, eps=1e-6)
        cos_fs = (d_fresh * d_stale).sum(dim=-1).clamp(-1.0, 1.0)
        fs_raw = 0.75 * 0.5 * (1.0 - cos_fs) + 0.25 * 0.25 * (v_fresh - v_stale).pow(2)
        fresh_vs_stale = torch.where(
            is_fresh.any(dim=-1) & is_stale.any(dim=-1),
            fs_raw.clamp(0.0, 1.0),
            torch.zeros(n_assets, device=device, dtype=calc_dtype),
        )

        # Cobertura y bandera de publicación macro (SOLO días de publicación age[:, 2] <= 1.0!)
        coverage_ratio = valid_f.mean(dim=-1)
        if n_mod > 2:
            macro_event_flag = (valid_bool[:, 2] & (safe_age[:, 2] <= 1.0)).to(dtype=calc_dtype)
        else:
            macro_event_flag = torch.zeros(n_assets, device=device, dtype=calc_dtype)

        # Delta de arbitraje ponderado por precisión (frescura + acuerdo + precisión aprendible)
        # Cero bypass crudo de h: consenso ponderado menos media uniforme de modalidades presentes
        eye_av = torch.eye(n_av, device=device, dtype=torch.bool).unsqueeze(0)
        other_valid = av_valid.unsqueeze(-2) & (~eye_av)  # [N, M_av, M_av]
        other_w = av_fresh.unsqueeze(-2) * other_valid.to(dtype=calc_dtype)
        mean_c_av = ((c_ij * other_w).sum(dim=-1) / other_w.sum(dim=-1).clamp_min(1e-6)) * (
            other_valid.any(dim=-1).to(dtype=calc_dtype)
        )
        agree_full = torch.ones((n_assets, n_mod), device=device, dtype=calc_dtype)
        agree_full.scatter_(1, av_idx_t.unsqueeze(0).expand(n_assets, -1), 1.0 - mean_c_av)

        mod_prec = torch.sigmoid(self.t7_mod_prec[:, :n_mod].to(device=device, dtype=calc_dtype))
        prec_score = valid_f * fresh * mod_prec * (0.5 + 0.5 * agree_full)
        prec_w = prec_score / prec_score.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        unif_w = valid_f / valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mod_consensus = torch.einsum("nm,nmd->nd", prec_w.to(dtype=dtype), safe_tok)
        mod_unif = torch.einsum("nm,nmd->nd", unif_w.to(dtype=dtype), safe_tok)
        arb_delta_base = mod_consensus - mod_unif

        return dict(
            av_votes=av_votes,
            av_fresh=av_fresh,
            overall_conflict=overall_conflict,
            mkt_vs_info=mkt_vs_info,
            fresh_vs_stale=fresh_vs_stale,
            coverage_ratio=coverage_ratio,
            macro_event_flag=macro_event_flag,
            mean_fresh=fresh.mean(dim=-1),
            arb_delta_base=arb_delta_base,
        )

    def arbitrate_modal_conflict(
        self,
        alpha0: torch.Tensor,
        vol0: torch.Tensor,
        *,
        mod_cache: dict[str, torch.Tensor] | None = None,
        mod_tokens: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
        modality_age: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool 7 (`v2_10tools`): auditoría dinámica de conflicto por paso `k` + delta arbitrado."""
        n_assets = alpha0.shape[0]
        device, dtype = alpha0.device, alpha0.dtype
        if mod_cache is None and mod_tokens is not None and modality_mask is not None:
            mod_cache = self.prepare_modality_cache(
                mod_tokens, modality_mask, modality_age=modality_age
            )
        if mod_cache is None:
            return (
                torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype),
                torch.zeros((n_assets, 8), device=device, dtype=dtype),
                torch.zeros(n_assets, device=device, dtype=dtype),
                torch.zeros(n_assets, device=device, dtype=dtype),
            )

        calc_dtype = _stable_calc_dtype(dtype)
        a_c = torch.nan_to_num(alpha0.to(dtype=calc_dtype), nan=0.0, posinf=0.0, neginf=0.0)
        v_c = torch.nan_to_num(
            vol0.detach().to(device=device, dtype=calc_dtype), nan=0.02, posinf=1.0, neginf=1e-4
        ).clamp_min(1e-4)
        alpha_dir = torch.tanh(a_c / v_c)  # [N] in [-1, 1]

        av_fresh = mod_cache["av_fresh"]
        av_votes = mod_cache["av_votes"]
        av_w = av_fresh / av_fresh.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        hyp_contradiction = (av_w * F.relu(-alpha_dir.unsqueeze(-1) * av_votes)).sum(dim=-1).clamp(
            0.0, 1.0
        )

        overall_c = mod_cache["overall_conflict"]
        mkt_info_c = mod_cache["mkt_vs_info"]
        conflict_score = (0.5 * overall_c + 0.3 * mkt_info_c + 0.2 * hyp_contradiction).clamp(
            0.0, 1.0
        )
        arb_delta = (0.25 + 0.75 * conflict_score.unsqueeze(-1)).to(dtype=dtype) * mod_cache[
            "arb_delta_base"
        ]
        scalars = torch.stack(
            [
                conflict_score,
                overall_c,
                mkt_info_c,
                mod_cache["fresh_vs_stale"],
                hyp_contradiction,
                mod_cache["coverage_ratio"],
                mod_cache["macro_event_flag"],
                mod_cache["mean_fresh"],
            ],
            dim=-1,
        ).to(dtype=dtype)
        return (
            arb_delta,
            scalars,
            conflict_score.to(dtype=dtype),
            hyp_contradiction.to(dtype=dtype),
        )

    def run_scratchpad_ops(
        self,
        h: torch.Tensor,
        notes: torch.Tensor,
        *,
        urgency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool 9 (`v2_10tools`): READ por atención cruzada + WRITE/COMPACT con novedad.

        Opera directamente sobre `cognitive_notes [8, D]` (slots `0..3` persistentes entre sesiones
        con tasa conservadora; slots `4..7` borrador intra-sesión; sin banco duplicado).
        """
        assert self.t9_h_norm is not None
        assert self.t9_note_norm is not None
        assert self.t9_slot_queries is not None
        assert self.t9_write_q is not None
        assert self.t9_write_k is not None
        assert self.t9_write_v is not None
        assert self.t9_write_norm is not None
        assert self.t9_nov_gamma is not None
        assert self.t9_nov_bias is not None
        assert self.t9_out_norm is not None
        assert self.t9_read_q is not None
        assert self.t9_read_k is not None
        assert self.t9_read_v is not None
        assert self.tool9_proj is not None

        device, dtype = h.device, h.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        n_slots = notes.shape[0]
        inv_sqrt_a = 1.0 / math.sqrt(float(max(self.t9_attn_dim, 1)))

        h_n = self.t9_h_norm(h)  # [N, D]
        notes_in = notes.to(device=device, dtype=dtype)
        notes_n = self.t9_note_norm(notes_in)  # [M, D]

        # 1) WRITE/COMPACT equivariante ponderado por urgencia + compuerta de novedad
        slot_q = self.t9_slot_queries[:n_slots].to(device=device, dtype=dtype)
        q_w = self.t9_write_q(notes_n + slot_q)  # [M, d_a]
        k_w = self.t9_write_k(h_n)  # [N, d_a]
        v_w = self.t9_write_v(h_n)  # [N, D]
        w_logits = (q_w @ k_w.T).to(dtype=calc_dtype) * inv_sqrt_a  # [M, N]
        if urgency is not None:
            urg_b = torch.log1p(
                urgency.detach().to(device=device, dtype=calc_dtype).clamp_min(0.0)
            ).unsqueeze(0)
            w_logits = w_logits + urg_b
        attn_w = torch.softmax(w_logits, dim=-1).to(dtype=dtype)  # [M, N]
        u_write = self.t9_write_norm(attn_w @ v_w)  # [M, D]

        cos_sim = F.cosine_similarity(
            u_write.to(dtype=calc_dtype), notes_in.to(dtype=calc_dtype), dim=-1, eps=1e-6
        )  # [M]
        nov_raw = (1.0 - cos_sim).clamp(0.0, 2.0)
        gamma = F.softplus(self.t9_nov_gamma.to(device=device, dtype=calc_dtype))
        g_nov = torch.sigmoid(
            gamma * nov_raw + self.t9_nov_bias.to(device=device, dtype=calc_dtype)
        )  # [M] in (0, 1)

        n_p = min(self.n_persistent_notes, n_slots)
        if n_p < n_slots:
            slot_rate = torch.cat([0.25 * g_nov[:n_p], g_nov[n_p:]], dim=0)
        else:
            slot_rate = 0.25 * g_nov
        rate_col = slot_rate.unsqueeze(-1).to(dtype=dtype)
        next_notes = self.t9_out_norm((1.0 - rate_col) * notes_in + rate_col * u_write)  # [M, D]

        # 2) READ por atención cruzada direccionada por contenido sobre el borrador actualizado
        q_r = self.t9_read_q(h_n)  # [N, d_a]
        k_r_prev = self.t9_read_k(notes_n)  # [M, d_a]
        k_r = self.t9_read_k(self.t9_note_norm(next_notes))  # [M, d_a]
        v_r = self.t9_read_v(next_notes)  # [M, D]
        attn_r = torch.softmax((q_r @ k_r.T).to(dtype=calc_dtype) * inv_sqrt_a, dim=-1)  # [N, M]
        scratch_read = attn_r.to(dtype=dtype) @ v_r  # [N, D]

        hit_score = (
            F.normalize(q_r.to(dtype=calc_dtype), dim=-1, eps=1e-6)
            @ F.normalize(k_r_prev.to(dtype=calc_dtype), dim=-1, eps=1e-6).T
        ).max(dim=-1).values
        nov_asset = (attn_r @ g_nov).to(dtype=dtype)
        t9_raw = self.tool9_proj(
            torch.cat(
                [scratch_read, hit_score.to(dtype=dtype).unsqueeze(-1), nov_asset.unsqueeze(-1)],
                dim=-1,
            )
        )
        return t9_raw, next_notes, hit_score.to(dtype=dtype), g_nov.mean().to(dtype=dtype)

    def forward(
        self,
        h: torch.Tensor,
        *,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        hrr_memory: HRRModernHopfieldMemory | None = None,
        hypergraph: RMTSpilloverHypergraph | None = None,
        pred_k: dict[str, torch.Tensor] | None = None,
        w_prev: torch.Tensor | None = None,
        regime_key: torch.Tensor | None = None,
        sector_key: torch.Tensor | None = None,
        ood_energy: torch.Tensor | None = None,
        factor_cache: dict[str, torch.Tensor | None] | None = None,
        horizon_mask: torch.Tensor | None = None,
        high_snr_context: torch.Tensor | None = None,
        mod_tokens: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
        modality_age: torch.Tensor | None = None,
        mod_cache: dict[str, torch.Tensor] | None = None,
        cognitive_notes: torch.Tensor | None = None,
        cf_placebo_mode: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Ejecuta las herramientas en dispositivo y proyecta salidas a `[N, T, D]`."""
        n_assets = h.shape[0]
        device, dtype = h.device, h.dtype
        hyp = self.estimate_hypothesis(h) if pred_k is None else pred_k
        f_cache = (
            self.prepare_factor_cache(
                n_assets=n_assets,
                device=device,
                dtype=dtype,
                exposures=exposures,
                rmt_loadings=rmt_loadings,
            )
            if factor_cache is None
            else factor_cache
        )

        # Extraer liquidez causal `liq_pct` de `high_snr_context[:, CTX_LIQ_PCT]` si está disponible
        if high_snr_context is not None and high_snr_context.shape[-1] > CTX_LIQ_PCT:
            liq_pct: torch.Tensor | None = high_snr_context[:, CTX_LIQ_PCT].to(
                device=device, dtype=dtype
            )
        else:
            liq_pct = None

        # Tool 0: delta puro de remoción factorial (-h_sys, ±6*MAD pre-OLS, cero bypass crudo de h)
        h_sys, h_idio = self.decompose_factors(
            h, exposures=exposures, rmt_loadings=rmt_loadings, factor_cache=f_cache
        )
        h_c_energy = (h - h.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1, keepdim=True)
        idio_ratio = h_idio.pow(2).sum(dim=-1, keepdim=True) / h_c_energy.clamp_min(1e-6)
        if self.is_v2:
            sys_energy = h_sys.pow(2).sum(dim=-1, keepdim=True)
            fac_crowd_norm = torch.log1p(sys_energy / h_c_energy.clamp_min(1e-6))
            t0 = self.tool0_proj(torch.cat([-h_sys, idio_ratio, fac_crowd_norm], dim=-1))
        else:
            t0 = self.tool0_proj(torch.cat([-h_sys, idio_ratio], dim=-1))

        # Tool 1: precedente episódico puro + OOD enlazando `regime_key` y `sector_key`
        eff_sector_key = (
            sector_key
            if sector_key is not None
            else self.derive_sector_key(
                factor_cache=f_cache,
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                device=device,
                dtype=dtype,
            )
        )
        retrieved, ood_eff = self.lookup_hrr_memory(
            h,
            hrr_memory,
            regime_key=regime_key,
            sector_key=eff_sector_key,
            fallback_ood=ood_energy,
        )
        t1 = self.tool1_proj(
            torch.cat([retrieved, torch.log1p(ood_eff.clamp_min(0.0)).unsqueeze(-1)], dim=-1)
        )

        # Tool 2: delta puro de propagación hipergráfica + dispersión de pares `peer_disp`
        h_spill, inc_ent, peer_disp = self.run_hypergraph_spillover(
            h, hypergraph=hypergraph, rmt_loadings=rmt_loadings, return_peer_disp=True
        )
        if self.is_v2:
            t2 = self.tool2_proj(
                torch.cat(
                    [
                        h_spill - h,
                        inc_ent.unsqueeze(-1),
                        torch.log1p(peer_disp.clamp_min(0.0)).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
        else:
            t2 = self.tool2_proj(torch.cat([h_spill - h, inc_ent.unsqueeze(-1)], dim=-1))

        # Tool 3: verificador de consistencia estructural financiera (P0-4 `horizon_mask`)
        struct_feats, e_struct = self.verify_structure(
            hyp,
            exposures=exposures,
            ood_energy=ood_eff,
            factor_cache=f_cache,
            horizon_mask=horizon_mask,
        )
        t3 = self.tool3_proj(struct_feats)

        # Tool 4: factibilidad de cartera P+ (P0-5 Triton FISTA fast path + impacto liquidez)
        alpha0 = hyp["alpha"][:, 0]
        vol0 = hyp["vol_pred"][:, 0]
        w_opt = self.run_portfolio_feasibility(
            alpha0,
            vol=vol0,
            exposures=exposures,
            w_prev=w_prev,
            liq_pct=liq_pct,
            factor_cache=f_cache,
        )
        w0 = (
            torch.zeros(n_assets, device=device, dtype=dtype)
            if w_prev is None
            else w_prev.detach().to(device=device, dtype=dtype)
        )
        delta_w = w_opt - w0
        turnover_asset = delta_w.abs()
        session_turnover = turnover_asset.sum().expand(n_assets)
        a_centered = alpha0 - alpha0.mean()
        mu_norm = a_centered / a_centered.abs().sum().clamp_min(1e-6)
        feas_gap = w_opt - mu_norm
        cap_util = w_opt.abs() / max(float(self.portfolio_layer.pos_cap), 1e-6)
        if liq_pct is not None:
            ill_d = (1.0 - liq_pct.detach().clamp(0.0, 1.0)).pow(2)
            liq_impact_proxy = turnover_asset * (1.0 + 1.5 * ill_d)
        else:
            liq_impact_proxy = turnover_asset
        port_feats = torch.stack(
            [
                w_opt,
                delta_w,
                turnover_asset,
                session_turnover,
                feas_gap,
                cap_util,
                liq_impact_proxy,
            ],
            dim=-1,
        )
        t4 = self.tool4_proj(port_feats)

        if not self.is_v2:
            # Tool 5 (v1): chequeo epistémico y margen de Energía Libre
            risk_feats, fe_margin, sigma_epist = self.check_epistemic_risk(
                hyp, ood_energy=ood_eff, structural_energy=e_struct
            )
            t5 = self.tool5_proj(risk_feats)
            raw_tools = (t0, t1, t2, t3, t4, t5)
            normed = [self.tool_norms[i](ti) for i, ti in enumerate(raw_tools)]
            stacked = torch.stack(normed, dim=1)  # [N, 6, D]
            aux: dict[str, torch.Tensor] = dict(
                ood_energy=ood_eff,
                structural_energy=e_struct,
                portfolio_weights=w_opt,
                turnover_pressure=turnover_asset,
                liq_impact_proxy=liq_impact_proxy,
                free_energy_margin=fe_margin,
                epistemic_uncertainty=sigma_epist,
                hypergraph_entropy=inc_ent,
                hypergraph_peer_dispersion=peer_disp,
                hypothesis=hyp,
            )
            return stacked, aux

        # --- Herramientas ortogonales v2 (T6, T7, T8, T9) ---
        assert self.tool6_proj is not None
        assert self.tool7_proj is not None
        assert self.cf_sim is not None
        assert self.tool8_proj is not None

        # Tool 6: calibrador de rangos transversales (gaussian_rank_features + MAD/Huber/liquidez)
        rank_feats, outlier_mag = self.calibrate_cross_sectional_ranks(
            alpha0, vol0, liq_pct=liq_pct, factor_cache=f_cache
        )
        t6 = self.tool6_proj(rank_feats)

        # Tool 7: árbitro de conflicto cross-modal (excluye macro; bandera día de publicación)
        arb_delta, conflict_scalars, conflict_score, hyp_contradiction = (
            self.arbitrate_modal_conflict(
                alpha0,
                vol0,
                mod_cache=mod_cache,
                mod_tokens=mod_tokens,
                modality_mask=modality_mask,
                modality_age=modality_age,
            )
        )
        t7 = self.tool7_proj(torch.cat([arb_delta, conflict_scalars], dim=-1))

        # Tool 8: simulador contrafactual de estrés compartido (+ soporte placebo_mode)
        cf_out = self.cf_sim(
            alpha0,
            vol0,
            high_snr_context=high_snr_context,
            exposures=exposures,
            rmt_loadings=rmt_loadings,
            w_opt=w_opt,
            quantiles=hyp.get("quantiles"),
            conflict=conflict_score,
            factor_cache=f_cache,
            placebo_mode=cf_placebo_mode,
        )
        e_cf = cf_out["e_cf"]
        p_flip = cf_out["p_flip"]
        worst_pnl = cf_out["worst_pnl"]
        t8 = self.tool8_proj(cf_out["cf_trace"])

        # Tool 5 (v2): chequeo epistémico unificado integrando `conflict_score` y `e_cf`
        risk_feats, fe_margin, sigma_epist = self.check_epistemic_risk(
            hyp,
            ood_energy=ood_eff,
            structural_energy=e_struct,
            conflict_score=conflict_score,
            cf_energy=e_cf,
        )
        t5 = self.tool5_proj(risk_feats)

        # Tool 9: operaciones sobre el borrador unificado `cognitive_notes [8, D]`
        notes_in = (
            cognitive_notes
            if cognitive_notes is not None
            else torch.zeros((self.n_notes, self.hidden_dim), device=device, dtype=dtype)
        )
        urgency = e_struct.detach() + e_cf.detach() + conflict_score.detach()
        t9, next_notes, scratch_hit, scratch_novelty = self.run_scratchpad_ops(
            h, notes_in, urgency=urgency
        )

        raw_tools_v2 = (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9)
        normed_v2 = [self.tool_norms[i](ti) for i, ti in enumerate(raw_tools_v2)]
        stacked_v2 = torch.stack(normed_v2, dim=1)  # [N, 10, D]
        aux_v2: dict[str, torch.Tensor] = dict(
            ood_energy=ood_eff,
            structural_energy=e_struct,
            portfolio_weights=w_opt,
            turnover_pressure=turnover_asset,
            liq_impact_proxy=liq_impact_proxy,
            free_energy_margin=fe_margin,
            epistemic_uncertainty=sigma_epist,
            hypergraph_entropy=inc_ent,
            hypergraph_peer_dispersion=peer_disp,
            outlier_mag=outlier_mag,
            conflict_score=conflict_score,
            hyp_contradiction=hyp_contradiction,
            e_cf=e_cf,
            p_flip=p_flip,
            worst_pnl=worst_pnl,
            cf_trace=cf_out["cf_trace"],
            scratchpad_hit_score=scratch_hit,
            scratchpad_novelty_mean=scratch_novelty,
            next_cognitive_notes=next_notes,
            hypothesis=hyp,
        )
        return stacked_v2, aux_v2


class ToolRouter(nn.Module):
    """Enrutador de herramientas (`v1` estático o `v2` lazo cerrado con fase y annealing)."""

    def __init__(
        self,
        context_dim: int,
        *,
        max_steps: int = DEFAULT_THINKING_STEPS,
        n_tools: int = 6,
        step_emb_dim: int = 8,
        router_dim: int = 32,
        mode: str = "soft",
        temperature: float = 1.0,
        closed_loop: bool = False,
        hidden_dim: int | None = None,
        diag_dim: int = 8,
        use_phase_prior: bool = False,
        anneal_temperature: bool = False,
        anneal_total_steps: int = 2000,
    ) -> None:
        super().__init__()
        if mode not in TOOL_ROUTER_MODES:
            raise ValueError(f"mode inválido {mode!r}; válidos: {TOOL_ROUTER_MODES}")
        if not (1 <= max_steps <= MAX_THINKING_STEPS):
            raise ValueError(f"max_steps debe satisfacer 1 <= max_steps <= {MAX_THINKING_STEPS}")
        if n_tools < 1 or temperature <= 0.0:
            raise ValueError("n_tools >= 1 y temperature > 0 son obligatorios")
        if closed_loop and (hidden_dim is None or hidden_dim < 8):
            raise ValueError("hidden_dim >= 8 es obligatorio cuando closed_loop=True")

        self.context_dim = context_dim
        self.max_steps = int(max_steps)
        self.n_tools = n_tools
        self.step_emb_dim = int(step_emb_dim)
        self.mode = mode
        self.temperature = float(temperature)
        self.closed_loop = bool(closed_loop)
        self.hidden_dim = hidden_dim
        self.diag_dim = int(diag_dim)
        self.use_phase_prior = bool(use_phase_prior)
        self.anneal_temperature = bool(anneal_temperature)
        self.anneal_total_steps = max(1, int(anneal_total_steps))

        if self.anneal_temperature:
            if self.mode == "sparsemax":
                self.temp_start = 2.0 * self.temperature
                self.temp_end = 1.0 * self.temperature
            else:
                self.temp_start = 1.5 * self.temperature
                self.temp_end = 0.7 * self.temperature
        else:
            self.temp_start = self.temperature
            self.temp_end = self.temperature

        self.ctx_norm = nn.LayerNorm(context_dim)
        self.ctx_proj = nn.Linear(context_dim, router_dim)
        self.step_embedding = nn.Embedding(self.max_steps, self.step_emb_dim)
        self.step_proj = nn.Linear(self.step_emb_dim, router_dim)

        if self.closed_loop:
            assert hidden_dim is not None
            self.h_summary_proj: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 16),
                nn.LayerNorm(16),
                nn.Linear(16, router_dim),
            )
            self.diag_proj: nn.Sequential | None = nn.Sequential(
                nn.Linear(self.diag_dim, router_dim),
                nn.SiLU(),
                nn.Linear(router_dim, router_dim),
            )
        else:
            self.h_summary_proj = None
            self.diag_proj = None

        if self.use_phase_prior:
            self.phase_prior: nn.Parameter | None = nn.Parameter(
                _init_phase_prior(self.max_steps, n_tools)
            )
        else:
            self.phase_prior = None

        self.router_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(router_dim, n_tools),
        )
        self.register_buffer("step_ids", torch.arange(self.max_steps, dtype=torch.long))
        self.register_buffer("balance_bias", torch.zeros(n_tools, dtype=torch.float32))
        self.register_buffer("anneal_step", torch.zeros((), dtype=torch.float32))

    def get_step_embedding(self, step_idx: int, *, dtype: torch.dtype) -> torch.Tensor:
        """Devuelve el embedding de paso `[step_emb_dim]` en dispositivo sin copias CPU."""
        clamped_idx = min(max(int(step_idx), 0), self.max_steps - 1)
        return self.step_embedding(self.step_ids[clamped_idx]).to(dtype=dtype)

    def current_temperature(self) -> torch.Tensor:
        """Temperatura efectiva del router (annealing coseno en GPU, 0 syncs, BPTT-safe)."""
        if not self.anneal_temperature:
            return torch.full(
                (), self.temperature, device=self.anneal_step.device, dtype=torch.float32
            )
        horizon = max(0.4 * float(self.anneal_total_steps), 1.0)
        frac = (self.anneal_step.detach().clone() / horizon).clamp(0.0, 1.0)
        cos_w = 0.5 * (1.0 + torch.cos(math.pi * frac))
        return self.temp_end + (self.temp_start - self.temp_end) * cos_w

    @torch.no_grad()
    def advance_temperature_schedule(self, steps: int = 1) -> None:
        """Avanza el calendario coseno de temperatura estrictamente en `train()` (0 syncs)."""
        if not self.training:
            return
        self.anneal_step.add_(float(max(int(steps), 0)))
        self.anneal_step.clamp_(0.0, float(self.anneal_total_steps))

    @torch.no_grad()
    def update_balance_bias(self, tool_loads: torch.Tensor, *, lr: float = 0.05) -> None:
        """Balanceo DeepSeek-V3 Aux-Loss-Free estrictamente en `train()` (clamped `[-3, 3]`)."""
        if not self.training:
            return
        target = 1.0 / float(self.n_tools)
        loads = tool_loads.detach().to(device=self.balance_bias.device, dtype=torch.float32)
        diff = target - loads
        step = lr * torch.sign(diff) * torch.clamp(diff.abs() * self.n_tools, 0.2, 2.0)
        self.balance_bias.add_(step)
        self.balance_bias.sub_(self.balance_bias.mean())
        self.balance_bias.clamp_(-3.0, 3.0)

    def forward(
        self,
        high_snr_context: torch.Tensor,
        step_idx: int,
        *,
        h_state: torch.Tensor | None = None,
        prev_diagnostics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve pesos de herramientas `[N, n_tools]` y entropía por activo `[N]`."""
        n_assets = high_snr_context.shape[0]
        device, dtype = high_snr_context.device, high_snr_context.dtype

        if self.mode == "no_tools":
            weights = torch.zeros((n_assets, self.n_tools), device=device, dtype=dtype)
            ent = torch.zeros(n_assets, device=device, dtype=dtype)
            return weights, ent

        if self.mode == "uniform":
            calc_dtype = _stable_calc_dtype(dtype)
            weights_c = torch.full(
                (n_assets, self.n_tools),
                1.0 / float(self.n_tools),
                device=device,
                dtype=calc_dtype,
            )
            ent = -(weights_c * torch.log(weights_c.clamp_min(1e-8))).sum(dim=-1)
            return weights_c.to(dtype=dtype), ent.to(dtype=dtype)

        calc_dtype = _stable_calc_dtype(dtype)
        clamped_idx = min(max(int(step_idx), 0), self.max_steps - 1)
        step_e = self.get_step_embedding(clamped_idx, dtype=dtype)
        h_ctx = self.ctx_proj(self.ctx_norm(high_snr_context))
        h_feat = h_ctx + self.step_proj(step_e).unsqueeze(0)

        if self.closed_loop and self.h_summary_proj is not None and self.diag_proj is not None:
            if h_state is not None:
                h_feat = h_feat + self.h_summary_proj(h_state.to(device=device, dtype=dtype))
            else:
                assert self.hidden_dim is not None
                h_feat = h_feat + self.h_summary_proj(
                    torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype)
                )
            if prev_diagnostics is not None:
                d_in = prev_diagnostics.to(device=device, dtype=dtype)
            else:
                d_in = torch.zeros((n_assets, self.diag_dim), device=device, dtype=dtype)
            h_feat = h_feat + self.diag_proj(d_in)

        raw_logits = self.router_head(h_feat).to(dtype=calc_dtype)
        if self.phase_prior is not None:
            raw_logits = raw_logits + self.phase_prior.index_select(
                0, self.step_ids[clamped_idx].reshape(1)
            ).to(device=device, dtype=calc_dtype)

        eff_temp = self.current_temperature().to(device=device, dtype=calc_dtype).clamp_min(1e-4)
        # `.clone()` obligatorio antes de sumar bias para impedir error de versión BPTT in-place!
        bias_c = self.balance_bias.detach().clone().to(device=device, dtype=calc_dtype).unsqueeze(0)
        logits = raw_logits / eff_temp + bias_c

        weights_c = (
            _sparsemax(logits, dim=-1)
            if self.mode == "sparsemax"
            else torch.softmax(logits, dim=-1)
        )
        ent = -(weights_c * torch.log(weights_c.clamp_min(1e-8))).sum(dim=-1)
        return weights_c.to(dtype=dtype), ent.to(dtype=dtype)


class AdaptiveHaltGate(nn.Module):
    """Compuerta aprendida de parada adaptativa (ACT/PonderNet) acoplada a consistencia monótona.

    Weight-tied entre todos los pasos de pensamiento `1..K` (`K <= 4`), 100 % diferenciable
    en pre-entrenamiento end-to-end y sin sincronizaciones CPU↔GPU ni ramas Python sobre datos.
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        step_emb_dim: int = 8,
        gate_dim: int = 16,
        monotonic_halt_boost: float = 2.0,
        use_difficulty: bool = False,
    ) -> None:
        super().__init__()
        self.monotonic_halt_boost = float(monotonic_halt_boost)
        self.use_difficulty = bool(use_difficulty)
        self.h_norm = nn.LayerNorm(hidden_dim)
        self.ctx_norm = nn.LayerNorm(context_dim)
        self.h_proj = nn.Linear(hidden_dim, gate_dim)
        self.meta_proj = nn.Linear(context_dim + step_emb_dim + 3, gate_dim)
        self.out_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(gate_dim, 1),
        )
        if self.use_difficulty:
            self.difficulty_defer_raw: nn.Parameter | None = nn.Parameter(
                torch.tensor(_inv_softplus(2.0), dtype=torch.float32)
            )
        else:
            self.difficulty_defer_raw = None

    def forward(
        self,
        h: torch.Tensor,
        delta_norm: torch.Tensor,
        consistency_energy: torch.Tensor,
        worsened_energy: torch.Tensor,
        high_snr_context: torch.Tensor,
        step_embedding: torch.Tensor,
        *,
        difficulty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Devuelve la probabilidad condicional de parada `lambda_k in (0, 1)` de forma `[N]`."""
        n_assets = h.shape[0]
        device, dtype = h.device, h.dtype
        ctx_n = self.ctx_norm(high_snr_context.to(device=device, dtype=dtype))
        step_exp = step_embedding.to(device=device, dtype=dtype).unsqueeze(0).expand(n_assets, -1)
        scalars = torch.stack(
            [
                torch.log1p(delta_norm.clamp_min(0.0)),
                torch.log1p(consistency_energy.clamp_min(0.0)),
                torch.log1p(worsened_energy.clamp_min(0.0)),
            ],
            dim=-1,
        )
        meta = torch.cat([ctx_n, step_exp, scalars], dim=-1)
        feat = self.h_proj(self.h_norm(h)) + self.meta_proj(meta)
        logit = self.out_head(feat).squeeze(-1)
        # Empuje monótono: si la consistencia empeora, aumenta de inmediato la probabilidad de parar
        logit = logit + self.monotonic_halt_boost * torch.log1p(worsened_energy.clamp_min(0.0))
        # Modulación por dificultad en v2: activos fáciles paran en k=1, difíciles iteran k=2..3
        if self.use_difficulty and self.difficulty_defer_raw is not None and difficulty is not None:
            defer = F.softplus(self.difficulty_defer_raw.to(device=device, dtype=dtype))
            rho_c = difficulty.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            logit = logit - defer * (2.0 * rho_c - 1.0)
        return torch.sigmoid(logit)


class _SpectralLinear(nn.Module):
    """Capa lineal con normalización espectral CUDA-Graph/`compile`-safe (cero mutación fwd)."""

    def __init__(self, in_features: int, out_features: int, *, n_power_iters: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(float(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        self.n_power_iters = int(n_power_iters)

        u = F.normalize(torch.randn(out_features), dim=0, eps=1e-12)
        v = F.normalize(torch.randn(in_features), dim=0, eps=1e-12)
        with torch.no_grad():
            for _ in range(20):
                v = F.normalize(self.weight.T @ u, dim=0, eps=1e-12)
                u = F.normalize(self.weight @ v, dim=0, eps=1e-12)
        self.register_buffer("u", u)
        self.register_buffer("v", v)

    @torch.no_grad()
    def update_spectral_buffers(self, *, n_iters: int | None = None) -> None:
        """Hook post-step seguro: actualiza `u`/`v` fuera de `forward` (compatible CUDA Graphs)."""
        iters = self.n_power_iters if n_iters is None else max(1, int(n_iters))
        calc_dtype = _stable_calc_dtype(self.weight.dtype)
        u_cur = self.u.to(device=self.weight.device, dtype=calc_dtype)
        v_cur = self.v.to(device=self.weight.device, dtype=calc_dtype)
        w_det = self.weight.detach().to(dtype=calc_dtype)
        for _ in range(iters):
            v_cur = F.normalize(w_det.T @ u_cur, dim=0, eps=1e-12)
            u_cur = F.normalize(w_det @ v_cur, dim=0, eps=1e-12)
        self.u.copy_(u_cur.to(dtype=self.u.dtype))
        self.v.copy_(v_cur.to(dtype=self.v.dtype))

    def normalized_weight(self, *, update_buffers: bool = False) -> torch.Tensor:
        """Devuelve `W / max(sigma_max(W), 1)` sin mutar buffers durante `forward`."""
        w_dtype = self.weight.dtype
        calc_dtype = _stable_calc_dtype(w_dtype)
        u_cur = self.u.to(device=self.weight.device, dtype=calc_dtype)
        v_cur = self.v.to(device=self.weight.device, dtype=calc_dtype)
        w_det = self.weight.detach().to(dtype=calc_dtype)
        with torch.no_grad():
            for _ in range(self.n_power_iters):
                v_cur = F.normalize(w_det.T @ u_cur, dim=0, eps=1e-12)
                u_cur = F.normalize(w_det @ v_cur, dim=0, eps=1e-12)
            if update_buffers:
                self.u.copy_(u_cur.to(dtype=self.u.dtype))
                self.v.copy_(v_cur.to(dtype=self.v.dtype))
        u_snap = u_cur.detach()
        v_snap = v_cur.detach()
        w_calc = self.weight.to(dtype=calc_dtype)
        sigma = torch.dot(u_snap, w_calc @ v_snap).abs().clamp_min(1.0)
        return (w_calc / sigma).to(dtype=w_dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cached_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = self.normalized_weight(update_buffers=False) if cached_weight is None else cached_weight
        return F.linear(x, w.to(dtype=x.dtype), self.bias.to(dtype=x.dtype))


class _WeightTiedContractiveCore(nn.Module):
    """Celda recurrente compacta (cuello `core_dim <= hidden_dim`) estrictamente contractiva."""

    def __init__(
        self,
        hidden_dim: int,
        core_dim: int,
        *,
        lipschitz_bound: float = 0.90,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.core_dim = core_dim
        self.lipschitz_bound = float(lipschitz_bound)

        self.sn_in_h = _SpectralLinear(hidden_dim, core_dim)
        self.sn_in_x0 = _SpectralLinear(hidden_dim, core_dim)
        self.sn_in_tool = _SpectralLinear(hidden_dim, core_dim)
        self.sn_core = _SpectralLinear(core_dim, core_dim)
        self.sn_out = _SpectralLinear(core_dim, hidden_dim)

        # Compuerta de herramienta dependiente solo de (x0, tool_msg) -> no altera J_h
        self.tool_gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.update_beta_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def update_spectral_buffers(self) -> None:
        """Actualiza buffers espectrales `u`/`v` de las 5 proyecciones tras `optimizer.step()`."""
        for layer in (self.sn_in_h, self.sn_in_x0, self.sn_in_tool, self.sn_core, self.sn_out):
            layer.update_spectral_buffers()

    def compute_spectral_weights(self, *, update_buffers: bool = False) -> tuple[torch.Tensor, ...]:
        """Calcula los 5 pesos normalizados espectralmente una vez por sesión (ahorro en GPU)."""
        return (
            self.sn_in_h.normalized_weight(update_buffers=update_buffers),
            self.sn_in_x0.normalized_weight(update_buffers=update_buffers),
            self.sn_in_tool.normalized_weight(update_buffers=update_buffers),
            self.sn_core.normalized_weight(update_buffers=update_buffers),
            self.sn_out.normalized_weight(update_buffers=update_buffers),
        )

    def lipschitz_upper_bound(self) -> torch.Tensor:
        """Cota superior certificada `||d h_next / d h||_2 < lipschitz_bound < 1` (tensor 0-D)."""
        beta = 0.15 + 0.45 * torch.sigmoid(self.update_beta_logit)
        return self.lipschitz_bound * (1.0 - beta * (1.0 - 1.0 / math.sqrt(3.0)))

    def forward(
        self,
        x0: torch.Tensor,
        h: torch.Tensor,
        tool_msg: torch.Tensor,
        *,
        cached_weights: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Aplica un paso contractivo weight-tied integrando `(x0, h, tool_msg)`."""
        w_h, w_x0, w_tool, w_core, w_out = (
            self.compute_spectral_weights(update_buffers=False)
            if cached_weights is None
            else cached_weights
        )
        tool_gate = torch.sigmoid(self.tool_gate_proj(torch.cat([x0, tool_msg], dim=-1)))
        gated_tool = tool_gate * tool_msg

        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        u = torch.tanh(
            (
                self.sn_in_h(h, cached_weight=w_h)
                + self.sn_in_x0(x0, cached_weight=w_x0)
                + self.sn_in_tool(gated_tool, cached_weight=w_tool)
            )
            * inv_sqrt3
        )
        u_mid = torch.tanh(self.sn_core(u, cached_weight=w_core))
        proposal = torch.tanh(self.sn_out(u_mid, cached_weight=w_out)) * self.lipschitz_bound

        beta = (0.15 + 0.45 * torch.sigmoid(self.update_beta_logit)).to(
            device=h.device, dtype=h.dtype
        )
        l_b = self.lipschitz_bound
        # Pesos suman 1.0 -> conserva escala de activaciones y garantiza contracción ||J_h|| < L < 1
        return l_b * (1.0 - beta) * h + (1.0 - l_b) * (1.0 - beta) * x0 + beta * proposal


class TitansCognitiveCore(nn.Module):
    """Núcleo Cognitivo Compacto weight-tied con parada adaptativa y bucle de herramientas."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        core_dim: int | None = None,
        thinking_steps: int = DEFAULT_THINKING_STEPS,
        tool_mode: str = "soft",
        toolbox_version: Literal["v1_6tools", "v2_10tools"] = "v1_6tools",
        n_modalities: int = 5,
        n_notes: int | None = None,
        n_persistent_notes: int | None = None,
        n_hyperedges: int = 16,
        rmt_dim: int = 5,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        lipschitz_bound: float = 0.90,
        router_temperature: float = 1.0,
        halt_prior_p: float = 0.45,
        ponder_weight: float = DEFAULT_PONDER_WEIGHT,
        monotonic_penalty: float = 5.0,
        use_context_engineer: bool = False,
        cf_placebo_mode: bool = False,
        tool_balance_weight: float = 0.005,
        tool_decor_weight: float = 0.01,
        tool_decor_margin: float = 0.85,
        router_anneal_total_steps: int = 2000,
        norm_kind: str = "layernorm",
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or context_dim < 1:
            raise ValueError("hidden_dim >= 8 y context_dim >= 1 son obligatorios")
        if toolbox_version not in TOOLBOX_VERSIONS:
            raise ValueError(
                f"toolbox_version inválido {toolbox_version!r}; válidos: {TOOLBOX_VERSIONS}"
            )
        resolved_core_dim = core_dim if core_dim is not None else max(8, hidden_dim // 2)
        if not (8 <= resolved_core_dim <= hidden_dim):
            raise ValueError(
                f"core_dim ({resolved_core_dim}) debe satisfacer 8 <= core_dim <= hidden_dim "
                f"({hidden_dim})"
            )
        if not (1 <= thinking_steps <= MAX_THINKING_STEPS):
            raise ValueError(
                f"thinking_steps ({thinking_steps}) debe cumplir el tope duro "
                f"1 <= thinking_steps <= {MAX_THINKING_STEPS}"
            )
        if not (0.1 <= lipschitz_bound < 1.0):
            raise ValueError("lipschitz_bound debe satisfacer 0.1 <= lipschitz_bound < 1.0")
        if tool_mode not in TOOL_ROUTER_MODES:
            raise ValueError(f"tool_mode inválido {tool_mode!r}; válidos: {TOOL_ROUTER_MODES}")
        if not (0.05 <= halt_prior_p <= 0.95):
            raise ValueError("halt_prior_p debe pertenecer a [0.05, 0.95]")
        if ponder_weight < 0.0:
            raise ValueError("ponder_weight debe ser >= 0.0")

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.core_dim = resolved_core_dim
        self.thinking_steps = int(thinking_steps)
        self.tool_mode = tool_mode
        self.toolbox_version = toolbox_version
        self.is_v2 = toolbox_version == "v2_10tools"
        self.tool_names = TOOL_NAMES_V2 if self.is_v2 else TOOL_NAMES
        self.n_tools = len(self.tool_names)
        self.lipschitz_bound = float(lipschitz_bound)
        self.ponder_weight = float(ponder_weight)
        self.monotonic_penalty = float(monotonic_penalty)
        self.use_context_engineer = bool(use_context_engineer)
        self.cf_placebo_mode = bool(cf_placebo_mode)
        self.tool_balance_weight = float(tool_balance_weight)
        self.tool_decor_weight = float(tool_decor_weight)
        self.tool_decor_margin = float(tool_decor_margin)
        self.norm_kind = str(norm_kind)

        resolved_n_notes = (8 if self.is_v2 else 4) if n_notes is None else int(n_notes)
        resolved_n_persist = (
            (min(4, resolved_n_notes) if self.is_v2 else resolved_n_notes)
            if n_persistent_notes is None
            else min(int(n_persistent_notes), resolved_n_notes)
        )
        self.n_notes = resolved_n_notes
        self.n_persistent_notes = resolved_n_persist

        self.context_engineer = (
            ContextEngineerGPU(
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                n_modalities=n_modalities,
                n_notes=resolved_n_notes,
                n_persistent_notes=resolved_n_persist,
                norm_kind=self.norm_kind,
            )
            if self.use_context_engineer
            else None
        )
        # Si use_context_engineer=True, T9 reutiliza `context_engineer.init_notes` (sin duplicar)
        if self.is_v2 and not self.use_context_engineer:
            self.scratchpad_init_notes: nn.Parameter | None = nn.Parameter(
                torch.zeros(resolved_n_notes, hidden_dim, dtype=torch.float32)
            )
        else:
            self.scratchpad_init_notes = None

        self.tool_router = ToolRouter(
            context_dim=context_dim,
            max_steps=self.thinking_steps,
            n_tools=self.n_tools,
            step_emb_dim=8,
            router_dim=max(16, min(32, resolved_core_dim)),
            mode=tool_mode,
            temperature=router_temperature,
            closed_loop=self.is_v2,
            hidden_dim=hidden_dim if self.is_v2 else None,
            diag_dim=8,
            use_phase_prior=self.is_v2,
            anneal_temperature=self.is_v2,
            anneal_total_steps=router_anneal_total_steps,
        )
        self.toolbox = DifferentiableQuantToolbox(
            hidden_dim=hidden_dim,
            core_dim=resolved_core_dim,
            toolbox_version=toolbox_version,
            rmt_dim=rmt_dim,
            n_hyperedges=n_hyperedges,
            n_modalities=n_modalities,
            n_notes=resolved_n_notes,
            n_persistent_notes=resolved_n_persist,
            horizons=horizons,
            cf_placebo_mode=cf_placebo_mode,
        )
        self.core_cell = _WeightTiedContractiveCore(
            hidden_dim=hidden_dim,
            core_dim=resolved_core_dim,
            lipschitz_bound=lipschitz_bound,
        )
        self.halt_gate = AdaptiveHaltGate(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            step_emb_dim=self.tool_router.step_emb_dim,
            gate_dim=16,
            use_difficulty=self.is_v2,
        )

        # Prior geométrico truncado en [1..thinking_steps] para regularización PonderNet KL (v1)
        prior_vals = []
        rem_p = 1.0
        for step_k in range(self.thinking_steps):
            if step_k + 1 < self.thinking_steps:
                pk = rem_p * float(halt_prior_p)
                rem_p *= 1.0 - float(halt_prior_p)
            else:
                pk = rem_p
            prior_vals.append(max(pk, 1e-6))
        prior_t = torch.tensor(prior_vals, dtype=torch.float32)
        self.register_buffer("halt_prior", prior_t / prior_t.sum())

    def update_spectral_buffers(self) -> None:
        """Actualiza buffers espectrales `u`/`v` fuera de `forward` (compatible CUDA Graphs)."""
        self.core_cell.update_spectral_buffers()

    def update_router_balance_bias(self, tool_loads: torch.Tensor, *, lr: float = 0.05) -> None:
        """Actualiza el sesgo DeepSeek-V3 del `ToolRouter` estrictamente en `train()`."""
        self.tool_router.update_balance_bias(tool_loads, lr=lr)

    def step_router_schedule(self, steps: int = 1) -> None:
        """Avanza el calendario coseno de temperatura del `ToolRouter` en `train()`."""
        self.tool_router.advance_temperature_schedule(steps=steps)

    def register_spectral_post_step_hook(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.utils.hooks.RemovableHandle:
        """Registra un hook post-step en `optimizer` para actualizar `_SpectralLinear`."""

        def _hook(_opt: torch.optim.Optimizer, _args: object, _kwargs: object) -> None:
            self.update_spectral_buffers()

        return optimizer.register_step_post_hook(_hook)

    def parameter_summary(self) -> dict[str, int]:
        """Resume la huella compacta de parámetros separando núcleo weight-tied, router y caja."""
        core_p = sum(p.numel() for p in self.core_cell.parameters())
        router_p = sum(p.numel() for p in self.tool_router.parameters())
        halt_p = sum(p.numel() for p in self.halt_gate.parameters())
        toolbox_p = sum(p.numel() for p in self.toolbox.parameters())
        ce_p = (
            sum(p.numel() for p in self.context_engineer.parameters())
            if self.context_engineer is not None
            else 0
        )
        sp_p = (
            self.scratchpad_init_notes.numel() if self.scratchpad_init_notes is not None else 0
        )
        return dict(
            total_params=core_p + router_p + halt_p + toolbox_p + ce_p + sp_p,
            core_params=core_p,
            router_params=router_p,
            halt_gate_params=halt_p,
            toolbox_params=toolbox_p,
            context_engineer_params=ce_p,
            thinking_steps=self.thinking_steps,
            core_dim=self.core_dim,
            hidden_dim=self.hidden_dim,
            n_tools=self.n_tools,
        )

    def compute_probe_aux_loss(
        self,
        step_predictions: list[dict[str, torch.Tensor]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Supervisión profunda auxiliar sobre `probe_head` cuando `head` externo es `None`."""
        if not step_predictions:
            return torch.zeros((), device=targets.device, dtype=targets.dtype)
        weights = [float(k + 1) for k in range(len(step_predictions))]
        w_sum = sum(weights)
        ref = step_predictions[0]["pred_return"]
        total = torch.zeros((), device=ref.device, dtype=ref.dtype)
        for wk, hyp_k in zip(weights, step_predictions, strict=True):
            total = total + (wk / w_sum) * self.toolbox.compute_probe_aux_loss(hyp_k, targets)
        return total

    @staticmethod
    def _compute_difficulty_index(
        *,
        e_struct: torch.Tensor,
        e_cf: torch.Tensor,
        conflict: torch.Tensor,
        sigma_epist: torch.Tensor,
        vol0: torch.Tensor,
        calc_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Índice de dificultad compuesta desacoplado `rho in (0, 1)` para `AdaptiveHaltGate v2`."""
        e_s = torch.log1p(e_struct.detach().to(dtype=calc_dtype).clamp_min(0.0))
        e_c = torch.log1p(e_cf.detach().to(dtype=calc_dtype).clamp_min(0.0))
        conf = conflict.detach().to(dtype=calc_dtype).clamp(0.0, 1.0)
        ep_r = torch.log1p(
            (
                sigma_epist.detach().to(dtype=calc_dtype)
                / vol0.detach().to(dtype=calc_dtype).clamp_min(1e-4)
            ).clamp_min(0.0)
        )
        score = 0.65 * e_s + 1.0 * e_c + 1.5 * conf + 0.55 * ep_r
        return torch.sigmoid(2.0 * (score - 1.0))

    def _init_standalone_scratchpad(
        self,
        prev_notes: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Inicializa `cognitive_notes [8, D]` cuando `use_context_engineer=False` en v2."""
        if self.scratchpad_init_notes is not None:
            init_n = self.scratchpad_init_notes.to(device=device, dtype=dtype)
        elif self.context_engineer is not None:
            init_n = self.context_engineer.init_notes.to(device=device, dtype=dtype)
        else:
            init_n = torch.zeros((self.n_notes, self.hidden_dim), device=device, dtype=dtype)
        if prev_notes is None:
            return init_n
        prev_t = prev_notes.to(device=device, dtype=dtype)
        n_p = min(self.n_persistent_notes, self.n_notes, prev_t.shape[0])
        if n_p >= self.n_notes:
            return prev_t[: self.n_notes]
        return torch.cat([prev_t[:n_p], init_n[n_p:]], dim=0)

    def forward(
        self,
        x0: torch.Tensor | None,
        high_snr_context: torch.Tensor,
        *,
        mod_tokens: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
        physics_emb: torch.Tensor | None = None,
        modality_age: torch.Tensor | None = None,
        prev_notes: torch.Tensor | None = None,
        state: dict[str, object] | None = None,
        pruned_context_ratio: torch.Tensor | None = None,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        hrr_memory: HRRModernHopfieldMemory | None = None,
        hypergraph: RMTSpilloverHypergraph | None = None,
        head: nn.Module | None = None,
        probe_targets: torch.Tensor | None = None,
        w_prev: torch.Tensor | None = None,
        regime_key: torch.Tensor | None = None,
        sector_key: torch.Tensor | None = None,
        ood_energy: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        cf_placebo_mode: bool | None = None,
        enable_early_exit: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, object]]:
        """Ejecuta `1..thinking_steps` iteraciones con parada adaptativa y consistencia monótona."""
        salience_weights: torch.Tensor | None = None
        state_notes = (
            state.get("cognitive_notes")
            if (state is not None and isinstance(state.get("cognitive_notes"), torch.Tensor))
            else None
        )
        eff_prev_notes = prev_notes if prev_notes is not None else state_notes
        context_notes: torch.Tensor | None = eff_prev_notes

        if mod_tokens is not None:
            if self.context_engineer is None:
                raise ValueError(
                    "use_context_engineer=True es obligatorio al pasar `mod_tokens` directamente"
                )
            if modality_mask is None or physics_emb is None:
                raise ValueError(
                    "`modality_mask` y `physics_emb` son obligatorios junto con `mod_tokens`"
                )
            anchor_x, salience_weights, pruned_ratio, context_notes = self.context_engineer(
                mod_tokens,
                modality_mask,
                physics_emb,
                high_snr_context,
                modality_age=modality_age,
                prev_notes=eff_prev_notes,
            )
        else:
            if x0 is None:
                raise ValueError("Debe suministrarse `x0` o `mod_tokens`")
            anchor_x = x0
            pruned_ratio = (
                pruned_context_ratio.to(device=anchor_x.device, dtype=anchor_x.dtype)
                if pruned_context_ratio is not None
                else torch.zeros((), device=anchor_x.device, dtype=anchor_x.dtype)
            )
            if self.is_v2:
                context_notes = self._init_standalone_scratchpad(
                    eff_prev_notes, device=anchor_x.device, dtype=anchor_x.dtype
                )

        n_assets = anchor_x.shape[0]
        device, dtype = anchor_x.device, anchor_x.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        ctx = high_snr_context.to(device=device, dtype=dtype)
        cached_weights = self.core_cell.compute_spectral_weights(update_buffers=False)
        inv_sqrt_d = 1.0 / math.sqrt(float(self.hidden_dim))

        # Hoist normalizaciones, Gram B^T B, operadores ridge y caché cross-modal 1 vez por sesión
        factor_cache = (
            self.toolbox.prepare_factor_cache(
                n_assets=n_assets,
                device=device,
                dtype=dtype,
                exposures=exposures,
                rmt_loadings=rmt_loadings,
            )
            if self.tool_mode != "no_tools"
            else None
        )
        mod_cache = (
            self.toolbox.prepare_modality_cache(
                mod_tokens, modality_mask, modality_age=modality_age
            )
            if (
                self.is_v2
                and self.tool_mode != "no_tools"
                and mod_tokens is not None
                and modality_mask is not None
            )
            else None
        )

        h = anchor_x
        h_acc = torch.zeros_like(anchor_x)
        remaining = torch.ones(n_assets, device=device, dtype=dtype)
        cum_halt = torch.zeros(n_assets, device=device, dtype=dtype)
        expected_steps = torch.zeros(n_assets, device=device, dtype=dtype)
        best_energy: torch.Tensor | None = None

        pred_cur: dict[str, torch.Tensor] | None = None
        if self.tool_mode != "no_tools":
            pred_cur = (
                head(h, exposures=exposures)
                if head is not None
                else self.toolbox.estimate_hypothesis(h)
            )

        notes_cur: torch.Tensor | None = context_notes
        prev_diag = torch.zeros((n_assets, 8), device=device, dtype=dtype)
        rho_0: torch.Tensor | None = None
        cf_sign_flip_k0 = torch.zeros((), device=device, dtype=dtype)
        p_flip_acc = torch.zeros(n_assets, device=device, dtype=dtype)
        p_flip_cur = torch.zeros(n_assets, device=device, dtype=dtype)

        step_hiddens: list[torch.Tensor] = []
        step_predictions: list[dict[str, torch.Tensor]] = []
        step_tool_weights: list[torch.Tensor] = []
        step_entropies: list[torch.Tensor] = []
        halt_probs_per_step: list[torch.Tensor] = []
        monotonic_accepts: list[torch.Tensor] = []
        decor_steps: list[torch.Tensor] = []
        corr_max_steps: list[torch.Tensor] = []
        corr_mean_steps: list[torch.Tensor] = []
        conflict_steps: list[torch.Tensor] = []
        novelty_steps: list[torch.Tensor] = []
        last_tool_aux: dict[str, torch.Tensor] = {}

        off_eye_t = 1.0 - torch.eye(self.n_tools, device=device, dtype=calc_dtype)
        n_off_t = float(max(self.n_tools * (self.n_tools - 1), 1))

        for step_idx in range(self.thinking_steps):
            tool_weights, router_ent = self.tool_router(
                ctx,
                step_idx,
                h_state=h if self.is_v2 else None,
                prev_diagnostics=prev_diag if self.is_v2 else None,
            )
            step_tool_weights.append(tool_weights)
            step_entropies.append(router_ent)

            if self.tool_mode == "no_tools":
                tool_msg = torch.zeros_like(h)
                if rho_0 is None:
                    # Proxy causal de dificultad desde contexto + OOD en modo no_tools
                    vol_p = (
                        ctx[:, CTX_VOL_PCT].to(dtype=calc_dtype).clamp(0.0, 1.0)
                        if ctx.shape[-1] > CTX_VOL_PCT
                        else torch.full((n_assets,), 0.5, device=device, dtype=calc_dtype)
                    )
                    ill_p = (
                        1.0 - ctx[:, CTX_LIQ_PCT].to(dtype=calc_dtype).clamp(0.0, 1.0)
                        if ctx.shape[-1] > CTX_LIQ_PCT
                        else torch.full((n_assets,), 0.5, device=device, dtype=calc_dtype)
                    )
                    wass_p = (
                        torch.sigmoid(ctx[:, CTX_WASS].to(dtype=calc_dtype) - 1.0)
                        if ctx.shape[-1] > CTX_WASS
                        else torch.zeros(n_assets, device=device, dtype=calc_dtype)
                    )
                    ood_p = (
                        torch.log1p(
                            ood_energy.detach().to(device=device, dtype=calc_dtype).clamp_min(0.0)
                        )
                        if ood_energy is not None
                        else torch.zeros(n_assets, device=device, dtype=calc_dtype)
                    )
                    rho_0 = torch.sigmoid(2.0 * (vol_p + ill_p + wass_p + 0.5 * ood_p - 1.5))
                rho_k = rho_0
            else:
                tool_outputs, last_tool_aux = self.toolbox(
                    h,
                    exposures=exposures,
                    rmt_loadings=rmt_loadings,
                    hrr_memory=hrr_memory,
                    hypergraph=hypergraph,
                    pred_k=pred_cur,
                    w_prev=w_prev,
                    regime_key=regime_key,
                    sector_key=sector_key,
                    ood_energy=ood_energy,
                    factor_cache=factor_cache,
                    horizon_mask=horizon_mask,
                    high_snr_context=ctx,
                    mod_cache=mod_cache,
                    cognitive_notes=notes_cur,
                    cf_placebo_mode=cf_placebo_mode,
                )
                tool_msg = torch.einsum("nt,ntd->nd", tool_weights, tool_outputs)
                if self.is_v2 and "next_cognitive_notes" in last_tool_aux:
                    notes_cur = last_tool_aux["next_cognitive_notes"]

                # Decorrelación de salidas entre herramientas (penaliza |cos| > tool_decor_margin)
                u_flat = tool_outputs.to(dtype=calc_dtype).transpose(0, 1).reshape(self.n_tools, -1)
                u_norm = F.normalize(u_flat, dim=-1, eps=1e-6)
                gram_u = (u_norm @ u_norm.T).abs()
                gram_off = gram_u * off_eye_t
                corr_max_steps.append(gram_off.max())
                corr_mean_steps.append(gram_off.sum() / n_off_t)
                decor_steps.append(
                    (F.relu(gram_off - self.tool_decor_margin).pow(2) * off_eye_t).sum() / n_off_t
                )

                if self.is_v2:
                    e_cf_k = last_tool_aux["e_cf"]
                    conf_k = last_tool_aux["conflict_score"]
                    p_flip_cur = last_tool_aux["p_flip"]
                    if step_idx == 0:
                        cf_sign_flip_k0 = p_flip_cur.mean()
                    conflict_steps.append(conf_k.mean())
                    novelty_steps.append(last_tool_aux["scratchpad_novelty_mean"])
                    rho_k = self._compute_difficulty_index(
                        e_struct=last_tool_aux["structural_energy"],
                        e_cf=e_cf_k,
                        conflict=conf_k,
                        sigma_epist=last_tool_aux["epistemic_uncertainty"],
                        vol0=last_tool_aux["hypothesis"]["vol_pred"][:, 0],
                        calc_dtype=calc_dtype,
                    )
                    if rho_0 is None:
                        rho_0 = rho_k
                else:
                    rho_k = self._compute_difficulty_index(
                        e_struct=last_tool_aux["structural_energy"],
                        e_cf=torch.zeros(n_assets, device=device, dtype=dtype),
                        conflict=torch.zeros(n_assets, device=device, dtype=dtype),
                        sigma_epist=last_tool_aux["epistemic_uncertainty"],
                        vol0=last_tool_aux["hypothesis"]["vol_pred"][:, 0],
                        calc_dtype=calc_dtype,
                    )
                    if rho_0 is None:
                        rho_0 = rho_k

            h_cand = self.core_cell(anchor_x, h, tool_msg, cached_weights=cached_weights)
            delta_norm = (h_cand - h).norm(dim=-1) * inv_sqrt_d

            # Consistencia monótona sin desfase: evalúa E_struct (v1) o E_struct + 0.5*E_cf (v2)
            if self.tool_mode == "no_tools":
                e_cand = delta_norm
                if best_energy is None:
                    best_energy = e_cand.detach()
                    worsened = torch.zeros_like(e_cand)
                    accept_k = torch.ones_like(e_cand)
                else:
                    worsened = F.relu(e_cand - best_energy)
                    accept_k = torch.exp(-self.monotonic_penalty * worsened)
                    best_energy = torch.minimum(best_energy, e_cand.detach())
                if (not self.training) and enable_early_exit and step_idx > 0:
                    frozen = cum_halt > 0.95
                    accept_k = torch.where(frozen, torch.zeros_like(accept_k), accept_k)
                if head is not None:
                    step_predictions.append(head(h_cand, exposures=exposures))
            else:
                if best_energy is None:
                    e_init = last_tool_aux["structural_energy"]
                    if self.is_v2:
                        e_init = e_init + 0.5 * last_tool_aux["e_cf"]
                    best_energy = e_init.detach()
                pred_cand = (
                    head(h_cand, exposures=exposures)
                    if head is not None
                    else self.toolbox.estimate_hypothesis(h_cand)
                )
                _, e_struct_cand = self.toolbox.verify_structure(
                    pred_cand,
                    exposures=exposures,
                    ood_energy=last_tool_aux.get("ood_energy"),
                    factor_cache=factor_cache,
                    horizon_mask=horizon_mask,
                )
                if self.is_v2 and self.toolbox.cf_sim is not None:
                    cf_cand = self.toolbox.cf_sim(
                        pred_cand["alpha"][:, 0],
                        pred_cand["vol_pred"][:, 0],
                        high_snr_context=ctx,
                        exposures=exposures,
                        rmt_loadings=rmt_loadings,
                        w_opt=last_tool_aux.get("portfolio_weights"),
                        quantiles=pred_cand.get("quantiles"),
                        conflict=last_tool_aux.get("conflict_score"),
                        factor_cache=factor_cache,
                        placebo_mode=cf_placebo_mode,
                    )
                    e_cand = e_struct_cand + 0.5 * cf_cand["e_cf"]
                    p_flip_cand = cf_cand["p_flip"]
                else:
                    e_cand = e_struct_cand
                    p_flip_cand = p_flip_cur

                worsened = F.relu(e_cand - best_energy)
                accept_k = torch.exp(-self.monotonic_penalty * worsened)
                best_energy = torch.minimum(best_energy, e_cand.detach())
                # Congelación zero-sync en evaluación cuando masa acumulada de parada > 0.95
                if (not self.training) and enable_early_exit and step_idx > 0:
                    frozen = cum_halt > 0.95
                    accept_k = torch.where(frozen, torch.zeros_like(accept_k), accept_k)

                # Reutilizar `pred_cand` (combinado por `accept_k`) -> reduce `head()` a K+1!
                assert pred_cur is not None
                pred_eff = _blend_head_predictions(pred_cand, pred_cur, accept_k)
                step_predictions.append(pred_eff)
                pred_cur = pred_eff
                p_flip_eff = accept_k * p_flip_cand + (1.0 - accept_k) * p_flip_cur

                # Actualizar estado diagnóstico de lazo cerrado para el paso k+1
                if self.is_v2:
                    vol_ref = pred_eff["vol_pred"][:, 0].detach().clamp_min(1e-4)
                    ep_ratio = (
                        last_tool_aux["epistemic_uncertainty"] / vol_ref
                    ).clamp_min(0.0)
                    prev_diag = torch.stack(
                        [
                            torch.log1p(e_struct_cand.clamp_min(0.0)),
                            torch.log1p(
                                (e_cand - e_struct_cand).clamp_min(0.0) * 2.0
                            ),
                            p_flip_eff.clamp(0.0, 1.0),
                            last_tool_aux["conflict_score"].clamp(0.0, 1.0),
                            last_tool_aux["hyp_contradiction"].clamp(0.0, 1.0),
                            torch.log1p(ep_ratio),
                            torch.log1p(last_tool_aux["outlier_mag"].clamp_min(0.0)),
                            torch.log1p(delta_norm.clamp_min(0.0)),
                        ],
                        dim=-1,
                    )

            h_eff = accept_k.unsqueeze(-1) * h_cand + (1.0 - accept_k.unsqueeze(-1)) * h
            monotonic_accepts.append(accept_k)

            # Parada adaptativa aprendida (`halt_gate`) con tope duro en `step_idx == K - 1`
            step_e = self.tool_router.get_step_embedding(step_idx, dtype=dtype)
            lambda_k = self.halt_gate(
                h_eff,
                delta_norm,
                e_cand,
                worsened,
                ctx,
                step_e,
                difficulty=rho_k if self.is_v2 else None,
            )
            if step_idx + 1 < self.thinking_steps:
                p_k = remaining * lambda_k
                remaining = remaining * (1.0 - lambda_k)
            else:
                p_k = remaining
            cum_halt = cum_halt + p_k

            h_acc = h_acc + p_k.unsqueeze(-1) * h_eff
            expected_steps = expected_steps + float(step_idx + 1) * p_k
            if self.is_v2 and self.tool_mode != "no_tools":
                p_flip_acc = p_flip_acc + p_k * p_flip_eff
            halt_probs_per_step.append(p_k)
            step_hiddens.append(h_eff)
            h = h_eff

        halting_dist = torch.stack(halt_probs_per_step, dim=-1)  # [N, K]
        hd_c = halting_dist.to(dtype=calc_dtype)
        assert rho_0 is not None

        if self.is_v2:
            # Prior geométrico adaptado por activo según dificultad desacoplada rho_{i,0}
            lam_prior = (0.80 - 0.65 * rho_0.detach()).clamp(0.15, 0.80)
            prior_cols: list[torch.Tensor] = []
            rem_pr = torch.ones(n_assets, device=device, dtype=calc_dtype)
            for sk in range(self.thinking_steps):
                if sk + 1 < self.thinking_steps:
                    pk_pr = rem_pr * lam_prior
                    rem_pr = rem_pr * (1.0 - lam_prior)
                else:
                    pk_pr = rem_pr
                prior_cols.append(pk_pr.clamp_min(1e-6))
            prior_c = torch.stack(prior_cols, dim=-1)
            prior_c = prior_c / prior_c.sum(dim=-1, keepdim=True)
        else:
            prior_c = self.halt_prior.to(device=device, dtype=calc_dtype).unsqueeze(0)

        ponder_kl = (
            (hd_c * (torch.log(hd_c.clamp_min(1e-8)) - torch.log(prior_c.clamp_min(1e-8))))
            .sum(dim=-1)
            .mean()
            .clamp_min(0.0)
            .to(dtype=dtype)
        )
        weighted_ponder_loss = ponder_kl * self.ponder_weight

        if probe_targets is not None and head is None and step_predictions:
            probe_aux_loss = self.compute_probe_aux_loss(step_predictions, probe_targets)
        else:
            probe_aux_loss = torch.zeros((), device=device, dtype=dtype)

        next_state: dict[str, object] = {} if state is None else dict(state)
        if notes_cur is not None:
            next_state["cognitive_notes"] = notes_cur.detach()

        stacked_weights = torch.stack(step_tool_weights, dim=0)  # [K, N, T]
        tool_weights_by_step = stacked_weights.mean(dim=1)  # [K, T]
        tool_usage_distribution = stacked_weights.mean(dim=(0, 1))  # [T]
        tool_entropy_per_asset = torch.stack(step_entropies, dim=0).mean(dim=0)  # [N]
        tool_entropy = tool_entropy_per_asset.mean()
        monotonic_accept_rate = torch.stack(monotonic_accepts, dim=0).mean()

        # Trío limpio anti-colapso (DeepSeek bias + batch-mean balance + decorrelación > 0.85)
        if self.tool_mode in ("soft", "sparsemax"):
            p_bar = tool_usage_distribution.to(dtype=calc_dtype)
            h_bar = -(p_bar * torch.log(p_bar.clamp_min(1e-8))).sum()
            ent_floor = F.relu(0.55 * math.log(float(self.n_tools)) - h_bar)
            cv_sq = (
                (p_bar - (1.0 / float(self.n_tools))).pow(2).mean()
                * float(self.n_tools * self.n_tools)
            )
            tool_balance_loss = (ent_floor + 0.5 * cv_sq).to(dtype=dtype)
        else:
            tool_balance_loss = torch.zeros((), device=device, dtype=dtype)

        if decor_steps:
            tool_decor_loss = torch.stack(decor_steps).mean().to(dtype=dtype)
            tool_corr_max = torch.stack(corr_max_steps).max().to(dtype=dtype)
            tool_corr_mean = torch.stack(corr_mean_steps).mean().to(dtype=dtype)
        else:
            tool_decor_loss = torch.zeros((), device=device, dtype=dtype)
            tool_corr_max = torch.zeros((), device=device, dtype=dtype)
            tool_corr_mean = torch.zeros((), device=device, dtype=dtype)

        tool_aux_loss = (
            self.tool_balance_weight * tool_balance_loss + self.tool_decor_weight * tool_decor_loss
            if self.tool_mode != "no_tools"
            else torch.zeros((), device=device, dtype=dtype)
        )

        # Métricas GPU 0D/1D para `telemetry.py` (cero host-syncs)
        if self.tool_mode == "no_tools":
            tool_load_max_min_ratio = torch.ones((), device=device, dtype=dtype)
            tool_active_count = torch.zeros((), device=device, dtype=dtype)
        else:
            p_bar_det = tool_usage_distribution.detach().to(dtype=calc_dtype)
            tool_load_max_min_ratio = (
                p_bar_det.max() / p_bar_det.min().clamp_min(1e-6)
            ).to(dtype=dtype)
            tool_active_count = (
                (stacked_weights > 0.05).to(dtype=calc_dtype).sum(dim=-1).mean().to(dtype=dtype)
            )

        # Cuartiles fácil (q25) vs difícil (q75) de pasos esperados (FP32 sort, 0 syncs, BF16-safe)
        rho_sorted = torch.sort(rho_0.detach()).values
        q25_thr = rho_sorted[(n_assets - 1) // 4]
        q75_thr = rho_sorted[((n_assets - 1) * 3) // 4]
        easy_m = (rho_0.detach() <= q25_thr).to(dtype=calc_dtype)
        hard_m = (rho_0.detach() >= q75_thr).to(dtype=calc_dtype)
        es_c = expected_steps.to(dtype=calc_dtype)
        expected_steps_easy_q25 = ((es_c * easy_m).sum() / easy_m.sum().clamp_min(1.0)).to(
            dtype=dtype
        )
        expected_steps_hard_q75 = ((es_c * hard_m).sum() / hard_m.sum().clamp_min(1.0)).to(
            dtype=dtype
        )
        expected_steps_mean = expected_steps.mean()

        cf_sign_flip_final = (
            p_flip_acc.mean()
            if (self.is_v2 and self.tool_mode != "no_tools")
            else torch.zeros((), device=device, dtype=dtype)
        )
        conflict_mean = (
            torch.stack(conflict_steps).mean().to(dtype=dtype)
            if conflict_steps
            else torch.zeros((), device=device, dtype=dtype)
        )
        scratchpad_novelty_mean = (
            torch.stack(novelty_steps).mean().to(dtype=dtype)
            if novelty_steps
            else torch.zeros((), device=device, dtype=dtype)
        )

        diagnostics: dict[str, object] = dict(
            toolbox_version=self.toolbox_version,
            tool_names=self.tool_names,
            tool_mode=self.tool_mode,
            thinking_steps=self.thinking_steps,
            tool_usage_distribution=tool_usage_distribution,
            tool_weights_mean=tool_usage_distribution,
            tool_weights_by_step=tool_weights_by_step,
            tool_weights_per_step=step_tool_weights,
            tool_load_max_min_ratio=tool_load_max_min_ratio,
            tool_active_count=tool_active_count,
            tool_corr_max=tool_corr_max,
            tool_corr_mean=tool_corr_mean,
            tool_entropy=tool_entropy,
            tool_entropy_per_asset=tool_entropy_per_asset,
            tool_balance_loss=tool_balance_loss,
            tool_decor_loss=tool_decor_loss,
            tool_aux_loss=tool_aux_loss,
            router_temperature=self.tool_router.current_temperature().to(dtype=dtype),
            pruned_context_ratio=pruned_ratio,
            lipschitz_upper_bound=self.core_cell.lipschitz_upper_bound(),
            halting_distribution=halting_dist,
            expected_thinking_steps=expected_steps_mean,
            expected_steps_mean=expected_steps_mean,
            expected_steps_easy_q25=expected_steps_easy_q25,
            expected_steps_hard_q75=expected_steps_hard_q75,
            expected_steps_per_asset=expected_steps,
            difficulty_index=rho_0.to(dtype=dtype),
            cf_sign_flip_k0=cf_sign_flip_k0,
            cf_sign_flip_final=cf_sign_flip_final,
            conflict_mean=conflict_mean,
            scratchpad_novelty_mean=scratchpad_novelty_mean,
            ponder_kl_loss=ponder_kl,
            weighted_ponder_loss=weighted_ponder_loss,
            ponder_weight=self.ponder_weight,
            probe_aux_loss=probe_aux_loss,
            step_predictions=step_predictions,
            monotonic_accept_rate=monotonic_accept_rate,
            monotonic_accepts_per_step=monotonic_accepts,
            salience_weights=salience_weights,
            context_notes=notes_cur,
            state=next_state,
            last_tool_aux=last_tool_aux,
        )
        return h_acc, step_hiddens, diagnostics


__all__ = [
    "DEFAULT_PONDER_WEIGHT",
    "DEFAULT_THINKING_STEPS",
    "MAX_THINKING_STEPS",
    "TOOLBOX_VERSIONS",
    "TOOL_NAMES",
    "TOOL_NAMES_V2",
    "TOOL_ROUTER_MODES",
    "AdaptiveHaltGate",
    "ContextEngineerGPU",
    "DifferentiableQuantToolbox",
    "TitansCognitiveCore",
    "ToolRouter",
]
