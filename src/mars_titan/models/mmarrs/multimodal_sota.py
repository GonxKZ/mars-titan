"""Fusión Multimodal con Detección de Conflicto, Alineación SSL y Cabeza Skew Student-t.

Componentes SOTA 2025–2026 nativos de GPU (cero sincronizaciones CPU↔GPU `.item()`/`.cpu()`/
`.numpy()`, compatibles con FP32 estricto, BF16 y FP16 en NVIDIA RTX PRO 6000 Blackwell `sm_120`):

  1. `CrossModalConflictFusion`:
     - Consume las 5 modalidades (`prices`, `fundamentals`, `macro`, `news`, `charts` en
       `[N, 5, D]`), `modality_mask [N, 5]`, `physics_emb [N, D]` y `modality_age [N, 5]`.
     - Ejecuta atención cruzada multimodal consciente de frescura mediante
       `F.scaled_dot_product_attention` (FlashAttention/cuDNN en Blackwell) aislando bit a bit
       las modalidades ausentes con `torch.where` previo y sesgo `-inf` sobre claves ausentes
       (con el token ancla `physics_emb` siempre activo para evitar filas totalmente `-inf`).
     - Calcula el conflicto explícito inter-modal (`conflict_score [N, 1]`,
       `conflict_per_pair [N, 10]`, `modality_votes [N, 5]`, `group_conflict [N, 1]`) combinando
       varianza direccional ponderada por precisión/frescura y desacuerdo angular por pares.

  2. `SelfSupervisedMultimodalAlignmentLoss`:
     - (a) Reconstrucción Enmascarada de Modalidades (leave-one-out estricto vía SDPA con máscara
       diagonal bloqueada: predice cada modalidad presente únicamente desde las restantes
       modalidades presentes, pérdida coseno + MSE sin fuga propia).
     - (b) Alineación Cross-Modal VICReg + Barlow Twins entre la vista de mercado
       (`prices + charts`) y la vista transversal variable por activo (`fundamentals + news`,
       excluyendo `macro` constante por sesión), penalizando colapso de varianza y redundancia
       dimensional fuera de la diagonal sin ramas dependientes de host, omitiendo `view_b` con
       `< 2` activos válidos y escalando el total mediante `ssl_weight` (defecto `0.02`).

  3. `SkewStudentTDistributionHead` + `skew_student_t_nll` + `skew_student_t_var_cvar`:
     - Distribución paramétrica asimétrica de colas pesadas Two-Piece Skew Student-t
       (Hansen 1994 / Fernández-Steel 1998) con parámetros `(loc, scale > 0, df > 2.2,
       skew in (-0.95, 0.95))`.
     - NLL analítico exacto, VaR y CVaR (Expected Shortfall) analíticos en 1 % y 5 % mediante
       inversa de Student-t diferenciable en GPU (Hill 1970, Algoritmo 396), 9 cuantiles
       estrictamente crecientes sin cruces y descomposición exacta libre de Monte Carlo en
       varianza aleatórica y varianza epistémica vía Normal-Inverse-Gamma Evidencial acoplado
       con ensamble multi-cabeza (`aleatoric_var + epistemic_var == total_var`).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import DEFAULT_HORIZONS, DEFAULT_QUANTILES

MODALITY_NAMES: tuple[str, ...] = ("prices", "fundamentals", "macro", "news", "charts")
MODALITY_PAIR_INDICES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)
MARKET_VIEW_INDICES: tuple[int, ...] = (0, 4)  # prices + charts
CONTEXT_VIEW_INDICES: tuple[int, ...] = (1, 2, 3)  # fundamentals + macro + news
ASSET_CONTEXT_VIEW_INDICES: tuple[int, ...] = (1, 3)  # fundamentals + news (asset-varying)


def _inv_softplus(value: float) -> float:
    """Inversa numéricamente estable de `softplus` para inicialización positiva."""
    v = max(float(value), 1e-5)
    if v > 20.0:
        return v
    return math.log(math.expm1(v))


def _safe_l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalización L2 suave con gradiente finito incluso en `x == 0`."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _log_student_t_const(df: torch.Tensor) -> torch.Tensor:
    """Log-constante normalizadora `log(Gamma((v+1)/2) / (sqrt(pi*v)*Gamma(v/2)))` en FP32."""
    df32 = df.float().clamp_min(2.05)
    return (
        torch.lgamma(0.5 * (df32 + 1.0))
        - torch.lgamma(0.5 * df32)
        - 0.5 * torch.log(math.pi * df32)
    )


def _student_t_first_abs_moment(df: torch.Tensor) -> torch.Tensor:
    """Primer momento absoluto exacto `m1(v) = E[|T_v|] = 2*v*c(v)/(v-1)` en FP32."""
    df32 = df.float().clamp_min(2.05)
    c = torch.exp(_log_student_t_const(df32))
    return (2.0 * df32 * c) / (df32 - 1.0)


def _skew_variance_factor(df: torch.Tensor, skew: torch.Tensor) -> torch.Tensor:
    """Factor exacto `k(v, lambda) = Var(Z_{v,lambda}) / Var(T_v) > 0` en FP32."""
    df32 = df.float().clamp_min(2.05)
    sk32 = skew.float().clamp(-0.99, 0.99)
    m1 = _student_t_first_abs_moment(df32)
    sk2 = sk32 * sk32
    k = (1.0 + 3.0 * sk2) - 4.0 * sk2 * (m1 * m1) * ((df32 - 2.0) / df32)
    return k.clamp_min(1e-4)


def student_t_ppf(p: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """Cuantil analítico diferenciable en GPU de Student-t estándar (Hill 1970, Algoritmo 396)."""
    p32 = p.float().clamp(1e-6, 1.0 - 1e-6)
    df32 = df.float().clamp_min(2.05)
    sign = torch.where(p32 < 0.5, -torch.ones_like(p32), torch.ones_like(p32))
    q = (2.0 * torch.minimum(p32, 1.0 - p32)).clamp(min=1e-6, max=1.0 - 1e-7)

    a = 1.0 / (df32 - 0.5)
    b = 48.0 / (a * a)
    c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36
    d = ((94.5 / (b + c) - 3.0) / b + 1.0) * torch.sqrt(a * (math.pi / 2.0)) * df32
    x = d * q
    y = x.pow(2.0 / df32)

    # Rama 1: expansión asintótica basada en cuantil normal (torch.erfinv nativo en GPU)
    z = math.sqrt(2.0) * torch.erfinv((1.0 - q).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    z2 = z * z
    c_mod = c + torch.where(df32 < 5.0, 0.3 * (df32 - 4.5) * (z + 0.6), torch.zeros_like(z))
    c_mod = (((0.05 * d * z - 5.0) * z - 7.0) * z - 2.0) * z + b + c_mod
    c_mod_safe = torch.where(
        c_mod.abs() < 1e-4,
        torch.where(c_mod >= 0.0, torch.full_like(c_mod, 1e-4), torch.full_like(c_mod, -1e-4)),
        c_mod,
    )
    y1_lin = (
        ((((0.4 * z2 + 6.3) * z2 + 36.0) * z2 + 94.5) / c_mod_safe - z2 - 3.0) / b + 1.0
    ) * z
    y1_lin = y1_lin.clamp(-1e4, 1e4)
    y1 = a * y1_lin * y1_lin
    y1_clamped = y1.clamp(min=0.0, max=40.0)
    y1 = torch.where(
        y1_clamped > 0.002,
        torch.expm1(y1_clamped),
        0.5 * y1_clamped * y1_clamped + y1_clamped,
    )

    # Rama 2: cola extrema (entradas saneadas para evitar NaN en gradiente de la rama inactiva)
    y_safe = y.clamp(min=1e-8, max=0.5)
    inner = ((df32 + 6.0) / (df32 * y_safe) - 0.089 * d - 0.822) * (df32 + 2.0) * 3.0
    inner_safe = torch.where(
        inner.abs() < 1e-4,
        torch.where(inner >= 0.0, torch.full_like(inner, 1e-4), torch.full_like(inner, -1e-4)),
        inner,
    )
    y2 = (
        ((1.0 / inner_safe + 0.5 / (df32 + 4.0)) * y_safe - 1.0)
        * (df32 + 1.0)
        / (df32 + 2.0)
        + 1.0 / y_safe
    )

    y_final = torch.where(y > 0.05 + a, y1, y2)
    t_abs = torch.sqrt(y_final.clamp_min(0.0) * df32 + 1e-12)
    is_center = (p32 - 0.5).abs() < 1e-7
    return torch.where(is_center, torch.zeros_like(t_abs), sign * t_abs)


def skew_student_t_nll(
    target: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    df: torch.Tensor,
    skew: torch.Tensor,
    *,
    reduction: str = "mean",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-verosimilitud negativa analítica exacta de Two-Piece Skew Student-t en FP32."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction inválido {reduction!r}; válidos: ('mean', 'sum', 'none')")

    y32 = target.to(device=loc.device, dtype=torch.float32)
    loc32 = loc.float()
    if y32.ndim == loc32.ndim - 1 and y32.shape == loc32.shape[:-1]:
        y32 = y32.unsqueeze(-1).expand_as(loc32)
    sc32 = scale.float().clamp_min(1e-6)
    df32 = df.float().clamp_min(2.05)
    sk32 = skew.float().clamp(-0.99, 0.99)

    # P0-3: intersectar máscara con finitud y sanear `y32` ANTES de calcular z/nll (evita NaN*0=NaN)
    finite_y = torch.isfinite(y32)
    if mask is not None:
        valid = (mask.to(device=loc32.device) > 0.5).expand_as(loc32) & finite_y
    else:
        valid = finite_y
    y32_safe = torch.where(valid, y32, loc32.detach())

    z = (y32_safe - loc32) / sc32
    piece_scale = torch.where(z >= 0.0, 1.0 + sk32, 1.0 - sk32)
    u = z / piece_scale
    log_c = _log_student_t_const(df32)
    nll = -log_c + torch.log(sc32) + 0.5 * (df32 + 1.0) * torch.log1p((u * u) / df32)

    m = valid.to(dtype=torch.float32)
    if reduction == "none":
        return nll * m
    if reduction == "sum":
        return (nll * m).sum()
    return (nll * m).sum() / m.sum().clamp_min(1.0)


def skew_student_t_var_cvar(
    loc: torch.Tensor,
    scale: torch.Tensor,
    df: torch.Tensor,
    skew: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VaR y CVaR (Expected Shortfall) analíticos exactos a nivel de cola `alpha in (0, 1)`."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha debe pertenecer a (0, 1), recibido: {alpha}")

    orig_dtype = loc.dtype
    loc32 = loc.float()
    sc32 = scale.float().clamp_min(1e-6)
    df32 = df.float().clamp_min(2.05)
    sk32 = skew.float().clamp(-0.99, 0.99)

    c = torch.exp(_log_student_t_const(df32))
    m1 = (2.0 * df32 * c) / (df32 - 1.0)

    a_t = torch.full_like(sk32, float(alpha))
    split = 0.5 * (1.0 - sk32)
    left_branch = a_t < split

    p1 = (a_t / (1.0 - sk32)).clamp(1e-6, 0.5)
    p2 = ((a_t + sk32) / (1.0 + sk32)).clamp(0.5, 1.0 - 1e-6)
    q1 = student_t_ppf(p1, df32)
    q2 = student_t_ppf(p2, df32)
    q_z = torch.where(left_branch, (1.0 - sk32) * q1, (1.0 + sk32) * q2)

    t_q1 = c * (1.0 + (q1 * q1) / df32).pow(-0.5 * (df32 + 1.0))
    t_q2 = c * (1.0 + (q2 * q2) / df32).pow(-0.5 * (df32 + 1.0))
    cvar_z_left = -((1.0 - sk32).pow(2) / a_t) * ((df32 + q1 * q1) / (df32 - 1.0)) * t_q1
    cvar_z_right = (
        2.0 * sk32 * m1 - (1.0 + sk32).pow(2) * ((df32 + q2 * q2) / (df32 - 1.0)) * t_q2
    ) / a_t
    cvar_z = torch.where(left_branch, cvar_z_left, cvar_z_right)

    var_y = (loc32 + sc32 * q_z).to(dtype=orig_dtype)
    cvar_y = (loc32 + sc32 * cvar_z).to(dtype=orig_dtype)
    return var_y, cvar_y


def skew_student_t_quantiles(
    loc: torch.Tensor,
    scale: torch.Tensor,
    df: torch.Tensor,
    skew: torch.Tensor,
    quantile_levels: tuple[float, ...] | torch.Tensor = DEFAULT_QUANTILES,
) -> torch.Tensor:
    """Cuantiles analíticos estrictamente crecientes `[..., Q]` de Two-Piece Skew Student-t."""
    orig_dtype = loc.dtype
    loc32 = loc.float().unsqueeze(-1)
    sc32 = scale.float().clamp_min(1e-6).unsqueeze(-1)
    df32 = df.float().clamp_min(2.05).unsqueeze(-1)
    sk32 = skew.float().clamp(-0.99, 0.99).unsqueeze(-1)

    if isinstance(quantile_levels, torch.Tensor):
        taus = quantile_levels.to(device=loc.device, dtype=torch.float32).reshape(-1)
    else:
        taus = torch.tensor(quantile_levels, device=loc.device, dtype=torch.float32).reshape(-1)
    view_shape = (1,) * loc.ndim + (taus.numel(),)
    taus = taus.view(view_shape)

    split = 0.5 * (1.0 - sk32)
    left_branch = taus < split
    p1 = (taus / (1.0 - sk32)).clamp(1e-6, 0.5)
    p2 = ((taus + sk32) / (1.0 + sk32)).clamp(0.5, 1.0 - 1e-6)
    q1 = student_t_ppf(p1, df32.expand_as(p1))
    q2 = student_t_ppf(p2, df32.expand_as(p2))
    q_z = torch.where(left_branch, (1.0 - sk32) * q1, (1.0 + sk32) * q2)
    return (loc32 + sc32 * q_z).to(dtype=orig_dtype)


class CrossModalConflictFusion(nn.Module):
    """Fusión multimodal consciente de frescura con SDPA nativo y detector de conflicto."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_modalities: int = 5,
        n_heads: int = 4,
        stale_halflife: float = 5.0,
        dropout: float = 0.0,
        conflict_attenuation: float = 0.5,
        market_indices: tuple[int, ...] = MARKET_VIEW_INDICES,
        context_indices: tuple[int, ...] = CONTEXT_VIEW_INDICES,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim >= 8 es obligatorio")
        if n_modalities < 2:
            raise ValueError("n_modalities >= 2 es obligatorio para detectar conflicto inter-modal")
        if n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError("n_heads >= 1 y hidden_dim divisible por n_heads son obligatorios")
        if stale_halflife <= 0.0:
            raise ValueError("stale_halflife debe ser estrictamente positivo")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout debe pertenecer a [0, 1)")
        if not (0.0 <= conflict_attenuation <= 1.0):
            raise ValueError("conflict_attenuation debe pertenecer a [0, 1]")

        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = float(dropout)
        self.conflict_attenuation = float(conflict_attenuation)

        # Pares no ordenados (i, j) con 0 <= i < j < M (10 pares para M=5)
        pair_i = [i for i in range(n_modalities) for j in range(i + 1, n_modalities)]
        pair_j = [j for i in range(n_modalities) for j in range(i + 1, n_modalities)]
        self.n_pairs = len(pair_i)
        self.register_buffer("pair_i", torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(pair_j, dtype=torch.long))

        # Máscaras de grupos de dominio (mercado/técnico vs fundamentales/macro/noticias)
        mkt_idx = tuple(i for i in market_indices if 0 <= i < n_modalities)
        ctx_idx = tuple(i for i in context_indices if 0 <= i < n_modalities)
        mkt_vec = torch.zeros(n_modalities, dtype=torch.float32)
        ctx_vec = torch.zeros(n_modalities, dtype=torch.float32)
        if mkt_idx:
            mkt_vec[list(mkt_idx)] = 1.0
        if ctx_idx:
            ctx_vec[list(ctx_idx)] = 1.0
        self.register_buffer("market_group_mask", mkt_vec.view(1, n_modalities))
        self.register_buffer("context_group_mask", ctx_vec.view(1, n_modalities))

        # Decaimiento de frescura aprendible por modalidad y precisión base
        init_rate = math.log(2.0) / float(stale_halflife)
        self.age_decay_raw = nn.Parameter(
            torch.full((1, n_modalities), _inv_softplus(init_rate), dtype=torch.float32)
        )
        self.modality_precision_raw = nn.Parameter(
            torch.full((1, n_modalities), _inv_softplus(1.0), dtype=torch.float32)
        )

        # Rama de conflicto direccional compartida (ortogonal, sin sesgo para antisimetría exacta)
        self.dir_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.dir_proj.weight)
        self.vote_proj = nn.Linear(hidden_dim, 1, bias=False)
        # ||dir_unit||_2 == 1 => std=1.0 produce varianza unitaria independiente de hidden_dim
        nn.init.normal_(self.vote_proj.weight, std=1.0)

        # Embeddings de tipo de modalidad + atención cruzada SDPA (ancla física + M modalidades)
        self.mod_type_embed = nn.Parameter(
            torch.randn(1, n_modalities, hidden_dim) * (0.02 / math.sqrt(float(hidden_dim)))
        )
        self.phys_norm = nn.LayerNorm(hidden_dim)
        self.tok_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Bloque residual SwiGLU compacto sobre tokens atendidos
        ffn_dim = hidden_dim * 2
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_gate = nn.Linear(hidden_dim, ffn_dim)
        self.ffn_up = nn.Linear(hidden_dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, hidden_dim)

        # Pooling consciente de frescura + compuerta modulada por conflicto
        self.salience_proj = nn.Linear(hidden_dim, 1)
        self.conflict_gate_proj = nn.Linear(2 * hidden_dim + 1, hidden_dim)
        self.conflict_embed = nn.Linear(1, hidden_dim)
        self.fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        mod_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        physics_emb: torch.Tensor,
        modality_age: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Fusiona modalidades con SDPA nativo y retorna `(fused_h, conflict_score, diag)`."""
        if mod_tokens.ndim != 3 or mod_tokens.shape[1] != self.n_modalities:
            raise ValueError(
                f"mod_tokens debe tener forma [N, {self.n_modalities}, {self.hidden_dim}]"
            )
        n_assets = mod_tokens.shape[0]
        device, orig_dtype = mod_tokens.device, mod_tokens.dtype

        # 1) Aislamiento bit a bit de modalidades ausentes ANTES de cualquier cálculo
        mask_bool = modality_mask.to(device=device) > 0.5  # [N, M]
        mask_f32 = mask_bool.to(dtype=torch.float32)
        zeros_tok = torch.zeros_like(mod_tokens)
        clean_tokens = torch.where(mask_bool.unsqueeze(-1), mod_tokens, zeros_tok)
        phys = physics_emb.to(device=device, dtype=orig_dtype)

        zeros_m32 = torch.zeros((n_assets, self.n_modalities), device=device, dtype=torch.float32)
        if modality_age is None:
            age_f32 = zeros_m32
        else:
            age_raw = modality_age.to(device=device, dtype=torch.float32)
            age_f32 = torch.where(mask_bool, age_raw, zeros_m32).clamp_min(0.0)

        # 2) Frescura exponencial y precisión por modalidad en FP32
        decay_rate = F.softplus(self.age_decay_raw.to(device=device, dtype=torch.float32)) + 1e-4
        age_decay = (decay_rate * age_f32).clamp(max=20.0)
        freshness = torch.where(mask_bool, torch.exp(-age_decay), zeros_m32)  # [N, M]
        log_fresh = torch.where(mask_bool, -age_decay, zeros_m32)

        base_prec = F.softplus(
            self.modality_precision_raw.to(device=device, dtype=torch.float32)
        ) + 1e-3
        prec = torch.where(mask_bool, base_prec * freshness, zeros_m32)  # [N, M]

        # 3) Conflicto inter-modal explícito (votos direccionales + desacuerdo angular por pares)
        dir_raw = self.dir_proj(clean_tokens).float()
        dir_unit = torch.where(
            mask_bool.unsqueeze(-1),
            _safe_l2_normalize(dir_raw),
            torch.zeros_like(dir_raw),
        )
        raw_votes = self.vote_proj(dir_unit.to(dtype=orig_dtype)).squeeze(-1).float()
        modality_votes = torch.where(mask_bool, torch.tanh(raw_votes), zeros_m32)  # [N, M]

        present_count = mask_f32.sum(dim=-1, keepdim=True)  # [N, 1]
        has_two = (present_count >= 1.5).to(torch.float32)

        prec_sum = prec.sum(dim=-1, keepdim=True)
        prec_w = torch.where(mask_bool, prec / prec_sum.clamp_min(1e-6), zeros_m32)
        mean_vote = (prec_w * modality_votes).sum(dim=-1, keepdim=True)
        vote_var = (prec_w * (modality_votes - mean_vote).pow(2)).sum(dim=-1, keepdim=True)
        vote_var = vote_var * has_two

        # Desacuerdo por cada uno de los C(M, 2) = 10 pares (unión probabilística angular + voto)
        pi, pj = self.pair_i, self.pair_j
        pair_mask = mask_bool[:, pi] & mask_bool[:, pj]  # [N, P]
        zeros_p = torch.zeros((n_assets, self.n_pairs), device=device, dtype=torch.float32)
        cos_ij = (dir_unit[:, pi, :] * dir_unit[:, pj, :]).sum(dim=-1).clamp(-1.0, 1.0)
        ang_ij = 0.5 * (1.0 - cos_ij)
        vdiff_ij = 0.25 * (modality_votes[:, pi] - modality_votes[:, pj]).pow(2)
        pair_union = 1.0 - (1.0 - ang_ij) * (1.0 - vdiff_ij)
        conflict_per_pair = torch.where(pair_mask, pair_union, zeros_p)

        pair_w = torch.where(pair_mask, prec[:, pi] * prec[:, pj], zeros_p)
        pair_w_sum = pair_w.sum(dim=-1, keepdim=True)
        has_pair = (pair_mask.any(dim=-1, keepdim=True)).to(torch.float32)
        pair_conflict = (
            (pair_w * conflict_per_pair).sum(dim=-1, keepdim=True) / pair_w_sum.clamp_min(1e-6)
        ) * has_pair

        conflict_score_f32 = (1.0 - (1.0 - pair_conflict) * (1.0 - vote_var)) * has_two

        # Conflicto de dominio explícito: vista mercado (prices+charts) vs vista contexto
        mkt_w = prec * self.market_group_mask.to(device=device, dtype=torch.float32)
        ctx_w = prec * self.context_group_mask.to(device=device, dtype=torch.float32)
        has_mkt = mkt_w.sum(dim=-1, keepdim=True) > 1e-8
        has_ctx = ctx_w.sum(dim=-1, keepdim=True) > 1e-8
        both_groups = (has_mkt & has_ctx).to(torch.float32)

        mkt_wn = mkt_w / mkt_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ctx_wn = ctx_w / ctx_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_mkt = _safe_l2_normalize(torch.einsum("nm,nmd->nd", mkt_wn, dir_unit))
        dir_ctx = _safe_l2_normalize(torch.einsum("nm,nmd->nd", ctx_wn, dir_unit))
        vote_mkt = (mkt_wn * modality_votes).sum(dim=-1, keepdim=True)
        vote_ctx = (ctx_wn * modality_votes).sum(dim=-1, keepdim=True)
        grp_ang = 0.5 * (1.0 - (dir_mkt * dir_ctx).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0))
        grp_vdiff = 0.25 * (vote_mkt - vote_ctx).pow(2)
        group_conflict = (1.0 - (1.0 - grp_ang) * (1.0 - grp_vdiff)) * both_groups

        # 4) Atención cruzada multimodal consciente de frescura vía F.scaled_dot_product_attention
        phys_tok = self.phys_norm(phys).unsqueeze(1)  # [N, 1, D]
        mod_normed = self.tok_norm(clean_tokens) + self.mod_type_embed.to(
            device=device, dtype=orig_dtype
        )
        mod_in = torch.where(mask_bool.unsqueeze(-1), mod_normed, zeros_tok)
        seq = torch.cat([phys_tok, mod_in], dim=1)  # [N, M + 1, D]
        seq_len = self.n_modalities + 1

        q = self.q_proj(seq).view(n_assets, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(seq).view(n_assets, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(seq).view(n_assets, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Defensa en profundidad: anular claves y valores en posiciones ausentes
        seq_valid = torch.cat(
            [torch.ones((n_assets, 1), device=device, dtype=torch.bool), mask_bool], dim=1
        )  # [N, M + 1]
        kv_valid = seq_valid.view(n_assets, 1, seq_len, 1)
        k = torch.where(kv_valid, k, torch.zeros_like(k))
        v = torch.where(kv_valid, v, torch.zeros_like(v))

        # Máscara aditiva SDPA: clave 0 (física) siempre 0.0; presentes = log_fresh; ausentes = -inf
        neg_inf = torch.full_like(log_fresh, float("-inf"))
        mod_bias = torch.where(mask_bool, log_fresh, neg_inf)
        phys_bias = torch.zeros((n_assets, 1), device=device, dtype=torch.float32)
        sdpa_mask = (
            torch.cat([phys_bias, mod_bias], dim=1)
            .to(dtype=orig_dtype)
            .view(n_assets, 1, 1, seq_len)
        )

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(n_assets, seq_len, self.hidden_dim)
        seq_h = seq + self.out_proj(attn_out)
        ffn_in = self.ffn_norm(seq_h)
        seq_h = seq_h + self.ffn_down(F.silu(self.ffn_gate(ffn_in)) * self.ffn_up(ffn_in))

        phys_att = seq_h[:, 0, :]  # [N, D]
        mod_att = torch.where(mask_bool.unsqueeze(-1), seq_h[:, 1:, :], zeros_tok)

        # 5) Pooling ponderado por frescura y saliencia (en FP32, cero NaN con máscara vacía)
        sal_logits = self.salience_proj(mod_att).squeeze(-1).float() + log_fresh
        sal_masked = torch.where(mask_bool, sal_logits, torch.full_like(sal_logits, -1e4))
        has_any = (present_count > 0.5).to(torch.float32)
        mod_weights_f32 = torch.softmax(sal_masked, dim=-1) * mask_f32 * has_any
        mod_weights_f32 = mod_weights_f32 / mod_weights_f32.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        mod_weights_f32 = mod_weights_f32 * has_any
        mod_weights = mod_weights_f32.to(dtype=orig_dtype)
        mod_pooled = torch.einsum("nm,nmd->nd", mod_weights, mod_att)

        # 6) Fusión final condicionada por conflicto (escala continua, nunca apagado binario)
        conflict_score = conflict_score_f32.to(dtype=orig_dtype)
        gate_in = torch.cat([phys_att, mod_pooled, conflict_score], dim=-1)
        conflict_gate = torch.sigmoid(self.conflict_gate_proj(gate_in))
        attenuated_mod = mod_pooled * (1.0 - self.conflict_attenuation * conflict_score)
        gated_mod = conflict_gate * attenuated_mod + self.conflict_embed(conflict_score)
        fused_h = self.out_norm(self.fusion_proj(torch.cat([gated_mod, phys_att], dim=-1)))

        diagnostics_tensors: dict[str, torch.Tensor] = {
            "conflict_per_pair": conflict_per_pair.to(dtype=orig_dtype),
            "modality_votes": modality_votes.to(dtype=orig_dtype),
            "vote_variance": vote_var.to(dtype=orig_dtype),
            "pairwise_angular_conflict": pair_conflict.to(dtype=orig_dtype),
            "group_conflict": group_conflict.to(dtype=orig_dtype),
            "modality_weights": mod_weights,
            "freshness": freshness.to(dtype=orig_dtype),
            "present_count": present_count.to(dtype=orig_dtype),
            "mean_conflict": conflict_score.mean(),
        }
        return fused_h, conflict_score, diagnostics_tensors


class SelfSupervisedMultimodalAlignmentLoss(nn.Module):
    """Objetivo SSL auxiliar: reconstrucción enmascarada leave-one-out + VICReg/Barlow Twins."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_modalities: int = 5,
        n_heads: int = 4,
        proj_dim: int | None = None,
        view_a_indices: tuple[int, ...] = MARKET_VIEW_INDICES,
        view_b_indices: tuple[int, ...] = ASSET_CONTEXT_VIEW_INDICES,
        recon_cos_weight: float = 1.0,
        recon_mse_weight: float = 1.0,
        vicreg_inv_weight: float = 25.0,
        vicreg_var_weight: float = 25.0,
        vicreg_cov_weight: float = 1.0,
        barlow_weight: float = 1.0,
        barlow_lambda: float = 5e-3,
        var_gamma: float = 1.0,
        recon_weight: float = 1.0,
        align_weight: float = 1.0,
        ssl_weight: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim >= 8 es obligatorio")
        if n_modalities < 2:
            raise ValueError("n_modalities >= 2 es obligatorio")
        if n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError("n_heads >= 1 y hidden_dim divisible por n_heads son obligatorios")
        if not view_a_indices or not view_b_indices:
            raise ValueError("view_a_indices y view_b_indices no pueden estar vacíos")
        if any(i < 0 or i >= n_modalities for i in view_a_indices + view_b_indices):
            raise ValueError("Índices de vistas fuera de rango [0, n_modalities)")
        if var_gamma <= 0.0:
            raise ValueError("var_gamma debe ser estrictamente positivo")
        if ssl_weight < 0.0:
            raise ValueError("ssl_weight debe ser no negativo")
        resolved_proj_dim = proj_dim if proj_dim is not None else hidden_dim
        if resolved_proj_dim < 2:
            raise ValueError("proj_dim >= 2 es obligatorio para regularización de covarianza")

        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.proj_dim = int(resolved_proj_dim)
        self.view_a_indices = tuple(view_a_indices)
        self.view_b_indices = tuple(view_b_indices)

        self.recon_cos_weight = float(recon_cos_weight)
        self.recon_mse_weight = float(recon_mse_weight)
        self.vicreg_inv_weight = float(vicreg_inv_weight)
        self.vicreg_var_weight = float(vicreg_var_weight)
        self.vicreg_cov_weight = float(vicreg_cov_weight)
        self.barlow_weight = float(barlow_weight)
        self.barlow_lambda = float(barlow_lambda)
        self.var_gamma = float(var_gamma)
        self.recon_weight = float(recon_weight)
        self.align_weight = float(align_weight)
        self.ssl_weight = float(ssl_weight)

        # Buffers de índices y máscaras estructurales en dispositivo (cero asignaciones por paso)
        self.register_buffer("view_a_idx", torch.tensor(self.view_a_indices, dtype=torch.long))
        self.register_buffer("view_b_idx", torch.tensor(self.view_b_indices, dtype=torch.long))
        self.register_buffer(
            "loo_eye_mask",
            ~torch.eye(n_modalities, dtype=torch.bool),  # True donde j != m
        )
        self.register_buffer(
            "off_diag_mask",
            1.0 - torch.eye(self.proj_dim, dtype=torch.float32),
        )

        # (a) Reconstructor enmascarado leave-one-out vía SDPA
        scale = 1.0 / math.sqrt(float(hidden_dim))
        self.ctx_norm = nn.LayerNorm(hidden_dim)
        self.ctx_enc = nn.Linear(hidden_dim, hidden_dim)
        self.mod_type_embed = nn.Parameter(torch.randn(1, n_modalities, hidden_dim) * scale * 0.1)
        self.mask_queries = nn.Parameter(torch.randn(1, n_modalities, hidden_dim) * scale)
        self.anchor_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.phys_proj = nn.Linear(hidden_dim, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.recon_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # (b) Proyectores VICReg / Barlow Twins (RMSNorm: soporta N=1 y sin fuga entre muestras)
        self.proj_a = nn.Sequential(
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, self.proj_dim),
            nn.SiLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        self.proj_b = nn.Sequential(
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, self.proj_dim),
            nn.SiLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        # Inicialización preservadora de varianza (ganancia ~sqrt(2) tras SiLU)
        for proj in (self.proj_a, self.proj_b):
            nn.init.kaiming_normal_(proj[1].weight, nonlinearity="relu")
            nn.init.zeros_(proj[1].bias)
            nn.init.orthogonal_(proj[3].weight, gain=math.sqrt(2.0))
            nn.init.zeros_(proj[3].bias)

    def reconstruct_masked_modalities(
        self,
        mod_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        physics_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruye cada modalidad `m` estrictamente desde las demás modalidades `j != m`."""
        n_assets = mod_tokens.shape[0]
        device, orig_dtype = mod_tokens.device, mod_tokens.dtype
        mask_bool = modality_mask.to(device=device) > 0.5
        mask_f32 = mask_bool.to(torch.float32)

        zeros_tok = torch.zeros_like(mod_tokens)
        clean_tokens = torch.where(mask_bool.unsqueeze(-1), mod_tokens, zeros_tok)

        src_raw = self.ctx_enc(self.ctx_norm(clean_tokens)) + self.mod_type_embed.to(
            device=device, dtype=orig_dtype
        )
        src = torch.where(mask_bool.unsqueeze(-1), src_raw, zeros_tok)

        anchor = self.anchor_token.to(device=device, dtype=orig_dtype).expand(n_assets, 1, -1)
        q_in = self.mask_queries.to(device=device, dtype=orig_dtype).expand(n_assets, -1, -1)
        if physics_emb is not None:
            phys_p = self.phys_proj(physics_emb.to(device=device, dtype=orig_dtype)).unsqueeze(1)
            anchor = anchor + phys_p
            q_in = q_in + phys_p

        kv_seq = torch.cat([anchor, src], dim=1)  # [N, M + 1, D]
        kv_len = self.n_modalities + 1

        q = (
            self.q_proj(q_in)
            .view(n_assets, self.n_modalities, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.k_proj(kv_seq).view(n_assets, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_seq).view(n_assets, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Defensa en profundidad: anular claves/valores en posiciones ausentes
        kv_valid = torch.cat(
            [torch.ones((n_assets, 1), device=device, dtype=torch.bool), mask_bool], dim=1
        ).view(n_assets, 1, kv_len, 1)
        k = torch.where(kv_valid, k, torch.zeros_like(k))
        v = torch.where(kv_valid, v, torch.zeros_like(v))

        # Máscara leave-one-out [N, 1, M, M + 1]: ancla 0 siempre activa; j+1 solo si j != m y vivo
        allow_mod = mask_bool.unsqueeze(1) & self.loo_eye_mask.to(device=device).unsqueeze(0)
        mod_shape = (n_assets, self.n_modalities, self.n_modalities)
        mod_bias = torch.where(
            allow_mod,
            torch.zeros(mod_shape, device=device, dtype=orig_dtype),
            torch.full(mod_shape, float("-inf"), device=device, dtype=orig_dtype),
        )
        anchor_bias = torch.zeros((n_assets, self.n_modalities, 1), device=device, dtype=orig_dtype)
        sdpa_mask = torch.cat([anchor_bias, mod_bias], dim=-1).unsqueeze(1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        attn = attn.transpose(1, 2).contiguous().view(n_assets, self.n_modalities, self.hidden_dim)
        pred_tokens = self.recon_decoder(q_in + self.out_proj(attn))

        # Válido estrictamente cuando `m` está presente y existe al menos otra modalidad `j != m`
        other_present = mask_f32.sum(dim=-1, keepdim=True) - mask_f32
        valid_recon = mask_bool & (other_present >= 0.5)
        return pred_tokens, valid_recon

    def forward(
        self,
        mod_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        physics_emb: torch.Tensor | None = None,
        *,
        ssl_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calcula reconstrucción enmascarada + alineación VICReg/Barlow sin sync CPU."""
        if mod_tokens.ndim != 3 or mod_tokens.shape[1] != self.n_modalities:
            raise ValueError(
                f"mod_tokens debe tener forma [N, {self.n_modalities}, {self.hidden_dim}]"
            )
        w_ssl = self.ssl_weight if ssl_weight is None else float(ssl_weight)
        if w_ssl < 0.0:
            raise ValueError("ssl_weight debe ser no negativo")
        device = mod_tokens.device
        mask_bool = modality_mask.to(device=device) > 0.5
        mask_f32 = mask_bool.to(torch.float32)
        clean_tokens = torch.where(
            mask_bool.unsqueeze(-1), mod_tokens, torch.zeros_like(mod_tokens)
        )

        # (a) Reconstrucción enmascarada leave-one-out (targets con stop-grad anti-colapso)
        pred_tokens, valid_recon = self.reconstruct_masked_modalities(
            clean_tokens, mask_f32, physics_emb=physics_emb
        )
        pred_f32 = pred_tokens.float()
        tgt_f32 = clean_tokens.detach().float()

        pred_u = _safe_l2_normalize(pred_f32)
        tgt_u = _safe_l2_normalize(tgt_f32)
        cos_slot = (1.0 - (pred_u * tgt_u).sum(dim=-1).clamp(-1.0, 1.0)).clamp_min(0.0)
        mse_slot = (pred_f32 - tgt_f32).pow(2).mean(dim=-1)

        valid_f32 = valid_recon.to(torch.float32)
        valid_denom = valid_f32.sum().clamp_min(1.0)
        recon_cosine = (cos_slot * valid_f32).sum() / valid_denom
        recon_mse = (mse_slot * valid_f32).sum() / valid_denom
        recon_loss = self.recon_cos_weight * recon_cosine + self.recon_mse_weight * recon_mse
        valid_recon_ratio = valid_f32.mean()

        # (b) Alineación Cross-Modal VICReg + Barlow Twins: vista mercado vs vista contexto
        idx_a = self.view_a_idx.to(device=device)
        idx_b = self.view_b_idx.to(device=device)
        mask_a = mask_f32[:, idx_a]
        mask_b = mask_f32[:, idx_b]
        cnt_a = mask_a.sum(dim=-1, keepdim=True)
        cnt_b = mask_b.sum(dim=-1, keepdim=True)

        pool_a = (
            (clean_tokens[:, idx_a, :].float() * mask_a.unsqueeze(-1)).sum(dim=1)
            / cnt_a.clamp_min(1.0)
        ).to(dtype=mod_tokens.dtype)
        pool_b = (
            (clean_tokens[:, idx_b, :].float() * mask_b.unsqueeze(-1)).sum(dim=1)
            / cnt_b.clamp_min(1.0)
        ).to(dtype=mod_tokens.dtype)

        pair_valid = (cnt_a.squeeze(-1) > 0.5) & (cnt_b.squeeze(-1) > 0.5)
        w = pair_valid.to(torch.float32)  # [N]
        n_v = w.sum()
        has_var = (n_v >= 2.0).to(torch.float32)
        valid_pair_ratio = w.mean()

        z_a = self.proj_a(pool_a).float()  # [N, P]
        z_b = self.proj_b(pool_b).float()  # [N, P]

        # 1) Invarianza entre vistas presentes (omitida con < 2 activos variables válidos)
        sq_diff = (z_a - z_b).pow(2).mean(dim=-1)
        inv_loss = has_var * ((sq_diff * w).sum() / n_v.clamp_min(1.0))

        # 2) Regularización de varianza (anti-colapso dimensional) sobre muestras válidas
        w_col = w.unsqueeze(-1)
        mu_a = (z_a * w_col).sum(dim=0, keepdim=True) / n_v.clamp_min(1.0)
        mu_b = (z_b * w_col).sum(dim=0, keepdim=True) / n_v.clamp_min(1.0)
        zc_a = (z_a - mu_a) * w_col
        zc_b = (z_b - mu_b) * w_col

        denom_cov = (n_v - 1.0).clamp_min(1.0)
        var_a = zc_a.pow(2).sum(dim=0) / denom_cov
        var_b = zc_b.pow(2).sum(dim=0) / denom_cov
        std_a = torch.sqrt(var_a + 1e-4)
        std_b = torch.sqrt(var_b + 1e-4)
        var_loss = has_var * 0.5 * (
            F.relu(self.var_gamma - std_a).mean() + F.relu(self.var_gamma - std_b).mean()
        )

        # 3) Regularización de covarianza fuera de diagonal (VICReg)
        off_diag = self.off_diag_mask.to(device=device, dtype=torch.float32)
        cov_a = (zc_a.T @ zc_a) / denom_cov
        cov_b = (zc_b.T @ zc_b) / denom_cov
        cov_loss = (
            has_var
            * 0.5
            * ((cov_a * off_diag).pow(2).sum() + (cov_b * off_diag).pow(2).sum())
            / float(self.proj_dim)
        )

        # 4) Reducción de redundancia cruzada (Barlow Twins)
        zn_a = zc_a / std_a.unsqueeze(0)
        zn_b = zc_b / std_b.unsqueeze(0)
        cross_corr = (zn_a.T @ zn_b) / denom_cov
        barlow_on = (torch.diagonal(cross_corr) - 1.0).pow(2).mean()
        n_off = float(max(self.proj_dim * (self.proj_dim - 1), 1))
        barlow_off = (cross_corr * off_diag).pow(2).sum() / n_off
        barlow_loss = has_var * (barlow_on + self.barlow_lambda * barlow_off)

        alignment_loss = (
            self.vicreg_inv_weight * inv_loss
            + self.vicreg_var_weight * var_loss
            + self.vicreg_cov_weight * cov_loss
            + self.barlow_weight * barlow_loss
        )
        raw_total = self.recon_weight * recon_loss + self.align_weight * alignment_loss
        ssl_w_t = raw_total.new_full((), w_ssl)
        total_loss = ssl_w_t * raw_total

        return {
            "total": total_loss,
            "raw_total": raw_total,
            "ssl_weight": ssl_w_t,
            "reconstruction": recon_loss,
            "recon_cosine": recon_cosine,
            "recon_mse": recon_mse,
            "alignment": alignment_loss,
            "vicreg_invariance": inv_loss,
            "vicreg_variance": var_loss,
            "vicreg_covariance": cov_loss,
            "barlow_twins": barlow_loss,
            "valid_recon_ratio": valid_recon_ratio,
            "valid_pair_ratio": valid_pair_ratio,
        }


class SkewStudentTDistributionHead(nn.Module):
    """Cabeza paramétrica Skew Student-t + NIG Evidencial con riesgo de cola analítico."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
        n_ensemble: int = 4,
        min_df: float = 2.22,
        max_df: float = 60.0,
        max_skew: float = 0.94,
        min_scale: float = 1e-4,
        evidential_reg_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim >= 8 es obligatorio")
        if not horizons:
            raise ValueError("horizons no puede estar vacío")
        if tuple(sorted(quantile_levels)) != tuple(quantile_levels):
            raise ValueError("quantile_levels debe ser estrictamente creciente")
        if n_ensemble < 1:
            raise ValueError("n_ensemble >= 1 es obligatorio")
        if not (2.2 < min_df < max_df):
            raise ValueError("Se requiere 2.2 < min_df < max_df")
        if not (0.0 < max_skew < 0.95):
            raise ValueError("max_skew debe pertenecer a (0, 0.95)")
        if min_scale <= 0.0:
            raise ValueError("min_scale debe ser estrictamente positivo")

        self.hidden_dim = hidden_dim
        self.horizons = tuple(horizons)
        self.quantile_levels = tuple(quantile_levels)
        self.n_h = len(self.horizons)
        self.n_q = len(self.quantile_levels)
        self.n_ensemble = int(n_ensemble)
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.max_skew = float(max_skew)
        self.min_scale = float(min_scale)
        self.evidential_reg_weight = float(evidential_reg_weight)
        # Buffer en dispositivo para cero copias H2D en forward (seguro para CUDA Graphs)
        self.register_buffer(
            "quantile_taus",
            torch.tensor(self.quantile_levels, dtype=torch.float32),
            persistent=False,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        # Ensamble multi-cabeza de localización (libre de MC, 1 solo GEMM)
        self.ensemble_loc_head = nn.Linear(hidden_dim, self.n_ensemble * self.n_h)
        # Parámetros evidenciales Skew-NIG: (raw_v, raw_df, raw_beta, raw_skew)
        self.evidential_head = nn.Linear(hidden_dim, 4 * self.n_h)
        # Sensibilidad positiva del grado de evidencia `v` al conflicto inter-modal
        self.conflict_evidence_decay = nn.Parameter(
            torch.tensor(_inv_softplus(1.0), dtype=torch.float32)
        )

        # Inicialización en escala financiera realista (vol diaria ~2 %, df ~6, skew ~0)
        nn.init.normal_(self.ensemble_loc_head.weight, std=0.01)
        nn.init.zeros_(self.ensemble_loc_head.bias)
        nn.init.normal_(self.evidential_head.weight, std=0.02)
        with torch.no_grad():
            bias = self.evidential_head.bias.view(self.n_h, 4)
            bias[:, 0].fill_(_inv_softplus(1.0))  # v_base ~ 1.0
            init_df_frac = (6.0 - self.min_df) / (self.max_df - self.min_df)
            init_df_frac = min(max(init_df_frac, 1e-3), 1.0 - 1e-3)
            bias[:, 1].fill_(math.log(init_df_frac / (1.0 - init_df_frac)))  # df ~ 6.0
            bias[:, 2].fill_(-2.75)  # softplus(-2.75) * 1e-2 ~ 6.2e-4 -> scale ~ 0.02
            bias[:, 3].zero_()  # skew ~ 0.0

    def forward(
        self,
        h: torch.Tensor,
        *,
        conflict_score: torch.Tensor | None = None,
        compute_tail_risk: bool | None = None,
        compute_quantiles: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predice Skew Student-t, varianza aleatórica/epistémica y VaR/CVaR (bajo demanda)."""
        n_assets = h.shape[0]
        device, orig_dtype = h.device, h.dtype
        run_tail_risk = (
            (not self.training) if compute_tail_risk is None else bool(compute_tail_risk)
        )
        run_quantiles = run_tail_risk if compute_quantiles is None else bool(compute_quantiles)
        hn = self.norm(h)

        # 1) Ensamble multi-cabeza para localización y varianza epistémica de ensamble
        ens_loc = self.ensemble_loc_head(hn).view(n_assets, self.n_ensemble, self.n_h).float()
        loc32 = ens_loc.mean(dim=1)  # [N, H]
        ensemble_var32 = (
            ens_loc.var(dim=1, unbiased=False)
            if self.n_ensemble > 1
            else torch.zeros_like(loc32)
        )

        # 2) Parámetros NIG Evidenciales + sesgo en FP32 (cotas exactas incluso bajo BF16/FP16)
        raw = self.evidential_head(hn).view(n_assets, self.n_h, 4).float()
        v_base = F.softplus(raw[..., 0]) + 1e-2
        if conflict_score is not None:
            c_sc = conflict_score.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            if c_sc.ndim == 1:
                c_sc = c_sc.unsqueeze(-1)
            decay = F.softplus(
                self.conflict_evidence_decay.to(device=device, dtype=torch.float32)
            ).clamp(max=10.0)
            v_raw = (v_base * torch.exp(-decay * c_sc)).clamp_min(1e-4)
        else:
            v_raw = v_base

        df32 = self.min_df + (self.max_df - self.min_df) * torch.sigmoid(raw[..., 1])
        alpha32 = 0.5 * df32  # estrictamente > 1.11
        skew32 = self.max_skew * torch.tanh(raw[..., 3])
        beta32 = F.softplus(raw[..., 2]) * 1e-2 + (self.min_scale**2) * alpha32

        # 3) Descomposición exacta unificada Skew-NIG + Ensamble:
        #    1/v_eff = 1/v_raw + (alpha - 1) * ensemble_var / (beta * k(df, skew))
        #    => aleatoric_var + epistemic_var == total_var == Var(SkewStudentT) idénticamente.
        k_skew = _skew_variance_factor(df32, skew32)
        aleatoric_var32 = k_skew * (beta32 / (alpha32 - 1.0))
        inv_v_eff = (1.0 / v_raw) + ((alpha32 - 1.0) * ensemble_var32) / (beta32 * k_skew)
        v_eff32 = 1.0 / inv_v_eff.clamp_min(1e-6)
        epistemic_var32 = k_skew * (beta32 / (v_eff32 * (alpha32 - 1.0)))
        total_var32 = aleatoric_var32 + epistemic_var32

        scale32 = torch.sqrt((beta32 * (1.0 + v_eff32)) / (v_eff32 * alpha32))
        m1 = _student_t_first_abs_moment(df32)
        mean32 = loc32 + scale32 * (2.0 * skew32 * m1)

        out: dict[str, torch.Tensor] = {
            "loc": loc32.to(dtype=orig_dtype),
            "mean": mean32.to(dtype=orig_dtype),
            "scale": scale32.to(dtype=orig_dtype),
            "df": df32.to(dtype=orig_dtype),
            "skew": skew32.to(dtype=orig_dtype),
            "nig_v": v_eff32.to(dtype=orig_dtype),
            "nig_alpha": alpha32.to(dtype=orig_dtype),
            "nig_beta": beta32.to(dtype=orig_dtype),
            "aleatoric_var": aleatoric_var32.to(dtype=orig_dtype),
            "epistemic_var": epistemic_var32.to(dtype=orig_dtype),
            "ensemble_var": ensemble_var32.to(dtype=orig_dtype),
            "total_var": total_var32.to(dtype=orig_dtype),
            "aleatoric_std": torch.sqrt(aleatoric_var32).to(dtype=orig_dtype),
            "epistemic_std": torch.sqrt(epistemic_var32).to(dtype=orig_dtype),
        }
        # 4) Riesgo de cola analítico 1%/5% + 9 cuantiles (omitido en train() por defecto)
        if run_tail_risk:
            var_01_32, cvar_01_32 = skew_student_t_var_cvar(loc32, scale32, df32, skew32, 0.01)
            var_05_32, cvar_05_32 = skew_student_t_var_cvar(loc32, scale32, df32, skew32, 0.05)
            out["var_01"] = var_01_32.to(dtype=orig_dtype)
            out["var_05"] = var_05_32.to(dtype=orig_dtype)
            out["cvar_01"] = cvar_01_32.to(dtype=orig_dtype)
            out["cvar_05"] = cvar_05_32.to(dtype=orig_dtype)
        if run_quantiles:
            out["quantiles"] = skew_student_t_quantiles(
                loc32, scale32, df32, skew32, quantile_levels=self.quantile_taus
            ).to(dtype=orig_dtype)
        return out

    def nll(
        self,
        dist: dict[str, torch.Tensor],
        targets: torch.Tensor,
        *,
        reduction: str = "mean",
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calcula el NLL analítico exacto Skew Student-t sobre la salida de `forward`."""
        return skew_student_t_nll(
            targets,
            dist["loc"],
            dist["scale"],
            dist["df"],
            dist["skew"],
            reduction=reduction,
            mask=mask,
        )

    def loss(
        self,
        dist: dict[str, torch.Tensor],
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        evidential_reg_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pérdida total: NLL Skew Student-t exacto + regularizador de evidencia NIG."""
        reg_w = (
            self.evidential_reg_weight
            if evidential_reg_weight is None
            else float(evidential_reg_weight)
        )
        nll = self.nll(dist, targets, reduction="mean", mask=mask)
        mean32 = dist["mean"].float()
        y32 = targets.to(device=mean32.device, dtype=torch.float32)
        if y32.ndim == mean32.ndim - 1 and y32.shape == mean32.shape[:-1]:
            y32 = y32.unsqueeze(-1).expand_as(mean32)
        finite_y = torch.isfinite(y32)
        if mask is not None:
            valid = (mask.to(device=mean32.device) > 0.5).expand_as(mean32) & finite_y
        else:
            valid = finite_y
        y32_safe = torch.where(valid, y32, mean32.detach())
        err = torch.sqrt((y32_safe - mean32).pow(2) + 1e-8)
        evidence = 2.0 * dist["nig_v"].float() + dist["nig_alpha"].float()
        reg_elem = err * evidence
        m = valid.to(dtype=torch.float32)
        evid_reg = (reg_elem * m).sum() / m.sum().clamp_min(1.0)
        total = nll + reg_w * evid_reg
        return {
            "total": total,
            "nll": nll,
            "evidential_reg": evid_reg,
        }

    def tail_risk(
        self,
        dist: dict[str, torch.Tensor],
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula `(VaR_alpha, CVaR_alpha)` analíticos para cualquier nivel `alpha in (0, 1)`."""
        return skew_student_t_var_cvar(
            dist["loc"], dist["scale"], dist["df"], dist["skew"], alpha
        )


__all__ = [
    "ASSET_CONTEXT_VIEW_INDICES",
    "CONTEXT_VIEW_INDICES",
    "MARKET_VIEW_INDICES",
    "MODALITY_NAMES",
    "MODALITY_PAIR_INDICES",
    "CrossModalConflictFusion",
    "SelfSupervisedMultimodalAlignmentLoss",
    "SkewStudentTDistributionHead",
    "skew_student_t_nll",
    "skew_student_t_quantiles",
    "skew_student_t_var_cvar",
    "student_t_ppf",
]
