"""Simulador contrafactual de estrés diferenciable en GPU compartido (`F3` y `F4 T8`).

Implementa el contrato bloqueado de `#133` Sección C como utilidad pura de tensores,
independiente (cero imports desde MoE / Holo3L / CogCore), sin parámetros entrenables por
omisión, ejecutada internamente en FP32 (`torch.amp.autocast(..., enabled=False)`), con
**cero host-syncs GPU** (`audit_zero_host_syncs == 0`) y 100 % compatible con
`torch.compile(mode="default", dynamic=True)` en paneles irregulares `N_t`.

Escenarios ortogonales (`S = 5`, todos continuos en `0` y libres de singularidades `1/||beta||`):
  - `S0` `vol_regime_spike`: compresión de señal + arrastre de vol (`CTX_VOL_PCT`, `CTX_WASS`).
  - `S1` `liquidity_freeze_impact`: impacto cuadrático por iliquidez y posición (`CTX_LIQ_PCT`).
  - `S2` `factor_crowding_unwind`: reversión `-1.5 * alpha_sys` sobre proyección factorial `B_n`.
  - `S3` `rmt_eigenmode_rotation`: rotación de automodos RMT suave en `beta_R = 0` (`R_n`).
  - `S4` `tail_conflict_shock`: choque adverso por conflicto cross-modal y asimetría de colas.

Disciplina de gradientes estricta: ÚNICAMENTE `alpha` transporta gradientes; `vol` y el resto de
entradas se desconectan (`.detach()`) internamente para no sabotear la calibración CRPS/volatilidad.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# Índices canónicos verificados de `high_snr_context [N, 8]` (`market_physics_v4_gpu.py:1568-1579`)
CTX_RMT_ABS: int = 0
CTX_RMT_ENT: int = 1
CTX_RMT_NMODES_FRAC: int = 2
CTX_WASS: int = 3
CTX_CROSS_DISP: int = 4
CTX_MACRO_STRESS: int = 5
CTX_VOL_PCT: int = 6
CTX_LIQ_PCT: int = 7

SCENARIO_NAMES: tuple[str, ...] = (
    "vol_regime_spike",
    "liquidity_freeze_impact",
    "factor_crowding_unwind",
    "rmt_eigenmode_rotation",
    "tail_conflict_shock",
)

N_SCENARIOS: int = len(SCENARIO_NAMES)
CF_TRACE_DIM: int = 10


def _stable_calc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promociona FP16/BF16 a FP32 para álgebra lineal/logsumexp y preserva FP32/FP64."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _standardize_factor_matrix(
    mat: torch.Tensor,
    *,
    device: torch.device,
    calc_dtype: torch.dtype,
) -> torch.Tensor:
    """Estandariza columnas de una matriz factorial `[N, K]` de forma causal y sin `NaN`."""
    m_clean = torch.nan_to_num(
        mat.detach().to(device=device, dtype=calc_dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    m_c = m_clean - m_clean.mean(dim=0, keepdim=True)
    m_std = m_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    return m_c / m_std


def prepare_cf_factor_cache(
    *,
    exposures: torch.Tensor | None = None,
    rmt_loadings: torch.Tensor | None = None,
    ridge: float = 1e-3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | None]:
    """Iza normalización y operadores ridge FP32 (`solve_ex`) una vez por sesión fuera del bucle."""
    cache: dict[str, torch.Tensor | None] = dict(
        cf_b_n=None,
        cf_b_proj=None,
        cf_r_n=None,
        cf_r_proj=None,
        cf_r_row=None,
    )
    ref_t = exposures if exposures is not None else rmt_loadings
    if ref_t is None:
        return cache

    dev = device if device is not None else ref_t.device
    calc_dtype = _stable_calc_dtype(dtype)
    reg = float(ridge)

    with torch.amp.autocast(device_type=dev.type, enabled=False):
        if (
            exposures is not None
            and exposures.ndim == 2
            and exposures.shape[0] >= 2
            and exposures.shape[-1] > 0
        ):
            b_n = _standardize_factor_matrix(exposures, device=dev, calc_dtype=calc_dtype)
            f_dim = b_n.shape[-1]
            gram_b = b_n.T @ b_n + reg * torch.eye(f_dim, device=dev, dtype=calc_dtype)
            proj_b = torch.linalg.solve_ex(gram_b, b_n.T, check_errors=False)[0]
            cache["cf_b_n"] = b_n
            cache["cf_b_proj"] = proj_b

        if (
            rmt_loadings is not None
            and rmt_loadings.ndim == 2
            and rmt_loadings.shape[0] >= 2
            and rmt_loadings.shape[-1] > 0
        ):
            r_n = _standardize_factor_matrix(rmt_loadings, device=dev, calc_dtype=calc_dtype)
            k_dim = r_n.shape[-1]
            gram_r = r_n.T @ r_n + reg * torch.eye(k_dim, device=dev, dtype=calc_dtype)
            proj_r = torch.linalg.solve_ex(gram_r, r_n.T, check_errors=False)[0]
            r_row = r_n.norm(dim=-1) / math.sqrt(float(max(k_dim, 1)))
            cache["cf_r_n"] = r_n
            cache["cf_r_proj"] = proj_r
            cache["cf_r_row"] = r_row

    return cache


class CounterfactualStressSimulator(nn.Module):
    """Simulador contrafactual diferenciable de `S = 5` escenarios ortogonales (0 parámetros)."""

    def __init__(
        self,
        pos_cap: float = 0.05,
        softmin_beta: float = 6.0,
        flip_tau: float = 4.0,
        placebo_mode: bool = False,
        *,
        ridge: float = 1e-3,
    ) -> None:
        super().__init__()
        if pos_cap <= 0.0:
            raise ValueError("pos_cap debe ser > 0.0")
        if softmin_beta <= 0.0:
            raise ValueError("softmin_beta debe ser > 0.0")
        if flip_tau <= 0.0:
            raise ValueError("flip_tau debe ser > 0.0")
        if ridge <= 0.0:
            raise ValueError("ridge debe ser > 0.0")

        self.pos_cap = float(pos_cap)
        self.softmin_beta = float(softmin_beta)
        self.flip_tau = float(flip_tau)
        self.placebo_mode = bool(placebo_mode)
        self.ridge = float(ridge)

    @staticmethod
    def prepare_factor_cache(
        *,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        ridge: float = 1e-3,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor | None]:
        """Precalcula operadores factoriales/RMT izados una vez por sesión."""
        return prepare_cf_factor_cache(
            exposures=exposures,
            rmt_loadings=rmt_loadings,
            ridge=ridge,
            device=device,
            dtype=dtype,
        )

    def _project_factor_signal(
        self,
        alpha_c: torch.Tensor,
        mat: torch.Tensor | None,
        *,
        cached_mat: torch.Tensor | None,
        cached_proj: torch.Tensor | None,
        cached_row_norm: torch.Tensor | None = None,
        need_row_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Proyecta `alpha_c [N]` sobre el subespacio factorial/RMT en FP32 vía `solve_ex`."""
        n_assets = alpha_c.shape[0]
        device, calc_dtype = alpha_c.device, alpha_c.dtype

        if (
            cached_mat is not None
            and cached_proj is not None
            and cached_mat.shape[0] == n_assets
            and cached_proj.shape[-1] == n_assets
        ):
            m_n = cached_mat.detach().to(device=device, dtype=calc_dtype)
            m_proj = cached_proj.detach().to(device=device, dtype=calc_dtype)
            beta = m_proj @ alpha_c
            sys_sig = m_n @ beta
            if not need_row_norm:
                return sys_sig, beta, None
            if cached_row_norm is not None and cached_row_norm.shape[0] == n_assets:
                row_n = cached_row_norm.detach().to(device=device, dtype=calc_dtype)
            else:
                k_dim = max(int(m_n.shape[-1]), 1)
                row_n = m_n.norm(dim=-1) / math.sqrt(float(k_dim))
            return sys_sig, beta, row_n

        if mat is None or mat.ndim != 2 or n_assets < 2 or mat.shape[-1] <= 0:
            zeros_n = torch.zeros_like(alpha_c)
            return zeros_n, None, (zeros_n if need_row_norm else None)

        m_n = _standardize_factor_matrix(mat, device=device, calc_dtype=calc_dtype)
        k_dim = m_n.shape[-1]
        gram = m_n.T @ m_n + self.ridge * torch.eye(k_dim, device=device, dtype=calc_dtype)
        rhs = (m_n.T @ alpha_c).unsqueeze(-1)
        beta = torch.linalg.solve_ex(gram, rhs, check_errors=False)[0].squeeze(-1)
        sys_sig = m_n @ beta
        row_n = (
            m_n.norm(dim=-1) / math.sqrt(float(max(int(k_dim), 1))) if need_row_norm else None
        )
        return sys_sig, beta, row_n

    def forward(
        self,
        alpha: torch.Tensor,
        vol: torch.Tensor,
        *,
        high_snr_context: torch.Tensor | None = None,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        w_opt: torch.Tensor | None = None,
        quantiles: torch.Tensor | None = None,
        conflict: torch.Tensor | None = None,
        factor_cache: dict[str, torch.Tensor | None] | None = None,
        placebo_mode: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evalúa los 5 escenarios contrafactuales diferenciables respecto únicamente a `alpha`."""
        orig_dtype = alpha.dtype
        device = alpha.device
        calc_dtype = _stable_calc_dtype(orig_dtype)
        use_placebo = self.placebo_mode if placebo_mode is None else bool(placebo_mode)

        alpha_1d = alpha[:, 0] if alpha.ndim == 2 else alpha.reshape(-1)
        vol_1d = vol[:, 0] if vol.ndim == 2 else vol.reshape(-1)
        n_assets = alpha_1d.shape[0]

        with torch.amp.autocast(device_type=device.type, enabled=False):
            alpha_c = alpha_1d.to(dtype=calc_dtype)
            alpha_det = alpha_c.detach()
            abs_alpha = alpha_det.abs()
            sgn = torch.sign(alpha_det)

            # Disciplina de gradientes: `vol` y auxiliares estrictamente `.detach()`
            vol_d = torch.nan_to_num(
                vol_1d.detach().to(device=device, dtype=calc_dtype),
                nan=0.02,
                posinf=1.0,
                neginf=1e-4,
            ).clamp_min(1e-4)

            if high_snr_context is not None:
                ctx_raw = torch.nan_to_num(
                    high_snr_context.detach().to(device=device, dtype=calc_dtype),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if ctx_raw.ndim == 1:
                    ctx_raw = ctx_raw.unsqueeze(0).expand(n_assets, -1)
                c_dim = ctx_raw.shape[-1]
                vol_pct = (
                    ctx_raw[:, CTX_VOL_PCT].clamp(0.0, 1.0)
                    if c_dim > CTX_VOL_PCT
                    else torch.full((n_assets,), 0.5, device=device, dtype=calc_dtype)
                )
                wass = (
                    ctx_raw[:, CTX_WASS]
                    if c_dim > CTX_WASS
                    else torch.zeros(n_assets, device=device, dtype=calc_dtype)
                )
                liq_pct = (
                    ctx_raw[:, CTX_LIQ_PCT].clamp(0.0, 1.0)
                    if c_dim > CTX_LIQ_PCT
                    else torch.full((n_assets,), 0.5, device=device, dtype=calc_dtype)
                )
            else:
                vol_pct = torch.full((n_assets,), 0.5, device=device, dtype=calc_dtype)
                wass = torch.zeros(n_assets, device=device, dtype=calc_dtype)
                liq_pct = torch.full((n_assets,), 0.5, device=device, dtype=calc_dtype)

            # S0: vol_regime_spike
            stress = torch.sigmoid(wass - 1.0)
            mu_0 = alpha_c / (1.0 + 0.5 * vol_pct * (1.0 + stress)) - 0.35 * sgn * vol_d * F.relu(
                vol_pct - 0.5
            )

            # S1: liquidity_freeze_impact
            ill = (1.0 - liq_pct).clamp(0.0, 1.0)
            if w_opt is not None:
                w_d = torch.nan_to_num(
                    w_opt.detach().reshape(n_assets).to(device=device, dtype=calc_dtype),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                pos_util = (w_d.abs() / self.pos_cap).clamp(0.0, 2.0)
            else:
                pos_util = torch.zeros(n_assets, device=device, dtype=calc_dtype)
            mu_1 = alpha_c - sgn * (0.004 * ill.pow(2) * (1.0 + pos_util))

            # S2: factor_crowding_unwind (exacto 0 si alpha ⊥ B_n, homogéneo en escala)
            cache = factor_cache if factor_cache is not None else {}
            alpha_sys, _, _ = self._project_factor_signal(
                alpha_c,
                exposures,
                cached_mat=cache.get("cf_b_n"),
                cached_proj=cache.get("cf_b_proj"),
                need_row_norm=False,
            )
            mu_2 = alpha_c - 1.5 * alpha_sys

            # S3: rmt_eigenmode_rotation (suave en beta_R=0, sin singularidad 1/||beta||)
            r_sys, beta_r, r_row = self._project_factor_signal(
                alpha_c,
                rmt_loadings,
                cached_mat=cache.get("cf_r_n"),
                cached_proj=cache.get("cf_r_proj"),
                cached_row_norm=cache.get("cf_r_row"),
                need_row_norm=True,
            )
            if beta_r is not None and r_row is not None:
                # sqrt(||beta||^2 + eps^2) - eps: valor 0 y derivada 0 en beta=0 (sin 0/0 NaN)
                beta_r_norm = (torch.sqrt((beta_r * beta_r).sum() + 1e-12) - 1e-6).clamp_min(0.0)
                mu_3 = alpha_c - r_sys - 0.5 * sgn * r_row * beta_r_norm
            else:
                mu_3 = alpha_c

            # S4: tail_conflict_shock
            if conflict is not None:
                conf_d = torch.nan_to_num(
                    conflict.detach().reshape(n_assets).to(device=device, dtype=calc_dtype),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
            else:
                conf_d = torch.zeros(n_assets, device=device, dtype=calc_dtype)

            if quantiles is not None and quantiles.shape[-1] >= 3:
                q_2d = quantiles[:, 0, :] if quantiles.ndim == 3 else quantiles
                q_d = torch.nan_to_num(
                    q_2d.detach().to(device=device, dtype=calc_dtype),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                n_q = q_d.shape[-1]
                idx_lo = 1 if n_q >= 7 else 0
                idx_med = n_q // 2
                idx_hi = n_q - 2 if n_q >= 7 else n_q - 1
                q_lo = q_d[:, idx_lo]
                q_med = q_d[:, idx_med]
                q_hi = q_d[:, idx_hi]
                adv_tail = F.relu(-sgn * ((q_hi - q_med) - (q_med - q_lo)))
            else:
                adv_tail = torch.zeros(n_assets, device=device, dtype=calc_dtype)
            mu_4 = alpha_c - sgn * (0.30 * conf_d * vol_d + 0.25 * adv_tail)

            mu_s = torch.stack([mu_0, mu_1, mu_2, mu_3, mu_4], dim=-1)  # [N, 5]

            # Placebo sin puntos fijos (`placebo_mode=True`): conserva marginales, rompe alineación
            if use_placebo:
                delta_s = mu_s - alpha_c.unsqueeze(-1)
                if n_assets >= 2:
                    shift = max(1, n_assets // 2)
                    delta_pl = torch.roll(delta_s.detach(), shifts=shift, dims=0)
                    alpha_sys_eff = torch.roll(alpha_sys.detach(), shifts=shift, dims=0)
                    r_sys_eff = torch.roll(r_sys.detach(), shifts=shift, dims=0)
                else:
                    delta_pl = delta_s.detach()
                    alpha_sys_eff = alpha_sys.detach()
                    r_sys_eff = r_sys.detach()
                mu_s = alpha_c.unsqueeze(-1) + delta_pl
            else:
                alpha_sys_eff = alpha_sys
                r_sys_eff = r_sys

            # Agregación invariante de escala y segura en alpha = 0 (anti-FM-7)
            scale_ref = abs_alpha + 0.25 * vol_d + 1e-4
            denom_dd = abs_alpha + vol_d + 1e-4
            signed_pnl = sgn.unsqueeze(-1) * mu_s  # [N, 5]

            worst_pnl = (
                -(1.0 / self.softmin_beta)
                * torch.logsumexp(
                    -self.softmin_beta * signed_pnl / scale_ref.unsqueeze(-1), dim=-1
                )
                * scale_ref
            )
            active_bet = torch.tanh(abs_alpha / (0.25 * vol_d + 1e-4))
            p_flip = active_bet * torch.sigmoid(-self.flip_tau * worst_pnl / scale_ref)
            drawdown_frac = F.relu((abs_alpha - worst_pnl) / denom_dd).clamp(0.0, 2.0)
            e_cf = (p_flip + 0.5 * drawdown_frac).clamp(0.0, 3.0)

            scenario_drawdowns = F.relu(
                (abs_alpha.unsqueeze(-1) - signed_pnl) / denom_dd.unsqueeze(-1)
            ).clamp(0.0, 2.0)

            cf_trace = torch.cat(
                [
                    torch.tanh(scenario_drawdowns),  # [N, 5]
                    (worst_pnl / scale_ref).clamp(-4.0, 4.0).unsqueeze(-1),  # [N, 1]
                    p_flip.unsqueeze(-1),  # [N, 1]
                    e_cf.unsqueeze(-1),  # [N, 1]
                    (alpha_sys_eff / denom_dd).clamp(-4.0, 4.0).unsqueeze(-1),  # [N, 1]
                    (r_sys_eff / denom_dd).clamp(-4.0, 4.0).unsqueeze(-1),  # [N, 1]
                ],
                dim=-1,
            )  # [N, 10]

            return dict(
                e_cf=e_cf.to(dtype=orig_dtype),
                p_flip=p_flip.to(dtype=orig_dtype),
                worst_pnl=worst_pnl.to(dtype=orig_dtype),
                cf_trace=cf_trace.to(dtype=orig_dtype),
                scenario_drawdowns=scenario_drawdowns.to(dtype=orig_dtype),
                alpha_sys=alpha_sys_eff.to(dtype=orig_dtype),
                r_sys=r_sys_eff.to(dtype=orig_dtype),
            )


__all__ = [
    "CF_TRACE_DIM",
    "CTX_CROSS_DISP",
    "CTX_LIQ_PCT",
    "CTX_MACRO_STRESS",
    "CTX_RMT_ABS",
    "CTX_RMT_ENT",
    "CTX_RMT_NMODES_FRAC",
    "CTX_VOL_PCT",
    "CTX_WASS",
    "N_SCENARIOS",
    "SCENARIO_NAMES",
    "CounterfactualStressSimulator",
    "prepare_cf_factor_cache",
]
