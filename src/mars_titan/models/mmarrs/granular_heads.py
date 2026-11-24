"""Cabezas Granulares Multi-Horizonte y Pérdida Multi-Tarea Consciente de Cartera (`MMARRS`).

Emite en todos los horizontes `h in (1, 2, 5, 10, 21, 63)` de una sola pasada (cero host-syncs):
  1. Retorno descompuesto en Factor Sistemático + Alfa Idiosincrático puro.
  2. 9 cuantiles estrictamente crecientes sin cruces (`0.01..0.99`) por construcción (`softplus`).
  3. Probabilidades Triple-Barrier triclase `(bajista, lateral, alcista)` + Probabilidad de Giro.
  4. Log-volatilidad rugosa predicha y auto-estimación de error epistémico esperado.

Entrenado con:
  - Supervisión Profunda sobre los pasos del doble bucle (`k = 1..K`).
  - `horizon_mask` estricto: cuando `y_ret.ndim == 1`, entrena SOLO `h=0` (jamás replica `y_1`
    sobre `h=63`, P0-4).
  - CRPS/Pinball fusionado vía `fused_pinball_crps_loss` (autograd Triton `sm_120`, P0-7) y modo
    gemelo FM-7 `noic_pinball_twin` (`w_ic = 0`).
  - Sharpe-Turnover normalizado por volatilidad `-net_ret / (port_vol + 1e-4)` (P0-3).
  - Decorrelación Bi-Hemisférica de residuos y pérdidas auxiliares (SSL / Skew-t / Ponder KL).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features
from mars_titan.evaluation.metacognitive_agent import DifferentiablePortfolioLayer

from .blackwell_kernels import fused_pinball_crps_loss
from .config import DEFAULT_HORIZONS, DEFAULT_QUANTILES
from .dl_training_sota import (
    build_norm_layer,
    cross_sectional_rank_contrastive_loss,
    tail_listnet_loss,
    vicreg_covariance_loss,
)
from .frontier_modules_sota import (
    polar_alpha_calibration_loss,
    polar_alpha_persistence_loss,
)


def infer_horizon_mask(
    y_ret: torch.Tensor,
    n_horizons: int,
) -> torch.Tensor:
    """Construye máscara `[H]` (1.0 en horizontes activos, 0.0 en inactivos; P0-4)."""
    active = 1 if y_ret.ndim == 1 else min(int(y_ret.shape[-1]), int(n_horizons))
    idx = torch.arange(n_horizons, device=y_ret.device)
    return (idx < active).to(dtype=y_ret.dtype if y_ret.is_floating_point() else torch.float32)


class MultiHorizonGranularHead(nn.Module):
    """Decodificador multi-cabeza y multi-horizonte con 9 cuantiles garantizados sin cruces."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
        exposures_dim: int = 9,
        tabm_k: int = 1,
        norm_kind: str = "layernorm",
    ) -> None:
        super().__init__()
        if tabm_k < 1:
            raise ValueError(f"tabm_k debe ser >= 1; recibido {tabm_k}")
        self.hidden_dim = hidden_dim
        self.horizons = horizons
        self.quantile_levels = quantile_levels
        self.n_h = len(horizons)
        self.n_q = len(quantile_levels)
        self.exposures_dim = exposures_dim
        self.tabm_k = int(tabm_k)
        self.norm_kind = str(norm_kind)

        self.norm: nn.Module = (
            nn.LayerNorm(hidden_dim)
            if self.norm_kind == "layernorm"
            else build_norm_layer(self.norm_kind, hidden_dim)  # type: ignore[arg-type]
        )
        # 1) Alfa idiosincrático + factor sistemático
        self.alpha_head = nn.Linear(hidden_dim, self.n_h)
        self.factor_load = (
            nn.Linear(exposures_dim, self.n_h, bias=False) if exposures_dim > 0 else None
        )
        # 2) 9 cuantiles estrictamente crecientes: mediana + saltos positivos (softplus)
        self.q_median = nn.Linear(hidden_dim, self.n_h)
        self.q_deltas_lo = nn.Linear(hidden_dim, self.n_h * (self.n_q // 2))
        self.q_deltas_hi = nn.Linear(hidden_dim, self.n_h * (self.n_q - 1 - self.n_q // 2))
        # 3) Triple-Barrier (3 clases) + Turning-Point (1 logit)
        self.tb_head = nn.Linear(hidden_dim, self.n_h * 3)
        self.turn_head = nn.Linear(hidden_dim, self.n_h)
        # 4) Volatilidad rugosa y error epistémico esperado
        self.vol_head = nn.Linear(hidden_dim, self.n_h)
        self.epistemic_head = nn.Linear(hidden_dim, self.n_h)

        # Escala inicial financiera realista (retornos diarios ~1-2 %)
        nn.init.normal_(self.alpha_head.weight, std=0.01)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.normal_(self.q_median.weight, std=0.01)
        nn.init.zeros_(self.q_median.bias)
        nn.init.constant_(self.vol_head.bias, -3.9)  # exp(-3.9) ~ 0.02 vol diaria

        # Adaptadores TabM BatchEnsemble rango-1 SOLO sobre `alpha_head` (P0-6; 0 extras si K=1)
        if self.tabm_k > 1:
            self.tabm_r: nn.Parameter | None = nn.Parameter(torch.empty(self.tabm_k, hidden_dim))
            self.tabm_s: nn.Parameter | None = nn.Parameter(torch.empty(self.tabm_k, self.n_h))
            nn.init.normal_(self.tabm_r, mean=1.0, std=0.05)
            nn.init.normal_(self.tabm_s, mean=1.0, std=0.05)
        else:
            self.register_parameter("tabm_r", None)
            self.register_parameter("tabm_s", None)

    def forward(
        self,
        h: torch.Tensor,
        *,
        exposures: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decodifica todas las cabezas granulares desde `h [N, D]`."""
        n_assets = h.shape[0]
        hn = self.norm(h)
        if self.tabm_k > 1 and self.tabm_r is not None and self.tabm_s is not None:
            r_k = self.tabm_r.to(device=hn.device, dtype=hn.dtype)
            s_k = self.tabm_s.to(device=hn.device, dtype=hn.dtype)
            hn_k = hn.unsqueeze(1) * r_k.unsqueeze(0)  # [N, K, D]
            raw_k = F.linear(hn_k, self.alpha_head.weight, None) * s_k.unsqueeze(0)
            if self.alpha_head.bias is not None:
                raw_k = raw_k + self.alpha_head.bias.view(1, 1, -1)
            # Centrar cada miembro transversalmente y promediar K -> alpha para pred_return/P+/CRPS
            alpha_members = raw_k - raw_k.mean(dim=0, keepdim=True)  # [N, K, H]
            alpha_c = alpha_members.mean(dim=1)  # [N, H]
        else:
            alpha = self.alpha_head(hn)  # [N, H]
            # Centrar alfa transversalmente para conservar neutralidad estructural
            alpha_c = alpha - alpha.mean(dim=0, keepdim=True)
            alpha_members = alpha_c.unsqueeze(1)  # [N, 1, H]

        if (
            self.factor_load is not None
            and exposures is not None
            and exposures.shape[-1] == self.exposures_dim
        ):
            factor_ret = self.factor_load(exposures.to(device=h.device, dtype=h.dtype)) * 0.01
        else:
            factor_ret = torch.zeros_like(alpha_c)
        pred_return = factor_ret + alpha_c

        # Construcción de cuantiles estrictamente ordenados sin cruces posibles
        med = self.q_median(hn)  # [N, H]
        n_lo = self.n_q // 2
        n_hi = self.n_q - 1 - n_lo
        d_lo = F.softplus(self.q_deltas_lo(hn).view(n_assets, self.n_h, n_lo)) * 0.01 + 1e-4
        d_hi = F.softplus(self.q_deltas_hi(hn).view(n_assets, self.n_h, n_hi)) * 0.01 + 1e-4
        lo_cum = torch.cumsum(d_lo, dim=-1).flip(dims=[-1])
        hi_cum = torch.cumsum(d_hi, dim=-1)
        q_lo = med.unsqueeze(-1) - lo_cum
        q_hi = med.unsqueeze(-1) + hi_cum
        quantiles = torch.cat([q_lo, med.unsqueeze(-1), q_hi], dim=-1)  # [N, H, Q]

        tb_logits = self.tb_head(hn).view(n_assets, self.n_h, 3)
        tb_probs = torch.softmax(tb_logits, dim=-1)
        turn_logits = self.turn_head(hn)
        turning_prob = torch.sigmoid(turn_logits)
        log_vol = torch.clamp(self.vol_head(hn), min=-6.0, max=0.5)
        vol_pred = torch.exp(log_vol)
        epistemic_error = F.softplus(self.epistemic_head(hn)) * 0.02

        return dict(
            pred_return=pred_return,
            alpha=alpha_c,
            alpha_members=alpha_members,
            factor_return=factor_ret,
            quantiles=quantiles,
            tb_logits=tb_logits,
            tb_probs=tb_probs,
            turn_logits=turn_logits,
            turning_prob=turning_prob,
            log_vol=log_vol,
            vol_pred=vol_pred,
            epistemic_error=epistemic_error,
        )


class MMARRSMultiTaskLoss(nn.Module):
    """Pérdida compuesta: Deep Supervision + IC neut (Gaussian/TabM) + CRPS + Sharpe + stab05."""

    def __init__(
        self,
        *,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
        w_ic: float = 1.0,
        w_crps: float = 15.0,
        w_tb: float = 0.25,
        w_turn: float = 0.15,
        w_vol: float = 0.20,
        w_epist: float = 5.0,
        w_energy: float = 0.25,
        w_sharpe: float = 0.20,
        w_decor: float = 0.15,
        l1_cost: float = 0.001,
        noic_pinball_twin: bool = False,
        use_fused_blackwell_kernels: bool = True,
        use_gaussian_rank_targets: bool = False,
        w_stab_downside: float = 0.0,
        stab_ema_decay: float = 0.90,
        w_listnet_tail: float = 0.0,
        w_latent_cov: float = 0.0,
        w_supcon: float = 0.0,
        w_polar_cal: float = 1.0,
        w_churn: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.register_buffer(
            "taus", torch.tensor(quantile_levels, dtype=torch.float32).view(1, 1, -1)
        )
        self.register_buffer("_ema_ic", torch.zeros((), dtype=torch.float32))
        self.register_buffer("_ema_ic_initialized", torch.zeros((), dtype=torch.float32))
        self.noic_pinball_twin = bool(noic_pinball_twin)
        self.w_ic = 0.0 if self.noic_pinball_twin else float(w_ic)
        self.w_crps = float(w_crps)
        self.w_tb = float(w_tb)
        self.w_turn = float(w_turn)
        self.w_vol = float(w_vol)
        self.w_epist = float(w_epist)
        self.w_energy = float(w_energy)
        self.w_sharpe = 0.0 if self.noic_pinball_twin else float(w_sharpe)
        self.w_decor = float(w_decor)
        self.l1_cost = float(l1_cost)
        self.use_fused_blackwell_kernels = bool(use_fused_blackwell_kernels)
        self.use_gaussian_rank_targets = bool(use_gaussian_rank_targets)
        self.w_stab_downside = 0.0 if self.noic_pinball_twin else float(w_stab_downside)
        self.stab_ema_decay = float(stab_ema_decay)
        self.w_listnet_tail = 0.0 if self.noic_pinball_twin else float(w_listnet_tail)
        self.w_latent_cov = float(w_latent_cov)
        self.w_supcon = float(w_supcon)
        self.w_polar_cal = float(w_polar_cal)
        self.w_churn = float(w_churn)
        self.portfolio_layer = DifferentiablePortfolioLayer(
            steps=6, risk_aversion=2.0, factor_penalty=8.0, l1_cost=l1_cost, gross_limit=1.0
        )

    @torch.no_grad()
    def reset_stab_ema(self) -> None:
        """Reinicia los buffers causales de EMA de IC para `stab05` (sin host-syncs)."""
        self._ema_ic.zero_()
        self._ema_ic_initialized.zero_()

    @staticmethod
    def _neutralized_ic_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        exposures: torch.Tensor | None = None,
        liquidity_weights: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        use_gaussian_rank_targets: bool = False,
    ) -> torch.Tensor:
        """Pérdida `1 - corr` en FP32 maestro (soporta `[N, H]` y TabM `[N, K, H]`, ties/NaNs)."""
        orig_dtype = pred.dtype
        calc_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            p_raw = pred.to(dtype=calc_dtype)
            y_raw = target.to(device=pred.device, dtype=calc_dtype)
            if p_raw.ndim == 1:
                p3 = p_raw.view(-1, 1, 1)
            elif p_raw.ndim == 2:
                p3 = p_raw.unsqueeze(1)  # [N, 1, H]
            elif p_raw.ndim == 3:
                p3 = p_raw  # [N, K, H]
            else:
                raise ValueError(f"pred debe tener 1, 2 o 3 dims; recibido ndim={p_raw.ndim}")

            y2 = y_raw.unsqueeze(-1) if y_raw.ndim == 1 else y_raw  # [N, H]
            n_assets, k_mem, n_h = p3.shape
            y2 = y2[:, :n_h]

            valid_y = torch.isfinite(y2)  # [N, H]
            valid_f = valid_y.to(dtype=calc_dtype)
            valid_3 = valid_y.unsqueeze(1)  # [N, 1, H]
            v_cnt = valid_f.sum(dim=0, keepdim=True).clamp_min(1.0)  # [1, H]
            v_cnt_3 = v_cnt.unsqueeze(1)  # [1, 1, H]

            # Aislar filas NaN antes de neutralizar factores para garantizar gradiente exacto 0
            p3 = torch.where(valid_3, p3, torch.zeros_like(p3))
            p3 = torch.where(
                valid_3, p3 - p3.sum(dim=0, keepdim=True) / v_cnt_3, torch.zeros_like(p3)
            )

            # Target gaussiano promediado en empates SOLO dentro de `_neutralized_ic_loss` (P0-6)
            if use_gaussian_rank_targets:
                y_clean = gaussian_rank_features(y2, mask=valid_y)
            else:
                y_zero = torch.where(valid_y, y2, torch.zeros_like(y2))
                y_clean = torch.where(
                    valid_y,
                    y_zero - y_zero.sum(dim=0, keepdim=True) / v_cnt,
                    torch.zeros_like(y_zero),
                )

            has_exp = (
                exposures is not None
                and exposures.shape[-1] > 0
                and n_assets > exposures.shape[-1]
            )
            if has_exp:
                b = exposures.to(device=pred.device, dtype=calc_dtype)
                b = torch.where(torch.isfinite(b), b, torch.zeros_like(b))
                b = b - b.mean(dim=0, keepdim=True)
                b_gram = b.T @ b + torch.eye(b.shape[-1], device=pred.device, dtype=calc_dtype)
                p_flat = p3.reshape(n_assets, k_mem * n_h)
                proj_w, _ = torch.linalg.solve_ex(b_gram, b.T @ p_flat, check_errors=False)
                p3 = (p_flat - b @ proj_w).reshape(n_assets, k_mem, n_h)
                p3 = torch.where(valid_3, p3, torch.zeros_like(p3))
                p3 = torch.where(
                    valid_3, p3 - p3.sum(dim=0, keepdim=True) / v_cnt_3, torch.zeros_like(p3)
                )

            if liquidity_weights is not None:
                lw = liquidity_weights.to(device=pred.device, dtype=calc_dtype).reshape(-1, 1)
                lw = torch.where(torch.isfinite(lw), lw, torch.zeros_like(lw)).clamp_min(1e-4)
                lw = lw / lw.mean().clamp_min(1e-6)
                sqrt_lw = torch.sqrt(lw).unsqueeze(-1)  # [N, 1, 1]
                p_w = p3 * sqrt_lw
                y_w = y_clean.unsqueeze(1) * sqrt_lw
            else:
                p_w = p3
                y_w = y_clean.unsqueeze(1)

            p_norm = p_w.norm(dim=0).clamp_min(1e-6)  # [K, H]
            y_norm = y_w.norm(dim=0).clamp_min(1e-6)  # [1, H]
            corr_kh = (p_w * y_w).sum(dim=0) / (p_norm * y_norm)  # [K, H]
            per_h = (1.0 - corr_kh).mean(dim=0)  # [H]
            if horizon_mask is not None:
                hm_c = (
                    horizon_mask.to(device=pred.device, dtype=calc_dtype).reshape(-1)[
                        : per_h.shape[0]
                    ]
                    > 0.5
                ).to(dtype=calc_dtype)
                return ((per_h * hm_c).sum() / hm_c.sum().clamp_min(1.0)).to(dtype=orig_dtype)
            return per_h.mean().to(dtype=orig_dtype)

    def _step_loss(
        self,
        pred: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        exposures: torch.Tensor | None,
        horizon_mask: torch.Tensor | None = None,
        liquidity_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pr = pred["pred_return"]
        device, dtype = pr.device, pr.dtype
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        y_ret = targets["returns"].to(device=device, dtype=dtype)
        # P0-4: JAMÁS replicar `y_1` 1D sobre `h=63`; recortar estrictamente a `n_h = 1`
        if y_ret.ndim == 1:
            y_ret = y_ret.unsqueeze(-1)
        n_h = min(y_ret.shape[-1], pr.shape[-1])
        y_h = y_ret[:, :n_h]
        p_h = pr[:, :n_h]
        q_h = pred["quantiles"][:, :n_h, :]

        if horizon_mask is not None:
            hm_bool = horizon_mask.to(device=device).reshape(-1)[:n_h] > 0.5
            hm = hm_bool.to(dtype=dtype)
        else:
            hm_bool = None
            hm = None

        # #187: máscara exacta por celda finita (cero 0*NaN, cero host-syncs!)
        fin_y = torch.isfinite(y_h)  # [N, H]
        cell_valid = (fin_y & hm_bool.unsqueeze(0)) if hm_bool is not None else fin_y
        cell_w = cell_valid.to(dtype=dtype)
        cell_denom = cell_w.sum().clamp_min(1.0)
        y_h_safe = torch.where(cell_valid, y_h, torch.zeros_like(y_h))
        p_h_safe = torch.where(cell_valid, p_h, torch.zeros_like(p_h))

        # Detectar PolarAlphaHead (#195 P0-1a/b/c: ranking sobre `alpha_u`, cal sobre `gamma_h`)
        is_polar = (
            "alpha_cal_for_cal" in pred
            and "alpha_cal_for_pplus" in pred
            and "alpha_u" in pred
        )

        # 1) IC neutralizado multi-horizonte (+ Polar P0-1a/b/c o TabM K miembros + Huber crudo)
        if is_polar:
            u_h = pred["alpha_u"][:, :n_h]
            u_eff = torch.where(cell_valid, u_h, torch.zeros_like(u_h))
            ic_term = self._neutralized_ic_loss(
                u_eff,
                y_h,
                exposures,
                liquidity_weights=liquidity_weights,
                horizon_mask=hm,
                use_gaussian_rank_targets=self.use_gaussian_rank_targets,
            )
            # P0-1a: ranking puro sobre `alpha_u` sin 5.0 * Huber distorsionando la dirección
            loss_ic = ic_term
            loss_polar_cal = polar_alpha_calibration_loss(
                pred["alpha_cal_for_cal"][:, :n_h],
                y_h,
                delta=0.03,
                demean_target=True,
                horizon_mask=hm,
            ).to(dtype=dtype)
        else:
            loss_polar_cal = torch.zeros((), device=device, dtype=dtype)
            alpha_members = pred.get("alpha_members")
            use_tabm = (
                isinstance(alpha_members, torch.Tensor)
                and alpha_members.ndim == 3
                and alpha_members.shape[1] > 1
            )
            if use_tabm:
                assert alpha_members is not None
                am_h = alpha_members[:, :, :n_h]
                f_ret = pred.get("factor_return")
                p_members = (
                    am_h + f_ret[:, :n_h].unsqueeze(1)
                    if isinstance(f_ret, torch.Tensor)
                    else am_h
                )
                cell_v3 = cell_valid.unsqueeze(1)  # [N, 1, H]
                pm_safe = torch.where(cell_v3, p_members, torch.zeros_like(p_members))
                y3_safe = y_h_safe.unsqueeze(1).expand_as(p_members)
                ic_term = self._neutralized_ic_loss(
                    pm_safe,
                    y_h,
                    exposures,
                    liquidity_weights=liquidity_weights,
                    horizon_mask=hm,
                    use_gaussian_rank_targets=self.use_gaussian_rank_targets,
                )
                huber_elem = F.huber_loss(pm_safe, y3_safe, delta=0.03, reduction="none")
                huber_term = (
                    torch.where(cell_v3, huber_elem, torch.zeros_like(huber_elem))
                    .mean(dim=1)
                    .sum()
                    / cell_denom
                )
                loss_ic = ic_term + 5.0 * huber_term
            else:
                ic_term = self._neutralized_ic_loss(
                    p_h_safe,
                    y_h,
                    exposures,
                    liquidity_weights=liquidity_weights,
                    horizon_mask=hm,
                    use_gaussian_rank_targets=self.use_gaussian_rank_targets,
                )
                huber_elem = F.huber_loss(p_h_safe, y_h_safe, delta=0.03, reduction="none")
                huber_term = (
                    torch.where(cell_valid, huber_elem, torch.zeros_like(huber_elem)).sum()
                    / cell_denom
                )
                loss_ic = ic_term + 5.0 * huber_term
        rho_ic = 1.0 - ic_term

        # 1b) Tail-ListNet simétrico opt-in (`dl_training_sota.py`, sobre `alpha_u` si Polar)
        if self.w_listnet_tail > 0.0:
            rank_in = pred["alpha_u"][:, :n_h] if is_polar else pred["alpha"][:, :n_h]
            loss_listnet_tail = tail_listnet_loss(
                rank_in,
                y_h,
                mask=cell_valid,
                liquidity_weights=liquidity_weights,
            ).to(dtype=dtype)
        else:
            loss_listnet_tail = torch.zeros((), device=device, dtype=dtype)

        # 2) Pinball / CRPS discreto sobre los 9 cuantiles (kernel Blackwell fusionado P0-7 + #187)
        q_h_safe = torch.where(cell_valid.unsqueeze(-1), q_h, torch.zeros_like(q_h))
        if self.use_fused_blackwell_kernels and hm is None:
            crps_el = fused_pinball_crps_loss(
                q_h_safe,
                y_h_safe,
                self.taus.reshape(-1),
                crps_scale=2.0,
                reduction="none",
                use_triton=True,
            )
            loss_crps = (
                torch.where(cell_valid, crps_el, torch.zeros_like(crps_el)).sum() / cell_denom
            )
        else:
            taus = self.taus.to(device=device, dtype=dtype)
            diff = y_h_safe.unsqueeze(-1) - q_h_safe
            pinball_nh = torch.maximum(taus * diff, (taus - 1.0) * diff).mean(dim=-1)  # [N, H]
            loss_crps = (
                2.0
                * torch.where(cell_valid, pinball_nh, torch.zeros_like(pinball_nh)).sum()
                / cell_denom
            )

        # 3) Triple-Barrier CE y Turning-Point BCE (enmascarado exacto por celda finita + horizonte)
        if "tb_class" in targets and targets["tb_class"] is not None:
            y_tb_raw = targets["tb_class"]
            if y_tb_raw.ndim == 1:
                y_tb_raw = y_tb_raw.unsqueeze(-1)
            y_tb = y_tb_raw[:, :n_h].to(device=device, dtype=torch.long)
            tb_valid = (
                (hm_bool.unsqueeze(0).expand_as(y_tb))
                if hm_bool is not None
                else torch.ones_like(y_tb, dtype=torch.bool)
            )
        else:
            # Varianza centrada dos pasadas sobre celdas finitas (inmune a NaN parciales!)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                y_c = y_h_safe.to(dtype=calc_dtype)
                vf = fin_y.to(dtype=calc_dtype)
                vcnt = vf.sum(dim=0, keepdim=True).clamp_min(1.0)
                ym = y_c.sum(dim=0, keepdim=True) / vcnt
                y_dev = torch.where(fin_y, y_c - ym, torch.zeros_like(y_c))
                y_var = y_dev.pow(2).sum(dim=0, keepdim=True) / (vcnt - 1.0).clamp_min(1.0)
                thr = torch.sqrt(y_var + 1e-12).clamp_min(1e-4).to(dtype=dtype) * 0.5
            y_tb = torch.where(y_h_safe > thr, 2, torch.where(y_h_safe < -thr, 0, 1)).long()
            tb_valid = cell_valid
        tb_el = F.cross_entropy(
            pred["tb_logits"][:, :n_h, :].reshape(-1, 3),
            y_tb.reshape(-1).clamp(0, 2),
            reduction="none",
        ).view(p_h.shape[0], n_h)
        loss_tb = (
            torch.where(tb_valid, tb_el, torch.zeros_like(tb_el)).sum()
            / tb_valid.to(dtype=dtype).sum().clamp_min(1.0)
        )

        if "turning" in targets and targets["turning"] is not None:
            y_turn_raw = targets["turning"]
            if y_turn_raw.ndim == 1:
                y_turn_raw = y_turn_raw.unsqueeze(-1)
            y_turn = y_turn_raw[:, :n_h].to(device=device, dtype=dtype)
            turn_fin = torch.isfinite(y_turn)
            turn_valid = (turn_fin & hm_bool.unsqueeze(0)) if hm_bool is not None else turn_fin
            y_turn_safe = torch.where(turn_valid, y_turn, torch.zeros_like(y_turn))
            t_logits_safe = torch.where(
                turn_valid, pred["turn_logits"][:, :n_h], torch.zeros_like(y_turn)
            )
            turn_el = F.binary_cross_entropy_with_logits(
                t_logits_safe, y_turn_safe, reduction="none"
            )
            loss_turn = (
                torch.where(turn_valid, turn_el, torch.zeros_like(turn_el)).sum()
                / turn_valid.to(dtype=dtype).sum().clamp_min(1.0)
            )
        else:
            loss_turn = torch.zeros((), device=device, dtype=dtype)

        # 4) Volatilidad y calibración de auto-error epistémico (enmascarado exacto por celda)
        if "vol" in targets and targets["vol"] is not None:
            y_vol_raw = targets["vol"]
            if y_vol_raw.ndim == 1:
                y_vol_raw = y_vol_raw.unsqueeze(-1)
            y_vol_t = y_vol_raw[:, :n_h].to(device=device, dtype=dtype)
            vol_fin = torch.isfinite(y_vol_t) & (y_vol_t > 0.0)
            vol_valid = (vol_fin & hm_bool.unsqueeze(0)) if hm_bool is not None else vol_fin
            y_vol_safe = torch.where(vol_valid, y_vol_t.clamp_min(1e-4), torch.ones_like(y_vol_t))
            log_vol_safe = torch.where(
                vol_valid, pred["log_vol"][:, :n_h], torch.zeros_like(y_vol_t)
            )
            vol_el = F.huber_loss(log_vol_safe, torch.log(y_vol_safe), delta=0.5, reduction="none")
            loss_vol = (
                torch.where(vol_valid, vol_el, torch.zeros_like(vol_el)).sum()
                / vol_valid.to(dtype=dtype).sum().clamp_min(1.0)
            )
        else:
            loss_vol = torch.zeros((), device=device, dtype=dtype)

        real_abs_err = torch.where(
            cell_valid, (y_h_safe - p_h.detach()).abs(), torch.zeros_like(y_h_safe)
        )
        ep_pred_safe = torch.where(
            cell_valid, pred["epistemic_error"][:, :n_h], torch.zeros_like(y_h_safe)
        )
        epist_el = F.huber_loss(ep_pred_safe, real_abs_err, delta=0.02, reduction="none")
        loss_epist = (
            torch.where(cell_valid, epist_el, torch.zeros_like(epist_el)).sum() / cell_denom
        )

        total = (
            self.w_ic * loss_ic
            + self.w_listnet_tail * loss_listnet_tail
            + (self.w_polar_cal * loss_polar_cal if is_polar else 0.0)
            + self.w_crps * loss_crps
            + self.w_tb * loss_tb
            + self.w_turn * loss_turn
            + self.w_vol * loss_vol
            + self.w_epist * loss_epist
        )
        return dict(
            total=total,
            ic=loss_ic,
            listnet_tail=loss_listnet_tail,
            polar_cal=loss_polar_cal,
            rho_ic=rho_ic,
            crps=loss_crps,
            tb=loss_tb,
            turn=loss_turn,
            vol=loss_vol,
            epist=loss_epist,
        )

    def forward(
        self,
        *,
        step_outputs: list[dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
        exposures: torch.Tensor | None = None,
        energy_total: torch.Tensor | None = None,
        w_prev: torch.Tensor | None = None,
        companion_pred: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        liquidity_weights: torch.Tensor | None = None,
        aux_losses: dict[str, torch.Tensor] | None = None,
        ema_ic_ref: torch.Tensor | None = None,
        final_hidden: torch.Tensor | None = None,
        prev_alpha_u: torch.Tensor | None = None,
        cur_asset_ids: torch.Tensor | None = None,
        prev_asset_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calcula la pérdida multi-tarea completa con Deep Supervision sobre los `K` pasos."""
        n_steps = len(step_outputs)
        weights = [float(k + 1) for k in range(n_steps)]
        w_sum = sum(weights)

        ref = step_outputs[0]["pred_return"]
        device, dtype = ref.device, ref.dtype
        last_parts: dict[str, torch.Tensor] = {}
        deep_total = torch.zeros((), device=device, dtype=dtype)
        for wk, pred_k in zip(weights, step_outputs, strict=True):
            parts = self._step_loss(
                pred_k,
                targets,
                exposures,
                horizon_mask=horizon_mask,
                liquidity_weights=liquidity_weights,
            )
            deep_total = deep_total + (wk / w_sum) * parts["total"]
            last_parts = parts

        final_pred = step_outputs[-1]
        y_ret = targets["returns"].to(device=device, dtype=dtype)
        y1 = y_ret[:, 0] if y_ret.ndim == 2 else y_ret

        # Volatilidad para P+ y normalización de Sharpe (realizada si provista, sanitizada #187)
        if "vol" in targets and targets["vol"] is not None:
            v_raw = targets["vol"].to(device=device, dtype=dtype)
            v_sel = v_raw[:, 0] if v_raw.ndim == 2 else v_raw
            v_fin = torch.isfinite(v_sel) & (v_sel > 0.0)
            asset_vol = (
                torch.where(v_fin, v_sel, torch.full_like(v_sel, 0.02))
                .detach()
                .clamp_min(1e-4)
            )
        else:
            v_sel = final_pred["vol_pred"][:, 0].detach()
            v_fin = torch.isfinite(v_sel) & (v_sel > 0.0)
            asset_vol = torch.where(v_fin, v_sel, torch.full_like(v_sel, 0.02)).clamp_min(1e-4)

        # Cartera convexa diferenciable P+: usa `alpha_cal_for_pplus` cuando Polar (#195 P0-1c!)
        pplus_alpha = (
            final_pred["alpha_cal_for_pplus"][:, 0]
            if "alpha_cal_for_pplus" in final_pred
            else final_pred["alpha"][:, 0]
        )
        w_t = self.portfolio_layer(
            pplus_alpha,
            vol=asset_vol,
            exposures=exposures,
            w_prev=w_prev,
        )
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        w_c = w_t.to(dtype=calc_dtype)
        y1_c = y1.to(dtype=calc_dtype)
        y1_fin = torch.isfinite(y1_c)
        y1_safe = torch.where(y1_fin, y1_c, torch.zeros_like(y1_c))
        w_fin_c = torch.where(y1_fin, w_c, torch.zeros_like(w_c))
        vol_c = asset_vol.to(dtype=calc_dtype)
        gross = (w_fin_c * y1_safe).sum()
        if w_prev is not None:
            w_prev_c = torch.nan_to_num(
                w_prev.to(device=device, dtype=calc_dtype), nan=0.0, posinf=0.0, neginf=0.0
            )
            turnover = (w_c - w_prev_c).abs().sum()
        else:
            turnover = w_c.abs().sum()
        net_ret = gross - self.l1_cost * turnover
        # P0-3: denominador de volatilidad de cartera (impide trampas por apalancamiento)
        port_vol = torch.sqrt((w_c * vol_c).pow(2).sum() + 1e-8)
        raw_sharpe = -net_ret / (port_vol + 1e-4)
        loss_sharpe = torch.where(
            y1_fin.any(), raw_sharpe, torch.zeros_like(raw_sharpe)
        ).to(dtype=dtype)

        # Semivarianza bajista `stab05` (P0-6): bisagra causal `0.5 * relu(ema_ic_det - rho_s)^2`
        rho_s = last_parts["rho_ic"]
        if self.w_stab_downside > 0.0:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                rho_c = rho_s.to(dtype=calc_dtype)
                ema_prior = self._ema_ic.detach().clone().to(device=device, dtype=calc_dtype)
                init_flag = (
                    self._ema_ic_initialized.detach().clone().to(device=device, dtype=calc_dtype)
                )
                if ema_ic_ref is not None:
                    ema_ic_det = ema_ic_ref.detach().to(device=device, dtype=calc_dtype).reshape(())
                else:
                    ema_ic_det = torch.where(init_flag > 0.5, ema_prior, rho_c.detach())
                loss_stab_downside = (0.5 * F.relu(ema_ic_det - rho_c).pow(2)).to(dtype=dtype)
                if self.training and ema_ic_ref is None:
                    with torch.no_grad():
                        rho_det = rho_c.detach().to(
                            device=self._ema_ic.device, dtype=self._ema_ic.dtype
                        )
                        rho_fin = torch.isfinite(rho_det)
                        rho_safe = torch.where(rho_fin, rho_det, self._ema_ic)
                        is_init = self._ema_ic_initialized > 0.5
                        d_ema = float(self.stab_ema_decay)
                        upd = torch.where(
                            is_init,
                            d_ema * self._ema_ic + (1.0 - d_ema) * rho_safe,
                            rho_safe,
                        )
                        self._ema_ic.copy_(torch.where(rho_fin, upd, self._ema_ic))
                        self._ema_ic_initialized.copy_(
                            torch.where(
                                rho_fin,
                                torch.ones_like(self._ema_ic_initialized),
                                self._ema_ic_initialized,
                            )
                        )
        else:
            ema_ic_det = self._ema_ic.detach().to(device=device, dtype=dtype)
            loss_stab_downside = torch.zeros_like(deep_total)

        # Regularizadores latentes SOTA opt-in (`dl_training_sota.py`: VICReg cov + Rank SupCon)
        if self.w_latent_cov > 0.0 and final_hidden is not None:
            loss_latent_cov = vicreg_covariance_loss(final_hidden).to(dtype=dtype)
        else:
            loss_latent_cov = torch.zeros_like(deep_total)

        if self.w_supcon > 0.0 and final_hidden is not None:
            loss_supcon = cross_sectional_rank_contrastive_loss(final_hidden, y1).to(dtype=dtype)
        else:
            loss_supcon = torch.zeros_like(deep_total)

        # Persistencia anti-churn en `alpha_u` alineada por `asset_ids` (`frontier_modules_sota`)
        if self.w_churn > 0.0 and prev_alpha_u is not None:
            cur_u = final_pred["alpha_u"] if "alpha_u" in final_pred else final_pred["alpha"]
            loss_churn = polar_alpha_persistence_loss(
                cur_u,
                prev_alpha_u.to(device=device, dtype=cur_u.dtype),
                cur_asset_ids=cur_asset_ids,
                prev_asset_ids=prev_asset_ids,
                horizon_mask=horizon_mask,
                w_churn=self.w_churn,
            ).to(dtype=dtype)
        else:
            loss_churn = torch.zeros_like(deep_total)

        # Energía estructural E_fin
        loss_energy = (
            energy_total.to(device=device, dtype=dtype)
            if energy_total is not None
            else torch.zeros_like(deep_total)
        )

        # Decorrelación Bi-Hemisférica de residuos (enmascarada ante NaN/Inf #187)
        if companion_pred is not None:
            comp = companion_pred.to(device=device, dtype=dtype).detach()
            dec_valid = torch.isfinite(y1) & torch.isfinite(comp)
            dec_cnt = dec_valid.to(dtype=dtype).sum().clamp_min(1.0)
            y1_d = torch.where(dec_valid, y1, torch.zeros_like(y1))
            comp_d = torch.where(dec_valid, comp, torch.zeros_like(comp))
            p0_d = torch.where(dec_valid, final_pred["pred_return"][:, 0], torch.zeros_like(y1))
            res_self = torch.where(dec_valid, y1_d - p0_d, torch.zeros_like(y1))
            res_comp = torch.where(dec_valid, y1_d - comp_d, torch.zeros_like(y1))
            rs = torch.where(
                dec_valid, res_self - res_self.sum() / dec_cnt, torch.zeros_like(res_self)
            )
            rc = torch.where(
                dec_valid, res_comp - res_comp.sum() / dec_cnt, torch.zeros_like(res_comp)
            )
            corr_res = (rs * rc).sum() / (rs.norm().clamp_min(1e-6) * rc.norm().clamp_min(1e-6))
            loss_decor = torch.where(
                dec_valid.sum() >= 2, F.relu(corr_res) ** 2, torch.zeros_like(deep_total)
            )
        else:
            loss_decor = torch.zeros_like(deep_total)

        # Pérdidas auxiliares (SSL multimodal, Skew-t NLL, PonderNet KL de Cognitive Core)
        loss_aux = torch.zeros_like(deep_total)
        if aux_losses:
            for aux_val in aux_losses.values():
                if aux_val is not None:
                    loss_aux = loss_aux + aux_val.to(device=device, dtype=dtype)

        total = (
            deep_total
            + self.w_energy * loss_energy
            + self.w_sharpe * loss_sharpe
            + self.w_decor * loss_decor
            + self.w_stab_downside * loss_stab_downside
            + self.w_latent_cov * loss_latent_cov
            + self.w_supcon * loss_supcon
            + loss_churn
            + loss_aux
        )
        return dict(
            total=total,
            deep_supervision=deep_total,
            ic=last_parts["ic"],
            listnet_tail=last_parts["listnet_tail"],
            polar_cal=last_parts["polar_cal"],
            latent_cov=loss_latent_cov,
            supcon=loss_supcon,
            churn=loss_churn,
            rho_ic=rho_s.detach(),
            ema_ic=ema_ic_det.detach(),
            stab_downside=loss_stab_downside,
            crps=last_parts["crps"],
            tb=last_parts["tb"],
            energy=loss_energy,
            sharpe_turnover=loss_sharpe,
            decor=loss_decor,
            aux=loss_aux,
            portfolio_weights=w_t.detach(),
        )
