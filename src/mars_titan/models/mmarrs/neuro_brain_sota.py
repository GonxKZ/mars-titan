"""Familia 2 Independiente de Frontera Neuro-IA: `StandaloneHolographicBrainModel`.

Adapta mecanismos rigurosos de Neurociencia Computacional a Deep Learning nativo en GPU
(NVIDIA RTX PRO 6000 Blackwell `sm_120`, FP32 estricto y `bfloat16`/`float16` seguros vía
`_stable_calc_dtype`, **cero sincronizaciones CPU↔GPU** en `forward`/`step_matured`/pérdida):
  1. `LateralInhibitionCompetition`: normalización divisiva canónica de Carandini & Heeger
     + inhibición lateral cortical suave para suprimir ruido común y decorrelacionar señales.
  2. `DendriticCompartmentLayer`: detección activa de coincidencia dendrítica basal × apical
     (Larkum) en `einsum` 3D fusionado (cero bucles Python) para conmutación rápida de régimen
     sin olvido catastrófico.
  3. `ThreeFactorDopaminePlasticity`: plasticidad neo-Hebbiana de 3 factores (Gerstner/Schultz)
     con trazas de elegibilidad causales en `t` y consolidación estrictamente diferida en `t+1`
     por señal de RPE/IC maduro dentro de la caja contractiva de Lyapunov `PT1`.
  4. `HippocampalSharpWaveReplay`: Sistemas de Aprendizaje Complementario (CLS / SWR) con
     buffer prioritario por sorpresa + `|RPE|` que preserva prototipos de crisis (2008/2020)
     y aporta regularización por repetición contra el olvido catastrófico en `2000–2023`.
  5. `StandaloneHolographicBrainModel`: arquitectura completa independiente de Familia 2
     (por defecto `use_moe_bridge=False`, sin Cognitive Core ni ToolRouter; puente opcional
     `use_moe_bridge=True` reutilizando `HighSNRMoELayer` para ablación controlada `soft` vs
     `uniform` `1/E`) con ciclos contractivos de Codificación Predictiva (`1..3`, tope duro `4`)
     y aceptación monótona sin desfase.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features

from .config import DEFAULT_HORIZONS, DEFAULT_QUANTILES, MOE_MODES
from .dl_training_sota import build_norm_layer
from .frontier_modules_sota import PolarAlphaHead
from .granular_heads import MultiHorizonGranularHead
from .hypergraph_hrr import (
    HolographicBrain3L,
    HRRModernHopfieldMemory,
    RMTSpilloverHypergraph,
)
from .mor_loom_pc import HighSNRMoELayer, _sparsemax

DEFAULT_BRAIN_CYCLES: int = 3
MAX_BRAIN_CYCLES: int = 4

NEURO_BRAIN_ABLATIONS: dict[str, dict[str, object]] = {
    "standalone_k1": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=1,
    ),
    "full": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
    ),
    "no_dendritic": dict(
        use_dendritic=False,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
    ),
    "no_lateral_inhibition": dict(
        use_dendritic=True,
        use_lateral_inhibition=False,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
    ),
    "no_three_factor": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=False,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
    ),
    "no_replay": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=False,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
    ),
    "bare_holographic_3l": dict(
        use_dendritic=False,
        use_lateral_inhibition=False,
        use_three_factor=False,
        use_replay=False,
        use_hrr_hopfield=False,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=1,
    ),
    "holographic_moe_soft_k3": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=True,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
    ),
    "holographic_moe_uniform_k3": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=True,
        moe_mode="uniform",
        max_cycles=DEFAULT_BRAIN_CYCLES,
    ),
    # Familia 2 v2 Computational Neuroscience SOTA (#128 §2 + #133 P0-4 & B)
    "nb_v2_fhrr_lgn_pi": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
        use_unitary_fhrr=True,
        use_lgn_dual_input=True,
        use_active_inference_pi=True,
        use_homeostatic_bcm=False,
        use_stratified_swr=False,
        noic_pinball_twin=False,
    ),
    "nb_v2_full": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
        use_unitary_fhrr=True,
        use_lgn_dual_input=True,
        use_active_inference_pi=True,
        use_homeostatic_bcm=True,
        use_stratified_swr=True,
        noic_pinball_twin=False,
    ),
    "nb_v2_noic_twin": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=False,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
        use_unitary_fhrr=True,
        use_lgn_dual_input=True,
        use_active_inference_pi=True,
        use_homeostatic_bcm=True,
        use_stratified_swr=True,
        noic_pinball_twin=True,
    ),
    "nb_v2_moe_soft_k3": dict(
        use_dendritic=True,
        use_lateral_inhibition=True,
        use_three_factor=True,
        use_replay=True,
        use_hrr_hopfield=True,
        use_moe_bridge=True,
        moe_mode="soft",
        max_cycles=DEFAULT_BRAIN_CYCLES,
        use_unitary_fhrr=True,
        use_lgn_dual_input=True,
        use_active_inference_pi=True,
        use_homeostatic_bcm=True,
        use_stratified_swr=True,
        noic_pinball_twin=False,
    ),
}


def _stable_calc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promociona FP16/BF16 a FP32 en reducciones/log/álgebra lineal y preserva FP32/FP64."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _inv_softplus(value: float) -> float:
    """Inversa numéricamente estable de `softplus` para inicialización positiva."""
    v = max(float(value), 1e-4)
    return v if v > 20.0 else math.log(math.expm1(v))


def _offdiag_abs_corr(x: torch.Tensor) -> torch.Tensor:
    """Media de `|corr_ij|` fuera de la diagonal para `x [N, D]` en `calc_dtype` (sin `.item()`)."""
    calc_dtype = _stable_calc_dtype(x.dtype)
    xc = x.to(dtype=calc_dtype)
    xc = xc - xc.mean(dim=0, keepdim=True)
    xn = xc / xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    n_rows = float(max(x.shape[0], 1))
    corr = (xn.T @ xn) / n_rows
    d_dim = corr.shape[0]
    if d_dim <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    eye_mask = 1.0 - torch.eye(d_dim, device=x.device, dtype=calc_dtype)
    val = (corr.abs() * eye_mask).sum() / float(d_dim * (d_dim - 1))
    return val.to(dtype=x.dtype)


def _fhrr_unitary_pair_norm(b: torch.Tensor) -> torch.Tensor:
    """Normaliza cada par complejo 2D `(re, im)` a módulo unitario (`#133` P0-4, 0 FFTs)."""
    d = b.shape[-1]
    n_pairs = d // 2
    if n_pairs == 0:
        return torch.ones((*b.shape[:-1], 0, 2), device=b.device, dtype=b.dtype)
    calc_dtype = _stable_calc_dtype(b.dtype)
    b_even = b[..., : 2 * n_pairs].to(dtype=calc_dtype)
    b_p = b_even.reshape(*b.shape[:-1], n_pairs, 2)
    sq = (b_p * b_p).sum(dim=-1, keepdim=True)
    norm = torch.sqrt(sq.clamp_min(1e-12))
    b_div = b_p / norm
    ident = torch.zeros_like(b_p)
    ident[..., 0] = 1.0
    b_u = torch.where(sq > 1e-12, b_div, ident)
    return b_u.to(dtype=b.dtype)


def _fhrr_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Enlace holográfico unitario FHRR en pares reales 2D (`O(D)`, 0 `rfft`, Inductor-fusable)."""
    d = a.shape[-1]
    n_pairs = d // 2
    if n_pairs == 0:
        return a
    calc_dtype = _stable_calc_dtype(a.dtype)
    a_c = a.to(dtype=calc_dtype)
    b_c = b.to(device=a.device, dtype=calc_dtype)
    a_p = a_c[..., : 2 * n_pairs].reshape(*a_c.shape[:-1], n_pairs, 2)
    b_u = _fhrr_unitary_pair_norm(b_c[..., : 2 * n_pairs])
    a_r, a_i = a_p[..., 0], a_p[..., 1]
    b_r, b_i = b_u[..., 0], b_u[..., 1]
    out_r = a_r * b_r - a_i * b_i
    out_i = a_r * b_i + a_i * b_r
    out_even = torch.stack([out_r, out_i], dim=-1).reshape(*a_c.shape[:-1], 2 * n_pairs)
    if 2 * n_pairs < d:
        out_c = torch.cat([out_even, a_c[..., 2 * n_pairs :]], dim=-1)
    else:
        out_c = out_even
    return out_c.to(dtype=a.dtype)


def _fhrr_unbind(bound: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    """Desenlace conjugado exacto FHRR (`_fhrr_unbind(_fhrr_bind(a, b), b) == a`, 0 `rfft`)."""
    d = bound.shape[-1]
    n_pairs = d // 2
    if n_pairs == 0:
        return bound
    calc_dtype = _stable_calc_dtype(bound.dtype)
    c_c = bound.to(dtype=calc_dtype)
    r_c = role.to(device=bound.device, dtype=calc_dtype)
    c_p = c_c[..., : 2 * n_pairs].reshape(*c_c.shape[:-1], n_pairs, 2)
    b_u = _fhrr_unitary_pair_norm(r_c[..., : 2 * n_pairs])
    c_r, c_i = c_p[..., 0], c_p[..., 1]
    b_r, b_i = b_u[..., 0], b_u[..., 1]
    out_r = c_r * b_r + c_i * b_i
    out_i = c_i * b_r - c_r * b_i
    out_even = torch.stack([out_r, out_i], dim=-1).reshape(*c_c.shape[:-1], 2 * n_pairs)
    if 2 * n_pairs < d:
        out_c = torch.cat([out_even, c_c[..., 2 * n_pairs :]], dim=-1)
    else:
        out_c = out_even
    return out_c.to(dtype=bound.dtype)


def _lgn_dual_transform(
    raw: torch.Tensor,
    *,
    present_row_mask: torch.Tensor | None = None,
    clip_val: float = 8.0,
) -> torch.Tensor:
    """Canal dual LGN `[clipped_z, masked_gaussian_rank]` (0 syncs, `torch.where` NaN-safe)."""
    calc_dtype = _stable_calc_dtype(raw.dtype)
    raw_c = raw.to(dtype=calc_dtype)
    fin_mask = torch.isfinite(raw_c)
    if present_row_mask is not None:
        row_m = present_row_mask.to(device=raw.device, dtype=torch.bool).unsqueeze(-1)
        valid_mask = fin_mask & row_m
    else:
        valid_mask = fin_mask
    safe_val = torch.where(valid_mask, raw_c, torch.zeros_like(raw_c))
    z_level = torch.where(
        valid_mask, safe_val.clamp(-clip_val, clip_val), torch.zeros_like(safe_val)
    )
    g_rank = gaussian_rank_features(safe_val, mask=valid_mask, clip_val=clip_val)
    return torch.cat([z_level, g_rank], dim=-1).to(dtype=raw.dtype)


class LateralInhibitionCompetition(nn.Module):
    """Normalización divisiva (Carandini & Heeger) + inhibición lateral cortical decorreladora."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        sigma_init: float = 0.5,
        inh_strength_init: float = 0.75,
        surround_strength_init: float = 0.85,
    ) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim >= 4 es obligatorio")
        self.hidden_dim = hidden_dim

        # Parámetros de normalización divisiva canónica
        self.sigma_raw = nn.Parameter(
            torch.tensor(_inv_softplus(sigma_init), dtype=torch.float32)
        )
        self.alpha_feat_raw = nn.Parameter(
            torch.tensor(_inv_softplus(0.5), dtype=torch.float32)
        )
        self.alpha_pop_raw = nn.Parameter(
            torch.tensor(_inv_softplus(0.5), dtype=torch.float32)
        )
        self.beta_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.div_gain = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))

        # Fuerza de inhibición lateral común y de entorno correlacionado
        self.inh_logit = nn.Parameter(
            torch.tensor(
                math.log(inh_strength_init / (1.0 - inh_strength_init)), dtype=torch.float32
            )
        )
        self.surround_logit = nn.Parameter(
            torch.tensor(
                math.log(surround_strength_init / (1.0 - surround_strength_init)),
                dtype=torch.float32,
            )
        )
        self.skip_logit = nn.Parameter(torch.tensor(-2.2, dtype=torch.float32))
        self.register_buffer("eye_d", torch.eye(hidden_dim, dtype=torch.float32))

    def _inhibit_2d(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = x.device, x.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        xc = x.to(dtype=calc_dtype)
        n_assets = x.shape[0]

        # 1) Normalización divisiva de Carandini & Heeger: acota energía local y poblacional
        sigma2 = F.softplus(self.sigma_raw.to(device=device, dtype=calc_dtype)).pow(2) + 1e-4
        a_feat = F.softplus(self.alpha_feat_raw.to(device=device, dtype=calc_dtype))
        a_pop = F.softplus(self.alpha_pop_raw.to(device=device, dtype=calc_dtype))
        beta = 0.35 + 0.30 * torch.sigmoid(self.beta_logit.to(device=device, dtype=calc_dtype))

        local_energy = xc.pow(2).mean(dim=-1, keepdim=True)  # [N, 1]
        pop_energy = local_energy.mean(dim=0, keepdim=True)  # [1, 1] (equivariante!)
        denom = (sigma2 + a_feat * local_energy + a_pop * pop_energy).pow(beta)
        gain = self.div_gain.to(device=device, dtype=calc_dtype).unsqueeze(0)
        y_div = (xc * gain) / denom

        # 2) Inhibición lateral cortical suave (resta modo común + entorno correlacionado)
        inh = torch.sigmoid(self.inh_logit.to(device=device, dtype=calc_dtype))
        s_strength = 0.95 * torch.sigmoid(self.surround_logit.to(device=device, dtype=calc_dtype))
        mu_pop = y_div.mean(dim=0, keepdim=True)
        y_c = y_div - mu_pop

        if n_assets > 1:
            # Matriz de correlación normalizada fuera de diagonal (estrictamente invariante en N)
            y_std = y_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            y_norm = y_c / y_std
            corr = (y_norm.T @ y_norm) / float(n_assets)  # [D, D]
            eye = self.eye_d.to(device=device, dtype=calc_dtype)
            corr_off = corr * (1.0 - eye)
            # Normalización por grado efectivo de competencia lateral para estabilidad garantizada
            deg = corr_off.abs().sum(dim=0, keepdim=True).clamp_min(1.0)
            surround = (y_norm @ (corr_off / deg)) * y_std
            y_inh = y_div - inh * mu_pop - s_strength * surround
        else:
            # Con 1 solo activo, aplica inhibición lateral intra-vector (sin cancelar todo a 0)
            feat_mean = y_div.mean(dim=-1, keepdim=True)
            y_inh = y_div - (0.5 * inh) * feat_mean

        skip = 0.15 * torch.sigmoid(self.skip_logit.to(device=device, dtype=calc_dtype))
        out_c = (1.0 - skip) * y_inh + skip * (xc / denom)
        return out_c.to(dtype=dtype), denom.mean().to(dtype=dtype), y_div, out_c

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        return_diagnostics: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Aplica normalización divisiva + inhibición lateral a `[N, D]` o `[N, A, D]`."""
        compute_corr = (
            (not self.training) if return_diagnostics is None else bool(return_diagnostics)
        )
        if x.ndim == 2:
            out, denom_mean, _, _ = self._inhibit_2d(x)
            if compute_corr:
                corr_in = _offdiag_abs_corr(x)
                corr_out = _offdiag_abs_corr(out)
            else:
                zero_s = torch.zeros((), device=x.device, dtype=x.dtype)
                corr_in = zero_s
                corr_out = zero_s
            return out, dict(
                mean_offdiag_corr_in=corr_in,
                mean_offdiag_corr_out=corr_out,
                divisive_denom_mean=denom_mean,
            )

        if x.ndim == 3:
            device, dtype = x.device, x.dtype
            calc_dtype = _stable_calc_dtype(dtype)
            m_bool = (
                torch.ones(x.shape[:2], device=device, dtype=torch.bool)
                if mask is None
                else (mask.to(device=device) > 0.5)
            )
            m = m_bool.to(dtype=calc_dtype)
            x_c = x.to(dtype=calc_dtype)
            xc = torch.where(m_bool.unsqueeze(-1), x_c, torch.zeros_like(x_c))

            sigma2 = F.softplus(self.sigma_raw.to(device=device, dtype=calc_dtype)).pow(2) + 1e-4
            a_feat = F.softplus(self.alpha_feat_raw.to(device=device, dtype=calc_dtype))
            a_pop = F.softplus(self.alpha_pop_raw.to(device=device, dtype=calc_dtype))
            beta = 0.35 + 0.30 * torch.sigmoid(self.beta_logit.to(device=device, dtype=calc_dtype))

            # Energía por área y energía poblacional entre áreas presentes (seguro ante NaN/Inf!)
            area_e = xc.pow(2).mean(dim=-1, keepdim=True)  # [N, A, 1]
            m_sum = m.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
            pop_e = (
                torch.where(m_bool.unsqueeze(-1), area_e, torch.zeros_like(area_e)).sum(
                    dim=1, keepdim=True
                )
                / m_sum
            )
            denom = (sigma2 + a_feat * area_e + a_pop * pop_e).pow(beta)
            gain = self.div_gain.to(device=device, dtype=calc_dtype).view(1, 1, -1)
            y_div = (xc * gain) / denom

            # Inhibición lateral entre áreas presentes (resta media común de otras áreas)
            inh = torch.sigmoid(self.inh_logit.to(device=device, dtype=calc_dtype))
            area_mean = (
                torch.where(m_bool.unsqueeze(-1), y_div, torch.zeros_like(y_div)).sum(
                    dim=1, keepdim=True
                )
                / m_sum
            )
            y_inh = y_div - inh * area_mean
            skip = 0.15 * torch.sigmoid(self.skip_logit.to(device=device, dtype=calc_dtype))
            out_raw = (1.0 - skip) * y_inh + skip * (xc / denom)
            out = torch.where(m_bool.unsqueeze(-1), out_raw, torch.zeros_like(out_raw)).to(
                dtype=dtype
            )
            if compute_corr:
                flat_in = xc.reshape(-1, self.hidden_dim)
                flat_out = out.reshape(-1, self.hidden_dim)
                corr_in = _offdiag_abs_corr(flat_in)
                corr_out = _offdiag_abs_corr(flat_out)
            else:
                zero_s = torch.zeros((), device=device, dtype=dtype)
                corr_in = zero_s
                corr_out = zero_s
            return out, dict(
                mean_offdiag_corr_in=corr_in,
                mean_offdiag_corr_out=corr_out,
                divisive_denom_mean=denom.mean().to(dtype=dtype),
            )

        raise ValueError(f"LateralInhibitionCompetition espera tensor 2D o 3D, recibido {x.ndim}D")


class DendriticCompartmentLayer(nn.Module):
    """Detección activa de coincidencia dendrítica basal × apical (Larkum) en `einsum` 3D."""

    def __init__(
        self,
        hidden_dim: int,
        apical_dim: int,
        *,
        n_branches: int = 8,
        branch_dim: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or apical_dim < 1 or n_branches < 2:
            raise ValueError("hidden_dim >= 8, apical_dim >= 1 y n_branches >= 2 son obligatorios")
        self.hidden_dim = hidden_dim
        self.apical_dim = apical_dim
        self.n_branches = n_branches
        self.branch_dim = branch_dim if branch_dim is not None else max(8, hidden_dim // 2)
        self.temperature = float(temperature)

        self.basal_norm = nn.LayerNorm(hidden_dim)
        # Escala RMS suave que preserva signo y nivel medio de shocks de régimen (sin borrar DC!)
        self.apical_scale = nn.Parameter(torch.ones(apical_dim, dtype=torch.float32))

        # Ramas dendríticas basales almacenadas como tensores 3D contiguos (0 bucles en sm_120!)
        b_scale = 1.0 / math.sqrt(float(hidden_dim))
        s_scale = 1.0 / math.sqrt(float(self.branch_dim * n_branches))
        self.basal_w = nn.Parameter(torch.randn(n_branches, hidden_dim, self.branch_dim) * b_scale)
        self.basal_b = nn.Parameter(torch.zeros(n_branches, self.branch_dim, dtype=torch.float32))
        self.soma_w = nn.Parameter(torch.randn(n_branches, self.branch_dim, hidden_dim) * s_scale)
        self.soma_b = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        # Penacho apical: genera compuerta de coincidencia Ca2+ y ganancia por rama
        ap_hid = max(16, min(64, n_branches * 4))
        self.apical_gate_net = nn.Sequential(
            nn.Linear(apical_dim, ap_hid),
            nn.SiLU(),
            nn.Linear(ap_hid, n_branches),
        )
        self.apical_mod_net = nn.Sequential(
            nn.Linear(apical_dim, ap_hid),
            nn.SiLU(),
            nn.Linear(ap_hid, n_branches),
        )
        self.soma_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        apical_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Ejecuta coincidencia basal × apical en 2 `einsum` fusionados sin bucles Python."""
        n_assets = x.shape[0]
        device, dtype = x.device, x.dtype
        ap = apical_ctx.to(device=device, dtype=dtype)
        if ap.ndim == 1:
            ap = ap.unsqueeze(0).expand(n_assets, -1)

        h_basal = self.basal_norm(x)
        calc_dtype = _stable_calc_dtype(dtype)
        ap_c = ap.to(dtype=calc_dtype)
        rms_inv = torch.rsqrt(ap_c.pow(2).mean(dim=-1, keepdim=True) + 1.0)
        ap_n = (ap_c * rms_inv).to(dtype=dtype) * self.apical_scale.to(device=device, dtype=dtype)

        # 1) Ramas basales en 1 sola contracción 3D: [N, B, K] (seguro bajo autocast BF16!)
        b_dtype = h_basal.dtype
        basal_act = F.silu(
            torch.einsum("nd,bdk->nbk", h_basal, self.basal_w.to(dtype=b_dtype))
            + self.basal_b.to(dtype=b_dtype).unsqueeze(0)
        )

        # 2) Compuerta de coincidencia Ca2+ del penacho apical + modulación por rama: [N, B]
        gate_logits = self.apical_gate_net(ap_n) / self.temperature
        apical_gates = torch.sigmoid(gate_logits).to(dtype=b_dtype)
        apical_mod = (1.0 + 0.5 * torch.tanh(self.apical_mod_net(ap_n))).to(dtype=b_dtype)
        coincidence = basal_act * (apical_gates * apical_mod).unsqueeze(-1)

        # 3) Integración somática en 1 contracción 3D: [N, D]
        soma_delta = (
            torch.einsum("nbk,bkd->nd", coincidence, self.soma_w.to(dtype=b_dtype))
            + self.soma_b.to(dtype=b_dtype)
        )
        out = x + self.soma_norm(soma_delta).to(dtype=dtype)

        diag = dict(
            apical_gates=apical_gates,
            active_branches_mean=apical_gates.sum(dim=-1).mean(),
            branch_sparsity=(apical_gates < 0.1).to(dtype=dtype).mean(),
        )
        return out, diag


class TurrigianoHomeostaticScaling(nn.Module):
    """Escalado homeostático sináptico de Turrigiano causal `t-1` (0 mutación en `forward`)."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        target_rate: float = 1.0,
        rate_decay: float = 0.98,
        gain_min: float = 0.5,
        gain_max: float = 2.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim >= 4 es obligatorio")
        self.hidden_dim = int(hidden_dim)
        self.target_rate = float(target_rate)
        self.rate_decay = float(rate_decay)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        # kappa in (0, 0.5), init ~0.35 para amortiguar colapso/saturación en crisis
        self.kappa_logit = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        rate_ema_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Escala `x [N, D]` con tasa causal `<= t-1` y devuelve `(out, next_rate_ema)`."""
        device, dtype = x.device, x.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        if rate_ema_prev is None:
            r_prev = torch.ones((self.hidden_dim,), device=device, dtype=calc_dtype)
        else:
            r_raw = rate_ema_prev.detach().to(device=device, dtype=calc_dtype)
            r_prev = torch.where(
                torch.isfinite(r_raw), r_raw.clamp_min(1e-3), torch.ones_like(r_raw)
            )
        kappa = 0.5 * torch.sigmoid(self.kappa_logit.to(device=device, dtype=calc_dtype))
        gain = torch.clamp(
            (self.target_rate / r_prev).pow(kappa), min=self.gain_min, max=self.gain_max
        )
        out = (x.to(dtype=calc_dtype) * gain.unsqueeze(0)).to(dtype=dtype)

        x_det = x.detach().to(dtype=calc_dtype)
        x_fin = torch.where(torch.isfinite(x_det), x_det, torch.zeros_like(x_det))
        cur_rms = torch.sqrt(x_fin.pow(2).mean(dim=0) + 1e-6)
        next_rate_ema = (self.rate_decay * r_prev + (1.0 - self.rate_decay) * cur_rms).detach()
        return out, next_rate_ema


class ActiveInferencePrecisionGate(nn.Module):
    """Precisión sensorial por activo y canal `Pi_{i,t} in [0.25, 1.0]` (Banach `< 0.90`)."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        n_modalities: int = 5,
        pi_min: float = 0.25,
        pi_max: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 4 or context_dim < 1:
            raise ValueError("hidden_dim >= 4 y context_dim >= 1 son obligatorios")
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.n_modalities = int(n_modalities)
        self.pi_min = float(pi_min)
        self.pi_max = float(pi_max)

        in_dim = self.context_dim + 3
        hid = max(16, min(64, self.hidden_dim // 2))
        self.age_decay = nn.Parameter(torch.zeros((self.n_modalities,), dtype=torch.float32))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, self.hidden_dim),
        )
        nn.init.normal_(self.net[-1].weight, std=0.05)
        nn.init.constant_(self.net[-1].bias, 0.5)

    def forward(
        self,
        *,
        mod_mask: torch.Tensor,
        ctx: torch.Tensor,
        ood_energy: torch.Tensor,
        pc_err_var_ema: torch.Tensor,
        modality_age: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calcula `Pi_{i,t} [N, D]` estrictamente en `[pi_min, pi_max]` sin mutar buffers."""
        device, dtype = ctx.device, ctx.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        m_c = mod_mask.to(device=device, dtype=calc_dtype)
        m_bool = m_c > 0.5
        mod_frac = m_c.mean(dim=-1, keepdim=True)  # [N, 1]

        if modality_age is None:
            age_c = torch.zeros_like(m_c)
        else:
            age_raw = modality_age.detach().to(device=device, dtype=calc_dtype)
            age_c = torch.where(
                torch.isfinite(age_raw), age_raw.clamp(0.0, 252.0), torch.zeros_like(age_raw)
            )
        decay = F.softplus(self.age_decay.to(device=device, dtype=calc_dtype)).unsqueeze(0)
        fresh_m = torch.exp(-decay * age_c)
        m_cnt = m_c.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_fresh = (
            torch.where(m_bool, fresh_m, torch.zeros_like(fresh_m)).sum(dim=-1, keepdim=True)
            / m_cnt
        )

        ood_c = ood_energy.detach().to(device=device, dtype=calc_dtype)
        ood_safe = torch.where(torch.isfinite(ood_c), ood_c.clamp_min(0.0), torch.zeros_like(ood_c))
        log_ood = torch.log1p(ood_safe).unsqueeze(-1)

        ctx_c = ctx.to(dtype=calc_dtype)
        ctx_safe = torch.where(
            torch.isfinite(ctx_c), ctx_c.clamp(-8.0, 8.0), torch.zeros_like(ctx_c)
        )

        phi = torch.cat([mod_frac, mean_fresh, log_ood, ctx_safe], dim=-1)
        net_dtype = self.net[1].weight.dtype
        logits = self.net(phi.to(dtype=net_dtype)).to(dtype=calc_dtype)
        g_asset = torch.sigmoid(logits)

        var_raw = pc_err_var_ema.detach().to(device=device, dtype=calc_dtype)
        var_ema = torch.where(
            torch.isfinite(var_raw), var_raw.clamp_min(1e-6), torch.ones_like(var_raw)
        )
        var_med = var_ema.median()
        pi_ch = torch.clamp(var_med / var_ema, min=self.pi_min, max=self.pi_max)

        pi_eff = self.pi_min + (self.pi_max - self.pi_min) * (g_asset * pi_ch.unsqueeze(0))
        diag = dict(
            pi_mean=pi_eff.mean().to(dtype=dtype),
            pi_min=pi_eff.min().to(dtype=dtype),
            pi_max=pi_eff.max().to(dtype=dtype),
            pi_std=pi_eff.std(unbiased=False).to(dtype=dtype),
            pi_ch=pi_ch.detach().to(dtype=dtype),
        )
        return pi_eff.to(dtype=dtype), diag


class ThreeFactorDopaminePlasticity(nn.Module):
    """Plasticidad neo-Hebbiana de 3 factores (Gerstner/Schultz) bajo cota Lyapunov PT1."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        plasticity_rank: int = 24,
        eligibility_decay: float = 0.85,
        rpe_baseline_decay: float = 0.90,
        pt1_alpha_lo: float = 0.05,
        pt1_theta_max: float = 0.25,
        pt1_w_max: float = 4.0,
        huber_delta: float = 1.0,
        use_homeostatic_bcm: bool = False,
        bcm_decay: float = 0.90,
        tag_decay: float = 0.80,
        tag_threshold: float = 0.30,
        tag_temperature: float = 0.12,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or plasticity_rank < 4:
            raise ValueError("hidden_dim >= 8 y plasticity_rank >= 4 son obligatorios")
        if not (0.0 < pt1_alpha_lo < 1.0) or not (0.0 < pt1_theta_max <= 1.0):
            raise ValueError("pt1_alpha_lo in (0,1) y pt1_theta_max in (0,1] son obligatorios")
        if pt1_w_max <= 0.0 or huber_delta <= 0.0:
            raise ValueError("pt1_w_max y huber_delta deben ser estrictamente positivos")

        self.hidden_dim = hidden_dim
        self.plasticity_rank = int(plasticity_rank)
        self.eligibility_decay = float(eligibility_decay)
        self.rpe_baseline_decay = float(rpe_baseline_decay)
        self.pt1_alpha_lo = float(pt1_alpha_lo)
        self.pt1_theta_max = float(pt1_theta_max)
        self.pt1_w_max = float(pt1_w_max)
        self.huber_delta = float(huber_delta)
        self.use_homeostatic_bcm = bool(use_homeostatic_bcm)
        self.bcm_decay = float(bcm_decay)
        self.tag_decay = float(tag_decay)
        self.tag_threshold = float(tag_threshold)
        self.tag_temperature = float(tag_temperature)

        # Codificadores pre/post sinápticos compartidos entre lectura (t) y traza (t)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.pre_proj = nn.Linear(hidden_dim, self.plasticity_rank)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.post_proj = nn.Linear(hidden_dim, self.plasticity_rank)
        self.out_proj = nn.Linear(self.plasticity_rank, hidden_dim)
        nn.init.normal_(self.out_proj.weight, std=0.05)
        nn.init.zeros_(self.out_proj.bias)
        # Escala sináptica lenta base (`W_eff = base_scale * I + W_fast`): gradientes vivos paso 0!
        self.base_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def init_state(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Inicializa pesos rápidos `[R, R]`, traza de elegibilidad `[R, R]` y línea base RPE."""
        calc_dtype = _stable_calc_dtype(dtype)
        r = self.plasticity_rank
        st = dict(
            fast_weights=torch.zeros((r, r), device=device, dtype=calc_dtype),
            eligibility=torch.zeros((r, r), device=device, dtype=calc_dtype),
            rpe_baseline=torch.zeros((), device=device, dtype=calc_dtype),
            last_pre=torch.zeros((1, r), device=device, dtype=calc_dtype),
            last_post=torch.zeros((1, r), device=device, dtype=calc_dtype),
            last_asset_ids=torch.zeros((0,), device=device, dtype=torch.long),
            has_last_pair=torch.zeros((), device=device, dtype=torch.bool),
        )
        if self.use_homeostatic_bcm:
            st["bcm_theta"] = torch.zeros((r,), device=device, dtype=calc_dtype)
            st["bcm_initialized"] = torch.zeros((), device=device, dtype=torch.bool)
            st["w_tag"] = torch.zeros((r, r), device=device, dtype=calc_dtype)
            st["post_rate_ema"] = torch.ones((r,), device=device, dtype=calc_dtype)
            st["pending_post_rate_ema"] = torch.ones((r,), device=device, dtype=calc_dtype)
        return st

    def read(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Lee `W_slow + W_fast` maduro de `<= t-1` (invariante a `N` y gradientes vivos paso 0)."""
        device, dtype = x.device, x.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        w_fast = (
            torch.zeros(
                (self.plasticity_rank, self.plasticity_rank),
                device=device,
                dtype=calc_dtype,
            )
            if state is None
            else state["fast_weights"].to(device=device, dtype=calc_dtype)
        )
        u = F.silu(self.pre_proj(self.pre_norm(x)))
        post_gate = torch.sigmoid(self.post_proj(self.post_norm(x)))
        u_c = u.to(dtype=calc_dtype)
        base_s = self.base_scale.to(device=device, dtype=calc_dtype)
        v_eff_c = u_c @ w_fast + base_s * u_c
        if self.use_homeostatic_bcm:
            if state is not None and "post_rate_ema" in state:
                r_raw = state["post_rate_ema"].detach().to(device=device, dtype=calc_dtype)
                r_prev = torch.where(
                    torch.isfinite(r_raw), r_raw.clamp_min(0.1), torch.ones_like(r_raw)
                )
            else:
                r_prev = torch.ones((self.plasticity_rank,), device=device, dtype=calc_dtype)
            g_syn = torch.clamp((1.0 / r_prev).pow(0.25), min=0.5, max=2.0)
            v_eff_c = v_eff_c * g_syn.unsqueeze(0)
        v_eff = v_eff_c.to(dtype=post_gate.dtype)
        delta = self.out_proj(v_eff * post_gate).to(dtype=dtype)
        out = x + delta
        w_norm = torch.linalg.matrix_norm(w_fast, ord="fro").to(dtype=dtype)
        diag = dict(fast_weight_norm=w_norm)
        if self.use_homeostatic_bcm:
            if state is not None and "w_tag" in state:
                w_tag_n = torch.linalg.matrix_norm(
                    state["w_tag"].to(device=device, dtype=calc_dtype), ord="fro"
                )
                theta_m = state["bcm_theta"].to(device=device, dtype=calc_dtype).mean()
            else:
                w_tag_n = torch.zeros((), device=device, dtype=calc_dtype)
                theta_m = torch.zeros((), device=device, dtype=calc_dtype)
            diag["w_tag_norm"] = w_tag_n.to(dtype=dtype)
            diag["bcm_theta_mean"] = theta_m.to(dtype=dtype)
        return out, diag

    @torch.no_grad()
    def record_eligibility(
        self,
        x_pre: torch.Tensor,
        h_post: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
        *,
        asset_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Actualiza la traza causal de elegibilidad en `t` SIN modificar `fast_weights`."""
        device, dtype = x_pre.device, x_pre.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        st = (
            self.init_state(device=device, dtype=calc_dtype)
            if state is None
            else {k: v.clone() for k, v in state.items()}
        )
        p_dtype = self.pre_proj.weight.dtype
        x_in = x_pre.detach().to(device=device, dtype=p_dtype)
        h_in = h_post.detach().to(device=device, dtype=p_dtype)
        x_safe = torch.where(torch.isfinite(x_in), x_in, torch.zeros_like(x_in))
        h_safe = torch.where(torch.isfinite(h_in), h_in, torch.zeros_like(h_in))
        u = F.silu(self.pre_proj(self.pre_norm(x_safe))).to(dtype=calc_dtype)
        v = F.silu(self.post_proj(self.post_norm(h_safe))).to(dtype=calc_dtype)
        u_n = F.normalize(u, dim=-1, eps=1e-6)
        v_n = F.normalize(v, dim=-1, eps=1e-6)

        n_assets = float(max(u_n.shape[0], 1))
        if self.use_homeostatic_bcm:
            r = self.plasticity_rank
            theta_prev = st.get(
                "bcm_theta", torch.zeros((r,), device=device, dtype=calc_dtype)
            ).to(device=device, dtype=calc_dtype)
            bcm_init = st.get(
                "bcm_initialized", torch.zeros((), device=device, dtype=torch.bool)
            ).to(device=device)
            w_tag_prev = st.get(
                "w_tag", torch.zeros((r, r), device=device, dtype=calc_dtype)
            ).to(device=device, dtype=calc_dtype)
            post_rate_prev = st.get(
                "post_rate_ema", torch.ones((r,), device=device, dtype=calc_dtype)
            ).to(device=device, dtype=calc_dtype)

            # Umbral deslizante BCM estrictamente causal t-1 (desacoplado, compile-safe!)
            v_eff_post = v_n - theta_prev.unsqueeze(0)
            hebb_outer = (u_n.T @ v_eff_post) / n_assets
            hebb_clip = torch.clamp(hebb_outer, min=-self.huber_delta, max=self.huber_delta)

            dec = self.eligibility_decay
            st["eligibility"] = dec * st["eligibility"].to(device=device, dtype=calc_dtype) + (
                1.0 - dec
            ) * hebb_clip
            w_tag_new = self.tag_decay * w_tag_prev + (1.0 - self.tag_decay) * hebb_clip
            tag_frob = torch.linalg.matrix_norm(w_tag_new, ord="fro")
            tag_scale = torch.clamp(self.pt1_w_max / tag_frob.clamp_min(1e-6), max=1.0)
            st["w_tag"] = w_tag_new * tag_scale

            v_mean_t = v_n.mean(dim=0)
            theta_next = torch.where(
                bcm_init,
                self.bcm_decay * theta_prev + (1.0 - self.bcm_decay) * v_mean_t,
                v_mean_t,
            )
            st["bcm_theta"] = theta_next.detach()
            st["bcm_initialized"] = torch.ones((), device=device, dtype=torch.bool)
            cur_post_rms = torch.sqrt(v.pow(2).mean(dim=0) + 1e-6)
            st["pending_post_rate_ema"] = (0.98 * post_rate_prev + 0.02 * cur_post_rms).detach()
            st["post_rate_ema"] = post_rate_prev
            st["last_pre"] = u_n
            st["last_post"] = v_eff_post
        else:
            hebb_outer = (u_n.T @ v_n) / n_assets  # [R, R], invariante por permutación en N!
            hebb_clip = torch.clamp(hebb_outer, min=-self.huber_delta, max=self.huber_delta)
            dec = self.eligibility_decay
            st["eligibility"] = dec * st["eligibility"].to(device=device, dtype=calc_dtype) + (
                1.0 - dec
            ) * hebb_clip
            st["last_pre"] = u_n
            st["last_post"] = v_n

        if asset_ids is not None and asset_ids.numel() == u_n.shape[0]:
            st["last_asset_ids"] = asset_ids.detach().to(device=device, dtype=torch.long).reshape(
                -1
            )
        else:
            st["last_asset_ids"] = torch.zeros((0,), device=device, dtype=torch.long)
        st["has_last_pair"] = torch.ones((), device=device, dtype=torch.bool)
        return st

    @torch.no_grad()
    def consolidate(
        self,
        state: dict[str, torch.Tensor],
        *,
        matured_rpe: torch.Tensor | None = None,
        matured_ic: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Consolida `W_fast` en `t+1` con alineación canónica `asset_ids` dentro de PT1."""
        st = {k: v.clone() for k, v in state.items()}
        w_fast = st["fast_weights"]
        device, calc_dtype = w_fast.device, w_fast.dtype
        elig = st["eligibility"].to(device=device, dtype=calc_dtype)
        baseline = st["rpe_baseline"].to(device=device, dtype=calc_dtype)

        # 1) Señal dopaminérgica RPE = recompensa madura - expectativa (inmune a NaN/Inf!)
        if matured_ic is not None:
            ic_raw = matured_ic.detach().to(device=device, dtype=calc_dtype)
            ic_fin = torch.isfinite(ic_raw)
            ic_safe = torch.where(ic_fin, ic_raw, torch.zeros_like(ic_raw))
            reward_scalar = ic_safe.sum() / ic_fin.to(dtype=calc_dtype).sum().clamp_min(1.0)
        elif matured_rpe is not None:
            rpe_raw = matured_rpe.detach().to(device=device, dtype=calc_dtype)
            rpe_fin = torch.isfinite(rpe_raw)
            rpe_safe = torch.where(rpe_fin, rpe_raw, torch.zeros_like(rpe_raw))
            reward_scalar = rpe_safe.sum() / rpe_fin.to(dtype=calc_dtype).sum().clamp_min(1.0)
        else:
            reward_scalar = torch.zeros((), device=device, dtype=calc_dtype)

        da_scalar = torch.tanh((reward_scalar - baseline) * 5.0)
        new_baseline = (
            self.rpe_baseline_decay * baseline + (1.0 - self.rpe_baseline_decay) * reward_scalar
        )

        # 2) Término de 3 factores: etiqueta sináptica W_tag (v2) o elegibilidad (v1) + asset_ids
        if self.use_homeostatic_bcm and "w_tag" in st:
            trace_src = st["w_tag"].to(device=device, dtype=calc_dtype)
        else:
            trace_src = elig
        delta_trace = trace_src * da_scalar
        last_ids_raw = st.get("last_asset_ids")
        use_id_align = (
            matured_rpe is not None
            and matured_rpe.shape[0] > 0
            and asset_ids is not None
            and asset_ids.numel() == matured_rpe.shape[0]
            and last_ids_raw is not None
            and last_ids_raw.numel() == st["last_pre"].shape[0]
            and last_ids_raw.numel() > 0
        )
        if use_id_align:
            assert matured_rpe is not None and asset_ids is not None and last_ids_raw is not None
            l_ids = last_ids_raw.to(device=device, dtype=torch.long).reshape(-1)
            order = torch.argsort(l_ids)
            sorted_l_ids = l_ids[order]
            u_sorted = st["last_pre"].to(device=device, dtype=calc_dtype)[order]
            v_sorted = st["last_post"].to(device=device, dtype=calc_dtype)[order]

            cur_ids = asset_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
            pos = torch.searchsorted(sorted_l_ids, cur_ids)
            safe_pos = pos.clamp_max(sorted_l_ids.shape[0] - 1)
            matched = (pos < sorted_l_ids.shape[0]) & (sorted_l_ids[safe_pos] == cur_ids)

            rpe_in = matured_rpe.detach().to(device=device, dtype=calc_dtype).reshape(-1)
            valid_mask = matched & torch.isfinite(rpe_in)
            rpe_vec = torch.where(valid_mask, torch.tanh(rpe_in), torch.zeros_like(rpe_in))
            valid_n = valid_mask.to(dtype=calc_dtype).sum().clamp_min(1.0)

            m_col = valid_mask.unsqueeze(-1)
            u_g = u_sorted[safe_pos]
            v_g = v_sorted[safe_pos]
            u_al = torch.where(m_col, u_g, torch.zeros_like(u_g))
            v_al = torch.where(m_col, v_g, torch.zeros_like(v_g))
            delta_asset = ((u_al * rpe_vec.unsqueeze(-1)).T @ v_al) / valid_n
            has_pair_b = st["has_last_pair"] & valid_mask.any()
            delta_3f = torch.where(has_pair_b, 0.5 * delta_trace + 0.5 * delta_asset, delta_trace)
        elif (
            matured_rpe is not None
            and st["last_pre"].shape[0] == matured_rpe.shape[0]
            and matured_rpe.shape[0] > 0
        ):
            rpe_in = matured_rpe.detach().to(device=device, dtype=calc_dtype)
            rpe_fin_v = torch.isfinite(rpe_in)
            rpe_vec = torch.where(rpe_fin_v, torch.tanh(rpe_in), torch.zeros_like(rpe_in))
            valid_n = rpe_fin_v.to(dtype=calc_dtype).sum().clamp_min(1.0)
            u_last = st["last_pre"].to(device=device, dtype=calc_dtype)
            v_last = st["last_post"].to(device=device, dtype=calc_dtype)
            delta_asset = ((u_last * rpe_vec.unsqueeze(-1)).T @ v_last) / valid_n
            has_pair_b = st["has_last_pair"] & rpe_fin_v.any()
            delta_3f = torch.where(has_pair_b, 0.5 * delta_trace + 0.5 * delta_asset, delta_trace)
        else:
            delta_3f = delta_trace

        # 3) Caja contractiva PT1 + captura tardía W_tag (Frey-Morris) + bola Frobenius W_max
        surprise_mag = da_scalar.abs()
        alpha = torch.clamp(
            self.pt1_alpha_lo + 0.20 * surprise_mag, min=self.pt1_alpha_lo, max=0.95
        )
        theta_base = torch.clamp(
            self.pt1_theta_max * (0.4 + 0.6 * surprise_mag),
            min=1e-4,
            max=self.pt1_theta_max,
        )
        if self.use_homeostatic_bcm:
            g_tag = torch.sigmoid(
                (surprise_mag - self.tag_threshold) / max(self.tag_temperature, 1e-3)
            )
            theta = torch.clamp(theta_base * g_tag, min=1e-5, max=self.pt1_theta_max)
        else:
            g_tag = torch.ones((), device=device, dtype=calc_dtype)
            theta = theta_base

        delta_clip = torch.clamp(delta_3f, min=-self.huber_delta, max=self.huber_delta)
        w_new = (1.0 - alpha) * w_fast + theta * delta_clip

        frob = torch.linalg.matrix_norm(w_new, ord="fro")
        scale = torch.clamp(self.pt1_w_max / frob.clamp_min(1e-6), max=1.0)
        w_new = w_new * scale

        st["fast_weights"] = w_new
        st["rpe_baseline"] = new_baseline
        st["eligibility"] = elig * 0.5
        if self.use_homeostatic_bcm:
            if "w_tag" in st:
                st["w_tag"] = st["w_tag"].to(device=device, dtype=calc_dtype) * (1.0 - 0.5 * g_tag)
            if "pending_post_rate_ema" in st:
                st["post_rate_ema"] = st["pending_post_rate_ema"].to(
                    device=device, dtype=calc_dtype
                )
        final_norm = torch.linalg.matrix_norm(w_new, ord="fro")
        diag = dict(
            alpha=alpha,
            theta=theta,
            dopamine_rpe=da_scalar,
            fast_weight_norm=final_norm,
        )
        if self.use_homeostatic_bcm:
            diag["tag_capture_gate"] = g_tag
            if "w_tag" in st:
                diag["w_tag_norm"] = torch.linalg.matrix_norm(st["w_tag"], ord="fro")
            if "bcm_theta" in st:
                diag["bcm_theta_mean"] = st["bcm_theta"].mean()
        return st, diag


class HippocampalSharpWaveReplay(nn.Module):
    """Buffer hipocampal prioritario (CLS / SWR) anti-olvido de regímenes de crisis en GPU."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        n_horizons: int = 6,
        capacity: int = 64,
        temperature: float = 1.0,
        use_stratified_swr: bool = False,
        dedup_cos_thresh: float = 0.85,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or capacity < 2:
            raise ValueError("hidden_dim >= 8 y capacity >= 2 son obligatorios")
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.n_horizons = n_horizons
        self.capacity = int(capacity)
        self.temperature = float(temperature)
        self.use_stratified_swr = bool(use_stratified_swr)
        self.dedup_cos_thresh = float(dedup_cos_thresh)

        # 3 estratos 2:1:1 (exacto 32/16/16 en capacity=64): crisis / sorpresa / calma
        self.cap_crisis = max(1, (self.capacity * 2) // 4)
        rem_cap = self.capacity - self.cap_crisis
        self.cap_surprise = max(1, rem_cap // 2) if rem_cap > 0 else 0
        self.cap_calm = max(0, self.capacity - self.cap_crisis - self.cap_surprise)

        self.register_buffer("proto_latent", torch.zeros(self.capacity, hidden_dim))
        self.register_buffer("proto_context", torch.zeros(self.capacity, context_dim))
        self.register_buffer("proto_target", torch.zeros(self.capacity, n_horizons))
        self.register_buffer(
            "proto_target_mask", torch.zeros(self.capacity, n_horizons, dtype=torch.bool)
        )
        self.register_buffer("proto_priority", torch.zeros(self.capacity))
        self.register_buffer("proto_filled", torch.zeros(self.capacity, dtype=torch.bool))
        # Nombre `write_ptr` alineado con `_EPISODIC_BUFFER_SUFFIXES` (auto-reset sin fugas!)
        self.register_buffer("write_ptr", torch.zeros(3, dtype=torch.long))

        # Cebado episódico ligero durante forward (con identidad exacta cuando el buffer está vacío)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.out_proj.weight, std=0.05)
        nn.init.zeros_(self.out_proj.bias)
        self.replay_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def _store_stratified_zero_sync(
        self,
        *,
        lat_safe: torch.Tensor,
        ctx_safe: torch.Tensor,
        tgt_safe: torch.Tensor,
        tgt_fin: torch.Tensor,
        surp_vec: torch.Tensor,
        rpe_vec: torch.Tensor,
        asset_valid: torch.Tensor,
        is_2d: bool,
        top_k: int | None,
    ) -> torch.Tensor:
        """Escritura 3-estratos (32/16/16) 100% branchless `torch.where` con dedup coseno."""
        device = self.proto_latent.device
        buf_dtype = self.proto_latent.dtype
        calc_dtype = _stable_calc_dtype(buf_dtype)
        c0 = self.cap_crisis
        c1 = c0 + self.cap_surprise
        n_items = lat_safe.shape[0]

        # 1) Estrato 0 [0:c0] — Crisis por prioridad con dedup coseno branchless `torch.where`
        asset_prio = surp_vec + rpe_vec + 1e-3
        sel_cr = torch.where(asset_valid, asset_prio, torch.full_like(asset_prio, -1e9))
        k_req = min(4, max(1, c0 // 2)) if top_k is None else max(1, int(top_k))
        k_cr = min(k_req, n_items, c0)
        _, cr_idx = torch.topk(sel_cr, k=k_cr, dim=0)
        cand_lat = lat_safe[cr_idx]
        cand_ctx = ctx_safe[cr_idx]
        cand_tgt = tgt_safe[cr_idx]
        cand_tmask = tgt_fin[cr_idx]
        cand_prio = asset_prio[cr_idx].to(dtype=buf_dtype)
        cand_valid = asset_valid[cr_idx]

        cr_lat = self.proto_latent[:c0]
        cr_ctx = self.proto_context[:c0]
        cr_tgt = self.proto_target[:c0]
        cr_tmask = self.proto_target_mask[:c0]
        cr_prio = self.proto_priority[:c0]
        cr_filled = self.proto_filled[:c0]

        cand_u = F.normalize(cand_lat.to(dtype=calc_dtype), dim=-1, eps=1e-6)
        slot_u = F.normalize(cr_lat.to(dtype=calc_dtype), dim=-1, eps=1e-6)
        sim_mat = cand_u @ slot_u.T  # [k_cr, c0]
        sim_masked = torch.where(cr_filled.unsqueeze(0), sim_mat, torch.full_like(sim_mat, -2.0))

        # Si un duplicado tiene mayor prioridad, refresca su ranura existente sin crear copia
        dup_mat = cand_valid.unsqueeze(1) & (sim_masked >= self.dedup_cos_thresh)
        dup_scores = torch.where(
            dup_mat,
            cand_prio.to(dtype=calc_dtype).unsqueeze(1),
            torch.full((k_cr, c0), -1e9, device=device, dtype=calc_dtype),
        )
        best_dup_prio, best_dup_k = dup_scores.max(dim=0)
        do_upgrade = cr_filled & (best_dup_prio > cr_prio.to(dtype=calc_dtype))
        up_m2 = do_upgrade.unsqueeze(-1)
        cr_lat_u = torch.where(up_m2, cand_lat[best_dup_k], cr_lat)
        cr_ctx_u = torch.where(up_m2, cand_ctx[best_dup_k], cr_ctx)
        cr_tgt_u = torch.where(up_m2, cand_tgt[best_dup_k], cr_tgt)
        cr_tmask_u = torch.where(up_m2, cand_tmask[best_dup_k], cr_tmask)
        cr_prio_u = torch.where(do_upgrade, cand_prio[best_dup_k], cr_prio)

        max_sim = sim_masked.max(dim=-1).values
        if k_cr > 1:
            self_sim = cand_u @ cand_u.T
            lower_tri = torch.tril(
                torch.ones((k_cr, k_cr), device=device, dtype=torch.bool), diagonal=-1
            )
            intra_max = torch.where(
                lower_tri, self_sim, torch.full_like(self_sim, -2.0)
            ).max(dim=-1).values
            is_intra_dup = intra_max >= self.dedup_cos_thresh
        else:
            is_intra_dup = torch.zeros((k_cr,), device=device, dtype=torch.bool)
        cand_novel = cand_valid & (~is_intra_dup) & (max_sim < self.dedup_cos_thresh)

        all_lat = torch.cat([cr_lat_u, cand_lat], dim=0)
        all_ctx = torch.cat([cr_ctx_u, cand_ctx], dim=0)
        all_tgt = torch.cat([cr_tgt_u, cand_tgt], dim=0)
        all_tmask = torch.cat([cr_tmask_u, cand_tmask], dim=0)
        all_prio = torch.cat([cr_prio_u, cand_prio], dim=0)
        all_filled = torch.cat([cr_filled, cand_novel], dim=0)

        eff_score = torch.where(
            all_filled,
            all_prio.to(dtype=calc_dtype),
            torch.full((all_prio.shape[0],), -1e9, device=device, dtype=calc_dtype),
        )
        _, keep_idx = torch.topk(eff_score, k=c0, dim=0, largest=True, sorted=True)
        keep_filled = all_filled[keep_idx]
        keep_m2 = keep_filled.unsqueeze(-1)
        self.proto_latent[:c0].copy_(
            torch.where(keep_m2, all_lat[keep_idx], torch.zeros_like(cr_lat))
        )
        self.proto_context[:c0].copy_(
            torch.where(keep_m2, all_ctx[keep_idx], torch.zeros_like(cr_ctx))
        )
        self.proto_target[:c0].copy_(
            torch.where(keep_m2, all_tgt[keep_idx], torch.zeros_like(cr_tgt))
        )
        self.proto_target_mask[:c0].copy_(all_tmask[keep_idx] & keep_m2)
        self.proto_priority[:c0].copy_(
            torch.where(keep_filled, all_prio[keep_idx], torch.zeros_like(cr_prio))
        )
        self.proto_filled[:c0].copy_(keep_filled)

        # 2) Estrato 1 [c0:c1] — Sorpresa reciente FIFO (0-sync CUDA: 1-D index_select + arange!)
        if self.cap_surprise > 0:
            su_score = torch.where(
                asset_valid, surp_vec + 0.25 * rpe_vec, torch.full_like(surp_vec, -1e9)
            )
            su_best_1d = su_score.argmax(dim=0).reshape(1)
            su_valid = asset_valid.index_select(0, su_best_1d).squeeze(0)
            su_surp = surp_vec.index_select(0, su_best_1d).squeeze(0)
            su_rpe = rpe_vec.index_select(0, su_best_1d).squeeze(0)
            su_all_filled = self.proto_filled[c0:c1].all()
            admit_su = su_valid & ((su_surp >= 0.35) | (~su_all_filled))

            su_ptr = self.write_ptr.narrow(0, 0, 1).squeeze(0)
            su_rel = su_ptr % self.cap_surprise
            su_oh = torch.arange(self.cap_surprise, device=device, dtype=torch.long) == su_rel
            su_write_mask = su_oh & admit_su
            su_m2 = su_write_mask.unsqueeze(-1)
            self.proto_latent[c0:c1].copy_(
                torch.where(
                    su_m2, lat_safe.index_select(0, su_best_1d), self.proto_latent[c0:c1]
                )
            )
            self.proto_context[c0:c1].copy_(
                torch.where(
                    su_m2, ctx_safe.index_select(0, su_best_1d), self.proto_context[c0:c1]
                )
            )
            self.proto_target[c0:c1].copy_(
                torch.where(
                    su_m2, tgt_safe.index_select(0, su_best_1d), self.proto_target[c0:c1]
                )
            )
            self.proto_target_mask[c0:c1].copy_(
                torch.where(
                    su_m2, tgt_fin.index_select(0, su_best_1d), self.proto_target_mask[c0:c1]
                )
            )
            su_prio_val = (su_surp + 0.25 * su_rpe + 1e-3).to(dtype=buf_dtype)
            self.proto_priority[c0:c1].copy_(
                torch.where(su_write_mask, su_prio_val, self.proto_priority[c0:c1])
            )
            self.proto_filled[c0:c1].copy_(self.proto_filled[c0:c1] | su_write_mask)
            next_su_ptr = (su_ptr + admit_su.to(torch.long)) % self.cap_surprise
            self.write_ptr.narrow(0, 0, 1).copy_(next_su_ptr.reshape(1))

        # 3) Estrato 2 [c1:C] — Anclas de calma FIFO (0-sync CUDA: 1-D index_select + arange!)
        if self.cap_calm > 0:
            ca_score = torch.where(asset_valid, -rpe_vec, torch.full_like(rpe_vec, -1e9))
            ca_best_1d = ca_score.argmax(dim=0).reshape(1)
            ca_valid = asset_valid.index_select(0, ca_best_1d).squeeze(0)
            ca_rpe = rpe_vec.index_select(0, ca_best_1d).squeeze(0)
            ca_surp = surp_vec.index_select(0, ca_best_1d).squeeze(0)
            if is_2d:
                admit_ca = ca_valid & (ca_rpe <= 1.5)
            else:
                admit_ca = ca_valid & (ca_rpe <= 1.0) & (ca_surp < 0.5)

            ca_ptr = self.write_ptr.narrow(0, 1, 1).squeeze(0)
            ca_rel = ca_ptr % self.cap_calm
            ca_oh = torch.arange(self.cap_calm, device=device, dtype=torch.long) == ca_rel
            ca_write_mask = ca_oh & admit_ca
            ca_m2 = ca_write_mask.unsqueeze(-1)
            self.proto_latent[c1:].copy_(
                torch.where(ca_m2, lat_safe.index_select(0, ca_best_1d), self.proto_latent[c1:])
            )
            self.proto_context[c1:].copy_(
                torch.where(ca_m2, ctx_safe.index_select(0, ca_best_1d), self.proto_context[c1:])
            )
            self.proto_target[c1:].copy_(
                torch.where(ca_m2, tgt_safe.index_select(0, ca_best_1d), self.proto_target[c1:])
            )
            self.proto_target_mask[c1:].copy_(
                torch.where(
                    ca_m2, tgt_fin.index_select(0, ca_best_1d), self.proto_target_mask[c1:]
                )
            )
            ca_prio_val = (1.0 / (1.0 + ca_rpe + ca_surp)).to(dtype=buf_dtype)
            self.proto_priority[c1:].copy_(
                torch.where(ca_write_mask, ca_prio_val, self.proto_priority[c1:])
            )
            self.proto_filled[c1:].copy_(self.proto_filled[c1:] | ca_write_mask)
            next_ca_ptr = (ca_ptr + admit_ca.to(torch.long)) % self.cap_calm
            self.write_ptr.narrow(0, 1, 1).copy_(next_ca_ptr.reshape(1))

        self.write_ptr.narrow(0, 2, 1).add_(1)
        return self.proto_filled.sum().to(dtype=buf_dtype)

    @torch.no_grad()
    def compute_surprise(self, h_session: torch.Tensor) -> torch.Tensor:
        """Sorpresa episódica `1 - max_cos_sim` de la sesión actual frente a la memoria SWR."""
        device = self.proto_latent.device
        calc_dtype = _stable_calc_dtype(h_session.dtype)
        proto = h_session.detach().to(device=device, dtype=calc_dtype)
        if proto.ndim == 2:
            proto = proto.mean(dim=0)
        p_unit = F.normalize(proto.unsqueeze(0), dim=-1, eps=1e-6)  # [1, D]
        k_unit = F.normalize(self.proto_latent.to(dtype=calc_dtype), dim=-1, eps=1e-6)  # [C, D]
        sims = (p_unit @ k_unit.T).squeeze(0)  # [C]
        masked_sims = torch.where(
            self.proto_filled, sims, torch.full_like(sims, -1.0)
        )
        has_any = self.proto_filled.any()
        max_sim = masked_sims.max()
        surprise = torch.where(
            has_any, torch.relu(1.0 - max_sim), torch.ones((), device=device, dtype=calc_dtype)
        )
        return surprise.to(dtype=h_session.dtype)

    @torch.no_grad()
    def store_episode(
        self,
        latent_proto: torch.Tensor,
        context_proto: torch.Tensor,
        target_proto: torch.Tensor,
        *,
        surprise: torch.Tensor,
        matured_rpe: torch.Tensor,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Almacena prototipos individuales top-K de alto `|RPE|` con targets reales (cero sync)."""
        device = self.proto_latent.device
        buf_dtype = self.proto_latent.dtype
        calc_dtype = _stable_calc_dtype(buf_dtype)

        lat_in = latent_proto.detach().to(device=device, dtype=buf_dtype)
        ctx_in = context_proto.detach().to(device=device, dtype=buf_dtype)
        tgt_in = target_proto.detach().to(device=device, dtype=buf_dtype)

        if lat_in.ndim == 2:
            n_assets = lat_in.shape[0]
            if n_assets == 0:
                return self.proto_filled.sum().to(dtype=buf_dtype)
            if ctx_in.ndim == 1:
                ctx_2d = ctx_in.unsqueeze(0).expand(n_assets, -1)
            elif ctx_in.shape[0] == n_assets:
                ctx_2d = ctx_in
            else:
                ctx_2d = ctx_in.mean(dim=0, keepdim=True).expand(n_assets, -1)

            if tgt_in.ndim == 1:
                if tgt_in.shape[0] == n_assets:
                    tgt_2d = tgt_in.unsqueeze(-1)
                else:
                    tgt_2d = tgt_in.unsqueeze(0).expand(n_assets, -1)
            elif tgt_in.shape[0] == n_assets:
                tgt_2d = tgt_in
            else:
                tgt_2d = tgt_in.mean(dim=0, keepdim=True).expand(n_assets, -1)

            n_h_in = tgt_2d.shape[-1]
            tgt_fin_raw = torch.isfinite(tgt_2d)
            if n_h_in < self.n_horizons:
                pad_h = self.n_horizons - n_h_in
                tgt_pad = F.pad(tgt_2d, (0, pad_h))
                tgt_fin = F.pad(tgt_fin_raw, (0, pad_h), value=False)
            else:
                tgt_pad = tgt_2d[:, : self.n_horizons]
                tgt_fin = tgt_fin_raw[:, : self.n_horizons]

            tgt_safe = torch.where(tgt_fin, tgt_pad, torch.zeros_like(tgt_pad))
            lat_fin = torch.isfinite(lat_in).all(dim=-1)
            lat_safe = torch.where(torch.isfinite(lat_in), lat_in, torch.zeros_like(lat_in))
            ctx_safe = torch.where(torch.isfinite(ctx_2d), ctx_2d, torch.zeros_like(ctx_2d))

            surp_in = surprise.detach().to(device=device, dtype=calc_dtype)
            if surp_in.ndim >= 1 and surp_in.shape[0] == n_assets:
                surp_v = surp_in.reshape(n_assets, -1).mean(dim=-1)
                surp_vec = torch.where(
                    torch.isfinite(surp_v), surp_v.clamp_min(0.0), torch.zeros_like(surp_v)
                )
            else:
                surp_s = surp_in.mean()
                surp_s = torch.where(
                    torch.isfinite(surp_s), surp_s.clamp_min(0.0), torch.zeros_like(surp_s)
                )
                surp_vec = surp_s.expand(n_assets)

            rpe_in = matured_rpe.detach().to(device=device, dtype=calc_dtype)
            if rpe_in.ndim >= 1 and rpe_in.shape[0] == n_assets:
                rpe_v = rpe_in.reshape(n_assets, -1).abs().mean(dim=-1)
                rpe_vec = torch.where(torch.isfinite(rpe_v), rpe_v, torch.zeros_like(rpe_v))
            else:
                rpe_s = rpe_in.abs().mean()
                rpe_s = torch.where(torch.isfinite(rpe_s), rpe_s, torch.zeros_like(rpe_s))
                tgt_mag = (
                    tgt_safe.to(dtype=calc_dtype).abs().sum(dim=-1)
                    / tgt_fin.to(dtype=calc_dtype).sum(dim=-1).clamp_min(1.0)
                )
                rpe_vec = rpe_s + tgt_mag * 50.0

            asset_valid = tgt_fin.any(dim=-1) & lat_fin
            if self.use_stratified_swr:
                return self._store_stratified_zero_sync(
                    lat_safe=lat_safe,
                    ctx_safe=ctx_safe,
                    tgt_safe=tgt_safe,
                    tgt_fin=tgt_fin,
                    surp_vec=surp_vec,
                    rpe_vec=rpe_vec,
                    asset_valid=asset_valid,
                    is_2d=True,
                    top_k=top_k,
                )

            asset_prio = surp_vec + rpe_vec + 1e-3
            sel_score = torch.where(
                asset_valid, asset_prio, torch.full_like(asset_prio, -1e9)
            )
            k_req = (
                min(4, max(1, self.capacity // 2))
                if top_k is None
                else max(1, int(top_k))
            )
            k_sel = min(k_req, n_assets, self.capacity)
            _, top_idx = torch.topk(sel_score, k=k_sel, dim=0)

            cand_lat = lat_safe[top_idx]
            cand_ctx = ctx_safe[top_idx]
            cand_tgt = tgt_safe[top_idx]
            cand_tmask = tgt_fin[top_idx]
            cand_prio = asset_prio[top_idx].to(dtype=buf_dtype)
            cand_valid = asset_valid[top_idx]
        else:
            ctx_1d = ctx_in.mean(dim=0) if ctx_in.ndim == 2 else ctx_in
            tgt_1d = tgt_in.mean(dim=0) if tgt_in.ndim == 2 else tgt_in
            tgt_fin_raw = torch.isfinite(tgt_1d)
            if tgt_1d.numel() < self.n_horizons:
                pad_h = self.n_horizons - tgt_1d.numel()
                tgt_pad = F.pad(tgt_1d, (0, pad_h))
                tgt_fin = F.pad(tgt_fin_raw, (0, pad_h), value=False)
            else:
                tgt_pad = tgt_1d[: self.n_horizons]
                tgt_fin = tgt_fin_raw[: self.n_horizons]

            tgt_safe = torch.where(tgt_fin, tgt_pad, torch.zeros_like(tgt_pad))
            lat_fin = torch.isfinite(lat_in).all()
            lat_safe = torch.where(torch.isfinite(lat_in), lat_in, torch.zeros_like(lat_in))
            ctx_safe = torch.where(torch.isfinite(ctx_1d), ctx_1d, torch.zeros_like(ctx_1d))

            surp_s = surprise.detach().to(device=device, dtype=calc_dtype).mean()
            surp_s = torch.where(
                torch.isfinite(surp_s), surp_s.clamp_min(0.0), torch.zeros_like(surp_s)
            )
            rpe_s = matured_rpe.detach().to(device=device, dtype=calc_dtype).abs().mean()
            rpe_s = torch.where(torch.isfinite(rpe_s), rpe_s, torch.zeros_like(rpe_s))

            if self.use_stratified_swr:
                return self._store_stratified_zero_sync(
                    lat_safe=lat_safe.unsqueeze(0),
                    ctx_safe=ctx_safe.unsqueeze(0),
                    tgt_safe=tgt_safe.unsqueeze(0),
                    tgt_fin=tgt_fin.unsqueeze(0),
                    surp_vec=surp_s.unsqueeze(0),
                    rpe_vec=rpe_s.unsqueeze(0),
                    asset_valid=(tgt_fin.any() & lat_fin).unsqueeze(0),
                    is_2d=False,
                    top_k=1,
                )

            new_prio = (surp_s + rpe_s + 1e-3).to(dtype=buf_dtype)

            cand_lat = lat_safe.unsqueeze(0)
            cand_ctx = ctx_safe.unsqueeze(0)
            cand_tgt = tgt_safe.unsqueeze(0)
            cand_tmask = tgt_fin.unsqueeze(0)
            cand_prio = new_prio.unsqueeze(0)
            cand_valid = (tgt_fin.any() & lat_fin).unsqueeze(0)

        # Fusión vectorizada de cola de prioridad Top-C (cero bucles, cero .item(), FP16-safe!)
        all_lat = torch.cat([self.proto_latent, cand_lat], dim=0)
        all_ctx = torch.cat([self.proto_context, cand_ctx], dim=0)
        all_tgt = torch.cat([self.proto_target, cand_tgt], dim=0)
        all_tmask = torch.cat([self.proto_target_mask, cand_tmask], dim=0)
        all_prio = torch.cat([self.proto_priority, cand_prio], dim=0)
        all_filled = torch.cat([self.proto_filled, cand_valid], dim=0)

        eff_score = torch.where(
            all_filled,
            all_prio.to(dtype=calc_dtype),
            torch.full((all_prio.shape[0],), -1e9, device=device, dtype=calc_dtype),
        )
        _, keep_idx = torch.topk(eff_score, k=self.capacity, dim=0, largest=True, sorted=True)
        keep_filled = all_filled[keep_idx]
        keep_m2 = keep_filled.unsqueeze(-1)
        self.proto_latent.copy_(
            torch.where(keep_m2, all_lat[keep_idx], torch.zeros_like(self.proto_latent))
        )
        self.proto_context.copy_(
            torch.where(keep_m2, all_ctx[keep_idx], torch.zeros_like(self.proto_context))
        )
        self.proto_target.copy_(
            torch.where(keep_m2, all_tgt[keep_idx], torch.zeros_like(self.proto_target))
        )
        self.proto_target_mask.copy_(all_tmask[keep_idx] & keep_m2)
        self.proto_priority.copy_(
            torch.where(keep_filled, all_prio[keep_idx], torch.zeros_like(self.proto_priority))
        )
        self.proto_filled.copy_(keep_filled)
        return self.proto_filled.sum().to(dtype=buf_dtype)

    def sample_replay_batch(
        self,
        batch_size: int = 8,
        *,
        temperature: float | None = None,
        sample_seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Selección Top-K (v1) o Gumbel-TopK determinista GPU 3-estratos (v2) sin `.item()`."""
        k = max(1, min(int(batch_size), self.capacity))
        temp = self.temperature if temperature is None else float(temperature)
        device = self.proto_latent.device
        buf_dtype = self.proto_latent.dtype
        calc_dtype = _stable_calc_dtype(buf_dtype)

        # Snapshot BPTT multi-sesión: evita error in-place cuando `step_matured` muta buffers (#141)
        p_filled = self.proto_filled.clone() if torch.is_grad_enabled() else self.proto_filled
        p_prio = self.proto_priority.clone() if torch.is_grad_enabled() else self.proto_priority
        p_lat = self.proto_latent.clone() if torch.is_grad_enabled() else self.proto_latent
        p_ctx = self.proto_context.clone() if torch.is_grad_enabled() else self.proto_context
        p_tgt = self.proto_target.clone() if torch.is_grad_enabled() else self.proto_target
        p_tmask = (
            self.proto_target_mask.clone() if torch.is_grad_enabled() else self.proto_target_mask
        )
        w_ptr = self.write_ptr.clone() if torch.is_grad_enabled() else self.write_ptr

        if self.use_stratified_swr:
            idx = torch.arange(self.capacity, device=device, dtype=torch.int64)
            step = w_ptr[2].to(device=device, dtype=torch.int64) + int(sample_seed)
            h = (idx + 1) * 2654435761 + (step + 1) * 2246822519
            h = h ^ (h >> 15)
            h = (h * 2246822519) & 0x7FFFFFFF
            h = h ^ (h >> 13)
            u = ((h & 0x00FFFFFF).to(dtype=calc_dtype) + 0.5) / float(0x01000000)
            gumbel = -torch.log(-torch.log(u.clamp(1e-6, 1.0 - 1e-6)))

            c0 = self.cap_crisis
            c1 = c0 + self.cap_surprise
            k_cr = min(self.cap_crisis, max(1, (k * 2) // 4))
            rem_k = k - k_cr
            k_su = (
                min(self.cap_surprise, max(1, rem_k // 2))
                if (rem_k > 0 and self.cap_surprise > 0)
                else 0
            )
            k_ca = min(self.cap_calm, max(0, k - k_cr - k_su)) if self.cap_calm > 0 else 0
            shortfall = k - (k_cr + k_su + k_ca)
            if shortfall > 0:
                add_cr = min(shortfall, self.cap_crisis - k_cr)
                k_cr += add_cr

            idx_parts: list[torch.Tensor] = []
            for lo, hi, k_s in ((0, c0, k_cr), (c0, c1, k_su), (c1, self.capacity, k_ca)):
                if k_s <= 0 or hi <= lo:
                    continue
                s_filled = p_filled[lo:hi]
                s_prio = p_prio[lo:hi].to(dtype=calc_dtype)
                s_logits = torch.log(s_prio.clamp_min(1e-6)) / max(temp, 1e-3) + gumbel[lo:hi]
                s_score = torch.where(s_filled, s_logits, torch.full_like(s_logits, -1e6))
                _, rel_idx = torch.topk(s_score, k=k_s, dim=0)
                idx_parts.append(rel_idx + lo)

            top_idx = torch.cat(idx_parts, dim=0)
            top_filled = p_filled[top_idx]
            top_vals = p_prio[top_idx].to(dtype=calc_dtype)
            top_filled_f = top_filled.to(dtype=calc_dtype)
            has_replay = top_filled.any()

            w_logits = torch.where(
                top_filled,
                0.25 * torch.log1p(top_vals.clamp_min(0.0)) / max(temp, 1e-3),
                torch.full_like(top_vals, -1e6),
            )
            weights = torch.softmax(w_logits, dim=0) * top_filled_f
            weights = weights / weights.sum().clamp_min(1e-8)
            weights = torch.where(has_replay, weights, torch.zeros_like(weights))
            return dict(
                latents=p_lat[top_idx],
                contexts=p_ctx[top_idx],
                targets=p_tgt[top_idx],
                target_mask=p_tmask[top_idx],
                priorities=torch.where(top_filled, top_vals, torch.zeros_like(top_vals)).to(
                    dtype=buf_dtype
                ),
                weights=weights.to(dtype=buf_dtype),
                filled_mask=top_filled,
                has_replay=has_replay,
            )

        eff_prio = torch.where(
            p_filled,
            p_prio.to(dtype=calc_dtype),
            torch.full((self.capacity,), -1e6, device=device, dtype=calc_dtype),
        )
        top_vals, top_idx = torch.topk(eff_prio, k=k, dim=0)
        top_filled = p_filled[top_idx]
        top_filled_f = top_filled.to(dtype=calc_dtype)
        has_replay = top_filled.any()

        logits = torch.where(
            top_filled,
            torch.log(top_vals.clamp_min(1e-6)) / max(temp, 1e-3),
            torch.full_like(top_vals, -1e6),
        )
        weights = torch.softmax(logits, dim=0) * top_filled_f
        weights = weights / weights.sum().clamp_min(1e-8)
        weights = torch.where(has_replay, weights, torch.zeros_like(weights))

        return dict(
            latents=p_lat[top_idx],
            contexts=p_ctx[top_idx],
            targets=p_tgt[top_idx],
            target_mask=p_tmask[top_idx],
            priorities=torch.where(top_filled, top_vals, torch.zeros_like(top_vals)).to(
                dtype=buf_dtype
            ),
            weights=weights.to(dtype=buf_dtype),
            filled_mask=top_filled,
            has_replay=has_replay,
        )

    def replay_regularization_loss(
        self,
        head: nn.Module,
        *,
        batch_size: int = 8,
        sample_seed: int = 0,
    ) -> torch.Tensor:
        """Pérdida de consolidación SWR por prioridad (0 exacto con grafo vivo si vacío)."""
        sample = self.sample_replay_batch(batch_size=batch_size, sample_seed=sample_seed)
        head_param = next(head.parameters())
        device, dtype = head_param.device, head_param.dtype
        calc_dtype = _stable_calc_dtype(dtype)

        latents = sample["latents"].to(device=device, dtype=dtype)
        targets = sample["targets"].to(device=device, dtype=calc_dtype)
        weights = sample["weights"].to(device=device, dtype=calc_dtype)
        has_replay = sample["has_replay"].to(device=device)

        pred = head(latents, exposures=None)["pred_return"].to(dtype=calc_dtype)
        n_h = min(pred.shape[-1], targets.shape[-1])
        t_mask = sample["target_mask"].to(device=device, dtype=calc_dtype)[:, :n_h]
        huber_raw = F.huber_loss(pred[:, :n_h], targets[:, :n_h], reduction="none", delta=0.05)
        per_proto = (huber_raw * t_mask).sum(dim=-1) / t_mask.sum(dim=-1).clamp_min(1.0)
        loss_raw = (weights * per_proto).sum()
        return torch.where(has_replay, loss_raw, loss_raw * 0.0).to(dtype=dtype)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Cebado episódico suave desde prototipos prioritarios almacenados (sin `.item()`)."""
        device, dtype = h.device, h.dtype
        calc_dtype = _stable_calc_dtype(dtype)
        filled = self.proto_filled.to(device=device)
        protos = self.proto_latent.to(device=device, dtype=dtype)
        prios = self.proto_priority.to(device=device, dtype=calc_dtype)
        # Snapshot BPTT multi-sesión: evita crash in-place cuando `step_matured` muta SWR (#141)
        if torch.is_grad_enabled():
            filled = filled.clone()
            protos = protos.clone()
            prios = prios.clone()
        filled_f = filled.to(dtype=calc_dtype)
        filled_cnt = filled_f.sum()
        has_any = filled_cnt > 0.5

        q = self.q_proj(h).to(dtype=calc_dtype)
        k = self.k_proj(protos).to(dtype=calc_dtype)
        v = self.v_proj(protos).to(dtype=calc_dtype)
        inv_sqrt_d = 1.0 / math.sqrt(float(self.hidden_dim))
        logits = (q @ k.T) * inv_sqrt_d + 0.25 * torch.log1p(prios.clamp_min(0.0)).unsqueeze(0)
        masked_logits = torch.where(filled.unsqueeze(0), logits, torch.full_like(logits, -1e6))
        attn = torch.softmax(masked_logits, dim=-1) * filled_f.unsqueeze(0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        primed_raw = self.out_proj((attn @ v).to(dtype=dtype))
        gate = torch.sigmoid(self.replay_gate.to(device=device, dtype=dtype))
        gated_primed = gate * primed_raw
        primed = torch.where(has_any, gated_primed, gated_primed * 0.0)

        masked_prios = torch.where(filled, prios, torch.zeros_like(prios))
        mean_prio = masked_prios.sum() / filled_cnt.clamp_min(1.0)
        max_prio = masked_prios.max()
        return h + primed, dict(
            filled_count=filled_cnt.to(dtype=dtype),
            mean_priority=torch.where(has_any, mean_prio, torch.zeros_like(mean_prio)).to(
                dtype=dtype
            ),
            max_priority=torch.where(has_any, max_prio, torch.zeros_like(max_prio)).to(dtype=dtype),
        )


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


class _ContractivePredictiveCodingCycle(nn.Module):
    """Ciclo bidireccional contractivo de Codificación Predictiva (`Lipschitz < 1`)."""

    def __init__(self, hidden_dim: int, *, lipschitz_bound: float = 0.90) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lipschitz_bound = float(lipschitz_bound)
        self.top_down = _SpectralLinear(hidden_dim, hidden_dim)
        self.sn_h = _SpectralLinear(hidden_dim, hidden_dim)
        self.sn_x0 = _SpectralLinear(hidden_dim, hidden_dim)
        self.sn_err = _SpectralLinear(hidden_dim, hidden_dim)
        self.sn_out = _SpectralLinear(hidden_dim, hidden_dim)
        self.beta_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def update_spectral_buffers(self) -> None:
        """Actualiza buffers espectrales `u`/`v` de las 5 proyecciones tras `optimizer.step()`."""
        for layer in (self.top_down, self.sn_h, self.sn_x0, self.sn_err, self.sn_out):
            layer.update_spectral_buffers()

    def compute_spectral_weights(self, *, update_buffers: bool = False) -> tuple[torch.Tensor, ...]:
        return (
            self.top_down.normalized_weight(update_buffers=update_buffers),
            self.sn_h.normalized_weight(update_buffers=update_buffers),
            self.sn_x0.normalized_weight(update_buffers=update_buffers),
            self.sn_err.normalized_weight(update_buffers=update_buffers),
            self.sn_out.normalized_weight(update_buffers=update_buffers),
        )

    def lipschitz_upper_bound(self) -> torch.Tensor:
        """Cota superior certificada `||d h_next / d h||_2 < lipschitz_bound < 1`."""
        beta = 0.15 + 0.45 * torch.sigmoid(self.beta_logit)
        # ||d(pc_err)/dh|| <= max(Pi) <= 1.0 cuando Pi in [0.25, 1.0]; el factor 1/sqrt(3) acota
        return self.lipschitz_bound * (1.0 - beta * (1.0 - 2.0 / 3.0))

    def forward(
        self,
        x0: torch.Tensor,
        h: torch.Tensor,
        precision_pi: torch.Tensor,
        *,
        cached_weights: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_td, w_h, w_x0, w_err, w_out = (
            self.compute_spectral_weights(update_buffers=False)
            if cached_weights is None
            else cached_weights
        )
        # Predicción generativa top-down y error de predicción ponderado por precisión Pi <= 1
        x_hat = self.top_down(h, cached_weight=w_td)
        pi_clamped = precision_pi.to(device=h.device, dtype=h.dtype).clamp(0.25, 1.0)
        pi_eff = pi_clamped.unsqueeze(0) if pi_clamped.ndim == 1 else pi_clamped
        pc_err = pi_eff * (x0 - x_hat)

        # Escala 1/3 asegura ||d u / dh||_2 <= (1 + 1)/3 = 2/3 < 1 incluso con top_down acoplado
        u = torch.tanh(
            (
                self.sn_h(h, cached_weight=w_h)
                + self.sn_x0(x0, cached_weight=w_x0)
                + self.sn_err(pc_err, cached_weight=w_err)
            )
            * (1.0 / 3.0)
        )
        proposal = torch.tanh(self.sn_out(u, cached_weight=w_out)) * self.lipschitz_bound
        beta = (0.15 + 0.45 * torch.sigmoid(self.beta_logit)).to(device=h.device, dtype=h.dtype)
        l_b = self.lipschitz_bound
        h_next = l_b * (1.0 - beta) * h + (1.0 - l_b) * (1.0 - beta) * x0 + beta * proposal
        return h_next, pc_err


def _run_hypergraph_zero_sync(
    hg: RMTSpilloverHypergraph,
    x: torch.Tensor,
    *,
    rmt_loadings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ejecuta `RMTSpilloverHypergraph` en GPU/BF16 con cero `.item()`."""
    device, dtype = x.device, x.dtype
    calc_dtype = _stable_calc_dtype(dtype)
    out = x
    scale = 1.0 / math.sqrt(float(hg.hidden_dim))
    norm_out = hg.node_norm(out)
    logits = (norm_out @ hg.edge_queries.to(dtype=norm_out.dtype).T) * scale
    if rmt_loadings is not None and rmt_loadings.shape[-1] == hg.rmt_dim:
        rmt_in = rmt_loadings.to(device=device, dtype=dtype)
    else:
        rmt_in = torch.zeros((x.shape[0], hg.rmt_dim), device=device, dtype=dtype)
    logits = logits + hg.rmt_proj(rmt_in)
    inc_c = torch.softmax(logits.to(dtype=calc_dtype), dim=-1)
    edge_deg = inc_c.sum(dim=0, keepdim=True).clamp_min(1e-6)
    inc = inc_c.to(dtype=dtype)
    inc_norm = (inc_c / edge_deg).to(dtype=dtype)

    for _ in range(hg.n_rounds):
        v = F.silu(hg.msg_in(hg.node_norm(out)))
        edge_state = inc_norm.T @ v
        node_msg = hg.msg_out(inc @ edge_state)
        out = out + hg.out_scale.to(dtype=dtype) * node_msg

    ent = -(inc_c * torch.log(inc_c.clamp_min(1e-8))).sum(dim=-1).mean().to(dtype=dtype)
    return out, ent


def _run_holographic_3l_zero_sync(
    hb: HolographicBrain3L,
    mod_tokens: torch.Tensor,
    modality_mask: torch.Tensor,
    *,
    regime_vec: torch.Tensor | None = None,
    rmt_loadings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ejecuta `HolographicBrain3L` completo (3 niveles) en GPU/BF16 con cero `.item()`."""
    n_assets, _, d_model = mod_tokens.shape
    device, dtype = mod_tokens.device, mod_tokens.dtype
    calc_dtype = _stable_calc_dtype(dtype)
    mask_bool = modality_mask.to(device=device) > 0.5
    present_tokens = torch.where(
        mask_bool.unsqueeze(-1), mod_tokens, torch.zeros_like(mod_tokens)
    )

    p_ab = torch.softmax(hb.membership_logits.to(device=device, dtype=calc_dtype), dim=-1).to(
        dtype=dtype
    )
    m_a = torch.einsum("ma,nmd->nad", p_ab, present_tokens)
    low_r = torch.einsum("nad,adr->nar", m_a, hb.area_v.to(dtype=dtype))
    lora_delta = torch.einsum("nar,adr->nad", low_r, hb.area_u.to(dtype=dtype))
    h_areas = hb.node_norm(m_a + lora_delta + hb.area_emb.to(dtype=dtype).unsqueeze(0))

    pt_norm = hb.node_norm(present_tokens)
    slot_logits = torch.einsum(
        "nmd,sd->nsm", pt_norm, hb.slot_keys.to(dtype=pt_norm.dtype)
    ) / math.sqrt(float(d_model))
    sl_c = slot_logits.to(dtype=calc_dtype)
    neg_inf = torch.where(
        mask_bool.unsqueeze(1), torch.zeros_like(sl_c), torch.full_like(sl_c, -1e4)
    )
    d_sm = torch.softmax(sl_c + neg_inf, dim=-1).to(dtype=dtype)
    slot_mix = torch.einsum("nsm,nmd->nsd", d_sm, present_tokens)
    h_slots = hb.node_norm(slot_mix + hb.slot_emb.to(dtype=dtype).unsqueeze(0))
    c_ms = torch.softmax(sl_c.transpose(1, 2), dim=-1).to(dtype=dtype)

    nodes = torch.cat([h_areas, h_slots], dim=1)
    init_nodes = nodes
    if regime_vec is not None:
        rv = regime_vec.to(device=device, dtype=dtype)
        if rv.ndim == 1:
            rv = rv.unsqueeze(0).expand(n_assets, -1)
        gamma, beta = hb.thalamus_film(rv).chunk(2, dim=-1)
        gamma = 1.0 + 0.5 * torch.tanh(gamma)
        beta = 0.5 * torch.tanh(beta)
    else:
        gamma = torch.ones((n_assets, d_model), device=device, dtype=dtype)
        beta = torch.zeros_like(gamma)

    for _ in range(hb.cycles):
        q, k, v = hb.qkv(hb.node_norm(nodes)).chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        ).squeeze(1)
        nodes = nodes + 0.25 * (gamma.unsqueeze(1) * hb.bridge_out(attn) + beta.unsqueeze(1))

    summary_in = nodes.mean(dim=1)
    summary_l2, hg_ent = _run_hypergraph_zero_sync(
        hb.hypergraph_l2, summary_in, rmt_loadings=rmt_loadings
    )
    l2_delta = summary_l2 - summary_in

    delta_nodes = nodes - init_nodes
    delta_areas = delta_nodes[:, : hb.n_cortical_areas, :]
    delta_slots = delta_nodes[:, hb.n_cortical_areas :, :]
    delta_mod = (
        torch.einsum("ma,nad->nmd", p_ab, delta_areas)
        + torch.einsum("nms,nsd->nmd", c_ms, delta_slots)
        + l2_delta.unsqueeze(1)
    )
    out_cand = present_tokens + hb.readback(delta_mod)
    out_tokens = torch.where(mask_bool.unsqueeze(-1), out_cand, torch.zeros_like(out_cand))
    return out_tokens, hg_ent


def _encode_hrr_key_unitary(
    query: torch.Tensor,
    *,
    regime_key: torch.Tensor | None = None,
    sector_key: torch.Tensor | None = None,
) -> torch.Tensor:
    """Codifica clave HRR con enlace unitario real-pair FHRR (0 llamadas `rfft`/`irfft`)."""
    k = F.normalize(query, dim=-1, eps=1e-6)
    if regime_key is not None:
        rk = regime_key.to(device=query.device, dtype=query.dtype)
        if rk.ndim == 1:
            rk = rk.unsqueeze(0).expand_as(k)
        k = _fhrr_bind(k, rk)
    if sector_key is not None:
        sk = sector_key.to(device=query.device, dtype=query.dtype)
        if sk.ndim == 1:
            sk = sk.unsqueeze(0).expand_as(k)
        k = _fhrr_bind(k, sk)
    return F.normalize(k, dim=-1, eps=1e-6)


def _run_hrr_lookup_zero_sync(
    hrr: HRRModernHopfieldMemory,
    h: torch.Tensor,
    *,
    regime_key: torch.Tensor | None = None,
    use_unitary_fhrr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ejecuta consulta `HRRModernHopfieldMemory` en GPU/BF16/FP16 (v1 FFT o v2 FHRR 0-FFT)."""
    device, dtype = h.device, h.dtype
    param_dtype = hrr.out_proj.weight.dtype
    h_in = h if h.dtype == param_dtype else h.to(dtype=param_dtype)
    if not use_unitary_fhrr:
        res = hrr.retrieve(h_in, regime_key=regime_key)
        return res["retrieved"].to(device=device, dtype=dtype), res["ood_energy"].to(
            device=device, dtype=dtype
        )

    # Ruta v2 100% libre de `rfft`/`irfft` (real-pair unitary FHRR, Inductor-fusable!)
    n_assets = h_in.shape[0]
    calc_dtype = _stable_calc_dtype(h_in.dtype)
    k_q = _encode_hrr_key_unitary(h_in, regime_key=regime_key)
    filled = hrr.slot_filled.to(device=device)
    keys = hrr.slot_keys.to(device=device, dtype=calc_dtype)
    vals = hrr.slot_vals.to(device=device, dtype=calc_dtype)
    traces_d = hrr.traces.to(device=device, dtype=h_in.dtype)
    if torch.is_grad_enabled():
        filled = filled.clone()
        keys = keys.clone()
        vals = vals.clone()
        traces_d = traces_d.clone()
    filled_f = filled.to(dtype=calc_dtype)
    filled_cnt = filled_f.sum()
    has_slots = filled_cnt > 0.5

    sim = k_q.to(dtype=calc_dtype) @ keys.T  # [N, M]
    scaled = hrr.beta * sim
    masked_scaled = torch.where(filled.unsqueeze(0), scaled, torch.full_like(scaled, -1e6))
    attn = torch.softmax(masked_scaled, dim=-1) * filled_f.unsqueeze(0)
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    attn = torch.where(has_slots, attn, torch.zeros_like(attn))
    retrieved_raw = (attn @ vals).to(dtype=h_in.dtype)

    lse = torch.logsumexp(masked_scaled, dim=-1) / hrr.beta
    ood_raw = torch.relu(1.0 + torch.log(filled_cnt.clamp_min(1.0)) / hrr.beta - lse)
    ood_energy = torch.where(has_slots, ood_raw, torch.zeros_like(ood_raw)).to(dtype=dtype)

    has_trace = traces_d.abs().sum() > 1e-12
    w_tr = torch.softmax(hrr.trace_weights.to(device=device, dtype=h_in.dtype), dim=0)
    tr_mix = (w_tr.unsqueeze(-1) * traces_d).sum(dim=0)
    tr_exp = tr_mix.unsqueeze(0).expand(n_assets, -1)
    tr_unbound = _fhrr_unbind(k_q, tr_exp)
    trace_readout_raw = hrr.trace_proj(tr_exp + 0.5 * tr_unbound)
    trace_readout = torch.where(has_trace, trace_readout_raw, trace_readout_raw * 0.0)

    hop_raw = hrr.out_proj(retrieved_raw)
    hop_out = torch.where(has_slots, hop_raw, hop_raw * 0.0)
    retrieved = (hop_out + trace_readout).to(device=device, dtype=dtype)
    return retrieved, ood_energy


@torch.no_grad()
def _store_hrr_zero_sync(
    hrr: HRRModernHopfieldMemory,
    query: torch.Tensor,
    value: torch.Tensor,
    *,
    regime_key: torch.Tensor | None = None,
    return_signal: torch.Tensor | None = None,
    use_unitary_fhrr: bool = False,
) -> torch.Tensor:
    """Almacena 2 prototipos comprimidos (invariantes a `N`) por sesión sin sobrescribir todo."""
    k_enc = (
        _encode_hrr_key_unitary(query.detach(), regime_key=regime_key)
        if use_unitary_fhrr
        else hrr._encode_key(query.detach(), regime_key=regime_key)
    )
    k_det = k_enc.to(device=hrr.slot_keys.device, dtype=hrr.slot_keys.dtype)
    v_det = value.detach().to(device=hrr.slot_vals.device, dtype=hrr.slot_vals.dtype)

    # 2 prototipos equivariantes por sesión (evita que N=2600 borre las 64 ranuras cada día!)
    proto_k_mean = F.normalize(k_det.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
    proto_v_mean = v_det.mean(dim=0, keepdim=True)
    if return_signal is not None and return_signal.shape[0] == k_det.shape[0]:
        sig = return_signal.detach().to(device=k_det.device, dtype=k_det.dtype)
        w_dir = torch.softmax(sig * 20.0, dim=0).unsqueeze(-1)
        proto_k_dir = F.normalize((w_dir * k_det).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)
        proto_v_dir = (w_dir * v_det).sum(dim=0, keepdim=True)
        k_write = torch.cat([proto_k_mean, proto_k_dir], dim=0)
        v_write = torch.cat([proto_v_mean, proto_v_dir], dim=0)
    else:
        k_write = proto_k_mean
        v_write = proto_v_mean

    n_in = min(k_write.shape[0], hrr.max_slots)
    proto = proto_k_mean.squeeze(0)
    for idx, dec in enumerate(hrr.trace_decays):
        hrr.traces[idx].mul_(dec).add_(proto * (1.0 - dec))

    offsets = torch.arange(n_in, device=hrr.slot_keys.device, dtype=torch.long)
    indices = (offsets + hrr.write_ptr) % hrr.max_slots
    hrr.slot_keys[indices] = k_write[:n_in]
    hrr.slot_vals[indices] = v_write[:n_in]
    hrr.slot_filled.index_fill_(0, indices, True)
    hrr.write_ptr.copy_((hrr.write_ptr + n_in) % hrr.max_slots)
    return hrr.slot_filled.sum().to(dtype=query.dtype)


def _blend_head_predictions(
    pred_cand: dict[str, torch.Tensor],
    pred_prev: dict[str, torch.Tensor],
    accept_k: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combina predicciones por `accept_k` (exacto cuando `accept_k=1`, evita 2x `head`)."""
    a2 = accept_k.unsqueeze(-1)
    b2 = 1.0 - a2
    a3 = a2.unsqueeze(-1)
    b3 = 1.0 - a3

    tb_logits = a3 * pred_cand["tb_logits"] + b3 * pred_prev["tb_logits"]
    turn_logits = a2 * pred_cand["turn_logits"] + b2 * pred_prev["turn_logits"]
    log_vol = a2 * pred_cand["log_vol"] + b2 * pred_prev["log_vol"]
    return dict(
        pred_return=a2 * pred_cand["pred_return"] + b2 * pred_prev["pred_return"],
        alpha=a2 * pred_cand["alpha"] + b2 * pred_prev["alpha"],
        factor_return=a2 * pred_cand["factor_return"] + b2 * pred_prev["factor_return"],
        quantiles=a3 * pred_cand["quantiles"] + b3 * pred_prev["quantiles"],
        tb_logits=tb_logits,
        tb_probs=torch.softmax(tb_logits, dim=-1),
        turn_logits=turn_logits,
        turning_prob=torch.sigmoid(turn_logits),
        log_vol=log_vol,
        vol_pred=torch.exp(log_vol),
        epistemic_error=a2 * pred_cand["epistemic_error"] + b2 * pred_prev["epistemic_error"],
    )


def _run_high_snr_moe_zero_sync(
    moe: HighSNRMoELayer,
    x: torch.Tensor,
    context: torch.Tensor,
    *,
    placebo_ctx_bank: torch.Tensor | None = None,
    placebo_ctx_filled: torch.Tensor | None = None,
    placebo_ctx_ptr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ejecuta `HighSNRMoELayer` en GPU/BF16 con cero `.item()` y placebo cross-session válido."""
    n_assets = x.shape[0]
    device, dtype = x.device, x.dtype
    calc_dtype = _stable_calc_dtype(dtype)
    h = moe.in_norm(x)
    h_dtype = h.dtype

    s_hid = F.silu(
        torch.einsum("nd,edf->nef", h, moe.shared_w1.to(dtype=h_dtype))
        + moe.shared_b1.to(dtype=h_dtype).unsqueeze(0)
    )
    s_out = (
        torch.einsum("nef,efd->ned", s_hid, moe.shared_w2.to(dtype=h_dtype))
        + moe.shared_b2.to(dtype=h_dtype).unsqueeze(0)
    ).mean(dim=1)

    if moe.mode == "uniform":
        probs_c = torch.full(
            (n_assets, moe.n_routed),
            1.0 / float(moe.n_routed),
            device=device,
            dtype=calc_dtype,
        )
    else:
        ctx = context.to(device=device, dtype=dtype)
        if moe.mode == "shuffled_context":
            # Permutación biyectiva exacta para TODO N >= 1 (incluso 7 | N): gcd(-1, N) = 1
            idx = torch.arange(n_assets, device=device, dtype=torch.long)
            perm_n = (n_assets - 1 - idx + (n_assets // 2)) % max(n_assets, 1)
            ctx_fallback = torch.roll(ctx[perm_n], shifts=max(1, ctx.shape[-1] // 2), dims=-1)
            if (
                placebo_ctx_bank is not None
                and placebo_ctx_filled is not None
                and placebo_ctx_ptr is not None
                and placebo_ctx_bank.shape[0] > 0
            ):
                p_filled = placebo_ctx_filled.to(device=device)
                has_hist = p_filled.any()
                n_filled = p_filled.to(dtype=torch.long).sum().clamp_min(1)
                ptr_l = placebo_ctx_ptr.to(device=device, dtype=torch.long)
                pick_idx = ((ptr_l * 13 + 5) % n_filled).clamp_max(placebo_ctx_bank.shape[0] - 1)
                past_mean = (
                    placebo_ctx_bank.to(device=device, dtype=dtype)
                    .index_select(0, pick_idx.reshape(1))
                    .squeeze(0)
                )
                ctx_dev = (ctx - ctx.mean(dim=0, keepdim=True))[perm_n]
                ctx_hist = past_mean.unsqueeze(0) + ctx_dev
                ctx = torch.where(has_hist, ctx_hist, ctx_fallback)
            else:
                ctx = ctx_fallback
        logits = (moe.router(moe.ctx_norm(ctx)) / moe.temperature).to(dtype=calc_dtype)
        b_bias = moe.balance_bias.to(device=device, dtype=calc_dtype)
        if torch.is_grad_enabled():
            b_bias = b_bias.clone()
        logits = logits + b_bias.unsqueeze(0)
        probs_c = (
            _sparsemax(logits, dim=-1)
            if moe.mode == "sparsemax"
            else torch.softmax(logits, dim=-1)
        )

    probs = probs_c.to(dtype=h_dtype)
    r_hid = F.silu(
        torch.einsum("nd,edf->nef", h, moe.routed_w1.to(dtype=h_dtype))
        + moe.routed_b1.to(dtype=h_dtype).unsqueeze(0)
    )
    r_all = (
        torch.einsum("nef,efd->ned", r_hid, moe.routed_w2.to(dtype=h_dtype))
        + moe.routed_b2.to(dtype=h_dtype).unsqueeze(0)
    )
    r_out = torch.einsum("ne,ned->nd", probs, r_all)

    g = torch.sigmoid(moe.mix_gate.to(device=device, dtype=dtype))
    out = x + g * r_out.to(dtype=dtype) + (1.0 - g) * s_out.to(dtype=dtype)
    ent = -(probs_c * torch.log(probs_c.clamp_min(1e-8))).sum(dim=-1).mean().to(dtype=dtype)
    loads = probs_c.mean(dim=0).to(dtype=dtype)
    return out, ent, loads


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


@dataclass(frozen=True)
class NeuroBrainConfig:
    """Contrato inmutable SHA-256 de la Familia 2 Independiente (`holographic_brain`)."""

    ablation: str = "full"
    prices_dim: int = 46
    fundamentals_dim: int = 32
    macro_dim: int = 24
    news_dim: int = 32
    charts_dim: int = 32
    physics_dim: int = 24
    context_dim: int = 12
    exposures_dim: int = 9
    rmt_dim: int = 5
    hidden_dim: int = 128
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    # Interruptores de ablación emparejada de Familia 2
    use_dendritic: bool = True
    use_lateral_inhibition: bool = True
    use_three_factor: bool = True
    use_replay: bool = True
    use_hrr_hopfield: bool = True
    use_moe_bridge: bool = False
    moe_mode: str = "soft"
    n_shared_experts: int = 2
    n_routed_experts: int = 6
    max_cycles: int = DEFAULT_BRAIN_CYCLES
    # Geometría cortical, dendrítica, plástica e hipocampal
    n_cortical_areas: int = 19
    n_hidden_slots: int = 4
    n_hyperedges: int = 16
    n_dendritic_branches: int = 8
    plasticity_rank: int = 24
    replay_capacity: int = 64
    hrr_max_slots: int = 64
    pt1_alpha_lo: float = 0.05
    pt1_theta_max: float = 0.25
    pt1_w_max: float = 4.0
    huber_delta: float = 1.0
    lipschitz_bound: float = 0.90
    monotonic_penalty: float = 5.0
    # Interruptores v2 Neuro-IA SOTA (#128 §2 + #133 P0-4 & B; False por defecto -> v1 intacto!)
    use_unitary_fhrr: bool = False
    use_lgn_dual_input: bool = False
    use_active_inference_pi: bool = False
    use_homeostatic_bcm: bool = False
    use_stratified_swr: bool = False
    noic_pinball_twin: bool = False
    # Interruptores Wave-3C opt-in (valores por defecto conservan 100% retrocompatibilidad)
    use_polar_alpha_head: bool = False
    norm_kind: str = "layernorm"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim >= 8 es obligatorio")
        if not (1 <= self.max_cycles <= MAX_BRAIN_CYCLES):
            raise ValueError(
                f"max_cycles ({self.max_cycles}) debe cumplir el tope duro "
                f"1 <= max_cycles <= {MAX_BRAIN_CYCLES}"
            )
        if not (0.1 <= self.lipschitz_bound < 1.0):
            raise ValueError("lipschitz_bound debe ser estrictamente < 1.0 (Banach)")
        if self.moe_mode not in MOE_MODES:
            raise ValueError(f"moe_mode inválido {self.moe_mode!r}; válidos: {MOE_MODES}")
        if self.norm_kind not in ("layernorm", "rmsnorm", "dyt"):
            raise ValueError(
                f"norm_kind inválido {self.norm_kind!r}; válidos: ('layernorm', 'rmsnorm', 'dyt')"
            )
        if not (0.0 < self.pt1_alpha_lo < 1.0) or not (0.0 < self.pt1_theta_max <= 1.0):
            raise ValueError("pt1_alpha_lo in (0,1) y pt1_theta_max in (0,1] son obligatorios")
        if tuple(sorted(self.quantile_levels)) != self.quantile_levels:
            raise ValueError("quantile_levels debe ser estrictamente creciente")

    def identity(self) -> dict[str, object]:
        """Representación canónica JSON-serializable de Familia 2."""
        return dict(schema_version=1, family="holographic_brain", **asdict(self))

    def fingerprint(self) -> str:
        """Huella SHA-256 determinista del contrato completo."""
        payload = json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


class StandaloneHolographicBrainModel(nn.Module):
    """Modelo Independiente Familia 2 (`holographic_brain`): Neurociencia Computacional en GPU."""

    def __init__(self, config: NeuroBrainConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        nk = config.norm_kind

        # 1) 5 codificadores multimodales + física de mercado v4 (con canal dual LGN opcional v2)
        lgn_mult = 2 if config.use_lgn_dual_input else 1
        self.enc_prices = _ModalityEncoder(config.prices_dim * lgn_mult, d, norm_kind=nk)
        self.enc_fundamentals = _ModalityEncoder(
            config.fundamentals_dim * lgn_mult, d, norm_kind=nk
        )
        self.enc_macro = _ModalityEncoder(config.macro_dim, d, norm_kind=nk)
        self.enc_news = _ModalityEncoder(config.news_dim, d, norm_kind=nk)
        self.enc_charts = _ModalityEncoder(config.charts_dim, d, norm_kind=nk)
        self.enc_physics = _ModalityEncoder(config.physics_dim * lgn_mult, d, norm_kind=nk)
        self.regime_proj = nn.Linear(config.context_dim, d)

        # 2) Cerebro Holográfico 3 Niveles (Cortical + Hipergrafo RMT + Tálamo FiLM)
        self.holographic_3l = HolographicBrain3L(
            d,
            n_modalities=5,
            n_cortical_areas=config.n_cortical_areas,
            n_hidden_slots=config.n_hidden_slots,
            n_hyperedges=config.n_hyperedges,
            regime_dim=config.context_dim,
            rmt_dim=config.rmt_dim,
        )
        self.fusion_proj = nn.Linear(2 * d, d)
        self.fusion_norm: nn.Module = (
            nn.LayerNorm(d)
            if nk == "layernorm"
            else build_norm_layer(nk, d)  # type: ignore[arg-type]
        )

        # 3) Módulos Neuro-IA de Frontera gobernados por los flags de ablación
        self.dendritic = (
            DendriticCompartmentLayer(
                d, config.context_dim, n_branches=config.n_dendritic_branches
            )
            if config.use_dendritic
            else None
        )
        self.lateral_inhibition = (
            LateralInhibitionCompetition(d) if config.use_lateral_inhibition else None
        )
        self.homeostatic_scaling = (
            TurrigianoHomeostaticScaling(d) if config.use_homeostatic_bcm else None
        )
        self.hrr_hopfield = (
            HRRModernHopfieldMemory(d, max_slots=config.hrr_max_slots)
            if config.use_hrr_hopfield
            else None
        )
        self.three_factor_plasticity = (
            ThreeFactorDopaminePlasticity(
                d,
                plasticity_rank=min(config.plasticity_rank, d),
                pt1_alpha_lo=config.pt1_alpha_lo,
                pt1_theta_max=config.pt1_theta_max,
                pt1_w_max=config.pt1_w_max,
                huber_delta=config.huber_delta,
                use_homeostatic_bcm=config.use_homeostatic_bcm,
            )
            if config.use_three_factor
            else None
        )
        self.replay_memory = (
            HippocampalSharpWaveReplay(
                d,
                config.context_dim,
                n_horizons=len(config.horizons),
                capacity=config.replay_capacity,
                use_stratified_swr=config.use_stratified_swr,
            )
            if config.use_replay
            else None
        )
        self.active_inference_pi = (
            ActiveInferencePrecisionGate(d, config.context_dim, n_modalities=5)
            if config.use_active_inference_pi
            else None
        )

        # 4) Puente MoE opcional (desactivado por defecto para Familia 2 puramente standalone!)
        self.moe_bridges = (
            nn.ModuleList(
                [
                    HighSNRMoELayer(
                        hidden_dim=d,
                        context_dim=config.context_dim,
                        n_shared_experts=config.n_shared_experts,
                        n_routed_experts=config.n_routed_experts,
                        mode=config.moe_mode,
                    )
                    for _ in range(config.max_cycles)
                ]
            )
            if config.use_moe_bridge
            else None
        )

        # 5) Codificación Predictiva Contractiva weight-tied (`1..3`, tope duro `4`)
        self.pc_cycle = _ContractivePredictiveCodingCycle(
            d, lipschitz_bound=config.lipschitz_bound
        )
        self.register_buffer("layer_precision", torch.ones(d, dtype=torch.float32))
        self.register_buffer("pc_err_var_ema", torch.ones(d, dtype=torch.float32))
        self.register_buffer(
            "horizon_scales",
            torch.tensor([math.sqrt(float(h)) for h in config.horizons], dtype=torch.float32),
        )

        # 6) Cabezas Granulares Multi-Horizonte (9 cuantiles sin cruces + TB + Vol + Epist)
        self.head = MultiHorizonGranularHead(
            hidden_dim=d,
            horizons=config.horizons,
            quantile_levels=config.quantile_levels,
            exposures_dim=config.exposures_dim,
            norm_kind=nk,
        )
        self.polar_alpha_head: PolarAlphaHead | None = (
            PolarAlphaHead(n_horizons=len(config.horizons), default_vol=0.015)
            if config.use_polar_alpha_head
            else None
        )

    def _encode_modalities(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Codifica las 5 modalidades aislando con 0 exacto las modalidades ausentes."""
        prices = batch["prices"]
        n_assets = prices.shape[0]
        device, dtype = prices.device, prices.dtype
        c = self.config

        def _get(key: str, dim: int) -> torch.Tensor:
            val = batch.get(key)
            if val is None:
                return torch.zeros((n_assets, dim), device=device, dtype=dtype)
            return val.to(device=device, dtype=dtype)

        mask = batch.get("modality_mask")
        if mask is None:
            mask_bool = torch.ones((n_assets, 5), device=device, dtype=torch.bool)
        else:
            mask_bool = mask.to(device=device) > 0.5
        mask_t = mask_bool.to(dtype=dtype)

        # Aislar entradas crudas ausentes con torch.where (seguro ante NaN/Inf * 0!)
        def _zero_absent(raw: torch.Tensor, col: int) -> torch.Tensor:
            return torch.where(mask_bool[:, col : col + 1], raw, torch.zeros_like(raw))

        p_raw = _zero_absent(prices.to(device=device, dtype=dtype), 0)
        f_raw = _zero_absent(_get("fundamentals", c.fundamentals_dim), 1)
        m_in = _zero_absent(_get("macro", c.macro_dim), 2)
        n_in = _zero_absent(_get("news", c.news_dim), 3)
        ch_in = _zero_absent(_get("charts", c.charts_dim), 4)

        if c.use_lgn_dual_input:
            p_in = _lgn_dual_transform(p_raw, present_row_mask=mask_bool[:, 0])
            f_in = _lgn_dual_transform(f_raw, present_row_mask=mask_bool[:, 1])
        else:
            p_in = p_raw
            f_in = f_raw

        tokens = torch.stack(
            [
                self.enc_prices(p_in),
                self.enc_fundamentals(f_in),
                self.enc_macro(m_in),
                self.enc_news(n_in),
                self.enc_charts(ch_in),
            ],
            dim=1,
        )
        return torch.where(mask_bool.unsqueeze(-1), tokens, torch.zeros_like(tokens)), mask_t

    def update_spectral_buffers(self) -> None:
        """Actualiza buffers espectrales `u`/`v` fuera de `forward` (compatible CUDA Graphs)."""
        self.pc_cycle.update_spectral_buffers()

    def register_spectral_post_step_hook(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.utils.hooks.RemovableHandle:
        """Registra un hook post-step en `optimizer` para actualizar `_SpectralLinear`."""

        def _hook(_opt: torch.optim.Optimizer, _args: object, _kwargs: object) -> None:
            self.update_spectral_buffers()

        return optimizer.register_step_post_hook(_hook)

    def _structural_energy(
        self,
        pred: dict[str, torch.Tensor],
        *,
        exposures: torch.Tensor | None = None,
        ood_energy: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energía estructural verificable en `t` por activo `[N]` respetando `horizon_mask`."""
        ret = pred["pred_return"]
        alpha = pred["alpha"]
        q = pred["quantiles"]
        vol = pred["vol_pred"]
        n_assets, n_h = ret.shape
        device, dtype = ret.device, ret.dtype
        calc_dtype = _stable_calc_dtype(dtype)

        ret_c = ret.to(dtype=calc_dtype)
        alpha_c = alpha.to(dtype=calc_dtype)
        q_c = q.to(dtype=calc_dtype)
        vol_c = vol.to(dtype=calc_dtype)

        h_mask: torch.Tensor | None = None
        if horizon_mask is not None:
            hm_raw = horizon_mask.to(device=device)
            hm_bool = hm_raw > 0.5
            if hm_bool.ndim == 1:
                h_mask = hm_bool[:n_h].to(dtype=calc_dtype).unsqueeze(0)
            else:
                h_mask = hm_bool[:, :n_h].to(dtype=calc_dtype)

        if n_h >= 2:
            scales = self.horizon_scales[:n_h].to(device=device, dtype=calc_dtype)
            norm_r = ret_c / scales.unsqueeze(0)
            pair_diff_sq = (norm_r[:, 1:] - norm_r[:, :-1]) ** 2
            if h_mask is not None:
                pair_m = h_mask[:, 1:] * h_mask[:, :-1]
                pair_cnt = pair_m.sum(dim=-1)
                e_hor = torch.where(
                    pair_cnt > 0.5,
                    (pair_diff_sq * pair_m).sum(dim=-1) / pair_cnt.clamp_min(1.0),
                    torch.zeros(n_assets, device=device, dtype=calc_dtype),
                )
            else:
                e_hor = pair_diff_sq.mean(dim=-1)
        else:
            e_hor = torch.zeros(n_assets, device=device, dtype=calc_dtype)

        cross_per_h = F.relu(q_c[..., :-1] - q_c[..., 1:]).pow(2).mean(dim=-1)
        iqr_sigma = (
            (q_c[..., 6] - q_c[..., 2]) / 2.5631
            if q_c.shape[-1] >= 9
            else (q_c[..., -1] - q_c[..., 0]) / 2.5631
        )
        vol_err_h = (iqr_sigma - vol_c) ** 2
        if h_mask is not None:
            h_cnt = h_mask.sum(dim=-1).clamp_min(1.0)
            cross_viol = (cross_per_h * h_mask).sum(dim=-1) / h_cnt
            e_cdf = cross_viol + (vol_err_h * h_mask).sum(dim=-1) / h_cnt
        else:
            cross_viol = cross_per_h.mean(dim=-1)
            e_cdf = cross_viol + vol_err_h.mean(dim=-1)

        a0 = alpha_c[:, 0]
        mean_pen = a0.mean().pow(2).expand(n_assets)
        if exposures is not None and exposures.shape[-1] > 0 and n_assets > 1:
            b = exposures.to(device=device, dtype=calc_dtype)
            b_c = b - b.mean(dim=0, keepdim=True)
            b_n = b_c / b_c.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-4)
            fac_pen = ((b_n.T @ a0) / float(n_assets)).pow(2).mean().expand(n_assets)
        else:
            fac_pen = torch.zeros(n_assets, device=device, dtype=calc_dtype)

        e_ood = (
            ood_energy.to(device=device, dtype=calc_dtype).clamp_min(0.0) * (a0**2)
            if ood_energy is not None
            else torch.zeros(n_assets, device=device, dtype=calc_dtype)
        )
        return (e_hor + e_cdf + 0.5 * (mean_pen + fac_pen) + 0.5 * e_ood).to(dtype=dtype)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        state: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Alias estándar `nn.Module.forward` hacia `forward_session` (para `torch.compile`)."""
        return self.forward_session(batch, state=state)

    def forward_session(
        self,
        batch: dict[str, torch.Tensor],
        state: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Ejecuta una sesión causal completa de Familia 2 en GPU con cero `.item()`."""
        st: dict[str, object] = {} if state is None else dict(state)
        mod_tokens, mod_mask = self._encode_modalities(batch)
        n_assets = mod_tokens.shape[0]
        device, dtype = mod_tokens.device, mod_tokens.dtype
        c = self.config

        phys_raw = batch.get("physics_v4")
        if phys_raw is None:
            phys_raw = torch.zeros((n_assets, c.physics_dim), device=device, dtype=dtype)
        phys_t = phys_raw.to(device=device, dtype=dtype)
        phys_in = _lgn_dual_transform(phys_t) if c.use_lgn_dual_input else phys_t
        phys_emb = self.enc_physics(phys_in)

        ctx = batch.get("high_snr_context")
        if ctx is None:
            ctx = torch.zeros((n_assets, c.context_dim), device=device, dtype=dtype)
        else:
            ctx = torch.nan_to_num(
                ctx.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0
            )

        regime_vec = batch.get("regime_vec")
        if regime_vec is None:
            regime_vec = ctx.mean(dim=0)
        else:
            regime_vec = torch.nan_to_num(
                regime_vec.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0
            )

        rmt_loadings = batch.get("rmt_loadings")
        exposures = batch.get("exposures")
        horizon_mask = batch.get("horizon_mask")
        asset_ids = batch.get("asset_ids")
        modality_age = batch.get("modality_age")

        # 1) Cerebro Holográfico 3 Niveles (zero-sync)
        mod_tokens, holo_l2_ent = _run_holographic_3l_zero_sync(
            self.holographic_3l,
            mod_tokens,
            mod_mask,
            regime_vec=regime_vec,
            rmt_loadings=rmt_loadings,
        )
        mod_mask_bool = mod_mask > 0.5
        denom = mod_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mod_pooled = (
            torch.where(mod_mask_bool.unsqueeze(-1), mod_tokens, torch.zeros_like(mod_tokens)).sum(
                dim=1
            )
            / denom
        )
        x0 = self.fusion_norm(self.fusion_proj(torch.cat([mod_pooled, phys_emb], dim=-1)))

        # 2) Coincidencia dendrítica basal × apical (Larkum)
        dend_diag: dict[str, torch.Tensor] = {}
        if self.dendritic is not None:
            x0, dend_diag = self.dendritic(x0, ctx)

        # 3) Normalización divisiva + inhibición lateral cortical (diagnósticos Gram solo en eval!)
        lat_diag: dict[str, torch.Tensor] = {}
        if self.lateral_inhibition is not None:
            x0, lat_diag = self.lateral_inhibition(x0, return_diagnostics=not self.training)

        # 3b) Escalado homeostático sináptico de Turrigiano causal t-1 (v2, 0 mutación en fwd!)
        if self.homeostatic_scaling is not None:
            prev_homeo = st.get("homeo_rate_ema")
            prev_homeo_t = prev_homeo if isinstance(prev_homeo, torch.Tensor) else None
            x0, next_homeo_ema = self.homeostatic_scaling(x0, prev_homeo_t)
            st["homeo_rate_ema"] = next_homeo_ema

        # 4) Recuperación asociativa HRR-Modern-Hopfield (zero-sync; FHRR unitario 0-FFT en v2!)
        ood_energy = torch.zeros(n_assets, device=device, dtype=dtype)
        if self.hrr_hopfield is not None:
            reg_key = self.regime_proj(regime_vec)
            retrieved, ood_energy = _run_hrr_lookup_zero_sync(
                self.hrr_hopfield,
                x0,
                regime_key=reg_key,
                use_unitary_fhrr=c.use_unitary_fhrr,
            )
            x0 = x0 + retrieved

        # 5) Lectura de Plasticidad de 3 Factores (estrictamente madura <= t-1)
        plast_diag: dict[str, torch.Tensor] = {}
        x_pre_plast = x0
        if self.three_factor_plasticity is not None:
            x0, plast_diag = self.three_factor_plasticity.read(x0, st.get("plasticity_state"))

        # 6) Cebado episódico hipocampal SWR
        replay_diag: dict[str, torch.Tensor] = {}
        if self.replay_memory is not None:
            x0, replay_diag = self.replay_memory(x0)

        # 6b) Precisión sensorial Active Inference Pi_{i,t} in [0.25, 1.0] (v2) o escalar (v1)
        ai_pi_diag: dict[str, torch.Tensor] = {}
        layer_pi_snap = (
            self.layer_precision.clone() if torch.is_grad_enabled() else self.layer_precision
        )
        if self.active_inference_pi is not None:
            err_var_snap = (
                self.pc_err_var_ema.clone() if torch.is_grad_enabled() else self.pc_err_var_ema
            )
            precision_pi, ai_pi_diag = self.active_inference_pi(
                mod_mask=mod_mask,
                ctx=ctx,
                ood_energy=ood_energy,
                pc_err_var_ema=err_var_snap,
                modality_age=modality_age,
            )
        else:
            precision_pi = layer_pi_snap

        # 7) Codificación Predictiva Contractiva (1..max_cycles) + reutilización exacta de `head`
        cached_pc = self.pc_cycle.compute_spectral_weights(update_buffers=False)
        h = x0
        pred_cur = self.head(x0, exposures=exposures)
        best_energy = self._structural_energy(
            pred_cur, exposures=exposures, ood_energy=ood_energy, horizon_mask=horizon_mask
        ).detach()

        placebo_bank: torch.Tensor | None = None
        placebo_filled: torch.Tensor | None = None
        placebo_ptr: torch.Tensor | None = None
        if self.moe_bridges is not None and c.moe_mode == "shuffled_context":
            raw_bank = st.get("placebo_ctx_bank")
            raw_filled = st.get("placebo_ctx_filled")
            raw_ptr = st.get("placebo_ctx_ptr")
            if (
                isinstance(raw_bank, torch.Tensor)
                and isinstance(raw_filled, torch.Tensor)
                and isinstance(raw_ptr, torch.Tensor)
            ):
                placebo_bank = raw_bank
                placebo_filled = raw_filled
                placebo_ptr = raw_ptr

        cycle_outputs: list[dict[str, torch.Tensor]] = []
        pc_errors: list[torch.Tensor] = []
        pc_sq_err_list: list[torch.Tensor] = []
        monotonic_accepts: list[torch.Tensor] = []
        energy_traj: list[torch.Tensor] = [best_energy.mean()]
        moe_entropies: list[torch.Tensor] = []
        moe_loads: list[torch.Tensor] = []

        calc_dtype = _stable_calc_dtype(dtype)
        pi_inv_det = (
            1.0 / precision_pi.detach().to(device=device, dtype=calc_dtype).clamp(0.25, 1.0)
        )
        if pi_inv_det.ndim == 1:
            pi_inv_det = pi_inv_det.unsqueeze(0)

        for cycle_idx in range(c.max_cycles):
            h_in = h
            if self.moe_bridges is not None:
                h_in, m_ent, m_loads = _run_high_snr_moe_zero_sync(
                    self.moe_bridges[cycle_idx],
                    h_in,
                    ctx,
                    placebo_ctx_bank=placebo_bank,
                    placebo_ctx_filled=placebo_filled,
                    placebo_ctx_ptr=placebo_ptr,
                )
                moe_entropies.append(m_ent)
                moe_loads.append(m_loads)

            h_cand, pc_err = self.pc_cycle(
                x0, h_in, precision_pi, cached_weights=cached_pc
            )
            pc_errors.append(pc_err.norm(dim=-1).mean())
            if self.active_inference_pi is not None:
                raw_err = pc_err.detach().to(dtype=calc_dtype) * pi_inv_det
                pc_sq_err_list.append(raw_err.pow(2).mean(dim=0))

            pred_cand = self.head(h_cand, exposures=exposures)
            e_cand = self._structural_energy(
                pred_cand, exposures=exposures, ood_energy=ood_energy, horizon_mask=horizon_mask
            )
            worsened = F.relu(e_cand - best_energy)
            accept_k = torch.exp(-c.monotonic_penalty * worsened)
            best_energy = torch.minimum(best_energy, e_cand.detach())

            h_eff = accept_k.unsqueeze(-1) * h_cand + (1.0 - accept_k.unsqueeze(-1)) * h
            pred_eff = _blend_head_predictions(pred_cand, pred_cur, accept_k)
            monotonic_accepts.append(accept_k)
            energy_traj.append(e_cand.mean())
            cycle_outputs.append(pred_eff)
            h = h_eff
            pred_cur = pred_eff

        if self.polar_alpha_head is not None:
            vol_det: torch.Tensor | None = None
            for v_key in ("realized_vol", "volatility", "hist_vol"):
                v_raw = batch.get(v_key)
                if isinstance(v_raw, torch.Tensor):
                    vol_det = v_raw.detach().to(device=device, dtype=dtype)
                    break
            polar_cycles: list[dict[str, torch.Tensor]] = []
            for cyc in cycle_outputs:
                v_step = vol_det
                if v_step is None:
                    v_cand = cyc.get("rough_vol")
                    if not isinstance(v_cand, torch.Tensor):
                        v_cand = cyc.get("vol_pred")
                    v_step = v_cand.detach() if isinstance(v_cand, torch.Tensor) else None
                p_out = self.polar_alpha_head(cyc["alpha"], vol=v_step)
                cyc_new = dict(cyc)
                cyc_new["alpha_raw"] = cyc["alpha"]
                cyc_new["alpha_u"] = p_out["u"]
                cyc_new["alpha_cal"] = p_out["alpha_cal"]
                cyc_new["alpha_cal_for_cal"] = p_out["alpha_cal_for_cal"]
                cyc_new["alpha_cal_for_pplus"] = p_out["alpha_cal_for_pplus"]
                cyc_new["alpha"] = p_out["alpha_cal"]
                # #233 P0-1: Pure calibrated zero-mean alpha (no untrained factor_return noise!)
                cyc_new["factor_return"] = torch.zeros_like(p_out["alpha_cal"])
                cyc_new["pred_return"] = p_out["alpha_cal"]
                polar_cycles.append(cyc_new)
            cycle_outputs = polar_cycles

        final_pred = cycle_outputs[-1]
        if pc_sq_err_list:
            st["last_pc_sq_err"] = torch.stack(pc_sq_err_list, dim=0).mean(dim=0).detach()

        # Actualizar banco placebo cross-session DESPUÉS del forward (estrictamente causal <= t)
        if self.moe_bridges is not None and c.moe_mode == "shuffled_context":
            bank_cap = 32
            if placebo_bank is None or placebo_filled is None or placebo_ptr is None:
                next_bank = torch.zeros((bank_cap, c.context_dim), device=device, dtype=dtype)
                next_filled = torch.zeros((bank_cap,), device=device, dtype=torch.bool)
                next_ptr = torch.zeros((), device=device, dtype=torch.long)
            else:
                next_bank = placebo_bank.to(device=device, dtype=dtype).clone()
                next_filled = placebo_filled.to(device=device).clone()
                next_ptr = placebo_ptr.to(device=device, dtype=torch.long).clone()
            slot = next_ptr % next_bank.shape[0]
            slot_oh = (
                torch.arange(next_bank.shape[0], device=device, dtype=torch.long) == slot
            ).to(dtype=dtype)
            ctx_mean_det = ctx.detach().mean(dim=0)
            next_bank = next_bank * (1.0 - slot_oh.unsqueeze(-1)) + ctx_mean_det.unsqueeze(
                0
            ) * slot_oh.unsqueeze(-1)
            next_filled = next_filled | (slot_oh > 0.5)
            st["placebo_ctx_bank"] = next_bank.detach()
            st["placebo_ctx_filled"] = next_filled.detach()
            st["placebo_ctx_ptr"] = (next_ptr + 1).detach()

        # Registrar traza de elegibilidad causal en t (sin modificar W_fast hasta step_matured!)
        if self.three_factor_plasticity is not None:
            st["plasticity_state"] = self.three_factor_plasticity.record_eligibility(
                x_pre_plast, h, st.get("plasticity_state"), asset_ids=asset_ids
            )
        st["last_x0"] = x0.detach()
        st["last_h"] = h.detach()
        st["last_ctx"] = ctx.detach()
        st["last_regime_vec"] = regime_vec.detach()
        if isinstance(asset_ids, torch.Tensor):
            st["last_asset_ids"] = asset_ids.detach()
        st["last_moe_loads"] = moe_loads
        st.setdefault("hrr_slots_used", torch.zeros((), device=device, dtype=dtype))
        st.setdefault("replay_slots_used", torch.zeros((), device=device, dtype=dtype))

        pi_diag_mean = ai_pi_diag.get(
            "pi_mean", layer_pi_snap.mean().to(device=device, dtype=dtype)
        )
        pi_diag_min = ai_pi_diag.get(
            "pi_min", layer_pi_snap.min().to(device=device, dtype=dtype)
        )
        pi_diag_max = ai_pi_diag.get(
            "pi_max", layer_pi_snap.max().to(device=device, dtype=dtype)
        )
        pi_diag_std = ai_pi_diag.get(
            "pi_std", layer_pi_snap.std(unbiased=False).to(device=device, dtype=dtype)
        )
        diagnostics: dict[str, object] = dict(
            family="holographic_brain",
            ablation=c.ablation,
            max_cycles=c.max_cycles,
            lipschitz_upper_bound=self.pc_cycle.lipschitz_upper_bound(),
            pc_error_norm=torch.stack(pc_errors).mean(),
            monotonic_accept_rate=torch.stack(monotonic_accepts, dim=0).mean(),
            monotonic_accepts_per_step=monotonic_accepts,
            structural_energy_trajectory=torch.stack(energy_traj),
            holo_l2_entropy=holo_l2_ent,
            ood_energy_mean=ood_energy.mean(),
            active_inference_pi_mean=pi_diag_mean,
            active_inference_pi_min=pi_diag_min,
            active_inference_pi_max=pi_diag_max,
            active_inference_pi_std=pi_diag_std,
            dendritic=dend_diag,
            lateral_inhibition=lat_diag,
            plasticity=plast_diag,
            replay=replay_diag,
            moe_router_entropy=(
                torch.stack(moe_entropies).mean()
                if moe_entropies
                else torch.zeros((), device=device, dtype=dtype)
            ),
            moe_expert_loads=moe_loads,
        )

        out: dict[str, object] = dict(
            pred_return=final_pred["pred_return"],
            alpha=final_pred["alpha"],
            factor_return=final_pred["factor_return"],
            quantiles=final_pred["quantiles"],
            tb_logits=final_pred["tb_logits"],
            tb_probs=final_pred["tb_probs"],
            turn_logits=final_pred["turn_logits"],
            turning_prob=final_pred["turning_prob"],
            log_vol=final_pred["log_vol"],
            vol_pred=final_pred["vol_pred"],
            epistemic_error=final_pred["epistemic_error"],
            final_hidden=h,
            cycle_outputs=cycle_outputs,
            diagnostics=diagnostics,
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
                out[p_key] = final_pred[p_key]
        return out

    @torch.no_grad()
    def step_matured(
        self,
        state: dict[str, object] | None,
        *,
        matured_targets: torch.Tensor,
        matured_preds: torch.Tensor | None = None,
        matured_ic: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Consolidación estrictamente causal en `t+1`: 3-factores, HRR, SWR y precisión Pi_l."""
        st: dict[str, object] = {} if state is None else dict(state)
        last_h = st.get("last_h")
        if last_h is None:
            return st
        device, dtype = last_h.device, last_h.dtype
        calc_dtype = _stable_calc_dtype(dtype)

        y_t = matured_targets.detach().to(device=device, dtype=calc_dtype)
        y1 = y_t[:, 0] if y_t.ndim == 2 else y_t
        fin_y1 = torch.isfinite(y1)
        fin_y1_cnt = fin_y1.to(dtype=calc_dtype).sum().clamp_min(1.0)
        has_fin_y1 = fin_y1.any()
        y1_safe = torch.where(fin_y1, y1, torch.zeros_like(y1))

        if matured_preds is not None:
            p_t = matured_preds.detach().to(device=device, dtype=calc_dtype)
            p1 = p_t[:, 0] if p_t.ndim == 2 else p_t
            fin_pair = fin_y1 & torch.isfinite(p1)
            pair_cnt = fin_pair.to(dtype=calc_dtype).sum().clamp_min(1.0)
            has_pair_fin = fin_pair.any()
            y1_p = torch.where(fin_pair, y1, torch.zeros_like(y1))
            p1_p = torch.where(fin_pair, p1, torch.zeros_like(p1))

            err_vec = torch.where(fin_pair, (y1_p - p1_p).abs(), torch.zeros_like(y1))
            err_raw = err_vec.sum() / pair_cnt
            err_scale = torch.where(
                has_pair_fin, err_raw, torch.full((), 0.02, device=device, dtype=calc_dtype)
            )
            y1_abs_m = (y1_p.abs().sum() / pair_cnt).clamp_min(1e-3)
            p1_abs_m = (p1_p.abs().sum() / pair_cnt).clamp_min(1e-3)
            rpe_raw = torch.clamp((p1_p * y1_p) / (y1_abs_m * p1_abs_m), -5.0, 5.0)
            rpe_vec = torch.where(fin_pair, rpe_raw, torch.zeros_like(y1))

            if matured_ic is None:
                yc = torch.where(fin_pair, y1_p - y1_p.sum() / pair_cnt, torch.zeros_like(y1))
                pc = torch.where(fin_pair, p1_p - p1_p.sum() / pair_cnt, torch.zeros_like(p1))
                ic_raw = (yc * pc).sum() / (
                    yc.norm().clamp_min(1e-6) * pc.norm().clamp_min(1e-6)
                )
                ic_val = torch.where(
                    pair_cnt > 1.5, ic_raw, torch.zeros((), device=device, dtype=calc_dtype)
                )
            else:
                ic_in = matured_ic.detach().to(device=device, dtype=calc_dtype).mean()
                ic_val = torch.where(
                    torch.isfinite(ic_in), ic_in, torch.zeros((), device=device, dtype=calc_dtype)
                )
            asset_rpe_score = err_vec * 50.0 + rpe_vec.abs() + ic_val.abs()
        else:
            y1_mean = y1_safe.sum() / fin_y1_cnt
            y1_dev = torch.where(fin_y1, y1_safe - y1_mean, torch.zeros_like(y1))
            y1_std = torch.sqrt((y1_dev.pow(2).sum() / fin_y1_cnt).clamp_min(1e-6))
            rpe_vec = torch.where(
                fin_y1, torch.clamp(y1_safe / y1_std, -5.0, 5.0), torch.zeros_like(y1)
            )
            err_scale = torch.where(
                has_fin_y1,
                y1_safe.abs().sum() / fin_y1_cnt,
                torch.full((), 0.02, device=device, dtype=calc_dtype),
            )
            if matured_ic is not None:
                ic_in = matured_ic.detach().to(device=device, dtype=calc_dtype).mean()
                ic_val = torch.where(
                    torch.isfinite(ic_in), ic_in, torch.zeros((), device=device, dtype=calc_dtype)
                )
            else:
                ic_val = torch.zeros((), device=device, dtype=calc_dtype)
            asset_rpe_score = y1_safe.abs() * 50.0 + rpe_vec.abs() + ic_val.abs()

        # 1) Consolidación de 3 factores con alineación canónica `asset_ids` bajo PT1
        if self.three_factor_plasticity is not None and st.get("plasticity_state") is not None:
            new_plast, plast_c_diag = self.three_factor_plasticity.consolidate(
                st["plasticity_state"],
                matured_rpe=rpe_vec,
                matured_ic=ic_val,
                asset_ids=asset_ids,
            )
            st["plasticity_state"] = new_plast
            st["last_plasticity_consolidation"] = plast_c_diag

        # Alinear estados (last_x0, last_h, last_ctx) con activos maduros si hay `asset_ids`
        last_x0 = st.get("last_x0")
        last_ctx = st.get("last_ctx")
        last_ids_raw = st.get("last_asset_ids")
        n_mat = y1.shape[0]
        if (
            asset_ids is not None
            and isinstance(last_ids_raw, torch.Tensor)
            and last_ids_raw.numel() == last_h.shape[0]
            and last_ids_raw.numel() > 0
            and asset_ids.numel() == n_mat
        ):
            l_ids = last_ids_raw.to(device=device, dtype=torch.long).reshape(-1)
            order = torch.argsort(l_ids)
            sorted_l_ids = l_ids[order]
            cur_ids = asset_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
            pos = torch.searchsorted(sorted_l_ids, cur_ids)
            safe_pos = pos.clamp_max(sorted_l_ids.shape[0] - 1)
            matched_mat = (pos < sorted_l_ids.shape[0]) & (sorted_l_ids[safe_pos] == cur_ids)
            h_aligned = last_h[order][safe_pos]
            x0_aligned = last_x0[order][safe_pos] if isinstance(last_x0, torch.Tensor) else None
            ctx_aligned = (
                last_ctx[order][safe_pos]
                if isinstance(last_ctx, torch.Tensor)
                else torch.zeros((n_mat, self.config.context_dim), device=device, dtype=dtype)
            )
            y1_aligned = torch.where(matched_mat, y1_safe, torch.zeros_like(y1_safe))
            y_store_base = y_t.unsqueeze(-1) if y_t.ndim == 1 else y_t
            nan_fill = torch.full_like(y_store_base, float("nan"))
            y_store = torch.where(matched_mat.unsqueeze(-1), y_store_base, nan_fill)
            rpe_store = torch.where(matched_mat, asset_rpe_score, torch.zeros_like(asset_rpe_score))
        else:
            n_common = min(last_h.shape[0], n_mat)
            h_aligned = last_h[:n_common]
            x0_aligned = last_x0[:n_common] if isinstance(last_x0, torch.Tensor) else None
            ctx_aligned = (
                last_ctx[:n_common]
                if isinstance(last_ctx, torch.Tensor)
                else torch.zeros((n_common, self.config.context_dim), device=device, dtype=dtype)
            )
            y1_aligned = y1_safe[:n_common]
            y_store_base = y_t.unsqueeze(-1) if y_t.ndim == 1 else y_t
            y_store = y_store_base[:n_common]
            rpe_store = asset_rpe_score[:n_common]

        # 2) Almacenamiento episódico en HRR-Modern-Hopfield (zero-sync, BF16-AMP safe)
        if self.hrr_hopfield is not None and x0_aligned is not None and x0_aligned.shape[0] > 0:
            reg_vec = st.get("last_regime_vec")
            proj_dtype = self.regime_proj.weight.dtype
            reg_key = (
                self.regime_proj(reg_vec.to(device=device, dtype=proj_dtype))
                if reg_vec is not None
                else None
            )
            x0_mem = x0_aligned.to(device=device, dtype=proj_dtype)
            val_mem = x0_mem * (
                1.0 + torch.tanh(y1_aligned.to(dtype=proj_dtype).unsqueeze(-1) * 20.0)
            )
            st["hrr_slots_used"] = _store_hrr_zero_sync(
                self.hrr_hopfield,
                x0_mem,
                val_mem,
                regime_key=reg_key,
                return_signal=y1_aligned,
                use_unitary_fhrr=self.config.use_unitary_fhrr,
            )

        # 3) Almacenamiento prioritario SWR: top-K activos individuales de alto |RPE| con targets!
        if self.replay_memory is not None and h_aligned.shape[0] > 0:
            surprise = self.replay_memory.compute_surprise(h_aligned)
            st["replay_slots_used"] = self.replay_memory.store_episode(
                h_aligned,
                ctx_aligned,
                y_store,
                surprise=surprise,
                matured_rpe=rpe_store,
            )

        # 4) Actualización causal de precisión Pi_l en dispositivo (escala-libre en v2, 0 sync)
        last_pc_sq = st.get("last_pc_sq_err")
        if self.config.use_active_inference_pi and isinstance(last_pc_sq, torch.Tensor):
            sq_c = last_pc_sq.detach().to(
                device=self.pc_err_var_ema.device, dtype=self.pc_err_var_ema.dtype
            )
            sq_safe = torch.where(
                torch.isfinite(sq_c), sq_c.clamp(1e-6, 1e4), self.pc_err_var_ema
            )
            self.pc_err_var_ema.mul_(0.9).add_(0.1 * sq_safe)
            var_ema = self.pc_err_var_ema.clamp_min(1e-6)
            pi_ch = torch.clamp(var_ema.median() / var_ema, min=0.25, max=1.0)
            self.layer_precision.copy_(pi_ch.to(dtype=self.layer_precision.dtype))
        else:
            pi_target = torch.clamp(0.02 / err_scale.clamp_min(0.005), min=0.25, max=1.0).to(
                dtype=self.layer_precision.dtype
            )
            self.layer_precision.mul_(0.9).add_(0.1 * pi_target)

        # 5) Balanceo Aux-Loss-Free del puente MoE SOLO durante training (congelado en eval()!)
        if (
            self.training
            and self.moe_bridges is not None
            and self.config.moe_mode in ("soft", "sparsemax")
        ):
            loads_list = st.get("last_moe_loads") or []
            for blk, loads in zip(self.moe_bridges, loads_list, strict=False):
                blk.update_balance_bias(loads, lr=0.05)

        return st

    def compute_replay_loss(self, *, batch_size: int = 8) -> torch.Tensor:
        """Calcula pérdida de consolidación hipocampal SWR (0 exacto si desactivado o vacío)."""
        ref_param = next(self.head.parameters())
        if self.replay_memory is None:
            return torch.zeros((), device=ref_param.device, dtype=ref_param.dtype)
        return self.replay_memory.replay_regularization_loss(self.head, batch_size=batch_size)

    def parameter_audit(self) -> dict[str, object]:
        """Audita parámetros totales, entrenables, ablación y huella SHA-256 de Familia 2."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(
            family="holographic_brain",
            ablation=self.config.ablation,
            fingerprint=self.config.fingerprint(),
            total_params=int(total),
            trainable_params=int(trainable),
            max_cycles=self.config.max_cycles,
            use_dendritic=self.config.use_dendritic,
            use_lateral_inhibition=self.config.use_lateral_inhibition,
            use_three_factor=self.config.use_three_factor,
            use_replay=self.config.use_replay,
            use_moe_bridge=self.config.use_moe_bridge,
            moe_mode=self.config.moe_mode,
            use_unitary_fhrr=self.config.use_unitary_fhrr,
            use_lgn_dual_input=self.config.use_lgn_dual_input,
            use_active_inference_pi=self.config.use_active_inference_pi,
            use_homeostatic_bcm=self.config.use_homeostatic_bcm,
            use_stratified_swr=self.config.use_stratified_swr,
            noic_pinball_twin=self.config.noic_pinball_twin,
        )


def build_neuro_brain_ablation(
    name: str = "full",
    **overrides: object,
) -> StandaloneHolographicBrainModel:
    """Fábrica oficial de las ablaciones emparejadas de Familia 2 (`holographic_brain`)."""
    if name not in NEURO_BRAIN_ABLATIONS:
        raise ValueError(
            f"Ablación desconocida {name!r}; válidas: {tuple(NEURO_BRAIN_ABLATIONS.keys())}"
        )
    spec: dict[str, object] = dict(NEURO_BRAIN_ABLATIONS[name])
    spec["ablation"] = name
    spec.update(overrides)
    return StandaloneHolographicBrainModel(NeuroBrainConfig(**spec))


__all__ = [
    "DEFAULT_BRAIN_CYCLES",
    "MAX_BRAIN_CYCLES",
    "NEURO_BRAIN_ABLATIONS",
    "ActiveInferencePrecisionGate",
    "DendriticCompartmentLayer",
    "HippocampalSharpWaveReplay",
    "LateralInhibitionCompetition",
    "NeuroBrainConfig",
    "StandaloneHolographicBrainModel",
    "ThreeFactorDopaminePlasticity",
    "TurrigianoHomeostaticScaling",
    "_fhrr_bind",
    "_fhrr_unbind",
    "_fhrr_unitary_pair_norm",
    "_lgn_dual_transform",
    "build_neuro_brain_ablation",
]
