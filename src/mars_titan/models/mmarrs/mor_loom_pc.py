"""MoE Recursivo de Doble Bucle (`MoR-LOOM-Fin`) + Verificador de Energía + `PC-TRM-Eq`.

Diseño 100 % vectorizado y libre de host-syncs GPU (`audit_zero_host_syncs == 0`) para NVIDIA
RTX PRO 6000 Blackwell (`sm_120`):
  1. `HighSNRMoELayer`: Expertos Compartidos + Enrutados en tensores 3D contiguos, combinación
     fusionada vía `fused_soft_moe_combine` (autograd Triton `sm_120`), balanceo Aux-Loss-Free
     congelado estrictamente en `eval()`, control uniforme `1/E` (v04) y placebo determinista de
     contexto barajado entre sesiones (v05, biyectivo incluso con `7 | N`).
  2. `FinancialEnergyVerifier`: 4 invariantes financieros causales con soporte explícito de
     `horizon_mask` (anula `e_horizon` cuando solo hay 1 horizonte activo, P0-4).
  3. `PCTRMRecursiveLoop`: Codificación Predictiva Bidireccional con `layer_precision <= 1.0`,
     contracción de Banach certificada (`Lipschitz < 1` garantizado), routers LOOM `R^(k)` por
     paso (`max_loops=3` por omisión, tope duro `<= 4`) y aceptación monótona sin host-syncs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .blackwell_kernels import fused_soft_moe_combine
from .counterfactual_sim import (
    CF_TRACE_DIM,
    CounterfactualStressSimulator,
    prepare_cf_factor_cache,
)
from .dl_training_sota import build_norm_layer
from .frontier_modules_sota import DiffInducedFactorCrossAttention

MAX_LOOP_STEPS: int = 4


def _stable_calc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promociona FP16/BF16 a FP32 para álgebra lineal/logsumexp y preserva FP32/FP64."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Proyección euclídea exacta sobre el símplex (Martins & Astudillo, ICML 2016)."""
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.ndim
    view_shape[dim] = -1
    k = k.view(*view_shape)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    support = (1.0 + k * z_sorted) > cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (torch.gather(cumsum, dim, k_z - 1) - 1.0) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


class HighSNRMoELayer(nn.Module):
    """Capa MoE fusionada 3D (`Shared + Routed`) con balanceo DeepSeek-V3 Aux-Loss-Free."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        n_shared_experts: int = 2,
        n_routed_experts: int = 6,
        expert_mult: float = 1.5,
        mode: str = "soft",
        temperature: float = 1.0,
        use_fused_blackwell_kernels: bool = True,
        build_default_router: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.n_shared = n_shared_experts
        self.n_routed = n_routed_experts
        self.d_ff = max(8, int(round(hidden_dim * expert_mult)))
        self.mode = mode
        self.temperature = float(temperature)
        self.use_fused_blackwell_kernels = bool(use_fused_blackwell_kernels)

        self.in_norm = nn.LayerNorm(hidden_dim)
        if build_default_router:
            self.ctx_norm: nn.LayerNorm | None = nn.LayerNorm(context_dim)
            self.router: nn.Linear | None = nn.Linear(context_dim, n_routed_experts)
            # Sesgo dinámico Aux-Loss-Free (solo se actualiza en `train()`, congelado en `eval()`)
            self.register_buffer(
                "balance_bias", torch.zeros(n_routed_experts, dtype=torch.float32)
            )
        else:
            self.ctx_norm = None
            self.router = None
            self.register_buffer("balance_bias", None)

        # Tensores 3D contiguos: evalúan TODOS los expertos en 2 einsums (188 SMs + L2 128 MB!)
        s_scale = 1.0 / math.sqrt(hidden_dim)
        f_scale = 1.0 / math.sqrt(self.d_ff)
        self.shared_w1 = nn.Parameter(
            torch.randn(n_shared_experts, hidden_dim, self.d_ff) * s_scale
        )
        self.shared_b1 = nn.Parameter(torch.zeros(n_shared_experts, self.d_ff))
        self.shared_w2 = nn.Parameter(
            torch.randn(n_shared_experts, self.d_ff, hidden_dim) * f_scale
        )
        self.shared_b2 = nn.Parameter(torch.zeros(n_shared_experts, hidden_dim))

        self.routed_w1 = nn.Parameter(
            torch.randn(n_routed_experts, hidden_dim, self.d_ff) * s_scale
        )
        self.routed_b1 = nn.Parameter(torch.zeros(n_routed_experts, self.d_ff))
        self.routed_w2 = nn.Parameter(
            torch.randn(n_routed_experts, self.d_ff, hidden_dim) * f_scale
        )
        self.routed_b2 = nn.Parameter(torch.zeros(n_routed_experts, hidden_dim))

        self.mix_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    @torch.no_grad()
    def update_balance_bias(self, expert_loads: torch.Tensor, *, lr: float = 0.05) -> None:
        """Balanceo DeepSeek-V3 Aux-Loss-Free: estrictamente congelado en `eval()` (P1-3)."""
        if not self.training or self.balance_bias is None:
            return
        target = 1.0 / float(self.n_routed)
        loads = expert_loads.detach().to(device=self.balance_bias.device, dtype=torch.float32)
        diff = target - loads
        step = lr * torch.sign(diff) * torch.clamp(diff.abs() * self.n_routed, 0.2, 2.0)
        self.balance_bias.add_(step)
        self.balance_bias.sub_(self.balance_bias.mean())
        self.balance_bias.clamp_(-4.0, 4.0)

    def _prepare_placebo_context(
        self,
        ctx: torch.Tensor,
        placebo_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Construye contexto placebo barajado entre sesiones (P1-3; biyectivo para todo N)."""
        n_assets, c_dim = ctx.shape
        if placebo_context is not None:
            p_ctx = placebo_context.to(device=ctx.device, dtype=ctx.dtype)
            if p_ctx.ndim == 1:
                p_ctx = p_ctx.unsqueeze(0).expand(n_assets, -1)
            elif p_ctx.shape[0] != n_assets:
                idx = torch.arange(n_assets, device=ctx.device, dtype=torch.long) % p_ctx.shape[0]
                p_ctx = p_ctx[idx]
            if p_ctx.shape[-1] == c_dim:
                return p_ctx
            if p_ctx.shape[-1] < c_dim:
                # Mantener columnas extra de realimentación barajadas biyectivamente
                extra = ctx[:, p_ctx.shape[-1] :]
                if n_assets > 1:
                    extra = torch.roll(extra.flip(0), shifts=max(1, n_assets // 2), dims=0)
                return torch.cat([p_ctx, extra], dim=-1)
            return p_ctx[:, :c_dim]

        # Respaldo determinista sin contexto de otra sesión: biyección exacta en filas + roll dims
        out = ctx
        if n_assets > 1:
            out = torch.roll(out.flip(0), shifts=max(1, n_assets // 2), dims=0)
        if c_dim > 1:
            # Destruye también columnas constantes de mercado en llamadas aisladas
            out = torch.roll(out, shifts=max(1, c_dim // 2), dims=-1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        placebo_context: torch.Tensor | None = None,
        router_override: nn.Module | None = None,
        ctx_norm_override: nn.Module | None = None,
        balance_bias_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Ejecuta `Shared + Routed` en paralelo sin bucles sobre expertos ni host-syncs GPU."""
        n_assets = x.shape[0]
        h = self.in_norm(x)

        # 1) Expertos compartidos (siempre activos, anclan física universal del mercado)
        s_hid = F.silu(torch.einsum("nd,edf->nef", h, self.shared_w1) + self.shared_b1.unsqueeze(0))
        s_out = (
            torch.einsum("nef,efd->ned", s_hid, self.shared_w2) + self.shared_b2.unsqueeze(0)
        ).mean(dim=1)

        # 2) Enrutamiento sobre contexto de alto SNR según modo (incluye controles v04 y v05)
        if self.mode == "uniform":
            probs = torch.full(
                (n_assets, self.n_routed),
                1.0 / float(self.n_routed),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            ctx = context.to(device=x.device, dtype=x.dtype)
            if self.mode == "shuffled_context":
                ctx = self._prepare_placebo_context(ctx, placebo_context)
            eff_ctx_norm = ctx_norm_override if ctx_norm_override is not None else self.ctx_norm
            eff_router = router_override if router_override is not None else self.router
            eff_bb = (
                balance_bias_override if balance_bias_override is not None else self.balance_bias
            )
            assert eff_ctx_norm is not None and eff_router is not None
            logits = eff_router(eff_ctx_norm(ctx)) / self.temperature
            if eff_bb is not None:
                bb = eff_bb.to(device=logits.device, dtype=logits.dtype)
                if torch.is_grad_enabled():
                    bb = bb.clone()
                logits = logits + bb.unsqueeze(0)
            probs = (
                _sparsemax(logits, dim=-1)
                if self.mode == "sparsemax"
                else torch.softmax(logits, dim=-1)
            )

        # 3) Expertos enrutados en 2 einsums fusionados: [N, E, D]
        r_hid = F.silu(torch.einsum("nd,edf->nef", h, self.routed_w1) + self.routed_b1.unsqueeze(0))
        r_all = torch.einsum("nef,efd->ned", r_hid, self.routed_w2) + self.routed_b2.unsqueeze(0)

        if self.use_fused_blackwell_kernels:
            out = fused_soft_moe_combine(
                r_all,
                probs,
                s_out,
                self.mix_gate,
                residual=x,
                apply_sigmoid=True,
                use_triton=True,
            )
        else:
            r_out = torch.einsum("ne,ned->nd", probs, r_all)
            g = torch.sigmoid(self.mix_gate.to(dtype=x.dtype))
            out = x + g * r_out + (1.0 - g) * s_out

        # Diagnósticos de salud 100% en dispositivo (cero `.item()`; omite O(N*D) corr en train!)
        with torch.no_grad():
            loads = probs.mean(dim=0)
            max_l = loads.max()
            min_l = loads.min().clamp_min(1e-6)
            load_ratio = max_l / min_l
            ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
            if self.training:
                mean_corr = torch.zeros((), device=x.device, dtype=x.dtype)
            else:
                flat_e = r_all.permute(1, 0, 2).reshape(self.n_routed, -1).float()
                flat_e = flat_e - flat_e.mean(dim=1, keepdim=True)
                norms = flat_e.norm(dim=1, keepdim=True).clamp_min(1e-8)
                corr_m = (flat_e / norms) @ (flat_e / norms).T
                off_diag = corr_m[~torch.eye(self.n_routed, dtype=torch.bool, device=x.device)]
                mean_corr = (
                    off_diag.mean().to(dtype=x.dtype)
                    if off_diag.numel()
                    else torch.zeros((), device=x.device, dtype=x.dtype)
                )

        diag: dict[str, object] = dict(
            router_probs=probs,
            router_entropy_per_asset=ent,
            router_entropy=ent.mean(),
            expert_loads=loads,
            load_max_min_ratio=load_ratio,
            expert_mean_corr=mean_corr,
        )
        return out, diag


class FinancialEnergyVerifier(nn.Module):
    """Verificador de energía estructural financiera `E_fin(y^(k), x_t)` (sin mirar el futuro)."""

    def __init__(
        self,
        horizons: tuple[int, ...] = (1, 2, 5, 10, 21, 63),
        *,
        w_horizon: float = 1.0,
        w_cdf_vol: float = 1.0,
        w_cross_section: float = 0.5,
        w_ood: float = 0.5,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.w_horizon = w_horizon
        self.w_cdf_vol = w_cdf_vol
        self.w_cross_section = w_cross_section
        self.w_ood = w_ood
        self.register_buffer(
            "horizon_scales",
            torch.tensor([math.sqrt(float(h)) for h in horizons], dtype=torch.float32),
        )

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        *,
        exposures: torch.Tensor | None = None,
        ood_energy: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calcula los 4 términos de energía verificables en `t` respetando `horizon_mask`."""
        ret = pred["pred_return"]  # [N, H]
        alpha = pred["alpha"]  # [N, H]
        q = pred["quantiles"]  # [N, H, Q]
        vol = pred["vol_pred"]  # [N, H]
        n_assets, n_h = ret.shape
        device, dtype = ret.device, ret.dtype

        if horizon_mask is not None:
            h_mask = horizon_mask.to(device=device, dtype=dtype).reshape(-1)[:n_h]
            h_mask = (h_mask > 0.5).to(dtype=dtype)
        else:
            h_mask = torch.ones(n_h, device=device, dtype=dtype)
        h_sum = h_mask.sum().clamp_min(1.0)

        # 1) Coherencia multi-horizonte: anulada en horizontes inactivos (P0-4, buffer en GPU)
        if n_h >= 2:
            scales = self.horizon_scales[:n_h].to(device=device, dtype=dtype)
            norm_ret = ret / scales.unsqueeze(0)
            pair_mask = h_mask[:-1] * h_mask[1:]
            pair_sum = pair_mask.sum()
            has_pairs = (pair_sum > 0.5).to(dtype=dtype)
            diffs_sq = (norm_ret[:, 1:] - norm_ret[:, :-1]) ** 2
            e_hor_asset = (
                (diffs_sq * pair_mask.unsqueeze(0)).sum(dim=-1) / pair_sum.clamp_min(1.0)
            ) * has_pairs
        else:
            e_hor_asset = torch.zeros(n_assets, device=device, dtype=dtype)

        # 2) Consistencia CDF (no-cruce) + coherencia entre anchura intercuantílica y vol rugosa
        cross_h = F.relu(q[..., :-1] - q[..., 1:]).pow(2).mean(dim=-1)  # [N, H]
        cross_viol = (cross_h * h_mask.unsqueeze(0)).sum(dim=-1) / h_sum
        if q.shape[-1] >= 9:
            iqr_sigma = (q[..., 6] - q[..., 2]) / 2.5631  # q90 - q10 ~ 2.563 * sigma
        else:
            iqr_sigma = (q[..., -1] - q[..., 0]) / 2.5631
        vol_match = (((iqr_sigma - vol) ** 2) * h_mask.unsqueeze(0)).sum(dim=-1) / h_sum
        e_cdf_asset = cross_viol + vol_match

        # 3) Conservación transversal: alfa de media cero + ortogonalidad a factores conocidos B_t
        a0 = alpha[:, 0]
        mean_pen = a0.mean().pow(2)
        if exposures is not None and exposures.shape[-1] > 0:
            b = exposures.to(device=device, dtype=dtype)
            b_c = b - b.mean(dim=0, keepdim=True)
            b_n = b_c / b_c.std(dim=0, keepdim=True).clamp_min(1e-4)
            proj = (b_n.T @ a0) / max(float(n_assets), 1.0)
            fac_pen = (proj**2).mean()
        else:
            fac_pen = torch.zeros((), device=device, dtype=dtype)
        e_cs_asset = torch.zeros(n_assets, device=device, dtype=dtype) + (mean_pen + fac_pen)

        # 4) Anclaje epistémico Hopfield OOD: frena alfas extremos bajo energía OOD alta
        if ood_energy is not None:
            ood = ood_energy.to(device=device, dtype=dtype).clamp_min(0.0)
            e_ood_asset = ood * (a0**2)
        else:
            e_ood_asset = torch.zeros(n_assets, device=device, dtype=dtype)

        per_asset = (
            self.w_horizon * e_hor_asset
            + self.w_cdf_vol * e_cdf_asset
            + self.w_cross_section * e_cs_asset
            + self.w_ood * e_ood_asset
        )
        return dict(
            total=per_asset.mean(),
            per_asset=per_asset,
            e_horizon=e_hor_asset.mean(),
            e_cdf_vol=e_cdf_asset.mean(),
            e_cross_section=e_cs_asset.mean(),
            e_ood=e_ood_asset.mean(),
        )


class _ContractiveStepBlock(nn.Module):
    """Bloque contractivo de 1 iteración con constante de Lipschitz garantizada `< 1.0`."""

    def __init__(self, hidden_dim: int, lipschitz_bound: float = 0.90) -> None:
        super().__init__()
        self.lipschitz_bound = float(lipschitz_bound)
        self.lin1 = spectral_norm(nn.Linear(hidden_dim * 3, hidden_dim))
        self.lin2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.top_down = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.ema_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def max_operator_norm(self, precision_pi: torch.Tensor | None = None) -> torch.Tensor:
        """Cota superior tensorial (cero syncs) de la constante de Lipschitz efectiva (< 1.0)."""
        beta = torch.sigmoid(self.ema_logit.detach()) * 0.5
        pi_max = (
            precision_pi.detach().clamp(min=0.0, max=1.0).max().to(dtype=beta.dtype)
            if precision_pi is not None
            else torch.ones((), device=beta.device, dtype=beta.dtype)
        )
        # ||d(cat/denom)/dz|| <= sqrt(1 + pi_max^2) / sqrt(2 + pi_max^2) < 1
        cat_gain = torch.sqrt((1.0 + pi_max.pow(2)) / (2.0 + pi_max.pow(2)))
        delta_lip = (self.lipschitz_bound**2) * cat_gain
        eff = (1.0 - beta) * self.lipschitz_bound + beta * delta_lip
        return torch.clamp(eff, max=self.lipschitz_bound)

    def forward(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        y_emb: torch.Tensor,
        precision_pi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # P1-3: limitar `layer_precision <= 1.0` y escalar `cat` por `sqrt(2 + max(pi)^2)`
        pi_clamped = precision_pi.clamp(min=0.0, max=1.0)
        x_hat = self.top_down(z)
        pc_err = pi_clamped.unsqueeze(0) * (x0 - x_hat)
        cat = torch.cat([x0, z, y_emb + pc_err], dim=-1)
        denom = torch.sqrt(2.0 + pi_clamped.max().pow(2))
        h = torch.tanh((self.lin1(cat) / denom) * self.lipschitz_bound)
        delta = torch.tanh(self.lin2(h)) * self.lipschitz_bound
        beta = 0.5 * torch.sigmoid(self.ema_logit.to(dtype=z.dtype))
        # Contracción estricta de Banach: ||dz_next/dz|| <= (1 - beta)*L + beta*L^2 <= L < 1.0
        z_next = self.lipschitz_bound * (1.0 - beta) * z + beta * delta
        return z_next, pc_err


class PCTRMRecursiveLoop(nn.Module):
    """Doble Bucle `PC-TRM-Eq` + `MoR-LOOM-Fin` con routers por iteración y cero host-syncs."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        max_loops: int = 3,
        use_moe: bool = True,
        n_shared_experts: int = 2,
        n_routed_experts: int = 6,
        moe_mode: str = "soft",
        ffn_mult: float = 2.0,
        lipschitz_bound: float = 0.90,
        early_exit_tol: float = 1e-4,
        use_fused_blackwell_kernels: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.max_loops = int(max_loops)
        self.use_moe = use_moe
        self.lipschitz_bound = float(lipschitz_bound)
        self.early_exit_tol = float(early_exit_tol)
        self.use_fused_blackwell_kernels = bool(use_fused_blackwell_kernels)

        # Precisiones causales por dimensión 0 < Pi_l <= 1.0 (acotadas para Banach < 1.0)
        self.register_buffer("layer_precision", torch.ones(hidden_dim, dtype=torch.float32))
        self.y_feedback = nn.Linear(4, hidden_dim)

        # Routers y bloques independientes por iteración k = 0..max_loops-1 (LOOM anti-colapso)
        self.step_blocks = nn.ModuleList()
        self.contractive_steps = nn.ModuleList()
        d_ff_dense = max(8, int(round(hidden_dim * ffn_mult)))
        for _ in range(self.max_loops):
            if use_moe:
                self.step_blocks.append(
                    HighSNRMoELayer(
                        hidden_dim,
                        context_dim + 2,  # contexto alto SNR + [pc_err_norm, energy_k]
                        n_shared_experts=n_shared_experts,
                        n_routed_experts=n_routed_experts,
                        mode=moe_mode,
                        use_fused_blackwell_kernels=self.use_fused_blackwell_kernels,
                    )
                )
            else:
                self.step_blocks.append(
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, d_ff_dense),
                        nn.SiLU(),
                        nn.Linear(d_ff_dense, hidden_dim),
                    )
                )
            self.contractive_steps.append(
                _ContractiveStepBlock(hidden_dim, lipschitz_bound=lipschitz_bound)
            )

    def _summarize_pred(self, pred: dict[str, torch.Tensor]) -> torch.Tensor:
        """Resumen compacto 4D de la hipótesis `y^(k)` para el bucle de auto-crítica."""
        r0 = pred["pred_return"][:, 0:1]
        a0 = pred["alpha"][:, 0:1]
        v0 = pred["vol_pred"][:, 0:1]
        tp0 = pred["turning_prob"][:, 0:1]
        return self.y_feedback(torch.cat([r0, a0, v0, tp0], dim=-1))

    def forward(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        *,
        head: nn.Module,
        verifier: FinancialEnergyVerifier,
        exposures: torch.Tensor | None = None,
        ood_energy: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        placebo_context: torch.Tensor | None = None,
        enable_early_exit: bool | None = None,
    ) -> dict[str, object]:
        """Ejecuta `1..max_loops` pasos con Deep Supervision en train y selección monótona GPU."""
        early_exit = (not self.training) if enable_early_exit is None else bool(enable_early_exit)
        device, dtype = x0.device, x0.dtype
        z = x0
        step_outputs: list[dict[str, torch.Tensor]] = []
        energy_traj: list[torch.Tensor] = []
        router_diags: list[dict[str, object]] = []

        best_pred: dict[str, torch.Tensor] | None = None
        best_energy: dict[str, torch.Tensor] | None = None
        best_energy_val = torch.full((), float("inf"), device=device, dtype=dtype)
        best_z = z

        active_flag = torch.ones((), device=device, dtype=torch.bool)
        k_used_t = torch.zeros((), device=device, dtype=dtype)
        accepted_drops = torch.zeros((), device=device, dtype=dtype)
        attempted_refines = torch.zeros((), device=device, dtype=dtype)

        prev_energy_vec = torch.zeros(x0.shape[0], 1, device=device, dtype=dtype)
        prev_pc_norm = torch.zeros(x0.shape[0], 1, device=device, dtype=dtype)
        prev_y0: torch.Tensor | None = None
        prev_e_total: torch.Tensor | None = None

        pi_cur = self.layer_precision.to(device=device, dtype=dtype).clamp(min=0.0, max=1.0)

        for k in range(self.max_loops):
            blk = self.step_blocks[k]
            if self.use_moe:
                ctx_k = torch.cat([context, prev_pc_norm, prev_energy_vec], dim=-1)
                h_k, r_diag = blk(z, ctx_k, placebo_context=placebo_context)
                router_diags.append(r_diag)
            else:
                h_k = z + blk(z)

            pred_k = head(h_k, exposures=exposures)
            en_k = verifier(
                pred_k,
                exposures=exposures,
                ood_energy=ood_energy,
                horizon_mask=horizon_mask,
            )
            e_val = en_k["total"].detach()
            step_outputs.append(pred_k)

            if k == 0 or not early_exit:
                best_pred, best_energy, best_energy_val, best_z = pred_k, en_k, e_val, h_k
                energy_traj.append(e_val)
                k_used_t = k_used_t + 1.0
                if k > 0 and prev_e_total is not None:
                    attempted_refines = attempted_refines + 1.0
                    accepted_drops = accepted_drops + (e_val <= prev_e_total + 1e-6).to(dtype)
            else:
                # Aceptación monótona 100% en dispositivo (cero `.item()`, CUDA-Graph safe!)
                attempted_refines = attempted_refines + active_flag.to(dtype)
                improved = e_val <= best_energy_val + 1e-6
                accept_step = active_flag & improved
                accepted_drops = accepted_drops + accept_step.to(dtype)
                k_used_t = k_used_t + accept_step.to(dtype)

                assert best_pred is not None and best_energy is not None
                best_pred = {
                    key: torch.where(accept_step, pred_k[key], best_pred[key])
                    for key in pred_k
                }
                best_energy = {
                    key: torch.where(accept_step, en_k[key], best_energy[key])
                    for key in en_k
                }
                best_z = torch.where(accept_step, h_k, best_z)
                best_energy_val = torch.where(accept_step, e_val, best_energy_val)
                energy_traj.append(best_energy_val)

                if prev_y0 is not None:
                    diff_y = (pred_k["pred_return"].detach() - prev_y0).abs().mean()
                    converged = diff_y < self.early_exit_tol
                else:
                    converged = torch.zeros((), device=device, dtype=torch.bool)
                active_flag = accept_step & (~converged)

            prev_y0 = pred_k["pred_return"].detach()
            prev_e_total = e_val
            prev_energy_vec = en_k["per_asset"].detach().unsqueeze(-1)

            # Refinamiento contractivo de estado dual hacia el siguiente paso (re-inyectando x0)
            if k + 1 < self.max_loops:
                y_emb = self._summarize_pred(pred_k)
                z, pc_err = self.contractive_steps[k](x0, h_k, y_emb, pi_cur)
                prev_pc_norm = pc_err.detach().norm(dim=-1, keepdim=True)

        max_spec = torch.stack(
            [s.max_operator_norm(pi_cur) for s in self.contractive_steps]
        ).max()
        de_neg_rate = torch.where(
            attempted_refines > 0.5,
            accepted_drops / attempted_refines.clamp_min(1.0),
            torch.ones((), device=device, dtype=dtype),
        )
        last_router = router_diags[-1] if router_diags else None
        z0 = torch.zeros((), device=device, dtype=dtype)
        one0 = torch.ones((), device=device, dtype=dtype)
        diagnostics: dict[str, object] = dict(
            k_used=k_used_t,
            energy_trajectory=energy_traj,
            delta_energy_neg_rate=de_neg_rate,
            max_spectral_norm=max_spec,
            router_entropy=last_router["router_entropy"] if last_router else z0,
            load_max_min_ratio=last_router["load_max_min_ratio"] if last_router else one0,
            expert_mean_corr=last_router["expert_mean_corr"] if last_router else z0,
            router_entropy_per_asset=(
                last_router["router_entropy_per_asset"]
                if last_router
                else torch.zeros(x0.shape[0], device=device, dtype=dtype)
            ),
            expert_loads=[d["expert_loads"] for d in router_diags] if router_diags else [],
        )
        return dict(
            final_pred=best_pred,
            final_hidden=best_z,
            step_outputs=step_outputs,
            energy=best_energy,
            diagnostics=diagnostics,
        )


class _RMSNorm(nn.Module):
    """RMSNorm estable en FP32 interno para atención QK y estado latente del bucle."""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        calc_dtype = _stable_calc_dtype(x.dtype)
        x_c = x.to(dtype=calc_dtype)
        inv_rms = torch.rsqrt(x_c.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_c * inv_rms).to(dtype=x.dtype) * self.scale.to(device=x.device, dtype=x.dtype)


class _SpectralLinear(nn.Module):
    """Capa lineal espectral CUDA-Graph/`torch.compile`-safe (cero mutación en `forward`)."""

    def __init__(self, in_features: int, out_features: int, *, n_power_iters: int = 3) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_power_iters = int(n_power_iters)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(float(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

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
        """Hook post-step explícito: actualiza `u`/`v` fuera de `forward()`."""
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
        """Devuelve `W / max(sigma_max(W), 1)` sin mutar buffers durante `forward()`."""
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


class _StepLOOMRouter(nn.Module):
    """Enrutador ligero por paso `R^(k)` con sesgo Aux-Loss-Free congelado en `eval()`."""

    def __init__(self, context_dim: int, n_routed_experts: int) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.n_routed = int(n_routed_experts)
        self.ctx_norm = nn.LayerNorm(self.context_dim)
        self.router = nn.Linear(self.context_dim, self.n_routed)
        self.register_buffer("balance_bias", torch.zeros(self.n_routed, dtype=torch.float32))

    @torch.no_grad()
    def update_balance_bias(self, expert_loads: torch.Tensor, *, lr: float = 0.05) -> None:
        """Balanceo DeepSeek-V3 Aux-Loss-Free: estrictamente congelado en `eval()`."""
        if not self.training:
            return
        target = 1.0 / float(self.n_routed)
        loads = expert_loads.detach().to(device=self.balance_bias.device, dtype=torch.float32)
        diff = target - loads
        step = lr * torch.sign(diff) * torch.clamp(diff.abs() * self.n_routed, 0.2, 2.0)
        self.balance_bias.add_(step)
        self.balance_bias.sub_(self.balance_bias.mean())
        self.balance_bias.clamp_(-4.0, 4.0)


def _blend_head_predictions(
    pred_cand: dict[str, torch.Tensor],
    pred_prev: dict[str, torch.Tensor],
    accept_k: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combina hipótesis `pred_cand` y `pred_prev` por activo `accept_k [N]` (reuso `pred_eff`)."""
    a2 = accept_k.unsqueeze(-1)
    b2 = 1.0 - a2
    a3 = a2.unsqueeze(-1)
    b3 = 1.0 - a3

    out: dict[str, torch.Tensor] = {}
    for key, t_cand in pred_cand.items():
        if key in ("tb_probs", "turning_prob", "vol_pred"):
            continue
        t_prev = pred_prev.get(key)
        if t_prev is None or t_prev.shape != t_cand.shape:
            out[key] = t_cand
        elif t_cand.ndim == 3:
            out[key] = a3 * t_cand + b3 * t_prev
        elif t_cand.ndim == 2:
            out[key] = a2 * t_cand + b2 * t_prev
        else:
            out[key] = accept_k * t_cand + (1.0 - accept_k) * t_prev

    if "tb_logits" in out:
        out["tb_probs"] = torch.softmax(out["tb_logits"], dim=-1)
    elif "tb_probs" in pred_cand and "tb_probs" in pred_prev:
        out["tb_probs"] = a3 * pred_cand["tb_probs"] + b3 * pred_prev["tb_probs"]

    if "turn_logits" in out:
        out["turning_prob"] = torch.sigmoid(out["turn_logits"])
    elif "turning_prob" in pred_cand and "turning_prob" in pred_prev:
        out["turning_prob"] = a2 * pred_cand["turning_prob"] + b2 * pred_prev["turning_prob"]

    if "log_vol" in out:
        out["vol_pred"] = torch.exp(out["log_vol"])
    elif "vol_pred" in pred_cand and "vol_pred" in pred_prev:
        out["vol_pred"] = a2 * pred_cand["vol_pred"] + b2 * pred_prev["vol_pred"]

    return out


def _select_head_predictions(
    pred_cand: dict[str, torch.Tensor],
    pred_prev: dict[str, torch.Tensor],
    accept_bool: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Selección dura por activo (`torch.where`, 0 syncs) garantizando 0% regresiones en `eval`."""
    n_assets = accept_bool.shape[0]
    out: dict[str, torch.Tensor] = {}
    for key, t_cand in pred_cand.items():
        t_prev = pred_prev.get(key)
        if t_prev is None or t_prev.shape != t_cand.shape:
            out[key] = t_cand
        else:
            mask_view = accept_bool.view([n_assets] + [1] * (t_cand.ndim - 1))
            out[key] = torch.where(mask_view, t_cand, t_prev)
    return out


def _standardize_factor_concat(
    *,
    exposures: torch.Tensor | None,
    rmt_loadings: torch.Tensor | None,
    n_assets: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Estandariza y concatena `[B_n, R_n]` de forma invariante a `F` y sin `NaN`/syncs."""
    cols: list[torch.Tensor] = []
    calc_dtype = _stable_calc_dtype(dtype)
    for raw in (exposures, rmt_loadings):
        if raw is None or raw.ndim != 2 or raw.shape[0] != n_assets or raw.shape[-1] <= 0:
            continue
        m = torch.nan_to_num(
            raw.detach().to(device=device, dtype=calc_dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        m_c = m - m.mean(dim=0, keepdim=True)
        m_std = m_c.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        cols.append((m_c / m_std).to(dtype=dtype))
    if not cols:
        return None
    return torch.cat(cols, dim=-1)


class InducedFactorCrossAttention(nn.Module):
    """Atención transversal `O(N * R)` estrictamente invariante a `F` con QK-RMSNorm y gate 0.05."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_landmarks: int = 16,
        n_heads: int = 4,
        context_dim: int = 0,
        init_gate: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim debe ser >= 4")
        if n_landmarks < 2:
            raise ValueError("n_landmarks debe ser >= 2")
        if n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) debe ser divisible por n_heads ({n_heads})"
            )

        self.hidden_dim = int(hidden_dim)
        self.n_landmarks = int(n_landmarks)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.context_dim = int(context_dim)

        self.in_norm = _RMSNorm(self.hidden_dim)
        self.landmarks = nn.Parameter(
            torch.randn(self.n_landmarks, self.hidden_dim) / math.sqrt(float(self.hidden_dim))
        )
        self.regime_to_land = (
            nn.Linear(self.context_dim, self.hidden_dim) if self.context_dim > 0 else None
        )

        # Fase 0 (F-invariante): landmarks aprendidos atienden tokens factoriales `B_n^T @ h / N`
        self.f_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.f_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.qf_norm = _RMSNorm(self.head_dim)
        self.kf_norm = _RMSNorm(self.head_dim)

        # Fase 1: Asset -> Landmark Gather O(N * R * D)
        self.q1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.q1_norm = _RMSNorm(self.head_dim)
        self.k1_norm = _RMSNorm(self.head_dim)

        # Fase 2: Landmark <-> Landmark Reasoning O(R^2 * D)
        self.norm_m = _RMSNorm(self.hidden_dim)
        self.qm = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.km = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.vm = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.qm_norm = _RMSNorm(self.head_dim)
        self.km_norm = _RMSNorm(self.head_dim)

        # Fase 3: Landmark -> Asset Broadcast O(N * R * D)
        self.norm_b = _RMSNorm(self.hidden_dim)
        self.q2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.q2_norm = _RMSNorm(self.head_dim)
        self.k2_norm = _RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Compuerta `single_zero_v1` (`0.05`): casi identidad al nacer con gradiente paso-0 vivo
        self.out_gate = nn.Parameter(torch.full((), float(init_gate), dtype=torch.float32))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [L, D] -> [1, H, L, d_h]
        return x.view(1, x.shape[0], self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [1, H, L, d_h] -> [L, D]
        return x.transpose(1, 2).contiguous().view(x.shape[2], self.hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        *,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        factor_mat: torch.Tensor | None = None,
        step_emb: torch.Tensor | None = None,
        regime_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Ejecuta atención transversal `O(N * R)` invariante a `F` y equivariante en activos."""
        n_assets = h.shape[0]
        device, dtype = h.device, h.dtype
        h_n = self.in_norm(h)

        # Consultas landmark aprendidas moduladas por `(step_emb, regime_vec)`
        q_land = self.landmarks.to(device=device, dtype=dtype)
        if step_emb is not None:
            q_land = q_land + step_emb.to(device=device, dtype=dtype).unsqueeze(0)
        if self.regime_to_land is not None and regime_vec is not None:
            reg = regime_vec.to(device=device, dtype=dtype).reshape(-1)
            q_land = q_land + self.regime_to_land(reg).unsqueeze(0)

        # Fase 0 (F-invariante): tokens factoriales `B_n^T @ h / max(N, 1)` -> `[M, D]`
        f_mat = (
            factor_mat
            if factor_mat is not None
            else _standardize_factor_concat(
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                n_assets=n_assets,
                device=device,
                dtype=dtype,
            )
        )
        if f_mat is not None and f_mat.shape[-1] > 0:
            f_tok = (f_mat.to(device=device, dtype=dtype).T @ h_n) / max(n_assets, 1)  # [M, D]
            qf = self.qf_norm(self._split_heads(q_land))
            kf = self.kf_norm(self._split_heads(self.f_k(f_tok)))
            vf = self._split_heads(self.f_v(f_tok))
            land_seed = q_land + self._merge_heads(F.scaled_dot_product_attention(qf, kf, vf))
        else:
            land_seed = q_land

        # Fase 1: Asset -> Landmark Gather
        q1 = self.q1_norm(self._split_heads(self.q1(land_seed)))
        k1 = self.k1_norm(self._split_heads(self.k1(h_n)))
        v1 = self._split_heads(self.v1(h_n))
        land = land_seed + self._merge_heads(F.scaled_dot_product_attention(q1, k1, v1))

        # Fase 2: Landmark <-> Landmark Reasoning (O(R^2 * D))
        land_m = self.norm_m(land)
        qm = self.qm_norm(self._split_heads(self.qm(land_m)))
        km = self.km_norm(self._split_heads(self.km(land_m)))
        vm = self._split_heads(self.vm(land_m))
        land_tilde = land + self._merge_heads(F.scaled_dot_product_attention(qm, km, vm))

        # Fase 3: Landmark -> Asset Broadcast (O(N * R * D))
        land_b = self.norm_b(land_tilde)
        q2 = self.q2_norm(self._split_heads(self.q2(h_n)))
        k2 = self.k2_norm(self._split_heads(self.k2(land_b)))
        v2 = self._split_heads(self.v2(land_b))
        bcast = self._merge_heads(F.scaled_dot_product_attention(q2, k2, v2))
        delta = self.out_proj(bcast)

        # Entropía de atención landmark en dispositivo (O(N * R), 0 host-syncs)
        with torch.no_grad():
            inv_s = 1.0 / math.sqrt(float(self.head_dim))
            logits_b = torch.matmul(q2.float(), k2.float().transpose(-1, -2)) * inv_s
            p_b = torch.softmax(logits_b, dim=-1)
            ent = -(p_b * torch.log(p_b.clamp_min(1e-8))).sum(dim=-1).mean().to(dtype=dtype)

        out = h + self.out_gate.to(device=device, dtype=dtype) * delta
        return out, dict(landmark_attn_entropy=ent)


class _WeightTiedContractiveIntegrator(nn.Module):
    """Integrador contractivo weight-tied con `_SpectralLinear` puro (0 mutación en `forward`)."""

    def __init__(self, hidden_dim: int, *, lipschitz_bound: float = 0.90) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lipschitz_bound = float(lipschitz_bound)
        self.sn_x0 = _SpectralLinear(self.hidden_dim, self.hidden_dim)
        self.sn_h = _SpectralLinear(self.hidden_dim, self.hidden_dim)
        self.sn_fb = _SpectralLinear(self.hidden_dim, self.hidden_dim)
        self.sn_out = _SpectralLinear(self.hidden_dim, self.hidden_dim)
        self.beta_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def update_spectral_buffers(self, *, n_iters: int | None = None) -> None:
        """Actualiza buffers espectrales `u`/`v` tras `optimizer.step()`."""
        for layer in (self.sn_x0, self.sn_h, self.sn_fb, self.sn_out):
            layer.update_spectral_buffers(n_iters=n_iters)

    def compute_spectral_weights(self, *, update_buffers: bool = False) -> tuple[torch.Tensor, ...]:
        """Iza los 4 pesos espectralmente normalizados UNA vez por sesión."""
        return (
            self.sn_x0.normalized_weight(update_buffers=update_buffers),
            self.sn_h.normalized_weight(update_buffers=update_buffers),
            self.sn_fb.normalized_weight(update_buffers=update_buffers),
            self.sn_out.normalized_weight(update_buffers=update_buffers),
        )

    def lipschitz_upper_bound(self) -> torch.Tensor:
        """Cota superior certificada `||d z_cand / d z||_2 < lipschitz_bound < 1` (0-D)."""
        beta = 0.15 + 0.45 * torch.sigmoid(self.beta_logit)
        return self.lipschitz_bound * (1.0 - beta * (1.0 - 1.0 / math.sqrt(3.0)))

    def forward(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        h_moe: torch.Tensor,
        fb: torch.Tensor,
        *,
        cached_weights: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Paso contractivo de Banach con conservación exacta de escala convexa (`sum = 1.0`)."""
        w_x0, w_h, w_fb, w_out = (
            self.compute_spectral_weights(update_buffers=False)
            if cached_weights is None
            else cached_weights
        )
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        pre = (
            self.sn_x0(x0, cached_weight=w_x0)
            + self.sn_h(h_moe, cached_weight=w_h)
            + self.sn_fb(fb, cached_weight=w_fb)
        ) * inv_sqrt3
        delta = torch.tanh(self.sn_out(torch.tanh(pre), cached_weight=w_out)) * self.lipschitz_bound
        beta = (0.15 + 0.45 * torch.sigmoid(self.beta_logit)).to(device=z.device, dtype=z.dtype)
        l_b = self.lipschitz_bound
        return l_b * (1.0 - beta) * z + (1.0 - l_b) * (1.0 - beta) * x0 + beta * delta


class AdaptiveLoopHaltGate(nn.Module):
    """Compuerta humana de parada por activo guiada por `[E_struct, E_cf, E_conflict, E_epist]`."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        step_emb_dim: int = 16,
        gate_dim: int = 24,
        halt_threshold: float = 0.75,
        risk_slope: float = 2.0,
        monotonic_boost: float = 3.0,
    ) -> None:
        super().__init__()
        self.halt_threshold = float(halt_threshold)
        self.risk_slope = float(risk_slope)
        self.monotonic_boost = float(monotonic_boost)

        self.h_norm = _RMSNorm(hidden_dim)
        self.ctx_norm = nn.LayerNorm(context_dim)
        self.h_proj = nn.Linear(hidden_dim, gate_dim)
        self.meta_proj = nn.Linear(context_dim + step_emb_dim + 4, gate_dim)
        self.out_lin = nn.Linear(gate_dim, 1)
        # Inicialización pequeña: el sesgo físico (calma para / estrés re-itera) domina al inicio
        nn.init.normal_(self.out_lin.weight, std=0.02)
        nn.init.zeros_(self.out_lin.bias)

    def forward(
        self,
        z_k: torch.Tensor,
        ctx: torch.Tensor,
        step_emb: torch.Tensor,
        diag_4: torch.Tensor,
        risk_k: torch.Tensor,
        worsened: torch.Tensor,
    ) -> torch.Tensor:
        """Devuelve probabilidad condicional de parada `lambda_k [N]` en `(0, 1)`."""
        n_assets = z_k.shape[0]
        device, dtype = z_k.device, z_k.dtype
        step_exp = step_emb.to(device=device, dtype=dtype).unsqueeze(0).expand(n_assets, -1)
        meta = torch.cat(
            [
                self.ctx_norm(ctx.to(device=device, dtype=dtype)),
                step_exp,
                diag_4.detach().to(device=device, dtype=dtype),
            ],
            dim=-1,
        )
        feat = F.silu(self.h_proj(self.h_norm(z_k)) + self.meta_proj(meta))
        mlp_logit = self.out_lin(feat).squeeze(-1)
        risk_det = risk_k.detach().to(device=device, dtype=dtype)
        wors_det = worsened.detach().to(device=device, dtype=dtype).clamp_min(0.0)
        logit = (
            mlp_logit
            - self.risk_slope * (risk_det - self.halt_threshold)
            + self.monotonic_boost * wors_det
        )
        return torch.sigmoid(logit)


class LoopedCrossAssetTransformer(nn.Module):
    """Familia 3 v2: Looped Transformer weight-tied + ciclo Verify-Measure-Counterfactual-Retry."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        *,
        max_loops: int = 3,
        use_cross_asset_attn: bool = True,
        use_diff_xasset_attn: bool = False,
        diff_xasset_zero_lambda_control: bool = False,
        n_landmarks: int = 16,
        n_attn_heads: int | None = None,
        use_moe: bool = True,
        n_shared_experts: int = 2,
        n_routed_experts: int = 6,
        expert_mult: float = 1.5,
        moe_mode: str = "soft",
        ffn_mult: float = 2.0,
        lipschitz_bound: float = 0.90,
        use_counterfactual: bool = True,
        placebo_mode: bool = False,
        pos_cap: float = 0.05,
        halt_threshold: float = 0.75,
        monotonic_penalty: float = 5.0,
        use_fused_blackwell_kernels: bool = True,
        norm_kind: str = "layernorm",
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or context_dim < 1:
            raise ValueError("hidden_dim >= 8 y context_dim >= 1 son obligatorios")
        if not (1 <= int(max_loops) <= MAX_LOOP_STEPS):
            raise ValueError(
                f"max_loops ({max_loops}) debe cumplir 1 <= max_loops <= {MAX_LOOP_STEPS}"
            )
        if not (0.1 <= float(lipschitz_bound) < 1.0):
            raise ValueError("lipschitz_bound debe pertenecer a [0.1, 1.0)")

        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.max_loops = int(max_loops)
        self.use_cross_asset_attn = bool(use_cross_asset_attn)
        self.use_diff_xasset_attn = bool(use_diff_xasset_attn)
        self.diff_xasset_zero_lambda_control = bool(diff_xasset_zero_lambda_control)
        self.use_moe = bool(use_moe)
        self.moe_mode = str(moe_mode)
        self.lipschitz_bound = float(lipschitz_bound)
        self.use_counterfactual = bool(use_counterfactual)
        self.placebo_mode = bool(placebo_mode)
        self.halt_threshold = float(halt_threshold)
        self.monotonic_penalty = float(monotonic_penalty)
        self.use_fused_blackwell_kernels = bool(use_fused_blackwell_kernels)
        self.norm_kind = str(norm_kind)

        # Compatibilidad de estado con `model.py:step_matured` (`layer_precision <= 1.0`)
        self.register_buffer("layer_precision", torch.ones(self.hidden_dim, dtype=torch.float32))

        # Embeddings de paso ligeros + AdaRMSNorm FiLM (añade ~0% params en K=3 vs K=1)
        self.step_emb_dim = 16
        self.step_embed = nn.Embedding(self.max_loops, self.step_emb_dim)
        self.step_emb_proj = nn.Linear(self.step_emb_dim, self.hidden_dim)
        self.step_film = nn.Linear(self.step_emb_dim, 2 * self.hidden_dim)
        nn.init.zeros_(self.step_film.weight)
        nn.init.zeros_(self.step_film.bias)

        # Inyección de entrada `x_0` (Huginn) + normalización de estado
        self.x0_inj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.inj_gate = nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        self.u_norm: nn.Module = (
            _RMSNorm(self.hidden_dim)
            if self.norm_kind == "layernorm"
            else build_norm_layer(self.norm_kind, self.hidden_dim)  # type: ignore[arg-type]
        )

        # Proyección de realimentación `[cf_trace (10), err_feedback (8)] -> D`
        fb_in_dim = CF_TRACE_DIM + 8
        fb_mid = max(16, min(64, self.hidden_dim // 2))
        self.feedback_proj = nn.Sequential(
            nn.LayerNorm(fb_in_dim),
            nn.Linear(fb_in_dim, fb_mid),
            nn.SiLU(),
            nn.Linear(fb_mid, self.hidden_dim),
        )
        self.feedback_gate = nn.Parameter(torch.full((), 0.25, dtype=torch.float32))

        # 1 sola atención transversal weight-tied `O(N * R)` invariante a `F` (F3 opt-in DiffAttn!)
        self.cross_asset_attn: InducedFactorCrossAttention | DiffInducedFactorCrossAttention | None
        if self.use_cross_asset_attn:
            if self.use_diff_xasset_attn:
                resolved_heads = (
                    int(n_attn_heads)
                    if n_attn_heads is not None
                    else (
                        4
                        if self.hidden_dim % 8 == 0
                        else (2 if self.hidden_dim % 4 == 0 else 1)
                    )
                )
                self.cross_asset_attn = DiffInducedFactorCrossAttention(
                    self.hidden_dim,
                    n_landmarks=n_landmarks,
                    n_heads=resolved_heads,
                    context_dim=self.context_dim,
                    init_gate=0.05,
                    zero_lambda_control=self.diff_xasset_zero_lambda_control,
                )
            else:
                resolved_heads = (
                    int(n_attn_heads)
                    if n_attn_heads is not None
                    else (
                        4
                        if self.hidden_dim % 4 == 0
                        else (2 if self.hidden_dim % 2 == 0 else 1)
                    )
                )
                self.cross_asset_attn = InducedFactorCrossAttention(
                    self.hidden_dim,
                    n_landmarks=n_landmarks,
                    n_heads=resolved_heads,
                    context_dim=self.context_dim,
                    init_gate=0.05,
                )
        else:
            self.cross_asset_attn = None

        # 1 solo banco MoE weight-tied con routers LOOM ligeros por paso `R^(k)`
        route_ctx_dim = self.context_dim + 4
        if self.use_moe:
            self.shared_moe: HighSNRMoELayer | None = HighSNRMoELayer(
                self.hidden_dim,
                route_ctx_dim,
                n_shared_experts=n_shared_experts,
                n_routed_experts=n_routed_experts,
                expert_mult=expert_mult,
                mode=moe_mode,
                use_fused_blackwell_kernels=self.use_fused_blackwell_kernels,
                build_default_router=False,
            )
            self.step_routers = nn.ModuleList(
                [_StepLOOMRouter(route_ctx_dim, n_routed_experts) for _ in range(self.max_loops)]
            )
            self.shared_ffn: nn.Module | None = None
        else:
            d_ff_dense = max(8, int(round(self.hidden_dim * ffn_mult)))
            self.shared_moe = None
            self.step_routers = nn.ModuleList()
            ffn_norm: nn.Module = (
                nn.LayerNorm(self.hidden_dim)
                if self.norm_kind == "layernorm"
                else build_norm_layer(self.norm_kind, self.hidden_dim)  # type: ignore[arg-type]
            )
            self.shared_ffn = nn.Sequential(
                ffn_norm,
                nn.Linear(self.hidden_dim, d_ff_dense),
                nn.SiLU(),
                nn.Linear(d_ff_dense, self.hidden_dim),
            )

        # 1 solo integrador contractivo weight-tied con `_SpectralLinear` puro
        self.contractive_integrator = _WeightTiedContractiveIntegrator(
            self.hidden_dim, lipschitz_bound=self.lipschitz_bound
        )

        # Simulador contrafactual compartido (0 parámetros) + compuerta humana de parada por activo
        self.cf_sim = CounterfactualStressSimulator(
            pos_cap=pos_cap, placebo_mode=self.placebo_mode
        )
        self.halt_gate = AdaptiveLoopHaltGate(
            self.hidden_dim,
            self.context_dim,
            step_emb_dim=self.step_emb_dim,
            gate_dim=24,
            halt_threshold=self.halt_threshold,
        )

    @property
    def step_blocks(self) -> nn.ModuleList:
        """Alias compatible con `model.py:step_matured` para actualizar sesgos Aux-Loss-Free."""
        return self.step_routers

    @torch.no_grad()
    def update_spectral_buffers(self, *, n_iters: int | None = None) -> None:
        """Hook post-step explícito: actualiza buffers `_SpectralLinear` fuera de `forward()`."""
        self.contractive_integrator.update_spectral_buffers(n_iters=n_iters)

    def register_spectral_post_step_hook(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.utils.hooks.RemovableHandle:
        """Registra un hook post-step en `optimizer` para actualizar `_SpectralLinear`."""

        def _hook(_opt: torch.optim.Optimizer, _args: object, _kwargs: object) -> None:
            self.update_spectral_buffers()

        return optimizer.register_step_post_hook(_hook)

    @torch.no_grad()
    def update_balance_biases(
        self,
        expert_loads_list: list[torch.Tensor],
        *,
        lr: float = 0.05,
    ) -> None:
        """Actualiza sesgos Aux-Loss-Free de cada router `R^(k)` solo en `train()`."""
        if not self.training:
            return
        for router_k, loads_k in zip(self.step_routers, expert_loads_list, strict=False):
            router_k.update_balance_bias(loads_k, lr=lr)

    @staticmethod
    def _measure_step(
        pred: dict[str, torch.Tensor],
        *,
        verifier: FinancialEnergyVerifier,
        cf_sim: CounterfactualStressSimulator,
        use_counterfactual: bool,
        eff_placebo: bool,
        ctx: torch.Tensor,
        exposures: torch.Tensor | None,
        rmt_loadings: torch.Tensor | None,
        ood_cur: torch.Tensor,
        horizon_mask: torch.Tensor | None,
        conflict_cur: torch.Tensor,
        w_opt: torch.Tensor | None,
        cf_cache: dict[str, torch.Tensor | None] | None,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Verifica `E_struct`, `E_cf`/`cf_trace`, `E_conflict`, `E_epist` y riesgo compuesto."""
        n_assets = pred["alpha"].shape[0]
        device, dtype = pred["alpha"].device, pred["alpha"].dtype

        en_k = verifier(
            pred,
            exposures=exposures,
            ood_energy=ood_cur,
            horizon_mask=horizon_mask,
        )
        e_struct = en_k["per_asset"]

        if use_counterfactual:
            cf_out = cf_sim(
                pred["alpha"][:, 0],
                pred["vol_pred"][:, 0],
                high_snr_context=ctx,
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                w_opt=w_opt,
                quantiles=pred.get("quantiles"),
                conflict=conflict_cur,
                factor_cache=cf_cache,
                placebo_mode=eff_placebo,
            )
            e_cf = cf_out["e_cf"]
            cf_trace = cf_out["cf_trace"]
            p_flip = cf_out["p_flip"]
            scen_dd = cf_out["scenario_drawdowns"]
        else:
            e_cf = torch.zeros(n_assets, device=device, dtype=dtype)
            cf_trace = torch.zeros((n_assets, CF_TRACE_DIM), device=device, dtype=dtype)
            p_flip = torch.zeros(n_assets, device=device, dtype=dtype)
            scen_dd = torch.zeros((n_assets, 5), device=device, dtype=dtype)

        ret0 = pred["pred_return"][:, 0].detach().abs()
        vol0 = pred["vol_pred"][:, 0].detach().clamp_min(1e-4)
        epist0 = pred["epistemic_error"][:, 0].clamp_min(0.0)
        e_epist = torch.log1p(epist0 / (ret0 + 0.25 * vol0 + 1e-4)) + 0.25 * torch.log1p(
            ood_cur.clamp_min(0.0)
        )

        e_struct_log = torch.log1p(e_struct.clamp_min(0.0))
        risk = e_struct_log + e_cf + 0.5 * conflict_cur + 0.5 * e_epist
        diag_4 = torch.stack([e_struct_log, e_cf, conflict_cur, e_epist], dim=-1)
        return en_k, e_cf, cf_trace, p_flip, scen_dd, e_epist, risk, diag_4

    @staticmethod
    def _build_err_feedback(
        pred: dict[str, torch.Tensor],
        diag_4: torch.Tensor,
    ) -> torch.Tensor:
        """Resumen compacto 8D `[ret, alpha, vol, turn, log1p(E_struct), E_cf, E_conf, E_epist]`."""
        r0 = pred["pred_return"][:, 0:1]
        a0 = pred["alpha"][:, 0:1]
        v0 = pred["vol_pred"][:, 0:1].detach()
        tp0 = pred["turning_prob"][:, 0:1]
        return torch.cat([r0, a0, v0, tp0, diag_4], dim=-1)

    def forward(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        *,
        head: nn.Module,
        verifier: FinancialEnergyVerifier,
        exposures: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
        ood_energy: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        conflict: torch.Tensor | None = None,
        w_opt: torch.Tensor | None = None,
        regime_vec: torch.Tensor | None = None,
        placebo_context: torch.Tensor | None = None,
        placebo_mode: bool | None = None,
        enable_early_exit: bool | None = None,
    ) -> dict[str, object]:
        """Ejecuta `1..max_loops` pasos weight-tied con Verify-Measure-CF-Retry y cero syncs."""
        early_exit = (not self.training) if enable_early_exit is None else bool(enable_early_exit)
        eff_placebo = self.placebo_mode if placebo_mode is None else bool(placebo_mode)
        n_assets = x0.shape[0]
        device, dtype = x0.device, x0.dtype

        ctx = torch.nan_to_num(
            context.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        reg_vec = (
            regime_vec.to(device=device, dtype=dtype).reshape(-1)
            if regime_vec is not None
            else ctx.mean(dim=0)
        )
        ood_cur = (
            torch.nan_to_num(
                ood_energy.detach().to(device=device, dtype=dtype).reshape(n_assets),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            if ood_energy is not None
            else torch.zeros(n_assets, device=device, dtype=dtype)
        )
        conflict_cur = (
            torch.nan_to_num(
                conflict.detach().to(device=device, dtype=dtype).reshape(n_assets),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            if conflict is not None
            else torch.zeros(n_assets, device=device, dtype=dtype)
        )

        # Izado UNA vez por sesión: pesos espectrales, matriz factorial `F_mat` y cache CF FP32
        cached_spec_w = self.contractive_integrator.compute_spectral_weights(update_buffers=False)
        hoisted_f_mat = (
            _standardize_factor_concat(
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                n_assets=n_assets,
                device=device,
                dtype=dtype,
            )
            if self.use_cross_asset_attn
            else None
        )
        cf_cache = (
            prepare_cf_factor_cache(
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                device=device,
                dtype=dtype,
            )
            if self.use_counterfactual
            else None
        )
        x0_inj_vec = torch.sigmoid(self.inj_gate.to(device=device, dtype=dtype)) * self.x0_inj(x0)

        z = x0
        h_acc = torch.zeros_like(x0)
        remaining = torch.ones(n_assets, device=device, dtype=dtype)
        expected_steps = torch.zeros(n_assets, device=device, dtype=dtype)
        active_mask = torch.ones(n_assets, device=device, dtype=torch.bool)
        cum_halt = torch.zeros(n_assets, device=device, dtype=dtype)

        prev_diag4 = torch.zeros((n_assets, 4), device=device, dtype=dtype)
        prev_cf_trace = torch.zeros((n_assets, CF_TRACE_DIM), device=device, dtype=dtype)
        prev_err_fb = torch.zeros((n_assets, 8), device=device, dtype=dtype)

        step_outputs: list[dict[str, torch.Tensor]] = []
        energy_traj: list[torch.Tensor] = []
        router_diags: list[dict[str, object]] = []
        router_probs_list: list[torch.Tensor] = []
        landmark_ents: list[torch.Tensor] = []
        diff_lambdas: list[torch.Tensor] = []
        halt_probs_list: list[torch.Tensor] = []
        accept_list: list[torch.Tensor] = []

        best_risk: torch.Tensor | None = None
        best_pred: dict[str, torch.Tensor] | None = None
        best_energy: dict[str, torch.Tensor] | None = None
        best_z = z
        best_e_cf = torch.zeros(n_assets, device=device, dtype=dtype)
        best_p_flip = torch.zeros(n_assets, device=device, dtype=dtype)
        best_scen_dd = torch.zeros((n_assets, 5), device=device, dtype=dtype)
        e_cf_k0 = torch.zeros(n_assets, device=device, dtype=dtype)
        risk_k0_det = torch.zeros(n_assets, device=device, dtype=dtype)

        for k in range(self.max_loops):
            step_e_raw = self.step_embed.weight[k].to(device=device, dtype=dtype)
            step_e = self.step_emb_proj(step_e_raw)
            film_g, film_b = self.step_film(step_e_raw).chunk(2, dim=-1)

            if k == 0:
                fb_vec = torch.zeros_like(z)
            else:
                fb_in = torch.cat([prev_cf_trace, prev_err_fb], dim=-1)
                fb_vec = self.feedback_gate.to(device=device, dtype=dtype) * self.feedback_proj(
                    fb_in
                )

            # Inyección Huginn + AdaRMSNorm FiLM por paso
            u_k = self.u_norm(z + x0_inj_vec + fb_vec)
            u_k = u_k * (1.0 + film_g.unsqueeze(0)) + film_b.unsqueeze(0) + step_e.unsqueeze(0)

            # Atención transversal O(N * R) weight-tied e invariante a `F`
            if self.cross_asset_attn is not None:
                u_attn, attn_diag = self.cross_asset_attn(
                    u_k,
                    factor_mat=hoisted_f_mat,
                    step_emb=step_e,
                    regime_vec=reg_vec,
                )
                landmark_ents.append(attn_diag["landmark_attn_entropy"])
                if "diff_lambda_mean" in attn_diag:
                    diff_lambdas.append(attn_diag["diff_lambda_mean"])
            else:
                u_attn = u_k

            # MoE weight-tied con router LOOM por paso `R^(k)` (o FFN denso weight-tied)
            if self.use_moe and self.shared_moe is not None:
                ctx_route = torch.cat([ctx, prev_diag4.detach()], dim=-1)
                step_r = self.step_routers[k]
                h_moe, r_diag = self.shared_moe(
                    u_attn,
                    ctx_route,
                    placebo_context=placebo_context,
                    router_override=step_r.router,
                    ctx_norm_override=step_r.ctx_norm,
                    balance_bias_override=step_r.balance_bias,
                )
                router_diags.append(r_diag)
                router_probs_list.append(r_diag["router_probs"].detach())  # type: ignore[union-attr]
            else:
                assert self.shared_ffn is not None
                h_moe = u_attn + self.shared_ffn(u_attn)

            # Integración contractiva de Banach (cero mutación de buffers en `forward`)
            z_cand = self.contractive_integrator(
                x0, z, h_moe, fb_vec, cached_weights=cached_spec_w
            )

            # Ciclo Verify -> Measure -> Counterfactual Stress-Test
            pred_cand = head(z_cand, exposures=exposures)
            (
                en_cand,
                e_cf_cand,
                cf_trace_cand,
                p_flip_cand,
                scen_dd_cand,
                _,
                risk_cand,
                diag4_cand,
            ) = self._measure_step(
                pred_cand,
                verifier=verifier,
                cf_sim=self.cf_sim,
                use_counterfactual=self.use_counterfactual,
                eff_placebo=eff_placebo,
                ctx=ctx,
                exposures=exposures,
                rmt_loadings=rmt_loadings,
                ood_cur=ood_cur,
                horizon_mask=horizon_mask,
                conflict_cur=conflict_cur,
                w_opt=w_opt,
                cf_cache=cf_cache,
            )

            if k == 0:
                e_cf_k0 = e_cf_cand
                risk_k0_det = risk_cand.detach()
                best_risk = risk_k0_det
                worsened = torch.zeros_like(risk_cand)
                accept_k = torch.ones(n_assets, device=device, dtype=dtype)
                z_eff = z_cand
                pred_eff = pred_cand
                best_z = z_cand
                best_pred = pred_cand
                best_energy = en_cand
                best_e_cf = e_cf_cand
                best_p_flip = p_flip_cand
                best_scen_dd = scen_dd_cand
                eff_cf_trace = cf_trace_cand
                eff_diag4 = diag4_cand
                energy_traj.append(best_risk.mean())
            elif early_exit:
                assert best_risk is not None and best_pred is not None and best_energy is not None
                worsened = F.relu(risk_cand - best_risk)
                improved = risk_cand.detach() <= best_risk + 1e-6
                accept_bool = active_mask & improved
                accept_k = accept_bool.to(dtype=dtype)
                accept_list.append(accept_k)

                best_z = torch.where(accept_bool.unsqueeze(-1), z_cand, best_z)
                best_pred = _select_head_predictions(pred_cand, best_pred, accept_bool)
                per_asset_e = torch.where(
                    accept_bool, en_cand["per_asset"], best_energy["per_asset"]
                )
                any_acc = accept_bool.any()
                best_energy = dict(
                    total=per_asset_e.mean(),
                    per_asset=per_asset_e,
                    e_horizon=torch.where(any_acc, en_cand["e_horizon"], best_energy["e_horizon"]),
                    e_cdf_vol=torch.where(any_acc, en_cand["e_cdf_vol"], best_energy["e_cdf_vol"]),
                    e_cross_section=torch.where(
                        any_acc,
                        en_cand["e_cross_section"],
                        best_energy["e_cross_section"],
                    ),
                    e_ood=torch.where(any_acc, en_cand["e_ood"], best_energy["e_ood"]),
                )
                best_e_cf = torch.where(accept_bool, e_cf_cand, best_e_cf)
                best_p_flip = torch.where(accept_bool, p_flip_cand, best_p_flip)
                best_scen_dd = torch.where(accept_bool.unsqueeze(-1), scen_dd_cand, best_scen_dd)
                best_risk = torch.where(accept_bool, risk_cand.detach(), best_risk)

                z_eff = best_z
                pred_eff = best_pred
                eff_cf_trace = torch.where(accept_bool.unsqueeze(-1), cf_trace_cand, prev_cf_trace)
                eff_diag4 = torch.where(accept_bool.unsqueeze(-1), diag4_cand, prev_diag4)
                energy_traj.append(best_risk.mean())
            else:
                assert best_risk is not None and best_pred is not None
                worsened = F.relu(risk_cand - best_risk)
                accept_k = torch.exp(-self.monotonic_penalty * worsened)
                accept_list.append(accept_k)
                a1 = accept_k.unsqueeze(-1)
                z_eff = a1 * z_cand + (1.0 - a1) * z
                pred_eff = _blend_head_predictions(pred_cand, best_pred, accept_k)
                best_pred = pred_eff
                best_z = z_eff
                best_energy = en_cand
                best_e_cf = accept_k * e_cf_cand + (1.0 - accept_k) * best_e_cf
                best_p_flip = accept_k * p_flip_cand + (1.0 - accept_k) * best_p_flip
                best_scen_dd = a1 * scen_dd_cand + (1.0 - a1) * best_scen_dd
                eff_cf_trace = a1 * cf_trace_cand + (1.0 - a1) * prev_cf_trace
                eff_diag4 = a1 * diag4_cand + (1.0 - a1) * prev_diag4
                best_risk = torch.minimum(best_risk, risk_cand.detach())
                energy_traj.append(risk_cand.detach().mean())

            # Compuerta adaptativa de parada por activo (`AdaptiveLoopHaltGate`)
            if k + 1 < self.max_loops:
                lambda_k = self.halt_gate(
                    z_eff, ctx, step_e_raw, eff_diag4, risk_cand, worsened
                )
                p_k = remaining * lambda_k
                remaining = remaining * (1.0 - lambda_k)
            else:
                p_k = remaining

            h_acc = h_acc + p_k.unsqueeze(-1) * z_eff
            expected_steps = expected_steps + float(k + 1) * p_k
            halt_probs_list.append(p_k)
            if early_exit:
                cum_halt = cum_halt + p_k.detach()
                active_mask = active_mask & (cum_halt < 0.95)

            step_outputs.append(pred_eff)
            prev_diag4 = eff_diag4
            prev_cf_trace = eff_cf_trace
            prev_err_fb = self._build_err_feedback(pred_eff, eff_diag4)
            z = z_eff

        # Estado y predicción final (reutilizando `pred_eff`: `<= K + 1` llamadas totales a `head`)
        assert best_pred is not None and best_energy is not None
        if early_exit or self.max_loops == 1:
            final_hidden = best_z
            final_pred = best_pred
            final_energy = best_energy
        else:
            final_hidden = h_acc
            final_pred = head(h_acc, exposures=exposures)
            step_outputs[-1] = final_pred
            final_energy = verifier(
                final_pred,
                exposures=exposures,
                ood_energy=ood_cur,
                horizon_mask=horizon_mask,
            )

        # PonderNet KL con prior geométrico guiado por riesgo (calma -> k=1, estrés -> k=2..3)
        halting_dist = torch.stack(halt_probs_list, dim=-1)  # [N, K]
        calc_dtype = _stable_calc_dtype(dtype)
        hd_c = halting_dist.to(dtype=calc_dtype)
        p_prior = torch.sigmoid(
            -1.5 * (risk_k0_det.to(dtype=calc_dtype) - self.halt_threshold)
        ).clamp(0.15, 0.80)
        prior_cols: list[torch.Tensor] = []
        rem_p = torch.ones(n_assets, device=device, dtype=calc_dtype)
        for k_idx in range(self.max_loops):
            if k_idx + 1 < self.max_loops:
                pk_p = rem_p * p_prior
                rem_p = rem_p * (1.0 - p_prior)
            else:
                pk_p = rem_p
            prior_cols.append(pk_p.clamp_min(1e-6))
        prior_dist = torch.stack(prior_cols, dim=-1)
        prior_dist = prior_dist / prior_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ponder_kl_loss = (
            (hd_c * (torch.log(hd_c.clamp_min(1e-8)) - torch.log(prior_dist.clamp_min(1e-8))))
            .sum(dim=-1)
            .mean()
            .clamp_min(0.0)
            .to(dtype=dtype)
        )

        # Regularizador acotado anti-empeoramiento CF (jamás premia encoger |alpha| -> 0)
        cf_fragility_loss = F.relu(best_e_cf - e_cf_k0.detach()).mean().to(dtype=dtype)

        # Divergencia Jensen-Shannon entre routers de paso `R^(k)` vs `R^(0)` (0 syncs)
        z0 = torch.zeros((), device=device, dtype=dtype)
        one0 = torch.ones((), device=device, dtype=dtype)
        if len(router_probs_list) >= 2:
            p0_c = router_probs_list[0].to(dtype=calc_dtype)
            js_terms: list[torch.Tensor] = []
            for pk_r in router_probs_list[1:]:
                pk_c = pk_r.to(dtype=calc_dtype)
                m_c = (0.5 * (p0_c + pk_c)).clamp_min(1e-8)
                kl0 = (p0_c * (torch.log(p0_c.clamp_min(1e-8)) - torch.log(m_c))).sum(dim=-1)
                kl1 = (pk_c * (torch.log(pk_c.clamp_min(1e-8)) - torch.log(m_c))).sum(dim=-1)
                js_terms.append((0.5 * (kl0 + kl1)).mean())
            loop_step_router_js_div = torch.stack(js_terms).mean().to(dtype=dtype)
        else:
            loop_step_router_js_div = z0

        loop_k_used_mean = expected_steps.detach().mean().to(dtype=dtype)
        loop_halt_at_k1_pct = halting_dist[:, 0].detach().mean().to(dtype=dtype)
        cf_fragility_k0 = e_cf_k0.detach().mean().to(dtype=dtype)
        cf_fragility_final = best_e_cf.detach().mean().to(dtype=dtype)
        cf_sign_flip_rate_final = best_p_flip.detach().mean().to(dtype=dtype)
        cf_worst_dd_final = best_scen_dd.detach().max(dim=-1).values.mean().to(dtype=dtype)
        loop_monotonic_accept_rate = (
            torch.stack(accept_list, dim=0).detach().mean().to(dtype=dtype)
            if accept_list
            else one0
        )
        landmark_attn_entropy = (
            torch.stack(landmark_ents, dim=0).detach().mean().to(dtype=dtype)
            if landmark_ents
            else z0
        )

        last_router = router_diags[-1] if router_diags else None
        max_spec = self.contractive_integrator.lipschitz_upper_bound().to(
            device=device, dtype=dtype
        )

        diagnostics: dict[str, object] = dict(
            # Diagnósticos 0-D v2 sin host-syncs
            loop_k_used_mean=loop_k_used_mean,
            loop_halt_at_k1_pct=loop_halt_at_k1_pct,
            cf_fragility_k0=cf_fragility_k0,
            cf_fragility_final=cf_fragility_final,
            cf_sign_flip_rate_final=cf_sign_flip_rate_final,
            cf_worst_dd_final=cf_worst_dd_final,
            loop_monotonic_accept_rate=loop_monotonic_accept_rate,
            loop_step_router_js_div=loop_step_router_js_div,
            landmark_attn_entropy=landmark_attn_entropy,
            ponder_kl_loss=ponder_kl_loss,
            cf_fragility_loss=cf_fragility_loss,
            halting_distribution=halting_dist,
            expected_steps_per_asset=expected_steps,
            # Claves retrocompatibles con `PCTRMRecursiveLoop` y `model.py`
            k_used=loop_k_used_mean,
            energy_trajectory=energy_traj,
            delta_energy_neg_rate=loop_monotonic_accept_rate,
            max_spectral_norm=max_spec,
            router_entropy=last_router["router_entropy"] if last_router else z0,
            load_max_min_ratio=last_router["load_max_min_ratio"] if last_router else one0,
            expert_mean_corr=last_router["expert_mean_corr"] if last_router else z0,
            router_entropy_per_asset=(
                last_router["router_entropy_per_asset"]
                if last_router
                else torch.zeros(n_assets, device=device, dtype=dtype)
            ),
            expert_loads=[d["expert_loads"] for d in router_diags] if router_diags else [],
        )
        if diff_lambdas:
            diagnostics["diff_lambda_mean"] = (
                torch.stack(diff_lambdas, dim=0).detach().mean().to(dtype=dtype)
            )
        return dict(
            final_pred=final_pred,
            final_hidden=final_hidden,
            step_outputs=step_outputs,
            energy=final_energy,
            diagnostics=diagnostics,
            ponder_kl_loss=ponder_kl_loss,
            cf_fragility_loss=cf_fragility_loss,
        )


__all__ = [
    "MAX_LOOP_STEPS",
    "AdaptiveLoopHaltGate",
    "FinancialEnergyVerifier",
    "HighSNRMoELayer",
    "InducedFactorCrossAttention",
    "LoopedCrossAssetTransformer",
    "PCTRMRecursiveLoop",
    "_SpectralLinear",
    "_sparsemax",
]
