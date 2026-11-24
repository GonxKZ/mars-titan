"""Memoria Titans Neuromodulada (ACh/NE + Gated DeltaNet-2 + Caja PT1 + Huber + Cascada 3x).

Innovación propia #1 de `MMARRS-TITAN` (cero host-syncs GPU, BPTT truncado y alineación canónica):
  1. Compuertas vectoriales desacopladas de borrado (`alpha_t`) y escritura (`theta_t`, `eta_t`)
     acotadas estrictamente dentro de la caja contractiva de Lyapunov `PT1`
     (`alpha >= alpha_lo`, `eta <= eta_hi`, `theta <= theta_max`).
  2. Doble Neuromodulación Causal (Yu & Dayan 2005; Piray & Daw 2024):
     - **ACh (ruido esperado local):** amortigua `theta` para evitar escribir ruido de corto plazo.
     - **NE (Locus Coeruleus, shock inesperado sistémico):** acelera el olvido `alpha` del banco
       rápido y transfiere lectura hacia el banco lento de régimen.
  3. Sorpresa asociativa recortada con pérdida **Huber (MIRAS-Yaad)** vía `fused_neuromod_update`
     (autograd analítico Triton `sm_120` para entrenamiento BPTT).
  4. **Cascada Benna-Fusi de 3 escalas** (`fast` ~5d, `medium` ~21d, `slow` ~126d) indexada por
     `asset_ids: int64[N]` canónicos (`gather`/`scatter` diferenciable; IPOs en 0; jamás borra
     bancos cuando `N_t != N_{t-1}`).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blackwell_kernels import fused_neuromod_update
from .hardware_blackwell import fused_pt1_huber_gate_update

_BANK_KEYS: tuple[str, ...] = ("fast", "fast_mom", "medium", "slow")
_ASSET_ID_SENTINEL: int = (1 << 62) - 1
_DEFAULT_MAX_ASSET_TABLE: int = 16384


class NeuromodulatedTitansMemory(nn.Module):
    """Memoria neuronal asociativa neuromodulada ACh/NE con persistencia canónica por `asset_id`."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        pt1_alpha_lo: float = 0.05,
        pt1_eta_hi: float = 0.95,
        pt1_theta_max: float = 0.25,
        huber_delta: float = 1.0,
        use_triton: bool = True,
        use_fused_blackwell_kernels: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pt1_alpha_lo = float(pt1_alpha_lo)
        self.pt1_eta_hi = float(pt1_eta_hi)
        self.pt1_theta_max = float(pt1_theta_max)
        self.huber_delta = float(huber_delta)
        self.use_triton = bool(use_triton)
        self.use_fused_blackwell_kernels = bool(use_fused_blackwell_kernels)

        # Proyección fusionada (1 solo GEMM para key, value, alpha, eta, theta -> mínimo HBM I/O)
        self.fused_proj = nn.Linear(hidden_dim, 5 * hidden_dim)
        self.ach_gain = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.ne_gain = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.scale_logits = nn.Parameter(torch.tensor([0.5, 0.2, -0.1], dtype=torch.float32))
        self.read_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Escala residual pequeña: identidad casi exacta con estado vacío y gradiente vivo paso 1
        nn.init.normal_(self.out_proj.weight, std=0.05)
        nn.init.zeros_(self.out_proj.bias)

    def init_state(
        self,
        n_assets: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        asset_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inicializa los 3 bancos Benna-Fusi (`fast`, `medium`, `slow`) y su momento."""
        z = torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype)
        st: dict[str, torch.Tensor] = dict(
            fast=z.clone(),
            fast_mom=z.clone(),
            medium=z.clone(),
            slow=z.clone(),
        )
        if asset_ids is not None:
            cur_ids = asset_ids.to(device=device, dtype=torch.long).reshape(-1)
            sort_idx = torch.argsort(cur_ids)
            st["asset_ids"] = cur_ids[sort_idx]
        return st

    @staticmethod
    def detach_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        """Desconecta del grafo autograd todos los bancos en los límites de BPTT truncado."""
        if state is None:
            return None
        return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}

    @staticmethod
    def _gather_by_asset_ids(
        state: dict[str, torch.Tensor],
        cur_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
    ) -> dict[str, torch.Tensor]:
        """Recoge bancos persistentes por `asset_id` vía `searchsorted` (cero syncs, IPOs -> 0)."""
        n_assets = cur_ids.shape[0]
        table_ids = state["asset_ids"].to(device=device, dtype=torch.long).reshape(-1)
        zeros = torch.zeros((n_assets, hidden_dim), device=device, dtype=dtype)
        if table_ids.shape[0] == 0:
            return {k: zeros.clone() for k in _BANK_KEYS}

        idx = torch.searchsorted(table_ids, cur_ids)
        idx_safe = idx.clamp(max=table_ids.shape[0] - 1)
        matched = (idx < table_ids.shape[0]) & (table_ids[idx_safe] == cur_ids)
        m_col = matched.unsqueeze(-1)
        gathered: dict[str, torch.Tensor] = {}
        for k in _BANK_KEYS:
            bank = state[k].to(device=device, dtype=dtype)
            gathered[k] = torch.where(m_col, bank[idx_safe], zeros)
        return gathered

    def read(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
        *,
        ne_signal: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Lee la cascada multiescala ponderada dinámicamente por el pulso noradrenérgico NE."""
        zero_s = torch.zeros((), device=x.device, dtype=x.dtype)
        if state is None:
            return x, dict(ne_slow_shift=zero_s, mem_norm=zero_s)

        if asset_ids is not None and "asset_ids" in state:
            cur_ids = asset_ids.to(device=x.device, dtype=torch.long).reshape(-1)
            banks = self._gather_by_asset_ids(
                state, cur_ids, device=x.device, dtype=x.dtype, hidden_dim=self.hidden_dim
            )
            fast_b = banks["fast"]
            med_b = banks["medium"]
            slow_b = banks["slow"]
        else:
            if state["fast"].shape[0] != x.shape[0]:
                return x, dict(ne_slow_shift=zero_s, mem_norm=zero_s)
            fast_b = state["fast"].to(device=x.device, dtype=x.dtype)
            med_b = state["medium"].to(device=x.device, dtype=x.dtype)
            slow_b = state["slow"].to(device=x.device, dtype=x.dtype)

        ne_scalar = (
            torch.zeros((), device=x.device, dtype=x.dtype)
            if ne_signal is None
            else ne_signal.to(device=x.device, dtype=x.dtype).mean()
        )
        # Cuando NE sube (shock de régimen), desplaza peso hacia el banco lento de régimen
        ne_gate = torch.sigmoid(ne_scalar)
        shift = torch.stack(
            [-0.8 * ne_gate, torch.zeros_like(ne_gate), 0.8 * ne_gate]
        ).to(dtype=x.dtype)
        w = torch.softmax(self.scale_logits.to(device=x.device, dtype=x.dtype) + shift, dim=0)
        mem = w[0] * fast_b + w[1] * med_b + w[2] * slow_b
        fused = x + self.out_proj(self.read_norm(mem))
        return fused, dict(
            ne_slow_shift=ne_gate.detach(),
            mem_norm=mem.detach().float().norm(dim=-1).mean().to(dtype=x.dtype),
        )

    def update(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
        *,
        ach_signal: torch.Tensor | None = None,
        ne_signal: torch.Tensor | None = None,
        target_hint: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Actualiza la cascada 3x de forma causal dentro de la caja PT1 (soporta BPTT y IDs)."""
        n_assets = x.shape[0]
        device, dtype = x.device, x.dtype

        prev_ids: torch.Tensor | None = None
        cur_ids: torch.Tensor | None = None
        if asset_ids is not None:
            cur_ids = asset_ids.to(device=device, dtype=torch.long).reshape(-1)
            if state is not None and "asset_ids" in state:
                prev_ids = state["asset_ids"].to(device=device, dtype=torch.long).reshape(-1)
                cur_banks = self._gather_by_asset_ids(
                    state, cur_ids, device=device, dtype=dtype, hidden_dim=self.hidden_dim
                )
            elif state is not None and state["fast"].shape[0] == n_assets:
                cur_banks = {k: state[k].to(device=device, dtype=dtype) for k in _BANK_KEYS}
            else:
                z = torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype)
                cur_banks = {k: z.clone() for k in _BANK_KEYS}
        else:
            if state is None or "asset_ids" in state or state["fast"].shape[0] != n_assets:
                cur_banks = self.init_state(n_assets, device=device, dtype=dtype)
            else:
                cur_banks = {k: state[k].to(device=device, dtype=dtype) for k in _BANK_KEYS}

        proj = self.fused_proj(x)
        key, val, a_raw, e_raw, t_raw = proj.chunk(5, dim=-1)

        # 1) Compuertas vectoriales base estrictamente dentro de la caja PT1
        alpha_base = self.pt1_alpha_lo + (1.0 - self.pt1_alpha_lo) * torch.sigmoid(a_raw) * 0.5
        eta = self.pt1_eta_hi * torch.sigmoid(e_raw)
        theta_base = self.pt1_theta_max * torch.sigmoid(t_raw)

        # 2) Modulación ACh (ruido esperado local -> amortigua tasa de escritura theta)
        if ach_signal is not None:
            ach = ach_signal.to(device=device, dtype=dtype)
            if ach.ndim == 1:
                ach = ach.unsqueeze(-1)
            ach_damp = torch.exp(
                -F.softplus(self.ach_gain.to(dtype=dtype)) * F.softplus(ach)
            )
            theta = torch.clamp(theta_base * ach_damp, min=1e-5, max=self.pt1_theta_max)
        else:
            theta = theta_base

        # 3) Modulación NE (shock inesperado sistémico -> acelera olvido alpha en banco rápido)
        if ne_signal is not None:
            ne = ne_signal.to(device=device, dtype=dtype).mean()
            ne_boost = torch.sigmoid(ne) * torch.sigmoid(self.ne_gain.to(dtype=dtype))
            alpha = torch.clamp(
                alpha_base + ne_boost * (0.95 - alpha_base),
                min=self.pt1_alpha_lo,
                max=0.98,
            )
        else:
            alpha = alpha_base

        # 4) Sorpresa asociativa (val - proyección de memoria actual) con recorte Huber (Yaad)
        key_unit = F.normalize(key, dim=-1, eps=1e-6)
        read_pred = (cur_banks["fast"] * key_unit).sum(dim=-1, keepdim=True) * key_unit
        target_vec = (
            val if target_hint is None else (val + target_hint.to(device=device, dtype=dtype))
        )
        surprise = target_vec - read_pred

        # 5) Actualización fusionada del banco rápido (autograd Triton sm_120 para BPTT)
        if self.use_fused_blackwell_kernels:
            fast_new, mom_new = fused_neuromod_update(
                cur_banks["fast"],
                cur_banks["fast_mom"],
                surprise,
                alpha,
                eta,
                theta,
                huber_delta=self.huber_delta,
                use_triton=self.use_triton,
            )
        else:
            fast_new, mom_new = fused_pt1_huber_gate_update(
                cur_banks["fast"],
                cur_banks["fast_mom"],
                surprise,
                alpha,
                eta,
                theta,
                huber_delta=self.huber_delta,
                use_triton=self.use_triton,
            )
        # Contracción suave para blindar contra explosión numérica prolongada
        fast_new = 10.0 * torch.tanh(fast_new / 10.0)
        mom_new = 5.0 * torch.tanh(mom_new / 5.0)

        # 6) Cascada difusiva Benna-Fusi hacia bancos medio (21d) y lento (126d)
        med_new = 0.90 * cur_banks["medium"] + 0.10 * fast_new
        slow_new = 0.98 * cur_banks["slow"] + 0.02 * med_new
        updated_active = dict(fast=fast_new, fast_mom=mom_new, medium=med_new, slow=slow_new)

        # 7) Empaquetado en tabla canónica por `asset_id` (0 host-syncs: sin `torch.unique`!)
        if cur_ids is not None:
            if (
                prev_ids is None
                or prev_ids.shape[0] == 0
                or state is None
                or "asset_ids" not in state
            ):
                sort_idx = torch.argsort(cur_ids, stable=True)
                c_sorted = cur_ids[sort_idx]
                c_is_dup = torch.zeros_like(c_sorted, dtype=torch.bool)
                if c_sorted.shape[0] > 1:
                    c_is_dup[:-1] = c_sorted[:-1] == c_sorted[1:]
                sentinel_t = torch.full_like(c_sorted, _ASSET_ID_SENTINEL)
                dedup_ids = torch.where(c_is_dup, sentinel_t, c_sorted)
                d_order = torch.argsort(dedup_ids, stable=True)
                sorted_ids = dedup_ids[d_order]
                is_real = (sorted_ids < _ASSET_ID_SENTINEL).unsqueeze(-1)
                new_state: dict[str, torch.Tensor] = {}
                for k in _BANK_KEYS:
                    b_sorted = updated_active[k][sort_idx][d_order]
                    new_state[k] = torch.where(is_real, b_sorted, torch.zeros_like(b_sorted))
                new_state["asset_ids"] = sorted_ids
            else:
                p_len = prev_ids.shape[0]
                c_order = torch.argsort(cur_ids, stable=True)
                c_sorted = cur_ids[c_order]
                c_is_dup = torch.zeros_like(c_sorted, dtype=torch.bool)
                if c_sorted.shape[0] > 1:
                    c_is_dup[:-1] = c_sorted[:-1] == c_sorted[1:]
                pos = torch.searchsorted(prev_ids, c_sorted)
                safe_pos = pos.clamp(max=p_len - 1)
                in_prev = (
                    (pos < p_len)
                    & (prev_ids[safe_pos] == c_sorted)
                    & (c_sorted < _ASSET_ID_SENTINEL)
                )
                upd_mask = in_prev & (~c_is_dup)
                dummy_pos = torch.full_like(safe_pos, p_len)
                scatter_idx = (
                    torch.where(upd_mask, safe_pos, dummy_pos)
                    .unsqueeze(-1)
                    .expand(-1, self.hidden_dim)
                )
                sentinel_t = torch.full_like(c_sorted, _ASSET_ID_SENTINEL)
                add_ids = torch.where(in_prev | c_is_dup, sentinel_t, c_sorted)
                merged_ids = torch.cat([prev_ids, add_ids], dim=0)
                m_order = torch.argsort(merged_ids, stable=True)
                max_cap = max(p_len, _DEFAULT_MAX_ASSET_TABLE, c_sorted.shape[0] * 4)
                if m_order.shape[0] > max_cap:
                    m_order = m_order[:max_cap]
                sorted_ids = merged_ids[m_order]
                is_real = (sorted_ids < _ASSET_ID_SENTINEL).unsqueeze(-1)
                pad_row = torch.zeros((1, self.hidden_dim), device=device, dtype=dtype)
                add_keep = (~(in_prev | c_is_dup)).unsqueeze(-1)
                new_state = {}
                for k in _BANK_KEYS:
                    prev_bank = state[k].to(device=device, dtype=dtype)
                    cur_sorted_bank = updated_active[k][c_order]
                    prev_ext = torch.cat([prev_bank, pad_row], dim=0)
                    prev_updated = prev_ext.scatter(0, scatter_idx, cur_sorted_bank)[:p_len]
                    add_bank = torch.where(
                        add_keep, cur_sorted_bank, torch.zeros_like(cur_sorted_bank)
                    )
                    merged_bank = torch.cat([prev_updated, add_bank], dim=0)
                    gathered_m = merged_bank[m_order]
                    new_state[k] = torch.where(is_real, gathered_m, torch.zeros_like(gathered_m))
                new_state["asset_ids"] = sorted_ids
        else:
            new_state = updated_active

        diag = dict(
            alpha_min=alpha.detach().min(),
            alpha_mean=alpha.detach().mean(),
            eta_max=eta.detach().max(),
            theta_max=theta.detach().max(),
            theta_mean=theta.detach().mean(),
        )
        return new_state, diag


def compact_long_running_state(
    state: dict[str, torch.Tensor] | dict[str, object] | None,
    *,
    fast_decay: float = 0.70,
    momentum_decay: float = 0.50,
    mkt_fast_decay: float = 1.0,
) -> dict[str, object]:
    """Compacta trazas rápidas (`fast`, `fast_mom`, `M_mkt_mom`) preservando bancos lentos.

    Acepta directamente un diccionario `titans_state` o un diccionario de estado causal externo que
    contenga `"titans_state"`. Preserva bit-a-bit `medium`, `slow`, `M_mkt_slow`, `asset_ids` y
    `last_ne_signal` con cero host-syncs.
    """
    if not (0.0 < fast_decay <= 1.0):
        raise ValueError("fast_decay debe pertenecer a (0, 1]")
    if not (0.0 <= momentum_decay < 1.0):
        raise ValueError("momentum_decay debe pertenecer a [0, 1)")
    if not (0.0 < mkt_fast_decay <= 1.0):
        raise ValueError("mkt_fast_decay debe pertenecer a (0, 1]")
    if state is None:
        return {}

    if "titans_state" in state and isinstance(state["titans_state"], dict):
        outer: dict[str, object] = {}
        for k, v in state.items():
            if k == "titans_state":
                outer[k] = compact_long_running_state(
                    v,  # type: ignore[arg-type]
                    fast_decay=fast_decay,
                    momentum_decay=momentum_decay,
                    mkt_fast_decay=mkt_fast_decay,
                )
            elif isinstance(v, torch.Tensor):
                outer[k] = v.detach().clone()
            else:
                outer[k] = v
        return outer

    compacted: dict[str, object] = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            compacted[k] = v
            continue
        t = v.detach()
        if k == "fast":
            compacted[k] = t * float(fast_decay)
        elif k in ("fast_mom", "M_mkt_mom"):
            compacted[k] = t * float(momentum_decay)
        elif k == "M_mkt" and mkt_fast_decay != 1.0:
            compacted[k] = t * float(mkt_fast_decay)
        else:
            compacted[k] = t.clone()
    return compacted


class DualAxisTitansMemory(nn.Module):
    """Memoria Asociativa Dual-Axis v2 (Matriz Cohorte Proximal FP32 + Banco Benna-Fusi por activo).

    Implementa el contrato exacto de `#128` §1 y `#133` P0-3 con 0 host-syncs y seguridad BF16:
      - **Axis 1 (Matriz compartida de régimen de mercado `M_mkt [H, d_h, d_h]` + cascada lenta
        `M_mkt_slow`):**
        * Claves normalizadas `||k_{i,h}||_2 = 1`; valores objetivo `v_i = val_i.detach() + target`
          (cero autotrampa en Axis 1, solo targets madurados `y_1d`).
        * Residuo recortado Row-Huber `E_{i,h} = psi_delta(v_{i,h} - k_{i,h} M_{t-1,h}^T)` +
          rechazo redescendiente **Tukey biweight** (`c = 4.685` sobre MAD robusto centrado) +
          ortogonalización transversal **CSOS** `E_tilde_{i,h} = E_{i,h} - sum_j c_j E_{j,h}`.
        * Ecuación normal proximal FP32 (`autocast(enabled=False)` + `torch.linalg.solve_ex`):
          `(I + theta_h K_h^T C K_h) M_{t,h}^T = (1 - alpha_h) M_{t-1,h}^T
           + theta_h K_h^T C (K_h M_{t-1,h}^T + E_tilde_h)`,
          transpuesta de vuelta `M_t = M_t^T.transpose(-1, -2)` y proyección Frobenius `<= M_max`.
        * Compuertas escalares por cabeza `alpha_h in [alpha_lo, 0.98]`, `theta_h in [0, theta_max]`
          garantizando contracción de Lyapunov `||D^+||_F <= (1 - alpha_h) ||D||_F` y
          `kappa_2 <= 1 + theta_max`.
      - **Axis 2 (Cascada Benna-Fusi por activo `fast/medium/slow`):**
        * Preserva la tabla canónica `asset_ids` (0 host-syncs, deduplicación sesión 0 sin
          `torch.unique`) y claves retrocompatibles (`fast`, `fast_mom`, `medium`, `slow`,
          `asset_ids`) junto a `M_mkt`, `M_mkt_slow`, `M_mkt_mom` y `last_ne_signal`.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        pt1_alpha_lo: float = 0.05,
        pt1_eta_hi: float = 0.95,
        pt1_theta_max: float = 0.25,
        huber_delta: float = 1.0,
        tukey_c: float = 4.685,
        m_max: float = 10.0,
        mkt_slow_decay: float = 0.98,
        mkt_mom_decay: float = 0.50,
        use_triton: bool = True,
        use_fused_blackwell_kernels: bool = True,
    ) -> None:
        super().__init__()
        if n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} debe ser divisible por n_heads={n_heads} (>= 1)"
            )
        if tukey_c <= 0.0:
            raise ValueError("tukey_c debe ser > 0")
        if m_max <= 0.0:
            raise ValueError("m_max debe ser > 0")

        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.pt1_alpha_lo = float(pt1_alpha_lo)
        self.pt1_eta_hi = float(pt1_eta_hi)
        self.pt1_theta_max = float(pt1_theta_max)
        self.huber_delta = float(huber_delta)
        self.tukey_c = float(tukey_c)
        self.m_max = float(m_max)
        self.mkt_slow_decay = float(mkt_slow_decay)
        self.mkt_mom_decay = float(mkt_mom_decay)
        self.use_triton = bool(use_triton)
        self.use_fused_blackwell_kernels = bool(use_fused_blackwell_kernels)

        # Proyecciones asociativas explícitas (BPTT entrenables)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Compuertas PT1: escalar-por-cabeza en Axis 1 y vectoriales por activo en Axis 2
        self.mkt_gate_proj = nn.Linear(self.hidden_dim, 2 * self.n_heads)
        nn.init.normal_(self.mkt_gate_proj.weight, std=0.05)
        with torch.no_grad():
            self.mkt_gate_proj.bias[: self.n_heads].fill_(-1.5)
            self.mkt_gate_proj.bias[self.n_heads :].fill_(1.0)
        self.asset_gate_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)

        # Neuromodulación ACh / NE y mezcla multiescala
        self.ach_gain = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.ne_gain = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.mkt_scale_logits = nn.Parameter(torch.tensor([0.6, -0.2], dtype=torch.float32))
        self.scale_logits = nn.Parameter(torch.tensor([0.5, 0.2, -0.1], dtype=torch.float32))

        # Lectura dual con compuerta multiplicativa para Axis 2 y proyección común `out_proj`
        self.mkt_read_norm = nn.LayerNorm(self.hidden_dim)
        self.read_norm = nn.LayerNorm(self.hidden_dim)
        self.read_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.normal_(self.read_gate.weight, std=0.05)
        nn.init.constant_(self.read_gate.bias, 1.0)

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.normal_(self.out_proj.weight, std=0.05)
        nn.init.zeros_(self.out_proj.bias)

    def init_state(
        self,
        n_assets: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        asset_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inicializa bancos compatibles Benna-Fusi + matrices asociativas Axis 1 + EMA NE."""
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        z_asset = torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype)
        z_mkt = torch.zeros(
            (self.n_heads, self.head_dim, self.head_dim), device=device, dtype=calc_dtype
        )
        st: dict[str, torch.Tensor] = dict(
            fast=z_asset.clone(),
            fast_mom=z_asset.clone(),
            medium=z_asset.clone(),
            slow=z_asset.clone(),
            M_mkt=z_mkt.clone(),
            M_mkt_slow=z_mkt.clone(),
            M_mkt_mom=z_mkt.clone(),
            last_ne_signal=torch.zeros((), device=device, dtype=calc_dtype),
            ne_ema_mean=torch.zeros((), device=device, dtype=calc_dtype),
            ne_ema_var=torch.full((), 1e-2, device=device, dtype=calc_dtype),
            ne_ema_steps=torch.zeros((), device=device, dtype=calc_dtype),
        )
        if asset_ids is not None:
            cur_ids = asset_ids.to(device=device, dtype=torch.long).reshape(-1)
            sort_idx = torch.argsort(cur_ids, stable=True)
            st["asset_ids"] = cur_ids[sort_idx]
        return st

    @staticmethod
    def detach_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        """Desconecta todos los tensores persistentes del grafo autograd en bordes BPTT."""
        return NeuromodulatedTitansMemory.detach_state(state)

    @staticmethod
    def compact_long_running_state(
        state: dict[str, torch.Tensor] | dict[str, object] | None,
        *,
        fast_decay: float = 0.70,
        momentum_decay: float = 0.50,
        mkt_fast_decay: float = 1.0,
    ) -> dict[str, object]:
        """Compactación periódica Zero-Drift: atenúa `M_mkt_mom`, `fast_mom` y `fast`."""
        return compact_long_running_state(
            state,
            fast_decay=fast_decay,
            momentum_decay=momentum_decay,
            mkt_fast_decay=mkt_fast_decay,
        )

    def _compute_tukey_weights(
        self,
        raw_res_h: torch.Tensor,
        *,
        write_mask: torch.Tensor | None = None,
        quality_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcula pesos redescendientes Tukey biweight (`c=4.685`) sobre sorpresa MAD robusta.

        Centra el residuo por la mediana transversal de coordenadas antes de medir su norma L2,
        de modo que un sesgo común de mercado no altera los pesos Tukey mientras los outliers
        idiosincráticos `20*sigma` (splits/halts/limit-locks) redescinden a peso exacto `0.0`.
        """
        with torch.amp.autocast(device_type=raw_res_h.device.type, enabled=False):
            calc_dtype = (
                torch.float64 if raw_res_h.dtype == torch.float64 else torch.float32
            )
            r_det = raw_res_h.detach().to(dtype=calc_dtype)  # [H, N, d_h]
            med_coord = r_det.median(dim=1, keepdim=True).values  # [H, 1, d_h]
            r_cent = r_det - med_coord
            err_i = torch.sqrt((r_cent * r_cent).sum(dim=-1) + 1e-12).mean(dim=0)  # [N]

            err_med = err_i.median()
            mad = (err_i - err_med).abs().median()
            sigma_mad = (1.4826 * mad).clamp_min(1e-6)
            u = F.relu(err_i - err_med) / (self.tukey_c * sigma_mad)
            u_clamped = u.clamp(0.0, 1.0)
            w_tukey = (1.0 - u_clamped * u_clamped).pow(2)  # exacto 0.0 si u >= 1

            w_raw = w_tukey
            if write_mask is not None:
                wm = write_mask.detach().to(device=r_det.device, dtype=calc_dtype).reshape(-1)
                wm = torch.where(torch.isfinite(wm), wm, torch.zeros_like(wm)).clamp(0.0, 1.0)
                w_raw = w_raw * wm
            if quality_weights is not None:
                qw = quality_weights.detach().to(device=r_det.device, dtype=calc_dtype).reshape(-1)
                qw = torch.where(torch.isfinite(qw), qw, torch.zeros_like(qw)).clamp_min(0.0)
                w_raw = w_raw * qw

            w_sum = w_raw.sum()
            c_weights = torch.where(
                w_sum > 1e-8,
                w_raw / w_sum.clamp_min(1e-8),
                torch.zeros_like(w_raw),
            )
            # Magnitud de sorpresa de sesión libre de outliers para el pulso causal NE
            err_raw_i = torch.sqrt((r_det * r_det).sum(dim=-1) + 1e-12).mean(dim=0)
            session_err = torch.where(
                w_sum > 1e-8,
                (err_raw_i * c_weights).sum(),
                err_raw_i.median(),
            )
            return w_tukey, c_weights, session_err

    def _proximal_cohort_step(
        self,
        m_prev: torch.Tensor,
        k_h: torch.Tensor,
        v_h: torch.Tensor,
        c_weights: torch.Tensor,
        alpha_h: torch.Tensor,
        theta_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paso proximal exacto FP32 `#133` P0-3 con Row-Huber + CSOS + proyección `<= M_max`.

        Resuelve por cabeza `h`:
          `(I + theta_h K_h^T C K_h) M_{t,h}^T = (1 - alpha_h) M_{t-1,h}^T
           + theta_h K_h^T C (K_h M_{t-1,h}^T + E_tilde_h)`
        y transpone de vuelta `M_t = M_t^T.transpose(-1, -2)`.
        """
        with torch.amp.autocast(device_type=k_h.device.type, enabled=False):
            calc_dtype = torch.float64 if k_h.dtype == torch.float64 else torch.float32
            m_p = m_prev.to(device=k_h.device, dtype=calc_dtype)
            k_c = k_h.to(dtype=calc_dtype)
            v_c = v_h.to(device=k_h.device, dtype=calc_dtype)
            c_c = c_weights.to(device=k_h.device, dtype=calc_dtype).reshape(-1)
            a_c = alpha_h.to(device=k_h.device, dtype=calc_dtype).view(self.n_heads, 1, 1)
            t_c = theta_h.to(device=k_h.device, dtype=calc_dtype).view(self.n_heads, 1, 1)

            m_p_t = m_p.transpose(-1, -2)  # [H, d_h, d_h]
            pred_v = torch.bmm(k_c, m_p_t)  # [H, N, d_h]
            raw_res = v_c - pred_v

            # Recorte Row-Huber por fila `(i, h)`
            r_norm = torch.sqrt((raw_res * raw_res).sum(dim=-1, keepdim=True) + 1e-12)
            e_h = raw_res * torch.clamp(self.huber_delta / r_norm, max=1.0)

            # Ortogonalización transversal CSOS: elimina deriva media del mercado
            c_col = c_c.view(1, -1, 1)
            e_mean = (c_col * e_h).sum(dim=1, keepdim=True)
            e_tilde = e_h - e_mean

            # Ecuación normal ridge proximal FP32 (0 host-syncs vía solve_ex)
            c_k = k_c * c_col  # [H, N, d_h]
            gram = torch.bmm(k_c.transpose(-1, -2), c_k)  # [H, d_h, d_h]
            eye = torch.eye(self.head_dim, device=k_c.device, dtype=calc_dtype).unsqueeze(0)
            a_mat = eye + t_c * gram
            v_eff = pred_v + e_tilde
            rhs = (1.0 - a_c) * m_p_t + t_c * torch.bmm(c_k.transpose(-1, -2), v_eff)
            m_next_t, _ = torch.linalg.solve_ex(a_mat, rhs, check_errors=False)
            m_next = m_next_t.transpose(-1, -2)

            # Proyección contractiva en bola de Frobenius `||M_{t,h}||_F <= M_max`
            frob = torch.sqrt((m_next * m_next).sum(dim=(-2, -1), keepdim=True) + 1e-12)
            m_next = m_next * torch.clamp(self.m_max / frob, max=1.0)

            cond_ub = 1.0 + t_c.detach().max()
            return m_next, cond_ub

    def read(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
        *,
        ne_signal: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Lee simultáneamente Axis 1 (`q_{i,h} M^{mkt,T}`) y Axis 2 (Benna-Fusi por activo)."""
        zero_s = torch.zeros((), device=x.device, dtype=x.dtype)
        if state is None or x.shape[0] == 0:
            return x, dict(
                ne_slow_shift=zero_s,
                mem_norm=zero_s,
                mkt_mem_norm=zero_s,
                asset_mem_norm=zero_s,
            )

        n_assets = x.shape[0]
        device, dtype = x.device, x.dtype

        # Pulso NE causal: prioriza argumento explícito y si es None usa `last_ne_signal` persistido
        if ne_signal is not None:
            ne_scalar = ne_signal.to(device=device, dtype=dtype).mean()
        elif isinstance(state.get("last_ne_signal"), torch.Tensor):
            ne_scalar = state["last_ne_signal"].to(device=device, dtype=dtype).mean()
        else:
            ne_scalar = zero_s
        ne_gate = torch.sigmoid(ne_scalar)

        # 1) Axis 1: lectura asociativa `q_{i,h} M^{mkt,T}_{t-1,h}` (2 escalas fast + slow)
        q_raw = self.q_proj(x)
        q_h = F.normalize(
            q_raw.view(n_assets, self.n_heads, self.head_dim).transpose(0, 1),
            dim=-1,
            eps=1e-6,
        )
        m_fast_raw = state.get("M_mkt")
        m_slow_raw = state.get("M_mkt_slow")
        if isinstance(m_fast_raw, torch.Tensor):
            m_fast = m_fast_raw.to(device=device, dtype=dtype)
        else:
            m_fast = torch.zeros(
                (self.n_heads, self.head_dim, self.head_dim), device=device, dtype=dtype
            )
        if isinstance(m_slow_raw, torch.Tensor):
            m_slow = m_slow_raw.to(device=device, dtype=dtype)
        else:
            m_slow = torch.zeros_like(m_fast)

        mkt_shift = torch.stack([-0.8 * ne_gate, 0.8 * ne_gate]).to(dtype=dtype)
        g_mkt = torch.softmax(
            self.mkt_scale_logits.to(device=device, dtype=dtype) + mkt_shift, dim=0
        )
        m_blend = g_mkt[0] * m_fast + g_mkt[1] * m_slow
        mkt_read_h = torch.bmm(q_h, m_blend.transpose(-1, -2))  # [H, N, d_h]
        mkt_read = mkt_read_h.transpose(0, 1).contiguous().view(n_assets, self.hidden_dim)

        # 2) Axis 2: lectura Benna-Fusi por activo (`asset_ids`) + compuerta multiplicativa
        if asset_ids is not None and "asset_ids" in state:
            cur_ids = asset_ids.to(device=device, dtype=torch.long).reshape(-1)
            banks = NeuromodulatedTitansMemory._gather_by_asset_ids(
                state, cur_ids, device=device, dtype=dtype, hidden_dim=self.hidden_dim
            )
            fast_b, med_b, slow_b = banks["fast"], banks["medium"], banks["slow"]
        elif "fast" in state and state["fast"].shape[0] == n_assets:
            fast_b = state["fast"].to(device=device, dtype=dtype)
            med_b = state["medium"].to(device=device, dtype=dtype)
            slow_b = state["slow"].to(device=device, dtype=dtype)
        else:
            zeros_a = torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype)
            fast_b = med_b = slow_b = zeros_a

        shift = torch.stack(
            [-0.8 * ne_gate, torch.zeros_like(ne_gate), 0.8 * ne_gate]
        ).to(dtype=dtype)
        w = torch.softmax(self.scale_logits.to(device=device, dtype=dtype) + shift, dim=0)
        asset_mem = w[0] * fast_b + w[1] * med_b + w[2] * slow_b

        mkt_term = self.mkt_read_norm(mkt_read)
        asset_term = F.silu(self.read_gate(x)) * self.read_norm(asset_mem)
        fused = x + self.out_proj(mkt_term + asset_term)

        mkt_norm = mkt_read.detach().float().norm(dim=-1).mean().to(dtype=dtype)
        asset_norm = asset_mem.detach().float().norm(dim=-1).mean().to(dtype=dtype)
        return fused, dict(
            ne_slow_shift=ne_gate.detach(),
            mem_norm=(mkt_norm + asset_norm),
            mkt_mem_norm=mkt_norm,
            asset_mem_norm=asset_norm,
        )

    def update(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
        *,
        ach_signal: torch.Tensor | None = None,
        ne_signal: torch.Tensor | None = None,
        target_hint: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
        write_mask: torch.Tensor | None = None,
        quality_weights: torch.Tensor | None = None,
        matured_error: torch.Tensor | None = None,
        detach_state: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Actualiza Axis 1 (matriz proximal FP32 + Tukey + CSOS) y Axis 2 (Benna-Fusi ragged)."""
        n_assets = x.shape[0]
        device, dtype = x.device, x.dtype
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        if n_assets == 0:
            base = (
                self.init_state(0, device=device, dtype=dtype, asset_ids=asset_ids)
                if state is None
                else {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
            )
            zero_c = torch.zeros((), device=device, dtype=calc_dtype)
            return (
                self.detach_state(base) if detach_state else base,  # type: ignore[return-value]
                dict(
                    alpha_min=torch.full((), self.pt1_alpha_lo, device=device, dtype=calc_dtype),
                    alpha_mean=torch.full((), self.pt1_alpha_lo, device=device, dtype=calc_dtype),
                    eta_max=zero_c,
                    theta_max=zero_c,
                    theta_mean=zero_c,
                    cond_upper_bound=torch.ones((), device=device, dtype=calc_dtype),
                    tukey_min_weight=torch.ones((), device=device, dtype=calc_dtype),
                    tukey_zero_frac=zero_c,
                    mkt_norm=zero_c,
                    last_ne_signal=zero_c,
                ),
            )

        # 0) Recoger estado previo de Axis 1 y Axis 2
        prev_ids: torch.Tensor | None = None
        cur_ids: torch.Tensor | None = None
        if asset_ids is not None:
            cur_ids = asset_ids.to(device=device, dtype=torch.long).reshape(-1)
            if state is not None and "asset_ids" in state:
                prev_ids = state["asset_ids"].to(device=device, dtype=torch.long).reshape(-1)
                cur_banks = NeuromodulatedTitansMemory._gather_by_asset_ids(
                    state, cur_ids, device=device, dtype=dtype, hidden_dim=self.hidden_dim
                )
            elif state is not None and "fast" in state and state["fast"].shape[0] == n_assets:
                cur_banks = {k: state[k].to(device=device, dtype=dtype) for k in _BANK_KEYS}
            else:
                z = torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype)
                cur_banks = {k: z.clone() for k in _BANK_KEYS}
        else:
            if (
                state is None
                or "asset_ids" in state
                or "fast" not in state
                or state["fast"].shape[0] != n_assets
            ):
                z = torch.zeros((n_assets, self.hidden_dim), device=device, dtype=dtype)
                cur_banks = {k: z.clone() for k in _BANK_KEYS}
            else:
                cur_banks = {k: state[k].to(device=device, dtype=dtype) for k in _BANK_KEYS}

        z_mkt = torch.zeros(
            (self.n_heads, self.head_dim, self.head_dim), device=device, dtype=calc_dtype
        )
        m_prev = (
            state["M_mkt"].to(device=device, dtype=calc_dtype)
            if (state is not None and isinstance(state.get("M_mkt"), torch.Tensor))
            else z_mkt
        )
        m_slow_prev = (
            state["M_mkt_slow"].to(device=device, dtype=calc_dtype)
            if (state is not None and isinstance(state.get("M_mkt_slow"), torch.Tensor))
            else z_mkt
        )
        m_mom_prev = (
            state["M_mkt_mom"].to(device=device, dtype=calc_dtype)
            if (state is not None and isinstance(state.get("M_mkt_mom"), torch.Tensor))
            else z_mkt
        )

        # 1) Claves unitarias y valores objetivo (P0-3: val_i.detach() en Axis 1!)
        k_raw = self.k_proj(x)
        val_raw = self.v_proj(x)
        k_h = F.normalize(
            k_raw.view(n_assets, self.n_heads, self.head_dim).transpose(0, 1),
            dim=-1,
            eps=1e-6,
        )
        hint_t = target_hint.to(device=device, dtype=dtype) if target_hint is not None else None
        v_mkt_flat = val_raw.detach() if hint_t is None else (val_raw.detach() + hint_t)
        v_h = v_mkt_flat.view(n_assets, self.n_heads, self.head_dim).transpose(0, 1)

        # 2) Residuo previo para Tukey biweight redescendiente (`c=4.685`) y pulso NE causal
        with torch.amp.autocast(device_type=device.type, enabled=False):
            k_h_f = k_h.to(dtype=calc_dtype)
            v_h_f = v_h.to(dtype=calc_dtype)
            raw_res_h = v_h_f - torch.bmm(k_h_f, m_prev.transpose(-1, -2))
            w_tukey, c_weights, session_err = self._compute_tukey_weights(
                raw_res_h, write_mask=write_mask, quality_weights=quality_weights
            )

            # EMA causal de sorpresa de sesión -> NE libre de escala (sin constante 0.015)
            e_cur = (
                matured_error.detach().to(device=device, dtype=calc_dtype).abs().mean()
                if matured_error is not None
                else session_err.detach()
            )
            prev_ne_mean = (
                state["ne_ema_mean"].detach().to(device=device, dtype=calc_dtype)
                if (state is not None and isinstance(state.get("ne_ema_mean"), torch.Tensor))
                else e_cur
            )
            prev_ne_var = (
                state["ne_ema_var"].detach().to(device=device, dtype=calc_dtype)
                if (state is not None and isinstance(state.get("ne_ema_var"), torch.Tensor))
                else torch.full((), 1e-2, device=device, dtype=calc_dtype)
            )
            prev_ne_steps = (
                state["ne_ema_steps"].detach().to(device=device, dtype=calc_dtype)
                if (state is not None and isinstance(state.get("ne_ema_steps"), torch.Tensor))
                else torch.zeros((), device=device, dtype=calc_dtype)
            )
            raw_auto_ne = torch.clamp(
                (e_cur - prev_ne_mean) / (torch.sqrt(prev_ne_var.clamp_min(1e-8)) + 1e-4),
                min=-4.0,
                max=4.0,
            )
            auto_ne = torch.where(
                prev_ne_steps > 0.5, raw_auto_ne, torch.zeros_like(raw_auto_ne)
            )
            is_first_ne = prev_ne_steps < 0.5
            new_ne_mean = torch.where(is_first_ne, e_cur, 0.95 * prev_ne_mean + 0.05 * e_cur)
            new_ne_var = torch.where(
                is_first_ne,
                prev_ne_var,
                0.95 * prev_ne_var + 0.05 * (e_cur - prev_ne_mean).pow(2),
            )
            new_ne_steps = prev_ne_steps + 1.0

            ne_scalar_f = (
                ne_signal.to(device=device, dtype=calc_dtype).mean()
                if ne_signal is not None
                else auto_ne
            )

            # ACh efectivo (si None -> ceros para mantener ach_gain vivo y PT1 estable)
            if ach_signal is not None:
                ach_f = ach_signal.to(device=device, dtype=calc_dtype)
                if ach_f.ndim == 1:
                    ach_f = ach_f.unsqueeze(-1)
            else:
                ach_f = torch.zeros((n_assets, 1), device=device, dtype=calc_dtype)

            # 3) Compuertas escalares por cabeza PT1 para Axis 1
            x_summary = x.to(dtype=calc_dtype).mean(dim=0)
            mkt_g = F.linear(
                x_summary,
                self.mkt_gate_proj.weight.to(dtype=calc_dtype),
                self.mkt_gate_proj.bias.to(dtype=calc_dtype),
            )
            a_h_raw, t_h_raw = mkt_g.chunk(2, dim=-1)
            alpha_h_base = self.pt1_alpha_lo + (1.0 - self.pt1_alpha_lo) * torch.sigmoid(
                a_h_raw
            ) * 0.5
            theta_h_base = self.pt1_theta_max * torch.sigmoid(t_h_raw)

            ach_damp_h = torch.exp(
                -F.softplus(self.ach_gain.to(device=device, dtype=calc_dtype))
                * F.softplus(ach_f).mean()
            )
            theta_h = torch.clamp(theta_h_base * ach_damp_h, min=1e-5, max=self.pt1_theta_max)

            ne_boost_h = torch.sigmoid(ne_scalar_f) * torch.sigmoid(
                self.ne_gain.to(device=device, dtype=calc_dtype)
            )
            alpha_h = torch.clamp(
                alpha_h_base + ne_boost_h * (0.95 - alpha_h_base),
                min=self.pt1_alpha_lo,
                max=0.98,
            )

            # 4) Actualización proximal Axis 1 + cascada lenta 2 escalas + momento acotado
            m_mkt_new, cond_ub = self._proximal_cohort_step(
                m_prev, k_h_f, v_h_f, c_weights, alpha_h, theta_h
            )
            m_slow_new = (
                self.mkt_slow_decay * m_slow_prev + (1.0 - self.mkt_slow_decay) * m_mkt_new
            )
            m_mom_raw = self.mkt_mom_decay * m_mom_prev + (m_mkt_new - m_prev)
            mom_frob = torch.sqrt((m_mom_raw * m_mom_raw).sum(dim=(-2, -1), keepdim=True) + 1e-12)
            m_mom_new = m_mom_raw * torch.clamp((0.5 * self.m_max) / mom_frob, max=1.0)

        # 5) Axis 2: actualización Benna-Fusi por activo (con rechazo Tukey + write_mask en theta)
        a_raw, e_raw, t_raw = self.asset_gate_proj(x).chunk(3, dim=-1)
        alpha_a_base = self.pt1_alpha_lo + (1.0 - self.pt1_alpha_lo) * torch.sigmoid(a_raw) * 0.5
        eta_a = self.pt1_eta_hi * torch.sigmoid(e_raw)
        theta_a_base = self.pt1_theta_max * torch.sigmoid(t_raw)

        ach_a = ach_f.to(dtype=dtype)
        ach_damp_a = torch.exp(
            -F.softplus(self.ach_gain.to(device=device, dtype=dtype)) * F.softplus(ach_a)
        )
        write_gate_a = w_tukey.to(dtype=dtype).unsqueeze(-1)
        if write_mask is not None:
            wm_a = write_mask.detach().to(device=device, dtype=dtype).reshape(-1, 1)
            wm_a = torch.where(torch.isfinite(wm_a), wm_a, torch.zeros_like(wm_a)).clamp(0.0, 1.0)
            write_gate_a = write_gate_a * wm_a
        theta_a = torch.clamp(
            torch.clamp(theta_a_base * ach_damp_a, min=1e-5, max=self.pt1_theta_max) * write_gate_a,
            min=0.0,
            max=self.pt1_theta_max,
        )

        ne_boost_a = torch.sigmoid(ne_scalar_f.to(dtype=dtype)) * torch.sigmoid(
            self.ne_gain.to(device=device, dtype=dtype)
        )
        alpha_a = torch.clamp(
            alpha_a_base + ne_boost_a * (0.95 - alpha_a_base),
            min=self.pt1_alpha_lo,
            max=0.98,
        )

        key_unit = F.normalize(k_raw, dim=-1, eps=1e-6)
        read_pred = (cur_banks["fast"] * key_unit).sum(dim=-1, keepdim=True) * key_unit
        target_vec = val_raw if hint_t is None else (val_raw + hint_t)
        surprise = target_vec - read_pred

        if self.use_fused_blackwell_kernels:
            fast_new, mom_new = fused_neuromod_update(
                cur_banks["fast"],
                cur_banks["fast_mom"],
                surprise,
                alpha_a,
                eta_a,
                theta_a,
                huber_delta=self.huber_delta,
                use_triton=self.use_triton,
            )
        else:
            fast_new, mom_new = fused_pt1_huber_gate_update(
                cur_banks["fast"],
                cur_banks["fast_mom"],
                surprise,
                alpha_a,
                eta_a,
                theta_a,
                huber_delta=self.huber_delta,
                use_triton=self.use_triton,
            )
        fast_new = 10.0 * torch.tanh(fast_new / 10.0)
        mom_new = 5.0 * torch.tanh(mom_new / 5.0)
        med_new = 0.90 * cur_banks["medium"] + 0.10 * fast_new
        slow_new = 0.98 * cur_banks["slow"] + 0.02 * med_new
        updated_active = dict(fast=fast_new, fast_mom=mom_new, medium=med_new, slow=slow_new)

        # 6) Empaquetado canónico por `asset_ids` (0 host-syncs, dedup sesión 0 con sentinel)
        if cur_ids is not None:
            if (
                prev_ids is None
                or prev_ids.shape[0] == 0
                or state is None
                or "asset_ids" not in state
            ):
                sort_idx = torch.argsort(cur_ids, stable=True)
                c_sorted = cur_ids[sort_idx]
                c_is_dup = torch.zeros_like(c_sorted, dtype=torch.bool)
                if c_sorted.shape[0] > 1:
                    c_is_dup[:-1] = c_sorted[:-1] == c_sorted[1:]
                sentinel_t = torch.full_like(c_sorted, _ASSET_ID_SENTINEL)
                dedup_ids = torch.where(c_is_dup, sentinel_t, c_sorted)
                d_order = torch.argsort(dedup_ids, stable=True)
                sorted_ids = dedup_ids[d_order]
                is_real = (sorted_ids < _ASSET_ID_SENTINEL).unsqueeze(-1)
                new_state: dict[str, torch.Tensor] = {}
                for k in _BANK_KEYS:
                    b_sorted = updated_active[k][sort_idx][d_order]
                    new_state[k] = torch.where(is_real, b_sorted, torch.zeros_like(b_sorted))
                new_state["asset_ids"] = sorted_ids
            else:
                p_len = prev_ids.shape[0]
                c_order = torch.argsort(cur_ids, stable=True)
                c_sorted = cur_ids[c_order]
                c_is_dup = torch.zeros_like(c_sorted, dtype=torch.bool)
                if c_sorted.shape[0] > 1:
                    c_is_dup[:-1] = c_sorted[:-1] == c_sorted[1:]
                pos = torch.searchsorted(prev_ids, c_sorted)
                safe_pos = pos.clamp(max=p_len - 1)
                in_prev = (
                    (pos < p_len)
                    & (prev_ids[safe_pos] == c_sorted)
                    & (c_sorted < _ASSET_ID_SENTINEL)
                )
                upd_mask = in_prev & (~c_is_dup)
                dummy_pos = torch.full_like(safe_pos, p_len)
                scatter_idx = (
                    torch.where(upd_mask, safe_pos, dummy_pos)
                    .unsqueeze(-1)
                    .expand(-1, self.hidden_dim)
                )
                sentinel_t = torch.full_like(c_sorted, _ASSET_ID_SENTINEL)
                add_ids = torch.where(in_prev | c_is_dup, sentinel_t, c_sorted)
                merged_ids = torch.cat([prev_ids, add_ids], dim=0)
                m_order = torch.argsort(merged_ids, stable=True)
                max_cap = max(p_len, _DEFAULT_MAX_ASSET_TABLE, c_sorted.shape[0] * 4)
                if m_order.shape[0] > max_cap:
                    m_order = m_order[:max_cap]
                sorted_ids = merged_ids[m_order]
                is_real = (sorted_ids < _ASSET_ID_SENTINEL).unsqueeze(-1)
                pad_row = torch.zeros((1, self.hidden_dim), device=device, dtype=dtype)
                add_keep = (~(in_prev | c_is_dup)).unsqueeze(-1)
                new_state = {}
                for k in _BANK_KEYS:
                    prev_bank = state[k].to(device=device, dtype=dtype)
                    cur_sorted_bank = updated_active[k][c_order]
                    prev_ext = torch.cat([prev_bank, pad_row], dim=0)
                    prev_updated = prev_ext.scatter(0, scatter_idx, cur_sorted_bank)[:p_len]
                    add_bank = torch.where(
                        add_keep, cur_sorted_bank, torch.zeros_like(cur_sorted_bank)
                    )
                    merged_bank = torch.cat([prev_updated, add_bank], dim=0)
                    gathered_m = merged_bank[m_order]
                    new_state[k] = torch.where(is_real, gathered_m, torch.zeros_like(gathered_m))
                new_state["asset_ids"] = sorted_ids
        else:
            new_state = updated_active

        # Añadir claves Axis 1 y pulso NE causal persistente
        new_state["M_mkt"] = m_mkt_new
        new_state["M_mkt_slow"] = m_slow_new
        new_state["M_mkt_mom"] = m_mom_new
        new_state["last_ne_signal"] = ne_scalar_f.detach()
        new_state["ne_ema_mean"] = new_ne_mean.detach()
        new_state["ne_ema_var"] = new_ne_var.detach()
        new_state["ne_ema_steps"] = new_ne_steps.detach()

        if detach_state:
            detached = self.detach_state(new_state)
            assert detached is not None
            new_state = detached

        diag = dict(
            alpha_min=torch.minimum(alpha_h.detach().min(), alpha_a.detach().to(calc_dtype).min()),
            alpha_mean=alpha_h.detach().mean(),
            eta_max=eta_a.detach().to(calc_dtype).max(),
            theta_max=torch.maximum(theta_h.detach().max(), theta_a.detach().to(calc_dtype).max()),
            theta_mean=theta_h.detach().mean(),
            cond_upper_bound=cond_ub,
            tukey_min_weight=w_tukey.detach().min(),
            tukey_zero_frac=(w_tukey.detach() <= 1e-8).to(dtype=calc_dtype).mean(),
            mkt_norm=m_mkt_new.detach().norm(dim=(-2, -1)).mean(),
            last_ne_signal=ne_scalar_f.detach(),
        )
        return new_state, diag
