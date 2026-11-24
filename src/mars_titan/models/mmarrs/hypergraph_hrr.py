"""Hipergrafo Transversal `O(N*K)`, Memoria `HRR-Hopfield` FFT y Cerebro Holográfico 3 Niveles.

Completa el mapa abierto en `holographic_network.py` (niveles 2 y 3) con cero host-syncs GPU
para máximo rendimiento en NVIDIA RTX PRO 6000 Blackwell (`sm_120`):
  1. `RMTSpilloverHypergraph`: paso de mensajes `O(N*K)` equivariante por permutación combinando
     modos espectrales RMT y ranuras de hiperaristas aprendidas (`out_scale = 0.05`, sin bloqueo
     triple-zero FM-5).
  2. `HRRModernHopfieldMemory`: enlace holográfico rol-relleno FFT exacto, capacidad por omisión
     `max_slots=512`, almacenamiento de `k_proto=4` prototipos comprimidos de sesión/régimen (jamás
     sobrescribe las ranuras alfabéticamente con `k[:64]`) y lectura activa de trazas multiescala
     (5d/21d/63d/252d) fusionada en `retrieve()`.
  3. `HolographicBrain3L`: Nivel 1 (áreas corticales con priors de dominio diversos en
     `membership_logits` + LoRA 3D `single_zero_v1` sin bloqueo FM-5), Nivel 2 (hipergrafo RMT)
     y Nivel 3 (Tálamo FiLM), con aislamiento bit-a-bit `torch.where` frente a `NaN` en ausentes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _domain_catalog_membership_prior(n_modalities: int, n_cortical_areas: int) -> torch.Tensor:
    """Genera priors de pertenencia anatómicos diversos por dominio (evita simetría 1/A)."""
    logits = torch.zeros(n_modalities, n_cortical_areas, dtype=torch.float32)
    if n_modalities == 5 and n_cortical_areas == 19:
        # Layout canónico Entrega 2 §4.1.3: 3 precios, 2 fund, 12 macro, 1 noticias, 1 gráficos
        ranges = [(0, 3), (3, 5), (5, 17), (17, 18), (18, 19)]
        for m_idx, (lo, hi) in enumerate(ranges):
            logits[m_idx, lo:hi] = 1.5
    else:
        for a_idx in range(n_cortical_areas):
            m_idx = (a_idx * n_modalities) // max(n_cortical_areas, 1)
            m_idx = min(m_idx, n_modalities - 1)
            logits[m_idx, a_idx] = 1.5
    # Ruptura determinista de simetría dentro de cada dominio
    area_ramp = torch.linspace(-0.20, 0.20, steps=n_cortical_areas, dtype=torch.float32)
    mod_ramp = torch.linspace(-0.15, 0.15, steps=n_modalities, dtype=torch.float32)
    logits = logits + area_ramp.unsqueeze(0) + mod_ramp.unsqueeze(1)
    return logits


class RMTSpilloverHypergraph(nn.Module):
    """Hipergrafo transversal lineal `O(N*K)` estrictamente equivariante por permutación."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_hyperedges: int = 16,
        rmt_dim: int = 5,
        n_rounds: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_hyperedges = n_hyperedges
        self.rmt_dim = rmt_dim
        self.n_rounds = n_rounds

        self.edge_queries = nn.Parameter(
            torch.randn(n_hyperedges, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.rmt_proj = nn.Linear(rmt_dim, n_hyperedges, bias=False)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.msg_in = nn.Linear(hidden_dim, hidden_dim)
        self.msg_out = nn.Linear(hidden_dim, hidden_dim)
        # P1-2 (single_zero_v1): pequeño valor no nulo para gradientes vivos desde paso 0
        self.out_scale = nn.Parameter(torch.full((), 0.05, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        *,
        rmt_loadings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Propaga mensajes vértice -> hiperarista -> vértice en `O(N*K*D)` (cero host-syncs)."""
        h = x
        scale = 1.0 / math.sqrt(self.hidden_dim)
        # Incidencia suave equivariante: logits [N, K]
        logits = (self.node_norm(h) @ self.edge_queries.T) * scale
        if rmt_loadings is not None and rmt_loadings.shape[-1] == self.rmt_dim:
            logits = logits + self.rmt_proj(rmt_loadings.to(device=h.device, dtype=h.dtype))
        inc = torch.softmax(logits, dim=-1)  # [N, K]
        # Normalización por grado de hiperarista (evita dominancia de grupos gigantes)
        edge_deg = inc.sum(dim=0, keepdim=True).clamp_min(1e-6)  # [1, K]
        inc_norm = inc / edge_deg

        out_s = self.out_scale.to(dtype=h.dtype)
        for _ in range(self.n_rounds):
            # Vértice -> Hiperarista: [K, D] = [K, N] @ [N, D]
            v = F.silu(self.msg_in(self.node_norm(h)))
            edge_state = inc_norm.T @ v
            # Hiperarista -> Vértice: [N, D] = [N, K] @ [K, D]
            node_msg = self.msg_out(inc @ edge_state)
            h = h + out_s * node_msg

        ent = -(inc * torch.log(inc.clamp_min(1e-8))).sum(dim=-1).mean()
        return h, dict(incidence_entropy=ent.detach())


class HRRModernHopfieldMemory(nn.Module):
    """Memoria asociativa holográfica HRR (FFT) + Modern Hopfield + Trazas Multiescala."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        max_slots: int = 512,
        k_proto: int = 4,
        beta: float = 6.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_slots = int(max_slots)
        self.k_proto = max(1, int(k_proto))
        self.beta = float(beta)

        self.register_buffer("slot_keys", torch.zeros(self.max_slots, hidden_dim))
        self.register_buffer("slot_vals", torch.zeros(self.max_slots, hidden_dim))
        self.register_buffer("slot_filled", torch.zeros(self.max_slots, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.zeros((), dtype=torch.long))
        # Trazas superpuestas multiescala (5d, 21d, 63d, 252d)
        self.register_buffer("traces", torch.zeros(4, hidden_dim))
        self.trace_decays = (0.80, 0.95, 0.985, 0.996)

        # Consultas prototipo ortogonales fijas para compresión equivariante de `k_proto` ranuras
        proto_q = torch.randn(max(self.k_proto, 4), hidden_dim)
        proto_q = F.normalize(proto_q, dim=-1, eps=1e-6)
        self.register_buffer("proto_queries", proto_q[: self.k_proto])

        # Mezcla aprendible de las 4 trazas temporales (5d/21d/63d/252d) fusionada en `retrieve()`
        self.trace_weights = nn.Parameter(torch.tensor([0.3, 0.2, 0.1, 0.0], dtype=torch.float32))
        self.trace_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.trace_proj.weight, std=0.05)
        nn.init.zeros_(self.trace_proj.bias)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Identidad exacta con buffer vacío y lectura activa inmediata tras `store()`
        nn.init.normal_(self.out_proj.weight, std=0.05)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def circular_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Enlace holográfico exacto `a ⊛ b = irfft(rfft(a) * rfft(b))` en O(d log d)."""
        d = a.shape[-1]
        fa = torch.fft.rfft(a.float(), n=d, dim=-1)
        fb = torch.fft.rfft(b.float(), n=d, dim=-1)
        out = torch.fft.irfft(fa * fb, n=d, dim=-1)
        return out.to(dtype=a.dtype)

    @staticmethod
    def circular_unbind(bound: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
        """Desenlace aproximado (correlación circular) `bound ★ role`."""
        d = bound.shape[-1]
        fc = torch.fft.rfft(bound.float(), n=d, dim=-1)
        fr = torch.fft.rfft(role.float(), n=d, dim=-1)
        out = torch.fft.irfft(fc * torch.conj(fr), n=d, dim=-1)
        return out.to(dtype=bound.dtype)

    def _encode_key(
        self,
        query: torch.Tensor,
        regime_key: torch.Tensor | None = None,
        sector_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = F.normalize(query, dim=-1, eps=1e-6)
        if regime_key is not None:
            rk = F.normalize(
                regime_key.to(device=query.device, dtype=query.dtype), dim=-1, eps=1e-6
            )
            if rk.ndim == 1:
                rk = rk.unsqueeze(0).expand_as(k)
            k = self.circular_bind(k, rk)
        if sector_key is not None:
            sk = F.normalize(
                sector_key.to(device=query.device, dtype=query.dtype), dim=-1, eps=1e-6
            )
            if sk.ndim == 1:
                sk = sk.unsqueeze(0).expand_as(k)
            k = self.circular_bind(k, sk)
        return F.normalize(k, dim=-1, eps=1e-6)

    def _compress_session_prototypes(
        self,
        k_det: torch.Tensor,
        v_det: torch.Tensor,
        *,
        return_signal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Comprime N activos en `k_proto` prototipos equivariantes de sesión/régimen (P0-2)."""
        n_assets = k_det.shape[0]
        k_target = min(self.k_proto, self.max_slots)
        if n_assets <= k_target:
            return k_det, v_det

        # 1) Prototipo 0: media global de sesión (ancla de régimen)
        proto_k_list: list[torch.Tensor] = [
            F.normalize(k_det.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
        ]
        proto_v_list: list[torch.Tensor] = [v_det.mean(dim=0, keepdim=True)]

        # 2) Prototipos selectivos M3 (positivos / negativos en retorno/sorpresa madura)
        if return_signal is not None and return_signal.shape[0] == n_assets and k_target >= 3:
            sig = return_signal.detach().to(device=k_det.device, dtype=k_det.dtype).reshape(-1)
            sig_std = sig.std().clamp_min(1e-4)
            z_sig = torch.clamp(sig / sig_std, -6.0, 6.0)
            w_pos = torch.softmax(z_sig * 2.0, dim=0).unsqueeze(-1)
            w_neg = torch.softmax(-z_sig * 2.0, dim=0).unsqueeze(-1)
            proto_k_list.append(
                F.normalize((w_pos * k_det).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)
            )
            proto_v_list.append((w_pos * v_det).sum(dim=0, keepdim=True))
            proto_k_list.append(
                F.normalize((w_neg * k_det).sum(dim=0, keepdim=True), dim=-1, eps=1e-6)
            )
            proto_v_list.append((w_neg * v_det).sum(dim=0, keepdim=True))

        # 3) Completar hasta `k_target` mediante clustering suave por atención sobre `proto_queries`
        rem = k_target - len(proto_k_list)
        if rem > 0:
            q_ref = self.proto_queries[:rem].to(device=k_det.device, dtype=k_det.dtype)  # [R, D]
            logits = (q_ref @ k_det.T) * 4.0  # [R, N]
            assign = torch.softmax(logits, dim=-1)  # [R, N]
            k_cl = F.normalize(assign @ k_det, dim=-1, eps=1e-6)
            v_cl = assign @ v_det
            proto_k_list.append(k_cl)
            proto_v_list.append(v_cl)

        k_out = torch.cat(proto_k_list, dim=0)[:k_target]
        v_out = torch.cat(proto_v_list, dim=0)[:k_target]
        return k_out, v_out

    @torch.no_grad()
    def store(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        *,
        regime_key: torch.Tensor | None = None,
        sector_key: torch.Tensor | None = None,
        return_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Almacena `k_proto` prototipos comprimidos por sesión en ranuras Hopfield y trazas HRR."""
        k_enc = self._encode_key(query.detach(), regime_key=regime_key, sector_key=sector_key)
        v_det = value.detach().to(device=self.slot_vals.device, dtype=self.slot_vals.dtype)
        k_det = k_enc.to(device=self.slot_keys.device, dtype=self.slot_keys.dtype)

        k_write, v_write = self._compress_session_prototypes(
            k_det, v_det, return_signal=return_signal
        )
        n_in = min(k_write.shape[0], self.max_slots)

        # Actualizar trazas exponenciales multiescala (5d/21d/63d/252d) con prototipo medio
        proto = k_det.mean(dim=0)
        for idx, dec in enumerate(self.trace_decays):
            self.traces[idx].mul_(dec).add_(proto * (1.0 - dec))

        # Escribir en buffer circular determinista sin host-syncs (cero `.item()`)
        offsets = torch.arange(n_in, device=self.slot_keys.device, dtype=torch.long)
        indices = (offsets + self.write_ptr) % self.max_slots
        self.slot_keys[indices] = k_write[:n_in]
        self.slot_vals[indices] = v_write[:n_in]
        self.slot_filled.index_fill_(0, indices, True)
        self.write_ptr.copy_((self.write_ptr + n_in) % self.max_slots)
        return self.slot_filled.sum()

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        regime_key: torch.Tensor | None = None,
        sector_key: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Recuperación Modern Hopfield (Ramsauer) + fusión activa de `self.traces` (cero syncs)."""
        n_assets = query.shape[0]
        device, dtype = query.device, query.dtype
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        k_q = self._encode_key(query, regime_key=regime_key, sector_key=sector_key)
        filled = self.slot_filled.to(device=device)
        keys = self.slot_keys.to(device=device, dtype=calc_dtype)
        vals = self.slot_vals.to(device=device, dtype=calc_dtype)
        traces_d = self.traces.to(device=device, dtype=dtype)
        # En BPTT multi-sesión `store()` muta buffers entre pasos: clonar snapshot
        if torch.is_grad_enabled():
            filled = filled.clone()
            keys = keys.clone()
            vals = vals.clone()
            traces_d = traces_d.clone()
        filled_f = filled.to(dtype=calc_dtype)
        filled_cnt = filled_f.sum()
        has_slots = filled_cnt > 0.5

        # Similitud coseno en [-1, 1] -> energía Ramsauer shifted >= 0 (enmascarada sin sync!)
        sim = k_q.to(dtype=calc_dtype) @ keys.T  # [N, M]
        scaled = self.beta * sim
        masked_scaled = torch.where(filled.unsqueeze(0), scaled, torch.full_like(scaled, -1e6))
        attn = torch.softmax(masked_scaled, dim=-1) * filled_f.unsqueeze(0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn = torch.where(has_slots, attn, torch.zeros_like(attn))
        retrieved_raw = (attn @ vals).to(dtype=dtype)

        # Energía de Ramsauer invariante a M y exactamente 0 con banco vacío
        lse = torch.logsumexp(masked_scaled, dim=-1) / self.beta
        ood_raw = torch.relu(1.0 + torch.log(filled_cnt.clamp_min(1.0)) / self.beta - lse)
        ood_energy = torch.where(has_slots, ood_raw, torch.zeros_like(ood_raw)).to(dtype=dtype)

        # Lectura activa de trazas multiescala (5d/21d/63d/252d) fusionada en retrieve() (P0-2)
        has_trace = traces_d.abs().sum() > 1e-12
        w_tr = torch.softmax(self.trace_weights.to(device=device, dtype=dtype), dim=0)
        tr_mix = (w_tr.unsqueeze(-1) * traces_d).sum(dim=0)  # [D]
        tr_exp = tr_mix.unsqueeze(0).expand(n_assets, -1)
        tr_unbound = self.circular_unbind(k_q, F.normalize(tr_exp, dim=-1, eps=1e-6))
        trace_readout_raw = self.trace_proj(tr_exp + 0.5 * tr_unbound)
        trace_readout = torch.where(
            has_trace, trace_readout_raw, torch.zeros_like(trace_readout_raw)
        )

        hop_out = torch.where(
            has_slots, self.out_proj(retrieved_raw), torch.zeros_like(retrieved_raw)
        )
        retrieved = hop_out + trace_readout
        return dict(
            retrieved=retrieved,
            ood_energy=ood_energy,
            attention=attn.to(dtype=dtype),
            trace_readout=trace_readout,
        )


class HolographicBrain3L(nn.Module):
    """Cerebro Holográfico Jerárquico de 3 Niveles (Cortical + Hipergrafo + Tálamo FiLM)."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_modalities: int = 5,
        n_cortical_areas: int = 19,
        n_hidden_slots: int = 4,
        n_hyperedges: int = 16,
        regime_dim: int = 12,
        rmt_dim: int = 5,
        lora_rank: int = 4,
        cycles: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.n_cortical_areas = n_cortical_areas
        self.n_hidden_slots = n_hidden_slots
        self.n_nodes = n_cortical_areas + n_hidden_slots
        self.cycles = cycles

        # Nivel 1: priors de dominio diversos + LoRA 3D `single_zero_v1` (rompe bloqueo FM-5!)
        self.membership_logits = nn.Parameter(
            _domain_catalog_membership_prior(n_modalities, n_cortical_areas)
        )
        self.area_u = nn.Parameter(
            torch.randn(n_cortical_areas, hidden_dim, lora_rank) * (0.05 / math.sqrt(lora_rank))
        )
        self.area_v = nn.Parameter(
            torch.randn(n_cortical_areas, hidden_dim, lora_rank) / math.sqrt(hidden_dim)
        )
        self.area_emb = nn.Parameter(
            torch.randn(n_cortical_areas, hidden_dim) * 0.02
        )
        self.slot_keys = nn.Parameter(
            torch.randn(n_hidden_slots, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.slot_emb = nn.Parameter(torch.randn(n_hidden_slots, hidden_dim) * 0.02)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # Puente cortical con SDPA (FlashAttention / cuDNN en Blackwell; MATH en fp32_strict)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.bridge_out = nn.Linear(hidden_dim, hidden_dim)

        # Nivel 2: Hipergrafo transversal entre activos O(N*K)
        self.hypergraph_l2 = RMTSpilloverHypergraph(
            hidden_dim, n_hyperedges=n_hyperedges, rmt_dim=rmt_dim, n_rounds=1
        )
        with torch.no_grad():
            self.hypergraph_l2.out_scale.fill_(0.25)

        # Nivel 3: Tálamo de Régimen Global (FiLM gamma/beta) — init pequeño no nulo (P1-2)
        self.thalamus_film = nn.Sequential(
            nn.Linear(regime_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.normal_(self.thalamus_film[-1].weight, std=0.05)
        nn.init.zeros_(self.thalamus_film[-1].bias)

        # Proyección de vuelta con escala pequeña (0.01): casi identidad al nacer y gradiente paso 0
        self.readback = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.readback.weight, std=0.01)
        nn.init.zeros_(self.readback.bias)

    def forward(
        self,
        mod_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        *,
        regime_vec: torch.Tensor | None = None,
        rmt_loadings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Transforma `[N, M, D]` respetando `modality_mask [N, M]` (aislamiento `NaN` seguro)."""
        n_assets, n_mod, d_model = mod_tokens.shape
        mask_bool = modality_mask.to(device=mod_tokens.device) > 0.5
        zeros_tok = torch.zeros_like(mod_tokens)
        present_tokens = torch.where(mask_bool.unsqueeze(-1), mod_tokens, zeros_tok)

        # --- NIVEL 1: Áreas Corticales (LoRA 3D fusionado) + Ranuras Soft-MoE ---
        p_ab = torch.softmax(self.membership_logits.to(dtype=mod_tokens.dtype), dim=-1)  # [M, A]
        m_a = torch.einsum("ma,nmd->nad", p_ab, present_tokens)
        low_r = torch.einsum("nad,adr->nar", m_a, self.area_v.to(dtype=mod_tokens.dtype))
        lora_delta = torch.einsum("nar,adr->nad", low_r, self.area_u.to(dtype=mod_tokens.dtype))
        h_areas = self.node_norm(
            m_a + lora_delta + self.area_emb.to(dtype=mod_tokens.dtype).unsqueeze(0)
        )

        # Ranuras ocultas Soft-MoE sobre modalidades presentes
        slot_logits = torch.einsum(
            "nmd,sd->nsm",
            self.node_norm(present_tokens),
            self.slot_keys.to(dtype=mod_tokens.dtype),
        ) / math.sqrt(d_model)
        neg_inf = torch.where(
            mask_bool.unsqueeze(1),
            torch.zeros_like(slot_logits),
            torch.full_like(slot_logits, -1e4),
        )
        d_sm = torch.softmax(slot_logits + neg_inf, dim=-1)  # [N, S, M]
        h_slots = self.node_norm(
            torch.einsum("nsm,nmd->nsd", d_sm, present_tokens)
            + self.slot_emb.to(dtype=mod_tokens.dtype).unsqueeze(0)
        )
        c_ms = torch.softmax(slot_logits.transpose(1, 2), dim=-1)  # [N, M, S]

        # --- NIVEL 3: Tálamo de Régimen FiLM modulando el puente ---
        nodes = torch.cat([h_areas, h_slots], dim=1)  # [N, A+S, D]
        init_nodes = nodes
        if regime_vec is not None:
            rv = regime_vec.to(device=mod_tokens.device, dtype=mod_tokens.dtype)
            if rv.ndim == 1:
                rv = rv.unsqueeze(0).expand(n_assets, -1)
            film = self.thalamus_film(rv)  # [N, 2D]
            gamma, beta = film.chunk(2, dim=-1)
            gamma = 1.0 + 0.5 * torch.tanh(gamma)
            beta = 0.5 * torch.tanh(beta)
        else:
            gamma = torch.ones(
                (n_assets, d_model), device=mod_tokens.device, dtype=mod_tokens.dtype
            )
            beta = torch.zeros_like(gamma)

        # Puente cortical multi-ciclo con SDPA
        for _ in range(self.cycles):
            q, k, v = self.qkv(self.node_norm(nodes)).chunk(3, dim=-1)
            attn = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
            ).squeeze(1)
            nodes = nodes + 0.25 * (
                gamma.unsqueeze(1) * self.bridge_out(attn) + beta.unsqueeze(1)
            )

        # --- NIVEL 2: Intercambio transversal entre activos vía Hipergrafo RMT ---
        summary_in = nodes.mean(dim=1)  # [N, D]
        summary_l2, hg_diag = self.hypergraph_l2(summary_in, rmt_loadings=rmt_loadings)
        l2_delta = summary_l2 - summary_in

        # Lectura de vuelta hacia modalidades presentes (conservando 0 exacto en ausentes)
        delta_nodes = nodes - init_nodes
        delta_areas = delta_nodes[:, : self.n_cortical_areas, :]
        delta_slots = delta_nodes[:, self.n_cortical_areas :, :]
        delta_mod = (
            torch.einsum("ma,nad->nmd", p_ab, delta_areas)
            + torch.einsum("nms,nsd->nmd", c_ms, delta_slots)
            + l2_delta.unsqueeze(1)
        )
        out_cand = present_tokens + self.readback(delta_mod)
        out_tokens = torch.where(mask_bool.unsqueeze(-1), out_cand, zeros_tok)
        return out_tokens, dict(levels=3, l2_incidence_entropy=hg_diag["incidence_entropy"])
