"""Interferómetro holográfico de sesión en fase dual (`holographic_session_interferometer_v1`).

Tras MAC o tras la rama técnica, cada activo i de la sesión S (mismo mercado y mismo instante
de decisión t) tiene su representación e_i ∈ R^H. El interferómetro construye un campo
holográfico compacto por banda m ∈ {1, …, M} usando fasores de módulo unitario en el toro de
Clifford T^{d_c} (FHRR exacta sin FFT ni colapso de norma):

    n_i = LayerNorm(e_i)
    θ_{i,m}, φ_{i,m} ∈ R^{d_c},   v_{i,m} ∈ R^{d_v},   g_{i,m} = 2 tanh(w^π_m · n_i)
    π_{i,m} = softmax_{i ∈ S}(g_{i,m})
    H^{Re}_m = Σ_{j ∈ S} π_{j,m} cos(θ_{j,m}) v_{j,m}ᵀ
    H^{Im}_m = Σ_{j ∈ S} π_{j,m} sin(θ_{j,m}) v_{j,m}ᵀ

Cada activo i desenlaza el holograma con su fasor conjugado e^{-i φ_{i,m}}, resta su propia
contribución (cancelación leave-one-out exacta en O(1) por activo) y centra frente a la media
leave-one-out de sus pares v̄^{(-i)}_m, obteniendo en coste lineal O(N M d_c d_v):

    Δ^{Re}_{i,m} = Σ_{j ∈ S ∖ {i}} π̃^{(i)}_{j,m} (K^{Re}_{ij,m} − ρ^{Re}_{i,m}) v_{j,m}
    Δ^{Im}_{i,m} = Σ_{j ∈ S ∖ {i}} π̃^{(i)}_{j,m} (K^{Im}_{ij,m} − ρ^{Im}_{i,m}) v_{j,m}

donde K^{Re}_{ij,m} = media_k cos(θ_{j,m,k} − φ_{i,m,k}) ∈ [−1, 1] es la interferencia
simétrica en fase/contrafase, K^{Im}_{ij,m} = media_k sin(θ_{j,m,k} − φ_{i,m,k}) ∈ [−1, 1] es
la interferencia antisimétrica en cuadratura y (ρ^{Re}_{i,m}, ρ^{Im}_{i,m}) mide la coherencia
de fase del activo i frente al resto de la sesión.

Propiedades algebraicas exactas:
- Suma cero en los pesos efectivos: Σ_{j≠i} π̃^{(i)}_{j,m} (K_{ij,m} − ρ_{i,m}) = 0. Un
  desplazamiento uniforme v_j → v_j + c se cancela idénticamente en Δ (sin colapso uniforme).
- Cero eco propio: ∂Δ_{i,m}/∂v_{i,m} = 0 y ∂ρ_{i,m}/∂θ_{i,m} = 0.
- Acotación numérica: g_{i,m} ∈ [−2, 2] asegura 1 − π_{i,m} ≥ 1/(1 + e^4) > 0 para N ≥ 2.
- Inicialización neutra sin doble cero: W_out se inicializa Kaiming (no nula) y la ganancia
  vectorial g arranca en cero, e'_i = e_i + g ⊙ (W_out LayerNorm(z_i)), de modo que el
  predictor parte bit a bit de su control y g recibe gradiente en el primer paso.

Modos con los mismos parámetros y la misma inicialización:
- `interferometric`: mecanismo completo (en fase, cuadratura y coherencia de fase).
- `in_phase_only`: ablación que anula el canal antisimétrico de cuadratura (Δ^{Im}, ρ^{Im}).
- `self_only`: control fila a fila sin lectura de otros activos (evalúa K_{ii} sobre sí mismo).
- `shuffled_peers`: control placebo que permuta deterministamente la lectura entre los activos
  de cada sesión (con semilla derivada del predictor, mercado e instante), conservando su
  distribución marginal y rompiendo su correspondencia con cada activo.
"""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from .config import bounded_integer, require_identity
from .state import check_finite

SESSION_HOLOGRAM = "holographic_session_interferometer_v1"
INTERFEROMETRIC = "interferometric"
IN_PHASE_ONLY = "in_phase_only"
SELF_ONLY = "self_only"
SHUFFLED_PEERS = "shuffled_peers"
MODES = (INTERFEROMETRIC, IN_PHASE_ONLY, SELF_ONLY, SHUFFLED_PEERS)
BANDS = (1, 2, 4, 8)
MIN_PHASE_DIM = 2
MAX_PHASE_DIM = 64
DENSITY_BOUND = 2.0
POSITIONS = dict(
    after_mac="after_mac_before_head",
    after_technical="after_technical_branch_before_head",
)


@dataclass(frozen=True)
class SessionHologramConfig:
    bands: int = 4
    phase_dim: int = 16
    mode: str = INTERFEROMETRIC
    position: str = "after_technical"

    def __post_init__(self):
        if type(self.bands) is not int or self.bands not in BANDS:
            raise ValueError(f"El interferómetro holográfico admite {BANDS} bandas")
        bounded_integer(
            self.phase_dim, "dimensión de fase del interferómetro", MIN_PHASE_DIM, MAX_PHASE_DIM
        )
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo del interferómetro debe ser uno de {MODES}")
        if type(self.position) is not str or self.position not in POSITIONS:
            raise ValueError("La posición del interferómetro debe ser after_mac o after_technical")

    def identity(self):
        groups = (
            "each_row_with_itself"
            if self.mode == SELF_ONLY
            else "same_market_and_prediction_instant_in_the_block"
        )
        channels = dict(
            interferometric="in_phase_and_quadrature_covariance_plus_phase_coherence",
            in_phase_only="in_phase_covariance_and_coherence_quadrature_zeroed",
            self_only="row_local_self_phase_modulation_without_peer_mixing",
            shuffled_peers="interferometric_readout_permuted_within_each_session_by_seed",
        )[self.mode]
        return dict(
            schema_version=1,
            kind=SESSION_HOLOGRAM,
            bands=self.bands,
            phase_dim=self.phase_dim,
            mode=self.mode,
            position=POSITIONS[self.position],
            density_bound=DENSITY_BOUND,
            groups=groups,
            binding="unitary_phase_clifford_torus_cos_sin",
            leave_one_out="exact_algebraic_self_subtraction",
            covariance="zero_sum_centered_over_leave_one_out_peers",
            channels=channels,
            normalization="pre_layer_norm_and_readout_layer_norm",
            output="residual_with_zero_initialized_vector_gate_over_kaiming_projection",
            state="none",
            parameter_updates="none_in_forward",
        )


def parse_session_hologram(raw):
    """Aceptar la configuración, sus valores declarados o None."""
    if raw is None:
        return None
    if type(raw) is SessionHologramConfig:
        raw.__post_init__()
        return raw
    required = {"bands", "phase_dim", "mode"}
    allowed = required | {"position"}
    if isinstance(raw, dict) and required <= set(raw) <= allowed:
        return SessionHologramConfig(**raw)
    raise ValueError(
        "session_hologram debe declarar bands, phase_dim y mode (y opcionalmente position), "
        "o ser None"
    )


class HolographicSessionInterferometer(nn.Module):
    """Interferencia holográfica de fase dual en O(N) sobre los activos de cada sesión."""

    session_level = True

    def __init__(self, hidden_size, config, *, seed, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión del interferómetro holográfico", 1, 4096)
        bounded_integer(seed, "semilla del interferómetro holográfico", 0, 2**32 - 1)
        if type(config) is not SessionHologramConfig:
            raise ValueError("El interferómetro holográfico necesita SessionHologramConfig")
        config.__post_init__()
        if hidden_size % config.bands:
            raise ValueError("La dimensión debe ser múltiplo del número de bandas")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("El interferómetro holográfico necesita precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        self.band_dim = hidden_size // config.bands
        self.readout_dim = 2 * hidden_size + 2 * config.bands
        phase_width = config.bands * config.phase_dim
        # Semilla aislada e independiente de mode/position: variante y controles arrancan iguales.
        derived = hashlib.sha256(f"{SESSION_HOLOGRAM}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, dtype=dtype)
            self.proj = nn.Linear(
                hidden_size, 2 * phase_width + hidden_size, bias=False, dtype=dtype
            )
            self.density = nn.Linear(hidden_size, config.bands, bias=False, dtype=dtype)
            self.readout_norm = nn.LayerNorm(self.readout_dim, dtype=dtype)
            self.out = nn.Linear(self.readout_dim, hidden_size, bias=False, dtype=dtype)
        # Densidad inicial uniforme (π = 1/N) y puerta vectorial a cero sobre W_out no nula.
        nn.init.zeros_(self.density.weight)
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def get_extra_state(self):
        state = dict(configuration=self.config.identity(), hidden_size=self.hidden_size)
        if self.config.mode == SHUFFLED_PEERS:
            state["permutation_seed"] = self.seed
        return state

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def groups(self, markets, prediction_at):
        """Clave y filas de cada sesión del bloque, en el orden de primera aparición."""
        sessions = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        return list(sessions.items())

    def _permutation(self, key, rows):
        market, instant = key
        text = f"{SESSION_HOLOGRAM}/{self.seed}/{market}/{instant}"
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(rows, generator=generator)

    def _session_readout(
        self, c_theta, s_theta, c_phi, s_phi, values, density_logits, self_re, self_im
    ):
        """Lectura interferométrica exacta O(N M d_c d_v) para una sesión completa de N filas."""
        rows = len(values)
        if rows == 1:
            zero_anchor = (values.sum() + density_logits.sum() + self_re.sum() + self_im.sum()) * 0
            return values.new_zeros((1, self.readout_dim)) + zero_anchor
        pi = torch.softmax(density_logits, dim=0)
        weighted = pi.unsqueeze(-1) * values
        holo_re = torch.einsum("rmk,rmd->mkd", c_theta, weighted)
        holo_im = torch.einsum("rmk,rmd->mkd", s_theta, weighted)
        bar_c = torch.einsum("rm,rmk->mk", pi, c_theta)
        bar_s = torch.einsum("rm,rmk->mk", pi, s_theta)
        bar_v = weighted.sum(dim=0)

        inv_dc = 1.0 / self.config.phase_dim
        full_re = (
            torch.einsum("rmk,mkd->rmd", c_phi, holo_re)
            + torch.einsum("rmk,mkd->rmd", s_phi, holo_im)
        ) * inv_dc
        full_im = (
            torch.einsum("rmk,mkd->rmd", c_phi, holo_im)
            - torch.einsum("rmk,mkd->rmd", s_phi, holo_re)
        ) * inv_dc
        align_re = (
            torch.einsum("rmk,mk->rm", c_phi, bar_c) + torch.einsum("rmk,mk->rm", s_phi, bar_s)
        ) * inv_dc
        align_im = (
            torch.einsum("rmk,mk->rm", c_phi, bar_s) - torch.einsum("rmk,mk->rm", s_phi, bar_c)
        ) * inv_dc

        loo_scale = 1.0 / (1.0 - pi)
        loo_scale_v = loo_scale.unsqueeze(-1)
        r_re = loo_scale_v * (full_re - self_re.unsqueeze(-1) * weighted)
        r_im = loo_scale_v * (full_im - self_im.unsqueeze(-1) * weighted)
        rho_re = loo_scale * (align_re - pi * self_re)
        rho_im = loo_scale * (align_im - pi * self_im)
        v_loo_mean = loo_scale_v * (bar_v.unsqueeze(0) - weighted)

        delta_re = r_re - rho_re.unsqueeze(-1) * v_loo_mean
        delta_im = r_im - rho_im.unsqueeze(-1) * v_loo_mean
        if self.config.mode == IN_PHASE_ONLY:
            delta_im = torch.zeros_like(delta_im)
            rho_im = torch.zeros_like(rho_im)
        return torch.cat(
            (
                delta_re.reshape(rows, self.hidden_size),
                delta_im.reshape(rows, self.hidden_size),
                rho_re,
                rho_im,
            ),
            dim=-1,
        )

    def forward(self, encoded, markets, prediction_at):
        reference = self.gate
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.layout != torch.strided
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError(
                "El interferómetro necesita una representación, un mercado y un instante por fila"
            )
        if torch.is_autocast_enabled(encoded.device.type):
            raise ValueError("El interferómetro holográfico exige FP32 o FP64 sin autocast")
        check_finite(encoded, "La representación de entrada al interferómetro holográfico")

        rows = len(encoded)
        bands, dc, dv = self.config.bands, self.config.phase_dim, self.band_dim
        phase_width = bands * dc
        normed = self.norm(encoded)
        projected = self.proj(normed)
        theta = projected[:, :phase_width].reshape(rows, bands, dc)
        phi = projected[:, phase_width : 2 * phase_width].reshape(rows, bands, dc)
        values = projected[:, 2 * phase_width :].reshape(rows, bands, dv)
        c_theta, s_theta = torch.cos(theta), torch.sin(theta)
        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        inv_dc = 1.0 / dc
        self_re = (c_phi * c_theta + s_phi * s_theta).sum(dim=-1) * inv_dc
        self_im = (c_phi * s_theta - s_phi * c_theta).sum(dim=-1) * inv_dc

        if self.config.mode == SELF_ONLY:
            readout = torch.cat(
                (
                    (self_re.unsqueeze(-1) * values).reshape(rows, self.hidden_size),
                    (self_im.unsqueeze(-1) * values).reshape(rows, self.hidden_size),
                    self_re,
                    self_im,
                ),
                dim=-1,
            )
        else:
            density_logits = DENSITY_BOUND * torch.tanh(self.density(normed))
            sessions = self.groups(markets, prediction_at)
            if len(sessions) == 1:
                key, _ = sessions[0]
                readout = self._session_readout(
                    c_theta, s_theta, c_phi, s_phi, values, density_logits, self_re, self_im
                )
                if self.config.mode == SHUFFLED_PEERS and rows > 1:
                    order = self._permutation(key, rows).to(encoded.device)
                    readout = readout.index_select(0, order)
            else:
                readout = encoded.new_zeros((rows, self.readout_dim))
                for key, members in sessions:
                    if encoded.is_cuda:
                        index = torch.tensor(members, dtype=torch.int64, pin_memory=True).to(
                            device=encoded.device, non_blocking=True
                        )
                    else:
                        index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
                    part = self._session_readout(
                        c_theta.index_select(0, index),
                        s_theta.index_select(0, index),
                        c_phi.index_select(0, index),
                        s_phi.index_select(0, index),
                        values.index_select(0, index),
                        density_logits.index_select(0, index),
                        self_re.index_select(0, index),
                        self_im.index_select(0, index),
                    )
                    if self.config.mode == SHUFFLED_PEERS and len(members) > 1:
                        order = self._permutation(key, len(members)).to(encoded.device)
                        part = part.index_select(0, order)
                    readout = readout.index_copy(0, index, part)

        residual = self.out(self.readout_norm(readout))
        check_finite(residual, "La proyección del interferómetro holográfico de sesión")
        result = encoded + self.gate * residual
        check_finite(result, "La representación con interferómetro holográfico de sesión")
        return result
