"""Campo holográfico persistente multiescala (`persistent_session_hologram_v1`, MPSH).

Extiende el interferómetro holográfico de sesión (fase unitaria en el toro de Clifford T^{d_c},
cancelación leave-one-out exacta y covarianza de suma cero en fase y cuadratura) con una memoria
holográfica persistente por mercado a L escalas temporales (semividas τ_ℓ ∈ {1, 5, 10} sesiones):

    ṽ_{j,t,m} = v_{j,t,m} − Σ_{r ∈ S_t} π_{r,t,m} v_{r,t,m}
    H^{Re}_{t,m} = Σ_{j ∈ S_t} π_{j,t,m} cos(θ_{j,t,m}) ṽ_{j,t,m}ᵀ
    H^{Im}_{t,m} = Σ_{j ∈ S_t} π_{j,t,m} sin(θ_{j,t,m}) ṽ_{j,t,m}ᵀ
    c̄_{t,m} = Σ_{j ∈ S_t} π_{j,t,m} cos(θ_{j,t,m}),   s̄_{t,m} = Σ_{j ∈ S_t} π_{j,t,m} sin(θ_{j,t,m})

Al cerrar la sesión t (solo en `commit`, después de emitir todas las predicciones de la sesión),
el estado del mercado se actualiza con decaimiento exponencial por semivida λ_ℓ = 2^{-1/τ_ℓ} y masa
efectiva exacta w_{t,ℓ} para corrección de sesgo en arranque frío:

    M^{Re}_{t,ℓ} = λ_ℓ M^{Re}_{t−1,ℓ} + (1 − λ_ℓ) H^{Re}_t
    M^{Im}_{t,ℓ} = λ_ℓ M^{Im}_{t−1,ℓ} + (1 − λ_ℓ) H^{Im}_t
    c̄^{mem}_{t,ℓ} = λ_ℓ c̄^{mem}_{t−1,ℓ} + (1 − λ_ℓ) c̄_t
    s̄^{mem}_{t,ℓ} = λ_ℓ s̄^{mem}_{t−1,ℓ} + (1 − λ_ℓ) s̄_t
    w_{t,ℓ} = λ_ℓ w_{t−1,ℓ} + (1 − λ_ℓ)

Al predecir la sesión t, cada activo i consulta de forma estrictamente causal el campo confirmado
de sesiones anteriores M̂_{t−1,ℓ} = M_{t−1,ℓ} / w_{t−1,ℓ} con su fasor de consulta actual φ_{i,t}.
Como cada holograma de sesión centra ṽ_{j,s} = v_{j,s} − v̄_s antes de superponerse, el desenlace
recupera de forma algebraicamente exacta la media exponencial insesgada sobre sesiones pasadas s < t
de las covarianzas transversales de suma cero (invariante frente a cualquier desplazamiento uniforme
c_s distinto en cada día s) y la compara con la lectura instantánea leave-one-out de la sesión t:

    z^{rot}_{i,t,ℓ} = z^{inst}_{i,t} − z^{mem}_{i,t,ℓ}

combinando los tres flujos (instantáneo, trayectoria histórica depurada y rotación entre sesiones)
mediante proyecciones Kaiming no nulas, pesos aprendidos por escala y canal, y una única puerta
vectorial inicializada a cero (sin doble cero en serie).

Controles negativos integrados con exactamente los mismos parámetros e inicialización:
- `persistent`: mecanismo completo multiescala (instantáneo + memoria persistente + rotación).
- `instant_only`: ablación sin persistencia entre sesiones (solo lectura instantánea de la sesión).
- `shuffled_history`: placebo que mantiene el mismo campo persistente confirmado pero permuta
  deterministamente la lectura histórica entre los activos de cada sesión antes de la rotación.
- `self_only`: control fila a fila sin mezcla transversal entre activos ni memoria entre sesiones.
"""

import hashlib
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .config import LAYER_NORM_EPS, bounded_integer, canonical, require_identity
from .state import check_differentiable, check_finite, require_true

PERSISTENT_SESSION_HOLOGRAM = "persistent_session_hologram_v1"
PERSISTENT = "persistent"
INSTANT_ONLY = "instant_only"
SHUFFLED_HISTORY = "shuffled_history"
SELF_ONLY = "self_only"
MODES = (PERSISTENT, INSTANT_ONLY, SHUFFLED_HISTORY, SELF_ONLY)
BANDS = (1, 2, 4, 8)
MIN_PHASE_DIM = 2
MAX_PHASE_DIM = 64
DENSITY_BOUND = 2.0
POSITIONS = dict(
    after_mac="after_mac_before_head",
    after_technical="after_technical_branch_before_head",
)


@dataclass(frozen=True)
class PersistentSessionHologramConfig:
    """Contrato del campo holográfico persistente transversal-temporal multiescala (MPSH)."""

    bands: int = 4
    phase_dim: int = 16
    half_lives: tuple[int, ...] = (1, 5, 10)
    mode: str = PERSISTENT
    position: str = "after_technical"
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.bands) is not int or self.bands not in BANDS:
            raise ValueError(f"El holograma persistente admite {BANDS} bandas")
        bounded_integer(
            self.phase_dim,
            "dimensión de fase del holograma persistente",
            MIN_PHASE_DIM,
            MAX_PHASE_DIM,
        )
        half_lives = (
            tuple(self.half_lives) if isinstance(self.half_lives, list) else self.half_lives
        )
        if (
            type(half_lives) is not tuple
            or not 1 <= len(half_lives) <= 4
            or any(type(value) is not int or not 1 <= value <= 252 for value in half_lives)
            or list(half_lives) != sorted(set(half_lives))
        ):
            raise ValueError(
                "half_lives debe tener de una a cuatro semividas crecientes en [1, 252]"
            )
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo del holograma persistente debe ser uno de {MODES}")
        if type(self.position) is not str or self.position not in POSITIONS:
            raise ValueError(
                "La posición del holograma persistente debe ser after_mac o after_technical"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla del holograma persistente debe ser un entero de 32 bits")
        object.__setattr__(self, "half_lives", half_lives)

    @property
    def decays(self) -> tuple[float, ...]:
        """Retención exponencial λ_ℓ = 2^{-1/τ_ℓ} de cada escala temporal."""
        return tuple(2.0 ** (-1.0 / tau) for tau in self.half_lives)

    def identity(self) -> dict:
        groups = (
            "each_row_with_itself"
            if self.mode == SELF_ONLY
            else "same_market_and_prediction_instant_in_the_block"
        )
        persistence = dict(
            persistent="unbiased_per_session_centered_multiscale_ema_with_rotation",
            instant_only="disabled_instantaneous_session_hologram_only",
            shuffled_history="unbiased_multiscale_ema_readout_permuted_within_session_by_seed",
            self_only="disabled_row_local_self_phase_modulation_only",
        )[self.mode]
        return dict(
            schema_version=1,
            kind=PERSISTENT_SESSION_HOLOGRAM,
            bands=self.bands,
            phase_dim=self.phase_dim,
            half_lives=list(self.half_lives),
            mode=self.mode,
            position=POSITIONS[self.position],
            seed=self.seed,
            density_bound=DENSITY_BOUND,
            groups=groups,
            binding="unitary_phase_clifford_torus_cos_sin",
            leave_one_out="exact_algebraic_self_subtraction_on_centered_hologram",
            covariance="zero_sum_centered_per_session_in_phase_and_quadrature",
            persistence=persistence,
            causality="committed_only_on_session_close_read_only_in_subsequent_sessions",
            market_isolation="strict_per_market_hologram_state",
            output="residual_with_zero_initialized_vector_gate_over_kaiming_projections",
            parameter_updates="none_in_forward",
        )


def parse_persistent_session_hologram(raw: object) -> PersistentSessionHologramConfig | None:
    """Aceptar PersistentSessionHologramConfig, su diccionario declarado o None."""
    if raw is None:
        return None
    if type(raw) is PersistentSessionHologramConfig:
        raw.__post_init__()
        return raw
    required = {"bands", "phase_dim", "half_lives", "mode"}
    allowed = required | {"position", "seed"}
    if isinstance(raw, dict) and required <= set(raw) <= allowed:
        return PersistentSessionHologramConfig(**raw)
    raise ValueError(
        "persistent_session_hologram debe declarar bands, phase_dim, half_lives y mode "
        "(y opcionalmente position y seed), o ser None"
    )


@dataclass(frozen=True)
class MarketHologramState:
    """Campo holográfico persistente confirmado de un mercado en las L escalas temporales."""

    holo_re: Tensor
    holo_im: Tensor
    mean_c: Tensor
    mean_s: Tensor
    weight: Tensor
    committed_at: int
    updates: int


@dataclass(frozen=True)
class PendingSessionHologram:
    """Holograma compacto de una sesión emitida, pendiente de confirmación al cierre."""

    decision_at: int
    holo_re: Tensor
    holo_im: Tensor
    mean_c: Tensor
    mean_s: Tensor


@dataclass(frozen=True)
class PersistentSessionHologramState:
    """Estado de MPSH: memorias confirmadas por mercado, cursor causal y sesiones pendientes."""

    config_id: str
    parameter_id: str
    markets: dict[str, MarketHologramState]
    pending: dict[str, PendingSessionHologram]
    committed_at: dict[str, int]


class PersistentSessionHologram(nn.Module):
    """Campo holográfico persistente transversal-temporal multiescala en O(N·L·M·d_c·d_v)."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        config: PersistentSessionHologramConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión del holograma persistente", 1, 4096)
        bounded_integer(seed, "semilla del holograma persistente", 0, 2**32 - 1)
        if type(config) is not PersistentSessionHologramConfig:
            raise ValueError("MPSH necesita PersistentSessionHologramConfig")
        config.__post_init__()
        if hidden_size % config.bands:
            raise ValueError("La dimensión debe ser múltiplo del número de bandas")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("El holograma persistente necesita precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        self.band_dim = hidden_size // config.bands
        self.readout_dim = 2 * hidden_size + 2 * config.bands
        scales = len(config.half_lives)
        phase_width = config.bands * config.phase_dim
        # Semilla derivada e independiente de mode/position/config.seed: variante y controles
        # arrancan con pesos bit a bit idénticos sin consumir el generador global.
        derived = hashlib.sha256(f"{PERSISTENT_SESSION_HOLOGRAM}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.proj = nn.Linear(
                hidden_size, 2 * phase_width + hidden_size, bias=False, dtype=dtype
            )
            self.density = nn.Linear(hidden_size, config.bands, bias=False, dtype=dtype)
            self.inst_norm = nn.LayerNorm(
                self.readout_dim, eps=LAYER_NORM_EPS, bias=False, dtype=dtype
            )
            self.mem_norm = nn.LayerNorm(
                self.readout_dim, eps=LAYER_NORM_EPS, bias=False, dtype=dtype
            )
            self.rot_norm = nn.LayerNorm(
                self.readout_dim, eps=LAYER_NORM_EPS, bias=False, dtype=dtype
            )
            self.out_inst = nn.Linear(self.readout_dim, hidden_size, bias=False, dtype=dtype)
            self.out_mem = nn.Linear(self.readout_dim, hidden_size, bias=False, dtype=dtype)
            self.out_rot = nn.Linear(self.readout_dim, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.density.weight)
        self.scale_mem = nn.Parameter(torch.full((scales, hidden_size), 1.0 / scales, dtype=dtype))
        self.scale_rot = nn.Parameter(torch.full((scales, hidden_size), 1.0 / scales, dtype=dtype))
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self._fingerprint = hashlib.sha256(canonical(config.identity()).encode()).hexdigest()

    def fingerprint(self) -> str:
        return self._fingerprint

    def get_extra_state(self) -> dict:
        state = dict(configuration=self.config.identity(), hidden_size=self.hidden_size)
        if self.config.mode == SHUFFLED_HISTORY:
            state["permutation_seed"] = self.seed
        return state

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _reference(self) -> Tensor:
        reference = self.gate
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("El holograma persistente admite precisión FP32 o FP64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("El holograma persistente exige FP32 o FP64 explícitos, sin autocast")
        return reference

    def initial_state(self, parameter_id: str) -> PersistentSessionHologramState:
        return PersistentSessionHologramState(self.fingerprint(), parameter_id, {}, {}, {})

    def validate_state(self, state: object, parameter_id: str) -> None:
        reference = self._reference()
        scales = len(self.config.half_lives)
        bands, dc, dv = self.config.bands, self.config.phase_dim, self.band_dim
        if (
            not isinstance(state, PersistentSessionHologramState)
            or state.config_id != self.fingerprint()
            or state.parameter_id != parameter_id
            or not isinstance(state.markets, dict)
            or not isinstance(state.pending, dict)
            or not isinstance(state.committed_at, dict)
        ):
            raise ValueError("El estado del holograma persistente no corresponde al módulo")
        for market, at in state.committed_at.items():
            if type(market) is not str or not market or type(at) is not int:
                raise ValueError("El cursor causal del holograma persistente no es válido")
        for market, stats in state.markets.items():
            if (
                type(market) is not str
                or not market
                or not isinstance(stats, MarketHologramState)
                or type(stats.updates) is not int
                or stats.updates < 1
                or type(stats.committed_at) is not int
                or state.committed_at.get(market) != stats.committed_at
                or not isinstance(stats.holo_re, Tensor)
                or not isinstance(stats.holo_im, Tensor)
                or not isinstance(stats.mean_c, Tensor)
                or not isinstance(stats.mean_s, Tensor)
                or not isinstance(stats.weight, Tensor)
                or stats.holo_re.shape != (scales, bands, dc, dv)
                or stats.holo_im.shape != (scales, bands, dc, dv)
                or stats.mean_c.shape != (scales, bands, dc)
                or stats.mean_s.shape != (scales, bands, dc)
                or stats.weight.shape != (scales,)
                or stats.holo_re.dtype != reference.dtype
                or stats.holo_im.dtype != reference.dtype
                or stats.mean_c.dtype != reference.dtype
                or stats.mean_s.dtype != reference.dtype
                or stats.weight.dtype != reference.dtype
                or stats.holo_re.device != reference.device
                or stats.holo_im.device != reference.device
                or stats.mean_c.device != reference.device
                or stats.mean_s.device != reference.device
                or stats.weight.device != reference.device
            ):
                raise ValueError(
                    "Los tensores de mercado del holograma persistente no conservan su contrato"
                )
            check_finite(stats.holo_re, "El campo real del holograma persistente")
            check_finite(stats.holo_im, "El campo imaginario del holograma persistente")
            check_finite(stats.mean_c, "El fasor coseno medio del holograma persistente")
            check_finite(stats.mean_s, "El fasor seno medio del holograma persistente")
            check_finite(stats.weight, "La masa efectiva del holograma persistente")
            require_true(
                (stats.weight > 0).all() & (stats.weight <= 1.0 + 1e-6).all(),
                "La masa efectiva del holograma persistente debe pertenecer a (0, 1]",
            )
        for market, entry in state.pending.items():
            if (
                type(market) is not str
                or not market
                or not isinstance(entry, PendingSessionHologram)
                or type(entry.decision_at) is not int
                or not isinstance(entry.holo_re, Tensor)
                or not isinstance(entry.holo_im, Tensor)
                or not isinstance(entry.mean_c, Tensor)
                or not isinstance(entry.mean_s, Tensor)
                or entry.holo_re.shape != (bands, dc, dv)
                or entry.holo_im.shape != (bands, dc, dv)
                or entry.mean_c.shape != (bands, dc)
                or entry.mean_s.shape != (bands, dc)
                or entry.holo_re.dtype != reference.dtype
                or entry.holo_im.dtype != reference.dtype
                or entry.mean_c.dtype != reference.dtype
                or entry.mean_s.dtype != reference.dtype
                or entry.holo_re.device != reference.device
                or entry.holo_im.device != reference.device
                or entry.mean_c.device != reference.device
                or entry.mean_s.device != reference.device
            ):
                raise ValueError(
                    "El holograma pendiente de sesión no conserva su contrato de forma y tipo"
                )
            check_finite(entry.holo_re, "El campo real pendiente del holograma persistente")
            check_finite(entry.holo_im, "El campo imaginario pendiente del holograma persistente")
            check_finite(entry.mean_c, "El fasor coseno pendiente del holograma persistente")
            check_finite(entry.mean_s, "El fasor seno pendiente del holograma persistente")

    def _shuffle_permutation(self, market: str, instant: int, count: int) -> Tensor:
        text = (
            f"{PERSISTENT_SESSION_HOLOGRAM}/shuffle/"
            f"{self.seed}/{self.config.seed}/{market}/{instant}"
        )
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(count, generator=generator)

    def _groups(
        self,
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: PersistentSessionHologramState,
    ) -> list[tuple[str, int, list[int]]]:
        """Agrupar por mercado validando una sola sesión estrictamente posterior por mercado."""
        by_market: dict[str, tuple[int, list[int]]] = {}
        for row, (market, at) in enumerate(zip(markets, prediction_at, strict=True)):
            if type(market) is not str or not market:
                raise ValueError("Cada fila del holograma persistente necesita un mercado válido")
            instant = int(at)
            entry = by_market.get(market)
            if entry is None:
                if market in state.pending:
                    raise ValueError("La sesión anterior del mercado sigue pendiente de confirmar")
                previous = state.committed_at.get(market)
                if previous is not None and instant <= previous:
                    raise ValueError(
                        "La sesión ya se confirmó o no es posterior a la memoria del mercado"
                    )
                by_market[market] = (instant, [row])
            else:
                if entry[0] != instant:
                    raise ValueError(
                        "Las filas de un mercado en un bloque deben pertenecer a la misma sesión"
                    )
                entry[1].append(row)
        return [(market, instant, rows) for market, (instant, rows) in by_market.items()]

    def _session_step(
        self,
        market: str,
        instant: int,
        c_theta: Tensor,
        s_theta: Tensor,
        c_phi: Tensor,
        s_phi: Tensor,
        values: Tensor,
        density_logits: Tensor,
        self_re: Tensor,
        self_im: Tensor,
        market_stats: MarketHologramState | None,
    ) -> tuple[Tensor, PendingSessionHologram]:
        """Residuo proyectado [N, H] (antes de `gate`) y holograma centrado pendiente."""
        rows = len(values)
        bands, dc, dv = self.config.bands, self.config.phase_dim, self.band_dim
        inv_dc = 1.0 / float(dc)

        if rows == 1:
            zero_anchor = (
                values.sum() + density_logits.sum() + self_re.sum() + self_im.sum()
            ) * 0.0
            holo_re = values.new_zeros((bands, dc, dv)) + zero_anchor
            holo_im = values.new_zeros((bands, dc, dv)) + zero_anchor
            bar_c = c_theta.squeeze(0)
            bar_s = s_theta.squeeze(0)
            z_inst = values.new_zeros((1, self.readout_dim)) + zero_anchor
        else:
            pi = torch.softmax(density_logits, dim=0)
            weighted = pi.unsqueeze(-1) * values
            bar_v = weighted.sum(dim=0)
            v_centered = values - bar_v.unsqueeze(0)
            weighted_centered = pi.unsqueeze(-1) * v_centered
            holo_re = torch.einsum("rmk,rmd->mkd", c_theta, weighted_centered)
            holo_im = torch.einsum("rmk,rmd->mkd", s_theta, weighted_centered)
            bar_c = torch.einsum("rm,rmk->mk", pi, c_theta)
            bar_s = torch.einsum("rm,rmk->mk", pi, s_theta)

            unbound_re = (
                torch.einsum("rmk,mkd->rmd", c_phi, holo_re)
                + torch.einsum("rmk,mkd->rmd", s_phi, holo_im)
            ) * inv_dc
            unbound_im = (
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
            rho_re = loo_scale * (align_re - pi * self_re)
            rho_im = loo_scale * (align_im - pi * self_im)
            delta_re = loo_scale.unsqueeze(-1) * (
                unbound_re - (self_re - rho_re).unsqueeze(-1) * weighted_centered
            )
            delta_im = loo_scale.unsqueeze(-1) * (
                unbound_im - (self_im - rho_im).unsqueeze(-1) * weighted_centered
            )
            z_inst = torch.cat(
                (
                    delta_re.reshape(rows, self.hidden_size),
                    delta_im.reshape(rows, self.hidden_size),
                    rho_re,
                    rho_im,
                ),
                dim=-1,
            )

        u_inst = self.out_inst(self.inst_norm(z_inst))
        pending_entry = PendingSessionHologram(instant, holo_re, holo_im, bar_c, bar_s)

        if self.config.mode == INSTANT_ONLY or market_stats is None:
            return u_inst, pending_entry

        scales = len(self.config.half_lives)
        inv_w = 1.0 / market_stats.weight
        hat_re = market_stats.holo_re * inv_w[:, None, None, None]
        hat_im = market_stats.holo_im * inv_w[:, None, None, None]
        hat_c = market_stats.mean_c * inv_w[:, None, None]
        hat_s = market_stats.mean_s * inv_w[:, None, None]

        mem_re = (
            torch.einsum("rmk,lmkd->rlmd", c_phi, hat_re)
            + torch.einsum("rmk,lmkd->rlmd", s_phi, hat_im)
        ) * inv_dc
        mem_im = (
            torch.einsum("rmk,lmkd->rlmd", c_phi, hat_im)
            - torch.einsum("rmk,lmkd->rlmd", s_phi, hat_re)
        ) * inv_dc
        rho_mem_re = (
            torch.einsum("rmk,lmk->rlm", c_phi, hat_c) + torch.einsum("rmk,lmk->rlm", s_phi, hat_s)
        ) * inv_dc
        rho_mem_im = (
            torch.einsum("rmk,lmk->rlm", c_phi, hat_s) - torch.einsum("rmk,lmk->rlm", s_phi, hat_c)
        ) * inv_dc
        z_mem = torch.cat(
            (
                mem_re.reshape(rows, scales, self.hidden_size),
                mem_im.reshape(rows, scales, self.hidden_size),
                rho_mem_re,
                rho_mem_im,
            ),
            dim=-1,
        )
        if self.config.mode == SHUFFLED_HISTORY and rows > 1:
            perm = self._shuffle_permutation(market, instant, rows).to(values.device)
            z_mem = z_mem.index_select(0, perm)
        z_rot = z_inst.unsqueeze(1) - z_mem

        u_mem = self.out_mem(self.mem_norm(z_mem))
        u_rot = self.out_rot(self.rot_norm(z_rot))
        y_mem = (u_mem * self.scale_mem.unsqueeze(0)).sum(dim=1)
        y_rot = (u_rot * self.scale_rot.unsqueeze(0)).sum(dim=1)
        return u_inst + y_mem + y_rot, pending_entry

    def forward(
        self,
        encoded: Tensor,
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: PersistentSessionHologramState,
    ) -> tuple[Tensor, PersistentSessionHologramState]:
        """Leer campo instantáneo y memoria confirmada previa, dejando la sesión pendiente."""
        reference = self._reference()
        if (
            not isinstance(encoded, Tensor)
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
                "El holograma persistente necesita representación, mercado e instante por fila"
            )
        if not isinstance(state, PersistentSessionHologramState):
            raise ValueError("El holograma persistente necesita su estado identificado")
        self.validate_state(state, state.parameter_id)
        check_finite(encoded, "La representación de entrada al holograma persistente")

        rows = len(encoded)
        bands, dc, dv = self.config.bands, self.config.phase_dim, self.band_dim
        phase_width = bands * dc
        sessions = self._groups(markets, prediction_at, state)

        normed = self.norm(encoded)
        projected = self.proj(normed)
        theta = projected[:, :phase_width].reshape(rows, bands, dc)
        phi = projected[:, phase_width : 2 * phase_width].reshape(rows, bands, dc)
        values = projected[:, 2 * phase_width :].reshape(rows, bands, dv)
        c_theta, s_theta = torch.cos(theta), torch.sin(theta)
        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        inv_dc = 1.0 / float(dc)
        self_re = (c_phi * c_theta + s_phi * s_theta).sum(dim=-1) * inv_dc
        self_im = (c_phi * s_theta - s_phi * c_theta).sum(dim=-1) * inv_dc

        pending = dict(state.pending)
        if self.config.mode == SELF_ONLY:
            z_self = torch.cat(
                (
                    (self_re.unsqueeze(-1) * values).reshape(rows, self.hidden_size),
                    (self_im.unsqueeze(-1) * values).reshape(rows, self.hidden_size),
                    self_re,
                    self_im,
                ),
                dim=-1,
            )
            residual = self.out_inst(self.inst_norm(z_self))
            zero_holo = encoded.new_zeros((bands, dc, dv))
            zero_phasor = encoded.new_zeros((bands, dc))
            for market, instant, _ in sessions:
                pending[market] = PendingSessionHologram(
                    instant, zero_holo, zero_holo, zero_phasor, zero_phasor
                )
        else:
            density_logits = DENSITY_BOUND * torch.tanh(self.density(normed))
            if len(sessions) == 1:
                market, instant, _ = sessions[0]
                residual, entry = self._session_step(
                    market,
                    instant,
                    c_theta,
                    s_theta,
                    c_phi,
                    s_phi,
                    values,
                    density_logits,
                    self_re,
                    self_im,
                    state.markets.get(market),
                )
                pending[market] = entry
            else:
                residual = encoded.new_zeros((rows, self.hidden_size))
                for market, instant, members in sessions:
                    if encoded.is_cuda:
                        index = torch.tensor(members, dtype=torch.int64, pin_memory=True).to(
                            device=encoded.device, non_blocking=True
                        )
                    else:
                        index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
                    part, entry = self._session_step(
                        market,
                        instant,
                        c_theta.index_select(0, index),
                        s_theta.index_select(0, index),
                        c_phi.index_select(0, index),
                        s_phi.index_select(0, index),
                        values.index_select(0, index),
                        density_logits.index_select(0, index),
                        self_re.index_select(0, index),
                        self_im.index_select(0, index),
                        state.markets.get(market),
                    )
                    residual = residual.index_copy(0, index, part)
                    pending[market] = entry

        check_finite(residual, "La proyección del holograma persistente")
        result = encoded + self.gate * residual
        check_finite(result, "La representación con holograma persistente")
        return result, replace(state, pending=pending)

    def commit(
        self,
        state: PersistentSessionHologramState,
        *,
        differentiable: bool = False,
    ) -> PersistentSessionHologramState:
        """Confirmar las sesiones pendientes al cierre, actualizando el campo persistente."""
        check_differentiable(differentiable)
        reference = self._reference()
        self.validate_state(state, state.parameter_id)
        if not state.pending:
            return state
        markets = dict(state.markets)
        committed_at = dict(state.committed_at)
        decays = torch.tensor(self.config.decays, dtype=reference.dtype, device=reference.device)
        one_minus = 1.0 - decays
        scales = len(self.config.half_lives)
        bands, dc, dv = self.config.bands, self.config.phase_dim, self.band_dim
        writes_memory = self.config.mode in (PERSISTENT, SHUFFLED_HISTORY)

        for market, entry in sorted(state.pending.items()):
            committed_at[market] = entry.decision_at
            if not writes_memory:
                continue
            h_re = entry.holo_re if differentiable else entry.holo_re.detach()
            h_im = entry.holo_im if differentiable else entry.holo_im.detach()
            m_c = entry.mean_c if differentiable else entry.mean_c.detach()
            m_s = entry.mean_s if differentiable else entry.mean_s.detach()
            previous = markets.get(market)
            if previous is None:
                prev_re = torch.zeros(
                    (scales, bands, dc, dv), dtype=reference.dtype, device=reference.device
                )
                prev_im = torch.zeros(
                    (scales, bands, dc, dv), dtype=reference.dtype, device=reference.device
                )
                prev_c = torch.zeros(
                    (scales, bands, dc), dtype=reference.dtype, device=reference.device
                )
                prev_s = torch.zeros(
                    (scales, bands, dc), dtype=reference.dtype, device=reference.device
                )
                prev_w = torch.zeros((scales,), dtype=reference.dtype, device=reference.device)
                updates = 1
            else:
                prev_re = previous.holo_re if differentiable else previous.holo_re.detach()
                prev_im = previous.holo_im if differentiable else previous.holo_im.detach()
                prev_c = previous.mean_c if differentiable else previous.mean_c.detach()
                prev_s = previous.mean_s if differentiable else previous.mean_s.detach()
                prev_w = previous.weight if differentiable else previous.weight.detach()
                updates = previous.updates + 1
            new_re = decays[:, None, None, None] * prev_re + one_minus[
                :, None, None, None
            ] * h_re.unsqueeze(0)
            new_im = decays[:, None, None, None] * prev_im + one_minus[
                :, None, None, None
            ] * h_im.unsqueeze(0)
            new_c = decays[:, None, None] * prev_c + one_minus[:, None, None] * m_c.unsqueeze(0)
            new_s = decays[:, None, None] * prev_s + one_minus[:, None, None] * m_s.unsqueeze(0)
            new_w = decays * prev_w + one_minus
            if not differentiable:
                new_re = new_re.detach()
                new_im = new_im.detach()
                new_c = new_c.detach()
                new_s = new_s.detach()
                new_w = new_w.detach()
            check_finite(new_re, "La actualización real del holograma persistente")
            check_finite(new_im, "La actualización imaginaria del holograma persistente")
            check_finite(new_c, "La actualización coseno del holograma persistente")
            check_finite(new_s, "La actualización seno del holograma persistente")
            check_finite(new_w, "La actualización de masa del holograma persistente")
            markets[market] = MarketHologramState(
                new_re, new_im, new_c, new_s, new_w, entry.decision_at, updates
            )
        return replace(state, markets=markets, pending={}, committed_at=committed_at)

    def detach_state(
        self, state: PersistentSessionHologramState, parameter_id: str | None = None
    ) -> PersistentSessionHologramState:
        """Cortar el grafo en las fronteras de truncamiento con todas las sesiones confirmadas."""
        if state.pending:
            raise ValueError(
                "Solo se corta el grafo del holograma persistente con todas las sesiones "
                "confirmadas"
            )
        markets = {
            market: MarketHologramState(
                stats.holo_re.detach(),
                stats.holo_im.detach(),
                stats.mean_c.detach(),
                stats.mean_s.detach(),
                stats.weight.detach(),
                stats.committed_at,
                stats.updates,
            )
            for market, stats in state.markets.items()
        }
        return replace(
            state,
            parameter_id=parameter_id or state.parameter_id,
            markets=markets,
            pending={},
            committed_at=dict(state.committed_at),
        )

    def export_state(self, state: PersistentSessionHologramState) -> dict:
        """Exportar copias CPU sin grafo, exigiendo todas las sesiones ya confirmadas."""
        detached = self.detach_state(state)
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            committed_at={market: at for market, at in sorted(detached.committed_at.items())},
            markets={
                market: dict(
                    holo_re=stats.holo_re.cpu().clone(),
                    holo_im=stats.holo_im.cpu().clone(),
                    mean_c=stats.mean_c.cpu().clone(),
                    mean_s=stats.mean_s.cpu().clone(),
                    weight=stats.weight.cpu().clone(),
                    committed_at=stats.committed_at,
                    updates=stats.updates,
                )
                for market, stats in sorted(detached.markets.items())
            },
        )

    def restore_state(
        self, payload: object, parameter_id: str, *, device: torch.device | str
    ) -> PersistentSessionHologramState:
        """Validar y restaurar el estado exportado sobre `device`."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "configuration", "committed_at", "markets"}
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.config.identity())
            or not isinstance(payload["committed_at"], dict)
            or not isinstance(payload["markets"], dict)
        ):
            raise ValueError(
                "El estado guardado del holograma persistente no corresponde al predictor"
            )
        reference = self._reference()
        target_device = torch.device(device)
        committed_at = {}
        for market, at in payload["committed_at"].items():
            if type(market) is not str or not market or type(at) is not int:
                raise ValueError("El cursor guardado del holograma persistente no es válido")
            committed_at[market] = at
        markets = {}
        expected_fields = {
            "holo_re",
            "holo_im",
            "mean_c",
            "mean_s",
            "weight",
            "committed_at",
            "updates",
        }
        for market, entry in payload["markets"].items():
            if not isinstance(entry, dict) or set(entry) != expected_fields:
                raise ValueError(
                    "Los tensores de mercado guardados del holograma persistente están incompletos"
                )
            markets[market] = MarketHologramState(
                entry["holo_re"].to(device=target_device, dtype=reference.dtype, copy=True),
                entry["holo_im"].to(device=target_device, dtype=reference.dtype, copy=True),
                entry["mean_c"].to(device=target_device, dtype=reference.dtype, copy=True),
                entry["mean_s"].to(device=target_device, dtype=reference.dtype, copy=True),
                entry["weight"].to(device=target_device, dtype=reference.dtype, copy=True),
                int(entry["committed_at"]),
                int(entry["updates"]),
            )
        state = PersistentSessionHologramState(
            self.fingerprint(), parameter_id, markets, {}, committed_at
        )
        self.validate_state(state, parameter_id)
        return state
