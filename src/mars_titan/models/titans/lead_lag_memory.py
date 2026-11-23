"""Memoria rápida de desfase dinámico entre grupos de activos `lead_lag_fast_memory_v1` (LL-FM).

Complementa las memorias rápidas no supervisadas por activo de Titans-MAC con una memoria rápida
de contagio direccional entre grupos latentes de activos por mercado, actualizada estrictamente
cuando maduran las etiquetas de sesiones pasadas (`decision_at < target_available_at <= event.at`).

Diseño orientado por la evidencia empírica de la campaña real (donde CRRM perdió −0,0036 de IC en
ajuste al regresar el rango maduro propio de cada activo sobre `encoded`):

1. **Aislamiento del tronco (stop-gradient de entrada)**: las asignaciones de grupo `p_{i,t}` y la
   receptividad continua `κ_{i,t}` leen `LayerNorm(sg(e_{i,t}))`. Así los metagradientes diferidos
   aprenden los parámetros de LL-FM sin distorsionar las representaciones de `TitansMAC` ni de
   `technical_branch`.
2. **Innovación transversal ortogonalizada** (sin regresar el rango propio ya explicado por los
   rasgos): al madurar la sesión `s` con rangos normales gaussianos `g_s` y predicción emitida
   estandarizada `ĝ_s`, se proyecta fuera el componente ya explicado por el tronco:

       β_s = ⟨g_s, ĝ_s⟩ / (‖ĝ_s‖_2^2 + ε_proj),     r_s^⊥ = g_s − β_s ĝ_s

   A diferencia de `g_s − ĝ_s` (correlación −0,67 con `ĝ_s` cuando IC ≈ 0,11), `r_s^⊥` es
   estrictamente ortogonal a `ĝ_s` y coincide suavemente con `g_s` cuando la cabeza arranca en cero.
3. **Contagio estrictamente entre grupos distintos (`diag(T) = 0`)**: cada activo se asigna a `G`
   grupos latentes blandos con logits centrados por sesión y acotados:

       p_{i,t} = softmax(B · tanh((z_{i,t} − z̄_{m,t}) / B)) ∈ Δ^G
       R_{s,a} = (Σ_{j=1}^{N_s} p_{j,s,a} r_{j,s}^⊥) / (Σ_{j=1}^{N_s} p_{j,s,a} + ε_c)

   La transición efectiva por horizonte `h` separa una matriz estructural aprendida `T_{0,h}` y una
   desviación rápida `ΔT_{m,h}` con diagonal exactamente nula (`M_off = 11ᵀ − I_G`):

       T_{m,h}^{eff} = (T_{0,h} + ΔT_{m,h}) ⊙ M_off

   Ningún grupo puede predecirse a sí mismo desde su propio rango pasado.
4. **Regla delta normalizada (NLMS) con olvido anclado y trazas de desfase multiescala**:
   para horizontes de desfase `τ_h` (p. ej. 1 y 5 sesiones, `λ_h = 0` si `τ_h = 1`, si no
   `exp(−1 / τ_h)`), se mantiene la traza causal de choques líderes `Z_{m,h} ∈ R^G` y se actualiza
   `ΔT_{m,h}` por filas mediante regla delta normalizada contractiva (inmune a explosiones en FP32
   y FP64) anclada hacia `T_{0,h}`:

       e_{s,h} = R_s − T_{m,h}^{eff} Z_{m,h}
       d_{h,b} = (‖Z_{m,h}‖_2^2 − Z_{m,h,b}^2)_+ + γ_T
       ΔT_{m,h} ← (1 − α_h) · (ΔT_{m,h} + (η_h / d_h) ⊙ (e_{s,h} Z_{m,h}ᵀ)) ⊙ M_off
       Z_{m,h} ← λ_h Z_{m,h} + (1 − λ_h) R_s

5. **Lectura rica de contagio direccional y flujo neto antisimétrico**:
   en el instante `t`, cada activo lee tanto el contagio dirigido fuera de diagonal
   `s_off = tanh(T^{eff} Z_m)` como el flujo neto antisimétrico
   `s_asym = tanh(½ (T^{eff} − (T^{eff})ᵀ) Z_m)` modulado por su receptividad continua
   `κ_{i,t} = σ(W_κ LN(sg(e_{i,t})) + b_κ)`, centrado transversalmente en la sesión y sumado
   mediante una ganancia vectorial aprendida `g` inicializada en cero sobre `W_out` Kaiming (sin
   doble cero en serie):

       f_{i,t,h} = [ (G p_{i,t}) ⊙ s_{off,h},  (G p_{i,t} ⊙ κ_{i,t}) ⊙ s_{asym,h} ]
       e'_{i,t} = e_{i,t} + g ⊙ (W_out (f_{i,t} − f̄_{m,t}))

Controles negativos integrados (mismos parámetros e inicialización bit a bit):
- `shuffled_targets`: permuta deterministamente `r_s^⊥` entre los activos de cada sesión madura con
  semilla causal derivada de `(seed, config.seed, mercado, decision_at)`.
- `static_prior`: actualiza las trazas `Z_{m,h}` con normalidad pero mantiene `ΔT_{m,h} = 0`,
  aislando la adaptación rápida en tiempo de prueba frente al prior estructural `T_0`.
- `eval_frozen`: actualiza en ajuste (`train=True`) y congela las escrituras en validación y
  evaluación (`train=False`), aislando cualquier efecto sobre el tronco durante el ajuste.
"""

import hashlib
import math
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn

from .config import LAYER_NORM_EPS, bounded_integer, canonical, require_identity
from .session_ic_objective import standardized
from .session_ranks import gaussian_ranks
from .state import check_differentiable, check_finite, require_true

LEAD_LAG_MEMORY = "lead_lag_fast_memory_v1"
LEAD_LAG = "lead_lag"
SHUFFLED_TARGETS = "shuffled_targets"
STATIC_PRIOR = "static_prior"
EVAL_FROZEN = "eval_frozen"
MODES = (LEAD_LAG, SHUFFLED_TARGETS, STATIC_PRIOR, EVAL_FROZEN)

MIN_GROUPS = 2
MAX_GROUPS = 64
MIN_LAG_SCALE = 1
MAX_LAG_SCALE = 63
DECAY_FLOOR = 1e-3
DECAY_CEILING = 0.5
MIN_DECAY = 0.005
MAX_DECAY = 0.49
LR_FLOOR = 1e-3
LR_CEILING = 0.95
MIN_LR = 0.005
MAX_LR = 0.9
MIN_DAMPING = 0.05
MAX_DAMPING = 100.0
LOGIT_BOUND = 4.0
GROUP_MASS_FLOOR = 1.0
PROJ_EPS = 1e-8
PRIOR_INIT_SCALE = 0.2


def _finite_float(value: object, low: float, high: float, message: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(message)
    return float(value)


def _bounded_sigmoid_inverse(value: float, low: float, high: float) -> float:
    """Inversa de `low + (high - low) * sigmoid(z)` para `low < value < high`."""
    ratio = (value - low) / (high - low)
    return math.log(ratio / (1.0 - ratio))


def _row_index(rows: list[int], device: torch.device) -> Tensor:
    if device.type != "cuda":
        return torch.tensor(rows, dtype=torch.int64, device=device)
    return torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
        device=device, non_blocking=True
    )


def _market(flow: str) -> str:
    if type(flow) is not str or "/" not in flow:
        raise ValueError("Cada flujo de LL-FM necesita su prefijo de mercado")
    return flow.split("/", 1)[0]


@dataclass(frozen=True)
class LeadLagMemoryConfig:
    """Contrato de la memoria rápida de desfase dinámico entre grupos de activos (LL-FM)."""

    groups: int = 16
    lag_scales: tuple[int, ...] = (1, 5)
    decay: float = 0.05
    learning_rate: float = 0.2
    damping: float = 1.0
    minimum_rows: int = 8
    mode: str = LEAD_LAG
    seed: int = 0

    def __post_init__(self) -> None:
        bounded_integer(self.groups, "número de grupos de LL-FM", MIN_GROUPS, MAX_GROUPS)
        scales = self.lag_scales
        scales = tuple(scales) if isinstance(scales, list) else scales
        if (
            type(scales) is not tuple
            or not 1 <= len(scales) <= 4
            or any(
                type(value) is not int or not MIN_LAG_SCALE <= value <= MAX_LAG_SCALE
                for value in scales
            )
            or list(scales) != sorted(set(scales))
        ):
            raise ValueError("lag_scales debe tener de una a cuatro escalas crecientes en [1, 63]")
        decay = _finite_float(
            self.decay,
            MIN_DECAY,
            MAX_DECAY,
            f"decay debe ser un número finito en [{MIN_DECAY}, {MAX_DECAY}]",
        )
        learning_rate = _finite_float(
            self.learning_rate,
            MIN_LR,
            MAX_LR,
            f"learning_rate debe ser un número finito en [{MIN_LR}, {MAX_LR}]",
        )
        damping = _finite_float(
            self.damping,
            MIN_DAMPING,
            MAX_DAMPING,
            f"damping debe ser un número finito en [{MIN_DAMPING}, {MAX_DAMPING}]",
        )
        bounded_integer(self.minimum_rows, "minimum_rows de LL-FM", 3, 4096)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de LL-FM debe ser uno de {MODES}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de LL-FM debe ser un entero de 32 bits")
        object.__setattr__(self, "lag_scales", scales)
        object.__setattr__(self, "decay", decay)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "damping", damping)

    @property
    def trace_decays(self) -> tuple[float, ...]:
        """Retención exponencial λ_h de cada traza de desfase (0 para desfase exacto 1)."""
        return tuple(0.0 if tau == 1 else math.exp(-1.0 / tau) for tau in self.lag_scales)

    def identity(self) -> dict:
        return dict(
            schema_version=1,
            kind=LEAD_LAG_MEMORY,
            groups=self.groups,
            lag_scales=list(self.lag_scales),
            decay=self.decay,
            learning_rate=self.learning_rate,
            damping=self.damping,
            minimum_rows=self.minimum_rows,
            mode=self.mode,
            seed=self.seed,
            decay_bounds=[DECAY_FLOOR, DECAY_CEILING],
            lr_bounds=[LR_FLOOR, LR_CEILING],
            logit_bound=LOGIT_BOUND,
            group_mass_floor=GROUP_MASS_FLOOR,
            projection_eps=PROJ_EPS,
            input_coupling="detached_encoded_stop_gradient",
            memberships="session_centered_bounded_tanh_softmax_latent_groups",
            targets="backbone_ols_orthogonalized_matured_session_gaussian_normal_ranks",
            transition="strictly_off_diagonal_learned_prior_plus_online_nlms_delta_fast_weight",
            readout="cross_section_centered_off_diagonal_and_receptor_gated_antisymmetric_tanh",
            output="residual_with_zero_initialized_vector_gate_over_kaiming_projection",
            position="after_technical_and_crrm_before_head",
            market_isolation="strict_per_market_transition_and_lag_trace",
            causality="updated_only_on_matured_labels_strictly_before_emission",
        )


def parse_lead_lag_memory(raw: object) -> LeadLagMemoryConfig | None:
    """Aceptar LeadLagMemoryConfig, su diccionario completo o None."""
    if raw is None:
        return None
    if type(raw) is LeadLagMemoryConfig:
        raw.__post_init__()
        return raw
    expected = {field.name for field in fields(LeadLagMemoryConfig)}
    if isinstance(raw, dict) and set(raw) == expected:
        return LeadLagMemoryConfig(**raw)
    raise ValueError(
        "lead_lag_memory debe declarar groups, lag_scales, decay, learning_rate, damping, "
        "minimum_rows, mode y seed, o ser None"
    )


@dataclass(frozen=True)
class MarketLeadLagState:
    """Estado rápido de desfase por mercado: desviación ΔT [H, G, G] y traza Z [H, G]."""

    delta_transition: Tensor
    trace: Tensor
    sessions: int
    updates: int


@dataclass(frozen=True)
class LeadLagMemoryState:
    """Estado de LL-FM: matrices rápidas por mercado y pertenencias emitidas pendientes."""

    config_id: str
    parameter_id: str
    markets: dict[str, MarketLeadLagState]
    # Clave: (flow, decision_at) -> (matriz de pertenencias de su emisión [filas, G], predicciones
    # emitidas de esa emisión [filas], fila). Guarda el tensor entero de la emisión para reunir una
    # sesión con una selección, con grafo O(1) en el número de filas y sin esperar a la GPU.
    pending: dict[tuple[str, int], tuple[Tensor, Tensor, int]]


def _gather_pending(entries: list[tuple[Tensor, Tensor, int]]) -> tuple[Tensor, Tensor]:
    """Reunir en orden las pertenencias [n, G] y predicciones emitidas [n] pendientes."""
    sources: dict[int, tuple[Tensor, Tensor, list[int], list[int]]] = {}
    for position, (probs, issued, row) in enumerate(entries):
        _, _, positions, picked = sources.setdefault(id(probs), (probs, issued, [], []))
        positions.append(position)
        picked.append(row)
    prob_parts, issued_parts, order = [], [], []
    for probs, issued, positions, picked in sources.values():
        index = _row_index(picked, probs.device)
        prob_parts.append(probs.index_select(0, index))
        issued_parts.append(issued.index_select(0, index))
        order.extend(positions)
    if len(prob_parts) == 1:
        return prob_parts[0], issued_parts[0]
    inverse = torch.empty(len(order), dtype=torch.int64)
    inverse[torch.tensor(order, dtype=torch.int64)] = torch.arange(len(order), dtype=torch.int64)
    inverse = inverse.to(prob_parts[0].device)
    return (
        torch.cat(prob_parts, dim=0).index_select(0, inverse),
        torch.cat(issued_parts, dim=0).index_select(0, inverse),
    )


class LeadLagMemory(nn.Module):
    """Memoria rápida de desfase dinámico entre grupos latentes de activos (LL-FM)."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        config: LeadLagMemoryConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión oculta de LL-FM", 1, 4096)
        bounded_integer(seed, "semilla de LL-FM", 0, 2**32 - 1)
        if type(config) is not LeadLagMemoryConfig:
            raise ValueError("LL-FM necesita LeadLagMemoryConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("LL-FM admite precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        scales, groups = len(config.lag_scales), config.groups
        self.readout_dim = 2 * scales * groups
        # Semilla aislada e independiente del modo: candidato y controles arrancan idénticos.
        derived = hashlib.sha256(f"{LEAD_LAG_MEMORY}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.group_proj = nn.Linear(hidden_size, groups, bias=False, dtype=dtype)
            self.receptor_proj = nn.Linear(hidden_size, groups, bias=True, dtype=dtype)
            prior_std = PRIOR_INIT_SCALE / math.sqrt(float(groups - 1))
            prior_init = torch.randn((scales, groups, groups), dtype=dtype) * prior_std
            off_diag = 1.0 - torch.eye(groups, dtype=dtype)
            prior_init = prior_init * off_diag.unsqueeze(0)
            self.prior_transition = nn.Parameter(prior_init)
            self.out = nn.Linear(self.readout_dim, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.receptor_proj.bias)
        decay_logit = _bounded_sigmoid_inverse(config.decay, DECAY_FLOOR, DECAY_CEILING)
        lr_logit = _bounded_sigmoid_inverse(config.learning_rate, LR_FLOOR, LR_CEILING)
        self.decay_logits = nn.Parameter(torch.full((scales,), decay_logit, dtype=dtype))
        self.lr_logits = nn.Parameter(torch.full((scales,), lr_logit, dtype=dtype))
        # Puerta vectorial en cero sobre W_out Kaiming (no nula): sin doble cero en serie.
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self._fingerprint = hashlib.sha256(canonical(config.identity()).encode()).hexdigest()

    def fingerprint(self) -> str:
        return self._fingerprint

    def get_extra_state(self) -> dict:
        return dict(configuration=self.config.identity(), hidden_size=self.hidden_size)

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _reference(self) -> Tensor:
        reference = self.gate
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("LL-FM admite precisión FP32 o FP64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("LL-FM exige FP32 o FP64 explícitos, sin autocast")
        return reference

    def _off_diag(self, reference: Tensor) -> Tensor:
        groups = self.config.groups
        return 1.0 - torch.eye(groups, dtype=reference.dtype, device=reference.device)

    def _effective_decay(self) -> Tensor:
        return DECAY_FLOOR + (DECAY_CEILING - DECAY_FLOOR) * torch.sigmoid(self.decay_logits)

    def _effective_lr(self) -> Tensor:
        return LR_FLOOR + (LR_CEILING - LR_FLOOR) * torch.sigmoid(self.lr_logits)

    def initial_state(self, parameter_id: str) -> LeadLagMemoryState:
        return LeadLagMemoryState(self.fingerprint(), parameter_id, {}, {})

    def validate_state(self, state: object, parameter_id: str) -> None:
        reference = self._reference()
        scales, groups = len(self.config.lag_scales), self.config.groups
        if (
            not isinstance(state, LeadLagMemoryState)
            or state.config_id != self.fingerprint()
            or state.parameter_id != parameter_id
            or not isinstance(state.markets, dict)
            or not isinstance(state.pending, dict)
        ):
            raise ValueError("El estado de memoria rápida de desfase no corresponde al módulo")
        for market, stats in state.markets.items():
            if (
                type(market) is not str
                or not market
                or not isinstance(stats, MarketLeadLagState)
                or type(stats.sessions) is not int
                or stats.sessions < 1
                or type(stats.updates) is not int
                or stats.updates < 0
                or stats.updates >= stats.sessions
                or not isinstance(stats.delta_transition, Tensor)
                or not isinstance(stats.trace, Tensor)
                or stats.delta_transition.shape != (scales, groups, groups)
                or stats.trace.shape != (scales, groups)
                or stats.delta_transition.dtype != reference.dtype
                or stats.trace.dtype != reference.dtype
                or stats.delta_transition.device != reference.device
                or stats.trace.device != reference.device
            ):
                raise ValueError("Los estadísticos de mercado de LL-FM no conservan su contrato")
            check_finite(stats.delta_transition, "La matriz rápida de transición de LL-FM")
            check_finite(stats.trace, "La traza de choques de grupos de LL-FM")
            diag = torch.diagonal(stats.delta_transition, dim1=-2, dim2=-1)
            require_true(
                (diag == 0).all(),
                "La diagonal de la transición de LL-FM debe permanecer estrictamente en cero",
            )

    def memberships(
        self,
        encoded: Tensor,
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
    ) -> Tensor:
        """Pertenencias blandas a grupos latentes [N, G] centradas por sesión y mercado."""
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
            raise ValueError("LL-FM necesita una representación, un mercado y un instante por fila")
        check_finite(encoded, "La representación de entrada de LL-FM")
        # Stop-gradient en `encoded`: protege al tronco frente a metagradientes ruidosos diferidos.
        normed = self.norm(encoded.detach())
        raw = self.group_proj(normed)
        sessions: dict[tuple[str, int], list[int]] = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        if len(sessions) == 1:
            centered = raw - raw.mean(dim=0, keepdim=True) if len(raw) > 1 else raw
            bounded = LOGIT_BOUND * torch.tanh(centered / LOGIT_BOUND)
            probs = torch.softmax(bounded, dim=-1)
        else:
            probs = torch.zeros_like(raw)
            for members in sessions.values():
                index = _row_index(members, raw.device)
                part = raw.index_select(0, index)
                if len(members) > 1:
                    part = part - part.mean(dim=0, keepdim=True)
                bounded = LOGIT_BOUND * torch.tanh(part / LOGIT_BOUND)
                probs = probs.index_copy(0, index, torch.softmax(bounded, dim=-1))
        check_finite(probs, "Las pertenencias a grupos de LL-FM")
        return probs

    def read_features(
        self,
        encoded: Tensor,
        probs: Tensor,
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: LeadLagMemoryState,
    ) -> Tensor:
        """Rasgos de contagio inter-grupo centrados por sesión [N, 2*H*G] antes de `out`."""
        reference = self._reference()
        groups, scales = self.config.groups, len(self.config.lag_scales)
        if (
            not isinstance(encoded, Tensor)
            or encoded.layout != torch.strided
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or not isinstance(probs, Tensor)
            or probs.layout != torch.strided
            or probs.ndim != 2
            or probs.shape != (len(encoded), groups)
            or probs.dtype != reference.dtype
            or probs.device != reference.device
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError(
                "La lectura de LL-FM necesita representación, pertenencias [filas, G], mercado "
                "e instante por fila"
            )
        check_finite(encoded, "La representación de consulta de LL-FM")
        check_finite(probs, "Las pertenencias de consulta de LL-FM")
        features = probs.new_zeros((len(probs), self.readout_dim))
        if not state.markets:
            return features
        off_diag = self._off_diag(reference).unsqueeze(0)
        kappa = torch.sigmoid(self.receptor_proj(self.norm(encoded.detach())))
        sessions: dict[tuple[str, int], list[int]] = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        group_scale = float(groups)
        for (market, _instant), members in sessions.items():
            stats = state.markets.get(market)
            if stats is None:
                continue
            contiguous = len(members) == len(probs)
            if contiguous:
                p_m, k_m = probs, kappa
            else:
                index = _row_index(members, probs.device)
                p_m = probs.index_select(0, index)
                k_m = kappa.index_select(0, index)
            t_eff = (self.prior_transition + stats.delta_transition) * off_diag
            t_asym = 0.5 * (t_eff - t_eff.transpose(-1, -2))
            s_off = torch.tanh(torch.einsum("hgk,hk->hg", t_eff, stats.trace))
            s_asym = torch.tanh(torch.einsum("hgk,hk->hg", t_asym, stats.trace))
            scaled_p = group_scale * p_m
            ch_off = (scaled_p[:, None, :] * s_off[None, :, :]).reshape(
                len(members), scales * groups
            )
            ch_asym = ((scaled_p * k_m)[:, None, :] * s_asym[None, :, :]).reshape(
                len(members), scales * groups
            )
            part = torch.cat((ch_off, ch_asym), dim=-1)
            if len(members) > 1:
                part = part - part.mean(dim=0, keepdim=True)
            if contiguous:
                features = part
            else:
                features = features.index_copy(0, index, part)
        check_finite(features, "Los rasgos de contagio de LL-FM")
        return features

    def read(
        self,
        encoded: Tensor,
        probs: Tensor,
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: LeadLagMemoryState,
    ) -> Tensor:
        """Residuo [N, H] sumado a la representación antes de la cabeza."""
        features = self.read_features(encoded, probs, markets, prediction_at, state)
        delta = self.gate * self.out(features)
        check_finite(delta, "El residuo de la memoria rápida de desfase")
        return delta

    def stage(
        self,
        state: LeadLagMemoryState,
        flow_ids: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        memberships: Tensor,
        issued_predictions: Tensor,
        *,
        differentiable: bool = False,
    ) -> LeadLagMemoryState:
        """Registrar pertenencias emitidas y predicciones puntuales a la espera de maduración."""
        check_differentiable(differentiable)
        reference = self._reference()
        if (
            not isinstance(memberships, Tensor)
            or memberships.ndim != 2
            or memberships.shape != (len(flow_ids), self.config.groups)
            or memberships.dtype != reference.dtype
            or memberships.device != reference.device
            or not isinstance(issued_predictions, Tensor)
            or issued_predictions.shape != (len(flow_ids),)
            or len(prediction_at) != len(flow_ids)
        ):
            raise ValueError(
                "El registro pendiente de LL-FM necesita pertenencias y predicciones por fila"
            )
        check_finite(memberships, "Las pertenencias pendientes de LL-FM")
        check_finite(issued_predictions, "Las predicciones emitidas para LL-FM")
        stored_probs = memberships if differentiable else memberships.detach()
        issued = issued_predictions.detach().to(device=reference.device, dtype=reference.dtype)
        pending = dict(state.pending)
        for row, (flow, at) in enumerate(zip(flow_ids, prediction_at, strict=True)):
            entry_key = (flow, int(at))
            if entry_key in pending:
                raise ValueError("LL-FM ya tiene registrada esa emisión pendiente")
            pending[entry_key] = (stored_probs, issued, row)
        return replace(state, pending=pending)

    def _shuffle_permutation(self, market: str, decision_at: int, count: int) -> Tensor:
        text = f"{LEAD_LAG_MEMORY}/shuffle/{self.seed}/{self.config.seed}/{market}/{decision_at}"
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(count, generator=generator)

    def settle(
        self,
        state: LeadLagMemoryState,
        matured: list[tuple[str, int, float]] | tuple[tuple[str, int, float], ...],
        *,
        differentiable: bool = False,
        train: bool = True,
    ) -> LeadLagMemoryState:
        """Actualizar las transiciones rápidas de desfase con etiquetas ya maduras."""
        check_differentiable(differentiable)
        if type(train) is not bool:
            raise ValueError("train debe ser booleano")
        if not matured:
            return state
        reference = self._reference()
        pending = dict(state.pending)
        grouped: dict[tuple[int, str], list[tuple[str, Tensor, Tensor, int, float]]] = {}
        for flow, decision_at, outcome in matured:
            if (
                isinstance(outcome, bool)
                or not isinstance(outcome, (int, float))
                or not math.isfinite(outcome)
            ):
                raise ValueError("El resultado maduro para LL-FM debe ser un real finito")
            entry_key = (flow, int(decision_at))
            if entry_key not in pending:
                raise ValueError("La etiqueta madura de LL-FM no tiene emisión pendiente")
            grouped.setdefault((int(decision_at), _market(flow)), []).append(
                (flow, *pending.pop(entry_key), float(outcome))
            )
        if self.config.mode == EVAL_FROZEN and not train:
            return replace(state, pending=pending)

        markets = dict(state.markets)
        scales, groups = len(self.config.lag_scales), self.config.groups
        off_diag = self._off_diag(reference).unsqueeze(0)
        trace_decays = torch.tensor(
            self.config.trace_decays, dtype=reference.dtype, device=reference.device
        ).unsqueeze(-1)
        one_minus_trace = 1.0 - trace_decays
        alpha = self._effective_decay()
        eta = self._effective_lr()
        prior = self.prior_transition if differentiable else self.prior_transition.detach()
        damping = float(self.config.damping)

        for decision_at, market in sorted(grouped):
            rows = sorted(grouped[decision_at, market], key=lambda item: item[0])
            if len(rows) < self.config.minimum_rows:
                continue
            p_mat, issued = _gather_pending([item[1:4] for item in rows])
            if not differentiable:
                p_mat = p_mat.detach()
            outcomes = torch.tensor(
                [item[4] for item in rows], dtype=reference.dtype, device=reference.device
            )
            present = torch.ones((len(rows), 1), dtype=torch.bool, device=reference.device)
            normal_ranks = gaussian_ranks(outcomes.unsqueeze(-1), present).squeeze(-1)
            issued_std = standardized(issued)
            # Proyección OLS exacta: innovación ortogonal a la predicción del tronco.
            beta = (normal_ranks * issued_std).sum() / (issued_std.square().sum() + PROJ_EPS)
            innovation = normal_ranks - beta * issued_std
            if self.config.mode == SHUFFLED_TARGETS:
                perm = self._shuffle_permutation(market, decision_at, len(rows)).to(
                    reference.device
                )
                innovation = innovation.index_select(0, perm)
            mass = p_mat.sum(dim=0) + GROUP_MASS_FLOOR
            group_r = (p_mat.transpose(0, 1) @ innovation) / mass
            check_finite(group_r, "Los rendimientos de innovación de grupo de LL-FM")

            previous = markets.get(market)
            if previous is None:
                new_delta = torch.zeros(
                    (scales, groups, groups), dtype=reference.dtype, device=reference.device
                )
                new_trace = group_r.unsqueeze(0).expand(scales, groups).clone()
                sessions = 1
                updates = 0
            else:
                prev_delta = (
                    previous.delta_transition
                    if differentiable
                    else previous.delta_transition.detach()
                )
                prev_trace = previous.trace if differentiable else previous.trace.detach()
                if self.config.mode == STATIC_PRIOR:
                    new_delta = prev_delta
                    updates = previous.updates
                else:
                    t_eff = (prior + prev_delta) * off_diag
                    pred_r = torch.einsum("hgk,hk->hg", t_eff, prev_trace)
                    err = group_r.unsqueeze(0) - pred_r
                    # Energía por fila excluyendo la diagonal b: contractividad exacta por fila.
                    total_sq = prev_trace.square().sum(dim=-1, keepdim=True)
                    denom = (total_sq - prev_trace.square()).clamp_min(0.0) + damping
                    step = (eta[:, None] / denom).unsqueeze(-1) * (
                        err.unsqueeze(-1) * prev_trace.unsqueeze(-2)
                    )
                    new_delta = (1.0 - alpha[:, None, None]) * (prev_delta + step) * off_diag
                    updates = previous.updates + 1
                new_trace = trace_decays * prev_trace + one_minus_trace * group_r.unsqueeze(0)
                sessions = previous.sessions + 1
            if not differentiable:
                new_delta = new_delta.detach()
                new_trace = new_trace.detach()
            check_finite(new_delta, "La actualización de transición de LL-FM")
            check_finite(new_trace, "La actualización de traza de LL-FM")
            markets[market] = MarketLeadLagState(new_delta, new_trace, sessions, updates)
        return replace(state, markets=markets, pending=pending)

    def detach_state(
        self, state: LeadLagMemoryState, parameter_id: str | None = None
    ) -> LeadLagMemoryState:
        """Cortar el grafo de autograd en las fronteras de truncamiento y compactar pendientes."""
        markets = {
            market: MarketLeadLagState(
                stats.delta_transition.detach(),
                stats.trace.detach(),
                stats.sessions,
                stats.updates,
            )
            for market, stats in state.markets.items()
        }
        remaining: dict[int, tuple[Tensor, Tensor, list[tuple[tuple[str, int], int]]]] = {}
        for key, (probs, issued, row) in state.pending.items():
            remaining.setdefault(id(probs), (probs, issued, []))[2].append((key, row))
        pending = {}
        for probs, issued, entries in remaining.values():
            if len(entries) < len(probs):
                index = _row_index([row for _, row in entries], probs.device)
                compact_probs = probs.detach().index_select(0, index)
                compact_issued = issued.index_select(0, index)
                for position, (key, _) in enumerate(entries):
                    pending[key] = (compact_probs, compact_issued, position)
            else:
                clean = probs.detach()
                for key, row in entries:
                    pending[key] = (clean, issued, row)
        pending = {key: pending[key] for key in state.pending}
        return replace(
            state,
            parameter_id=parameter_id or state.parameter_id,
            markets=markets,
            pending=pending,
        )

    def export_state(self, state: LeadLagMemoryState) -> dict:
        """Exportar copias CPU sin grafo para checkpoints deterministas."""
        detached = self.detach_state(state)
        keys_sorted = sorted(detached.pending)
        if keys_sorted:
            gathered_probs, gathered_issued = _gather_pending(
                [detached.pending[k] for k in keys_sorted]
            )
            pending_memberships = gathered_probs.cpu().clone()
            pending_issued = gathered_issued.to(dtype=torch.float64).cpu().clone()
        else:
            pending_memberships = torch.empty(
                (0, self.config.groups), dtype=self._reference().dtype
            )
            pending_issued = torch.empty((0,), dtype=torch.float64)
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            markets={
                market: dict(
                    delta_transition=stats.delta_transition.cpu().clone(),
                    trace=stats.trace.cpu().clone(),
                    sessions=stats.sessions,
                    updates=stats.updates,
                )
                for market, stats in sorted(detached.markets.items())
            },
            pending=dict(
                flows=[flow for flow, _ in keys_sorted],
                decision_at=torch.tensor([at for _, at in keys_sorted], dtype=torch.int64),
                memberships=pending_memberships,
                issued=pending_issued,
            ),
        )

    def restore_state(
        self, payload: object, parameter_id: str, *, device: torch.device | str
    ) -> LeadLagMemoryState:
        """Validar y restaurar el estado exportado sobre `device`."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "configuration", "markets", "pending"}
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.config.identity())
            or not isinstance(payload["markets"], dict)
            or not isinstance(payload["pending"], dict)
            or set(payload["pending"]) != {"flows", "decision_at", "memberships", "issued"}
        ):
            raise ValueError("El estado guardado de LL-FM no corresponde al predictor")
        reference = self._reference()
        target_device = torch.device(device)
        markets = {}
        for market, entry in payload["markets"].items():
            if not isinstance(entry, dict) or set(entry) != {
                "delta_transition",
                "trace",
                "sessions",
                "updates",
            }:
                raise ValueError("Los estadísticos de mercado guardados de LL-FM están incompletos")
            markets[market] = MarketLeadLagState(
                entry["delta_transition"].to(
                    device=target_device, dtype=reference.dtype, copy=True
                ),
                entry["trace"].to(device=target_device, dtype=reference.dtype, copy=True),
                int(entry["sessions"]),
                int(entry["updates"]),
            )
        raw_pending = payload["pending"]
        flows = raw_pending["flows"]
        moments = raw_pending["decision_at"].tolist()
        issued = raw_pending["issued"].to(device=target_device, dtype=reference.dtype, copy=True)
        memberships = raw_pending["memberships"].to(
            device=target_device, dtype=reference.dtype, copy=True
        )
        if not (len(flows) == len(moments) == len(issued) == len(memberships)):
            raise ValueError("Las pertenencias pendientes guardadas de LL-FM no concilian")
        pending = {
            (flow, int(at)): (memberships, issued, row)
            for row, (flow, at) in enumerate(zip(flows, moments, strict=True))
        }
        state = LeadLagMemoryState(self.fingerprint(), parameter_id, markets, pending)
        self.validate_state(state, parameter_id)
        return state
