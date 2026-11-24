"""Consolidación sináptica de tres factores de rango bajo `three_factor_consolidation_v1` (D3F-LR).

Complementa las memorias rápidas no supervisadas por activo de Titans-MAC con una regla causal de
consolidación sináptica de tres factores (Gerstner et al., 2018) factorizada a rango bajo d << H:

1. **Trazas de elegibilidad de rango 1 por activo** (sobre claves y valores transversales centrados
   frente a pares de la misma sesión y mercado, con norma L2 suave < 1):

       x_{i,t} = LN(stopgrad(e_{i,t}))
       k̂_{i,t} = norm_L2(W_k x_{i,t} − media_{S_{m,t}}(W_k x)) ∈ R^d
       v̂_{i,t} = norm_L2(W_v x_{i,t} − media_{S_{m,t}}(W_v x)) ∈ R^d
       a_{i,t} = (1 − γ_e) a_{i,t−1} + γ_e k̂_{i,t}
       b_{i,t} = (1 − γ_e) b_{i,t−1} + γ_e v̂_{i,t}

   La traza E_{i,t} = a_{i,t} b_{i,t}ᵀ vive en O(d) por flujo y se calienta de forma puramente no
   supervisada incluso durante el periodo de calentamiento de cada fase.

2. **Tercer factor diferido: crédito marginal al IC de la sesión** (solo al madurar etiquetas con
   `decision_at < target_available_at <= event.at` y N_s >= minimum_rows):

       g_{j,s} = Φ⁻¹((rango(y_{j,s}) − 1/2) / N_s)
       δ_{j,s} = g̃_{j,s} · p̃_{j,s} − (1/N_s) Σ_l g̃_{l,s} · p̃_{l,s}
       m_{j,s} = tanh(β · δ_{j,s}) ∈ (−1, 1)

   donde g̃ y p̃ son los rangos normales y predicciones emitidas estandarizados en la sesión. El
   tercer factor mide congruencia predictiva (segundo orden), no el rango bruto del activo.

3. **Consolidación dual (transversal de alta SNR + traza lenta local por flujo)**:
   - Sinapsis transversal compartida por mercado W_m ∈ R^{d×d} (resuelve la SNR ~50× menor del
     crédito aislado por acción-día al promediar sobre los N_s pares de la sesión):

       ΔW_{m,s} = (1/N_s) Σ_{j∈S_{m,s}} m_{j,s} a_{j,s} b_{j,s}ᵀ
       W_m ← (1 − α_w) W_m + α_w ΔW_{m,s},      w_m ← (1 − α_w) w_m + α_w

   - Vector consolidado lento por activo c_i ∈ R^d (acción O(d) de E_{i,s} sobre k̂_{i,s}):

       ψ_{j,s} = (a_{j,s}ᵀ k̂_{j,s}) b_{j,s}
       c_j ← (1 − α_c) c_j + α_c m_{j,s} ψ_{j,s},   u_j ← (1 − α_c) u_j + α_c

4. **Lectura acotada, centrada y protegida del tronco**:
   Al emitir en t (antes de incorporar etiquetas de t), se leen las dos ramas sobre el estado ya
   consolidado (con corrección de sesgo EMA y peso por masa efectiva, nula antes de la primera
   maduración), se normalizan sin sesgo aditivo (RMSNorm) y se inyectan mediante puerta vectorial
   acotada inicializada en cero sobre W_out Kaiming no nula (sin doble cero en serie):

       r^{cross}_{i,t} = k̂_{i,t}ᵀ Ŵ_m,   s^{cross}_{i,t} = v̂_{i,t}ᵀ r^{cross}_{i,t}
       r^{local}_{i,t} = ĉ_{i,t−1},       s^{local}_{i,t} = v̂_{i,t}ᵀ ĉ_{i,t−1}
       e'_{i,t} = e_{i,t} + (g_max · tanh(g)) ⊙ W_out [z̃^{cross}_{i,t}, z̃^{local}_{i,t}]

   `stopgrad(e_{i,t})` aísla al tronco y a la rama técnica de metagradientes ruidosos por etiquetas
   diferidas (mitigando el fallo medido en CRRM durante el ajuste) mientras todos los parámetros
   propios de D3F-LR se entrenan de extremo a extremo dentro de `truncation = 2`.

Controles y ablaciones con exactamente los mismos parámetros e inicialización:
- `three_factor`: mecanismo completo (consolidación transversal + local).
- `cross_only`: solo la sinapsis transversal W_m (anula la rama local).
- `local_only`: solo la consolidación local por flujo c_i (anula la rama transversal).
- `shuffled_credit`: placebo causal que permuta deterministamente m_{j,s} dentro de cada sesión
  madura con semilla derivada de (predictor_seed, config.seed, mercado, decision_at).
- `eval_frozen`: actualiza en ajuste (`train=True`) y congela las escrituras en validación y
  evaluación (`train=False`), separando regularización en ajuste de consolidación en prueba.
"""

import hashlib
import math
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LAYER_NORM_EPS, bounded_integer, canonical, require_identity
from .session_ic_objective import standardized
from .session_ranks import gaussian_ranks
from .state import check_differentiable, check_finite, require_true

THREE_FACTOR_CONSOLIDATION = "three_factor_consolidation_v1"
THREE_FACTOR = "three_factor"
CROSS_ONLY = "cross_only"
LOCAL_ONLY = "local_only"
SHUFFLED_CREDIT = "shuffled_credit"
EVAL_FROZEN = "eval_frozen"
MODES = (THREE_FACTOR, CROSS_ONLY, LOCAL_ONLY, SHUFFLED_CREDIT, EVAL_FROZEN)

MIN_DIM = 4
MAX_DIM = 64
MIN_TAU = 1.0
MAX_TAU = 252.0
MIN_CREDIT_SCALE = 0.05
MAX_CREDIT_SCALE = 10.0
MIN_MAX_GAIN = 0.01
MAX_MAX_GAIN = 2.0
KEY_NORM_EPS = 1e-8
READOUT_NORM_EPS = 1e-5
MASS_EPS = 1e-8
CREDIT_SCALE_FLOOR = 1e-2
DECAY_MIN = 1e-3
DECAY_MAX = 0.999
MAX_UNCOMPACTED_FLOW_BLOCKS = 8


def _finite_float(value: object, low: float, high: float, message: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(message)
    return float(value)


def _softplus_inverse(value: float) -> float:
    """softplus⁻¹(x) = x + log(1 − exp(−x)), estable en todo (0, ∞)."""
    return value + math.log(-math.expm1(-value))


def _decay_logit_from_tau(tau: float) -> float:
    """Logit inicial tal que DECAY_MIN + (DECAY_MAX − DECAY_MIN)·σ(z) = 1 − exp(−1/τ)."""
    target = -math.expm1(-1.0 / tau)
    prob = (target - DECAY_MIN) / (DECAY_MAX - DECAY_MIN)
    return math.log(prob / (1.0 - prob))


def _row_index(rows: list[int], device: torch.device) -> Tensor:
    if device.type != "cuda":
        return torch.tensor(rows, dtype=torch.int64, device=device)
    return torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
        device=device, non_blocking=True
    )


def _market(flow: str) -> str:
    if type(flow) is not str or "/" not in flow:
        raise ValueError("Cada flujo de D3F-LR necesita su prefijo de mercado")
    return flow.split("/", 1)[0]


def _rms_norm(values: Tensor, weight: Tensor) -> Tensor:
    """Normalización RMS sin sesgo aditivo: preserva el cero exacto con cualquier peso."""
    inv_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + READOUT_NORM_EPS)
    return values * inv_rms * weight


@dataclass(frozen=True)
class ThreeFactorConsolidationConfig:
    """Contrato de la consolidación sináptica de tres factores de rango bajo (D3F-LR)."""

    dim: int = 16
    trace_tau: float = 5.0
    consolidation_tau: float = 21.0
    credit_scale: float = 1.0
    max_gain: float = 0.25
    minimum_rows: int = 8
    mode: str = THREE_FACTOR
    seed: int = 0

    def __post_init__(self) -> None:
        bounded_integer(self.dim, "dimensión de D3F-LR", MIN_DIM, MAX_DIM)
        trace_tau = _finite_float(
            self.trace_tau,
            MIN_TAU,
            MAX_TAU,
            f"trace_tau debe ser un número finito en [{MIN_TAU}, {MAX_TAU}]",
        )
        consolidation_tau = _finite_float(
            self.consolidation_tau,
            MIN_TAU,
            MAX_TAU,
            f"consolidation_tau debe ser un número finito en [{MIN_TAU}, {MAX_TAU}]",
        )
        credit_scale = _finite_float(
            self.credit_scale,
            MIN_CREDIT_SCALE,
            MAX_CREDIT_SCALE,
            f"credit_scale debe ser un número finito en [{MIN_CREDIT_SCALE}, {MAX_CREDIT_SCALE}]",
        )
        max_gain = _finite_float(
            self.max_gain,
            MIN_MAX_GAIN,
            MAX_MAX_GAIN,
            f"max_gain debe ser un número finito en [{MIN_MAX_GAIN}, {MAX_MAX_GAIN}]",
        )
        bounded_integer(self.minimum_rows, "minimum_rows de D3F-LR", 3, 4096)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de D3F-LR debe ser uno de {MODES}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de D3F-LR debe ser un entero de 32 bits")
        object.__setattr__(self, "trace_tau", trace_tau)
        object.__setattr__(self, "consolidation_tau", consolidation_tau)
        object.__setattr__(self, "credit_scale", credit_scale)
        object.__setattr__(self, "max_gain", max_gain)

    def identity(self) -> dict:
        return dict(
            schema_version=1,
            kind=THREE_FACTOR_CONSOLIDATION,
            dim=self.dim,
            trace_tau=self.trace_tau,
            consolidation_tau=self.consolidation_tau,
            credit_scale=self.credit_scale,
            max_gain=self.max_gain,
            minimum_rows=self.minimum_rows,
            mode=self.mode,
            seed=self.seed,
            input_gradient="detached_from_backbone_to_prevent_meta_gradient_interference",
            eligibility="per_flow_rank1_factorized_ema_of_cross_section_centered_l2_kv",
            third_factor="tanh_of_learned_scale_times_centered_marginal_session_ic_credit",
            consolidation=(
                "dual_cross_asset_market_synapse_and_per_flow_slow_vector_with_ema_bias_correction"
            ),
            readout="bias_free_rmsnorm_mass_gated_kaiming_proj_with_bounded_zero_tanh_gate",
            key_norm_eps=KEY_NORM_EPS,
            readout_norm_eps=READOUT_NORM_EPS,
            mass_eps=MASS_EPS,
            market_isolation="strict_per_market_synapse_and_session_centering",
            causality="traces_unsupervised_consolidation_only_on_matured_labels_before_emission",
        )


def parse_three_factor_consolidation(raw: object) -> ThreeFactorConsolidationConfig | None:
    """Aceptar ThreeFactorConsolidationConfig, su diccionario completo o None."""
    if raw is None:
        return None
    if type(raw) is ThreeFactorConsolidationConfig:
        raw.__post_init__()
        return raw
    expected = {field.name for field in fields(ThreeFactorConsolidationConfig)}
    if isinstance(raw, dict) and set(raw) == expected:
        return ThreeFactorConsolidationConfig(**raw)
    raise ValueError(
        "three_factor_consolidation debe declarar dim, trace_tau, consolidation_tau, "
        "credit_scale, max_gain, minimum_rows, mode y seed, o ser None"
    )


@dataclass(frozen=True)
class MarketConsolidationState:
    """Sinapsis transversal consolidada W_m ∈ R^{d×d} y masa efectiva w_m de un mercado."""

    synapse: Tensor
    mass: Tensor
    updates: int


@dataclass(frozen=True)
class ThreeFactorConsolidationState:
    """Estado de D3F-LR: sinapsis por mercado, trazas por flujo y emisiones pendientes.

    `flows` y `pending` guardan las matrices completas de cada emisión junto al índice de fila:
    así una sesión de miles de activos se reúne con una sola selección, el grafo de autograd es
    O(1) respecto al número de filas y ninguna operación espera a la GPU en el camino caliente.
    """

    config_id: str
    parameter_id: str
    markets: dict[str, MarketConsolidationState]
    # flow_id -> (a_block [M, d], b_block [M, d], c_block [M, d], u_block [M], row_idx)
    flows: dict[str, tuple[Tensor, Tensor, Tensor, Tensor, int]]
    # (flow_id, decision_at) -> (k_block [M, d], a_block [M, d], b_block [M, d], issued [M], row)
    pending: dict[tuple[str, int], tuple[Tensor, Tensor, Tensor, Tensor, int]]


def _gather_flow_records(
    flow_map: dict[str, tuple[Tensor, Tensor, Tensor, Tensor, int]],
    flow_ids: tuple[str, ...] | list[str],
    *,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reunir (a, b, c, u) para `flow_ids` en O(bloques) nodos de autograd."""
    count = len(flow_ids)
    if count == 0:
        empty = torch.empty((0, dim), dtype=dtype, device=device)
        return empty, empty.clone(), empty.clone(), torch.empty((0,), dtype=dtype, device=device)
    sources: dict[
        tuple[int, int, int, int], tuple[Tensor, Tensor, Tensor, Tensor, list[int], list[int]]
    ] = {}
    missing_positions: list[int] = []
    for position, flow in enumerate(flow_ids):
        entry = flow_map.get(flow)
        if entry is None:
            missing_positions.append(position)
            continue
        a_blk, b_blk, c_blk, u_blk, row = entry
        key = (id(a_blk), id(b_blk), id(c_blk), id(u_blk))
        bucket = sources.get(key)
        if bucket is None:
            bucket = (a_blk, b_blk, c_blk, u_blk, [], [])
            sources[key] = bucket
        bucket[4].append(position)
        bucket[5].append(row)

    if not sources:
        zeros = torch.zeros((count, dim), dtype=dtype, device=device)
        return (
            zeros,
            zeros.clone(),
            zeros.clone(),
            torch.zeros((count,), dtype=dtype, device=device),
        )

    # Camino rápido sin copia si todas las filas coinciden exactamente con un único bloque.
    if not missing_positions and len(sources) == 1:
        a_blk, b_blk, c_blk, u_blk, positions, picked = next(iter(sources.values()))
        if len(picked) == len(a_blk) and picked == list(range(len(a_blk))):
            return a_blk, b_blk, c_blk, u_blk
        index = _row_index(picked, device)
        return (
            a_blk.index_select(0, index),
            b_blk.index_select(0, index),
            c_blk.index_select(0, index),
            u_blk.index_select(0, index),
        )

    a_parts, b_parts, c_parts, u_parts, order = [], [], [], [], []
    for a_blk, b_blk, c_blk, u_blk, positions, picked in sources.values():
        index = _row_index(picked, device)
        a_parts.append(a_blk.index_select(0, index))
        b_parts.append(b_blk.index_select(0, index))
        c_parts.append(c_blk.index_select(0, index))
        u_parts.append(u_blk.index_select(0, index))
        order.extend(positions)
    if missing_positions:
        m = len(missing_positions)
        zeros = torch.zeros((m, dim), dtype=dtype, device=device)
        a_parts.append(zeros)
        b_parts.append(zeros)
        c_parts.append(zeros)
        u_parts.append(torch.zeros((m,), dtype=dtype, device=device))
        order.extend(missing_positions)
    inverse = torch.empty(count, dtype=torch.int64)
    inverse[torch.tensor(order, dtype=torch.int64)] = torch.arange(count, dtype=torch.int64)
    inverse = inverse.to(device)
    return (
        torch.cat(a_parts, dim=0).index_select(0, inverse),
        torch.cat(b_parts, dim=0).index_select(0, inverse),
        torch.cat(c_parts, dim=0).index_select(0, inverse),
        torch.cat(u_parts, dim=0).index_select(0, inverse),
    )


def _gather_pending(
    entries: list[tuple[Tensor, Tensor, Tensor, Tensor, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reunir (k, a, b, issued) en el orden pedido con O(emisiones) nodos de autograd."""
    sources: dict[int, tuple[Tensor, Tensor, Tensor, Tensor, list[int], list[int]]] = {}
    for position, (k_blk, a_blk, b_blk, p_blk, row) in enumerate(entries):
        bucket = sources.setdefault(id(k_blk), (k_blk, a_blk, b_blk, p_blk, [], []))
        bucket[4].append(position)
        bucket[5].append(row)
    k_parts, a_parts, b_parts, p_parts, order = [], [], [], [], []
    for k_blk, a_blk, b_blk, p_blk, positions, picked in sources.values():
        index = _row_index(picked, k_blk.device)
        k_parts.append(k_blk.index_select(0, index))
        a_parts.append(a_blk.index_select(0, index))
        b_parts.append(b_blk.index_select(0, index))
        p_parts.append(p_blk.index_select(0, index))
        order.extend(positions)
    if len(k_parts) == 1:
        return k_parts[0], a_parts[0], b_parts[0], p_parts[0]
    inverse = torch.empty(len(order), dtype=torch.int64)
    inverse[torch.tensor(order, dtype=torch.int64)] = torch.arange(len(order), dtype=torch.int64)
    inverse = inverse.to(k_parts[0].device)
    return (
        torch.cat(k_parts, dim=0).index_select(0, inverse),
        torch.cat(a_parts, dim=0).index_select(0, inverse),
        torch.cat(b_parts, dim=0).index_select(0, inverse),
        torch.cat(p_parts, dim=0).index_select(0, inverse),
    )


def _compact_flow_map(
    flow_map: dict[str, tuple[Tensor, Tensor, Tensor, Tensor, int]],
    *,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor, int]]:
    """Compactar todos los flujos vivos en un solo bloque contiguo desacoplado."""
    if not flow_map:
        return {}
    ordered = sorted(flow_map)
    a_all, b_all, c_all, u_all = _gather_flow_records(
        flow_map, ordered, dim=dim, dtype=dtype, device=device
    )
    a_all, b_all, c_all, u_all = (
        a_all.detach().contiguous(),
        b_all.detach().contiguous(),
        c_all.detach().contiguous(),
        u_all.detach().contiguous(),
    )
    return {flow: (a_all, b_all, c_all, u_all, idx) for idx, flow in enumerate(ordered)}


class ThreeFactorConsolidation(nn.Module):
    """Consolidación sináptica de tres factores de rango bajo (D3F-LR)."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        config: ThreeFactorConsolidationConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión oculta de D3F-LR", 1, 4096)
        bounded_integer(seed, "semilla de D3F-LR", 0, 2**32 - 1)
        if type(config) is not ThreeFactorConsolidationConfig:
            raise ValueError("D3F-LR necesita ThreeFactorConsolidationConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("D3F-LR admite precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        branch_dim = config.dim + 1
        # Semilla aislada e independiente de `mode`: todas las variantes arrancan iguales.
        derived = hashlib.sha256(f"{THREE_FACTOR_CONSOLIDATION}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.key_proj = nn.Linear(hidden_size, config.dim, bias=False, dtype=dtype)
            self.val_proj = nn.Linear(hidden_size, config.dim, bias=False, dtype=dtype)
            self.out = nn.Linear(2 * branch_dim, hidden_size, bias=False, dtype=dtype)
        # Escalas RMSNorm sin sesgo y puerta vectorial a cero sobre W_out Kaiming (sin doble cero).
        self.cross_norm_weight = nn.Parameter(torch.ones(branch_dim, dtype=dtype))
        self.local_norm_weight = nn.Parameter(torch.ones(branch_dim, dtype=dtype))
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.trace_decay_logit = nn.Parameter(
            torch.tensor(_decay_logit_from_tau(config.trace_tau), dtype=dtype)
        )
        self.cross_decay_logit = nn.Parameter(
            torch.tensor(_decay_logit_from_tau(config.consolidation_tau), dtype=dtype)
        )
        self.local_decay_logit = nn.Parameter(
            torch.tensor(_decay_logit_from_tau(config.consolidation_tau), dtype=dtype)
        )
        self.credit_scale_logit = nn.Parameter(
            torch.tensor(_softplus_inverse(config.credit_scale - CREDIT_SCALE_FLOOR), dtype=dtype)
        )
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
            raise ValueError("D3F-LR admite precisión FP32 o FP64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("D3F-LR exige FP32 o FP64 explícitos, sin autocast")
        return reference

    def _trace_decay(self) -> Tensor:
        return DECAY_MIN + (DECAY_MAX - DECAY_MIN) * torch.sigmoid(self.trace_decay_logit)

    def _cross_decay(self) -> Tensor:
        return DECAY_MIN + (DECAY_MAX - DECAY_MIN) * torch.sigmoid(self.cross_decay_logit)

    def _local_decay(self) -> Tensor:
        return DECAY_MIN + (DECAY_MAX - DECAY_MIN) * torch.sigmoid(self.local_decay_logit)

    def _credit_scale(self) -> Tensor:
        return CREDIT_SCALE_FLOOR + F.softplus(self.credit_scale_logit)

    def initial_state(self, parameter_id: str) -> ThreeFactorConsolidationState:
        return ThreeFactorConsolidationState(self.fingerprint(), parameter_id, {}, {}, {})

    def validate_state(self, state: object, parameter_id: str) -> None:
        reference = self._reference()
        dim = self.config.dim
        if (
            not isinstance(state, ThreeFactorConsolidationState)
            or state.config_id != self.fingerprint()
            or state.parameter_id != parameter_id
            or not isinstance(state.markets, dict)
            or not isinstance(state.flows, dict)
            or not isinstance(state.pending, dict)
        ):
            raise ValueError("El estado de consolidación de tres factores no corresponde al módulo")
        for market, stats in state.markets.items():
            if (
                type(market) is not str
                or not market
                or not isinstance(stats, MarketConsolidationState)
                or type(stats.updates) is not int
                or stats.updates < 1
                or not isinstance(stats.synapse, Tensor)
                or not isinstance(stats.mass, Tensor)
                or stats.synapse.shape != (dim, dim)
                or stats.mass.shape != ()
                or stats.synapse.dtype != reference.dtype
                or stats.mass.dtype != reference.dtype
                or stats.synapse.device != reference.device
                or stats.mass.device != reference.device
            ):
                raise ValueError(
                    "El estado sináptico por mercado de D3F-LR no conserva su contrato"
                )
            check_finite(stats.synapse, "La sinapsis transversal de D3F-LR")
            check_finite(stats.mass, "La masa efectiva de mercado de D3F-LR")
            require_true(
                (stats.mass >= 0) & (stats.mass <= 1.0 + 1e-6),
                "La masa efectiva de mercado de D3F-LR debe permanecer en [0, 1]",
            )
        checked_blocks: set[tuple[int, int, int, int]] = set()
        for flow, entry in state.flows.items():
            if type(flow) is not str or not isinstance(entry, tuple) or len(entry) != 5:
                raise ValueError("El estado por flujo de D3F-LR no conserva su contrato")
            a_blk, b_blk, c_blk, u_blk, row = entry
            if type(row) is not int or row < 0:
                raise ValueError("El índice de fila de D3F-LR no es válido")
            block_key = (id(a_blk), id(b_blk), id(c_blk), id(u_blk))
            if block_key not in checked_blocks:
                if (
                    not isinstance(a_blk, Tensor)
                    or not isinstance(b_blk, Tensor)
                    or not isinstance(c_blk, Tensor)
                    or not isinstance(u_blk, Tensor)
                    or a_blk.ndim != 2
                    or b_blk.shape != a_blk.shape
                    or c_blk.shape != a_blk.shape
                    or a_blk.shape[1] != dim
                    or u_blk.shape != (a_blk.shape[0],)
                    or a_blk.dtype != reference.dtype
                    or b_blk.dtype != reference.dtype
                    or c_blk.dtype != reference.dtype
                    or u_blk.dtype != reference.dtype
                    or a_blk.device != reference.device
                    or b_blk.device != reference.device
                    or c_blk.device != reference.device
                    or u_blk.device != reference.device
                ):
                    raise ValueError("Los bloques de traza de D3F-LR no conservan su contrato")
                check_finite(a_blk, "La traza a de D3F-LR")
                check_finite(b_blk, "La traza b de D3F-LR")
                check_finite(c_blk, "El vector consolidado local c de D3F-LR")
                check_finite(u_blk, "La masa efectiva local de D3F-LR")
                require_true(
                    (u_blk >= 0).all() & (u_blk <= 1.0 + 1e-6).all(),
                    "La masa efectiva local de D3F-LR debe permanecer en [0, 1]",
                )
                checked_blocks.add(block_key)
            if row >= a_blk.shape[0]:
                raise ValueError("El índice de fila de D3F-LR excede su bloque")

    def _project_kv(
        self,
        encoded: Tensor,
        groups: list[tuple[tuple[str, int], list[int]]],
    ) -> tuple[Tensor, Tensor]:
        """Claves y valores centrados por sesión y mercado sobre el tronco desacoplado."""
        normed = self.norm(encoded.detach())
        raw_k = self.key_proj(normed)
        raw_v = self.val_proj(normed)
        if len(groups) == 1:
            if len(raw_k) > 1:
                raw_k = raw_k - raw_k.mean(dim=0, keepdim=True)
                raw_v = raw_v - raw_v.mean(dim=0, keepdim=True)
            k_hat = raw_k / torch.sqrt(raw_k.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
            v_hat = raw_v / torch.sqrt(raw_v.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
            return k_hat, v_hat
        k_hat = torch.zeros_like(raw_k)
        v_hat = torch.zeros_like(raw_v)
        for _, members in groups:
            index = _row_index(members, raw_k.device)
            pk = raw_k.index_select(0, index)
            pv = raw_v.index_select(0, index)
            if len(members) > 1:
                pk = pk - pk.mean(dim=0, keepdim=True)
                pv = pv - pv.mean(dim=0, keepdim=True)
            k_hat = k_hat.index_copy(
                0, index, pk / torch.sqrt(pk.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
            )
            v_hat = v_hat.index_copy(
                0, index, pv / torch.sqrt(pv.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
            )
        return k_hat, v_hat

    def read_and_trace(
        self,
        encoded: Tensor,
        flow_ids: tuple[str, ...] | list[str],
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: ThreeFactorConsolidationState,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Leer el residuo causal consolidado [N, H] y calcular la traza (k̂, a_t, b_t)."""
        reference = self._reference()
        dim = self.config.dim
        if (
            not isinstance(encoded, Tensor)
            or encoded.layout != torch.strided
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or len(flow_ids) != len(encoded)
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError(
                "D3F-LR necesita una representación, un flujo, un mercado y un instante por fila"
            )
        check_finite(encoded, "La representación de entrada de D3F-LR")
        rows = len(encoded)
        session_map: dict[tuple[str, int], list[int]] = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            session_map.setdefault(key, []).append(row)
        groups = list(session_map.items())

        k_hat, v_hat = self._project_kv(encoded, groups)
        a_prev, b_prev, c_prev, u_prev = _gather_flow_records(
            state.flows, flow_ids, dim=dim, dtype=reference.dtype, device=reference.device
        )

        # Rama transversal por mercado (mejora la SNR y captura interacción entre pares).
        branch_dim = dim + 1
        z_cross = encoded.new_zeros((rows, branch_dim))
        if state.markets:
            for (market, _), members in groups:
                stats = state.markets.get(market)
                if stats is None:
                    continue
                contiguous = len(members) == rows
                if contiguous:
                    k_m, v_m = k_hat, v_hat
                else:
                    index = _row_index(members, encoded.device)
                    k_m = k_hat.index_select(0, index)
                    v_m = v_hat.index_select(0, index)
                w_hat = stats.synapse / (stats.mass + MASS_EPS)
                r_cross = k_m @ w_hat
                s_cross = (v_m * r_cross).sum(dim=-1, keepdim=True)
                part = torch.cat((r_cross, s_cross), dim=-1)
                if len(members) > 1:
                    part = part - part.mean(dim=0, keepdim=True)
                part = stats.mass * _rms_norm(part, self.cross_norm_weight)
                if contiguous:
                    z_cross = part
                else:
                    z_cross = z_cross.index_copy(0, index, part)

        # Rama local consolidada por flujo (traza lenta idiosincrásica, centrada por sesión).
        c_hat = c_prev / (u_prev.unsqueeze(-1) + MASS_EPS)
        s_local = (v_hat * c_hat).sum(dim=-1, keepdim=True)
        z_local_raw = torch.cat((c_hat, s_local), dim=-1)
        z_local = u_prev.unsqueeze(-1) * _rms_norm(z_local_raw, self.local_norm_weight)
        if len(groups) == 1:
            if rows > 1:
                z_local = z_local - z_local.mean(dim=0, keepdim=True)
        else:
            centered_local = torch.zeros_like(z_local)
            for _, members in groups:
                index = _row_index(members, encoded.device)
                part = z_local.index_select(0, index)
                if len(members) > 1:
                    part = part - part.mean(dim=0, keepdim=True)
                centered_local = centered_local.index_copy(0, index, part)
            z_local = centered_local

        if self.config.mode == LOCAL_ONLY:
            z_cross = torch.zeros_like(z_cross)
        elif self.config.mode == CROSS_ONLY:
            z_local = torch.zeros_like(z_local)

        features = torch.cat((z_cross, z_local), dim=-1)
        residual = self.out(features)
        gain = self.config.max_gain * torch.tanh(self.gate)
        delta = gain * residual

        # Actualización no supervisada de la traza de elegibilidad de rango 1 (a_{i,t}, b_{i,t}).
        gamma_e = self._trace_decay()
        a_next = (1.0 - gamma_e) * a_prev + gamma_e * k_hat
        b_next = (1.0 - gamma_e) * b_prev + gamma_e * v_hat

        check_finite(delta, "El residuo de consolidación de tres factores")
        check_finite(a_next, "La traza de elegibilidad a de D3F-LR")
        check_finite(b_next, "La traza de elegibilidad b de D3F-LR")
        return delta, (k_hat, a_next, b_next)

    def stage(
        self,
        state: ThreeFactorConsolidationState,
        flow_ids: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        emission: tuple[Tensor, Tensor, Tensor],
        issued_predictions: Tensor,
        *,
        record_pending: bool = True,
        differentiable: bool = False,
    ) -> ThreeFactorConsolidationState:
        """Avanzar trazas de elegibilidad y, fuera de calentamiento, registrar emisión pendiente."""
        check_differentiable(differentiable)
        if type(record_pending) is not bool:
            raise ValueError("record_pending debe ser booleano")
        reference = self._reference()
        dim = self.config.dim
        if (
            not isinstance(emission, tuple)
            or len(emission) != 3
            or any(
                not isinstance(t, Tensor)
                or t.shape != (len(flow_ids), dim)
                or t.dtype != reference.dtype
                or t.device != reference.device
                for t in emission
            )
            or not isinstance(issued_predictions, Tensor)
            or issued_predictions.shape != (len(flow_ids),)
            or len(prediction_at) != len(flow_ids)
        ):
            raise ValueError(
                "El registro de D3F-LR necesita (k̂, a, b) y predicciones emitidas por fila"
            )
        k_hat, a_next, b_next = emission
        check_finite(k_hat, "Las claves de emisión de D3F-LR")
        check_finite(a_next, "La traza a de emisión de D3F-LR")
        check_finite(b_next, "La traza b de emisión de D3F-LR")
        check_finite(issued_predictions, "Las predicciones emitidas para D3F-LR")

        if not differentiable:
            k_hat, a_next, b_next = k_hat.detach(), a_next.detach(), b_next.detach()
        _, _, c_prev, u_prev = _gather_flow_records(
            state.flows, flow_ids, dim=dim, dtype=reference.dtype, device=reference.device
        )
        if not differentiable:
            c_prev, u_prev = c_prev.detach(), u_prev.detach()

        flows = dict(state.flows)
        for row, flow in enumerate(flow_ids):
            flows[flow] = (a_next, b_next, c_prev, u_prev, row)
        if not differentiable and len({id(e[0]) for e in flows.values()}) > (
            MAX_UNCOMPACTED_FLOW_BLOCKS
        ):
            flows = _compact_flow_map(
                flows, dim=dim, dtype=reference.dtype, device=reference.device
            )

        if not record_pending:
            return replace(state, flows=flows)

        issued = issued_predictions.detach().to(device=reference.device, dtype=reference.dtype)
        pending = dict(state.pending)
        for row, (flow, at) in enumerate(zip(flow_ids, prediction_at, strict=True)):
            entry_key = (flow, int(at))
            if entry_key in pending:
                raise ValueError("D3F-LR ya tiene registrada esa emisión pendiente")
            pending[entry_key] = (k_hat, a_next, b_next, issued, row)
        return replace(state, flows=flows, pending=pending)

    def _shuffle_permutation(self, market: str, decision_at: int, count: int) -> Tensor:
        text = (
            f"{THREE_FACTOR_CONSOLIDATION}/shuffle/"
            f"{self.seed}/{self.config.seed}/{market}/{decision_at}"
        )
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(count, generator=generator)

    def settle(
        self,
        state: ThreeFactorConsolidationState,
        matured: list[tuple[str, int, float]] | tuple[tuple[str, int, float], ...],
        *,
        differentiable: bool = False,
        train: bool = True,
    ) -> ThreeFactorConsolidationState:
        """Consolidar sinapsis transversal y vector local con el crédito marginal al IC maduro."""
        check_differentiable(differentiable)
        if type(train) is not bool:
            raise ValueError("train debe ser booleano")
        if not matured:
            return state
        reference = self._reference()
        dim = self.config.dim
        pending = dict(state.pending)
        grouped: dict[tuple[int, str], list[tuple[str, Tensor, Tensor, Tensor, Tensor, int, float]]]
        grouped = {}
        for flow, decision_at, outcome in matured:
            if (
                isinstance(outcome, bool)
                or not isinstance(outcome, (int, float))
                or not math.isfinite(outcome)
            ):
                raise ValueError("El resultado maduro para D3F-LR debe ser un real finito")
            entry_key = (flow, int(decision_at))
            if entry_key not in pending:
                raise ValueError("La etiqueta madura de D3F-LR no tiene emisión pendiente")
            grouped.setdefault((int(decision_at), _market(flow)), []).append(
                (flow, *pending.pop(entry_key), float(outcome))
            )
        if self.config.mode == EVAL_FROZEN and not train:
            return replace(state, pending=pending)

        markets = dict(state.markets)
        flows = dict(state.flows)
        alpha_w = self._cross_decay()
        alpha_c = self._local_decay()
        beta = self._credit_scale()
        if not differentiable:
            alpha_w, alpha_c, beta = alpha_w.detach(), alpha_c.detach(), beta.detach()

        for decision_at, market in sorted(grouped):
            rows = sorted(grouped[decision_at, market], key=lambda item: item[0])
            if len(rows) < self.config.minimum_rows:
                continue
            flow_names = [item[0] for item in rows]
            k_mat, a_mat, b_mat, issued = _gather_pending([item[1:6] for item in rows])
            if not differentiable:
                k_mat, a_mat, b_mat = k_mat.detach(), a_mat.detach(), b_mat.detach()
            outcomes = torch.tensor(
                [item[6] for item in rows], dtype=reference.dtype, device=reference.device
            )
            present = torch.ones((len(rows), 1), dtype=torch.bool, device=reference.device)
            normal_ranks = gaussian_ranks(outcomes.unsqueeze(-1), present).squeeze(-1)
            prod = standardized(normal_ranks) * standardized(issued)
            delta_ic = prod - prod.mean()
            third_factor = torch.tanh(beta * delta_ic)
            if self.config.mode == SHUFFLED_CREDIT:
                perm = self._shuffle_permutation(market, decision_at, len(rows)).to(
                    reference.device
                )
                third_factor = third_factor.index_select(0, perm)

            # 1) Sinapsis transversal de rango bajo W_m ∈ R^{d×d} (promedio sobre los N_s pares).
            inv_n = 1.0 / float(len(rows))
            dw = (a_mat.transpose(0, 1) @ (third_factor.unsqueeze(-1) * b_mat)) * inv_n
            previous = markets.get(market)
            if previous is None:
                prev_w = torch.zeros((dim, dim), dtype=reference.dtype, device=reference.device)
                prev_m = torch.zeros((), dtype=reference.dtype, device=reference.device)
                updates = 1
            else:
                prev_w = previous.synapse if differentiable else previous.synapse.detach()
                prev_m = previous.mass if differentiable else previous.mass.detach()
                updates = previous.updates + 1
            new_w = (1.0 - alpha_w) * prev_w + alpha_w * dw
            new_m = (1.0 - alpha_w) * prev_m + alpha_w
            if not differentiable:
                new_w, new_m = new_w.detach(), new_m.detach()
            check_finite(new_w, "La actualización sináptica transversal de D3F-LR")
            check_finite(new_m, "La masa efectiva de mercado de D3F-LR")
            markets[market] = MarketConsolidationState(new_w, new_m, updates)

            # 2) Consolidación lenta local c_j ∈ R^d sobre la acción O(d) de E_{j,s} = a bᵀ en k̂.
            a_cur, b_cur, c_cur, u_cur = _gather_flow_records(
                flows, flow_names, dim=dim, dtype=reference.dtype, device=reference.device
            )
            if not differentiable:
                a_cur, b_cur, c_cur, u_cur = (
                    a_cur.detach(),
                    b_cur.detach(),
                    c_cur.detach(),
                    u_cur.detach(),
                )
            psi = (a_mat * k_mat).sum(dim=-1, keepdim=True) * b_mat
            c_new = (1.0 - alpha_c) * c_cur + alpha_c * (third_factor.unsqueeze(-1) * psi)
            u_new = (1.0 - alpha_c) * u_cur + alpha_c
            if not differentiable:
                c_new, u_new = c_new.detach(), u_new.detach()
            check_finite(c_new, "El vector consolidado local de D3F-LR")
            check_finite(u_new, "La masa efectiva local de D3F-LR")
            for idx, flow in enumerate(flow_names):
                flows[flow] = (a_cur, b_cur, c_new, u_new, idx)

        if not differentiable and len({id(e[0]) for e in flows.values()}) > (
            MAX_UNCOMPACTED_FLOW_BLOCKS
        ):
            flows = _compact_flow_map(
                flows, dim=dim, dtype=reference.dtype, device=reference.device
            )
        return replace(state, markets=markets, flows=flows, pending=pending)

    def detach_state(
        self, state: ThreeFactorConsolidationState, parameter_id: str | None = None
    ) -> ThreeFactorConsolidationState:
        """Cortar el grafo en las fronteras de truncamiento y compactar bloques vivos."""
        reference = self._reference()
        markets = {
            market: MarketConsolidationState(
                stats.synapse.detach(), stats.mass.detach(), stats.updates
            )
            for market, stats in state.markets.items()
        }
        flows = _compact_flow_map(
            state.flows, dim=self.config.dim, dtype=reference.dtype, device=reference.device
        )
        remaining: dict[
            int, tuple[Tensor, Tensor, Tensor, Tensor, list[tuple[tuple[str, int], int]]]
        ] = {}
        for key, (k_blk, a_blk, b_blk, p_blk, row) in state.pending.items():
            remaining.setdefault(id(k_blk), (k_blk, a_blk, b_blk, p_blk, []))[4].append((key, row))
        pending = {}
        for k_blk, a_blk, b_blk, p_blk, entries in remaining.values():
            if len(entries) < len(k_blk):
                index = _row_index([row for _, row in entries], k_blk.device)
                ck = k_blk.detach().index_select(0, index)
                ca = a_blk.detach().index_select(0, index)
                cb = b_blk.detach().index_select(0, index)
                cp = p_blk.detach().index_select(0, index)
                for pos, (key, _) in enumerate(entries):
                    pending[key] = (ck, ca, cb, cp, pos)
            else:
                ck, ca, cb, cp = k_blk.detach(), a_blk.detach(), b_blk.detach(), p_blk.detach()
                for key, row in entries:
                    pending[key] = (ck, ca, cb, cp, row)
        pending = {key: pending[key] for key in state.pending}
        return replace(
            state,
            parameter_id=parameter_id or state.parameter_id,
            markets=markets,
            flows=flows,
            pending=pending,
        )

    def export_state(self, state: ThreeFactorConsolidationState) -> dict:
        """Exportar copias CPU sin grafo para checkpoints deterministas."""
        reference = self._reference()
        detached = self.detach_state(state)
        flow_ids = sorted(detached.flows)
        if flow_ids:
            a_all, b_all, c_all, u_all = _gather_flow_records(
                detached.flows,
                flow_ids,
                dim=self.config.dim,
                dtype=reference.dtype,
                device=reference.device,
            )
            flow_payload = dict(
                flow_ids=flow_ids,
                a=a_all.cpu().clone(),
                b=b_all.cpu().clone(),
                c=c_all.cpu().clone(),
                u=u_all.cpu().clone(),
            )
        else:
            empty = torch.empty((0, self.config.dim), dtype=reference.dtype)
            flow_payload = dict(
                flow_ids=[],
                a=empty,
                b=empty.clone(),
                c=empty.clone(),
                u=torch.empty((0,), dtype=reference.dtype),
            )
        keys_sorted = sorted(detached.pending)
        if keys_sorted:
            gk, ga, gb, gp = _gather_pending([detached.pending[k] for k in keys_sorted])
            pending_payload = dict(
                flows=[flow for flow, _ in keys_sorted],
                decision_at=torch.tensor([at for _, at in keys_sorted], dtype=torch.int64),
                keys=gk.cpu().clone(),
                trace_a=ga.cpu().clone(),
                trace_b=gb.cpu().clone(),
                issued=gp.to(dtype=torch.float64).cpu().clone(),
            )
        else:
            empty = torch.empty((0, self.config.dim), dtype=reference.dtype)
            pending_payload = dict(
                flows=[],
                decision_at=torch.empty((0,), dtype=torch.int64),
                keys=empty,
                trace_a=empty.clone(),
                trace_b=empty.clone(),
                issued=torch.empty((0,), dtype=torch.float64),
            )
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            markets={
                market: dict(
                    synapse=stats.synapse.cpu().clone(),
                    mass=stats.mass.cpu().clone(),
                    updates=stats.updates,
                )
                for market, stats in sorted(detached.markets.items())
            },
            flows=flow_payload,
            pending=pending_payload,
        )

    def restore_state(
        self, payload: object, parameter_id: str, *, device: torch.device | str
    ) -> ThreeFactorConsolidationState:
        """Validar y restaurar el estado exportado sobre `device`."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "configuration", "markets", "flows", "pending"}
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.config.identity())
            or not isinstance(payload["markets"], dict)
            or not isinstance(payload["flows"], dict)
            or set(payload["flows"]) != {"flow_ids", "a", "b", "c", "u"}
            or not isinstance(payload["pending"], dict)
            or set(payload["pending"])
            != {"flows", "decision_at", "keys", "trace_a", "trace_b", "issued"}
        ):
            raise ValueError("El estado guardado de D3F-LR no corresponde al predictor")
        reference = self._reference()
        target_device = torch.device(device)
        markets = {}
        for market, entry in payload["markets"].items():
            if not isinstance(entry, dict) or set(entry) != {"synapse", "mass", "updates"}:
                raise ValueError(
                    "Los estadísticos de mercado guardados de D3F-LR están incompletos"
                )
            markets[market] = MarketConsolidationState(
                entry["synapse"].to(device=target_device, dtype=reference.dtype, copy=True),
                entry["mass"].to(device=target_device, dtype=reference.dtype, copy=True),
                int(entry["updates"]),
            )
        raw_flows = payload["flows"]
        flow_ids = raw_flows["flow_ids"]
        a_all = raw_flows["a"].to(device=target_device, dtype=reference.dtype, copy=True)
        b_all = raw_flows["b"].to(device=target_device, dtype=reference.dtype, copy=True)
        c_all = raw_flows["c"].to(device=target_device, dtype=reference.dtype, copy=True)
        u_all = raw_flows["u"].to(device=target_device, dtype=reference.dtype, copy=True)
        if not (len(flow_ids) == len(a_all) == len(b_all) == len(c_all) == len(u_all)):
            raise ValueError("Las trazas por flujo guardadas de D3F-LR no concilian")
        flows = {flow: (a_all, b_all, c_all, u_all, idx) for idx, flow in enumerate(flow_ids)}

        raw_pending = payload["pending"]
        p_flows = raw_pending["flows"]
        moments = raw_pending["decision_at"].tolist()
        k_blk = raw_pending["keys"].to(device=target_device, dtype=reference.dtype, copy=True)
        pa_blk = raw_pending["trace_a"].to(device=target_device, dtype=reference.dtype, copy=True)
        pb_blk = raw_pending["trace_b"].to(device=target_device, dtype=reference.dtype, copy=True)
        issued = raw_pending["issued"].to(device=target_device, dtype=reference.dtype, copy=True)
        if not (
            len(p_flows) == len(moments) == len(k_blk) == len(pa_blk) == len(pb_blk) == len(issued)
        ):
            raise ValueError("Las emisiones pendientes guardadas de D3F-LR no concilian")
        pending = {
            (flow, int(at)): (k_blk, pa_blk, pb_blk, issued, row)
            for row, (flow, at) in enumerate(zip(p_flows, moments, strict=True))
        }
        state = ThreeFactorConsolidationState(
            self.fingerprint(), parameter_id, markets, flows, pending
        )
        self.validate_state(state, parameter_id)
        return state
