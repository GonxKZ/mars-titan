"""Memoria neuronal de Titans supervisada por correlación de rango con retención anclada
(`ic_titans_mlp_v1`, IC-Titans-MLP).

Mantiene una memoria neuronal MLP de 2 capas compartida por mercado M_{m,IC} = (W_{1,m}, W_{2,m})
con la misma arquitectura residual normalizada que `NeuralMemory` (`residual_layer_norm = True`):

    M_m(x) = x + LayerNorm(W_{2,m} GELU(W_{1,m} x))

En el instante de decisión t, para las N_t filas del mercado m con representación e_{i,t} ∈ R^H
antes de la cabeza, se construyen claves transversales centradas por sesión y de norma acotada:

    u_{i,t} = W_k LN(e_{i,t}) ∈ R^d
    ũ_{i,t} = u_{i,t} − (1/N_t) Σ_{j∈S_{m,t}} u_{j,t}           (si N_t > 1)
    k̂_{i,t} = ũ_{i,t} / sqrt(‖ũ_{i,t}‖_2^2 + ε_k)               (‖k̂_{i,t}‖_2 < 1)

Al madurar las N_s ≥ minimum_rows etiquetas {y_{j,s}} de una sesión pasada emitida s
(`decision_at < target_available_at <= event.at`), se forman sus puntuaciones normales de rango
g_{j,s} = Φ⁻¹((rango(y_{j,s}) − 1/2) / N_s) y se actualiza la memoria del mercado m con pérdida
interna de correlación negativa regularizada y anclaje a M_0 = (W_{1,0}, W_{2,0}):

    ℓ_IC(M; K̂_s, g_s) = −ρ(ŵ_readᵀ M(K̂_s), g_s) + Σ_{ℓ=1}^2 (μ_ℓ / 2) ‖W_ℓ − W_{ℓ,0}‖_F^2
    G̃_ℓ = recorte_F(∇_{W_ℓ} ℓ_IC(M_m; K̂_s, g_s), C)
    S_{m,ℓ} ← η_ℓ S_{m,ℓ} − (θ_ℓ / K) G̃_ℓ
    W_{m,ℓ} ← (1 − α_ℓ) W_{m,ℓ} + α_ℓ W_{ℓ,0} + S_{m,ℓ}     (paso 0; pasos > 0 sin olvido extra)

Diseño para evitar los riesgos señalados para memorias MLP de una sola pasada:
1. Gradiente interno analítico cerrado (`_inner_ic_gradients`): reproduce bit a bit el gradiente
   exacto de `torch.autograd.grad(..., create_graph=True)` sin autograd de segundo orden,
   permitiendo evaluación en `torch.no_grad()` y metagradientes de primer orden mucho más rápidos.
2. Estabilidad y acondicionamiento frente a θ bajo ruido financiero:
   - vector de lectura normalizado ŵ_read (elimina la deriva de escala que amplifica ∇_s ρ);
   - recorte suave de norma Frobenius C (estilo PT1, finito incluso en gradiente nulo);
   - puertas aprendidas por capa en caja acotada α_ℓ ∈ [α_min, 1), η_ℓ ∈ (0, η_max),
     θ_ℓ ∈ (0, θ_max), μ_ℓ > μ_min, acotando ‖M_{m,ℓ} − W_{ℓ,0}‖_F para cualquier horizonte;
   - pasos internos opcionales `inner_steps` ∈ {1..4} (1 reproduce la ecuación literal; >1 mejora
     la convergencia no lineal en dos capas sin repetir el olvido de la sesión).
3. Inicialización neutra sin doble cero: `out.weight`, `w_read` e `initial_weights` arrancan no
   nulos y únicamente la ganancia vectorial `gate` arranca en cero:
       e'_{i,t} = e_{i,t} + gate ⊙ (W_out LayerNorm(M_m(k̂_{i,t})))
   Así el predictor coincide bit a bit con su control al inicio y `gate` recibe gradiente en el
   primer paso.

Controles negativos y ablación integrados (exactamente los mismos parámetros e inicialización):
- `matured_rank`: mecanismo completo con etiquetas maduras y retención anclada a M_0.
- `shuffled_targets`: placebo que permuta deterministamente g_s dentro de cada sesión madura con
  semilla causal derivada de (seed, config.seed, mercado, decision_at).
- `eval_frozen`: actualiza en ajuste (`train=True`) y congela las escrituras en evaluación
  (`train=False`), aislando la adaptación en tiempo de prueba de la capacidad estática M_0.
- `unanchored`: sustituye el anclaje hacia M_0 por olvido y regularización L2 hacia cero,
  contrastando directamente el eje de retención anclada.
"""

import hashlib
import math
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .associative_gradient import gelu_derivative
from .config import LAYER_NORM_EPS, bounded_integer, canonical, require_identity
from .session_ic_objective import RELATIVE_FLOOR, standardized
from .session_ranks import gaussian_ranks
from .state import check_differentiable, check_finite

IC_TITANS_MLP = "ic_titans_mlp_v1"
MATURED_RANK = "matured_rank"
SHUFFLED_TARGETS = "shuffled_targets"
EVAL_FROZEN = "eval_frozen"
UNANCHORED = "unanchored"
MODES = (MATURED_RANK, SHUFFLED_TARGETS, EVAL_FROZEN, UNANCHORED)

DEPTH = 2
KEY_NORM_EPS = 1e-8
ALPHA_FLOOR = 1e-3
ETA_MAX = 0.9
THETA_MAX = 0.5
ANCHOR_FLOOR = 1e-3

MIN_DIM = 4
MAX_DIM = 128
MIN_HALF_LIFE = 1.0
MAX_HALF_LIFE = 252.0
MIN_ANCHOR = 0.01
MAX_ANCHOR = 100.0
MIN_GRAD_CLIP = 0.01
MAX_GRAD_CLIP = 100.0
MIN_INNER_STEPS = 1
MAX_INNER_STEPS = 4


def _finite_float(
    value: object, low: float, high: float, message: str, *, strict_low=False, strict_high=False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(message)
    low_ok = value > low if strict_low else value >= low
    high_ok = value < high if strict_high else value <= high
    if not (low_ok and high_ok):
        raise ValueError(message)
    return float(value)


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


def _softplus_inverse(value: float) -> float:
    """softplus⁻¹(x) = x + log(1 − exp(−x)), estable para x pequeño y grande."""
    return value + math.log(-math.expm1(-value))


def _row_index(rows: list[int], device: torch.device) -> Tensor:
    if device.type != "cuda":
        return torch.tensor(rows, dtype=torch.int64, device=device)
    return torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
        device=device, non_blocking=True
    )


def _clip_frobenius(gradient: Tensor, limit: float) -> Tensor:
    """Reescalar una matriz de gradiente para que ‖G‖_F ≤ limit sin perder finitud en cero."""
    maximum = gradient.detach().abs().amax()
    scale = maximum.clamp(min=limit)
    normalized = gradient / scale
    squared = normalized.square().sum()
    needs_clip = (maximum > limit) | (squared > 1.0)
    inverse_norm = torch.where(needs_clip, squared, torch.ones_like(squared)).rsqrt()
    clipped = normalized * limit if limit <= 1.0 else gradient * (limit / scale)
    return torch.where(needs_clip, clipped * inverse_norm, gradient)


def _apply_mlp(keys: Tensor, weights: tuple[Tensor, Tensor]) -> Tensor:
    """Aplicar la memoria MLP de 2 capas con `residual_layer_norm = True`: x + LN(W₂ GELU(W₁ x))."""
    hidden = keys @ weights[0].transpose(0, 1)
    activated = F.gelu(hidden)
    output = activated @ weights[1].transpose(0, 1)
    return keys + F.layer_norm(output, (keys.shape[-1],), eps=LAYER_NORM_EPS)


def _corr_score_gradient(scores: Tensor, target_std: Tensor) -> Tensor:
    """Gradiente exacto ∂(−ρ(s, g))/∂s con el suelo regularizado de `standardized`."""
    epsilon = torch.finfo(scores.dtype).eps
    maximum = scores.detach().abs().amax().clamp_min(epsilon)
    _, exponent = torch.frexp(maximum)
    magnitude = torch.ldexp(torch.ones_like(maximum), exponent - 1)
    scaled = scores / magnitude
    centered = scaled - scaled.mean()
    floor = RELATIVE_FLOOR * scaled.square().mean() + (epsilon / magnitude).square()
    variance = centered.square().mean() + floor
    inv_std = torch.rsqrt(variance)
    unit_scores = centered * inv_std
    rho = (unit_scores * target_std).mean()
    inv_n = 1.0 / float(scores.numel())
    g_centered = target_std - target_std.mean()
    d_scaled = (inv_n * inv_std) * (-g_centered + rho * unit_scores) + (
        (inv_n * RELATIVE_FLOOR) * (rho / variance)
    ) * scaled
    return d_scaled / magnitude


def _inner_ic_gradients(
    weights: tuple[Tensor, Tensor],
    anchor_weights: tuple[Tensor, Tensor],
    keys: Tensor,
    target_std: Tensor,
    unit_w_read: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gradientes cerrados ∇_{W₁, W₂} ℓ_IC(W₁, W₂; K̂_s, g_s) diferenciables en primer orden."""
    w1, w2 = weights
    hidden = keys @ w1.transpose(0, 1)
    activated = F.gelu(hidden)
    output = activated @ w2.transpose(0, 1)
    normalized = F.layer_norm(output, (output.shape[-1],), eps=LAYER_NORM_EPS)
    inv_std = torch.rsqrt(output.var(-1, unbiased=False, keepdim=True) + LAYER_NORM_EPS)
    prediction = keys + normalized
    scores = prediction @ unit_w_read
    d_scores = _corr_score_gradient(scores, target_std)
    d_y = d_scores.unsqueeze(-1) * unit_w_read.unsqueeze(0)
    d_z = inv_std * (
        d_y - d_y.mean(-1, keepdim=True) - normalized * (d_y * normalized).mean(-1, keepdim=True)
    )
    grad_w2 = d_z.transpose(0, 1) @ activated + mu[1] * (w2 - anchor_weights[1])
    d_a = d_z @ w2
    d_h = d_a * gelu_derivative(hidden)
    grad_w1 = d_h.transpose(0, 1) @ keys + mu[0] * (w1 - anchor_weights[0])
    return grad_w1, grad_w2


@dataclass(frozen=True)
class ICTitansMLPConfig:
    """Contrato de la memoria neuronal supervisada por IC con retención anclada."""

    dim: int = 32
    alpha_half_life: float = 21.0
    eta: float = 0.5
    theta: float = 0.05
    anchor: float = 0.1
    grad_clip: float = 1.0
    inner_steps: int = 1
    minimum_rows: int = 8
    mode: str = MATURED_RANK
    seed: int = 0

    def __post_init__(self) -> None:
        bounded_integer(self.dim, "dimensión de IC-Titans-MLP", MIN_DIM, MAX_DIM)
        alpha_half_life = _finite_float(
            self.alpha_half_life,
            MIN_HALF_LIFE,
            MAX_HALF_LIFE,
            f"alpha_half_life debe ser un número finito en [{MIN_HALF_LIFE}, {MAX_HALF_LIFE}]",
        )
        eta = _finite_float(
            self.eta,
            0.0,
            ETA_MAX,
            f"eta debe ser un número finito en (0, {ETA_MAX})",
            strict_low=True,
            strict_high=True,
        )
        theta = _finite_float(
            self.theta,
            0.0,
            THETA_MAX,
            f"theta debe ser un número finito en (0, {THETA_MAX})",
            strict_low=True,
            strict_high=True,
        )
        anchor = _finite_float(
            self.anchor,
            MIN_ANCHOR,
            MAX_ANCHOR,
            f"anchor debe ser un número finito en [{MIN_ANCHOR}, {MAX_ANCHOR}]",
        )
        grad_clip = _finite_float(
            self.grad_clip,
            MIN_GRAD_CLIP,
            MAX_GRAD_CLIP,
            f"grad_clip debe ser un número finito en [{MIN_GRAD_CLIP}, {MAX_GRAD_CLIP}]",
        )
        bounded_integer(
            self.inner_steps, "inner_steps de IC-Titans-MLP", MIN_INNER_STEPS, MAX_INNER_STEPS
        )
        bounded_integer(self.minimum_rows, "minimum_rows de IC-Titans-MLP", 3, 4096)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de IC-Titans-MLP debe ser uno de {MODES}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de IC-Titans-MLP debe ser un entero de 32 bits")
        object.__setattr__(self, "alpha_half_life", alpha_half_life)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "grad_clip", grad_clip)

    def identity(self) -> dict:
        retention = (
            "anchored_forgetting_and_frobenius_penalty_toward_initial_weights_m0"
            if self.mode != UNANCHORED
            else "unanchored_forgetting_and_l2_penalty_toward_zero"
        )
        return dict(
            schema_version=1,
            kind=IC_TITANS_MLP,
            dim=self.dim,
            depth=DEPTH,
            residual_layer_norm=True,
            alpha_half_life=self.alpha_half_life,
            alpha_floor=ALPHA_FLOOR,
            eta=self.eta,
            eta_max=ETA_MAX,
            theta=self.theta,
            theta_max=THETA_MAX,
            anchor=self.anchor,
            anchor_floor=ANCHOR_FLOOR,
            grad_clip=self.grad_clip,
            inner_steps=self.inner_steps,
            minimum_rows=self.minimum_rows,
            mode=self.mode,
            seed=self.seed,
            key_space="layer_norm_linear_cross_section_centered_smooth_l2_normalized",
            key_norm_eps=KEY_NORM_EPS,
            inner_objective="negative_regularized_session_rank_correlation_with_unit_w_read",
            inner_gradient="closed_form_analytic_differentiable_with_frobenius_clip",
            retention=retention,
            readout="layer_norm_kaiming_projection_with_zero_initialized_vector_gate_before_head",
            market_isolation="strict_per_market_fast_mlp_weights_and_momentum",
            causality="updated_only_on_matured_labels_strictly_before_emission",
        )


def parse_ic_titans_mlp(raw: object) -> ICTitansMLPConfig | None:
    """Aceptar ICTitansMLPConfig, su diccionario completo o None."""
    if raw is None:
        return None
    if type(raw) is ICTitansMLPConfig:
        raw.__post_init__()
        return raw
    expected = {field.name for field in fields(ICTitansMLPConfig)}
    if isinstance(raw, dict) and set(raw) == expected:
        return ICTitansMLPConfig(**raw)
    raise ValueError(
        "ic_titans_mlp debe declarar dim, alpha_half_life, eta, theta, anchor, grad_clip, "
        "inner_steps, minimum_rows, mode y seed, o ser None"
    )


@dataclass(frozen=True)
class MarketICMemoryState:
    """Pesos rápidos y momentum (W₁, W₂, S₁, S₂) de la memoria IC de un mercado."""

    weights: tuple[Tensor, Tensor]
    momentum: tuple[Tensor, Tensor]
    updates: int


@dataclass(frozen=True)
class ICTitansMLPState:
    """Estado de IC-Titans-MLP: memorias MLP rápidas por mercado y claves pendientes."""

    config_id: str
    parameter_id: str
    markets: dict[str, MarketICMemoryState]
    # Clave: (flow, decision_at) -> (matriz de claves emitidas [filas, dim], fila).
    # Guarda la matriz de la emisión y la fila, no un slice por fila: una sesión se reúne con
    # una sola selección y el grafo del metagradiente no crece con el número de activos.
    pending: dict[tuple[str, int], tuple[Tensor, int]]


def _gather_pending(entries: list[tuple[Tensor, int]]) -> Tensor:
    """Reunir en el orden pedido las claves pendientes [n, dim] con una selección por emisión."""
    sources: dict[int, tuple[Tensor, list[int], list[int]]] = {}
    for position, (keys, row) in enumerate(entries):
        _, positions, picked = sources.setdefault(id(keys), (keys, [], []))
        positions.append(position)
        picked.append(row)
    parts, order = [], []
    for keys, positions, picked in sources.values():
        if len(picked) == len(keys) and picked == list(range(len(keys))):
            parts.append(keys)
        else:
            index = _row_index(picked, keys.device)
            parts.append(keys.index_select(0, index))
        order.extend(positions)
    if len(parts) == 1:
        return parts[0]
    inverse = torch.empty(len(order), dtype=torch.int64)
    inverse[torch.tensor(order, dtype=torch.int64)] = torch.arange(len(order), dtype=torch.int64)
    inverse = inverse.to(parts[0].device)
    return torch.cat(parts, dim=0).index_select(0, inverse)


def _market(flow: str) -> str:
    if type(flow) is not str or "/" not in flow:
        raise ValueError("Cada flujo de IC-Titans-MLP necesita su prefijo de mercado")
    return flow.split("/", 1)[0]


class ICTitansMLP(nn.Module):
    """Memoria neuronal MLP de 2 capas compartida por mercado supervisada por IC de rangos."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        config: ICTitansMLPConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión oculta de IC-Titans-MLP", 1, 4096)
        bounded_integer(seed, "semilla de IC-Titans-MLP", 0, 2**32 - 1)
        if type(config) is not ICTitansMLPConfig:
            raise ValueError("IC-Titans-MLP necesita ICTitansMLPConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("IC-Titans-MLP admite precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        # Semilla aislada e independiente de `mode`: candidato y placebos arrancan idénticos.
        derived = hashlib.sha256(f"{IC_TITANS_MLP}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.key_proj = nn.Linear(hidden_size, config.dim, bias=False, dtype=dtype)
            self.initial_weights = nn.ParameterList(
                nn.Parameter(torch.empty(config.dim, config.dim, dtype=dtype)) for _ in range(DEPTH)
            )
            for weight in self.initial_weights:
                nn.init.xavier_uniform_(weight)
            self.w_read = nn.Parameter(torch.empty(config.dim, dtype=dtype))
            nn.init.normal_(self.w_read)
            with torch.no_grad():
                self.w_read.div_(self.w_read.norm().clamp_min(KEY_NORM_EPS))
            self.readout_norm = nn.LayerNorm(config.dim, eps=LAYER_NORM_EPS, dtype=dtype)
            self.out = nn.Linear(config.dim, hidden_size, bias=False, dtype=dtype)
        # Única ganancia inicializada a cero sobre `out` Kaiming: sin doble cero en serie.
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        rate = math.log(2.0) / config.alpha_half_life
        alpha_init = -math.expm1(-rate)
        alpha_logit = _logit((alpha_init - ALPHA_FLOOR) / (1.0 - ALPHA_FLOOR))
        eta_logit = _logit(config.eta / ETA_MAX)
        theta_logit = _logit(config.theta / THETA_MAX)
        anchor_logit = _softplus_inverse(config.anchor - ANCHOR_FLOOR)
        self.alpha_logits = nn.Parameter(torch.full((DEPTH,), alpha_logit, dtype=dtype))
        self.eta_logits = nn.Parameter(torch.full((DEPTH,), eta_logit, dtype=dtype))
        self.theta_logits = nn.Parameter(torch.full((DEPTH,), theta_logit, dtype=dtype))
        self.anchor_logits = nn.Parameter(torch.full((DEPTH,), anchor_logit, dtype=dtype))
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
            raise ValueError("IC-Titans-MLP admite precisión FP32 o FP64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("IC-Titans-MLP exige FP32 o FP64 explícitos, sin autocast")
        return reference

    def _gates(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * torch.sigmoid(self.alpha_logits)
        eta = ETA_MAX * torch.sigmoid(self.eta_logits)
        theta = THETA_MAX * torch.sigmoid(self.theta_logits)
        mu = ANCHOR_FLOOR + F.softplus(self.anchor_logits)
        return alpha, eta, theta, mu

    def initial_state(self, parameter_id: str) -> ICTitansMLPState:
        return ICTitansMLPState(self.fingerprint(), parameter_id, {}, {})

    def validate_state(self, state: object, parameter_id: str) -> None:
        reference = self._reference()
        dim = self.config.dim
        if (
            not isinstance(state, ICTitansMLPState)
            or state.config_id != self.fingerprint()
            or state.parameter_id != parameter_id
            or not isinstance(state.markets, dict)
            or not isinstance(state.pending, dict)
        ):
            raise ValueError("El estado de IC-Titans-MLP no corresponde al módulo")
        for market, entry in state.markets.items():
            if (
                type(market) is not str
                or not market
                or not isinstance(entry, MarketICMemoryState)
                or type(entry.updates) is not int
                or entry.updates < 1
                or type(entry.weights) is not tuple
                or type(entry.momentum) is not tuple
                or len(entry.weights) != DEPTH
                or len(entry.momentum) != DEPTH
            ):
                raise ValueError("El estado de mercado de IC-Titans-MLP no conserva su contrato")
            for tensor in (*entry.weights, *entry.momentum):
                if (
                    not isinstance(tensor, Tensor)
                    or tensor.layout != torch.strided
                    or tensor.shape != (dim, dim)
                    or tensor.dtype != reference.dtype
                    or tensor.device != reference.device
                ):
                    raise ValueError(
                        "Los tensores de mercado de IC-Titans-MLP no conservan su contrato"
                    )
                check_finite(tensor, "El estado rápido de IC-Titans-MLP")

    def keys(
        self,
        encoded: Tensor,
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
    ) -> Tensor:
        """Claves transversales centradas por sesión y mercado con norma L2 suave < 1."""
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
                "IC-Titans-MLP necesita una representación, un mercado y un instante por fila"
            )
        check_finite(encoded, "La representación de entrada de IC-Titans-MLP")
        raw = self.key_proj(self.norm(encoded))
        groups: dict[tuple[str, int], list[int]] = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            groups.setdefault(key, []).append(row)
        if len(groups) == 1:
            centered = raw - raw.mean(dim=0, keepdim=True) if len(raw) > 1 else raw
            norm = torch.sqrt(centered.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
            result = centered / norm
        else:
            result = torch.zeros_like(raw)
            for members in groups.values():
                index = _row_index(members, raw.device)
                part = raw.index_select(0, index)
                if len(members) > 1:
                    part = part - part.mean(dim=0, keepdim=True)
                norm = torch.sqrt(part.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
                result = result.index_copy(0, index, part / norm)
        check_finite(result, "Las claves transversales de IC-Titans-MLP")
        return result

    def read_memory(
        self,
        keys: Tensor,
        markets: tuple[str, ...] | list[str],
        state: ICTitansMLPState,
    ) -> Tensor:
        """Leer M_m(k̂_{i,t}) ∈ R^d para cada activo desde el estado de su mercado (o M_0)."""
        reference = self._reference()
        dim = self.config.dim
        if (
            not isinstance(keys, Tensor)
            or keys.layout != torch.strided
            or keys.ndim != 2
            or keys.shape[1] != dim
            or len(keys) == 0
            or keys.dtype != reference.dtype
            or keys.device != reference.device
            or len(markets) != len(keys)
        ):
            raise ValueError(
                "La lectura de IC-Titans-MLP necesita claves [filas, dim] y un mercado por fila"
            )
        check_finite(keys, "Las claves de consulta de IC-Titans-MLP")
        default_weights = (self.initial_weights[0], self.initial_weights[1])
        by_market: dict[str, list[int]] = {}
        for row, market in enumerate(markets):
            by_market.setdefault(market, []).append(row)
        if len(by_market) == 1:
            (market,) = by_market
            entry = state.markets.get(market)
            weights = default_weights if entry is None else entry.weights
            readout = _apply_mlp(keys, weights)
        else:
            readout = keys.new_zeros((len(keys), dim))
            for market, members in by_market.items():
                entry = state.markets.get(market)
                weights = default_weights if entry is None else entry.weights
                index = _row_index(members, keys.device)
                part = _apply_mlp(keys.index_select(0, index), weights)
                readout = readout.index_copy(0, index, part)
        check_finite(readout, "La lectura interna de IC-Titans-MLP")
        return readout

    def read(
        self,
        keys: Tensor,
        markets: tuple[str, ...] | list[str],
        state: ICTitansMLPState,
    ) -> Tensor:
        """Residuo [N, H] sumado a la representación antes de la cabeza."""
        readout = self.read_memory(keys, markets, state)
        delta = self.gate * self.out(self.readout_norm(readout))
        check_finite(delta, "El residuo de IC-Titans-MLP")
        return delta

    def stage(
        self,
        state: ICTitansMLPState,
        flow_ids: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        keys: Tensor,
        *,
        differentiable: bool = False,
    ) -> ICTitansMLPState:
        """Registrar las claves emitidas de una sesión a la espera de maduración."""
        check_differentiable(differentiable)
        reference = self._reference()
        if (
            not isinstance(keys, Tensor)
            or keys.ndim != 2
            or keys.shape != (len(flow_ids), self.config.dim)
            or keys.dtype != reference.dtype
            or keys.device != reference.device
            or len(prediction_at) != len(flow_ids)
        ):
            raise ValueError(
                "El registro pendiente de IC-Titans-MLP necesita claves [filas, dim] por fila"
            )
        check_finite(keys, "Las claves pendientes de IC-Titans-MLP")
        stored_keys = keys if differentiable else keys.detach()
        pending = dict(state.pending)
        for row, (flow, at) in enumerate(zip(flow_ids, prediction_at, strict=True)):
            entry_key = (flow, int(at))
            if entry_key in pending:
                raise ValueError("IC-Titans-MLP ya tiene registrada esa emisión pendiente")
            pending[entry_key] = (stored_keys, row)
        return replace(state, pending=pending)

    def _shuffle_permutation(self, market: str, decision_at: int, count: int) -> Tensor:
        text = f"{IC_TITANS_MLP}/shuffle/{self.seed}/{self.config.seed}/{market}/{decision_at}"
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(count, generator=generator)

    def settle(
        self,
        state: ICTitansMLPState,
        matured: list[tuple[str, int, float]] | tuple[tuple[str, int, float], ...],
        *,
        differentiable: bool = False,
        train: bool = True,
    ) -> ICTitansMLPState:
        """Actualizar las memorias MLP rápidas con etiquetas ya maduras en orden canónico."""
        check_differentiable(differentiable)
        if type(train) is not bool:
            raise ValueError("train debe ser booleano")
        if not matured:
            return state
        reference = self._reference()
        pending = dict(state.pending)
        grouped: dict[tuple[int, str], list[tuple[str, Tensor, int, float]]] = {}
        for flow, decision_at, outcome in matured:
            if (
                isinstance(outcome, bool)
                or not isinstance(outcome, (int, float))
                or not math.isfinite(outcome)
            ):
                raise ValueError("El resultado maduro para IC-Titans-MLP debe ser un real finito")
            entry_key = (flow, int(decision_at))
            if entry_key not in pending:
                raise ValueError("La etiqueta madura de IC-Titans-MLP no tiene clave pendiente")
            grouped.setdefault((int(decision_at), _market(flow)), []).append(
                (flow, *pending.pop(entry_key), float(outcome))
            )
        if self.config.mode == EVAL_FROZEN and not train:
            return replace(state, pending=pending)

        markets = dict(state.markets)
        anchored = self.config.mode != UNANCHORED
        step_scale = 1.0 / float(self.config.inner_steps)
        with torch.set_grad_enabled(differentiable):
            alpha, eta, theta, mu = self._gates()
            unit_w_read = self.w_read / torch.sqrt(self.w_read.square().sum() + KEY_NORM_EPS)
            w0 = (self.initial_weights[0], self.initial_weights[1])
            zero_anchor = (torch.zeros_like(w0[0]), torch.zeros_like(w0[1]))
            anchor_weights = w0 if anchored else zero_anchor
            for decision_at, market in sorted(grouped):
                rows = sorted(grouped[decision_at, market], key=lambda item: item[0])
                if len(rows) < self.config.minimum_rows:
                    continue
                k_mat = _gather_pending([(item[1], item[2]) for item in rows])
                if not differentiable:
                    k_mat = k_mat.detach()
                outcomes = torch.tensor(
                    [item[3] for item in rows], dtype=reference.dtype, device=reference.device
                )
                present = torch.ones((len(rows), 1), dtype=torch.bool, device=reference.device)
                normal_ranks = gaussian_ranks(outcomes.unsqueeze(-1), present).squeeze(-1)
                if self.config.mode == SHUFFLED_TARGETS:
                    perm = self._shuffle_permutation(market, decision_at, len(rows)).to(
                        reference.device
                    )
                    normal_ranks = normal_ranks.index_select(0, perm)
                target_std = standardized(normal_ranks).detach()
                previous = markets.get(market)
                if previous is None:
                    cur_w = w0 if differentiable else (w0[0].detach(), w0[1].detach())
                    cur_s = (torch.zeros_like(w0[0]), torch.zeros_like(w0[1]))
                    updates = 1
                else:
                    cur_w = (
                        previous.weights
                        if differentiable
                        else (previous.weights[0].detach(), previous.weights[1].detach())
                    )
                    cur_s = (
                        previous.momentum
                        if differentiable
                        else (previous.momentum[0].detach(), previous.momentum[1].detach())
                    )
                    updates = previous.updates + 1
                for inner in range(self.config.inner_steps):
                    grads = _inner_ic_gradients(
                        cur_w, anchor_weights, k_mat, target_std, unit_w_read, mu
                    )
                    next_w, next_s = [], []
                    for idx in range(DEPTH):
                        clipped = _clip_frobenius(grads[idx], self.config.grad_clip)
                        s_new = eta[idx] * cur_s[idx] - (theta[idx] * step_scale) * clipped
                        if inner == 0:
                            if anchored:
                                w_new = (
                                    (1.0 - alpha[idx]) * cur_w[idx] + alpha[idx] * w0[idx] + s_new
                                )
                            else:
                                w_new = (1.0 - alpha[idx]) * cur_w[idx] + s_new
                        else:
                            w_new = cur_w[idx] + s_new
                        next_s.append(s_new)
                        next_w.append(w_new)
                    cur_w = (next_w[0], next_w[1])
                    cur_s = (next_s[0], next_s[1])
                if not differentiable:
                    cur_w = (cur_w[0].detach(), cur_w[1].detach())
                    cur_s = (cur_s[0].detach(), cur_s[1].detach())
                for tensor in (*cur_w, *cur_s):
                    check_finite(tensor, "La actualización rápida de IC-Titans-MLP")
                markets[market] = MarketICMemoryState(cur_w, cur_s, updates)
        return replace(state, markets=markets, pending=pending)

    def detach_state(
        self, state: ICTitansMLPState, parameter_id: str | None = None
    ) -> ICTitansMLPState:
        """Cortar el grafo de autograd en las fronteras de truncamiento y compactar pendientes."""
        markets = {
            market: MarketICMemoryState(
                (entry.weights[0].detach(), entry.weights[1].detach()),
                (entry.momentum[0].detach(), entry.momentum[1].detach()),
                entry.updates,
            )
            for market, entry in state.markets.items()
        }
        remaining: dict[int, tuple[Tensor, list[tuple[tuple[str, int], int]]]] = {}
        for key, (keys, row) in state.pending.items():
            remaining.setdefault(id(keys), (keys, []))[1].append((key, row))
        pending = {}
        for keys, entries in remaining.values():
            if len(entries) < len(keys):
                index = _row_index([row for _, row in entries], keys.device)
                compact_keys = keys.detach().index_select(0, index)
                for position, (key, _) in enumerate(entries):
                    pending[key] = (compact_keys, position)
            else:
                clean = keys.detach()
                for key, row in entries:
                    pending[key] = (clean, row)
        pending = {key: pending[key] for key in state.pending}
        return replace(
            state,
            parameter_id=parameter_id or state.parameter_id,
            markets=markets,
            pending=pending,
        )

    def export_state(self, state: ICTitansMLPState) -> dict:
        """Exportar copias CPU sin grafo para checkpoints deterministas."""
        detached = self.detach_state(state)
        keys_sorted = sorted(detached.pending)
        if keys_sorted:
            pending_keys = _gather_pending([detached.pending[k] for k in keys_sorted]).cpu().clone()
        else:
            pending_keys = torch.empty((0, self.config.dim), dtype=self._reference().dtype)
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            markets={
                market: dict(
                    weights=tuple(w.cpu().clone() for w in entry.weights),
                    momentum=tuple(m.cpu().clone() for m in entry.momentum),
                    updates=entry.updates,
                )
                for market, entry in sorted(detached.markets.items())
            },
            pending=dict(
                flows=[flow for flow, _ in keys_sorted],
                decision_at=torch.tensor([at for _, at in keys_sorted], dtype=torch.int64),
                keys=pending_keys,
            ),
        )

    def restore_state(
        self, payload: object, parameter_id: str, *, device: torch.device | str
    ) -> ICTitansMLPState:
        """Validar y restaurar el estado exportado sobre `device`."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "configuration", "markets", "pending"}
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.config.identity())
            or not isinstance(payload["markets"], dict)
            or not isinstance(payload["pending"], dict)
            or set(payload["pending"]) != {"flows", "decision_at", "keys"}
        ):
            raise ValueError("El estado guardado de IC-Titans-MLP no corresponde al predictor")
        reference = self._reference()
        target_device = torch.device(device)
        markets = {}
        for market, entry in payload["markets"].items():
            if (
                not isinstance(entry, dict)
                or set(entry) != {"weights", "momentum", "updates"}
                or not isinstance(entry["weights"], (tuple, list))
                or not isinstance(entry["momentum"], (tuple, list))
                or len(entry["weights"]) != DEPTH
                or len(entry["momentum"]) != DEPTH
            ):
                raise ValueError("El estado de mercado guardado de IC-Titans-MLP está incompleto")
            weights = tuple(
                w.to(device=target_device, dtype=reference.dtype, copy=True)
                for w in entry["weights"]
            )
            momentum = tuple(
                m.to(device=target_device, dtype=reference.dtype, copy=True)
                for m in entry["momentum"]
            )
            markets[market] = MarketICMemoryState(
                (weights[0], weights[1]),
                (momentum[0], momentum[1]),
                int(entry["updates"]),
            )
        raw_pending = payload["pending"]
        flows = raw_pending["flows"]
        moments = raw_pending["decision_at"].tolist()
        key_tensor = raw_pending["keys"].to(device=target_device, dtype=reference.dtype, copy=True)
        if not (len(flows) == len(moments) == len(key_tensor)):
            raise ValueError("Las claves pendientes guardadas de IC-Titans-MLP no concilian")
        pending = {
            (flow, int(at)): (key_tensor, row)
            for row, (flow, at) in enumerate(zip(flows, moments, strict=True))
        }
        state = ICTitansMLPState(self.fingerprint(), parameter_id, markets, pending)
        self.validate_state(state, parameter_id)
        return state
