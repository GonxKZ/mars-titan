"""Memoria asociativa con sesgo atencional contrastivo transversal (InfoNCE-Titans, Idea 5).

En el marco MIRAS (Behrouz et al., 2025, arXiv:2504.13173), Titans elige un sesgo atencional L2
punto a punto por flujo. Esta extensión añade una memoria asociativa neuronal compartida por
mercado M_cs (MLP residual de dos capas con LayerNorm, igual que la memoria rápida del núcleo)
cuya pérdida interna al cerrar cada sesión S_t sustituye L2 por InfoNCE transversal frente a P
prototipos deterministas de la sesión:

    ŷ_i = norm(M_cs(k_i)),        v̂_i = norm(v_i),
    ℓ_cs(M_cs; S_t) = −(1/N_t) Σ_i log [ exp(ŷ_i·v̂_i / τ)
                                         / (exp(ŷ_i·v̂_i / τ) + Σ_{p=1}^P exp(ŷ_i·v̄_p / τ)) ]

con norm(x) = x / √(‖x‖₂² + ε) suave y finita incluso en cero. Para evitar el coste O(N²) de
todos los pares entre ~2.700–4.193 acciones, cuando N_t ≤ P se usan todos los valores v̂_i y
cuando N_t > P se construyen P centroides esféricos blandos deterministas a partir de P semillas
esféricas máximamente separadas (k-center determinista por mínima similitud coseno máxima,
partiendo del vector más alejado de la media de la sesión), en una sola pasada matricial sin
operaciones atómicas ni dispersión no determinista. El gradiente interno ∂ℓ_cs/∂W se escribe en
forma cerrada diferenciable: coincide con `torch.autograd.grad(..., create_graph=True)` hasta
precisión de máquina en FP64, evita autograd de segundo orden en el bucle interno y corre en
`no_grad` durante validación y evaluación.

Causalidad: al emitir la sesión t solo se consulta el estado confirmado de sesiones estrictamente
anteriores (s < t), aislado por mercado (US y CN). La escritura ocurre una sola vez por sesión
después de emitir todas sus filas, ordenadas canónicamente por identificador de flujo para que la
partición en bloques no altere los pesos confirmados. La lectura entra como residuo tras MAC y la
rama técnica mediante una única proyección lineal inicializada a cero, de modo que el modelo
arranca bit a bit igual que su control y recibe gradiente desde el primer paso sin encadenar dos
ceros en serie.

Controles negativos integrados (exactamente los mismos parámetros e inicialización):
- `infonce`: candidato principal (InfoNCE transversal en ajuste y evaluación).
- `shuffled_values`: placebo con los mismos prototipos pero valores positivos emparejados mediante
  un desarreglo determinista de Sattolo sin puntos fijos (rompe la correspondencia clave-valor).
- `eval_frozen`: placebo idéntico en ajuste pero congelado en M_0 durante evaluación (sin
  escritura en tiempo de prueba).
- `l2_shared`: ablación que conserva arquitectura, normalización y posición pero sustituye InfoNCE
  por la pérdida L2 normalizada punto a punto sin repulsión contrastiva.
"""

import hashlib
import math
import random
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .associative_gradient import gelu_derivative
from .config import (
    LAYER_NORM_EPS,
    GateBias,
    bounded_integer,
    canonical,
    require_identity,
)
from .state import check_differentiable, check_finite

CONTRASTIVE_MEMORY = "contrastive_memory_v1"
MODES = ("infonce", "shuffled_values", "eval_frozen", "l2_shared")
MARKETS = ("US", "CN")
MAX_CONTRASTIVE_ROWS = 16384
THETA_MAX = 0.1
NORM_EPS = 1e-6
PROTOTYPE_TEMPERATURE = 0.1

_ROLES = {
    "infonce": "cross_sectional_contrastive_memory_variant",
    "shuffled_values": "shuffled_session_values_placebo",
    "eval_frozen": "eval_frozen_placebo",
    "l2_shared": "normalized_l2_shared_ablation",
}


def _seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{CONTRASTIVE_MEMORY}/{name}/{seed}".encode()).digest()
    return int.from_bytes(digest[:7], "little")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    return float(value)


def _moved(value, device):
    if isinstance(value, Tensor):
        return value.detach().to(device=device, copy=True).contiguous()
    if isinstance(value, tuple):
        return tuple(_moved(item, device) for item in value)
    if isinstance(value, list):
        return [_moved(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _moved(item, device) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ContrastiveMemoryConfig:
    """Hiperparámetros identificados de InfoNCE-Titans y sus controles negativos."""

    prototypes: int = 32
    temperature: float = 0.2
    mode: str = "infonce"

    def __post_init__(self) -> None:
        bounded_integer(self.prototypes, "prototypes de la memoria contrastiva", 2, 256)
        tau = _finite_number(self.temperature, "temperature de la memoria contrastiva")
        if not 0.05 <= tau <= 10.0:
            raise ValueError("temperature de la memoria contrastiva debe pertenecer a [0,05, 10]")
        object.__setattr__(self, "temperature", tau)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"mode de la memoria contrastiva debe ser uno de {MODES}")

    def identity(self) -> dict:
        return dict(
            schema_version=1,
            kind=CONTRASTIVE_MEMORY,
            prototypes=self.prototypes,
            temperature=self.temperature,
            mode=self.mode,
            role=_ROLES[self.mode],
            memory_architecture="two_layer_residual_layer_norm_mlp_gelu_exact",
            read="previous_committed_sessions_only_l2_normalized_output",
            write="once_per_session_after_emission_canonical_flow_order",
            inner_gradient="closed_form_analytic_matching_autograd_v1",
            prototype_rule=(
                "all_rows_if_at_most_p_else_farthest_cosine_seeded_soft_spherical_centroids"
            ),
            prototype_temperature=PROTOTYPE_TEMPERATURE,
            norm_eps=NORM_EPS,
            gates="mean_session_token_sigmoid_alpha_eta_and_scaled_theta",
            gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05, theta_max=THETA_MAX),
            position="after_mac_and_technical_branch_before_head",
            output="residual_with_zero_initialized_linear_projection",
            market_isolation="separate_fast_state_per_market",
        )


def parse_contrastive_memory(raw: object) -> ContrastiveMemoryConfig | None:
    """Aceptar ContrastiveMemoryConfig, su diccionario completo o None."""
    if raw is None:
        return None
    if type(raw) is ContrastiveMemoryConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == {"prototypes", "temperature", "mode"}:
        return ContrastiveMemoryConfig(**raw)
    raise ValueError("contrastive_memory debe declarar prototypes, temperature y mode, o ser None")


def safe_l2_normalize(values: Tensor, eps: float = NORM_EPS) -> tuple[Tensor, Tensor]:
    """Normalización L2 suave x / √(‖x‖₂² + ε), inmune a cero y a desbordamiento en FP32.

    Devuelve `(x_hat, inv_r)` donde `inv_r = (‖x‖₂² + ε)^(−1/2)`. Su jacobiano exacto en todo R^d
    es `inv_r * (I − x_hat x_hatᵀ)`, finito también cuando `x = 0`.
    """
    scale = values.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    scaled = values / scale
    inv_norm = torch.rsqrt(scaled.square().sum(dim=-1, keepdim=True) + (eps / scale) / scale)
    return scaled * inv_norm, inv_norm / scale


def session_prototypes(v_hat: Tensor, count: int) -> Tensor:
    """Construir hasta `count` prototipos deterministas sobre valores normalizados [N, d].

    Si N ≤ P devuelve todos los valores normalizados de la sesión. Si N > P elige P semillas
    esféricas máximamente separadas (k-center determinista por mínima similitud coseno máxima,
    partiendo del vector más alejado de la media de la sesión) y aplica un paso matricial de
    centroides esféricos blandos sin operaciones atómicas ni generador aleatorio.
    """
    rows = v_hat.shape[0]
    if rows <= count:
        return v_hat
    detached = v_hat.detach()
    mean_dir, _ = safe_l2_normalize(detached.mean(dim=0, keepdim=True))
    first = (detached @ mean_dir.squeeze(0)).argmin().unsqueeze(0)
    seed_rows = [detached.index_select(0, first)]
    max_sim = detached @ seed_rows[0].squeeze(0)
    for _ in range(1, count):
        next_index = max_sim.argmin().unsqueeze(0)
        seed = detached.index_select(0, next_index)
        seed_rows.append(seed)
        max_sim = torch.maximum(max_sim, detached @ seed.squeeze(0))
    seeds = torch.cat(seed_rows, dim=0)
    similarities = detached @ seeds.T
    assignments = torch.softmax(similarities / PROTOTYPE_TEMPERATURE, dim=-1)
    centroids = assignments.T @ v_hat
    prototypes, _ = safe_l2_normalize(centroids)
    return prototypes


def _index_tensor(indices: list[int], device: torch.device) -> Tensor:
    if device.type == "cuda":
        return torch.tensor(indices, dtype=torch.int64, pin_memory=True).to(
            device=device, non_blocking=True
        )
    return torch.tensor(indices, dtype=torch.int64, device=device)


def derange_values(values: Tensor, *, seed: int, market: int, moment: int) -> Tensor:
    """Desarreglo determinista de Sattolo sin puntos fijos (N ≥ 2) para el placebo."""
    count = values.shape[0]
    if count <= 1:
        return values
    digest = hashlib.sha256(
        f"{CONTRASTIVE_MEMORY}/derange/{seed}/{market}/{moment}/{count}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    order = list(range(count))
    for i in range(count - 1, 0, -1):
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    index = _index_tensor(order, values.device)
    return values.index_select(0, index)


def contrastive_inner_gradients(
    weights: tuple[Tensor, Tensor],
    keys: Tensor,
    v_pos: Tensor,
    prototypes: Tensor,
    *,
    temperature: float,
    loss_mode: str = "infonce",
) -> tuple[tuple[Tensor, Tensor], Tensor]:
    """Gradientes internos cerrados (∂ℓ/∂W₁, ∂ℓ/∂W₂) y pérdida escalar sobre la sesión.

    Todas las operaciones son diferenciables de primer orden: con grafo exterior, su
    retropropagación coincide con el metagradiente de segundo orden de
    `torch.autograd.grad(..., create_graph=True)` sin invocar doble autograd y sin materializar
    tensores [N, d, d].
    """
    if len(weights) != 2:
        raise ValueError("La memoria contrastiva requiere exactamente dos capas de pesos")
    if loss_mode not in ("infonce", "l2_shared"):
        raise ValueError("loss_mode debe ser infonce o l2_shared")
    w1, w2 = weights
    rows, dim = keys.shape
    hidden = keys @ w1.T
    activated = F.gelu(hidden)
    output = activated @ w2.T
    normalized = F.layer_norm(output, (dim,), eps=LAYER_NORM_EPS)
    inv_std = torch.rsqrt(output.var(-1, unbiased=False, keepdim=True) + LAYER_NORM_EPS)
    y = keys + normalized
    y_hat, inv_r = safe_l2_normalize(y)

    if loss_mode == "infonce":
        pos = (y_hat * v_pos).sum(dim=-1, keepdim=True) / temperature
        neg = (y_hat @ prototypes.T) / temperature
        logits = torch.cat((pos, neg), dim=-1)
        loss = (torch.logsumexp(logits, dim=-1) - logits[:, 0]).mean()
        probs = torch.softmax(logits, dim=-1)
        # ∂ℓ_i/∂z_{i,0} = p_{i,0} − 1 = −Σ_{p≥1} p_{i,p}; evita cancelación cuando p_{i,0} ≈ 1
        # y anula exactamente cualquier modo común compartido entre v_pos y prototypes.
        neg_probs = probs[:, 1:]
        grad_y_hat = (neg_probs @ prototypes - neg_probs.sum(dim=-1, keepdim=True) * v_pos) / (
            rows * temperature
        )
    else:
        diff = y_hat - v_pos
        loss = diff.square().sum(dim=-1).mean()
        grad_y_hat = (2.0 / rows) * diff

    dot = (grad_y_hat * y_hat).sum(dim=-1, keepdim=True)
    grad_y = (grad_y_hat - dot * y_hat) * inv_r
    grad_output = inv_std * (
        grad_y
        - grad_y.mean(dim=-1, keepdim=True)
        - normalized * (grad_y * normalized).mean(dim=-1, keepdim=True)
    )
    grad_w2 = grad_output.T @ activated
    grad_hidden = (grad_output @ w2) * gelu_derivative(hidden)
    grad_w1 = grad_hidden.T @ keys
    return (grad_w1, grad_w2), loss


@dataclass(frozen=True)
class ContrastiveMarketState:
    """Pesos rápidos, momentum y contador de sesiones confirmadas de un mercado."""

    weights: tuple[Tensor, Tensor]
    momentum: tuple[Tensor, Tensor]
    steps: int


@dataclass(frozen=True)
class ContrastiveMemoryState:
    """Estado explícito de la memoria contrastiva por mercado y tokens pendientes de sesión."""

    config_id: str
    parameter_id: str
    committed_at: tuple[int, ...]
    pending_at: tuple[int, ...]
    pending_flows: tuple[tuple[str, ...], ...]
    pending_tokens: tuple[Tensor | None, ...]
    markets: tuple[ContrastiveMarketState, ...]


class ContrastiveMemory(nn.Module):
    """Memoria asociativa transversal InfoNCE-Titans compartida por mercado y escrita por sesión."""

    def __init__(
        self,
        hidden_size: int,
        config: ContrastiveMemoryConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión de la memoria contrastiva", 2, 4096)
        bounded_integer(seed, "semilla de la memoria contrastiva", 0, 2**32 - 1)
        if type(config) is not ContrastiveMemoryConfig:
            raise ValueError("La memoria contrastiva necesita ContrastiveMemoryConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("La memoria contrastiva exige precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        gate_logits = GateBias(alpha_half_life=256.0, eta=0.5, theta=0.05).logits(THETA_MAX)

        # Generador derivado aislado en CPU: no consume RNG global y es idéntico en los 4 modos.
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_seed(seed, "parameters"))
            self.initial_weights = nn.ParameterList(
                nn.Parameter(torch.empty(hidden_size, hidden_size, dtype=dtype)) for _ in range(2)
            )
            for weight in self.initial_weights:
                nn.init.xavier_uniform_(weight)
            self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
            self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
            self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
            self.alpha_projection = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
            self.eta_projection = nn.Linear(hidden_size, 1, bias=True, dtype=dtype)
            self.theta_projection = nn.Linear(hidden_size, 1, bias=True, dtype=dtype)
            with torch.no_grad():
                for gate, value in zip(
                    (self.alpha_projection, self.eta_projection, self.theta_projection),
                    gate_logits,
                    strict=True,
                ):
                    gate.bias.fill_(value)
            # Única proyección lineal a cero: arranca igual que el control sin dos ceros en serie.
            self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
            nn.init.zeros_(self.out_projection.weight)

        self._fingerprint = hashlib.sha256(canonical(self.config.identity()).encode()).hexdigest()

    def fingerprint(self) -> str:
        return self._fingerprint

    def get_extra_state(self) -> dict:
        return dict(configuration=self.config.identity(), hidden_size=self.hidden_size)

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _reference(self) -> Tensor:
        reference = self.out_projection.weight
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("La memoria contrastiva exige precisión FP32 o FP64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("La memoria contrastiva exige FP32 o FP64 explícitos, sin autocast")
        return reference

    def _initial_market(self, *, differentiable: bool) -> ContrastiveMarketState:
        with torch.set_grad_enabled(differentiable):
            weights = tuple(
                w.clone() if differentiable else w.detach().clone() for w in self.initial_weights
            )
            momentum = tuple(torch.zeros_like(w) for w in weights)
        return ContrastiveMarketState(weights, momentum, 0)

    def initial_state(
        self, parameter_id: str, *, differentiable: bool = False
    ) -> ContrastiveMemoryState:
        check_differentiable(differentiable)
        self._reference()
        empty = (-1,) * len(MARKETS)
        markets = tuple(self._initial_market(differentiable=differentiable) for _ in MARKETS)
        return ContrastiveMemoryState(
            self.fingerprint(),
            parameter_id,
            empty,
            empty,
            ((),) * len(MARKETS),
            (None,) * len(MARKETS),
            markets,
        )

    def validate(
        self, state: object, parameter_id: str, *, device: torch.device | None = None
    ) -> None:
        reference = self._reference()
        expected_device = reference.device if device is None else torch.device(device)
        if (
            not isinstance(state, ContrastiveMemoryState)
            or state.config_id != self.fingerprint()
            or state.parameter_id != parameter_id
            or len(state.committed_at) != len(MARKETS)
            or len(state.pending_at) != len(MARKETS)
            or len(state.pending_flows) != len(MARKETS)
            or len(state.pending_tokens) != len(MARKETS)
            or len(state.markets) != len(MARKETS)
        ):
            raise ValueError("El estado de la memoria contrastiva no corresponde al predictor")
        expected_shape = (self.hidden_size, self.hidden_size)
        storages: set[tuple[torch.device, int]] = set()
        for market in state.markets:
            if (
                not isinstance(market, ContrastiveMarketState)
                or type(market.weights) is not tuple
                or type(market.momentum) is not tuple
                or len(market.weights) != 2
                or len(market.momentum) != 2
                or type(market.steps) is not int
                or market.steps < 0
            ):
                raise ValueError("El estado de mercado de la memoria contrastiva no es válido")
            for tensor in (*market.weights, *market.momentum):
                if (
                    not isinstance(tensor, Tensor)
                    or tensor.layout != torch.strided
                    or tensor.shape != expected_shape
                    or tensor.dtype != reference.dtype
                    or tensor.device != expected_device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "Los tensores de la memoria contrastiva no conservan forma, tipo "
                        "ni dispositivo"
                    )
                key = (tensor.device, tensor.untyped_storage().data_ptr())
                if key in storages:
                    raise ValueError(
                        "Los tensores de la memoria contrastiva no pueden compartir almacenamiento"
                    )
                storages.add(key)
                check_finite(tensor, "El estado de la memoria contrastiva")

    @staticmethod
    def _market(flow: str) -> int:
        prefix = flow.split("/", 1)[0]
        if prefix not in MARKETS:
            raise ValueError("El flujo de la memoria contrastiva no pertenece a US ni a CN")
        return MARKETS.index(prefix)

    def _session_rows(
        self,
        tokens: Tensor,
        flow_ids: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: ContrastiveMemoryState,
    ):
        reference = self._reference()
        if (
            not isinstance(tokens, Tensor)
            or tokens.layout != torch.strided
            or tokens.ndim != 2
            or tokens.shape != (len(flow_ids), self.hidden_size)
            or len(flow_ids) != len(prediction_at)
            or not flow_ids
            or tokens.dtype != reference.dtype
            or tokens.device != reference.device
        ):
            raise ValueError("Las filas de entrada no corresponden a la memoria contrastiva")
        check_finite(tokens, "Las entradas de la memoria contrastiva")
        markets = [self._market(flow) for flow in flow_ids]
        for market in sorted(set(markets)):
            rows = [index for index, value in enumerate(markets) if value == market]
            moments = {prediction_at[index] for index in rows}
            if len(moments) != 1:
                raise ValueError("Las filas de un mercado en un bloque deben ser de una sesión")
            (moment,) = moments
            if moment <= state.committed_at[market]:
                raise ValueError("La sesión ya se confirmó: no puede leer ni escribir de nuevo")
            if state.pending_at[market] not in (-1, moment):
                raise ValueError("La sesión anterior sigue pendiente de confirmar")
            names = tuple(flow_ids[index] for index in rows)
            yield market, rows, moment, names

    def _apply_memory(self, queries: Tensor, weights: tuple[Tensor, Tensor]) -> Tensor:
        w1, w2 = weights
        hidden = F.gelu(queries @ w1.T)
        output = hidden @ w2.T
        return queries + F.layer_norm(output, (self.hidden_size,), eps=LAYER_NORM_EPS)

    def read(
        self,
        encoded: Tensor,
        flow_ids: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: ContrastiveMemoryState,
    ) -> tuple[Tensor, ContrastiveMemoryState]:
        """Leer solo sesiones confirmadas (s < t) y preparar las filas de t para `commit`."""
        sessions = tuple(self._session_rows(encoded, flow_ids, prediction_at, state))
        context = torch.zeros_like(encoded)
        pending_at = list(state.pending_at)
        pending_flows = list(state.pending_flows)
        pending_tokens = list(state.pending_tokens)
        for market, rows, moment, names in sessions:
            if len(set(names) | set(pending_flows[market])) != len(names) + len(
                pending_flows[market]
            ):
                raise ValueError("Un flujo no puede aparecer dos veces en la misma sesión")
            contiguous = rows[-1] - rows[0] + 1 == len(rows)
            if contiguous:
                start, stop = rows[0], rows[-1] + 1
                selected = encoded[start:stop]
            else:
                index = _index_tensor(rows, encoded.device)
                selected = encoded.index_select(0, index)
            q_hat, _ = safe_l2_normalize(self.query_projection(selected))
            retrieved, _ = safe_l2_normalize(
                self._apply_memory(q_hat, state.markets[market].weights)
            )
            if contiguous:
                if len(sessions) == 1:
                    context = retrieved
                else:
                    context = context.clone()
                    context[start:stop] = retrieved
            else:
                context = context.index_copy(0, index, retrieved)
            pending_at[market] = moment
            pending_flows[market] = pending_flows[market] + names
            pending_tokens[market] = (
                selected
                if pending_tokens[market] is None
                else torch.cat((pending_tokens[market], selected), dim=0)
            )
        extra = self.out_projection(context)
        check_finite(extra, "La lectura de la memoria contrastiva")
        next_state = replace(
            state,
            pending_at=tuple(pending_at),
            pending_flows=tuple(pending_flows),
            pending_tokens=tuple(pending_tokens),
        )
        return extra, next_state

    def needs_commit(
        self,
        state: ContrastiveMemoryState,
        flow_ids: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
    ) -> bool:
        """Indica si algún mercado del bloque cambia de sesión respecto a su sesión pendiente."""
        for flow, moment in zip(flow_ids, prediction_at, strict=True):
            market = self._market(flow)
            pending = state.pending_at[market]
            if pending != -1 and pending != moment:
                return True
        return False

    def commit(
        self, state: ContrastiveMemoryState, *, differentiable: bool = False
    ) -> ContrastiveMemoryState:
        """Escribir cada mercado con sesión pendiente en orden canónico de flujos."""
        check_differentiable(differentiable)
        self._reference()
        if torch.is_inference_mode_enabled():
            raise ValueError("inference_mode es incompatible con la memoria contrastiva")
        markets = list(state.markets)
        committed = list(state.committed_at)
        freeze_eval = self.config.mode == "eval_frozen" and not self.training
        loss_mode = "l2_shared" if self.config.mode == "l2_shared" else "infonce"

        grad_context = torch.enable_grad() if differentiable else torch.no_grad()
        with grad_context:
            for market_index, tokens in enumerate(state.pending_tokens):
                if tokens is None:
                    continue
                flows = state.pending_flows[market_index]
                moment = state.pending_at[market_index]
                order = sorted(range(len(flows)), key=flows.__getitem__)
                if len(order) > MAX_CONTRASTIVE_ROWS:
                    raise ValueError("La sesión supera las filas admitidas por la escritura")
                committed[market_index] = moment
                if freeze_eval:
                    continue
                if order == list(range(len(flows))):
                    ordered = tokens
                else:
                    idx = _index_tensor(order, tokens.device)
                    ordered = tokens.index_select(0, idx)
                if not differentiable:
                    ordered = ordered.detach()
                keys, _ = safe_l2_normalize(self.key_projection(ordered))
                v_hat, _ = safe_l2_normalize(self.value_projection(ordered))
                prototypes = session_prototypes(v_hat, self.config.prototypes)
                if self.config.mode == "shuffled_values":
                    v_pos = derange_values(
                        v_hat, seed=self.seed, market=market_index, moment=moment
                    )
                else:
                    v_pos = v_hat
                mean_token = ordered.mean(dim=0)
                alpha = self.alpha_projection(mean_token).sigmoid().unsqueeze(-1)
                eta = self.eta_projection(mean_token).sigmoid().unsqueeze(-1)
                theta = THETA_MAX * self.theta_projection(mean_token).sigmoid().unsqueeze(-1)
                current = markets[market_index]
                gradients, loss = contrastive_inner_gradients(
                    current.weights,
                    keys,
                    v_pos,
                    prototypes,
                    temperature=self.config.temperature,
                    loss_mode=loss_mode,
                )
                check_finite(loss, "La pérdida asociativa contrastiva")
                next_momentum = tuple(
                    eta * prev_m - theta * grad
                    for prev_m, grad in zip(current.momentum, gradients, strict=True)
                )
                next_weights = tuple(
                    (1.0 - alpha) * prev_w + mom
                    for prev_w, mom in zip(current.weights, next_momentum, strict=True)
                )
                for value in (*next_weights, *next_momentum):
                    check_finite(value, "La actualización de la memoria contrastiva")
                if not differentiable:
                    next_weights = tuple(w.detach().contiguous() for w in next_weights)
                    next_momentum = tuple(m.detach().contiguous() for m in next_momentum)
                markets[market_index] = ContrastiveMarketState(
                    next_weights, next_momentum, current.steps + 1
                )

        empty = (-1,) * len(MARKETS)
        return replace(
            state,
            committed_at=tuple(committed),
            pending_at=empty,
            pending_flows=((),) * len(MARKETS),
            pending_tokens=(None,) * len(MARKETS),
            markets=tuple(markets),
        )

    def detach(
        self,
        state: ContrastiveMemoryState,
        parameter_id: str,
        *,
        reattach_initial: bool = False,
    ) -> ContrastiveMemoryState:
        """Cortar el grafo en una barrera de truncamiento, exigiendo sesiones confirmadas."""
        if any(tokens is not None for tokens in state.pending_tokens):
            raise ValueError("Solo se corta el grafo con todas las sesiones confirmadas")
        markets = []
        for market_index, market in enumerate(state.markets):
            if reattach_initial and market.steps == 0 and state.committed_at[market_index] == -1:
                markets.append(self._initial_market(differentiable=True))
            else:
                markets.append(
                    ContrastiveMarketState(
                        tuple(w.detach().clone().contiguous() for w in market.weights),
                        tuple(m.detach().clone().contiguous() for m in market.momentum),
                        market.steps,
                    )
                )
        return replace(state, parameter_id=parameter_id, markets=tuple(markets))

    def export_state(self, state: ContrastiveMemoryState) -> dict:
        """Exportar copias CPU sin grafo, solo con todas las sesiones confirmadas."""
        detached = self.detach(state, state.parameter_id)
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            committed_at=list(detached.committed_at),
            markets=[
                dict(
                    weights=tuple(_moved(w, "cpu") for w in market.weights),
                    momentum=tuple(_moved(m, "cpu") for m in market.momentum),
                    steps=market.steps,
                )
                for market in detached.markets
            ],
        )

    def restore_state(
        self, payload: object, parameter_id: str, *, device: torch.device
    ) -> ContrastiveMemoryState:
        """Restaurar el estado verificado desde CPU hacia `device`, bit a bit."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "configuration", "committed_at", "markets"}
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.config.identity())
            or not isinstance(payload["committed_at"], list)
            or len(payload["committed_at"]) != len(MARKETS)
            or not isinstance(payload["markets"], list)
            or len(payload["markets"]) != len(MARKETS)
        ):
            raise ValueError(
                "El estado guardado de la memoria contrastiva no corresponde al módulo"
            )
        markets = []
        for entry in payload["markets"]:
            if not isinstance(entry, dict) or set(entry) != {"weights", "momentum", "steps"}:
                raise ValueError("El estado de mercado guardado tiene campos incompatibles")
            weights = entry["weights"]
            momentum = entry["momentum"]
            steps = entry["steps"]
            if (
                not isinstance(weights, (tuple, list))
                or not isinstance(momentum, (tuple, list))
                or len(weights) != 2
                or len(momentum) != 2
                or type(steps) is not int
                or steps < 0
            ):
                raise ValueError("El estado de mercado guardado tiene capas o pasos inválidos")
            markets.append(
                ContrastiveMarketState(
                    tuple(_moved(w, device) for w in weights),
                    tuple(_moved(m, device) for m in momentum),
                    steps,
                )
            )
        committed = tuple(int(value) for value in payload["committed_at"])
        empty = (-1,) * len(MARKETS)
        state = ContrastiveMemoryState(
            self.fingerprint(),
            parameter_id,
            committed,
            empty,
            ((),) * len(MARKETS),
            (None,) * len(MARKETS),
            tuple(markets),
        )
        self.validate(state, parameter_id, device=device)
        return state
