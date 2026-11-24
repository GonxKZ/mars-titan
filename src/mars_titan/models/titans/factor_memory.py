r"""Memoria rápida factorizada con blanqueo espectral de sesión `low_rank_factor_memory_sw_v1`.

Implementación de la Idea 4 (LFM-SW — *Low-Rank Factor Memory with Spectral Whitening*).

Mantiene una base factorial latente de mercado $F_{m,t} \in \mathbb{R}^{K \times D}$ ($K < D$),
separada estrictamente por mercado $m \in \{\text{US}, \text{CN}\}$ y actualizada de forma causal
al cierre de cada sesión tras emitir todas sus predicciones:

1. Normalización de la representación pre-cabeza $e_{i,t} \in \mathbb{R}^D$:
   $$x_{i,t} = \operatorname{LayerNorm}(e_{i,t})$$

2. Lectura en el instante $t$ con la base confirmada $F_{m,t}$ (de sesiones $< t$):
   $$A_{m,t} = \tfrac{1}{2}(F_{m,t} F_{m,t}^\top + (F_{m,t} F_{m,t}^\top)^\top) + \gamma I_K$$
   $$P_{m,t} = A_{m,t}^{-1} F_{m,t}, \qquad \hat{l}_{i,t} = x_{i,t} P_{m,t}^\top, \qquad
     r_{i,t} = x_{i,t} - \hat{l}_{i,t} F_{m,t}$$
   con $\gamma = \gamma_{\min} + \operatorname{softplus}(z_\gamma) > 0$, garantizando que $A_{m,t}$
   es estrictamente simétrica definida positiva y resuelta mediante Cholesky diferenciable.

3. Inyección residual antes de la cabeza con proyección lineal inicializada en cero:
   $$e'_{i,t} = e_{i,t} + W_{\text{out}}\, r_{i,t}, \qquad W_{\text{out}} = 0 \text{ al inicio}$$

4. Actualización causal de $F_{m,t} \to F_{m,t+1}$ para sesiones posteriores ($> t$), en orden
   canónico de identificadores de flujo para conservar invarianza exacta frente a permutaciones:
   $$G_F = \frac{1}{N_{m,t}} \hat{L}_{m,t}^\top (\hat{L}_{m,t} F_{m,t} - X_{m,t})
         = -\frac{1}{N_{m,t}} \hat{L}_{m,t}^\top R_{m,t}$$
   $$F_{m,t+1} = (1 - \alpha_F) F_{m,t} - \theta_F \operatorname{NS}_s(G_F)$$
   donde $\operatorname{NS}_s$ aplica $s$ iteraciones Newton-Schulz quínticas estilo Muon
   (coeficientes $a = 3{,}4445,\; b = -4{,}7750,\; c = 2{,}0315$) sobre
   $Y_0 = G_F / \sqrt{\|G_F\|_F^2 + \varepsilon}$ con $\varepsilon = 10^{-12}$, que es suave
   y finita en $G_F = 0$.

Modos declarados con los mismos parámetros, formas e inicialización bit a bit:

- `online`: candidato completo, actualiza $F_{m,t}$ en ajuste, calentamiento y evaluación.
- `eval_frozen`: placebo de adaptación en evaluación; actualiza en ajuste (`training=True`) pero
  congela la base en $F_0$ en validación, calibración y evaluación (`training=False`).
- `random_basis`: placebo estructural; sustituye $F_{m,t}$ en cada sesión $(m, t)$ por una base
  ortonormal por filas aleatoria y determinista (sembrada por semilla, mercado e instante),
  conservando exactamente la misma capacidad $W_{\text{out}}$, normalización y rango $K$ sin
  rastrear la estructura factorial transversal del mercado.
"""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LAYER_NORM_EPS, bounded_integer, canonical, require_identity
from .state import check_differentiable, check_finite, require_true

FACTOR_MEMORY = "low_rank_factor_memory_sw_v1"
ONLINE = "online"
EVAL_FROZEN = "eval_frozen"
RANDOM_BASIS = "random_basis"
MODES = (ONLINE, EVAL_FROZEN, RANDOM_BASIS)
MARKETS = ("US", "CN")

# Cotas estrictas que acotan el radio espectral de F_{m,t} y el condicionamiento de Cholesky.
ALPHA_MIN = 1e-3
THETA_MAX = 0.5
GAMMA_MIN = 1e-2
GAMMA_MAX = 100.0
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-12

_REQUIRED_FIELDS = {"factors", "mode"}
_CONFIG_FIELDS = {"factors", "ns_steps", "mode", "alpha_init", "theta_init", "gamma_init"}
_STATE_PAYLOAD_FIELDS = {"schema_version", "configuration", "factors", "last_updated_at", "updates"}


def _finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    return float(value)


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


def _inv_softplus(value: float) -> float:
    if value > 20.0:
        return value + math.log(-math.expm1(-value))
    return math.log(math.expm1(value))


def _derived_seed(*parts: object) -> int:
    text = "/".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:7], "little")


def _orthonormal_rows(rows: int, cols: int, *, seed: int, dtype: torch.dtype) -> Tensor:
    """Base determinista de filas ortonormales [rows, cols] (rows <= cols) generada en CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(cols, rows, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw, mode="reduced")
    # Fijar los signos de la diagonal de R para invarianza frente a variantes de LAPACK.
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    q = q * signs.unsqueeze(0)
    return q.T.contiguous().to(dtype=dtype)


def newton_schulz(matrix: Tensor, *, steps: int = 3, eps: float = NS_EPS) -> Tensor:
    """Ortogonalización espectral Newton-Schulz quíntica (estilo Muon) para matriz [K, D].

    Normaliza por $\\sqrt{\\|G\\|_F^2 + \\varepsilon}$ para que todos los valores singulares sean
    menores que 1 y la derivada sea suave y finita incluso en $G = 0$.
    """
    if (
        not isinstance(matrix, Tensor)
        or matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] > matrix.shape[1]
    ):
        raise ValueError("Newton-Schulz necesita una matriz 2D con K <= D")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError("Newton-Schulz exige FP32 o FP64")
    if torch.is_autocast_enabled(matrix.device.type):
        raise ValueError("Newton-Schulz exige FP32 o FP64 explícitos, sin autocast")
    bounded_integer(steps, "iteraciones Newton-Schulz", 1, 8)
    norm = torch.sqrt((matrix * matrix).sum() + eps)
    y = matrix / norm
    a, b, c = NS_COEFFS
    for _ in range(steps):
        gram = y @ y.T
        y = a * y + (b * gram + c * (gram @ gram)) @ y
    return y


@dataclass(frozen=True)
class FactorMemoryConfig:
    """Configuración identificada de la memoria rápida factorizada LFM-SW."""

    factors: int = 16
    ns_steps: int = 3
    mode: str = ONLINE
    alpha_init: float = 0.05
    theta_init: float = 0.1
    gamma_init: float = 0.1

    def __post_init__(self) -> None:
        bounded_integer(self.factors, "factores de LFM-SW", 1, 256)
        bounded_integer(self.ns_steps, "iteraciones Newton-Schulz de LFM-SW", 1, 8)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de LFM-SW debe ser uno de {MODES}")
        alpha = _finite_float(self.alpha_init, "alpha_init")
        theta = _finite_float(self.theta_init, "theta_init")
        gamma = _finite_float(self.gamma_init, "gamma_init")
        if not ALPHA_MIN < alpha < 1.0:
            raise ValueError(f"alpha_init debe pertenecer a ({ALPHA_MIN}, 1.0)")
        if not 0.0 < theta < THETA_MAX:
            raise ValueError(f"theta_init debe pertenecer a (0.0, {THETA_MAX})")
        if not GAMMA_MIN < gamma <= GAMMA_MAX:
            raise ValueError(f"gamma_init debe pertenecer a ({GAMMA_MIN}, {GAMMA_MAX}]")
        object.__setattr__(self, "alpha_init", alpha)
        object.__setattr__(self, "theta_init", theta)
        object.__setattr__(self, "gamma_init", gamma)

    def identity(self) -> dict:
        state_semantics = {
            ONLINE: "per_market_factor_basis_updated_after_session_emission",
            EVAL_FROZEN: "per_market_factor_basis_updated_in_train_frozen_at_initial_in_eval",
            RANDOM_BASIS: "deterministic_orthonormal_random_basis_per_market_and_instant",
        }[self.mode]
        return dict(
            schema_version=1,
            kind=FACTOR_MEMORY,
            factors=self.factors,
            ns_steps=self.ns_steps,
            mode=self.mode,
            alpha_init=self.alpha_init,
            theta_init=self.theta_init,
            gamma_init=self.gamma_init,
            alpha_min=ALPHA_MIN,
            theta_max=THETA_MAX,
            gamma_min=GAMMA_MIN,
            ns_coefficients=list(NS_COEFFS),
            ns_eps=NS_EPS,
            position="after_mac_and_technical_branch_before_head",
            normalization="layer_norm_before_projection",
            projection="ridge_cholesky_solve_with_learned_gamma",
            update="muon_quintic_newton_schulz_on_canonical_flow_order",
            output="residual_with_zero_initialized_projection",
            state=state_semantics,
            causality="read_previous_sessions_only_update_after_emission",
        )


def parse_factor_memory(raw: object) -> FactorMemoryConfig | None:
    """Aceptar FactorMemoryConfig, diccionario JSON válido o None."""
    if raw is None:
        return None
    if type(raw) is FactorMemoryConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and _REQUIRED_FIELDS <= set(raw) <= _CONFIG_FIELDS:
        return FactorMemoryConfig(**raw)
    raise ValueError(
        "factor_memory debe declarar al menos factors y mode (y opcionalmente ns_steps, "
        "alpha_init, theta_init y gamma_init), o ser None"
    )


@dataclass(frozen=True)
class FactorMemoryState:
    """Estado rápido por mercado de LFM-SW: base factorial [K, D] y cursor causal."""

    config_id: str
    parameter_id: str
    factors: tuple[Tensor, ...]
    last_updated_at: tuple[int, ...]
    updates: tuple[int, ...]


class LowRankFactorMemory(nn.Module):
    """Residualización idiosincrásica en línea contra una base factorial blanqueada."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        config: FactorMemoryConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión de LFM-SW", 2, 4096)
        bounded_integer(seed, "semilla de LFM-SW", 0, 2**32 - 1)
        if type(config) is not FactorMemoryConfig:
            raise ValueError("LFM-SW necesita FactorMemoryConfig")
        config.__post_init__()
        if config.factors >= hidden_size:
            raise ValueError("factors debe ser menor que hidden_size para dejar residuo")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("LFM-SW necesita precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed

        # Inicialización determinista sin tocar el generador global. Los tres modos nacen con
        # exactamente los mismos parámetros, formas y valores bit a bit.
        init_seed = _derived_seed(FACTOR_MEMORY, "params", seed)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.out = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.out.weight)

        basis_seed = _derived_seed(FACTOR_MEMORY, "initial_factors", seed)
        self.initial_factors = nn.Parameter(
            _orthonormal_rows(config.factors, hidden_size, seed=basis_seed, dtype=dtype)
        )
        alpha_frac = (config.alpha_init - ALPHA_MIN) / (1.0 - ALPHA_MIN)
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(alpha_frac), dtype=dtype))
        self.theta_logit = nn.Parameter(
            torch.tensor(_logit(config.theta_init / THETA_MAX), dtype=dtype)
        )
        self.gamma_logit = nn.Parameter(
            torch.tensor(_inv_softplus(config.gamma_init - GAMMA_MIN), dtype=dtype)
        )
        self._fingerprint = hashlib.sha256(canonical(self.identity()).encode()).hexdigest()

    def identity(self) -> dict:
        return dict(configuration=self.config.identity(), hidden_size=self.hidden_size)

    def fingerprint(self) -> str:
        return self._fingerprint

    def get_extra_state(self) -> dict:
        return self.identity()

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def alpha(self) -> Tensor:
        """Tasa de olvido acotada en (ALPHA_MIN, 1)."""
        return ALPHA_MIN + (1.0 - ALPHA_MIN) * torch.sigmoid(self.alpha_logit)

    def theta(self) -> Tensor:
        """Paso de actualización Newton-Schulz acotado en (0, THETA_MAX)."""
        return THETA_MAX * torch.sigmoid(self.theta_logit)

    def gamma(self) -> Tensor:
        """Regularización ridge estrictamente positiva (>= GAMMA_MIN)."""
        return GAMMA_MIN + F.softplus(self.gamma_logit)

    def random_basis(
        self, market: str, moment: int, *, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """Base aleatoria ortonormal por filas determinista para el placebo `random_basis`."""
        basis_seed = _derived_seed(FACTOR_MEMORY, RANDOM_BASIS, self.seed, market, moment)
        return _orthonormal_rows(
            self.config.factors, self.hidden_size, seed=basis_seed, dtype=dtype
        ).to(device=device)

    def initial_state(
        self, parameter_id: str, *, differentiable: bool = False
    ) -> FactorMemoryState:
        check_differentiable(differentiable)
        base = self.initial_factors if differentiable else self.initial_factors.detach()
        return FactorMemoryState(
            self._fingerprint,
            parameter_id,
            tuple(base.clone() for _ in MARKETS),
            (-1,) * len(MARKETS),
            (0,) * len(MARKETS),
        )

    def validate_state(
        self, state: object, parameter_id: str, *, device: torch.device | None = None
    ) -> None:
        reference = self.out.weight
        expected_device = reference.device if device is None else torch.device(device)
        expected_shape = (self.config.factors, self.hidden_size)
        if (
            not isinstance(state, FactorMemoryState)
            or state.config_id != self._fingerprint
            or state.parameter_id != parameter_id
            or type(state.factors) is not tuple
            or len(state.factors) != len(MARKETS)
            or type(state.last_updated_at) is not tuple
            or len(state.last_updated_at) != len(MARKETS)
            or type(state.updates) is not tuple
            or len(state.updates) != len(MARKETS)
        ):
            raise ValueError("El estado de la memoria factorial no corresponde al predictor")
        storages = set()
        for factor, last_at, count in zip(
            state.factors, state.last_updated_at, state.updates, strict=True
        ):
            if (
                type(count) is not int
                or count < 0
                or type(last_at) is not int
                or (count == 0 and last_at != -1)
                or (count > 0 and last_at < 0)
            ):
                raise ValueError("Los cursores de la memoria factorial no son válidos")
            if (
                not isinstance(factor, Tensor)
                or factor.layout != torch.strided
                or factor.shape != expected_shape
                or factor.dtype != reference.dtype
                or factor.device != expected_device
            ):
                raise ValueError("Las bases de la memoria factorial no conservan forma o tipo")
            key = (factor.device, factor.untyped_storage().data_ptr())
            if key in storages:
                raise ValueError("Las bases de mercados distintos no pueden compartir memoria")
            storages.add(key)
            check_finite(factor, "La base factorial de la memoria")

    def detach_state(self, state: FactorMemoryState, parameter_id: str) -> FactorMemoryState:
        """Cortar el grafo en una barrera y sincronizar el nuevo parameter_id."""
        detached = tuple(
            self.initial_factors.detach().clone() if count == 0 else factor.detach().clone()
            for factor, count in zip(state.factors, state.updates, strict=True)
        )
        return FactorMemoryState(
            state.config_id,
            parameter_id,
            detached,
            state.last_updated_at,
            state.updates,
        )

    def export_state(self, state: FactorMemoryState, parameter_id: str) -> dict:
        """Exportar copias CPU sin grafo para puntos de control."""
        self.validate_state(state, parameter_id)
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            factors=[
                factor.detach().to(device="cpu", copy=True).contiguous() for factor in state.factors
            ],
            last_updated_at=list(state.last_updated_at),
            updates=list(state.updates),
        )

    def restore_state(
        self, payload: object, parameter_id: str, *, device: torch.device
    ) -> FactorMemoryState:
        """Validar y restaurar el estado guardado en el dispositivo del predictor."""
        if (
            not isinstance(payload, dict)
            or set(payload) != _STATE_PAYLOAD_FIELDS
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.config.identity())
            or not isinstance(payload["factors"], (list, tuple))
            or len(payload["factors"]) != len(MARKETS)
            or not isinstance(payload["last_updated_at"], (list, tuple))
            or len(payload["last_updated_at"]) != len(MARKETS)
            or not isinstance(payload["updates"], (list, tuple))
            or len(payload["updates"]) != len(MARKETS)
            or any(type(v) is not int for v in payload["last_updated_at"])
            or any(type(v) is not int for v in payload["updates"])
            or any(not isinstance(t, Tensor) for t in payload["factors"])
        ):
            raise ValueError("El estado guardado de la memoria factorial no corresponde al módulo")
        factors = tuple(
            tensor.detach().to(device=device, copy=True).contiguous()
            for tensor in payload["factors"]
        )
        state = FactorMemoryState(
            self._fingerprint,
            parameter_id,
            factors,
            tuple(payload["last_updated_at"]),
            tuple(payload["updates"]),
        )
        self.validate_state(state, parameter_id, device=device)
        return state

    def step(
        self,
        encoded: Tensor,
        flow_ids: tuple[str, ...] | list[str],
        markets: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        state: FactorMemoryState,
        *,
        differentiable: bool = False,
    ) -> tuple[Tensor, FactorMemoryState]:
        """Leer el residuo idiosincrásico con $F_{m,t}$ y actualizar $F_{m,t+1}$ tras emitir."""
        check_differentiable(differentiable)
        reference = self.out.weight
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
                "LFM-SW necesita una representación, un flujo, un mercado y un instante por fila"
            )
        if torch.is_autocast_enabled(encoded.device.type):
            raise ValueError("LFM-SW exige FP32 o FP64 explícitos, sin autocast")
        self.validate_state(state, state.parameter_id)
        check_finite(encoded, "La representación de entrada a la memoria factorial")

        normalized = self.norm(encoded)
        alpha = self.alpha()
        theta = self.theta()
        gamma = self.gamma()
        eye = torch.eye(self.config.factors, dtype=encoded.dtype, device=encoded.device)

        next_factors = [
            factor.clone() if differentiable else factor.detach().clone()
            for factor in state.factors
        ]
        next_last_at = list(state.last_updated_at)
        next_updates = list(state.updates)

        unique_markets = sorted(set(markets))
        single_market = len(unique_markets) == 1
        residuals = None if single_market else torch.zeros_like(encoded)

        for market in unique_markets:
            if market not in MARKETS:
                raise ValueError(f"Mercado desconocido para la memoria factorial: {market}")
            m_idx = MARKETS.index(market)
            if single_market:
                n_rows = len(encoded)
                m_flows = tuple(flow_ids)
                moments = set(prediction_at)
                x_m = normalized
            else:
                rows = [i for i, value in enumerate(markets) if value == market]
                n_rows = len(rows)
                m_flows = tuple(flow_ids[i] for i in rows)
                moments = {prediction_at[i] for i in rows}
                contiguous = rows[-1] - rows[0] + 1 == n_rows and rows == list(
                    range(rows[0], rows[0] + n_rows)
                )
                if contiguous:
                    x_m = normalized[rows[0] : rows[-1] + 1]
                else:
                    index = (
                        torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
                            device=encoded.device, non_blocking=True
                        )
                        if encoded.is_cuda
                        else torch.tensor(rows, dtype=torch.int64, device=encoded.device)
                    )
                    x_m = normalized.index_select(0, index)
            if len(moments) != 1:
                raise ValueError("Las filas de un mercado deben pertenecer a una sola sesión")
            (moment,) = moments
            if type(moment) is not int or moment < 0 or moment <= state.last_updated_at[m_idx]:
                raise ValueError(
                    "El instante de la sesión debe ser estrictamente posterior al estado factorial"
                )
            if len(set(m_flows)) != len(m_flows):
                raise ValueError("Un flujo no puede repetirse dentro de la misma sesión")

            # Si el mercado aún no observó sesiones, su base de partida es la meta-inicial F_0.
            if state.updates[m_idx] == 0:
                f_prev = self.initial_factors if differentiable else self.initial_factors.detach()
            else:
                f_prev = state.factors[m_idx]

            mode = self.config.mode
            if mode == ONLINE or (mode == EVAL_FROZEN and self.training):
                f_read = f_prev
                should_update = True
            elif mode == EVAL_FROZEN:
                f_read = self.initial_factors.detach()
                should_update = False
            else:
                f_read = self.random_basis(
                    market, moment, dtype=encoded.dtype, device=encoded.device
                )
                should_update = False

            gram = f_read @ f_read.T
            gram = 0.5 * (gram + gram.T)
            # Sin esperar a la GPU: el fallo de la factorización se comprueba diferido.
            chol, info = torch.linalg.cholesky_ex(gram + gamma * eye, check_errors=False)
            require_true(
                (info == 0).all(),
                "El sistema regularizado de la memoria factorial no es definido positivo",
            )
            proj = torch.cholesky_solve(f_read, chol)
            loadings = x_m @ proj.T
            r_m = x_m - loadings @ f_read

            if single_market:
                residuals = r_m
            elif contiguous:
                residuals[rows[0] : rows[-1] + 1] = r_m
            else:
                residuals = residuals.index_copy(0, index, r_m)

            if should_update:
                if all(m_flows[i] < m_flows[i + 1] for i in range(n_rows - 1)):
                    l_ord, r_ord = loadings, r_m
                else:
                    order = sorted(range(n_rows), key=m_flows.__getitem__)
                    perm = (
                        torch.tensor(order, dtype=torch.int64, pin_memory=True).to(
                            device=encoded.device, non_blocking=True
                        )
                        if encoded.is_cuda
                        else torch.tensor(order, dtype=torch.int64, device=encoded.device)
                    )
                    l_ord = loadings.index_select(0, perm)
                    r_ord = r_m.index_select(0, perm)
                # G_F = (1/N) L^T (L F - X) = -(1/N) L^T R
                grad_f = -(l_ord.T @ r_ord) / n_rows
                f_next = (1.0 - alpha) * f_prev - theta * newton_schulz(
                    grad_f, steps=self.config.ns_steps
                )
                if not differentiable:
                    f_next = f_next.detach()
            else:
                f_next = f_prev if differentiable else f_prev.detach()

            check_finite(f_next, "La base factorial actualizada")
            next_factors[m_idx] = f_next.clone()
            next_last_at[m_idx] = moment
            next_updates[m_idx] = state.updates[m_idx] + 1

        result = encoded + self.out(residuals)
        check_finite(result, "La representación con memoria factorial")
        next_state = FactorMemoryState(
            self._fingerprint,
            state.parameter_id,
            tuple(next_factors),
            tuple(next_last_at),
            tuple(next_updates),
        )
        return result, next_state
