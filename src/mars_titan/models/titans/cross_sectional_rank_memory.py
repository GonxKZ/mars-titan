"""Memoria transversal de rango y residuo maduro `cross_sectional_rank_memory_v1` (CRRM).

Complementa las memorias rápidas no supervisadas por activo de Titans-MAC con una memoria rápida
supervisada compartida por mercado, actualizada estrictamente cuando maduran las etiquetas de cada
sesión pasada (`decision_at < target_available_at <= event.at`).

En el instante de decisión t, para las N_t filas del mercado m con representación e_{i,t} ∈ R^H
antes de la cabeza:

    u_{i,t} = W_k LN(e_{i,t}) ∈ R^{d_m}
    ũ_{i,t} = u_{i,t} − (1/N_t) Σ_{j∈S_{m,t}} u_{j,t}           (si N_t > 1)
    k̂_{i,t} = ũ_{i,t} / sqrt(‖ũ_{i,t}‖_2^2 + ε_k)               (‖k̂_{i,t}‖_2 < 1)

Al madurar las N_s ≥ minimum_rows etiquetas {y_{j,s}} de una sesión emitida s, se forman dos
canales transversales libres de nivel: el rango normal gaussiano de la etiqueta y el residuo de
orden del tronco respecto a su predicción emitida desacoplada:

    g_{j,s} = Φ⁻¹((rango(y_{j,s}) − 1/2) / N_s),     ĝ_{j,s} = estandarizar_s(ŷ_{j,s})
    v_{j,s} = [ g_{j,s},  g_{j,s} − ĝ_{j,s} ] ∈ R^2

y se actualizan los estadísticos suficientes de R escalas temporales con retención exponencial
λ_r = exp(−1 / τ_r):

    C_s = (1/N_s) K̂_sᵀ K̂_s,    B_s = (1/N_s) K̂_sᵀ V_s
    P_{m,r} ← λ_r P_{m,r} + (1 − λ_r) C_s
    Q_{m,r} ← λ_r Q_{m,r} + (1 − λ_r) B_s
    w_{m,r} ← λ_r w_{m,r} + (1 − λ_r)

Al consultar en t con k̂_{i,t}, se resuelve exactamente la regresión ridge en forma cerrada mediante
Cholesky con penalización positiva aprendida γ̃_r = (γ_min + softplus(ρ_r)) / d_m, obteniendo a la
vez los pesos óptimos M*_{m,r} y la confianza epistémica de apalancamiento c_{i,t}^{(r)} ∈ [0, 1]:

    L_{m,r} L_{m,r}ᵀ = P_{m,r} + γ̃_r I_{d_m}
    M*_{m,r} = L_{m,r}^{-T} L_{m,r}^{-1} Q_{m,r}
    c_{i,t}^{(r)} = recorte(‖k̂_{i,t}‖_2^2 − γ̃_r ‖L_{m,r}^{-1} k̂_{i,t}‖_2^2, 0, 1)
                  = k̂_{i,t}ᵀ (P_{m,r} + γ̃_r I)^{-1} P_{m,r} k̂_{i,t}
    a_{i,t}^{(r)} = c_{i,t}^{(r)} · tanh(k̂_{i,t}ᵀ M*_{m,r})
    e'_{i,t} = e_{i,t} + W_out [a_{i,t}^{(1)}, ..., a_{i,t}^{(R)}]

W_out arranca en cero, así que el predictor comienza emitiendo lo mismo que su control; en el
primer segmento BPTT de 2 sesiones las etiquetas de la sesión 1 maduran antes de la 2 y el
gradiente alcanza W_out en el primer paso (y atraviesa Cholesky hacia W_k y ρ_r desde el segundo).

Controles negativos integrados (mismos parámetros e inicialización):
- `shuffled_targets`: permuta deterministamente las filas de V_s dentro de cada sesión madura con
  una semilla causal derivada de (seed, mercado, decision_at). Conserva P_{m,r}, la distribución
  marginal y el coste, pero destruye la asociación clave-rango.
- `eval_frozen`: actualiza con normalidad en ajuste (`train=True`) y congela las escrituras en
  validación/evaluación (`train=False`), aislando la ganancia de memorización en tiempo de prueba.
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

CROSS_SECTIONAL_RANK_MEMORY = "cross_sectional_rank_memory_v1"
MATURED_RANK = "matured_rank"
SHUFFLED_TARGETS = "shuffled_targets"
EVAL_FROZEN = "eval_frozen"
MODES = (MATURED_RANK, SHUFFLED_TARGETS, EVAL_FROZEN)
TARGET_CHANNELS = 2
KEY_NORM_EPS = 1e-8
RIDGE_FLOOR = 1e-2
MIN_RIDGE = 0.05
MAX_RIDGE = 100.0
MIN_DIM = 4
MAX_DIM = 128


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
    """softplus⁻¹(x) = x + log(1 − exp(−x)), estable para x pequeño y grande."""
    return value + math.log(-math.expm1(-value))


@dataclass(frozen=True)
class CrossSectionalRankMemoryConfig:
    """Contrato de la memoria transversal de rango y residuo maduro (CRRM)."""

    dim: int = 32
    time_constants: tuple[int, ...] = (5, 21, 63)
    ridge: float = 1.0
    minimum_rows: int = 8
    mode: str = MATURED_RANK
    seed: int = 0

    def __post_init__(self) -> None:
        bounded_integer(self.dim, "dimensión de CRRM", MIN_DIM, MAX_DIM)
        constants = self.time_constants
        constants = tuple(constants) if isinstance(constants, list) else constants
        if (
            type(constants) is not tuple
            or not 1 <= len(constants) <= 4
            or any(type(value) is not int or not 1 <= value <= 252 for value in constants)
            or list(constants) != sorted(set(constants))
        ):
            raise ValueError("time_constants debe tener de una a cuatro sesiones crecientes")
        ridge = _finite_float(
            self.ridge,
            MIN_RIDGE,
            MAX_RIDGE,
            f"ridge debe ser un número finito en [{MIN_RIDGE}, {MAX_RIDGE}]",
        )
        bounded_integer(self.minimum_rows, "minimum_rows de CRRM", 3, 4096)
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de CRRM debe ser uno de {MODES}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de CRRM debe ser un entero de 32 bits")
        object.__setattr__(self, "time_constants", constants)
        object.__setattr__(self, "ridge", ridge)

    @property
    def decays(self) -> tuple[float, ...]:
        """Retención exponencial λ_r = exp(−1 / τ_r) de cada escala."""
        return tuple(math.exp(-1.0 / tau) for tau in self.time_constants)

    def identity(self) -> dict:
        return dict(
            schema_version=1,
            kind=CROSS_SECTIONAL_RANK_MEMORY,
            dim=self.dim,
            time_constants=list(self.time_constants),
            ridge=self.ridge,
            ridge_floor=RIDGE_FLOOR,
            minimum_rows=self.minimum_rows,
            mode=self.mode,
            seed=self.seed,
            key_space="layer_norm_linear_cross_section_centered_smooth_l2_normalized",
            key_norm_eps=KEY_NORM_EPS,
            targets="matured_session_gaussian_normal_ranks_and_backbone_standardized_rank_residual",
            solver="batched_spd_cholesky_closed_form_multi_timescale_exponential_ridge",
            confidence="spectral_leverage_norm_sq_minus_ridge_times_cholesky_mahalanobis_sq",
            readout="leverage_gated_tanh_zero_initialized_residual_before_head",
            market_isolation="strict_per_market_sufficient_statistics",
            causality="updated_only_on_matured_labels_strictly_before_emission",
        )


def parse_cross_sectional_rank_memory(raw: object) -> CrossSectionalRankMemoryConfig | None:
    """Aceptar CrossSectionalRankMemoryConfig, su diccionario completo o None."""
    if raw is None:
        return None
    if type(raw) is CrossSectionalRankMemoryConfig:
        raw.__post_init__()
        return raw
    expected = {field.name for field in fields(CrossSectionalRankMemoryConfig)}
    if isinstance(raw, dict) and set(raw) == expected:
        return CrossSectionalRankMemoryConfig(**raw)
    raise ValueError(
        "cross_sectional_rank_memory debe declarar dim, time_constants, ridge, minimum_rows, "
        "mode y seed, o ser None"
    )


@dataclass(frozen=True)
class MarketRankStatistics:
    """Estadísticos suficientes (P, Q, w) de un mercado en las R escalas temporales."""

    cov: Tensor
    rhs: Tensor
    weight: Tensor
    updates: int


@dataclass(frozen=True)
class CrossSectionalRankMemoryState:
    """Estado de la memoria transversal de rango: estadísticos por mercado y claves pendientes."""

    config_id: str
    parameter_id: str
    markets: dict[str, MarketRankStatistics]
    # Clave: (flow, decision_at) -> (matriz de claves de su emisión [filas, dim], predicciones
    # emitidas de esa emisión [filas], fila). Se guardan la matriz y el vector de la emisión, no un
    # valor por fila: así una sesión se reúne con una selección, el grafo del metagradiente no
    # crece con el número de activos y registrar una emisión no espera a la GPU.
    pending: dict[tuple[str, int], tuple[Tensor, Tensor, int]]


def _gather_pending(entries: list[tuple[Tensor, Tensor, int]]) -> tuple[Tensor, Tensor]:
    """Reunir, en el orden dado, las claves [n, dim] y las predicciones emitidas [n] pendientes."""
    sources: dict[int, tuple[Tensor, Tensor, list[int], list[int]]] = {}
    for position, (keys, issued, row) in enumerate(entries):
        _, _, positions, picked = sources.setdefault(id(keys), (keys, issued, [], []))
        positions.append(position)
        picked.append(row)
    key_parts, issued_parts, order = [], [], []
    for keys, issued, positions, picked in sources.values():
        index = torch.tensor(picked, dtype=torch.int64, device=keys.device)
        key_parts.append(keys.index_select(0, index))
        issued_parts.append(issued.index_select(0, index))
        order.extend(positions)
    if len(key_parts) == 1:
        return key_parts[0], issued_parts[0]
    inverse = torch.empty(len(order), dtype=torch.int64)
    inverse[torch.tensor(order, dtype=torch.int64)] = torch.arange(len(order), dtype=torch.int64)
    inverse = inverse.to(key_parts[0].device)
    return (
        torch.cat(key_parts, dim=0).index_select(0, inverse),
        torch.cat(issued_parts, dim=0).index_select(0, inverse),
    )


def _market(flow: str) -> str:
    if type(flow) is not str or "/" not in flow:
        raise ValueError("Cada flujo de CRRM necesita su prefijo de mercado")
    return flow.split("/", 1)[0]


class CrossSectionalRankMemory(nn.Module):
    """Memoria rápida supervisada de rango transversal con Ridge recursivo diferenciable."""

    session_level = True

    def __init__(
        self,
        hidden_size: int,
        config: CrossSectionalRankMemoryConfig,
        *,
        seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        bounded_integer(hidden_size, "dimensión oculta de CRRM", 1, 4096)
        bounded_integer(seed, "semilla de CRRM", 0, 2**32 - 1)
        if type(config) is not CrossSectionalRankMemoryConfig:
            raise ValueError("CRRM necesita CrossSectionalRankMemoryConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("CRRM admite precisión FP32 o FP64")
        self.config = config
        self.hidden_size = hidden_size
        self.seed = seed
        derived = hashlib.sha256(f"{CROSS_SECTIONAL_RANK_MEMORY}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS, dtype=dtype)
            self.key_proj = nn.Linear(hidden_size, config.dim, bias=False, dtype=dtype)
            self.out = nn.Linear(
                len(config.time_constants) * TARGET_CHANNELS,
                hidden_size,
                bias=False,
                dtype=dtype,
            )
        nn.init.zeros_(self.out.weight)
        init_logit = _softplus_inverse(config.ridge - RIDGE_FLOOR)
        self.ridge_logits = nn.Parameter(
            torch.full((len(config.time_constants),), init_logit, dtype=dtype)
        )
        self._fingerprint = hashlib.sha256(canonical(config.identity()).encode()).hexdigest()

    def fingerprint(self) -> str:
        return self._fingerprint

    def get_extra_state(self) -> dict:
        return dict(configuration=self.config.identity(), hidden_size=self.hidden_size)

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _reference(self) -> Tensor:
        reference = self.out.weight
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("CRRM admite precisión FP32 o FP64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("CRRM exige FP32 o FP64 explícitos, sin autocast")
        return reference

    def initial_state(self, parameter_id: str) -> CrossSectionalRankMemoryState:
        return CrossSectionalRankMemoryState(self.fingerprint(), parameter_id, {}, {})

    def validate_state(self, state: object, parameter_id: str) -> None:
        reference = self._reference()
        scales, dim = len(self.config.time_constants), self.config.dim
        if (
            not isinstance(state, CrossSectionalRankMemoryState)
            or state.config_id != self.fingerprint()
            or state.parameter_id != parameter_id
            or not isinstance(state.markets, dict)
            or not isinstance(state.pending, dict)
        ):
            raise ValueError("El estado de memoria transversal de rango no corresponde al módulo")
        for market, stats in state.markets.items():
            if (
                type(market) is not str
                or not market
                or not isinstance(stats, MarketRankStatistics)
                or type(stats.updates) is not int
                or stats.updates < 1
                or not isinstance(stats.cov, Tensor)
                or not isinstance(stats.rhs, Tensor)
                or not isinstance(stats.weight, Tensor)
                or stats.cov.shape != (scales, dim, dim)
                or stats.rhs.shape != (scales, dim, TARGET_CHANNELS)
                or stats.weight.shape != (scales,)
                or stats.cov.dtype != reference.dtype
                or stats.rhs.dtype != reference.dtype
                or stats.weight.dtype != reference.dtype
                or stats.cov.device != reference.device
                or stats.rhs.device != reference.device
                or stats.weight.device != reference.device
            ):
                raise ValueError("Los estadísticos de mercado de CRRM no conservan su contrato")
            check_finite(stats.cov, "La covarianza de CRRM")
            check_finite(stats.rhs, "La covarianza cruzada de CRRM")
            check_finite(stats.weight, "La masa efectiva de CRRM")
            require_true(
                (stats.weight >= 0).all() & (stats.weight <= 1.0 + 1e-6).all(),
                "La masa efectiva de CRRM debe permanecer en [0, 1]",
            )

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
            raise ValueError("CRRM necesita una representación, un mercado y un instante por fila")
        check_finite(encoded, "La representación de entrada de CRRM")
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
                index = torch.tensor(members, dtype=torch.int64, device=raw.device)
                part = raw.index_select(0, index)
                if len(members) > 1:
                    part = part - part.mean(dim=0, keepdim=True)
                norm = torch.sqrt(part.square().sum(dim=-1, keepdim=True) + KEY_NORM_EPS)
                result = result.index_copy(0, index, part / norm)
        check_finite(result, "Las claves transversales de CRRM")
        return result

    def _effective_ridge(self) -> Tensor:
        """Regularización positiva aprendida γ̃_r = (γ_min + softplus(ρ_r)) / d_m."""
        return (RIDGE_FLOOR + F.softplus(self.ridge_logits)) / float(self.config.dim)

    def read_features(
        self,
        keys: Tensor,
        markets: tuple[str, ...] | list[str],
        state: CrossSectionalRankMemoryState,
    ) -> Tensor:
        """Leer los rasgos modulados por apalancamiento [N, 2R] antes de la proyección `out`."""
        reference = self._reference()
        dim, scales = self.config.dim, len(self.config.time_constants)
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
                "La lectura de CRRM necesita claves [filas, dim] y un mercado por fila"
            )
        check_finite(keys, "Las claves de consulta de CRRM")
        features = keys.new_zeros((len(keys), scales * TARGET_CHANNELS))
        if not state.markets:
            return features
        ridge = self._effective_ridge()
        eye = torch.eye(dim, dtype=keys.dtype, device=keys.device).unsqueeze(0)
        by_market: dict[str, list[int]] = {}
        for row, market in enumerate(markets):
            by_market.setdefault(market, []).append(row)
        for market, members in by_market.items():
            stats = state.markets.get(market)
            if stats is None:
                continue
            contiguous = len(members) == len(keys)
            if contiguous:
                k_m = keys
            else:
                index = torch.tensor(members, dtype=torch.int64, device=keys.device)
                k_m = keys.index_select(0, index)
            sym = 0.5 * (stats.cov + stats.cov.transpose(-1, -2))
            system = sym + ridge[:, None, None] * eye
            # Sin esperar a la GPU: el fallo de la factorización se comprueba diferido.
            chol, info = torch.linalg.cholesky_ex(system, check_errors=False)
            require_true(
                (info == 0).all(), "El sistema regularizado de CRRM no es definido positivo"
            )
            weights = torch.cholesky_solve(stats.rhs, chol)
            raw_pred = torch.einsum("nd,rdc->nrc", k_m, weights)
            k_rhs = k_m.transpose(0, 1).unsqueeze(0).expand(scales, dim, len(members))
            whitened = torch.linalg.solve_triangular(chol, k_rhs, upper=False)
            k_norm_sq = k_m.square().sum(dim=-1).unsqueeze(0)
            leverage = ridge[:, None] * whitened.square().sum(dim=1)
            conf = (k_norm_sq - leverage).clamp(0.0, 1.0).transpose(0, 1).unsqueeze(-1)
            part = (conf * torch.tanh(raw_pred)).reshape(len(members), scales * TARGET_CHANNELS)
            if contiguous:
                features = part
            else:
                features = features.index_copy(0, index, part)
        check_finite(features, "Los rasgos asociativos de CRRM")
        return features

    def read(
        self,
        keys: Tensor,
        markets: tuple[str, ...] | list[str],
        state: CrossSectionalRankMemoryState,
    ) -> Tensor:
        """Residuo [N, H] sumado a la representación antes de la cabeza."""
        features = self.read_features(keys, markets, state)
        delta = self.out(features)
        check_finite(delta, "El residuo de la memoria transversal de rango")
        return delta

    def stage(
        self,
        state: CrossSectionalRankMemoryState,
        flow_ids: tuple[str, ...] | list[str],
        prediction_at: tuple[int, ...] | list[int],
        keys: Tensor,
        issued_predictions: Tensor,
        *,
        differentiable: bool = False,
    ) -> CrossSectionalRankMemoryState:
        """Registrar claves emitidas y predicciones puntuales a la espera de maduración."""
        check_differentiable(differentiable)
        reference = self._reference()
        if (
            not isinstance(keys, Tensor)
            or keys.ndim != 2
            or keys.shape != (len(flow_ids), self.config.dim)
            or keys.dtype != reference.dtype
            or keys.device != reference.device
            or not isinstance(issued_predictions, Tensor)
            or issued_predictions.shape != (len(flow_ids),)
            or len(prediction_at) != len(flow_ids)
        ):
            raise ValueError(
                "El registro pendiente de CRRM necesita claves y predicciones por fila"
            )
        check_finite(keys, "Las claves pendientes de CRRM")
        check_finite(issued_predictions, "Las predicciones emitidas para CRRM")
        stored_keys = keys if differentiable else keys.detach()
        issued = issued_predictions.detach().to(device=reference.device, dtype=reference.dtype)
        pending = dict(state.pending)
        for row, (flow, at) in enumerate(zip(flow_ids, prediction_at, strict=True)):
            entry_key = (flow, int(at))
            if entry_key in pending:
                raise ValueError("CRRM ya tiene registrada esa emisión pendiente")
            pending[entry_key] = (stored_keys, issued, row)
        return replace(state, pending=pending)

    def _shuffle_permutation(self, market: str, decision_at: int, count: int) -> Tensor:
        text = (
            f"{CROSS_SECTIONAL_RANK_MEMORY}/shuffle/"
            f"{self.seed}/{self.config.seed}/{market}/{decision_at}"
        )
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(count, generator=generator)

    def settle(
        self,
        state: CrossSectionalRankMemoryState,
        matured: list[tuple[str, int, float]] | tuple[tuple[str, int, float], ...],
        *,
        differentiable: bool = False,
        train: bool = True,
    ) -> CrossSectionalRankMemoryState:
        """Actualizar los estadísticos suficientes con etiquetas ya maduras en orden canónico."""
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
                raise ValueError("El resultado maduro para CRRM debe ser un real finito")
            entry_key = (flow, int(decision_at))
            if entry_key not in pending:
                raise ValueError("La etiqueta madura de CRRM no tiene clave emitida pendiente")
            grouped.setdefault((int(decision_at), _market(flow)), []).append(
                (flow, *pending.pop(entry_key), float(outcome))
            )
        if self.config.mode == EVAL_FROZEN and not train:
            return replace(state, pending=pending)

        markets = dict(state.markets)
        decays = torch.tensor(self.config.decays, dtype=reference.dtype, device=reference.device)
        one_minus = 1.0 - decays
        scales, dim = len(self.config.time_constants), self.config.dim
        for decision_at, market in sorted(grouped):
            rows = sorted(grouped[decision_at, market], key=lambda item: item[0])
            if len(rows) < self.config.minimum_rows:
                continue
            k_mat, issued = _gather_pending([item[1:4] for item in rows])
            if not differentiable:
                k_mat = k_mat.detach()
            outcomes = torch.tensor(
                [item[4] for item in rows], dtype=reference.dtype, device=reference.device
            )
            present = torch.ones((len(rows), 1), dtype=torch.bool, device=reference.device)
            normal_ranks = gaussian_ranks(outcomes.unsqueeze(-1), present).squeeze(-1)
            rank_residual = normal_ranks - standardized(issued)
            v_mat = torch.stack((normal_ranks, rank_residual), dim=-1)
            if self.config.mode == SHUFFLED_TARGETS:
                perm = self._shuffle_permutation(market, decision_at, len(rows)).to(
                    reference.device
                )
                v_mat = v_mat.index_select(0, perm)
            inv_n = 1.0 / float(len(rows))
            cov_s = (k_mat.transpose(0, 1) @ k_mat) * inv_n
            cov_s = 0.5 * (cov_s + cov_s.transpose(0, 1))
            rhs_s = (k_mat.transpose(0, 1) @ v_mat) * inv_n
            previous = markets.get(market)
            if previous is None:
                prev_cov = torch.zeros(
                    (scales, dim, dim), dtype=reference.dtype, device=reference.device
                )
                prev_rhs = torch.zeros(
                    (scales, dim, TARGET_CHANNELS), dtype=reference.dtype, device=reference.device
                )
                prev_weight = torch.zeros((scales,), dtype=reference.dtype, device=reference.device)
                updates = 1
            else:
                prev_cov = previous.cov if differentiable else previous.cov.detach()
                prev_rhs = previous.rhs if differentiable else previous.rhs.detach()
                prev_weight = previous.weight if differentiable else previous.weight.detach()
                updates = previous.updates + 1
            new_cov = decays[:, None, None] * prev_cov + one_minus[:, None, None] * cov_s.unsqueeze(
                0
            )
            new_rhs = decays[:, None, None] * prev_rhs + one_minus[:, None, None] * rhs_s.unsqueeze(
                0
            )
            new_weight = decays * prev_weight + one_minus
            if not differentiable:
                new_cov, new_rhs, new_weight = (
                    new_cov.detach(),
                    new_rhs.detach(),
                    new_weight.detach(),
                )
            check_finite(new_cov, "La actualización de covarianza de CRRM")
            check_finite(new_rhs, "La actualización cruzada de CRRM")
            check_finite(new_weight, "La masa efectiva de CRRM")
            markets[market] = MarketRankStatistics(new_cov, new_rhs, new_weight, updates)
        return replace(state, markets=markets, pending=pending)

    def detach_state(
        self, state: CrossSectionalRankMemoryState, parameter_id: str | None = None
    ) -> CrossSectionalRankMemoryState:
        """Cortar el grafo de autograd en las fronteras de truncamiento y actualizar huella."""
        markets = {
            market: MarketRankStatistics(
                stats.cov.detach(), stats.rhs.detach(), stats.weight.detach(), stats.updates
            )
            for market, stats in state.markets.items()
        }
        # Cada matriz de emisión se corta una vez. Si de una emisión solo quedan pendientes algunas
        # filas, se guardan esas filas y no la matriz entera, para que no se acumule.
        remaining: dict[int, tuple[Tensor, Tensor, list[tuple[tuple[str, int], int]]]] = {}
        for key, (keys, issued, row) in state.pending.items():
            remaining.setdefault(id(keys), (keys, issued, []))[2].append((key, row))
        pending = {}
        for keys, issued, entries in remaining.values():
            if len(entries) < len(keys):
                index = torch.tensor(
                    [row for _, row in entries], dtype=torch.int64, device=keys.device
                )
                compact_keys = keys.detach().index_select(0, index)
                compact_issued = issued.index_select(0, index)
                for position, (key, _) in enumerate(entries):
                    pending[key] = (compact_keys, compact_issued, position)
            else:
                clean = keys.detach()
                for key, row in entries:
                    pending[key] = (clean, issued, row)
        pending = {key: pending[key] for key in state.pending}
        return replace(
            state,
            parameter_id=parameter_id or state.parameter_id,
            markets=markets,
            pending=pending,
        )

    def export_state(self, state: CrossSectionalRankMemoryState) -> dict:
        """Exportar copias CPU sin grafo para checkpoints deterministas."""
        detached = self.detach_state(state)
        keys_sorted = sorted(detached.pending)
        if keys_sorted:
            gathered_keys, gathered_issued = _gather_pending(
                [detached.pending[k] for k in keys_sorted]
            )
            pending_keys = gathered_keys.cpu().clone()
            pending_issued = gathered_issued.to(dtype=torch.float64).cpu().clone()
        else:
            pending_keys = torch.empty((0, self.config.dim), dtype=self._reference().dtype)
            pending_issued = torch.empty((0,), dtype=torch.float64)
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            markets={
                market: dict(
                    cov=stats.cov.cpu().clone(),
                    rhs=stats.rhs.cpu().clone(),
                    weight=stats.weight.cpu().clone(),
                    updates=stats.updates,
                )
                for market, stats in sorted(detached.markets.items())
            },
            pending=dict(
                flows=[flow for flow, _ in keys_sorted],
                decision_at=torch.tensor([at for _, at in keys_sorted], dtype=torch.int64),
                keys=pending_keys,
                issued=pending_issued,
            ),
        )

    def restore_state(
        self, payload: object, parameter_id: str, *, device: torch.device | str
    ) -> CrossSectionalRankMemoryState:
        """Validar y restaurar el estado exportado sobre `device`."""
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "configuration", "markets", "pending"}
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.config.identity())
            or not isinstance(payload["markets"], dict)
            or not isinstance(payload["pending"], dict)
            or set(payload["pending"]) != {"flows", "decision_at", "keys", "issued"}
        ):
            raise ValueError("El estado guardado de CRRM no corresponde al predictor")
        reference = self._reference()
        target_device = torch.device(device)
        markets = {}
        for market, entry in payload["markets"].items():
            if not isinstance(entry, dict) or set(entry) != {"cov", "rhs", "weight", "updates"}:
                raise ValueError("Los estadísticos de mercado guardados de CRRM están incompletos")
            markets[market] = MarketRankStatistics(
                entry["cov"].to(device=target_device, dtype=reference.dtype, copy=True),
                entry["rhs"].to(device=target_device, dtype=reference.dtype, copy=True),
                entry["weight"].to(device=target_device, dtype=reference.dtype, copy=True),
                int(entry["updates"]),
            )
        raw_pending = payload["pending"]
        flows = raw_pending["flows"]
        moments = raw_pending["decision_at"].tolist()
        issued = raw_pending["issued"].to(device=target_device, dtype=reference.dtype, copy=True)
        key_tensor = raw_pending["keys"].to(device=target_device, dtype=reference.dtype, copy=True)
        if not (len(flows) == len(moments) == len(issued) == len(key_tensor)):
            raise ValueError("Las claves pendientes guardadas de CRRM no concilian")
        pending = {
            (flow, int(at)): (key_tensor, issued, row)
            for row, (flow, at) in enumerate(zip(flows, moments, strict=True))
        }
        state = CrossSectionalRankMemoryState(self.fingerprint(), parameter_id, markets, pending)
        self.validate_state(state, parameter_id)
        return state
