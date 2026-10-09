"""Contratos identificados y presupuestos del núcleo de memoria neuronal."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass


def bounded_integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}")


def canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("La configuración no tiene un contrato JSON válido") from error


def require_identity(actual: object, expected: dict) -> None:
    if canonical(actual) != canonical(expected):
        raise ValueError("La configuración no coincide con el contrato del módulo")


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    return float(value)


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


@dataclass(frozen=True)
class GateBias:
    """Tasas iniciales declaradas para entrada nula. Los pesos conservan la dependencia."""

    alpha_half_life: float = 256.0
    eta: float = 0.5
    theta: float = 0.05

    def __post_init__(self) -> None:
        half_life = _finite_number(self.alpha_half_life, "alpha_half_life")
        eta = _finite_number(self.eta, "eta")
        theta = _finite_number(self.theta, "theta")
        if not 1 <= half_life <= 1e6:
            raise ValueError("alpha_half_life debe pertenecer a [1, 1e6] observaciones")
        if not 0 < eta < 1:
            raise ValueError("eta inicial debe pertenecer a (0, 1)")
        if not 0 < theta < 1:
            raise ValueError("theta inicial debe pertenecer a (0, 1)")
        object.__setattr__(self, "alpha_half_life", half_life)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "theta", theta)

    def logits(self, theta_max: float) -> tuple[float, float, float]:
        """Bias de α, η y θ. Con α constante, (1 − α)^h = 1/2 tras h observaciones."""
        if not self.theta < theta_max:
            raise ValueError("theta inicial debe ser menor que theta_max")
        rate = math.log(2) / self.alpha_half_life
        # logit(α) = log(α) − log(1 − α), con α = 1 − 2^(−1/h) calculado sin cancelación.
        alpha = math.log(-math.expm1(-rate)) + rate
        return alpha, _logit(self.eta), _logit(self.theta / theta_max)


@dataclass(frozen=True)
class MemoryConfig:
    dim: int
    depth: int = 2
    normalize_qk: bool = True
    theta_max: float = 0.1
    max_batch: int = 256
    max_tokens: int = 64
    max_state_bytes: int = 64 * 1024 * 1024
    parameter_seed: int = 42
    gate_bias: GateBias | None = None

    def __post_init__(self) -> None:
        bounded_integer(self.dim, "dim", 1, 512)
        bounded_integer(self.depth, "depth", 1, 2)
        bounded_integer(self.max_batch, "max_batch", 1, 256)
        bounded_integer(self.max_tokens, "max_tokens", 1, 256)
        bounded_integer(self.max_state_bytes, "max_state_bytes", 1, 256 * 1024 * 1024)
        bounded_integer(self.parameter_seed, "parameter_seed", 0, 2**63 - 1)
        if type(self.normalize_qk) is not bool:
            raise ValueError("normalize_qk debe ser booleano")
        if (
            type(self.theta_max) not in (int, float)
            or not math.isfinite(self.theta_max)
            or not 0 < self.theta_max <= 1
        ):
            raise ValueError("theta_max debe ser finito y pertenecer a (0, 1]")
        object.__setattr__(self, "theta_max", float(self.theta_max))
        if self.gate_bias is not None:
            if not isinstance(self.gate_bias, GateBias):
                raise ValueError("gate_bias debe ser GateBias o None")
            self.gate_bias.logits(self.theta_max)

    def identity(self) -> dict:
        fields = asdict(self)
        gate_bias = fields.pop("gate_bias")
        # Sin bias se conserva literalmente la identidad v1 y su huella.
        extension = (
            {}
            if gate_bias is None
            else {"gate_bias": {**gate_bias, "init": "constant_logit_bias_v1_weight_draws"}}
        )
        return {
            **fields,
            **extension,
            "schema_version": 1,
            "architecture": "square_mlp",
            "hidden_activation": "gelu_exact",
            "bias": False,
            "residual": False,
            "layer_norm": False,
            "normalization_eps": 1e-12,
            "loss_reduction": "sum_per_token_per_flow",
            "forgetting_broadcast": "output_rows_each_matrix",
            "rates": "sigmoid_alpha_vector_eta_scalar_scaled_theta_scalar",
            "update_order": "sequential_current_weights",
            "inner_derivative": "partial_at_fixed_observation",
            "gradient_policy": "explicit_differentiable_or_detached",
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical(self.identity()).encode()).hexdigest()


@dataclass(frozen=True)
class MACConfig:
    memory: MemoryConfig
    heads: int = 1
    persistent_tokens: int = 4
    max_segment: int = 32
    memory_mode: str = "online"
    max_attention_elements: int = 8 * 1024 * 1024
    parameter_seed: int = 43

    def __post_init__(self) -> None:
        if not isinstance(self.memory, MemoryConfig):
            raise ValueError("memory debe ser una configuración de memoria")
        bounded_integer(self.heads, "heads", 1, self.memory.dim)
        bounded_integer(self.persistent_tokens, "persistent_tokens", 0, 64)
        bounded_integer(self.max_segment, "max_segment", 1, self.memory.max_tokens)
        bounded_integer(self.max_attention_elements, "max_attention_elements", 1, 64 * 1024 * 1024)
        bounded_integer(self.parameter_seed, "parameter_seed", 0, 2**63 - 1)
        if self.memory.dim % self.heads:
            raise ValueError("heads debe dividir dim")
        if self.memory_mode not in ("online", "frozen", "disabled"):
            raise ValueError("memory_mode debe ser online, frozen o disabled")

    def identity(self) -> dict:
        return {
            **asdict(self),
            "memory": self.memory.identity(),
            "schema_version": 1,
            "layout": "persistent_memory_segment",
            "output_gate": "hadamard",
            "output_granularity": "segment_close",
            "write_positions": "segment_only",
            "attention_mask": "triangular_on_packed_sequence",
            "attention_bias": False,
            "dropout": 0.0,
            "disabled_policy": "attention_only_without_prefix_or_gate",
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical(self.identity()).encode()).hexdigest()
