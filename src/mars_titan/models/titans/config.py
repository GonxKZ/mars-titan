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


# Valor por defecto de torch.nn.LayerNorm. Las fuentes no fijan el épsilon de la memoria.
LAYER_NORM_EPS = 1e-5
# La sección 4.4 del artículo no fija el núcleo de la convolución. Se adopta como decisión
# propia el valor de Gated DeltaNet y de ShortConvolution en flash-linear-attention.
PAPER_CONVOLUTION_KERNEL = 4
# Nombre de la identidad que reúne SiLU, la convolución causal de núcleo 4 y la norma L2 de
# q y k, tal como las describe la sección 4.4.
PAPER_PROJECTIONS = "titans_mac_paper_projections_v2"
# Flujos por bloque y bytes de estado admitidos. Los valores por defecto siguen en 256 y
# 64 MiB: un bloque mayor solo existe si la receta lo declara y cambia su identidad.
MAX_BLOCK_ROWS = 4096
MAX_STATE_BYTES = 1024**3


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

    def logits(
        self, theta_max: float, stability: "MemoryStability | None" = None
    ) -> tuple[float, float, float]:
        """Bias de α, η y θ para que una entrada nula produzca las tasas declaradas.

        Con α constante, (1 − α)^h = 1/2 tras h observaciones. Cuando PT1 activa la caja, las
        puertas son α = α_lo + (1 − α_lo)σ(z) y η = η_hi σ(z), así que el bias invierte esas
        funciones y no la sigmoide simple. Si la tasa declarada queda fuera de la caja no hay
        ningún bias que la produzca y la configuración se rechaza.
        """
        if not self.theta < theta_max:
            raise ValueError("theta inicial debe ser menor que theta_max")
        rate = math.log(2) / self.alpha_half_life
        if stability is not None and stability.gate_box:
            # α se calcula como −expm1(−rate) para no perder cifras al restar 1 − 2^(−1/h).
            alpha = -math.expm1(-rate)
            if not stability.alpha_floor < alpha:
                raise ValueError("La semivida inicial exige un olvido mayor que alpha_floor")
            if not self.eta < stability.eta_ceiling:
                raise ValueError("eta inicial debe ser menor que eta_ceiling")
            fraction = (alpha - stability.alpha_floor) / (1 - stability.alpha_floor)
            return (
                _logit(fraction),
                _logit(self.eta / stability.eta_ceiling),
                _logit(self.theta / theta_max),
            )
        # logit(α) = log(α) − log(1 − α), con α = 1 − 2^(−1/h) calculado sin cancelación.
        alpha = math.log(-math.expm1(-rate)) + rate
        return alpha, _logit(self.eta), _logit(self.theta / theta_max)


# Cajas (alpha_floor, eta_ceiling, theta_max) en las que el certificado encuentra una función
# de Lyapunov común para la memoria lineal de profundidad 1. Una caja contenida en otra hereda
# la garantía. La memoria de dos capas con LayerNorm no tiene certificado. La derivación está
# en docs/research/titans-memory-certificate.md.
CERTIFIED_GATE_BOXES = {"pt1": (1 / 500, 3 / 10, 1 / 10), "b2": (1 / 100, 1 / 2, 1 / 10)}


@dataclass(frozen=True)
class MemoryStability:
    """Primera propuesta posterior a Titans (PT1), con dos piezas que se activan por separado.

    Con `gate_box`, las puertas pasan a α = α_lo + (1 − α_lo)σ(·) y η = η_hi σ(·), mientras
    que θ = θ_max σ(·) no cambia. Con `gradient_clip` igual a G, cada fila de cada matriz del
    gradiente asociativo se reescala para que su norma euclídea no supere G antes de entrar
    en el momentum. Así la Proposición 5 del certificado acota el estado con cualquier
    profundidad. Las ecuaciones, su fuente y sus límites están en
    docs/engineering/titans-memory-stability.md.
    """

    alpha_floor: float = 1 / 500
    eta_ceiling: float = 3 / 10
    gradient_clip: float | None = None
    gate_box: bool = True

    def __post_init__(self) -> None:
        floor = _finite_number(self.alpha_floor, "alpha_floor")
        ceiling = _finite_number(self.eta_ceiling, "eta_ceiling")
        if not 0 < floor < 1:
            raise ValueError("alpha_floor debe pertenecer a (0, 1)")
        if not 0 < ceiling < 1:
            raise ValueError("eta_ceiling debe pertenecer a (0, 1)")
        if type(self.gate_box) is not bool:
            raise ValueError("gate_box debe ser booleano")
        clip = self.gradient_clip
        if clip is not None:
            clip = _finite_number(clip, "gradient_clip")
            if not 0 < clip <= 1e6:
                raise ValueError("gradient_clip debe pertenecer a (0, 1e6]")
        if not self.gate_box and clip is None:
            raise ValueError("La estabilidad declarada debe activar la caja o el recorte")
        object.__setattr__(self, "alpha_floor", floor)
        object.__setattr__(self, "eta_ceiling", ceiling)
        object.__setattr__(self, "gradient_clip", clip)

    def certified_box(self, theta_max: float) -> str | None:
        """Nombre de la caja certificada que contiene la declarada, o None si no hay ninguna.

        La garantía solo vale para la memoria lineal, así que no certifica la memoria de dos
        capas con LayerNorm que usa la campaña.
        """
        if not self.gate_box:
            return None
        for name, (floor, ceiling, theta) in CERTIFIED_GATE_BOXES.items():
            if self.alpha_floor >= floor and self.eta_ceiling <= ceiling and theta_max <= theta:
                return name
        return None

    def identity(self) -> dict:
        return {
            **asdict(self),
            "gate_map": "alpha_floor_plus_scaled_sigmoid_and_eta_scaled_sigmoid_v1"
            if self.gate_box
            else "unchanged",
            "clip_rule": "inner_gradient_row_l2_before_momentum_v1"
            if self.gradient_clip is not None
            else "none",
            "source": "post_titans_pt1_v1",
        }


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
    # M(x) = x + LN(MLP(x)), sección 3.3 de las actas. LN sin afinidad aprendida.
    residual_layer_norm: bool = False
    # La sección 4.4 del artículo aplica SiLU al calcular consultas, claves y valores.
    qkv_silu: bool = False
    # Tamaño K del núcleo de la convolución causal que sigue a cada proyección. Con 0 no hay
    # convolución.
    qkv_convolution: int = 0
    # Sin PT1 (None) la identidad y el cálculo son los de la versión anterior, bit a bit.
    stability: MemoryStability | None = None

    def __post_init__(self) -> None:
        bounded_integer(self.dim, "dim", 1, 512)
        if type(self.qkv_silu) is not bool:
            raise ValueError("qkv_silu debe ser booleano")
        if self.qkv_convolution != 0:
            bounded_integer(self.qkv_convolution, "qkv_convolution", 2, 8)
        elif type(self.qkv_convolution) is not int:
            raise ValueError("qkv_convolution debe ser 0 o un entero entre 2 y 8")
        if type(self.residual_layer_norm) is not bool:
            raise ValueError("residual_layer_norm debe ser booleano")
        if self.residual_layer_norm and self.dim < 2:
            raise ValueError("LayerNorm sobre una sola dimensión anula siempre la lectura")
        bounded_integer(self.depth, "depth", 1, 2)
        bounded_integer(self.max_batch, "max_batch", 1, MAX_BLOCK_ROWS)
        bounded_integer(self.max_tokens, "max_tokens", 1, 256)
        bounded_integer(self.max_state_bytes, "max_state_bytes", 1, MAX_STATE_BYTES)
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
        if self.stability is not None and not isinstance(self.stability, MemoryStability):
            raise ValueError("stability debe ser MemoryStability o None")
        if self.gate_bias is not None:
            if not isinstance(self.gate_bias, GateBias):
                raise ValueError("gate_bias debe ser GateBias o None")
            self.gate_bias.logits(self.theta_max, self.stability)

    @property
    def window(self) -> int:
        """Entradas anteriores que guarda cada ventana causal, K − 1, o 0 sin convolución."""
        return max(self.qkv_convolution - 1, 0)

    def identity(self) -> dict:
        fields = asdict(self)
        gate_bias = fields.pop("gate_bias")
        residual = fields.pop("residual_layer_norm")
        silu = fields.pop("qkv_silu")
        kernel = fields.pop("qkv_convolution")
        fields.pop("stability")
        # Sin bias ni residual se conserva literalmente la identidad v1 y su huella.
        extension = (
            {}
            if gate_bias is None
            else {"gate_bias": {**gate_bias, "init": "constant_logit_bias_v1_weight_draws"}}
        )
        if residual:
            extension["memory_function"] = {
                "form": "input_plus_layer_norm_of_mlp",
                "source": "neurips_2025_section_3_3",
                "layer_norm_eps": LAYER_NORM_EPS,
                "layer_norm_affine": False,
                "expansion": 1,
            }
        # Sin SiLU ni convolución se conserva también la identidad anterior y su huella.
        if silu:
            extension["qkv_activation"] = {
                "function": "silu",
                "streams": ["query", "key", "value"],
                "source": "arxiv_2501_00663_section_4_4",
            }
        if kernel:
            extension["qkv_convolution"] = {
                "kernel": kernel,
                "form": "causal_depthwise_per_channel",
                "streams": ["query", "key", "value"],
                "bias": False,
                "initial_window": "zeros",
                "state": "previous_projections_per_flow",
                "init": "conv1d_default_after_previous_draws",
                "source": "arxiv_2501_00663_section_4_4",
                "kernel_source": "gated_deltanet_and_fla_short_convolution",
            }
        if silu or kernel:
            extension["projection_order"] = "linear_convolution_silu_then_l2_query_key"
        if silu and kernel == PAPER_CONVOLUTION_KERNEL and self.normalize_qk:
            extension["projections"] = PAPER_PROJECTIONS
        if self.stability is not None:
            extension["stability"] = self.stability.identity()
        return {
            **fields,
            **extension,
            "schema_version": 1,
            "architecture": "square_mlp",
            "hidden_activation": "gelu_exact",
            "bias": False,
            "residual": residual,
            "layer_norm": residual,
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
