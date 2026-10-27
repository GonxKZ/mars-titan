"""Escala de Laplace ajustada con β-NLL y β = 1 (F1 de #749, `laplace_scale_head_v1`).

La cabeza escalar del predictor sigue emitiendo la mediana μ. Una proyección lineal nueva del
mismo estado codificado h da la escala

    b = softplus(w·h + c) + b_min

y el ajuste sustituye el error absoluto por la NLL de Laplace ponderada con la escala sin
gradiente, la β-NLL de Seitzer et al. (2022) con β = 1:

    L = (1/N) Σ_i sg(b_i) · (|y_i − μ_i| / b_i + log b_i)

Por fila, antes de la media,

    ∂L/∂μ = sign(μ − y)
    ∂L/∂b = sg(b) · (1/b) · (1 − |y − μ| / b)

La mediana recibe el mismo gradiente que con el error absoluto del control, y la escala
conserva su óptimo en b = E|y − μ|. El tronco recibe además el gradiente de la escala.

Desviaciones declaradas: el artículo plantea la β-NLL para la gaussiana con el peso
sg(σ^(2β)). Aquí se aplica a la Laplace con el peso sg(b), que es el que deja el gradiente de
μ igual al del error absoluto. Se omite la constante log 2 de la NLL, que no cambia ningún
gradiente. El suelo b_min es una decisión propia para que 1/b y log b sean finitos.

Fuente: Seitzer, M., Tavakoli, A., Antic, D. y Martius, G. (2022). On the pitfalls of
heteroscedastic uncertainty estimation with probabilistic neural networks. ICLR 2022.
arXiv:2203.09168.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from mars_titan.models.quantile_head import LEVELS

LAPLACE_SCALE_HEAD = "laplace_scale_head_v1"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    return float(value)


@dataclass(frozen=True)
class LaplaceScaleHead:
    # Suelo de la escala en unidades del objetivo. Mantiene finitos 1/b y log b.
    floor: float = 1e-6
    # Escala inicial común. Los pesos de la proyección empiezan en cero.
    initial_scale: float = 1e-2

    def __post_init__(self):
        floor = _number(self.floor, "floor")
        initial = _number(self.initial_scale, "initial_scale")
        if not 1e-12 <= floor <= 1e-2 or not 2 * floor <= initial <= 1.0:
            raise ValueError("La escala necesita 1e-12 <= floor, 2·floor <= initial_scale <= 1")
        object.__setattr__(self, "floor", floor)
        object.__setattr__(self, "initial_scale", initial)

    def identity(self):
        return dict(
            schema_version=1,
            name=LAPLACE_SCALE_HEAD,
            floor=self.floor,
            initial_scale=self.initial_scale,
            scale="softplus_of_linear_on_encoded_plus_floor",
            initial="zero_weight_constant_bias",
            loss="beta_nll_laplace_mean_of_detached_scale_times_nll",
            beta=1.0,
            nll_constant="log_2_omitted",
            point="scalar_head_median",
            emitted_quantiles="laplace_quantiles_at_quantile_head_levels",
        )


class LaplaceScale(nn.Module):
    """Proyección lineal del estado codificado a la escala b de la Laplace."""

    def __init__(self, hidden_size, config, *, dtype=torch.float32):
        super().__init__()
        if type(config) is not LaplaceScaleHead:
            raise ValueError("La escala necesita su configuración de Laplace")
        self.config = config
        # Inicialización determinista, sin el generador global: el resto del predictor
        # conserva los pesos del control con la misma semilla.
        shifted = config.initial_scale - config.floor
        # softplus⁻¹(x) = x + log(1 − exp(−x)), estable también cuando x es grande.
        bias = shifted + math.log(-math.expm1(-shifted))
        self.weight = nn.Parameter(torch.zeros(1, hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.tensor([bias], dtype=dtype))

    def forward(self, encoded):
        logit = functional.linear(encoded, self.weight, self.bias).squeeze(-1)
        return functional.softplus(logit) + self.config.floor


def laplace_beta_nll(prediction, target):
    """Media por fila de sg(b) · (|y − μ| / b + log b). `prediction` es [filas, 2] con μ y b.

    Se escribe como |y − μ| · (sg(b) / b) + sg(b) · log b, que tiene el mismo valor. El
    cociente sg(b) / b vale exactamente 1 en coma flotante, así que la derivada respecto a μ
    es exactamente la del error absoluto, sin el redondeo de multiplicar y dividir por b.
    """
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.ndim != 2
        or prediction.shape != (len(target), 2)
        or target.ndim != 1
        or len(target) == 0
        or prediction.dtype != target.dtype
        or prediction.device != target.device
        or not prediction.dtype.is_floating_point
    ):
        raise ValueError("La pérdida de Laplace necesita mediana y escala por etiqueta")
    point, scale = prediction.unbind(-1)
    weight = scale.detach()
    return ((target - point).abs() * (weight / scale) + weight * torch.log(scale)).mean()


# Cuantil de la Laplace estándar en cada nivel de la cabeza de cuantiles: log(2τ) por debajo
# de la mediana y −log(2(1 − τ)) por encima.
_STANDARD_QUANTILES = tuple(
    math.log(2 * level) if level <= 0.5 else -math.log(2 * (1 - level)) for level in LEVELS
)


def laplace_quantiles(point, scale):
    """Cuantiles μ + b·q_τ en los niveles de `quantile_head_v1`, [filas, 5] y ordenados."""
    if point.shape != scale.shape or point.ndim != 1:
        raise ValueError("Los cuantiles necesitan una mediana y una escala por fila")
    standard = torch.tensor(_STANDARD_QUANTILES, dtype=point.dtype, device=point.device)
    return point.unsqueeze(-1) + scale.unsqueeze(-1) * standard
