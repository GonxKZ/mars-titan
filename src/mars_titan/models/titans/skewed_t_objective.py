"""t de Student asimétrica ajustada con β-NLL (L2 de #787, `skewed_t_objective_v1`).

La cabeza escalar del predictor sigue emitiendo la posición μ. Tres proyecciones lineales
nuevas del mismo estado codificado h dan la escala, la asimetría y los grados de libertad

    σ = softplus(a_σ) + σ_min,   ξ = ξ_max^tanh(a_ξ),   ν = ν_min + softplus(a_ν),

con a = W·h + c y W = 0 al construirse, así que cada parámetro empieza en el valor declarado.
Con z = (y − μ)/σ, la densidad es la de la ecuación (7) de Michańków (2025), que parte la t
de Student estándar t_ν en cero con la transformación de Fernández y Steel:

    f(y) = 2 / (σ (ξ + 1/ξ)) · t_ν(ξ z)   si z < 0,
    f(y) = 2 / (σ (ξ + 1/ξ)) · t_ν(z / ξ) si z ≥ 0.

La masa por debajo de μ es 1/(1 + ξ²), así que ξ > 1 alarga la cola derecha y ξ = 1 da la t
simétrica. El ajuste sustituye el error absoluto por la NLL de la ecuación (16), ponderada en
cada fila por la escala sin gradiente, como la β-NLL de F1:

    L = (1/N) Σ_i sg(σ_i) · NLL_i.

Con x = ξ z si z < 0 y x = z/ξ si z ≥ 0, el gradiente de μ en cada fila es

    ∂L/∂μ = −(1/N) · s · (ν + 1) x / (ν + x²),   s = ξ o 1/ξ según el lado,

la puntuación acotada de la t, que no depende de la escala de la fila. La escala conserva su
óptimo porque el peso sg(σ) es positivo y no tiene gradiente.

La predicción emitida es la mediana de la distribución ajustada, como la de la Laplace de F1
y la de la cabeza de cuantiles, así que el contraste con F1 cambia la familia y no el
funcional del punto. μ es la moda y solo coincide con la mediana cuando ξ = 1. La mediana y
los cinco niveles emitidos son cuantiles exactos de la distribución, calculados sin gradiente
con la cuantílica de Student de SciPy, que PyTorch no ofrece. La pérdida solo usa las salidas
[μ, σ, ξ, ν], así que el punto emitido no necesita gradiente.

Desviaciones declaradas: el artículo no fija las activaciones de la capa de salida, así que
σ_min, la cota ξ_max de la asimetría y el suelo ν_min de los grados de libertad son decisiones
propias. Con ν_min = 2 la varianza siempre existe. El peso sg(σ) de la β-NLL tampoco está en
el artículo. Se usa como en F1, para que la comparación entre las dos cabezas cambie la
distribución y no la ponderación de las filas.

Fuentes: Michańków, J. (2025). Forecasting Probability Distributions of Financial Returns with
Deep Neural Networks. arXiv:2508.18921. Seitzer, M., Tavakoli, A., Antic, D. y Martius, G.
(2022). On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural
networks. ICLR 2022. arXiv:2203.09168.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import special
from torch import nn
from torch.nn import functional

from mars_titan.models.quantile_head import LEVELS

SKEWED_T_OBJECTIVE = "skewed_t_objective_v1"
MEDIAN = (0.5,)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    return float(value)


def _inverse_softplus(value):
    # softplus⁻¹(x) = x + log(1 − exp(−x)), estable también cuando x es grande.
    return value + math.log(-math.expm1(-value))


@dataclass(frozen=True)
class SkewedTObjective:
    # Suelo de la escala en unidades del objetivo. Mantiene finitos 1/σ y log σ.
    floor: float = 1e-6
    # Escala inicial común, la misma que la de F1.
    initial_scale: float = 1e-2
    # Grados de libertad mínimos e iniciales. Con más de 2 la varianza existe.
    dof_floor: float = 2.0
    initial_dof: float = 5.0
    # La asimetría queda entre 1/ξ_max y ξ_max y empieza en 1, la t simétrica.
    max_skew: float = 3.0

    def __post_init__(self):
        floor = _number(self.floor, "floor")
        initial = _number(self.initial_scale, "initial_scale")
        dof_floor = _number(self.dof_floor, "dof_floor")
        dof = _number(self.initial_dof, "initial_dof")
        skew = _number(self.max_skew, "max_skew")
        if not 1e-12 <= floor <= 1e-2 or not 2 * floor <= initial <= 1.0:
            raise ValueError("La escala necesita 1e-12 <= floor, 2·floor <= initial_scale <= 1")
        if not 2.0 <= dof_floor < dof <= 100.0:
            raise ValueError("Los grados de libertad necesitan 2 <= dof_floor < initial_dof <= 100")
        if not 1.0 < skew <= 10.0:
            raise ValueError("La asimetría necesita 1 < max_skew <= 10")
        for name, value in (
            ("floor", floor),
            ("initial_scale", initial),
            ("dof_floor", dof_floor),
            ("initial_dof", dof),
            ("max_skew", skew),
        ):
            object.__setattr__(self, name, value)

    @property
    def name(self):
        return SKEWED_T_OBJECTIVE

    def identity(self):
        return dict(
            schema_version=1,
            name=SKEWED_T_OBJECTIVE,
            floor=self.floor,
            initial_scale=self.initial_scale,
            dof_floor=self.dof_floor,
            initial_dof=self.initial_dof,
            max_skew=self.max_skew,
            distribution="fernandez_steel_skewed_student_t",
            scale="softplus_of_linear_on_encoded_plus_floor",
            skew="max_skew_to_the_power_tanh_of_linear_on_encoded",
            dof="dof_floor_plus_softplus_of_linear_on_encoded",
            initial="zero_weights_constant_biases",
            loss="beta_nll_mean_of_detached_scale_times_full_nll",
            beta=1.0,
            point="median_of_the_fitted_distribution_with_scipy",
            emitted_quantiles="exact_quantiles_at_quantile_head_levels_with_scipy",
        )

    def build(self, hidden_size, *, seed, dtype):
        return SkewedT(hidden_size, self, dtype=dtype)


class SkewedT(nn.Module):
    """Proyecciones del estado codificado a la escala, la asimetría y los grados de libertad.

    Es el módulo del objetivo de salida L2: las salidas del ajuste son [μ, σ, ξ, ν] por fila,
    la predicción emitida es la mediana de la distribución y las filas emitidas llevan sus
    cuantiles.
    """

    session_level = False
    row_decomposable = True
    # El punto emitido es la mediana ajustada, no la salida de la cabeza escalar.
    scalar_point = False

    def __init__(self, hidden_size, config, *, dtype=torch.float32):
        super().__init__()
        if type(config) is not SkewedTObjective:
            raise ValueError("La t asimétrica necesita su configuración")
        self.config = config
        # Inicialización determinista, sin el generador global: el resto del predictor
        # conserva los pesos del control con la misma semilla. tanh(0) = 0 da ξ = 1.
        bias = (
            _inverse_softplus(config.initial_scale - config.floor),
            0.0,
            _inverse_softplus(config.initial_dof - config.dof_floor),
        )
        self.weight = nn.Parameter(torch.zeros(3, hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=dtype))

    def parameters_of(self, encoded):
        """Escala, asimetría y grados de libertad [filas] del estado codificado."""
        scale, skew, dof = functional.linear(encoded, self.weight, self.bias).unbind(-1)
        config = self.config
        return (
            functional.softplus(scale) + config.floor,
            torch.exp(math.log(config.max_skew) * torch.tanh(skew)),
            functional.softplus(dof) + config.dof_floor,
        )

    def forward(self, point, encoded, markets):
        outputs = torch.stack((point, *self.parameters_of(encoded)), dim=-1)
        return outputs, skewed_t_quantiles(outputs.detach(), MEDIAN)[:, 0]

    def loss(self, outputs, target, sessions):
        return skewed_t_beta_nll(outputs, target)

    def levels(self, outputs):
        """Cuantiles emitidos [filas, 5] a partir de las salidas [μ, σ, ξ, ν] sin gradiente."""
        return skewed_t_quantiles(outputs.detach(), LEVELS)


def _checked(outputs, target):
    if (
        not isinstance(outputs, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or outputs.ndim != 2
        or outputs.shape != (len(target), 4)
        or target.ndim != 1
        or len(target) == 0
        or outputs.dtype != target.dtype
        or outputs.device != target.device
        or not outputs.dtype.is_floating_point
    ):
        raise ValueError(
            "La t asimétrica necesita posición, escala, asimetría y grados por etiqueta"
        )


def skewed_t_nll(outputs, target):
    """NLL completa por fila de la t asimétrica. `outputs` es [filas, 4] con μ, σ, ξ y ν."""
    _checked(outputs, target)
    point, scale, skew, dof = outputs.unbind(-1)
    z = (target - point) / scale
    # ξ estrecha el lado izquierdo y 1/ξ ensancha el derecho cuando ξ > 1.
    x = torch.where(z < 0, z * skew, z / skew)
    log_t = (
        torch.lgamma((dof + 1) / 2)
        - torch.lgamma(dof / 2)
        - 0.5 * torch.log(dof * math.pi)
        - (dof + 1) / 2 * torch.log1p(x.square() / dof)
    )
    return torch.log(skew + 1 / skew) - math.log(2) + torch.log(scale) - log_t


def skewed_t_beta_nll(outputs, target):
    """Media por fila de sg(σ) · NLL, la β-NLL de F1 aplicada a la t asimétrica."""
    nll = skewed_t_nll(outputs, target)
    return (outputs[:, 1].detach() * nll).mean()


def skewed_t_quantiles(outputs, levels):
    """Cuantiles exactos [filas, niveles] de las salidas [μ, σ, ξ, ν], sin gradiente.

    Por debajo de la masa 1/(1 + ξ²) de la izquierda, el nivel τ cae en z = T⁻¹(τ(1 + ξ²)/2)/ξ.
    Por encima, en z = ξ · T⁻¹(1 − (1 − τ)(1 + ξ²)/(2ξ²)), donde T⁻¹ es la cuantílica de la t
    estándar con ν grados. SciPy la calcula en float64 sobre la CPU.
    """
    if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2 or outputs.shape[1] != 4:
        raise ValueError("Los cuantiles necesitan posición, escala, asimetría y grados por fila")
    if outputs.requires_grad:
        raise ValueError("Los cuantiles emitidos se calculan sin gradiente")
    values = outputs.to("cpu", torch.float64).numpy()
    point, scale, skew, dof = (column[:, None] for column in values.T)
    tau = np.asarray(levels, dtype=np.float64)[None, :]
    mass = 1 + skew**2
    left = tau < 1 / mass
    # Las dos ramas se calculan en todas las celdas y np.where elige la de cada nivel.
    with np.errstate(invalid="ignore"):
        below = special.stdtrit(dof, np.clip(tau * mass / 2, 0, 1)) / skew
        above = skew * special.stdtrit(dof, np.clip(1 - (1 - tau) * mass / (2 * skew**2), 0, 1))
    quantiles = point + scale * np.where(left, below, above)
    if not np.isfinite(quantiles).all():
        raise ValueError("Los cuantiles de la t asimétrica no son finitos")
    return torch.from_numpy(quantiles).to(outputs.device, outputs.dtype)
