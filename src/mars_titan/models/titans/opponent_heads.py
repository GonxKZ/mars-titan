"""Cabezas oponentes con un integrador neutral fijo (A4 de #749, `opponent_heads_v1`).

La cabeza escalar del predictor da el centro c del estado codificado h. Una proyección
lineal nueva da la semiamplitud d, con pesos y sesgo iniciales nulos. Las dos cabezas son

    pesimista = c − d        optimista = c + d

y el integrador neutral emite su media, c. Cada cabeza se ajusta con su nivel asimétrico,
τ y 1 − τ con τ < 0,5, en la familia declarada:

    expectil   ρ_τ(u) = |τ − 1(u < 0)| · u²
    cuantil    ρ_τ(u) = (τ − 1(u < 0)) · u

con u = y − cabeza y la pérdida (1/N) Σ_i [ρ_τ(y_i − c_i + d_i) + ρ_{1−τ}(y_i − c_i − d_i)].

Con d = 0 la suma de los dos niveles vale u² en la familia de expectiles y |u| en la de
cuantiles, así que c recibe al empezar el gradiente del error cuadrático o del absoluto. Por
fila, la derivada respecto a d en d = 0 es −2|u|(1 − 2τ) con expectiles y −(1 − 2τ) con
cuantiles si u ≠ 0. Es negativa para τ < 0,5, así que el descenso aumenta d y separa las dos
cabezas. Nada impide que d sea negativa en alguna fila. Las cabezas se cruzarían, pero el
punto emitido no cambia.

(c, d) ↦ (c − d, c + d) es biyectivo, así que en el mínimo cada cabeza alcanza su nivel y c
estima el punto medio de los dos, (q_τ + q_{1−τ})/2. Con cuantiles y τ = 0,25 es el centro
intercuartílico. Solo coincide con la mediana, que optimiza el MAE de la selección, si la
distribución condicional es simétrica.

El control declara τ = 0,5 en ambas cabezas, con los mismos parámetros. Entonces la
derivada respecto a d en d = 0 es exactamente cero, d no se mueve de cero y c se ajusta con
la pérdida simétrica de su familia. Con cuantiles es el error absoluto de la referencia. Con
expectiles es el error cuadrático, que cambia también la escala del gradiente respecto a la
receta de error absoluto. Por eso cada familia se compara con su propio control.

Fuentes: Newey y Powell (1987), Asymmetric least squares estimation and testing,
Econometrica 55(4), https://doi.org/10.2307/1911031, para los expectiles. Dabney et al.
(2020), A distributional code for value in dopamine-based reinforcement learning, Nature
577, 671-675, https://doi.org/10.1038/s41586-019-1924-6, para la idea de neuronas
optimistas y pesimistas con niveles asimétricos. El integrador fijo y su uso como predicción
puntual son una decisión del proyecto.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

OPPONENT_HEADS = "opponent_heads_v1"
FAMILIES = ("expectile", "quantile")


@dataclass(frozen=True)
class OpponentHeads:
    # Familia de la pérdida asimétrica. El control usa la misma familia con τ = 0,5.
    family: str = "expectile"
    # Nivel de la cabeza pesimista. La optimista usa 1 − τ y el integrador es su media.
    level: float = 0.25

    def __post_init__(self):
        if type(self.family) is not str or self.family not in FAMILIES:
            raise ValueError("Las cabezas oponentes usan expectiles o cuantiles")
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, (int, float))
            or not math.isfinite(self.level)
            or not 0.0 < self.level <= 0.5
        ):
            raise ValueError("El nivel pesimista debe estar en (0, 0,5]")
        object.__setattr__(self, "level", float(self.level))

    @property
    def name(self):
        return OPPONENT_HEADS

    @property
    def control(self):
        """El control de la familia: las mismas cabezas sin asimetría."""
        return self.level == 0.5

    def identity(self):
        return dict(
            schema_version=1,
            name=OPPONENT_HEADS,
            family=self.family,
            levels=[self.level, 1.0 - self.level],
            role="symmetric_control" if self.control else "opponent_variant",
            heads="scalar_head_center_minus_and_plus_zero_initialized_half_width",
            integrator="fixed_mean_of_both_heads",
            loss=f"mean_of_asymmetric_{self.family}_losses_of_both_heads",
        )

    def build(self, hidden_size, *, seed, dtype):
        return OpponentSpread(hidden_size, self, dtype=dtype)


class OpponentSpread(nn.Module):
    """Proyección lineal del estado codificado a la semiamplitud d, nula al empezar.

    Es el módulo del objetivo de salida A4: las salidas del ajuste son [c, d] por fila y la
    predicción emitida es el centro c de la cabeza escalar, la media fija de las dos cabezas.
    """

    session_level = False
    row_decomposable = True
    # El integrador emite la cabeza escalar sin transformar.
    scalar_point = True

    def __init__(self, hidden_size, config, *, dtype=torch.float32):
        super().__init__()
        if type(config) is not OpponentHeads:
            raise ValueError("La semiamplitud necesita la configuración de las cabezas")
        self.config = config
        # Sin el generador global: el resto del predictor conserva los pesos del control.
        self.weight = nn.Parameter(torch.zeros(1, hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype))

    def spread(self, encoded):
        return functional.linear(encoded, self.weight, self.bias).squeeze(-1)

    def forward(self, point, encoded, markets):
        return torch.stack((point, self.spread(encoded)), dim=-1), point

    def loss(self, outputs, target, sessions):
        return opponent_loss(outputs, target, self.config)


def opponent_loss(prediction, target, config):
    """Media por fila de las pérdidas de ambas cabezas. `prediction` es [filas, 2] con c y d."""
    if (
        type(config) is not OpponentHeads
        or not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.ndim != 2
        or prediction.shape != (len(target), 2)
        or target.ndim != 1
        or len(target) == 0
        or prediction.dtype != target.dtype
        or prediction.device != target.device
        or not prediction.dtype.is_floating_point
    ):
        raise ValueError("La pérdida oponente necesita centro y semiamplitud por etiqueta")
    center, spread = prediction.unbind(-1)
    total = 0.0
    for head, level in ((center - spread, config.level), (center + spread, 1.0 - config.level)):
        error = target - head
        weight = torch.where(error < 0, 1.0 - level, level)
        total = total + (weight * (error * error if config.family == "expectile" else error.abs()))
    return total.mean()


def opponent_heads(center, spread):
    """Cabezas pesimista y optimista [filas, 2]. El integrador emite su media, el centro."""
    if center.shape != spread.shape or center.ndim != 1:
        raise ValueError("Las cabezas necesitan un centro y una semiamplitud por fila")
    return torch.stack((center - spread, center + spread), dim=-1)
