"""Cabeza común `quantile_head_v1`: cinco cuantiles ordenados y media de pinball.

La parametrización replica `Candidate::quantiles` de `native/src/candidate.cpp`:
una proyección lineal produce cinco valores sin restricción. El tercero es la
mediana libre y los demás se convierten en incrementos positivos con softplus,
acumulados desde la mediana hacia cada cola. El orden no decreciente se cumple
por construcción, también en coma flotante, porque restar o sumar un valor no
negativo nunca cruza el operando al redondear.

La pérdida es la media de pinball con pesos iguales sobre los cinco niveles y la
predicción puntual es la mediana. En un residuo nulo se usa el subgradiente
medio (tau - 1/2), el mismo que da `torch.maximum` al repartir el empate. Así el
término de la mediana vale exactamente la mitad de `l1_loss`, también en el cero.
"""

import functools

import torch
from torch import nn
from torch.nn import functional

QUANTILE_HEAD = "quantile_head_v1"
PINBALL = "pinball"
LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
MEDIAN_INDEX = 2
# Columnas de las tablas de predicción, en el orden de LEVELS y del panel de evaluación.
QUANTILE_COLUMNS = (
    "quantile_0025",
    "quantile_0100",
    "quantile_0500",
    "quantile_0900",
    "quantile_0975",
)
CONTRACT = dict(
    schema_version=1,
    name=QUANTILE_HEAD,
    levels=list(LEVELS),
    columns=list(QUANTILE_COLUMNS),
    parametrization="free_median_softplus_increments_as_native_candidate",
    softplus=dict(beta=1, threshold=20),
    point_prediction="median",
    loss="mean_pinball_equal_level_weights",
    zero_residual_subgradient="midpoint_tau_minus_half",
)


def ordered_quantiles(raw):
    """Convertir [..., 5] valores libres en cinco cuantiles no decrecientes."""
    if not isinstance(raw, torch.Tensor) or raw.ndim < 1 or raw.shape[-1] != len(LEVELS):
        raise ValueError("La cabeza de cuantiles necesita cinco valores por fila")
    median = raw[..., MEDIAN_INDEX]
    lower = median - functional.softplus(raw[..., 1])
    upper = median + functional.softplus(raw[..., 3])
    return torch.stack(
        (
            lower - functional.softplus(raw[..., 0]),
            lower,
            median,
            upper,
            upper + functional.softplus(raw[..., 4]),
        ),
        dim=-1,
    )


def median(quantiles):
    """Predicción puntual de la cabeza, sin copiar ni recalcular."""
    if (
        not isinstance(quantiles, torch.Tensor)
        or quantiles.ndim < 1
        or quantiles.shape[-1] != len(LEVELS)
    ):
        raise ValueError("Se esperan cinco cuantiles por fila")
    return quantiles[..., MEDIAN_INDEX]


@functools.cache
def _levels(dtype, device):
    """Niveles en el dispositivo, creados una vez. Copiarlos en cada llamada sincronizaba la GPU.

    Solo se leen, así que compartir el tensor no cambia ningún valor ni gradiente. Se
    crean fuera de `inference_mode` para que el ajuste pueda guardarlos en su grafo.
    """
    with torch.inference_mode(False):
        return torch.tensor(LEVELS, dtype=dtype, device=device)


def pinball_loss(quantiles, target, *, reduction="mean"):
    """Media de pinball de los cinco niveles por fila y, opcionalmente, entre filas.

    Con residuo u = y - q, cada nivel aporta max(tau u, (tau - 1) u). `reduction="none"`
    devuelve una pérdida por fila para que el llamante pueda ponderar mercados.
    """
    if reduction not in ("mean", "none"):
        raise ValueError("La reducción debe ser mean o none")
    if (
        not isinstance(quantiles, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or quantiles.ndim != 2
        or quantiles.shape != (len(target), len(LEVELS))
        or target.ndim != 1
        or len(target) < 1
    ):
        raise ValueError("Los cuantiles deben tener forma [filas, 5] y el objetivo [filas]")
    if quantiles.dtype != target.dtype or quantiles.device != target.device:
        raise ValueError("Cuantiles y objetivo deben compartir precisión y dispositivo")
    if not quantiles.dtype.is_floating_point:
        raise ValueError("La pérdida pinball necesita valores reales")
    levels = _levels(quantiles.dtype, quantiles.device)
    residual = target.unsqueeze(-1) - quantiles
    per_row = torch.maximum(levels * residual, (levels - 1) * residual).mean(dim=-1)
    return per_row if reduction == "none" else per_row.mean()


class QuantileHead(nn.Linear):
    """Proyección lineal a cinco valores seguida de la parametrización ordenada.

    Hereda de `nn.Linear` para conservar `weight` y `bias`, su inicialización y los
    nombres del estado (`head.weight`, `head.bias`), como `head_weight` y `head_bias`
    en el candidato nativo. La salida tiene forma [filas, 5].
    """

    def __init__(self, in_features, *, device=None, dtype=None):
        if type(in_features) is not int or not 1 <= in_features <= 4096:
            raise ValueError("La anchura de entrada de la cabeza no es válida")
        super().__init__(in_features, len(LEVELS), device=device, dtype=dtype)

    def forward(self, value):
        return ordered_quantiles(super().forward(value))
