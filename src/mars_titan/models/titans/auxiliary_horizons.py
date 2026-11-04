"""Cabezas auxiliares de horizontes de 5 y 20 sesiones (F5 de #749, `auxiliary_horizons_v1`).

Sean o(t) el ordinal de la decisión t entre las decisiones de su mercado en el recorrido e
y_{f,t} la etiqueta principal del flujo f, el residuo de la sesión siguiente con sus α̂ y β̂
estimados hasta su propia decisión. La etiqueta auxiliar de horizonte h de la decisión con
ordinal s es la suma de las h etiquetas principales consecutivas del mismo flujo:

    y⁽ʰ⁾_{f,s} = Σ_{j=0}^{h−1} y_{f,s+j}

La derivación reutiliza la residualización del objetivo principal y no lee ni escribe
columnas nuevas. La etiqueta madura con su último sumando. No existe si el flujo no decidió en
alguno de esos ordinales, si un sumando no madura dentro del tramo o si madura más de
`maturity_slack` decisiones de mercado después del último ordinal de la ventana.

Cada cabeza es una proyección lineal del token fusionado x, antes de MAC: a⁽ʰ⁾ = W_h x + c_h.
Con A las etiquetas auxiliares maduras desde la actualización anterior, el ajuste suma

    L_aux = w · (1/|A|) Σ_{(f,s,h)∈A} |a⁽ʰ⁾_{f,s} − y⁽ʰ⁾_{f,s}| / √h

a la pérdida principal sin cambiarla. El divisor √h iguala aproximadamente la escala de
horizontes cuyos sumandos fueran independientes. El control declarado son las mismas
cabezas con w = 0, que no reciben gradiente.

Desviaciones declaradas: la suma de residuos diarios no es un residuo de h sesiones con α̂ y
β̂ reestimados para ese horizonte. El token de cada decisión se vuelve a calcular al madurar
su etiqueta, con los parámetros vigentes y las entradas guardadas al emitirla, porque su grafo
ya no existe tras el truncamiento. Por eso el término auxiliar ajusta codificadores, fusión y
cabezas auxiliares, pero no la memoria ni la atención de MAC. Es una derivación propia de
aprendizaje multitarea con objetivos auxiliares (Caruana, 1997) y no reproduce ningún
artículo.

Fuente: Caruana, R. (1997). Multitask learning. Machine Learning, 28(1), 41-75.
https://doi.org/10.1023/A:1007379606734
"""

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional

AUXILIARY_HORIZONS = "auxiliary_horizons_v1"


@dataclass(frozen=True)
class AuxiliaryHorizons:
    horizons: tuple[int, ...] = (5, 20)
    weight: float = 0.5
    # Decisiones de mercado que puede tardar en madurar el último sumando de una ventana.
    maturity_slack: int = 2

    def __post_init__(self):
        horizons = tuple(self.horizons) if isinstance(self.horizons, list) else self.horizons
        if (
            type(horizons) is not tuple
            or not 1 <= len(horizons) <= 4
            or any(type(h) is not int or not 2 <= h <= 64 for h in horizons)
            or list(horizons) != sorted(set(horizons))
        ):
            raise ValueError("horizons debe tener de uno a cuatro enteros crecientes entre 2 y 64")
        weight = self.weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 10
        ):
            raise ValueError("weight debe ser un número finito entre 0 y 10")
        if type(self.maturity_slack) is not int or not 0 <= self.maturity_slack <= 16:
            raise ValueError("maturity_slack debe ser un entero entre 0 y 16")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "weight", float(weight))

    @property
    def span(self):
        """Ordinales anteriores al actual que aún pueden completar una ventana."""
        return self.horizons[-1] - 1 + self.maturity_slack

    def identity(self):
        return dict(
            asdict(self),
            horizons=list(self.horizons),
            schema_version=1,
            name=AUXILIARY_HORIZONS,
            target="sum_of_primary_labels_of_consecutive_market_decisions_of_the_flow",
            maturity="last_component_within_maturity_slack_market_decisions",
            loss="weight_times_mean_over_matured_labels_of_absolute_error_over_sqrt_horizon",
            head="linear_on_fused_token_before_mac",
            gradient="token_recomputed_from_stored_inputs_with_current_parameters_at_maturity",
            evaluation="unchanged_primary_prediction",
            control="same_heads_with_weight_zero",
        )


class AuxiliaryHeads(nn.Module):
    """Una salida lineal por horizonte sobre el token fusionado."""

    def __init__(self, hidden_size, config, *, seed, dtype=torch.float32):
        super().__init__()
        if type(config) is not AuxiliaryHorizons:
            raise ValueError("Las cabezas auxiliares necesitan su configuración")
        self.config = config
        # Generador propio derivado de la semilla, como la rama ancha de PT6, para no consumir
        # el global ni cambiar los pesos del resto del predictor. Misma escala que nn.Linear.
        digest = hashlib.sha256(f"{AUXILIARY_HORIZONS}/{seed}".encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        bound = 1 / math.sqrt(hidden_size)
        shape = (len(config.horizons), hidden_size)
        weight = torch.empty(shape, dtype=dtype).uniform_(-bound, bound, generator=generator)
        bias = torch.empty(shape[0], dtype=dtype).uniform_(-bound, bound, generator=generator)
        self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)

    def forward(self, token):
        return functional.linear(token, self.weight, self.bias)


def auxiliary_errors(outputs, targets, heads, horizons):
    """|a⁽ʰ⁾ − y⁽ʰ⁾| / √h por etiqueta. `heads` es el índice de la cabeza de cada fila."""
    if (
        outputs.ndim != 2
        or outputs.shape[1] != len(horizons)
        or targets.shape != (len(outputs),)
        or heads.shape != (len(outputs),)
        or targets.dtype != outputs.dtype
    ):
        raise ValueError("Cada etiqueta auxiliar necesita su fila y su horizonte")
    selected = outputs.gather(1, heads.unsqueeze(-1)).squeeze(-1)
    scale = torch.tensor(horizons, dtype=outputs.dtype, device=outputs.device).sqrt()
    return (selected - targets).abs() / scale[heads]
