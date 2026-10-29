"""Pérdida asociativa de Huber por canal `huber_memory_v1`, R1 de #786.

Yaad, una de las memorias de MIRAS (Behrouz, Razaviyayn, Zhong y Mirrokni, ICLR 2026,
secciones 5.1 y 5.3), sustituye la pérdida L2 de Titans por la de Huber con un umbral δ_t por
canal que depende de la entrada. Dentro del umbral escribe con el gradiente de ℓ2 y fuera con
el de ℓ1 escalado por δ_t. Con r = M(k) − v el residuo de un token y un flujo, aquí la pérdida es

    ℓ(r) = Σ_c r_c²                 si |r_c| ≤ δ_c
           Σ_c 2δ_c|r_c| − δ_c²     si |r_c| > δ_c,

es decir, 2·H_δ(r_c) en cada canal. El factor 2 conserva la escala de la pérdida del núcleo,
‖r‖², de modo que dentro del umbral el gradiente es exactamente el de L2, 2r, y fuera vale
2δ_c·sign(r_c), el residuo recortado. El umbral es

    δ_c = κ_c ρ,    κ_c = softplus(w·x + b_c),    ρ = ‖v‖₂ / √d sin gradiente,

con x el token que también da las puertas α, η y θ. El artículo no fija la parametrización
de δ_t, así que esta es propia: una dependencia de rango uno con la entrada, como η y θ, un
desplazamiento por canal y la escala del valor, para que κ se lea como un error relativo. w
empieza en cero y b en softplus⁻¹(κ₀), sin consumir ningún generador, así que al empezar el
umbral de todos los canales es κ₀ρ. La escala no lleva gradiente porque así el gradiente
interno es el de Huber con el umbral fijo. La memoria compartida de PT4 conserva L2.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

HUBER_MEMORY = "huber_memory_v1"


@dataclass(frozen=True)
class HuberMemory:
    """Umbral inicial κ₀ en unidades de la magnitud cuadrática media del valor."""

    initial_ratio: float = 1.0

    def __post_init__(self) -> None:
        ratio = self.initial_ratio
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0.01 <= ratio <= 100:
            raise ValueError("initial_ratio debe ser un número finito en [0,01, 100]")
        object.__setattr__(self, "initial_ratio", float(ratio))

    @property
    def bias(self) -> float:
        """softplus⁻¹(κ₀) = log(eˣ − 1), calculado con expm1 para no perder cifras."""
        return math.log(math.expm1(self.initial_ratio))

    def identity(self) -> dict:
        return dict(
            name=HUBER_MEMORY,
            initial_ratio=self.initial_ratio,
            loss="twice_channelwise_huber_equal_to_l2_inside_the_threshold",
            threshold="softplus_rank_one_input_plus_channel_bias_times_value_rms",
            scale="value_rms_without_gradient",
            initialization="zero_input_weights_and_constant_bias_without_rng",
            shared_memory="l2_unchanged",
            source="arxiv_2504_13173_yaad_sections_5_1_and_5_3",
        )


def huber_threshold(weight: Tensor, bias: Tensor, token: Tensor, values: Tensor) -> Tensor:
    """δ = softplus(w·x + b) · ‖v‖₂/√d, con forma [filas, canales]."""
    scale = values.detach().square().mean(dim=-1, keepdim=True).sqrt()
    return torch.nn.functional.softplus(torch.nn.functional.linear(token, weight) + bias) * scale


def huber_elements(residual: Tensor, threshold: Tensor) -> Tensor:
    """2·H_δ por canal: r² dentro del umbral y 2δ|r| − δ² fuera."""
    absolute = residual.abs()
    return torch.where(
        absolute <= threshold, residual.square(), 2 * threshold * absolute - threshold.square()
    )
