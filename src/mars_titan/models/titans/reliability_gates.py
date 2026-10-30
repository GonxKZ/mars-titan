"""Puertas de Titans moduladas por la fiabilidad madura (#791, `mature_reliability_signal_v1`).

La señal r_{c,k} de cada fila es la del mercado de su flujo para el consumidor c y la
velocidad k. La mantiene `mars_titan.memory.mature_reliability` fuera del predictor y solo
cambia con resultados ya conocidos. Este módulo la convierte en un desplazamiento acotado de
los logits de las puertas de escritura θ y de olvido α de la memoria neuronal:

    u_{c,k} = sign(r_{c,k}) · max(|r_{c,k}| − μ, 0)
    δ_c = b · tanh(Σ_k w_{c,k} · u_{c,k})
    θ_t = θ_max · σ(z_θ + δ_escritura)        α_t = σ(z_α + δ_olvido)

con z_θ y z_α los logits de Titans para la entrada x_t. El margen μ descarta las
desviaciones pequeñas de una señal ruidosa y la cota b limita cada desplazamiento a ±b en
unidades de logit. Los dos se fijan antes de ejecutar. Los pesos w empiezan en cero, así que
δ = 0 y el predictor parte exactamente de su referencia. Con la caja de PT1 el desplazamiento
se suma al logit antes de la caja y α conserva su suelo. La memoria compartida de PT4 tiene
sus propias puertas, que la señal no modula.

Con w = 0 la derivada de δ_c respecto a w_{c,k} es b · u_{c,k}. Es nula mientras la señal no
supera el margen, por ejemplo antes de que madure la primera etiqueta.

Es una adaptación propia. Las puertas dependientes de la entrada son las de Titans (Behrouz,
Zhong y Mirrokni, 2024, arXiv 2501.00663). La idea de que un resultado conocido con retraso
module la plasticidad como tercer factor viene de las reglas de tres factores (Gerstner y
otros, 2018, https://doi.org/10.3389/fncir.2018.00053). La sorpresa asociativa de Titans
sigue siendo otra señal y solo sustituye a la madura en su control declarado.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .state import require_true

MATURE_RELIABILITY = "mature_reliability_signal_v1"
# La señal principal y sus dos controles de fuente. El tercer control, con señales
# independientes por consumidor, se declara con `independent`.
SOURCES = ("mature_directional_hit", "shuffled_in_time", "titans_surprise")
CONSUMERS = ("write", "forget")


def _number(value, low, high, message):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(message)
    return float(value)


@dataclass(frozen=True)
class MatureReliability:
    source: str = "mature_directional_hit"
    # Constantes de tiempo en eventos de maduración del mercado: sesión, semana y mes.
    time_constants: tuple[int, ...] = (1, 5, 21)
    margin: float = 0.05
    bound: float = 1.0
    consumers: tuple[str, ...] = CONSUMERS
    # Control de señales independientes: cada consumidor lee solo las etiquetas de su parte
    # de los flujos, repartidos con la semilla.
    independent: bool = False
    # Semilla del reparto de flujos y del orden barajado del control temporal.
    seed: int = 0

    def __post_init__(self):
        if type(self.source) is not str or self.source not in SOURCES:
            raise ValueError("La fuente de la señal no está declarada")
        constants = self.time_constants
        constants = tuple(constants) if isinstance(constants, list) else constants
        if (
            type(constants) is not tuple
            or not 1 <= len(constants) <= 4
            or any(type(value) is not int or not 1 <= value <= 252 for value in constants)
            or list(constants) != sorted(set(constants))
        ):
            raise ValueError("time_constants debe tener de una a cuatro sesiones crecientes")
        consumers = self.consumers
        consumers = tuple(consumers) if isinstance(consumers, list) else consumers
        if (
            type(consumers) is not tuple
            or not consumers
            or any(type(name) is not str for name in consumers)
            or list(consumers) != [name for name in CONSUMERS if name in consumers]
        ):
            raise ValueError("consumers debe ser una parte ordenada de write y forget")
        if type(self.independent) is not bool or (self.independent and len(consumers) < 2):
            raise ValueError("Las señales independientes necesitan al menos dos consumidores")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de la señal debe ser un entero de 32 bits")
        margin = _number(self.margin, 0.0, 0.5, "El margen debe estar en [0, 0,5]")
        bound = _number(self.bound, 0.0, 4.0, "La cota debe estar en (0, 4]")
        if bound == 0.0:
            raise ValueError("La cota debe estar en (0, 4]")
        object.__setattr__(self, "time_constants", constants)
        object.__setattr__(self, "consumers", consumers)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "bound", bound)

    @property
    def decays(self):
        """Factor de cada velocidad: v ← λ v + (1 − λ) s con λ = exp(−1/τ)."""
        return tuple(math.exp(-1.0 / value) for value in self.time_constants)

    @property
    def channels(self):
        """Acumuladores por mercado: uno compartido o uno por consumidor."""
        return len(self.consumers) if self.independent else 1

    @property
    def role(self):
        if self.independent:
            return "independent_signals_per_consumer_control"
        return dict(
            mature_directional_hit="shared_signal_variant",
            shuffled_in_time="shuffled_in_time_control",
            titans_surprise="titans_surprise_control",
        )[self.source]

    def identity(self):
        return dict(
            schema_version=1,
            name=MATURE_RELIABILITY,
            source=self.source,
            role=self.role,
            time_constants=list(self.time_constants),
            margin=self.margin,
            bound=self.bound,
            consumers=list(self.consumers),
            independent=self.independent,
            seed=self.seed,
            score=dict(
                mature_directional_hit="sign_of_issued_times_sign_of_matured_outcome",
                shuffled_in_time="score_of_a_uniformly_drawn_earlier_or_current_maturity_event",
                titans_surprise="relative_gap_of_associative_loss_to_its_slowest_average",
            )[self.source],
            aggregation="market_mean_per_event_with_exact_summation",
            speeds="exponential_average_per_event_with_decay_exp_minus_one_over_tau",
            prior="zero_until_the_first_event",
            reset="each_pass_with_the_fast_state",
            shrinkage="soft_threshold_by_margin",
            shift="bound_times_tanh_of_zero_initialized_mixing_of_speeds",
            gates="added_to_write_theta_and_forget_alpha_logits_before_the_gate_box",
            shared_memory="not_modulated",
        )


class ReliabilityGates(nn.Module):
    """Mezcla aprendida de las velocidades de la señal para cada puerta, nula al empezar."""

    def __init__(self, config, *, dtype=torch.float32):
        super().__init__()
        if type(config) is not MatureReliability:
            raise ValueError("Las puertas necesitan la configuración de la señal")
        self.config = config
        # Ceros sin el generador global: el resto del predictor conserva sus pesos.
        shape = (len(config.consumers), len(config.time_constants))
        self.mixing = nn.Parameter(torch.zeros(shape, dtype=dtype))

    def validate(self, signal, rows):
        """Forma [filas, consumidores, velocidades], precisión y valores en [−1, 1]."""
        if (
            not isinstance(signal, torch.Tensor)
            or signal.shape != (rows, *self.mixing.shape)
            or signal.dtype != self.mixing.dtype
            or signal.device != self.mixing.device
            or signal.requires_grad
        ):
            raise ValueError("La señal de fiabilidad no corresponde a las filas ni a las puertas")
        require_true(
            torch.isfinite(signal).all() & (signal.abs() <= 1).all(),
            "La señal de fiabilidad debe ser finita y estar en [−1, 1]",
        )

    def forward(self, signal):
        """Desplazamiento de logit por consumidor, cada uno con forma [filas, 1]."""
        margin = self.config.margin
        shrunk = signal.sign() * (signal.abs() - margin).clamp(min=0.0)
        shifts = self.config.bound * torch.tanh((shrunk * self.mixing).sum(-1))
        return {
            name: shifts[:, index : index + 1] for index, name in enumerate(self.config.consumers)
        }
