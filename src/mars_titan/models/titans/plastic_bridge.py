"""Configuración del puente hebbiano de tres factores (#784, P1 `three_factor_bridge_v1`).

P1 añade un peso rápido a cada arista entre dos áreas distintas del puente de H1 o H2. El peso
del mercado de la fila se suma al logit de atención de la consulta i sobre la clave j en todos
los pasos del puente, y la red devuelve la media de esas probabilidades como traza de
elegibilidad. El estado de los pesos, las trazas pendientes y la regla de actualización con
el error maduro están en `mars_titan.memory.three_factor_bridge`, fuera del predictor. P1 no
añade parámetros y con pesos nulos el predictor es la referencia.

El tercer factor tiene dos fuentes declaradas. `relative_error_gap` es la de la issue: la
diferencia relativa entre el error absoluto esperado del mercado y el de la predicción.
`mature_directional_hit` toma el acierto direccional de la señal común de #791 frente a su
velocidad más lenta, así que P1 se convierte en otro consumidor de la fiabilidad madura, como
decidió la segunda ola. Las dos tienen el mismo control, que sustituye δ por su media.
"""

import math
from dataclasses import dataclass

THREE_FACTOR_BRIDGE = "three_factor_bridge_v1"
MEAN_THIRD_FACTOR = "mean_third_factor_v1"
KINDS = (THREE_FACTOR_BRIDGE, MEAN_THIRD_FACTOR)
RELATIVE_ERROR_GAP = "relative_error_gap"
MATURE_DIRECTIONAL_HIT = "mature_directional_hit"
SOURCES = (RELATIVE_ERROR_GAP, MATURE_DIRECTIONAL_HIT)


def _number(value, low, high, message):
    # Los límites son finitos, así que la comparación también rechaza NaN e infinito.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low < value <= high:
        raise ValueError(message)
    return float(value)


@dataclass(frozen=True)
class ThreeFactorBridge:
    kind: str = THREE_FACTOR_BRIDGE
    source: str = RELATIVE_ERROR_GAP
    # η: un evento con δ = 1 y traza máxima cambia cada peso como mucho en η unidades de logit.
    learning_rate: float = 0.05
    # B: suma máxima de los valores absolutos de las aristas que llegan a un área.
    budget: float = 1.0
    # τ de la expectativa del error en eventos de maduración del mercado, un mes de sesiones.
    # Con la señal común no se usa, porque la expectativa es su velocidad más lenta.
    time_constant: int | None = 21

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError("El tipo del puente de tres factores no está declarado")
        if type(self.source) is not str or self.source not in SOURCES:
            raise ValueError("La fuente del tercer factor no está declarada")
        rate = _number(self.learning_rate, 0.0, 1.0, "La tasa del puente debe estar en (0, 1]")
        budget = _number(self.budget, 0.0, 4.0, "La cota del puente debe estar en (0, 4]")
        if self.source == MATURE_DIRECTIONAL_HIT:
            if self.time_constant is not None:
                raise ValueError("Con la señal común la constante de la expectativa es None")
        elif type(self.time_constant) is not int or not 1 <= self.time_constant <= 252:
            raise ValueError("La constante de la expectativa debe estar entre 1 y 252 eventos")
        object.__setattr__(self, "learning_rate", rate)
        object.__setattr__(self, "budget", budget)

    @property
    def decay(self):
        return math.exp(-1.0 / self.time_constant)

    @property
    def role(self):
        return "variant" if self.kind == THREE_FACTOR_BRIDGE else "mean_third_factor_control"

    def identity(self):
        common = self.source == MATURE_DIRECTIONAL_HIT
        return dict(
            schema_version=1,
            name=THREE_FACTOR_BRIDGE,
            kind=self.kind,
            role=self.role,
            source=self.source,
            learning_rate=self.learning_rate,
            budget=self.budget,
            time_constant=self.time_constant,
            edges="distinct_area_pairs_joined_by_the_bridge_topology",
            read="fast_weight_added_to_area_to_area_attention_logits_in_every_bridge_step",
            trace="mean_attention_probability_of_present_query_areas_held_with_the_pending",
            third_factor=(
                "half_of_directional_hit_minus_expected_hit_of_the_common_signal"
                if common
                else "clipped_gap_of_absolute_error_to_its_expectation_over_the_expectation"
            ),
            expectation=(
                "slowest_speed_of_the_mature_reliability_signal_before_the_event"
                if common
                else "market_exponential_average_of_absolute_error_per_maturation_event"
            ),
            credit=(
                "each_matured_label"
                if self.kind == THREE_FACTOR_BRIDGE
                else "running_mean_of_the_market_third_factor"
            ),
            update="eta_times_mean_over_the_market_labels_of_third_factor_times_trace",
            homeostasis="proportional_scaling_of_the_edges_reaching_each_area_to_the_budget",
            summation="float64_in_order_of_flow_and_decision",
            reset="each_pass_with_the_fast_state",
        )


def parse_three_factor_bridge(raw):
    """None, la configuración o un diccionario con todos sus campos."""
    if raw is None or type(raw) is ThreeFactorBridge:
        return raw
    if not isinstance(raw, dict) or set(raw) != set(ThreeFactorBridge.__dataclass_fields__):
        raise ValueError("three_factor_bridge debe declarar todos sus campos")
    return ThreeFactorBridge(**raw)
