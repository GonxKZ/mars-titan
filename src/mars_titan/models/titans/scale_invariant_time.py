"""Semividas de olvido en escala geométrica `scale_invariant_time_v1`, U2 de #786.

SITH y DeepSITH (Jacques, Tiganj, Howard y Sederberg, NeurIPS 2021) representan la historia
con integradores cuyas constantes de tiempo están espaciadas en escala logarítmica, de modo
que la representación no cambia al reescalar el tiempo. En Titans el olvido α ya es un vector
por fila de salida de cada matriz de la memoria, y `GateBias` fija para todas las filas la
misma semivida inicial h. U2 reparte esas semividas en escala geométrica alrededor de h:

    h_i = h · 2^{o(2i/(n−1) − 1)},    i = 0, …, n − 1,

de h/2^o a h·2^o, con media geométrica h. Con o = 6 y h = 256 sesiones van de 4 a 16.384. El
bias de la fila i es el logit de α_i = 1 − 2^{−1/h_i} que calcula `GateBias.logits`, también
con la caja de PT1, que rechaza las semividas que exigen un olvido inferior a su suelo. Como
en `GateBias`, los bias son el valor inicial de parámetros aprendidos, la dependencia de la
entrada no cambia y no se consume ningún generador. η, θ y la memoria compartida de PT4 no
cambian. Con o = 0 todas las filas reciben la semivida del control y el cálculo coincide bit a
bit con él.
"""

import math
from dataclasses import dataclass, replace

SCALE_INVARIANT_TIME = "scale_invariant_time_v1"


@dataclass(frozen=True)
class ScaleInvariantTime:
    """Octavas o a cada lado de la semivida declarada en `GateBias`."""

    octaves: float = 6.0

    def __post_init__(self) -> None:
        value = self.octaves
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 8:
            raise ValueError("octaves debe ser un número finito en [0, 8]")
        object.__setattr__(self, "octaves", float(value))

    def half_lives(self, center: float, rows: int) -> tuple[float, ...]:
        """Semividas crecientes por fila. Con o = 0, 2^0 = 1 y todas valen exactamente h."""
        if type(rows) is not int or rows < 2:
            raise ValueError("Las escalas de tiempo necesitan al menos dos filas")
        return tuple(center * 2.0 ** (self.octaves * (2 * i / (rows - 1) - 1)) for i in range(rows))

    def alpha_logits(self, gate_bias, rows, theta_max, stability=None) -> tuple[float, ...]:
        """Bias de α por fila con la misma función que el control."""
        return tuple(
            replace(gate_bias, alpha_half_life=half_life).logits(theta_max, stability)[0]
            for half_life in self.half_lives(gate_bias.alpha_half_life, rows)
        )

    def identity(self) -> dict:
        return dict(
            name=SCALE_INVARIANT_TIME,
            octaves=self.octaves,
            center="gate_bias_alpha_half_life",
            spacing="log2_uniform_ascending_over_alpha_rows",
            learned="initial_bias_of_the_input_dependent_alpha",
            unchanged=["eta", "theta", "shared_memory"],
            source="deepsith_neurips_2021",
        )
