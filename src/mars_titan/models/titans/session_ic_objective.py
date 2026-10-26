"""Pérdida de IC por sesión con cabeza de escala separada (F4 de #749, `session_ic_objective_v1`).

La cabeza escalar del predictor deja de emitir la predicción y pasa a emitir una puntuación s
por fila. Sea T el conjunto de sesiones (mercado e instante de decisión) con al menos
`minimum_rows` etiquetas maduras en el tramo, g = Φ⁻¹((r − 1/2)/n) las puntuaciones normales de
los rangos de sus objetivos y ρ_t la correlación de Pearson dentro de la sesión t:

    L = −(1/|T|) Σ_{t∈T} ρ_t(s, g) + w · (1/N) Σ_i |y_i − ŷ_i|

La predicción emitida es ŷ_i = softplus(κ) · z_i, con z_i = (s_i − s̄_t) / sqrt(var_t(s) + ε_t)
calculado sobre todas las filas de su sesión en la emisión y con s sin gradiente. Así el
primer término solo ajusta la puntuación y el tronco, y el segundo solo la escala κ. Como κ
es común y softplus(κ) > 0, ŷ conserva el orden de s dentro de cada sesión y el IC emitido es
el de la puntuación.

La correlación de Pearson no depende del nivel ni de la escala de s dentro de cada sesión.
La pérdida usa una aproximación regularizada y no conserva exactamente esa invariancia. El suelo
ε_t = 10⁻¹⁰ · media(s²) + eps(dtype)² trata como falta de información una
dispersión menor que 10⁻⁵ veces la magnitud de las puntuaciones, unas cien veces el redondeo
de FP32, en lugar de convertir ese redondeo en una predicción de escala completa. El término
absoluto limita la normalización de puntuaciones nulas. La misma normalización entra en la
correlación regularizada del ajuste. No garantiza la finitud de todos los gradientes del
predictor. Sin la escala, la
predicción no tendría unidades de retorno y el MAE sin calibrar no tendría sentido. κ empieza
en softplus(κ₀) = `initial_scale`, cerca de predecir cero.

Desviaciones declaradas: es una derivación propia del contraste principal de la criba (IC
medio por sesión), no la reproducción de un artículo. La correlación usa las puntuaciones
normales de los rangos del objetivo y la puntuación continua del modelo, que sí es derivable.
No es el IC de Spearman, que necesitaría también los rangos de s. Las sesiones con menos de
`minimum_rows` etiquetas maduras en el tramo no entran en el primer término.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional

from .session_ranks import gaussian_ranks
from .state import require_finite

SESSION_IC_OBJECTIVE = "session_ic_objective_v1"
# Fracción de la media de s² que se suma a la varianza al estandarizar.
RELATIVE_FLOOR = 1e-10


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} debe ser un número finito")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} debe ser un número finito") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} debe ser un número finito")
    return result


@dataclass(frozen=True)
class SessionICObjective:
    minimum_rows: int = 8
    scale_weight: float = 1.0
    initial_scale: float = 1e-3

    def __post_init__(self):
        if type(self.minimum_rows) is not int or not 3 <= self.minimum_rows <= 4096:
            raise ValueError("minimum_rows debe ser un entero entre 3 y 4096")
        for name in ("scale_weight", "initial_scale"):
            value = _number(getattr(self, name), name)
            if not 0 < value <= 1e3:
                raise ValueError(f"{name} debe ser positivo y acotado")
            object.__setattr__(self, name, value)

    @property
    def name(self):
        return SESSION_IC_OBJECTIVE

    def identity(self):
        return dict(
            **asdict(self),
            schema_version=1,
            name=SESSION_IC_OBJECTIVE,
            score="scalar_head_output_trained_by_session_correlation",
            correlation="regularized_correlation_of_score_and_target_normal_scores_per_session",
            constant_target_session="zero_contribution",
            target_scores="inverse_normal_of_average_rank_minus_half_among_matured_rows",
            session_weighting="equal_mean_over_sessions_with_minimum_rows",
            point="softplus_scale_times_session_standardized_detached_score",
            scale_objective="mae_of_point_trains_only_the_scale",
            standardization_floor=dict(relative=RELATIVE_FLOOR, absolute="dtype_eps_squared"),
        )

    def build(self, hidden_size, *, seed, dtype):
        return SessionScale(self, dtype=dtype)


def standardized_scores(scores, markets):
    """Centrar y escalar las puntuaciones de cada mercado del lote, que es una sesión."""
    if scores.ndim != 1 or len(markets) != len(scores):
        raise ValueError("Las puntuaciones necesitan un mercado por fila")
    result = torch.zeros_like(scores)
    for market in sorted(set(markets)):
        rows = torch.tensor(
            [i for i, value in enumerate(markets) if value == market],
            dtype=torch.int64,
            device=scores.device,
        )
        result = result.index_copy(0, rows, standardized(scores.index_select(0, rows)))
    return result


def session_correlations(scores, target, sessions, minimum_rows):
    """Correlación regularizada por sesión entre puntuaciones y rangos normales del objetivo.

    Usa el suelo de `standardized`. Una sesión con objetivo constante aporta cero. Devuelve
    una entrada por sesión con suficientes filas, en el orden de primera aparición.
    """
    if (
        scores.ndim != 1
        or target.shape != scores.shape
        or len(sessions) != len(scores)
        or scores.dtype != target.dtype
    ):
        raise ValueError("Puntuaciones, objetivos y sesiones deben ir fila a fila")
    groups = {}
    for row, key in enumerate(sessions):
        groups.setdefault(key, []).append(row)
    values = []
    for rows in groups.values():
        if len(rows) < minimum_rows:
            continue
        index = torch.tensor(rows, dtype=torch.int64, device=scores.device)
        score = scores.index_select(0, index)
        labels = target.index_select(0, index).unsqueeze(-1)
        normal = gaussian_ranks(labels, torch.ones_like(labels, dtype=torch.bool)).squeeze(-1)
        values.append((standardized(score) * standardized(normal)).mean())
    if not values:
        return scores.new_zeros(0)
    return torch.stack(values)


def standardized(values):
    """Normalización con suelo relativo y eps², sin elevar valores grandes al cuadrado.

    Se divide por una potencia de dos próxima a max(|v|, eps) antes de calcular las medias.
    Este reescalado es exacto en binario y evita perder dispersión al centrar. El factor se
    cancela en el cociente, incluido el suelo absoluto, y no necesita gradiente propio.
    """
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 1
        or values.numel() == 0
        or values.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("La normalización necesita un vector no vacío en FP32 o FP64")
    require_finite(values, "Las puntuaciones de la sesión deben ser finitas")
    epsilon = torch.finfo(values.dtype).eps
    maximum = values.detach().abs().amax().clamp_min(epsilon)
    _, exponent = torch.frexp(maximum)
    magnitude = torch.ldexp(torch.ones_like(maximum), exponent - 1)
    scaled = values / magnitude
    centered = scaled - scaled.mean()
    floor = RELATIVE_FLOOR * scaled.square().mean() + (epsilon / magnitude).square()
    return centered / torch.sqrt(centered.square().mean() + floor)


def session_ic_loss(outputs, target, sessions, config):
    """Pérdida de la variante. `outputs` es [filas, 2] con la puntuación y la predicción."""
    if outputs.ndim != 2 or outputs.shape[1] != 2 or target.shape != (len(outputs),):
        raise ValueError("La pérdida de IC necesita puntuación y predicción por fila")
    correlations = session_correlations(outputs[:, 0], target, sessions, config.minimum_rows)
    # Sin sesiones válidas el primer término vale cero y no aporta gradiente al tronco.
    ranking = -correlations.mean() if len(correlations) else outputs[:, 0].sum() * 0
    return ranking + config.scale_weight * (target - outputs[:, 1]).abs().mean()


class SessionScale(nn.Module):
    """Escala κ común que lleva la puntuación estandarizada a unidades de retorno."""

    # La emisión estandariza dentro de la sesión y la pérdida agrupa sesiones completas.
    session_level = True
    row_decomposable = False

    def __init__(self, config, *, dtype=torch.float32):
        super().__init__()
        if type(config) is not SessionICObjective:
            raise ValueError("La escala necesita su configuración de IC por sesión")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("La escala necesita precisión FP32 o FP64")
        if config.initial_scale < torch.finfo(dtype).tiny:
            raise ValueError("initial_scale está por debajo del mínimo normal de la precisión")
        self.config = config
        # softplus⁻¹(x) = x + log(1 − exp(−x)), estable también cuando x es grande.
        initial = config.initial_scale + math.log(-math.expm1(-config.initial_scale))
        self.scale = nn.Parameter(torch.tensor(initial, dtype=dtype))

    def forward(self, score, encoded, markets):
        point = functional.softplus(self.scale) * standardized_scores(score.detach(), markets)
        return torch.stack((score, point), dim=-1), point

    def loss(self, outputs, target, sessions):
        return session_ic_loss(outputs, target, sessions, self.config)
