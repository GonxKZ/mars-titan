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
from .state import require_finite, require_true

SESSION_IC_OBJECTIVE = "session_ic_objective_v1"
SESSION_IC_TERM = "session_ic_term_v1"
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


@dataclass(frozen=True)
class SessionICTerm:
    """Término de IC por sesión que una receta suma a su pérdida (`session_ic_term_v1`).

    La pérdida declarada, pinball con la cabeza de cuantiles o una pérdida puntual, mide el
    error fila a fila en unidades de retorno. El contraste de la campaña es el IC medio por
    sesión, que solo depende del orden de las predicciones dentro de cada sesión. Con este
    término la pérdida del tramo es

        L = L_declarada − w · (1/|T|) Σ_{t∈T} ρ_t(p, g)

    p es el punto emitido, la mediana con la cabeza de cuantiles, y g, T y ρ_t son los de
    `session_ic_objective_v1`: las puntuaciones normales de los rangos del objetivo, las
    sesiones con al menos `minimum_rows` etiquetas maduras en el tramo y la correlación
    regularizada dentro de cada una. La pérdida declarada sigue fijando el nivel y la escala
    de las predicciones, que conservan sus unidades y sus cuantiles. El término no depende
    de ellos y solo empuja el orden dentro de cada sesión. La cabeza, la predicción emitida
    y sus consumidores no cambian, a diferencia del objetivo de salida, que sustituye la
    cabeza escalar y emite una puntuación estandarizada.

    Es una derivación propia. Sin sesiones con suficientes filas el término vale cero.

    Con `liquidity_column` y `liquidity_power` la correlación de cada sesión pondera sus
    filas por liquidez: con q_i la posición de la fila entre las de su sesión al ordenarlas por
    la columna declarada del bloque del gráfico (su rango medio menos 1/2, entre n), el peso
    es (0,05 + q_i)^p, normalizado a suma uno, y la media, la dispersión y el producto de la
    correlación son los ponderados. Los rangos del objetivo no cambian. Así el término cuenta
    más el orden entre los activos que se pueden operar. Con p = 0 sería la correlación sin
    ponderar. La columna es una entrada disponible al emitir y no lleva gradiente.

    Con `offset_column` y `offset_scale` el término mide el orden de lo que se puede operar.
    La etiqueta resta un desplazamiento a_i que se conoce al predecir (el alfa del modelo de
    mercado, una columna del bloque del gráfico por su escala). El rendimiento operable es
    y_i + a_i y la predicción del modelo para él es p_i + a_i, así que el término pasa a ser

        −w · (1/|T|) Σ_{t∈T} ρ_t(p + a, g(y + a))

    con g las puntuaciones normales de los rangos de y + a. A diferencia del término sin
    desplazamiento, depende de la escala de p: sumada a a, una predicción con otra escala
    ordena de otra forma. Por eso también ajusta la escala de la mediana. Donde la columna no
    es finita (un activo sin modelo de mercado) el desplazamiento vale cero. No lleva gradiente.
    """

    weight: float = 1.0
    minimum_rows: int = 8
    # Columna del bloque del gráfico con la liquidez de cada fila, contada desde el final.
    liquidity_column: int | None = None
    liquidity_power: float | None = None
    # Columna del bloque del gráfico con el desplazamiento conocido de la etiqueta, contada
    # desde el final, y el factor que la lleva a las unidades de la etiqueta.
    offset_column: int | None = None
    offset_scale: float | None = None

    def __post_init__(self):
        if type(self.minimum_rows) is not int or not 3 <= self.minimum_rows <= 4096:
            raise ValueError("minimum_rows debe ser un entero entre 3 y 4096")
        weight = _number(self.weight, "weight")
        if not 0 < weight <= 1e3:
            raise ValueError("weight debe ser positivo y acotado")
        object.__setattr__(self, "weight", weight)
        if (self.liquidity_column is None) != (self.liquidity_power is None):
            raise ValueError("La liquidez del término declara su columna y su exponente")
        if self.liquidity_column is not None:
            if type(self.liquidity_column) is not int or not -4096 <= self.liquidity_column < 0:
                raise ValueError("liquidity_column cuenta desde el final del bloque: es negativa")
            power = _number(self.liquidity_power, "liquidity_power")
            if not 0 < power <= 8:
                raise ValueError("liquidity_power debe estar entre 0 y 8")
            object.__setattr__(self, "liquidity_power", power)
        if (self.offset_column is None) != (self.offset_scale is None):
            raise ValueError("El desplazamiento del término declara su columna y su escala")
        if self.offset_column is not None:
            if type(self.offset_column) is not int or not -4096 <= self.offset_column < 0:
                raise ValueError("offset_column cuenta desde el final del bloque: es negativa")
            scale = _number(self.offset_scale, "offset_scale")
            if not 0 < scale <= 1e3:
                raise ValueError("offset_scale debe ser positiva y acotada")
            object.__setattr__(self, "offset_scale", scale)

    @property
    def weighted(self):
        """Si la correlación pondera las filas por liquidez."""
        return self.liquidity_column is not None

    @property
    def shifted(self):
        """Si el término suma al punto y al objetivo el desplazamiento conocido de la etiqueta."""
        return self.offset_column is not None

    def identity(self):
        result = dict(
            weight=self.weight,
            minimum_rows=self.minimum_rows,
            schema_version=1,
            name=SESSION_IC_TERM,
            point="median_of_the_quantile_head_or_scalar_output",
            correlation="regularized_correlation_of_point_and_target_normal_scores_per_session",
            constant_target_session="zero_contribution",
            target_scores="inverse_normal_of_average_rank_minus_half_among_matured_rows",
            session_weighting="equal_mean_over_sessions_with_minimum_rows",
            sign="weight_times_mean_correlation_subtracted_from_the_declared_loss",
            standardization_floor=dict(relative=RELATIVE_FLOOR, absolute="dtype_eps_squared"),
        )
        # Sin la liquidez se conserva literalmente la identidad anterior.
        if self.weighted:
            result.update(
                liquidity_column=self.liquidity_column,
                liquidity_power=self.liquidity_power,
                row_weighting=(
                    "offset_0.05_plus_session_position_of_the_chart_column_to_the_power_"
                    "normalized_to_sum_one"
                ),
                liquidity_source="chart_block_column_counted_from_the_end_as_received",
            )
        # Y sin el desplazamiento, también.
        if self.shifted:
            result.update(
                offset_column=self.offset_column,
                offset_scale=self.offset_scale,
                offset="known_label_offset_added_to_point_and_target_before_the_correlation",
                offset_source=(
                    "chart_block_column_counted_from_the_end_as_received_times_the_scale_"
                    "with_zero_where_it_is_not_finite"
                ),
            )
        return result

    def term(self, point, target, sessions, liquidity=None, offset=None):
        """−w por la correlación media de las sesiones del tramo, o cero si no hay ninguna."""
        if self.weighted != (liquidity is not None):
            raise ValueError("El término recibe la liquidez de cada fila solo si la declara")
        if self.shifted != (offset is not None) or (
            offset is not None and offset.shape != point.shape
        ):
            raise ValueError("El término recibe el desplazamiento de cada fila solo si lo declara")
        if offset is not None:
            # Lo operable: la etiqueta y su predicción con el desplazamiento conocido sumado.
            offset = offset.detach().to(point.dtype)
            point, target = point + offset, target + offset
        correlations = session_correlations(
            point,
            target,
            sessions,
            self.minimum_rows,
            liquidity=liquidity,
            power=self.liquidity_power,
        )
        if not len(correlations):
            return point.sum() * 0
        return -self.weight * correlations.mean()


def parse_session_ic_term(raw):
    """Aceptar la configuración, sus valores o None."""
    if raw is None:
        return None
    if type(raw) is SessionICTerm:
        raw.__post_init__()
        return raw
    if type(raw) is dict and {"weight", "minimum_rows"} <= set(raw):
        extra = set(raw) - {"weight", "minimum_rows"}
        liquidity, offset = (
            {"liquidity_column", "liquidity_power"},
            {"offset_column", "offset_scale"},
        )
        if extra in (set(), liquidity, offset, liquidity | offset):
            return SessionICTerm(**raw)
    raise ValueError(
        "session_ic_term debe declarar weight y minimum_rows (y liquidity_column con "
        "liquidity_power, u offset_column con offset_scale), o ser None"
    )


def known_offset(values, scale):
    """Desplazamiento conocido de cada fila: su columna por la escala, y cero si no es finita."""
    values = values.detach()
    return torch.where(torch.isfinite(values), values * scale, torch.zeros_like(values))


def liquidity_weights(values, power):
    """Pesos de las filas de una sesión: (0,05 + posición por liquidez)^p, con suma uno."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("La liquidez necesita un valor por fila de la sesión")
    require_finite(values, "La liquidez de la sesión debe ser finita")
    ordered = values.sort().values
    lower = torch.searchsorted(ordered, values, right=False)
    upper = torch.searchsorted(ordered, values, right=True)
    # Rango medio de las filas empatadas, como en los rangos del objetivo.
    position = ((lower + upper + 1).to(values.dtype) / 2 - 0.5) / values.numel()
    weights = (0.05 + position) ** power
    return weights / weights.sum()


def weighted_standardized(values, weights):
    """Como `standardized`, con la media y la dispersión ponderadas. Los pesos suman uno."""
    if (
        not isinstance(values, torch.Tensor)
        or not isinstance(weights, torch.Tensor)
        or values.ndim != 1
        or values.numel() == 0
        or values.dtype not in (torch.float32, torch.float64)
        or weights.shape != values.shape
        or weights.dtype != values.dtype
        or weights.device != values.device
    ):
        raise ValueError("La normalización ponderada necesita un peso por valor en FP32 o FP64")
    require_finite(values, "Las puntuaciones de la sesión deben ser finitas")
    require_finite(weights, "Los pesos de la sesión deben ser finitos")
    require_true(
        (weights >= 0).all() & (weights.sum() > 0),
        "Los pesos de la sesión deben ser no negativos y sumar positivo",
    )
    epsilon = torch.finfo(values.dtype).eps
    maximum = values.detach().abs().amax().clamp_min(epsilon)
    _, exponent = torch.frexp(maximum)
    magnitude = torch.ldexp(torch.ones_like(maximum), exponent - 1)
    scaled = values / magnitude
    centered = scaled - (weights * scaled).sum()
    floor = RELATIVE_FLOOR * (weights * scaled.square()).sum() + (epsilon / magnitude).square()
    return centered / torch.sqrt((weights * centered.square()).sum() + floor)


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


def session_correlations(scores, target, sessions, minimum_rows, *, liquidity=None, power=None):
    """Correlación regularizada por sesión entre puntuaciones y rangos normales del objetivo.

    Usa el suelo de `standardized`. Una sesión con objetivo constante aporta cero. Devuelve
    una entrada por sesión con suficientes filas, en el orden de primera aparición. Con
    `liquidity` y `power`, cada sesión pondera sus filas con `liquidity_weights`.
    """
    if (
        scores.ndim != 1
        or target.shape != scores.shape
        or len(sessions) != len(scores)
        or scores.dtype != target.dtype
    ):
        raise ValueError("Puntuaciones, objetivos y sesiones deben ir fila a fila")
    if (liquidity is None) != (power is None) or (
        liquidity is not None and liquidity.shape != scores.shape
    ):
        raise ValueError("La ponderación necesita la liquidez de cada fila y su exponente")
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
        if liquidity is None:
            values.append((standardized(score) * standardized(normal)).mean())
            continue
        weights = liquidity_weights(liquidity.detach().index_select(0, index), power)
        product = weighted_standardized(score, weights) * weighted_standardized(normal, weights)
        values.append((weights * product).sum())
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
    # El punto emitido es la puntuación estandarizada por la escala, no la cabeza escalar.
    scalar_point = False

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
