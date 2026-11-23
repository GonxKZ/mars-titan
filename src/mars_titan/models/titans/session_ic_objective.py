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

    Con `plain_weight`, que solo se declara junto al desplazamiento, el término suma además la
    correlación de siempre, sin desplazamiento, con ese peso:

        −w · (1/|T|) Σ ρ_t(p + a, g(y + a)) − w_plain · (1/|T|) Σ ρ_t(p, g(y))

    El primero ordena lo que se puede operar y el segundo conserva el orden de la etiqueta, que
    es el contraste de la campaña. En la red tabular, el término desplazado solo gana IC
    operable y pierde el de la etiqueta; los dos juntos conservan los dos.

    Con `liquidity_column` y `liquidity_weight` (número ≥ 1, por defecto ausente = 1) las filas
    del tercio más líquido de cada sesión con liquidez conocida pesan `liquidity_weight` y el
    resto (incluidas las filas sin liquidez finita conocida) pesan 1, tanto en la correlación
    ponderada del término como en la media ponderada de la pérdida declarada (pinball). El
    umbral se elige por sesión con la entrada `(2 · m) // 3` de los `m` valores finitos
    ordenados (empates incluidos, invariante al orden de las filas). Con valor 1 el cálculo es
    exactamente el actual y no entra en `identity()`. No se combina con `liquidity_power`.

    Con `stability` (número ≥ 0, por defecto ausente = 0) al término se le suma
    `stability · weight · σ(ρ)` cuando hay al menos dos sesiones válidas en el grupo de
    acumulación, donde `σ(ρ)` es la desviación típica muestral regularizada (con suelo `eps²`
    para garantizar gradientes finitos también con correlaciones iguales o predicción constante).
    Con `plain_weight` se aplica la misma regla a cada rama con su propio peso. Con valor 0 el
    cálculo es exactamente el actual y no entra en `identity()`.
    """

    weight: float = 1.0
    minimum_rows: int = 8
    # Columna del bloque del gráfico con la liquidez de cada fila, contada desde el final.
    liquidity_column: int | None = None
    liquidity_power: float | None = None
    # Factor de peso (>= 1) del tercio más líquido de cada sesión en correlación y pinball.
    liquidity_weight: float | None = None
    # Penalización (>= 0) de la desviación típica de las correlaciones entre sesiones del tramo.
    stability: float = 0.0
    # Columna del bloque del gráfico con el desplazamiento conocido de la etiqueta, contada
    # desde el final, y el factor que la lleva a las unidades de la etiqueta.
    offset_column: int | None = None
    offset_scale: float | None = None
    # Peso del mismo término sobre la etiqueta tal cual, que se suma al desplazado.
    plain_weight: float | None = None

    def __post_init__(self):
        if type(self.minimum_rows) is not int or not 3 <= self.minimum_rows <= 4096:
            raise ValueError("minimum_rows debe ser un entero entre 3 y 4096")
        weight = _number(self.weight, "weight")
        if not 0 < weight <= 1e3:
            raise ValueError("weight debe ser positivo y acotado")
        object.__setattr__(self, "weight", weight)
        if self.liquidity_weight is not None:
            liq_weight = _number(self.liquidity_weight, "liquidity_weight")
            if not 1.0 <= liq_weight <= 1e3:
                raise ValueError("liquidity_weight debe ser un número entre 1 y 1000")
            object.__setattr__(self, "liquidity_weight", liq_weight)
        if self.liquidity_power is not None:
            power = _number(self.liquidity_power, "liquidity_power")
            if not 0 < power <= 8:
                raise ValueError("liquidity_power debe estar entre 0 y 8")
            object.__setattr__(self, "liquidity_power", power)
        if self.liquidity_column is not None:
            if type(self.liquidity_column) is not int or not -4096 <= self.liquidity_column < 0:
                raise ValueError("liquidity_column cuenta desde el final del bloque: es negativa")
        if self.liquidity_power is not None and self.liquidity_column is None:
            raise ValueError("La liquidez del término declara su columna y su exponente")
        if self.top_third_weighted and self.liquidity_column is None:
            raise ValueError("liquidity_weight mayor que 1 requiere liquidity_column")
        if (
            self.liquidity_column is not None
            and self.liquidity_power is None
            and self.liquidity_weight is None
        ):
            raise ValueError("La liquidez del término declara su columna y su exponente o su peso")
        if self.liquidity_power is not None and self.top_third_weighted:
            raise ValueError("liquidity_weight no se combina con liquidity_power")
        stability = _number(self.stability, "stability")
        if not 0.0 <= stability <= 1e3:
            raise ValueError("stability debe ser un número entre 0 y 1000")
        object.__setattr__(self, "stability", stability)
        if (self.offset_column is None) != (self.offset_scale is None):
            raise ValueError("El desplazamiento del término declara su columna y su escala")
        if self.offset_column is not None:
            if type(self.offset_column) is not int or not -4096 <= self.offset_column < 0:
                raise ValueError("offset_column cuenta desde el final del bloque: es negativa")
            scale = _number(self.offset_scale, "offset_scale")
            if not 0 < scale <= 1e3:
                raise ValueError("offset_scale debe ser positiva y acotada")
            object.__setattr__(self, "offset_scale", scale)
        if self.plain_weight is not None:
            if self.offset_column is None:
                raise ValueError("plain_weight acompaña al desplazamiento: sin él sobra")
            plain = _number(self.plain_weight, "plain_weight")
            if not 0 < plain <= 1e3:
                raise ValueError("plain_weight debe ser positivo y acotado")
            object.__setattr__(self, "plain_weight", plain)

    @property
    def top_third_weighted(self):
        """Si el tercio más líquido de cada sesión pesa más que el resto."""
        return self.liquidity_weight is not None and self.liquidity_weight > 1.0

    @property
    def weighted(self):
        """Si la correlación pondera las filas por liquidez."""
        return self.liquidity_power is not None or self.top_third_weighted

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
        if self.liquidity_power is not None:
            result.update(
                liquidity_column=self.liquidity_column,
                liquidity_power=self.liquidity_power,
                row_weighting=(
                    "offset_0.05_plus_session_position_of_the_chart_column_to_the_power_"
                    "normalized_to_sum_one"
                ),
                liquidity_source="chart_block_column_counted_from_the_end_as_received",
            )
        elif self.top_third_weighted:
            result.update(
                liquidity_column=self.liquidity_column,
                liquidity_weight=self.liquidity_weight,
                row_weighting=(
                    "top_third_of_known_liquidity_per_session_at_ge_sorted_2n_over_3_"
                    "weighted_by_liquidity_weight_others_and_unknown_1_for_correlation_and_loss"
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
        if self.plain_weight is not None:
            result.update(
                plain_weight=self.plain_weight,
                plain="same_correlation_without_the_offset_subtracted_with_its_own_weight",
            )
        if self.stability > 0.0:
            result.update(
                stability=self.stability,
                stability_penalty=(
                    "stability_times_weight_times_regularized_sample_std_of_session_"
                    "correlations_when_at_least_two_sessions"
                ),
            )
        return result

    def _combine(self, correlations, weight):
        value = -weight * correlations.mean()
        if self.stability > 0.0 and len(correlations) >= 2:
            value = value + self.stability * weight * session_correlation_std(correlations)
        return value

    def term(self, point, target, sessions, liquidity=None, offset=None):
        """−w por la correlación media de las sesiones del tramo, o cero si no hay ninguna."""
        if self.weighted:
            if liquidity is None:
                raise ValueError("El término recibe la liquidez de cada fila solo si la declara")
        elif liquidity is not None:
            if self.liquidity_column is None:
                raise ValueError("El término recibe la liquidez de cada fila solo si la declara")
            if liquidity.shape != point.shape:
                raise ValueError("La liquidez debe ir fila a fila con el punto")
        if self.shifted != (offset is not None) or (
            offset is not None and offset.shape != point.shape
        ):
            raise ValueError("El término recibe el desplazamiento de cada fila solo si lo declara")
        active_liquidity = liquidity if self.weighted else None
        active_weight = self.liquidity_weight if self.top_third_weighted else None
        plain = None
        if offset is not None:
            if self.plain_weight is not None:
                # La etiqueta tal cual, antes de sumar el desplazamiento.
                plain = session_correlations(
                    point,
                    target,
                    sessions,
                    self.minimum_rows,
                    liquidity=active_liquidity,
                    power=self.liquidity_power,
                    liquidity_weight=active_weight,
                )
            if offset.device != point.device:
                raise ValueError(
                    "El desplazamiento conocido debe estar en el dispositivo del punto"
                )
            # Lo operable: la etiqueta y su predicción con el desplazamiento conocido sumado.
            offset = offset.detach().to(point.dtype)
            require_finite(offset, "El desplazamiento conocido debe ser finito")
            point, target = point + offset, target + offset
        correlations = session_correlations(
            point,
            target,
            sessions,
            self.minimum_rows,
            liquidity=active_liquidity,
            power=self.liquidity_power,
            liquidity_weight=active_weight,
        )
        if not len(correlations):
            return point.sum() * 0
        value = self._combine(correlations, self.weight)
        if plain is not None:
            value = value + self._combine(plain, self.plain_weight)
        return value


_OPTIONAL_TERM_KEYS = frozenset(
    {
        "liquidity_column",
        "liquidity_power",
        "liquidity_weight",
        "stability",
        "offset_column",
        "offset_scale",
        "plain_weight",
    }
)


def parse_session_ic_term(raw):
    """Aceptar la configuración, sus valores o None."""
    if raw is None:
        return None
    if type(raw) is SessionICTerm:
        raw.__post_init__()
        return raw
    required = {"weight", "minimum_rows"}
    if (
        type(raw) is dict
        and required <= set(raw) <= required | _OPTIONAL_TERM_KEYS
        and all(raw[key] is not None for key in raw if key in _OPTIONAL_TERM_KEYS)
    ):
        return SessionICTerm(**raw)
    raise ValueError(
        "session_ic_term debe declarar weight y minimum_rows (y opcionalmente liquidity_column "
        "con liquidity_power o liquidity_weight, stability, u offset_column con offset_scale y, "
        "con ellas, plain_weight), o ser None"
    )


def known_offset(values, scale):
    """Desplazamiento conocido de cada fila: su columna por la escala, y cero si no es finita."""
    # La finitud se comprueba después de escalar: un valor finito puede desbordar al multiplicarlo.
    scaled = values.detach() * scale
    return torch.where(torch.isfinite(scaled), scaled, torch.zeros_like(scaled))


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


def top_third_weights(values, weight):
    """Pesos brutos de una sesión: `weight` en el tercio superior conocido y 1 en el resto.

    El umbral es la entrada `(2 · m) // 3` de los `m` valores finitos ordenados (empates
    incluidos), invariante al orden de las filas. Las filas con liquidez no finita (o todas
    si ninguna es finita) reciben peso 1.
    """
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 1
        or values.numel() == 0
        or values.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("La liquidez necesita un vector no vacío en FP32 o FP64")
    weight = _number(weight, "liquidity_weight")
    if not 1.0 <= weight <= 1e3:
        raise ValueError("liquidity_weight debe ser un número entre 1 y 1000")
    values = values.detach()
    ones = torch.ones_like(values)
    if weight == 1.0:
        return ones
    known = torch.isfinite(values)
    # Las filas sin liquidez conocida van al final y nunca fijan ni superan el umbral.
    infinite = torch.full((), torch.inf, dtype=values.dtype, device=values.device)
    ordered = torch.where(known, values, infinite).sort().values
    count = known.sum(dtype=torch.int64)
    index = ((2 * count) // 3).clamp(max=values.numel() - 1)
    threshold = ordered.index_select(0, index.unsqueeze(0)).squeeze(0)
    finite_values = torch.where(known, values, -infinite)
    top = known & (finite_values >= threshold)
    return torch.where(top, torch.full_like(values, weight), ones)


def session_row_weights(liquidity, sessions, weight):
    """Pesos brutos fila a fila del tramo, eligiendo el tercio líquido dentro de cada sesión."""
    if (
        not isinstance(liquidity, torch.Tensor)
        or liquidity.ndim != 1
        or len(sessions) != len(liquidity)
        or liquidity.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Los pesos por sesión necesitan la liquidez de cada fila en FP32 o FP64")
    weight = _number(weight, "liquidity_weight")
    if not 1.0 <= weight <= 1e3:
        raise ValueError("liquidity_weight debe ser un número entre 1 y 1000")
    source = liquidity.detach()
    if len(source) == 0:
        return source.clone()
    if weight == 1.0:
        return torch.ones_like(source)
    groups = {}
    for row, key in enumerate(sessions):
        groups.setdefault(key, []).append(row)
    result = torch.ones_like(source)
    for rows in groups.values():
        if rows[-1] - rows[0] + 1 == len(rows):
            start, stop = rows[0], rows[-1] + 1
            result[start:stop] = top_third_weights(source[start:stop], weight)
        else:
            index = _row_index(rows, source.device)
            result.index_copy_(0, index, top_third_weights(source.index_select(0, index), weight))
    return result


def session_correlation_std(correlations):
    """Desviación típica muestral regularizada de las correlaciones por sesión (>= 2).

    El suelo eps² de la precisión garantiza un gradiente finito y nulo cuando todas las
    sesiones comparten la misma correlación, incluida la predicción constante.
    """
    if (
        not isinstance(correlations, torch.Tensor)
        or correlations.ndim != 1
        or correlations.numel() < 2
        or correlations.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("La desviación típica necesita al menos dos correlaciones en FP32 o FP64")
    centered = correlations - correlations.mean()
    variance = centered.square().sum() / (correlations.numel() - 1)
    epsilon = torch.finfo(correlations.dtype).eps
    return torch.sqrt(variance + epsilon * epsilon)


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


def _row_index(rows, device):
    if device.type != "cuda":
        return torch.tensor(rows, dtype=torch.int64, device=device)
    return torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
        device=device, non_blocking=True
    )


def standardized_scores(scores, markets):
    """Centrar y escalar las puntuaciones de cada mercado del lote, que es una sesión."""
    if scores.ndim != 1 or len(markets) != len(scores):
        raise ValueError("Las puntuaciones necesitan un mercado por fila")
    if len(scores) == 0:
        return torch.zeros_like(scores)
    first = markets[0]
    if all(value == first for value in markets):
        return standardized(scores)
    result = torch.zeros_like(scores)
    for market in sorted(set(markets)):
        positions = [i for i, value in enumerate(markets) if value == market]
        if positions[-1] - positions[0] + 1 == len(positions):
            start, stop = positions[0], positions[-1] + 1
            result = result.clone()
            result[start:stop] = standardized(scores[start:stop])
        else:
            rows = _row_index(positions, scores.device)
            result = result.index_copy(0, rows, standardized(scores.index_select(0, rows)))
    return result


def session_correlations(
    scores, target, sessions, minimum_rows, *, liquidity=None, power=None, liquidity_weight=None
):
    """Correlación regularizada por sesión entre puntuaciones y rangos normales del objetivo.

    Usa el suelo de `standardized`. Una sesión con objetivo constante aporta cero. Devuelve
    una entrada por sesión con suficientes filas, en el orden de primera aparición. Con
    `liquidity` y `power`, cada sesión pondera sus filas con `liquidity_weights`. Con
    `liquidity` y `liquidity_weight > 1`, pondera con `top_third_weights` normalizados a suma uno.
    """
    if (
        scores.ndim != 1
        or target.shape != scores.shape
        or len(sessions) != len(scores)
        or scores.dtype != target.dtype
    ):
        raise ValueError("Puntuaciones, objetivos y sesiones deben ir fila a fila")
    if liquidity_weight is not None:
        liquidity_weight = _number(liquidity_weight, "liquidity_weight")
        if not 1.0 <= liquidity_weight <= 1e3:
            raise ValueError("liquidity_weight debe ser un número entre 1 y 1000")
    uses_power = power is not None
    uses_top_third = liquidity_weight is not None and liquidity_weight > 1.0
    if uses_power and uses_top_third:
        raise ValueError("No se combinan power y liquidity_weight en la correlación ponderada")
    if uses_power or uses_top_third:
        if (
            liquidity is None
            or liquidity.shape != scores.shape
            or liquidity.device != scores.device
        ):
            raise ValueError("La ponderación necesita la liquidez de cada fila y su factor")
    elif liquidity is not None:
        if liquidity_weight is None:
            raise ValueError("La ponderación necesita la liquidez de cada fila y su exponente")
        if liquidity.shape != scores.shape or liquidity.device != scores.device:
            raise ValueError("La liquidez debe ir fila a fila con las puntuaciones")
    groups = {}
    for row, key in enumerate(sessions):
        groups.setdefault(key, []).append(row)
    values = []
    for rows in groups.values():
        if len(rows) < minimum_rows:
            continue
        if rows[-1] - rows[0] + 1 == len(rows):
            start, stop = rows[0], rows[-1] + 1
            score = scores[start:stop]
            labels = target[start:stop].unsqueeze(-1)
            liq = None if not (uses_power or uses_top_third) else liquidity.detach()[start:stop]
        else:
            index = _row_index(rows, scores.device)
            score = scores.index_select(0, index)
            labels = target.index_select(0, index).unsqueeze(-1)
            liq = (
                None
                if not (uses_power or uses_top_third)
                else liquidity.detach().index_select(0, index)
            )
        normal = gaussian_ranks(labels, torch.ones_like(labels, dtype=torch.bool)).squeeze(-1)
        if liq is None:
            values.append((standardized(score) * standardized(normal)).mean())
            continue
        if uses_power:
            weights = liquidity_weights(liq, power)
        else:
            raw = top_third_weights(liq, liquidity_weight).to(score.dtype)
            weights = raw / raw.sum()
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
