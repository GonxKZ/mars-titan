"""Rangos gaussianos por sesión de las entradas numéricas (F4 de #749, `session_rank_inputs_v1`).

Cada columna numérica se sustituye por la puntuación normal de su rango entre las filas de la
misma sesión, es decir, del mismo mercado y del mismo instante de decisión:

    g_i = Φ⁻¹((r_i − 1/2) / n)

r_i es el rango de la fila i entre las n filas en que la columna está presente, con los empates
promediados. Es la transformación inversa normal basada en rangos con desplazamiento c = 1/2
(Beasley, Erickson y Allison, 2009, doi:10.1007/s10519-009-9281-0). Gu, Kelly y Xiu (2020,
doi:10.1093/rfs/hhaa009) ordenan las características de cada mes en sección cruzada antes de
ajustar sus modelos. Aquí se usa la puntuación normal en lugar del rango lineal en [−1, 1].

Solo se transforman los cinco canales de la ventana de precios, columna a columna por sesión
de la ventana, y los valores de los fundamentales cuya máscara vale uno. Las máscaras, las
edades, los bits de presencia, los vectores de noticias y gráficos y el contexto macro, igual
para todos los activos de una sesión, no cambian. Una posición ausente sigue valiendo cero.

La transformación solo lee filas del lote, que tiene un único instante de decisión y entradas
disponibles en él. Si el lote no contiene la sesión completa, los rangos serían los de un
subconjunto. Por eso el predictor la exige con `session_complete=True`, que solo declara quien
ha comprobado que el bloque contiene todas las filas de su instante.
"""

import torch

from .state import require_finite

SESSION_RANK_INPUTS = "session_rank_inputs_v1"
PRICE_VALUE_CHANNELS = 5


def identity():
    return dict(
        schema_version=1,
        name=SESSION_RANK_INPUTS,
        session="rows_of_one_market_and_decision_instant_in_one_complete_block",
        normal_scores="inverse_normal_of_rank_minus_half_over_present_rows",
        ties="average_rank",
        transformed=dict(
            prices="five_value_channels_per_window_step_among_rows_present_at_that_step",
            windows_without_presence_channel="every_step_present",
            fundamentals="values_among_rows_with_concept_mask",
        ),
        unchanged=[
            "price_presence",
            "fundamental_masks",
            "fundamental_ages",
            "news",
            "charts",
            "macro",
            "presence",
        ],
        absent_value=0.0,
    )


def gaussian_ranks(values, present):
    """Puntuaciones normales de los rangos de cada columna entre sus filas presentes.

    `values` y `present` tienen forma [filas, columnas]. Devuelve el mismo dtype y cero en
    las posiciones ausentes. Con una sola fila presente la puntuación es cero.
    """
    if (
        not isinstance(values, torch.Tensor)
        or not isinstance(present, torch.Tensor)
        or values.ndim != 2
        or present.shape != values.shape
        or present.dtype != torch.bool
        or not values.dtype.is_floating_point
    ):
        raise ValueError("Los rangos necesitan valores reales [filas, columnas] y su presencia")
    if values.dtype not in (torch.float32, torch.float64):
        raise ValueError("La precisión de los rangos debe ser FP32 o FP64")
    if present.device != values.device:
        raise ValueError("La presencia y los valores necesitan el mismo dispositivo")
    if values.shape[0] == 0:
        return values.clone()
    observed = torch.where(present, values, 0.0)
    require_finite(observed, "Los valores presentes de los rangos deben ser finitos")
    infinite = torch.full((), torch.inf, dtype=values.dtype, device=values.device)
    # Las posiciones ausentes se ordenan al final y no cuentan como menores que ninguna presente.
    masked = torch.where(present, values, infinite).T.contiguous()
    ordered = masked.sort(dim=-1).values
    lower = torch.searchsorted(ordered, masked, right=False)
    upper = torch.searchsorted(ordered, masked, right=True)
    # Las posiciones lower + 1, ..., upper comparten valor y su rango medio es la media de ambas.
    rank = (lower + upper + 1).to(values.dtype) / 2
    count = present.sum(dim=0, keepdim=True).T.to(values.dtype).clamp_min(1)
    probability = ((rank - 0.5) / count).T
    # No evaluar ndtri fuera de (0, 1) en las posiciones ausentes. ndtri(0,5) vale cero.
    return torch.special.ndtri(torch.where(present, probability, 0.5))


def _ranked_inputs(inputs, rows):
    """Transformar las filas `rows` de un mercado y devolver sus valores derivados."""
    prices = inputs["prices"].index_select(0, rows)
    size, steps, channels = prices.shape
    # Las ventanas anteriores a la v3.1 no tienen bit de presencia y todas sus sesiones existen.
    step_present = (
        prices[..., -1] > 0
        if channels > PRICE_VALUE_CHANNELS
        else torch.ones((size, steps), dtype=torch.bool, device=prices.device)
    )
    present = step_present.unsqueeze(-1).expand(-1, -1, PRICE_VALUE_CHANNELS)
    values = prices[..., :PRICE_VALUE_CHANNELS].reshape(size, -1)
    ranked = gaussian_ranks(values, present.reshape(size, -1)).reshape(
        size, steps, PRICE_VALUE_CHANNELS
    )
    numeric = inputs["fundamentals"].index_select(0, rows)
    data, masks, ages = numeric.chunk(3, dim=1)
    fundamentals = torch.cat((gaussian_ranks(data, masks > 0), masks, ages), dim=1)
    return torch.cat((ranked, prices[..., PRICE_VALUE_CHANNELS:]), dim=-1), fundamentals


def rank_session_inputs(inputs, markets):
    """Entradas con precios y fundamentales en rangos gaussianos por mercado del lote."""
    prices = inputs["prices"]
    if prices.ndim != 3 or prices.shape[-1] not in (PRICE_VALUE_CHANNELS, PRICE_VALUE_CHANNELS + 1):
        raise ValueError("Los rangos por sesión necesitan cinco canales y, si existe, la presencia")
    if inputs["fundamentals"].shape[1] % 3 or len(markets) != len(prices):
        raise ValueError("Los fundamentales necesitan valores, máscaras y edades por fila")
    result = dict(inputs)
    ranked_prices, ranked_fundamentals = prices.clone(), inputs["fundamentals"].clone()
    for market in sorted(set(markets)):
        rows = torch.tensor(
            [i for i, value in enumerate(markets) if value == market],
            dtype=torch.int64,
            device=prices.device,
        )
        price_values, fundamental_values = _ranked_inputs(inputs, rows)
        ranked_prices.index_copy_(0, rows, price_values)
        ranked_fundamentals.index_copy_(0, rows, fundamental_values)
    result.update(prices=ranked_prices, fundamentals=ranked_fundamentals)
    return result
