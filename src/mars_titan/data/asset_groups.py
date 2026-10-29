"""Grupos de activos de una sesión con la ventana pasada, para H4 de #783.

H4 (`asset_group_membership_v1`) reparte los activos de cada sesión entre grupos con una
pertenencia fraccionaria que solo usa la ventana de 64 sesiones que termina en la sesión de la
decisión. Con c, h y l el logaritmo del cierre, del máximo y del mínimo respecto del ancla de la
ventana, y r_t = c_t − c_{t−1} cuando las dos sesiones están presentes, cada fila tiene cinco
medidas:

- **Volatilidad.** Desviación típica de r con ddof = 1.
- **Momentum.** c de la última sesión presente menos c de la primera, el rendimiento
  logarítmico de la ventana.
- **Diferencial.** Estimador CHL de Abdi y Ranaldo (2017) en su versión corregida por dos días,
  la media de sqrt(max(4 (c_t − η_t)(c_t − η_{t+1}), 0)) con η = (h + l)/2. Se descartan los
  pares con una sesión ausente o sin rango (h = l), como en las simulaciones del artículo. En
  su aplicación empírica el artículo arrastra los precios del día anterior, y aquí no se imputa
  ningún precio. El volumen de la ventana está dividido por su media en cada activo y no permite
  comparar activos, por eso la liquidez sale de los precios.
- **Beta.** Pendiente de MCO de r frente a m, la media equiponderada de los rendimientos de las
  filas válidas de la sesión en cada paso.
- **Correlación.** Con el residuo centrado e = (r − r̄) − β (m − m̄) normalizado, z_i · z_j es la
  correlación de los residuos de los activos i y j, porque la v3.1 solo admite ausencias de todo
  el mercado y las filas de una sesión tienen las mismas sesiones presentes. Las K primeras
  componentes principales de z, los vectores propios de ZᵀZ de mayor valor propio, dan una
  inmersión que se normaliza por filas como en el paso 4 de Ng, Jordan y Weiss (2002). Así,
  |y_i − y_j|² = 2 (1 − ρ̂_ij), con ρ̂ la correlación reconstruida con las K componentes, y dos
  filas quedan más lejos cuanto menos se parecen sus residuos. A diferencia de Ng, Jordan y
  Weiss, la matriz de la que salen los vectores propios es la correlación de los residuos, que
  puede ser negativa, y no una afinidad gaussiana normalizada por el grado. Las c-medias
  difusas de Bezdek (1981), con m = 2, reparten la inmersión en K grupos desde los centros del
  recorrido más lejano de Gonzalez (1985). Agrupar los residuos y no los rendimientos evita que
  el modo común del mercado decida los grupos. Agrupar la inmersión de dimensión K y no las 63
  sesiones responde a que, en dimensión alta, los prototipos de las c-medias difusas acaban en
  el centro de gravedad de los datos (Winkler, Klawonn y Kruse, 2011).

Cada medida necesita al menos `min_observations` rendimientos o pares válidos. El valor por
defecto, 36, aplica a los 63 pares de la ventana la proporción entre los 12 días aplicables que
el artículo de Abdi y Ranaldo exige en un mes y los meses de 21 sesiones de sus simulaciones.
Las cuatro primeras medidas se convierten en el rango medio u ∈ (0, 1) entre las filas válidas
de la sesión, y u en una partición difusa de Ruspini (1969) con Q cubos: con
p = clamp(uQ − 1/2, 0, Q − 1), la fila pertenece al cubo ⌊p⌋ con 1 − (p − ⌊p⌋) y al siguiente
con el resto. Una fila sin observaciones suficientes no pertenece a ningún grupo de esa familia.
Las pertenencias de cada familia suman uno en las demás.

Todo se calcula sin gradiente y en el dispositivo de la ventana, sin copias al anfitrión. Con
menos filas válidas que K, algunos centros se repiten y cada fila queda en su propio grupo.
"""

import math

import torch
from torch.nn import functional as F

from .price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS

BUCKET_FAMILIES = ("volatility", "momentum", "liquidity_spread", "beta")
FAMILIES = (*BUCKET_FAMILIES, "correlation")
_OPEN, _HIGH, _LOW, _CLOSE = range(4)
# Las distancias al cuadrado entre puntos de norma uno no superan 4. Por debajo de este valor
# el punto coincide con el centro y recibe casi toda su pertenencia.
_TINY = 1e-12


def _validate(prices):
    if (
        not isinstance(prices, torch.Tensor)
        or prices.ndim != 3
        or prices.shape[1:] != (CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS))
        or len(prices) == 0
        or prices.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError(
            "H4 necesita ventanas de 64 sesiones con el bit de presencia en FP32 o FP64"
        )


def window_measures(prices, min_observations):
    """Medidas de cada fila de una sesión y si cada una tiene observaciones suficientes.

    Devuelve un diccionario con las cuatro medidas de los cubos, el residuo normalizado de la
    correlación con forma [filas, 63] y una máscara de validez por familia.
    """
    _validate(prices)
    with torch.no_grad():
        present = prices[..., -1] > 0
        close, high, low = prices[..., _CLOSE], prices[..., _HIGH], prices[..., _LOW]
        valid = present[:, 1:] & present[:, :-1]
        returns = torch.where(valid, close[:, 1:] - close[:, :-1], 0.0)
        count = valid.sum(1)
        mean = returns.sum(1) / count.clamp_min(1)
        centered = torch.where(valid, returns - mean[:, None], 0.0)
        volatility = (centered.square().sum(1) / (count - 1).clamp_min(1)).sqrt()
        enough = count >= min_observations

        first = present.to(torch.int64).argmax(1)
        last = present.shape[1] - 1 - present.flip(1).to(torch.int64).argmax(1)
        momentum = (close.gather(1, last[:, None]) - close.gather(1, first[:, None])).squeeze(1)

        mid = (high + low) / 2
        ranged = present & (high > low)
        pairs = ranged[:, 1:] & ranged[:, :-1]
        squared = 4 * (close[:, :-1] - mid[:, :-1]) * (close[:, :-1] - mid[:, 1:])
        spread_terms = torch.where(pairs, squared.clamp_min(0).sqrt(), 0.0)
        pair_count = pairs.sum(1)
        spread = spread_terms.sum(1) / pair_count.clamp_min(1)

        # El mercado de cada paso es la media de las filas con rendimientos suficientes.
        weight = valid & enough[:, None]
        members = weight.sum(0)
        market = (returns * weight).sum(0) / members.clamp_min(1)
        covered = valid & (members > 0)[None, :]
        used = covered.sum(1)
        returns_mean = torch.where(covered, returns, 0.0).sum(1) / used.clamp_min(1)
        market_mean = torch.where(covered, market[None, :], 0.0).sum(1) / used.clamp_min(1)
        deviation = torch.where(covered, returns - returns_mean[:, None], 0.0)
        market_deviation = torch.where(covered, market[None, :] - market_mean[:, None], 0.0)
        variance = market_deviation.square().sum(1)
        regular = enough & (used >= min_observations) & (variance > 0)
        beta = (deviation * market_deviation).sum(1) / torch.where(regular, variance, 1.0)
        beta = torch.where(regular, beta, 0.0)
        residual = deviation - beta[:, None] * market_deviation
        norm = residual.norm(dim=1)
        correlated = regular & (norm > 0)
        normalized = torch.where(
            correlated[:, None], residual / torch.where(correlated, norm, 1.0)[:, None], 0.0
        )
    return dict(
        volatility=volatility,
        momentum=momentum,
        liquidity_spread=spread,
        beta=beta,
        residual=normalized,
        valid=dict(
            volatility=enough,
            momentum=enough,
            liquidity_spread=pair_count >= min_observations,
            beta=regular,
            correlation=correlated,
        ),
    )


def midranks(values, valid):
    """Rango medio u ∈ (0, 1) entre las filas válidas, con los empates en el mismo valor.

    Las filas no válidas reciben 0 y no cuentan para las demás.
    """
    with torch.no_grad():
        dtype = values.dtype
        smaller = ((values[None, :] < values[:, None]) & valid[None, :]).sum(1).to(dtype)
        equal = ((values[None, :] == values[:, None]) & valid[None, :]).sum(1).to(dtype)
        rows = valid.sum().clamp_min(1).to(dtype)
        return torch.where(valid, (smaller + 0.5 * equal) / rows, 0.0)


def fuzzy_buckets(ranks, valid, buckets):
    """Partición difusa de Ruspini de Q cubos: cada fila válida reparte uno entre dos vecinos."""
    with torch.no_grad():
        position = (ranks * buckets - 0.5).clamp(0, buckets - 1)
        lower = position.floor()
        fraction = position - lower
        lower = lower.to(torch.int64)
        upper = (lower + 1).clamp_max(buckets - 1)
        result = F.one_hot(lower, buckets).to(ranks.dtype) * (1 - fraction)[:, None]
        result = result + F.one_hot(upper, buckets).to(ranks.dtype) * fraction[:, None]
        return torch.where(valid[:, None], result, 0.0)


def _squared_distances(points, centers):
    # La diferencia directa evita el atajo con productos de `cdist`, que pierde precisión.
    return (points[:, None, :] - centers[None, :, :]).square().sum(-1)


def farthest_first(points, valid, count):
    """Centros del recorrido más lejano de Gonzalez (1985) entre las filas válidas.

    El primero es el punto más alejado de la media y cada siguiente el más alejado de los ya
    elegidos. Los índices se quedan en el dispositivo, sin sincronizar con el anfitrión.
    """
    with torch.no_grad():
        rows = valid.sum().clamp_min(1).to(points.dtype)
        mean = torch.where(valid[:, None], points, 0.0).sum(0) / rows
        excluded = torch.full_like(points[:, 0], -math.inf)
        distance = torch.where(valid, (points - mean).square().sum(1), excluded)
        chosen = []
        for step in range(count):
            index = distance.argmax().reshape(1)
            chosen.append(index)
            gap = (points - points.index_select(0, index)).square().sum(1)
            # Tras el primer centro, cada fila guarda la distancia al centro elegido más cercano.
            distance = torch.where(
                valid, gap if step == 0 else torch.minimum(distance, gap), excluded
            )
        return points.index_select(0, torch.cat(chosen))


def _fuzzy_weights(points, centers):
    """Pertenencias de las c-medias difusas con m = 2: u_ik ∝ 1 / d²(x_i, c_k)."""
    inverse = _squared_distances(points, centers).clamp_min(_TINY).reciprocal()
    return inverse / inverse.sum(1, keepdim=True)


def fuzzy_clusters(points, valid, clusters, iterations):
    """C-medias difusas de Bezdek (1981) con m = 2 e iteraciones fijas sobre las filas válidas."""
    with torch.no_grad():
        centers = farthest_first(points, valid, clusters)
        for _ in range(iterations):
            weights = _fuzzy_weights(points, centers).square() * valid[:, None]
            # Con m = 2 cada centro es la media de los puntos ponderada por u².
            centers = (weights.T @ points) / weights.sum(0).clamp_min(_TINY)[:, None]
        return torch.where(valid[:, None], _fuzzy_weights(points, centers), 0.0)


def correlation_embedding(residual, valid, dimensions):
    """K primeras componentes principales del residuo normalizado, con norma uno por fila."""
    with torch.no_grad():
        masked = torch.where(valid[:, None], residual, 0.0)
        # eigh devuelve los valores propios en orden creciente. Las componentes de valor propio
        # nulo dan columnas nulas, así que su base arbitraria no cambia la inmersión.
        _, vectors = torch.linalg.eigh(masked.T @ masked)
        scores = masked @ vectors[:, -dimensions:]
        norm = scores.norm(dim=1)
        kept = valid & (norm > 0)
        embedded = torch.where(kept[:, None], scores / torch.where(kept, norm, 1.0)[:, None], 0.0)
        return embedded, kept


def session_memberships(prices, *, buckets, clusters, min_observations, iterations):
    """Pertenencias de las filas de una sesión, con las cinco familias en el orden de FAMILIES.

    Devuelve [filas, 4Q + K]: Q cubos por cada medida y K grupos de correlación.
    """
    measures = window_measures(prices, min_observations)
    parts = []
    for family in BUCKET_FAMILIES:
        valid = measures["valid"][family]
        parts.append(fuzzy_buckets(midranks(measures[family], valid), valid, buckets))
    points, kept = correlation_embedding(
        measures["residual"], measures["valid"]["correlation"], clusters
    )
    parts.append(fuzzy_clusters(points, kept, clusters, iterations))
    return torch.cat(parts, dim=1)


def family_slices(buckets, clusters):
    """Columnas de cada familia en las pertenencias de `session_memberships`."""
    widths = [buckets] * len(BUCKET_FAMILIES) + [clusters]
    starts = [sum(widths[:index]) for index in range(len(widths))]
    return tuple(slice(start, start + width) for start, width in zip(starts, widths, strict=True))
