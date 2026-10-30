"""Regiones por correlación de series observadas en fecha, comunes a C3 y C4 de #751.

C4 agrupa los residuos ya maduros de cada activo. C3 y el sector estadístico de B3 (#750)
agrupan los rendimientos diarios de su ventana de precios. Todas usan la misma regla:

    serie estandarizada con la media y la desviación de sus propias observaciones
    ρ_ab  correlación del par sobre sus observaciones comunes
    regiones  enlace medio sobre 1 − ρ cortado en k grupos

Un par con menos de `min_observations` observaciones comunes no aporta evidencia y su
correlación vale cero. Las regiones se numeran desde 1 por el menor activo que contienen, así
que no dependen del orden de las filas. La región 0 reúne los activos sin `min_observations`
observaciones o con la serie constante.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .price_windows import PRICE_FEATURES, PRICE_WINDOW_CHANNELS

NO_HISTORY = "0"
CLOSE = PRICE_FEATURES.index("log_close_to_anchor_close")


def window_returns(window):
    """Rendimientos diarios del cierre de ventanas [activos, sesiones, canales] y su máscara.

    Con el canal de presencia solo cuenta el rendimiento entre dos sesiones presentes. Las
    ventanas de cinco canales no tienen huecos. La ventana se lee en FP64.
    """
    window = np.asarray(window, dtype=np.float64)
    observed = (
        window[..., -1] > 0
        if window.shape[-1] == len(PRICE_WINDOW_CHANNELS)
        else np.ones(window.shape[:2], dtype=bool)
    )
    returns = np.diff(window[..., CLOSE], axis=1)
    return returns, observed[:, 1:] & observed[:, :-1]


def correlation(series, mask, *, min_observations):
    """Correlación por pares sobre las observaciones comunes de series ya estandarizadas.

    `series` es [activos, instantes] con ceros donde `mask` es falso. Un par con menos de
    `min_observations` instantes comunes vale cero.
    """
    values = np.where(mask, series, 0.0)
    present = mask.astype(np.float64)
    numerator = values @ values.T
    squares = (values * values) @ present.T
    denominator = np.sqrt(squares * squares.T)
    common = present @ present.T
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denominator > 0, numerator / denominator, 0.0)
    result = np.where(common >= min_observations, np.clip(result, -1.0, 1.0), 0.0)
    np.fill_diagonal(result, 1.0)
    return result


def correlation_regions(assets, series, mask, *, clusters, min_observations):
    """Región de cada activo a partir de su serie [activos, instantes] y su máscara.

    `assets` está ordenado sin repeticiones y nombra las filas de `series`. Las posiciones
    sin observar no cuentan, valgan lo que valgan.
    """
    assets = list(assets)
    mask = np.asarray(mask, dtype=bool)
    series = np.asarray(series, dtype=np.float64)
    if (
        series.ndim != 2
        or mask.shape != series.shape
        or len(assets) != len(series)
        or assets != sorted(set(assets))
    ):
        raise ValueError("Las regiones necesitan activos ordenados con una serie y su máscara")
    series = np.where(mask, series, 0.0)
    if not np.isfinite(series).all():
        raise ValueError("Las series de las regiones contienen valores no finitos")
    counts = mask.sum(axis=1)
    mean = np.divide(series.sum(axis=1), counts, out=np.zeros(len(assets)), where=counts > 0)
    centered = np.where(mask, series - mean[:, None], 0.0)
    spread = np.sqrt(
        np.divide((centered**2).sum(axis=1), counts, out=np.zeros(len(assets)), where=counts > 0)
    )
    usable = (counts >= min_observations) & (spread > 0)
    result = dict.fromkeys(assets, NO_HISTORY)
    chosen = np.flatnonzero(usable)
    if len(chosen) == 0:
        return result
    if len(chosen) == 1:
        result[assets[chosen[0]]] = "1"
        return result
    standardized = centered[chosen] / spread[chosen, None]
    rho = correlation(standardized, mask[chosen], min_observations=min_observations)
    distance = squareform(1.0 - rho, checks=False)
    groups = fcluster(linkage(distance, method="average"), t=clusters, criterion="maxclust")
    first = {}
    for position, group in zip(chosen, groups, strict=True):
        first.setdefault(int(group), assets[position])
    order = {group: str(rank + 1) for rank, group in enumerate(sorted(first, key=first.get))}
    for position, group in zip(chosen, groups, strict=True):
        result[assets[position]] = order[int(group)]
    return result


def permuted_regions(regions, seed):
    """Las mismas etiquetas y tamaños repartidos al azar entre los activos con historia."""
    names = sorted(name for name, region in regions.items() if region != NO_HISTORY)
    generator = np.random.default_rng(np.random.SeedSequence(seed))
    shuffled = [regions[names[index]] for index in generator.permutation(len(names))]
    return {**regions, **dict(zip(names, shuffled, strict=True))}
