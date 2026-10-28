"""Puerta de reescalado por regiones en fecha `pit_region_gate_v1` (C4 del estudio, #751).

La etapa agrupa los activos de cada mercado por la correlación de sus residuos ya maduros y
reescala la predicción con un factor por región. No cambia el modelo ni sus pesos:

    ρ_ab  correlación de los residuos de a y b en las W últimas madureces anteriores a t_G
    regiones  enlace medio sobre 1 − ρ cortado en k grupos, en cada mercado
    λ_{g,r} = argmin_{λ∈[0,1]} Σ_i w_i·|y_i − λ·x_i| en las filas de G de la región r
    ŷ_i = λ_{g,r(i)}·x_i

t_G es la primera emisión de la calibración (G), así que las regiones solo usan etiquetas que
ya habían madurado entonces. Regiones y factores quedan congelados antes de abrir la
evaluación. w_i es el peso 1/n de la sesión y λ se elige como en A1 (`scale_diagnostics`). Los
activos sin historia suficiente forman la región 0. Un activo ausente de la calibración, o una
región sin factor, usa el factor de su mercado, y un mercado sin factor predice cero como en A1.

`regions` declara la variante o uno de sus controles:

- `residual_correlation`: la variante, con las regiones de la correlación pasada.
- `permuted`: las mismas regiones y tamaños, repartidas al azar entre los activos con historia.
- `current_sector`: el sector de las fichas actuales, que no es una clasificación en fecha.
  Mide cuánto aportaría una etiqueta con fuga, nunca una evaluación válida por sí misma.
- `market`: una sola región por mercado, que reproduce A1.

La correlación usa cada serie estandarizada con sus propias etiquetas de la ventana y se
calcula sobre las madureces comunes a cada par. Un par con menos de `min_labels` madureces
comunes no aporta evidencia y su correlación vale cero.
"""

import hashlib
import json

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import scale_diagnostics
from mars_titan.evaluation.forecast_panel import ForecastPanel

IDENTITY = "pit_region_gate_v1"
SCHEMA_VERSION = 1
REGIONS = ("residual_correlation", "permuted", "current_sector", "market")
NO_HISTORY = "0"
# Unas 126 sesiones son los seis meses de la validación de G que preceden a la calibración.
DEFAULT = dict(
    regions="residual_correlation",
    clusters=8,
    window=126,
    min_labels=60,
    seed=20261027,
)
_RECORD = {
    "schema_version",
    "identity",
    "config",
    "fitted_until",
    "regions",
    "factors",
    "market_factors",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_config(config):
    """Validar la configuración declarada de la etapa y devolver una copia."""
    _require(
        isinstance(config, dict) and set(config) == set(DEFAULT),
        f"La puerta por regiones declara exactamente {sorted(DEFAULT)}",
    )
    _require(
        type(config["regions"]) is str and config["regions"] in REGIONS,
        f"Las regiones deben ser una de {REGIONS}",
    )
    _require(
        type(config["clusters"]) is int and 2 <= config["clusters"] <= 32,
        "El número de regiones debe ser un entero entre 2 y 32",
    )
    window = config["window"]
    _require(
        type(window) is int
        and 20 <= window <= 2520
        and type(config["min_labels"]) is int
        and 10 <= config["min_labels"] <= window,
        "La ventana y su mínimo de etiquetas no son válidos",
    )
    _require(
        type(config["seed"]) is int and 0 <= config["seed"] < 2**32,
        "La semilla del control permutado debe ser un entero de 32 bits",
    )
    return dict(config)


def _instants(values, rows, name):
    values = np.asarray(values)
    _require(
        values.shape == (rows,) and values.dtype.kind in "iu",
        f"{name} debe tener un instante en microsegundos UTC por fila",
    )
    values = values.astype(np.int64)
    _require(np.all(values < FINAL_TEST_START_US), f"{name} entra en el test reservado de 2024")
    return values


def _assets(panel, assets):
    _require(isinstance(panel, ForecastPanel), "La puerta por regiones necesita un panel")
    assets = np.asarray(assets, dtype=str)
    _require(assets.shape == (panel.rows,), "Se necesita el activo de cada fila del panel")
    return assets


def panel_assets(panel):
    """Activo de cada fila a partir de identidades `mercado/activo/instante`."""
    _require(isinstance(panel, ForecastPanel), "La puerta por regiones necesita un panel")
    identities = panel.row_id.to_pylist()
    _require(
        all(isinstance(value, str) for value in identities),
        "Las identidades del panel no nombran el activo",
    )
    result = []
    for value, code in zip(identities, panel.market, strict=True):
        market, separator, rest = value.partition("/")
        asset, slash, _ = rest.rpartition("/")
        _require(
            separator and slash and asset and market == panel.markets[code],
            "Las identidades del panel deben ser mercado/activo/instante",
        )
        result.append(asset)
    return np.asarray(result, dtype=str)


def correlation(series, mask, *, min_labels):
    """Correlación por pares sobre las madureces comunes de series ya estandarizadas.

    `series` es [activos, instantes] con ceros donde `mask` es falso. Un par con menos de
    `min_labels` instantes comunes vale cero.
    """
    values = np.where(mask, series, 0.0)
    present = mask.astype(np.float64)
    numerator = values @ values.T
    squares = (values * values) @ present.T
    denominator = np.sqrt(squares * squares.T)
    common = present @ present.T
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denominator > 0, numerator / denominator, 0.0)
    result = np.where(common >= min_labels, np.clip(result, -1.0, 1.0), 0.0)
    np.fill_diagonal(result, 1.0)
    return result


def residual_regions(assets, history, cutoff, *, clusters, window, min_labels):
    """Región de cada activo con las etiquetas que han madurado en `cutoff` o antes.

    `history` tiene `asset_id`, `target` y `matures_at` alineados. La ventana son las
    `window` últimas madureces de esos activos. Las regiones se numeran desde 1 por el
    menor activo que contienen. La región 0 reúne los activos sin `min_labels` etiquetas
    o con residuos constantes.
    """
    assets = sorted(set(np.asarray(assets, dtype=str).tolist()))
    labels = np.asarray(history["asset_id"], dtype=str)
    target = np.asarray(history["target"], dtype=np.float64)
    matures_at = _instants(history["matures_at"], len(labels), "La madurez del historial")
    _require(
        labels.ndim == 1 and target.shape == labels.shape,
        "El historial necesita activo, objetivo y madurez alineados",
    )
    _require(np.isfinite(target).all(), "El historial contiene objetivos no finitos")
    known = np.isin(labels, assets) & (matures_at <= cutoff)
    labels, target, matures_at = labels[known], target[known], matures_at[known]
    instants = np.unique(matures_at)[-window:]
    keep = np.isin(matures_at, instants)
    labels, target, matures_at = labels[keep], target[keep], matures_at[keep]
    row = np.searchsorted(assets, labels)
    column = np.searchsorted(instants, matures_at)
    flat = row * max(len(instants), 1) + column
    _require(
        len(np.unique(flat)) == len(flat),
        "Un activo tiene dos etiquetas que maduran en el mismo instante",
    )
    series = np.zeros((len(assets), len(instants)))
    mask = np.zeros_like(series, dtype=bool)
    series[row, column], mask[row, column] = target, True
    counts = mask.sum(axis=1)
    mean = np.divide(series.sum(axis=1), counts, out=np.zeros(len(assets)), where=counts > 0)
    centered = np.where(mask, series - mean[:, None], 0.0)
    spread = np.sqrt(
        np.divide((centered**2).sum(axis=1), counts, out=np.zeros(len(assets)), where=counts > 0)
    )
    usable = (counts >= min_labels) & (spread > 0)
    result = dict.fromkeys(assets, NO_HISTORY)
    chosen = np.flatnonzero(usable)
    if len(chosen) == 0:
        return result
    if len(chosen) == 1:
        result[assets[chosen[0]]] = "1"
        return result
    standardized = centered[chosen] / spread[chosen, None]
    rho = correlation(standardized, mask[chosen], min_labels=min_labels)
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


def sector_regions(assets, sectors):
    """Regiones numeradas por el nombre de sector. Sin sector, el activo va a la región 0."""
    _require(isinstance(sectors, dict), "El control sectorial necesita el sector de cada activo")
    names = sorted({sectors[a] for a in assets if sectors.get(a) is not None})
    number = {name: str(rank + 1) for rank, name in enumerate(names)}
    return {a: number.get(sectors.get(a), NO_HISTORY) for a in sorted(set(assets))}


def fit(panel, matures_at, assets, history, config=None, *, sectors=None):
    """Fijar las regiones con etiquetas maduras antes de G y ajustar λ con las filas de G.

    `panel` es la calibración, `matures_at` la madurez de cada una de sus etiquetas y
    `assets` el activo de cada fila. `sectors` solo lo usa el control `current_sector`.
    """
    config = checked_config(DEFAULT if config is None else config)
    assets = _assets(panel, assets)
    matures_at = _instants(matures_at, panel.rows, "La madurez de calibración")
    _require(
        np.all(matures_at > panel.prediction_at),
        "Cada etiqueta debe madurar después de su predicción",
    )
    cutoff = int(panel.prediction_at.min())
    kind = config["regions"]
    _require(
        (kind == "current_sector") == (sectors is not None),
        "Solo el control current_sector recibe los sectores actuales",
    )
    regions = {}
    for code, market in enumerate(panel.markets):
        members = assets[panel.market == code]
        if not len(members):
            continue
        if kind == "market":
            mapping = dict.fromkeys(sorted(set(members.tolist())), "1")
        elif kind == "current_sector":
            mapping = sector_regions(members.tolist(), sectors)
        else:
            mapping = residual_regions(
                members,
                history,
                cutoff,
                clusters=config["clusters"],
                window=config["window"],
                min_labels=config["min_labels"],
            )
            if kind == "permuted":
                mapping = permuted_regions(mapping, [config["seed"], code])
        regions[market] = mapping
    prediction = np.asarray(panel.prediction, dtype=np.float64)
    weights = scale_diagnostics.session_weights(panel)
    factors = {}
    for code, market in enumerate(panel.markets):
        if market not in regions:
            continue
        rows = panel.market == code
        labels = np.array([regions[market][a] for a in assets[rows]])
        factors[market] = {
            region: scale_diagnostics.shrinkage_factor(
                panel.target[rows][labels == region],
                prediction[rows][labels == region],
                weights[rows][labels == region],
            )
            for region in sorted(set(labels.tolist()))
        }
    return dict(
        schema_version=SCHEMA_VERSION,
        identity=IDENTITY,
        config=config,
        fitted_until=int(matures_at.max()),
        regions=regions,
        factors=factors,
        market_factors=scale_diagnostics.market_shrinkage(panel, prediction),
    )


def checked_record(record):
    """Comprobar un registro congelado, también tras leerlo de JSON."""
    _require(
        isinstance(record, dict)
        and set(record) == _RECORD
        and record["schema_version"] == SCHEMA_VERSION
        and record["identity"] == IDENTITY
        and type(record["fitted_until"]) is int
        and record["fitted_until"] < FINAL_TEST_START_US,
        "El registro de la puerta por regiones no conserva su contrato",
    )
    checked_config(record["config"])
    return record


def stage_sha256(record):
    """Huella de la etapa congelada: configuración, regiones, factores y madurez del ajuste."""
    return hashlib.sha256(_canonical(checked_record(record)).encode()).hexdigest()


def apply(record, panel, assets):
    """Predicción reescalada de un tramo posterior, sin leer sus objetivos."""
    record = checked_record(record)
    assets = _assets(panel, assets)
    _require(
        int(panel.prediction_at.min()) >= record["fitted_until"],
        "La puerta usa etiquetas que maduran después de una emisión evaluada",
    )
    scale = np.zeros(panel.rows)
    for code, market in enumerate(panel.markets):
        rows = panel.market == code
        if not rows.any():
            continue
        fallback = record["market_factors"].get(market)
        regions = record["regions"].get(market, {})
        factors = record["factors"].get(market, {})
        names, inverse = np.unique(assets[rows], return_inverse=True)
        chosen = [factors.get(regions[name]) if name in regions else None for name in names]
        chosen = [fallback if factor is None else factor for factor in chosen]
        scale[rows] = np.array([0.0 if f is None else f for f in chosen])[inverse]
    return np.asarray(panel.prediction, dtype=np.float64) * scale
