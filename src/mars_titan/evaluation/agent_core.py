# ruff: noqa: E501
"""Núcleo agéntico sobre las emisiones: abstención, intervalos conformes, atribución y costes.

Es el punto A del encargo del coordinador (22 de noviembre de 2026, 20:50 UTC).
- Medidas descriptivas y secundarias, fuera de la familia primaria del registro confirmatorio.
- Se calculan sobre las emisiones ya hechas, sin reentrenar nada.
- Van por origen, con la configuración de `configs/experiments/posttraining-agent-core.json`,
  cuya huella exige la orden.

Sistemas, los mismos de `organ_aggregation`:
- S1: la media de rangos de titans_mac_nogain y lightgbm_technical.
- S2: ridge_cov_active sobre esos dos y la señal del brazo combinado de los órganos.

Medidas, en la etiqueta y en lo operable salvo donde se dice:
- **A1, abstención selectiva.** El agente se abstiene donde no sabe.
  - Por activo, donde más discrepan los rangos tipificados de la red y de los árboles.
  - Por sesión, donde menos se parecen sus órdenes.
  - La cobertura se elige en calibración, por el IC con los abstenidos a cero.
  - En evaluación se informan el IC del conjunto cubierto y el IC con los abstenidos a cero, frente
    a dos controles a igual cobertura en cada sesión: abstenerse en lo de menor magnitud y al azar.
  - También van la curva de riesgo y cobertura y los tercios de liquidez.
- **A2, intervalos conformes**, solo en la etiqueta: sumar α desplaza por igual el intervalo y la
  etiqueta, así que en lo operable la cobertura es la misma.
  - Se comparan tres cosas: los cuantiles del núcleo tal cual; esos mismos conformalizados en
    calibración; y los del sistema combinado.
  - Los del sistema combinado salen de una regresión cuantílica lineal ajustada en validación sobre
    la media de rangos y la anchura del núcleo, conformalizada en calibración.
  - Cobertura y anchura al 80 % y al 95 % en evaluación: global, por tercio de liquidez, por decil
    de volatilidad y por trimestre.
- **A3, atribución de Shapley exacta** del IC medio del conjunto entre sus componentes.
  - El placebo (la señal barajada) debe salir en cero.
  - Hay dos juegos: la media de rangos y ridge_cov_active.
  - Se calcula sesión a sesión, con su t HAC.
- **A4, costes**, solo en lo operable.
  - Carteras largas y cortas de los deciles extremos, con pesos iguales, de S1, S2 y cada modelo solo.
  - Se informan la rotación diaria, el coste de equilibrio por unidad negociada, las medidas de
    riesgo y el Sharpe neto a 1, 2 y 3 veces el coste base.

Nada se elige en evaluación. Las entradas, la espera y la publicación son las de
`organ_aggregation`.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm, rankdata

from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots

KIND = "posttraining_agent_core_v1"
# Huella canónica de configs/experiments/posttraining-agent-core.json, la declaración del núcleo.
REGISTERED_SHA256 = "7019689c04b8c4ecaf8a09d4d691f1de24dc02f60d12b552e6cda4a60826a6ad"
# Variantes registradas: la misma configuración con otros nombres de modelo. La prueba comprueba
# campo a campo que solo cambian el padre, los árboles, el órgano y la descripción.
VARIANT_SHA256 = {
    # Brazo v3 de H9 (rasgos long_horizon_v3): los modelos llevan el sufijo _v3.
    "6587e4877fa0659a2c16f837952a51921195ed8e3b681240aa7c2db973cf642c": (
        "posttraining-agent-core-v3.json"
    ),
}
SPLITS = og.SPLITS
AGGREGATED = og.AGGREGATED
QUANTILES = ("quantile_0025", "quantile_0100", "quantile_0500", "quantile_0900", "quantile_0975")
PLAYERS = ("parent", "tree", "signal", "placebo")
GAMMA = 0.5772156649015329


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_declaration(document):
    """La declaración registrada del núcleo y su huella, o un error que dice por qué no lo es."""
    _require(isinstance(document, dict), "La declaración del núcleo es un objeto JSON")
    digest = hashlib.sha256(pilots._canonical(document).encode()).hexdigest()
    _require(
        digest == REGISTERED_SHA256 or digest in VARIANT_SHA256,
        f"La declaración no es la registrada para el núcleo (huella {digest[:12]}, registrada "
        f"{REGISTERED_SHA256[:12]} o una de sus variantes con otros nombres de modelo): la "
        "configuración de las medidas no cambia",
    )
    grid = document["aggregator"]
    _require(
        tuple(grid["lambdas"]) == tuple(oa.RIDGE_LAMBDAS)
        and tuple(grid["half_lives"]) == tuple(oa.RIDGE_HALF_LIVES),
        "El agregador del código no usa la rejilla de ridge_cov_active declarada",
    )
    return document, digest


# ---------------------------------------------------------------------------------------------
# Piezas puras: sesiones, rangos e IC


def bounds_of(sessions):
    """Inicio y fin de cada sesión de un tramo alineado."""
    return [(int(start), int(end)) for _, _, start, end in sessions]


def session_z(values, bounds):
    """Rangos medios tipificados en cada sesión: media 0 y desviación 1 (0 si no hay orden)."""
    result = np.zeros(len(values), dtype=np.float64)
    for start, end in bounds:
        ranks = rankdata(values[start:end])
        spread = ranks.std()
        if spread > 0:
            result[start:end] = (ranks - ranks.mean()) / spread
    return result


def rank_ic(prediction, target):
    """Correlación de rangos medios; NaN si alguno de los dos no tiene orden."""
    left, right = rankdata(prediction), rankdata(target)
    left, right = left - left.mean(), right - right.mean()
    scale = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right) / scale if scale > 0 else math.nan


def session_ics(prediction, target, bounds, keys, *, min_assets, keep=None):
    """IC de rangos de cada sesión con al menos `min_assets` filas conservadas: {clave: IC}."""
    result = {}
    for key, (start, end) in zip(keys, bounds, strict=True):
        mask = slice(start, end) if keep is None else np.flatnonzero(keep[start:end]) + start
        count = end - start if keep is None else len(mask)
        if count < min_assets:
            continue
        value = rank_ic(prediction[mask], target[mask])
        if math.isfinite(value):
            result[key] = value
    return result


def _mean(values):
    return float(np.mean(list(values.values()))) if values else None


# ---------------------------------------------------------------------------------------------
# A1: abstención selectiva


def asset_confidence(parent, tree, bounds):
    """Confianza por activo: menos la discrepancia de los rangos tipificados de las dos familias."""
    return -np.abs(session_z(parent, bounds) - session_z(tree, bounds))


def zeroed(score_z, keep):
    """La predicción tipificada con los abstenidos en el centro de la sesión (cero)."""
    return np.where(keep, score_z, 0.0)


def choose_asset_coverage(confidence, score_z, target, bounds, keys, coverages, *, min_assets):
    """Cobertura y umbral elegidos en calibración por el IC medio con los abstenidos a cero.

    El umbral es el cuantil de la confianza de calibración que deja esa cobertura. En un empate
    gana la cobertura mayor. Solo mira el tramo que recibe.
    """
    best = None
    for coverage in sorted(coverages):
        threshold = -np.inf if coverage >= 1 else float(np.quantile(confidence, 1 - coverage))
        keep = confidence >= threshold
        value = _mean(
            session_ics(zeroed(score_z, keep), target, bounds, keys, min_assets=min_assets)
        )
        if best is None or (value is not None and value >= best[2]):
            best = (coverage, threshold, value)
    return dict(coverage=best[0], threshold=best[1], calibration_ic=best[2])


def equal_count_controls(score_z, keep, bounds, *, draws, seed):
    """Controles a igual cobertura en cada sesión: menor magnitud |z| y azar (varios sorteos)."""
    magnitude = np.ones(len(keep), dtype=bool)
    rng = np.random.default_rng(seed)
    random = np.ones((draws, len(keep)), dtype=bool)
    for start, end in bounds:
        dropped = int((~keep[start:end]).sum())
        if dropped == 0:
            continue
        order = np.argsort(np.abs(score_z[start:end]), kind="stable")
        magnitude[start + order[:dropped]] = False
        for draw in range(draws):
            random[draw, start + rng.permutation(end - start)[:dropped]] = False
    return magnitude, random


def _random_mean(score_z, target, bounds, keys, random, *, covered, min_assets):
    per_draw = []
    for keep in random:
        prediction = score_z if covered else zeroed(score_z, keep)
        per_draw.append(
            session_ics(
                prediction,
                target,
                bounds,
                keys,
                min_assets=min_assets,
                keep=keep if covered else None,
            )
        )
    common = set.intersection(*(set(d) for d in per_draw)) if per_draw else set()
    # En orden de sesión: el de un conjunto depende de la semilla de hash del proceso.
    return {key: float(np.mean([d[key] for d in per_draw])) for key in sorted(common)}


def asset_abstention(cal, ev, settings, tool, *, min_assets):
    """A1 por activo de un sistema en un modo: elección en calibración y medida en evaluación."""
    choice = choose_asset_coverage(
        cal["confidence"],
        cal["z"],
        cal["target"],
        cal["bounds"],
        cal["keys"],
        settings["coverages"],
        min_assets=min_assets,
    )
    keep = ev["confidence"] >= choice["threshold"]
    magnitude, random = equal_count_controls(
        ev["z"], keep, ev["bounds"], draws=settings["random_draws"], seed=settings["random_seed"]
    )
    args = (ev["target"], ev["bounds"], ev["keys"])
    base = session_ics(ev["z"], *args, min_assets=min_assets)
    series = dict(
        base=base,
        zeroed=session_ics(zeroed(ev["z"], keep), *args, min_assets=min_assets),
        zeroed_magnitude=session_ics(zeroed(ev["z"], magnitude), *args, min_assets=min_assets),
        zeroed_random=_random_mean(ev["z"], *args, random, covered=False, min_assets=min_assets),
        covered=session_ics(ev["z"], *args, min_assets=min_assets, keep=keep),
        covered_magnitude=session_ics(ev["z"], *args, min_assets=min_assets, keep=magnitude),
        covered_random=_random_mean(ev["z"], *args, random, covered=True, min_assets=min_assets),
    )
    contrasts = dict(
        zeroed_vs_base=tool.paired(series["zeroed"], base),
        zeroed_vs_magnitude=tool.paired(series["zeroed"], series["zeroed_magnitude"]),
        zeroed_vs_random=tool.paired(series["zeroed"], series["zeroed_random"]),
        covered_vs_base=tool.paired(series["covered"], base),
        covered_vs_magnitude=tool.paired(series["covered"], series["covered_magnitude"]),
        covered_vs_random=tool.paired(series["covered"], series["covered_random"]),
    )
    curve = []
    for coverage in sorted(settings["coverages"]):
        threshold = (
            -np.inf if coverage >= 1 else float(np.quantile(cal["confidence"], 1 - coverage))
        )
        kept = ev["confidence"] >= threshold
        mag, _ = equal_count_controls(ev["z"], kept, ev["bounds"], draws=0, seed=0)
        curve.append(
            dict(
                coverage=coverage,
                realized=float(kept.mean()),
                covered=_mean(session_ics(ev["z"], *args, min_assets=min_assets, keep=kept)),
                zeroed=_mean(session_ics(zeroed(ev["z"], kept), *args, min_assets=min_assets)),
                covered_magnitude=_mean(
                    session_ics(ev["z"], *args, min_assets=min_assets, keep=mag)
                ),
            )
        )
    terciles = []
    groups = within_session_groups(ev["liquidity"], ev["bounds"], 3)
    for group in range(3):
        member = groups == group
        terciles.append(
            dict(
                group=group,
                coverage=float(keep[member].mean()) if member.any() else None,
                covered=_mean(
                    session_ics(ev["z"], *args, min_assets=min_assets, keep=member & keep)
                ),
                all=_mean(session_ics(ev["z"], *args, min_assets=min_assets, keep=member)),
            )
        )
    summary = dict(
        choice=choice,
        realized_coverage=float(keep.mean()),
        means={name: _mean(values) for name, values in series.items()},
        contrasts=contrasts,
        curve=curve,
        liquidity_terciles=terciles,
    )
    return summary, series


def session_confidence(parent, tree, bounds):
    """Confianza por sesión: la correlación de rangos de las dos familias en la sesión."""
    return np.array([rank_ic(parent[s:e], tree[s:e]) for s, e in bounds])


def session_abstention(cal, ev, settings, tool, *, min_assets):
    """A1 por sesión: abstenerse en las sesiones donde las dos familias menos coinciden."""

    def ics(view):
        return np.array([rank_ic(view["z"][s:e], view["target"][s:e]) for s, e in view["bounds"]])

    cal_ic, ev_ic = ics(cal), ics(ev)
    best = None
    for coverage in sorted(settings["coverages"]):
        threshold = (
            -np.inf
            if coverage >= 1
            else float(np.quantile(cal["session_confidence"], 1 - coverage))
        )
        value = float(np.mean(np.where(cal["session_confidence"] >= threshold, cal_ic, 0.0)))
        if best is None or value >= best[2]:
            best = (coverage, threshold, value)
    keep = ev["session_confidence"] >= best[1]
    dropped = int((~keep).sum())
    magnitude = np.ones(len(keep), dtype=bool)
    magnitude[np.argsort(ev["session_spread"], kind="stable")[:dropped]] = False
    realized = float(keep.mean())
    keys = ev["keys"]
    series = dict(
        base=dict(zip(keys, ev_ic, strict=True)),
        zeroed=dict(zip(keys, np.where(keep, ev_ic, 0.0), strict=True)),
        zeroed_magnitude=dict(zip(keys, np.where(magnitude, ev_ic, 0.0), strict=True)),
        zeroed_random=dict(zip(keys, realized * ev_ic, strict=True)),
    )
    covered = [float(v) for v, k in zip(ev_ic, keep, strict=True) if k]
    covered_magnitude = [float(v) for v, k in zip(ev_ic, magnitude, strict=True) if k]
    summary = dict(
        choice=dict(coverage=best[0], threshold=best[1], calibration_ic=best[2]),
        realized_coverage=realized,
        means=dict(
            base=float(ev_ic.mean()),
            zeroed=float(np.mean(np.where(keep, ev_ic, 0.0))),
            zeroed_magnitude=float(np.mean(np.where(magnitude, ev_ic, 0.0))),
            zeroed_random=float(realized * ev_ic.mean()),
            covered=float(np.mean(covered)) if covered else None,
            covered_magnitude=float(np.mean(covered_magnitude)) if covered_magnitude else None,
        ),
        contrasts=dict(
            zeroed_vs_base=tool.paired(series["zeroed"], series["base"]),
            zeroed_vs_magnitude=tool.paired(series["zeroed"], series["zeroed_magnitude"]),
            zeroed_vs_random=tool.paired(series["zeroed"], series["zeroed_random"]),
        ),
    )
    return summary, series


def within_session_groups(values, bounds, groups):
    """Grupo de cada fila por cuantiles de `values` dentro de su sesión (-1 si no es finito)."""
    result = np.full(len(values), -1, dtype=np.int64)
    for start, end in bounds:
        segment = values[start:end]
        finite = np.isfinite(segment)
        if finite.sum() < groups:
            continue
        ranks = rankdata(segment[finite]) - 1
        result[start + np.flatnonzero(finite)] = np.minimum(
            (ranks * groups / finite.sum()).astype(np.int64), groups - 1
        )
    return result


# ---------------------------------------------------------------------------------------------
# A2: intervalos conformes


def conformal_correction(scores, level):
    """Cuantil de muestra finita ⌈(n + 1)·nivel⌉/n de las puntuaciones de conformidad."""
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    rank = math.ceil((len(scores) + 1) * level)
    return math.inf if rank > len(scores) else float(scores[rank - 1])


def conformity(lower, upper, target):
    return np.maximum(lower - target, target - upper)


def interval_summary(lower, upper, target, groups=None):
    """Cobertura y anchura media y mediana, global y por grupo."""
    inside = (target >= lower) & (target <= upper)
    width = upper - lower
    result = dict(
        coverage=float(inside.mean()),
        width=float(width.mean()),
        median_width=float(np.median(width)),
    )
    if groups is not None:
        result["groups"] = {
            str(name): dict(
                coverage=float(inside[groups == name].mean()),
                width=float(width[groups == name].mean()),
            )
            for name in np.unique(groups)
            if (groups == name).any() and not (isinstance(name, (int, np.integer)) and name < 0)
        }
    return result


def quantile_lines(features, target, levels, *, max_rows, seed):
    """Regresiones cuantílicas lineales de la etiqueta, una por cuantil de cada nivel."""
    from statsmodels.regression.quantile_regression import QuantReg

    rows = np.arange(len(target))
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=max_rows, replace=False))
    coefficients = {}
    for level in levels:
        for tau in ((1 - level) / 2, 1 - (1 - level) / 2):
            fit = QuantReg(target[rows], features[rows]).fit(q=tau, max_iter=5000)
            coefficients[round(tau, 4)] = np.asarray(fit.params, dtype=np.float64)
    return coefficients


def combined_features(score_z, width):
    return np.column_stack([np.ones(len(width)), score_z * width, width])


# ---------------------------------------------------------------------------------------------
# A3: Shapley


def shapley(players, values):
    """Valores de Shapley exactos de un juego dado por {coalición: valor}; ∅ vale cero."""
    n = len(players)
    result = {}
    for player in players:
        others = [p for p in players if p != player]
        total = 0.0
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in itertools.combinations(others, size):
                coalition = frozenset(subset)
                total = total + weight * (values[coalition | {player}] - values[coalition])
        result[player] = total
    return result


def shapley_by_orderings(players, values):
    """Los mismos valores como media de las contribuciones marginales en todos los órdenes."""
    totals = {p: 0.0 for p in players}
    orders = list(itertools.permutations(players))
    for order in orders:
        coalition = frozenset()
        for player in order:
            totals[player] = totals[player] + values[coalition | {player}] - values[coalition]
            coalition = coalition | {player}
    return {p: total / len(orders) for p, total in totals.items()}


def session_game(per_coalition, players):
    """Juego sesión a sesión sobre las sesiones comunes: {coalición: vector}, con ∅ = 0."""
    keys = sorted(set.intersection(*(set(v) for v in per_coalition.values())))
    values = {c: np.array([per_coalition[c][k] for k in keys]) for c in per_coalition}
    values[frozenset()] = np.zeros(len(keys))
    _require(
        all(
            frozenset(c) in values
            for r in range(1, len(players) + 1)
            for c in itertools.combinations(players, r)
        ),
        "Falta el valor de alguna coalición del juego de Shapley",
    )
    return keys, values


# ---------------------------------------------------------------------------------------------
# A4: costes


def tail_book(score, assets, bounds, tail):
    """Pesos de la cartera larga y corta de las colas de cada sesión: lista de {activo: peso}."""
    books = []
    for start, end in bounds:
        count = end - start
        size = max(1, int(math.floor(count * tail)))
        order = np.argsort(score[start:end], kind="stable")
        book = {assets[start + i]: -1.0 / size for i in order[:size]}
        book.update({assets[start + i]: 1.0 / size for i in order[count - size :]})
        books.append(book)
    return books


def book_series(books, returns_by_session):
    """Rendimiento de cada sesión y rotación respecto a la anterior (la primera no cuenta)."""
    gross, turnover = [], []
    previous = None
    for book, returns in zip(books, returns_by_session, strict=True):
        gross.append(sum(weight * returns[asset] for asset, weight in book.items()))
        if previous is None:
            turnover.append(math.nan)
        else:
            assets = set(book) | set(previous)
            turnover.append(sum(abs(book.get(a, 0.0) - previous.get(a, 0.0)) for a in assets))
        previous = book
    return np.asarray(gross), np.asarray(turnover)


def risk_metrics(daily, *, annualization, cvar):
    """Sharpe, Sortino, caída máxima de la suma acumulada, CVaR y aciertos de una serie diaria."""
    daily = np.asarray(daily, dtype=np.float64)
    mean, std = float(daily.mean()), float(daily.std(ddof=1))
    downside = daily[daily < 0]
    down = float(np.sqrt(np.mean(downside**2))) if len(downside) else 0.0
    cumulative = np.cumsum(daily)
    drawdown = float(np.max(np.maximum.accumulate(np.r_[0.0, cumulative])[1:] - cumulative))
    worst = np.sort(daily)[: max(1, int(math.ceil(len(daily) * cvar)))]
    return dict(
        mean=mean,
        std=std,
        sharpe=mean / std * math.sqrt(annualization) if std > 0 else None,
        sortino=mean / down * math.sqrt(annualization) if down > 0 else None,
        max_drawdown=drawdown,
        cvar=float(worst.mean()),
        hit_rate=float((daily > 0).mean()),
    )


def deflated_sharpe(daily, sharpes, *, trials):
    """Sharpe deflactado (Bailey y López de Prado) de una serie entre `trials` ensayos."""
    daily = np.asarray(daily, dtype=np.float64)
    sr = daily.mean() / daily.std(ddof=1)
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    expected = (
        math.sqrt(variance)
        * ((1 - GAMMA) * norm.ppf(1 - 1 / trials) + GAMMA * norm.ppf(1 - 1 / (trials * math.e)))
        if trials > 1
        else 0.0
    )
    centered = daily - daily.mean()
    skew = float(np.mean(centered**3) / np.mean(centered**2) ** 1.5)
    kurtosis = float(np.mean(centered**4) / np.mean(centered**2) ** 2)
    denominator = math.sqrt(max(1e-12, 1 - skew * sr + (kurtosis - 1) / 4 * sr**2))
    return float(norm.cdf((sr - expected) * math.sqrt(len(daily) - 1) / denominator))


# ---------------------------------------------------------------------------------------------
# Covariables conocidas al decidir


def known_rows(state, origin, tables, view):
    """Alfa, liquidez (log del volumen en dólares de 21 sesiones), volatilidad (log de la de 21
    sesiones) y madurez exacta de la vista de cada fila, por tramo y en el orden de las tablas.

    Los rasgos son los de la criba (`technical_features`) del día de la decisión, como el alfa
    conocido de `KnownOffset`; cada activo se calcula una sola vez para todos los tramos.
    """
    tool = og._screening()
    og.checked_factors(
        Path(state) / "cohorts" / str(origin) / "manifest.json", og.markets_of(tables)
    )
    base = tool.KnownOffset(Path(state), origin)
    names = base.module.NAMES
    columns = {
        split: [np.full(t.num_rows, np.nan) for _ in range(3)] for split, t in tables.items()
    }
    places = {}
    for split, table in tables.items():
        assets = table.column("asset_id").to_pylist()
        instants = table.column("prediction_at").cast(pa.int64()).to_numpy()
        for row, asset in enumerate(assets):
            places.setdefault(asset, []).append((split, row, instants[row]))
    for asset, rows in places.items():
        market, symbol = asset.split("/", 1)
        prices, days = base._table(base.prepared / market / symbol / "prices.parquet")
        features = base.module.technical_features(
            prices, days, *base.factors.get(market, (None, None))
        )
        known = features[:, names.index("has_market_model")] > 0
        values = (
            np.where(
                known, features[:, names.index("alpha_percent")].astype(np.float64) / 100, np.nan
            ),
            features[:, names.index("log_dollar_volume_21")].astype(np.float64),
            (
                features[:, names.index("log_volatility_252")]
                + features[:, names.index("volatility_21_over_252")]
            ).astype(np.float64),
        )
        wanted = np.array([r[2] for r in rows]).astype("datetime64[us]").astype("datetime64[D]")
        position = np.searchsorted(days, wanted)
        inside = position < len(days)
        inside[inside] = days[position[inside]] == wanted[inside]
        for (split, row, _), found, at in zip(rows, inside, position, strict=True):
            if found:
                for column, value in zip(columns[split], values, strict=True):
                    column[row] = value[at]
    maturity = pilots._maturity(Path(state) / "cohorts" / str(origin) / "manifest.json", view)
    matures = {}
    for split, table in tables.items():
        _require(split in maturity, f"La vista {view[:12]} no da la madurez de {split}")
        rows = maturity[split]
        lookup = dict(
            zip(
                zip(
                    rows.column("asset_id").to_pylist(),
                    rows.column("prediction_at").cast(pa.int64()).to_pylist(),
                    strict=True,
                ),
                rows.column("matures_at").cast(pa.int64()).to_pylist(),
                strict=True,
            )
        )
        keys = zip(
            table.column("asset_id").to_pylist(),
            table.column("prediction_at").cast(pa.int64()).to_pylist(),
            strict=True,
        )
        values = [lookup.get(key) for key in keys]
        _require(
            all(v is not None for v in values), f"Hay filas de {split} sin madurez en la vista"
        )
        matures[split] = np.asarray(values, dtype=np.int64)
    alpha = {s: c[0] for s, c in columns.items()}
    liquidity = {s: c[1] for s, c in columns.items()}
    volatility = {s: c[2] for s, c in columns.items()}
    # Oráculo: el alfa coincide con el de KnownOffset en una muestra fija de activos.
    for asset in sorted(places)[:: max(1, len(places) // 25)]:
        for split, table in tables.items():
            rows = [r for s, r, _ in places[asset] if s == split]
            if not rows:
                continue
            instants = table.column("prediction_at").cast(pa.int64()).to_numpy()[rows]
            expected = base.known(asset, instants)[0]
            _require(
                np.array_equal(alpha[split][rows], expected, equal_nan=True),
                f"El alfa de {asset} en {split} no es el de KnownOffset",
            )
    return alpha, liquidity, volatility, matures


# ---------------------------------------------------------------------------------------------
# Orden por origen


def _attention(declaration, state, origin):
    folder = (
        Path(state)
        / "runs"
        / declaration["attention"]
        / str(origin)
        / f"seed-{declaration['seed']}"
    )
    try:
        status = json.loads((folder / "run.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None
    ready = status == "completed" and all(
        og._readable(folder / f"{s}-predictions.parquet") for s in SPLITS
    )
    return folder if ready else None


def _sessions_of(table):
    market = np.asarray(table.column("market").to_pylist())
    instant = table.column("prediction_at").cast(pa.int64()).to_numpy()
    change = np.r_[True, (market[1:] != market[:-1]) | (instant[1:] != instant[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(instant)]
    return [(market[s], int(instant[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _view(aligned, prediction, positions, covariates, index):
    """Lo que A1 y A4 necesitan de un tramo alineado de un modo."""
    sessions = aligned["sessions"]
    bounds = bounds_of(sessions)
    rows = np.array([index[s] for s in aligned["sample_id"]], dtype=np.int64)
    parent = aligned["predictions"][:, positions["parent"]].astype(np.float64)
    tree = aligned["predictions"][:, positions["tree"]].astype(np.float64)
    return dict(
        bounds=bounds,
        keys=[(m, int(t)) for m, t, _, _ in sessions],
        target=np.asarray(aligned["target"], dtype=np.float64),
        z=session_z(np.asarray(prediction, dtype=np.float64), bounds),
        score=np.asarray(prediction, dtype=np.float64),
        confidence=asset_confidence(parent, tree, bounds),
        session_confidence=session_confidence(parent, tree, bounds),
        session_spread=np.array([parent[s:e].std() for s, e in bounds]),
        liquidity=covariates["liquidity"][rows],
        assets=list(aligned["asset_id"]),
        predictions=aligned["predictions"],
    )


def _series_table(rows):
    columns = {}
    for row in rows:
        for name in row:
            columns.setdefault(name, [])
    for row in rows:
        for name in columns:
            columns[name].append(row.get(name))
    return pa.table(columns)


def _a1(declaration, tool, views, modes):
    settings, min_assets = declaration["abstention"], declaration["min_assets"]
    summary, rows = {}, []
    for mode in modes:
        for system in ("s1", "s2"):
            cal, ev = views[(mode, system, "calibration")], views[(mode, system, "evaluation")]
            asset, asset_series = asset_abstention(cal, ev, settings, tool, min_assets=min_assets)
            session, session_series = session_abstention(
                cal, ev, settings, tool, min_assets=min_assets
            )
            summary[f"{mode}/{system}"] = dict(asset=asset, session=session)
            for kind, series in (("asset", asset_series), ("session", session_series)):
                for name, values in series.items():
                    rows += [
                        dict(
                            mode=mode,
                            system=system,
                            kind=kind,
                            series=name,
                            market=str(k[0]),
                            prediction_at=int(k[1]),
                            value=float(v),
                        )
                        for k, v in values.items()
                    ]
    return summary, rows


def _a2(declaration, parent, tree, covariates):
    settings = declaration["conformal"]
    scores, widths, groups = {}, {}, {}
    for split in SPLITS:
        sessions = _sessions_of(parent[split])
        bounds = bounds_of(sessions)
        own = parent[split].column("prediction").to_numpy().astype(np.float64)
        other = tree[split].column("prediction").to_numpy().astype(np.float64)
        scores[split] = (session_z(own, bounds) + session_z(other, bounds)) / 2
        widths[split] = parent[split].column("quantile_0900").to_numpy().astype(
            np.float64
        ) - parent[split].column("quantile_0100").to_numpy().astype(np.float64)
        groups[split] = dict(
            liquidity=within_session_groups(
                covariates[split]["liquidity"], bounds, settings["liquidity_groups"]
            ),
            volatility=within_session_groups(
                covariates[split]["volatility"], bounds, settings["volatility_groups"]
            ),
            quarter=np.array(
                [
                    f"{d.astype('datetime64[Y]').astype(int) + 1970}T{(d.astype('datetime64[M]').astype(int) % 12) // 3 + 1}"
                    for d in parent[split]
                    .column("prediction_at")
                    .cast(pa.int64())
                    .to_numpy()
                    .astype("datetime64[us]")
                ]
            ),
        )
    target = {s: parent[s].column("target").to_numpy().astype(np.float64) for s in SPLITS}
    fit, conformal = settings["fit_split"], settings["conformal_split"]
    coefficients = quantile_lines(
        combined_features(scores[fit], widths[fit]),
        target[fit],
        settings["levels"],
        max_rows=settings["max_fit_rows"],
        seed=settings["fit_seed"],
    )
    result = dict(
        crossing_core=float((widths["evaluation"] <= 0).mean()),
        coefficients={str(k): v.tolist() for k, v in coefficients.items()},
        levels={},
    )
    for level in settings["levels"]:
        low_name, high_name = settings["core"][str(level)]
        low_tau, high_tau = round((1 - level) / 2, 4), round(1 - (1 - level) / 2, 4)
        bands = {}
        for split in (conformal, "evaluation"):
            core = (
                parent[split].column(low_name).to_numpy().astype(np.float64),
                parent[split].column(high_name).to_numpy().astype(np.float64),
            )
            features = combined_features(scores[split], widths[split])
            combined = (features @ coefficients[low_tau], features @ coefficients[high_tau])
            bands[split] = dict(core=core, combined=combined)
        corrections = {
            name: conformal_correction(
                conformity(*bands[conformal][name], target[conformal]), level
            )
            for name in ("core", "combined")
        }
        evaluation = bands["evaluation"]
        systems = dict(
            core_raw=evaluation["core"],
            core_cqr=(
                evaluation["core"][0] - corrections["core"],
                evaluation["core"][1] + corrections["core"],
            ),
            combined_qr=evaluation["combined"],
            combined_cqr=(
                evaluation["combined"][0] - corrections["combined"],
                evaluation["combined"][1] + corrections["combined"],
            ),
        )
        result["levels"][str(level)] = dict(
            corrections=corrections,
            systems={
                name: dict(
                    **interval_summary(low, high, target["evaluation"]),
                    liquidity=interval_summary(
                        low, high, target["evaluation"], groups["evaluation"]["liquidity"]
                    )["groups"],
                    volatility=interval_summary(
                        low, high, target["evaluation"], groups["evaluation"]["volatility"]
                    )["groups"],
                    quarter=interval_summary(
                        low, high, target["evaluation"], groups["evaluation"]["quarter"]
                    )["groups"],
                )
                for name, (low, high) in systems.items()
            },
        )
    return result


def _a3(tool, games, players, modes):
    summary, rows = {}, []
    for mode in modes:
        for game, per_coalition in games[mode].items():
            keys, values = session_game(per_coalition, players)
            phi = shapley(players, values)
            check = shapley_by_orderings(players, values)
            _require(
                all(np.allclose(phi[p], check[p], rtol=0, atol=1e-12) for p in players),
                "El Shapley exacto no coincide con la media de los órdenes",
            )
            whole = values[frozenset(players)]
            _require(
                np.allclose(sum(phi.values()), whole, rtol=0, atol=1e-12),
                "El Shapley no es eficiente",
            )
            summary[f"{mode}/{game}"] = dict(
                sessions=len(keys),
                whole=float(whole.mean()),
                players={
                    p: dict(
                        mean=float(phi[p].mean()),
                        share=float(phi[p].mean() / whole.mean()) if whole.mean() else None,
                        t=tool.summary(dict(zip(keys, phi[p].tolist(), strict=True)))["t_nw"],
                    )
                    for p in players
                },
            )
            for p in players:
                rows += [
                    dict(
                        mode=mode,
                        game=game,
                        player=p,
                        market=str(k[0]),
                        prediction_at=int(k[1]),
                        value=float(v),
                    )
                    for k, v in zip(keys, phi[p], strict=True)
                ]
    return summary, rows


def cost_table(gross, turnover, settings):
    """Riesgo bruto y neto de una cartera en los mismos días: los que tienen rotación.

    El primer día no tiene cartera anterior; queda fuera del bruto y del neto, así que el
    Sharpe neto solo se separa del bruto por el coste.
    """
    days = np.isfinite(np.asarray(turnover, dtype=np.float64))
    gross = np.asarray(gross, dtype=np.float64)[days]
    turnover = np.asarray(turnover, dtype=np.float64)[days]
    traded = float(turnover.mean())
    risk = risk_metrics(gross, annualization=settings["annualization"], cvar=settings["cvar"])
    breakeven = float(gross.mean() / traded) * 1e4 if traded > 0 else None
    net = {
        str(multiplier): risk_metrics(
            gross - settings["base_cost_bps"] * multiplier / 1e4 * turnover,
            annualization=settings["annualization"],
            cvar=settings["cvar"],
        )["sharpe"]
        for multiplier in settings["multipliers"]
    }
    return dict(
        risk=risk,
        mean_bps=risk["mean"] * 1e4,
        turnover=traded,
        breakeven_bps=breakeven,
        safety_margin=None if breakeven is None else breakeven / settings["base_cost_bps"],
        net_sharpe=net,
    )


def _a4(declaration, tool, views):
    settings = declaration["costs"]
    books, rows = {}, []
    sources = {
        "s1": views[("operable", "s1", "evaluation")],
        "s2": views[("operable", "s2", "evaluation")],
    }
    base = sources["s1"]
    for name, column in (("parent", "parent"), ("tree", "tree")):
        sources[name] = dict(
            base, score=base["predictions"][:, base["positions"][column]].astype(np.float64)
        )
    series = {}
    for name, view in sources.items():
        bounds, assets, target = view["bounds"], view["assets"], view["target"]
        returns = [{assets[i]: target[i] for i in range(s, e)} for s, e in bounds]
        book = tail_book(view["score"], assets, bounds, settings["tail"])
        gross, turnover = book_series(book, returns)
        series[name] = (view["keys"], gross, turnover, book, view)
    traded_days = {n: np.isfinite(series[n][2]) for n in series}
    sharpes = [
        series[n][1][traded_days[n]].mean() / series[n][1][traded_days[n]].std(ddof=1)
        for n in series
    ]
    for name, (keys, gross, turnover, book, view) in series.items():
        table = cost_table(gross, turnover, settings)
        deciles = within_session_groups(view["score"], view["bounds"], 10)
        by_decile = [
            float(
                np.mean(
                    [
                        view["target"][s:e][deciles[s:e] == d].mean()
                        for s, e in view["bounds"]
                        if (deciles[s:e] == d).any()
                    ]
                )
            )
            for d in range(10)
        ]
        liquidity = view["liquidity"]
        legs = [
            liquidity[s + np.flatnonzero([view["assets"][i] in b for i in range(s, e)])]
            for (s, e), b in zip(view["bounds"], book, strict=True)
        ]
        books[name] = dict(
            **table,
            deflated_sharpe=deflated_sharpe(gross[traded_days[name]], sharpes, trials=len(series)),
            deciles_bps=[v * 1e4 for v in by_decile],
            monotonicity=rank_ic(np.arange(10.0), np.array(by_decile)),
            median_log_dollar_volume=float(np.nanmedian(np.concatenate(legs))) if legs else None,
        )
        rows += [
            dict(
                book=name,
                market=str(k[0]),
                prediction_at=int(k[1]),
                gross=float(g),
                turnover=None if math.isnan(t) else float(t),
            )
            for k, g, t in zip(keys, gross, turnover, strict=True)
        ]
    daily = {
        name: dict(zip(series[name][0], series[name][1].tolist(), strict=True)) for name in series
    }
    contrasts = dict(
        s1_vs_parent=tool.paired(daily["s1"], daily["parent"]),
        s1_vs_tree=tool.paired(daily["s1"], daily["tree"]),
        s2_vs_s1=tool.paired(daily["s2"], daily["s1"]),
    )
    return dict(books=books, contrasts=contrasts), rows


def measure(
    declaration, state, reference, organs, origin, output, *, wait_hours, poll, log, markets=None
):
    """A1–A4 de un origen, publicados en `<output>/<origen>`; devuelve el resumen.

    `markets` limita la medida a esos mercados (`<output>/<origen>-<mercados>`). Un estado con
    varios mercados exige elegirlos: la lectura es por mercado.
    """
    declaration, digest = checked_declaration(declaration)
    sources = og.folders(declaration, state, reference, organs, origin)
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    missing = og.wait_for_inputs(
        declaration, sources, manifest, hours=wait_hours, poll=poll, say=log
    )
    if missing:
        raise og.MissingInputs(missing)
    attention = _attention(declaration, state, origin)
    if attention is not None:
        sources = dict(sources, attention=attention)
    fingerprint = og._fingerprint(sources, manifest)
    target = Path(output) / og.target_name(origin, markets)
    if target.exists():
        return og._already(target, digest, fingerprint, log)
    started = time.monotonic()

    def say(text):
        log(f"{time.monotonic() - started:6.0f} s {text}")

    records = {
        role: json.loads((folder / "run.json").read_text(encoding="utf-8"))
        for role, folder in sources.items()
    }
    og._checked_sources(declaration, origin, sources, records, fingerprint)
    if "attention" in records:
        _require(
            records["attention"].get("final_test_opened") is False,
            "La atención abrió el test final",
        )
    tool = og._screening()
    schema = pq.read_schema(sources["parent"] / "evaluation-predictions.parquet").names
    _require(
        all(q in schema for q in QUANTILES),
        f"{sources['parent']} no emite los cuantiles {QUANTILES}",
    )
    parent = {
        s: pq.read_table(
            sources["parent"] / f"{s}-predictions.parquet", columns=og.COLUMNS + list(QUANTILES)
        ).sort_by(og.ORDER)
        for s in SPLITS
    }
    tables = dict(
        tree={s: og._table(sources["tree"], s) for s in SPLITS},
        organ={s: og._table(sources["organ"], s) for s in AGGREGATED},
    )
    if "attention" in sources:
        tables["attention"] = {s: og._table(sources["attention"], s) for s in SPLITS}
    present = og.markets_of(parent)
    measured = og.measured_markets(present, markets)
    if measured != present:
        parent = {s: og.only(table, measured) for s, table in parent.items()}
        tables = {
            role: {s: og.only(table, measured) for s, table in splits.items()}
            for role, splits in tables.items()
        }
    say(f"mercados medidos: {measured} (el estado tiene {present})")
    for role, splits in tables.items():
        for split, table in splits.items():
            for name in og.KEYS:
                _require(
                    table.column(name).equals(parent[split].column(name)),
                    f"{sources[role]} no tiene las filas del padre en {split} (columna {name})",
                )
    view = records["parent"]["request"]["view_sha256"]
    alpha, liquidity, volatility, matures = known_rows(state, origin, parent, view)
    covariates = {s: dict(liquidity=liquidity[s], volatility=volatility[s]) for s in SPLITS}
    say("covariables conocidas y madurez")
    own = {s: parent[s].column("prediction").to_numpy().astype(np.float64) for s in SPLITS}
    signal = {
        s: tables["organ"][s].column("prediction").to_numpy().astype(np.float64) - own[s]
        for s in AGGREGATED
    }
    predictions = dict(
        parent=own,
        tree={
            s: tables["tree"][s].column("prediction").to_numpy().astype(np.float64) for s in SPLITS
        },
        signal=signal,
        placebo={
            s: og.shuffled_within_sessions(
                signal[s],
                np.asarray(parent[s].column("market").to_pylist()),
                parent[s].column("prediction_at").cast(pa.int64()).to_numpy(),
                declaration["placebo"]["seed"],
            )
            for s in AGGREGATED
        },
    )
    players = PLAYERS
    if "attention" in sources:
        predictions["attention"] = {
            s: tables["attention"][s].column("prediction").to_numpy().astype(np.float64)
            for s in SPLITS
        }
        players = (*PLAYERS, "attention")
    names = dict(
        parent=declaration["parent"],
        tree=declaration["tree"],
        signal=og.SIGNAL,
        placebo=og.PLACEBO,
        attention=declaration["attention"],
    )
    player_of = {names[p]: p for p in players}
    modes = tuple(declaration["modes"])
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        (work / "informes").mkdir(parents=True)
        inputs = {
            p: og._write_component(
                work / "entradas" / names[p], parent, predictions[p], alpha, matures
            )
            for p in players
        }
        coalitions = [
            frozenset(c)
            for r in range(2, len(players) + 1)
            for c in itertools.combinations(players, r)
        ]
        games = {mode: dict(rank_average={}, ridge_cov_active={}) for mode in modes}
        views = {}
        factors = {}
        for mode in modes:
            everyone, systems, rows_of = None, {}, {}
            for coalition in coalitions:
                members = [p for p in players if p in coalition]
                result = oa.run_aggregation(
                    {names[p]: inputs[p] for p in members},
                    splits=tuple(declaration["splits"]),
                    selection_split=declaration["selection_split"],
                    min_assets=declaration["min_assets"],
                    align=declaration["align"],
                    operable=mode == "operable",
                    alpha_col="alpha",
                    maturity_col="matures_at",
                    placebo_seed=declaration["aggregator_placebo_seed"],
                    operable_scales=tuple(declaration["operable_scales"]),
                )
                for game in games[mode]:
                    games[mode][game][coalition] = result["aggregators"][game][
                        "evaluation_session_ics"
                    ]
                if coalition == frozenset(("parent", "tree")):
                    systems["s1"] = result["aggregators"]["rank_average"]["split_predictions"]
                    rows_of["s1"] = result["aligned"]
                    (work / "informes" / f"{mode}-s1.md").write_text(
                        oa.render_markdown(oa._json_safe_report(result)), encoding="utf-8"
                    )
                if coalition == frozenset(("parent", "tree", "signal")):
                    systems["s2"] = result["aggregators"]["ridge_cov_active"]["split_predictions"]
                    rows_of["s2"] = result["aligned"]
                    (work / "informes" / f"{mode}-s2.md").write_text(
                        oa.render_markdown(oa._json_safe_report(result)), encoding="utf-8"
                    )
                if coalition == frozenset(players):
                    everyone = result["aligned"]
                    factors[mode] = (result["operable"] or {}).get("scales")
            say(f"{mode}: {len(coalitions)} conjuntos agregados")
            for system, aligned in rows_of.items():
                for split in AGGREGATED:
                    _require(
                        list(aligned[split]["sample_id"]) == list(everyone[split]["sample_id"])
                        and len(systems[system][split]) == len(everyone[split]["sample_id"]),
                        f"{system} en {mode}/{split} no tiene las filas alineadas de todos los "
                        "componentes",
                    )
            positions = {
                player_of[n]: i for i, n in enumerate(everyone["evaluation"]["model_names"])
            }
            for split in AGGREGATED:
                ev = everyone[split]
                for p in players:
                    if split == "evaluation":
                        ics = oa.session_rank_ic(
                            ev,
                            ev["predictions"][:, positions[p]],
                            min_assets=declaration["min_assets"],
                        )
                        for game in games[mode]:
                            games[mode][game][frozenset((p,))] = ics
                index = {
                    sample: i
                    for i, sample in enumerate(parent[split].column("sample_id").to_pylist())
                }
                for system, split_predictions in systems.items():
                    item = _view(ev, split_predictions[split], positions, covariates[split], index)
                    item["positions"] = positions
                    views[(mode, system, split)] = item
        a1, a1_rows = _a1(declaration, tool, views, modes)
        say("A1 abstención")
        a2 = _a2(declaration, parent, tables["tree"], covariates)
        say("A2 intervalos conformes")
        a3, a3_rows = _a3(tool, games, players, modes)
        say("A3 Shapley")
        a4, a4_rows = _a4(declaration, tool, views) if "operable" in modes else ({}, [])
        say("A4 costes")
        shutil.rmtree(work / "entradas")
        summary = og._plain(
            dict(
                schema_version=1,
                kind=KIND,
                status="completed",
                origin=origin,
                declaration_sha256=digest,
                code=og._code_identity(),
                inputs=fingerprint,
                players=list(players),
                markets=measured,
                operable_factors=factors,
                alpha_known={s: float(np.isfinite(alpha[s]).mean()) for s in SPLITS},
                a1=a1,
                a2=a2,
                a3=a3,
                a4=a4,
                seconds=round(time.monotonic() - started),
            )
        )
        pq.write_table(_series_table(a1_rows), work / "a1-sesiones.parquet")
        pq.write_table(_series_table(a3_rows), work / "a3-sesiones.parquet")
        pq.write_table(_series_table(a4_rows), work / "a4-dias.parquet")
        (work / "resumen.md").write_text(markdown(summary), encoding="utf-8")
        (work / "resumen.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.rename(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say(f"publicado en {target}")
    return summary


def _n(value, digits=4):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}".replace(".", ",").replace("-", "−")


def _c(entry):
    if not entry or entry.get("mean") is None:
        return "—"
    return f"{_n(entry['mean'])} (t {_n(entry['t'], 1)})"


def markdown(summary):
    """Resumen legible de A1–A4 de un origen."""
    lines = [
        f"# Núcleo agéntico del origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration_sha256'][:12]}; componentes {', '.join(summary['players'])}. "
        "Diferencias de evaluación pareadas por sesión, con t HAC.",
        "",
        "## A1, abstención selectiva",
        "",
        "| Modo / sistema | Por | Cobertura elegida (real) | IC base | Con abstenidos a cero − base | − control de magnitud | − azar | Cubierto − base |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, item in summary["a1"].items():
        for kind in ("asset", "session"):
            part = item[kind]
            lines.append(
                f"| {key} | {'activo' if kind == 'asset' else 'sesión'} | {part['choice']['coverage']:.1f} ({part['realized_coverage']:.2f}) | "
                f"{_n(part['means']['base'])} | {_c(part['contrasts']['zeroed_vs_base'])} | {_c(part['contrasts']['zeroed_vs_magnitude'])} | "
                f"{_c(part['contrasts']['zeroed_vs_random'])} | {_c(part['contrasts'].get('covered_vs_base'))} |"
            )
    lines += [
        "",
        "## A2, intervalos conformes (etiqueta, evaluación)",
        "",
        "| Nivel | Sistema | Cobertura | Anchura media | Cobertura por tercio de liquidez | Peor decil de volatilidad |",
        "|---|---|---|---|---|---|",
    ]
    for level, item in summary["a2"]["levels"].items():
        for name, part in item["systems"].items():
            terciles = ", ".join(f"{v['coverage']:.3f}" for v in part["liquidity"].values())
            worst = (
                min(part["volatility"].values(), key=lambda v: v["coverage"])["coverage"]
                if part["volatility"]
                else None
            )
            lines.append(
                f"| {level} | {name} | {part['coverage']:.3f} | {part['width']:.4f} | {terciles} | {worst:.3f} |"
            )
    lines += [
        "",
        "## A3, Shapley del IC medio de evaluación",
        "",
        "| Modo / juego | IC del conjunto | " + " | ".join(summary["players"]) + " |",
        "|---|---|" + "---|" * len(summary["players"]),
    ]
    for key, item in summary["a3"].items():
        lines.append(
            f"| {key} | {_n(item['whole'])} | "
            + " | ".join(
                f"{_n(item['players'][p]['mean'])} (t {_n(item['players'][p]['t'], 1)})"
                for p in summary["players"]
            )
            + " |"
        )
    if summary["a4"]:
        lines += [
            "",
            "## A4, colas en lo operable (evaluación, por sesión)",
            "",
            "| Cartera | Media (pb) | Sharpe | Rotación | Equilibrio (pb) | Sharpe neto 1×/2×/3× | DSR | Caída máx. |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, book in summary["a4"]["books"].items():
            net = "/".join(_n(v, 2) if v is not None else "—" for v in book["net_sharpe"].values())
            lines.append(
                f"| {name} | {_n(book['mean_bps'], 2)} | {_n(book['risk']['sharpe'], 2)} | {book['turnover']:.3f} | "
                f"{_n(book['breakeven_bps'], 2)} | {net} | {book['deflated_sharpe']:.3f} | {book['risk']['max_drawdown']:.4f} |"
            )
        lines.append("")
        lines.append(
            "Diferencias diarias: "
            + "; ".join(f"{k} {_c(v)}" for k, v in summary["a4"]["contrasts"].items())
            + "."
        )
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("medir", help="Calcular A1–A4 de un origen")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    run.add_argument("--reference", type=Path, required=True, help="Raíz de los árboles")
    run.add_argument("--organs", type=Path, required=True, help="Raíz de los órganos")
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--output", type=Path, required=True, help="Raíz propia del núcleo")
    run.add_argument("--wait-hours", type=float, default=0.0)
    run.add_argument("--poll-seconds", type=float, default=300.0)
    run.add_argument(
        "--market",
        action="append",
        dest="markets",
        metavar="MERCADO",
        help="Medir solo este mercado (repetible); obligatorio en un estado con varios",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)

    def log(line):
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {line}", flush=True)

    try:
        declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
        measure(
            declaration,
            args.state,
            args.reference,
            args.organs,
            args.origin,
            args.output,
            wait_hours=args.wait_hours,
            poll=args.poll_seconds,
            log=log,
            markets=args.markets,
        )
    except og.MissingInputs as missing:
        print(
            f"Faltan entradas del núcleo para el origen {args.origin} tras esperar "
            f"{args.wait_hours:g} h; no se escribe nada:",
            file=sys.stderr,
        )
        for item in missing.items:
            print(f"- {item}", file=sys.stderr)
        return og.MISSING_EXIT
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
