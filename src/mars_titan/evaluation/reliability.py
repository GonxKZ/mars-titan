# ruff: noqa: E501
"""F, fiabilidad de las emisiones: F1 (las lecturas de R-1 como evaluación) y F2 (confianza direccional).

Encargo del coordinador del 23 de noviembre de 2026 (línea F, «más del 90 % de precisión y
fiabilidad»). Es una evaluación descriptiva y secundaria, fuera de la familia primaria del registro
confirmatorio, con la configuración fijada en `configs/experiments/posttraining-reliability.json`,
cuya huella exige la orden. Mide cuatro predictores de un origen: S1, S2, los árboles y el padre.
S1 y S2 se construyen como en la orden de S2: las mismas entradas, la misma alineación y el mismo
factor operable k, elegido en calibración. En lo operable va además un control: el alfa conocido
solo como puntuación, que ya ordena y + α por sí mismo.

- **F1, fiabilidad.** En la etiqueta y en lo operable (k·p + α frente a y + α), con todas las filas
  y con el tercio más líquido de cada sesión:
  - la fracción de sesiones con IC de rangos > 0 y con decil alto − decil bajo > 0, y la de ventanas
    de 5 y 21 sesiones seguidas con esas medias > 0, todas con intervalos de Wilson al 95 %;
  - la curva de precisión selectiva: acierto de signo y respecto a la mediana de la sesión de las
    colas, según la fracción de nombres en la que se actúa;
  - el artefacto de R-1 (nombres con un alfa conocido patológico), con una regla fijada antes de
    medir: tope de |α| en su cuantil 0,99 de calibración. La curva se da también sin esas filas,
    junto con la parte de nombres elegidos que lo superan;
  - la cobertura de los intervalos (los de A2: el núcleo tal cual, conformalizado y combinado),
    total, por tercio de liquidez, por decil de volatilidad realizada de 21 sesiones conocida antes
    de la sesión y por trimestre, y el acierto de signo donde el intervalo del padre excluye el cero.
  - Cada orden mide un mercado; la lectura por mercado sale de una orden por mercado.
- **F2, confianza direccional calibrada,** solo en lo operable y sin las filas con |α| sobre el tope:
  - P(y + α > 0) se calibra con una regresión isótona creciente sobre la puntuación operable, en la
    primera mitad (en el tiempo) de las sesiones de calibración;
  - el umbral de confianza max(P, 1 − P) se elige en la segunda mitad con control de riesgo
    (Learn-then-Test): en cada umbral de una rejilla fija se contrasta «precisión < 0,90» con una
    prueba binomial por filas y otra normal por conglomerados de sesión, cada una a nivel
    δ / tamaño de la rejilla (Bonferroni), con δ = 0,05. El elegido es el umbral válido más bajo;
  - en evaluación se informan la cobertura, la precisión (Wilson y por conglomerados de sesión), el
    Brier frente al de la tasa base y la tabla de calibración. Si ningún umbral pasa, la cobertura
    es 0 y se dice tal cual.

Nada se elige en evaluación. 2013 es desarrollo; los orígenes limpios se miden con la misma orden.
"""

import argparse
import hashlib
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
from scipy.stats import binom, norm

from mars_titan.evaluation import agent_core as ac
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots

KIND = "posttraining_reliability_v1"
# Huella canónica de configs/experiments/posttraining-reliability.json, la declaración de F.
REGISTERED_SHA256 = "cdb1c5111e2b1d47199f28c82cbd6f6b19f66ecda768ebe02effcbc0a8a93f15"
SPLITS, AGGREGATED = og.SPLITS, og.AGGREGATED
PREDICTORS = ("s1", "s2", "tree", "parent")
# Control de lo operable: el alfa conocido solo, como puntuación. Ya ordena y + α por sí mismo
# (R-2: IC 0,045 con todos los nombres), así que lo que un predictor aporte se lee frente a él.
CONTROLS = dict(operable=("alpha",))
SCOPES = ("all", "liquid")
SYSTEMS = (("s1", ("parent", "tree")), ("s2", ("parent", "tree", "signal")))
GAMES = dict(s1="rank_average", s2="ridge_cov_active")
Z95 = 1.959963984540054


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_declaration(document):
    """La declaración registrada de F y su huella, o un error que dice por qué no lo es."""
    _require(isinstance(document, dict), "La declaración de F es un objeto JSON")
    digest = hashlib.sha256(pilots._canonical(document).encode()).hexdigest()
    _require(
        digest == REGISTERED_SHA256,
        f"La declaración no es la registrada para F (huella {digest[:12]}, registrada "
        f"{REGISTERED_SHA256[:12]}): la configuración de las medidas no cambia",
    )
    grid = document["aggregator"]
    _require(
        tuple(grid["lambdas"]) == tuple(oa.RIDGE_LAMBDAS)
        and tuple(grid["half_lives"]) == tuple(oa.RIDGE_HALF_LIVES),
        "El agregador del código no usa la rejilla de ridge_cov_active declarada",
    )
    return document, digest


# ---------------------------------------------------------------------------------------------
# F1: piezas puras


def wilson(hits, total, z=Z95):
    """Proporción con su intervalo de Wilson, o None sin observaciones."""
    hits, total = int(hits), int(total)
    if total == 0:
        return None
    p = hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return dict(rate=p, low=centre - half, high=centre + half, hits=hits, total=total)


def liquid_rows(liquidity, bounds):
    """Máscara del tercio más líquido de cada sesión; una fila sin liquidez conocida queda fuera."""
    keep = np.zeros(len(liquidity), dtype=bool)
    for start, end in bounds:
        values = liquidity[start:end]
        known = np.flatnonzero(np.isfinite(values))
        order = known[np.argsort(values[known], kind="mergesort")]
        keep[start + order[len(order) - len(order) // 3 :]] = True
    return keep


def _rows(start, end, keep):
    return np.arange(start, end) if keep is None else start + np.flatnonzero(keep[start:end])


def session_measures(tool, score, target, bounds, *, min_assets, keep=None):
    """(posición, IC de rangos, decil alto − decil bajo) de cada sesión con al menos `min_assets`
    filas; una sesión sin orden en la señal o en el objetivo no se puntúa, como en la criba."""
    result = []
    for index, (start, end) in enumerate(bounds):
        rows = _rows(start, end, keep)
        if len(rows) < min_assets:
            continue
        s, y = score[rows], target[rows]
        value = tool.correlation(tool.average_ranks(s), tool.average_ranks(y))
        if math.isnan(value):
            continue
        order = np.argsort(s, kind="stable")
        tenth = max(1, len(rows) // 10)
        result.append(
            (index, float(value), float(y[order[-tenth:]].mean() - y[order[:tenth]].mean()))
        )
    return result


def block_interval(values, keys, settings):
    """Media (una fracción si son indicadores) con su intervalo por arranque estacionario.

    Politis y Romano (1994): bloques de longitud geométrica con media `mean_block` sesiones,
    circulares dentro de cada mercado y en el orden del tiempo, `draws` remuestras con semilla
    fija; el intervalo son los percentiles 2,5 y 97,5. Wilson supone sesiones independientes; este
    no.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(settings["seed"])
    draws, p = settings["draws"], 1.0 / settings["mean_block"]
    total = np.zeros(draws)
    markets = {}
    for i, key in enumerate(keys):
        markets.setdefault(key[0], []).append((key[1], i))
    for market in sorted(markets):
        x = values[[i for _, i in sorted(markets[market])]]
        n = len(x)
        new = rng.random((draws, n)) < p
        new[:, 0] = True
        first = np.maximum.accumulate(np.where(new, np.arange(n), 0), axis=1)
        begin = np.take_along_axis(rng.integers(n, size=(draws, n)), first, axis=1)
        total += x[(begin + np.arange(n) - first) % n].sum(axis=1)
    low, high = np.quantile(total / len(values), [0.025, 0.975])
    return dict(rate=float(values.mean()), low=float(low), high=float(high))


def window_rates(measured, keys, width):
    """Ventanas de `width` sesiones puntuadas seguidas de cada mercado, sin solaparse (la última
    incompleta se descarta): fracción con IC medio > 0 y con diferencia de deciles media > 0."""
    by_market = {}
    for index, value, spread in measured:
        by_market.setdefault(keys[index][0], []).append((keys[index][1], value, spread))
    ic_hits = spread_hits = count = 0
    for market in sorted(by_market):
        series = sorted(by_market[market])
        for block in range(len(series) // width):
            chunk = series[block * width : (block + 1) * width]
            ic_hits += int(np.mean([c[1] for c in chunk]) > 0)
            spread_hits += int(np.mean([c[2] for c in chunk]) > 0)
            count += 1
    return dict(windows=count, ic=wilson(ic_hits, count), spread=wilson(spread_hits, count))


def selective_precision(score, target, bounds, coverages, *, min_assets, keep=None, flagged=None):
    """Acierto de las colas de cada sesión según la fracción de nombres en la que se actúa.

    Con cobertura c se toman los k = max(1, round(n·c/2)) nombres más altos de la señal (largos) y
    los k más bajos (cortos). El acierto de signo cuenta largos con objetivo > 0 y cortos con
    objetivo < 0; el relativo, respecto a la mediana de la sesión. Con `flagged` se cuenta además
    la parte de nombres elegidos marcados (los de |α| sobre el tope).
    """
    counts = {c: np.zeros(4, dtype=np.int64) for c in coverages}
    payoff = {c: 0.0 for c in coverages}
    sessions = 0
    for start, end in bounds:
        rows = _rows(start, end, keep)
        n = len(rows)
        if n < min_assets:
            continue
        sessions += 1
        s, y = score[rows], target[rows]
        order = np.argsort(s, kind="stable")
        median = float(np.median(y))
        for c in coverages:
            k = max(1, int(round(n * c / 2)))
            low, high = order[:k], order[-k:]
            marked = 0 if flagged is None else int(flagged[rows][np.r_[low, high]].sum())
            counts[c] += (
                int((y[high] > 0).sum() + (y[low] < 0).sum()),
                int((y[high] > median).sum() + (y[low] < median).sum()),
                2 * k,
                marked,
            )
            payoff[c] += float(y[high].mean() - y[low].mean())
    curve = {}
    for c in coverages:
        sign, relative, names, marked = (int(v) for v in counts[c])
        curve[str(c)] = dict(
            sign=wilson(sign, names),
            relative=wilson(relative, names),
            names=names,
            sessions=sessions,
            spread_bps=1e4 * payoff[c] / sessions if sessions else None,
            flagged=None if flagged is None else (marked / names if names else None),
        )
    return curve


def reliability(tool, view, settings, *, min_assets, keep=None):
    """Lecturas de F1 de una señal sobre las filas de un tramo (todas o `keep`), y sus sesiones."""
    measured = session_measures(
        tool, view["score"], view["target"], view["bounds"], min_assets=min_assets, keep=keep
    )
    ics = np.array([m[1] for m in measured])
    spreads = np.array([m[2] for m in measured])
    sessions = len(measured)
    scored_keys = [view["keys"][i] for i, _, _ in measured]
    clean = ~view["outlier"] if keep is None else keep & ~view["outlier"]
    curve = dict(min_assets=min_assets)
    result = dict(
        sessions=sessions,
        ic_mean=float(ics.mean()) if sessions else None,
        ic_std=float(ics.std(ddof=1)) if sessions > 1 else None,
        ic_worst=float(ics.min()) if sessions else None,
        ic_positive=wilson(int((ics > 0).sum()), sessions),
        ic_positive_block=block_interval(ics > 0, scored_keys, settings["block_interval"]),
        spread_positive=wilson(int((spreads > 0).sum()), sessions),
        spread_positive_block=block_interval(spreads > 0, scored_keys, settings["block_interval"]),
        spread_mean_bps=1e4 * float(spreads.mean()) if sessions else None,
        windows={str(w): window_rates(measured, view["keys"], w) for w in settings["windows"]},
        selective=selective_precision(
            view["score"],
            view["target"],
            view["bounds"],
            settings["coverages"],
            keep=keep,
            flagged=view["outlier"],
            **curve,
        ),
        selective_without_alpha_outliers=selective_precision(
            view["score"],
            view["target"],
            view["bounds"],
            settings["coverages"],
            keep=clean,
            **curve,
        ),
    )
    rows = [
        dict(market=str(view["keys"][i][0]), prediction_at=int(view["keys"][i][1]), ic=v, spread=s)
        for i, v, s in measured
    ]
    return result, rows, measured


def improvement(tool, measured, baseline, keys, windows, block):
    """Mejora por sesión de una señal sobre otra en las sesiones que puntúan las dos: diferencia de
    IC (media, t de Newey-West de la criba e intervalo al 95 %), de deciles, fracción de sesiones
    con mejora > 0 (Wilson) y ventanas con mejora media > 0."""
    base = {index: (value, spread) for index, value, spread in baseline}
    common = [(i, v - base[i][0], s - base[i][1]) for i, v, s in measured if i in base]
    if not common:
        return dict(sessions=0, mean=None, t_nw=None, interval=None, positive=None, windows={})
    differences = np.array([c[1] for c in common])
    stats = tool.summary({keys[i]: d for i, d, _ in common})
    t = stats["t_nw"]
    se = abs(stats["mean"] / t) if t not in (None, 0) and math.isfinite(t) else None
    return dict(
        sessions=len(common),
        mean=stats["mean"],
        t_nw=t,
        interval=None if se is None else [stats["mean"] - Z95 * se, stats["mean"] + Z95 * se],
        spread_mean_bps=1e4 * float(np.mean([c[2] for c in common])),
        positive=wilson(int((differences > 0).sum()), len(differences)),
        positive_block=block_interval(differences > 0, [keys[i] for i, _, _ in common], block),
        windows={str(w): window_rates(common, keys, w) for w in windows},
    )


def _claim(entry, low, block=None):
    """Demostrado si el límite inferior de Wilson (y el de bloques, si lo hay) llega a `low`;
    compatible si solo llega el punto."""
    if not entry:
        return "sin datos"
    if entry["low"] >= low and (block is None or block["low"] >= low):
        return "demostrado"
    return "compatible" if entry["rate"] >= low else "no"


def verdicts(summary, rules):
    """Lectura mecánica de las reglas de entrega declaradas, sin juicio después de medir.

    - F1: una fiabilidad ≥ 0,90 se da por demostrada si el límite inferior de Wilson llega a 0,90 y
      por compatible si solo llega el punto. En lo operable, se atribuye al modelo si su mejora
      sobre el alfa solo tiene media > 0 y t ≥ el declarado.
    - F2 entra como capa de confianza del sistema declarado si todas sus condiciones se cumplen.
    """
    low, min_t = rules["f1_claim_wilson_low"], rules["f1_attribution_min_t"]
    f1 = {}
    for key, item in summary["f1"].items():
        mode, predictor, scope = key.split("/")
        reading = dict(
            sessions=_claim(item["ic_positive"], low, item.get("ic_positive_block")),
            **{f"windows_{w}": _claim(v["ic"], low) for w, v in item["windows"].items()},
        )
        versus = summary["versus_alpha"].get(f"{predictor}/{scope}") if mode == "operable" else None
        if versus is not None:
            reading["over_alpha"] = bool(
                versus["mean"] is not None
                and versus["mean"] > 0
                and versus["t_nw"] is not None
                and versus["t_nw"] >= min_t
            )
        f1[key] = reading
    system, scope = rules["f2_system"], rules["f2_scope"]
    item, alpha = summary["f2"][f"{system}/{scope}"], summary["f2"][f"alpha/{scope}"]
    at, alpha_at = item["evaluation"]["at_threshold"], alpha["evaluation"]["at_threshold"]
    reasons = []
    if item["chosen_threshold"] is None:
        reasons.append("ningún umbral pasa el control de riesgo en calibración")
    else:
        if (
            at["coverage"] < rules["f2_min_coverage"]
            or at["sessions_with_action"] < rules["f2_min_sessions"]
        ):
            reasons.append("cobertura prácticamente cero")
        if at["precision"] is None or at["precision"]["rate"] < rules["f2_min_precision"]:
            reasons.append("la precisión de evaluación no llega a la declarada")
        clustered = at["clustered"]
        if (
            clustered is None
            or clustered["low"] is None
            or clustered["low"] < rules["f2_min_clustered_low"]
        ):
            reasons.append("el límite inferior por sesiones no llega al declarado")
        if not at["coverage"] > (alpha_at["coverage"] if alpha_at else 0.0):
            reasons.append("no cubre más que el alfa solo con su propio umbral")
    target_precision = rules["f2_target_precision"]
    seen = item["parts"]["evaluation"]["range"]
    if not seen.get("at_target_rows"):
        statement = f"no hay casos al {target_precision:.0%}".replace("%", " %")
    else:
        statement = (
            f"hay {seen['at_target_rows']} filas con confianza ≥ {target_precision} y acierto "
            f"{seen['at_target_precision']['rate']:.3f}; el control de riesgo "
            + ("las valida" if item["chosen_threshold"] is not None else "no las valida")
        )
    adaptive = summary["intervals"]["adaptive"]
    return dict(
        f1=f1,
        f2=dict(
            system=f"{system}/{scope}",
            enters=not reasons,
            reasons=reasons,
            confidence=dict(
                emitted="probabilidad calibrada (isótona con bloques mínimos)",
                range=[seen.get("min"), seen.get("max")],
                max_confidence=seen.get("max_confidence"),
                statement=statement,
                ece={part: v["calibration_error"]["ece"] for part, v in item["parts"].items()},
            ),
        ),
        intervals={
            level: dict(
                delivered=entry["delivered"],
                confirmed=entry["confirmed"],
                reasons=entry["candidates"][entry["delivered"]]["reasons"],
            )
            for level, entry in adaptive.items()
        },
    )


def zero_excluding(low, high, target, keep=None):
    """Acierto de signo donde el intervalo excluye el cero: largo si low > 0, corto si high < 0."""
    keep = np.ones(len(target), dtype=bool) if keep is None else keep
    long, short = keep & (low > 0), keep & (high < 0)
    acted = int(long.sum() + short.sum())
    return dict(
        rows=acted,
        share=acted / int(keep.sum()) if keep.any() else None,
        long=int(long.sum()),
        short=int(short.sum()),
        sign=wilson(int((target[long] > 0).sum() + (target[short] < 0).sum()), acted),
    )


# ---------------------------------------------------------------------------------------------
# F2: piezas puras


def isotonic_fit(x, y, min_rows=1):
    """Regresión isótona creciente de `y` sobre `x` (fusión de bloques adyacentes).

    Devuelve el borde derecho de cada bloque y su valor: una función escalonada creciente. Los
    empates en `x` se agrupan antes, así que el resultado no depende del orden de las filas.
    Con `min_rows` > 1, los bloques consecutivos se funden de izquierda a derecha hasta reunir al
    menos esas filas, y un resto final se une al último: la función sigue siendo creciente y
    ningún escalón descansa en un puñado de filas (en 2013, un bloque extremo de pocas filas daba
    P = 1 con un acierto real de 0,59).
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    _require(
        len(x) == len(y) and len(x) > 0 and np.isfinite(x).all() and np.isfinite(y).all(),
        "La regresión isótona necesita x e y finitos, de la misma longitud y no vacíos",
    )
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y, minlength=len(unique))
    weights = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    total, weight, right = [], [], []
    for value_sum, count, edge in zip(sums, weights, unique, strict=True):
        total.append(value_sum)
        weight.append(count)
        right.append(edge)
        while len(total) > 1 and total[-2] / weight[-2] >= total[-1] / weight[-1]:
            total[-2] += total[-1]
            weight[-2] += weight[-1]
            right[-2] = right[-1]
            del total[-1], weight[-1], right[-1]
    if min_rows > 1:
        groups, acc_total, acc_weight = [], 0.0, 0.0
        for value_sum, count, edge in zip(total, weight, right, strict=True):
            acc_total, acc_weight = acc_total + value_sum, acc_weight + count
            if acc_weight >= min_rows:
                groups.append([acc_total, acc_weight, edge])
                acc_total, acc_weight = 0.0, 0.0
        if acc_weight > 0:
            if groups:
                groups[-1][0] += acc_total
                groups[-1][1] += acc_weight
                groups[-1][2] = right[-1]
            else:
                groups.append([acc_total, acc_weight, right[-1]])
        total = [g[0] for g in groups]
        weight = [g[1] for g in groups]
        right = [g[2] for g in groups]
    return np.array(right), np.array(total) / np.array(weight)


def isotonic_predict(model, x):
    """Valor de la función escalonada: el del primer bloque cuyo borde derecho es ≥ x (el último
    por encima de todos)."""
    rights, values = model
    position = np.searchsorted(rights, np.asarray(x, dtype=np.float64), side="left")
    return values[np.clip(position, 0, len(values) - 1)]


def acted(probability, threshold):
    """Filas en que se actúa (confianza max(P, 1 − P) ≥ umbral) y su dirección (+1 o −1)."""
    confidence = np.maximum(probability, 1 - probability)
    return confidence >= threshold, np.where(probability >= 0.5, 1, -1)


def correct(target, direction):
    """Acierto de signo de cada fila; un objetivo nulo es un error en las dos direcciones."""
    return ((direction > 0) & (target > 0)) | ((direction < 0) & (target < 0))


def session_index(bounds, rows):
    """Sesión de cada fila de un tramo."""
    index = np.empty(rows, dtype=np.int64)
    for i, (start, end) in enumerate(bounds):
        index[start:end] = i
    return index


def clustered_precision(hits, acted_rows, z=Z95):
    """Precisión agregada (aciertos / actuaciones) con su error por conglomerados de sesión."""
    hits = np.asarray(hits, dtype=np.float64)
    acted_rows = np.asarray(acted_rows, dtype=np.float64)
    total = float(acted_rows.sum())
    if total == 0:
        return None
    p = float(hits.sum()) / total
    used = acted_rows > 0
    k = int(used.sum())
    if k < 2:
        return dict(rate=p, low=None, high=None, se=None, sessions=k)
    residual = hits[used] - p * acted_rows[used]
    se = math.sqrt(k / (k - 1) * float(np.sum(residual**2))) / total
    return dict(rate=p, low=p - z * se, high=p + z * se, se=se, sessions=k)


def _at(probability, target, session, sessions, threshold):
    mask, direction = acted(probability, threshold)
    good = correct(target, direction) & mask
    acted_by = np.bincount(session[mask], minlength=sessions)
    hits_by = np.bincount(session[good], minlength=sessions)
    return mask, int(good.sum()), hits_by, acted_by


def risk_tests(probability, target, session, sessions, thresholds, *, precision, delta):
    """Control de riesgo sobre una rejilla fija: cada umbral es válido si las dos pruebas de
    «precisión < `precision`» (binomial por filas y normal por conglomerados de sesión) dan
    p ≤ δ / tamaño de la rejilla. Devuelve el umbral válido más bajo (o None) y las pruebas."""
    level = delta / len(thresholds)
    tests = []
    for threshold in thresholds:
        mask, hits, hits_by, acted_by = _at(probability, target, session, sessions, threshold)
        n = int(mask.sum())
        binomial = float(binom.cdf(n - hits, n, 1 - precision)) if n else 1.0
        clustered = clustered_precision(hits_by, acted_by)
        if clustered is None or clustered["se"] is None:
            cluster_p = 1.0
        elif clustered["se"] == 0:
            cluster_p = 0.0 if clustered["rate"] > precision else 1.0
        else:
            cluster_p = float(norm.sf((clustered["rate"] - precision) / clustered["se"]))
        tests.append(
            dict(
                threshold=threshold,
                acted=n,
                precision=hits / n if n else None,
                binomial_p=binomial,
                clustered_p=cluster_p,
                valid=binomial <= level and cluster_p <= level,
            )
        )
    valid = [t["threshold"] for t in tests if t["valid"]]
    return (min(valid) if valid else None), tests


def calibration_error(probability, outcome, bins):
    """Tabla de calibración por intervalos iguales de P y su error esperado (ECE) y máximo (MCE)."""
    group = np.clip((probability * bins).astype(np.int64), 0, bins - 1)
    table, ece, mce, total = [], 0.0, None, len(probability)
    for b in range(bins):
        inside = group == b
        rows = int(inside.sum())
        predicted = float(probability[inside].mean()) if rows else None
        observed = float(outcome[inside].mean()) if rows else None
        table.append(
            dict(
                low=b / bins, high=(b + 1) / bins, rows=rows, predicted=predicted, observed=observed
            )
        )
        if rows:
            gap = abs(predicted - observed)
            ece += rows / total * gap
            mce = gap if mce is None else max(mce, gap)
    return dict(rows=total, ece=ece if total else None, mce=mce, table=table)


def probability_range(probability, target, target_precision):
    """Rango real de P y las filas cuya confianza llega a la precisión buscada, con su acierto."""
    if not len(probability):
        return dict(rows=0)
    confidence = np.maximum(probability, 1 - probability)
    mask, direction = acted(probability, target_precision)
    hits = int((correct(target, direction) & mask).sum())
    return dict(
        rows=len(probability),
        min=float(probability.min()),
        max=float(probability.max()),
        quantiles={
            str(q): float(np.quantile(probability, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        max_confidence=float(confidence.max()),
        at_target_rows=int(mask.sum()),
        at_target_precision=wilson(hits, int(mask.sum())),
    )


def directional(cal, ev, settings):
    """F2 de una señal operable: calibración, umbral con control de riesgo y lectura en evaluación.

    La probabilidad calibrada es también la confianza de signo que se emite: van su rango real y su
    error de calibración en la mitad de ajuste (dentro de muestra), en la de riesgo y en evaluación.
    """
    _require(len(cal["bounds"]) >= 2, "F2 necesita al menos dos sesiones de calibración")
    order = sorted(range(len(cal["bounds"])), key=lambda i: (cal["keys"][i][1], cal["keys"][i][0]))
    half = len(order) // 2
    fit = np.concatenate([np.arange(*cal["bounds"][i]) for i in order[:half]])
    risk = np.concatenate([np.arange(*cal["bounds"][i]) for i in order[half:]])
    min_rows = max(1, math.ceil(settings["min_block_share"] * len(fit)))
    model = isotonic_fit(cal["score"][fit], (cal["target"][fit] > 0).astype(np.float64), min_rows)
    thresholds = list(settings["thresholds"])
    cal_session = session_index(cal["bounds"], len(cal["score"]))
    risk_probability = isotonic_predict(model, cal["score"][risk])
    chosen, tests = risk_tests(
        risk_probability,
        cal["target"][risk],
        np.unique(cal_session[risk], return_inverse=True)[1],
        len(order) - half,
        thresholds,
        precision=settings["target_precision"],
        delta=settings["delta"],
    )
    probability = isotonic_predict(model, ev["score"])
    outcome = (ev["target"] > 0).astype(np.float64)
    bins, target_precision = settings["reliability_bins"], settings["target_precision"]
    parts = {}
    for name, p, y in (
        ("fit", isotonic_predict(model, cal["score"][fit]), cal["target"][fit]),
        ("risk", risk_probability, cal["target"][risk]),
        ("evaluation", probability, ev["target"]),
    ):
        parts[name] = dict(
            range=probability_range(p, y, target_precision),
            calibration_error=calibration_error(p, (y > 0).astype(np.float64), bins),
        )
    ev_session, sessions = session_index(ev["bounds"], len(ev["score"])), len(ev["bounds"])
    curve = []
    for threshold in thresholds:
        mask, hits, hits_by, acted_by = _at(
            probability, ev["target"], ev_session, sessions, threshold
        )
        n = int(mask.sum())
        curve.append(
            dict(
                threshold=threshold,
                coverage=n / len(mask) if len(mask) else None,
                precision=wilson(hits, n),
                clustered=clustered_precision(hits_by, acted_by),
                sessions_with_action=int((acted_by > 0).sum()),
            )
        )
    base = float(outcome.mean()) if len(outcome) else None
    return dict(
        calibrator=dict(
            blocks=len(model[1]),
            fit_rows=len(fit),
            risk_rows=len(risk),
            min_rows=min_rows,
            edges=model[0].tolist(),
            values=model[1].tolist(),
        ),
        chosen_threshold=chosen,
        risk_tests=tests,
        parts=parts,
        evaluation=dict(
            rows=len(probability),
            sessions=sessions,
            base_rate=base,
            brier=float(np.mean((probability - outcome) ** 2)) if len(outcome) else None,
            brier_base=float(np.mean((base - outcome) ** 2)) if len(outcome) else None,
            at_threshold=None if chosen is None else curve[thresholds.index(chosen)],
            curve=curve,
        ),
    )


def confidence_table(view, item, cap_flag):
    """La confianza de signo que emite un sistema: P calibrada por fila, con su dirección."""
    model = (np.asarray(item["calibrator"]["edges"]), np.asarray(item["calibrator"]["values"]))
    probability = isotonic_predict(model, view["score"])
    return pa.table(
        dict(
            sample_id=pa.array(view["sample_id"], pa.string()),
            probability_up=pa.array(probability, pa.float64()),
            confidence=pa.array(np.maximum(probability, 1 - probability), pa.float64()),
            direction=pa.array(np.where(probability >= 0.5, 1, -1).astype(np.int8)),
            alpha_over_cap=pa.array(np.asarray(cap_flag, dtype=bool)),
        )
    )


# ---------------------------------------------------------------------------------------------
# Intervalos adaptativos


def _conformal_quantile(ordered, level):
    """Cuantil ⌈(n + 1)·nivel⌉/n de unas puntuaciones ordenadas, con el nivel acotado a
    [1/(n + 1), n/(n + 1)] para que la banda sea siempre finita."""
    n = len(ordered)
    level = min(max(level, 1 / (n + 1)), n / (n + 1))
    return float(ordered[min(max(math.ceil((n + 1) * level), 1), n) - 1])


def adaptive_bands(
    lower, upper, target, sessions, maturity, scores, level, gamma, groups=None, score_groups=None
):
    """CQR por grupo con ACI por sesión (Gibbs y Candès, 2021) sobre unas bandas.

    `sessions` son (mercado, instante, inicio, fin) en el orden de las filas, y `maturity` el
    instante en que madura la última etiqueta de cada una. Cada mercado y cada grupo llevan su
    propio nivel α, que empieza en 1 − nivel. Al llegar a una sesión entran en la recursión
    α ← α + γ·(1 − nivel − fallo) las sesiones anteriores ya maduras, en orden; el fallo es la
    parte de filas del grupo que su banda no cubrió. La corrección de la sesión es el cuantil
    conforme 1 − α de las puntuaciones de calibración del grupo (las de todos si el grupo no tiene).
    Con γ = 0 es la CQR estática; sin grupos, la global.
    """
    groups = np.zeros(len(target), dtype=np.int64) if groups is None else np.asarray(groups)
    score_groups = (
        np.zeros(len(scores), dtype=np.int64) if score_groups is None else np.asarray(score_groups)
    )
    pooled = np.sort(np.asarray(scores, dtype=np.float64))
    ordered = {}
    for g in np.unique(groups).tolist():
        own = np.sort(np.asarray(scores, dtype=np.float64)[score_groups == g])
        ordered[g] = own if len(own) else pooled
    low, high = np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)
    nominal = 1 - level
    by_market, levels = {}, []
    for index, (market, instant, start, end) in enumerate(sessions):
        by_market.setdefault(market, []).append((int(instant), index, int(start), int(end)))
    for market in sorted(by_market):
        alpha = dict.fromkeys(ordered, nominal)
        pending = []
        for instant, index, start, end in sorted(by_market[market]):
            ready = sorted(p for p in pending if p[0] <= instant)
            pending = [p for p in pending if p[0] > instant]
            for _, _, g, miss in ready:
                alpha[g] += gamma * (nominal - miss)
            segment = groups[start:end]
            for g in np.unique(segment).tolist():
                rows = start + np.flatnonzero(segment == g)
                q = _conformal_quantile(ordered[g], 1 - alpha[g])
                low[rows], high[rows] = lower[rows] - q, upper[rows] + q
                inside = (target[rows] >= low[rows]) & (target[rows] <= high[rows])
                pending.append((int(maturity[index]), instant, g, 1.0 - float(inside.mean())))
            levels.append(float(np.mean(list(alpha.values()))))
    return (
        low,
        high,
        dict(
            alpha_mean=float(np.mean(levels)) if levels else None,
            alpha_min=float(np.min(levels)) if levels else None,
            alpha_max=float(np.max(levels)) if levels else None,
        ),
    )


def interval_rule(item, level, rule):
    """Regla de R-1: a menos de `total_tolerance` de la nominal en total, y ningún decil de
    volatilidad ni mercado más de `group_shortfall` por debajo."""
    reasons = []
    if abs(item["coverage"] - level) > rule["total_tolerance"]:
        reasons.append(f"cobertura total {item['coverage']:.3f} frente a {level}")
    worst = min(g["coverage"] for g in item["volatility"].values())
    if worst < level - rule["group_shortfall"]:
        reasons.append(f"decil de volatilidad con {worst:.3f}")
    for market, part in item["market"].items():
        if part["coverage"] < level - rule["group_shortfall"]:
            reasons.append(f"mercado {market} con {part['coverage']:.3f}")
    return reasons


def _quarters(table):
    instants = table.column("prediction_at").cast(pa.int64()).to_numpy().astype("datetime64[us]")
    years = instants.astype("datetime64[Y]").astype(int) + 1970
    quarters = (instants.astype("datetime64[M]").astype(int) % 12) // 3 + 1
    return np.array([f"{y}T{q}" for y, q in zip(years, quarters, strict=True)])


def adaptive_intervals(parent, covariates, matures, settings):
    """Candidatos de intervalo sobre los cuantiles del padre, medidos en evaluación con la regla."""
    sessions = {s: ac._sessions_of(parent[s]) for s in AGGREGATED}
    bounds = {s: ac.bounds_of(sessions[s]) for s in AGGREGATED}
    target = {s: parent[s].column("target").to_numpy().astype(np.float64) for s in AGGREGATED}
    volatility = {
        s: ac.within_session_groups(
            covariates[s]["volatility"], bounds[s], settings["volatility_groups"]
        )
        for s in AGGREGATED
    }
    liquidity = ac.within_session_groups(
        covariates["evaluation"]["liquidity"], bounds["evaluation"], settings["liquidity_groups"]
    )
    markets = np.asarray(parent["evaluation"].column("market").to_pylist())
    quarters = _quarters(parent["evaluation"])
    maturity = np.array(
        [int(np.max(matures["evaluation"][start:end])) for start, end in bounds["evaluation"]]
    )
    result = {}
    for level in settings["levels"]:
        low_name, high_name = settings["core"][str(level)]
        band = {
            s: (
                parent[s].column(low_name).to_numpy().astype(np.float64),
                parent[s].column(high_name).to_numpy().astype(np.float64),
            )
            for s in AGGREGATED
        }
        scores = ac.conformity(*band["calibration"], target["calibration"])
        candidates = {}
        for name, gamma, grouped in (
            ("core_cqr", 0.0, False),
            ("core_cqr_aci", settings["aci_gamma"], False),
            ("core_mondrian", 0.0, True),
            ("core_mondrian_aci", settings["aci_gamma"], True),
        ):
            low, high, path = adaptive_bands(
                *band["evaluation"],
                target["evaluation"],
                sessions["evaluation"],
                maturity,
                scores,
                level,
                gamma,
                groups=volatility["evaluation"] if grouped else None,
                score_groups=volatility["calibration"] if grouped else None,
            )
            y = target["evaluation"]
            item = dict(
                **ac.interval_summary(low, high, y),
                volatility=ac.interval_summary(low, high, y, volatility["evaluation"])["groups"],
                liquidity=ac.interval_summary(low, high, y, liquidity)["groups"],
                quarter=ac.interval_summary(low, high, y, quarters)["groups"],
                market=ac.interval_summary(low, high, y, markets)["groups"],
                aci=path,
            )
            item["reasons"] = interval_rule(item, level, settings["rule"])
            item["passes"] = not item["reasons"]
            candidates[name] = item
        delivered = settings["delivered"]
        result[str(level)] = dict(
            candidates=candidates,
            delivered=delivered,
            confirmed=candidates[delivered]["passes"],
            first_passing=next(
                (n for n in settings["candidates"] if candidates[n]["passes"]), None
            ),
        )
    return result


ROW_ARRAYS = ("score", "target", "liquidity", "alpha", "outlier")


def restricted(view, keep):
    """La misma vista con solo las filas de `keep`; las sesiones que se quedan sin filas salen."""
    bounds, keys, start = [], [], 0
    for key, (first, last) in zip(view["keys"], view["bounds"], strict=True):
        count = int(keep[first:last].sum())
        if count:
            bounds.append((start, start + count))
            keys.append(key)
            start += count
    index = np.flatnonzero(keep)
    return dict(
        view,
        bounds=bounds,
        keys=keys,
        **{name: np.asarray(view[name])[index] for name in ROW_ARRAYS},
    )


# ---------------------------------------------------------------------------------------------
# La orden por origen


def alpha_cap(alpha, settings):
    """Tope de |α|: su cuantil declarado entre las filas con alfa conocido del tramo declarado."""
    known = np.asarray(alpha[settings["alpha_cap_split"]], dtype=np.float64)
    known = known[np.isfinite(known)]
    _require(
        len(known) > 0,
        f"Sin alfa conocido en {settings['alpha_cap_split']} no hay tope de |α|: F no se mide",
    )
    return float(np.quantile(np.abs(known), settings["alpha_cap_quantile"]))


def _views(mode, results, names, parent, liquidity, alpha, cap):
    """Vistas de los cuatro predictores en calibración y evaluación de un modo."""
    views = {}
    for split in AGGREGATED:
        aligned = results["s2"]["aligned"][split]
        _require(
            list(results["s1"]["aligned"][split]["sample_id"]) == list(aligned["sample_id"]),
            f"S1 y S2 no tienen las mismas filas en {mode}/{split}",
        )
        position = {name: i for i, name in enumerate(aligned["model_names"])}
        index = {s: i for i, s in enumerate(parent[split].column("sample_id").to_pylist())}
        rows = np.array([index[s] for s in aligned["sample_id"]], dtype=np.int64)
        known = np.asarray(alpha[split], dtype=np.float64)[rows]
        if mode == "operable":
            _require(
                np.allclose(known, aligned["alpha"], rtol=1e-6, atol=1e-9),
                f"Las filas operables de {split} no llevan el alfa conocido de sus muestras",
            )
        base = dict(
            bounds=ac.bounds_of(aligned["sessions"]),
            keys=[(str(m), int(t)) for m, t, _, _ in aligned["sessions"]],
            target=np.asarray(aligned["target"], dtype=np.float64),
            liquidity=np.asarray(liquidity[split], dtype=np.float64)[rows],
            alpha=known,
            outlier=np.abs(np.nan_to_num(known, nan=0.0)) > cap,
            sample_id=list(aligned["sample_id"]),
        )
        scores = dict(
            s1=results["s1"]["aggregators"][GAMES["s1"]]["split_predictions"][split],
            s2=results["s2"]["aggregators"][GAMES["s2"]]["split_predictions"][split],
            tree=aligned["predictions"][:, position[names["tree"]]],
            parent=aligned["predictions"][:, position[names["parent"]]],
        )
        if mode == "operable":
            scores["alpha"] = known
        for predictor, score in scores.items():
            values = np.asarray(score, dtype=np.float64)
            _require(
                len(values) == len(base["target"]) and np.isfinite(values).all(),
                f"{predictor} en {mode}/{split} no tiene una puntuación finita por fila alineada",
            )
            views[(predictor, split)] = dict(base, score=values)
    return views


def measure(
    declaration, state, reference, organs, origin, output, *, wait_hours, poll, log, markets=None
):
    """F1 y F2 de un origen, publicados en `<output>/<origen>` (o `<origen>-<mercados>`)."""
    declaration, digest = checked_declaration(declaration)
    sources = og.folders(declaration, state, reference, organs, origin)
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    missing = og.wait_for_inputs(
        declaration, sources, manifest, hours=wait_hours, poll=poll, say=log
    )
    if missing:
        raise og.MissingInputs(missing)
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
    tool = og._screening()
    schema = pq.read_schema(sources["parent"] / "evaluation-predictions.parquet").names
    _require(
        all(q in schema for q in ac.QUANTILES),
        f"{sources['parent']} no emite los cuantiles {ac.QUANTILES}",
    )
    parent = {
        s: pq.read_table(
            sources["parent"] / f"{s}-predictions.parquet", columns=og.COLUMNS + list(ac.QUANTILES)
        ).sort_by(og.ORDER)
        for s in SPLITS
    }
    tables = dict(
        tree={s: og._table(sources["tree"], s) for s in SPLITS},
        organ={s: og._table(sources["organ"], s) for s in AGGREGATED},
    )
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
    alpha, liquidity, volatility, matures = ac.known_rows(state, origin, parent, view)
    covariates = {s: dict(liquidity=liquidity[s], volatility=volatility[s]) for s in SPLITS}
    settings, min_assets = declaration["reliability"], declaration["min_assets"]
    cap = alpha_cap(alpha, settings)
    say(f"covariables conocidas y madurez; tope de |α| {cap:.6f}")
    own = {s: parent[s].column("prediction").to_numpy().astype(np.float64) for s in SPLITS}
    predictions = dict(
        parent=own,
        tree={
            s: tables["tree"][s].column("prediction").to_numpy().astype(np.float64) for s in SPLITS
        },
        signal={
            s: tables["organ"][s].column("prediction").to_numpy().astype(np.float64) - own[s]
            for s in AGGREGATED
        },
    )
    names = dict(parent=declaration["parent"], tree=declaration["tree"], signal=og.SIGNAL)
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        work.mkdir(parents=True)
        raw_base = {
            s: oa.build_raw_split_base(
                parent[s], alpha=alpha[s], matures_at=matures[s], already_sorted=True
            )
            for s in AGGREGATED
        }
        f1, f1_rows, f2, versus, factors = {}, [], {}, {}, None
        decl_splits = tuple(declaration["splits"])
        decl_scales = tuple(declaration["operable_scales"])
        for mode in declaration["modes"]:
            is_operable = mode == "operable"
            raw_by_split = {
                s: {
                    names[p]: oa.build_raw_split(
                        raw_base[s], predictions[p][s], include_alpha=is_operable
                    )
                    for p in names
                }
                for s in AGGREGATED
            }
            cache = (
                oa.build_shared_panel_cache(
                    raw_by_split,
                    splits=decl_splits,
                    min_assets=min_assets,
                    align=declaration["align"],
                    operable=is_operable,
                    operable_scales=decl_scales,
                )
                if declaration["align"] == "strict"
                else None
            )
            results = {
                system: oa.run_aggregation(
                    {names[p]: names[p] for p in members},
                    splits=decl_splits,
                    selection_split=declaration["selection_split"],
                    min_assets=min_assets,
                    align=declaration["align"],
                    operable=is_operable,
                    alpha_col="alpha",
                    maturity_col="matures_at",
                    placebo_seed=declaration["aggregator_placebo_seed"],
                    operable_scales=decl_scales,
                    aggregators=("rank_average", "ridge_cov_active"),
                    include_placebos=False,
                    include_asset_abstention=False,
                    preloaded_raw=raw_by_split,
                    precomputed_cache=cache,
                )
                for system, members in SYSTEMS
            }
            if mode == "operable":
                scales = {s: results[s]["operable"]["scales"] for s in results}
                _require(
                    all(
                        scales["s1"][names[p]] == scales["s2"][names[p]] for p in ("parent", "tree")
                    ),
                    "El factor operable de un componente cambia con el conjunto",
                )
                factors = scales["s2"]
            views = _views(mode, results, names, parent, liquidity, alpha, cap)
            say(f"{mode}: S1 y S2 agregados")
            sessions_of = {}
            ordered_predictors = CONTROLS.get(mode, ()) + PREDICTORS
            first_ev = views[(ordered_predictors[0], "evaluation")]
            first_cal = views[(ordered_predictors[0], "calibration")]
            liquid_shared = dict(
                evaluation=liquid_rows(first_ev["liquidity"], first_ev["bounds"]),
                calibration=(
                    liquid_rows(first_cal["liquidity"], first_cal["bounds"])
                    if mode == "operable"
                    else None
                ),
            )
            # Los controles van primero: la mejora de cada predictor se mide frente a ellos.
            for predictor in ordered_predictors:
                ev, cal = views[(predictor, "evaluation")], views[(predictor, "calibration")]
                for scope in SCOPES:
                    keep = None if scope == "all" else liquid_shared["evaluation"]
                    item, rows, scored = reliability(
                        tool, ev, settings, min_assets=min_assets, keep=keep
                    )
                    f1[f"{mode}/{predictor}/{scope}"] = item
                    f1_rows += [
                        dict(mode=mode, predictor=predictor, scope=scope, **r) for r in rows
                    ]
                    sessions_of[(predictor, scope)] = scored
                    if mode == "operable" and predictor not in CONTROLS[mode]:
                        versus[f"{predictor}/{scope}"] = improvement(
                            tool,
                            scored,
                            sessions_of[("alpha", scope)],
                            ev["keys"],
                            settings["windows"],
                            settings["block_interval"],
                        )
                if mode != "operable":
                    continue
                for scope in SCOPES:
                    pair = []
                    for split, item in (("calibration", cal), ("evaluation", ev)):
                        keep = ~item["outlier"]
                        if scope == "liquid":
                            keep &= liquid_shared[split]
                        pair.append(restricted(item, keep))
                    f2[f"{predictor}/{scope}"] = directional(*pair, declaration["directional"])
                emit = declaration["directional"]["emit"]
                if predictor == emit["system"]:
                    item = f2[f"{predictor}/{emit['scope']}"]
                    for split, view_ in (("calibration", cal), ("evaluation", ev)):
                        pq.write_table(
                            confidence_table(view_, item, view_["outlier"]),
                            work / f"{predictor}-confianza-{split}.parquet",
                        )
            say(f"{mode}: F1{' y F2' if mode == 'operable' else ''}")
        intervals = ac._a2(declaration, parent, tables["tree"], covariates)
        intervals["adaptive"] = adaptive_intervals(
            parent,
            covariates,
            matures,
            dict(declaration["conformal"], **declaration["adaptive_intervals"]),
        )
        y = parent["evaluation"].column("target").to_numpy().astype(np.float64)
        clean = ~(np.abs(np.nan_to_num(np.asarray(alpha["evaluation"], dtype=np.float64))) > cap)
        intervals["zero_excluding"] = {}
        for level, (low_name, high_name) in declaration["conformal"]["core"].items():
            low = parent["evaluation"].column(low_name).to_numpy().astype(np.float64)
            high = parent["evaluation"].column(high_name).to_numpy().astype(np.float64)
            intervals["zero_excluding"][level] = dict(
                all=zero_excluding(low, high, y),
                without_alpha_outliers=zero_excluding(low, high, y, keep=clean),
            )
        say("intervalos")
        code = og._code_identity()
        code["files"]["reliability"] = pilots._sha256(Path(__file__))
        code["files"]["agent_core"] = pilots._sha256(Path(ac.__file__))
        summary = og._plain(
            dict(
                schema_version=1,
                kind=KIND,
                status="completed",
                origin=origin,
                declaration_sha256=digest,
                code=code,
                inputs=fingerprint,
                markets=measured,
                predictors=dict(
                    s1="S1",
                    s2="S2",
                    tree=names["tree"],
                    parent=names["parent"],
                    alpha="alfa conocido solo (control, en lo operable)",
                ),
                operable_factors=factors,
                alpha_known={s: float(np.isfinite(alpha[s]).mean()) for s in SPLITS},
                alpha_cap=cap,
                f1=f1,
                versus_alpha=versus,
                f2=f2,
                intervals=intervals,
                seconds=round(time.monotonic() - started),
            )
        )
        summary["verdicts"] = verdicts(summary, declaration["delivery"])
        pq.write_table(ac._series_table(f1_rows), work / "f1-sesiones.parquet")
        (work / "resumen.md").write_text(markdown(summary), encoding="utf-8")
        (work / "resumen.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.rename(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say(f"publicado en {target}")
    return summary


# ---------------------------------------------------------------------------------------------
# Resumen legible y orden


def _f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _p(entry, digits=3):
    if not entry or entry.get("low") is None:
        return "—" if not entry else _f(entry["rate"], digits)
    return f"{_f(entry['rate'], digits)} [{_f(entry['low'], digits)}; {_f(entry['high'], digits)}]"


def markdown(summary):
    """Resumen legible de F1 y F2 de un origen."""
    lines = [
        f"# Fiabilidad (F1 y F2) del origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration_sha256'][:12]}; mercados {', '.join(summary['markets'])}; "
        f"evaluación, con intervalos de Wilson al 95 %. Tope de |α| (cuantil 0,99 de calibración): "
        f"{_f(summary['alpha_cap'], 5)}. Factores operables: {summary['operable_factors']}.",
        "",
        "## F1, fiabilidad por sesión y en ventanas",
        "",
        "| Modo / predictor / filas | Sesiones | IC medio | Peor IC | IC > 0 (Wilson) | IC > 0 (bloques) | Deciles > 0 | Ventanas de 5 con IC > 0 | Ventanas de 21 con IC > 0 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, item in summary["f1"].items():
        windows = item["windows"]
        lines.append(
            f"| {key} | {item['sessions']} | {_f(item['ic_mean'])} | {_f(item['ic_worst'])} | "
            f"{_p(item['ic_positive'])} | {_p(item['ic_positive_block'])} | {_p(item['spread_positive'])} | "
            f"{_p(windows['5']['ic'])} | {_p(windows['21']['ic'])} |"
        )
    lines += [
        "",
        "## F1, precisión selectiva (acierto de signo de las colas de cada sesión)",
        "",
        "| Modo / predictor / filas | Todo | 20 % | 5 % | 1 % | 1 % sin |α| sobre el tope | Parte del 1 % sobre el tope |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, item in summary["f1"].items():
        sel, clean = item["selective"], item["selective_without_alpha_outliers"]
        lines.append(
            f"| {key} | {_p(sel['1.0']['sign'])} | {_p(sel['0.2']['sign'])} | "
            f"{_p(sel['0.05']['sign'])} | {_p(sel['0.01']['sign'])} | {_p(clean['0.01']['sign'])} | "
            f"{_f(sel['0.01']['flagged'], 3)} |"
        )
    lines += [
        "",
        "## Mejora sobre el alfa solo, en lo operable (IC por sesión, pareado)",
        "",
        "| Predictor / filas | Sesiones | Media | t | Intervalo al 95 % | Sesiones con mejora | Ventanas de 5 con mejora |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, item in summary["versus_alpha"].items():
        interval = item["interval"]
        lines.append(
            f"| {key} | {item['sessions']} | {_f(item['mean'])} | {_f(item['t_nw'], 2)} | "
            f"{'—' if interval is None else f'[{_f(interval[0])}; {_f(interval[1])}]'} | "
            f"{_p(item['positive'])} | {_p((item['windows'].get('5') or {}).get('ic'))} |"
        )
    lines += [
        "",
        "## F2, confianza direccional calibrada en lo operable (sin |α| sobre el tope)",
        "",
        "| Predictor / filas | Umbral elegido | Cobertura | Precisión (Wilson) | Precisión (sesiones) | Brier / tasa base |",
        "|---|---|---|---|---|---|",
    ]
    for key, item in summary["f2"].items():
        at, ev = item["evaluation"]["at_threshold"], item["evaluation"]
        if at is None:
            cells = "ninguno pasa | 0 | — | —"
        else:
            cells = (
                f"{_f(item['chosen_threshold'], 3)} | {_f(at['coverage'], 5)} | "
                f"{_p(at['precision'])} | {_p(at['clustered'])}"
            )
        lines.append(f"| {key} | {cells} | {_f(ev['brier'])} / {_f(ev['brier_base'])} |")
    verdict = summary["verdicts"]["f2"]
    confidence = verdict["confidence"]
    lines += [
        "",
        "| Predictor / filas | Rango de P en evaluación | Confianza máxima | Filas con confianza ≥ 0,90 (acierto) | ECE ajuste / riesgo / evaluación |",
        "|---|---|---|---|---|",
    ]
    for key, item in summary["f2"].items():
        seen = item["parts"]["evaluation"]["range"]
        ece = [item["parts"][p]["calibration_error"]["ece"] for p in ("fit", "risk", "evaluation")]
        lines.append(
            f"| {key} | [{_f(seen.get('min'), 3)}; {_f(seen.get('max'), 3)}] | "
            f"{_f(seen.get('max_confidence'), 3)} | {seen.get('at_target_rows', 0)} "
            f"({_p(seen.get('at_target_precision'))}) | {' / '.join(_f(e, 4) for e in ece)} |"
        )
    lines += [
        "",
        f"Confianza de signo que emite {verdict['system']}: {confidence['emitted']}, con rango "
        f"[{_f(confidence['range'][0], 3)}; {_f(confidence['range'][1], 3)}] en evaluación; "
        f"{confidence['statement']}.",
        "",
        f"F2 en la entrega ({verdict['system']}): "
        + ("entra." if verdict["enters"] else "no entra: " + "; ".join(verdict["reasons"]) + "."),
        "",
        "F1, fiabilidad ≥ 0,90 demostrada (límite de Wilson) o compatible (solo el punto): "
        + "; ".join(
            f"{key} {name}: {value}"
            for key, reading in summary["verdicts"]["f1"].items()
            for name, value in reading.items()
            if value in ("demostrado", "compatible")
        )
        + ".",
    ]
    lines += ["", "## Intervalos en evaluación (etiqueta)", ""]
    for level, item in summary["intervals"]["levels"].items():
        for name, part in item["systems"].items():
            deciles = [g["coverage"] for g in part["volatility"].values()]
            lines.append(
                f"- {level}, {name}: cobertura {_f(part['coverage'], 3)}, anchura media "
                f"{_f(part['width'], 5)}; por decil de volatilidad, de {_f(min(deciles), 3)} a "
                f"{_f(max(deciles), 3)}"
            )
    lines += [
        "",
        "| Nivel / candidato | Cobertura | Anchura media | Peor decil de volatilidad | α medio de ACI | Pasa la regla |",
        "|---|---|---|---|---|---|",
    ]
    for level, entry in summary["intervals"]["adaptive"].items():
        for name, item in entry["candidates"].items():
            worst = min(g["coverage"] for g in item["volatility"].values())
            lines.append(
                f"| {level} / {name} | {_f(item['coverage'], 3)} | {_f(item['width'], 5)} | "
                f"{_f(worst, 3)} | {_f(item['aci']['alpha_mean'], 3)} | "
                f"{'sí' if item['passes'] else 'no: ' + '; '.join(item['reasons'])} |"
            )
        lines.append(
            f"| {level} / entregado (fijo, elegido con 2013) | {entry['delivered']} | | | | "
            + ("confirma la regla |" if entry["confirmed"] else "no confirma la regla |")
        )
    for level, item in summary["intervals"]["zero_excluding"].items():
        for scope, part in item.items():
            lines.append(
                f"- Intervalo del padre al {level} que excluye el cero ({scope}): {part['rows']} filas "
                f"({_f(part['share'], 4)} del tramo), acierto de signo {_p(part['sign'])}"
            )
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("medir", help="Calcular F1 y F2 de un origen")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    run.add_argument("--reference", type=Path, required=True, help="Raíz de los árboles")
    run.add_argument("--organs", type=Path, required=True, help="Raíz de los órganos")
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--output", type=Path, required=True, help="Raíz propia de F")
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
            f"Faltan entradas de F para el origen {args.origin} tras esperar "
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
