# ruff: noqa: E501
"""F7, las métricas de error de las emisiones y su calibración.

Encargo del dueño, por el coordinador (23 de noviembre de 2026), con el protocolo del programa
(§5.13) y la declaración `configs/experiments/posttraining-error-metrics.json`, cuya huella exige
la orden.

- **Métricas.** Son las que el repositorio calcula siempre, con `forecast_scores.score_sessions` y
  cada sesión con el mismo peso:
  - MAE y RMSE;
  - pinball de cada cuantil y su media;
  - error de cobertura de los intervalos centrales del 80 y del 95 %;
  - ECE de la probabilidad de subida implícita en los cuantiles.

  Fuera de la regla, como información, van la puntuación de intervalo, el Brier de signo, la
  frecuencia de cada nivel y el IC de rangos.
- **Predictores**, en la etiqueta y en lo operable, sobre las filas de S2 de cada modo:
  - el padre;
  - la media de sus semillas (`seed_ensemble.ordered_mean` en orden de semilla, nivel a nivel en
    los cuantiles);
  - S1 y S2;
  - como referencias, si están emitidas, el núcleo de la criba y el modelo entrenado solo con
    pinball (noic), que da el mejor nivel con estos datos.

  En lo operable, la puntuación de los modelos es α + k·p (en FP64, con el k del criterio del
  agregador) y sus cuantiles son α + q.
- **Controles.** El cero, con la climatología de calibración como cuantiles. En lo operable,
  además, el alfa solo, con α más la climatología de la etiqueta.
- **Calibración.** Se ajusta en calibración, con las etiquetas maduras antes de la primera sesión de
  evaluación de su mercado, y se aplica sin cambios.
  - El punto pasa a λ·s, con el factor LAD de `scale_diagnostics`. λ > 0, así que el orden de la
    sesión no cambia.
  - Los cuantiles se centran en λ·s. Las anchuras de la cabeza se multiplican por el factor de cada
    nivel que minimiza exactamente la pinball.
  - Si dos niveles se cruzan, la fila se ordena. Ese reordenamiento nunca sube la pinball sumada.
- **Regla.** Una versión calibrada entra si, en evaluación, no empeora ninguna métrica de la regla
  frente a la sin calibrar (comparación estricta). Se entrega la más completa que pase. S1 y S2
  emiten puntuaciones z sin unidades, así que se comparan con el control de su modo.

Con un IC de Pearson ρ, el mejor predictor lineal solo rebaja el RMSE de predecir cero en
1 − √(1 − ρ²). El informe da esa cota con el ρ medido.
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

from mars_titan.evaluation import agent_core as ac
from mars_titan.evaluation import forecast_scores as fs
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.evaluation import reliability as rel
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation import seed_ensemble as se
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS

KIND = "posttraining_error_metrics_v1"
# Huella canónica de configs/experiments/posttraining-error-metrics.json, la declaración de F7.
REGISTERED_SHA256 = "606db891582cf1a6acd62414f9e1e1c18a1599be0a14073d5180229a039c93b8"
AGGREGATED = og.AGGREGATED
MODES = ("label", "operable")
HEADS = ("parent", "seeds", "core", "noic")
REFERENCES = ("core", "noic")
PREDICTORS = (*HEADS, "s1", "s2")
VERSIONS = ("raw", "point", "full")
LEVELS = tuple(float(level) for level in LEVELS)
MEDIAN = LEVELS.index(0.5)
NOMINALS = (0.8, 0.95)
CONTROLS = dict(label=("zero",), operable=("zero", "alpha"))
# Con qué se compara un predictor que no tiene versión sin calibrar.
REFERENCE = dict(label="zero", operable="alpha")
PAIRED = ("mae", "mse", "pinball")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_declaration(document):
    """La declaración registrada de F7 y su huella, o un error que dice por qué no lo es."""
    _require(isinstance(document, dict), "La declaración de F7 es un objeto JSON")
    digest = hashlib.sha256(pilots._canonical(document).encode()).hexdigest()
    _require(
        digest == REGISTERED_SHA256,
        f"La declaración no es la registrada para F7 (huella {digest[:12]}, registrada "
        f"{REGISTERED_SHA256[:12]}): el protocolo no cambia",
    )
    grid = document["aggregator"]
    _require(
        tuple(grid["lambdas"]) == tuple(oa.RIDGE_LAMBDAS)
        and tuple(grid["half_lives"]) == tuple(oa.RIDGE_HALF_LIVES),
        "El agregador del código no usa la rejilla de ridge_cov_active declarada",
    )
    metrics = document["metrics"]
    _require(
        tuple(metrics["levels"]) == LEVELS
        and tuple(metrics["intervals"]) == NOMINALS
        and metrics["sign_probability"] == fs.SIGN_PROBABILITY_RULE,
        "Los niveles, los intervalos o la regla de signo declarados no son los del repositorio",
    )
    predictors = document["predictors"]
    _require(
        predictors["parent"]["model"] == document["parent"]
        and predictors["parent"]["seeds"] == [document["seed"]]
        and predictors["seeds"]["model"] == document["parent"]
        and document["seed"] in predictors["seeds"]["seeds"]
        and len(set(predictors["seeds"]["seeds"])) >= 2
        and all(
            len(predictors[r]["seeds"]) == 1 and predictors[r]["optional"] is True
            for r in REFERENCES
        )
        and document["references"]["roles"] == list(REFERENCES),
        "Los predictores declarados no son el padre de S2, sus semillas y las dos referencias",
    )
    layer = document["delivery"]["coverage_layer"]
    _require(
        document["delivery"]["widths_source"] == ["noic", "core"]
        and tuple(layer["levels"]) == (0.8, 0.9, 0.95)
        and all(set(pair) <= set(QUANTILE_COLUMNS) for pair in layer["bands"].values())
        and list(layer["bands"]) == [str(level) for level in layer["levels"]],
        "La entrega de F7-b no declara las anchuras ni la capa de cobertura de F",
    )
    return document, digest


# ---------------------------------------------------------------------------------------------
# Piezas puras


def weighted_quantile(values, weights, level):
    """El menor minimizador de Σ w·ρ_level(v − c): el primer valor ordenado cuyo peso acumulado
    llega a level·Σw."""
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    _require(
        values.shape == weights.shape and values.ndim == 1 and len(values) >= 1,
        "Valores y pesos deben ser vectores alineados y no vacíos",
    )
    _require(np.isfinite(values).all() and np.all(weights > 0), "Valores finitos y pesos positivos")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, level * cumulative[-1], side="left"))
    return float(values[order][min(index, len(values) - 1)])


def level_scales(residual, widths, weights):
    """Factor s_τ ≥ 0 de cada nivel que minimiza exactamente Σ w·ρ_τ(r − s·anchura).

    Con anchura a > 0, ρ_τ(r − s·a) = a·ρ_τ(r/a − s): el mínimo es el cuantil τ de r/a con pesos
    w·a. Con a < 0 es el cuantil 1 − τ con pesos w·|a|. Las filas de anchura nula no dependen de s.
    La mediana tiene anchura cero y factor 1.
    """
    residual, widths = np.asarray(residual, dtype=np.float64), np.asarray(widths, dtype=np.float64)
    _require(widths.shape == (len(residual), len(LEVELS)), "Anchuras de forma [filas, niveles]")
    scales = np.ones(len(LEVELS))
    for j, level in enumerate(LEVELS):
        if j == MEDIAN:
            continue
        width = widths[:, j]
        active = width != 0
        _require(
            np.all(width[active] < 0) if level < 0.5 else np.all(width[active] > 0),
            f"Las anchuras del nivel {level} no tienen su signo: la cabeza cruza la mediana",
        )
        if not active.any():
            continue
        ratio = residual[active] / width[active]
        side = level if level > 0.5 else 1.0 - level
        best = weighted_quantile(ratio, weights[active] * np.abs(width[active]), side)
        scales[j] = max(0.0, best)
    return scales


def recalibrated(center, widths, scales):
    """Cuantiles centrados en `center` con las anchuras por su factor; cada fila, ordenada."""
    values = np.asarray(center, dtype=np.float64)[:, None] + widths * scales
    return np.sort(values, axis=1)


def climatology(target, weights):
    """Cuantiles del objetivo con el peso de cada fila: los de la sesión repartidos por igual."""
    return np.array([weighted_quantile(target, weights, level) for level in LEVELS])


def metrics(scores):
    """Las métricas de un panel ya puntuado, planas y con nombres estables."""
    summary = scores.summary(market_weighting="session")
    point, quantiles = summary["point"], summary["quantiles"]
    result = dict(
        sessions=summary["sessions"],
        mae=point["mae"],
        mse=point["mse"],
        rmse=point["rmse"],
        rank_ic=point["rank_ic"],
        pinball=None,
    )
    if quantiles is None:
        return result
    intervals = {str(round(row["nominal"], 4)): row for row in quantiles["intervals"]}
    sign = quantiles["sign_probability"]
    result.update(
        pinball=dict(quantiles["pinball"]),
        mean_pinball=quantiles["mean_pinball"],
        coverage={key: row["coverage"] for key, row in intervals.items()},
        coverage_error={key: abs(row["coverage_gap"]) for key, row in intervals.items()},
        interval_score={key: row["interval_score"] for key, row in intervals.items()},
        sign_ece=sign["ece"],
        sign_brier=sign["brier"],
        level_frequency_gap=dict(quantiles["level_frequency_gap"]),
    )
    return result


def rule_values(item):
    """Las métricas de la regla de una versión, planas: el punto siempre y, con cuantiles, el
    resto."""
    values = dict(mae=item["mae"], rmse=item["rmse"])
    if item.get("pinball") is not None:
        values.update({f"pinball@{key}": value for key, value in item["pinball"].items()})
        values["mean_pinball"] = item["mean_pinball"]
        values.update({f"coverage_error@{k}": v for k, v in item["coverage_error"].items()})
        values["sign_ece"] = item["sign_ece"]
    return values


def worse(candidate, reference):
    """Métricas de la regla en las que `candidate` es estrictamente peor que `reference`.

    Se comparan las que tienen las dos versiones. Una métrica sin definir en la candidata cuenta
    como peor.
    """
    mine, theirs = rule_values(candidate), rule_values(reference)
    return [
        name
        for name, value in mine.items()
        if name in theirs and theirs[name] is not None and (value is None or value > theirs[name])
    ]


def ladder(versions, reference):
    """La versión más completa que no empeora ninguna métrica de la regla frente a `reference`."""
    failing = {
        name: worse(versions[name], reference) for name in ("full", "point") if name in versions
    }
    delivered = next(
        (name for name in ("full", "point") if name in failing and not failing[name]), None
    )
    return dict(delivered=delivered, worse=failing)


def relative(item, control):
    """Mejora relativa 1 − M/M_c en las pérdidas y diferencia M − M_c en la calibración."""

    def ratio(mine, theirs):
        return None if mine is None or not theirs else 1.0 - mine / theirs

    result = dict(mae=ratio(item["mae"], control["mae"]), rmse=ratio(item["rmse"], control["rmse"]))
    if item.get("pinball") is not None and control.get("pinball") is not None:
        result["mean_pinball"] = ratio(item["mean_pinball"], control["mean_pinball"])
        result["pinball"] = {k: ratio(v, control["pinball"][k]) for k, v in item["pinball"].items()}
        result["coverage_error"] = {
            k: v - control["coverage_error"][k] for k, v in item["coverage_error"].items()
        }
        result["sign_ece"] = item["sign_ece"] - control["sign_ece"]
    return result


def paired(tool, left, right, metric):
    """`left` − `right` por sesión en una métrica (mae, mse o pinball): media y t de Newey-West."""
    _require(left.cohort_sha256 == right.cohort_sha256, "Un contraste necesita las mismas filas")
    a, b = left.series(metric), right.series(metric)
    keep = a.defined & b.defined
    keys = [
        (left.markets[int(m)], int(t))
        for m, t in zip(left.session_market, left.session_time, strict=True)
    ]
    stats = tool.summary(
        {k: float(x - y) for k, x, y, d in zip(keys, a.values, b.values, keep, strict=True) if d}
    )
    return dict(sessions=int(keep.sum()), mean=stats["mean"], t_nw=stats["t_nw"])


def rmse_bound(pearson):
    """La rebaja máxima del RMSE de predecir cero con un predictor lineal de correlación ρ."""
    return (
        None if pearson is None or abs(pearson) >= 1 else 1.0 - math.sqrt(1.0 - pearson * pearson)
    )


# ---------------------------------------------------------------------------------------------
# Entradas


def folders(declaration, state, reference, organs, origin):
    """Las carpetas obligatorias (las de S2 y las otras semillas del padre) y las de referencia."""
    sources = og.folders(declaration, state, reference, organs, origin)
    seeds = declaration["predictors"]["seeds"]
    for seed in seeds["seeds"]:
        if seed != declaration["seed"]:
            sources[f"seed/{seed}"] = (
                Path(state) / "runs" / seeds["model"] / str(origin) / f"seed-{seed}"
            )
    references = {}
    for role in REFERENCES:
        item = declaration["predictors"][role]
        references[role] = (
            Path(state) / "runs" / item["model"] / str(origin) / f"seed-{item['seeds'][0]}"
        )
    return sources, references


def checked_extra(declaration, sources, records):
    """Las otras semillas y las referencias: terminadas, sin test final y con la vista del padre."""
    view = (records["parent"].get("request") or {}).get("view_sha256")
    for role, record in records.items():
        if not (role.startswith("seed/") or role in REFERENCES):
            continue
        if role.startswith("seed/"):
            seed = int(role.split("/")[1])
        else:
            seed = declaration["predictors"][role]["seeds"][0]
        request = record.get("request") or {}
        _require(
            record.get("status") == "completed"
            and record.get("final_test_opened") is False
            and request.get("view_sha256") == view
            and request.get("seed") == seed,
            f"{sources[role]} no es una ejecución terminada, sin test final, de la semilla {seed} "
            "con la vista del padre",
        )


def _ordered(frame):
    """El marco en el orden canónico de ForecastPanel, comprobado con una sonda."""

    def probe(item):
        return ForecastPanel.from_columns(
            pa.array(item["ids"]),
            item["market"],
            item["instants"],
            item["target"],
            np.arange(len(item["ids"]), dtype=np.float64),
            markets=item["markets"],
        ).prediction.astype(np.int64)

    order = probe(frame)
    result = _subset(frame, order)
    _require(
        np.array_equal(probe(result), np.arange(len(order))), "El orden canónico no es estable"
    )
    return result


def _subset(frame, rows):
    """Las filas `rows` de un marco, en ese orden."""
    count = len(frame["ids"])
    result = {
        key: (value[rows] if isinstance(value, np.ndarray) and len(value) == count else value)
        for key, value in frame.items()
    }
    result["ids"] = [frame["ids"][i] for i in rows]
    result["scores"] = {k: v[rows] for k, v in frame["scores"].items()}
    result["heads"] = {k: v[rows] for k, v in frame["heads"].items()}
    return result


def _panel(frame, point, quantiles=None):
    return ForecastPanel.from_columns(
        pa.array(frame["ids"]),
        frame["market"],
        frame["instants"],
        frame["target"],
        np.asarray(point, dtype=np.float64),
        quantiles=quantiles,
        levels=None if quantiles is None else LEVELS,
        markets=frame["markets"],
    )


def _by_market(values, market, markets):
    """Un valor por fila desde {mercado: valor}; un vector por fila si el valor es un vector."""
    first = np.asarray(next(iter(values.values())), dtype=np.float64)
    result = np.zeros((len(market), *first.shape))
    for name in markets:
        rows = market == name
        if rows.any():
            result[rows] = values[name]
    return result


def _frame(
    mode,
    split,
    results,
    table,
    points,
    quantiles,
    alpha,
    volatility,
    matures,
    index,
    factors,
    markets,
):
    """Las filas de S2 de un modo y un tramo, con el objetivo, las puntuaciones y las cabezas."""
    aligned = results["s2"]["aligned"][split]
    ids = list(aligned["sample_id"])
    rows = np.array([index[split][s] for s in ids], dtype=np.int64)
    market = np.array(table[split].column("market").to_pylist(), dtype=object)[rows]
    instants = table[split].column("prediction_at").cast(pa.int64()).to_numpy()[rows]
    label = table[split].column("target").to_numpy().astype(np.float64)[rows]
    known = np.asarray(alpha[split], dtype=np.float64)[rows]
    operable = mode == "operable"
    target = label + known if operable else label
    _require(
        np.allclose(target, np.asarray(aligned["target"], dtype=np.float64), rtol=1e-5, atol=1e-7),
        f"El objetivo de {mode}/{split} no es el de las filas de S2",
    )
    for m, t, start, end in aligned["sessions"]:
        _require(
            np.all(market[start:end] == m) and np.all(instants[start:end] == int(t)),
            f"Las sesiones de S2 de {mode}/{split} no son las de sus filas",
        )
    scores = {
        p: (known + factors[p] * v[split][rows] if operable else v[split][rows])
        for p, v in points.items()
    }
    for system in ("s1", "s2"):
        game = results[system]["aggregators"][rel.GAMES[system]]
        scores[system] = np.asarray(game["split_predictions"][split], dtype=np.float64)
    heads = {
        p: q[split][rows] + (known[:, None] if operable else 0.0) for p, q in quantiles.items()
    }
    for p, values in {**scores, **heads}.items():
        _require(
            np.isfinite(values).all(), f"{p} en {mode}/{split} no es finito en todas las filas"
        )
    return _ordered(
        dict(
            markets=markets,
            ids=ids,
            market=market,
            instants=instants,
            target=target,
            label=label,
            alpha=known,
            matures=np.asarray(matures[split], dtype=np.int64)[rows],
            volatility=np.asarray(volatility[split], dtype=np.float64)[rows],
            scores=scores,
            heads=heads,
        )
    )


def _versions(cal, ev, predictor, min_assets):
    """Ajuste en calibración y paneles de evaluación de las versiones de un predictor."""
    head = predictor if predictor in HEADS else "parent"
    cal_panel = _panel(cal, cal["scores"][predictor])
    factors = sd.market_shrinkage(cal_panel, cal_panel.prediction)
    weights = sd.session_weights(cal_panel)
    widths_cal = cal["heads"][head] - cal["heads"][head][:, [MEDIAN]]
    widths_ev = ev["heads"][head] - ev["heads"][head][:, [MEDIAN]]
    present = [m for m in cal["markets"] if np.any(cal["market"] == m)]
    usable = all(factors.get(m) is not None and factors[m] > 0 for m in present)
    parameters = dict(lad=factors, level_scales={})
    panels = {}
    if predictor in HEADS:
        panels["raw"] = _panel(ev, ev["scores"][predictor], ev["heads"][predictor])
    if usable:
        center_cal = cal["scores"][predictor] * _by_market(factors, cal["market"], present)
        for m in present:
            rows = cal["market"] == m
            parameters["level_scales"][m] = level_scales(
                cal["target"][rows] - center_cal[rows], widths_cal[rows], weights[rows]
            ).tolist()
        center = ev["scores"][predictor] * _by_market(factors, ev["market"], present)
        scales = _by_market(parameters["level_scales"], ev["market"], present)
        head_ev = ev["heads"][predictor] if predictor in HEADS else None
        panels["point"] = _panel(ev, center, head_ev)
        panels["full"] = _panel(ev, center, recalibrated(center, widths_ev, scales))
    score_panel = _panel(ev, ev["scores"][predictor])
    pearson, defined = sd.session_correlations(
        score_panel, score_panel.prediction, min_assets=min_assets
    )
    rho = float(pearson[defined].mean()) if defined.any() else None
    scored = {
        name: fs.score_sessions(panel, rank_ic_min_assets=min_assets)
        for name, panel in panels.items()
    }
    return scored, parameters, usable, rho


def reaches(item, level):
    """Si una versión iguala o mejora el nivel de la referencia noic tal como se emite."""
    mae = item["mae"] <= level["mae"]
    pinball = None if item.get("pinball") is None else item["mean_pinball"] <= level["mean_pinball"]
    return dict(
        mae=mae,
        mean_pinball=pinball,
        mae_versus=1.0 - item["mae"] / level["mae"],
        mean_pinball_versus=(
            None
            if item.get("pinball") is None
            else 1.0 - item["mean_pinball"] / level["mean_pinball"]
        ),
        rank_ic=item["rank_ic"],
        rank_ic_reference=level["rank_ic"],
    )


def _session_rows(mode, predictor, version, scores, markets):
    return [
        dict(
            mode=mode,
            predictor=predictor,
            version=version,
            market=markets[int(m)],
            prediction_at=int(t),
            mae=float(scores.mae[k]),
            mse=float(scores.mse[k]),
            pinball=None if scores.pinball is None else float(scores.pinball[k].mean()),
        )
        for k, (m, t) in enumerate(zip(scores.session_market, scores.session_time, strict=True))
    ]


def _matured(frames, measured, mode):
    """Las filas de calibración cuya etiqueta madura antes de la primera evaluación de su mercado."""
    cal_all, ev = frames["calibration"], frames["evaluation"]
    first = {
        m: int(ev["instants"][ev["market"] == m].min())
        for m in measured
        if np.any(ev["market"] == m)
    }
    matured = np.array(
        [
            m in first and t <= first[m]
            for m, t in zip(cal_all["market"], cal_all["matures"], strict=True)
        ]
    )
    _require(
        matured.any(), f"Ninguna etiqueta de calibración de {mode} madura antes de la evaluación"
    )
    return _subset(cal_all, np.flatnonzero(matured))


def _mode(tool, mode, frames, min_assets, measured, predictors):
    """Controles, versiones, contrastes y regla de un modo."""
    operable = mode == "operable"
    cal_all, ev = frames["calibration"], frames["evaluation"]
    cal = _matured(frames, measured, mode)
    weights = sd.session_weights(_panel(cal, np.zeros(len(cal["ids"]))))
    present = [m for m in measured if np.any(cal["market"] == m)]

    def climate(values):
        return {
            m: climatology(values[cal["market"] == m], weights[cal["market"] == m]) for m in present
        }

    controls = dict(
        zero=_panel(
            ev, np.zeros(len(ev["ids"])), _by_market(climate(cal["target"]), ev["market"], present)
        )
    )
    if operable:
        label = _by_market(climate(cal["label"]), ev["market"], present)
        controls["alpha"] = _panel(ev, ev["alpha"], ev["alpha"][:, None] + label)
    control_scores = {
        name: fs.score_sessions(panel, rank_ic_min_assets=min_assets)
        for name, panel in controls.items()
    }
    control_metrics = {name: metrics(s) for name, s in control_scores.items()}
    entry = dict(
        rows=dict(
            calibration=len(cal_all["ids"]), matured=len(cal["ids"]), evaluation=len(ev["ids"])
        ),
        controls=control_metrics,
        predictors={},
    )
    rows = []
    for name, s in control_scores.items():
        rows += _session_rows(mode, f"control_{name}", "control", s, measured)
    for predictor in predictors:
        scored, parameters, usable, rho = _versions(cal, ev, predictor, min_assets)
        versions = {}
        for name, s in scored.items():
            item = metrics(s)
            item["versus"] = {c: relative(item, control_metrics[c]) for c in CONTROLS[mode]}
            others = list(control_scores.items())
            if name != "raw" and "raw" in scored:
                others.insert(0, ("raw", scored["raw"]))
            item["paired"] = {
                other: {
                    m: paired(tool, s, theirs, m)
                    for m in PAIRED
                    if m != "pinball" or (s.pinball is not None and theirs.pinball is not None)
                }
                for other, theirs in others
            }
            versions[name] = item
            rows += _session_rows(mode, predictor, name, s, measured)
        if "raw" in versions and "point" in versions:
            _require(
                math.isclose(
                    versions["raw"]["rank_ic"], versions["point"]["rank_ic"], abs_tol=1e-12
                ),
                f"La calibración de {predictor} en {mode} cambió el orden de la sesión",
            )
        reference = "raw" if "raw" in versions else REFERENCE[mode]
        decision = ladder(
            versions, versions["raw"] if "raw" in versions else control_metrics[reference]
        )
        if decision["delivered"] is None and "raw" in versions:
            decision["delivered"] = "raw"
        decision.update(reference=reference, calibrated=usable)
        entry["predictors"][predictor] = dict(
            parameters=parameters,
            pearson_ic=rho,
            rmse_bound=rmse_bound(rho),
            versions=versions,
            verdict=decision,
        )
    if "noic" in entry["predictors"]:
        level = entry["predictors"]["noic"]["versions"]["raw"]
        for predictor, item in entry["predictors"].items():
            delivered = item["verdict"]["delivered"]
            if predictor != "noic" and delivered is not None:
                item["verdict"]["noic"] = reaches(item["versions"][delivered], level)
    return entry, rows


def measure(
    declaration, state, reference, organs, origin, output, *, wait_hours, poll, log, markets=None
):
    """F7 de un origen, publicado en `<output>/<origen>` (o `<origen>-<mercados>`)."""
    declaration, digest = checked_declaration(declaration)
    sources, references = folders(declaration, state, reference, organs, origin)
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    missing = og.wait_for_inputs(
        declaration, sources, manifest, hours=wait_hours, poll=poll, say=log
    )
    if missing:
        raise og.MissingInputs(missing)
    absent = {}
    for role, folder in references.items():
        lacking = og.missing_inputs(declaration, {role: folder}, manifest)
        if lacking:
            absent[role] = lacking[0]
        else:
            sources[role] = folder
    fingerprint = og._fingerprint(sources, manifest)
    target = Path(output) / og.target_name(origin, markets)
    if target.exists():
        return og._already(target, digest, fingerprint, log)
    started = time.monotonic()

    def say(text):
        log(f"{time.monotonic() - started:6.0f} s {text}")

    for role, reason in absent.items():
        say(f"referencia {role} ausente: {reason}")
    records = {
        role: json.loads((folder / "run.json").read_text(encoding="utf-8"))
        for role, folder in sources.items()
    }
    og._checked_sources(declaration, origin, sources, records, fingerprint)
    checked_extra(declaration, sources, records)
    tool = og._screening()
    seed_roles = sorted(
        (r for r in sources if r.startswith("seed/")), key=lambda r: int(r.split("/")[1])
    )
    included = [r for r in REFERENCES if r in sources]
    model_roles = ["parent", *seed_roles, *included]
    full = {
        role: {
            s: pq.read_table(sources[role] / f"{s}-predictions.parquet").sort_by(og.ORDER)
            for s in AGGREGATED
        }
        for role in model_roles
    }
    tables = dict(
        tree={s: og._table(sources["tree"], s) for s in AGGREGATED},
        organ={s: og._table(sources["organ"], s) for s in AGGREGATED},
    )
    present = og.markets_of(full["parent"])
    measured = og.measured_markets(present, markets)
    if measured != present:
        full = {
            r: {s: og.only(t, measured) for s, t in splits.items()} for r, splits in full.items()
        }
        tables = {
            r: {s: og.only(t, measured) for s, t in splits.items()} for r, splits in tables.items()
        }
    say(f"mercados medidos: {measured} (el estado tiene {present})")
    parent = {s: full["parent"][s].select(og.COLUMNS) for s in AGGREGATED}
    for role, splits in [*tables.items(), *((r, full[r]) for r in model_roles[1:])]:
        for split, table in splits.items():
            for name in og.KEYS:
                _require(
                    table.column(name).equals(parent[split].column(name)),
                    f"{sources[role]} no tiene las filas del padre en {split} (columna {name})",
                )
    for role in model_roles:
        _require(
            all(set(QUANTILE_COLUMNS) <= set(full[role][s].column_names) for s in AGGREGATED),
            f"{sources[role]} no emite los cuantiles {QUANTILE_COLUMNS}",
        )
    view = records["parent"]["request"]["view_sha256"]
    alpha, _, volatility, matures = ac.known_rows(state, origin, parent, view)
    say("alfa conocido y madurez")

    def column(role, split, name):
        return full[role][split].column(name).to_numpy().astype(np.float64)

    def head(role, split):
        return np.column_stack([column(role, split, q) for q in QUANTILE_COLUMNS])

    # La media de semillas del repositorio: en orden ascendente de semilla, nivel a nivel.
    seed_order = sorted(
        ["parent", *seed_roles],
        key=lambda r: declaration["seed"] if r == "parent" else int(r.split("/")[1]),
    )
    points = dict(
        parent={s: column("parent", s, "prediction") for s in AGGREGATED},
        seeds={
            s: se.ordered_mean([column(r, s, "prediction") for r in seed_order]) for s in AGGREGATED
        },
        **{r: {s: column(r, s, "prediction") for s in AGGREGATED} for r in included},
    )
    quantiles = dict(
        parent={s: head("parent", s) for s in AGGREGATED},
        seeds={s: se.ordered_mean([head(r, s) for r in seed_order]) for s in AGGREGATED},
        **{r: {s: head(r, s) for s in AGGREGATED} for r in included},
    )
    own = points["parent"]
    predictions = dict(
        parent=own,
        tree={
            s: tables["tree"][s].column("prediction").to_numpy().astype(np.float64)
            for s in AGGREGATED
        },
        signal={
            s: tables["organ"][s].column("prediction").to_numpy().astype(np.float64) - own[s]
            for s in AGGREGATED
        },
        seeds=points["seeds"],
        **{r: points[r] for r in included},
    )
    names = dict(
        parent=declaration["parent"],
        tree=declaration["tree"],
        signal=og.SIGNAL,
        seeds=f"{declaration['parent']}_semillas",
        **{r: declaration["predictors"][r]["model"] for r in included},
    )
    index = {
        s: {sid: i for i, sid in enumerate(parent[s].column("sample_id").to_pylist())}
        for s in AGGREGATED
    }
    min_assets = declaration["min_assets"]
    present_predictors = [p for p in PREDICTORS if p in points or p in ("s1", "s2")]
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        work.mkdir(parents=True)
        inputs = {
            p: og._write_component(
                work / "entradas" / names[p], parent, predictions[p], alpha, matures
            )
            for p in names
        }

        def aggregate(members, operable):
            return oa.run_aggregation(
                {names[p]: inputs[p] for p in members},
                splits=tuple(declaration["splits"]),
                selection_split=declaration["selection_split"],
                min_assets=min_assets,
                align=declaration["align"],
                operable=operable,
                alpha_col="alpha",
                maturity_col="matures_at",
                placebo_seed=declaration["aggregator_placebo_seed"],
                operable_scales=tuple(declaration["operable_scales"]),
            )

        modes, rows, operable_factors, kept = {}, [], {}, {}
        for mode in MODES:
            operable = mode == "operable"
            results = {system: aggregate(members, operable) for system, members in rel.SYSTEMS}
            factors = {p: 1.0 for p in points}
            if operable:
                # El k de cada modelo con el criterio del agregador; el del padre debe ser el de S2.
                chosen = aggregate(("parent", "seeds", *included), True)["operable"]["scales"]
                factors = {p: float(chosen[names[p]]) for p in points}
                _require(
                    factors["parent"]
                    == float(results["s2"]["operable"]["scales"][names["parent"]]),
                    "El factor operable del padre no es el de S2",
                )
                operable_factors = factors
            say(f"{mode}: S1 y S2 agregados")
            frames = {
                split: _frame(
                    mode,
                    split,
                    results,
                    full["parent"],
                    points,
                    quantiles,
                    alpha,
                    volatility,
                    matures,
                    index,
                    factors,
                    tuple(measured),
                )
                for split in AGGREGATED
            }
            kept[mode] = frames
            modes[mode], mode_rows = _mode(
                tool, mode, frames, min_assets, tuple(measured), present_predictors
            )
            rows += mode_rows
            for predictor, item in modes[mode]["predictors"].items():
                say(f"{mode}/{predictor}: entrega {item['verdict']['delivered']}")
        handed = delivery(kept, declaration, min_assets, tuple(measured), included)
        say(f"F7-b: {handed.get('verdict', handed.get('status'))}")
        shutil.rmtree(work / "entradas")
        code = og._code_identity()
        for name, module in (
            ("error_metrics", sys.modules[__name__]),
            ("forecast_scores", fs),
            ("scale_diagnostics", sd),
        ):
            code["files"][name] = pilots._sha256(Path(module.__file__))
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
                references=dict(included=included, absent=absent),
                operable_factors=operable_factors,
                modes=modes,
                delivery=handed,
                seconds=round(time.monotonic() - started),
            )
        )
        pq.write_table(ac._series_table(rows), work / "sesiones.parquet")
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
# F7-b: la entrega con el orden de S2, el nivel LAD y las anchuras del modelo solo pinball


def _sessions_rows(frame):
    """(mercado, instante, inicio, fin) de las sesiones de un marco en orden canónico."""
    market, instants = frame["market"], frame["instants"]
    change = np.r_[True, (market[1:] != market[:-1]) | (instants[1:] != instants[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(instants)]
    return [
        (str(market[s]), int(instants[s]), int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
    ]


def halves(frame):
    """Las dos mitades en el tiempo de un tramo: los ⌊n/2⌋ primeros instantes y el resto."""
    instants = np.unique(frame["instants"])
    _require(len(instants) >= 4, "La calibración necesita al menos cuatro sesiones para partirse")
    cut = int(instants[len(instants) // 2 - 1])
    return (
        _subset(frame, np.flatnonzero(frame["instants"] <= cut)),
        _subset(frame, np.flatnonzero(frame["instants"] > cut)),
        cut,
    )


def candidate_fit(fit, source, present):
    """λ de S2 y factores de nivel de las anchuras de `source`, ajustados en `fit`."""
    panel = _panel(fit, fit["scores"]["s2"])
    factors = sd.market_shrinkage(panel, panel.prediction)
    _require(
        all(factors.get(m) is not None and factors[m] > 0 for m in present),
        "El factor LAD de S2 es nulo: no hay entrega en unidades de rendimiento",
    )
    weights = sd.session_weights(panel)
    center = fit["scores"]["s2"] * _by_market(factors, fit["market"], present)
    widths = fit["heads"][source] - fit["heads"][source][:, [MEDIAN]]
    scales = {}
    for m in present:
        rows = fit["market"] == m
        scales[m] = level_scales(
            fit["target"][rows] - center[rows], widths[rows], weights[rows]
        ).tolist()
    return dict(lad=factors, level_scales=scales)


def candidate_head(frame, fitted, source, present):
    """Centro λ·S2 y cabeza candidata de un marco con los parámetros ajustados."""
    center = frame["scores"]["s2"] * _by_market(fitted["lad"], frame["market"], present)
    widths = frame["heads"][source] - frame["heads"][source][:, [MEDIAN]]
    return center, recalibrated(
        center, widths, _by_market(fitted["level_scales"], frame["market"], present)
    )


def coverage_layer(fit, run, fit_head, run_head, layer):
    """Mondrian+ACI de F sobre una cabeza: puntuaciones conformes en `fit` y ACI en `run`."""
    fit_sessions, sessions = _sessions_rows(fit), _sessions_rows(run)
    groups_fit = ac.within_session_groups(
        fit["volatility"], ac.bounds_of(fit_sessions), layer["volatility_groups"]
    )
    groups = ac.within_session_groups(
        run["volatility"], ac.bounds_of(sessions), layer["volatility_groups"]
    )
    maturity = np.array([int(run["matures"][s:e].max()) for _, _, s, e in sessions])
    markets = np.asarray(run["market"]).astype(str)
    result = {}
    for level in layer["levels"]:
        low, high = (QUANTILE_COLUMNS.index(name) for name in layer["bands"][str(level)])
        scores = ac.conformity(fit_head[:, low], fit_head[:, high], fit["target"])
        lower, upper, path = rel.adaptive_bands(
            run_head[:, low],
            run_head[:, high],
            run["target"],
            sessions,
            maturity,
            scores,
            level,
            layer["aci_gamma"],
            groups=groups,
            score_groups=groups_fit,
        )
        item = dict(
            **ac.interval_summary(lower, upper, run["target"]),
            volatility=ac.interval_summary(lower, upper, run["target"], groups)["groups"],
            market=ac.interval_summary(lower, upper, run["target"], markets)["groups"],
            aci=path,
        )
        item["coverage_error"] = abs(item["coverage"] - level)
        item["interval_score"] = float(
            np.mean(pilots.interval_score_rows(lower, upper, run["target"], level))
        )
        item["reasons"] = rel.interval_rule(item, level, layer["rule"])
        result[str(level)] = item
    return result


def raw_bands(frame, head, layer):
    """Cobertura de las bandas de una cabeza sin capa, en los niveles que tienen sus dos cuantiles."""
    result = {}
    for level in layer["levels"]:
        low, high = (QUANTILE_COLUMNS.index(name) for name in layer["bands"][str(level)])
        if not math.isclose(LEVELS[high] - LEVELS[low], level):
            continue
        item = ac.interval_summary(head[:, low], head[:, high], frame["target"])
        result[str(level)] = dict(
            coverage=item["coverage"],
            coverage_error=abs(item["coverage"] - level),
            width=item["width"],
        )
    return result


def _flat(entry):
    """Las métricas que decide la regla de F7-b, planas."""
    values = {f"pinball@{k}": v for k, v in entry["pinball"].items()}
    values["mean_pinball"] = entry["mean_pinball"]
    for key in ("sign_ece", "sign_ece_operable", "mae", "rmse"):
        if key in entry:
            values[key] = entry[key]
    values.update({f"coverage_error@{k}": v["coverage_error"] for k, v in entry["layer"].items()})
    return values


def delivery(frames, declaration, min_assets, measured, included):
    """F7-b: la candidata (orden de S2, nivel LAD, anchuras del noic o del núcleo) frente a lo que
    ya entra (cabeza del padre con el Mondrian+ACI de F)."""
    settings = declaration["delivery"]
    source = next((r for r in settings["widths_source"] if r in included), None)
    if source is None:
        return dict(status="absent", reason="no están emitidos ni el noic ni el núcleo")
    layer = settings["coverage_layer"]
    label, operable = frames["label"], frames["operable"]
    cal, ev = _matured(label, measured, "label"), label["evaluation"]
    present = [m for m in measured if np.any(cal["market"] == m)]
    # Decisión en calibración: se ajusta en la primera mitad y se puntúa en la segunda.
    first, second, cut = halves(cal)
    fitted = candidate_fit(first, source, present)
    _, fit_head = candidate_head(first, fitted, source, present)
    center, run_head = candidate_head(second, fitted, source, present)
    decision = {}
    for name, heads in (
        ("incumbent", (first["heads"]["parent"], second["heads"]["parent"])),
        ("candidate", (fit_head, run_head)),
    ):
        scored = metrics(
            fs.score_sessions(_panel(second, center, heads[1]), rank_ic_min_assets=min_assets)
        )
        decision[name] = dict(
            mean_pinball=scored["mean_pinball"],
            pinball=scored["pinball"],
            layer=coverage_layer(first, second, heads[0], heads[1], layer),
        )
    decision["candidate_without_layer"] = raw_bands(second, run_head, layer)
    keys = ["mean_pinball", *(f"coverage_error@{level}" for level in layer["levels"])]
    mine, theirs = _flat(decision["candidate"]), _flat(decision["incumbent"])
    worse_calibration = [k for k in keys if mine[k] > theirs[k]]
    chosen = "incumbent" if worse_calibration else "candidate"
    # Evaluación: todo se ajusta otra vez con la calibración entera, como en F.
    fitted_all = candidate_fit(cal, source, present)
    _, cal_head = candidate_head(cal, fitted_all, source, present)
    center_ev, ev_head = candidate_head(ev, fitted_all, source, present)
    op_ev = operable["evaluation"]
    position = {sid: i for i, sid in enumerate(ev["ids"])}
    rows = np.array([position[s] for s in op_ev["ids"]], dtype=np.int64)
    evaluation = {}
    for name, heads in (
        ("incumbent", (cal["heads"]["parent"], ev["heads"]["parent"])),
        ("candidate", (cal_head, ev_head)),
    ):
        scored = metrics(
            fs.score_sessions(_panel(ev, center_ev, heads[1]), rank_ic_min_assets=min_assets)
        )
        shifted = metrics(
            fs.score_sessions(
                _panel(
                    op_ev,
                    op_ev["alpha"] + center_ev[rows],
                    op_ev["alpha"][:, None] + heads[1][rows],
                ),
                rank_ic_min_assets=min_assets,
            )
        )
        evaluation[name] = dict(
            mae=scored["mae"],
            rmse=scored["rmse"],
            mean_pinball=scored["mean_pinball"],
            pinball=scored["pinball"],
            sign_ece=scored["sign_ece"],
            sign_ece_operable=shifted["sign_ece"],
            layer=coverage_layer(cal, ev, heads[0], heads[1], layer),
        )
    evaluation["candidate_without_layer"] = raw_bands(ev, ev_head, layer)
    mine, theirs = _flat(evaluation["candidate"]), _flat(evaluation["incumbent"])
    worse_evaluation = [k for k, v in mine.items() if v > theirs[k]]
    superfluous = {
        level: bool(
            decision["candidate_without_layer"][level]["coverage_error"]
            <= decision["candidate"]["layer"][level]["coverage_error"]
            and evaluation["candidate_without_layer"][level]["coverage_error"]
            <= evaluation["candidate"]["layer"][level]["coverage_error"]
        )
        for level in decision["candidate_without_layer"]
    }
    enters = chosen == "candidate" and not worse_evaluation
    model = declaration["predictors"][source]["model"]
    return dict(
        status="measured",
        widths_source=source,
        widths_model=model,
        calibration=dict(
            cut=cut,
            first_rows=len(first["ids"]),
            second_rows=len(second["ids"]),
            fitted=fitted,
            **decision,
        ),
        chosen_in_calibration=chosen,
        worse_in_calibration=worse_calibration,
        evaluation=dict(fitted=fitted_all, **evaluation),
        worse_in_evaluation=worse_evaluation,
        enters=enters,
        layer_superfluous=superfluous,
        verdict="entra" if enters else "no entra",
        provenance=dict(
            order="S2 (ridge_cov_active de padre, árboles y señal del órgano)",
            level="escala LAD de S2",
            widths=f"{model}, semilla {declaration['predictors'][source]['seeds'][0]}",
            coverage="Mondrian+ACI de F",
        ),
    )


def _f(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _pct(value):
    return "—" if value is None else f"{100 * value:+.1f} %".replace(".", ",").replace("-", "−")


NAMES = dict(
    parent="padre (titans_mac_nogain)",
    seeds="semillas (media de 3)",
    core="núcleo de la criba (titans_mac)",
    noic="solo pinball (titans_mac_noic)",
    s1="S1",
    s2="S2",
)
VERSION_NAMES = dict(raw="sin calibrar", point="punto", full="completa")
MODE_NAMES = dict(label="Etiqueta", operable="Operable")


def _yes(value):
    return "—" if value is None else ("sí" if value else "no")


def markdown(summary):
    """Resumen legible de F7 de un origen."""
    lines = [
        f"# F7, las métricas de error, en el origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration_sha256'][:12]}; mercados {', '.join(summary['markets'])}; "
        "evaluación. Las métricas son las de `forecast_scores`, con cada sesión con el mismo peso. "
        "La mejora sobre un control es 1 − M/M_control en las pérdidas y la diferencia "
        "M − M_control en la cobertura y el ECE.",
        "",
        "## Lo que cabe esperar",
        "",
    ]
    for predictor, item in summary["modes"]["label"]["predictors"].items():
        lines.append(
            f"- {NAMES[predictor]}: IC de Pearson medio {_f(item['pearson_ic'], 3)}. Con ese ρ, el "
            "mejor predictor lineal rebaja el RMSE de predecir cero como mucho "
            f"1 − √(1 − ρ²) = {_pct(item['rmse_bound']).lstrip('+')}."
        )
    absent = summary["references"]["absent"]
    lines += [
        "- La etiqueta es casi todo ruido. Las mejoras posibles de MAE y RMSE sobre predecir cero "
        "son del orden del 1 %, y este informe no promete más.",
        *(f"- Referencia ausente, {NAMES[r]}: {why}." for r, why in absent.items()),
        "",
    ]
    lines += _delivery_lines(summary.get("delivery") or {})
    for mode, entry in summary["modes"].items():
        controls = CONTROLS[mode]
        lines += [
            f"## {MODE_NAMES[mode]}",
            "",
            f"Filas: calibración {entry['rows']['calibration']} ({entry['rows']['matured']} con la "
            f"etiqueta madura antes de la evaluación) y evaluación {entry['rows']['evaluation']}.",
            "",
            "| Predictor | Versión | MAE | "
            + " | ".join(f"MAE vs {c}" for c in controls)
            + " | RMSE | "
            + " | ".join(f"RMSE vs {c}" for c in controls)
            + " | Pinball media | Pinball vs climatología | Error de cobertura 80 | "
            "Error de cobertura 95 | ECE | IC de rangos |",
            "|---|---|---|"
            + "---|" * len(controls)
            + "---|"
            + "---|" * len(controls)
            + "---|---|---|---|---|---|",
        ]
        rows = [("control", name, value, None) for name, value in entry["controls"].items()]
        for predictor, item in entry["predictors"].items():
            rows += [
                (predictor, v, value, value["versus"]) for v, value in item["versions"].items()
            ]
        for predictor, version, value, versus in rows:
            name = f"control {version}" if predictor == "control" else NAMES[predictor]
            vs = versus or {}
            pin = value.get("pinball") is not None
            quantile_cells = (
                f"{_f(value['mean_pinball'])} | {_pct((vs.get('zero') or {}).get('mean_pinball'))} | "
                f"{_f(value['coverage_error']['0.8'], 4)} | {_f(value['coverage_error']['0.95'], 4)} | "
                f"{_f(value['sign_ece'], 4)}"
                if pin
                else "— | — | — | — | —"
            )
            lines.append(
                f"| {name} | {'—' if predictor == 'control' else VERSION_NAMES[version]} | "
                f"{_f(value['mae'])} | "
                + " | ".join(_pct((vs.get(c) or {}).get("mae")) for c in controls)
                + f" | {_f(value['rmse'])} | "
                + " | ".join(_pct((vs.get(c) or {}).get("rmse")) for c in controls)
                + f" | {quantile_cells} | {_f(value['rank_ic'], 4)} |"
            )
        lines += [
            "",
            "**Entrega por la regla** (estricta: ninguna métrica de la regla peor que la referencia):",
            "",
        ]
        for predictor, item in entry["predictors"].items():
            verdict = item["verdict"]
            best = verdict["delivered"]
            shown = VERSION_NAMES[best] if best else "ninguna en unidades de rendimiento"
            reasons = "; ".join(
                f"la {VERSION_NAMES[v]} empeora {', '.join(w)}"
                for v, w in verdict["worse"].items()
                if w
            )
            lambdas = ", ".join(f"{m} {_f(v, 4)}" for m, v in item["parameters"]["lad"].items())
            text = f"- {NAMES[predictor]}: entrega **{shown}** (referencia: {verdict['reference']}; λ {lambdas})."
            if reasons:
                text += f" {reasons[0].upper()}{reasons[1:]}."
            if best in ("point", "full") and verdict["reference"] == "raw":
                contrast = item["versions"][best]["paired"]["raw"]
                text += (
                    " Frente a la sin calibrar, por sesión: "
                    + "; ".join(
                        f"{m} {_f(v['mean'], 7)} (t {_f(v['t_nw'], 1)})"
                        for m, v in contrast.items()
                    )
                    + "."
                )
            lines.append(text)
        level = [
            (p, i["verdict"]["noic"])
            for p, i in entry["predictors"].items()
            if "noic" in i["verdict"]
        ]
        if level:
            lines += [
                "",
                "**Nivel de referencia: el noic tal como se emite** (el mejor error con estos datos). "
                "La versión entregada lo iguala si su MAE y su pinball media no son mayores; el orden "
                "del modelo no cambia con la calibración.",
                "",
                "| Predictor (entregado) | MAE vs noic | ¿Iguala el MAE? | Pinball vs noic | ¿Iguala la pinball? | IC de rangos | IC del noic |",
                "|---|---|---|---|---|---|---|",
            ]
            for predictor, item in level:
                version = entry["predictors"][predictor]["verdict"]["delivered"]
                lines.append(
                    f"| {NAMES[predictor]} ({VERSION_NAMES[version]}) | {_pct(item['mae_versus'])} | "
                    f"{_yes(item['mae'])} | {_pct(item['mean_pinball_versus'])} | "
                    f"{_yes(item['mean_pinball'])} | {_f(item['rank_ic'], 4)} | "
                    f"{_f(item['rank_ic_reference'], 4)} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def _delivery_lines(handed):
    """La sección de F7-b: de qué modelo sale cada pieza, la decisión en calibración y la regla."""
    lines = ["## F7-b, la entrega y de dónde sale cada pieza", ""]
    if handed.get("status") != "measured":
        return lines + [f"No se mide: {handed.get('reason', 'sin datos')}.", ""]
    source = handed["provenance"]
    lines += [
        f"- Orden: {source['order']}.",
        f"- Nivel: {source['level']} (λ {', '.join(f'{m} {_f(v, 5)}' for m, v in handed['evaluation']['fitted']['lad'].items())}).",
        f"- Anchuras: {source['widths']}, multiplicadas por su factor de pinball en cada nivel. Esto "
        "obliga a que la base emita un modelo solo pinball por origen y semilla.",
        f"- Cobertura: {source['coverage']}, encima de las anchuras elegidas.",
        "",
        f"**Decisión en calibración** (ajuste hasta el instante {handed['calibration']['cut']}, "
        f"{handed['calibration']['first_rows']} filas; puntuación en las "
        f"{handed['calibration']['second_rows']} siguientes): se queda "
        f"**{'la candidata' if handed['chosen_in_calibration'] == 'candidate' else 'lo que ya entra'}**"
        + (
            f"; la candidata empeora {', '.join(handed['worse_in_calibration'])}."
            if handed["worse_in_calibration"]
            else "."
        ),
        "",
        "| Tramo | Versión | Pinball media | Error de cobertura 80 | Error de cobertura 90 | Error de cobertura 95 | Anchura 90 | Peor decil de volatilidad 90 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, part in (
        ("calibración, 2.ª mitad", handed["calibration"]),
        ("evaluación", handed["evaluation"]),
    ):
        for name, shown in (("incumbent", "lo que ya entra"), ("candidate", "candidata")):
            item = part[name]
            layer = item["layer"]
            worst = min(g["coverage"] for g in layer["0.9"]["volatility"].values())
            lines.append(
                f"| {split} | {shown} | {_f(item['mean_pinball'])} | {_f(layer['0.8']['coverage_error'], 4)} | "
                f"{_f(layer['0.9']['coverage_error'], 4)} | {_f(layer['0.95']['coverage_error'], 4)} | "
                f"{_f(layer['0.9']['width'], 5)} | {_f(worst, 3)} |"
            )
        raw = part["candidate_without_layer"]
        lines.append(
            f"| {split} | candidata sin capa | — | {_f(raw['0.8']['coverage_error'], 4)} | — | "
            f"{_f(raw['0.95']['coverage_error'], 4)} | — | — |"
        )
    ev = handed["evaluation"]
    lines += [
        "",
        f"ECE de la probabilidad de signo en evaluación: lo que ya entra {_f(ev['incumbent']['sign_ece'], 4)} "
        f"(operable {_f(ev['incumbent']['sign_ece_operable'], 4)}); la candidata {_f(ev['candidate']['sign_ece'], 4)} "
        f"(operable {_f(ev['candidate']['sign_ece_operable'], 4)}).",
        "",
        "**Regla en evaluación** (ninguna métrica de F7 peor que lo que ya entra): "
        + ("**entra**." if handed["enters"] else "**no entra**")
        + (
            f"; empeora {', '.join(handed['worse_in_evaluation'])}."
            if handed["worse_in_evaluation"]
            else ""
        )
        + (
            ""
            if handed["enters"] or handed["worse_in_evaluation"]
            else ", porque en calibración se quedó lo que ya entra."
        ),
        "",
        "Capa de cobertura: el 90 % solo existe con ella, porque la cabeza no tiene los niveles 0,05 y "
        "0,95. "
        + "; ".join(
            f"al {float(level) * 100:.0f} % {'sobraría' if flag else 'hace falta'}"
            for level, flag in handed["layer_superfluous"].items()
        )
        + " (sin capa no peor en calibración y en evaluación).",
        "",
    ]
    return lines


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("medir", help="Calcular F7 de un origen")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    run.add_argument("--reference", type=Path, required=True, help="Raíz de los árboles")
    run.add_argument("--organs", type=Path, required=True, help="Raíz de los órganos")
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--output", type=Path, required=True, help="Raíz propia de F7")
    run.add_argument("--wait-hours", type=float, default=0.0)
    run.add_argument("--poll-seconds", type=float, default=300.0)
    run.add_argument(
        "--market",
        action="append",
        dest="markets",
        metavar="MERCADO",
        help="Medir solo este mercado (repetible); obligatorio en un estado con varios",
    )
    render = commands.add_parser("informe", help="Rehacer resumen.md desde resumen.json, sin medir")
    render.add_argument("--summary", type=Path, required=True, help="resumen.json de un origen")
    return result


def main(argv=None):
    args = parser().parse_args(argv)

    def log(line):
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {line}", flush=True)

    if args.command == "informe":
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        if summary.get("kind") != KIND:
            print(
                f"Error: {args.summary} no es un resumen de F7 ({summary.get('kind')!r})",
                file=sys.stderr,
            )
            return 1
        (args.summary.parent / "resumen.md").write_text(markdown(summary), encoding="utf-8")
        log(f"rehecho {args.summary.parent / 'resumen.md'}")
        return 0
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
            f"Faltan entradas de F7 para el origen {args.origin} tras esperar "
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
