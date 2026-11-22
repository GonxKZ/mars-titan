"""Pilotos de post-entrenamiento en CPU sobre las emisiones reales de una campaña DGX.

Mide, sin entrenar nada ni abrir el test final, dos familias de post-entrenamiento sobre las
predicciones que ya emitieron los trabajos terminados de un estado de `comparison_campaign`
(validación, calibración y evaluación de un origen y una semilla):

1. Escala e intervalos. `raw` (lo emitido), `cqr_static` (CQR del repositorio ajustada en
   calibración sobre la cabeza de cuantiles), `scale_cqr_static` (la mediana pasa a λ·p con el
   factor LAD de calibración y la CQR se ajusta sobre los cuantiles recentrados),
   `screening_scale_cqr` (la receta del informe de criba, con margen por CDF ponderada por
   sesión), `cqr_online` y
   `scale_cqr_online` (PT2: la corrección sigue la cobertura con etiquetas maduras, tasa
   γ = κ·B̂) y, para los modelos declarados, `regime_context` y `regime_uniform` (conformal
   ponderado por contextos parecidos y su control uniforme con el mismo conjunto pasado).
2. Memoria por activo del residuo de rango maduro (`memory.asset_residual_memory`): la rejilla
   declarada de semividas y pesos κ, la configuración fija del laboratorio tabular, la
   configuración elegida con el IC de calibración, el placebo que lee la memoria de otro
   activo y el control que guarda la etiqueta en lugar del residuo.

La combinación en línea de modelos se mide con `evaluation.cognitive_agent` y su declaración.

Contrato causal. La madurez de cada fila es la `target_available_at` de las etiquetas de la vista
(`online_calibration.view_maturity`), y el objetivo de cada fila emitida debe coincidir bit a bit
con el de la vista. Las calibraciones en línea y la memoria solo reciben una etiqueta cuando su
cohorte ya maduró en el instante de la decisión. Todo factor, corrección, escala de contexto,
tasa o configuración se fija con la calibración (o con la validación y la calibración cuando
el método necesita un tramo de arranque y otro de elección); la evaluación se mira una vez.

Estadística. El IC de rangos por sesión, el mínimo de 30 activos, el contraste pareado sesión
a sesión y el error HAC de Newey-West con retardo automático son los del informe de criba
(`scripts/dgx_screening_report.py`), que se carga tal cual. Cada informe dice cuántos orígenes y
semillas sostienen cada cifra.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.calibration import online_conformal as oc
from mars_titan.calibration import regime_weighted_conformal as rwc
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.memory import asset_residual_memory as arm
from mars_titan.memory import neighbour_residual_memory as nrm

KIND = "posttraining_pilots_v1"
SCHEMA_VERSION = 1
SPLITS = ("validation", "calibration", "evaluation")
QUANTILE_COLUMNS = (
    "quantile_0025",
    "quantile_0100",
    "quantile_0500",
    "quantile_0900",
    "quantile_0975",
)
LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
NOMINALS = (0.8, 0.95)
MEMORY_CONTROLS = ("label", "sign")
MAX_GRID = 16
SCREENING_PATH = Path(__file__).resolve().parents[3] / "scripts" / "dgx_screening_report.py"
_DECLARATION = {
    "schema_version",
    "kind",
    "final_test_access",
    "description",
    "market",
    "min_assets",
    "runs",
    "memory",
    "calibration",
}
_MEMORY = {"value", "half_lives", "kappas", "lab", "placebo_seed", "controls"}
_CALIBRATION = {"levels", "nominals", "min_rows", "online_rates", "regime_models"}
_SCREENING = None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def screening():
    """El módulo del informe de criba, cargado una sola vez desde `scripts/`."""
    global _SCREENING
    if _SCREENING is None:
        _require(SCREENING_PATH.is_file(), f"No está el informe de criba en {SCREENING_PATH}")
        spec = importlib.util.spec_from_file_location("dgx_screening_report", SCREENING_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SCREENING = module
    return _SCREENING


def _sorted_unique(values, low, high, label, integer=False):
    kinds = (int,) if integer else (int, float)
    _require(
        isinstance(values, list)
        and 1 <= len(values) <= MAX_GRID
        and all(type(v) in kinds and math.isfinite(v) and low <= v <= high for v in values)
        and values == sorted(set(values)),
        f"{label} es una lista creciente, sin repetir, de 1 a {MAX_GRID} valores"
        f" en [{low}, {high}]",
    )
    return values


def checked_declaration(document):
    """Validar la declaración de los pilotos antes de leer ninguna emisión."""
    _require(isinstance(document, dict), "La declaración de los pilotos es un objeto JSON")
    _require(
        document.get("kind") == KIND and document.get("schema_version") == SCHEMA_VERSION,
        f"La declaración debe ser {KIND} con schema_version {SCHEMA_VERSION}",
    )
    _require(
        set(document) == _DECLARATION,
        f"La declaración tiene exactamente los campos {sorted(_DECLARATION)}",
    )
    _require(document["final_test_access"] is False, "Los pilotos no abren el test final")
    _require(
        isinstance(document["description"], str) and document["description"],
        "Falta la descripción",
    )
    _require(isinstance(document["market"], str) and document["market"], "El mercado es un texto")
    _require(
        document["min_assets"] == screening().MIN_ASSETS,
        f"El mínimo de activos por sesión es el del informe de criba ({screening().MIN_ASSETS})",
    )
    runs = document["runs"]
    _require(isinstance(runs, list) and runs, "Se declara al menos un grupo de trabajos")
    for group in runs:
        _require(
            isinstance(group, dict)
            and set(group) == {"state", "models", "origins", "seeds"}
            and isinstance(group["state"], str)
            and group["state"]
            and isinstance(group["models"], list)
            and group["models"]
            and all(isinstance(m, str) and m for m in group["models"])
            and len(set(group["models"])) == len(group["models"]),
            "Cada grupo declara estado, modelos distintos, orígenes y semillas",
        )
        _sorted_unique(group["origins"], 2000, 2023, "Los orígenes", integer=True)
        _sorted_unique(group["seeds"], 0, 2**31, "Las semillas", integer=True)
    memory = document["memory"]
    _require(
        isinstance(memory, dict) and set(memory) == _MEMORY, f"memory declara {sorted(_MEMORY)}"
    )
    _require(memory["value"] in arm.VALUES, f"El valor de la memoria es uno de {arm.VALUES}")
    _sorted_unique(memory["half_lives"], 1, 10_000, "Las semividas")
    _sorted_unique(memory["kappas"], 0, 10, "Los pesos κ")
    lab = memory["lab"]
    _require(
        isinstance(lab, dict)
        and set(lab) == {"half_life", "kappa"}
        and lab["half_life"] in memory["half_lives"]
        and lab["kappa"] in memory["kappas"]
        and lab["kappa"] > 0,
        "La configuración del laboratorio pertenece a la rejilla declarada y tiene κ > 0",
    )
    _require(
        type(memory["placebo_seed"]) is int and memory["placebo_seed"] >= 0,
        "La semilla del placebo es un entero no negativo",
    )
    _require(
        isinstance(memory["controls"], list)
        and set(memory["controls"]) <= set(MEMORY_CONTROLS)
        and len(set(memory["controls"])) == len(memory["controls"]),
        f"Los controles de la memoria son de {MEMORY_CONTROLS}",
    )
    calibration = document["calibration"]
    _require(
        isinstance(calibration, dict) and set(calibration) == _CALIBRATION,
        f"calibration declara {sorted(_CALIBRATION)}",
    )
    _require(tuple(calibration["levels"]) == LEVELS, f"Los niveles emitidos son {LEVELS}")
    _require(tuple(calibration["nominals"]) == NOMINALS, f"Los intervalos centrales son {NOMINALS}")
    _require(
        type(calibration["min_rows"]) is int and calibration["min_rows"] >= 1,
        "El mínimo de filas de la CQR es un entero positivo",
    )
    _sorted_unique(calibration["online_rates"], 0, 1, "Las tasas κ de PT2")
    _require(
        isinstance(calibration["regime_models"], list)
        and all(isinstance(m, str) and m for m in calibration["regime_models"]),
        "Los modelos del conformal por regímenes son textos",
    )
    return document


@dataclass(frozen=True)
class Split:
    """Un tramo emitido en orden (mercado, instante, activo), con la madurez de la vista."""

    name: str
    market: np.ndarray
    asset: np.ndarray
    prediction_at: np.ndarray
    matures_at: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    quantiles: np.ndarray | None
    sessions: tuple

    @property
    def rows(self):
        return len(self.target)

    def data(self, prediction=None):
        """El formato del informe de criba: objetivo, predicción y sesiones."""
        values = self.prediction if prediction is None else prediction
        return dict(target=self.target, prediction=values, sessions=list(self.sessions))


def replace_target(split, target):
    """El mismo tramo con otro objetivo (para comprobar qué lee cada método)."""
    target = np.asarray(target, dtype=np.float64)
    _require(target.shape == split.target.shape, "El objetivo nuevo no tiene las filas del tramo")
    return replace(split, target=target)


def _strings(column):
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=str)


def _first(mask):
    return int(np.flatnonzero(mask)[0])


def emission_order(table):
    """Activo, mercado e instante de cada fila emitida y su orden (mercado, instante, activo)."""
    asset = _strings(table["asset_id"])
    market = _strings(table["market"])
    moments = table["prediction_at"].cast(pa.int64()).to_numpy()
    return asset, market, moments, np.lexsort((asset, moments, market))


def load_split(table, maturity, name):
    """Ordenar un tramo emitido, comprobarlo y unirle la madurez de las etiquetas de la vista.

    Falla con el activo y el instante de la primera fila que no cumple el contrato: columna
    ausente o nula, valor no finito, cuantiles cruzados, fila repetida, fila sin etiqueta en
    la vista, objetivo distinto del de la vista o etiqueta que no madura después de su decisión.
    """
    needed = ("sample_id", "asset_id", "market", "prediction_at", "target", "prediction")
    _require(
        all(c in table.column_names for c in needed),
        f"La emisión de {name} no tiene las columnas {needed}",
    )
    _require(
        table.schema.field("prediction_at").type == pa.timestamp("us", tz="UTC"),
        f"Los instantes de {name} deben ser timestamp UTC en microsegundos",
    )
    has_quantiles = all(c in table.column_names for c in QUANTILE_COLUMNS)
    columns = needed + (QUANTILE_COLUMNS if has_quantiles else ())
    for column in columns:
        _require(table[column].null_count == 0, f"La columna {column} de {name} tiene nulos")
    asset, market, moments, order = emission_order(table)
    asset, market, moments = asset[order], market[order], moments[order].astype(np.int64)
    target = table["target"].to_numpy().astype(np.float64)[order]
    prediction = table["prediction"].to_numpy().astype(np.float64)[order]
    quantiles = None
    if has_quantiles:
        quantiles = np.column_stack(
            [table[c].to_numpy().astype(np.float64) for c in QUANTILE_COLUMNS]
        )[order]
    _require(len(target) >= 1, f"La emisión de {name} está vacía")
    _require(int(moments.max()) < FINAL_TEST_START_US, f"La emisión de {name} llega al test final")
    repeated = (
        (asset[1:] == asset[:-1]) & (moments[1:] == moments[:-1]) & (market[1:] == market[:-1])
    )
    if repeated.any():
        i = _first(repeated)
        raise ValueError(f"Fila repetida en {name}: {asset[i]} {moments[i]}")
    for values, label in ((target, "objetivo"), (prediction, "predicción")):
        bad = ~np.isfinite(values)
        if bad.any():
            i = _first(bad)
            raise ValueError(f"Valor de {label} no finito en {name}: {asset[i]} {moments[i]}")
    if quantiles is not None:
        bad = ~np.isfinite(quantiles).all(axis=1)
        if bad.any():
            i = _first(bad)
            raise ValueError(f"Cuantil no finito en {name}: {asset[i]} {moments[i]}")
        crossed = np.any(np.diff(quantiles, axis=1) < 0, axis=1)
        if crossed.any():
            i = _first(crossed)
            raise ValueError(f"Cuantiles cruzados en {name}: {asset[i]} {moments[i]}")
    v_asset = _strings(maturity["asset_id"])
    v_market = _strings(maturity["market"])
    v_moments = maturity["prediction_at"].cast(pa.int64()).to_numpy().astype(np.int64)
    v_order = np.lexsort((v_asset, v_moments, v_market))
    v_asset, v_market, v_moments = v_asset[v_order], v_market[v_order], v_moments[v_order]
    same = (
        len(v_asset) == len(asset)
        and np.array_equal(v_asset, asset)
        and np.array_equal(v_moments, moments)
        and np.array_equal(v_market, market)
    )
    if not same:
        known = set(zip(v_market.tolist(), v_asset.tolist(), v_moments.tolist(), strict=True))
        for m, a, t in zip(market.tolist(), asset.tolist(), moments.tolist(), strict=True):
            _require((m, a, t) in known, f"La fila {a} {t} de {name} no tiene etiqueta en la vista")
        raise ValueError(
            f"La vista tiene {len(v_asset)} etiquetas de {name} y la emisión {len(asset)} filas"
        )
    v_target = maturity["target"].to_numpy().astype(np.float64)[v_order]
    differs = v_target != target
    if differs.any():
        i = _first(differs)
        raise ValueError(f"El objetivo de {name} no es el de la vista en {asset[i]} {moments[i]}")
    matures_at = maturity["matures_at"].to_numpy().astype(np.int64)[v_order]
    early = matures_at <= moments
    if early.any():
        i = _first(early)
        raise ValueError(f"La etiqueta de {asset[i]} {moments[i]} no madura después de su decisión")
    change = np.ones(len(target), dtype=bool)
    change[1:] = (moments[1:] != moments[:-1]) | (market[1:] != market[:-1])
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(target)]
    sessions = tuple(
        (str(market[s]), int(moments[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)
    )
    return Split(name, market, asset, moments, matures_at, target, prediction, quantiles, sessions)


# --- métricas -------------------------------------------------------------------------------


def session_ic(split, values):
    """IC de rangos por sesión con el criterio del informe de criba: {(mercado, instante): IC}."""
    return screening().session_ic(split.data(np.asarray(values, dtype=np.float64)))


def _session_means(split, values):
    return {
        (market, instant): float(np.mean(values[start:end]))
        for market, instant, start, end in split.sessions
    }


def pinball_rows(quantiles, target, levels=LEVELS):
    """Pérdida pinball media de los niveles declarados, fila a fila."""
    diff = target[:, None] - quantiles
    tau = np.asarray(levels, dtype=np.float64)[None, :]
    return np.maximum(tau * diff, (tau - 1.0) * diff).mean(axis=1)


def interval_score_rows(lower, upper, target, nominal):
    """Puntuación de intervalo de Gneiting y Raftery (2007) para el nivel central nominal."""
    alpha = 1.0 - nominal
    return (
        (upper - lower)
        + (2.0 / alpha) * np.maximum(lower - target, 0.0)
        + (2.0 / alpha) * np.maximum(target - upper, 0.0)
    )


def _pairs(levels, nominals):
    _, pairs, median = cqr.interval_pairs(list(levels), list(nominals))
    return pairs, median


def calibration_metrics(split, point, quantiles, levels=LEVELS, nominals=NOMINALS):
    """Escala, cobertura, compromiso de signo, ECE y pinball de una emisión calibrada."""
    target = split.target
    pairs, median = _pairs(levels, nominals)
    mae_zero = float(np.mean(np.abs(target)))
    result = dict(
        mae_ratio=float(np.mean(np.abs(target - point)) / mae_zero),
        pinball=float(np.mean(pinball_rows(quantiles, target, levels))),
    )
    per_session = dict(
        pinball=_session_means(split, pinball_rows(quantiles, target, levels)),
        mae=_session_means(split, np.abs(target - point)),
    )
    for nominal, lower, upper in pairs:
        key = f"{round(nominal * 100):d}"
        low, high = quantiles[:, lower], quantiles[:, upper]
        result[f"cov{key}"] = float(np.mean((target >= low) & (target <= high)))
        scores = interval_score_rows(low, high, target, nominal)
        result[f"is{key}"] = float(np.mean(scores))
        per_session[f"is{key}"] = _session_means(split, scores)
        if math.isclose(nominal, 0.8):
            up, down = low > 0.0, high < 0.0
            committed = up | down
            result["committed80"] = float(np.mean(committed))
            wrong = (up & (target <= 0.0)) | (down & (target >= 0.0))
            result["cse80"] = float(np.mean(wrong[committed])) if committed.any() else None
    module = screening()
    result["ece"] = float(module._ece_10bin(module._implied_up_probability(quantiles), target))
    return result, per_session


def recenter(quantiles, center):
    """Los cuantiles con la mediana movida a `center` y las anchuras de la cabeza."""
    return quantiles - quantiles[:, [2]] + np.asarray(center, dtype=np.float64)[:, None]


def online_conformal(start, run, kappa, levels, nominals, min_rows, centers=None):
    """PT2: CQR de `start` que sigue en `run` la cobertura de las cohortes ya maduras."""
    start_q = start.quantiles if centers is None else recenter(start.quantiles, centers[0])
    run_q = run.quantiles if centers is None else recenter(run.quantiles, centers[1])
    calibrator = oc.OnlineConformal.start(
        start.target,
        start_q,
        start.market,
        levels=list(levels),
        nominals=list(nominals),
        min_rows=min_rows,
        rate_fraction=float(kappa),
    )
    emitted, _, _ = oc.replay_online_conformal(
        calibrator, run_q, run.market, run.prediction_at, run.target, run.matures_at
    )
    return emitted


def _interval_criterion(split, quantiles, levels, nominals):
    pairs, _ = _pairs(levels, nominals)
    return float(
        sum(
            np.mean(interval_score_rows(quantiles[:, lo], quantiles[:, hi], split.target, n))
            for n, lo, hi in pairs
        )
    )


def select_online_rate(splits, section, scaled=False):
    """Elegir κ de PT2 arrancando en validación y puntuando la calibración (nunca la evaluación).

    El criterio es la suma de las puntuaciones de intervalo medias de los intervalos centrales
    declarados; con empate gana el κ menor. Con `scaled` la mediana recentrada usa el factor
    LAD de validación, el único conocido al arrancar.
    """
    validation, calibration = splits["validation"], splits["calibration"]
    levels, nominals, min_rows = section["levels"], section["nominals"], section["min_rows"]
    centers = None
    if scaled:
        factor = screening()._lad_scale(validation.prediction, validation.target)
        centers = (factor * validation.prediction, factor * calibration.prediction)
    best = None
    for kappa in section["online_rates"]:
        emitted = online_conformal(
            validation, calibration, kappa, levels, nominals, min_rows, centers=centers
        )
        score = _interval_criterion(calibration, emitted, levels, nominals)
        if best is None or score < best[0]:
            best = (score, float(kappa))
    return best[1]


def _panel(split):
    rows = [
        f"{a}/{t}" for a, t in zip(split.asset.tolist(), split.prediction_at.tolist(), strict=True)
    ]
    panel = ForecastPanel.from_columns(
        pa.array(rows),
        split.market,
        split.prediction_at,
        split.target,
        split.prediction,
        quantiles=split.quantiles,
        levels=LEVELS,
        markets=tuple(sorted(set(split.market.tolist()))),
    )
    position = {row: i for i, row in enumerate(rows)}
    order = np.fromiter(
        (position[r] for r in panel.row_id.to_pylist()), dtype=np.int64, count=len(rows)
    )
    return panel, order


def regime_conformal(splits, weighting):
    """Conformal ponderado por contextos de calibración sobre la evaluación, en orden del tramo."""
    calibration, evaluation = splits["calibration"], splits["evaluation"]
    history = dict(
        market=np.concatenate([s.market for s in splits.values()]),
        target=np.concatenate([s.target for s in splits.values()]),
        matures_at=np.concatenate([s.matures_at for s in splits.values()]),
    )
    cal_panel, cal_order = _panel(calibration)
    eval_panel, eval_order = _panel(evaluation)
    record = rwc.fit(cal_panel, calibration.matures_at[cal_order], history)
    emitted, summary = rwc.replay(
        record,
        (cal_panel, calibration.matures_at[cal_order]),
        (eval_panel, evaluation.matures_at[eval_order]),
        history,
        weighting=weighting,
    )
    result = np.empty_like(emitted)
    result[eval_order] = emitted
    np.testing.assert_array_equal(eval_panel.target, evaluation.target[eval_order])
    return result, summary


def _contrast(candidate, control):
    return screening().paired(candidate, control)


def calibration_pilot(splits, section, regime=False):
    """Métodos de escala e intervalos ajustados sin la evaluación y medidos en ella."""
    calibration, evaluation = splits["calibration"], splits["evaluation"]
    levels, nominals, min_rows = section["levels"], section["nominals"], section["min_rows"]
    pairs, median = _pairs(levels, nominals)
    module = screening()
    factor = float(module._lad_scale(calibration.prediction, calibration.target))
    center_cal, center_eval = factor * calibration.prediction, factor * evaluation.prediction
    methods = {"raw": (evaluation.prediction, evaluation.quantiles)}
    record = cqr.fit_conformal_quantiles(
        calibration.target, calibration.quantiles, calibration.market,
        levels=list(levels), nominals=list(nominals), min_rows=min_rows,
    )
    quantiles, _ = cqr.apply_conformal_quantiles(record, evaluation.quantiles, evaluation.market)
    methods["cqr_static"] = (evaluation.prediction, quantiles)
    shifted_cal = recenter(calibration.quantiles, center_cal)
    shifted_eval = recenter(evaluation.quantiles, center_eval)
    _, _, quantiles = _scale_cqr_static(calibration, evaluation, levels, nominals, min_rows)
    methods["scale_cqr_static"] = (center_eval, quantiles)
    margins = np.array(
        [
            module._session_cdf_cqr_margin(
                shifted_cal[:, lo], shifted_cal[:, hi], calibration.target,
                list(calibration.sessions), n,
            )
            for n, lo, hi in pairs
        ]
    )
    corrections = np.broadcast_to(margins, (evaluation.rows, len(pairs)))
    quantiles, _ = cqr.widen_in_order(shifted_eval, pairs, median, np.array(corrections))
    methods["screening_scale_cqr"] = (center_eval, quantiles)
    rate = select_online_rate(splits, section)
    methods["cqr_online"] = (
        evaluation.prediction,
        online_conformal(calibration, evaluation, rate, levels, nominals, min_rows),
    )
    scaled_rate = select_online_rate(splits, section, scaled=True)
    methods["scale_cqr_online"] = (
        center_eval,
        online_conformal(
            calibration, evaluation, scaled_rate, levels, nominals, min_rows,
            centers=(center_cal, center_eval),
        ),
    )
    regimes = {}
    if regime:
        for weighting in ("context", "uniform"):
            quantiles, summary = regime_conformal(splits, weighting)
            methods[f"regime_{weighting}"] = (evaluation.prediction, quantiles)
            regimes[weighting] = summary
    metrics, sessions = {}, {}
    for name, (point, quantiles) in methods.items():
        metrics[name], sessions[name] = calibration_metrics(
            evaluation, point, quantiles, levels, nominals
        )
    contrasts = {}
    for name in methods:
        for control in ("raw", "cqr_static"):
            if name == control or (control == "cqr_static" and name == "raw"):
                continue
            contrasts[f"{name}-{control}"] = {
                metric: _contrast(sessions[name][metric], sessions[control][metric])
                for metric in sessions[name]
            }
    series = [
        ("evaluation", market, instant, name, metric, value)
        for name, per_metric in sessions.items()
        for metric, values in per_metric.items()
        for (market, instant), value in values.items()
    ]
    return dict(
        selection=dict(lad_scale=factor, online_rate=rate, scaled_online_rate=scaled_rate),
        methods=metrics,
        contrasts=contrasts,
        regimes=regimes,
        series=series,
    )


# --- memoria por activo ------------------------------------------------------------------------


def derangement(assets, seed):
    """Mapa del placebo: cada activo lee la memoria de otro, en un ciclo reproducible."""
    names = sorted(set(assets))
    _require(len(names) >= 2, "El placebo necesita al menos dos activos")
    order = [names[i] for i in np.random.default_rng(seed).permutation(len(names))]
    return {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}


def memory_pilot(splits, section, *, min_assets):
    """Rejilla, configuración del laboratorio, elección en calibración, placebo y controles."""
    _require(
        min_assets == screening().MIN_ASSETS, "El mínimo de activos es el del informe de criba"
    )
    parts = [splits[name] for name in SPLITS]
    joined = dict(
        markets=np.concatenate([p.market for p in parts]),
        assets=np.concatenate([p.asset for p in parts]),
        prediction_at=np.concatenate([p.prediction_at for p in parts]),
        matures_at=np.concatenate([p.matures_at for p in parts]),
        prediction=np.concatenate([p.prediction for p in parts]),
        target=np.concatenate([p.target for p in parts]),
    )
    bounds = np.cumsum([0] + [p.rows for p in parts])
    read_map = derangement(joined["assets"].tolist(), section["placebo_seed"])
    kappas = tuple(sorted(set(section["kappas"]) | {0.0}))

    def replay(half_life, value, placebo, chosen=kappas):
        config = arm.AssetMemoryConfig(half_life=float(half_life), value=value)
        memory = arm.AssetResidualMemory(config)
        return arm.replay(memory, kappas=chosen, read_map=read_map if placebo else None, **joined)

    def ics(values):
        return {
            name: session_ic(splits[name], values[bounds[i] : bounds[i + 1]])
            for i, name in enumerate(SPLITS)
            if name != "validation"
        }

    def mean(series):
        return float(np.mean(list(series.values()))) if series else None

    grid = {}
    for half_life in section["half_lives"]:
        for kappa, values in replay(half_life, section["value"], False).items():
            grid[(half_life, kappa)] = ics(values)
    parent = ics(joined["prediction"])
    ranking = sorted(
        grid, key=lambda key: (-mean(grid[key]["calibration"]), key[1], key[0])
    )
    chosen = ranking[0]
    lab = (section["lab"]["half_life"], float(section["lab"]["kappa"]))
    variants = {
        "parent": parent,
        "kappa_0": grid[(lab[0], 0.0)],
        "lab": grid[lab],
        "selected": grid[chosen],
    }
    for name, (half_life, kappa) in (("lab", lab), ("selected", chosen)):
        if kappa == 0.0:
            variants[f"placebo_{name}"] = variants["kappa_0"]
            continue
        values = replay(half_life, section["value"], True, (kappa,))[kappa]
        variants[f"placebo_{name}"] = ics(values)
    for control in section["controls"]:
        values = replay(lab[0], control, False, (lab[1],))[lab[1]]
        variants[f"{control}_lab"] = ics(values)
    pairs = [
        ("lab", "parent"),
        ("selected", "parent"),
        ("placebo_lab", "parent"),
        ("lab", "placebo_lab"),
        ("selected", "placebo_selected"),
    ] + [(f"{c}_lab", "parent") for c in section["controls"]] + [
        ("lab", f"{c}_lab") for c in section["controls"]
    ]
    contrasts = {
        f"{a}-{b}": _contrast(variants[a]["evaluation"], variants[b]["evaluation"])
        for a, b in pairs
    }
    calibration_contrasts = {
        f"{a}-{b}": _contrast(variants[a]["calibration"], variants[b]["calibration"])
        for a, b in pairs
    }
    series = [
        (split, market, instant, name, "ic", value)
        for name, per_split in variants.items()
        for split, values in per_split.items()
        for (market, instant), value in values.items()
    ]
    return dict(
        selection=dict(
            half_life=chosen[0], kappa=chosen[1], calibration_ic=mean(grid[chosen]["calibration"])
        ),
        grid={
            f"h{h:g}_k{k:g}": dict(
                calibration=mean(v["calibration"]), evaluation=mean(v["evaluation"])
            )
            for (h, k), v in grid.items()
        },
        calibration={name: mean(v["calibration"]) for name, v in variants.items()},
        evaluation={name: mean(v["evaluation"]) for name, v in variants.items()},
        sessions={name: len(v["evaluation"]) for name, v in variants.items()},
        contrasts=contrasts,
        calibration_contrasts=calibration_contrasts,
        series=series,
    )


# --- estudio de memorias sobre varios modelos (activo, vecinos y su combinación) ---------------

STUDY_KIND = "posttraining_memories_v1"
DAY_US = 86_400_000_000
_STUDY = {
    "schema_version",
    "kind",
    "final_test_access",
    "description",
    "market",
    "min_assets",
    "runs",
    "models",
    "inputs",
    "asset",
    "neighbours",
    "combination",
    "placebo_seed",
}


def checked_study(document):
    """Validar la declaración del estudio de memorias antes de leer ninguna emisión."""
    _require(isinstance(document, dict), "La declaración del estudio es un objeto JSON")
    _require(
        document.get("kind") == STUDY_KIND and document.get("schema_version") == SCHEMA_VERSION,
        f"La declaración debe ser {STUDY_KIND} con schema_version {SCHEMA_VERSION}",
    )
    _require(
        _STUDY <= set(document) <= _STUDY | {"addendum"},
        f"El estudio tiene exactamente los campos {sorted(_STUDY)} y, si acaso, addendum",
    )
    _require(document["final_test_access"] is False, "El estudio no abre el test final")
    _require(
        document["min_assets"] == screening().MIN_ASSETS,
        f"El mínimo de activos por sesión es el del informe de criba ({screening().MIN_ASSETS})",
    )
    runs = document["runs"]
    _require(
        isinstance(runs, list)
        and runs
        and all(
            isinstance(r, dict)
            and set(r) == {"state", "origin", "seed"}
            and isinstance(r["state"], str)
            and type(r["origin"]) is int
            and 2000 <= r["origin"] <= 2023
            and type(r["seed"]) is int
            for r in runs
        ),
        "Cada ejecución declara estado, origen y semilla",
    )
    checked_memories(document, "tabular_lab_cache_v1", "la caché del laboratorio tabular")
    _checked_addendum(document.get("addendum"))
    return document


def checked_memories(document, source, where):
    """Modelos, claves, rejillas y placebo: lo común al estudio de memorias y a sus órganos."""
    models = document["models"]
    _require(
        isinstance(models, list)
        and 1 <= len(models) <= nrm.MAX_MODELS
        and all(isinstance(m, str) and m for m in models)
        and len(set(models)) == len(models),
        f"Se declaran entre 1 y {nrm.MAX_MODELS} modelos distintos",
    )
    inputs = document["inputs"]
    _require(
        isinstance(inputs, dict)
        and set(inputs) == {"source", "columns", "projection_dim", "projection_seed"}
        and inputs["source"] == source
        and inputs["columns"] == 92
        and type(inputs["projection_dim"]) is int
        and inputs["projection_dim"] >= 1
        and type(inputs["projection_seed"]) is int,
        f"Las claves salen de {where} ({source}, 92 columnas) con su proyección",
    )
    asset = document["asset"]
    _require(
        isinstance(asset, dict) and set(asset) == {"half_lives", "kappas", "lab"},
        "asset declara semividas, pesos y la configuración del laboratorio",
    )
    _sorted_unique(asset["half_lives"], 1, 10_000, "Las semividas")
    _sorted_unique(asset["kappas"], 0, 10, "Los pesos κ de la memoria por activo")
    _require(
        asset["lab"]["half_life"] in asset["half_lives"]
        and asset["lab"]["kappa"] in asset["kappas"],
        "La configuración del laboratorio pertenece a la rejilla declarada",
    )
    neighbours = document["neighbours"]
    _require(
        isinstance(neighbours, dict)
        and set(neighbours) == {"window", "k", "tau", "clip", "kappas"},
        "neighbours declara ventana, vecinos, temperatura, recorte y pesos",
    )
    nrm.NeighbourConfig(
        window=neighbours["window"],
        k=neighbours["k"],
        tau=neighbours["tau"],
        clip=neighbours["clip"],
    )
    _sorted_unique(neighbours["kappas"], 0, 10, "Los pesos κ de la memoria por vecinos")
    _sorted_unique(document["combination"]["kappas"], 0, 10, "Los pesos de la combinación")
    _require(
        set(document["combination"]) == {"kappas"} and 0.0 in document["combination"]["kappas"],
        "La rejilla de la combinación incluye el cero (cada memoria sola)",
    )
    _require(
        type(document["placebo_seed"]) is int and document["placebo_seed"] >= 0,
        "La semilla del placebo es un entero no negativo",
    )
    return document


_ADDENDUM = {"exclude_own", "quantile_shift", "cold_start"}


def _checked_addendum(addendum):
    """Mediciones añadidas: vecinos de otros activos, cuantiles movidos y arranque en frío."""
    if addendum is None:
        return None
    shift = addendum.get("quantile_shift") if isinstance(addendum, dict) else None
    _require(
        isinstance(addendum, dict)
        and set(addendum) <= _ADDENDUM
        and all(type(addendum.get(key, False)) is bool for key in ("exclude_own", "cold_start"))
        and (
            shift is None
            or (
                isinstance(shift, dict)
                and set(shift) == {"min_rows"}
                and type(shift["min_rows"]) is int
                and shift["min_rows"] >= 1
            )
        ),
        "addendum declara exclude_own y cold_start como booleanos y quantile_shift con min_rows",
    )
    return addendum


def lab_inputs(cache, names, assets, days):
    """Entradas del laboratorio tabular (46 rasgos y sus rangos por sesión) por fila emitida.

    `cache` es la caché ordenada del laboratorio (X, XR, asset, day, bounds y session_part) y
    `names` su `names.json`. Cada fila emitida (activo, día UTC de su decisión) debe estar una
    sola vez entre las filas posteriores al ajuste del laboratorio; si no, falla con la fila.
    """
    cache = Path(cache)
    part = np.load(cache / "session_part.npy")
    bounds = np.load(cache / "bounds.npy")
    post = np.flatnonzero(part >= 1)
    _require(
        post.size and np.all(part[post[0] :] >= 1), "Las sesiones posteriores no son contiguas"
    )
    first, last = int(bounds[post[0]]), int(bounds[post[-1] + 1])
    catalogue = np.asarray(json.loads(Path(names).read_text(encoding="utf-8"))["assets"])
    lab_asset = catalogue[np.asarray(np.load(cache / "asset.npy", mmap_mode="r")[first:last])]
    lab_day = np.asarray(np.load(cache / "day.npy", mmap_mode="r")[first:last]).astype(np.int64)
    position = {}
    for i, key in enumerate(zip(lab_asset.tolist(), lab_day.tolist(), strict=True)):
        _require(key not in position, f"Fila repetida en el laboratorio: {key}")
        position[key] = i
    index = np.empty(len(assets), dtype=np.int64)
    wanted = zip(np.asarray(assets).tolist(), np.asarray(days).tolist(), strict=True)
    for j, key in enumerate(wanted):
        found = position.get(key)
        _require(found is not None, f"La fila {key[0]} del día {key[1]} no está en el laboratorio")
        index[j] = found
    _require(
        len(np.unique(index)) == len(index), "Dos filas emitidas comparten fila del laboratorio"
    )
    x = np.asarray(np.load(cache / "X.npy", mmap_mode="r")[first:last])[index]
    ranks = np.asarray(np.load(cache / "XR.npy", mmap_mode="r")[first:last])[index]
    inputs = np.hstack([x, ranks]).astype(np.float32)
    digests = {
        name: _sha256(cache / name)
        for name in ("X.npy", "XR.npy", "asset.npy", "day.npy", "bounds.npy", "session_part.npy")
    }
    digests["names.json"] = _sha256(Path(names))
    return inputs, digests


def _session_slices(splits):
    offset, slices = 0, []
    for name in SPLITS:
        for market, instant, start, end in splits[name].sessions:
            slices.append((name, market, instant, offset + start, offset + end))
        offset += splits[name].rows
    return slices


def _zscore_sessions(values, slices):
    result = np.empty(len(values))
    for _, _, _, start, end in slices:
        result[start:end] = arm.zscore(np.asarray(values[start:end], dtype=np.float64))
    return result


def _session_dispersion(values, slices):
    """Desviación poblacional de cada sesión, repetida en sus filas: la de `arm.zscore`."""
    result = np.empty(len(values))
    for _, _, _, start, end in slices:
        result[start:end] = np.asarray(values[start:end], dtype=np.float64).std()
    return result


def shift_quantiles(quantiles, dispersion, signal):
    """Los cinco cuantiles movidos juntos por la lectura en unidades de retorno.

    La memoria suma κ por su lectura tipificada a la puntuación tipificada de la predicción.
    En unidades de retorno es sumar la dispersión de la sesión por esa misma cantidad a cada
    cuantil: el orden de la sesión es el de la puntuación combinada y los cuantiles no se
    cruzan porque todos se mueven lo mismo.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    dispersion = np.asarray(dispersion, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    _require(
        quantiles.ndim == 2
        and dispersion.shape == signal.shape == (len(quantiles),)
        and bool(np.all(np.isfinite(dispersion)) and np.all(dispersion >= 0))
        and bool(np.all(np.isfinite(signal))),
        "La dispersión por sesión es finita y no negativa y la lectura es finita en cada fila",
    )
    return quantiles + (dispersion * signal)[:, None]


def _scale_cqr_static(calibration, evaluation, levels, nominals, min_rows):
    """Escala LAD y CQR estáticas ajustadas en calibración y aplicadas a la evaluación."""
    factor = float(screening()._lad_scale(calibration.prediction, calibration.target))
    center_cal, center_eval = factor * calibration.prediction, factor * evaluation.prediction
    record = cqr.fit_conformal_quantiles(
        calibration.target,
        recenter(calibration.quantiles, center_cal),
        calibration.market,
        levels=list(levels),
        nominals=list(nominals),
        min_rows=min_rows,
    )
    quantiles, _ = cqr.apply_conformal_quantiles(
        record, recenter(evaluation.quantiles, center_eval), evaluation.market
    )
    return factor, center_eval, quantiles


def _shift_report(splits, prediction, quantiles, dispersion, signal, min_rows):
    """La memoria como corrección de los cinco cuantiles frente al padre en la evaluación.

    En bruto y con escala y CQR estáticas ajustadas en calibración (cada versión con su propio
    ajuste). Los contrastes son pareados por sesión: movida menos padre.
    """
    parts, offset = {}, 0
    for name in SPLITS:
        rows = slice(offset, offset + splits[name].rows)
        offset += splits[name].rows
        parent = replace(splits[name], prediction=prediction[rows], quantiles=quantiles[rows])
        moved = replace(
            splits[name],
            prediction=prediction[rows] + dispersion[rows] * signal[rows],
            quantiles=shift_quantiles(quantiles[rows], dispersion[rows], signal[rows]),
        )
        parts[name] = dict(parent=parent, shifted=moved)
    report = {}
    for label in ("raw", "scale_cqr_static"):
        metrics, sessions = {}, {}
        for side in ("parent", "shifted"):
            calibration, evaluation = parts["calibration"][side], parts["evaluation"][side]
            point, emitted = evaluation.prediction, evaluation.quantiles
            if label == "scale_cqr_static":
                _, point, emitted = _scale_cqr_static(
                    calibration, evaluation, LEVELS, NOMINALS, min_rows
                )
            metrics[side], sessions[side] = calibration_metrics(evaluation, point, emitted)
        report[label] = dict(
            parent=metrics["parent"],
            shifted=metrics["shifted"],
            contrasts={
                metric: _contrast(sessions["shifted"][metric], sessions["parent"][metric])
                for metric in sessions["shifted"]
            },
        )
    moved = parts["evaluation"]["shifted"]
    report["evaluation_ic"] = float(np.mean(list(session_ic(moved, moved.prediction).values())))
    return report


def _agreement(left, right, slices, min_assets):
    module = screening()
    found = {}
    for name, _, _, start, end in slices:
        if end - start < min_assets:
            continue
        value = module.correlation(
            module.average_ranks(left[start:end]), module.average_ranks(right[start:end])
        )
        if math.isfinite(value):
            found.setdefault(name, []).append(value)
    return {name: float(np.mean(values)) for name, values in found.items()}


def _best(grid, keys):
    """La clave con mayor IC medio de calibración; empates a los pesos menores."""
    def score(key):
        values = grid[key]["calibration"]
        return float(np.mean(list(values.values())))

    return sorted(keys, key=lambda key: (-score(key), key))[0]


def memory_study(
    splits, predictions, keys, section, *, quantiles=None, threads=1, log=print, organs=False
):
    """Memoria por activo, memoria por vecinos y su combinación sobre varios modelos.

    `splits` son los tramos de referencia (las filas son las mismas para todos los modelos),
    `predictions` {modelo: predicción de las filas unidas en el orden validación, calibración
    y evaluación} y `keys` las claves FP32 de esas filas. Toda elección usa el IC de calibración.
    Con `addendum` en la sección se miden además los vecinos de otros activos, la memoria como
    corrección de los cinco cuantiles (`quantiles` {modelo: cuantiles de las filas unidas}) y
    la memoria por activo que empieza vacía en la evaluación.

    Con `organs` devuelve además, para los órganos de memoria, la lectura de cada brazo en las
    filas unidas (la combinada con la semivida y los pesos elegidos en calibración y la fija del
    laboratorio), su elección, la dispersión de la predicción en cada sesión y el estado de su
    memoria por activo al terminar, junto al estado de la memoria por vecinos de todos los modelos.
    """
    addendum = _checked_addendum(section.get("addendum")) or {}
    _require(
        not addendum.get("quantile_shift") or set(quantiles or {}) == set(predictions),
        "Los cuantiles movidos necesitan los cuantiles de cada modelo",
    )
    parts = [splits[name] for name in SPLITS]
    joined = dict(
        markets=np.concatenate([p.market for p in parts]),
        assets=np.concatenate([p.asset for p in parts]),
        prediction_at=np.concatenate([p.prediction_at for p in parts]),
        matures_at=np.concatenate([p.matures_at for p in parts]),
        target=np.concatenate([p.target for p in parts]),
    )
    rows = len(joined["target"])
    _require(len(keys) == rows, "Las claves no tienen las filas de las emisiones")
    bounds = np.cumsum([0] + [p.rows for p in parts])
    slices = _session_slices(splits)
    models = list(predictions)
    matrix = np.column_stack([np.asarray(predictions[m], dtype=np.float64) for m in models])
    _require(matrix.shape == (rows, len(models)), "Cada modelo emite una predicción por fila")
    config = nrm.NeighbourConfig(
        window=section["neighbours"]["window"],
        k=section["neighbours"]["k"],
        tau=section["neighbours"]["tau"],
        clip=section["neighbours"]["clip"],
    )
    started = time.monotonic()
    neighbour_memory = nrm.NeighbourMemory(config, models=len(models), threads=threads)
    reads, placebo_reads = nrm.replay(
        neighbour_memory,
        markets=joined["markets"],
        prediction_at=joined["prediction_at"],
        matures_at=joined["matures_at"],
        keys=keys,
        predictions=matrix,
        target=joined["target"],
        placebo_seed=section["placebo_seed"],
    )
    log(f"vecinos de {len(models)} modelos: {time.monotonic() - started:.0f} s")
    other_reads = other_placebo = None
    if addendum.get("exclude_own"):
        started = time.monotonic()
        codes = np.unique(joined["assets"], return_inverse=True)[1].astype(np.int64)
        other_reads, other_placebo = nrm.replay(
            nrm.NeighbourMemory(
                replace(config, exclude_own=True), models=len(models), threads=threads
            ),
            markets=joined["markets"],
            prediction_at=joined["prediction_at"],
            matures_at=joined["matures_at"],
            keys=keys,
            predictions=matrix,
            target=joined["target"],
            placebo_seed=section["placebo_seed"],
            assets=codes,
        )
        log(f"vecinos de otros activos: {time.monotonic() - started:.0f} s")
    read_map = derangement(joined["assets"].tolist(), section["placebo_seed"])

    def ics(values):
        return {
            name: session_ic(splits[name], values[bounds[i] : bounds[i + 1]])
            for i, name in enumerate(SPLITS)
            if name != "validation"
        }

    def asset_reads(prediction, half_life, placebo):
        memory = arm.AssetResidualMemory(arm.AssetMemoryConfig(half_life=float(half_life)))
        out = arm.replay(
            memory,
            kappas=(0.0, 1.0),
            read_map=read_map if placebo else None,
            prediction=prediction,
            **{k: v for k, v in joined.items()},
        )
        return out[0.0], out[1.0] - out[0.0], memory

    results, series, exported = {}, [], {}
    for j, model in enumerate(models):
        started = time.monotonic()
        prediction = matrix[:, j]
        z = None
        variants = {"parent": ics(prediction)}
        asset, memories = {}, {}
        for half_life in section["asset"]["half_lives"]:
            z, asset[half_life], memories[half_life] = asset_reads(prediction, half_life, False)
        asset_grid = {
            (h, k): ics(z + k * asset[h])
            for h in section["asset"]["half_lives"]
            for k in section["asset"]["kappas"]
            if k > 0
        }
        chosen_asset = _best(asset_grid, list(asset_grid))
        lab = (section["asset"]["lab"]["half_life"], section["asset"]["lab"]["kappa"])
        variants["asset_lab"] = asset_grid[lab]
        variants["asset_selected"] = asset_grid[chosen_asset]
        for name, (h, k) in (("asset_placebo_lab", lab), ("asset_placebo_selected", chosen_asset)):
            _, fake, _ = asset_reads(prediction, h, True)
            variants[name] = ics(z + k * fake)
        near = _zscore_sessions(reads[:, j], slices)
        fake_near = _zscore_sessions(placebo_reads[:, j], slices)
        knn_grid = {k: ics(z + k * near) for k in section["neighbours"]["kappas"] if k > 0}
        chosen_knn = _best(knn_grid, list(knn_grid))
        variants["neighbours_selected"] = knn_grid[chosen_knn]
        variants["neighbours_placebo"] = ics(z + chosen_knn * fake_near)
        own = asset[chosen_asset[0]]
        weights = section["combination"]["kappas"]
        combo = {(a, b): ics(z + a * own + b * near) for a in weights for b in weights}
        both = _best(combo, [c for c in combo if c[0] > 0 and c[1] > 0])
        only_asset = _best(combo, [c for c in combo if c[0] > 0 and c[1] == 0])
        only_near = _best(combo, [c for c in combo if c[0] == 0 and c[1] > 0])
        variants["combined"] = combo[both]
        variants["combined_asset_only"] = combo[only_asset]
        variants["combined_neighbours_only"] = combo[only_near]
        if organs:
            exported[model] = dict(
                dispersion=_session_dispersion(prediction, slices),
                arms=dict(
                    combined=dict(
                        signal=both[0] * own + both[1] * near,
                        selection=dict(
                            selected_on="calibration",
                            half_life=chosen_asset[0],
                            kappa_asset=both[0],
                            kappa_neighbours=both[1],
                        ),
                        state=memories[chosen_asset[0]].export(),
                    ),
                    asset_lab=dict(
                        signal=lab[1] * asset[lab[0]],
                        selection=dict(selected_on="lab", half_life=lab[0], kappa=lab[1]),
                        state=memories[lab[0]].export(),
                    ),
                ),
            )
        extra_selection, extra_pairs, extra = {}, [], {}
        if other_reads is not None:
            other = _zscore_sessions(other_reads[:, j], slices)
            fake_other = _zscore_sessions(other_placebo[:, j], slices)
            other_grid = {k: ics(z + k * other) for k in section["neighbours"]["kappas"] if k > 0}
            chosen_other = _best(other_grid, list(other_grid))
            variants["neighbours_other_selected"] = other_grid[chosen_other]
            variants["neighbours_other_placebo"] = ics(z + chosen_other * fake_other)
            combo_other = {
                (a, b): ics(z + a * own + b * other) for a in weights for b in weights if a and b
            }
            both_other = _best(combo_other, list(combo_other))
            variants["combined_other"] = combo_other[both_other]
            extra_selection = dict(
                neighbours_other=dict(kappa=chosen_other),
                combined_other=dict(kappa_asset=both_other[0], kappa_neighbours=both_other[1]),
            )
            extra_pairs = [
                ("neighbours_other_selected", "parent"),
                ("neighbours_other_placebo", "parent"),
                ("neighbours_selected", "neighbours_other_selected"),
                ("combined_other", "combined_asset_only"),
            ]
            extra["agreement_other"] = _agreement(own, other, slices, screening().MIN_ASSETS)
        if addendum.get("cold_start"):
            start, end = bounds[SPLITS.index("evaluation")], bounds[-1]
            half_life, kappa = chosen_asset
            memory = arm.AssetResidualMemory(arm.AssetMemoryConfig(half_life=float(half_life)))
            cold = arm.replay(
                memory,
                kappas=(0.0, 1.0),
                prediction=prediction[start:end],
                **{key: value[start:end] for key, value in joined.items()},
            )
            values = session_ic(splits["evaluation"], cold[0.0] + kappa * (cold[1.0] - cold[0.0]))
            extra["cold_start"] = dict(
                evaluation=float(np.mean(list(values.values()))),
                contrast=_contrast(values, variants["asset_selected"]["evaluation"]),
            )
            series += [
                (model, "evaluation", market, instant, "asset_selected_cold", value)
                for (market, instant), value in values.items()
            ]
        if addendum.get("quantile_shift"):
            dispersion = _session_dispersion(prediction, slices)
            signals = dict(
                asset_selected=chosen_asset[1] * asset[chosen_asset[0]],
                neighbours_selected=chosen_knn * near,
                combined=both[0] * own + both[1] * near,
            )
            extra["quantile_shift"] = {
                name: _shift_report(
                    splits,
                    prediction,
                    np.asarray(quantiles[model], dtype=np.float64),
                    dispersion,
                    signal,
                    addendum["quantile_shift"]["min_rows"],
                )
                for name, signal in signals.items()
            }
        pairs = [
            ("asset_lab", "parent"),
            ("asset_selected", "parent"),
            ("asset_placebo_lab", "parent"),
            ("asset_placebo_selected", "parent"),
            ("neighbours_selected", "parent"),
            ("neighbours_placebo", "parent"),
            ("combined", "parent"),
            ("combined", "combined_asset_only"),
            ("combined", "combined_neighbours_only"),
            *extra_pairs,
        ]
        results[model] = dict(
            selection=dict(
                asset=dict(half_life=chosen_asset[0], kappa=chosen_asset[1]),
                neighbours=dict(kappa=chosen_knn),
                combined=dict(kappa_asset=both[0], kappa_neighbours=both[1]),
                asset_only=dict(kappa=only_asset[0]),
                neighbours_only=dict(kappa=only_near[1]),
                **extra_selection,
            ),
            calibration={
                name: float(np.mean(list(v["calibration"].values())))
                for name, v in variants.items()
            },
            evaluation={
                name: float(np.mean(list(v["evaluation"].values())))
                for name, v in variants.items()
            },
            sessions={name: len(v["evaluation"]) for name, v in variants.items()},
            contrasts={
                f"{a}-{b}": _contrast(variants[a]["evaluation"], variants[b]["evaluation"])
                for a, b in pairs
            },
            calibration_contrasts={
                f"{a}-{b}": _contrast(variants[a]["calibration"], variants[b]["calibration"])
                for a, b in pairs
            },
            agreement=_agreement(own, near, slices, screening().MIN_ASSETS),
            **extra,
        )
        series += [
            (model, split, market, instant, name, value)
            for name, per_split in variants.items()
            for split, values in per_split.items()
            for (market, instant), value in values.items()
        ]
        log(f"{model}: {time.monotonic() - started:.0f} s")
    model_, split, market, instant, variant, value = zip(*series, strict=True)
    table = pa.table(
        dict(
            model=pa.array(model_, type=pa.string()),
            split=pa.array(split, type=pa.string()),
            market=pa.array(market, type=pa.string()),
            instant=pa.array(instant, type=pa.int64()),
            variant=pa.array(variant, type=pa.string()),
            value=pa.array(value, type=pa.float64()),
        )
    )
    if organs:
        organ = dict(models=exported, order=models, neighbours=neighbour_memory.export())
        return results, table, organ
    return results, table


def run_memory_study(study, states, cache, names, output, *, threads=1, log=print):
    """Cargar los modelos de cada ejecución de uno en uno y medir las memorias sobre ellos."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_canonical(study).encode()).hexdigest()
    saved = output / "declaration.json"
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        _require(
            hashlib.sha256(_canonical(previous).encode()).hexdigest() == digest,
            f"{output} pertenece a otra declaración",
        )
    else:
        saved.write_text(json.dumps(study, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for run in study["runs"]:
        stem = f"{run['state']}--{run['origin']}--seed-{run['seed']}"
        if (output / f"{stem}.json").exists():
            log(f"{stem}: ya medido")
            continue
        root = Path(states[run["state"]])
        folders = {
            m: root / "runs" / m / str(run["origin"]) / f"seed-{run['seed']}"
            for m in study["models"]
        }
        for model, folder in folders.items():
            _require((folder / "run.json").is_file(), f"Falta el trabajo {folder}")
        first = json.loads((folders[study["models"][0]] / "run.json").read_text(encoding="utf-8"))
        view_sha256 = first["request"]["view_sha256"]
        maturity = _maturity(root / "cohorts" / str(run["origin"]) / "manifest.json", view_sha256)
        reference, predictions, inputs_sha = None, {}, {}
        shifted = bool((study.get("addendum") or {}).get("quantile_shift"))
        quantiles = {} if shifted else None
        for model, folder in folders.items():
            record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            _require(record.get("status") == "completed", f"El trabajo {folder} no está terminado")
            _require(
                record["request"]["view_sha256"] == view_sha256,
                f"El trabajo {folder} no usa la vista de los demás",
            )
            splits = {}
            for name in SPLITS:
                path = folder / f"{name}-predictions.parquet"
                inputs_sha[f"{model}/{name}"] = _sha256(path)
                splits[name] = load_split(pq.read_table(path), maturity[name], name)
            if reference is None:
                reference = splits
            for name in SPLITS:
                same = all(
                    np.array_equal(getattr(splits[name], f), getattr(reference[name], f))
                    for f in ("market", "asset", "prediction_at", "target")
                )
                _require(same, f"Las filas de {name} de {model} no son las de {study['models'][0]}")
            predictions[model] = np.concatenate([splits[name].prediction for name in SPLITS])
            if shifted:
                quantiles[model] = np.concatenate([splits[name].quantiles for name in SPLITS])
            log(f"cargado {model}")
        del maturity
        assets = np.concatenate([reference[name].asset for name in SPLITS])
        days = np.concatenate([reference[name].prediction_at for name in SPLITS]) // DAY_US
        started = time.monotonic()
        raw, lab_sha = lab_inputs(cache, names, assets, days)
        matrix = nrm.projection(
            raw.shape[1], study["inputs"]["projection_dim"], seed=study["inputs"]["projection_seed"]
        )
        _require(raw.shape[1] == study["inputs"]["columns"], "Las entradas no tienen 92 columnas")
        keys = nrm.keys_from_inputs(raw, matrix)
        del raw
        log(f"claves: {time.monotonic() - started:.0f} s")
        results, table = memory_study(
            reference, predictions, keys, study, quantiles=quantiles, threads=threads, log=log
        )
        report = dict(
            kind=f"{STUDY_KIND}_report",
            final_test_opened=False,
            declaration_sha256=digest,
            run=run,
            view_sha256=view_sha256,
            inputs=dict(emissions=inputs_sha, lab=lab_sha),
            rows={name: reference[name].rows for name in SPLITS},
            sessions={name: len(reference[name].sessions) for name in SPLITS},
            models=results,
        )
        pq.write_table(table, output / f"{stem}.parquet")
        temporary = output / f"{stem}.json.tmp"
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output / f"{stem}.json")
        log(f"{stem}: terminado")


# --- un trabajo y la orden -----------------------------------------------------------------------


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_run(folder, maturity, declaration, model=None):
    """Los dos pilotos sobre un trabajo terminado: informe JSON y serie por sesión."""
    folder = Path(folder)
    names = [f"{name}-predictions.parquet" for name in SPLITS] + ["run.json"]
    for name in names:
        _require((folder / name).is_file(), f"Falta {folder / name}")
    inputs = {name: _sha256(folder / name) for name in names}
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    _require(run.get("status") == "completed", f"El trabajo {folder} no está terminado")
    splits = {
        name: load_split(
            pq.read_table(folder / f"{name}-predictions.parquet"), maturity[name], name
        )
        for name in SPLITS
    }
    for earlier, later in zip(SPLITS[:-1], SPLITS[1:], strict=True):
        _require(
            splits[earlier].prediction_at.max() < splits[later].prediction_at.min(),
            f"Los tramos {earlier} y {later} se solapan",
        )
    memory = memory_pilot(splits, declaration["memory"], min_assets=declaration["min_assets"])
    calibration = None
    if all(split.quantiles is not None for split in splits.values()):
        regime = model is not None and model in declaration["calibration"]["regime_models"]
        calibration = calibration_pilot(splits, declaration["calibration"], regime=regime)
    rows = list(memory.pop("series"))
    if calibration is not None:
        rows += calibration.pop("series")
    split, market, instant, variant, metric, value = zip(*rows, strict=True)
    series = pa.table(
        dict(
            split=pa.array(split, type=pa.string()),
            market=pa.array(market, type=pa.string()),
            instant=pa.array(instant, type=pa.int64()),
            variant=pa.array(variant, type=pa.string()),
            metric=pa.array(metric, type=pa.string()),
            value=pa.array(value, type=pa.float64()),
        )
    )
    request = run.get("request", {})
    report = dict(
        kind=f"{KIND}_run",
        final_test_opened=False,
        run_id=run.get("run_id"),
        window=request.get("window"),
        view_sha256=request.get("view_sha256"),
        inputs=inputs,
        rows={name: s.rows for name, s in splits.items()},
        sessions={name: len(s.sessions) for name, s in splits.items()},
        memory=memory,
        calibration=calibration,
    )
    return report, series


def _canonical(document):
    return json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _maturity(manifest, view_sha256):
    """Madurez de las etiquetas de la vista con que se ajustaron los trabajos (su cohorte)."""
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.evaluation.online_calibration import view_maturity
    from mars_titan.training.corpus_inputs import CorpusDataset

    _require(manifest.is_file(), f"No está la vista {manifest}")
    dataset = CorpusDataset(manifest, cache_bytes=0, input_policy=HISTORICAL_MASKED)
    _require(
        dataset.identity == view_sha256,
        f"La vista {manifest} no es la de los trabajos ({dataset.identity} ≠ {view_sha256})",
    )
    return view_maturity(
        manifest, HISTORICAL_MASKED, view_sha256, partitions=SPLITS, dataset=dataset
    )


def _jobs(declaration, states):
    jobs = []
    for group in declaration["runs"]:
        root = Path(states[group["state"]])
        for origin in group["origins"]:
            for seed in group["seeds"]:
                for model in group["models"]:
                    folder = root / "runs" / model / str(origin) / f"seed-{seed}"
                    jobs.append((group["state"], model, origin, seed, folder))
    return jobs


def run_pilots(declaration, states, output, *, log=print):
    """Recorrer los trabajos declarados de uno en uno; reanuda los que ya tienen informe."""
    output = Path(output)
    (output / "runs").mkdir(parents=True, exist_ok=True)
    declaration_sha256 = hashlib.sha256(_canonical(declaration).encode()).hexdigest()
    saved = output / "declaration.json"
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        _require(
            hashlib.sha256(_canonical(previous).encode()).hexdigest() == declaration_sha256,
            f"{output} pertenece a otra declaración",
        )
    else:
        text = json.dumps(declaration, indent=2, ensure_ascii=False) + "\n"
        saved.write_text(text, encoding="utf-8")
    jobs = _jobs(declaration, states)
    for state, _, _, _, folder in jobs:
        _require((folder / "run.json").is_file(), f"El trabajo {folder} de {state} no existe")
    # La vista de cada trabajo es la cohorte que el estado escribió para su origen; los
    # trabajos que comparten vista comparten su madurez, que se lee una sola vez.
    views = {}
    for job in jobs:
        run = json.loads((job[4] / "run.json").read_text(encoding="utf-8"))
        manifest = Path(states[job[0]]) / "cohorts" / str(job[2]) / "manifest.json"
        views.setdefault(run["request"]["view_sha256"], (manifest, []))[1].append(job)
    done = 0
    for view_sha256, (manifest, group) in sorted(views.items(), key=lambda item: str(item[1][0])):
        maturity = None
        for state, model, origin, seed, folder in group:
            stem = f"{state}--{model}--{origin}--seed-{seed}"
            report_path = output / "runs" / f"{stem}.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                inputs = {name: _sha256(folder / name) for name in report["inputs"]}
                same = report.get("declaration_sha256") == declaration_sha256
                if same and inputs == report["inputs"]:
                    done += 1
                    log(f"[{done}/{len(jobs)}] {stem}: ya medido")
                    continue
            if maturity is None:
                started = time.monotonic()
                maturity = _maturity(manifest, view_sha256)
                log(f"madurez de {manifest}: {time.monotonic() - started:.0f} s")
            started = time.monotonic()
            report, series = analyze_run(folder, maturity, declaration, model=model)
            report.update(
                declaration_sha256=declaration_sha256,
                state=state,
                model=model,
                origin=origin,
                seed=seed,
            )
            pq.write_table(series, output / "runs" / f"{stem}.parquet")
            temporary = report_path.with_suffix(".json.tmp")
            text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, report_path)
            done += 1
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            elapsed = time.monotonic() - started
            log(f"[{done}/{len(jobs)}] {stem}: {elapsed:.0f} s, RSS {peak:.0f} MiB")
        del maturity
    return declaration_sha256


def _states(items):
    states = {}
    for item in items:
        name, _, path = item.partition("=")
        _require(name and path and name not in states, f"Estado mal declarado: {item!r}")
        states[name] = Path(path).resolve()
    return states


def _private_cache(path, states):
    from mars_titan.training.corpus_inputs import DIGEST_CACHE_ENV

    cache = Path(path).resolve()
    _require(
        not any(cache.is_relative_to(root) for root in states.values()),
        "La caché de huellas no puede estar dentro de un estado de campaña",
    )
    os.environ[DIGEST_CACHE_ENV] = str(cache)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pilots = commands.add_parser("pilotos", help="Escala, intervalos y memoria por activo")
    memories = commands.add_parser("memorias", help="Memorias por activo y por vecinos")
    for sub in (pilots, memories):
        sub.add_argument("--declaration", type=Path, required=True)
        sub.add_argument("--state", action="append", required=True, help="NOMBRE=RUTA de un estado")
        sub.add_argument("--output", type=Path, required=True, help="Carpeta del informe")
        sub.add_argument(
            "--digest-cache", type=Path, required=True,
            help="Caché privada de huellas de la vista (nunca la de un estado de campaña)",
        )
    memories.add_argument("--inputs-cache", type=Path, required=True, help="Caché del laboratorio")
    memories.add_argument(
        "--inputs-names", type=Path, required=True, help="names.json del laboratorio"
    )
    memories.add_argument(
        "--threads", type=int, default=1,
        help="Hilos de la lectura por vecinos (no cambian el resultado)",
    )
    args = parser.parse_args(argv)
    document = json.loads(args.declaration.read_text(encoding="utf-8"))
    states = _states(args.state)

    def log(line):
        print(line, flush=True)

    if args.command == "pilotos":
        declaration = checked_declaration(document)
        _require(
            {group["state"] for group in declaration["runs"]} <= set(states),
            "Falta la ruta de algún estado declarado",
        )
        _private_cache(args.digest_cache, states)
        run_pilots(declaration, states, args.output, log=log)
    else:
        study = checked_study(document)
        _require(
            {run["state"] for run in study["runs"]} <= set(states),
            "Falta la ruta de algún estado declarado",
        )
        _private_cache(args.digest_cache, states)
        run_memory_study(
            study, states, args.inputs_cache, args.inputs_names, args.output,
            threads=args.threads, log=log,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
