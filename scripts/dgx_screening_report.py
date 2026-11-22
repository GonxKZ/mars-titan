"""Informe de la criba de una campaña DGX: IC por sesión, contrastes pareados y combinaciones.

Lee de un estado de `comparison_campaign` las predicciones emitidas por cada modelo terminado en
uno o varios orígenes y una semilla (validación, calibración y evaluación) y escribe un informe
JSON y otro Markdown con:

- el IC de rangos por sesión de cada modelo y tramo (la métrica de selección de la campaña: media
  de la correlación de Spearman entre predicción y objetivo en cada sesión con al menos 30
  activos; una sesión es un mercado en un instante), por mercado si hay varios, su ICIR y la
  desviación de su curva de validación por época;
- el contraste pareado de cada variante con su control: la diferencia de IC sesión a sesión, que
  cancela el ruido común de las sesiones, con su media, su error típico y su t. Se decide solo con
  validación y calibración; la evaluación se informa como evidencia fuera de muestra del origen.
  Con varios orígenes, el contraste agregado reúne las sesiones de todos;
- combinaciones de modelos por media de rangos dentro de cada sesión, elegidas hacia delante solo
  con validación y calibración;
- con `--compare`, la diferencia entre cada modelo de este estado y el mismo modelo de otro (la
  campaña conjunta frente a la de EE. UU.) en sus sesiones comunes.

Con `--offset` mide además el desplazamiento conocido de la etiqueta. La etiqueta es
`r(t+1) - alfa(t) - beta(t) * m(t+1)` y alfa se conoce al predecir (`data.technical_features`):
el informe da el IC de `-alfa` por sí solo, el de cada modelo con la etiqueta sin alfa y el de
su combinación con `-alfa`, con el peso elegido en validación y calibración. La etiqueta sin
alfa, `r - beta * m`, es el rendimiento que se puede operar. Como la etiqueta es exactamente esa
menos alfa, la predicción de un modelo para ella es la suya más alfa, en unidades de retorno:
ese es el «IC sin alfa». Solo vale tal cual si la predicción tiene la escala de los retornos,
así que se da también con la predicción multiplicada por el factor que más IC sin alfa da en
validación y calibración, junto a la dispersión de la predicción frente a la de la etiqueta.
Las dos medidas se dan también en el tercio más líquido de cada sesión (volumen medio en dólares
de 21 sesiones), y los IC de evaluación llevan su t de Newey-West.

Con `--inference MODELO` escribe `sessions.csv`, el IC de evaluación de cada modelo en cada
sesión, y añade la inferencia de `dgx_ic_inference.py` frente a ese modelo de referencia:
intervalos por bootstrap estacionario, Diebold-Mariano, el contraste SPA de Hansen (¿supera
algún modelo a la referencia una vez descontado que se han probado todos?) y el conjunto de
confianza de modelos de Hansen, Lunde y Nason. Si `dgx_ic_stability.py` está junto a esta
herramienta, añade con la misma tabla la estabilidad de cada modelo: ICIR, sesiones y meses
con IC positivo, peor mes, caída máxima del IC acumulado y peor ventana de 21 sesiones.

Solo lee: no toca los estados ni abre ningún tramo que la campaña no haya emitido.

Uso:

    python scripts/dgx_screening_report.py --state ESTADO --year 2013 --seed 42 \
        --controls CONTROLES.json --output CARPETA [--reference OTRO] [--compare OTRO]
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SPLITS = ("validation", "calibration", "evaluation")
# La decisión entre modelos usa los dos tramos que el ajuste no ve como etiquetas. La escala,
# los márgenes de los intervalos y el peso de -alfa se ajustan solo con calibración, que la
# parada temprana no ha tocado.
DECISION = ("validation", "calibration")
CALIBRATION_SPLIT = ("calibration",)
MIN_ASSETS = 30
COLUMNS = ["sample_id", "market", "prediction_at", "target", "prediction"]
QUANTILE_COLUMNS = (
    "quantile_0025",
    "quantile_0100",
    "quantile_0500",
    "quantile_0900",
    "quantile_0975",
)
QUANTILE_LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)


def average_ranks(values):
    """Rangos medios (los empates comparten el rango medio), como en la correlación de Spearman."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2
    return ranks


def correlation(left, right):
    left = left - left.mean()
    right = right - right.mean()
    scale = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right) / scale if scale > 0 else float("nan")


def _technical_module():
    """`technical_features`, junto a esta herramienta instalada o en el repositorio."""
    here = Path(__file__).resolve()
    for path in (
        here.with_name("technical_features.py"),
        here.parents[1] / "src/mars_titan/data/technical_features.py",
    ):
        if path.is_file():
            spec = importlib.util.spec_from_file_location("technical_features", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("Falta technical_features.py para medir el desplazamiento de la etiqueta")


class KnownOffset:
    """Alfa de la etiqueta y liquidez de cada fila, con los precios preparados de la cohorte."""

    def __init__(self, state, year):
        manifest = json.loads((state / "cohorts" / str(year) / "manifest.json").read_text())
        self.prepared = Path(manifest["roots"]["prepared"])
        self.module = _technical_module()
        self.factors, self.assets = {}, {}
        for market, declared in (manifest.get("market_factors") or {}).items():
            path = Path(declared.get("prices_path", ""))
            if path.is_file():
                self.factors[market] = self._intraday(path)

    @staticmethod
    def _table(path):
        table = pq.read_table(
            path, columns=["open", "high", "low", "close", "volume", "available_at"]
        )
        days = table.column("available_at").cast("int64").to_numpy().astype("datetime64[us]")
        columns = ("open", "high", "low", "close", "volume")
        return np.column_stack([table.column(c).to_numpy() for c in columns]), days.astype(
            "datetime64[D]"
        )

    def _intraday(self, path):
        prices, days = self._table(path)
        with np.errstate(divide="ignore", invalid="ignore"):
            valid = (prices[:, 0] > 0) & (prices[:, 3] > 0)
            return days, np.where(valid, prices[:, 3] / prices[:, 0] - 1, np.nan)

    def alpha(self, asset, instants):
        """Alfa de un activo en cada instante de predicción, o NaN sin modelo de mercado."""
        return self.known(asset, instants)[0]

    def known(self, asset, instants):
        """Alfa (NaN sin modelo de mercado) y volumen en dólares de 21 sesiones de cada instante."""
        if asset not in self.assets:
            market, symbol = asset.split("/", 1)
            prices, days = self._table(self.prepared / market / symbol / "prices.parquet")
            features = self.module.technical_features(
                prices, days, *self.factors.get(market, (None, None))
            )
            names = self.module.NAMES
            known = features[:, names.index("has_market_model")] > 0
            values = features[:, names.index("alpha_percent")].astype(np.float64) / 100
            liquidity = features[:, names.index("log_dollar_volume_21")].astype(np.float64)
            self.assets[asset] = days, np.where(known, values, np.nan), liquidity
        days, values, liquidity = self.assets[asset]
        wanted = instants.astype("datetime64[us]").astype("datetime64[D]")
        position = np.searchsorted(days, wanted)
        inside = position < len(days)
        inside[inside] = days[position[inside]] == wanted[inside]
        result, volume = np.full(len(instants), np.nan), np.full(len(instants), np.nan)
        result[inside] = values[position[inside]]
        volume[inside] = liquidity[position[inside]]
        return result, volume

    def rows(self, run, split):
        """Alfa y liquidez de las filas de un tramo, en el orden de `load_split`."""
        table = pq.read_table(
            run / f"{split}-predictions.parquet",
            columns=["sample_id", "asset_id", "market", "prediction_at"],
        ).sort_by(
            [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
        )
        assets = np.asarray(table.column("asset_id").to_pylist())
        instants = table.column("prediction_at").cast("int64").to_numpy()
        result, volume = np.full(len(assets), np.nan), np.full(len(assets), np.nan)
        order = np.argsort(assets, kind="mergesort")
        bounds = np.r_[0, np.flatnonzero(assets[order][1:] != assets[order][:-1]) + 1, len(assets)]
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            rows = order[start:end]
            result[rows], volume[rows] = self.known(str(assets[rows[0]]), instants[rows])
        return result, volume


def _z(values):
    ranks = average_ranks(values)
    scale = ranks.std()
    return (ranks - ranks.mean()) / scale if scale > 0 else ranks * 0


def offset_sessions(data, alpha, weight, *, free, prediction=None):
    """IC por sesión de la mezcla de rangos del modelo con el desplazamiento conocido.

    Con `free` la etiqueta es la que no resta alfa y el desplazamiento entra con signo positivo.
    Sin `prediction` es el desplazamiento solo. Solo cuentan las filas con alfa conocido.
    """
    result = {}
    for market, instant, start, end in data["sessions"]:
        known = np.isfinite(alpha[start:end])
        if known.sum() < MIN_ASSETS:
            continue
        offset = alpha[start:end][known]
        target = data["target"][start:end][known]
        truth = _z(target + offset if free else target)
        signal = _z(offset if free else -offset)
        if prediction is not None:
            signal = _z(weight * _z(prediction[start:end][known]) + (1 - weight) * signal)
        result[(market, instant)] = float(np.mean(signal * truth))
    return result


def implied_sessions(data, alpha, liquidity=None, *, free, prediction=None):
    """IC por sesión en unidades de retorno, con todas las filas o en el tercio más líquido.

    Con `free` se mide la etiqueta sin alfa con la predicción más alfa, que es la predicción
    del modelo para ese objetivo. Sin `free`, la etiqueta con la predicción tal cual. Sin
    `prediction` se mide el desplazamiento solo. Con `liquidity` solo entra el tercio de filas
    con más volumen en dólares de cada sesión. Solo cuentan las filas con alfa conocido.
    """
    result = {}
    for market, instant, start, end in data["sessions"]:
        known = np.isfinite(alpha[start:end])
        if liquidity is not None:
            known &= np.isfinite(liquidity[start:end])
        if known.sum() < MIN_ASSETS:
            continue
        offset = alpha[start:end][known]
        target = data["target"][start:end][known]
        signal = np.zeros(len(offset)) if prediction is None else prediction[start:end][known]
        if free:
            target, signal = target + offset, signal + offset
        elif prediction is None:
            signal = -offset
        if liquidity is not None:
            order = np.argsort(liquidity[start:end][known], kind="mergesort")
            top = order[len(order) - len(order) // 3 :]
            if len(top) < MIN_ASSETS:
                continue
            target, signal = target[top], signal[top]
        value = correlation(average_ranks(signal), average_ranks(target))
        if not math.isnan(value):
            result[(market, instant)] = value
    return result


WEIGHTS = tuple(round(0.1 * step, 1) for step in range(11))
# Factores que se prueban para la predicción de lo operable, k·p + alfa.
SCALES = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)


def best_scale(splits, alphas, scales=SCALES):
    """Factor de la predicción con más IC sin alfa estrictamente en calibración.

    La predicción de un modelo para lo operable es la suya más alfa solo si su escala es la de
    los retornos. El factor se elige exclusivamente con el tramo limpio de calibración (sin
    mezclar validación, que tiene sesgo de selección de época), y la evaluación se mide con él.
    """

    def decision(factor):
        values = [
            value
            for split in CALIBRATION_SPLIT
            for value in implied_sessions(
                splits[split],
                alphas[split],
                free=True,
                prediction=factor * splits[split]["prediction"],
            ).values()
        ]
        return float(np.mean(values)) if values else -1.0

    return max(scales, key=decision)


def dispersion(data, alpha=None):
    """Media por sesión de la dispersión de la predicción entre la de la etiqueta."""
    ratios = []
    for _, _, start, end in data["sessions"]:
        if alpha is None:
            if end - start < MIN_ASSETS:
                continue
            target_slice = data["target"][start:end]
            pred_slice = data["prediction"][start:end]
        else:
            known = np.isfinite(alpha[start:end])
            if known.sum() < MIN_ASSETS:
                continue
            target_slice = data["target"][start:end][known]
            pred_slice = data["prediction"][start:end][known]
        spread = target_slice.std()
        if spread > 0:
            ratios.append(float(pred_slice.std() / spread))
    return float(np.mean(ratios)) if ratios else None


def _lad_scale(prediction, target):
    """Factor de escala óptimo en norma L1 por el origen (k >= 0)."""
    pred = np.asarray(prediction, dtype=np.float64)
    targ = np.asarray(target, dtype=np.float64)
    nonzero = pred != 0.0
    if not np.any(nonzero):
        return 0.0
    ratios = targ[nonzero] / pred[nonzero]
    weights = np.abs(pred[nonzero])
    order = np.argsort(ratios, kind="mergesort")
    sorted_ratios = ratios[order]
    cum_weights = np.cumsum(weights[order])
    idx = int(np.searchsorted(cum_weights, 0.5 * cum_weights[-1], side="left"))
    idx = min(idx, sorted_ratios.size - 1)
    value = float(sorted_ratios[idx])
    return max(0.0, value) if math.isfinite(value) else 0.0


def _session_cdf_cqr_margin(lower, upper, target, sessions, coverage):
    """Margen CQR mediante CDF ponderada por sesión con corrección finita (1 + 1/S)."""
    scores = np.maximum(lower - target, target - upper)
    valid_sessions = [(start, end) for _, _, start, end in sessions if end > start]
    n_sessions = len(valid_sessions)
    if n_sessions == 0:
        return 0.0
    row_scores = []
    row_weights = []
    for start, end in valid_sessions:
        seg = scores[start:end]
        row_scores.append(seg)
        row_weights.append(np.full(seg.size, 1.0 / (n_sessions * seg.size), dtype=np.float64))
    all_scores = np.concatenate(row_scores)
    all_weights = np.concatenate(row_weights)
    order = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[order]
    cum_weights = np.cumsum(all_weights[order])
    level = min(1.0, coverage * (1.0 + 1.0 / n_sessions))
    idx = int(np.searchsorted(cum_weights, level, side="left"))
    idx = min(idx, sorted_scores.size - 1)
    return float(sorted_scores[idx])


def _implied_up_probability(quantiles):
    """Probabilidad implícita P(y > 0) interpolada en los 5 cuantiles, acotada a [0.025, 0.975].

    Vectorizada: en una fila no decreciente, el número de cuantiles menores o iguales que cero es
    el índice de inserción de cero por la derecha.
    """
    levels = np.asarray(QUANTILE_LEVELS, dtype=np.float64)
    values = np.asarray(quantiles, dtype=np.float64)
    count = values.shape[0]
    if count == 0:
        return np.empty(0, dtype=np.float64)
    high = np.clip((values <= 0.0).sum(axis=1), 1, len(levels) - 1)
    low = high - 1
    rows = np.arange(count)
    below, above = values[rows, low], values[rows, high]
    width = above - below
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (0.0 - below) / width
    inside = np.where(
        width > 0.0,
        levels[low] + fraction * (levels[high] - levels[low]),
        0.5 * (levels[low] + levels[high]),
    )
    p_le_zero = np.where(
        values[:, 0] >= 0.0, levels[0], np.where(values[:, -1] <= 0.0, levels[-1], inside)
    )
    return np.clip(1.0 - p_le_zero, 0.025, 0.975)


def _ece_10bin(probs, targets):
    """Error de calibración esperado en 10 intervalos iguales para P(y > 0)."""
    if len(probs) == 0:
        return None
    y_up = (targets > 0.0).astype(np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    total = float(len(probs))
    for idx in range(10):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (probs >= lo) & (probs <= hi if idx == 9 else probs < hi)
        count = int(np.sum(mask))
        if count > 0:
            ece += (count / total) * abs(float(np.mean(probs[mask])) - float(np.mean(y_up[mask])))
    return float(ece)


def _calibration_diagnostics(cal_split, eval_split):
    """Diagnósticos de escala y calibración ajustados en `calibration` y medidos en `evaluation`."""
    y_cal = np.asarray(cal_split["target"], dtype=np.float64)
    p_cal = np.asarray(cal_split["prediction"], dtype=np.float64)
    y_eval = np.asarray(eval_split["target"], dtype=np.float64)
    p_eval = np.asarray(eval_split["prediction"], dtype=np.float64)

    mae_eval = float(np.mean(np.abs(y_eval - p_eval))) if y_eval.size else None
    mae_zero = float(np.mean(np.abs(y_eval))) if y_eval.size else None
    mae_ratio = (
        float(mae_eval / mae_zero)
        if mae_eval is not None and mae_zero is not None and mae_zero > 0.0
        else None
    )
    scale_lad = _lad_scale(p_cal, y_cal) if y_cal.size else None
    disp_cal = dispersion(cal_split)

    q_cal = cal_split.get("quantiles")
    q_eval = eval_split.get("quantiles")
    cov80_raw = cov95_raw = cov80_cqr = cov95_cqr = cse80_raw = ece_raw = None
    if q_cal is not None and q_eval is not None and y_cal.size and y_eval.size:
        cov80_raw = float(np.mean((y_eval >= q_eval[:, 1]) & (y_eval <= q_eval[:, 3])))
        cov95_raw = float(np.mean((y_eval >= q_eval[:, 0]) & (y_eval <= q_eval[:, 4])))

        center_cal = scale_lad * p_cal
        center_eval = scale_lad * p_eval
        q50_cal = q_cal[:, 2]
        q50_eval = q_eval[:, 2]

        low80_cal = center_cal - (q50_cal - q_cal[:, 1])
        hi80_cal = center_cal + (q_cal[:, 3] - q50_cal)
        m80 = _session_cdf_cqr_margin(low80_cal, hi80_cal, y_cal, cal_split["sessions"], 0.80)
        low80_eval = center_eval - (q50_eval - q_eval[:, 1]) - m80
        hi80_eval = center_eval + (q_eval[:, 3] - q50_eval) + m80
        cov80_cqr = float(np.mean((y_eval >= low80_eval) & (y_eval <= hi80_eval)))

        low95_cal = center_cal - (q50_cal - q_cal[:, 0])
        hi95_cal = center_cal + (q_cal[:, 4] - q50_cal)
        m95 = _session_cdf_cqr_margin(low95_cal, hi95_cal, y_cal, cal_split["sessions"], 0.95)
        low95_eval = center_eval - (q50_eval - q_eval[:, 0]) - m95
        hi95_eval = center_eval + (q_eval[:, 4] - q50_eval) + m95
        cov95_cqr = float(np.mean((y_eval >= low95_eval) & (y_eval <= hi95_eval)))

        commit_up = q_eval[:, 1] > 0.0
        commit_dn = q_eval[:, 3] < 0.0
        committed = commit_up | commit_dn
        if np.any(committed):
            wrong = (commit_up & (y_eval <= 0.0)) | (commit_dn & (y_eval >= 0.0))
            cse80_raw = float(np.mean(wrong[committed]))
        ece_raw = _ece_10bin(_implied_up_probability(q_eval), y_eval)

    return dict(
        mae_eval=mae_eval,
        mae_zero=mae_zero,
        mae_ratio=mae_ratio,
        lad_scale=scale_lad,
        dispersion=disp_cal,
        cov80_raw=cov80_raw,
        cov95_raw=cov95_raw,
        cov80_cqr=cov80_cqr,
        cov95_cqr=cov95_cqr,
        cse80_raw=cse80_raw,
        ece_raw=ece_raw,
    )


def offset_report(reports, provider, year, seed):
    """Desplazamiento conocido: su IC, la habilidad sin él y la combinación de cada modelo."""
    alphas, baseline = {}, None
    models = {}
    market_sessions = {}
    for name, report in sorted(reports.items()):
        run = Path(report["state"]) / "runs" / name / str(year) / f"seed-{seed}"
        per_split, volumes = {}, {}
        for split in SPLITS:
            digest = report["_splits"][split]["sample"]
            if (split, digest) not in alphas:
                alphas[split, digest] = provider.rows(run, split)
            per_split[split], volumes[split] = alphas[split, digest]

        eval_data = report["_splits"]["evaluation"]
        eval_alpha = per_split["evaluation"]
        for market, instant, start, end in eval_data["sessions"]:
            if end - start < MIN_ASSETS:
                continue
            entry = market_sessions.setdefault(market, [set(), set()])
            entry[0].add(instant)
            if np.isfinite(eval_alpha[start:end]).sum() >= MIN_ASSETS:
                entry[1].add(instant)

        if not any(np.isfinite(values).any() for values in per_split.values()):
            continue

        def ics(split, weight, free, model=True, report=report, per_split=per_split):
            data = report["_splits"][split]
            prediction = data["prediction"] if model else None
            return offset_sessions(data, per_split[split], weight, free=free, prediction=prediction)

        def units(
            split,
            free,
            liquid=False,
            model=True,
            report=report,
            per_split=per_split,
            volumes=volumes,
        ):
            data = report["_splits"][split]
            return summary(
                implied_sessions(
                    data,
                    per_split[split],
                    volumes[split] if liquid else None,
                    free=free,
                    prediction=data["prediction"] if model else None,
                )
            )

        if baseline is None:
            baseline = dict(
                label={s: summary(ics(s, 0.0, False, model=False)) for s in SPLITS},
                free={s: summary(ics(s, 0.0, True, model=False)) for s in SPLITS},
                liquid_label=units("evaluation", False, liquid=True, model=False),
                liquid_free=units("evaluation", True, liquid=True, model=False),
            )

        def decision(weight, ics=ics):
            values = [v for s in CALIBRATION_SPLIT for v in ics(s, weight, False).values()]
            return float(np.mean(values)) if values else -1.0

        weight = max(WEIGHTS, key=decision)
        scale = best_scale(report["_splits"], per_split)

        def scaled(
            split,
            liquid=False,
            report=report,
            per_split=per_split,
            volumes=volumes,
            scale=scale,
        ):
            data = report["_splits"][split]
            return summary(
                implied_sessions(
                    data,
                    per_split[split],
                    volumes[split] if liquid else None,
                    free=True,
                    prediction=scale * data["prediction"],
                )
            )

        models[name] = dict(
            label=summary(ics("evaluation", 1.0, False)),
            # La predicción del modelo para la etiqueta sin alfa es la suya más alfa.
            free={s: units(s, True) for s in SPLITS},
            # Los rangos de la predicción tal cual frente a la etiqueta sin alfa.
            free_ranks={s: summary(ics(s, 1.0, True)) for s in SPLITS},
            liquid_label=units("evaluation", False, liquid=True),
            liquid_free=units("evaluation", True, liquid=True),
            # Lo mismo con la predicción multiplicada por el factor elegido en calibración.
            scale=scale,
            free_scaled={s: scaled(s) for s in SPLITS},
            liquid_free_scaled=scaled("evaluation", liquid=True),
            dispersion=dispersion(report["_splits"]["calibration"], per_split["calibration"]),
            weight=weight,
            combined=summary(ics("evaluation", weight, False)),
        )

    offset_coverage = dict(
        by_market={
            market: dict(total_sessions=len(total), known_alpha_sessions=len(known))
            for market, (total, known) in sorted(market_sessions.items())
        },
        covered_markets=sorted(
            market for market, (_, known) in market_sessions.items() if len(known) > 0
        ),
        dropped_markets=sorted(
            market
            for market, (total, known) in market_sessions.items()
            if len(total) > 0 and len(known) == 0
        ),
    )
    if baseline is None and not offset_coverage["dropped_markets"]:
        return None
    return dict(baseline=baseline, models=models, offset_coverage=offset_coverage)


def _inference_module():
    """`dgx_ic_inference`, junto a esta herramienta instalada o en el repositorio."""
    path = Path(__file__).resolve().with_name("dgx_ic_inference.py")
    if not path.is_file():
        raise ValueError("Falta dgx_ic_inference.py para la inferencia entre modelos")
    spec = importlib.util.spec_from_file_location("dgx_ic_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stability_module():
    """`dgx_ic_stability`, junto a esta herramienta, o None si no está instalada."""
    path = Path(__file__).resolve().with_name("dgx_ic_stability.py")
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("dgx_ic_stability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _interval_calibration_module():
    """`dgx_interval_calibration`, junto a esta herramienta, o None si no está instalada."""
    path = Path(__file__).resolve().with_name("dgx_interval_calibration.py")
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("dgx_interval_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_interval_calibration(report, ic_mod):
    """Recalibra intervalos CQR-r Mondrian en `calibration` y evalúa en `evaluation`."""
    cal_split = report["_splits"]["calibration"]
    eval_split = report["_splits"]["evaluation"]
    if cal_split.get("quantiles") is None or eval_split.get("quantiles") is None:
        return None

    def _pack(split_data):
        sess = np.empty(len(split_data["target"]), dtype=np.int64)
        for s_idx, (_, _, start, end) in enumerate(split_data["sessions"]):
            sess[start:end] = s_idx
        return dict(
            target=split_data["target"],
            prediction=split_data["prediction"],
            sessions=sess,
            quantiles=split_data["quantiles"],
        )

    cal_data = _pack(cal_split)
    eval_data = _pack(eval_split)
    calibrated = ic_mod.recalibrate_mondrian_cqr_r(cal_data)
    return ic_mod.evaluate(calibrated, eval_data)


def session_table_with_diagnostics(origins, benchmark):
    """IC de evaluación por sesión y lista explícita de modelos excluidos de la inferencia."""
    series = {}
    for origin in origins:
        for name, report in origin["reports"].items():
            values = series.setdefault(name, {})
            for (market, instant), value in report["_ics"]["evaluation"].items():
                values[(origin["year"], instant, market)] = value
    if benchmark not in series or len(series[benchmark]) < 2:
        dropped = [
            dict(model=name, reason="benchmark_missing_or_insufficient")
            for name in sorted(series)
            if name != benchmark
        ]
        return None, dropped
    keys = sorted(series[benchmark])
    key_set = set(keys)
    names = []
    dropped = []
    for name in sorted(series):
        values = series[name]
        missing = len(key_set - set(values))
        extra = len(set(values) - key_set)
        if missing == 0 and extra == 0:
            names.append(name)
        else:
            dropped.append(
                dict(
                    model=name,
                    present_sessions=len(values),
                    benchmark_sessions=len(keys),
                    missing_sessions=missing,
                    extra_sessions=extra,
                    reason="session_mismatch_with_benchmark",
                )
            )
    if len(names) < 2:
        return None, dropped
    matrix = np.array([[series[name][key] for name in names] for key in keys], dtype=np.float64)
    return (keys, names, matrix), dropped


def session_table(origins, benchmark):
    """IC de evaluación por sesión de los modelos que comparten las sesiones de la referencia."""
    table, _ = session_table_with_diagnostics(origins, benchmark)
    return table


def write_sessions(path, table):
    keys, names, matrix = table
    lines = [",".join(["session", *names])]
    for (year, instant, market), row in zip(keys, matrix, strict=True):
        lines.append(",".join([f"{year}:{market}:{instant}", *(repr(float(v)) for v in row)]))
    path.write_text("\n".join(lines) + "\n")


def sample_digest(column):
    """Huella del orden de las muestras de un tramo: dos modelos se combinan si coincide."""
    digest = hashlib.sha256()
    for chunk in column.chunks:
        for buffer in chunk.buffers():
            if buffer is not None:
                digest.update(memoryview(buffer))
    return digest.hexdigest()


def load_split(run, split):
    """Predicciones de un tramo ordenadas por mercado, instante y muestra, con sus sesiones."""
    path = run / f"{split}-predictions.parquet"
    if not path.is_file():
        return None
    available = set(pq.read_schema(path).names)
    has_quantiles = all(col in available for col in QUANTILE_COLUMNS)
    read_cols = [*COLUMNS, *QUANTILE_COLUMNS] if has_quantiles else COLUMNS
    table = pq.read_table(path, columns=read_cols).sort_by(
        [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
    )
    market = table.column("market").combine_chunks().dictionary_encode()
    codes = market.indices.to_numpy(zero_copy_only=False)
    instant = table.column("prediction_at").cast("int64").to_numpy()
    change = (instant[1:] != instant[:-1]) | (codes[1:] != codes[:-1])
    starts = np.r_[0, np.flatnonzero(change) + 1]
    names = market.dictionary.to_pylist()
    bounds = list(zip(starts, np.r_[starts[1:], len(instant)], strict=True))
    quantiles = (
        np.column_stack(
            [table.column(col).to_numpy().astype(np.float64) for col in QUANTILE_COLUMNS]
        )
        if has_quantiles
        else None
    )
    return dict(
        sample=sample_digest(table.column("sample_id")),
        target=table.column("target").to_numpy(),
        prediction=table.column("prediction").to_numpy(),
        quantiles=quantiles,
        sessions=[(names[codes[start]], int(instant[start]), start, end) for start, end in bounds],
    )


def session_ic(data, prediction=None):
    """IC de rangos de cada sesión con suficientes activos: {(mercado, instante): IC}."""
    prediction = data["prediction"] if prediction is None else prediction
    result = {}
    for market, instant, start, end in data["sessions"]:
        if end - start < MIN_ASSETS:
            continue
        value = correlation(
            average_ranks(prediction[start:end]), average_ranks(data["target"][start:end])
        )
        if not math.isnan(value):
            result[(market, instant)] = value
    return result


def automatic_newey_west_lags(n_obs):
    """Regla automática de Newey y West (1994): ``floor(4 * (T / 100) ** (2 / 9))``."""
    return min(int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))), n_obs - 1)


def _newey_west_hac(values, lags=None, *, blocks=None):
    """Devuelve ``(t_hac, se_hac, max_lags)`` con el núcleo de Bartlett segmentado por bloques."""
    arr = np.asarray(values, dtype=np.float64)
    count = int(arr.size)
    if count < 2:
        return None, None, 0
    if lags is not None and (lags < 0 or lags >= count):
        raise ValueError(f"lags debe estar en [0, {count - 1}]; se recibió {lags}.")
    mean = float(np.mean(arr))
    if bool(np.all(arr == arr[0])):
        return None, 0.0, 0

    centered = arr - mean
    if blocks is None:
        partitions = [np.arange(count, dtype=np.int64)]
    else:
        grouped = {}
        for idx, blk in enumerate(blocks):
            grouped.setdefault(blk, []).append(idx)
        partitions = [np.asarray(idxs, dtype=np.int64) for idxs in grouped.values()]

    total_v = 0.0
    max_lags = 0
    for idxs in partitions:
        u = centered[idxs]
        tb = int(u.size)
        if tb == 0:
            continue
        l_b = (
            (automatic_newey_west_lags(tb) if tb > 1 else 0) if lags is None else min(lags, tb - 1)
        )
        max_lags = max(max_lags, l_b)
        v_b = float(np.dot(u, u))
        for k in range(1, l_b + 1):
            weight = 1.0 - k / (l_b + 1.0)
            v_b += 2.0 * weight * float(np.dot(u[k:], u[:-k]))
        total_v += max(v_b, 0.0)

    lrv = total_v / count
    if lrv <= 0.0:
        return None, 0.0, max_lags
    se = math.sqrt(lrv / count)
    return float(mean / se), float(se), int(max_lags)


def newey_west_t(values, lags=None, *, blocks=None):
    """t de la media con varianza HAC de Newey-West (rezago automático compartido), o None.

    Usa exactamente la regla automática ``L = floor(4 * (T / 100) ** (2 / 9))`` de
    ``dgx_ic_inference.newey_west_t`` y segmenta por ``blocks`` para no cruzar mercados ni
    saltos de origen.
    """
    t_hac, _, _ = _newey_west_hac(values, lags=lags, blocks=blocks)
    return t_hac


def _series_and_blocks(values, blocks=None):
    """Extrae array ordenado y bloques ``key[:-1]`` cuando ``values`` es un dict de sesiones."""
    if isinstance(values, dict):
        keys = sorted(values)
        arr = np.asarray([values[key] for key in keys], dtype=np.float64)
        if blocks is None and keys and all(isinstance(k, tuple) and len(k) >= 2 for k in keys):
            blocks = [key[:-1] for key in keys]
        return arr, blocks
    return np.asarray(list(values), dtype=np.float64), blocks


def summary(values, *, blocks=None):
    """Media, dispersión, ICIR y t de Newey-West segmentado por bloque de unas sesiones."""
    arr, use_blocks = _series_and_blocks(values, blocks=blocks)
    if len(arr) == 0:
        return dict(sessions=0, mean=None, std=None, icir=None, t_nw=None)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return dict(
        sessions=int(len(arr)),
        mean=float(arr.mean()),
        std=std,
        icir=float(arr.mean() / std) if std > 0 else None,
        t_nw=newey_west_t(arr, blocks=use_blocks),
    )


def paired(candidate, control):
    """Diferencia de IC sesión a sesión con estadístico t HAC de Newey-West por bloque."""
    keys = sorted(set(candidate) & set(control))
    if len(keys) < 2:
        return dict(
            sessions=len(keys),
            mean=None,
            se=None,
            se_hac=None,
            se_iid=None,
            t=None,
            t_iid=None,
            hac_lags=0,
        )
    blocks = (
        [key[:-1] for key in keys]
        if all(isinstance(key, tuple) and len(key) >= 2 for key in keys)
        else None
    )
    difference = np.asarray([candidate[key] - control[key] for key in keys], dtype=np.float64)
    mean = float(difference.mean())
    se_iid = float(difference.std(ddof=1) / math.sqrt(len(keys)))
    t_iid = float(mean / se_iid) if se_iid > 0 else None
    t_hac, se_hac, hac_lags = _newey_west_hac(difference, blocks=blocks)
    return dict(
        sessions=len(keys),
        mean=mean,
        se=se_hac if se_hac is not None else se_iid,
        se_hac=se_hac,
        se_iid=se_iid,
        t=t_hac,
        t_iid=t_iid,
        hac_lags=hac_lags,
    )


def pooled(ics, splits=DECISION):
    """Sesiones de varios tramos juntas. Con los de decisión es lo único con lo que se elige."""
    merged = {}
    for split in splits:
        merged.update({(split, *key): value for key, value in ics[split].items()})
    return merged


def epoch_curve(run):
    """IC de rangos de validación por época, del informe del ajuste."""
    path = run / "fit" / "run.json"
    if not path.is_file():
        return []
    history = json.loads(path.read_text()).get("history") or []
    return [
        (int(entry["epoch"]), float(entry["validation"]["session_rank_ic"]))
        for entry in history
        if (entry.get("validation") or {}).get("session_rank_ic") is not None
    ]


def completed(state, year, seed):
    """Registros de la cola de los trabajos terminados de un origen y una semilla."""
    records = {}
    for path in sorted((state / "queue").glob(f"*--{year}--seed-{seed}.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "completed":
            records[record["model"]] = record
    return records


def model_report(state, record, year, seed, decision_splits=DECISION):
    run = state / "runs" / record["model"] / str(year) / f"seed-{seed}"
    splits = {split: load_split(run, split) for split in SPLITS}
    if any(data is None for data in splits.values()):
        return None
    ics = {split: session_ic(data) for split, data in splits.items()}
    curve = epoch_curve(run)
    late = [value for epoch, value in curve if epoch >= 2]
    # Un trabajo reanudado reparte su tiempo entre varios intentos: se suman todos.
    seconds = [item["seconds"] for item in record.get("attempts") or [] if item.get("seconds")]
    result = record.get("result") or {}
    markets = sorted({key[0] for values in ics.values() for key in values})
    by_market = {
        market: {
            split: summary({key: v for key, v in ics[split].items() if key[0] == market})
            for split in SPLITS
        }
        for market in markets
    }
    score = result.get("best_score")
    diagnostics = _calibration_diagnostics(splits["calibration"], splits["evaluation"])
    return dict(
        model=record["model"],
        tier=record.get("tier"),
        state=str(state),
        best_epoch=result.get("best_epoch"),
        selected_validation_ic=None if score is None else 1 - 2 * score,
        hours=sum(seconds) / 3600 if seconds else None,
        epochs=curve,
        epoch_std=float(np.std(late, ddof=1)) if len(late) > 1 else None,
        ic={split: summary(values) for split, values in ics.items()},
        decision_ic=summary(pooled(ics, decision_splits))["mean"],
        ic_by_market=by_market if len(markets) > 1 else {},
        **diagnostics,
        _ics=ics,
        _splits=splits,
        _ranks={},
    )


def normalized_ranks(report, split):
    """Rangos de las predicciones dentro de cada sesión, entre 0 y 1. Se calculan una vez."""
    if split not in report["_ranks"]:
        data = report["_splits"][split]
        ranks = np.zeros(len(data["target"]), dtype=np.float64)
        for _, _, start, end in data["sessions"]:
            ranks[start:end] = average_ranks(data["prediction"][start:end]) / (end - start)
        report["_ranks"][split] = ranks
    return report["_ranks"][split]


def blend(members, split):
    """IC por sesión de la media de rangos de varios modelos, o None si no comparten muestras."""
    base = members[0]["_splits"][split]
    if any(member["_splits"][split]["sample"] != base["sample"] for member in members[1:]):
        return None
    total = sum(normalized_ranks(member, split) for member in members)
    return session_ic(base, total / len(members))


def forward_selection(reports, limit, decision_splits=DECISION):
    """Combinación elegida hacia delante por el IC de los tramos de decisión declarados."""
    chosen, steps, best = [], [], None
    while len(chosen) < limit:
        candidates = []
        for name in sorted(reports):
            if name in chosen:
                continue
            members = [reports[item] for item in (*chosen, name)]
            blends = {split: blend(members, split) for split in decision_splits}
            if any(item is None for item in blends.values()):
                continue
            candidates.append((summary(pooled(blends, decision_splits))["mean"], name))
        if not candidates:
            break
        score, name = max(candidates)
        if score is None or (best is not None and score <= best):
            break
        best = score
        chosen.append(name)
        members = [reports[item] for item in chosen]
        evaluation = blend(members, "evaluation")
        steps.append(
            dict(
                members=list(chosen),
                decision_ic=score,
                evaluation=summary(evaluation) if evaluation is not None else None,
            )
        )
    return steps


def contrast(name, control, group, candidate, base, decision_splits=DECISION):
    return dict(
        model=name,
        control=control,
        group=group,
        decision=paired(
            pooled(candidate["_ics"], decision_splits), pooled(base["_ics"], decision_splits)
        ),
        validation=paired(candidate["_ics"]["validation"], base["_ics"]["validation"]),
        calibration=paired(candidate["_ics"]["calibration"], base["_ics"]["calibration"]),
        evaluation=paired(candidate["_ics"]["evaluation"], base["_ics"]["evaluation"]),
    )


def origin_report(
    state,
    year,
    seed,
    controls,
    groups,
    references,
    ensemble_limit,
    offset=False,
    decision_splits=DECISION,
    calibration_report=False,
):
    records = completed(state, year, seed)
    reports = {}
    missing_splits = []
    for name, record in records.items():
        report = model_report(state, record, year, seed, decision_splits=decision_splits)
        if report is not None:
            reports[name] = report
        else:
            run = state / "runs" / name / str(year) / f"seed-{seed}"
            absent = [s for s in SPLITS if not (run / f"{s}-predictions.parquet").is_file()]
            missing_splits.append(dict(model=name, state=str(state), missing=absent))
    for reference in references:
        for name, record in completed(reference, year, seed).items():
            if name not in reports:
                report = model_report(
                    reference, record, year, seed, decision_splits=decision_splits
                )
                if report is not None:
                    reports[name] = report
                else:
                    run = reference / "runs" / name / str(year) / f"seed-{seed}"
                    absent = [s for s in SPLITS if not (run / f"{s}-predictions.parquet").is_file()]
                    missing_splits.append(dict(model=name, state=str(reference), missing=absent))
    contrasts = [
        contrast(
            name,
            control,
            groups.get(name),
            reports[name],
            reports[control],
            decision_splits=decision_splits,
        )
        for name, control in sorted(controls.items())
        if name in reports and control in reports
    ]
    ensembles = (
        forward_selection(reports, ensemble_limit, decision_splits=decision_splits)
        if len(reports) > 1
        else []
    )
    known = offset_report(reports, KnownOffset(state, year), year, seed) if offset else None
    interval_cal = None
    if calibration_report:
        ic_mod = _interval_calibration_module()
        if ic_mod is not None:
            interval_cal = {}
            for name, rep in sorted(reports.items()):
                ev_cal = _model_interval_calibration(rep, ic_mod)
                if ev_cal is not None:
                    interval_cal[name] = ev_cal
    for report in reports.values():
        # Los contrastes entre orígenes solo necesitan el IC por sesión: se liberan las filas.
        report["_splits"], report["_ranks"] = None, None
    return dict(
        year=year,
        completed=sorted(records),
        missing_splits=missing_splits,
        reports=reports,
        contrasts=contrasts,
        ensembles=ensembles,
        offset=known,
        offset_coverage=known["offset_coverage"] if known is not None else None,
        interval_calibration=interval_cal,
    )


def aggregate(origins, controls, groups, decision_splits=DECISION):
    """Contraste con las sesiones de todos los orígenes en los que están los dos modelos."""
    rows = []
    for name, control in sorted(controls.items()):
        merged = {kind: ({}, {}) for kind in ("decision", "evaluation")}
        years, means = [], []
        for origin in origins:
            reports = origin["reports"]
            if name not in reports or control not in reports:
                continue
            years.append(origin["year"])
            decision = (
                pooled(reports[name]["_ics"], decision_splits),
                pooled(reports[control]["_ics"], decision_splits),
            )
            means.append(paired(*decision)["mean"])
            sides = dict(
                decision=decision,
                evaluation=(
                    reports[name]["_ics"]["evaluation"],
                    reports[control]["_ics"]["evaluation"],
                ),
            )
            for kind, (left, right) in sides.items():
                merged[kind][0].update({(origin["year"], *key): v for key, v in left.items()})
                merged[kind][1].update({(origin["year"], *key): v for key, v in right.items()})
        if not years:
            continue
        rows.append(
            dict(
                model=name,
                control=control,
                group=groups.get(name),
                origins=years,
                origin_means=means,
                mean_of_origins=float(np.mean([m for m in means if m is not None]))
                if means
                else None,
                decision=paired(*merged["decision"]),
                evaluation=paired(*merged["evaluation"]),
            )
        )
    return rows


def comparison(origins, other, seed, decision_splits=DECISION):
    """Cada modelo de este estado frente al mismo modelo de otro, en sus sesiones comunes."""
    rows = []
    for origin in origins:
        records = completed(other, origin["year"], seed)
        for name in sorted(set(origin["reports"]) & set(records)):
            mine = origin["reports"][name]
            # Un modelo tomado del otro estado como referencia no se compara consigo mismo.
            if mine["state"] == str(other):
                continue
            report = model_report(
                other, records[name], origin["year"], seed, decision_splits=decision_splits
            )
            if report is None:
                continue
            rows.append(
                dict(
                    model=name,
                    year=origin["year"],
                    decision=paired(
                        pooled(mine["_ics"], decision_splits),
                        pooled(report["_ics"], decision_splits),
                    ),
                    evaluation=paired(mine["_ics"]["evaluation"], report["_ics"]["evaluation"]),
                )
            )
    return rows


def build(
    state,
    years,
    seed,
    controls,
    groups,
    references=(),
    compare=None,
    ensemble_limit=6,
    offset=False,
    inference=None,
    decision_splits=DECISION,
    calibration_report=False,
):
    decision_splits = tuple(decision_splits)
    origins = [
        origin_report(
            state,
            year,
            seed,
            controls,
            groups,
            references,
            ensemble_limit,
            offset,
            decision_splits=decision_splits,
            calibration_report=calibration_report,
        )
        for year in years
    ]
    sessions, dropped_from_inference = (
        session_table_with_diagnostics(origins, inference) if inference else (None, [])
    )
    document = dict(
        schema_version=1,
        kind="dgx_screening_report",
        state=str(state),
        references=[str(path) for path in references],
        seed=seed,
        min_assets=MIN_ASSETS,
        decision_splits=list(decision_splits),
        origins=[
            dict(
                year=origin["year"],
                completed=origin["completed"],
                missing_splits=origin["missing_splits"],
                models={
                    name: {key: value for key, value in report.items() if not key.startswith("_")}
                    for name, report in origin["reports"].items()
                },
                contrasts=origin["contrasts"],
                ensembles=origin["ensembles"],
                offset=origin["offset"],
                offset_coverage=origin["offset_coverage"],
                interval_calibration=origin.get("interval_calibration"),
            )
            for origin in origins
        ],
        aggregate=aggregate(origins, controls, groups, decision_splits=decision_splits)
        if len(origins) > 1
        else [],
        comparison=dict(
            state=str(compare),
            rows=comparison(origins, compare, seed, decision_splits=decision_splits),
        )
        if compare
        else None,
        inference=None,
        dropped_from_inference=dropped_from_inference,
        stability=None,
    )
    if inference:
        # La tabla de sesiones no se guarda en el informe: `main` la escribe en su propio archivo.
        document["_sessions"] = sessions
    return document


def _f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _contrast_rows(rows, lines):
    lines += [
        "| Variante | Control | Diferencia de decisión | t | Diferencia en evaluación | t |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted(rows, key=lambda item: -(item["decision"]["mean"] or -9)):
        lines.append(
            f"| `{item['model']}` | `{item['control']}` | {_f(item['decision']['mean'])} | "
            f"{_f(item['decision']['t'], 2)} | {_f(item['evaluation']['mean'])} | "
            f"{_f(item['evaluation']['t'], 2)} |"
        )
    lines.append("")


def markdown(report):
    years = ", ".join(str(origin["year"]) for origin in report["origins"])
    lines = [
        f"# Criba de los orígenes {years}, semilla {report['seed']}",
        "",
        f"Estado: `{report['state']}`. IC de rangos por sesión. Se decide solo con validación y "
        "calibración; la evaluación es la evidencia fuera de muestra de cada origen.",
        "",
    ]
    if report["aggregate"]:
        lines += ["## Contrastes pareados con todos los orígenes", ""]
        _contrast_rows(report["aggregate"], lines)
    for origin in report["origins"]:
        lines += [
            f"## Origen {origin['year']}: {len(origin['models'])} modelos",
            "",
            "| Modelo | Nivel | Decisión | Validación | Calibración | Evaluación | "
            "ICIR de evaluación | Desviación por época | Horas |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        ordered = sorted(origin["models"].values(), key=lambda item: -(item["decision_ic"] or -9))
        for item in ordered:
            ic = item["ic"]
            cells = [
                _f(item["decision_ic"]),
                _f(ic["validation"]["mean"]),
                _f(ic["calibration"]["mean"]),
                _f(ic["evaluation"]["mean"]),
                _f(ic["evaluation"]["icir"], 2),
                _f(item["epoch_std"]),
                _f(item["hours"], 2),
            ]
            lines.append(f"| `{item['model']}` | {item['tier']} | " + " | ".join(cells) + " |")
        markets = sorted({market for item in ordered for market in item["ic_by_market"]})
        if markets:
            lines += ["", "IC de evaluación por mercado:", ""]
            lines.append("| Modelo | " + " | ".join(markets) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in markets) + " |")
            for item in ordered:
                cells = [
                    _f((item["ic_by_market"].get(market) or {}).get("evaluation", {}).get("mean"))
                    for market in markets
                ]
                lines.append(f"| `{item['model']}` | " + " | ".join(cells) + " |")
        lines += ["", f"### Contrastes pareados del origen {origin['year']}", ""]
        _contrast_rows(origin["contrasts"], lines)
        lines += [
            f"### Combinaciones por media de rangos del origen {origin['year']}",
            "",
            "| Miembros | IC de decisión | IC de evaluación | ICIR de evaluación |",
            "| --- | --- | --- | --- |",
        ]
        for step in origin["ensembles"]:
            members = ", ".join(f"`{name}`" for name in step["members"])
            evaluation = step["evaluation"] or {}
            lines.append(
                f"| {members} | {_f(step['decision_ic'])} | {_f(evaluation.get('mean'))} | "
                f"{_f(evaluation.get('icir'), 2)} |"
            )
        lines.append("")
        known = origin.get("offset")
        if known:
            base = known.get("baseline")
            coverage = known.get("offset_coverage") or {}
            dropped_mkts = coverage.get("dropped_markets") or []
            covered_mkts = coverage.get("covered_markets") or []
            lines += [
                f"### Desplazamiento conocido de la etiqueta en el origen {origin['year']}",
                "",
            ]
            if dropped_mkts:
                dropped_str = ", ".join(f"`{m}`" for m in dropped_mkts)
                covered_str = ", ".join(f"`{m}`" for m in covered_mkts) or "ninguno"
                lines += [
                    f"> **AVISO**: mercados excluidos de la tabla de desplazamiento conocido por "
                    f"ausencia de modelo de mercado (`has_market_model == 0`): {dropped_str} "
                    f"(mercados cubiertos: {covered_str}).",
                    "",
                ]
            if base is not None:
                lines += [
                    "La etiqueta resta el alfa de las 252 sesiones anteriores, que se conoce al "
                    "predecir. `-alfa` por sí solo tiene un IC de evaluación de "
                    f"{_f(base['label']['evaluation']['mean'])} "
                    f"({_f(base['liquid_label']['mean'])} en el tercio más líquido). La etiqueta "
                    "sin alfa es el rendimiento que se puede operar, y la predicción de un modelo "
                    "para ella es la suya más alfa: `+alfa` por sí solo tiene "
                    f"{_f(base['free']['evaluation']['mean'])} "
                    f"({_f(base['liquid_free']['mean'])} en el tercio más líquido).",
                    "",
                    "Sumar alfa solo da la predicción de lo operable si la del modelo tiene la "
                    "escala de los retornos. «Dispersión» es la de la predicción entre la de la "
                    "etiqueta por sesión, en calibración: un modelo calibrado queda muy por debajo "
                    "de uno, y uno ajustado para ordenar cada sesión puede quedar por encima. Por "
                    "eso se da también la medida «a escala», con la predicción multiplicada por el "
                    "factor que da más IC sin alfa estrictamente en calibración.",
                    "",
                    "| Modelo | IC de evaluación | t NW | IC sin alfa | t NW | Dispersión | "
                    "Factor | Sin alfa a escala | t NW | Líquido | Líquido sin alfa | Líquido sin "
                    "alfa a escala | Peso | Con -alfa |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
                    "--- | --- |",
                ]
                ordered_known = sorted(
                    known["models"].items(), key=lambda item: -(item[1]["combined"]["mean"] or -9)
                )
                for name, item in ordered_known:
                    free, scaled = item["free"]["evaluation"], item["free_scaled"]["evaluation"]
                    lines.append(
                        f"| `{name}` | {_f(item['label']['mean'])} | "
                        f"{_f(item['label']['t_nw'], 1)} | "
                        f"{_f(free['mean'])} | {_f(free['t_nw'], 1)} | "
                        f"{_f(item['dispersion'], 2)} | {_f(item['scale'], 2)} | "
                        f"{_f(scaled['mean'])} | {_f(scaled['t_nw'], 1)} | "
                        f"{_f(item['liquid_label']['mean'])} | "
                        f"{_f(item['liquid_free']['mean'])} | "
                        f"{_f(item['liquid_free_scaled']['mean'])} | "
                        f"{_f(item['weight'], 1)} | {_f(item['combined']['mean'])} |"
                    )
                lines.append("")
    dropped_inf = report.get("dropped_from_inference") or []
    inference = report.get("inference")
    if inference:
        meta, spa, mcs = inference["metadata"], inference["spa"], inference["mcs"]
        lines += [
            f"## Inferencia entre modelos frente a `{meta['benchmark']}`",
            "",
            f"IC de evaluación de {meta['n_models']} modelos en {meta['n_sessions']} sesiones "
            f"comunes (`sessions.csv`). Bootstrap estacionario con {meta['bootstrap']} réplicas y "
            f"bloque medio de {_f(meta['bootstrap_block'], 1)} sesiones, las mismas para todos "
            "los modelos.",
            "",
        ]
        if dropped_inf:
            names_dropped = ", ".join(f"`{entry['model']}`" for entry in dropped_inf)
            lines += [
                f"> **AVISO**: modelos excluidos de la inferencia por sesiones incompletas "
                f"frente a la referencia: {names_dropped}.",
                "",
            ]
        lines += [
            f"- Contraste SPA de Hansen: p = {_f(spa['p_consistent'], 3)} (entre "
            f"{_f(spa['p_lower'], 3)} y {_f(spa['p_upper'], 3)}) de que ningún modelo supera a "
            f"la referencia. El de mayor IC es `{spa['best_candidate']}`.",
            f"- Conjunto de confianza de modelos al {round(100 * (1 - mcs['alpha']))} %: "
            f"{len(mcs['included'])} de {meta['n_models']} modelos no se distinguen del mejor.",
            "",
            "| Modelo | IC de evaluación | t NW | Intervalo al 95 % | Diferencia con la "
            "referencia | p DM | p MCS |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        ordered_models = sorted(inference["models"].items(), key=lambda item: -item[1]["mean"])
        for name, item in ordered_models:
            interval, test = item["bootstrap_ci_95"], item["diebold_mariano"]
            mark = "" if name in mcs["included"] else " (fuera)"
            lines.append(
                f"| `{name}` | {_f(item['mean'])} | {_f(item['t_newey_west'], 1)} | "
                f"{_f(interval['low'])} a {_f(interval['high'])} | {_f(test['mean'])} | "
                f"{_f(test['p_value'], 3)} | {_f(mcs['p_values'][name], 3)}{mark} |"
            )
        lines.append("")
    elif dropped_inf:
        names_dropped = ", ".join(f"`{entry['model']}`" for entry in dropped_inf)
        lines += [
            f"> **AVISO**: modelos excluidos de la inferencia: {names_dropped}.",
            "",
        ]
    stability = report.get("stability")
    if stability:
        ordered = sorted(stability.items(), key=lambda item: -item[1]["all"]["mean"])
        lines += [
            "## Estabilidad del IC de evaluación",
            "",
            "Con las mismas sesiones (`sessions.csv`). ICIR es la media del IC entre su "
            "desviación por sesión. La caída máxima es la mayor pérdida del IC acumulado desde "
            "un máximo anterior, la racha cuenta sesiones seguidas con IC nulo o negativo y la "
            "peor ventana es la media más baja de 21 sesiones seguidas.",
            "",
            "| Modelo | Ámbito | Sesiones | ICIR | Sesiones con IC positivo | Meses con IC "
            "positivo | Peor mes | Caída máxima | Racha | Peor ventana | Segunda mitad menos "
            "primera |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, scopes in ordered:
            for scope, item in scopes.items():
                month, window, halves = item["worst_month"], item["worst_window"], item["halves"]
                lines.append(
                    f"| `{name}` | {'todos' if scope == 'all' else scope} | {item['sessions']} | "
                    f"{_f(item['icir'], 2)} | {_f(100 * item['positive_fraction'], 1)} % | "
                    f"{_f(100 * item['months_positive_fraction'], 1)} % | "
                    f"{month['month']} ({_f(month['mean'])}) | "
                    f"{_f(item['max_drawdown']['depth'], 2)} | {item['longest_negative_run']} | "
                    f"{_f(window['mean']) if window else '—'} | "
                    f"{_f(halves['difference']) if halves else '—'} |"
                )
        lines.append("")
    if report["comparison"]:
        lines += [
            f"## Este estado frente a `{report['comparison']['state']}` en sus sesiones comunes",
            "",
            "| Modelo | Origen | Diferencia de decisión | t | Diferencia en evaluación | t |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in report["comparison"]["rows"]:
            lines.append(
                f"| `{item['model']}` | {item['year']} | {_f(item['decision']['mean'])} | "
                f"{_f(item['decision']['t'], 2)} | {_f(item['evaluation']['mean'])} | "
                f"{_f(item['evaluation']['t'], 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--controls", type=Path, help="JSON con controls {variante: control} y groups"
    )
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--compare", type=Path)
    parser.add_argument(
        "--offset", action="store_true", help="Medir el alfa conocido de la etiqueta"
    )
    parser.add_argument(
        "--inference", metavar="MODELO", help="Inferencia entre modelos frente a esta referencia"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--decision-split",
        choices=("pooled", "calibration", "validation"),
        default="pooled",
        help="Tramos de decisión: validación y calibración juntas (por defecto) o uno solo.",
    )
    parser.add_argument(
        "--calibration-report",
        action="store_true",
        help="Incluir recalibración CQR-r Mondrian de intervalos ajustada solo en calibración.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    declared = json.loads(args.controls.read_text()) if args.controls else {}
    report = build(
        args.state,
        args.year,
        args.seed,
        declared.get("controls", {}),
        declared.get("groups", {}),
        references=args.reference,
        compare=args.compare,
        offset=args.offset,
        inference=args.inference,
        decision_splits=DECISION if args.decision_split == "pooled" else (args.decision_split,),
        calibration_report=args.calibration_report,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    sessions = report.pop("_sessions", None)
    if sessions is not None:
        write_sessions(args.output / "sessions.csv", sessions)
        report["inference"] = _inference_module().analizar_tabla_ic(
            args.output / "sessions.csv", args.inference, bootstrap=args.bootstrap, seed=0
        )
        stability = _stability_module()
        if stability is not None:
            report["stability"] = stability.stability_table(args.output / "sessions.csv")
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    )
    (args.output / "report.md").write_text(markdown(report))
    count = sum(len(origin["models"]) for origin in report["origins"])
    print(f"{count} modelos en {len(report['origins'])} orígenes: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
