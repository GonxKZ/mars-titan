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
ese es el «IC sin alfa». Las dos medidas se dan también en el tercio más líquido de cada sesión
(volumen medio en dólares de 21 sesiones), y los IC de evaluación llevan su t de Newey-West.

Con `--inference MODELO` escribe `sessions.csv`, el IC de evaluación de cada modelo en cada
sesión, y añade la inferencia de `dgx_ic_inference.py` frente a ese modelo de referencia:
intervalos por bootstrap estacionario, Diebold-Mariano, el contraste SPA de Hansen (¿supera
algún modelo a la referencia una vez descontado que se han probado todos?) y el conjunto de
confianza de modelos de Hansen, Lunde y Nason.

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
DECISION = ("validation", "calibration")
MIN_ASSETS = 30
COLUMNS = ["sample_id", "market", "prediction_at", "target", "prediction"]


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


def offset_report(reports, provider, year, seed):
    """Desplazamiento conocido: su IC, la habilidad sin él y la combinación de cada modelo."""
    alphas, baseline = {}, None
    models = {}
    for name, report in sorted(reports.items()):
        run = Path(report["state"]) / "runs" / name / str(year) / f"seed-{seed}"
        per_split, volumes = {}, {}
        for split in SPLITS:
            digest = report["_splits"][split]["sample"]
            if (split, digest) not in alphas:
                alphas[split, digest] = provider.rows(run, split)
            per_split[split], volumes[split] = alphas[split, digest]
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
                ).values()
            )

        if baseline is None:
            baseline = dict(
                label={s: summary(ics(s, 0.0, False, model=False).values()) for s in SPLITS},
                free={s: summary(ics(s, 0.0, True, model=False).values()) for s in SPLITS},
                liquid_label=units("evaluation", False, liquid=True, model=False),
                liquid_free=units("evaluation", True, liquid=True, model=False),
            )

        def decision(weight, ics=ics):
            values = [v for s in DECISION for v in ics(s, weight, False).values()]
            return float(np.mean(values)) if values else -1.0

        weight = max(WEIGHTS, key=decision)
        models[name] = dict(
            label=summary(ics("evaluation", 1.0, False).values()),
            # La predicción del modelo para la etiqueta sin alfa es la suya más alfa.
            free={s: units(s, True) for s in SPLITS},
            # Los rangos de la predicción tal cual frente a la etiqueta sin alfa.
            free_ranks={s: summary(ics(s, 1.0, True).values()) for s in SPLITS},
            liquid_label=units("evaluation", False, liquid=True),
            liquid_free=units("evaluation", True, liquid=True),
            weight=weight,
            combined=summary(ics("evaluation", weight, False).values()),
        )
    return dict(baseline=baseline, models=models) if baseline is not None else None


def _inference_module():
    """`dgx_ic_inference`, junto a esta herramienta instalada o en el repositorio."""
    path = Path(__file__).resolve().with_name("dgx_ic_inference.py")
    if not path.is_file():
        raise ValueError("Falta dgx_ic_inference.py para la inferencia entre modelos")
    spec = importlib.util.spec_from_file_location("dgx_ic_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_table(origins, benchmark):
    """IC de evaluación por sesión de los modelos que comparten las sesiones de la referencia.

    Devuelve las sesiones (origen, instante y mercado) en orden temporal, los nombres de los
    modelos y la matriz [sesiones, modelos], o None si la referencia no está en los orígenes.
    """
    series = {}
    for origin in origins:
        for name, report in origin["reports"].items():
            values = series.setdefault(name, {})
            for (market, instant), value in report["_ics"]["evaluation"].items():
                values[(origin["year"], instant, market)] = value
    if benchmark not in series or len(series[benchmark]) < 2:
        return None
    keys = sorted(series[benchmark])
    names = sorted(
        name
        for name, values in series.items()
        if len(values) == len(keys) and all(key in values for key in keys)
    )
    if len(names) < 2:
        return None
    matrix = np.array([[series[name][key] for name in names] for key in keys], dtype=np.float64)
    return keys, names, matrix


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
    table = pq.read_table(path, columns=COLUMNS).sort_by(
        [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
    )
    market = table.column("market").combine_chunks().dictionary_encode()
    codes = market.indices.to_numpy(zero_copy_only=False)
    instant = table.column("prediction_at").cast("int64").to_numpy()
    change = (instant[1:] != instant[:-1]) | (codes[1:] != codes[:-1])
    starts = np.r_[0, np.flatnonzero(change) + 1]
    names = market.dictionary.to_pylist()
    bounds = list(zip(starts, np.r_[starts[1:], len(instant)], strict=True))
    return dict(
        sample=sample_digest(table.column("sample_id")),
        target=table.column("target").to_numpy(),
        prediction=table.column("prediction").to_numpy(),
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


NEWEY_WEST_LAGS = 10


def newey_west_t(values, lags=NEWEY_WEST_LAGS):
    """t de la media con la varianza de Newey y West (núcleo de Bartlett), o None.

    Los IC de sesiones consecutivas no son independientes: el desplazamiento conocido de la
    etiqueta comparte 251 de sus 252 sesiones de un día al siguiente.
    """
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count < 3 * lags:
        return None
    centered = values - values.mean()
    variance = float(centered @ centered) / count
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * float(centered[lag:] @ centered[:-lag]) / count
    return float(values.mean() / math.sqrt(variance / count)) if variance > 0 else None


def summary(values):
    """Media, dispersión e ICIR de los IC de unas sesiones, en su orden temporal."""
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return dict(sessions=0, mean=None, std=None, icir=None, t_nw=None)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return dict(
        sessions=int(len(values)),
        mean=float(values.mean()),
        std=std,
        icir=float(values.mean() / std) if std > 0 else None,
        t_nw=newey_west_t(values),
    )


def paired(candidate, control):
    """Diferencia de IC sesión a sesión entre dos modelos sobre sus sesiones comunes."""
    keys = sorted(set(candidate) & set(control))
    if len(keys) < 2:
        return dict(sessions=len(keys), mean=None, se=None, t=None)
    difference = np.asarray([candidate[key] - control[key] for key in keys])
    se = float(difference.std(ddof=1) / math.sqrt(len(keys)))
    return dict(
        sessions=len(keys),
        mean=float(difference.mean()),
        se=se,
        t=float(difference.mean() / se) if se > 0 else None,
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


def model_report(state, record, year, seed):
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
            split: summary(v for key, v in ics[split].items() if key[0] == market)
            for split in SPLITS
        }
        for market in markets
    }
    score = result.get("best_score")
    return dict(
        model=record["model"],
        tier=record.get("tier"),
        state=str(state),
        best_epoch=result.get("best_epoch"),
        selected_validation_ic=None if score is None else 1 - 2 * score,
        hours=sum(seconds) / 3600 if seconds else None,
        epochs=curve,
        epoch_std=float(np.std(late, ddof=1)) if len(late) > 1 else None,
        ic={split: summary(values.values()) for split, values in ics.items()},
        decision_ic=summary(pooled(ics).values())["mean"],
        ic_by_market=by_market if len(markets) > 1 else {},
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


def forward_selection(reports, limit):
    """Combinación elegida hacia delante por el IC de validación más calibración."""
    chosen, steps, best = [], [], None
    while len(chosen) < limit:
        candidates = []
        for name in sorted(reports):
            if name in chosen:
                continue
            members = [reports[item] for item in (*chosen, name)]
            blends = {split: blend(members, split) for split in DECISION}
            if any(item is None for item in blends.values()):
                continue
            candidates.append((summary(pooled(blends).values())["mean"], name))
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
                evaluation=summary(evaluation.values()) if evaluation is not None else None,
            )
        )
    return steps


def contrast(name, control, group, candidate, base):
    return dict(
        model=name,
        control=control,
        group=group,
        decision=paired(pooled(candidate["_ics"]), pooled(base["_ics"])),
        validation=paired(candidate["_ics"]["validation"], base["_ics"]["validation"]),
        evaluation=paired(candidate["_ics"]["evaluation"], base["_ics"]["evaluation"]),
    )


def origin_report(state, year, seed, controls, groups, references, ensemble_limit, offset=False):
    records = completed(state, year, seed)
    reports = {}
    for name, record in records.items():
        report = model_report(state, record, year, seed)
        if report is not None:
            reports[name] = report
    for reference in references:
        for name, record in completed(reference, year, seed).items():
            if name not in reports:
                report = model_report(reference, record, year, seed)
                if report is not None:
                    reports[name] = report
    contrasts = [
        contrast(name, control, groups.get(name), reports[name], reports[control])
        for name, control in sorted(controls.items())
        if name in reports and control in reports
    ]
    ensembles = forward_selection(reports, ensemble_limit) if len(reports) > 1 else []
    known = offset_report(reports, KnownOffset(state, year), year, seed) if offset else None
    for report in reports.values():
        # Los contrastes entre orígenes solo necesitan el IC por sesión: se liberan las filas.
        report["_splits"], report["_ranks"] = None, None
    return dict(
        year=year,
        completed=sorted(records),
        reports=reports,
        contrasts=contrasts,
        ensembles=ensembles,
        offset=known,
    )


def aggregate(origins, controls, groups):
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
            decision = (pooled(reports[name]["_ics"]), pooled(reports[control]["_ics"]))
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


def comparison(origins, other, seed):
    """Cada modelo de este estado frente al mismo modelo de otro, en sus sesiones comunes."""
    rows = []
    for origin in origins:
        records = completed(other, origin["year"], seed)
        for name in sorted(set(origin["reports"]) & set(records)):
            mine = origin["reports"][name]
            # Un modelo tomado del otro estado como referencia no se compara consigo mismo.
            if mine["state"] == str(other):
                continue
            report = model_report(other, records[name], origin["year"], seed)
            if report is None:
                continue
            rows.append(
                dict(
                    model=name,
                    year=origin["year"],
                    decision=paired(pooled(mine["_ics"]), pooled(report["_ics"])),
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
):
    origins = [
        origin_report(state, year, seed, controls, groups, references, ensemble_limit, offset)
        for year in years
    ]
    sessions = session_table(origins, inference) if inference else None
    document = dict(
        schema_version=1,
        kind="dgx_screening_report",
        state=str(state),
        references=[str(path) for path in references],
        seed=seed,
        min_assets=MIN_ASSETS,
        decision_splits=list(DECISION),
        origins=[
            dict(
                year=origin["year"],
                completed=origin["completed"],
                models={
                    name: {key: value for key, value in report.items() if not key.startswith("_")}
                    for name, report in origin["reports"].items()
                },
                contrasts=origin["contrasts"],
                ensembles=origin["ensembles"],
                offset=origin["offset"],
            )
            for origin in origins
        ],
        aggregate=aggregate(origins, controls, groups) if len(origins) > 1 else [],
        comparison=dict(state=str(compare), rows=comparison(origins, compare, seed))
        if compare
        else None,
        inference=None,
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
            base = known["baseline"]
            lines += [
                f"### Desplazamiento conocido de la etiqueta en el origen {origin['year']}",
                "",
                "La etiqueta resta el alfa de las 252 sesiones anteriores, que se conoce al "
                "predecir. `-alfa` por sí solo tiene un IC de evaluación de "
                f"{_f(base['label']['evaluation']['mean'])} "
                f"({_f(base['liquid_label']['mean'])} en el tercio más líquido). La etiqueta "
                "sin alfa es el rendimiento que se puede operar, y la predicción de un modelo "
                "para ella es la suya más alfa: `+alfa` por sí solo tiene "
                f"{_f(base['free']['evaluation']['mean'])} "
                f"({_f(base['liquid_free']['mean'])} en el tercio más líquido).",
                "",
                "| Modelo | IC de evaluación | t NW | IC sin alfa | t NW | Líquido | "
                "Líquido sin alfa | Peso | Con -alfa |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            ordered_known = sorted(
                known["models"].items(), key=lambda item: -(item[1]["combined"]["mean"] or -9)
            )
            for name, item in ordered_known:
                free = item["free"]["evaluation"]
                lines.append(
                    f"| `{name}` | {_f(item['label']['mean'])} | {_f(item['label']['t_nw'], 1)} | "
                    f"{_f(free['mean'])} | {_f(free['t_nw'], 1)} | "
                    f"{_f(item['liquid_label']['mean'])} | {_f(item['liquid_free']['mean'])} | "
                    f"{_f(item['weight'], 1)} | {_f(item['combined']['mean'])} |"
                )
            lines.append("")
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
    )
    args.output.mkdir(parents=True, exist_ok=True)
    sessions = report.pop("_sessions", None)
    if sessions is not None:
        write_sessions(args.output / "sessions.csv", sessions)
        report["inference"] = _inference_module().analizar_tabla_ic(
            args.output / "sessions.csv", args.inference, bootstrap=args.bootstrap, seed=0
        )
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    )
    (args.output / "report.md").write_text(markdown(report))
    count = sum(len(origin["models"]) for origin in report["origins"])
    print(f"{count} modelos en {len(report['origins'])} orígenes: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
