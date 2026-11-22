#!/usr/bin/env python3
"""Paridad entre equipos: el mismo modelo, origen y semilla ajustado en dos máquinas.

Dos equipos con otra GPU, otra CPU y otras bibliotecas no dan los mismos bits. Antes de mezclar
sus resultados en un análisis, este guion comprueba que se comportan igual: las filas y las
etiquetas son las mismas y el IC de rangos por sesión no difiere más que un margen declarado.

Por cada tramo (validación, calibración y evaluación) calcula el IC de rangos de cada sesión en
los dos equipos, sobre las mismas filas, y contrasta la diferencia por sesión con error típico
con autocorrelación (Newey-West, sin cruzar tramos) y dos contrastes unilaterales (TOST). Las
semillas comparten sesiones, así que se promedian por sesión antes del contraste en lugar de
contarse como réplicas independientes.

Uso:
    colab_parity.py --a <estado A> --b <estado B> --model titans_mac --origin 2013 \
        --seed 42 [--seed 43 ...] [--margin 0.002] [--output paridad.json]

Sale con 0 si los equipos son equivalentes dentro del margen, con 2 si no lo son o no se puede
concluir, y con 1 si las filas o las etiquetas no coinciden.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPLITS = ("validation", "calibration", "evaluation")
MIN_ASSETS = 30
DEFAULT_MARGIN = 0.002


def average_ranks(values):
    """Rangos medios (los empates comparten el promedio de sus posiciones)."""
    count = len(values)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    first = np.empty(count, dtype=bool)
    first[:1] = True
    first[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(first)
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1:] = count
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = ((starts + 1 + ends) / 2)[np.cumsum(first) - 1]
    return ranks


def rank_correlation(prediction, target):
    """Correlación de rangos. Una predicción o una etiqueta constante no ordena: cero."""
    left, right = average_ranks(prediction), average_ranks(target)
    left, right = left - left.mean(), right - right.mean()
    scale = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right) / scale if scale > 0 else 0.0


def read_predictions(path):
    """Filas de un fichero de predicciones, ordenadas por su identificador.

    Las columnas se quedan en Arrow y NumPy: un origen tiene más de un millón de filas y
    convertir sus identificadores en objetos de Python ocuparía varios gigas.
    """
    columns = ["sample_id", "market", "prediction_at", "target", "prediction"]
    table = pq.read_table(path, columns=columns)
    if pc.count_distinct(table.column("sample_id")).as_py() != table.num_rows:
        raise ValueError(f"Hay filas repetidas en {path}")
    table = table.take(pc.sort_indices(table, sort_keys=[("sample_id", "ascending")]))
    market = pc.dictionary_encode(table.column("market").combine_chunks())
    return dict(
        sample=table.column("sample_id").combine_chunks(),
        market=market.indices.to_numpy(zero_copy_only=False),
        markets=market.dictionary.to_pylist(),
        instant=table.column("prediction_at").cast("int64").to_numpy(zero_copy_only=False),
        target=table.column("target").to_numpy(zero_copy_only=False),
        prediction=table.column("prediction").to_numpy(zero_copy_only=False),
    )


def session_ic(rows):
    """IC de rangos de cada sesión (mercado e instante) con al menos MIN_ASSETS filas.

    La clave de cada sesión lleva el nombre del mercado, no su código, que es propio de cada
    fichero.
    """
    order = np.lexsort((rows["instant"], rows["market"]))
    market, instant = rows["market"][order], rows["instant"][order]
    prediction, target = rows["prediction"][order], rows["target"][order]
    changed = (market[1:] != market[:-1]) | (instant[1:] != instant[:-1])
    bounds = np.flatnonzero(np.r_[True, changed, True])
    result = {}
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        if stop - start >= MIN_ASSETS:
            key = f"{rows['markets'][int(market[start])]}|{int(instant[start])}"
            result[key] = rank_correlation(prediction[start:stop], target[start:stop])
    return result


def newey_west_lags(count):
    return int(math.floor(4 * (count / 100.0) ** (2.0 / 9.0))) if count > 1 else 0


def hac_standard_error(values, blocks):
    """Error típico de la media con núcleo de Bartlett, sin cruzar las fronteras de los bloques."""
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count < 2:
        return float("nan")
    centred = values.copy()
    total = 0.0
    for block in dict.fromkeys(blocks):
        index = np.flatnonzero(np.asarray(blocks, dtype=object) == block)
        # Cada bloque se centra en su media: su nivel propio no es ruido de la diferencia.
        part = centred[index] - centred[index].mean()
        lags = min(newey_west_lags(len(part)), len(part) - 1)
        variance = float(part @ part)
        for lag in range(1, lags + 1):
            variance += 2.0 * (1.0 - lag / (lags + 1.0)) * float(part[lag:] @ part[:-lag])
        total += max(variance, 0.0)
    return math.sqrt(total) / count


def normal_cdf(value):
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def tost(mean, standard_error, margin):
    """Mayor de los dos valores p unilaterales de la equivalencia dentro de ±margin."""
    if not standard_error > 0 or not math.isfinite(standard_error):
        return 0.0 if abs(mean) < margin else 1.0
    lower = 1.0 - normal_cdf((mean + margin) / standard_error)
    upper = normal_cdf((mean - margin) / standard_error)
    return max(lower, upper)


def verdict(mean, standard_error, margin, alpha=0.05):
    """equivalente, distinto o no concluyente, con el intervalo del 90 % de la diferencia."""
    half = 1.6448536269514722 * standard_error if math.isfinite(standard_error) else float("inf")
    low, high = mean - half, mean + half
    if tost(mean, standard_error, margin) <= alpha:
        label = "equivalente"
    elif low > margin or high < -margin:
        label = "distinto"
    else:
        label = "no concluyente"
    return label, [low, high]


def compare_runs(folder_a, folder_b, margin=DEFAULT_MARGIN):
    """Comparar una ejecución (modelo, origen y semilla) de dos equipos, tramo a tramo."""
    report = dict(folder_a=str(folder_a), folder_b=str(folder_b), splits={})
    for split in SPLITS:
        left = read_predictions(Path(folder_a) / f"{split}-predictions.parquet")
        right = read_predictions(Path(folder_b) / f"{split}-predictions.parquet")
        same_rows = left["sample"].equals(right["sample"])
        entry = dict(rows_a=len(left["sample"]), rows_b=len(right["sample"]), same_rows=same_rows)
        if same_rows:
            entry["same_targets"] = bool(np.array_equal(left["target"], right["target"]))
            difference = left["prediction"] - right["prediction"]
            entry["prediction_max_abs_difference"] = float(np.abs(difference).max())
            entry["prediction_correlation"] = float(
                np.corrcoef(left["prediction"], right["prediction"])[0, 1]
            )
            ic_a, ic_b = session_ic(left), session_ic(right)
            keys = sorted(ic_a, key=lambda key: (key.split("|")[0], int(key.split("|")[1])))
            entry.update(
                sessions=len(keys),
                ic_a=float(np.mean([ic_a[key] for key in keys])),
                ic_b=float(np.mean([ic_b[key] for key in keys])),
                keys=keys,
                difference=[ic_a[key] - ic_b[key] for key in keys],
            )
        report["splits"][split] = entry
    return report


def summarize(reports, margin=DEFAULT_MARGIN):
    """Veredicto por tramo y conjunto, promediando las semillas por sesión."""
    summary = dict(margin=margin, seeds=len(reports), splits={})
    identical = all(
        entry.get("same_rows") and entry.get("same_targets")
        for report in reports
        for entry in report["splits"].values()
    )
    summary["same_rows_and_targets"] = identical
    if not identical:
        summary["verdict"] = "filas o etiquetas distintas"
        return summary
    pooled_values, pooled_blocks = [], []
    for split in SPLITS:
        keys = reports[0]["splits"][split]["keys"]
        if any(report["splits"][split]["keys"] != keys for report in reports):
            summary["verdict"] = "las semillas no comparten sesiones"
            return summary
        matrix = np.array([report["splits"][split]["difference"] for report in reports])
        series = matrix.mean(axis=0)
        blocks = [key.split("|")[0] for key in keys]
        mean = float(series.mean())
        error = hac_standard_error(series, blocks)
        label, interval = verdict(mean, error, margin)
        summary["splits"][split] = dict(
            sessions=len(keys),
            mean_difference=mean,
            standard_error=error,
            interval_90=interval,
            tost_p=tost(mean, error, margin),
            verdict=label,
            per_seed_mean_difference=[float(row.mean()) for row in matrix],
        )
        pooled_values.extend(series.tolist())
        pooled_blocks.extend(f"{split}|{block}" for block in blocks)
    mean = float(np.mean(pooled_values))
    error = hac_standard_error(pooled_values, pooled_blocks)
    label, interval = verdict(mean, error, margin)
    summary["pooled"] = dict(
        sessions=len(pooled_values),
        mean_difference=mean,
        standard_error=error,
        interval_90=interval,
        tost_p=tost(mean, error, margin),
        verdict=label,
    )
    summary["verdict"] = label
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--a", type=Path, required=True, help="Estado de campaña del equipo A")
    parser.add_argument("--b", type=Path, required=True, help="Estado de campaña del equipo B")
    parser.add_argument("--model", required=True)
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.margin < 1:
        parser.error("El margen es una diferencia de IC entre 0 y 1")
    reports = []
    for seed in dict.fromkeys(args.seed):
        relative = Path("runs") / args.model / str(args.origin) / f"seed-{seed}"
        reports.append(compare_runs(args.a / relative, args.b / relative, args.margin))
    summary = summarize(reports, args.margin)
    document = dict(
        schema_version=1,
        kind="machine_parity_v1",
        model=args.model,
        origin=args.origin,
        seeds=list(dict.fromkeys(args.seed)),
        summary=summary,
        runs=[
            dict(
                report,
                splits={
                    name: {k: v for k, v in entry.items() if k not in ("keys", "difference")}
                    for name, entry in report["splits"].items()
                },
            )
            for report in reports
        ],
    )
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not summary["same_rows_and_targets"]:
        return 1
    return 0 if summary["verdict"] == "equivalente" else 2


if __name__ == "__main__":
    sys.exit(main())
