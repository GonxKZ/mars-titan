"""Escáner de fugas de información futura en las columnas de entrada, sin modelo.

Una entrada que contiene el objetivo futuro, aunque sea con ruido o transformada de forma
monótona, ordena los activos de una sesión casi igual que el objetivo. Las señales reales
sobre rendimientos residuales diarios tienen correlaciones de rangos por sesión del orden de
centésimas. Este escáner calcula, para cada columna y cada sesión (mercado, instante):
- la correlación de Spearman con el objetivo de esa fila (`future`);
- la misma correlación con un placebo pasado: el objetivo del mismo activo `lag` decisiones
  antes, que ya ha madurado si `lag` cubre el horizonte de la etiqueta (`past`).

Una columna se marca si la media de |IC| con el objetivo supera `mean_threshold` o si alguna
sesión con al menos `min_assets` activos supera `session_threshold`. El placebo pasado no
decide nada. Sirve para distinguir una entrada que copia rendimientos pasados (legítima,
alta con `past`) de una que copia el futuro (alta con `future`).

Los valores ausentes de una columna se excluyen de esa columna en cada sesión, y el objetivo
se ordena solo entre las filas presentes. Los umbrales son declaraciones previas, no
estimaciones: un escáner que no marca nada no demuestra la ausencia de fugas sutiles.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KIND = "input_leakage_scan"
KEYS = ("market", "prediction_at")
DEFAULTS = dict(mean_threshold=0.2, session_threshold=0.95, min_assets=20, chunk=32)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def placebo_past_target(frame, lag):
    """Objetivo del mismo activo `lag` decisiones antes, ordenando por instante."""
    _require(type(lag) is int and lag >= 1, "El retardo del placebo debe ser un entero positivo")
    ordered = frame.sort_values(["asset_id", "prediction_at"], kind="mergesort")
    shifted = ordered.groupby("asset_id", sort=False)["target"].shift(lag)
    return shifted.reindex(frame.index)


def _session_ic(sessions, values, target, min_assets):
    """Spearman por sesión entre cada columna de `values` y `target`, con ausencias por columna.

    Devuelve dos matrices (sesiones × columnas): la correlación y el número de filas presentes.
    """
    mask = values.notna().to_numpy() & target.notna().to_numpy()[:, None]
    present = pd.DataFrame(mask, index=values.index, columns=values.columns)
    masked_target = pd.DataFrame(
        np.where(mask, target.to_numpy()[:, None], np.nan),
        index=values.index,
        columns=values.columns,
    )
    masked_values = values.where(present)
    ranks_x = masked_values.groupby(sessions, sort=True).rank(method="average")
    ranks_y = masked_target.groupby(sessions, sort=True).rank(method="average")
    count = present.groupby(sessions, sort=True).sum()
    # Sumas centradas por sesión: una columna constante da varianza exactamente cero, sin el
    # residuo de redondeo de la fórmula de suma de cuadrados.
    centered_x = ranks_x - ranks_x.groupby(sessions, sort=True).transform("mean")
    centered_y = ranks_y - ranks_y.groupby(sessions, sort=True).transform("mean")
    covariance = (centered_x * centered_y).groupby(sessions, sort=True).sum().to_numpy()
    variance_x = (centered_x * centered_x).groupby(sessions, sort=True).sum().to_numpy()
    variance_y = (centered_y * centered_y).groupby(sessions, sort=True).sum().to_numpy()
    n = count.to_numpy(dtype=np.float64)
    defined = (n >= min_assets) & (variance_x > 0) & (variance_y > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = covariance / np.sqrt(variance_x * variance_y)
    return np.where(defined, ic, np.nan), n


def scan(frame, feature_columns, *, lag, **options):
    """Resumen por columna de la correlación de rangos con el objetivo y con el placebo."""
    settings = {**DEFAULTS, **options}
    _require(
        set(settings) == set(DEFAULTS), f"Opciones desconocidas: {set(options) - set(DEFAULTS)}"
    )
    required = {*KEYS, "asset_id", "target"}
    _require(required <= set(frame.columns), f"Faltan columnas: {sorted(required - set(frame))}")
    columns = list(feature_columns)
    _require(columns and len(set(columns)) == len(columns), "Columnas de entrada repetidas")
    _require(not set(columns) & required, "Una columna de entrada coincide con una clave")
    frame = frame.reset_index(drop=True)
    _require(np.isfinite(frame["target"]).all(), "El objetivo debe ser finito en todas las filas")
    sessions = frame.groupby(list(KEYS), sort=True).ngroup()
    past = placebo_past_target(frame, lag)
    findings = []
    for start in range(0, len(columns), settings["chunk"]):
        names = columns[start : start + settings["chunk"]]
        values = frame[names].astype(np.float64)
        values = values.where(np.isfinite(values))
        future_ic, future_n = _session_ic(sessions, values, frame["target"], settings["min_assets"])
        past_ic, _ = _session_ic(sessions, values, past, settings["min_assets"])
        for j, name in enumerate(names):
            ic, placebo = future_ic[:, j], past_ic[:, j]
            defined = np.isfinite(ic)
            mean_abs = float(np.mean(np.abs(ic[defined]))) if defined.any() else None
            worst = float(np.max(np.abs(ic[defined]))) if defined.any() else None
            flagged = mean_abs is not None and (
                mean_abs >= settings["mean_threshold"] or worst >= settings["session_threshold"]
            )
            placebo_defined = np.isfinite(placebo)
            findings.append(
                dict(
                    column=name,
                    sessions=int(defined.sum()),
                    rows=int(future_n[:, j].sum()),
                    mean_ic=float(np.mean(ic[defined])) if defined.any() else None,
                    mean_abs_ic=mean_abs,
                    max_abs_ic=worst,
                    past_mean_abs_ic=float(np.mean(np.abs(placebo[placebo_defined])))
                    if placebo_defined.any()
                    else None,
                    flagged=bool(flagged),
                )
            )
    flagged = [entry["column"] for entry in findings if entry["flagged"]]
    return dict(
        schema_version=1,
        kind=KIND,
        created_at_utc=datetime.now(UTC).isoformat(),
        rows=len(frame),
        sessions=int(sessions.nunique()),
        placebo_lag=lag,
        thresholds={k: v for k, v in settings.items() if k != "chunk"},
        columns=findings,
        flagged=flagged,
        passed=not flagged,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "table", help="Parquet con market, asset_id, prediction_at, target y entradas"
    )
    parser.add_argument("--lag", type=int, required=True)
    parser.add_argument("--columns", nargs="*", help="Por defecto, todas las que no son claves")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    frame = pq.read_table(args.table).to_pandas()
    keys = {*KEYS, "asset_id", "target"}
    columns = args.columns or [name for name in frame.columns if name not in keys]
    result = scan(frame, columns, lag=args.lag)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
