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


class _Sessions:
    """Filas ordenadas por sesión, para ordenar y sumar cada sesión con numpy.

    Las sesiones son los grupos ordenados de (mercado, instante). Todo el cálculo por columna
    trabaja en este orden, así que cada sesión es un tramo contiguo de filas.
    """

    def __init__(self, codes):
        self.order = np.argsort(codes, kind="stable")
        codes = np.asarray(codes)
        small = np.uint16 if codes.max(initial=0) < 2**16 else np.int64
        self.codes = codes.astype(small)[self.order]
        self.starts = np.flatnonzero(np.r_[True, self.codes[1:] != self.codes[:-1]])
        self.lengths = np.diff(np.r_[self.starts, len(self.codes)])
        self.first = np.repeat(self.starts, self.lengths)

    def ranks(self, values, mask):
        """Rangos medios dentro de cada sesión de las filas de `mask`, como `rank("average")`.

        Las filas fuera de `mask` se ordenan al final de su sesión y quedan sin rango, así
        que cada fila presente se ordena solo entre las presentes de su sesión.
        """
        key = np.where(mask, values, np.inf)
        # Igual que lexsort por (sesión, valor): primero por valor y después por sesión con un
        # orden estable, que para enteros pequeños es un radix y cuesta mucho menos.
        by_value = np.argsort(key)
        order = by_value[np.argsort(self.codes[by_value], kind="stable")]
        ordered, codes = key[order], self.codes[order]
        rows = len(key)
        new = np.ones(rows, dtype=bool)
        new[1:] = (ordered[1:] != ordered[:-1]) | (codes[1:] != codes[:-1])
        index = np.arange(rows)
        start = np.maximum.accumulate(np.where(new, index, 0))
        closes = np.ones(rows, dtype=bool)
        closes[:-1] = new[1:]
        end = np.minimum.accumulate(np.where(closes, index, rows)[::-1])[::-1]
        ranks = np.empty(rows)
        ranks[order] = (start + end) / 2 - self.first + 1
        return np.where(mask, ranks, np.nan)

    def sums(self, values):
        return np.add.reduceat(values, self.starts)

    def centered(self, ranks, mask, count):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.sums(np.where(mask, ranks, 0.0)) / count
        return np.where(mask, ranks - np.repeat(mean, self.lengths), 0.0)


def _session_ic(sessions, values, target, min_assets, cache):
    """Spearman por sesión entre cada columna de `values` y `target`, con ausencias por columna.

    `values` y `target` siguen el orden de `sessions`. Devuelve dos matrices (sesiones ×
    columnas): la correlación y el número de filas presentes. Los rangos del objetivo se
    reutilizan entre columnas con las mismas filas presentes (`cache`), que en una modalidad
    son casi siempre las mismas.
    """
    observed = np.isfinite(target)
    correlations, counts = [], []
    for column in values.T:
        mask = np.isfinite(column) & observed
        count = sessions.sums(mask.astype(np.float64))
        key = np.packbits(mask).tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = sessions.centered(sessions.ranks(target, mask), mask, count)
        centered_y = cache[key]
        centered_x = sessions.centered(sessions.ranks(column, mask), mask, count)
        # Sumas centradas por sesión: una columna constante da varianza exactamente cero, sin
        # el residuo de redondeo de la fórmula de suma de cuadrados.
        covariance = sessions.sums(centered_x * centered_y)
        variance_x = sessions.sums(centered_x * centered_x)
        variance_y = sessions.sums(centered_y * centered_y)
        defined = (count >= min_assets) & (variance_x > 0) & (variance_y > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = covariance / np.sqrt(variance_x * variance_y)
        correlations.append(np.where(defined, ic, np.nan))
        counts.append(count)
    return np.column_stack(correlations), np.column_stack(counts)


def scan(frame, feature_columns, *, lag, **options):
    """Resumen por columna de la correlación de rangos con el objetivo y con el placebo."""
    settings = _settings(options)
    columns = list(feature_columns)
    _require(set(columns) <= set(frame.columns), "Faltan columnas de entrada en la tabla")
    frame = frame.reset_index(drop=True)
    chunks = [
        columns[start : start + settings["chunk"]]
        for start in range(0, len(columns), settings["chunk"])
    ]
    return scan_blocks(frame, ((names, frame[names]) for names in chunks), lag=lag, **options)


def _settings(options):
    settings = {**DEFAULTS, **options}
    _require(
        set(settings) == set(DEFAULTS), f"Opciones desconocidas: {set(options) - set(DEFAULTS)}"
    )
    return settings


def scan_blocks(keys, blocks, *, lag, **options):
    """Escanear columnas que llegan por bloques, alineadas con las filas de `keys`.

    `keys` lleva las claves, `asset_id` y `target`. Cada bloque es un par (nombres, valores)
    con una columna de valores por nombre. Así una matriz mayor que la memoria se recorre por
    columnas desde disco con el mismo cálculo que una tabla en memoria.
    """
    settings = _settings(options)
    required = {*KEYS, "asset_id", "target"}
    _require(required <= set(keys.columns), f"Faltan columnas: {sorted(required - set(keys))}")
    keys = keys.reset_index(drop=True)
    _require(np.isfinite(keys["target"]).all(), "El objetivo debe ser finito en todas las filas")
    sessions = _Sessions(keys.groupby(list(KEYS), sort=True).ngroup().to_numpy())
    target = keys["target"].to_numpy(dtype=np.float64)[sessions.order]
    past = placebo_past_target(keys, lag).to_numpy(dtype=np.float64)[sessions.order]
    findings, seen, caches = [], set(), ({}, {})
    for names, block in blocks:
        names = list(names)
        _require(
            names and len(set(names)) == len(names) and not set(names) & seen,
            "Columnas de entrada repetidas",
        )
        _require(not set(names) & required, "Una columna de entrada coincide con una clave")
        seen.update(names)
        values = np.asarray(block, dtype=np.float64)
        _require(
            values.shape == (len(keys), len(names)), "Un bloque no tiene las filas de las claves"
        )
        values = values[sessions.order]
        future_ic, future_n = _session_ic(
            sessions, values, target, settings["min_assets"], caches[0]
        )
        past_ic, _ = _session_ic(sessions, values, past, settings["min_assets"], caches[1])
        for j, name in enumerate(names):
            findings.append(
                _finding(name, future_ic[:, j], future_n[:, j], past_ic[:, j], settings)
            )
    _require(findings, "No hay columnas de entrada")
    flagged = [entry["column"] for entry in findings if entry["flagged"]]
    return dict(
        schema_version=1,
        kind=KIND,
        created_at_utc=datetime.now(UTC).isoformat(),
        rows=len(keys),
        sessions=len(sessions.starts),
        placebo_lag=lag,
        thresholds={k: v for k, v in settings.items() if k != "chunk"},
        columns=findings,
        flagged=flagged,
        passed=not flagged,
    )


def _finding(name, ic, present, placebo, settings):
    defined = np.isfinite(ic)
    mean_abs = float(np.mean(np.abs(ic[defined]))) if defined.any() else None
    worst = float(np.max(np.abs(ic[defined]))) if defined.any() else None
    flagged = mean_abs is not None and (
        mean_abs >= settings["mean_threshold"] or worst >= settings["session_threshold"]
    )
    placebo_defined = np.isfinite(placebo)
    return dict(
        column=name,
        sessions=int(defined.sum()),
        rows=int(present.sum()),
        mean_ic=float(np.mean(ic[defined])) if defined.any() else None,
        mean_abs_ic=mean_abs,
        max_abs_ic=worst,
        past_mean_abs_ic=float(np.mean(np.abs(placebo[placebo_defined])))
        if placebo_defined.any()
        else None,
        flagged=bool(flagged),
    )


def _within_sessions(sessions, rng):
    """Orden que permuta las filas dentro de cada sesión, en el orden de `sessions`."""
    return np.lexsort((rng.random(len(sessions.codes)), sessions.codes))


def permutation_null(keys, values, names, *, permutations, seed, **options):
    """Lo que marcaría la regla declarada si el objetivo no tuviera relación con las columnas.

    Repite el cálculo por sesión del escáner con el objetivo real y después con el objetivo
    permutado dentro de cada sesión. La permutación conserva la composición de cada sesión y
    los empates de cada columna, que son los que hacen grande un |IC| por azar cuando hay pocos
    activos o pocos valores distintos. Por permutación guarda cuántas columnas se marcan y la
    mayor media de |IC|. El placebo no decide la marca, así que aquí no se calcula. Una
    asociación real con el objetivo, sea una fuga o una exposición legítima a un factor, queda
    fuera del nulo: este contraste solo dice si el azar basta para explicar la marca.
    """
    settings = _settings(options)
    _require(type(permutations) is int and permutations >= 1, "Hace falta al menos una permutación")
    names = list(names)
    keys = keys.reset_index(drop=True)
    values = np.asarray(values, dtype=np.float64)
    _require(
        names and values.shape == (len(keys), len(names)),
        "Las columnas no tienen las filas de las claves",
    )
    _require(np.isfinite(keys["target"]).all(), "El objetivo debe ser finito en todas las filas")
    sessions = _Sessions(keys.groupby(list(KEYS), sort=True).ngroup().to_numpy())
    target = keys["target"].to_numpy(dtype=np.float64)[sessions.order]
    values = values[sessions.order]
    unused = np.full(len(sessions.starts), np.nan)

    def score(order):
        ic, present = _session_ic(sessions, values, target[order], settings["min_assets"], {})
        found = [
            _finding(name, ic[:, j], present[:, j], unused, settings)
            for j, name in enumerate(names)
        ]
        means = [entry["mean_abs_ic"] for entry in found if entry["mean_abs_ic"] is not None]
        return ic, found, max(means, default=None)

    ic, found, strongest = score(np.arange(len(target)))
    _require(strongest is not None, "Ninguna sesión tiene una correlación definida")
    flagged = [entry["column"] for entry in found if entry["flagged"]]
    rng = np.random.default_rng(seed)
    counts, maxima = [], []
    for _ in range(permutations):
        _, null, top = score(_within_sessions(sessions, rng))
        counts.append(sum(entry["flagged"] for entry in null))
        maxima.append(top)
    counts, maxima = np.array(counts), np.array(maxima, dtype=np.float64)
    quantiles = (0.05, 0.5, 0.95)
    defined = np.isfinite(ic)
    positive = ((ic > 0) & defined).sum(axis=0) / np.maximum(defined.sum(axis=0), 1)
    columns = [
        dict(
            {key: entry[key] for key in ("column", "sessions", "mean_ic", "mean_abs_ic")},
            positive_sessions=None if entry["sessions"] == 0 else float(positive[j]),
            flagged=entry["flagged"],
        )
        for j, entry in enumerate(found)
    ]
    return dict(
        rows=len(keys),
        sessions=len(sessions.starts),
        thresholds={k: v for k, v in settings.items() if k != "chunk"},
        observed=dict(flagged=flagged, max_mean_abs_ic=strongest, columns=columns),
        null=dict(
            permutations=permutations,
            seed=seed,
            flagged_counts=counts.tolist(),
            flagged_quantiles={str(q): float(np.quantile(counts, q)) for q in quantiles},
            share_flagging_at_least_observed=float((counts >= len(flagged)).mean()),
            max_mean_abs_ic_quantiles={str(q): float(np.nanquantile(maxima, q)) for q in quantiles},
            share_max_at_least_observed=float((maxima >= strongest).mean()),
        ),
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
