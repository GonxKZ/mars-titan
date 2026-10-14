"""Recalcular una comparación walk-forward publicada y cotejarla con su informe.

La comparación escribe `comparison.json` y `sessions.parquet`. Esta comprobación vuelve a
leer las predicciones crudas de cada modelo, semilla y ventana desde el manifiesto de
fuentes, con sus huellas, y recalcula las métricas por sesión con
`integrity.independent_scores`, después de quitar las filas que el diseño conjunto excluye
de la ventana con la misma regla declarada que `integrity.row_identity`. Después exige:

- las mismas sesiones (mercado e instante) en la tabla publicada y en el recálculo;
- recuentos y estados idénticos;
- valores reales dentro de la tolerancia declarada, con los indefinidos en las mismas
  sesiones;
- el MAE, el MSE y la dirección agregados de cada vista con la ponderación declarada.

Solo se cotejan los cuantiles en bruto. Los calibrados dependen del calibrador común,
que no tiene todavía una segunda implementación, y el informe lo dice. El control cero no
tiene archivo de predicciones: se recalcula con predicción nula sobre los objetivos de la
ventana, que `row_identity` exige comunes a todos los modelos.

Un archivo compactado por la retención v2 se lee con los mismos bits. Uno liberado detiene
el recálculo hasta regenerarlo (`run_masked_campaign.py regenerate`).
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data import prediction_files
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.integrity import row_identity
from mars_titan.integrity.independent_scores import (
    COUNT_COLUMNS,
    SIGN_BINS,
    session_mean,
    session_scores,
)
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS, QUANTILE_HEAD

ZERO_CONTROL = comparison.ZERO_CONTROL

KIND = "walk_forward_score_recheck"
RTOL = 1e-10
ATOL = 1e-13
KEYS = ("market", "prediction_at")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _microseconds(moments):
    """Instantes UTC con zona horaria a microsegundos enteros desde la época."""
    naive = moments.dt.tz_convert("UTC").dt.tz_localize(None)
    return naive.astype("datetime64[us]").astype(np.int64)


def _close(left, right, rtol, atol):
    """Igualdad con tolerancia en la que NaN solo coincide con NaN."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    both_nan = np.isnan(left) & np.isnan(right)
    close = np.abs(left - right) <= atol + rtol * np.abs(right)
    return both_nan | (close & ~np.isnan(left) & ~np.isnan(right))


def compare_sessions(independent, published, *, rtol=RTOL, atol=ATOL):
    """Cotejar dos tablas por sesión y devolver un resumen con cada discrepancia contada."""
    published = published.copy()
    if published["prediction_at"].dtype.kind == "M":
        published["prediction_at"] = _microseconds(published["prediction_at"])
    left = independent.set_index(list(KEYS)).sort_index()
    right = published.set_index(list(KEYS)).sort_index()
    only_left = left.index.difference(right.index)
    only_right = right.index.difference(left.index)
    findings = dict(
        sessions=len(left),
        sessions_only_recomputed=len(only_left),
        sessions_only_published=len(only_right),
        columns={},
    )
    if len(only_left) or len(only_right):
        findings["passed"] = False
        return findings
    right = right.loc[left.index]
    for name in left.columns:
        if name not in right.columns:
            findings["columns"][name] = dict(missing_in_published=True)
            continue
        if name in COUNT_COLUMNS:
            mismatched = int(np.count_nonzero(left[name].to_numpy() != right[name].to_numpy()))
        elif name == "rank_ic_status":
            mismatched = int(np.count_nonzero(left[name].to_numpy() != right[name].to_numpy()))
        elif name.startswith("sign_bin_"):
            a = np.vstack(left[name].to_numpy()).astype(np.float64)
            b = np.vstack([np.asarray(item, dtype=np.float64) for item in right[name]])
            _require(a.shape[1] == b.shape[1] == SIGN_BINS, "Intervalos de signo distintos")
            mismatched = int(np.count_nonzero(~_close(a, b, rtol, atol).all(axis=1)))
        else:
            a = left[name].to_numpy(dtype=np.float64, na_value=np.nan)
            b = right[name].to_numpy(dtype=np.float64, na_value=np.nan)
            mismatched = int(np.count_nonzero(~_close(a, b, rtol, atol)))
            if mismatched:
                gap = np.abs(a - b)
                findings["columns"][name] = dict(
                    mismatched_sessions=mismatched,
                    max_abs_difference=float(np.nanmax(gap)) if np.isfinite(gap).any() else None,
                )
                continue
        findings["columns"][name] = dict(mismatched_sessions=mismatched)
    findings["passed"] = all(
        entry.get("mismatched_sessions", 1) == 0 for entry in findings["columns"].values()
    )
    return findings


def _aggregates(scores, weighting):
    """MAE, MSE y dirección agregados con la ponderación declarada."""
    markets = scores["market"].to_numpy()
    everywhere = np.ones(len(scores), dtype=bool)
    eligible = scores["direction_eligible"].to_numpy()
    accuracy = scores["direction_hits"].to_numpy() / np.maximum(eligible, 1)
    return dict(
        mae=session_mean(scores["mae"], everywhere, markets, weighting),
        mse=session_mean(scores["mse"], everywhere, markets, weighting),
        direction_accuracy=session_mean(accuracy, eligible > 0, markets, weighting),
    )


def _check_point(recomputed, point, rtol, atol):
    result = {}
    for name, value in recomputed.items():
        reported = point.get(name)
        if value is None or reported is None:
            result[name] = dict(passed=value is None and reported is None)
        else:
            result[name] = dict(
                passed=bool(_close(value, reported, rtol, atol)),
                recomputed=value,
                published=reported,
            )
    return result


def independent_window(config, sources, arm, seed, window_id):
    """Sesiones de la evaluación de un modelo y semilla recalculadas desde sus filas.

    Aplica antes la exclusión declarada del diseño conjunto (`row_identity.kept_rows`) y
    devuelve también el recuento por mercado de las filas quitadas.
    """
    quantile = sources["arms"][arm]["output"] == QUANTILE_HEAD
    record = sources["files"][arm, seed, window_id]["evaluation"]
    columns = ["market", "prediction_at", "target", "prediction"]
    # Comprueba la huella y lee igual un archivo presente o compactado por la retención v2.
    table = prediction_files.read(
        record["path"], record["sha256"], columns + (list(QUANTILE_COLUMNS) if quantile else [])
    )
    table, excluded = row_identity.kept_rows(
        table, row_identity.dropped_markets(sources, window_id, arm)
    )
    frame = table.to_pandas()
    frame["prediction_at"] = _microseconds(frame["prediction_at"])
    scores = session_scores(
        frame,
        levels=LEVELS if quantile else None,
        quantile_columns=QUANTILE_COLUMNS if quantile else None,
        rank_ic_min_assets=config["metrics"].get("rank_ic_min_assets", 3),
    )
    return scores, excluded


def zero_arms(sources):
    """Controles cero del ámbito, que no tienen archivo de predicciones."""
    return [name for name, arm in sources["arms"].items() if arm["output"] == ZERO_CONTROL]


def independent_zero(config, sources, window_id):
    """Sesiones del control cero de una ventana: predicción nula sobre los objetivos comunes.

    Los objetivos salen del primer archivo de evaluación de la ventana, tras la exclusión
    declarada. `row_identity` exige que todos los modelos compartan esas filas y objetivos.
    """
    arm, seed, _ = min(key for key in sources["files"] if key[2] == window_id)
    record = sources["files"][arm, seed, window_id]["evaluation"]
    columns = ["market", "prediction_at", "target"]
    table = prediction_files.read(record["path"], record["sha256"], columns)
    table, _ = row_identity.kept_rows(table, row_identity.dropped_markets(sources, window_id, arm))
    frame = table.to_pandas()
    frame["prediction_at"] = _microseconds(frame["prediction_at"])
    frame["prediction"] = 0.0
    return session_scores(frame, rank_ic_min_assets=config["metrics"].get("rank_ic_min_assets", 3))


def _compare(independent, raw, mask, rtol, atol):
    part = raw.loc[mask, [name for name in independent.columns if name in raw.columns]]
    return compare_sessions(independent, part, rtol=rtol, atol=atol)


def check_files(config, sources, raw, *, rtol=RTOL, atol=ATOL):
    """Cotejar las sesiones en bruto publicadas (`raw`) con el recálculo de cada archivo.

    `raw` tiene las columnas de la tabla de sesiones y `arm`, `seed` y `window`. El control
    cero se recalcula con predicción nula sobre los objetivos de la ventana. Devuelve el
    resultado de cada archivo y de cada control cero y el número de discrepancias.
    """
    checks, failures = [], 0
    for arm, seed, window_id in sorted(sources["files"]):
        independent, excluded = independent_window(config, sources, arm, seed, window_id)
        mask = (raw["arm"] == arm) & (raw["seed"] == seed) & (raw["window"] == window_id)
        outcome = _compare(independent, raw, mask, rtol, atol)
        outcome.update(arm=arm, seed=seed, window=window_id, excluded_rows=excluded)
        failures += not outcome["passed"]
        checks.append(outcome)
    zeros = zero_arms(sources)
    for window_id in sorted({window for _, _, window in sources["files"]}):
        independent = None
        for name in zeros:
            if independent is None:
                independent = independent_zero(config, sources, window_id)
            mask = (raw["arm"] == name) & (raw["window"] == window_id)
            outcome = _compare(independent, raw, mask, rtol, atol)
            outcome.update(arm=name, seed=None, window=window_id)
            failures += not outcome["passed"]
            checks.append(outcome)
    return checks, failures


def check_summaries(config, sources, raw, report, *, rtol=RTOL, atol=ATOL):
    """Volver a agregar MAE, MSE y dirección de cada vista y cotejarlos con el informe."""
    weighting = config["metrics"]["market_weighting"]
    aggregates, failures, everything = [], 0, "+".join(sources["markets"])
    pairs = sorted({(arm, seed) for arm, seed, _ in sources["files"]})
    pairs += [(name, None) for name in zero_arms(sources)]
    for arm, seed in pairs:
        if seed is None:
            pooled = raw[raw["arm"] == arm]
            overall = report["arms"][arm]["seeds"]["deterministic"]["overall"]
        else:
            pooled = raw[(raw["arm"] == arm) & (raw["seed"] == seed)]
            overall = report["arms"][arm]["seeds"][str(seed)]["overall"]
        for view, entry in overall.items():
            selected = pooled if view == everything else pooled[pooled["market"] == view]
            point = entry["summary"]["point"]
            result = _check_point(_aggregates(selected, weighting), point, rtol, atol)
            failures += sum(not item["passed"] for item in result.values())
            aggregates.append(dict(arm=arm, seed=seed, view=view, metrics=result))
    return aggregates, failures


def published_raw(folder, report):
    """Sesiones en bruto de la tabla publicada, tras comprobar su huella en el informe."""
    sessions_path = Path(folder) / "sessions.parquet"
    _require(
        report["artifacts"]["sessions.parquet"] == _sha256(sessions_path),
        "La tabla de sesiones no coincide con la huella del informe",
    )
    published = pq.read_table(sessions_path).to_pandas()
    return published[published["quantiles"] == "raw"]


def recheck(config_path, sources_path, scope, comparison_dir, *, rtol=RTOL, atol=ATOL):
    """Recalcular todas las sesiones en bruto de una comparación publicada."""
    folder = Path(comparison_dir)
    report = json.loads((folder / "comparison.json").read_text())
    config = comparison.resolve_config(config_path)
    _require(report["configuration"]["sha256"] == config["sha256"], "Otra configuración")
    sources = comparison.load_sources(sources_path, config, scope)
    _require(report["sources_sha256"] == sources["sha256"], "Otro manifiesto de fuentes")
    raw = published_raw(folder, report)
    checks, failures = check_files(config, sources, raw, rtol=rtol, atol=atol)
    aggregates, wrong = check_summaries(config, sources, raw, report, rtol=rtol, atol=atol)
    failures += wrong
    return dict(
        schema_version=1,
        kind=KIND,
        created_at_utc=datetime.now(UTC).isoformat(),
        comparison_sha256=_sha256(folder / "comparison.json"),
        tolerance=dict(rtol=rtol, atol=atol),
        scope=scope,
        coverage=dict(
            raw_sessions=True,
            calibrated_quantiles=False,
            calibrated_reason="El calibrador común aún no tiene segunda implementación",
            zero_control=True,
        ),
        windows=checks,
        aggregates=aggregates,
        failures=failures,
        passed=failures == 0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("sources")
    parser.add_argument("scope")
    parser.add_argument("comparison")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = recheck(args.config, args.sources, args.scope, args.comparison)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
