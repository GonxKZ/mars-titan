"""Escala del pico de la matriz de ajuste de XGBoost y estimación para la ventana completa.

Ajusta por mínimos cuadrados el pico de RSS de la construcción (fase `matrix_build`) frente
a las filas de ajuste, después de restar la caché del lector y las claves de cada fila, que
se miden aparte. Extrapola a las filas de la ventana completa con la caché del lector en su
límite y suma la validación residente de `memory-runs.json`, en el orden de
`external_corpus._execute`: la matriz sigue viva mientras se lee la validación.

El resultado es una estimación con un intervalo para la pendiente, no una medida de la
ventana completa. Solo usa ejecuciones de 64 bins sin memray, que son las que se repitieron
en los tres tamaños.

Uso: matrix_scaling.py EJECUCIONES VISTAS MEMORIA CONFIGURACIÓN SALIDA

EJECUCIONES son los resultados de `memory_profiles.py matrix`, VISTAS un objeto JSON con el
`summary.json` de cada vista de `view_subset.py`, MEMORIA `evidence/memory-runs.json` y
CONFIGURACIÓN la declaración de recursos de la campaña.
"""

import argparse
import inspect
import json
import statistics
from pathlib import Path

from scipy import stats

GIB = 1024**3
MIB = 1024**2


def least_squares(points):
    """Recta y error típico de la pendiente de una lista de pares (x, y)."""
    xs, ys = zip(*points, strict=True)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    intercept = mean_y - slope * mean_x
    residual = sum((y - intercept - slope * x) ** 2 for x, y in points)
    error = (residual / (len(points) - 2) / sxx) ** 0.5
    return intercept, slope, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("views", type=Path)
    parser.add_argument("memory", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from mars_titan.training.corpus_inputs import CorpusDataset

    # Una lista JSON o las líneas que escribe `memory_profiles.py --output`.
    text = args.runs.read_text(encoding="utf-8")
    runs = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    views = json.loads(args.views.read_text(encoding="utf-8"))
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    declared = json.loads(args.config.read_text(encoding="utf-8"))["models"]["xgboost"]
    cache_limit = inspect.signature(CorpusDataset).parameters["cache_bytes"].default

    used = [
        run
        for run in runs
        if run["result"]["construction"]["max_bin"] == 64
        and run["arrow_memory_pool"] == "mimalloc"
        and "reader_cached_bytes" in run["result"]
    ]
    key_bytes = {run["result"]["row_keys_bytes"] / run["result"]["rows"] for run in used}
    if len(key_bytes) != 1:
        raise SystemExit("Las claves de las filas no ocupan lo mismo por fila en cada ejecución")
    key_bytes = key_bytes.pop()
    points = [
        (
            run["result"]["rows"],
            run["sampled_peak_by_phase"]["matrix_build"]
            - run["result"]["reader_cached_bytes"]
            - run["result"]["row_keys_bytes"],
        )
        for run in used
    ]
    intercept, slope, error = least_squares(points)
    quantile = stats.t.ppf(0.975, len(points) - 2)
    slopes = dict(low=slope - quantile * error, central=slope, high=slope + quantile * error)

    sources = {view["source_counts"]["train"] for view in views.values()}
    validations = {view["source_counts"]["validation"] for view in views.values()}
    if len(sources) != 1 or len(validations) != 1:
        raise SystemExit("Las vistas no proceden de la misma ventana")
    train_rows, validation_rows = sources.pop(), validations.pop()

    # Validación sola sobre la vista del 82,7 %: lo que queda al liberarla es la caché del
    # lector y el resto del proceso, así que el transitorio es lo que no explican ni lo
    # residente ni lo que queda.
    plain = [run for run in memory if run["label"] in {"validation-plain", "validation-plain-r2"}]
    per_row = {run["result"]["resident_bytes"] / run["result"]["rows"] for run in plain}
    if len(per_row) != 1:
        raise SystemExit("La validación residente no ocupa lo mismo por fila")
    validation_row_bytes = per_row.pop()
    transients = [
        run["peak_rss_bytes"]
        - run["sampled_peak_by_phase"]["validation_release"]
        - run["result"]["resident_bytes"]
        for run in plain
    ]
    validation_bytes = validation_rows * validation_row_bytes
    transient = max(0, *transients)
    start = statistics.fmean(run["sampled_peak_by_phase"]["start"] for run in used)

    estimates = {}
    for name, value in slopes.items():
        matrix = intercept + cache_limit + key_bytes * train_rows + value * train_rows
        joint = matrix + validation_bytes + transient
        estimates[name] = dict(
            slope_bytes_per_row=value,
            matrix_peak_gib=matrix / GIB,
            joint_peak_gib=joint / GIB,
            joint_increment_over_start_gib=(joint - start) / GIB,
        )

    by_bins = {}
    for run in runs:
        result = run["result"]
        bins = str(result["construction"]["max_bin"])
        entry = by_bins.setdefault(
            bins, dict(build_device_mib_by_rows={}, disk_bytes_per_row=set(), plan_error_bytes=[])
        )
        # Solo la construcción: después, la reserva de CuPy conserva la validación liberada.
        device = entry["build_device_mib_by_rows"]
        rows = str(result["rows"])
        device[rows] = max(device.get(rows, 0), run["device_peak_mib_by_phase"]["matrix_build"])
        entry["disk_bytes_per_row"].add(round(result["disk_bytes_per_row"], 2))
        entry["plan_error_bytes"].append(
            result["disk_bytes"] - result["plan"]["disk_cache_bytes_estimate"]
        )
    for entry in by_bins.values():
        per_row_disk = max(entry["disk_bytes_per_row"])
        entry.update(
            disk_bytes_per_row=sorted(entry["disk_bytes_per_row"]),
            plan_error_bytes=[min(entry["plan_error_bytes"]), max(entry["plan_error_bytes"])],
            full_disk_gb=per_row_disk * train_rows / 1e9,
        )
    largest = max(run["result"]["rows"] for run in used)
    build_seconds = [
        run["result"]["build_seconds"] for run in used if run["result"]["rows"] == largest
    ]
    payload = dict(
        runs_used=[run["label"] for run in used],
        rows_measured=sorted({run["result"]["rows"] for run in used}),
        fit=dict(
            intercept_gib=intercept / GIB,
            slope_bytes_per_row=slope,
            slope_standard_error=error,
            t_quantile=quantile,
            degrees_of_freedom=len(points) - 2,
        ),
        components=dict(
            reader_cache_limit_bytes=cache_limit,
            row_key_bytes_per_row=key_bytes,
            validation_bytes_per_row=validation_row_bytes,
            validation_transient_bytes=transients,
            process_start_gib=start / GIB,
        ),
        full_window=dict(train_rows=train_rows, validation_rows=validation_rows),
        estimates=estimates,
        declared=dict(host_mib=declared["host_mib"], vram_mib=declared["vram_mib"]),
        device_and_disk_by_bins=by_bins,
        build_seconds_per_million_rows=[
            seconds / largest * 1e6 for seconds in sorted(build_seconds)
        ],
    )
    args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(payload["estimates"], indent=1))


if __name__ == "__main__":
    main()
