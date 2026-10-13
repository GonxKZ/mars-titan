"""Coste de la abstención con su cobertura al puntuar una serie de A v2 (#442).

Construye con `measure_stage_aggregates.table` las predicciones de evaluación de una serie
de US+CN con la forma real de la ventana (filas por mercado y decisiones del calendario).
Son sintéticas: no salen de ningún modelo, no se lee la edición ni ninguna vista real y no
se ajusta nada. Mide después, alternando las variantes en cada repetición:

- `score_sessions` y `to_table`, el camino de los agregados y de la comparación, con los
  aciertos selectivos y sin ellos (`_selective_hits` sustituido por una función nula);
- `independent_scores.session_scores`, el recálculo del informe de integridad, con
  `asset_id` (calcula los aciertos selectivos) y sin esa columna (no los calcula).

Antes de medir, `score_recheck.compare_sessions` coteja las dos implementaciones sesión a
sesión, también los aciertos de cada cobertura, y la medida se detiene si no coinciden.

    PYTHONPATH=src uv run --no-sync python \
        reports/engineering/publication-abstention-20261013/measure_abstention_cost.py \
        --window fold-018 --repetitions 5 --output <medida.json>
"""

import argparse
import importlib.util
import json
import os
import platform
import resource
import statistics
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import forecast_scores
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.integrity import independent_scores, score_recheck
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS
from mars_titan.posttraining import stage_comparison

ROOT = Path(__file__).resolve().parents[3]
SHAPE = ROOT / "reports/engineering/campaign-publication-20261010/measure_stage_aggregates.py"


def _shape_module():
    spec = importlib.util.spec_from_file_location("measure_stage_aggregates", SHAPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def series(window):
    """Tabla de evaluación de una serie con la forma real de la ventana."""
    shape = _shape_module()
    loaded = stage_comparison.load_declaration(shape.DECLARATION)
    counts = shape.market_counts(json.loads(shape.COUNTS.read_text()), window)
    first = next(iter(loaded["configs"].values()))
    scope = walk.restrict_windows(first, shape.SCOPE, [window])["resolved_scopes"][shape.SCOPE]
    rows = {market: value["evaluation"] for market, value in counts.items()}
    seed = zlib.crc32(f"{window}/evaluation".encode())
    data, width = shape.table(rows, scope["windows"][window]["evaluation"], 0, seed)
    return data, rows, width


def _panel(data):
    times = data["prediction_at"].cast(pa.int64()).to_numpy()
    quantiles = np.column_stack([data[name].to_numpy() for name in QUANTILE_COLUMNS])
    return ForecastPanel.from_columns(
        walk._row_id(data),
        data["market"].to_numpy(zero_copy_only=False),
        times,
        data["target"].to_numpy(),
        data["prediction"].to_numpy(),
        quantiles=quantiles,
        levels=LEVELS,
    )


def _frame(data, with_asset):
    frame = data.to_pandas()
    frame["prediction_at"] = data["prediction_at"].cast(pa.int64()).to_numpy()
    return frame if with_asset else frame.drop(columns="asset_id")


def _scores(panel, selective):
    if selective:
        scores = forecast_scores.score_sessions(panel)
    else:
        real = forecast_scores._selective_hits
        forecast_scores._selective_hits = lambda *args: None
        try:
            scores = forecast_scores.score_sessions(panel)
        finally:
            forecast_scores._selective_hits = real
    scores.to_table()
    return scores


def _independent(frame):
    return independent_scores.session_scores(
        frame, levels=LEVELS, quantile_columns=QUANTILE_COLUMNS
    )


def _summary(values):
    return dict(
        seconds=[round(value, 4) for value in values],
        median=round(statistics.median(values), 4),
        minimum=round(min(values), 4),
        maximum=round(max(values), 4),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, rows, width = series(args.window)
    panel = _panel(data)
    frames = {flag: _frame(data, flag) for flag in (True, False)}
    # Las dos variantes dan las mismas puntuaciones salvo los aciertos selectivos.
    with_hits, without = _scores(panel, True), _scores(panel, False)
    assert np.array_equal(with_hits.direction_hits, without.direction_hits)
    assert np.array_equal(with_hits.selective_hits[:, -1], with_hits.direction_hits)
    # El recálculo del informe coteja cada sesión por (mercado, instante), como al publicar.
    recheck = score_recheck.compare_sessions(
        _independent(frames[True]), with_hits.to_table().to_pandas()
    )
    assert recheck["passed"] and recheck["columns"]["selective_hits"]["mismatched_sessions"] == 0
    timings = {
        f"{path}_{variant}": []
        for path in ("scores", "independent")
        for variant in ("with", "without")
    }
    for _ in range(args.repetitions):
        for selective in (True, False):
            began = time.perf_counter()
            _scores(panel, selective)
            timings["scores_with" if selective else "scores_without"].append(
                time.perf_counter() - began
            )
        for flag in (True, False):
            began = time.perf_counter()
            _independent(frames[flag])
            name = "independent_with" if flag else "independent_without"
            timings[name].append(time.perf_counter() - began)
    summary = {name: _summary(values) for name, values in timings.items()}
    added = {
        path: round(summary[f"{path}_with"]["median"] - summary[f"{path}_without"]["median"], 4)
        for path in ("scores", "independent")
    }
    document = dict(
        schema_version=1,
        kind="abstention_coverage_cost_measure",
        window=args.window,
        rows=rows,
        sessions=int(panel.sessions),
        assets_per_session_max=width,
        coverages=list(forecast_scores.SELECTIVE_COVERAGES),
        recheck=dict(
            sessions=recheck["sessions"],
            columns=len(recheck["columns"]),
            passed=recheck["passed"],
        ),
        repetitions=args.repetitions,
        timings=summary,
        added_seconds_per_series=added,
        relative_increase={
            path: round(added[path] / summary[f"{path}_without"]["median"], 4)
            for path in ("scores", "independent")
        },
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        host=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            pyarrow=pa.__version__,
            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            load_average=os.getloadavg(),
        ),
        measured_at_utc=datetime.now(UTC).isoformat(),
        training_executed=False,
        final_test_opened=False,
    )
    atomic_json(args.output, document)
    print(json.dumps({k: v for k, v in document.items() if k != "host"}, indent=1))


if __name__ == "__main__":
    main()
