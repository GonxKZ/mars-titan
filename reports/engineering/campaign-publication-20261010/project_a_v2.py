"""Disco de los agregados de la comparación postentrenada en todo el recorrido de A v2.

Cuenta las series (brazo y semilla) de cada ventana con `rolling_storage.comparison_series`
sobre el plan de la etapa de adaptadores de A v2, comprueba que coinciden con las de la
declaración de la comparación y las multiplica por los bytes por serie medidos en
`fold-018` con `measure_stage_aggregates.py`. Esa ventana es la de más sesiones y filas, así
que el total es una cota superior. No lee datos ni modelos.

    PYTHONPATH=src uv run --no-sync python \
        reports/engineering/campaign-publication-20261010/project_a_v2.py \
        --strata <medida con estratos> --plain <medida sin estratos> --output <proyección.json>
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.posttraining import campaign_stage, stage_comparison
from mars_titan.training.rolling_storage import comparison_series

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "configs/posttraining/historical-masked-adapter-stage-a-v2.json"
DECLARATION = ROOT / "configs/posttraining/historical-masked-adapter-comparison-a-v2.json"
GB = 1e9


def declared_series(loaded, scope, window):
    """Series de la comparación de cada padre en la ventana, según su configuración."""
    return sum(
        len(arm["seeds"])
        for base_arm, config in loaded["configs"].items()
        if window in stage_comparison.compared_windows(loaded, scope, base_arm)
        for name, arm in config["arms"].items()
        if arm["output"] != "zero_control"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--strata", "--plain", "--output"):
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args()
    measures = {key: json.loads(getattr(args, key).read_text()) for key in ("strata", "plain")}
    stage = campaign_stage.load_stage(STAGE)
    jobs = defaultdict(list)
    for job in campaign_stage.plan_stage(stage):
        jobs[job["scope"], job["window"]].append(job)
    loaded = stage_comparison.load_declaration(DECLARATION)
    windows = []
    for (scope, window), members in jobs.items():
        series = comparison_series(members)
        declared = declared_series(loaded, scope, window)
        if series != declared:
            raise SystemExit(
                f"{scope}/{window}: {series} series en el plan y {declared} declaradas"
            )
        windows.append(dict(scope=scope, window=window, series=series))
    measured = measures["strata"]
    if measured["series"] != next(
        w["series"] for w in windows if w["window"] == measured["window"]
    ):
        raise SystemExit("La medida no tiene las series de su ventana")
    totals = {}
    for key, measure in measures.items():
        per_series = math.ceil(measure["bytes_per_series_max"])
        total = sum(w["series"] for w in windows) * per_series
        totals[key] = dict(bytes_per_series=per_series, bytes=total, gb=round(total / GB, 2))
    result = dict(
        schema_version=1,
        kind="stage_comparison_aggregates_projection",
        stage=str(STAGE.relative_to(ROOT)),
        stage_sha256=sha256(STAGE),
        declaration=str(DECLARATION.relative_to(ROOT)),
        declaration_sha256=sha256(DECLARATION),
        measured_window=measured["window"],
        windows=windows,
        series=sum(w["series"] for w in windows),
        totals=totals,
        bound="upper_measured_window_with_most_sessions",
        training_executed=False,
        final_test_opened=False,
    )
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}, indent=1))


if __name__ == "__main__":
    main()
