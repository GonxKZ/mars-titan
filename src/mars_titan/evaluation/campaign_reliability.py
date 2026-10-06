"""Calibrar errores congelados y evaluar fiabilidad por modelo sin abrir la reserva final."""

import argparse
import csv
import os
import re
import resource
import time
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.compute as pc

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.macro_coverage import _publish_directory
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.comparison_sources import predictive_sources
from mars_titan.evaluation.forecast_reliability import (
    TARGET_KIND,
    calibrate_absolute_error,
    directional_diagnostics,
    interval_diagnostics,
)
from mars_titan.evaluation.prediction_statistics import (
    InitialPolicyCache,
    _bounds,
    review_predictions,
)

LEVELS = (0.9, 0.95)
_IDENTITY = (
    "fold",
    "id",
    "stage",
    "phase",
    "family",
    "method",
    "seed",
    "included",
    "primary",
    "parent_id",
    "checkpoint_sha256",
)
_RATES = (
    "coverage",
    "call_coverage",
    "conditional_accuracy",
    "direction_accuracy",
    "positive_precision",
    "negative_precision",
    "positive_recall",
    "negative_recall",
    "nonzero_prediction_fraction",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(sources):
    _require(0 < len(sources) <= 8192, "La fiabilidad supera el presupuesto de predicciones")
    grouped = defaultdict(dict)
    for source in sources:
        _bounds(source)
        for name in ("checkpoint_sha256", "source_report_sha256", "sha256"):
            _require(
                isinstance(source.get(name), str) and re.fullmatch(r"[a-f0-9]{64}", source[name]),
                "Falta una huella de procedencia válida",
            )
        key = source["fold"], source["id"]
        partition = source["partition"]
        _require(partition not in grouped[key], "Hay particiones duplicadas para un modelo")
        grouped[key][partition] = source
    result = []
    for _, partitions in sorted(grouped.items()):
        _require(
            set(partitions) == {"calibration", "evaluation"},
            "Falta el par completo de calibración y evaluación",
        )
        calibration, evaluation = partitions["calibration"], partitions["evaluation"]
        _require(
            all(calibration[name] == evaluation[name] for name in _IDENTITY)
            and calibration["metadata"] == evaluation["metadata"],
            "Las particiones no pertenecen al mismo modelo y checkpoint congelados",
        )
        _require(
            _bounds(calibration)[1] <= _bounds(evaluation)[0]
            and calibration["path"].resolve() != evaluation["path"].resolve()
            and calibration["sha256"] != evaluation["sha256"],
            "La calibración debe ser anterior y distinta de la evaluación",
        )
        result.append((calibration, evaluation))
    return result


def _scores(target, prediction, calibration):
    if calibration is None:
        return directional_diagnostics(target, prediction) | dict(
            confidence=None,
            coverage=None,
            covered=None,
            intervals=None,
            mean_width=None,
            reason=None,
        )
    intervals = interval_diagnostics(target, prediction, calibration)
    direction = intervals.pop("direction")
    if direction is None:
        direction = {name: None for name in directional_diagnostics([], [])}
    return direction | intervals


def _sessions(cohort):
    markets = cohort["market"].to_numpy()
    moments = cohort["prediction_at"].to_numpy().astype(np.int64)
    order = np.lexsort((moments, markets))
    changes = (markets[order][1:] != markets[order][:-1]) | (
        moments[order][1:] != moments[order][:-1]
    )
    return order, np.r_[0, np.flatnonzero(changes) + 1, len(order)]


def _session_means(target, prediction, calibration, groups):
    means, counts = dict.fromkeys(_RATES, 0.0), dict.fromkeys(_RATES, 0)
    order, boundaries = groups
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = order[begin:end]
        result = _scores(target[selected], prediction[selected], calibration)
        for name in _RATES:
            value = result[name]
            if value is not None:
                counts[name] += 1
                means[name] += (value - means[name]) / counts[name]
    return {
        **{f"mean_session_{name}": means[name] if counts[name] else None for name in _RATES},
        **{f"defined_sessions_{name}": counts[name] for name in _RATES},
    }


def _case_rows(calibration_source, evaluation_source, calibration, evaluation):
    target = evaluation["cohort"]["target"].to_numpy()
    prediction = evaluation["prediction"]
    groups = _sessions(evaluation["cohort"])
    calibrators = [None] + [
        calibrate_absolute_error(
            calibration["cohort"]["target"].to_numpy(),
            calibration["prediction"],
            confidence=confidence,
            partition=calibration_source["partition"],
        )
        for confidence in LEVELS
    ]
    base = {name: evaluation_source[name] for name in _IDENTITY}
    for partition, source, reviewed in (
        ("calibration", calibration_source, calibration),
        ("evaluation", evaluation_source, evaluation),
    ):
        base.update(
            {
                f"{partition}_predictions_sha256": source["sha256"],
                f"{partition}_source_report_sha256": source["source_report_sha256"],
                f"{partition}_start": source["bounds"][0],
                f"{partition}_end": source["bounds"][1],
                f"{partition}_samples": reviewed["metrics"]["prediction"]["samples"],
                f"{partition}_sessions": reviewed["metrics"]["prediction"]["session_count"],
            }
        )
    for fitted in calibrators:
        yield (
            base
            | dict(
                prediction_kind="point" if fitted is None else "interval",
                radius=None if fitted is None else fitted["radius"],
                order_statistic=None if fitted is None else fitted["order_statistic"],
                coverage_guaranteed=False,
            )
            | _scores(target, prediction, fitted)
            | _session_means(target, prediction, fitted, groups)
        )


def evaluate_campaign_reliability(reference, completion, output):
    """Guardar diagnósticos por modelo con las fuentes intactas y sin promediar semillas.

    Los cocientes por fila conservan sus conteos. Las medias por sesión dan el
    mismo peso a cada par de mercado e instante con denominador definido y
    declaran cuántas sesiones entran en cada media. No se mezclan folds.
    """
    started = time.perf_counter()
    reference, completion, output = map(Path, (reference, completion, output))
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    for source in (reference, completion):
        outside_source(source, output)
        outside_source(output, source)
    sources, provenance = predictive_sources(reference, completion)
    pairs = _pairs(sources)
    initial_cache = InitialPolicyCache()
    current_fold, cohorts, rows, hashed_bytes = None, {}, [], 0
    for calibration_source, evaluation_source in pairs:
        if evaluation_source["fold"] != current_fold:
            current_fold, cohorts = evaluation_source["fold"], {}
        calibration = review_predictions(
            calibration_source, cohort=cohorts.get("calibration"), initial_cache=initial_cache
        )
        evaluation = review_predictions(
            evaluation_source, cohort=cohorts.get("evaluation"), initial_cache=initial_cache
        )
        if not cohorts:
            repeated = pc.is_in(
                calibration["cohort"]["sample_id"], value_set=evaluation["cohort"]["sample_id"]
            )
            _require(not pc.any(repeated).as_py(), "Calibración y evaluación reutilizan muestras")
            cohorts = {"calibration": calibration["cohort"], "evaluation": evaluation["cohort"]}
        rows.extend(_case_rows(calibration_source, evaluation_source, calibration, evaluation))
        hashed_bytes += (
            calibration_source["path"].stat().st_size + evaluation_source["path"].stat().st_size
        )
    root = Path(__file__).parents[1]
    report = dict(
        schema_version=1,
        kind="frozen_campaign_reliability",
        status="completed",
        created_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        target_kind=TARGET_KIND,
        coverage_guaranteed=False,
        provenance=provenance,
        counts=dict(
            models=len(pairs),
            prediction_files=len(sources),
            included_models=sum(pair[0]["included"] for pair in pairs),
            folds=len(provenance["folds"]),
        ),
        method=dict(
            calibration_partition="calibration",
            evaluation_partition="evaluation",
            confidence_levels=list(LEVELS),
            calibration_weighting="rows",
            case_rates="ratios_of_row_counts",
            direction_denominator="nonzero_targets",
            conditional_denominator="nonzero_targets_with_a_call",
            interval_sign_rule="lower_gt_zero_or_upper_lt_zero",
            session_weighting="equal_weight_of_sessions_with_defined_denominator",
            seed_or_fold_pooling=False,
            selection="frozen_before_calibration",
            limitations=[
                "La dependencia temporal impide afirmar una garantía de cobertura.",
                "La dirección corresponde al retorno residual, no a la subida bruta del activo.",
            ],
        ),
        cases=rows,
        resources=dict(
            hashed_prediction_bytes=hashed_bytes,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            gpu_used=False,
        ),
        versions={name: version(name) for name in ("numpy", "scipy", "pyarrow")},
        analysis_source_sha256={
            name: sha256(root / name)
            for name in (
                "evaluation/campaign_reliability.py",
                "evaluation/forecast_reliability.py",
                "evaluation/prediction_statistics.py",
                "evaluation/comparison_sources.py",
                "posttraining/parent_selection.py",
                "data/macro_coverage.py",
            )
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        with (stage / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(dict.fromkeys(name for row in rows for name in row))
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        report["artifacts"] = {"cases.csv": sha256(stage / "cases.csv")}
        report["resources"]["elapsed_before_receipt_seconds"] = time.perf_counter() - started
        atomic_json(stage / "reliability.json", report)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "completion", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_campaign_reliability(args.reference, args.completion, args.output)
    print(f"Fiabilidad evaluada para {report['counts']['models']} modelos. Test final cerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
