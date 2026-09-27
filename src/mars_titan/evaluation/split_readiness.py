"""Contrastar ventanas temporales con cobertura observada antes de preparar modelos."""

import argparse
import calendar
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation.splits import PARTITIONS, FoldPartitioner, build_folds


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError("Se necesita un JSON regular menor de 8 MiB")
    return json.loads(path.read_text())


def _admission(report, config):
    if (
        report.get("schema_version") != 1
        or report.get("policy") != "all_catalog_indicators_valid"
        or report.get("market") != config["market"]
    ):
        raise ValueError("La admisión no corresponde a la política y mercado requeridos")
    for field in ("source_sha256", "catalog_sha256", "complete_decisions_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(report.get(field))):
            raise ValueError("La admisión necesita huellas de procedencia completas")
    for field in ("total_decisions", "complete_decisions", "excluded_decisions"):
        if type(report.get(field)) is not int or report[field] < 0:
            raise ValueError("Los recuentos de admisión deben ser enteros no negativos")
    ids = report.get("required_indicator_ids", [])
    if (
        not ids
        or not all(isinstance(value, str) for value in ids)
        or len(set(ids)) != len(ids)
        or report.get("required_indicator_count") != len(ids)
    ):
        raise ValueError("El catálogo no concilia con los indicadores exigidos")
    if (
        report["total_decisions"] != report["complete_decisions"] + report["excluded_decisions"]
        or type(report.get("population_ready")) is not bool
        or report["population_ready"] != (report["complete_decisions"] > 0)
    ):
        raise ValueError("El estado de admisión no concilia con los recuentos")
    start, end = date.fromisoformat(report["start"]), date.fromisoformat(report["end"])
    if start > date.fromisoformat(config["train_start"]) or end < (
        date.fromisoformat(config["final_test_start"]) - timedelta(days=1)
    ):
        raise ValueError("La admisión no cubre todo el periodo de desarrollo")


def _complete_dates(path, report, config):
    artifact = Path(path).parent / "complete-decisions.parquet"
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or artifact.stat().st_size > 16 * 1024**2
        or sha256(artifact) != report["complete_decisions_sha256"]
    ):
        raise ValueError("El índice de sesiones no conserva su huella o excede el presupuesto")
    columns = ["prediction_at", "macro_available_at", "indicator_count"]
    with pq.ParquetFile(artifact, pre_buffer=False) as file:
        if (
            file.metadata.num_rows > 100_000
            or file.metadata.num_rows != report["complete_decisions"]
        ):
            raise ValueError("El índice de sesiones no concilia con su informe")
        if (
            sum(file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups))
            > 16 * 1024**2
            or file.schema_arrow.field("indicator_count").type != pa.int32()
        ):
            raise ValueError("El índice expandido excede el presupuesto o no tiene conteos int32")
        if any(
            file.schema_arrow.field(name).type != pa.timestamp("us", tz="UTC")
            for name in columns[:2]
        ):
            raise ValueError("El índice debe conservar fechas UTC en microsegundos")
        table = file.read(columns=columns, use_threads=False)
    if any(column.null_count for column in table.columns):
        raise ValueError("El índice contiene valores ausentes")
    prediction = table["prediction_at"].cast(pa.int64()).to_numpy()
    availability = table["macro_available_at"].cast(pa.int64()).to_numpy()
    count = table["indicator_count"].to_numpy()
    lower = np.datetime64(report["start"], "us").astype(np.int64)
    upper = np.datetime64(config["final_test_start"], "us").astype(np.int64)
    if (
        np.any(prediction[1:] <= prediction[:-1])
        or np.any(availability > prediction)
        or np.any(count != report["required_indicator_count"])
        or np.any(prediction < lower)
        or np.any(prediction >= upper)
    ):
        raise ValueError(
            "Las sesiones están duplicadas, fuera de periodo o usan información futura"
        )
    return prediction, availability


def prepare_readiness(protocol_path, admission_path, output):
    """Comprobar cobertura macro por ventana, sin abrir etiquetas ni afirmar entrenamiento."""
    protocol_path, admission_path, output = map(Path, (protocol_path, admission_path, output))
    for source in (protocol_path, admission_path):
        outside_source(source.parent, output)
    config, admission = _read(protocol_path), _read(admission_path)
    folds = build_folds(config)
    _admission(admission, config)
    prediction, availability = _complete_dates(admission_path, admission, config)
    clock = MarketClock(config["market"], config["train_start"], config["final_test_end"])
    sessions = np.array([int(t.timestamp() * 1_000_000) for t in clock.decisions], dtype=np.int64)
    positions = np.searchsorted(sessions, prediction)
    if np.any(positions >= len(sessions) - 1) or not np.array_equal(
        sessions[positions], prediction
    ):
        raise ValueError("El índice contiene decisiones ajenas al calendario")
    # Este límite usa una sesión de horizonte, aún no las etiquetas reales por activo.
    maturity = sessions[positions + 1]
    ready = []
    for fold in folds:
        assigned = FoldPartitioner(fold, clock, config).assign(prediction, availability, maturity)
        counts = {name: int(np.sum(assigned["partition"] == name)) for name in PARTITIONS}
        fold["macro_sessions"] = counts
        history_ready = False
        if counts["train"]:
            first = date.fromisoformat(
                str(np.datetime64(int(prediction[assigned["partition"] == "train"][0]), "us"))[:10]
            )
            month = first.year * 12 + first.month - 1 + config["minimum_train_months"]
            year, number = month // 12, month % 12 + 1
            minimum_end = date(year, number, min(first.day, calendar.monthrange(year, number)[1]))
            boundary = np.datetime64(fold["validation"][0], "us").astype(np.int64)
            index = np.searchsorted(sessions, boundary)
            first_validation = clock.decisions[int(index)].date()
            history_ready = first_validation >= minimum_end
        fold["minimum_history_satisfied"] = history_ready
        if all(counts.values()) and history_ready:
            ready.append(fold["id"])
    status = "macro_windows_ready" if ready else "blocked"
    reason = (
        "requires_multimodal_rows_and_actual_label_purge"
        if ready
        else "no_complete_macro_decisions"
        if not len(prediction)
        else "insufficient_macro_windows"
    )
    result = dict(
        schema_version=1,
        status=status,
        reason=reason,
        folds=folds,
        macro_ready_folds=ready,
        runnable_folds=[],
        protocol_sha256=sha256(protocol_path),
        admission_sha256=sha256(admission_path),
        macro_source_sha256=admission["source_sha256"],
        catalog_sha256=admission["catalog_sha256"],
        final_test=dict(
            start=config["final_test_start"],
            end=config["final_test_end"],
            opened=False,
            materialized=False,
        ),
        scientific_training_started=False,
        required_per_fold_fits=[
            "normalization",
            "action_grid",
            "parent",
            "optimizer",
            "calibration",
        ],
        limits=[
            "La cobertura macro no acredita las otras modalidades ni las etiquetas por activo.",
            "Los conteos previos usan un horizonte de una sesión. "
            "La asignación definitiva exige la maduración real.",
            "Los padres, normalizadores y rejillas anteriores no se pueden reutilizar "
            "si usaron fechas posteriores al corte.",
        ],
    )
    atomic_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_readiness(args.protocol, args.admission, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
